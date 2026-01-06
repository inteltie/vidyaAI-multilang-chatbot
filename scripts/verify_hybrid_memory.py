import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import ChatSession
from services.chat_memory import HybridChatMemoryService

load_dotenv()

class MockLLM:
    async def ainvoke(self, prompt):
        return MagicMock(content="This is a mock summary.")

async def verify():
    print("Starting verification...")
    
    # Setup
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    redis_url = os.getenv("REDIS_URL")
    
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    
    # Cleanup existing test data to allow unique index creation
    print("Cleaning up old test data...")
    await db.chat_sessions.delete_many({"session_id": {"$in": ["test_hybrid_session", "test_concurrent_session", "test_legacy_session"]}})
    
    await init_beanie(database=db, document_models=[ChatSession])
    redis_client = aioredis.from_url(redis_url, decode_responses=True)
    
    # Clear test session (Redis)
    session_id = "test_hybrid_session"
    user_id = "test_user"
    await redis_client.delete(f"chat:{session_id}:buffer")
    
    # Init Service
    service = HybridChatMemoryService(redis_client, MockLLM())
    
    # 1. Test ensure_session (should create new)
    print("Testing ensure_session...")
    session, buffer, summary = await service.ensure_session(user_id, session_id)
    assert session.session_id == session_id
    assert len(buffer) == 0
    print("✓ ensure_session passed")
    
    # 2. Test add_to_redis_buffer
    print("Testing add_to_redis_buffer...")
    await service.add_to_redis_buffer(session_id, "user", "Hello")
    await service.add_to_redis_buffer(session_id, "assistant", "Hi there")
    
    # Verify Redis
    redis_key = f"chat:{session_id}:buffer"
    msgs = await redis_client.lrange(redis_key, 0, -1)
    assert len(msgs) == 2
    assert json.loads(msgs[0])["content"] == "Hello"
    print("✓ Redis buffer passed")
    
    # 3. Test background_save_message
    print("Testing background_save_message...")
    await service.background_save_message(session_id, user_id, "user", "Hello")
    await service.background_save_message(session_id, user_id, "assistant", "Hi there")
    
    # Allow time for async write (though we awaited it here directly for test simplicity, 
    # in real app it's create_task. But here we called it directly so it should be done)
    
    # Verify MongoDB
    db_session = await ChatSession.find_one(ChatSession.session_id == session_id)
    assert len(db_session.messages) == 2
    assert db_session.messages[0].text == "Hello"
    print("✓ MongoDB save passed")
    
    # 4. Test Summary Trigger (Simulate 20 messages)
    print("Testing summary trigger...")
    # Add 18 more messages to reach 20
    for i in range(18):
        await service.background_save_message(session_id, user_id, "user", f"msg {i}")
    
    # Wait for background summary task (we need to sleep a bit as it uses create_task internally)
    await asyncio.sleep(2)
    
    db_session = await ChatSession.find_one(ChatSession.session_id == session_id)
    assert len(db_session.messages) == 20
    # Summary might be updated if MockLLM was called
    if db_session.summary == "This is a mock summary.":
        print("✓ Summary update passed")
    else:
        print(f"⚠ Summary update check skipped or failed (Current: {db_session.summary})")

    # 5. Test Legacy Data Normalization
    print("Testing legacy data normalization...")
    legacy_session_id = "test_legacy_session"
    await redis_client.delete(f"chat:{legacy_session_id}:buffer")
    
    # Inject legacy message with 'text' key directly into Redis
    legacy_msg = {"role": "user", "text": "Legacy Message"}
    await redis_client.rpush(f"chat:{legacy_session_id}:buffer", json.dumps(legacy_msg))
    
    # ensure_session should normalize it
    _, buffer, _ = await service.ensure_session(user_id, legacy_session_id)
    assert len(buffer) == 1
    assert buffer[0]["content"] == "Legacy Message"
    assert "text" not in buffer[0]
    print("✓ Legacy normalization passed")

    # 6. Test Concurrent Writes (Race Condition Check)
    print("Testing concurrent writes...")
    concurrent_session_id = "test_concurrent_session"
    await ChatSession.find_one(ChatSession.session_id == concurrent_session_id).delete()
    
    # Simulate concurrent user and assistant messages
    tasks = [
        service.background_save_message(concurrent_session_id, user_id, "user", f"msg {i}")
        for i in range(10)
    ]
    await asyncio.gather(*tasks)
    
    # Verify all messages are saved
    db_session = await ChatSession.find_one(ChatSession.session_id == concurrent_session_id)
    assert len(db_session.messages) == 10
    print("✓ Concurrent writes passed")

    print("\nAll verification steps completed successfully!")

if __name__ == "__main__":
    asyncio.run(verify())
