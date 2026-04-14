import asyncio
import json
import uuid
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from config import settings
from services.chat_memory import MemoryService
from models.chat import ChatSession
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from langchain_openai import ChatOpenAI

async def test_memory():
    # 1. Setup
    print("Setting up test environments...")
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    mongo_client = AsyncIOMotorClient(settings.mongo_uri)
    await init_beanie(
        database=mongo_client[settings.mongo_db_name],
        document_models=[ChatSession]
    )
    
    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key)
    llm_summary = ChatOpenAI(model=settings.summary_model_name, api_key=settings.openai_api_key)
    memory_service = MemoryService(redis_client, llm, llm_summary)
    
    user_id = "test_user"
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    
    print(f"Testing session: {session_id}")
    
    # Check if session exists (to clean up if needed)
    session = await ChatSession.find_one(ChatSession.session_id == session_id)
    if not session:
        session = ChatSession(session_id=session_id, user_id=user_id)
        await session.insert()
        
    # 2. Add 10 messages (5 turns) to trigger summarization
    messages = [
        ("user", "Hi, my name is John."),
        ("assistant", "Hello John! How can I help you today?"),
        ("user", "I want to learn about photosynthesis."),
        ("assistant", "Sure! Photosynthesis is how plants make food using sunlight."),
        ("user", "I have trouble understanding the role of chlorophyll."),
        ("assistant", "Chlorophyll is the green pigment that absorbs light energy."),
        ("user", "Oh, like a solar panel?"),
        ("assistant", "Exactly! It converts light into chemical energy."),
        ("user", "Cool. I struggle specifically with the Calvin cycle part."),
        ("assistant", "The Calvin cycle is the light-independent phase. We can dive into that.")
    ]
    
    print("Adding messages to session...")
    for role, content in messages:
        await memory_service.add_message(session_id, role, content)
        # We need to call the background save manually or wait for it
        # In the real app, it's a task. Here we'll just insert into DB for the test.
        session = await ChatSession.find_one(ChatSession.session_id == session_id)
        # Add message to Mongo
        from models.chat import ChatMessage
        await session.update({"$push": {"messages": ChatMessage(role=role, text=content)}})

    # 3. Trigger summarization
    print("Triggering structured summarization...")
    await memory_service.background_update_summary(session_id)
    
    # 4. Verify result
    session = await ChatSession.find_one(ChatSession.session_id == session_id)
    print("\n--- STICKY SUMMARY VERIFICATION ---")
    if session.structured_summary:
        print("✅ Structured summary found!")
        print(json.dumps(session.structured_summary, indent=2))
        
        # Check entities
        entities = session.structured_summary.get("key_entities", {})
        if "John" in str(entities.get("name", "")):
            print("✅ Successfully extracted name: John")
        if "Photosynthesis" in str(entities.get("topic", "")):
            print("✅ Successfully extracted topic: Photosynthesis")
        # Check if gaps were extracted
        gaps = str(entities.get("identified_gaps", ""))
        if "Calvin cycle" in gaps or "Calvin" in gaps:
            print("✅ Successfully identified gaps: Calvin cycle")
    else:
        print("❌ Structured summary NOT found.")
        print(f"Fallback summary: {session.summary}")

    # 5. Cleanup
    await session.delete()
    await redis_client.delete(f"chat:{session_id}:buffer")
    await redis_client.aclose()
    mongo_client.close()

if __name__ == "__main__":
    asyncio.run(test_memory())
