
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import MagicMock, AsyncMock

from services.chat_memory import MemoryService
from models.chat import ChatSession, IST

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from services.chat_memory import MemoryService

# Mock IST for consistency
class MockIST:
    utcoffset = lambda self, dt: timedelta(hours=5, minutes=30)
    tzname = lambda self, dt: "IST"
    dst = lambda self, dt: timedelta(0)
IST = MockIST()

async def test_session_restart():
    print("Testing Session Restart logic...")
    
    with patch("services.chat_memory.ChatSession") as MockChatSession:
        # 1. Setup Mock MemoryService
        redis_mock = AsyncMock()
        redis_mock.llen.return_value = 0 # Default to empty buffer
        redis_mock.lrange.return_value = []
        
        llm_mock = MagicMock()
        memory_service = MemoryService(redis_mock, llm_mock)
        
        # Helper to create mock session
        def create_mock_session(updated_at_ist, message_count=1):
            session = MagicMock()
            session.session_id = "test_sid"
            session.user_id = "test_uid"
            session.updated_at = updated_at_ist
            
            # Create mock message objects
            mock_msgs = []
            for _ in range(message_count):
                m = MagicMock()
                m.role = "user"
                m.text = "Hello"
                mock_msgs.append(m)
                
            session.messages = mock_msgs
            session.summary = "Old summary"
            session.insert = AsyncMock()
            return session

        now = datetime.utcnow()

        # Case A: Brand new session (No restart)
        async def mock_find_none(*args, **kwargs): return None
        MockChatSession.find_one = mock_find_none
        
        MockChatSession.return_value = create_mock_session(None, 0)
        _, _, _, is_restart = await memory_service.ensure_session("test_uid", "new_sid")
        print(f"New session: is_restart={is_restart}")
        assert is_restart is False
        
        # Case B: Recent session (1 hour ago) (No restart)
        one_hour_ago = now - timedelta(hours=1)
        async def mock_find_recent(*args, **kwargs): return create_mock_session(one_hour_ago)
        MockChatSession.find_one = mock_find_recent
        
        _, _, _, is_restart = await memory_service.ensure_session("test_uid", "recent_sid")
        print(f"Recent session (1h ago): is_restart={is_restart}")
        assert is_restart is False
        
        # Case C: Old session (3 hours ago) (RESTART)
        three_hours_ago = now - timedelta(hours=3)
        async def mock_find_old(*args, **kwargs): return create_mock_session(three_hours_ago)
        MockChatSession.find_one = mock_find_old
        
        _, _, _, is_restart = await memory_service.ensure_session("test_uid", "old_sid")
        print(f"Old session (3h ago): is_restart={is_restart}")
        assert is_restart is True
        
        # Case D: Old session but no messages (Empty session) (No restart)
        async def mock_find_old_empty(*args, **kwargs): return create_mock_session(three_hours_ago, message_count=0)
        MockChatSession.find_one = mock_find_old_empty
        
        _, _, _, is_restart = await memory_service.ensure_session("test_uid", "old_empty_sid")
        print(f"Old empty session (3h ago): is_restart={is_restart}")
        assert is_restart is False

    print("\n✅ Session Restart logic verified successfully!")

if __name__ == "__main__":
    asyncio.run(test_session_restart())
