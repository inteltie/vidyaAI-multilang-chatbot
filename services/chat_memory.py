import asyncio
import json
import logging
from typing import List, Optional, Tuple, Dict, Any

import redis.asyncio as aioredis
from pymongo.errors import DuplicateKeyError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages

from models import ChatSession, ChatMessage
from state import ConversationTurn
from config import settings

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Enhanced memory service with token-based trimming and background summarization.
    Uses Redis for short-term buffer and MongoDB for long-term storage.
    """

    def __init__(self, redis_client: aioredis.Redis, llm: BaseChatModel) -> None:
        self._redis = redis_client
        self._llm = llm
        self._memory_token_limit = getattr(settings, "memory_token_limit", 2000)

    async def ensure_session(self, user_id: str, session_id: str) -> Tuple[ChatSession, List[Dict[str, str]], Optional[str]]:
        """
        Ensure session exists, load summary, and prepare Redis buffer.
        Returns: (session_object, redis_buffer_messages, summary)
        """
        # 1. Get or create MongoDB session
        session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not session:
            session = ChatSession(session_id=session_id, user_id=user_id)
            await session.insert()
        
        summary = session.summary

        # 2. Check Redis buffer
        redis_key = f"chat:{session_id}:buffer"
        buffer_len = await self._redis.llen(redis_key)

        if buffer_len > 0:
            # Load from Redis
            raw_msgs = await self._redis.lrange(redis_key, 0, -1)
            buffer = []
            for m in raw_msgs:
                msg = json.loads(m)
                if "content" not in msg and "text" in msg:
                    msg["content"] = msg.pop("text")
                buffer.append(msg)
        else:
            # Rebuild from MongoDB (seed Redis)
            recent_msgs = session.messages[-settings.memory_buffer_size:]
            buffer = [{"role": m.role, "content": m.text} for m in recent_msgs]
            
            if buffer:
                await self._redis.rpush(redis_key, *[json.dumps(m) for m in buffer])
                await self._redis.expire(redis_key, 3600) # 1 hour TTL

        return session, buffer, summary

    async def get_context(self, session_id: str) -> Tuple[Optional[str], List[BaseMessage]]:
        """
        Retrieve structured context: (Summary, Token-Trimmed Messages).
        This is what agents should consume.
        """
        session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not session:
            return None, []

        summary = session.summary
        
        # Convert buffer/history to BaseMessage format for trimming
        # Use Redis as the source of truth for recent history to avoid DB latency
        redis_key = f"chat:{session_id}:buffer"
        raw_msgs = await self._redis.lrange(redis_key, 0, -1)
        
        messages: List[BaseMessage] = []
        for rm in raw_msgs:
            m = json.loads(rm)
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Perform token-based trimming (keep last N tokens)
        trimmed_history = trim_messages(
            messages,
            max_tokens=self._memory_token_limit,
            strategy="last",
            token_counter=self._llm, # Uses the model's tokenizer if available
            start_on="human",
            include_system=False, # Summary will be in system prompt
        )

        return summary, trimmed_history

    async def add_message(self, session_id: str, role: str, content: str):
        """Add message to Redis buffer and MongoDB."""
        # 1. Redis Buffer
        redis_key = f"chat:{session_id}:buffer"
        msg = {"role": role, "content": content}
        
        # Check last message in Redis to prevent duplicates
        last_msg_json = await self._redis.lindex(redis_key, -1)
        if last_msg_json:
            last_msg = json.loads(last_msg_json)
            if last_msg.get("role") == role and last_msg.get("content") == content:
                logger.warning(f"Duplicate Redis message detected for session {session_id}. Skipping.")
                return

        async with self._redis.pipeline() as pipe:
            await pipe.rpush(redis_key, json.dumps(msg))
            # Keep Redis buffer slightly larger than default for safety
            redis_buffer_limit = settings.memory_buffer_size + 10
            await pipe.ltrim(redis_key, -redis_buffer_limit, -1)
            await pipe.expire(redis_key, 3600)
            await pipe.execute()

    async def background_save_message(self, session_id: str, user_id: str, role: str, content: str):
        """Background task to save message to MongoDB and handle summarization."""
        try:
            session = await ChatSession.find_one(ChatSession.session_id == session_id)
            if not session:
                session = ChatSession(session_id=session_id, user_id=user_id)
                try:
                    await session.insert()
                except DuplicateKeyError:
                    session = await ChatSession.find_one(ChatSession.session_id == session_id)
            
            # Check for duplicate message in MongoDB
            if session.messages:
                last_msg = session.messages[-1]
                if last_msg.role == role and last_msg.text == content:
                    logger.warning(f"Duplicate MongoDB message detected for session {session_id}. Skipping.")
                    return

            await session.add_message(role, content)
            
            # Update summary every 10 messages for better context (increased frequency)
            if len(session.messages) % 10 == 0:
                asyncio.create_task(self.background_update_summary(session_id))
                
        except Exception as e:
            logger.error(f"Failed to save message to MongoDB: {e}")

    async def background_update_summary(self, session_id: str):
        """Background task to generate and save session summary."""
        try:
            session = await ChatSession.find_one(ChatSession.session_id == session_id)
            if not session or not session.messages:
                return

            # Incremental summarization: Use previous summary + last 20 messages
            recent_messages = session.messages[-20:]
            messages_text = "\n".join([f"{m.role}: {m.text}" for m in recent_messages])
            
            if session.summary:
                prompt = (
                    f"Here is a summary of the conversation so far:\n{session.summary}\n\n"
                    f"Here are the latest 20 messages:\n{messages_text}\n\n"
                    "Update the summary to include the new information, keeping it concise (3-5 sentences). "
                    "Focus on key topics and user preferences."
                )
            else:
                prompt = (
                    "Summarize the following conversation in 3-5 sentences, capturing key topics and user preferences.\n\n"
                    f"{messages_text}"
                )
            
            response = await self._llm.ainvoke(prompt)
            summary = response.content.strip()
            
            session.summary = summary
            await session.save()
            logger.info(f"Updated summary for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to update summary for session {session_id}: {e}")

    async def get_history(self, session_id: str) -> List[ConversationTurn]:
        """Get history from Redis buffer (for compatibility)."""
        redis_key = f"chat:{session_id}:buffer"
        raw_msgs = await self._redis.lrange(redis_key, 0, -1)
        return [
            ConversationTurn(role=m["role"], content=m["content"])
            for m in [json.loads(rm) for rm in raw_msgs]
        ]
