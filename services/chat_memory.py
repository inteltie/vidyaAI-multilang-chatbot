import asyncio
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple, Dict, Any

import redis.asyncio as aioredis
from pymongo.errors import (
    ConnectionFailure,
    DuplicateKeyError,
    NetworkTimeout,
    ServerSelectionTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, trim_messages
import tiktoken

from models import ChatSession, ChatMessage, StructuredSummary, UploadedImageRecord
from models.chat import GeneratedImageRecord, get_ist_now
from state import ConversationTurn
from config import settings
from services.message_modality import modality_to_kwargs

logger = logging.getLogger(__name__)

_MONGO_TRANSIENT_ERRORS = (
    NetworkTimeout,
    ServerSelectionTimeoutError,
    ConnectionFailure,
)


class MemoryService:
    """
    Enhanced memory service with token-based trimming and background summarization.
    Uses Redis for short-term buffer and MongoDB for long-term storage.
    """

    def __init__(self, redis_client: aioredis.Redis, llm: BaseChatModel, summary_llm: Optional[BaseChatModel] = None) -> None:
        self._redis = redis_client
        self._llm = llm
        self._summary_llm = summary_llm or llm
        self._memory_token_limit = getattr(settings, "memory_token_limit", 2000)
        self.SESSION_RESTART_THRESHOLD = 7200  # 2 hours in seconds
        self._tokenizer_warmed = False
        self._summary_max_tokens = getattr(settings, "summary_max_tokens", 400)
        self._summary_encoder = None

    def _get_summary_encoder(self):
        if self._summary_encoder is not None:
            return self._summary_encoder
        model_name = getattr(self._llm, "model_name", None) or settings.model_name
        try:
            self._summary_encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            self._summary_encoder = tiktoken.get_encoding("cl100k_base")
        return self._summary_encoder

    def _truncate_summary(self, text: str) -> str:
        if not text:
            return text
        enc = self._get_summary_encoder()
        tokens = enc.encode(text)
        if len(tokens) <= self._summary_max_tokens:
            return text
        truncated = enc.decode(tokens[: self._summary_max_tokens])
        return truncated.rstrip() + " …"

    async def warmup(self):
        """Pre-load tiktoken encodings to avoid cold-start delays on first use."""
        if self._tokenizer_warmed:
            return
        try:
            logger.info("Warming up tokenizer...")
            # This triggers the download/loading of the bpe files
            await asyncio.to_thread(self._llm.get_num_tokens, "Warmup text for tiktoken.")
            self._tokenizer_warmed = True
            logger.info("Tokenizer warmup complete.")
        except Exception as e:
            logger.warning(f"Tokenizer warmup failed: {e}")

    async def ensure_session(self, user_id: str, session_id: str) -> Tuple[ChatSession, List[Dict[str, str]], Optional[str], bool]:
        """
        Ensure session exists, load summary, and prepare Redis buffer.
        Returns: (session_object, redis_buffer_messages, summary, is_restart)
        """
        # 1. Get or create MongoDB session
        session = await ChatSession.find_one(ChatSession.session_id == session_id)
        is_restart = False
        
        if not session:
            session = ChatSession(session_id=session_id, user_id=user_id)
            await session.insert()
        else:
            # Check for restart condition: updated_at > threshold
            if session.updated_at:
                # updated_at might be offset-aware or naive depending on driver/data
                # Standardize both to naive UTC for safe comparison
                now = datetime.utcnow()
                
                # If session.updated_at is aware, convert to UTC and make naive
                updated_at = session.updated_at
                if updated_at.tzinfo is not None:
                    updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                
                diff = (now - updated_at).total_seconds()
                if diff > self.SESSION_RESTART_THRESHOLD and len(session.messages) > 0:
                    is_restart = True
                    logger.info("Session %s detected as restart (Inactivity: %.1fs)", session_id, diff)
        
        summary = session.summary

        # 2. Check Redis buffer
        redis_key = f"chat:{session_id}:buffer"
        buffer = []
        try:
            buffer_len = await self._redis.llen(redis_key)
            if buffer_len > 0:
                # Load from Redis
                raw_msgs = await self._redis.lrange(redis_key, 0, -1)
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
                    await self._redis.expire(redis_key, settings.memory_buffer_ttl)
        except Exception as e:
            logger.warning(f"Redis fallback in ensure_session for {session_id}: {e}")
            # Fallback: Load directly from MongoDB
            recent_msgs = session.messages[-settings.memory_buffer_size:]
            buffer = [{"role": m.role, "content": m.text} for m in recent_msgs]

        return session, buffer, summary, is_restart

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
        messages: List[BaseMessage] = []
        try:
            raw_msgs = await self._redis.lrange(redis_key, 0, -1)
            for rm in raw_msgs:
                m = json.loads(rm)
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        except Exception as e:
            logger.warning(f"Redis fallback in get_context for {session_id}: {e}")
            # Fallback to MongoDB messages if Redis fails
            recent_msgs = session.messages[-settings.memory_buffer_size:]
            for m in recent_msgs:
                if m.role == "user":
                    messages.append(HumanMessage(content=m.text))
                else:
                    messages.append(AIMessage(content=m.text))

        # Perform token-based trimming (keep last N tokens)
        # Enforce turn limit (buffer_size) BEFORE token limit
        trimmed_history = trim_messages(
            messages[-settings.memory_buffer_size:],
            max_tokens=self._memory_token_limit,
            strategy="last",
            token_counter=self._llm,
            start_on="human",
            include_system=False,
        )

        return summary, trimmed_history

    @staticmethod
    def _buffer_row(
        role: str,
        content: str,
        *,
        has_image_input: bool = False,
        image_input_ids: Optional[List[str]] = None,
        image_input_urls: Optional[List[str]] = None,
        has_generated_image: bool = False,
        generated_image_ids: Optional[List[str]] = None,
        generated_image_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "role": role,
            "content": content,
            "has_image_input": has_image_input,
            "image_input_ids": list(image_input_ids or []),
            "image_input_urls": list(image_input_urls or []),
            "has_generated_image": has_generated_image,
            "generated_image_ids": list(generated_image_ids or []),
            "generated_image_urls": list(generated_image_urls or []),
        }

    @staticmethod
    def _mongo_row_to_buffer(msg: ChatMessage) -> Dict[str, Any]:
        return MemoryService._buffer_row(
            msg.role,
            msg.text,
            has_image_input=bool(getattr(msg, "has_image_input", False)),
            image_input_ids=list(getattr(msg, "image_input_ids", None) or []),
            image_input_urls=list(getattr(msg, "image_input_urls", None) or []),
            has_generated_image=bool(getattr(msg, "has_generated_image", False)),
            generated_image_ids=list(getattr(msg, "generated_image_ids", None) or []),
            generated_image_urls=list(getattr(msg, "generated_image_urls", None) or []),
        )

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        has_image_input: bool = False,
        image_input_ids: Optional[List[str]] = None,
        image_input_urls: Optional[List[str]] = None,
        has_generated_image: bool = False,
        generated_image_ids: Optional[List[str]] = None,
        generated_image_urls: Optional[List[str]] = None,
    ):
        """Add message to Redis buffer (Mongo persistence via background_save_message)."""
        redis_key = f"chat:{session_id}:buffer"
        msg = self._buffer_row(
            role,
            content,
            has_image_input=has_image_input,
            image_input_ids=image_input_ids,
            image_input_urls=image_input_urls,
            has_generated_image=has_generated_image,
            generated_image_ids=generated_image_ids,
            generated_image_urls=generated_image_urls,
        )
        
        try:
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
                await pipe.expire(redis_key, settings.memory_buffer_ttl)
                await pipe.execute()
        except Exception as e:
            logger.warning(f"Skipping Redis update for {session_id} due to connection error: {e}")

    async def background_save_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        *,
        has_image_input: bool = False,
        image_input_ids: Optional[List[str]] = None,
        image_input_urls: Optional[List[str]] = None,
        has_generated_image: bool = False,
        generated_image_ids: Optional[List[str]] = None,
        generated_image_urls: Optional[List[str]] = None,
    ):
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

            await session.add_message(
                role,
                content,
                has_image_input=has_image_input,
                image_input_ids=image_input_ids,
                image_input_urls=image_input_urls,
                has_generated_image=has_generated_image,
                generated_image_ids=generated_image_ids,
                generated_image_urls=generated_image_urls,
            )
            
            # Update summary every 10 messages for better context (increased frequency)
            if len(session.messages) % 10 == 0:
                asyncio.create_task(self.background_update_summary(session_id))
                
        except Exception as e:
            logger.error(f"Failed to save message to MongoDB: {e}")

    async def background_update_summary(self, session_id: str):
        """Background task to generate and save session summary with atomic locks."""
        try:
            # 1. Try to acquire summarization lock
            # We only start if is_summarizing is False
            session = await ChatSession.find_one(
                ChatSession.session_id == session_id,
                ChatSession.is_summarizing == False
            )
            
            if not session or not session.messages:
                return

            # Atomic lock set
            await session.update({"$set": {"is_summarizing": True}})
            
            try:
                # Incremental summarization: Use previous summary + last 20 messages
                recent_messages = session.messages[-20:]
                messages_text = "\n".join([f"{m.role}: {m.text}" for m in recent_messages])
                
                # Use structured summary if available, fallback to simple summary
                current_summary = session.structured_summary or {"summary": session.summary or ""}
                if hasattr(current_summary, "model_dump"):
                    current_summary_payload = current_summary.model_dump()
                elif isinstance(current_summary, dict):
                    current_summary_payload = current_summary
                else:
                    current_summary_payload = {"summary": str(current_summary)}
                
                prompt = (
                    "You are a memory compaction system. Your goal is to maintain a structured summary of the conversation.\n"
                    "Extract and update the following information in JSON format:\n"
                    "1. summary: A 3-5 sentence overview of the conversation.\n"
                    "2. key_entities: {name, class_level, topic, identified_gaps, others}: "
                    "Extract the student's name, academic class level (e.g., 'Class 10'), the main subjects discussed, and specific learning gaps identified.\n"
                    "3. user_preferences: Any explicit preferences or constraints the user mentioned.\n\n"
                    f"Previous Data: {json.dumps(current_summary_payload)}\n\n"
                    f"New Messages:\n{messages_text}\n\n"
                    "Output ONLY valid JSON."
                )
                
                # Use dedicated summary LLM (Nano)
                response = await self._summary_llm.ainvoke(
                    prompt,
                    config={"max_tokens": self._summary_max_tokens},
                )
                
                raw_content = response.content.strip()
                # Remove triple backticks ONLY if both opening and closing fences are present
                if raw_content.startswith("```json") and raw_content.endswith("```"):
                    raw_content = raw_content[7:-3].strip()
                elif raw_content.startswith("```") and raw_content.endswith("```"):
                    raw_content = raw_content[3:-3].strip()
                else:
                    # Best-effort JSON extraction when model prepends/appends prose.
                    match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
                    if match:
                        raw_content = match.group(0).strip()

                last_valid_structured = (
                    current_summary_payload if isinstance(current_summary_payload, dict) else {}
                )
                try:
                    structured_summary_raw = json.loads(raw_content)
                except json.JSONDecodeError:
                    logger.error("Failed to parse structured summary JSON for %s. Content: %s", session_id, raw_content)
                    structured_summary = last_valid_structured or None
                    if last_valid_structured:
                        summary_str = self._truncate_summary(str(last_valid_structured))
                    else:
                        summary_str = self._truncate_summary(raw_content)
                else:
                    try:
                        validated = StructuredSummary.model_validate(structured_summary_raw)
                        structured_summary = validated.model_dump(exclude_none=True)
                        summary_str = self._truncate_summary(structured_summary.get("summary", ""))
                    except Exception as exc:
                        logger.error(
                            "Structured summary schema validation failed for %s: %s. Keeping previous structured summary.",
                            session_id,
                            exc,
                        )
                        structured_summary = last_valid_structured or None
                        fallback_summary = ""
                        if isinstance(last_valid_structured, dict):
                            fallback_summary = str(last_valid_structured.get("summary", ""))
                        summary_str = self._truncate_summary(fallback_summary)

                # Track token usage for background summarization
                usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage", {})
                if usage:
                    i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                    o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                    total_tokens = usage.get("total_tokens") or (i_tokens + o_tokens)
                    try:
                        from services.cache_service import CacheService
                        key = f"bg_tokens:{session_id}"
                        await CacheService.incr_hash(key, "input_tokens", int(i_tokens))
                        await CacheService.incr_hash(key, "output_tokens", int(o_tokens))
                        await CacheService.incr_hash(key, "total_tokens", int(total_tokens))
                    except Exception as e:
                        logger.debug("Failed to record background token usage: %s", e)
                
                # 2. Atomic update for summary AND lock release
                update_data = {"is_summarizing": False}
                if summary_str:
                    update_data["summary"] = summary_str
                if structured_summary:
                    update_data["structured_summary"] = structured_summary
                
                await ChatSession.find_one(ChatSession.session_id == session_id).update(
                    {"$set": update_data}
                )
                logger.info(f"Updated structured summary for session {session_id}")
            except Exception as e:
                # Ensure lock is released even on LLM failure
                await ChatSession.find_one(ChatSession.session_id == session_id).update(
                    {"$set": {"is_summarizing": False}}
                )
                raise e
                
        except Exception as e:
            logger.error(f"Failed to update summary for session {session_id}: {e}")

    async def _fetch_session_document(self, session_id: str) -> tuple[Optional[ChatSession], bool]:
        """Load session from Mongo; return (session, mongo_reachable)."""
        try:
            session = await ChatSession.find_one(ChatSession.session_id == session_id)
            return session, True
        except _MONGO_TRANSIENT_ERRORS as exc:
            logger.warning(
                "MongoDB timeout loading session %s (using Redis-only fallback): %s",
                session_id,
                exc,
            )
            return None, False
        except Exception as exc:
            logger.warning(
                "MongoDB error loading session %s (using Redis-only fallback): %s",
                session_id,
                exc,
            )
            return None, False

    async def _load_redis_buffer(self, session_id: str) -> List[Dict[str, Any]]:
        """Load chat buffer from Redis only (fast path)."""
        redis_key = f"chat:{session_id}:buffer"
        try:
            raw_msgs = await self._redis.lrange(redis_key, 0, -1)
            if not raw_msgs:
                return []
            buffer = [json.loads(m) for m in raw_msgs]
            for msg in buffer:
                if "content" not in msg and "text" in msg:
                    msg["content"] = msg.pop("text")
            return buffer
        except Exception as exc:
            logger.debug("Redis buffer load failed for %s: %s", session_id, exc)
            return []

    async def _load_chat_buffer(
        self,
        session_id: str,
        session: Optional[ChatSession],
    ) -> List[Dict[str, Any]]:
        """Load recent messages from Redis, seeding from Mongo when possible."""
        buffer = await self._load_redis_buffer(session_id)
        if buffer:
            return buffer

        redis_key = f"chat:{session_id}:buffer"
        try:
            if session and session.messages:
                recent_msgs = session.messages[-settings.memory_buffer_size :]
                buffer = [self._mongo_row_to_buffer(m) for m in recent_msgs]
                if buffer:
                    await self._redis.rpush(redis_key, *[json.dumps(m) for m in buffer])
                    await self._redis.expire(redis_key, settings.memory_buffer_ttl)
        except Exception as exc:
            logger.warning("Redis buffer load failed for %s: %s", session_id, exc)
            if session and session.messages:
                recent_msgs = session.messages[-settings.memory_buffer_size :]
                buffer = [self._mongo_row_to_buffer(m) for m in recent_msgs]
        return buffer

    async def load_session_full(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Optimized single-pass session load: Returns everything needed for LoadMemoryNode.
        Redis buffer first; Mongo metadata with a short timeout. Falls back to Redis-only
        when MongoDB is slow or unreachable.
        """
        buffer = await self._load_redis_buffer(session_id)

        session: Optional[ChatSession] = None
        mongo_ok = True
        try:
            session, mongo_ok = await asyncio.wait_for(
                self._fetch_session_document(session_id),
                timeout=settings.mongo_session_load_timeout_s,
            )
        except asyncio.TimeoutError:
            mongo_ok = False
            logger.warning(
                "MongoDB session load exceeded %.1fs for %s (using Redis-first fallback)",
                settings.mongo_session_load_timeout_s,
                session_id,
            )

        is_restart = False
        summary = None

        if not session:
            if mongo_ok:
                try:
                    session = ChatSession(session_id=session_id, user_id=user_id)
                    await session.insert()
                except Exception as exc:
                    mongo_ok = False
                    session = None
                    logger.warning(
                        "Could not create Mongo session %s: %s",
                        session_id,
                        exc,
                    )
        else:
            if session.updated_at:
                now = datetime.utcnow()
                updated_at = session.updated_at
                if updated_at.tzinfo is not None:
                    updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)

                diff = (now - updated_at).total_seconds()
                if diff > self.SESSION_RESTART_THRESHOLD and len(session.messages) > 0:
                    is_restart = True
                    logger.info(
                        "Session %s detected as restart (Inactivity: %.1fs)",
                        session_id,
                        diff,
                    )
            summary = session.summary

        if not buffer:
            buffer = await self._load_chat_buffer(session_id, session)

        # 3. Convert to BaseMessage and Trim
        messages: List[BaseMessage] = []
        for m in buffer:
            role = m.get("role", "user")
            content = m.get("content", "")
            kwargs = modality_to_kwargs(m)
            if role == "user":
                messages.append(HumanMessage(content=content, additional_kwargs=kwargs))
            elif role == "assistant":
                messages.append(AIMessage(content=content, additional_kwargs=kwargs))

        trimmed_history = trim_messages(
            messages[-settings.memory_buffer_size:],
            max_tokens=self._memory_token_limit,
            strategy="last",
            token_counter=self._llm,
            start_on="human",
            include_system=False,
        )

        modality_timeline: List[Dict[str, Any]] = []
        for msg in trimmed_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            kwargs = getattr(msg, "additional_kwargs", None) or {}
            modality_timeline.append(
                {
                    "role": role,
                    "content_preview": str(getattr(msg, "content", ""))[:120],
                    **modality_to_kwargs(kwargs),
                }
            )

        remembered: List[Dict[str, Any]] = []
        if session and getattr(session, "uploaded_images", None):
            for rec in session.uploaded_images:
                if hasattr(rec, "model_dump"):
                    remembered.append(rec.model_dump())
                elif isinstance(rec, dict):
                    remembered.append(rec)

        remembered_generated: List[Dict[str, Any]] = []
        if session and getattr(session, "generated_images", None):
            for rec in session.generated_images:
                if hasattr(rec, "model_dump"):
                    remembered_generated.append(rec.model_dump(mode="json"))
                elif isinstance(rec, dict):
                    remembered_generated.append(rec)

        structured_summary = (
            getattr(session, "structured_summary", None) if session else None
        )
        session_metadata: Dict[str, Any] = {}
        if summary or structured_summary:
            session_metadata = {
                "summary": summary,
                "structured_summary": structured_summary,
            }
        if not mongo_ok:
            session_metadata["memory_degraded"] = True
            session_metadata["memory_degraded_reason"] = "mongodb_unreachable"

        return {
            "conversation_history": trimmed_history,
            "message_modality_timeline": modality_timeline,
            "is_session_restart": is_restart,
            "remembered_images": remembered,
            "remembered_generated_images": remembered_generated,
            "session_metadata": session_metadata,
            "memory_degraded": not mongo_ok,
        }

    async def append_uploaded_images(
        self,
        session_id: str,
        user_id: str,
        records: List[Dict[str, Any]],
    ) -> None:
        """Persist new image analyzer records and enforce session cap."""
        if not records:
            return

        session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not session:
            session = ChatSession(session_id=session_id, user_id=user_id)
            await session.insert()

        existing = list(session.uploaded_images or [])
        for raw in records:
            existing.append(UploadedImageRecord(**raw))

        cap = settings.max_remembered_images
        if len(existing) > cap:
            existing = existing[-cap:]

        await session.update(
            {
                "$set": {
                    "uploaded_images": existing,
                    "updated_at": get_ist_now(),
                }
            }
        )

    async def append_generated_images(
        self,
        session_id: str,
        user_id: str,
        records: List[Dict[str, Any]],
    ) -> None:
        """Persist generated diagram metadata (no base64) and enforce session cap."""
        if not records:
            return

        session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not session:
            session = ChatSession(session_id=session_id, user_id=user_id)
            await session.insert()

        existing = list(session.generated_images or [])
        for raw in records:
            existing.append(GeneratedImageRecord(**raw))

        cap = settings.max_remembered_generated_images
        if len(existing) > cap:
            existing = existing[-cap:]

        await session.update(
            {
                "$set": {
                    "generated_images": existing,
                    "updated_at": get_ist_now(),
                }
            }
        )

    async def get_generated_images(self, session_id: str) -> List[Dict[str, Any]]:
        """Return persisted generated image metadata for a session (no base64)."""
        session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not session or not session.generated_images:
            return []

        out: List[Dict[str, Any]] = []
        for rec in session.generated_images:
            if hasattr(rec, "model_dump"):
                out.append(rec.model_dump(mode="json"))
            elif isinstance(rec, dict):
                out.append(rec)
        return out

    async def update_user_preferences(
        self,
        session_id: str,
        user_id: str,
        *,
        response_format: str | None = None,
    ) -> None:
        """
        Persist lightweight user preferences immediately (not waiting for background summarization).

        Stored inside ChatSession.structured_summary.user_preferences (string) for backward compatibility.
        """
        if not response_format:
            return
        response_format = str(response_format).strip().lower()
        if response_format not in {"bullets", "paragraph", "steps"}:
            return

        session = await ChatSession.find_one(ChatSession.session_id == session_id)
        if not session:
            session = ChatSession(session_id=session_id, user_id=user_id)
            try:
                await session.insert()
            except DuplicateKeyError:
                session = await ChatSession.find_one(ChatSession.session_id == session_id)

        structured = getattr(session, "structured_summary", None)
        if structured and hasattr(structured, "model_dump"):
            structured_payload = structured.model_dump(exclude_none=True)
        elif isinstance(structured, dict):
            structured_payload = structured
        else:
            structured_payload = {"summary": session.summary or ""}

        prefs = str(structured_payload.get("user_preferences") or "").strip()
        # Remove older response_format lines if present.
        prefs_lines = [ln for ln in prefs.splitlines() if ln.strip()]
        prefs_lines = [ln for ln in prefs_lines if not ln.lower().startswith("response_format:")]
        prefs_lines.append(f"response_format: {response_format}")
        structured_payload["user_preferences"] = "\n".join(prefs_lines).strip()

        # Validate via StructuredSummary if possible; if not, still store dict.
        try:
            validated = StructuredSummary.model_validate(structured_payload)
            structured_payload = validated.model_dump(exclude_none=True)
        except Exception:
            pass

        await session.update(
            {"$set": {"structured_summary": structured_payload, "updated_at": get_ist_now()}}
        )

    async def get_history(self, session_id: str) -> List[ConversationTurn]:
        """Get history from Redis buffer (for compatibility) with DB fallback."""
        redis_key = f"chat:{session_id}:buffer"
        try:
            raw_msgs = await self._redis.lrange(redis_key, 0, -1)
            if not raw_msgs:
                # Fallback to DB
                session = await ChatSession.find_one(ChatSession.session_id == session_id)
                if session:
                    recent = session.messages[-settings.memory_buffer_size:]
                    return [self._turn_from_mongo(m) for m in recent]
                return []

            return [self._turn_from_buffer(json.loads(rm)) for rm in raw_msgs]
        except Exception as e:
            logger.warning(f"Redis failure in get_history for {session_id}: {e}")
            session = await ChatSession.find_one(ChatSession.session_id == session_id)
            if session:
                recent = session.messages[-settings.memory_buffer_size:]
                return [self._turn_from_mongo(m) for m in recent]
            return []

    @staticmethod
    def _turn_from_buffer(row: Dict[str, Any]) -> ConversationTurn:
        return ConversationTurn(
            role=row.get("role", "user"),
            content=row.get("content", ""),
            has_image_input=bool(row.get("has_image_input")),
            image_input_ids=list(row.get("image_input_ids") or []),
            image_input_urls=list(row.get("image_input_urls") or []),
            has_generated_image=bool(row.get("has_generated_image")),
            generated_image_ids=list(row.get("generated_image_ids") or []),
            generated_image_urls=list(row.get("generated_image_urls") or []),
        )

    @staticmethod
    def _turn_from_mongo(msg: ChatMessage) -> ConversationTurn:
        return ConversationTurn(
            role=msg.role,
            content=msg.text,
            has_image_input=bool(getattr(msg, "has_image_input", False)),
            image_input_ids=list(getattr(msg, "image_input_ids", None) or []),
            image_input_urls=list(getattr(msg, "image_input_urls", None) or []),
            has_generated_image=bool(getattr(msg, "has_generated_image", False)),
            generated_image_ids=list(getattr(msg, "generated_image_ids", None) or []),
            generated_image_urls=list(getattr(msg, "generated_image_urls", None) or []),
        )
