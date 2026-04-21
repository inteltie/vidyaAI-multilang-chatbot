"""Conversational agent for handling acknowledgments and social interactions."""

import logging
import re
from collections.abc import Mapping
from langchain_openai import ChatOpenAI
from state import AgentState
from config import settings

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """Handles acknowledgments, greetings, and social interactions."""
    
    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    @staticmethod
    def _extract_known_name(session_metadata: dict, history) -> str | None:
        structured = (session_metadata or {}).get("structured_summary")
        entities = {}
        if structured:
            if isinstance(structured, Mapping):
                entities = dict(structured.get("key_entities") or {})
            elif hasattr(structured, "key_entities"):
                key_entities = getattr(structured, "key_entities")
                entities = key_entities.model_dump() if hasattr(key_entities, "model_dump") else (key_entities or {})

        name = entities.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

        # Fallback extraction from recent user messages.
        for msg in reversed(history or []):
            role = getattr(msg, "type", None)
            content = getattr(msg, "content", "")
            if role != "human":
                continue
            match = re.search(r"\b(?:i am|i'm|my name is)\s+([a-zA-Z][a-zA-Z\s]{1,30})", str(content), re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().split(",")[0].strip()
                if candidate:
                    return candidate.title()
        return None

    @staticmethod
    def _extract_known_class_level(session_metadata: dict, history) -> str | None:
        structured = (session_metadata or {}).get("structured_summary")
        entities = {}
        if structured:
            if isinstance(structured, Mapping):
                entities = dict(structured.get("key_entities") or {})
            elif hasattr(structured, "key_entities"):
                key_entities = getattr(structured, "key_entities")
                entities = key_entities.model_dump() if hasattr(key_entities, "model_dump") else (key_entities or {})

        for key in ("class_level",):
            value = entities.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Fallback to topic text and recent user history.
        topic_blob = entities.get("topic")
        if isinstance(topic_blob, str):
            match = re.search(r"\bclass\s*(\d{1,2})\b", topic_blob.lower())
            if match:
                return f"Class {match.group(1)}"

        for msg in reversed(history or []):
            role = getattr(msg, "type", None)
            content = str(getattr(msg, "content", ""))
            if role != "human":
                continue
            match = re.search(r"\bclass\s*(\d{1,2})\b", content.lower())
            if match:
                return f"Class {match.group(1)}"
        return None

    @staticmethod
    def _normalize_chat_token(text: str) -> str:
        cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"([a-z])\1{2,}", r"\1\1", cleaned)
        return cleaned

    @classmethod
    def _is_acknowledgment_only(cls, text: str) -> bool:
        normalized = cls._normalize_chat_token(text)
        if not normalized:
            return False
        if "?" in (text or ""):
            return False
        ack_tokens = {
            "ok", "okay", "alright", "sure", "fine", "k", "yep", "yes", "no",
            "cool", "nice", "great", "got it", "understood", "hmm", "hm", "ohk",
            "thx", "thanks", "thank you",
        }
        return normalized in ack_tokens
    
    async def __call__(self, state: AgentState) -> dict:
        """Generate friendly conversational response."""
        raw_query = state["query"]
        query = raw_query.lower()

        # Deterministic clarification path from AnalyzeQueryNode
        if state.get("needs_context") and state.get("clarification_message"):
            return {
                "response": state["clarification_message"],
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        memory_intent = state.get("memory_intent")
        if memory_intent in {"name", "class_level"}:
            history = state.get("conversation_history", [])
            session_metadata = state.get("session_metadata", {})

            if memory_intent == "name":
                known_name = self._extract_known_name(session_metadata, history)
                if known_name:
                    return {
                        "response": f"Your name is {known_name}.",
                        "llm_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                return {
                    "response": "I do not have your name in this chat yet. Please tell me your name once, and I will remember it for this session.",
                    "llm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

            known_class = self._extract_known_class_level(session_metadata, history)
            if known_class:
                return {
                    "response": f"You mentioned you are in {known_class}.",
                    "llm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            return {
                "response": "I do not have your class level in this chat yet. Please tell me your class, for example 'I am in class 10'.",
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        
        # Detection of Session Restart
        is_restart = state.get("is_session_restart", False)
        prefix = "Welcome back! " if is_restart else ""
        
        updates = {
            "response": "",
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0
        }
        target_lang = state.get("language", "en")
        if state.get("language_preference_locked") and state.get("preferred_response_language"):
            target_lang = state.get("preferred_response_language")
        
        # Template-based responses for common cases
        if self._is_acknowledgment_only(raw_query):
            updates["response"] = f"{prefix}Got it. I am here whenever you need help."
        elif any(word in query for word in ["thanks", "thank you", "thx"]):
            updates["response"] = f"{prefix}I'm glad I could help! Feel free to ask if you have more questions."
        elif any(word in query for word in ["bye", "goodbye", "see you"]):
            updates["response"] = f"{prefix}Goodbye! Happy learning! 📚"
        elif any(phrase in query for phrase in ["solved", "clear now", "got it", "understood"]):
            updates["response"] = f"{prefix}Great! I'm here whenever you need help with your studies. Is there anything else you'd like to know?"
        else:
            # Use LLM with history and summary for personalized responses
            history = state.get("conversation_history", [])
            session_metadata = state.get("session_metadata", {})
            summary = session_metadata.get("summary", "")
            structured_summary = session_metadata.get("structured_summary")
            structured_summary_data = {}
            if structured_summary:
                if isinstance(structured_summary, Mapping):
                    structured_summary_data = dict(structured_summary)
                elif hasattr(structured_summary, "model_dump"):
                    structured_summary_data = structured_summary.model_dump()
                else:
                    structured_summary_data = {
                        "key_entities": getattr(structured_summary, "key_entities", {}),
                    }

            key_entities = structured_summary_data.get("key_entities") or {}
            if hasattr(key_entities, "model_dump"):
                key_entities = key_entities.model_dump()
            elif not isinstance(key_entities, Mapping):
                key_entities = {}

            topic = key_entities.get("topic", "your studies")
            
            # Aggressive Truncation: Keep only last 4 messages for conversational context
            recent_history = history[-4:]
            history_text = ""
            for m in recent_history: 
                msg_type = getattr(m, "type", "")
                msg_content = getattr(m, "content", "")
                role = "STUDENT" if msg_type == "human" else "VIDYA"
                history_text += f"{role}: {msg_content}\n"
            
            # Log history tokens
            try:
                history_tokens = self._llm.get_num_tokens_from_messages(recent_history)
                logger.info("[TOKEN_USAGE] Context: chat_history_tokens=%d", history_tokens)
            except Exception as e:
                logger.debug("Failed to calculate history tokens: %s", e)

            has_history = len(history) > 0
            # Different prompts for fresh vs mid-conversation greetings
            if not has_history:
                # Fresh conversation - simple welcome
                prompt = (
                    f"You are Vidya, a friendly educational assistant. "
                    f"The user just greeted you for the first time. "
                    f"Respond in **{target_lang}** with a warm, brief welcome.\n\n"
                    f"User Greeting: {state['query']}\n\n"
                    f"Rules:\n"
                    f"- Be warm and welcoming. Use emojis (👋, 📚).\n"
                    f"- Introduce yourself briefly as Vidya, their learning companion.\n"
                    f"- Ask how you can help them today.\n"
                    f"- Keep it brief and inviting (<60 tokens).\n"
                    f"- DO NOT mention any previous topics or discussions.")
            else:
                # Format memory context from structured summary
                memory_context = ""
                if structured_summary_data:
                    memory_context = (
                        f"Student Name: {key_entities.get('name', 'User')}\n"
                        f"Current Topic: {key_entities.get('topic', 'Not set')}\n"
                        f"Identified Gaps: {key_entities.get('identified_gaps', 'None')}\n"
                    )
                elif summary:
                    memory_context = f"Overview: {summary}\n"

                # Mid-conversation greeting - reconnect pattern
                prompt = (
                    f"You are Vidya, a friendly educational assistant. "
                    f"The user is greeting you mid-conversation. "
                    f"Respond in **{target_lang}** using a natural reconnect pattern:\n\n"
                    f"1. Greet the user warmly.\n"
                    f"2. Optionally, acknowledge they are returning to the conversation, OR briefly mention the ongoing topic if relevant.\n"
                    f"3. Ask how you can help them right now, or if they want to continue.\n\n"
                    f"Memory Context:\n{memory_context}\n"
                    f"Recent History Snippet:\n{history_text}\n"
                    f"User Greeting: {state['query']}\n\n"
                    f"Rules:\n"
                    f"- IMPORTANT: Vary your phrasing! DO NOT always say 'We were just discussing...'. "
                    f"Be natural. (e.g., 'Welcome back!', 'Hi again! Still working on {topic if structured_summary_data else 'your topics'}?').\n"
                    f"- Use emojis (👋, 📚).\n"
                    f"- NEVER say generic 'I see you are greeting me'.\n"
                    f"- Keep it brief and inviting (<60 tokens).")
            max_tokens = settings.main_response_tokens
            if len(self._normalize_chat_token(raw_query).split()) <= 3:
                max_tokens = min(max_tokens, 120)
            resp = await self._llm.ainvoke(prompt, config={"max_tokens": max_tokens})
            updates["response"] = resp.content.strip()
            
            # Log token usage
            usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", {}).get("token_usage", {})
            if usage:
                i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                updates["input_tokens"] = i_tokens
                updates["output_tokens"] = o_tokens
                
                # Separate primary LLM (Mini) tokens from total/aux (Nano) for breakdown tracking
                updates["primary_input_tokens"] = i_tokens
                updates["primary_output_tokens"] = o_tokens
                logger.info(
                    "[TOKEN_USAGE] ConversationalAgent: input_tokens=%s, output_tokens=%s, total_tokens=%s, model=%s",
                    i_tokens,
                    o_tokens,
                    usage.get("total_tokens"),
                    self._llm.model_name
                )
            
            updates["llm_calls"] = 1
        
        logger.info("Conversational agent handled query%s", " (session restart)" if is_restart else "")
        return updates
