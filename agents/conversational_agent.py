"""Conversational agent for handling acknowledgments and social interactions."""

import logging
import re
from datetime import datetime, timezone
from collections.abc import Mapping
from langchain_openai import ChatOpenAI
from state import AgentState
from config import settings
from tools.web_search_tool import WebSearchTool
from services.costing import merge_patches, parse_web_search_result, patch_component_usage
from services.prompt_policy import (
    GLOBAL_GUARDRAIL_RULES,
    IDENTITY_BLURB,
    is_identity_or_tech_question,
)

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
            "cool", "nice", "great", "got it", "understood", "hmm", "hm", "ohk", "ohkk",
            "thx", "thanks", "thank you",
        }
        return normalized in ack_tokens

    @staticmethod
    def _looks_like_greeting(text: str) -> bool:
        normalized = ConversationalAgent._normalize_chat_token(text)
        greeting_tokens = {
            "hi", "hello", "hey", "hii", "heyy", "good morning",
            "good evening", "good afternoon", "namaste",
        }
        return normalized in greeting_tokens

    @staticmethod
    def _looks_like_thanks(text: str) -> bool:
        q = (text or "").lower()
        return any(word in q for word in ["thanks", "thank you", "thx"])

    @staticmethod
    def _looks_like_goodbye(text: str) -> bool:
        q = (text or "").lower()
        return any(word in q for word in ["bye", "goodbye", "see you"])

    @staticmethod
    def _looks_like_completion(text: str) -> bool:
        q = (text or "").lower()
        return any(phrase in q for phrase in ["solved", "clear now", "got it", "understood"])

    @staticmethod
    def _looks_like_no_more_questions(text: str) -> bool:
        q = re.sub(r"\s+", " ", (text or "").lower()).strip()
        phrases = (
            "no further questions",
            "i have no further questions",
            "no more questions",
            "i have no more questions",
            "that's all",
            "thats all",
            "that is all",
            "nothing else",
            "no questions",
            "all good now",
            "all clear now",
            "we can stop here",
            "lets stop here",
            "let's stop here",
        )
        return any(phrase in q for phrase in phrases)

    @classmethod
    def _is_informational_conversational(cls, text: str) -> bool:
        if cls._looks_like_greeting(text):
            return False
        if cls._is_acknowledgment_only(text):
            return False
        if cls._looks_like_thanks(text):
            return False
        if cls._looks_like_goodbye(text):
            return False
        if cls._looks_like_completion(text):
            return False
        if cls._looks_like_no_more_questions(text):
            return False
        tokens = cls._normalize_chat_token(text).split()
        filtered_tokens = [tok for tok in tokens if re.search(r"[a-zA-Z0-9]", tok)]
        return len(tokens) >= 4 or ("?" in (text or "") and len(filtered_tokens) >= 1)
    
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
        if state.get("response"):
            return {
                "response": state["response"],
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
        
        # Template-based responses for common social cases with strict cadence.
        if self._looks_like_greeting(raw_query):
            if is_restart:
                updates["response"] = f"{prefix}How can I help you today?"
            else:
                updates["response"] = "How can I help you with your question today?"
        elif self._is_acknowledgment_only(raw_query):
            updates["response"] = f"{prefix}Got it. I am here whenever you need help."
        elif self._looks_like_thanks(raw_query):
            updates["response"] = f"{prefix}I'm glad I could help! Feel free to ask if you have more questions."
        elif self._looks_like_goodbye(raw_query):
            updates["response"] = f"{prefix}Goodbye! Happy learning! 📚"
        elif self._looks_like_completion(raw_query):
            updates["response"] = f"{prefix}Great! I'm here whenever you need help with your studies. Is there anything else you'd like to know?"
        elif self._looks_like_no_more_questions(raw_query):
            updates["response"] = f"{prefix}Great work today. I am here whenever you want to continue."
        elif is_identity_or_tech_question(raw_query):
            # Identity / underlying-technology questions: respond with the branded
            # blurb and never disclose internals or run a web search.
            updates["response"] = IDENTITY_BLURB
        elif self._is_informational_conversational(raw_query):
            # Conversational informational path: web-backed + date-grounded.
            today_iso = datetime.now(timezone.utc).date().isoformat()
            web_tool = WebSearchTool()
            try:
                web_observation, web_i_tokens, web_o_tokens, web_billable = (
                    await web_tool.execute(query=raw_query)
                )
            except Exception as exc:
                logger.error(
                    "WebSearchTool failed for raw_query=%r with error=%s",
                    raw_query,
                    exc,
                )
                web_observation, web_i_tokens, web_o_tokens, web_billable = "", 0, 0, False
            prompt = (
                f"You are Vidya, a friendly educational assistant. "
                f"Today is {today_iso} (UTC). "
                f"Answer the user's informational question concisely and clearly in {target_lang}. "
                f"Use the web observation when relevant, and if information may change quickly, mention the date context briefly.\n\n"
                f"User query: {raw_query}\n\n"
                f"Web observation:\n{web_observation}\n\n"
                "Rules:\n"
                "- Do not add a greeting.\n"
                "- Keep response practical and easy to understand.\n"
                "- Keep answer under 120 words unless user explicitly asks for detail.\n"
                f"{GLOBAL_GUARDRAIL_RULES}"
            )
            max_tokens = min(settings.main_response_tokens, 180)
            resp = await self._llm.ainvoke(prompt, config={"max_tokens": max_tokens})
            updates["response"] = resp.content.strip()
            usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", {}).get("token_usage", {})
            i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            updates["input_tokens"] = i_tokens + web_i_tokens
            updates["output_tokens"] = o_tokens + web_o_tokens
            updates["primary_input_tokens"] = i_tokens
            updates["primary_output_tokens"] = o_tokens
            updates["aux_input_tokens"] = web_i_tokens
            updates["aux_output_tokens"] = web_o_tokens
            web_model = settings.web_search_model_name or "gpt-4o-mini"
            updates.update(
                merge_patches(
                    patch_component_usage("chat", self._llm.model_name, i_tokens, o_tokens),
                    patch_component_usage(
                        "web_search",
                        web_model,
                        web_i_tokens,
                        web_o_tokens,
                        billable_web_call=web_billable,
                    ),
                )
            )
            updates["llm_calls"] = 1
        else:
            updates["response"] = "Please share your question in one line, and I will help."
        
        logger.info("Conversational agent handled query%s", " (session restart)" if is_restart else "")
        return updates
