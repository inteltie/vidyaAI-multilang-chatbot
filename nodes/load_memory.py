"""LangGraph node: load_memory."""

from __future__ import annotations

from typing import Any, Dict
from time import perf_counter
import re

from services import MemoryService
from state import AgentState


class LoadMemoryNode:
    """LangGraph node: load_memory."""

    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service

    @staticmethod
    def _preferred_language_from_metadata(session_metadata: Dict[str, Any]) -> str | None:
        structured = (session_metadata or {}).get("structured_summary") or {}
        if hasattr(structured, "model_dump"):
            structured = structured.model_dump()
        pref_blob = (structured or {}).get("user_preferences")
        if not pref_blob:
            return None
        pref_text = str(pref_blob).lower()
        if re.search(r"\benglish\b", pref_text):
            return "en"
        return None

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        user_session_id = state["user_session_id"]
        user_id = state["user_id"]
        
        # Single DB/Redis pass to get history and context
        updates = await self._memory_service.load_session_full(user_id, user_session_id)
        session_metadata = updates.get("session_metadata", {})
        preferred_lang = self._preferred_language_from_metadata(session_metadata)
        if preferred_lang:
            updates["preferred_response_language"] = preferred_lang
            updates["language_preference_locked"] = True
        
        duration = perf_counter() - start
        updates["timings"] = {"load_memory": duration}
            
        return updates


