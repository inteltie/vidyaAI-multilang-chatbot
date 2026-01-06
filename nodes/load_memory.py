"""LangGraph node: load_memory."""

from __future__ import annotations

from time import perf_counter

from services import MemoryService
from state import AgentState


class LoadMemoryNode:
    """LangGraph node: load_memory."""

    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service

    async def __call__(self, state: AgentState) -> AgentState:
        start = perf_counter()
        user_session_id = state["user_session_id"]
        user_id = state["user_id"]
        
        # Load session, buffer, and summary
        await self._memory_service.ensure_session(user_id, user_session_id)
        
        # Get structured context (summary + token-trimmed messages)
        summary, messages = await self._memory_service.get_context(user_session_id)
        
        # Update state
        state["conversation_history"] = messages
        if summary:
            # Metadata merge logic in ParseSessionContextNode will handle this
            if "session_metadata" not in state:
                state["session_metadata"] = {}
            state["session_metadata"]["summary"] = summary
            
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["load_memory"] = duration
        state["timings"] = timings
        return state


