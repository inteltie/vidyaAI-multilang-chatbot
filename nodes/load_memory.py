"""LangGraph node: load_memory."""

from __future__ import annotations

from typing import Any, Dict
from time import perf_counter

from services import MemoryService
from state import AgentState


class LoadMemoryNode:
    """LangGraph node: load_memory."""

    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        user_session_id = state["user_session_id"]
        user_id = state["user_id"]
        
        # Single DB/Redis pass to get history and context
        updates = await self._memory_service.load_session_full(user_id, user_session_id)
        
        duration = perf_counter() - start
        updates["timings"] = {"load_memory": duration}
            
        return updates


