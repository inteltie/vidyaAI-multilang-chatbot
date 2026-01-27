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
        
        # Concurrently load session and context
        import asyncio
        ensure_task = self._memory_service.ensure_session(user_id, user_session_id)
        context_task = self._memory_service.get_context(user_session_id)
        
        results = await asyncio.gather(ensure_task, context_task)
        
        # Unpack results
        _, _, _, is_restart = results[0]
        summary, messages = results[1]
        
        duration = perf_counter() - start
        
        # Prepare updates
        updates = {
            "conversation_history": messages,
            "is_session_restart": is_restart,
            "timings": {"load_memory": duration}
        }
        
        if summary:
            # Metadata merge logic in ParseSessionContextNode will handle this
            updates["session_metadata"] = {"summary": summary}
            
        return updates


