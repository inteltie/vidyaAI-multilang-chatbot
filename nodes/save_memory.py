"""LangGraph node: save_memory."""

from __future__ import annotations
import asyncio
from time import perf_counter
from typing import Any, Dict, List

from services import MemoryService
from state import AgentState, ConversationTurn


class SaveMemoryNode:
    """LangGraph node: save_memory."""

    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service
    
    async def _save_messages_sequentially(self, session_id: str, user_id: str, query: str, response: str):
        """Save user message first, then AI message to ensure correct timestamp order."""
        await self._memory_service.background_save_message(session_id, user_id, "user", query)
        await self._memory_service.background_save_message(session_id, user_id, "assistant", response)

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        user_session_id = state["user_session_id"]
        user_id = state["user_id"]
        query = state["query"]
        response = state.get("response", "")

        # 1. Update Redis buffer (await - fast)
        await self._memory_service.add_message(user_session_id, "user", query)
        await self._memory_service.add_message(user_session_id, "assistant", response)

        # 2. Save to MongoDB sequentially (background)
        asyncio.create_task(self._save_messages_sequentially(user_session_id, user_id, query, response))

        duration = perf_counter() - start
        return {
            "timings": {"save_memory": duration}
        }


