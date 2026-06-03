"""LangGraph node: translate_response."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Dict

from agents.translation_agent import TranslationAgent
from state import AgentState

logger = logging.getLogger(__name__)


class TranslateResponseNode:
    """LangGraph node: translate_response."""

    def __init__(self, translation_agent: TranslationAgent) -> None:
        self._agent = translation_agent

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        updates = await self._agent(state)
        duration = perf_counter() - start
        timings = dict(updates.get("timings") or {})
        timings["translate_response"] = duration
        updates["timings"] = timings
        return updates
