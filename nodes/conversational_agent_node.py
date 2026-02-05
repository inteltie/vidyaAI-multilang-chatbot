"""Node wrapper for conversational agent."""

import logging
from typing import Any, Dict
from time import perf_counter
from agents import ConversationalAgent
from state import AgentState

logger = logging.getLogger(__name__)


class ConversationalAgentNode:
    """LangGraph node: conversational_agent."""
    
    def __init__(self, agent: ConversationalAgent) -> None:
        self._agent = agent
    
    async def __call__(self, state: AgentState) -> dict:
        start = perf_counter()
        
        # Run agent - now returns a dict of updates
        updates = await self._agent(state)
        
        duration = perf_counter() - start
        updates["timings"] = {"conversational_agent": duration}
        
        return updates


__all__ = ["ConversationalAgentNode"]
