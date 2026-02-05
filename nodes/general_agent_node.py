"""Node wrapper for general knowledge agent."""

import logging
from time import perf_counter
from agents import GeneralAgent
from state import AgentState

logger = logging.getLogger(__name__)


class GeneralAgentNode:
    """LangGraph node: general_agent."""
    
    def __init__(self, agent: GeneralAgent) -> None:
        self._agent = agent
    
    async def __call__(self, state: AgentState) -> dict:
        start = perf_counter()
        
        # Run agent - now returns a dict of updates
        updates = await self._agent(state)
        
        duration = perf_counter() - start
        updates["timings"] = {"general_agent": duration}
        
        return updates


__all__ = ["GeneralAgentNode"]
