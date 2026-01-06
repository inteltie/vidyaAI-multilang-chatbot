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
    
    async def __call__(self, state: AgentState) -> AgentState:
        start = perf_counter()
        
        state = await self._agent(state)
        
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["general_agent"] = duration
        state["timings"] = timings
        
        return state


__all__ = ["GeneralAgentNode"]
