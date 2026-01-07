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
        
        initial_llm_calls = state.get("llm_calls", 0)
        new_state = await self._agent(state)
        
        duration = perf_counter() - start
        calls_made = new_state.get("llm_calls", 0) - initial_llm_calls
        
        return {
            "response": new_state.get("response", ""),
            "llm_calls": max(0, calls_made),
            "timings": {"general_agent": duration}
        }


__all__ = ["GeneralAgentNode"]
