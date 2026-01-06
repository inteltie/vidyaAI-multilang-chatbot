"""Node wrapper for conversational agent."""

import logging
from time import perf_counter
from agents import ConversationalAgent
from state import AgentState

logger = logging.getLogger(__name__)


class ConversationalAgentNode:
    """LangGraph node: conversational_agent."""
    
    def __init__(self, agent: ConversationalAgent) -> None:
        self._agent = agent
    
    async def __call__(self, state: AgentState) -> AgentState:
        start = perf_counter()
        
        state = await self._agent(state)
        
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["conversational_agent"] = duration
        state["timings"] = timings
        
        return state


__all__ = ["ConversationalAgentNode"]
