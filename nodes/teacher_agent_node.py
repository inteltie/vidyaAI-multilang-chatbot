"""Teacher agent node for LangGraph integration."""

import logging
from time import perf_counter
from agents import TeacherAgent
from state import AgentState

logger = logging.getLogger(__name__)


class TeacherAgentNode:
    """LangGraph node that runs the Teacher agent."""
    
    def __init__(self, teacher_agent: TeacherAgent):
        self._agent = teacher_agent
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Execute Teacher agent and update state."""
        start = perf_counter()
        
        logger.info("Running Teacher agent with ReAct reasoning")
        
        # Run the Teacher agent (which internally uses ReAct)
        state = await self._agent(state)
        
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["teacher_agent"] = duration
        state["timings"] = timings
        
        logger.info("Teacher agent completed in %.2fs", duration)
        
        return state


__all__ = ["TeacherAgentNode"]
