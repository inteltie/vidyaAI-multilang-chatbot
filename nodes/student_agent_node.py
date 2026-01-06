"""Student agent node for LangGraph integration."""

import logging
from time import perf_counter
from agents import StudentAgent
from state import AgentState

logger = logging.getLogger(__name__)


class StudentAgentNode:
    """LangGraph node that runs the Student agent."""
    
    def __init__(self, student_agent: StudentAgent):
        self._agent = student_agent
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Execute Student agent and update state."""
        start = perf_counter()
        
        logger.info("Running Student agent with ReAct reasoning")
        
        # Run the Student agent (which internally uses ReAct)
        state = await self._agent(state)
        
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["student_agent"] = duration
        state["timings"] = timings
        
        logger.info("Student agent completed in %.2fs", duration)
        
        return state


__all__ = ["StudentAgentNode"]
