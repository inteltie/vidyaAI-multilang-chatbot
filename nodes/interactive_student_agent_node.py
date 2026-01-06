"""Interactive Student agent node for LangGraph integration."""

import logging
from time import perf_counter
from agents import InteractiveStudentAgent
from state import AgentState

logger = logging.getLogger(__name__)


class InteractiveStudentAgentNode:
    """LangGraph node that runs the Interactive Student agent."""
    
    def __init__(self, interactive_student_agent: InteractiveStudentAgent):
        self._agent = interactive_student_agent
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Execute Interactive Student agent and update state."""
        start = perf_counter()
        
        logger.info("Running Interactive Student agent with step-by-step reasoning")
        
        # Run the agent
        state = await self._agent(state)
        
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["interactive_student_agent"] = duration
        state["timings"] = timings
        
        logger.info("Interactive Student agent completed in %.2fs", duration)
        
        return state


__all__ = ["InteractiveStudentAgentNode"]
