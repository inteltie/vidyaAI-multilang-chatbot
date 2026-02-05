"""Interactive Student agent node for LangGraph integration."""

import logging
from typing import Any, Dict
from time import perf_counter
from agents import InteractiveStudentAgent
from state import AgentState

logger = logging.getLogger(__name__)


class InteractiveStudentAgentNode:
    """LangGraph node that runs the Interactive Student agent."""
    
    def __init__(self, interactive_student_agent: InteractiveStudentAgent):
        self._agent = interactive_student_agent
    
    async def __call__(self, state: AgentState) -> dict:
        """Execute Interactive Student agent and update state."""
        start = perf_counter()
        
        logger.info("Running Interactive Student agent with step-by-step reasoning")
        
        # Run the agent
        updates = await self._agent(state)
        
        duration = perf_counter() - start
        updates["timings"] = {"interactive_student_agent": duration}
        
        return updates


__all__ = ["InteractiveStudentAgentNode"]
