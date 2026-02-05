"""Student agent node for LangGraph integration."""

import logging
from typing import Any, Dict
from time import perf_counter
from agents import StudentAgent
from state import AgentState

logger = logging.getLogger(__name__)


class StudentAgentNode:
    """LangGraph node that runs the Student agent."""
    
    def __init__(self, student_agent: StudentAgent):
        self._agent = student_agent
    
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Execute Student agent and update state."""
        start = perf_counter()
        
        logger.info("Running Student agent with ReAct reasoning")
        
        # Run the Student agent with a 25s timeout
        import asyncio
        try:
            updates = await asyncio.wait_for(self._agent(state), timeout=25.0)
        except asyncio.TimeoutError:
            logger.warning("Student agent execution timed out after 25s")
            updates = {
                "response": "I'm sorry, but it's taking me longer than usual to process your request. Please try again or ask a simpler question.",
                "is_correction": False
            }
        
        duration = perf_counter() - start
        updates["timings"] = {"student_agent": duration}
        
        return updates


__all__ = ["StudentAgentNode"]
