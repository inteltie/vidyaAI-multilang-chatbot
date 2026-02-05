"""Teacher agent node for LangGraph integration."""

import logging
from typing import Any, Dict
from time import perf_counter
from agents import TeacherAgent
from state import AgentState

logger = logging.getLogger(__name__)


class TeacherAgentNode:
    """LangGraph node that runs the Teacher agent."""
    
    def __init__(self, teacher_agent: TeacherAgent):
        self._agent = teacher_agent
    
    async def __call__(self, state: AgentState) -> dict:
        """Execute Teacher agent and update state."""
        start = perf_counter()
        
        logger.info("Running Teacher agent with ReAct reasoning")
        
        # Run the Teacher agent (which internally uses ReAct)
        updates = await self._agent(state)
        
        duration = perf_counter() - start
        updates["timings"] = {"teacher_agent": duration}
        
        return updates


__all__ = ["TeacherAgentNode"]
