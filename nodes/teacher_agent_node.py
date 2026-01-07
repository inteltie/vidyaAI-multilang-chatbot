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
        
        initial_llm_calls = state.get("llm_calls", 0)
        logger.info("Running Teacher agent with ReAct reasoning")
        
        # Run the Teacher agent (which internally uses ReAct)
        new_state = await self._agent(state)
        
        duration = perf_counter() - start
        
        # Return ONLY the updates to avoid conflicts with parallel RAG node
        calls_made = new_state.get("llm_calls", 0) - initial_llm_calls
        
        return {
            "response": new_state.get("response", ""),
            "citations": new_state.get("citations", []),
            "llm_calls": max(0, calls_made),
            "final_language": new_state.get("final_language"),
            "is_translated": new_state.get("is_translated", False),
            "timings": {"teacher_agent": duration}
        }


__all__ = ["TeacherAgentNode"]
