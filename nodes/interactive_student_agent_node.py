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
        
        initial_llm_calls = state.get("llm_calls", 0)
        logger.info("Running Interactive Student agent with step-by-step reasoning")
        
        # Run the agent
        new_state = await self._agent(state)
        
        duration = perf_counter() - start
        
        # Return ONLY the updates to avoid conflicts with parallel RAG node
        calls_made = new_state.get("llm_calls", 0) - initial_llm_calls
        
        return {
            "response": new_state.get("response", ""),
            "citations": new_state.get("citations", []),
            "llm_calls": max(0, calls_made),
            "final_language": new_state.get("final_language"),
            "is_correction": new_state.get("is_correction", False),
            "timings": {"interactive_student_agent": duration}
        }


__all__ = ["InteractiveStudentAgentNode"]
