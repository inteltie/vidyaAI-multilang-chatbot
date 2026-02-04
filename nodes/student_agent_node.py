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
        
        initial_llm_calls = state.get("llm_calls", 0)
        logger.info("Running Student agent with ReAct reasoning")
        
        # Run the Student agent (which internally uses ReAct) with a 25s timeout
        import asyncio
        try:
            new_state = await asyncio.wait_for(self._agent(state), timeout=25.0)
        except asyncio.TimeoutError:
            logger.warning("Student agent execution timed out after 25s")
            # Return partial state or fallback
            new_state = state.copy()
            new_state["response"] = "I'm sorry, but it's taking me longer than usual to process your request. Please try again or ask a simpler question."
            new_state["is_correction"] = False
        
        duration = perf_counter() - start
        
        # Return ONLY the updates to avoid conflicts with parallel RAG node
        # For llm_calls, we return the DELTA (calls made in THIS node)
        calls_made = new_state.get("llm_calls", 0) - initial_llm_calls
        
        return {
            "response": new_state.get("response", ""),
            "citations": new_state.get("citations", []),
            "llm_calls": max(0, calls_made),
            "final_language": new_state.get("final_language"),
            "is_correction": new_state.get("is_correction", False),
            "timings": {"student_agent": duration}
        }


__all__ = ["StudentAgentNode"]
