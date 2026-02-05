"""ReAct agent node for LangGraph integration."""

import logging
from time import perf_counter
from agents import ReActAgent
from state import AgentState

logger = logging.getLogger(__name__)


class ReActAgentNode:
    """LangGraph node that runs the ReAct reasoning agent."""
    
    def __init__(self, react_agent: ReActAgent):
        self._agent = react_agent
    
    async def __call__(self, state: AgentState) -> dict:
        """Execute ReAct agent and update state."""
        start = perf_counter()
        
        query = state["query_en"]
        history = state.get("conversation_history", [])
        
        logger.info("Running ReAct agent for query: %s", query[:100])
        
        # Run the ReAct loop
        result = await self._agent.run(query, history)
        
        duration = perf_counter() - start
        
        return {
            "response": result["answer"],
            "reasoning_chain": result.get("reasoning_chain", []),
            "react_iterations": result.get("iterations", 0),
            "llm_calls": result.get("iterations", 0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "timings": {"react_agent": duration}
        }


__all__ = ["ReActAgentNode"]
