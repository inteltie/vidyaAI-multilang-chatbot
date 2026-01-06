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
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Execute ReAct agent and update state."""
        start = perf_counter()
        
        query = state["query_en"]
        history = state.get("conversation_history", [])
        
        logger.info("Running ReAct agent for query: %s", query[:100])
        
        # Run the ReAct loop
        result = await self._agent.run(query, history)
        
        # Update state with results
        state["response"] = result["answer"]
        state["reasoning_chain"] = result.get("reasoning_chain", [])
        state["react_iterations"] = result.get("iterations", 0)
        
        # Track LLM calls (each iteration uses 1 LLM call for reasoning)
        state["llm_calls"] = state.get("llm_calls", 0) + result.get("iterations", 0)
        
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["react_agent"] = duration
        state["timings"] = timings
        
        logger.info(
            "ReAct agent completed in %.2fs with %d iterations",
            duration,
            result.get("iterations", 0),
        )
        
        return state


__all__ = ["ReActAgentNode"]
