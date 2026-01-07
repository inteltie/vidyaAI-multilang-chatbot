"""LangGraph node: groundedness_check."""

from typing import Any, Dict
import logging
from state import AgentState
from services.response_validator import ResponseValidator

logger = logging.getLogger(__name__)


class GroundednessCheckNode:
    """
    Node to validate agent responses against retrieved documents and intent.
    """

    def __init__(self, validator: ResponseValidator):
        self._validator = validator

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        from time import perf_counter
        start = perf_counter()
        
        # Skip validation if mode is disabled
        from config import settings
        if settings.validation_mode == "disabled":
            duration = perf_counter() - start
            return {"timings": {"groundedness_check": duration}}

        # Skip validation if response is a fallback or conversational
        query_type = state.get("query_type")
        response = state.get("response")
        
        # If fast mode, skip conversational queries
        if settings.validation_mode == "fast" and query_type == "conversational":
            duration = perf_counter() - start
            return {"timings": {"groundedness_check": duration}}

        if not response:
            duration = perf_counter() - start
            return {"timings": {"groundedness_check": duration}}

        # Don't loop infinitely - limit to 1 correction
        if state.get("is_correction"):
            logger.info("Skipping validation for correction turn to prevent infinite loops.")
            duration = perf_counter() - start
            return {"timings": {"groundedness_check": duration}}

        docs = state.get("documents", [])
        intent_subjects = state.get("subjects", [])
        query = state.get("query_en", state.get("query"))

        logger.info("Running groundedness check for query: %s", query[:50])
        
        result = await self._validator.validate(
            query=query,
            response=response,
            documents=docs,
            intent_subjects=intent_subjects
        )

        state["validation_results"] = result.dict()

        if result.needs_clarification:
            logger.info("Guardian detected ambiguity. Requesting clarification.")
            state["response"] = result.clarification_question
            state["clarification_message"] = result.clarification_question
            # This will cause _route_after_validation to return "pass" 
            # but we've overridden the response, so it behaves like HITL.
        elif not result.is_valid:
            logger.warning("Response invalid! Feedback: %s", result.feedback)
        else:
            logger.info("Response validated successfully.")

        # Track validation LLM call
        state["llm_calls"] = state.get("llm_calls", 0) + 1

        duration = perf_counter() - start
        
        # Prepare updates
        updates = {
            "timings": {"groundedness_check": duration}
        }
        
        # Only include fields that were potentially modified
        if "validation_results" in locals() or "result" in locals():
            updates["validation_results"] = state.get("validation_results")
            updates["llm_calls"] = 1
            
            if state.get("clarification_message"):
                updates["response"] = state["response"]
                updates["clarification_message"] = state["clarification_message"]
        
        return updates
