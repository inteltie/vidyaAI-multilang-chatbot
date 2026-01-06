"""LangGraph node: groundedness_check."""

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

    async def __call__(self, state: AgentState) -> AgentState:
        from time import perf_counter
        start = perf_counter()
        
        # Skip validation if response is a fallback or conversational
        query_type = state.get("query_type")
        response = state.get("response")
        
        if query_type == "conversational" or not response:
            duration = perf_counter() - start
            timings = state.get("timings") or {}
            timings["groundedness_check"] = duration
            state["timings"] = timings
            return state

        # Don't loop infinitely - limit to 1 correction
        if state.get("is_correction"):
            logger.info("Skipping validation for correction turn to prevent infinite loops.")
            duration = perf_counter() - start
            timings = state.get("timings") or {}
            timings["groundedness_check"] = duration
            state["timings"] = timings
            return state

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

        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["groundedness_check"] = duration
        state["timings"] = timings
        
        return state
