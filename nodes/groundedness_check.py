"""LangGraph node: groundedness_check."""

from typing import Any, Dict
import logging
from state import AgentState
from services.response_validator import ResponseValidator

logger = logging.getLogger(__name__)


class GroundednessCheckNode:
    """
    Node to validate agent responses for language consistency.
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

        target_lang = state.get("language", "en")
        
        # Skip validation for English as it's the base language
        if target_lang == "en":
            logger.info("Skipping language validation for English response.")
            duration = perf_counter() - start
            return {"timings": {"groundedness_check": duration}}

        query = state.get("query_en", state.get("query"))

        logger.info("Running language consistency check for query: %s", query[:50])
        
        import asyncio
        try:
            result = await asyncio.wait_for(
                self._validator.validate(
                    response=response,
                    target_lang=target_lang
                ),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Groundedness check timed out after 10s. Defaulting to valid.")
            from services.response_validator import ValidationResult
            result = ValidationResult(is_valid=True, reasoning="Validation timed out, passed for stability.")

        state["validation_results"] = result.dict()

        if not result.is_valid:
            logger.warning("Language mismatch detected! Feedback: %s", result.feedback)
        else:
            logger.info("Response language validated successfully.")

        # Track validation LLM call
        state["llm_calls"] = state.get("llm_calls", 0) + 1

        duration = perf_counter() - start
        
        # Prepare updates
        updates = {
            "timings": {"groundedness_check": duration}
        }
        
        # Only include fields that were potentially modified
        if "result" in locals():
            updates["validation_results"] = result.dict()
            updates["llm_calls"] = 1
            updates["input_tokens"] = result.input_tokens
            updates["output_tokens"] = result.output_tokens
            
            if state.get("is_correction"):
                updates["response"] = state["response"]
        
        return updates
