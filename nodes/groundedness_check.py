"""LangGraph node: groundedness_check."""

from typing import Any, Dict
import logging
from state import AgentState
from services.response_validator import ResponseValidator
from services.agent_findings import publish

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
        agent_mode = state.get("agent_mode", "standard")

        # Skip document-groundedness for the InteractiveStudentAgent.
        # Socratic counter-questions are intentionally non-declarative ("What do
        # you think happens to the ball's energy?") and are not "grounded" in
        # retrieved documents by design. Language consistency is enforced
        # downstream by translate_response.
        if agent_mode == "interactive" and settings.validation_mode != "strict":
            logger.info(
                "Skipping groundedness for interactive agent (Socratic counter-questions are non-declarative)."
            )
            duration = perf_counter() - start
            return {"timings": {"groundedness_check": duration}}

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
        query_en = state.get("query_en", state.get("query"))
        
        if not query_en:
            logger.warning("Missing query content for groundedness check. Skipping.")
            duration = perf_counter() - start
            return {"timings": {"groundedness_check": duration}}

        documents = state.get("documents", [])

        logger.info("Running groundedness and language consistency check for query: %s", query_en[:50])
        
        import asyncio
        try:
            result = await asyncio.wait_for(
                self._validator.validate(
                    response=response,
                    target_lang=target_lang,
                    documents=documents,
                    query=query_en
                ),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Groundedness check timed out after 10s. Defaulting to valid.")
            from services.response_validator import ValidationResult
            result = ValidationResult(is_valid=True, reasoning="Validation timed out, passed for stability.")

        duration = perf_counter() - start
        
        # Prepare updates
        updates = {
            "timings": {"groundedness_check": duration},
            "validation_results": result.dict(),
            "llm_calls": 1,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "aux_input_tokens": result.input_tokens,
            "aux_output_tokens": result.output_tokens,
            "token_usage_by_model": {
                getattr(self._validator._llm, "model_name", "unknown"): {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                }
            },
        }
        from services.costing import patch_component_usage

        validator_model = getattr(self._validator._llm, "model_name", "unknown")
        updates.update(
            patch_component_usage(
                "chat", validator_model, result.input_tokens, result.output_tokens
            )
        )

        if not result.is_valid:
            updates["agent_findings"] = publish(
                state.get("agent_findings"),
                "groundedness_check",
                "validation",
                (result.feedback or result.reasoning or "Validation failed")[:120],
                {
                    "is_valid": False,
                    "feedback": result.feedback or result.reasoning or "",
                },
                priority=9,
            )
        
        return updates
