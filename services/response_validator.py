"""Service for validating agent responses against retrieved documents and user intent."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from state import Document

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of a response validation check."""
    is_valid: bool = Field(description="Whether the response matches the expected language.")
    reasoning: str = Field(description="Explanation for the validation result.")
    feedback: Optional[str] = Field(default=None, description="Corrective feedback if language mismatch.")
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)


class ResponseValidator:
    """Validates agent responses for groundedness and intent alignment."""

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm
        self._validator = llm.with_structured_output(ValidationResult, include_raw=True)

    async def validate(
        self,
        response: str,
        target_lang: str,
    ) -> ValidationResult:
        """
        Verify if the response matches the target language.
        """
        prompt = f"""You are a Language Consistency Checker. Your ONLY job is to verify if the AI Agent's response is in the CORRECT language.

Target Language: {target_lang}

Agent's Response:
{response}

VERIFICATION TASK:
1. Is the response ENTIRELY or PREDOMINANTLY in {target_lang}? 
2. If the response contains mixed languages or is in a completely different language (e.g., English instead of Hindi), mark `is_valid` as False.
3. Technical terms or formulas in English are acceptable if they are commonly used.

RETRY FEEDBACK:
- If `is_valid` is False, provide `feedback` like "Translate the response into {target_lang}."
"""

        from config import settings
        try:
            output = await self._validator.ainvoke(prompt, config={"max_tokens": settings.validation_tokens})
            result: ValidationResult = output["parsed"]
            raw_response = output["raw"]
            
            # Log token usage
            usage = getattr(raw_response, "usage_metadata", None) or getattr(raw_response, "response_metadata", {}).get("token_usage", {})
            if usage:
                 i_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                 o_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                 logger.info(
                     "[TOKEN_USAGE] ResponseValidator: input_tokens=%s, output_tokens=%s, total_tokens=%s, model=%s",
                     i_tokens,
                     o_tokens,
                     usage.get("total_tokens") or (i_tokens + o_tokens),
                     self._llm.model_name
                 )
            
            # Populate token counts in result
            if usage:
                result.input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                result.output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0

            logger.info("Validation result: valid=%s, reasoning=%s", result.is_valid, result.reasoning)
            return result
        except Exception as e:
            logger.error("Response validation failed: %s", e)
            # Default to valid to avoid blocking on technical errors
            return ValidationResult(is_valid=True, reasoning=f"Validation error: {e}")
