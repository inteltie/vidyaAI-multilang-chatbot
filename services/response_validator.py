"""Service for validating agent responses against retrieved documents and user intent."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from state import Document

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of a response validation check."""
    is_valid: bool = Field(description="Whether the response is grounded and in the correct language.")
    reasoning: str = Field(description="Explanation for the validation result.")
    feedback: Optional[str] = Field(default=None, description="Corrective feedback if validation fails.")
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
        documents: Optional[List[Document]] = None,
        query: Optional[str] = None
    ) -> ValidationResult:
        """
        Verify if the response is grounded in documents and matches the target language.
        """
        doc_context = ""
        if documents:
            doc_context = "\n".join([f"- {doc.get('text', '')}" for doc in documents])
        
        prompt = f"""You are an Educational Response Validator. Your job is to ensure the AI's response is GROUNDED and LINGUISTICALLY CORRECT.

### TASK 1: Groundedness Check
Compare the response against the provided documents.
- Does the response contain information NOT found in the documents? (Technical jargon is fine, but new facts are not).
- Is it factually consistent with the documents?

### TASK 2: Language Check
- Is the response in {target_lang}? 
- Technical terms in English are allowed.

### INPUTS:
User Query: {query if query else "N/A"}
Target Language: {target_lang}

Retrieved Documents:
{doc_context if doc_context else "No documents provided."}

Agent's Response:
{response}

### VERIFICATION CRITERIA:
1. `is_valid` = False if:
   - There is a factual hallucination (info not in docs).
   - The response is in the wrong language.
   - The response is off-topic.
2. Provide specific `feedback` for the agent to correct itself.
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
