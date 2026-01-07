"""Service for validating agent responses against retrieved documents and user intent."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from state import Document

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of a response validation check."""
    is_valid: bool = Field(description="Whether the response is factually grounded and matches intent.")
    needs_clarification: bool = Field(default=False, description="Whether the user needs to be asked for clarification due to ambiguity.")
    reasoning: str = Field(description="Explanation for the validation result.")
    feedback: Optional[str] = Field(default=None, description="Corrective feedback for the agent if invalid.")
    clarification_question: Optional[str] = Field(default=None, description="The specific question to ask the user if needs_clarification is True.")


class ResponseValidator:
    """Validates agent responses for groundedness and intent alignment."""

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm
        self._validator = llm.with_structured_output(ValidationResult)

    async def validate(
        self,
        query: str,
        response: str,
        documents: List[Document],
        intent_subjects: List[str],
    ) -> ValidationResult:
        """
        Perform groundedness and intent-alignment validation.
        """
        if not documents:
            return ValidationResult(is_valid=True, reasoning="No documents to validate against.")

        doc_context = "\n\n".join([
            f"Doc {i+1} (Subject: {d.get('metadata', {}).get('subject', 'N/A')}):\n{d.get('text', '')}"
            for i, d in enumerate(documents[:5])
        ])

        prompt = f"""You are a strict EDUCATIONAL GUARDIAN. Your job is to verify if an AI Agent's response is CORRECT and ALIGNED with the user's intent.

User Query: {query}
Detected Intent Subjects: {', '.join(intent_subjects)}

Retrieved Documents:
{doc_context}

Agent's Response:
{response}

VERIFICATION TASKS:
1. **Groundedness**: Is the answer supported by the provided documents? 
2. **Intent Alignment**: If the documents contain multiple subjects (e.g., Transformers in AI vs Electrical), did the agent pick the correct one matching '{intent_subjects[0] if intent_subjects else 'General'}'?
3. **Ambiguity Detection (CRITICAL)**: If the documents show multiple distinct and valid interpretations of the query, and the Agent's choice seems like a guess or excludes a likely alternative, set `needs_clarification` to True.
4. **NO EXTERNAL LINKS (MANDATORY)**: Check if the response contains ANY links to external websites (e.g., YouTube, Wikipedia, Khan Academy). If it does, mark `is_valid` as False and provide feedback.

RULES for `needs_clarification`:
- Set to True if the query is significantly ambiguous (e.g., "Transformers") and the retrieved documents have strong evidence for multiple contexts.
- Provide a `clarification_question` that briefly describes the detected contexts and asks the user which one they want (e.g., "I found information on both electrical transformers and neural networks. Which one should I explain?").

RETRY FEEDBACK:
- If `needs_clarification` is False and `is_valid` is False, provide `feedback` to fix the hallucination or remove external links.
"""

        try:
            result = await self._validator.ainvoke(prompt)
            logger.info("Validation result: valid=%s, reasoning=%s", result.is_valid, result.reasoning)
            return result
        except Exception as e:
            logger.error("Response validation failed: %s", e)
            # Default to valid to avoid blocking on technical errors
            return ValidationResult(is_valid=True, reasoning=f"Validation error: {e}")
