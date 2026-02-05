"""LangGraph node: analyze_query."""

from __future__ import annotations

import logging
from typing import Any, Dict
from time import perf_counter

from services.query_classifier import QueryClassifier, QueryClassification
from state import AgentState

logger = logging.getLogger(__name__)


class AnalyzeQueryNode:
    """
    LangGraph node: analyze_query.
    
    Combines language detection, translation, and intent classification
    into a single step/LLM call.
    """

    def __init__(self, classifier: QueryClassifier, retriever: RetrieverService | None = None) -> None:
        self._classifier = classifier
        self._retriever = retriever

    async def __call__(self, state: AgentState) -> dict:
        start = perf_counter()
        updates = {
            "input_tokens": 0,
            "output_tokens": 0
        }
        
        query = state["query"]
        history = state.get("conversation_history", [])
        
        # Phase 5: Query Summarization for large inputs
        final_query = query
        try:
             if len(query) > 5000:
                 logger.info("Query exceeds 5000 chars. Summarizing for efficiency...")
                 summary_prompt = (
                     "Summarize the following user request into a concise search query, "
                     "preserving all technical constraints and educational context:\n\n"
                     f"{query}"
                 )
                 from config import settings
                 resp = await self._classifier._llm.ainvoke(summary_prompt, config={"max_tokens": settings.query_analysis_tokens})
                 
                 # Track tokens for summarization
                 usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", {}).get("token_usage", {})
                 if usage:
                     updates["input_tokens"] += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                     updates["output_tokens"] += usage.get("output_tokens") or usage.get("completion_tokens") or 0

                 final_query = resp.content.strip()
                 logger.info("Summarized query: %s", final_query[:100])
             elif len(query) > 2000:
                 logger.info("Query between 2000-5000 chars. Using as-is (no summarization needed for model context window)")
                 final_query = query
        except Exception as e:
            logger.error("Query summarization failed: %s", e)

        # Phase 6: Sequential Analysis (Speculative RAG disabled for stability)
        import asyncio
        try:
            # Classification task with 15s timeout
            result = await asyncio.wait_for(self._classifier.analyze(final_query, history), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("Query classification timed out after 15s")
            from services.query_classifier import QueryClassification
            result = QueryClassification(
                query_type="curriculum_specific",
                translated_query=final_query,
                confidence=0.0,
                reasoning="Fallback due to timeout",
                subjects=["General"]
            )
        except Exception as e:
            logger.error("Query classification failed: %s", e)
            from services.query_classifier import QueryClassification
            result = QueryClassification(
                query_type="curriculum_specific",
                translated_query=final_query,
                confidence=0.0,
                reasoning=f"Fallback due to error: {e}"
            )
        
        # Prepare updates
        updates.update({
            "query_en": result.translated_query,
            "query_type": result.query_type,
            "subjects": result.subjects,
            "llm_calls": 1,
            "input_tokens": updates["input_tokens"] + result.input_tokens,
            "output_tokens": updates["output_tokens"] + result.output_tokens,
        })
        
        
        # Language is ALWAYS from the request - no AI overrides
        # detected_language is used internally to ensure agents work in English
        updates["detected_language"] = state.get("language", "en")
        
        # Session metadata extraction
        if result.query_type != "conversational":
            current_meta = state.get("session_metadata", {})
            new_meta = current_meta.copy()
                
            if not new_meta.get("class_level") and result.class_level:
                new_meta["class_level"] = result.class_level
            if not new_meta.get("subject") and result.extracted_subject:
                new_meta["subject"] = result.extracted_subject
            if not new_meta.get("topics") and result.chapter:
                new_meta["topics"] = result.chapter
            if not new_meta.get("lecture_id") and result.lecture_id:
                new_meta["lecture_id"] = result.lecture_id
            
            updates["session_metadata"] = new_meta
            
        # If we had a heuristic match, mark that we skipped the LLM
        if result.confidence >= 1.0 and "Matched" in result.reasoning:
            updates["llm_calls"] = 0
        
        duration = perf_counter() - start
        updates["timings"] = {"analyze_query": duration}
        
        return updates
