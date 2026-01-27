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
        updates = {}
        
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
                 resp = await self._classifier._llm.ainvoke(summary_prompt)
                 final_query = resp.content.strip()
                 logger.info("Summarized query: %s", final_query[:100])
             elif len(query) > 2000:
                 logger.info("Query between 2000-5000 chars. Using as-is (no summarization needed for model context window)")
                 final_query = query
        except Exception as e:
            logger.error("Query summarization failed: %s", e)

        # Phase 6: Speculative Parallelism
        import asyncio
        
        # Start classification task
        classifier_task = self._classifier.analyze(final_query, history)
        
        # Start speculative RAG only if it looks like an educational query (heuristics)
        # Avoid speculative RAG for very short queries that are likely conversational
        retrieval_task = None
        is_likely_educational = len(final_query.split()) > 2 or any(kw in final_query.lower() for kw in ["how", "what", "why", "explain", "describe", "define"])
        
        if self._retriever and is_likely_educational:
            logger.info("Triggering speculative RAG for query: %s", final_query[:50])
            from models import QueryIntent
            retrieval_task = self._retriever.retrieve(
                query_en=final_query,
                filters=state.get("request_filters"),
                intent=state.get("intent", QueryIntent.CONCEPT_EXPLANATION)
            )
        
        if retrieval_task:
            result, speculative_docs = await asyncio.gather(classifier_task, retrieval_task)
            # We'll store these in state for RetrieveDocumentsNode to pick up if valid
            updates["speculative_documents"] = speculative_docs
        else:
            result = await classifier_task
        
        # Prepare updates
        updates.update({
            "query_en": result.translated_query,
            "query_type": result.query_type,
            "subjects": result.subjects,
            "llm_calls": 1,
        })
        
        # Update language if user specifically switched it
        if result.response_language:
            updates["language"] = result.response_language
            updates["detected_language"] = result.response_language
        
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
