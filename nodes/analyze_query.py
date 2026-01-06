"""LangGraph node: analyze_query."""

from __future__ import annotations

import logging
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

    def __init__(self, classifier: QueryClassifier, retriever=None) -> None:
        self._classifier = classifier
        self._retriever = retriever

    async def __call__(self, state: AgentState) -> AgentState:
        start = perf_counter()
        
        query = state["query"]
        history = state.get("conversation_history", [])
        
        # Phase 5: Query Summarization for large inputs (> 512 tokens)
        final_query = query
        try:
             # Rough estimation if token_counter not easily accessible here
             # Standard: ~4 chars per token. 512 tokens ~= 2048 chars
             if len(query) > 2000:
                 logger.info("Query exceeds threshold. Summarizing for efficiency...")
                 summary_prompt = (
                     "Summarize the following user request into a concise search query, "
                     "preserving all technical constraints and educational context:\n\n"
                     f"{query}"
                 )
                 # We use the classifier's LLM or just a generic one
                 resp = await self._classifier._llm.ainvoke(summary_prompt)
                 final_query = resp.content.strip()
                 logger.info("Summarized query: %s", final_query[:100])
        except Exception as e:
            logger.error("Query summarization failed: %s", e)

        # Perform combined analysis using potentially summarized query
        result: QueryClassification = await self._classifier.analyze(final_query, history)
        
        # Update state
        state["query_en"] = result.translated_query
        state["query_type"] = result.query_type
        state["subjects"] = result.subjects
        
        # Proactive RAG (Optimization for speed)
        state["prefilled_observations"] = []
        if result.query_type == "curriculum_specific" and self._retriever:
            try:
                from models import QueryIntent
                from tools import RetrievalTool
                
                logger.info("Performing proactive RAG fetch for query: %s", result.translated_query[:50])
                # Note: We use the request_filters if available in state
                docs = await self._retriever.retrieve(
                    query_en=result.translated_query,
                    filters=state.get("request_filters"),
                    intent=QueryIntent.CONCEPT_EXPLANATION
                )
                if docs:
                    obs_text = RetrievalTool.format_documents(docs)
                    state["prefilled_observations"].append({
                        "tool": "retrieve_documents",
                        "args": {"query": result.translated_query, "filters": state.get("request_filters")},
                        "observation": obs_text
                    })
                    
                    # PROACTIVE SUFFICIENCY ASSESSMENT
                    # High quality if top score > 0.8 and multiple docs found
                    top_score = docs[0].get("score", 0)
                    if top_score > 0.85:
                        state["rag_quality"] = "high"
                    elif top_score > 0.7:
                        state["rag_quality"] = "medium"
                    else:
                        state["rag_quality"] = "low"
                        
                    logger.info("Proactive RAG fetch found %d docs. Quality: %s (Score: %.3f)", 
                                len(docs), state["rag_quality"], top_score)
                else:
                    state["rag_quality"] = "low"
            except Exception as e:
                logger.error("Proactive RAG fetch failed: %s", e)
                state["rag_quality"] = "low"
        
        # Track LLM call (1 call)
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        
        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["analyze_query"] = duration
        state["timings"] = timings
        
        logger.info(
            "Analyzed query: type=%s (%.3fs)", 
            result.query_type, 
            duration
        )
        
        return state
