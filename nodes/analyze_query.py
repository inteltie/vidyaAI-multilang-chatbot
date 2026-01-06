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
        
        # Phase 1 Optimization: Merged session metadata extraction
        if result.query_type != "conversational":
            if "session_metadata" not in state:
                state["session_metadata"] = {}
                
            # Merge extracted values as fallbacks (UI/existing session data takes precedence)
            meta = state["session_metadata"]
            if not meta.get("class_level") and result.class_level:
                meta["class_level"] = result.class_level
                logger.info("Extracted class_level: %s", result.class_level)
            if not meta.get("subject") and result.extracted_subject:
                meta["subject"] = result.extracted_subject
                logger.info("Extracted subject: %s", result.extracted_subject)
            if not meta.get("topics") and result.chapter:
                meta["topics"] = result.chapter # Mapping chapter to topics for DB consistency
                logger.info("Extracted topics (chapter): %s", result.chapter)
            if not meta.get("lecture_id") and result.lecture_id:
                meta["lecture_id"] = result.lecture_id
                logger.info("Extracted lecture_id: %s", result.lecture_id)
        
        # Proactive RAG (Optimization for speed)
        state["prefilled_observations"] = []
        if result.query_type == "curriculum_specific" and self._retriever:
            try:
                from models import QueryIntent
                from tools import RetrievalTool
                
                logger.info("Performing proactive RAG fetch for query: %s", result.translated_query[:50])
                
                # Sanitize filters: Strip additionalProp1
                raw_filters = state.get("request_filters", {})
                clean_filters = {k: v for k, v in raw_filters.items() if k != "additionalProp1"}
                
                docs = await self._retriever.retrieve(
                    query_en=result.translated_query,
                    filters=clean_filters if clean_filters else None,
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
                    
                # PROACTIVE WEB SEARCH: If quality is low or external cues detected
                if state.get("rag_quality") == "low" or any(w in query.lower() for w in ["latest", "recent", "news", "current"]):
                    logger.info("Triggering proactive web search...")
                    from tools.web_search_tool import WebSearchTool
                    # We use a shared instance if possible, but for node simplicity we can instantiate
                    # Note: In production, consider inject this tool
                    web_tool = WebSearchTool()
                    web_results = await web_tool.execute(query=result.translated_query)
                    state["prefilled_observations"].append({
                        "tool": "web_search",
                        "args": {"query": result.translated_query},
                        "observation": web_results
                    })
                    logger.info("Proactive web search completed.")

            except Exception as e:
                logger.error("Proactive fetch failed: %s", e)
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
