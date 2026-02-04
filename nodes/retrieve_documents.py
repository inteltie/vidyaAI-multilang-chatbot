"""LangGraph node: retrieve_documents."""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict
from time import perf_counter

from services import RetrieverService
from state import AgentState

logger = logging.getLogger(__name__)


class RetrieveDocumentsNode:
    """LangGraph node: retrieve_documents."""

    def __init__(self, retriever: RetrieverService) -> None:
        self._retriever = retriever

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        
        query_en = state["query_en"]
        query_raw = state.get("query", "")
        # Use explicit request filters provided in the request body
        request_filters = state.get("request_filters", {})
        
        logger.info("Retrieving documents for query: %s", query_en[:50])
        
        # Sanitize filters: Strip additionalProp1
        clean_filters = {k: v for k, v in request_filters.items() if k != "additionalProp1"}
        
        from models import QueryIntent
        intent = state.get("intent", QueryIntent.CONCEPT_EXPLANATION)

        # Proactive Web Search trigger: If keywords detected, we start it PARALLEL with RAG
        needs_web_proactive = any(w in query_raw.lower() for w in ["latest", "recent", "news", "current"])
        
        # Speculative RAG reuse logic
        speculative_docs = state.get("speculative_documents")
        can_use_speculative = False
        
        if speculative_docs is not None:
             # Logic: if query_en is basically the same as the one used for speculative RAG, reuse it.
             # We can't easily check what query was used in AnalyzeQueryNode without storing it, 
             # but we can assume it was the raw 'query' field.
             raw_query = state.get("query", "")
             if query_en.lower() == raw_query.lower() or (len(query_en) > 0 and query_en in raw_query):
                 logger.info("Reusing speculative RAG results (queries match).")
                 can_use_speculative = True
             else:
                 logger.info("Speculative RAG mismatch: raw='%s', analyzed='%s'. Re-fetching.", raw_query[:20], query_en[:20])

        # Define tasks (or use speculative result)
        rag_timeout = 15.0
        try:
            if can_use_speculative:
                docs = speculative_docs
                web_results = None # We'll check for proactive web search below
            else:
                rag_task = self._retriever.retrieve(
                    query_en=query_en,
                    filters=clean_filters if clean_filters else None,
                    intent=intent
                )
                
                web_task = None
                if needs_web_proactive:
                    from tools.web_search_tool import WebSearchTool
                    logger.info("Triggering proactive web search in parallel with RAG...")
                    web_tool = WebSearchTool()
                    web_task = web_tool.execute(query=query_en)
        
                # Execute parallelized tasks with timeout
                if web_task:
                    docs, web_results = await asyncio.wait_for(asyncio.gather(rag_task, web_task), timeout=rag_timeout)
                else:
                    docs = await asyncio.wait_for(rag_task, timeout=rag_timeout)
                    web_results = None
        except asyncio.TimeoutError:
            logger.warning("Retrieval tasks timed out after %.1fs", rag_timeout)
            docs = []
            web_results = None

        state["prefilled_observations"] = []
        
        # Process RAG results
        if docs:
            # Filter for state["documents"] as well to ensure validator uses high-score docs
            high_score_docs = [d for d in docs if d.get("score", 0.0) >= 0.40]
            state["documents"] = high_score_docs
            
            from tools import RetrievalTool
            obs_text = RetrievalTool.format_documents(docs, min_score=0.40)
            state["prefilled_observations"].append({
                "tool": "retrieve_documents",
                "args": {"query": query_en, "filters": request_filters},
                "observation": obs_text
            })
            
            # PROACTIVE SUFFICIENCY ASSESSMENT
            # Adjusted thresholds to reduce unnecessary web searches
            top_score = docs[0].get("score", 0)
            if top_score > 0.65:
                state["rag_quality"] = "high"
            elif top_score > 0.45:
                state["rag_quality"] = "medium"
            else:
                state["rag_quality"] = "low"
                
            logger.info("Found %d docs. Quality: %s (Score: %.3f)", 
                        len(docs), state["rag_quality"], top_score)
        else:
            state["documents"] = []
            state["rag_quality"] = "low"
            logger.warning("No documents retrieved")

        # Handle Web Search results (either from parallel task or needed due to low quality)
        llm_calls_in_node = 0
        
        if web_results:
            # Web search was done in parallel
            state["prefilled_observations"].append({
                "tool": "web_search",
                "args": {"query": query_en},
                "observation": web_results
            })
            llm_calls_in_node += 1
            logger.info("Proactive web search results added.")
        elif state.get("rag_quality") == "low":
            # RAG failed, and we didn't start web search in parallel. 
            # Check if we have enough time left (Stability Phase 2: Skip web search on slow paths)
            current_elapsed = perf_counter() - start
            if current_elapsed > 10.0: # If RAG already took > 10s, skip web search
                logger.warning("RAG quality low but RAG was slow (%.2fs). Skipping reactive web search to save time.", current_elapsed)
            else:
                logger.info("RAG quality low. Triggering reactive web search...")
                from tools.web_search_tool import WebSearchTool
                web_tool = WebSearchTool()
                try:
                    web_results = await asyncio.wait_for(web_tool.execute(query=query_en), timeout=10.0)
                    state["prefilled_observations"].append({
                        "tool": "web_search",
                        "args": {"query": query_en},
                        "observation": web_results
                    })
                    llm_calls_in_node += 1
                    logger.info("Reactive web search completed.")
                except asyncio.TimeoutError:
                    logger.warning("Reactive web search timed out after 10s")

        # Build lightweight citations
        citations = []
        for d in state["documents"]:
            meta = d.get("metadata", {}) or {}
            citations.append({
                "id": d.get("id", ""),
                "score": d.get("score", 0.0),
                "subject": meta.get("subject"),
                "topics": str(meta.get("topics")) if meta.get("topics") is not None else None,
                "class_id": meta.get("class_id"),
                "lecture_id": str(meta.get("lecture_id")) if meta.get("lecture_id") is not None else None,
                "transcript_id": str(meta.get("transcript_id")) if meta.get("transcript_id") is not None else None,
                "chunk_id": str(meta.get("chunk_id")) if meta.get("chunk_id") is not None else None,
            })
        state["citations"] = citations

        duration = perf_counter() - start
        
        return {
            "documents": state["documents"],
            "prefilled_observations": state["prefilled_observations"],
            "rag_quality": state["rag_quality"],
            "citations": state["citations"],
            "llm_calls": llm_calls_in_node,
            "timings": {"retrieve_documents": duration}
        }


__all__ = ["RetrieveDocumentsNode"]
