"""LangGraph node: retrieve_documents."""

from __future__ import annotations

import logging
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
        docs = await self._retriever.retrieve(
            query_en=query_en,
            filters=clean_filters if clean_filters else None,
            intent=state.get("intent", QueryIntent.CONCEPT_EXPLANATION)
        )
        
        state["prefilled_observations"] = []
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
            top_score = docs[0].get("score", 0)
            if top_score > 0.85:
                state["rag_quality"] = "high"
            elif top_score > 0.7:
                state["rag_quality"] = "medium"
            else:
                state["rag_quality"] = "low"
                
            logger.info("Found %d docs. Quality: %s (Score: %.3f)", 
                        len(docs), state["rag_quality"], top_score)
        else:
            state["documents"] = []
            state["rag_quality"] = "low"
            logger.warning("No documents retrieved")

        # PROACTIVE WEB SEARCH: If quality is low or external cues detected
        if state.get("rag_quality") == "low" or any(w in query_raw.lower() for w in ["latest", "recent", "news", "current"]):
            logger.info("Triggering proactive web search...")
            from tools.web_search_tool import WebSearchTool
            web_tool = WebSearchTool()
            web_results = await web_tool.execute(query=query_en)
            state["prefilled_observations"].append({
                "tool": "web_search",
                "args": {"query": query_en},
                "observation": web_results
            })
            state["llm_calls"] = state.get("llm_calls", 0) + 1
            logger.info("Proactive web search completed.")

        # Build lightweight citations
        citations = []
        for d in state["documents"]:
            meta = d.get("metadata", {}) or {}
            citations.append({
                "id": d.get("id", ""),
                "score": d.get("score", 0.0),
                "subject": meta.get("subject_id"),
                "topics": meta.get("topics"),
                "class_id": meta.get("class_id"),
                "lecture_id": str(meta.get("lecture_id")) if meta.get("lecture_id") is not None else None,
            })
        state["citations"] = citations

        duration = perf_counter() - start
        
        # Return ONLY the updates to avoid conflicts with parallel agent node
        # For llm_calls, we return the DELTA (calls made in THIS node)
        # so the operator.add reducer in AgentState works correctly.
        calls_made = 1 if (state.get("rag_quality") == "low" or any(w in query_raw.lower() for w in ["latest", "recent", "news", "current"])) else 0

        return {
            "documents": state["documents"],
            "prefilled_observations": state["prefilled_observations"],
            "rag_quality": state["rag_quality"],
            "citations": state["citations"],
            "llm_calls": calls_made,
            "timings": {"retrieve_documents": duration}
        }


