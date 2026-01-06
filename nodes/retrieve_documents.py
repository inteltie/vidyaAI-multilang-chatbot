"""LangGraph node: retrieve_documents."""

from __future__ import annotations

import logging
from time import perf_counter

from services import RetrieverService
from state import AgentState

logger = logging.getLogger(__name__)


class RetrieveDocumentsNode:
    """LangGraph node: retrieve_documents."""

    def __init__(self, retriever: RetrieverService) -> None:
        self._retriever = retriever

    async def __call__(self, state: AgentState) -> AgentState:
        start = perf_counter()
        
        # ONLY use explicit request filters provided in the request body
        request_filters = state.get("request_filters", {})
        
        logger.info("Retrieving documents for strict filters: %s", request_filters)
        docs = await self._retriever.retrieve(
            query_en=state["query_en"],
            filters=request_filters, # Strictly use request body filters
            intent=state["intent"],
        )
        state["documents"] = docs
        logger.info("Retrieved %d documents", len(docs))

        # Log retrieval results for debugging
        if docs:
            logger.info("=" * 80)
            logger.info("RETRIEVAL RESULTS: Retrieved %d documents", len(docs))
            for i, d in enumerate(docs, 1):
                meta = d.get("metadata", {}) or {}
                logger.info(
                    "  Doc %d: score=%.3f session_id=%s subject=%s chapter=%s",
                    i,
                    d.get("score", 0.0),
                    meta.get("session_id"),
                    meta.get("subject"),
                    meta.get("chapter"),
                )
                # Log first 100 chars of text
                text = d.get("text", "")[:100]
                logger.info("    Text preview: %s...", text)
            logger.info("=" * 80)
        else:
            logger.warning("RETRIEVAL RESULTS: No documents retrieved - LLM will use general knowledge")

        # Build lightweight citations from document metadata.
        citations = []
        for d in docs:
            meta = d.get("metadata", {}) or {}
            citations.append(
                {
                    "id": d.get("id", ""),
                    "score": d.get("score", 0.0),
                    "subject": meta.get("subject_id"),
                    "topics": meta.get("topics"),
                    "class_id": meta.get("class_id"),
                    "lecture_id": str(meta.get("lecture_id")) if meta.get("lecture_id") is not None else None,
                    "transcript_id": str(meta.get("transcript_id")) if meta.get("transcript_id") is not None else None,
                    "chunk_id": str(meta.get("chunk_id")) if meta.get("chunk_id") is not None else None,
                    "teacher_id": meta.get("teacher_id"),
                }
            )
        state["citations"] = citations

        duration = perf_counter() - start
        timings = state.get("timings") or {}
        timings["retrieve_documents"] = duration
        state["timings"] = timings
        return state


