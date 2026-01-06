"""Hybrid Pinecone retriever service."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from models import QueryIntent
from state import Document
from config import settings

logger = logging.getLogger(__name__)


def _hybrid_scale(
    dense: List[float],
    sparse: Dict[str, List[float]],
    alpha: float,
) -> tuple[list[float], Dict[str, List[float]]]:
    """Scale dense and sparse query vectors using a convex combination.

    Following Pinecone's hybrid search docs for a single hybrid index:
    score ≈ alpha * dense + (1 - alpha) * sparse.
    We apply alpha to the dense query vector and (1-alpha) to the sparse query
    vector before sending both to Pinecone.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {
        "indices": sparse.get("indices", []),
        "values": [v * (1.0 - alpha) for v in sparse.get("values", [])],
    }
    return scaled_dense, scaled_sparse


class RetrieverService:
    """Pinecone-based hybrid retriever using dense + BM25-based sparse vectors."""

    def __init__(self, settings) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            # Match the 1536-dimension Pinecone index using the large embedding model.
            dimensions=1536,
        )
        pc = Pinecone(api_key=settings.pinecone_api_key)
        # Type is provided by the Pinecone client; we don't depend on it here.
        self._index = pc.Index(settings.pinecone_index)

        # Load pre-trained BM25 encoder from JSON on disk for sparse query vectors.
        encoder_path = Path(__file__).resolve().parent.parent / "bm25_encoder.json"
        try:
            # Mirror your existing pattern: create an instance and call .load(path)
            encoder = BM25Encoder()
            encoder.load(str(encoder_path))
            self._bm25_encoder = encoder
            logger.info("Loaded BM25 encoder from %s", encoder_path)
        except Exception as exc:
            logger.warning("Failed to load BM25 encoder from %s: %s", encoder_path, exc)
            self._bm25_encoder = None

    async def _embed(self, text: str) -> List[float]:
        """Compute dense embedding for the query."""
        # LangChain embeddings are sync today; run in thread executor if needed.
        from anyio.to_thread import run_sync

        return await run_sync(self._embeddings.embed_query, text)

    async def retrieve(
        self,
        query_en: str,
        filters: Optional[Dict[str, Any]] = None,
        intent: QueryIntent = QueryIntent.CONCEPT_EXPLANATION,
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query_en: English query text
            class_level: Optional class/batch filter (from UI)
            subject: Optional subject filter (from UI)
            chapter: Optional chapter filter (from UI)
            lecture_id: Optional lecture session ID filter (from UI)
            intent: Query intent for weighting
        """
        start_time = time.time()
        
        # Generate dense embedding
        embed_start = time.time()
        try:
            dense = await self._embed(query_en)
            embed_time = time.time() - embed_start
            logger.info("⏱️  Embedding generation took %.3f seconds", embed_time)
        except Exception as exc:  # pragma: no cover
            logger.warning("Embedding failed: %s", exc)
            return []

        # Intent-based weights
        if intent == QueryIntent.CONCEPT_EXPLANATION:
            alpha = 0.7
        elif intent == QueryIntent.HOMEWORK_HELP:
            alpha = 0.4
        elif intent == QueryIntent.EXAM_PREP:
            alpha = 0.5
        else:
            alpha = 0.6

        # Use provided filters directly for Pinecone with transformation and whitelisting
        metadata_filter: Dict[str, Any] = {}
        
        # Valid metadata fields in current vector DB schema
        WHITELIST = {
            "class_id", "class_name", "subject", "subject_id", 
            "lecture_id", "teacher_id", "teacher_name", 
            "transcript_id", "is_ingested", "topics", "chapter"
        }
        
        # Numeric fields that MUST be integers for Pinecone filters to work
        INT_FIELDS = {
            "class_id", "subject_id", "lecture_id", 
            "teacher_id", "transcript_id", "chunk_index"
        }

        if filters:
            for key, value in filters.items():
                if key not in WHITELIST:
                    logger.debug("Skipping non-whitelisted filter: %s", key)
                    continue
                    
                # Automatic type casting for numeric IDs
                if key in INT_FIELDS and isinstance(value, str):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        logger.warning("Failed to cast filter %s='%s' to int", key, value)
                        continue

                if isinstance(value, dict):
                    # Assume already in operator format (e.g., {"$gt": 5})
                    metadata_filter[key] = value
                elif isinstance(value, (list, tuple)):
                    # Use $in for lists, ensure elements are cast if needed
                    if key in INT_FIELDS:
                        casted_list = []
                        for item in value:
                            try:
                                casted_list.append(int(item))
                            except (ValueError, TypeError):
                                casted_list.append(item)
                        value = casted_list
                    metadata_filter[key] = {"$in": value}
                else:
                    # Use $eq for scalars
                    metadata_filter[key] = {"$eq": value}

        # Encode sparse query vector using BM25 encoder if available
        sparse_start = time.time()
        sparse_vector: Optional[Dict[str, List[float]]] = None
        if self._bm25_encoder is not None:
            try:
                # BM25Encoder.encode_queries returns a list of {"indices": [...], "values": [...]}
                logger.debug("Query passed to sparse encoder: %s", [query_en])
                sparse_vecs = self._bm25_encoder.encode_queries([query_en])
                if sparse_vecs:
                    sparse_vector = sparse_vecs[0]
                sparse_time = time.time() - sparse_start
                logger.info("⏱️  BM25 encoding took %.3f seconds", sparse_time)
            except Exception as exc:  # pragma: no cover
                logger.warning("BM25 encoding failed, falling back to dense-only search: %s", exc)
                sparse_vector = None

        # Apply alpha weighting if we have a valid sparse query; otherwise fall back to dense-only search.
        if sparse_vector is not None and sparse_vector.get("indices") and sparse_vector.get("values"):
            dense_query, sparse_query = _hybrid_scale(dense, sparse_vector, alpha)
            query_kwargs: Dict[str, Any] = {
                "vector": dense_query,
                "sparse_vector": sparse_query,
                "top_k": settings.retriever_top_k,
                "filter": metadata_filter if metadata_filter else None,
                "include_metadata": True,
                "include_values": False,
            }
        else:
            # Dense-only fallback
            dense_query = [v * alpha for v in dense]
            query_kwargs: Dict[str, Any] = {
                "vector": dense_query,
                "top_k": settings.retriever_top_k,
                "filter": metadata_filter if metadata_filter else None,
                "include_metadata": True,
                "include_values": False,
            }
        
        if metadata_filter:
            logger.info("Applying Pinecone metadata filters: %s", metadata_filter)

        # Query Pinecone
        pinecone_start = time.time()
        try:
            res = self._index.query(**query_kwargs)
            pinecone_time = time.time() - pinecone_start
            logger.info("⏱️  Pinecone query took %.3f seconds", pinecone_time)
            # logger.info("Pinecone query result: %s", res)
        except Exception as exc:  # pragma: no cover
            logger.warning("Pinecone query failed: %s", exc)
            return []

        documents: List[Document] = []
        for match in getattr(res, "matches", []) or []:
            metadata = getattr(match, "metadata", {}) or {}
            text = metadata.get("text", "")
            documents.append(
                Document(
                    id=str(getattr(match, "id", "")),
                    score=float(getattr(match, "score", 0.0)),
                    text=text,
                    metadata=metadata,
                )
            )
        
        total_time = time.time() - start_time
        logger.info("🎯 Vector DB retrieval completed in %.3f seconds (found %d documents)", total_time, len(documents))
        
        return documents


