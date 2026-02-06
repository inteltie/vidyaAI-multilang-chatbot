"""Hybrid Pinecone retriever service."""

from __future__ import annotations

import asyncio
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
            max_retries=1, # Reduced retries for stability
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
        """Compute dense embedding for the query with Redis caching."""
        from services.cache_service import CacheService
        import hashlib

        # Generate deterministic cache key
        text_hash = hashlib.sha256(text.lower().strip().encode()).hexdigest()
        cache_key = f"embed:{self._embeddings.model}:{text_hash}"

        # Try cache first
        cached_vector = await CacheService.get(cache_key)
        if cached_vector:
            logger.info("Found embedding in cache for: %s", text[:30])
            return cached_vector

        # LangChain embeddings are sync today; run in thread executor
        from anyio.to_thread import run_sync
        
        vector = await run_sync(self._embeddings.embed_query, text)
        
        # Save to cache (TTL 24 hours for embeddings)
        await CacheService.set(cache_key, vector, ttl=86400)
        
        return vector

    async def retrieve(
        self,
        query_en: str,
        filters: Optional[Dict[str, Any]] = None,
        intent: QueryIntent = QueryIntent.CONCEPT_EXPLANATION,
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search with result caching.
        """
        start_time = time.time()
        from services.cache_service import CacheService
        import hashlib
        import json

        # Intent-based weights (affects cache key)
        if intent == QueryIntent.CONCEPT_EXPLANATION:
            alpha = 0.7
        elif intent == QueryIntent.HOMEWORK_HELP:
            alpha = 0.4
        elif intent == QueryIntent.EXAM_PREP:
            alpha = 0.5
        else:
            alpha = 0.6

        # 1. Check result cache first (Phase 3: Cost & Scale)
        filter_str = json.dumps(filters or {}, sort_keys=True)
        cache_payload = (
            f"{query_en.lower().strip()}||{filter_str}||{intent.value}||"
            f"{settings.pinecone_index}||{settings.embedding_model}||"
            f"{settings.retriever_top_k}||{alpha}||{bool(self._bm25_encoder)}"
        )
        retrieval_hash = hashlib.sha256(cache_payload.encode()).hexdigest()
        retrieval_cache_key = f"rag_res:{retrieval_hash}"
        
        cached_docs = await CacheService.get(retrieval_cache_key)
        if cached_docs:
            logger.info("🎯 Found Pinecone results in cache for query: %s", query_en[:30])
            # CacheService returns deserialized JSON (list of dicts)
            return cached_docs

        # 2. Generate dense embedding (with its own cache)
        embed_start = time.time()
        try:
            dense = await self._embed(query_en)
            embed_time = time.time() - embed_start
        except Exception as exc:  # pragma: no cover
            logger.warning("Embedding failed: %s", exc)
            return []

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
                sparse_vecs = await asyncio.to_thread(self._bm25_encoder.encode_queries, [query_en])
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
            res = await asyncio.to_thread(self._index.query, **query_kwargs)
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
        
        for i, doc in enumerate(documents, 1):
            text_snippet = doc.text[:100].replace("\n", " ") if hasattr(doc, 'text') else str(doc.get("text", ""))[:100].replace("\n", " ")
            logger.info("[RAG_RESULT] Doc %d: score=%.4f, id=%s, text=%s...", i, doc.score if hasattr(doc, 'score') else doc.get("score"), doc.id if hasattr(doc, 'id') else doc.get("id"), text_snippet)
        
        # Save to cache (Phase 3: Cost & Scale)
        if documents:
            await CacheService.set(retrieval_cache_key, documents, ttl=3600)
            
        return documents

