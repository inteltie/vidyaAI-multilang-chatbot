"""LangGraph node: retrieve_documents."""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict
from time import perf_counter
import tiktoken

from services import RetrieverService
from services.web_query_rewriter import WebQueryRewriter
from services.agent_findings import publish
from services.costing import (
    merge_usage_by_component,
    parse_web_search_result,
    patch_component_usage,
)
from state import AgentState
from config import settings

logger = logging.getLogger(__name__)


class RetrieveDocumentsNode:
    """LangGraph node: retrieve_documents."""

    def __init__(self, retriever: RetrieverService, rewriter: WebQueryRewriter | None = None) -> None:
        self._retriever = retriever
        self._rewriter = rewriter

    async def _prepare_web_query(
        self,
        state: AgentState,
        original_query: str,
        fallback_reason: str,
    ) -> tuple[str, int, int, dict[str, object]]:
        if not settings.web_rewrite_enabled or self._rewriter is None:
            return original_query, 0, 0, {
                "accepted": True,
                "reason": "rewrite_disabled",
                "overlap_ratio": 1.0,
            }
        history = state.get("conversation_history", []) or []
        snippets: list[str] = []
        for msg in history[-2:]:
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("text") or ""
            else:
                content = getattr(msg, "content", "") or ""
            text = str(content).strip()
            if text:
                snippets.append(text)
        rewritten, i_t, o_t, guard = await self._rewriter.rewrite(
            query_en=original_query,
            context_snippets=snippets,
            fallback_reason=fallback_reason,
        )
        return (rewritten or original_query), int(i_t or 0), int(o_t or 0), guard

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        
        query_en = state.get("query_en")
        query_raw = state.get("query", "")
        # Use explicit request filters provided in the request body
        request_filters = state.get("request_filters", {})
        orchestration_route = str(state.get("orchestration_route") or "retriever")
        fallback_reason = None
        embedding_tokens = 0
        rewrite_input_tokens = 0
        rewrite_output_tokens = 0
        web_query_original = query_en or ""
        web_query_used = query_en or ""
        rewrite_guard_reason = ""
        rewrite_overlap_ratio = 0.0
        rewrite_used_original_due_to_guard = False

        def _apply_guard(guard: dict[str, object], used_query: str) -> None:
            nonlocal rewrite_guard_reason, rewrite_overlap_ratio, rewrite_used_original_due_to_guard
            reason = str((guard or {}).get("reason", "") or "")
            overlap = float((guard or {}).get("overlap_ratio", 0.0) or 0.0)
            if not reason:
                reason = "accepted"
            rewrite_guard_reason = reason
            rewrite_overlap_ratio = overlap
            rewrite_used_original_due_to_guard = (
                bool(settings.web_rewrite_enabled)
                and str(used_query or "").strip() == str(query_en or "").strip()
                and reason not in ("rewrite_disabled", "accepted", "no_original_tokens")
            )
        
        if not query_en:
            logger.warning("Missing query_en for document retrieval. Skipping.")
            return {
                "documents": [],
                "prefilled_observations": [],
                "rag_quality": "low",
                "web_fallback_reason": "missing_query_en",
                "citations": [],
                "llm_calls": 0,
                "timings": {"retrieve_documents": perf_counter() - start}
            }

        if orchestration_route in {"none", "image_generate"}:
            reason = (
                "route_none_skip_retrieval"
                if orchestration_route == "none"
                else "route_image_generation_skip_retrieval"
            )
            return {
                "documents": [],
                "prefilled_observations": [],
                "rag_quality": "low",
                "web_fallback_reason": reason,
                "citations": [],
                "llm_calls": 0,
                "web_search_calls": 0,
                "timings": {"retrieve_documents": perf_counter() - start},
            }

        logger.info("Retrieving documents for query: %s", query_en[:50])
        try:
            try:
                encoder = tiktoken.encoding_for_model(settings.embedding_model)
            except Exception:
                encoder = tiktoken.get_encoding("cl100k_base")
            embedding_tokens = len(encoder.encode(query_en))
        except Exception as exc:
            logger.debug("Failed to estimate embedding tokens: %s", exc)
            embedding_tokens = 0
        
        # Sanitize filters: Strip additionalProp1
        clean_filters = {k: v for k, v in request_filters.items() if k != "additionalProp1"}
        
        from models import QueryIntent
        intent = state.get("intent", QueryIntent.CONCEPT_EXPLANATION)

        # Proactive Web Search trigger: If recency/current-affairs cues are detected.
        recency_terms = [
            "latest", "recent", "news", "current", "today", "now", "currently",
            "this week", "this month", "this year", "breaking", "update", "updates",
            "current affairs", "right now", "live", "headline", "headlines",
        ]
        query_raw_lower = query_raw.lower()
        needs_web_proactive = orchestration_route in {"web", "hybrid"} or (
            orchestration_route == "retriever"
            and any(term in query_raw_lower for term in recency_terms)
        )
        
        # Speculative RAG reuse logic
        speculative_docs = state.get("documents")
        can_use_speculative = False
        
        if speculative_docs:
             # Logic: if we have documents from AnalyzeQueryNode, reuse them.
             logger.info("Reusing speculative RAG results (found %d docs in state).", len(speculative_docs))
             can_use_speculative = True

        # Define tasks (or use speculative result)
        rag_timeout = 15.0
        try:
            if can_use_speculative:
                docs = speculative_docs
                web_results = None 
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
                    web_query_used, rw_i, rw_o, guard = await self._prepare_web_query(
                        state=state,
                        original_query=query_en,
                        fallback_reason="recency",
                    )
                    _apply_guard(guard, web_query_used)
                    rewrite_input_tokens += rw_i
                    rewrite_output_tokens += rw_o
                    web_task = web_tool.execute(query=web_query_used)
        
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
            min_score = settings.retriever_score_threshold
            high_score_threshold = settings.rag_quality_high_score

            # Ensure deterministic ordering by score
            docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)

            # Filter for state["documents"] as well to ensure validator uses high-score docs
            high_score_docs = [d for d in docs if d.get("score", 0.0) >= min_score]
            state["documents"] = high_score_docs
            
            top_score = docs[0].get("score", 0.0)
            
            # Only add RAG results to prefilled_observations if they pass the threshold
            if top_score >= min_score:
                # Log the selection
                for d in high_score_docs:
                    logger.info("[PROACTIVE_RAG] Selected: id=%s, score=%.4f", d.get("id"), d.get("score"))
                
                from tools import RetrievalTool
                obs_text = RetrievalTool.format_documents(docs, min_score=min_score)
                state["prefilled_observations"].append({
                    "tool": "retrieve_documents",
                    "args": {"query": query_en, "filters": request_filters},
                    "observation": obs_text
                })
                
                # Assess RAG quality
                if top_score >= high_score_threshold:
                    state["rag_quality"] = "high"
                else:
                    state["rag_quality"] = "medium"
            else:
                state["rag_quality"] = "low"
                logger.info("Top RAG score %.3f below %.2f threshold. Bypassing RAG documents.", top_score, min_score)
            
            logger.info("Found %d docs. Quality: %s (Score: %.3f)", 
                        len(docs), state["rag_quality"], top_score)
        else:
            state["documents"] = []
            state["rag_quality"] = "low"
            logger.warning("No documents retrieved")

        # Handle Web Search results (either from parallel task or needed due to low quality)
        llm_calls_in_node = 0
        input_tokens = 0
        output_tokens = 0
        token_usage_by_model: dict[str, dict[str, int]] = {}
        usage_by_component: dict[str, dict[str, Any]] = {}
        web_search_calls = 0
        web_model = settings.web_search_model_name or "gpt-4o-mini"
        analyzer_model = settings.analyzer_model_name or "gpt-4.1-nano"

        if web_results:
            obs_text, i_t, o_t, billable = parse_web_search_result(web_results)
            input_tokens = i_t
            output_tokens = o_t

            state["prefilled_observations"].append({
                "tool": "web_search",
                "args": {"query": web_query_used},
                "observation": obs_text
            })
            fallback_reason = "recency"
            llm_calls_in_node += 1
            token_usage_by_model[web_model] = {
                "input": token_usage_by_model.get(web_model, {}).get("input", 0) + int(input_tokens or 0),
                "output": token_usage_by_model.get(web_model, {}).get("output", 0) + int(output_tokens or 0),
            }
            usage_by_component = merge_usage_by_component(
                usage_by_component,
                patch_component_usage(
                    "web_search", web_model, i_t, o_t, billable_web_call=billable
                ).get("usage_by_component", {}),
            )
            if billable:
                web_search_calls += 1
            if rewrite_input_tokens or rewrite_output_tokens:
                token_usage_by_model[analyzer_model] = {
                    "input": token_usage_by_model.get(analyzer_model, {}).get("input", 0) + int(rewrite_input_tokens),
                    "output": token_usage_by_model.get(analyzer_model, {}).get("output", 0) + int(rewrite_output_tokens),
                }
                usage_by_component = merge_usage_by_component(
                    usage_by_component,
                    patch_component_usage(
                        "chat", analyzer_model, rewrite_input_tokens, rewrite_output_tokens
                    ).get("usage_by_component", {}),
                )
                input_tokens += int(rewrite_input_tokens)
                output_tokens += int(rewrite_output_tokens)
            logger.info("Proactive web search results added. reason=%s", fallback_reason)
        elif state.get("rag_quality") == "low" and orchestration_route in {"retriever", "hybrid", "web"}:
            # RAG failed, and we didn't start web search in parallel. 
            # Check if we have enough time left (Stability Phase 2: Skip web search on slow paths)
            current_elapsed = perf_counter() - start
            if current_elapsed > 10.0: # If RAG already took > 10s, skip web search
                logger.warning("RAG quality low but RAG was slow (%.2fs). Skipping reactive web search to save time.", current_elapsed)
                fallback_reason = "timeout_skip"
            else:
                logger.info("RAG quality low. Triggering reactive web search...")
                from tools.web_search_tool import WebSearchTool
                web_tool = WebSearchTool()
                try:
                    web_query_used, rw_i, rw_o, guard = await self._prepare_web_query(
                        state=state,
                        original_query=query_en,
                        fallback_reason="low_rag",
                    )
                    _apply_guard(guard, web_query_used)
                    rewrite_input_tokens += rw_i
                    rewrite_output_tokens += rw_o
                    web_results = await asyncio.wait_for(web_tool.execute(query=web_query_used), timeout=10.0)
                    obs_text, i_t, o_t, billable = parse_web_search_result(web_results)
                    input_tokens = i_t
                    output_tokens = o_t

                    state["prefilled_observations"].append({
                        "tool": "web_search",
                        "args": {"query": web_query_used},
                        "observation": obs_text
                    })
                    fallback_reason = "low_rag"
                    if orchestration_route != "hybrid":
                        fallback_reason = "low_rag_single_fallback"
                    llm_calls_in_node += 1
                    token_usage_by_model[web_model] = {
                        "input": token_usage_by_model.get(web_model, {}).get("input", 0) + int(input_tokens or 0),
                        "output": token_usage_by_model.get(web_model, {}).get("output", 0) + int(output_tokens or 0),
                    }
                    usage_by_component = merge_usage_by_component(
                        usage_by_component,
                        patch_component_usage(
                            "web_search", web_model, i_t, o_t, billable_web_call=billable
                        ).get("usage_by_component", {}),
                    )
                    if billable:
                        web_search_calls += 1
                    if rewrite_input_tokens or rewrite_output_tokens:
                        token_usage_by_model[analyzer_model] = {
                            "input": token_usage_by_model.get(analyzer_model, {}).get("input", 0) + int(rewrite_input_tokens),
                            "output": token_usage_by_model.get(analyzer_model, {}).get("output", 0) + int(rewrite_output_tokens),
                        }
                        usage_by_component = merge_usage_by_component(
                            usage_by_component,
                            patch_component_usage(
                                "chat", analyzer_model, rewrite_input_tokens, rewrite_output_tokens
                            ).get("usage_by_component", {}),
                        )
                        input_tokens += int(rewrite_input_tokens)
                        output_tokens += int(rewrite_output_tokens)
                    logger.info("Reactive web search completed. reason=%s", fallback_reason)
                except asyncio.TimeoutError:
                    logger.warning("Reactive web search timed out after 10s")
                    fallback_reason = "web_timeout"
        elif needs_web_proactive:
            # Proactive web path was relevant, but we may have skipped running it due to
            # speculative RAG reuse. Distinguish skip vs attempted-without-results.
            if can_use_speculative and web_results is None:
                fallback_reason = "recency_skipped_speculative"
            else:
                fallback_reason = "recency_no_result"

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

        top_titles = []
        for d in (state.get("documents") or [])[:3]:
            meta = d.get("metadata") or {}
            title = meta.get("title") or meta.get("topics") or d.get("id")
            if title:
                top_titles.append(str(title))

        findings = publish(
            state.get("agent_findings"),
            "retrieve_documents",
            "curriculum",
            f"rag_quality={state.get('rag_quality', 'low')}",
            {
                "rag_quality": state.get("rag_quality"),
                "top_doc_titles": top_titles,
                "document_count": len(state.get("documents") or []),
            },
        )
        
        return {
            "documents": state["documents"],
            "agent_findings": findings,
            "prefilled_observations": state["prefilled_observations"],
            "rag_quality": state["rag_quality"],
            "web_fallback_reason": fallback_reason,
            "citations": state["citations"],
            "llm_calls": llm_calls_in_node,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_usage_by_model": token_usage_by_model,
            "usage_by_component": usage_by_component,
            "web_search_calls": web_search_calls,
            "embedding_tokens": embedding_tokens,
            "web_rewrite_enabled": bool(settings.web_rewrite_enabled),
            "web_query_original": web_query_original,
            "web_query_used": web_query_used,
            "rewrite_input_tokens": int(rewrite_input_tokens),
            "rewrite_output_tokens": int(rewrite_output_tokens),
            "rewrite_guard_reason": rewrite_guard_reason,
            "rewrite_overlap_ratio": float(rewrite_overlap_ratio),
            "rewrite_used_original_due_to_guard": bool(rewrite_used_original_due_to_guard),
            "timings": {"retrieve_documents": duration}
        }


__all__ = ["RetrieveDocumentsNode"]
