"""LangGraph node: orchestrate_turn.

The Turn Orchestrator is the single, context-aware decision point for a turn.
It reads the FULL conversation and emits one structured ``TurnPlan`` (intent,
route, modality, standalone query, response strategy). All downstream routing
(``route_decision``) and enrichment (``analyze_query``) consume this plan rather
than re-deriving intent from keywords.

Decision priority:
1. Deterministic pre-gate for trivially obvious turns (empty / pure greeting) ->
   skip the LLM entirely.
2. Otherwise call the LLM planner (cheap analyzer model) over the full history.
3. On any failure/timeout, fall back to the heuristic classifier so behaviour
   degrades gracefully and is never worse than before.
"""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any, Dict, Optional

from config import settings
from services.query_classifier import QueryClassifier, QueryClassification
from services.query_understanding import QueryUnderstandingService, TurnPlan
from state import AgentState

logger = logging.getLogger(__name__)


class OrchestratorNode:
    """Produce an authoritative, context-aware TurnPlan for the current turn."""

    def __init__(
        self,
        classifier: QueryClassifier,
        understanding: Optional[QueryUnderstandingService] = None,
    ) -> None:
        self._classifier = classifier
        self._understanding = understanding

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _strip_context_wrapper(query: str) -> str:
        """Normalize runner-injected ``Context:...\\nCurrent question:`` wrappers."""
        if "\nCurrent question:" in query and query.strip().lower().startswith("context:"):
            try:
                tail = query.split("\nCurrent question:", 1)[1].strip()
                if tail:
                    return tail
            except Exception:
                pass
        return query

    @staticmethod
    def _is_pure_greeting(query: str) -> bool:
        from services.utils import is_greeting

        return is_greeting(query)

    def _conversational_plan(self, standalone: str, *, reasoning: str) -> TurnPlan:
        return TurnPlan(
            intent="greeting",
            route="conversational",
            query_type="conversational",
            needs_curriculum=False,
            needs_web=False,
            is_time_sensitive=False,
            standalone_query_en=standalone,
            subjects=["General"],
            confidence=1.0,
            reasoning=reasoning,
        )

    def _accumulate_tokens(
        self, updates: Dict[str, Any], model_name: str, usage: Dict[str, int]
    ) -> None:
        in_tok = int((usage or {}).get("input", 0) or 0)
        out_tok = int((usage or {}).get("output", 0) or 0)
        if not (in_tok or out_tok):
            return
        updates["input_tokens"] += in_tok
        updates["output_tokens"] += out_tok
        updates["aux_input_tokens"] = updates.get("aux_input_tokens", 0) + in_tok
        updates["aux_output_tokens"] = updates.get("aux_output_tokens", 0) + out_tok
        prev = updates["token_usage_by_model"].get(model_name, {})
        merged = {
            "input": prev.get("input", 0) + in_tok,
            "output": prev.get("output", 0) + out_tok,
        }
        updates["token_usage_by_model"][model_name] = merged
        updates["usage_by_component"] = {"chat": {model_name: dict(merged)}}

    async def _fallback_plan(
        self,
        updates: Dict[str, Any],
        model_name: str,
        *,
        query: str,
        history: Any,
        turn_ctx: Any,
        is_time_sensitive: bool,
        is_current_affairs: bool,
    ) -> TurnPlan:
        """Heuristic + classifier fallback when the LLM planner is unavailable."""
        from services.image_modality_policy import looks_like_image_explanation_request

        quality = self._classifier.assess_query_quality(query)
        normalized = quality.normalized_query or query
        explain_existing = looks_like_image_explanation_request(query)
        wants_visual = bool(turn_ctx.wants_visual_output) and not explain_existing

        heuristic = self._classifier._check_heuristics(normalized)
        if heuristic is not None:
            query_type = heuristic.query_type
            subjects = heuristic.subjects or ["General"]
            standalone = heuristic.translated_query or normalized
            class_level = None
            confidence = heuristic.confidence
            reasoning = f"Fallback heuristic: {heuristic.reasoning}"
        else:
            try:
                cls = await self._classifier.analyze(normalized, history)
                self._accumulate_tokens(
                    updates,
                    model_name,
                    {"input": cls.input_tokens, "output": cls.output_tokens},
                )
                updates["llm_calls"] = updates.get("llm_calls", 0) + 1
                query_type = cls.query_type
                subjects = cls.subjects or ["General"]
                standalone = cls.translated_query or normalized
                class_level = cls.class_level
                confidence = cls.confidence
                reasoning = f"Fallback classifier: {cls.reasoning}"
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Fallback classifier failed: %s", exc)
                query_type = "curriculum_specific"
                subjects = ["General"]
                standalone = normalized
                class_level = None
                confidence = 0.0
                reasoning = f"Fallback default (error: {exc})."

        route = "conversational" if query_type == "conversational" else "curriculum"
        needs_curriculum = query_type == "curriculum_specific"
        needs_web = bool(is_time_sensitive or is_current_affairs)

        if wants_visual:
            route = "image_generate"
            query_type = "image_generation"
            needs_curriculum = False
        elif needs_web and route == "curriculum":
            route = "hybrid"
        elif needs_web and route != "conversational":
            route = "web"

        return TurnPlan(
            intent="request_new_image" if wants_visual else (
                "greeting" if query_type == "conversational" else "new_question"
            ),
            route=route,
            query_type=query_type,
            needs_curriculum=needs_curriculum,
            needs_web=needs_web,
            is_time_sensitive=bool(is_time_sensitive),
            standalone_query_en=standalone,
            subjects=subjects,
            class_level=class_level,
            resolved_topic=(turn_ctx.last_user_topic or ""),
            wants_new_image=wants_visual,
            wants_image_explanation=explain_existing,
            wants_flowchart=bool(turn_ctx.wants_flowchart),
            response_strategy=turn_ctx.response_strategy,
            confidence=confidence,
            reasoning=reasoning,
        )

    # ---- main ------------------------------------------------------------

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        model_name = getattr(self._classifier._llm, "model_name", None) or "unknown"
        updates: Dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "token_usage_by_model": {},
        }

        raw_query = state.get("query", "") or ""
        query = self._strip_context_wrapper(raw_query)
        has_images = bool(state.get("has_images") or (state.get("images") or []))
        agent_mode = (state.get("agent_mode") or "standard").strip().lower()
        history = state.get("conversation_history", []) or []

        # Keyword guardrails for freshness (the LLM also decides this; we OR them).
        is_time_sensitive_kw = bool(
            getattr(self._classifier, "detect_time_sensitive_query", lambda _q: False)(query)
        )
        is_current_affairs_kw = bool(
            getattr(self._classifier, "detect_current_affairs_query", lambda _q: False)(query)
        )

        from services.context_resolver import resolve_turn_context

        turn_ctx = resolve_turn_context(
            query,
            history,
            session_metadata=state.get("session_metadata") or {},
        )

        plan: Optional[TurnPlan] = None

        # 1. Deterministic pre-gate (no LLM) for trivially obvious turns.
        if not query.strip():
            plan = self._conversational_plan(query, reasoning="Empty query.")
        elif self._is_pure_greeting(query) and not has_images:
            plan = self._conversational_plan(query, reasoning="Pure greeting heuristic.")

        # 2. LLM planner over the full conversation.
        if plan is None and self._understanding is not None:
            try:
                from services.generated_image_context import session_has_generated_diagram

                has_prior_diagram = session_has_generated_diagram(
                    remembered_generated_images=state.get("remembered_generated_images"),
                    conversation_history=history,
                    message_modality_timeline=state.get("message_modality_timeline"),
                )
                plan, usage = await self._understanding.understand(
                    query,
                    conversation_history=history,
                    agent_mode=agent_mode,
                    has_uploaded_images=has_images,
                    session_has_generated_diagram=has_prior_diagram,
                    topic_hint=(turn_ctx.last_user_topic or ""),
                )
                if plan is not None:
                    self._accumulate_tokens(updates, model_name, usage)
                    updates["llm_calls"] = updates.get("llm_calls", 0) + 1
            except Exception as exc:
                logger.warning("Turn orchestrator LLM failed: %s", exc)
                plan = None

        # 3. Heuristic fallback when the LLM is off/failed.
        if plan is None:
            plan = await self._fallback_plan(
                updates,
                model_name,
                query=query,
                history=history,
                turn_ctx=turn_ctx,
                is_time_sensitive=is_time_sensitive_kw,
                is_current_affairs=is_current_affairs_kw,
            )

        # Merge freshness guardrails into the plan (LLM OR keyword).
        is_time_sensitive = bool(plan.is_time_sensitive or is_time_sensitive_kw)
        is_current_affairs = bool(is_current_affairs_kw)
        if is_time_sensitive or is_current_affairs:
            plan.needs_web = True
            plan.is_time_sensitive = is_time_sensitive

        standalone = (plan.standalone_query_en or "").strip() or query

        updates["turn_plan"] = plan.model_dump()
        updates["query_understanding"] = plan.model_dump()
        updates["query_type"] = plan.query_type
        updates["query_en"] = standalone
        updates["subjects"] = plan.subjects or ["General"]
        updates["wants_visual_output"] = bool(plan.wants_new_image)
        updates["wants_flowchart"] = bool(plan.wants_flowchart)
        updates["response_strategy"] = plan.response_strategy or turn_ctx.response_strategy
        if plan.preferred_response_format:
            updates["preferred_response_format"] = plan.preferred_response_format
        updates["effective_query"] = turn_ctx.effective_query
        updates["context_resolution_block"] = turn_ctx.context_block
        updates["is_context_resolved_followup"] = turn_ctx.is_contextual_followup
        updates["is_time_sensitive_query"] = is_time_sensitive
        updates["is_current_affairs_query"] = is_current_affairs

        logger.info(
            "TurnPlan: route=%s query_type=%s needs_web=%s wants_image=%s conf=%.2f topic=%r",
            plan.route,
            plan.query_type,
            plan.needs_web,
            plan.wants_new_image,
            plan.confidence or 0.0,
            (plan.resolved_topic or "")[:80],
        )

        updates["timings"] = {"orchestrate_turn": perf_counter() - start}
        return updates


__all__ = ["OrchestratorNode"]
