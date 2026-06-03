"""LangGraph node: route_decision.

This node is now a *pure, deterministic* mapper from the context-aware
``TurnPlan`` (produced by ``orchestrate_turn``) to a single ``orchestration_route``.
It contains NO keyword scans for intent — that decision lives in the orchestrator.
Its only job is to enforce HARD policy guardrails that must never be left to the
LLM (image uploads, agent-mode gating for generation, and safety) and translate
the plan into the route value consumed by ``retrieve_documents`` and ``graph``.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict

from config import settings
from services.image_modality_policy import (
    evaluate_image_modality,
    is_image_generation_blocked,
    looks_like_image_generation_query,
)
from state import AgentState


class RouteDecisionNode:
    """Map the TurnPlan + hard policy gates to a single orchestration route."""

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        plan: Dict[str, Any] = state.get("turn_plan") or {}
        query = str(state.get("query") or "")
        # query_type may have been refined by deterministic overrides in analyze_query.
        query_type = str(state.get("query_type") or plan.get("query_type") or "curriculum_specific")
        has_images = bool(state.get("has_images") or (state.get("images") or []))
        agent_mode = str(state.get("agent_mode") or "standard").strip().lower()
        history = state.get("conversation_history") or []

        explain_existing = bool(plan.get("wants_image_explanation"))
        # Generation intent: primarily the orchestrator plan, with a keyword guardrail
        # so an explicit "draw/generate ... diagram" is never missed (defense in depth).
        wants_visual = bool(
            state.get("wants_visual_output")
            or plan.get("wants_new_image")
            or query_type == "image_generation"
            or (looks_like_image_generation_query(query) and not explain_existing)
        )
        is_time_sensitive = bool(
            state.get("is_time_sensitive_query") or plan.get("is_time_sensitive")
        )
        is_current_affairs = bool(state.get("is_current_affairs_query"))
        needs_web = bool(plan.get("needs_web") or is_time_sensitive or is_current_affairs)
        needs_curriculum = bool(
            plan.get("needs_curriculum", query_type == "curriculum_specific")
        )
        plan_route = str(plan.get("route") or "")

        # --- Hard guardrail: image modality (uploads, agent-mode, safety) ---
        image_modality = evaluate_image_modality(
            query,
            has_images=has_images,
            agent_mode=agent_mode,
            conversation_history=history,
            wants_visual_output=wants_visual,
            remembered_generated_images=state.get("remembered_generated_images"),
            message_modality_timeline=state.get("message_modality_timeline"),
            suppress_generation=explain_existing,
        )
        # Honor the orchestrator's explicit generation decision even when the
        # keyword policy is conservative, provided the hard gates allow it.
        explicit_generate = bool(
            plan.get("wants_new_image") or query_type == "image_generation"
        )
        if (
            image_modality == "none"
            and explicit_generate
            and not has_images
            and not explain_existing
            and agent_mode == "interactive"
            and settings.image_generation_enabled
        ):
            image_modality = "generate"
        if image_modality == "generate" and is_image_generation_blocked(query)[0]:
            # Safety refusal -> do not generate; answer textually instead.
            image_modality = "none"

        updates: Dict[str, Any] = {"image_modality": image_modality}
        route = "retriever"
        reason = "Default educational route."
        confidence = float(plan.get("confidence") or 0.6)
        fallback_reason = ""

        if has_images or image_modality == "analyze":
            route = "image_analyze"
            reason = "Image input present; run vision analysis."
            confidence = 1.0
        elif image_modality == "generate":
            route = "image_generate"
            reason = "Image generation intent detected."
            confidence = 1.0
            updates["query_type"] = "image_generation"
        elif (wants_visual and not explain_existing) and agent_mode != "interactive":
            # Generation asked for outside interactive mode: answer textually and
            # recommend interactive mode (the translator appends the recommendation).
            route = "retriever"
            reason = (
                "Image generation requested in standard mode; "
                "answer textually and recommend interactive mode."
            )
            confidence = 0.95
            fallback_reason = "interactive_mode_required_for_image_generation"
            if query_type == "image_generation":
                updates["query_type"] = "curriculum_specific"
        elif query_type == "conversational" or plan_route == "conversational":
            route = "none"
            reason = "Conversational query; no retrieval required."
            confidence = 0.9
        elif needs_web and needs_curriculum:
            route = "hybrid"
            reason = "Educational query needing fresh info; retriever + web."
            confidence = 0.85
        elif plan_route == "hybrid" or is_current_affairs or is_time_sensitive:
            route = "hybrid"
            reason = "Time-sensitive/current-affairs query; use retriever + web fallback."
            confidence = 0.85
        elif needs_web or plan_route == "web":
            route = "web"
            reason = "Fresh external information required."
            confidence = 0.7

        updates.update(
            {
                "orchestration_route": route,
                "route_reason": reason,
                "route_confidence": confidence,
                "route_fallback_reason": fallback_reason,
                "timings": {"route_decision": perf_counter() - start},
            }
        )
        return updates


__all__ = ["RouteDecisionNode"]
