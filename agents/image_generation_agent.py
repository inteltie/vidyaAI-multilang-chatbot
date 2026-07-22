"""Agent that generates detailed educational illustrations for interactive mode (Google Gemini)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from services.costing import merge_patches, patch_component_usage
from services.generated_image_store import save_generated_image
from services.image_modality_policy import (
    is_contextual_image_generation_request,
    is_explicit_image_generation_request,
    is_image_generation_blocked,
)
from services.image_prompt_composer import _fallback_plan
from services.image_prompt_composer import (
    ImageGenerationPlanner,
    build_fallback_exploration_zones,
    enrich_student_response_for_exploration,
    exploration_zones_to_dicts,
    format_brief_as_prompt,
)
from tools.image_generation_tool import ImageGenerationTool

logger = logging.getLogger(__name__)


class ImageGenerationAgent:
    """Generate images from student queries (interactive mode only)."""

    def __init__(
        self,
        tool: ImageGenerationTool | None = None,
        planner: ImageGenerationPlanner | None = None,
        llm: ChatOpenAI | None = None,
    ) -> None:
        self._tool = tool or ImageGenerationTool()
        self._planner = planner or ImageGenerationPlanner(llm=llm)

    _GENERIC_CAPTION_PHRASES = (
        "based on your request",
        "educational diagram",
        "you can use it to study",
        "click on",
        "click different",
        "tap on",
    )

    @classmethod
    def _is_generic_caption(cls, caption: str) -> bool:
        lower = caption.lower()
        return any(phrase in lower for phrase in cls._GENERIC_CAPTION_PHRASES)

    @classmethod
    def _merge_response(cls, student_response: str, tool_caption: str) -> str:
        student = (student_response or "").strip()
        caption = (tool_caption or "").strip()
        if caption and cls._is_generic_caption(caption):
            caption = ""
        if student and caption and caption.lower() not in student.lower():
            return f"{student}\n\n{caption}"
        return student or caption

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = (state.get("query") or "").strip()
        agent_mode = str(state.get("agent_mode") or "standard").strip().lower()
        grade = str(state.get("student_grade") or "B")
        subjects = state.get("subjects") or []
        subject_hint = ", ".join(subjects[:3]) if isinstance(subjects, list) else ""

        session_id = str(state.get("user_session_id") or "").strip()

        if agent_mode != "interactive":
            return {
                "response": (
                    "Image generation is available in interactive mode. "
                    "Switch `agent_mode` to `interactive` and try again."
                ),
                "generated_images": [],
                "new_generated_image_records": [],
                "image_generation_calls": 0,
                "image_generation_backend": "google",
                "image_model_used": "",
                "image_generation_status": "mode_restricted",
                "image_generation_error_reason": "interactive_mode_required",
                "llm_calls": 0,
            }

        blocked, reason = is_image_generation_blocked(query)
        if blocked:
            return {
                "response": (
                    "I can only create school-appropriate educational illustrations. "
                    f"{reason}"
                ),
                "generated_images": [],
                "new_generated_image_records": [],
                "image_generation_calls": 0,
                "image_generation_backend": "google",
                "image_model_used": "",
                "image_generation_status": "blocked",
                "image_generation_error_reason": "safety_policy_blocked",
                "llm_calls": 0,
            }

        plan = await self._planner.plan(state)
        llm_calls = 1 if (plan.input_tokens or plan.output_tokens) else 0

        explicit_gen = (
            bool(state.get("wants_visual_output"))
            or is_explicit_image_generation_request(query)
            or is_contextual_image_generation_request(
                query, state.get("conversation_history")
            )
        )
        if explicit_gen and plan.action != "generate_diagram":
            logger.warning(
                "Planner chose action=%s for explicit image request; forcing generate_diagram via fallback brief",
                plan.action,
            )
            plan = _fallback_plan(state)
            plan.input_tokens = plan.input_tokens or 0
            plan.output_tokens = plan.output_tokens or 0
            llm_calls = max(llm_calls, 1)

        if plan.action != "generate_diagram" or not plan.brief:
            status = "clarification_needed" if plan.action == "ask_clarification" else "text_only"
            patches: list[Dict[str, Any]] = [
                {
                    "response": plan.student_response,
                    "generated_images": [],
                    "new_generated_image_records": [],
                    "image_generation_backend": "google",
                    "image_model_used": "",
                    "image_generation_status": status,
                    "image_generation_error_reason": "",
                    "llm_calls": llm_calls,
                    "input_tokens": plan.input_tokens,
                    "output_tokens": plan.output_tokens,
                },
            ]
            if plan.model_name and (plan.input_tokens or plan.output_tokens):
                patches.append(
                    patch_component_usage(
                        "chat",
                        plan.model_name,
                        plan.input_tokens,
                        plan.output_tokens,
                    )
                )
            return merge_patches(*patches)

        assert plan.brief is not None
        if not plan.brief.exploration_zones:
            plan.brief.exploration_zones = build_fallback_exploration_zones(plan.brief)

        generation_prompt = format_brief_as_prompt(plan.brief)
        exploration_payload = exploration_zones_to_dicts(plan.brief.exploration_zones)
        logger.info(
            "Image generation planned action=%s source=%s topic=%s",
            plan.action,
            plan.brief.source,
            plan.brief.topic[:120],
        )

        (
            caption,
            images,
            i_tokens,
            o_tokens,
            billable,
            backend,
            model_used,
            image_fee_usd,
            gen_status,
            gen_reason,
        ) = await self._tool.execute(
            generation_prompt,
            grade=grade,
            subject_hint=subject_hint,
            visual_kind="flowchart" if state.get("wants_flowchart") else "illustration",
        )

        if not images:
            logger.warning(
                "Image generation produced no image bytes (status=%s reason=%s topic=%s)",
                gen_status,
                gen_reason,
                plan.brief.topic[:80],
            )
            response = (
                "I couldn't render the illustration this time—the image service returned no picture. "
                "Please try again in a moment, or rephrase with the topic you want illustrated "
                f"(e.g. a labeled diagram of {plan.brief.topic[:60]})."
            )
        else:
            response = enrich_student_response_for_exploration(
                self._merge_response(plan.student_response, caption),
                plan.brief.exploration_zones,
            )

        new_records: list[Dict[str, Any]] = []
        for img in images:
            img.setdefault("prompt_used", generation_prompt)
            img["exploration_zones"] = exploration_payload
            if plan.brief.interactive_title:
                img["interactive_title"] = plan.brief.interactive_title
            if not img.get("data") or not session_id:
                continue
            stored = save_generated_image(
                session_id,
                img["data"],
                img.get("mime_type", "image/png"),
                prompt_used=img.get("prompt_used", generation_prompt),
            )
            if stored.get("image_id"):
                img["image_id"] = stored["image_id"]
            if stored.get("url"):
                img["url"] = stored["url"]
            if stored.get("created_at"):
                img["created_at"] = stored["created_at"]
            if stored.get("url"):
                new_records.append(
                    {
                        "image_id": stored["image_id"],
                        "url": stored["url"],
                        "mime_type": stored.get("mime_type", img.get("mime_type", "image/png")),
                        "prompt_used": stored.get("prompt_used", generation_prompt),
                        "created_at": stored.get("created_at", ""),
                        "exploration_zones": exploration_payload,
                        "interactive_title": plan.brief.interactive_title or "",
                    }
                )

        logger.info(
            "Image generation complete backend=%s model=%s images=%d fee_usd=%.4f",
            backend,
            model_used,
            len(images),
            image_fee_usd,
        )

        patches = [
            {
                "response": response,
                "generated_images": images,
                "new_generated_image_records": new_records,
                "image_generation_backend": backend,
                "image_model_used": model_used,
                "image_generation_status": gen_status,
                "image_generation_error_reason": gen_reason,
                "llm_calls": llm_calls,
                "input_tokens": plan.input_tokens + i_tokens,
                "output_tokens": plan.output_tokens + o_tokens,
            },
        ]

        if plan.model_name and (plan.input_tokens or plan.output_tokens):
            patches.append(
                patch_component_usage(
                    "chat",
                    plan.model_name,
                    plan.input_tokens,
                    plan.output_tokens,
                )
            )

        if billable:
            patches.append(
                {
                    "usage_by_component": {
                        "image_generation": {
                            "calls": 1,
                            "fee_usd": image_fee_usd,
                            "backend": backend,
                            "model": model_used,
                        },
                    },
                }
            )

        return merge_patches(*patches)
