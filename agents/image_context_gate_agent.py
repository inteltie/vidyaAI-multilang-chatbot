"""Decide how much remembered image context to inject into the tutor prompt."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from config import settings
from services.agent_findings import build_image_inject_block, publish
from services.image_context_policy import evaluate_image_context_policy
from services.costing import patch_component_usage

logger = logging.getLogger(__name__)


class ImageContextGateAgent:
    """Rules-first gate with optional nano fallback for ambiguous cases."""

    def __init__(self, llm_fast: ChatOpenAI | None = None) -> None:
        self._llm_fast = llm_fast
        self._last_nano_usage: Dict[str, Any] = {}

    async def _resolve_ambiguous(
        self, query: str, remembered: List[Dict[str, Any]]
    ) -> str:
        if not self._llm_fast or not settings.image_gate_use_nano:
            return "compact"
        prompt = (
            "Does a tutor need stored image descriptions to answer this student message?\n"
            f'Student message: "{query}"\n'
            f"Session has {len(remembered)} remembered image(s).\n"
            'Reply JSON only: {"inject_level": "none"|"compact"|"full"}'
        )
        try:
            resp = await self._llm_fast.ainvoke(prompt)
            raw = (resp.content or "").strip()
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "", 1)
            parsed = json.loads(raw.strip())
            level = str(parsed.get("inject_level", "compact")).lower()
            if level in ("none", "compact", "full"):
                self._last_nano_usage = getattr(resp, "usage_metadata", None) or getattr(
                    resp, "response_metadata", {}
                ).get("token_usage", {})
                return level
        except Exception as exc:
            logger.debug("Image gate nano fallback failed: %s", exc)
        self._last_nano_usage = {}
        return "compact"

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        remembered_prior = list(state.get("remembered_images") or [])
        new_records = list(state.get("new_uploaded_image_records") or [])
        # Only deduplicate on image_id when it is present; records lacking an
        # image_id are treated as distinct rather than collapsing to one key.
        prior_ids = {str(r.get("image_id")) for r in remembered_prior if r.get("image_id")}
        remembered = remembered_prior + [
            r
            for r in new_records
            if not r.get("image_id") or str(r.get("image_id")) not in prior_ids
        ]

        history = state.get("conversation_history") or []
        turn_type = state.get("interactive_turn_type") or "opening"
        has_new_upload = bool(state.get("images")) or bool(new_records)

        decision = evaluate_image_context_policy(
            query=state.get("query") or "",
            has_new_images=has_new_upload,
            remembered_images=remembered,
            new_upload_records=new_records,
            conversation_history=history,
            is_topic_shift=bool(state.get("is_topic_shift")),
            turn_type=turn_type,
        )

        inject_level = decision.inject_level
        nano_usage_patch: Dict[str, Any] = {}
        if inject_level == "ambiguous":
            self._last_nano_usage = {}
            inject_level = await self._resolve_ambiguous(
                state.get("query") or "", remembered
            )
            decision.reason = f"Nano resolved ambiguous → {inject_level}"
            usage = self._last_nano_usage or {}
            i_t = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            o_t = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            if i_t or o_t:
                gate_model = getattr(self._llm_fast, "model_name", settings.analyzer_model_name)
                nano_usage_patch = patch_component_usage("image_gate", gate_model, i_t, o_t)

        visual_payload = {}
        for f in state.get("agent_findings") or []:
            if f.get("kind") == "visual":
                visual_payload = dict(f.get("payload") or {})
                break

        block = build_image_inject_block(
            inject_level,
            remembered,
            decision.selected_image_ids,
            current_visual=visual_payload,
        )

        findings = publish(
            state.get("agent_findings"),
            "image_gate",
            "visual_policy",
            decision.reason,
            {
                "inject_level": inject_level,
                "selected_image_ids": decision.selected_image_ids,
                "image_inject_block": block,
                "reason": decision.reason,
            },
            priority=8,
        )

        result = {
            "image_inject_level": inject_level,
            "image_inject_block": block,
            "image_gate_reason": decision.reason,
            "agent_findings": findings,
        }
        if nano_usage_patch:
            usage = nano_usage_patch.get("usage_by_component", {}).get("image_gate", {})
            gate_model = next(
                (k for k in usage if k not in ("calls", "images")),
                settings.analyzer_model_name,
            )
            model_usage = usage.get(gate_model, {})
            result.update(nano_usage_patch)
            result["input_tokens"] = int(model_usage.get("input", 0) or 0)
            result["output_tokens"] = int(model_usage.get("output", 0) or 0)
            result["aux_input_tokens"] = result["input_tokens"]
            result["aux_output_tokens"] = result["output_tokens"]
            result["llm_calls"] = 1
        return result
