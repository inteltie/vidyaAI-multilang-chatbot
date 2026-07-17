"""Vision agent: describe uploaded images for interactive tutoring."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import settings
from services.agent_findings import publish
from services.image_input import build_effective_query, prepare_images_for_analysis
from services.input_image_store import attach_input_image_urls
from services.costing import merge_patches, patch_component_usage

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


class ImageAnalyzerAgent:
    """Analyze images (S3 URL or base64) and produce educational context for the tutor."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raw_images = state.get("images") or []
        if not raw_images:
            return {}

        query = (state.get("query") or "").strip()
        prepared = await prepare_images_for_analysis(raw_images)
        upload_id = uuid.uuid4().hex[:12]
        session_id = str(state.get("user_session_id") or "").strip()

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are an educational vision assistant. Describe what appears in the "
                    f"attached image(s) to help a tutor understand the student's question.\n"
                    f"Student query: {query}\n\n"
                    "Rules:\n"
                    "- Do NOT answer the question or teach.\n"
                    "- Extract visible text, labels, equations, diagram structure, and subject hints.\n"
                    "- For flowcharts: list stages left-to-right or top-to-bottom; quote visible labels verbatim.\n"
                    "- If text is too small to read, say so; do not invent labels.\n"
                    "- Do not interpret medical scans (X-ray, CT, MRI); only describe non-medical homework visuals.\n"
                    "- Note ambiguity if unclear.\n"
                    "- Respond as JSON only:\n"
                    '{"image_context": "...", "images": [{"description": "...", "summary_line": "..."}]}\n'
                    "One entry in images per attached image, in the same order.\n"
                    "Each description must refer ONLY to that single image — do not mix content from other images.\n"
                    "image_context may briefly relate multiple images but must not invent details."
                ),
            }
        ]
        vision_detail = settings.vision_detail if settings.vision_detail in ("low", "high", "auto") else "high"
        for item in prepared:
            # Always send inline base64 to OpenAI — private S3 URLs are not fetchable by the API.
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": item["data_url"], "detail": vision_detail},
                }
            )

        response = await self._llm.ainvoke([HumanMessage(content=content)])
        raw = (response.content or "").strip()

        image_context = ""
        per_image: List[Dict[str, str]] = []
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            image_context = str(parsed.get("image_context") or "")
            per_image = list(parsed.get("images") or [])
        except (json.JSONDecodeError, TypeError):
            image_context = raw
            per_image = [{"description": raw, "summary_line": raw[:200]}]

        descriptions: List[str] = []
        records: List[Dict[str, Any]] = []
        now_iso = datetime.now(IST).isoformat()

        for idx, item in enumerate(prepared):
            img_meta = per_image[idx] if idx < len(per_image) else {}
            desc = str(img_meta.get("description") or "").strip()
            if not desc and len(prepared) == 1:
                desc = str(image_context or "Image content").strip()
            if not desc:
                desc = f"Image {idx + 1}: content could not be read clearly."
            summary = str(img_meta.get("summary_line") or desc[:200])
            image_id = uuid.uuid4().hex[:12]
            descriptions.append(desc)
            records.append(
                {
                    "image_id": image_id,
                    "upload_id": upload_id,
                    "user_query": query,
                    "url": str(item.get("source_url") or "").strip(),
                    "descriptions": [desc],
                    "summary_line": summary,
                    "image_context": desc,
                    "uploaded_at": now_iso,
                }
            )

        if session_id:
            records = await attach_input_image_urls(session_id, prepared, records)

        effective_query = build_effective_query(query, image_context)

        usage = getattr(response, "usage_metadata", None) or getattr(
            response, "response_metadata", {}
        ).get("token_usage", {})
        i_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        o_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        model_name = getattr(self._llm, "model_name", settings.model_name)

        findings = publish(
            state.get("agent_findings"),
            "image_analyzer",
            "visual",
            image_context[:120] if image_context else "Images analyzed",
            {
                "image_context": image_context,
                "descriptions": descriptions,
                "image_ids": [r["image_id"] for r in records],
                "upload_id": upload_id,
                "summary_line": records[0]["summary_line"] if records else "",
            },
            priority=10,
        )

        base = {
            "image_descriptions": descriptions,
            "image_context": image_context,
            "effective_query": effective_query,
            "new_uploaded_image_records": records,
            "has_images": True,
            "agent_findings": findings,
            "input_tokens": i_tokens,
            "output_tokens": o_tokens,
            "aux_input_tokens": i_tokens,
            "aux_output_tokens": o_tokens,
            "llm_calls": 1,
        }
        return merge_patches(
            base,
            patch_component_usage(
                "vision",
                model_name,
                i_tokens,
                o_tokens,
                images_analyzed=len(prepared),
            ),
        )
