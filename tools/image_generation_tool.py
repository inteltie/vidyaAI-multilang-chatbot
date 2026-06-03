"""Google Gemini image generation for detailed realistic educational visuals."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from services.image_prompt_composer import college_grade_style

logger = logging.getLogger(__name__)

_BACKEND = "google"

_LABEL_SPELLING_RULES = (
    "CRITICAL LABEL SPELLING (non-negotiable):\n"
    "- Every on-image label and heading MUST be spelled correctly in standard English.\n"
    "- ALWAYS use the correct scientific/engineering spelling even if the specification or the "
    "original request contains a typo (e.g. render 'Photosynthesis', never 'Photosunthesis'; "
    "'Mitochondria', never 'Mitocondria'). Silently correct misspellings—do not copy them.\n"
    "- Keep the meaning of each specified label, but fix any spelling/grammar errors in it.\n"
    "- Double-check engineering terms (e.g. torque, stress, modulus, impedance, PID, PLC, "
    "turbulence, Reynolds number, heat flux, microcontroller, gearbox, weld joint).\n"
    "- Prefer short callout labels (2-5 words). If unsure of spelling, use a simpler correct synonym.\n"
    "- No typos, no invented abbreviations, no garbled text.\n"
)

_ENGINEERING_DETAIL_RULES = (
    "ENGINEERING-LEVEL DESCRIPTIVE CONTENT (required for university engineering students):\n"
    "- Highly detailed realistic illustration: photorealistic or high-fidelity technical render "
    "(engineering textbook / lab manual quality)—not a simple schematic or icon.\n"
    "- Show every subsystem from the specification: components, interfaces, materials, fasteners, "
    "sensors, wiring/piping, forces, fluid/heat/electrical paths as applicable.\n"
    "- Prefer cutaways, exploded views, or sectional views when they teach mechanism or assembly.\n"
    "- Include at least 10–14 readable callout labels with correct technical terms and SI units "
    "where standard; leader lines to exact parts.\n"
    "- Depict scale, orientation, and spatial relationships; avoid unexplained blank regions.\n"
    "- Do NOT produce: minimalist icon, single-block flowchart, clipart, stick figure, or "
    "oversimplified diagram with fewer than 8 labeled parts.\n"
)

_INTERACTIVE_CURIOSITY_RULES = (
    "INTERACTIVE & CURIOSITY-DRIVEN LAYOUT (required):\n"
    "- Design the image as an explorable engineering poster, not a static slide.\n"
    "- Render every exploration zone as a bold circled number (1, 2, 3, …) on the correct "
    "physical region, each with a distinct accent color and a short nearby label.\n"
    "- Use visual hooks: arrows pointing to 'discover here' areas, subtle highlights, cutaways "
    "that reveal hidden mechanisms, before/after contrasts, or question-mark cues on ambiguous parts.\n"
    "- Include the interactive_title banner text legibly at the top if specified.\n"
    "- Make regions visually scannable so a student wants to ask 'what is zone 3 doing?'.\n"
    "- Keep numbers large and readable; do not overlap zone badges.\n"
)


def google_image_fee_usd() -> float:
    """Per-image USD estimate for Gemini image output."""
    return float(settings.image_generation_fee_per_call_usd or 0.039)


def _build_educational_prompt(
    prompt: str,
    *,
    grade: str,
    subject_hint: str,
    visual_kind: str = "illustration",
) -> str:
    subject_line = f" Subject context: {subject_hint}." if subject_hint else ""
    complexity = college_grade_style(grade)
    if (visual_kind or "").strip().lower() == "flowchart":
        return (
            "Clear labeled process flowchart for university engineering students "
            f"(B.Tech/B.E. level). {complexity}{subject_line}\n"
            "Generate ONE high-quality 2D flowchart: start/end, process boxes, decision diamonds, "
            "directed arrows, readable English labels on every step. Logical left-to-right or "
            "top-to-bottom flow. School-appropriate only; no watermarks, no logos.\n"
            f"{_LABEL_SPELLING_RULES}\n"
            f"Flowchart specification:\n{prompt}"
        )
    return (
        "Detailed realistic engineering illustration for university engineering students "
        "(B.Tech/B.E. level). "
        f"{complexity}{subject_line}\n"
        "Generate ONE high-quality, maximally descriptive technical visual that teaches through "
        "realistic component-level detail—not a simple schematic or block diagram. "
        "Prioritize subsystem labels, interfaces, materials, and accurate engineering structure. "
        "School-appropriate only; no watermarks, no logos, no gore.\n"
        f"{_ENGINEERING_DETAIL_RULES}\n"
        f"{_INTERACTIVE_CURIOSITY_RULES}\n"
        f"{_LABEL_SPELLING_RULES}\n"
        f"Visual specification:\n{prompt}"
    )


def _build_image_config():
    """Build Gemini ImageConfig from settings."""
    from google.genai import types

    size = (settings.google_image_size or "1K").strip().upper()
    if size not in {"512", "1K", "2K", "4K"}:
        size = "1K"
    aspect = (settings.google_image_aspect_ratio or "16:9").strip()
    return types.ImageConfig(aspect_ratio=aspect, image_size=size)


def _iter_response_parts(response: Any) -> List[Any]:
    """
    Collect content parts from a GenerateContent response.

    Prefer the first candidate's content parts only. Also reading response.parts
    duplicates the same inline images (Gemini SDK exposes both views).
    """
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        candidate_parts = getattr(content, "parts", None) if content else None
        if candidate_parts:
            return list(candidate_parts)

    direct = getattr(response, "parts", None)
    if direct:
        return list(direct)
    return []


def _dedupe_images(images: List[Dict[str, str]], *, max_images: int = 1) -> List[Dict[str, str]]:
    """Drop duplicate inline images (same bytes) and cap count per request."""
    seen: set[str] = set()
    unique: List[Dict[str, str]] = []
    for img in images:
        data = (img.get("data") or "").strip()
        if not data:
            continue
        fingerprint = data[:256] if len(data) > 256 else data
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(img)
        if len(unique) >= max_images:
            break
    if len(images) > len(unique):
        logger.info(
            "Deduplicated Gemini image parts: %d -> %d",
            len(images),
            len(unique),
        )
    return unique


def _bytes_to_b64(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        return base64.b64encode(raw).decode("ascii")
    return ""


def _field(obj: Any, key: str) -> Any:
    """Read either attribute-style SDK fields or dict-style payload fields."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_image_from_part(part: Any) -> Optional[Dict[str, str]]:
    """Extract one inline image from a Gemini response part, if present."""
    inline = _field(part, "inline_data") or _field(part, "inlineData")
    if inline is not None:
        raw = _field(inline, "data")
        if raw:
            mime = _field(inline, "mime_type") or _field(inline, "mimeType") or "image/png"
            b64 = _bytes_to_b64(raw)
            if b64:
                return {"mime_type": mime, "data": b64}

    file_data = _field(part, "file_data") or _field(part, "fileData")
    if file_data is not None:
        raw_bytes = _field(file_data, "data")
        if raw_bytes:
            mime = _field(file_data, "mime_type") or _field(file_data, "mimeType") or "image/png"
            b64 = _bytes_to_b64(raw_bytes)
            if b64:
                return {"mime_type": mime, "data": b64}

    # google-genai SDK helper when available
    try:
        as_image = part.as_image()
        if as_image is not None:
            import io

            buf = io.BytesIO()
            as_image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            if b64:
                return {"mime_type": "image/png", "data": b64}
    except Exception:
        pass

    return None


def _parse_gemini_response(
    response: Any,
    *,
    max_images: int = 1,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Return (images with base64 data, text caption parts)."""
    images: List[Dict[str, str]] = []
    text_parts: List[str] = []

    for part in _iter_response_parts(response):
        extracted = _extract_image_from_part(part)
        if extracted:
            images.append(extracted)
            continue
        text = _field(part, "text")
        if text:
            text_parts.append(str(text).strip())

    images = _dedupe_images(images, max_images=max_images)
    return images, text_parts


async def _call_gemini_image(
    client: Any,
    model: str,
    full_prompt: str,
    *,
    modalities: List[str],
) -> Any:
    from google.genai import types

    image_config = _build_image_config()
    config = types.GenerateContentConfig(
        response_modalities=modalities,
        image_config=image_config,
    )
    timeout_s = float(settings.image_generation_timeout_s or 90.0)
    return await asyncio.wait_for(
        client.aio.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)],
                )
            ],
            config=config,
        ),
        timeout=timeout_s,
    )


class ImageGenerationTool:
    """Generate educational diagrams via Google Gemini (default: gemini-3.1-flash-image)."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = (api_key or settings.gemini_api_key or "").strip()

    async def execute(
        self,
        prompt: str,
        *,
        grade: str = "B",
        subject_hint: str = "",
        visual_kind: str = "illustration",
    ) -> Tuple[str, List[Dict[str, str]], int, int, bool, str, str, float, str, str]:
        """
        Returns (caption, images, input_tokens, output_tokens, billable,
                 backend, model_used, image_fee_usd).
        """
        return await self._execute_google(
            prompt,
            grade=grade,
            subject_hint=subject_hint,
            visual_kind=visual_kind,
        )

    async def _execute_google(
        self,
        prompt: str,
        *,
        grade: str,
        subject_hint: str,
        visual_kind: str = "illustration",
    ) -> Tuple[str, List[Dict[str, str]], int, int, bool, str, str, float, str, str]:
        model = settings.google_image_model or "gemini-3.1-flash-image"
        api_key = self._api_key

        if not api_key:
            logger.error("GEMINI_API_KEY is not configured")
            return (
                "Image generation is not configured. Please set GEMINI_API_KEY.",
                [],
                0,
                0,
                False,
                _BACKEND,
                model,
                0.0,
                "disabled",
                "missing_gemini_api_key",
            )

        full_prompt = _build_educational_prompt(
            prompt,
            grade=grade,
            subject_hint=subject_hint,
            visual_kind=visual_kind,
        )
        logger.info(
            "Gemini image generation model=%s kind=%s prompt=%s",
            model,
            visual_kind,
            prompt[:80],
        )

        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            image_config = _build_image_config()
            logger.info(
                "Gemini image config aspect_ratio=%s image_size=%s",
                image_config.aspect_ratio,
                image_config.image_size,
            )
            response = await _call_gemini_image(
                client,
                model,
                full_prompt,
                modalities=["IMAGE", "TEXT"],
            )
            images, text_parts = _parse_gemini_response(response)
            if not images:
                part_kinds = [
                    type(p).__name__
                    for p in _iter_response_parts(response)
                ]
                logger.warning(
                    "Gemini returned no image on first attempt (part_types=%s); retry IMAGE-only",
                    part_kinds,
                )
                response_retry = await _call_gemini_image(
                    client,
                    model,
                    full_prompt,
                    modalities=["IMAGE"],
                )
                retry_images, retry_text = _parse_gemini_response(response_retry)
                if retry_images:
                    images = retry_images
                    if retry_text:
                        text_parts = retry_text
                else:
                    text_parts = text_parts or retry_text
        except asyncio.TimeoutError:
            timeout_s = float(settings.image_generation_timeout_s or 90.0)
            logger.error(
                "Gemini image generation timed out after %.0fs model=%s",
                timeout_s,
                model,
            )
            return (
                "Image generation took too long and was stopped. Please try again "
                "(a simpler diagram or flowchart request may be faster).",
                [],
                0,
                0,
                False,
                _BACKEND,
                model,
                0.0,
                "provider_timeout",
                "gemini_timeout",
            )
        except Exception as exc:
            logger.exception("Gemini image generation failed: %s", exc)
            return (
                "I could not generate a diagram right now. Please try again in a moment.",
                [],
                0,
                0,
                False,
                _BACKEND,
                model,
                0.0,
                "provider_error",
                "provider_exception",
            )

        billable = bool(images)
        fee = google_image_fee_usd() if billable else 0.0

        if images:
            # Prefer planner student_response; only pass through non-empty model text.
            caption = " ".join(text_parts).strip()
        else:
            caption = " ".join(text_parts).strip() or (
                "I could not generate a diagram for that request. "
                "Try rephrasing with the topic and diagram type you want."
            )

        status = "success" if images else "no_image_returned"
        reason = "" if images else "model_returned_no_image_parts"
        return caption, images, 0, 0, billable, _BACKEND, model, fee, status, reason


# Backward-compatible alias for tests
gpt_image_2_fee_usd = google_image_fee_usd

__all__ = [
    "ImageGenerationTool",
    "google_image_fee_usd",
    "gpt_image_2_fee_usd",
    "_parse_gemini_response",
]
