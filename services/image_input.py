"""Validation and encoding helpers for base64 image chat input."""

from __future__ import annotations

import base64
import binascii
from typing import Any, Dict, List

from config import settings
from models.domain import ALLOWED_IMAGE_MIME_TYPES


def build_data_url(mime_type: str, data: str) -> str:
    return f"data:{mime_type};base64,{data}"


def validate_and_decode_images(images: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Validate image payloads and return list of {mime_type, data_url, data}.

    Raises ValueError on invalid mime, empty data, or oversized decode.
    """
    if not images:
        return []

    max_count = settings.max_images_per_request
    if len(images) > max_count:
        raise ValueError(f"At most {max_count} images allowed per request")

    max_bytes = settings.max_image_bytes
    decoded: List[Dict[str, str]] = []

    for idx, img in enumerate(images):
        mime = (img.get("mime_type") or "").strip().lower()
        if mime not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError(f"Image {idx + 1}: unsupported mime_type '{mime}'")

        raw_b64 = (img.get("data") or "").strip()
        if not raw_b64:
            raise ValueError(f"Image {idx + 1}: empty data")

        # Strip data-URL prefix if client sent full URL
        if raw_b64.startswith("data:"):
            comma = raw_b64.find(",")
            if comma >= 0:
                raw_b64 = raw_b64[comma + 1 :]

        try:
            payload = base64.b64decode(raw_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Image {idx + 1}: invalid base64") from exc

        if len(payload) > max_bytes:
            raise ValueError(
                f"Image {idx + 1}: exceeds max size of {max_bytes} bytes"
            )

        if mime == "image/jpeg" and not payload.startswith(b"\xff\xd8"):
            raise ValueError(
                f"Image {idx + 1}: data is not a valid JPEG (check base64 encoding; "
                "do not use jq @base64 on binary files — use Python or file upload tools)"
            )
        if mime == "image/png" and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Image {idx + 1}: data is not a valid PNG")
        if mime == "image/webp" and not (
            payload.startswith(b"RIFF") and len(payload) >= 12 and payload[8:12] == b"WEBP"
        ):
            raise ValueError(f"Image {idx + 1}: data is not a valid WEBP")

        decoded.append(
            {
                "mime_type": mime,
                "data": raw_b64,
                "data_url": build_data_url(mime, raw_b64),
            }
        )

    return decoded


def build_effective_query(query: str, image_context: str) -> str:
    """Merge user query with analyzer image context for memory/RAG."""
    q = (query or "").strip()
    ctx = (image_context or "").strip()
    if not ctx:
        return q
    if not q:
        return f"[Attached images]\n{ctx}"
    return f"{q}\n\n[Attached images]\n{ctx}"
