"""Persist generated diagram bytes to local disk and build public URLs."""

from __future__ import annotations

import base64
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9_-]+")

_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}


def _sanitize_segment(value: str, *, fallback: str = "unknown") -> str:
    cleaned = _SAFE_SEGMENT.sub("", (value or "").strip())
    return cleaned or fallback


def _extension_for_mime(mime_type: str) -> str:
    normalized = (mime_type or "image/png").strip().lower()
    return _MIME_EXT.get(normalized, ".png")


def ensure_generated_image_dir() -> Path:
    """Create the generated image root directory if missing."""
    root = Path(settings.generated_image_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _public_url(session_id: str, image_id: str, ext: str) -> str:
    base = (settings.public_base_url or "").rstrip("/")
    safe_session = _sanitize_segment(session_id)
    safe_image = _sanitize_segment(image_id)
    return f"{base}/generated/{safe_session}/{safe_image}{ext}"


def save_generated_image(
    session_id: str,
    b64_data: str,
    mime_type: str,
    *,
    prompt_used: str = "",
) -> Dict[str, Any]:
    """
    Decode base64 image data, write to disk, and return metadata for API/Mongo.

    On failure, returns partial metadata (no url) so callers can still return base64.
    """
    image_id = uuid.uuid4().hex[:12]
    safe_session = _sanitize_segment(session_id)
    safe_image = _sanitize_segment(image_id, fallback=image_id)
    ext = _extension_for_mime(mime_type)
    created_at = datetime.now(IST).isoformat()

    meta: Dict[str, Any] = {
        "image_id": safe_image,
        "url": "",
        "mime_type": (mime_type or "image/png").strip().lower() or "image/png",
        "prompt_used": prompt_used or "",
        "created_at": created_at,
    }

    raw_b64 = (b64_data or "").strip()
    if not raw_b64:
        logger.warning("save_generated_image: empty base64 for session=%s", session_id)
        return meta

    try:
        image_bytes = base64.b64decode(raw_b64, validate=True)
    except Exception as exc:
        logger.error("save_generated_image: invalid base64 session=%s: %s", session_id, exc)
        return meta

    try:
        root = ensure_generated_image_dir()
        session_dir = root / safe_session
        session_dir.mkdir(parents=True, exist_ok=True)
        file_path = session_dir / f"{safe_image}{ext}"
        file_path.write_bytes(image_bytes)
        meta["url"] = _public_url(session_id, safe_image, ext)
        logger.info(
            "Saved generated image session=%s image_id=%s path=%s",
            safe_session,
            safe_image,
            file_path,
        )
    except Exception as exc:
        logger.exception(
            "save_generated_image: write failed session=%s image_id=%s: %s",
            session_id,
            safe_image,
            exc,
        )

    return meta


__all__ = [
    "ensure_generated_image_dir",
    "save_generated_image",
]
