"""Persist user-provided chat image URLs (S3 upload or passthrough)."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, List

from services.s3_storage import s3_configured, upload_image_bytes

logger = logging.getLogger(__name__)


async def attach_input_image_urls(
    session_id: str,
    prepared_images: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Set ``url`` on each uploaded-image record.

    - URL input (already on S3): reuse ``source_url``, no re-upload.
    - Base64 input: upload to S3 when configured.
    """
    if not prepared_images or not records:
        return records

    updated: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        rec = dict(record)
        if idx >= len(prepared_images):
            updated.append(rec)
            continue

        item = prepared_images[idx]
        source_url = str(item.get("source_url") or "").strip()
        skip_upload = bool(item.get("skip_s3_upload"))

        if skip_upload and source_url:
            rec["url"] = source_url
            updated.append(rec)
            continue

        if not s3_configured():
            updated.append(rec)
            continue

        raw_b64 = (item.get("data") or "").strip()
        mime_type = (item.get("mime_type") or "image/png").strip().lower()
        image_id = str(rec.get("image_id") or "").strip()
        upload_id = str(rec.get("upload_id") or "").strip()
        if not raw_b64 or not image_id or not session_id:
            updated.append(rec)
            continue

        try:
            image_bytes = base64.b64decode(raw_b64, validate=True)
        except Exception as exc:
            logger.warning(
                "attach_input_image_urls: invalid base64 session=%s image_id=%s: %s",
                session_id,
                image_id,
                exc,
            )
            updated.append(rec)
            continue

        url = await asyncio.to_thread(
            upload_image_bytes,
            image_bytes,
            mime_type,
            session_id=session_id,
            image_id=image_id,
            kind="upload",
            upload_id=upload_id,
        )
        if url:
            rec["url"] = url
        updated.append(rec)

    return updated


# Backward-compatible alias
upload_input_images = attach_input_image_urls

__all__ = ["attach_input_image_urls", "upload_input_images"]
