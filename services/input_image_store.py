"""Upload user-provided chat images to S3 and attach URLs to analyzer records."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, List

from services.s3_storage import s3_configured, upload_image_bytes

logger = logging.getLogger(__name__)


async def upload_input_images(
    session_id: str,
    decoded_images: List[Dict[str, str]],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Upload decoded input images to S3 and set ``url`` on each matching record.

    Records and decoded_images must be in the same order.
    """
    if not decoded_images or not records or not s3_configured():
        return records

    updated: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        rec = dict(record)
        if idx >= len(decoded_images):
            updated.append(rec)
            continue

        item = decoded_images[idx]
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
                "upload_input_images: invalid base64 session=%s image_id=%s: %s",
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


__all__ = ["upload_input_images"]
