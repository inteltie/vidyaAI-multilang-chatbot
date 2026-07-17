"""Validation and preparation helpers for chat image input (URL or base64)."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from typing import Any, Dict, List

import httpx

from config import settings
from models.domain import is_supported_image_mime, normalize_image_mime
from services.s3_storage import download_image_bytes_from_s3_url

logger = logging.getLogger(__name__)


def build_data_url(mime_type: str, data: str) -> str:
    return f"data:{mime_type};base64,{data}"


def _encode_payload(mime_type: str, payload: bytes) -> Dict[str, str]:
    raw_b64 = base64.b64encode(payload).decode("ascii")
    return {
        "mime_type": mime_type,
        "data": raw_b64,
        "data_url": build_data_url(mime_type, raw_b64),
    }


def _validate_image_payload(payload: bytes, mime: str, idx: int) -> None:
    max_bytes = settings.max_image_bytes
    if len(payload) > max_bytes:
        raise ValueError(f"Image {idx + 1}: exceeds max size of {max_bytes} bytes")

    if mime == "image/jpeg" and not payload.startswith(b"\xff\xd8"):
        raise ValueError(
            f"Image {idx + 1}: data is not a valid JPEG (check encoding or mime_type)"
        )
    if mime == "image/png" and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Image {idx + 1}: data is not a valid PNG")
    if mime == "image/webp" and not (
        payload.startswith(b"RIFF") and len(payload) >= 12 and payload[8:12] == b"WEBP"
    ):
        raise ValueError(f"Image {idx + 1}: data is not a valid WEBP")


def _decode_base64_image(raw_b64: str, mime: str, idx: int) -> bytes:
    raw_b64 = (raw_b64 or "").strip()
    if not raw_b64:
        raise ValueError(f"Image {idx + 1}: empty data")

    if raw_b64.startswith("data:"):
        comma = raw_b64.find(",")
        if comma >= 0:
            raw_b64 = raw_b64[comma + 1 :]

    try:
        payload = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Image {idx + 1}: invalid base64") from exc

    _validate_image_payload(payload, mime, idx)
    return payload


async def _download_https_image(url: str) -> bytes:
    timeout = float(getattr(settings, "image_download_timeout_s", 30.0) or 30.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def _download_image_from_url(url: str, idx: int) -> bytes:
    """Download image bytes from S3 (IAM/boto3) or generic HTTPS."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise ValueError(f"Image {idx + 1}: empty url")

    payload = await asyncio.to_thread(download_image_bytes_from_s3_url, cleaned)
    if payload is None:
        try:
            payload = await _download_https_image(cleaned)
        except httpx.HTTPError as exc:
            raise ValueError(f"Image {idx + 1}: failed to download url") from exc

    if not payload:
        raise ValueError(f"Image {idx + 1}: downloaded image is empty")
    return payload


async def prepare_images_for_analysis(
    images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Prepare images for vision analysis.

    Accepts either:
    - ``url``: HTTPS/S3 link (image already stored; no re-upload needed)
    - ``data``: legacy base64 payload (uploaded to S3 by the service when configured)

    Returns dicts with mime_type, data, data_url, source_url, skip_s3_upload.
    """
    if not images:
        return []

    max_count = settings.max_images_per_request
    if len(images) > max_count:
        raise ValueError(f"At most {max_count} images allowed per request")

    prepared: List[Dict[str, Any]] = []

    for idx, img in enumerate(images):
        raw_mime = (img.get("mime_type") or "").strip().lower()
        if not is_supported_image_mime(raw_mime):
            raise ValueError(f"Image {idx + 1}: unsupported mime_type '{raw_mime}'")
        mime = normalize_image_mime(raw_mime)

        source_url = (img.get("url") or "").strip()
        raw_b64 = (img.get("data") or "").strip()

        if source_url and raw_b64:
            raise ValueError(f"Image {idx + 1}: provide either 'url' or 'data', not both")
        if not source_url and not raw_b64:
            raise ValueError(f"Image {idx + 1}: either 'url' or 'data' is required")

        if source_url:
            payload = await _download_image_from_url(source_url, idx)
            _validate_image_payload(payload, mime, idx)
            item = _encode_payload(mime, payload)
            item["source_url"] = source_url
            item["skip_s3_upload"] = True
        else:
            payload = _decode_base64_image(raw_b64, mime, idx)
            item = _encode_payload(mime, payload)
            item["source_url"] = ""
            item["skip_s3_upload"] = False

        prepared.append(item)

    return prepared


def validate_and_decode_images(images: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Validate legacy base64-only image payloads.

    Prefer :func:`prepare_images_for_analysis` for URL or base64 input.
    """
    if not images:
        return []

    max_count = settings.max_images_per_request
    if len(images) > max_count:
        raise ValueError(f"At most {max_count} images allowed per request")

    decoded: List[Dict[str, str]] = []

    for idx, img in enumerate(images):
        source_url = (img.get("url") or "").strip()
        if source_url:
            raise ValueError(
                f"Image {idx + 1}: url-based images require prepare_images_for_analysis()"
            )

        raw_mime = (img.get("mime_type") or "").strip().lower()
        if not is_supported_image_mime(raw_mime):
            raise ValueError(f"Image {idx + 1}: unsupported mime_type '{raw_mime}'")
        mime = normalize_image_mime(raw_mime)

        payload = _decode_base64_image(str(img.get("data") or ""), mime, idx)
        item = _encode_payload(mime, payload)
        decoded.append(item)

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
