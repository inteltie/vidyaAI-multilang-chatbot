"""Upload chat images (user uploads and generated diagrams) to Amazon S3."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import unquote, urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import settings

logger = logging.getLogger(__name__)

_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9_-]+")
_S3_VIRTUAL_HOST_RE = re.compile(
    r"^([a-z0-9][a-z0-9.-]*)\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com$",
    re.IGNORECASE,
)

_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}


def s3_configured() -> bool:
    """True when an S3 bucket is configured (credentials via IAM role or env keys)."""
    return bool((settings.s3_bucket or "").strip())


def _explicit_aws_credentials() -> tuple[str, str]:
    """Return (access_key_id, secret) when both are set for local/dev use."""
    key = (settings.aws_access_key_id or "").strip()
    secret = (settings.aws_secret_access_key or "").strip()
    return key, secret


def _sanitize_segment(value: str, *, fallback: str = "unknown") -> str:
    cleaned = _SAFE_SEGMENT.sub("", (value or "").strip())
    return cleaned or fallback


def _extension_for_mime(mime_type: str) -> str:
    normalized = (mime_type or "image/png").strip().lower()
    return _MIME_EXT.get(normalized, ".png")


def _object_key(
    session_id: str,
    image_id: str,
    ext: str,
    *,
    kind: Literal["upload", "generated"],
    upload_id: str = "",
) -> str:
    prefix = (settings.s3_prefix or "chat-images").strip().strip("/")
    safe_session = _sanitize_segment(session_id)
    safe_image = _sanitize_segment(image_id, fallback=image_id)
    if kind == "upload":
        # e.g. chat-images/uploads/965000/b935825cca97.jpg
        return f"{prefix}/uploads/{safe_session}/{safe_image}{ext}"
    return f"{prefix}/generated/{safe_session}/{safe_image}{ext}"


def _public_url(key: str) -> str:
    base = (settings.s3_public_base_url or "").strip().rstrip("/")
    if base:
        return f"{base}/{key.lstrip('/')}"
    bucket = (settings.s3_bucket or "").strip()
    region = (settings.aws_region or "ap-south-1").strip()
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


@lru_cache(maxsize=1)
def _s3_client():
    """
    Build S3 client.

    - Production: omit AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY; boto3 uses the
      instance/task IAM role via the default credential chain.
    - Local dev: set both keys in .env when not using a shared credentials file.
    """
    region = (settings.aws_region or "ap-south-1").strip()
    client_kwargs: dict = {"region_name": region}
    access_key, secret_key = _explicit_aws_credentials()

    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
        logger.info("S3 client: using explicit AWS credentials from environment")
    elif access_key or secret_key:
        logger.warning(
            "Only one of AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY is set; "
            "falling back to default AWS credential chain (IAM role / ~/.aws/credentials)"
        )
        logger.info("S3 client: using default AWS credential chain")
    else:
        logger.info("S3 client: using default AWS credential chain (IAM role)")

    return boto3.client("s3", **client_kwargs)


def upload_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    *,
    session_id: str,
    image_id: str,
    kind: Literal["upload", "generated"],
    upload_id: str = "",
) -> Optional[str]:
    """
    Upload image bytes to S3 and return the public URL.

    Returns None when S3 is not configured or the upload fails.
    """
    if not image_bytes:
        logger.warning("upload_image_bytes: empty payload session=%s", session_id)
        return None
    if not s3_configured():
        logger.debug("S3 not configured; skipping upload session=%s", session_id)
        return None

    ext = _extension_for_mime(mime_type)
    key = _object_key(
        session_id,
        image_id,
        ext,
        kind=kind,
        upload_id=upload_id,
    )
    content_type = (mime_type or "image/png").strip().lower() or "image/png"

    try:
        client = _s3_client()
        client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=image_bytes,
            ContentType=content_type,
        )
        url = _public_url(key)
        logger.info(
            "Uploaded %s image to S3 session=%s image_id=%s key=%s",
            kind,
            _sanitize_segment(session_id),
            _sanitize_segment(image_id),
            key,
        )
        return url
    except (BotoCoreError, ClientError) as exc:
        logger.exception(
            "S3 upload failed session=%s image_id=%s kind=%s: %s",
            session_id,
            image_id,
            kind,
            exc,
        )
        return None


def parse_s3_https_url(url: str) -> tuple[str, str] | None:
    """
    Parse a virtual-hosted-style S3 HTTPS URL into (bucket, key).

    Supports: https://bucket.s3.region.amazonaws.com/path/to/key
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None

    match = _S3_VIRTUAL_HOST_RE.match(parsed.netloc)
    if not match:
        return None

    bucket = match.group(1)
    key = unquote((parsed.path or "").lstrip("/"))
    if not key:
        return None
    return bucket, key


def download_s3_object(bucket: str, key: str) -> bytes:
    """Download object bytes from S3 using the configured credential chain."""
    client = _s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    body = response.get("Body")
    if body is None:
        raise ValueError(f"S3 object s3://{bucket}/{key} returned no body")
    return body.read()


def download_image_bytes_from_s3_url(url: str) -> bytes | None:
    """Download image bytes when the URL points at an S3 object."""
    parsed = parse_s3_https_url(url)
    if not parsed:
        return None
    bucket, key = parsed
    try:
        return download_s3_object(bucket, key)
    except (BotoCoreError, ClientError, ValueError) as exc:
        logger.exception("Failed to download S3 image bucket=%s key=%s: %s", bucket, key, exc)
        return None


__all__ = [
    "download_image_bytes_from_s3_url",
    "download_s3_object",
    "parse_s3_https_url",
    "s3_configured",
    "upload_image_bytes",
]
