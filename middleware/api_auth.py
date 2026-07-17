"""FastAPI dependencies and OpenAPI wiring for API key authentication."""

from __future__ import annotations

import secrets
from typing import Any, List

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

API_KEY_SCHEME_NAME = "ApiKeyAuth"


def build_api_auth(
    expected_key: str,
    header_name: str = "X-API-Key",
    *,
    enabled: bool = True,
) -> tuple[List[Any], Any]:
    """Return route dependencies that enforce API key auth when enabled and configured."""
    if not enabled or not expected_key:
        return [], None

    api_key_header_scheme = APIKeyHeader(
        name=header_name,
        scheme_name=API_KEY_SCHEME_NAME,
        auto_error=False,
        description="Shared API key issued to trusted clients.",
    )

    async def verify_api_key(
        request: Request,
        api_key: str | None = Security(api_key_header_scheme),
    ) -> None:
        provided = (api_key or "").strip()
        if not provided:
            authorization = request.headers.get("authorization", "")
            if authorization.lower().startswith("bearer "):
                provided = authorization[7:].strip()

        if not provided or not secrets.compare_digest(provided, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
            )

    return [Depends(verify_api_key)], api_key_header_scheme
