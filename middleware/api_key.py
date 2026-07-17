"""API key authentication middleware."""

from __future__ import annotations

import logging
import secrets
from typing import Callable, Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

DEFAULT_EXEMPT_PATHS = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests that do not present a valid shared API key in headers."""

    def __init__(
        self,
        app,
        *,
        api_key: str,
        header_name: str = "X-API-Key",
        enabled: bool = True,
        exempt_paths: Iterable[str] | None = None,
        exempt_prefixes: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._enabled = enabled
        self._api_key = api_key
        self._header_name = header_name.lower()
        self._exempt_paths = frozenset(exempt_paths or DEFAULT_EXEMPT_PATHS)
        self._exempt_prefixes = tuple(exempt_prefixes or ("/generated",))

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        if not self._enabled or not self._api_key:
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in self._exempt_paths or any(
            path.startswith(prefix) for prefix in self._exempt_prefixes
        ):
            return await call_next(request)

        provided_key = self._extract_api_key(request)
        if not provided_key or not secrets.compare_digest(provided_key, self._api_key):
            logger.warning(
                "Rejected request without valid API key: %s %s",
                request.method,
                path,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
            )

        return await call_next(request)

    def _extract_api_key(self, request: Request) -> str | None:
        header_value = request.headers.get(self._header_name)
        if header_value:
            return header_value.strip()

        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()

        return None
