"""
API Key Authentication Middleware (Phase 53).
Protects sensitive endpoints (chat, voice, document upload) with API key auth.
Public endpoints (health, docs, static files) remain open.

Configuration:
    Set MEDICAL_API_KEYS env var with comma-separated keys.
    If not set, all requests are allowed (dev mode).

Usage:
    Authorization: Bearer <api-key>
    X-API-Key: <api-key>
"""
import hmac
import logging
import os
import time
from typing import Optional, Set

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── Module-level singleton reference (set in __init__) ──
_middleware_instance: Optional["APIKeyAuthMiddleware"] = None

# ── Paths that require API key auth ──
_PROTECTED_PREFIXES: tuple = (
    "/chat",
    "/generate-speech",
    "/transcribe",
    "/upload-and-validate",
    "/skin-lesion",
)

# ── Paths always public (no auth needed) ──
_PUBLIC_PREFIXES: tuple = (
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/cache/stats",
    "/api/v1/",
    "/data/",
    "/uploads/",
    "/favicon.ico",
)


def _load_api_keys() -> Set[str]:
    """Load API keys from MEDICAL_API_KEYS env var (comma-separated)."""
    raw = os.environ.get("MEDICAL_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def _extract_key(request: Request) -> Optional[str]:
    """Extract API key from Authorization: Bearer <key> or X-API-Key header."""
    # 1. Authorization: Bearer <key>
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # 2. X-API-Key header
    xkey = request.headers.get("x-api-key", "")
    if xkey:
        return xkey.strip()
    return None


def _is_protected(path: str) -> bool:
    """Check if the request path requires authentication."""
    # Exact match root is public
    if path == "/":
        return False
    # Check public prefixes first
    for prefix in _PUBLIC_PREFIXES:
        if prefix == "/":
            continue  # "/" is handled by exact match above
        if path.startswith(prefix):
            return False
    # Check protected prefixes
    for prefix in _PROTECTED_PREFIXES:
        if path.startswith(prefix):
            return True
    # Unknown paths: allow (could be static files, frontend routes, etc.)
    return False


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that enforces API key authentication on protected endpoints.

    If MEDICAL_API_KEYS is not set, operates in dev mode (all requests allowed).
    """

    def __init__(self, app, api_keys: Optional[Set[str]] = None):
        global _middleware_instance
        super().__init__(app)
        self._api_keys = api_keys if api_keys is not None else _load_api_keys()
        _middleware_instance = self
        self._dev_mode = len(self._api_keys) == 0
        self._auth_failures = 0
        self._auth_successes = 0

        if self._dev_mode:
            logger.info("[APIKeyAuth] Dev mode: no MEDICAL_API_KEYS set, all requests allowed")
        else:
            logger.info("[APIKeyAuth] Production mode: %d API key(s) loaded", len(self._api_keys))

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth check for non-protected endpoints
        if not _is_protected(path):
            return await call_next(request)

        # Dev mode: allow everything
        if self._dev_mode:
            request.state.api_authenticated = True
            request.state.api_key_id = "dev-mode"
            return await call_next(request)

        # Extract API key
        key = _extract_key(request)
        if not key:
            self._auth_failures += 1
            logger.warning("[APIKeyAuth] Missing API key for %s %s from %s",
                           request.method, path, request.client.host if request.client else "unknown")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "authentication_required",
                    "message": "API key required. Set Authorization: Bearer <key> or X-API-Key header.",
                    "docs": "/docs for API documentation",
                },
            )

        # Validate key using constant-time comparison
        valid = any(hmac.compare_digest(key, allowed) for allowed in self._api_keys)
        if not valid:
            self._auth_failures += 1
            logger.warning("[APIKeyAuth] Invalid API key for %s %s from %s",
                           request.method, path, request.client.host if request.client else "unknown")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "invalid_api_key",
                    "message": "The provided API key is not valid.",
                },
            )

        # Success
        self._auth_successes += 1
        request.state.api_authenticated = True
        request.state.api_key_id = key[:8] + "..."  # partial key for logging
        return await call_next(request)

    def get_stats(self) -> dict:
        """Return auth middleware statistics."""
        return {
            "dev_mode": self._dev_mode,
            "keys_configured": len(self._api_keys),
            "auth_successes": self._auth_successes,
            "auth_failures": self._auth_failures,
            "protected_prefixes": list(_PROTECTED_PREFIXES),
        }


def get_auth_stats() -> dict:
    """Module-level accessor for API key auth middleware statistics."""
    if _middleware_instance is None:
        return {"dev_mode": True, "keys_configured": 0, "auth_successes": 0, "auth_failures": 0}
    return _middleware_instance.get_stats()
