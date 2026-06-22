"""
Tests for middleware/api_key_auth.py — Phase 53: API Key authentication.
"""
import asyncio
import importlib.util
import os
import sys

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

# Load module directly
AUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "middleware", "api_key_auth.py")
spec = importlib.util.spec_from_file_location("api_key_auth", AUTH_PATH)
auth_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth_mod)
APIKeyAuthMiddleware = auth_mod.APIKeyAuthMiddleware
_extract_key = auth_mod._extract_key
_is_protected = auth_mod._is_protected


# ── Fixtures ──

def _make_app(api_keys: str = ""):
    """Create a minimal Starlette app with the auth middleware."""
    os.environ["MEDICAL_API_KEYS"] = api_keys

    async def chat_endpoint(request):
        return JSONResponse({"ok": True, "path": "/chat"})

    async def health_endpoint(request):
        return JSONResponse({"status": "healthy"})

    async def docs_endpoint(request):
        return JSONResponse({"openapi": True})

    async def stream_endpoint(request):
        return JSONResponse({"ok": True, "path": "/chat/stream"})

    app = Starlette(routes=[
        Route("/chat", chat_endpoint),
        Route("/chat/stream", stream_endpoint),
        Route("/health", health_endpoint),
        Route("/docs", docs_endpoint),
    ])
    app.add_middleware(APIKeyAuthMiddleware)
    return app


# ── Tests: Dev mode (no keys configured) ──

class TestDevMode:
    """When MEDICAL_API_KEYS is empty, all requests should be allowed."""

    def test_dev_mode_allows_protected(self):
        app = _make_app(api_keys="")
        client = TestClient(app)
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_dev_mode_allows_public(self):
        app = _make_app(api_keys="")
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200


# ── Tests: Key mode (keys configured) ──

class TestKeyMode:
    """When MEDICAL_API_KEYS is set, protected endpoints require valid key."""

    KEYS = "test-key-abc,test-key-xyz"

    def test_no_key_returns_401(self):
        app = _make_app(api_keys=self.KEYS)
        client = TestClient(app)
        resp = client.get("/chat")
        assert resp.status_code == 401
        assert "API key required" in resp.json()["message"]

    def test_invalid_key_returns_403(self):
        app = _make_app(api_keys=self.KEYS)
        client = TestClient(app)
        resp = client.get("/chat", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_valid_bearer_token(self):
        app = _make_app(api_keys=self.KEYS)
        client = TestClient(app)
        resp = client.get("/chat", headers={"Authorization": "Bearer test-key-abc"})
        assert resp.status_code == 200

    def test_valid_x_api_key_header(self):
        app = _make_app(api_keys=self.KEYS)
        client = TestClient(app)
        resp = client.get("/chat", headers={"X-API-Key": "test-key-xyz"})
        assert resp.status_code == 200

    def test_bearer_takes_precedence_over_x_api_key(self):
        app = _make_app(api_keys=self.KEYS)
        client = TestClient(app)
        # Both headers present; bearer is checked first
        resp = client.get("/chat", headers={
            "Authorization": "Bearer test-key-abc",
            "X-API-Key": "wrong-key"
        })
        assert resp.status_code == 200

    def test_public_endpoints_no_key_needed(self):
        app = _make_app(api_keys=self.KEYS)
        client = TestClient(app)
        for path in ["/health", "/docs"]:
            resp = client.get(path)
            assert resp.status_code == 200, f"{path} should be public"

    def test_chat_stream_protected(self):
        app = _make_app(api_keys=self.KEYS)
        client = TestClient(app)
        resp = client.get("/chat/stream")
        assert resp.status_code == 401

        resp2 = client.get("/chat/stream", headers={"Authorization": "Bearer test-key-abc"})
        assert resp2.status_code == 200


# ── Tests: Stats ──

class TestStats:
    def test_get_stats_returns_config(self):
        middleware = APIKeyAuthMiddleware.__new__(APIKeyAuthMiddleware)
        middleware._dev_mode = True
        middleware._api_keys = set()
        middleware._auth_successes = 0
        middleware._auth_failures = 0
        stats = middleware.get_stats()
        assert "dev_mode" in stats
        assert "keys_configured" in stats
        assert "protected_prefixes" in stats
        assert stats["dev_mode"] is True


# ── Tests: Key extraction edge cases ──

class TestKeyExtraction:
    """Test the module-level _extract_key function."""

    def test_extract_from_bearer(self):
        from starlette.requests import Request as StarletteRequest
        scope = {"type": "http", "method": "GET", "path": "/", "headers": [
            (b"authorization", b"Bearer my-secret-key")
        ]}
        req = StarletteRequest(scope)
        assert _extract_key(req) == "my-secret-key"

    def test_extract_from_x_api_key(self):
        from starlette.requests import Request as StarletteRequest
        scope = {"type": "http", "method": "GET", "path": "/", "headers": [
            (b"x-api-key", b"my-secret-key")
        ]}
        req = StarletteRequest(scope)
        assert _extract_key(req) == "my-secret-key"

    def test_extract_empty_when_no_headers(self):
        from starlette.requests import Request as StarletteRequest
        scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
        req = StarletteRequest(scope)
        assert _extract_key(req) is None

    def test_extract_rejects_basic_auth(self):
        """Basic auth scheme should not be accepted as API key."""
        from starlette.requests import Request as StarletteRequest
        scope = {"type": "http", "method": "GET", "path": "/", "headers": [
            (b"authorization", b"Basic dXNlcjpwYXNz")
        ]}
        req = StarletteRequest(scope)
        assert _extract_key(req) is None
