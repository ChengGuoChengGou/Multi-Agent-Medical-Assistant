"""
Tests for error_handlers: content-negotiation, structured error responses.

Run: python -m pytest tests/test_error_handlers.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

# ── Test helpers ──


def _make_app():
    """Build minimal FastAPI app with error handlers registered."""
    from error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @app.get("/http-error")
    async def http_error():
        raise HTTPException(status_code=404, detail="Resource not found")

    @app.get("/server-error")
    async def server_error():
        raise RuntimeError("something broke")

    class Item(BaseModel):
        name: str
        price: float

    @app.post("/validate")
    async def validate(item: Item):
        return {"ok": True}

    return app


# ── Tests ──


class TestContentNegotiation:
    """Error responses should be HTML for browsers, JSON for API clients."""

    def test_api_client_gets_json_404(self):
        """API client (Accept: application/json) should get JSON error."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/http-error", headers={"Accept": "application/json"})
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == 404
        assert "Resource not found" in data["error"]["message"]

    def test_browser_gets_html_404(self):
        """Browser (Accept: text/html) should get HTML error page."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/http-error", headers={"Accept": "text/html"})
        assert resp.status_code == 404
        assert "text/html" in resp.headers.get("content-type", "")
        assert "404" in resp.text

    def test_api_client_gets_json_500(self):
        """Unhandled exception should return 500 JSON with no internal details."""
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/server-error", headers={"Accept": "application/json"})
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"]["code"] == 500
        # Must NOT leak internal error details
        assert "something broke" not in data["error"]["message"]
        assert "Internal server error" in data["error"]["message"]

    def test_browser_gets_html_500(self):
        """Browser should get HTML page for unhandled exception."""
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/server-error", headers={"Accept": "text/html"})
        assert resp.status_code == 500
        assert "text/html" in resp.headers.get("content-type", "")
        assert "500" in resp.text


class TestErrorResponseStructure:
    """All error responses must have consistent structure."""

    def test_error_has_request_id(self):
        """JSON error should include request_id field."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/http-error", headers={"Accept": "application/json"})
        data = resp.json()
        assert "request_id" in data["error"]

    def test_error_has_timestamp(self):
        """JSON error should include timestamp field."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/http-error", headers={"Accept": "application/json"})
        data = resp.json()
        ts = data["error"]["timestamp"]
        assert isinstance(ts, int)
        assert ts > 1700000000  # after 2023

    def test_validation_error_returns_422(self):
        """Pydantic validation error should return structured 422."""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/validate", json={"name": "test"}, headers={"Accept": "application/json"})
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == 422
        assert "validation_errors" in data["error"].get("details", {})

    def test_validation_error_field_detail(self):
        """Validation error should list the offending field."""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/validate", json={"name": "test"}, headers={"Accept": "application/json"})
        data = resp.json()
        errors = data["error"]["details"]["validation_errors"]
        assert len(errors) >= 1
        assert "price" in errors[0]["field"]


class TestIsBrowserDetection:
    """_is_browser should correctly detect browser vs API client."""

    def test_detects_browser(self):
        from starlette.requests import Request as StarletteRequest

        from error_handlers import _is_browser

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"accept", b"text/html,application/xhtml+xml")],
        }
        req = StarletteRequest(scope, receive=lambda: None)
        assert _is_browser(req) is True

    def test_detects_api_client(self):
        from starlette.requests import Request as StarletteRequest

        from error_handlers import _is_browser

        scope = {"type": "http", "method": "GET", "path": "/", "headers": [(b"accept", b"application/json")]}
        req = StarletteRequest(scope, receive=lambda: None)
        assert _is_browser(req) is False

    def test_json_in_accept_disqualifies_browser(self):
        """If both text/html and application/json present, treat as API."""
        from starlette.requests import Request as StarletteRequest

        from error_handlers import _is_browser

        scope = {"type": "http", "method": "GET", "path": "/", "headers": [(b"accept", b"text/html, application/json")]}
        req = StarletteRequest(scope, receive=lambda: None)
        assert _is_browser(req) is False
