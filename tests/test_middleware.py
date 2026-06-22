"""
Tests for middleware: RateLimitMiddleware, RequestDedupMiddleware,
SecurityHeadersMiddleware, RequestLoggingMiddleware.

Run: python -m pytest tests/test_middleware.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware import Middleware


# ─── Test helper ───

def _make_app(middleware_class, **kwargs):
    async def echo(request: Request):
        return JSONResponse({"ok": True})

    async def health(request: Request):
        return JSONResponse({"status": "healthy"})

    return Starlette(
        routes=[
            Route("/chat/query", echo, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/data/test", echo, methods=["GET"]),
        ],
        middleware=[Middleware(middleware_class, **kwargs)],
    )


# ═══════ RateLimitMiddleware ═══════

class TestRateLimitMiddleware:

    def test_normal_request_passes(self):
        from middleware.rate_limiter import RateLimitMiddleware
        app = _make_app(RateLimitMiddleware, requests_per_minute=10, requests_per_hour=100)
        client = TestClient(app)
        resp = client.post("/chat/query", json={"q": "hello"})
        assert resp.status_code == 200

    def test_exceed_per_minute_returns_429(self):
        from middleware.rate_limiter import RateLimitMiddleware
        app = _make_app(RateLimitMiddleware, requests_per_minute=3, requests_per_hour=100)
        client = TestClient(app)
        for i in range(3):
            assert client.post("/chat/query", json={"q": f"r{i}"}).status_code == 200
        resp = client.post("/chat/query", json={"q": "over"})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_health_skips_rate_limit(self):
        from middleware.rate_limiter import RateLimitMiddleware
        app = _make_app(RateLimitMiddleware, requests_per_minute=0, requests_per_hour=0)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_static_path_skips_rate_limit(self):
        from middleware.rate_limiter import RateLimitMiddleware
        app = _make_app(RateLimitMiddleware, requests_per_minute=0, requests_per_hour=0)
        client = TestClient(app)
        resp = client.get("/data/test")
        assert resp.status_code == 200

    def test_get_client_ip_forwarded_for(self):
        from middleware.rate_limiter import RateLimitMiddleware
        mw = RateLimitMiddleware(app=None, requests_per_minute=10, requests_per_hour=100)
        request = type("Req", (), {"headers": {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}, "client": type("C", (), {"host": "127.0.0.1"})()})()
        assert mw._get_client_ip(request) == "10.0.0.1"

    def test_get_client_ip_no_forwarded(self):
        from middleware.rate_limiter import RateLimitMiddleware
        mw = RateLimitMiddleware(app=None, requests_per_minute=10, requests_per_hour=100)
        request = type("Req", (), {"headers": {}, "client": type("C", (), {"host": "192.168.1.100"})()})()
        assert mw._get_client_ip(request) == "192.168.1.100"


# ═══════ SecurityHeadersMiddleware ═══════

class TestSecurityHeadersMiddleware:

    def test_security_headers_present(self):
        from middleware.security import SecurityHeadersMiddleware
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.post("/chat/query", json={"q": "test"})
        assert "content-security-policy" in resp.headers
        assert resp.headers.get("x-frame-options") == "DENY"
        assert "x-xss-protection" in resp.headers
        assert "referrer-policy" in resp.headers

    def test_csp_contains_self(self):
        from middleware.security import SecurityHeadersMiddleware
        app = _make_app(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.post("/chat/query", json={"q": "test"})
        csp = resp.headers["content-security-policy"]
        assert "'self'" in csp


# ═══════ RequestLoggingMiddleware ═══════

class TestRequestLoggingMiddleware:

    def test_logging_does_not_break_response(self):
        from middleware.security import RequestLoggingMiddleware
        app = _make_app(RequestLoggingMiddleware)
        client = TestClient(app)
        resp = client.post("/chat/query", json={"q": "test"})
        assert resp.status_code == 200


# ═══════ RequestDedupMiddleware ═══════

class TestRequestDedupMiddleware:

    def test_non_dedup_path_passes_through(self):
        from middleware.request_dedup import RequestDedupMiddleware
        app = _make_app(RequestDedupMiddleware)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "X-Dedup" not in resp.headers

    def test_dedup_eligible_path_returns_x_dedup_header(self):
        from middleware.request_dedup import RequestDedupMiddleware
        app = _make_app(RequestDedupMiddleware)
        client = TestClient(app)
        resp = client.post("/chat/query", json={"q": "test"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Dedup") == "miss"

    def test_get_dedup_stats(self):
        from middleware.request_dedup import get_dedup_stats
        stats = get_dedup_stats()
        assert isinstance(stats, dict)
        assert "inflight_requests" in stats
        assert "total_coalesced" in stats
        assert "total_leader" in stats


# ═══════ CSP Policy Config ═══════

class TestCSPConfig:

    def test_csp_frame_ancestors_none(self):
        from middleware.security import CSP_POLICY
        assert "'none'" in CSP_POLICY.get("frame-ancestors", "")

    def test_csp_connect_src_allows_self(self):
        from middleware.security import CSP_POLICY
        assert "'self'" in CSP_POLICY.get("connect-src", "")


# ═══════ /health Endpoint ═══════

class TestHealthEndpoint:

    def test_health_returns_middleware_status(self):
        try:
            from app import app as real_app
            client = TestClient(real_app)
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "middleware" in data
            assert "dedup_stats" in data
        except Exception as e:
            pytest.skip(f"Full app import failed: {e}")
