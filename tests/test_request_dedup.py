"""Tests for middleware/request_dedup.py — request deduplication middleware."""

import asyncio
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from middleware.request_dedup import (
    RequestDedupMiddleware,
    _InFlightEntry,
    _make_dedup_key,
    get_dedup_stats,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_dedup():
    """FastAPI app with RequestDedupMiddleware."""
    app = FastAPI()
    app.add_middleware(RequestDedupMiddleware)

    call_count = 0

    @app.post("/chat/query")
    async def chat_query():
        nonlocal call_count
        call_count += 1
        # Simulate some processing
        await asyncio.sleep(0.05)
        return {"response": f"answer_{call_count}", "count": call_count}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/other")
    async def other():
        return {"status": "other"}

    return app


@pytest.fixture
def client(app_with_dedup):
    return TestClient(app_with_dedup)


# ---------------------------------------------------------------------------
# Test: _make_dedup_key
# ---------------------------------------------------------------------------


class TestMakeDedupKey:
    def test_same_request_same_key(self):
        """Identical method+path+body should produce same key."""
        req1 = MagicMock(spec=Request)
        req1.method = "POST"
        req1.url.path = "/chat/query"
        req1.query_params = {}

        req2 = MagicMock(spec=Request)
        req2.method = "POST"
        req2.url.path = "/chat/query"
        req2.query_params = {}

        body = b'{"query": "test"}'
        k1 = _make_dedup_key(req1, body)
        k2 = _make_dedup_key(req2, body)
        assert k1 == k2

    def test_different_body_different_key(self):
        req = MagicMock(spec=Request)
        req.method = "POST"
        req.url.path = "/chat/query"
        req.query_params = {}

        k1 = _make_dedup_key(req, b'{"query": "a"}')
        k2 = _make_dedup_key(req, b'{"query": "b"}')
        assert k1 != k2

    def test_different_path_different_key(self):
        req1 = MagicMock(spec=Request)
        req1.method = "POST"
        req1.url.path = "/chat/query"
        req1.query_params = {}

        req2 = MagicMock(spec=Request)
        req2.method = "POST"
        req2.url.path = "/other"
        req2.query_params = {}

        k1 = _make_dedup_key(req1, b"{}")
        k2 = _make_dedup_key(req2, b"{}")
        assert k1 != k2

    def test_empty_body(self):
        req = MagicMock(spec=Request)
        req.method = "POST"
        req.url.path = "/chat/query"
        req.query_params = {}
        key = _make_dedup_key(req, b"")
        assert len(key) == 24  # sha256[:24]


# ---------------------------------------------------------------------------
# Test: _InFlightEntry
# ---------------------------------------------------------------------------


class TestInFlightEntry:
    def test_initial_state(self):
        entry = _InFlightEntry()
        assert entry.event is not None
        assert not entry.event.is_set()
        assert entry.result is None
        assert entry.error is None
        assert entry.waiters == 0
        assert entry.result_status == 200
        assert isinstance(entry.started_at, float)


# ---------------------------------------------------------------------------
# Test: non-eligible routes pass through
# ---------------------------------------------------------------------------


class TestNonEligibleRoutes:
    def test_get_passes_through(self, client):
        """GET /health should not be deduped."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "X-Dedup" not in resp.headers

    def test_non_dedup_path_passes_through(self, client):
        """POST /other should not be deduped."""
        resp = client.post("/other", json={"data": "test"})
        assert resp.status_code == 200
        assert "X-Dedup" not in resp.headers


# ---------------------------------------------------------------------------
# Test: dedup eligible routes
# ---------------------------------------------------------------------------


class TestDedupEligibleRoutes:
    def test_first_request_has_miss_header(self, client):
        """First request to /chat/query should have X-Dedup: miss."""
        resp = client.post("/chat/query", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Dedup") == "miss"


# ---------------------------------------------------------------------------
# Test: get_dedup_stats
# ---------------------------------------------------------------------------


class TestGetDedupStats:
    def test_stats_structure(self):
        stats = get_dedup_stats()
        assert "inflight_requests" in stats
        assert "total_coalesced" in stats
        assert "total_leader" in stats
        assert isinstance(stats["inflight_requests"], int)
        assert isinstance(stats["total_coalesced"], int)
        assert isinstance(stats["total_leader"], int)


# ---------------------------------------------------------------------------
# Test: middleware config
# ---------------------------------------------------------------------------


class TestMiddlewareConfig:
    def test_dedup_paths_frozen(self):
        assert "/chat/query" in RequestDedupMiddleware.DEDUP_PATHS
        assert isinstance(RequestDedupMiddleware.DEDUP_PATHS, frozenset)

    def test_max_wait_seconds(self):
        assert RequestDedupMiddleware.MAX_WAIT_SECONDS == 120.0
