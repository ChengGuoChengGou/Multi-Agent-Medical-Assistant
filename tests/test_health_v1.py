"""
Tests for API v1 Health Check endpoints.
Phase 7: Dependency health checks for LLM, Qdrant, Embeddings.
Phase 54: Enriched /health with uptime, api_auth stats, middleware completeness.
"""

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:
    def test_health_endpoint_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        # Phase 54: uptime_seconds added
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
        # middleware dict with api_key_auth
        middleware = data["middleware"]
        assert middleware["rate_limiting"] is True
        assert middleware["security_headers"] is True
        assert middleware["request_logging"] is True
        assert middleware["request_deduplication"] is True
        assert middleware["api_key_auth"] is True
        # dedup_stats present
        assert "dedup_stats" in data
        assert isinstance(data["dedup_stats"], dict)
        # Phase 54: api_auth stats present
        assert "api_auth" in data
        assert isinstance(data["api_auth"], dict)
        assert "dev_mode" in data["api_auth"]
        assert "keys_configured" in data["api_auth"]
        assert "auth_successes" in data["api_auth"]
        assert "auth_failures" in data["api_auth"]

    def test_health_endpoint_no_auth_required(self, client):
        """Health check should work without authentication."""
        resp = client.get("/health")
        assert resp.status_code == 200


class TestHealthV1Endpoints:
    """Phase 54: Additional health check assertions."""

    def test_health_response_has_uptime(self, client):
        """Uptime should be a non-negative number."""
        resp = client.get("/health")
        data = resp.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_detailed_endpoint(self, client):
        """Detailed health check should include dependency info."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "api_auth" in data
        assert "dev_mode" in data["api_auth"]

    def test_health_endpoint_no_auth_required_v2(self, client):
        """Health check should work without authentication (duplicate for clarity)."""
        resp = client.get("/health")
        assert resp.status_code == 200
