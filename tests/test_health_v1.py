"""
Tests for API v1 Health Check endpoints.
Phase 7: Dependency health checks for LLM, Qdrant, Embeddings.
"""

import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from app import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Synchronous test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthDetailed:
    """Test /api/v1/health/detailed endpoint."""

    def test_endpoint_exists(self, client):
        """Detailed health endpoint should return 200."""
        resp = client.get("/api/v1/health/detailed")
        assert resp.status_code == 200

    def test_response_schema(self, client):
        """Response should have required fields."""
        resp = client.get("/api/v1/health/detailed")
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "dependencies" in data
        assert data["version"] == "2.0"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_dependencies_schema(self, client):
        """Each dependency should have name, status. latency_ms is optional."""
        resp = client.get("/api/v1/health/detailed")
        data = resp.json()
        deps = data["dependencies"]
        assert isinstance(deps, list)
        assert len(deps) >= 1
        for dep in deps:
            assert "name" in dep
            assert "status" in dep
            assert dep["status"] in ("healthy", "degraded", "unhealthy")
            # latency_ms may be None when not measured
            if dep.get("latency_ms") is not None:
                assert isinstance(dep["latency_ms"], (int, float))

    def test_overall_status_values(self, client):
        """Overall status should be one of the valid values."""
        resp = client.get("/api/v1/health/detailed")
        data = resp.json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    def test_at_least_one_dependency_returned(self, client):
        """At least one dependency check should be present."""
        resp = client.get("/api/v1/health/detailed")
        data = resp.json()
        dep_names = [d["name"] for d in data["dependencies"]]
        assert len(dep_names) >= 1
        # Known deps: llm_primary, qdrant, embedding, file_storage
        known = {"llm_primary", "qdrant", "embedding", "file_storage"}
        assert any(n in known for n in dep_names)


class TestHealthDetailedEdgeCases:
    """Edge cases for health check."""

    def test_multiple_calls_consistent_schema(self, client):
        """Multiple calls should return consistent schema."""
        for _ in range(3):
            resp = client.get("/api/v1/health/detailed")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert "dependencies" in data

    def test_health_endpoint_no_auth_required(self, client):
        """Health check should work without authentication."""
        resp = client.get("/api/v1/health/detailed")
        assert resp.status_code == 200
