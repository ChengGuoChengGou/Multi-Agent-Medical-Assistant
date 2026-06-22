"""Tests for Phase 9: System monitoring aggregation endpoint."""
import time
import pytest
from fastapi.testclient import TestClient

from app import app
from api.monitoring import set_app_start_time


client = TestClient(app)


class TestMonitoringEndpoint:

    def test_admin_stats_returns_200(self):
        set_app_start_time(time.time() - 60)
        resp = client.get("/api/v1/admin/stats")
        assert resp.status_code == 200

    def test_admin_stats_structure(self):
        set_app_start_time(time.time() - 120)
        data = client.get("/api/v1/admin/stats").json()
        assert "uptime_seconds" in data
        assert "circuit_breakers" in data
        assert "agent_metrics" in data
        assert "system" in data
        assert data["uptime_seconds"] >= 100  # ~120s

    def test_circuit_breakers_structure(self):
        data = client.get("/api/v1/admin/stats").json()
        cb = data["circuit_breakers"]
        assert "breakers" in cb
        # Should have llm, mcp, web_search, image breakers
        for name in ("llm", "mcp", "web_search", "image"):
            assert name in cb["breakers"]
            breaker = cb["breakers"][name]
            assert "state" in breaker
            assert breaker["state"] in ("closed", "open", "half_open")

    def test_system_info_structure(self):
        data = client.get("/api/v1/admin/stats").json()
        sys_info = data["system"]
        assert "python_version" in sys_info
        assert "pid" in sys_info
        assert sys_info["pid"] > 0

    def test_agent_metrics_structure(self):
        data = client.get("/api/v1/admin/stats").json()
        am = data["agent_metrics"]
        # agent_metrics may be empty {} if no agents have been tracked yet
        # That's valid - just verify it's a dict
        assert isinstance(am, dict)

    def test_uptime_positive(self):
        set_app_start_time(time.time() - 300)
        data = client.get("/api/v1/admin/stats").json()
        assert data["uptime_seconds"] >= 290
