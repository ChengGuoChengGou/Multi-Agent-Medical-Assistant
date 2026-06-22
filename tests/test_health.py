"""Tests for api/health.py - Dependency Health Check Endpoint."""
import os
from unittest.mock import MagicMock, patch
import pytest
from api.health import (
    DependencyStatus,
    DetailedHealthResponse,
    _check_llm_primary,
    _check_qdrant,
    _check_embedding,
    _check_file_storage,
)


class TestDependencyStatus:
    """Test DependencyStatus model."""

    def test_healthy_status(self):
        ds = DependencyStatus(name="test", status="healthy", latency_ms=1.5, detail="ok")
        assert ds.name == "test"
        assert ds.status == "healthy"
        assert ds.latency_ms == 1.5
        assert ds.detail == "ok"

    def test_unhealthy_status_no_latency(self):
        ds = DependencyStatus(name="db", status="unhealthy", detail="connection refused")
        assert ds.latency_ms is None
        assert ds.status == "unhealthy"

    def test_degraded_status(self):
        ds = DependencyStatus(name="cache", status="degraded", latency_ms=50.0)
        assert ds.status == "degraded"

    def test_optional_fields(self):
        ds = DependencyStatus(name="x", status="healthy")
        assert ds.latency_ms is None
        assert ds.detail is None


class TestDetailedHealthResponse:
    """Test DetailedHealthResponse model."""

    def test_healthy_response(self):
        deps = [DependencyStatus(name="a", status="healthy")]
        resp = DetailedHealthResponse(
            status="healthy", version="2.0", uptime_seconds=100.0, dependencies=deps
        )
        assert resp.status == "healthy"
        assert resp.version == "2.0"
        assert len(resp.dependencies) == 1

    def test_degraded_response(self):
        deps = [
            DependencyStatus(name="a", status="healthy"),
            DependencyStatus(name="b", status="degraded"),
        ]
        resp = DetailedHealthResponse(
            status="degraded", version="2.0", uptime_seconds=50.0, dependencies=deps
        )
        assert resp.status == "degraded"

    def test_unhealthy_response(self):
        deps = [DependencyStatus(name="a", status="unhealthy")]
        resp = DetailedHealthResponse(
            status="unhealthy", version="2.0", uptime_seconds=0.0, dependencies=deps
        )
        assert resp.status == "unhealthy"


class TestCheckFunctions:
    """Test individual check functions with mocks."""

    @patch("config.Config")
    def test_check_llm_primary_success(self, mock_cfg_cls):
        mock_cfg = MagicMock()
        mock_cfg._make_llm.return_value = MagicMock(model_name="test-model")
        mock_cfg.llm_provider = "xiaomi-mimo"
        mock_cfg_cls.return_value = mock_cfg

        result = _check_llm_primary()
        assert result.status == "healthy"
        assert "test-model" in result.detail

    @patch("config.Config")
    def test_check_llm_primary_failure(self, mock_cfg_cls):
        mock_cfg_cls.side_effect = Exception("No API key")
        result = _check_llm_primary()
        assert result.status == "unhealthy"
        assert "No API key" in result.detail

    @patch("requests.get")
    @patch("config.Config")
    def test_check_qdrant_success(self, mock_cfg_cls, mock_requests):
        mock_cfg = MagicMock()
        mock_cfg.rag = MagicMock(qdrant_url="http://localhost:6333")
        mock_cfg_cls.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {"collections": [{"name": "test"}]}}
        mock_requests.return_value = mock_resp

        result = _check_qdrant()
        assert result.status == "healthy"
        assert "1 collections" in result.detail

    @patch("requests.get")
    @patch("config.Config")
    def test_check_qdrant_no_url(self, mock_cfg_cls, mock_requests):
        mock_cfg = MagicMock()
        mock_cfg.rag = MagicMock(spec=[])  # no qdrant_url attr
        # Also make getattr return None
        type(mock_cfg.rag).qdrant_url = None
        mock_cfg_cls.return_value = mock_cfg

        result = _check_qdrant()
        # Should return unhealthy with "not configured" or handle gracefully
        assert result.status in ("unhealthy", "degraded")

    @patch("requests.get")
    @patch("config.Config")
    def test_check_qdrant_connection_error(self, mock_cfg_cls, mock_requests):
        mock_cfg = MagicMock()
        mock_cfg.rag = MagicMock(qdrant_url="http://localhost:6333")
        mock_cfg_cls.return_value = mock_cfg
        mock_requests.get.side_effect = Exception("Connection refused")

        result = _check_qdrant()
        assert result.status in ("unhealthy", "degraded")

    def test_check_file_storage_healthy(self, tmp_path):
        """Test file storage check with valid dirs."""
        import os
        from api.health import _check_file_storage as check

        # Create required dirs
        for d in ["uploads/backend", "uploads/frontend", "data"]:
            os.makedirs(os.path.join(tmp_path, d), exist_ok=True)

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = check()
            assert result.status == "healthy"
            assert "accessible" in result.detail
        finally:
            os.chdir(original_dir)

    def test_check_file_storage_missing_dirs(self, tmp_path):
        """Test file storage check with missing dirs."""
        from api.health import _check_file_storage as check

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = check()
            # Should be degraded since dirs are missing
            assert result.status == "degraded"
            assert "missing" in result.detail.lower()
        finally:
            os.chdir(original_dir)
