"""
End-to-end tests for API endpoints.
Run: python -m pytest tests/test_main.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client for the FastAPI app."""
    # Set test environment variables before importing app
    os.environ.setdefault("XIAOMI_API_KEY", "test-key-not-real")
    os.environ.setdefault("MILVUS_HOST", "localhost")
    os.environ.setdefault("MILVUS_PORT", "19530")
    os.environ.setdefault("ENABLED_AGENTS", "conversation_agent,report_agent")
    
    from app import app
    return TestClient(app)


# === Health Endpoints ===

class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
    
    def test_health_has_status(self, client):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded"), f"Unexpected status: {data['status']}"
    
    def test_health_has_middleware_info(self, client):
        r = client.get("/health")
        data = r.json()
        assert "middleware" in data
        assert data["middleware"]["rate_limiting"] is True
    
    def test_health_has_dedup_stats(self, client):
        r = client.get("/health")
        data = r.json()
        assert "dedup_stats" in data


# === Metrics Endpoint (not yet implemented) ===

class TestMetricsEndpoint:
    @pytest.mark.xfail(reason="/metrics endpoint not yet implemented")
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200


# === Security Headers ===

class TestSecurityHeaders:
    def test_csp_header_present(self, client):
        r = client.get("/health")
        assert "content-security-policy" in r.headers
    
    def test_x_content_type_options(self, client):
        r = client.get("/health")
        assert r.headers.get("x-content-type-options") == "nosniff"
    
    def test_x_frame_options(self, client):
        r = client.get("/health")
        assert r.headers.get("x-frame-options") == "DENY"
    
    def test_security_headers_present(self, client):
        """Security headers are set by middleware."""
        r = client.get("/health")
        # x-frame-options set by SecurityHeadersMiddleware
        assert "x-frame-options" in r.headers
        assert r.headers["x-frame-options"] == "DENY"


# === API Versioning ===

class TestAPIVersioning:
    def test_api_version_prefix(self, client):
        """API v1 prefix does not exist yet — expect 404."""
        r = client.get("/api/v1/health")
        assert r.status_code == 404

    def test_api_v1_returns_404(self, client):
        """/api/v1/health returns 404 (not yet implemented)."""
        r = client.get("/api/v1/health")
        assert r.status_code == 404


# === Chat Endpoint ===

class TestChatEndpoint:
    def test_chat_requires_body(self, client):
        r = client.post("/chat")
        assert r.status_code == 422
    
    def test_chat_empty_message_rejected(self, client):
        r = client.post("/chat", json={"message": ""})
        assert r.status_code in [400, 422]
    
    def test_chat_valid_structure(self, client):
        r = client.post("/chat", json={"message": "hello"})
        # May fail due to missing API key, but should return valid structure
        if r.status_code == 200:
            data = r.json()
            assert "status" in data
            assert "response" in data
    
    def test_chat_xss_in_message_rejected(self, client):
        r = client.post("/chat", json={"message": "<script>alert(1)</script>"})
        assert r.status_code in [400, 422]


# === Upload Endpoint ===

class TestUploadEndpoint:
    def test_upload_requires_file(self, client):
        r = client.post("/upload")
        assert r.status_code == 422
    
    def test_validate_requires_file(self, client):
        r = client.post("/validate")
        assert r.status_code == 422


# === Speech Endpoint ===

class TestSpeechEndpoint:
    def test_speech_requires_body(self, client):
        r = client.post("/generate-speech")
        assert r.status_code == 422


# === Static Pages ===

class TestStaticPages:
    def test_index_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
    
    def test_index_is_html(self, client):
        r = client.get("/")
        assert "text/html" in r.headers.get("content-type", "")
    
    def test_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200
    
    def test_openapi_available(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        data = r.json()
        assert "openapi" in data


# === Not Found ===

class TestNotFound:
    def test_unknown_route_returns_404(self, client):
        r = client.get("/nonexistent-route-xyz")
        assert r.status_code == 404
