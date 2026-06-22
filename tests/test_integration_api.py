"""
Integration tests for API endpoints using FastAPI TestClient.
Covers: health, metrics, chat, validate, upload, session cookie, middleware behaviors.

Note: conftest.py mocks heavy deps (agents.rag_agent etc.) but process_query
return shape must match what /chat endpoint expects:
  {"messages": [obj_with_.content], "agent_name": str}
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_process_query_return(text="mock reply", agent="GENERAL_AGENT"):
    """Build a mock return value matching process_query's expected shape."""
    msg = MagicMock()
    msg.content = text
    return {"messages": [msg], "agent_name": agent}


@pytest.fixture
def client():
    """Create a TestClient with all middleware and routes active."""
    from fastapi.testclient import TestClient

    from app import app

    return TestClient(app)


@pytest.fixture
def mock_chat():
    """Patch process_query + semantic cache for /chat endpoint."""
    mock_return = _make_process_query_return()
    with (
        patch("app.process_query", return_value=mock_return) as m,
        patch("app.semantic_get", return_value=None),
        patch("app.semantic_set"),
    ):
        yield m


@pytest.fixture
def mock_validate():
    """Patch process_query + semantic cache for /validate endpoint."""
    mock_return = _make_process_query_return(text="validation response")
    with (
        patch("app.process_query", return_value=mock_return),
        patch("app.semantic_get", return_value=None),
        patch("app.semantic_set"),
    ):
        yield


# ── Health & System Endpoints ─────────────────────────────────────────────


class TestHealthEndpoint:
    """Vertical slice: GET /health → full response contract."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_has_uptime(self, client):
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_has_middleware_status(self, client):
        data = client.get("/health").json()
        mw = data.get("middleware", {})
        assert mw.get("rate_limiting") is True
        assert mw.get("security_headers") is True
        assert mw.get("request_logging") is True
        assert mw.get("request_deduplication") is True
        assert mw.get("api_key_auth") is True

    def test_health_has_dedup_stats(self, client):
        data = client.get("/health").json()
        assert "dedup_stats" in data
        assert isinstance(data["dedup_stats"], dict)


class TestMetricsEndpoint:
    """Vertical slice: GET /metrics → Prometheus text format."""

    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_uptime(self, client):
        text = client.get("/metrics").text
        assert "medical_app_uptime_seconds" in text

    def test_metrics_contains_dedup_counters(self, client):
        text = client.get("/metrics").text
        assert "medical_app_dedup_total" in text
        assert "medical_app_dedup_hits" in text
        assert "medical_app_dedup_pending" in text

    def test_metrics_is_valid_exposition_format(self, client):
        """Lines with HELP/TYPE should precede value lines."""
        text = client.get("/metrics").text
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        help_lines = [l for l in lines if l.startswith("# HELP")]
        type_lines = [l for l in lines if l.startswith("# TYPE")]
        assert len(help_lines) >= 1
        assert len(type_lines) >= 1


# ── Index (HTML) ─────────────────────────────────────────────────────────


class TestIndexEndpoint:
    """Vertical slice: GET / → HTML template."""

    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]


# ── Chat Endpoint ─────────────────────────────────────────────────────────


class TestChatEndpoint:
    """Vertical slice: POST /chat → agent routing + session cookie."""

    def test_chat_success(self, client, mock_chat):
        resp = client.post("/chat", json={"query": "What is diabetes?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "response" in data
        assert "agent" in data

    def test_chat_sets_session_cookie(self, client, mock_chat):
        resp = client.post("/chat", json={"query": "test"})
        assert "session_id" in resp.cookies

    def test_chat_returns_agent_name(self, client, mock_chat):
        data = client.post("/chat", json={"query": "test"}).json()
        assert isinstance(data["agent"], str)
        assert len(data["agent"]) > 0

    def test_chat_empty_query(self, client, mock_chat):
        """Empty query should still route (agent decides)."""
        resp = client.post("/chat", json={"query": ""})
        assert resp.status_code == 200

    def test_chat_with_conversation_history(self, client, mock_chat):
        resp = client.post(
            "/chat",
            json={
                "query": "Follow up question",
                "conversation_history": [
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"},
                ],
            },
        )
        assert resp.status_code == 200


# ── Validate Endpoint ─────────────────────────────────────────────────────


class TestValidateEndpoint:
    """Vertical slice: POST /validate → human-in-the-loop validation."""

    def test_validate_approved(self, client, mock_validate):
        resp = client.post("/validate", data={"validation_result": "yes", "comments": "Looks correct"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "validated"
        assert "message" in data

    def test_validate_rejected(self, client, mock_validate):
        resp = client.post("/validate", data={"validation_result": "no", "comments": "Needs correction"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"
        assert data["comments"] == "Needs correction"

    def test_validate_without_comments(self, client, mock_validate):
        resp = client.post("/validate", data={"validation_result": "yes"})
        assert resp.status_code == 200

    def test_validate_sets_session_cookie(self, client, mock_validate):
        resp = client.post("/validate", data={"validation_result": "yes"})
        assert "session_id" in resp.cookies


# ── Upload Endpoint ───────────────────────────────────────────────────────


class TestUploadEndpoint:
    """Vertical slice: POST /upload → file type + size validation."""

    def test_upload_unsupported_extension(self, client):
        """Non-image file should be rejected with 400."""
        resp = client.post(
            "/upload", files={"image": ("test.txt", b"not an image", "text/plain")}, data={"text": "analyze this"}
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["status"] == "error"
        assert "Unsupported file type" in data["response"]

    @patch("app.config.api.max_image_upload_size", 1)  # 1 MB limit
    def test_upload_file_too_large(self, client):
        """File exceeding size limit should be rejected with 413."""
        large_content = b"x" * (2 * 1024 * 1024)  # 2 MB
        resp = client.post("/upload", files={"image": ("big.png", large_content, "image/png")}, data={"text": ""})
        assert resp.status_code == 413
        data = resp.json()
        assert data["status"] == "error"
        assert "File too large" in data["response"]


# ── Transcribe Endpoint ───────────────────────────────────────────────────


class TestTranscribeEndpoint:
    """Vertical slice: POST /transcribe → empty file rejection."""

    def test_transcribe_no_filename(self, client):
        """Empty filename should be rejected (400 by endpoint or 422 by FastAPI validation)."""
        resp = client.post("/transcribe", files={"audio": ("", b"", "audio/wav")})
        assert resp.status_code in (400, 422)


# ── Middleware Behavior ────────────────────────────────────────────────────


class TestSecurityHeaders:
    """Integration: security headers present on every response."""

    def test_has_x_content_type_options(self, client):
        resp = client.get("/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_has_x_frame_options(self, client):
        resp = client.get("/health")
        assert resp.headers.get("x-frame-options") in ("DENY", "SAMEORIGIN")

    def test_has_csp_header(self, client):
        resp = client.get("/health")
        assert "content-security-policy" in {k.lower() for k in resp.headers.keys()}


class TestCORS:
    """Integration: CORS headers for allowed origins."""

    def test_cors_preflight(self, client):
        resp = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in {k.lower() for k in resp.headers.keys()}


# ── OpenAPI Schema ────────────────────────────────────────────────────────


class TestOpenAPI:
    """Integration: OpenAPI schema is accessible and well-formed."""

    def test_openapi_returns_200(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_has_paths(self, client):
        schema = client.get("/openapi.json").json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/chat" in schema["paths"]

    def test_openapi_has_tags(self, client):
        schema = client.get("/openapi.json").json()
        tag_names = [t["name"] for t in schema.get("tags", [])]
        assert "System" in tag_names
        assert "Chat" in tag_names

    def test_docs_page_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_accessible(self, client):
        resp = client.get("/redoc")
        assert resp.status_code == 200
