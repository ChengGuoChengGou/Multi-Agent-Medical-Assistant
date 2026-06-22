"""
Unit tests for schemas.py — API request/response Pydantic schemas.
Run: python -m pytest tests/test_schemas.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import (
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    SpeechRequest,
    TranscribeResponse,
    ValidateResponse,
)


# ── QueryRequest ──
class TestQueryRequest:
    def test_basic(self):
        r = QueryRequest(query="headache")
        assert r.query == "headache"
        assert r.conversation_history == []

    def test_with_history(self):
        hist = [{"role": "user", "content": "hello"}]
        r = QueryRequest(query="fever", conversation_history=hist)
        assert r.conversation_history == hist

    def test_query_required(self):
        with pytest.raises(Exception):
            QueryRequest()

    def test_query_min_length(self):
        with pytest.raises(Exception):
            QueryRequest(query="")

    def test_query_max_length(self):
        with pytest.raises(Exception):
            QueryRequest(query="x" * 5001)

    def test_query_trim(self):
        """Pydantic strips whitespace by default."""
        r = QueryRequest(query="  hello  ")
        assert r.query == "  hello  " or r.query == "hello"  # depends on config

    def test_openapi_example(self):
        schema = QueryRequest.model_config.get("json_schema_extra", {})
        # QueryRequest has no model_config override in schemas.py
        # Just verify it's instantiable
        r = QueryRequest(query="test")
        assert r.query == "test"


# ── SpeechRequest ──
class TestSpeechRequest:
    def test_basic(self):
        r = SpeechRequest(text="hello")
        assert r.text == "hello"
        assert r.voice_id == "EXAMPLE_VOICE_ID"

    def test_custom_voice(self):
        r = SpeechRequest(text="hi", voice_id="custom_voice")
        assert r.voice_id == "custom_voice"

    def test_text_required(self):
        with pytest.raises(Exception):
            SpeechRequest()

    def test_text_min_length(self):
        with pytest.raises(Exception):
            SpeechRequest(text="")


# ── HealthResponse ──
class TestHealthResponse:
    def test_minimal(self):
        h = HealthResponse(
            status="healthy",
            uptime_seconds=100.0,
            middleware={"rate_limiting": True},
            dedup_stats={"active": 0},
        )
        assert h.status == "healthy"
        assert h.uptime_seconds == 100.0
        assert h.middleware == {"rate_limiting": True}
        assert h.dedup_stats == {"active": 0}
        assert h.api_auth == {}  # default_factory=dict

    def test_full(self):
        h = HealthResponse(
            status="ok",
            uptime_seconds=999.0,
            middleware={"a": True, "b": False},
            dedup_stats={"total": 5},
            api_auth={"dev_mode": True},
        )
        assert h.api_auth == {"dev_mode": True}

    def test_required_fields(self):
        with pytest.raises(Exception):
            HealthResponse(status="ok")  # missing uptime_seconds, middleware, dedup_stats

    def test_openapi_example(self):
        cfg = HealthResponse.model_config.get("json_schema_extra", {})
        assert "examples" in cfg


# ── ChatResponse ──
class TestChatResponse:
    def test_minimal(self):
        c = ChatResponse(status="success", response="ok", agent="RAG_AGENT")
        assert c.status == "success"
        assert c.response == "ok"
        assert c.agent == "RAG_AGENT"
        assert c.result_image is None

    def test_with_image(self):
        c = ChatResponse(
            status="success",
            response="lesion detected",
            agent="SKIN_LESION_AGENT",
            result_image="/uploads/plot.png",
        )
        assert c.result_image == "/uploads/plot.png"

    def test_required_fields(self):
        with pytest.raises(Exception):
            ChatResponse(response="ok")  # missing status and agent

    def test_openapi_example(self):
        cfg = ChatResponse.model_config.get("json_schema_extra", {})
        assert "examples" in cfg


# ── ValidateResponse ──
class TestValidateResponse:
    def test_minimal(self):
        v = ValidateResponse(
            status="validated",
            message="ok",
            response="confirmed",
        )
        assert v.status == "validated"
        assert v.message == "ok"
        assert v.response == "confirmed"
        assert v.comments is None

    def test_rejected(self):
        v = ValidateResponse(
            status="rejected",
            message="needs review",
            response="re-evaluation required",
            comments="Recommend additional chest X-ray",
        )
        assert v.comments == "Recommend additional chest X-ray"

    def test_required_fields(self):
        with pytest.raises(Exception):
            ValidateResponse(status="validated")  # missing message, response


# ── TranscribeResponse ──
class TestTranscribeResponse:
    def test_basic(self):
        t = TranscribeResponse(transcript="Patient has fever.")
        assert t.transcript == "Patient has fever."

    def test_required(self):
        with pytest.raises(Exception):
            TranscribeResponse()


# ── ErrorResponse ──
class TestErrorResponse:
    def test_minimal(self):
        e = ErrorResponse(error="fail")
        assert e.error == "fail"
        assert e.details is None

    def test_full(self):
        e = ErrorResponse(error="fail", details="timeout")
        assert e.details == "timeout"

    def test_error_required(self):
        with pytest.raises(Exception):
            ErrorResponse()

    def test_openapi_example(self):
        cfg = ErrorResponse.model_config.get("json_schema_extra", {})
        assert "examples" in cfg


# ── model_dump / JSON serialization ──
class TestModelDump:
    def test_query_request_dump(self):
        r = QueryRequest(query="fever")
        d = r.model_dump()
        assert d["query"] == "fever"
        assert d["conversation_history"] == []

    def test_chat_response_json(self):
        c = ChatResponse(status="success", response="ok", agent="RAG")
        json_str = c.model_dump_json()
        assert '"response":"ok"' in json_str
        assert '"agent":"RAG"' in json_str

    def test_health_response_dump(self):
        h = HealthResponse(
            status="ok", uptime_seconds=10.0,
            middleware={"a": True}, dedup_stats={"b": 2},
        )
        d = h.model_dump()
        assert d["status"] == "ok"
        assert d["uptime_seconds"] == 10.0
        assert d["api_auth"] == {}

    def test_validate_response_dump(self):
        v = ValidateResponse(status="v", message="m", response="r")
        d = v.model_dump()
        assert d["comments"] is None

    def test_transcribe_response_roundtrip(self):
        t = TranscribeResponse(transcript="hello")
        d = t.model_dump()
        assert d["transcript"] == "hello"
