"""
Unit tests for models.py — Pydantic request/response models & structured output schemas.
Run: python -m pytest tests/test_models.py -v
"""

import os
import sys
from typing import Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    AgentRouteDecision,
    AgentRouteInfo,
    ChatResponse,
    ConversationMessage,
    ErrorResponse,
    HealthResponse,
    MedicalDiagnosis,
    MedicalReport,
    QueryRequest,
    SpeechRequest,
    api_error,
    api_success,
    get_structured_output_schema,
)

# ── QueryRequest ────────────────────────────────────────────────────


class TestQueryRequest:
    def test_valid(self):
        r = QueryRequest(query="What is diabetes?")
        assert r.query == "What is diabetes?"
        assert r.conversation_history == []

    def test_strips_whitespace(self):
        r = QueryRequest(query="  hello  ")
        assert r.query == "hello"

    def test_rejects_empty_after_strip(self):
        with pytest.raises(Exception):
            QueryRequest(query="   ")

    def test_rejects_none(self):
        with pytest.raises(Exception):
            QueryRequest(query=None)

    def test_rejects_too_long(self):
        with pytest.raises(Exception):
            QueryRequest(query="x" * 4097)

    def test_accepts_max_length(self):
        r = QueryRequest(query="x" * 4096)
        assert len(r.query) == 4096

    def test_conversation_history_defaults_empty(self):
        r = QueryRequest(query="hi")
        assert r.conversation_history == []

    def test_conversation_history_custom(self):
        r = QueryRequest(query="hi", conversation_history=[{"role": "user", "content": "prev"}])
        assert len(r.conversation_history) == 1


# ── SpeechRequest ───────────────────────────────────────────────────


class TestSpeechRequest:
    def test_valid(self):
        r = SpeechRequest(text="Hello")
        assert r.text == "Hello"
        assert r.voice_id == "zh-CN-XiaoxiaoNeural"

    def test_custom_voice(self):
        r = SpeechRequest(text="hi", voice_id="en-US-AriaNeural")
        assert r.voice_id == "en-US-AriaNeural"

    def test_rejects_empty(self):
        with pytest.raises(Exception):
            SpeechRequest(text="")

    def test_rejects_too_long(self):
        with pytest.raises(Exception):
            SpeechRequest(text="x" * 5001)


# ── ChatResponse ────────────────────────────────────────────────────


class TestChatResponse:
    def test_defaults(self):
        r = ChatResponse()
        assert r.status == "success"
        assert r.response == ""
        assert r.agent == "unknown"
        assert r.cached is None
        assert r.result_image is None
        assert r.request_id is None

    def test_custom(self):
        r = ChatResponse(status="success", response="hi", agent="rag", cached=True, request_id="abc")
        assert r.response == "hi"
        assert r.agent == "rag"
        assert r.cached is True


# ── ErrorResponse ───────────────────────────────────────────────────


class TestErrorResponse:
    def test_defaults(self):
        r = ErrorResponse(detail="something broke")
        assert r.status == "error"
        assert r.detail == "something broke"
        assert r.agent == "System"

    def test_custom_agent(self):
        r = ErrorResponse(detail="fail", agent="rag_agent")
        assert r.agent == "rag_agent"


# ── HealthResponse ──────────────────────────────────────────────────


class TestHealthResponse:
    def test_valid_statuses(self):
        for s in ("healthy", "degraded", "unhealthy"):
            r = HealthResponse(status=s)
            assert r.status == s

    def test_invalid_status_raises(self):
        with pytest.raises(Exception):
            HealthResponse(status="ok")

    def test_version_default(self):
        r = HealthResponse(status="healthy")
        assert r.version == "3.3.0"

    def test_checks_default(self):
        r = HealthResponse(status="healthy")
        assert r.checks == {}

    def test_checks_custom(self):
        r = HealthResponse(status="degraded", checks={"llm": "ok", "db": "fail"})
        assert r.checks["db"] == "fail"

    def test_timestamp_auto_set(self):
        r = HealthResponse(status="healthy")
        assert r.timestamp > 0


# ── AgentRouteInfo ──────────────────────────────────────────────────


class TestAgentRouteInfo:
    def test_valid(self):
        r = AgentRouteInfo(agent_name="rag_agent")
        assert r.agent_name == "rag_agent"
        assert r.confidence == 1.0

    def test_custom_confidence(self):
        r = AgentRouteInfo(agent_name="x", confidence=0.75)
        assert r.confidence == 0.75

    def test_confidence_out_of_range(self):
        with pytest.raises(Exception):
            AgentRouteInfo(agent_name="x", confidence=1.5)

    def test_confidence_negative(self):
        with pytest.raises(Exception):
            AgentRouteInfo(agent_name="x", confidence=-0.1)


# ── ConversationMessage ─────────────────────────────────────────────


class TestConversationMessage:
    def test_valid_roles(self):
        for role in ("user", "assistant", "system"):
            m = ConversationMessage(role=role, content="hi")
            assert m.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(Exception):
            ConversationMessage(role="admin", content="hi")

    def test_content_required(self):
        with pytest.raises(Exception):
            ConversationMessage(role="user")


# ── MedicalDiagnosis ────────────────────────────────────────────────


class TestMedicalDiagnosis:
    def test_valid(self):
        d = MedicalDiagnosis(
            condition="Common Cold",
            confidence=0.85,
            symptoms=["runny nose"],
            recommendations=["rest"],
            urgency="low",
        )
        assert d.condition == "Common Cold"
        assert d.confidence == 0.85
        assert d.urgency == "low"

    def test_urgency_case_insensitive(self):
        d = MedicalDiagnosis(condition="x", confidence=0.5, urgency="HIGH")
        assert d.urgency == "high"

    def test_urgency_invalid_defaults_medium(self):
        d = MedicalDiagnosis(condition="x", confidence=0.5, urgency="unknown")
        assert d.urgency == "medium"

    def test_confidence_boundary(self):
        d = MedicalDiagnosis(condition="x", confidence=0.0)
        assert d.confidence == 0.0
        d2 = MedicalDiagnosis(condition="x", confidence=1.0)
        assert d2.confidence == 1.0

    def test_confidence_out_of_range(self):
        with pytest.raises(Exception):
            MedicalDiagnosis(condition="x", confidence=1.1)

    def test_defaults(self):
        d = MedicalDiagnosis(condition="x", confidence=0.5)
        assert d.symptoms == []
        assert d.recommendations == []
        assert d.differential == []
        assert "not a medical diagnosis" in d.disclaimer.lower()

    def test_strip_whitespace(self):
        d = MedicalDiagnosis(condition="  cold  ", confidence=0.5)
        assert d.condition == "cold"


# ── MedicalReport ───────────────────────────────────────────────────


class TestMedicalReport:
    def test_valid(self):
        r = MedicalReport(title="Report", summary="All good")
        assert r.title == "Report"
        assert r.summary == "All good"
        assert r.sections == []
        assert r.key_findings == []

    def test_full(self):
        r = MedicalReport(
            title="Full",
            summary="s",
            sections=[{"title": "s1", "content": "c1"}],
            key_findings=["f1"],
            recommendations=["r1"],
        )
        assert len(r.sections) == 1
        assert r.key_findings == ["f1"]

    def test_strip_whitespace(self):
        r = MedicalReport(title="  t  ", summary="  s  ")
        assert r.title == "t"
        assert r.summary == "s"


# ── AgentRouteDecision ──────────────────────────────────────────────


class TestAgentRouteDecision:
    def test_valid(self):
        d = AgentRouteDecision(agent="rag_agent")
        assert d.agent == "rag_agent"
        assert d.reasoning == ""
        assert d.confidence == 0.5
        assert d.requires_vision is False
        assert d.requires_search is False

    def test_custom(self):
        d = AgentRouteDecision(
            agent="image_agent",
            reasoning="image detected",
            confidence=0.9,
            requires_vision=True,
            requires_search=False,
        )
        assert d.requires_vision is True

    def test_confidence_out_of_range(self):
        with pytest.raises(Exception):
            AgentRouteDecision(agent="x", confidence=2.0)


# ── api_success / api_error factory functions ───────────────────────


class TestApiFactory:
    def test_api_success_basic(self):
        d = api_success("hello")
        assert d["status"] == "success"
        assert d["response"] == "hello"
        assert d["agent"] == "unknown"

    def test_api_success_with_agent(self):
        d = api_success("hi", agent="rag")
        assert d["agent"] == "rag"

    def test_api_success_excludes_none(self):
        d = api_success("hi")
        assert "cached" not in d
        assert "result_image" not in d
        assert "request_id" not in d

    def test_api_success_with_all_fields(self):
        d = api_success("hi", agent="a", cached=True, result_image="img.png", request_id="r1")
        assert d["cached"] is True
        assert d["result_image"] == "img.png"
        assert d["request_id"] == "r1"

    def test_api_error_basic(self):
        d = api_error("something broke")
        assert d["status"] == "error"
        assert d["detail"] == "something broke"
        assert d["agent"] == "System"

    def test_api_error_excludes_none(self):
        d = api_error("fail")
        assert "request_id" not in d

    def test_api_error_with_request_id(self):
        d = api_error("fail", request_id="r1")
        assert d["request_id"] == "r1"


# ── get_structured_output_schema ────────────────────────────────────


class TestGetStructuredOutputSchema:
    def test_returns_dict(self):
        schema = get_structured_output_schema(MedicalDiagnosis)
        assert isinstance(schema, dict)

    def test_has_required_fields(self):
        schema = get_structured_output_schema(MedicalDiagnosis)
        assert "properties" in schema
        assert "condition" in schema["properties"]
        assert "confidence" in schema["properties"]

    def test_report_schema(self):
        schema = get_structured_output_schema(MedicalReport)
        assert "title" in schema["properties"]
        assert "summary" in schema["properties"]

    def test_route_decision_schema(self):
        schema = get_structured_output_schema(AgentRouteDecision)
        assert "agent" in schema["properties"]
        assert "confidence" in schema["properties"]
