"""
Unit tests for exceptions.py — Structured exception hierarchy.
Run: python -m pytest tests/test_exceptions.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exceptions import (
    AgentError,
    FileUploadError,
    MCPError,
    MedicalAssistantError,
    RateLimitError,
    TTSError,
    ValidationError,
)

# ── MedicalAssistantError (base) ────────────────────────────────────

class TestMedicalAssistantError:
    def test_defaults(self):
        e = MedicalAssistantError("boom")
        assert e.message == "boom"
        assert e.status_code == 500
        assert e.error_type == "internal_error"
        assert e.details == {}
        assert str(e) == "boom"

    def test_custom_fields(self):
        e = MedicalAssistantError("bad", status_code=418, error_type="teapot", details={"x": 1})
        assert e.status_code == 418
        assert e.error_type == "teapot"
        assert e.details == {"x": 1}

    def test_is_exception(self):
        assert issubclass(MedicalAssistantError, Exception)

    def test_to_dict_minimal(self):
        e = MedicalAssistantError("err")
        d = e.to_dict()
        assert d == {"error": "internal_error", "message": "err"}

    def test_to_dict_with_details(self):
        e = MedicalAssistantError("err", error_type="custom", details={"key": "val"})
        d = e.to_dict()
        assert d == {"error": "custom", "message": "err", "details": {"key": "val"}}

    def test_to_dict_empty_details_excluded(self):
        e = MedicalAssistantError("no details")
        d = e.to_dict()
        assert "details" not in d

    def test_catch_as_base(self):
        """All subclasses should be catchable as MedicalAssistantError."""
        for exc_cls in [AgentError, ValidationError, RateLimitError, FileUploadError, TTSError, MCPError]:
            try:
                raise exc_cls("test")
            except MedicalAssistantError as caught:
                assert caught.message == "test"


# ── AgentError ──────────────────────────────────────────────────────

class TestAgentError:
    def test_defaults(self):
        e = AgentError("agent fail")
        assert e.status_code == 500
        assert e.error_type == "agent_error"
        assert e.details["agent"] == "unknown"

    def test_custom_agent_name(self):
        e = AgentError("fail", agent_name="rag_agent")
        assert e.details["agent"] == "rag_agent"

    def test_extra_details_merged(self):
        e = AgentError("fail", agent_name="x", details={"code": 42})
        assert e.details["agent"] == "x"
        assert e.details["code"] == 42

    def test_to_dict_structure(self):
        e = AgentError("fail", agent_name="report")
        d = e.to_dict()
        assert d["error"] == "agent_error"
        assert d["details"]["agent"] == "report"


# ── ValidationError ─────────────────────────────────────────────────

class TestValidationError:
    def test_defaults(self):
        e = ValidationError("bad input")
        assert e.status_code == 400
        assert e.error_type == "validation_error"
        assert e.details == {}

    def test_with_field(self):
        e = ValidationError("required", field="query")
        assert e.details == {"field": "query"}

    def test_to_dict_structure(self):
        e = ValidationError("missing", field="email")
        d = e.to_dict()
        assert d["error"] == "validation_error"
        assert d["details"]["field"] == "email"


# ── RateLimitError ──────────────────────────────────────────────────

class TestRateLimitError:
    def test_defaults(self):
        e = RateLimitError()
        assert e.status_code == 429
        assert e.error_type == "rate_limit_exceeded"
        assert "Too many" in e.message
        assert e.details["retry_after_seconds"] == 60

    def test_custom_retry_after(self):
        e = RateLimitError(retry_after=120)
        assert e.details["retry_after_seconds"] == 120

    def test_custom_message(self):
        e = RateLimitError(message="slow down")
        assert e.message == "slow down"

    def test_to_dict_structure(self):
        e = RateLimitError(retry_after=30)
        d = e.to_dict()
        assert d["error"] == "rate_limit_exceeded"
        assert d["details"]["retry_after_seconds"] == 30


# ── FileUploadError ─────────────────────────────────────────────────

class TestFileUploadError:
    def test_defaults(self):
        e = FileUploadError("bad file")
        assert e.status_code == 400
        assert e.error_type == "file_upload_error"
        assert e.details == {}

    def test_with_filename(self):
        e = FileUploadError("too large", filename="test.pdf")
        assert e.details == {"filename": "test.pdf"}

    def test_to_dict_structure(self):
        e = FileUploadError("invalid", filename="bad.exe")
        d = e.to_dict()
        assert d["error"] == "file_upload_error"
        assert d["details"]["filename"] == "bad.exe"


# ── TTSError ────────────────────────────────────────────────────────

class TestTTSError:
    def test_defaults(self):
        e = TTSError("speech fail")
        assert e.status_code == 500
        assert e.error_type == "tts_error"
        assert e.details["engine"] == "unknown"

    def test_custom_engine(self):
        e = TTSError("fail", engine="edge-tts")
        assert e.details["engine"] == "edge-tts"

    def test_to_dict_structure(self):
        e = TTSError("timeout", engine="gtts")
        d = e.to_dict()
        assert d["error"] == "tts_error"
        assert d["details"]["engine"] == "gtts"


# ── MCPError ────────────────────────────────────────────────────────

class TestMCPError:
    def test_defaults(self):
        e = MCPError("mcp down")
        assert e.status_code == 503
        assert e.error_type == "mcp_error"
        assert e.details == {}

    def test_with_server(self):
        e = MCPError("timeout", server="medical-mcp")
        assert e.details == {"server": "medical-mcp"}

    def test_to_dict_structure(self):
        e = MCPError("fail", server="search-mcp")
        d = e.to_dict()
        assert d["error"] == "mcp_error"
        assert d["details"]["server"] == "search-mcp"

    def test_no_server_excludes_from_details(self):
        e = MCPError("fail")
        d = e.to_dict()
        assert "server" not in d.get("details", {})


# ── Inheritance Chain ───────────────────────────────────────────────

class TestInheritanceChain:
    def test_all_subclasses_inherit_from_base(self):
        for cls in [AgentError, ValidationError, RateLimitError, FileUploadError, TTSError, MCPError]:
            assert issubclass(cls, MedicalAssistantError), f"{cls.__name__} not subclass"

    def test_catch_all_with_base(self):
        """One except clause catches all custom errors."""
        errors = [
            AgentError("a"),
            ValidationError("v"),
            RateLimitError(),
            FileUploadError("f"),
            TTSError("t"),
            MCPError("m"),
        ]
        caught = []
        for err in errors:
            try:
                raise err
            except MedicalAssistantError as e:
                caught.append(e)
        assert len(caught) == 6
