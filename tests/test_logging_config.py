"""
Unit tests for utils/logging_config.py — JSON/Human formatters, setup_logging, get_logger.
Run: python -m pytest tests/test_logging_config.py -v
"""
import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import (
    HumanFormatter,
    JSONFormatter,
    get_logger,
    request_id_var,
    setup_logging,
)


# ── JSONFormatter ──
class TestJSONFormatter:
    def _make_record(self, level=logging.INFO, msg="test message", **extra):
        record = logging.LogRecord(
            name="test.logger", level=level, pathname="test.py",
            lineno=1, msg=msg, args=(), exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_basic_json_output(self):
        fmt = JSONFormatter()
        output = fmt.format(self._make_record())
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["msg"] == "test message"
        assert data["logger"] == "test.logger"
        assert "ts" in data

    def test_request_id_propagation(self):
        token = request_id_var.set("req-abc-123")
        try:
            fmt = JSONFormatter()
            data = json.loads(fmt.format(self._make_record()))
            assert data["request_id"] == "req-abc-123"
        finally:
            request_id_var.reset(token)

    def test_request_id_default(self):
        fmt = JSONFormatter()
        data = json.loads(fmt.format(self._make_record()))
        assert data["request_id"] == "-"

    def test_extra_fields_included(self):
        fmt = JSONFormatter()
        data = json.loads(fmt.format(self._make_record(
            duration_ms=123.4, status_code=200, method="POST", path="/chat",
        )))
        assert data["duration_ms"] == 123.4
        assert data["status_code"] == 200
        assert data["method"] == "POST"
        assert data["path"] == "/chat"

    def test_extra_fields_not_present_when_none(self):
        fmt = JSONFormatter()
        data = json.loads(fmt.format(self._make_record()))
        assert "duration_ms" not in data
        assert "status_code" not in data

    def test_exception_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="t.py",
            lineno=1, msg="err", args=(), exc_info=exc_info,
        )
        fmt = JSONFormatter()
        data = json.loads(fmt.format(record))
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_unicode(self):
        fmt = JSONFormatter()
        data = json.loads(fmt.format(self._make_record(msg="头痛")))
        assert data["msg"] == "头痛"


# ── HumanFormatter ──
class TestHumanFormatter:
    def test_basic_output(self):
        fmt = HumanFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="t.py",
            lineno=1, msg="hello", args=(), exc_info=None,
        )
        output = fmt.format(record)
        assert "INFO" in output
        assert "hello" in output
        assert "test" in output

    def test_with_request_id(self):
        token = request_id_var.set("req-xyz-789")
        try:
            fmt = HumanFormatter()
            record = logging.LogRecord(
                name="test", level=logging.DEBUG, pathname="t.py",
                lineno=1, msg="debug msg", args=(), exc_info=None,
            )
            output = fmt.format(record)
            assert "req-xyz-" in output  # first 8 chars
        finally:
            request_id_var.reset(token)


# ── setup_logging ──
class TestSetupLogging:
    def test_setup_replaces_handlers(self):
        root = logging.getLogger()
        initial_count = len(root.handlers)
        setup_logging(level="WARNING", json_format=True)
        # Should have exactly 1 handler after setup
        assert len(root.handlers) == 1

    def test_setup_human_format(self):
        setup_logging(level="DEBUG", json_format=False)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, HumanFormatter)

    def test_setup_json_format(self):
        setup_logging(level="INFO", json_format=True)
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_noisy_libraries_quieted(self):
        setup_logging(level="INFO")
        for name in ("httpx", "httpcore", "urllib3", "langchain", "openai", "chromadb"):
            assert logging.getLogger(name).level == logging.WARNING


# ── get_logger ──
class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("my_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my_module"

    def test_returns_different_instances(self):
        l1 = get_logger("mod_a")
        l2 = get_logger("mod_b")
        assert l1 is not l2
