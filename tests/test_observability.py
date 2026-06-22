"""
Unit tests for observability.py — MetricsCollector, AgentMetrics, JSONFormatter.
Run: python -m pytest tests/test_observability.py -v
"""

import json
import logging
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability import AgentMetrics, JSONFormatter, MetricsCollector, setup_json_logging


# ── MetricsCollector ──
class TestMetricsCollector:
    def test_inc_default(self):
        m = MetricsCollector()
        m.inc("requests_total")
        assert m._counters["requests_total"] == 1.0

    def test_inc_with_value(self):
        m = MetricsCollector()
        m.inc("requests_total", 5.0)
        assert m._counters["requests_total"] == 5.0

    def test_inc_with_labels(self):
        m = MetricsCollector()
        m.inc("requests_total", 1.0, labels={"path": "/chat"})
        key = 'requests_total{path="/chat"}'
        assert m._counters[key] == 1.0

    def test_set_gauge(self):
        m = MetricsCollector()
        m.set_gauge("memory_mb", 512.0)
        assert m._gauges["memory_mb"] == 512.0

    def test_set_gauge_with_labels(self):
        m = MetricsCollector()
        m.set_gauge("cpu", 75.0, labels={"core": "0"})
        key = 'cpu{core="0"}'
        assert m._gauges[key] == 75.0

    def test_observe(self):
        m = MetricsCollector()
        m.observe("latency", 0.1)
        m.observe("latency", 0.2)
        assert len(m._histograms["latency"]) == 2
        assert m._histograms["latency"] == [0.1, 0.2]

    def test_timer(self):
        m = MetricsCollector()
        with m.timer("request_duration"):
            time.sleep(0.01)
        assert len(m._histograms["request_duration"]) == 1
        assert m._histograms["request_duration"][0] >= 0

    def test_metric_key_no_labels(self):
        m = MetricsCollector()
        assert m._metric_key("foo") == "foo"

    def test_metric_key_sorted_labels(self):
        m = MetricsCollector()
        key = m._metric_key("foo", {"b": "2", "a": "1"})
        assert key == 'foo{a="1",b="2"}'

    def test_render_contains_uptime(self):
        m = MetricsCollector()
        output = m.render()
        assert "medical_app_uptime_seconds" in output

    def test_render_contains_counters(self):
        m = MetricsCollector()
        m.inc("test_counter")
        output = m.render()
        assert "medical_test_counter" in output
        assert "counter" in output

    def test_render_contains_gauges(self):
        m = MetricsCollector()
        m.set_gauge("test_gauge", 42.0)
        output = m.render()
        assert "medical_test_gauge 42" in output
        assert "gauge" in output

    def test_render_contains_histogram(self):
        m = MetricsCollector()
        m.observe("test_hist", 0.5)
        m.observe("test_hist", 1.5)
        output = m.render()
        assert "medical_test_hist_bucket" in output
        assert "histogram" in output
        assert "0.5" in output
        assert "0.9" in output
        assert "0.99" in output


# ── AgentMetrics ──
class TestAgentMetrics:
    def test_track_success(self):
        am = AgentMetrics()
        with am.track("RAG"):
            pass  # no error
        stats = am.get_stats("RAG")
        assert stats["calls"] == 1
        assert stats["failures"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["avg_latency_ms"] >= 0

    def test_track_failure(self):
        am = AgentMetrics()
        with pytest.raises(ValueError), am.track("RAG"):
            raise ValueError("boom")
        stats = am.get_stats("RAG")
        assert stats["calls"] == 1
        assert stats["failures"] == 1
        assert stats["success_rate"] == 0.0

    def test_multiple_tracks(self):
        am = AgentMetrics()
        with am.track("RAG"):
            pass
        with am.track("RAG"):
            pass
        with pytest.raises(RuntimeError), am.track("RAG"):
            raise RuntimeError("fail")
        stats = am.get_stats("RAG")
        assert stats["calls"] == 3
        assert stats["failures"] == 1
        assert abs(stats["success_rate"] - 0.6667) < 0.01

    def test_max_min_latency(self):
        am = AgentMetrics()
        with am.track("X"):
            time.sleep(0.02)
        with am.track("X"):
            time.sleep(0.001)
        stats = am.get_stats("X")
        assert stats["max_latency_ms"] > stats["min_latency_ms"]

    def test_get_stats_unknown_agent(self):
        am = AgentMetrics()
        assert am.get_stats("NONEXISTENT") == {}

    def test_get_all_stats(self):
        am = AgentMetrics()
        with am.track("A"):
            pass
        with am.track("B"):
            pass
        all_stats = am.get_all_stats()
        assert "A" in all_stats
        assert "B" in all_stats

    def test_reset(self):
        am = AgentMetrics()
        with am.track("X"):
            pass
        assert am.get_stats("X")["calls"] == 1
        am.reset()
        assert am.get_all_stats() == {}

    def test_thread_safety(self):
        """Multiple threads tracking same agent should not corrupt stats."""
        am = AgentMetrics()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    with am.track("MT"):
                        pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert am.get_stats("MT")["calls"] == 200


# ── JSONFormatter ──
class TestJSONFormatter:
    def test_basic_format(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["msg"] == "hello"

    def test_with_exception(self):
        fmt = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error",
            args=(),
            exc_info=exc_info,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_with_request_id(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="req",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc-123"
        output = fmt.format(record)
        data = json.loads(output)
        assert data["request_id"] == "abc-123"

    def test_with_duration_ms(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="done",
            args=(),
            exc_info=None,
        )
        record.duration_ms = 42.5
        output = fmt.format(record)
        data = json.loads(output)
        assert data["duration_ms"] == 42.5

    def test_unicode(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="中文测试",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data["msg"] == "中文测试"


# ── setup_json_logging ──
class TestSetupJsonLogging:
    def test_setup_replaces_formatters(self):
        # Ensure at least one handler exists
        root = logging.getLogger()
        if not root.handlers:
            root.addHandler(logging.StreamHandler())
        setup_json_logging()
        for handler in root.handlers:
            assert isinstance(handler.formatter, JSONFormatter)
