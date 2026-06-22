"""Observability utilities: Prometheus-style metrics + structured JSON logging.

Provides /metrics endpoint for monitoring and JSON-formatted logs
for production log aggregation (ELK/Loki/etc).
"""

import json
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Optional

# ─── Prometheus-style Metrics ───────────────────────────────────────


class MetricsCollector:
    """In-process metrics collector (no prometheus_client dependency).

    Tracks counters, gauges, and histograms for key endpoints.
    Exposes /metrics in a Prometheus-compatible text format.
    """

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._start_time = time.time()

    def inc(self, name: str, value: float = 1.0, labels: Optional[Dict] = None):
        """Increment a counter."""
        key = self._metric_key(name, labels)
        self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Set a gauge value."""
        key = self._metric_key(name, labels)
        self._gauges[key] = value

    def observe(self, name: str, value: float, labels: Optional[Dict] = None):
        """Record a histogram observation."""
        key = self._metric_key(name, labels)
        self._histograms[key].append(value)

    @contextmanager
    def timer(self, name: str, labels: Optional[Dict] = None):
        """Context manager to time a block and record as histogram."""
        start = time.monotonic()
        yield
        elapsed = time.monotonic() - start
        self.observe(name, elapsed, labels)

    def _metric_key(self, name: str, labels: Optional[Dict] = None) -> str:
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def render(self) -> str:
        """Render all metrics in Prometheus text format."""
        lines = []
        uptime = time.time() - self._start_time

        # Uptime
        lines.append("# HELP medical_app_uptime_seconds Application uptime")
        lines.append("# TYPE medical_app_uptime_seconds gauge")
        lines.append(f"medical_app_uptime_seconds {uptime:.1f}")

        # Counters
        for key, val in sorted(self._counters.items()):
            name = key.split("{")[0]
            lines.append(f"# TYPE {name} counter")
            lines.append(f"medical_{key} {val}")

        # Gauges
        for key, val in sorted(self._gauges.items()):
            name = key.split("{")[0]
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"medical_{key} {val}")

        # Histograms
        for key, values in sorted(self._histograms.items()):
            name = key.split("{")[0]
            lines.append(f"# TYPE {name} histogram")
            if values:
                sorted_v = sorted(values)
                for q, label in [(0.5, "0.5"), (0.9, "0.9"), (0.99, "0.99")]:
                    idx = min(int(len(sorted_v) * q), len(sorted_v) - 1)
                    lines.append(f'medical_{name}_bucket{{le="{label}"}} {sorted_v[idx]:.4f}')
                lines.append(f"medical_{name}_sum {sum(values):.4f}")
                lines.append(f"medical_{name}_count {len(values)}")

        return "\n".join(lines) + "\n"


# Global singleton
metrics = MetricsCollector()


# ─── Structured JSON Logging ───────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Emit logs as single-line JSON for log aggregation systems."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        return json.dumps(log_entry, ensure_ascii=False)


def setup_json_logging(level: int = logging.INFO):
    """Replace default formatter with JSON on all handlers."""
    for handler in logging.root.handlers:
        handler.setFormatter(JSONFormatter())
    logging.root.setLevel(level)


# ─── Per-Agent Metrics (Phase 47) ────────────────────────────────────
# Tracks latency, call count, success/failure rate for each agent node.


class AgentMetrics:
    """Per-agent latency and throughput tracker.

    Usage:
        agent_metrics = AgentMetrics()
        with agent_metrics.track("RAG_AGENT"):
            result = run_rag_agent(state)
        print(agent_metrics.get_all_stats())
    """

    def __init__(self):
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "failures": 0,
                "total_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "min_latency_ms": float("inf"),
                "last_call_ts": 0.0,
            }
        )
        self._lock = threading.Lock()

    @contextmanager
    def track(self, agent_name: str):
        """Context manager to track agent call latency."""
        start = time.perf_counter()
        try:
            yield
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record(agent_name, elapsed_ms, success=True)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record(agent_name, elapsed_ms, success=False)
            raise

    def _record(self, agent_name: str, latency_ms: float, success: bool):
        with self._lock:
            s = self._stats[agent_name]
            s["calls"] += 1
            if not success:
                s["failures"] += 1
            s["total_latency_ms"] += latency_ms
            s["max_latency_ms"] = max(s["max_latency_ms"], latency_ms)
            s["min_latency_ms"] = min(s["min_latency_ms"], latency_ms)
            s["last_call_ts"] = time.time()

    def get_stats(self, agent_name: str) -> dict:
        """Get stats for a specific agent."""
        with self._lock:
            s = self._stats.get(agent_name)
            if not s:
                return {}
            calls = s["calls"]
            return {
                "calls": calls,
                "failures": s["failures"],
                "success_rate": round((calls - s["failures"]) / calls, 4) if calls else 0,
                "avg_latency_ms": round(s["total_latency_ms"] / calls, 2) if calls else 0,
                "max_latency_ms": round(s["max_latency_ms"], 2),
                "min_latency_ms": round(s["min_latency_ms"], 2) if s["min_latency_ms"] != float("inf") else 0,
            }

    def get_all_stats(self) -> dict:
        """Get stats for all tracked agents."""
        return {name: self.get_stats(name) for name in self._stats}

    def reset(self):
        """Reset all stats."""
        with self._lock:
            self._stats.clear()


# Global singleton instance
agent_metrics = AgentMetrics()
