"""
Structured logging configuration for Medical Assistant.

Features:
- JSON-formatted logs for production (machine-parseable)
- Human-readable format for development
- request_id propagation via contextvars
- Automatic log level from LOG_LEVEL env var

Usage:
    from utils.logging_config import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
    logger.info("query_processed", extra={"query": "headache", "duration_ms": 1234})
"""

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# ── request_id context (injected by middleware, read by logger) ──
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per line (for log aggregators: ELK, Loki, CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get("-"),
        }
        # Attach any extra fields passed via logger.info(..., extra={...})
        for key in ("duration_ms", "status_code", "method", "path", "client_ip", "query", "session_id", "error_type"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Human-readable format for local development."""

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get("-")
        short_rid = rid[:8] if rid != "-" else "-"
        return (
            f"{datetime.fromtimestamp(record.created).strftime('%H:%M:%S')} "
            f"[{record.levelname:5s}] [{short_rid}] "
            f"{record.name}: {record.getMessage()}"
        )


def setup_logging(level: str | None = None, json_format: bool | None = None):
    """Configure root logger. Call once at app startup.

    Args:
        level: Log level (default from LOG_LEVEL env or INFO)
        json_format: True=JSON, False=human (default: JSON if LOG_FORMAT=json env)
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if json_format is None:
        json_format = os.environ.get("LOG_FORMAT", "json").lower() == "json"

    root = logging.getLogger()
    # Remove existing handlers to avoid duplicates on reload
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_format else HumanFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))

    # Quiet noisy libraries
    for noisy in ("httpx", "httpcore", "urllib3", "langchain", "openai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialized", extra={"level": level, "json": json_format})


def get_logger(name: str) -> logging.Logger:
    """Get a named logger (convenience wrapper)."""
    return logging.getLogger(name)
