"""Pydantic request/response models for structured API contracts.

Phase 20: Centralized models for consistent API responses.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Request Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Chat query request with optional conversation history."""
    query: str = Field(..., min_length=1, max_length=4096, description="User query text")
    conversation_history: List = Field(default=[], description="Previous conversation messages")


class SpeechRequest(BaseModel):
    """TTS speech generation request."""
    text: str = Field(..., description="Text to synthesize")
    voice_id: str = Field(default="zh-CN-XiaoxiaoNeural", description="Edge TTS voice identifier")


# ─── Response Models ────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Standard chat response envelope."""
    status: str = "success"
    response: str = ""
    agent: str = "unknown"
    cached: Optional[bool] = None
    result_image: Optional[str] = None
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response envelope."""
    status: str = "error"
    detail: str
    agent: str = "System"
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response with dependency status."""
    status: str  # "healthy" | "degraded"
    version: str = "3.3.0"
    checks: Dict[str, str] = {}
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class MetricsResponse(BaseModel):
    """Prometheus-compatible metrics (rendered as text/plain)."""
    pass  # Rendered by observability module directly


# ─── Response Factory ───────────────────────────────────────────────

def api_success(
    response: str,
    agent: str = "unknown",
    cached: bool = False,
    result_image: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardized success response dict."""
    resp: Dict[str, Any] = {
        "status": "success",
        "response": response,
        "agent": agent,
    }
    if cached:
        resp["cached"] = True
    if result_image:
        resp["result_image"] = result_image
    if request_id:
        resp["request_id"] = request_id
    return resp


def api_error(
    detail: str,
    agent: str = "System",
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardized error response dict."""
    resp: Dict[str, Any] = {
        "status": "error",
        "detail": detail,
        "agent": agent,
    }
    if request_id:
        resp["request_id"] = request_id
    return resp
