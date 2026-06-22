"""Pydantic request/response models for structured API contracts.

Phase 20: Centralized models for consistent API responses.
Phase 54: Pydantic V2 modernization (model_config, field_validator, model_dump).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ─── Request Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Chat query request with optional conversation history."""

    model_config = {"json_schema_extra": {"examples": [{"query": "What is hypertension?"}]}}

    query: str = Field(..., min_length=1, max_length=4096, description="User query text")
    conversation_history: List = Field(default=[], description="Previous conversation messages")

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be empty or whitespace-only")
        return v


class SpeechRequest(BaseModel):
    """TTS speech generation request."""

    model_config = {"json_schema_extra": {"examples": [{"text": "Hello, how can I help you?"}]}}

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: str = Field(default="zh-CN-XiaoxiaoNeural", description="Edge TTS voice identifier")


# ─── Response Models ────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Standard chat response envelope."""

    model_config = {"json_schema_extra": {"examples": [{"status": "success", "response": "...", "agent": "decision"}]}}

    status: str = "success"
    response: str = ""
    agent: str = "unknown"
    cached: Optional[bool] = None
    result_image: Optional[str] = None
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    model_config = {"json_schema_extra": {"examples": [{"status": "error", "detail": "...", "agent": "System"}]}}

    status: str = "error"
    detail: str
    agent: str = "System"
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response with dependency status."""

    model_config = {"json_schema_extra": {"examples": [{"status": "healthy", "version": "3.3.0", "checks": {"llm": "ok"}}]}}

    status: str  # "healthy" | "degraded"
    version: str = "3.3.0"
    checks: Dict[str, str] = Field(default_factory=dict)
    timestamp: int = Field(default_factory=lambda: int(time.time()))

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in ("healthy", "degraded", "unhealthy"):
            raise ValueError(f"status must be healthy/degraded/unhealthy, got {v}")
        return v


class MetricsResponse(BaseModel):
    """Prometheus-compatible metrics (rendered as text/plain)."""
    pass  # Rendered by observability module directly


# ─── Agent Routing Models ───────────────────────────────────────────

class AgentRouteInfo(BaseModel):
    """Internal model for agent routing decisions."""

    agent_name: str = Field(..., description="Target agent identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Routing confidence score")
    reasoning: Optional[str] = Field(default=None, description="Why this agent was selected")


class ConversationMessage(BaseModel):
    """Typed conversation message (replaces raw dicts in history)."""

    role: str = Field(..., description="Message role: user/assistant/system")
    content: str = Field(..., min_length=1, description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"user", "assistant", "system"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got {v}")
        return v


# ─── Response Factory ───────────────────────────────────────────────

def api_success(
    response: str,
    agent: str = "unknown",
    cached: bool = False,
    result_image: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardized success response dict.

    Uses ChatResponse model for validation, then returns dict for backward compat.
    """
    model = ChatResponse(
        response=response,
        agent=agent,
        cached=cached if cached else None,
        result_image=result_image,
        request_id=request_id,
    )
    # model_dump with exclude_none for clean JSON
    return model.model_dump(exclude_none=True)


def api_error(
    detail: str,
    agent: str = "System",
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardized error response dict.

    Uses ErrorResponse model for validation, then returns dict for backward compat.
    """
    model = ErrorResponse(
        detail=detail,
        agent=agent,
        request_id=request_id,
    )
    return model.model_dump(exclude_none=True)


# ─── Structured Output Models (Phase 55) ────────────────────────────
# Pydantic models for enforcing structured LLM output via JSON mode.
# Used with agent_decision.py JsonOutputParser for consistent schemas.

class MedicalDiagnosis(BaseModel):
    """Structured diagnosis output from LLM."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "description": "Structured medical diagnosis for agent output",
            "examples": [{
                "condition": "Common Cold",
                "confidence": 0.85,
                "symptoms": ["runny nose", "sore throat"],
                "recommendations": ["rest", "hydration"],
                "urgency": "low",
                "disclaimer": "This is AI-generated guidance, not a medical diagnosis.",
            }],
        },
    )

    condition: str = Field(..., description="Suspected condition or diagnosis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    symptoms: List[str] = Field(default_factory=list, description="Identified symptoms")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    urgency: str = Field(default="medium", description="Urgency level: low/medium/high/critical")
    differential: List[str] = Field(default_factory=list, description="Differential diagnoses to consider")
    disclaimer: str = Field(
        default="This is AI-generated guidance, not a medical diagnosis. Please consult a healthcare professional.",
        description="Medical disclaimer",
    )

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "critical"}
        v = v.lower().strip()
        if v not in allowed:
            return "medium"
        return v


class MedicalReport(BaseModel):
    """Structured report output from LLM."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"description": "Structured medical report for report agent"},
    )

    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary")
    sections: List[Dict[str, str]] = Field(default_factory=list, description="Report sections [{title, content}]")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    disclaimer: str = Field(
        default="This report is AI-generated and should be reviewed by a medical professional.",
    )


class AgentRouteDecision(BaseModel):
    """Structured routing decision from the decision agent."""
    model_config = ConfigDict(str_strip_whitespace=True)

    agent: str = Field(..., description="Target agent name")
    reasoning: str = Field(default="", description="Why this agent was chosen")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    requires_vision: bool = Field(default=False, description="Whether image analysis is needed")
    requires_search: bool = Field(default=False, description="Whether web search is needed")


def get_structured_output_schema(model_class: type[BaseModel]) -> dict:
    """Get JSON Schema for a Pydantic model (useful for LLM tool definitions).

    Example usage with OpenAI API:
        tools = [{"type": "function", "function": {"name": "diagnose", "parameters": get_structured_output_schema(MedicalDiagnosis)}}]
    """
    return model_class.model_json_schema()
