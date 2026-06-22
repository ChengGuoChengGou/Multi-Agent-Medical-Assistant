"""Pydantic models for API request/response schemas (OpenAPI documentation).

All models use Field(examples=...) for rich OpenAPI schema rendering.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


# ── Request Models ──

class QueryRequest(BaseModel):
    """Request body for chat endpoint."""
    query: str = Field(
        ..., description="User's medical question or message",
        min_length=1, max_length=5000,
        examples=["What are the early symptoms of type 2 diabetes?"]
    )
    conversation_history: List[Dict[str, str]] = Field(
        default=[], description="Previous conversation turns for context",
        examples=[[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi, how can I help?"}]]
    )


class SpeechRequest(BaseModel):
    """Request body for text-to-speech endpoint."""
    text: str = Field(
        ..., description="Text to convert to speech", min_length=1,
        examples=["The patient was diagnosed with mild pneumonia."]
    )
    voice_id: str = Field(
        default="EXAMPLE_VOICE_ID", description="ElevenLabs voice ID to use",
        examples=["21m00Tcm4TlvDq8ikWAM"]
    )


# ── Response Models ──

class HealthResponse(BaseModel):
    """Health check response for Docker / load balancer probes.

    Returns middleware stack status and request deduplication statistics.
    """
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "status": "healthy",
            "middleware": {
                "rate_limiting": True,
                "security_headers": True,
                "request_logging": True,
                "request_deduplication": True
            },
            "dedup_stats": {
                "active_requests": 2,
                "total_deduped": 15,
                "cache_size": 8
            }
        }]
    })
    status: str = Field(..., description="Service status", examples=["healthy"])
    middleware: Dict[str, bool] = Field(..., description="Middleware stack status")
    dedup_stats: Dict[str, Any] = Field(..., description="Request deduplication statistics")


class ChatResponse(BaseModel):
    """Response from the medical chat endpoint.

    The `agent` field indicates which specialized agent handled the query
    (e.g. SKIN_LESION_AGENT, RAG_AGENT, WEB_SEARCH_AGENT).
    The optional `result_image` is returned only for image analysis queries.
    """
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "status": "success",
            "response": "Type 2 diabetes early symptoms include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision.",
            "agent": "RAG_AGENT"
        }]
    })
    status: str = Field(..., description="Response status", examples=["success"])
    response: str = Field(..., description="Agent's medical response text")
    agent: str = Field(..., description="Name of the agent that handled the query", examples=["RAG_AGENT"])
    result_image: Optional[str] = Field(
        default=None,
        description="URL path to result image (e.g. skin lesion segmentation plot)",
        examples=["/uploads/skin_lesion_output/segmentation_plot.png"]
    )


class ValidateResponse(BaseModel):
    """Response from human validation endpoint.

    Called when a doctor validates or rejects an AI-generated diagnosis.
    The `status` field reflects the validation outcome, and `comments`
    is populated when the diagnosis is rejected.
    """
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "status": "validated",
            "message": "Diagnosis validated successfully",
            "response": "The diagnosis has been confirmed and recorded.",
            "comments": None
        }]
    })
    status: str = Field(..., description="Validation outcome: 'validated' or 'rejected'", examples=["validated"])
    message: str = Field(..., description="Human-readable status message")
    response: str = Field(..., description="Agent response after processing validation")
    comments: Optional[str] = Field(
        default=None, description="Validator comments (present when rejected)",
        examples=["Recommend additional chest X-ray for confirmation"]
    )


class TranscribeResponse(BaseModel):
    """Response from speech-to-text endpoint (OpenAI Whisper)."""
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"transcript": "Patient presents with persistent cough and fever for 3 days."}]
    })
    transcript: str = Field(..., description="Transcribed text from the audio")


class ErrorResponse(BaseModel):
    """Standard error response returned by all endpoints on failure."""
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"error": "Unsupported file type", "details": "Allowed formats: PNG, JPG, JPEG"}]
    })
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
