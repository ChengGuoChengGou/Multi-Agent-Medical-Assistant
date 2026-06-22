"""Pydantic models for API request/response schemas (OpenAPI documentation)."""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# ── Request Models ──

class QueryRequest(BaseModel):
    """Request body for chat endpoint."""
    query: str = Field(..., description="User's medical question or message", min_length=1, max_length=5000)
    conversation_history: list = Field(default=[], description="Previous conversation turns for context")


class SpeechRequest(BaseModel):
    """Request body for text-to-speech endpoint."""
    text: str = Field(..., description="Text to convert to speech", min_length=1)
    voice_id: str = Field(default="EXAMPLE_VOICE_ID", description="ElevenLabs voice ID to use")


# ── Response Models ──

class HealthResponse(BaseModel):
    """Health check response for Docker / load balancer probes."""
    status: str = Field(..., example="healthy")
    middleware: Dict[str, bool] = Field(..., description="Middleware stack status")
    dedup_stats: Dict[str, Any] = Field(..., description="Request deduplication statistics")


class ChatResponse(BaseModel):
    """Response from the medical chat endpoint."""
    status: str = Field(..., example="success", description="Response status")
    response: str = Field(..., description="Agent's medical response text")
    agent: str = Field(..., description="Name of the agent that handled the query")
    result_image: Optional[str] = Field(default=None, description="URL to result image (e.g. skin lesion segmentation)")


class ValidateResponse(BaseModel):
    """Response from human validation endpoint."""
    status: str = Field(..., example="validated", description="Validation outcome: 'validated' or 'rejected'")
    message: str = Field(..., description="Human-readable status message")
    response: str = Field(..., description="Agent response after processing validation")
    comments: Optional[str] = Field(default=None, description="Validator comments (present if rejected)")


class TranscribeResponse(BaseModel):
    """Response from speech-to-text endpoint."""
    transcript: str = Field(..., description="Transcribed text from the audio")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
