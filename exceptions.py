"""Structured exception classes for the Medical Assistant API.

Provides consistent error responses across all endpoints with proper
HTTP status codes and machine-readable error types.
"""

from typing import Any, Dict, Optional


class MedicalAssistantError(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "internal_error",
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        result = {"error": self.error_type, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


class AgentError(MedicalAssistantError):
    """Errors from agent processing (router, LLM calls, etc.)."""

    def __init__(self, message: str, agent_name: str = "unknown", details: dict | None = None):
        super().__init__(
            message=message,
            status_code=500,
            error_type="agent_error",
            details={"agent": agent_name, **(details or {})},
        )


class ValidationError(MedicalAssistantError):
    """Input validation errors."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(
            message=message,
            status_code=400,
            error_type="validation_error",
            details={"field": field} if field else {},
        )


class RateLimitError(MedicalAssistantError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Too many requests. Please wait and try again.", retry_after: int = 60):
        super().__init__(
            message=message,
            status_code=429,
            error_type="rate_limit_exceeded",
            details={"retry_after_seconds": retry_after},
        )


class FileUploadError(MedicalAssistantError):
    """File upload validation errors."""

    def __init__(self, message: str, filename: str | None = None):
        super().__init__(
            message=message,
            status_code=400,
            error_type="file_upload_error",
            details={"filename": filename} if filename else {},
        )


class TTSError(MedicalAssistantError):
    """Text-to-speech generation errors."""

    def __init__(self, message: str, engine: str = "unknown"):
        super().__init__(
            message=message,
            status_code=500,
            error_type="tts_error",
            details={"engine": engine},
        )


class MCPError(MedicalAssistantError):
    """MCP client/server errors."""

    def __init__(self, message: str, server: str | None = None):
        super().__init__(
            message=message,
            status_code=503,
            error_type="mcp_error",
            details={"server": server} if server else {},
        )
