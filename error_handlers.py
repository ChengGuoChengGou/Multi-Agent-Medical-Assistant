"""
Structured error response handlers (Phase 27).
Consistent JSON error format across all HTTP error codes.
"""
import logging
import time
import uuid
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger("medical_chatbot.errors")


def _error_body(code: int, message: str, request_id: str = None, details: dict = None) -> dict:
    """Build standardized error response body."""
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id or str(uuid.uuid4())[:8],
            "timestamp": int(time.time()),
        }
    }
    if details:
        body["error"]["details"] = details
    return body


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTPException with structured JSON response."""
    request_id = getattr(request.state, "request_id", None)
    body = _error_body(exc.status_code, str(exc.detail), request_id)
    logger.warning(f"[{request_id}] HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content=body)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors with structured JSON."""
    request_id = getattr(request.state, "request_id", None)
    errors = []
    for err in exc.errors():
        loc = " → ".join(str(l) for l in err.get("loc", []))
        errors.append({"field": loc, "message": err.get("msg", ""), "type": err.get("type", "")})
    body = _error_body(422, "Request validation failed", request_id, details={"validation_errors": errors})
    logger.warning(f"[{request_id}] Validation error: {errors}")
    return JSONResponse(status_code=422, content=body)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions (no internal details leaked)."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"[{request_id}] Unhandled exception: {type(exc).__name__}: {exc}", exc_info=True)
    body = _error_body(500, "Internal server error", request_id)
    return JSONResponse(status_code=500, content=body)


def register_error_handlers(app):
    """Register all structured error handlers on the FastAPI app."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("[Errors] Structured error handlers registered")
