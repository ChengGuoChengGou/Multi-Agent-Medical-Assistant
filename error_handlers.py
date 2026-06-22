"""
Structured error response handlers (Phase 27 + Phase 41).
Content-negotiation: HTML for browsers, JSON for API clients.
"""

import logging
import time
import uuid

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("medical_chatbot.errors")

# ===== HTML templates (Phase 41) =====
_ERROR_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>{code} {title}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;background:#f8f9fa;color:#333}}
.box{{text-align:center;padding:2rem;max-width:500px}}
h1{{font-size:6rem;margin:0;color:#e74c3c}}
h2{{margin:0.5rem 0 1rem;color:#555}}
p{{color:#777;line-height:1.6}}
code{{background:#eee;padding:2px 6px;border-radius:3px;font-size:0.85rem}}
a{{color:#3498db;text-decoration:none}}a:hover{{text-decoration:underline}}
</style></head>
<body><div class="box">
<h1>{code}</h1><h2>{title}</h2><p>{message}</p>
{extra}<p><a href="/">← 返回首页</a></p>
</div></body></html>"""


def _is_browser(request: Request) -> bool:
    """Detect if request comes from a browser (Accept header contains text/html)."""
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept


def _error_body(code: int, message: str, request_id: str | None = None, details: dict | None = None) -> dict:
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
    """Handle HTTPException - HTML for browsers, JSON for API clients."""
    request_id = getattr(request.state, "request_id", None)
    body = _error_body(exc.status_code, str(exc.detail), request_id)
    logger.warning(f"[{request_id}] HTTP {exc.status_code}: {exc.detail}")
    if _is_browser(request):
        title_map = {404: "页面未找到", 403: "禁止访问", 401: "未授权"}
        return HTMLResponse(
            status_code=exc.status_code,
            content=_ERROR_HTML_TEMPLATE.format(
                code=exc.status_code,
                title=title_map.get(exc.status_code, "请求错误"),
                message=str(exc.detail),
                extra=f"<p>请求ID: <code>{request_id}</code></p>" if request_id else "",
            ),
        )
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
    if _is_browser(request):
        return HTMLResponse(
            status_code=500,
            content=_ERROR_HTML_TEMPLATE.format(
                code=500,
                title="服务器内部错误",
                message="服务器遇到了意外错误，请稍后重试。",
                extra=f"<p>请求ID: <code>{request_id}</code></p>" if request_id else "",
            ),
        )
    return JSONResponse(status_code=500, content=body)


def register_error_handlers(app):
    """Register all structured error handlers on the FastAPI app."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("[Errors] Structured error handlers registered")
