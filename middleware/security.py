"""
Security Middleware for Medical Assistant Application
Implements CSP, security headers, CSRF protection, and input sanitization.
"""
import os
import hmac
import hashlib
import secrets
import time
import logging
import re
import html
from typing import Optional, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ============================================================
# 1. CSP (Content Security Policy) Configuration
# ============================================================
CSP_POLICY = {
    "default-src": ["'self'"],
    "script-src": ["'self'", "'unsafe-inline'"],
    "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com", "https://cdnjs.cloudflare.com"],
    "font-src": ["'self'", "https://fonts.gstatic.com", "https://cdnjs.cloudflare.com"],
    "img-src": ["'self'", "data:", "blob:"],
    "media-src": ["'self'", "blob:"],
    "connect-src": ["'self'"],
    "frame-ancestors": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
}

def build_csp_header() -> str:
    """Build CSP header string from policy dict."""
    parts = []
    for directive, sources in CSP_POLICY.items():
        parts.append(f"{directive} {' '.join(sources)}")
    return "; ".join(parts)


# ============================================================
# 2. CSRF Token Management
# ============================================================
class CSRFProtection:
    """Simple CSRF token implementation using HMAC."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key

    def generate_token(self, session_id: str) -> str:
        """Generate a CSRF token for the given session."""
        timestamp = str(int(time.time()))
        message = f"{session_id}:{timestamp}"
        signature = hmac.new(self.secret_key, message.encode(), hashlib.sha256).hexdigest()
        return f"{timestamp}.{signature}"

    def validate_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate a CSRF token (valid for max_age seconds)."""
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return False

            timestamp_str, signature = parts
            timestamp = int(timestamp_str)

            # Check token age
            if time.time() - timestamp > max_age:
                return False

            # Verify signature
            message = f"{session_id}:{timestamp_str}"
            expected_signature = hmac.new(self.secret_key, message.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature, expected_signature)
        except (ValueError, AttributeError):
            return False


# ============================================================
# 3. Security Headers Middleware
# ============================================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Core security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # CSP header
        response.headers["Content-Security-Policy"] = build_csp_header()

        # HSTS (only if serving over HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Remove server header if present
        if "server" in response.headers:
            del response.headers["server"]

        return response


# ============================================================
# 4. Input Sanitization
# ============================================================
def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user input to prevent XSS and limit length."""
    if not isinstance(text, str):
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove null bytes
    text = text.replace('\x00', '')

    # HTML escape
    text = html.escape(text, quote=True)

    return text.strip()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    # Remove path separators and null bytes
    filename = os.path.basename(filename)
    filename = filename.replace('\x00', '')

    # Only allow safe characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]

    return f"{name}{ext}"


# ============================================================
# 5. MIME Type Validation
# ============================================================
ALLOWED_MIME_TYPES = {
    'image/png': ['.png'],
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/gif': ['.gif'],
    'image/bmp': ['.bmp'],
    'image/tiff': ['.tiff', '.tif'],
    'application/dicom': ['.dcm'],
    'application/octet-stream': ['.nii', '.nii.gz', '.mha'],
}

def validate_mime_type(content: bytes, filename: str) -> bool:
    """Validate file MIME type against content and extension."""
    # Check file magic bytes for common image formats
    if content[:8] == b'\x89PNG\r\n\x1a\n':
        return filename.lower().endswith('.png')
    if content[:2] == b'\xff\xd8':
        return filename.lower().endswith(('.jpg', '.jpeg'))
    if content[:4] == b'GIF8':
        return filename.lower().endswith('.gif')
    if content[:2] == b'BM':
        return filename.lower().endswith('.bmp')

    # For medical formats, allow based on extension (no reliable magic bytes)
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.nii', '.gz', '.mha', '.dcm'):
        return True

    # For other image types, check extension
    for mime, extensions in ALLOWED_MIME_TYPES.items():
        if ext in extensions:
            return True

    return False


# ============================================================
# 6. Secure Error Responses
# ============================================================
ERROR_MESSAGES = {
    400: "Bad request. Please check your input.",
    401: "Authentication required.",
    403: "Access denied.",
    404: "Resource not found.",
    413: "File too large.",
    415: "Unsupported media type.",
    429: "Too many requests. Please try again later.",
    500: "An internal error occurred. Please try again later.",
}

def secure_error_response(status_code: int, detail: str = None, log_error: bool = True) -> JSONResponse:
    """Return a secure error response that doesn't leak internal details."""
    safe_message = ERROR_MESSAGES.get(status_code, ERROR_MESSAGES[500])

    if log_error and detail:
        logger.error(f"Error {status_code}: {detail}")

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error": safe_message
        }
    )


# ============================================================
# 7. Request Logging Middleware
# ============================================================
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests for security auditing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")

        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(f"Response: {response.status_code} ({duration:.3f}s)")

        return response


# ============================================================
# 8. Exported Utilities
# ============================================================
__all__ = [
    'CSP_POLICY',
    'build_csp_header',
    'CSRFProtection',
    'SecurityHeadersMiddleware',
    'RequestLoggingMiddleware',
    'sanitize_input',
    'sanitize_filename',
    'validate_mime_type',
    'secure_error_response',
]
