"""
Middleware package for Medical Assistant Application.

Provides:
- SecurityHeadersMiddleware: CSP, XSS, frame protection headers
- RequestLoggingMiddleware: Structured request/response logging
- RateLimitMiddleware: Sliding window rate limiting per client IP
- RequestDeduplicationMiddleware: Coalesce identical in-flight requests
- APIKeyAuthMiddleware: API key authentication for protected endpoints
- CSRFProtection: CSRF token generation and validation
- sanitize_input / sanitize_filename / validate_mime_type: Input safety utilities
"""
from .security import (
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    CSRFProtection,
    sanitize_input,
    sanitize_filename,
    validate_mime_type,
    secure_error_response,
    CSP_POLICY,
    build_csp_header,
)
from .rate_limiter import RateLimitMiddleware
from .request_dedup import RequestDedupMiddleware, get_dedup_stats
from .api_key_auth import APIKeyAuthMiddleware, get_auth_stats

__all__ = [
    "SecurityHeadersMiddleware",
    "RequestLoggingMiddleware",
    "CSRFProtection",
    "RateLimitMiddleware",
    "RequestDedupMiddleware",
    "APIKeyAuthMiddleware",
    "sanitize_input",
    "sanitize_filename",
    "validate_mime_type",
    "secure_error_response",
    "CSP_POLICY",
    "build_csp_header",
    "get_dedup_stats",
    "get_auth_stats",
]
