"""
Rate Limiting Middleware for Medical Assistant Application
Implements sliding window rate limiting per client IP.
"""
import time
import asyncio
import logging
from collections import defaultdict
from typing import Dict, List

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter per client IP.
    
    Args:
        requests_per_minute: Max requests allowed per minute per IP
        requests_per_hour: Max requests allowed per hour per IP
    """

    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 500):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For for reverse proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = self._get_client_ip(request)
        now = time.time()

        # Skip rate limiting for static files and health checks
        path = request.url.path
        if path.startswith(("/static", "/data", "/uploads", "/favicon", "/health")):
            return await call_next(request)

        async with self._lock:
            timestamps = self._requests[client_ip]

            # Clean old entries (sliding window)
            cutoff_hour = now - 3600
            timestamps[:] = [t for t in timestamps if t > cutoff_hour]

            # Check hourly limit
            if len(timestamps) >= self.rph:
                logger.warning(f"Rate limit (hourly) exceeded for {client_ip}: {len(timestamps)} requests")
                return JSONResponse(
                    status_code=429,
                    content={"status": "error", "error": "Too many requests. Please try again later."},
                    headers={"Retry-After": "60"}
                )

            # Check per-minute limit
            cutoff_minute = now - 60
            recent_count = sum(1 for t in timestamps if t > cutoff_minute)
            if recent_count >= self.rpm:
                logger.warning(f"Rate limit (per-minute) exceeded for {client_ip}: {recent_count} requests/min")
                return JSONResponse(
                    status_code=429,
                    content={"status": "error", "error": "Too many requests. Please slow down."},
                    headers={"Retry-After": "10"}
                )

            # Record this request
            timestamps.append(now)

        response = await call_next(request)
        return response


def get_rate_limit_stats(requests_store: dict) -> dict:
    """Return rate limit statistics for monitoring."""
    active_ips = len(requests_store)
    return {
        "active_client_ips": active_ips,
    }
