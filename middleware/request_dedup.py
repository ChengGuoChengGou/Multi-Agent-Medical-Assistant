"""Request Deduplication Middleware (Phase 55).

Coalesces identical in-flight requests: if the same query is already being
processed, subsequent callers wait for the first result instead of spawning
duplicate LLM calls. Uses asyncio.Event for wait/notify pattern.

Inspired by: GPTCache dedup pattern, early-arrival coalescing in distributed
systems (Netflix Hystrix, Spring Cloud Gateway).

Zero external dependencies — stdlib asyncio only.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("medical_chatbot.dedup")

# In-flight request registry: key → _InFlightEntry
_inflight: dict[str, _InFlightEntry] = {}
_inflight_lock = asyncio.Lock()

# Stats
_dedup_hits: int = 0  # requests that waited for an existing in-flight
_dedup_misses: int = 0  # requests that proceeded normally


class _InFlightEntry:
    """Tracks an in-flight request."""

    __slots__ = ("error", "event", "result", "result_headers", "result_status", "started_at", "waiters")

    def __init__(self):
        self.event: asyncio.Event = asyncio.Event()
        self.result: bytes | None = None
        self.result_headers: dict = {}
        self.result_status: int = 200
        self.error: Exception | None = None
        self.started_at: float = time.monotonic()
        self.waiters: int = 0


def _make_dedup_key(request: Request, body: bytes) -> str:
    """Generate dedup key from method + path + body hash."""
    parts = [
        request.method,
        str(request.url.path),
        str(request.query_params),
    ]
    if body:
        parts.append(hashlib.sha256(body).hexdigest()[:16])
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class RequestDedupMiddleware(BaseHTTPMiddleware):
    """Middleware that coalesces identical concurrent requests.

    Only applies to POST /chat/query (the main LLM endpoint).
    Other routes pass through unchanged.
    """

    # Routes eligible for dedup
    DEDUP_PATHS = frozenset({"/chat/query"})

    # Max time to wait for an in-flight request (seconds)
    MAX_WAIT_SECONDS = 120.0

    async def dispatch(self, request: Request, call_next):
        global _dedup_hits, _dedup_misses

        # Only dedup eligible POST routes
        if request.method != "POST" or request.url.path not in self.DEDUP_PATHS:
            return await call_next(request)

        # Read body for key generation
        body = await request.body()
        key = _make_dedup_key(request, body)

        async with _inflight_lock:
            entry = _inflight.get(key)

            if entry is not None and not entry.event.is_set():
                # Existing in-flight request found — wait for it
                entry.waiters += 1
                _inflight_lock.release()
                try:
                    logger.info("[Dedup] Coalescing request key=%s (waiters=%d)", key, entry.waiters)
                    await asyncio.wait_for(entry.event.wait(), timeout=self.MAX_WAIT_SECONDS)

                    if entry.error:
                        raise entry.error

                    _dedup_hits += 1
                    logger.info("[Dedup] Cache hit for key=%s (saved 1 LLM call)", key)
                    return Response(
                        content=entry.result,
                        media_type="application/json",
                        headers={**entry.result_headers, "X-Dedup": "hit"},
                        status_code=entry.result_status,
                    )
                except TimeoutError:
                    logger.warning("[Dedup] Timeout waiting for key=%s, falling through", key)
                except Exception:
                    pass
                finally:
                    await _inflight_lock.acquire()
                    entry.waiters -= 1
                    if entry.waiters <= 0:
                        _inflight.pop(key, None)
                # Fall through to process normally
            else:
                # No in-flight — register this request as the leader
                entry = _InFlightEntry()
                _inflight[key] = entry

        # Process as the leader
        try:
            # Re-construct request with body for downstream
            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive

            response = await call_next(request)

            # Read response body for caching
            resp_body = b""
            async for chunk in response.body_iterator:
                if isinstance(chunk, str):
                    resp_body += chunk.encode()
                else:
                    resp_body += chunk

            entry.result = resp_body
            entry.result_status = response.status_code
            entry.result_headers = {
                k: v for k, v in response.headers.items() if k.lower() not in ("content-length", "transfer-encoding")
            }
            entry.event.set()

            _dedup_misses += 1

            return Response(
                content=resp_body,
                status_code=response.status_code,
                headers={**entry.result_headers, "X-Dedup": "miss"},
                media_type=response.media_type,
            )

        except Exception as e:
            entry.error = e
            entry.event.set()
            raise
        finally:
            # Cleanup if no waiters remain
            async with _inflight_lock:
                if entry.waiters <= 0:
                    _inflight.pop(key, None)


def get_dedup_stats() -> dict:
    """Return deduplication statistics (useful for /health endpoint)."""
    return {
        "inflight_requests": len(_inflight),
        "total_coalesced": _dedup_hits,
        "total_leader": _dedup_misses,
    }
