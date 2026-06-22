"""Circuit Breaker pattern for external service calls (Phase 47).

Prevents cascade failures when external services (LLM, MCP, web search) are down.
States: CLOSED (normal) → OPEN (failing, fast-fail) → HALF_OPEN (testing recovery).

Usage:
    breaker = CircuitBreaker("llm_service", failure_threshold=5, recovery_timeout=30)
    result = breaker.call(lambda: llm.invoke(query))
"""

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import TypeVar

logger = logging.getLogger("medical_chatbot.circuit_breaker")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are fast-failed
    HALF_OPEN = "half_open"  # Testing recovery, one request allowed through


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is in OPEN state."""

    def __init__(self, service_name: str, remaining_seconds: float):
        self.service_name = service_name
        self.remaining_seconds = remaining_seconds
        super().__init__(f"Circuit breaker '{service_name}' is OPEN. Retry in {remaining_seconds:.1f}s.")


class CircuitBreaker:
    """Thread-safe circuit breaker for protecting external service calls.

    Args:
        name: Service identifier (for logging/metrics)
        failure_threshold: Number of consecutive failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exceptions: Exception types that count as failures
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: tuple = (Exception,),
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._success_count = 0
        self._total_calls = 0
        self._total_failures = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state, auto-transitioning OPEN→HALF_OPEN if timeout elapsed."""
        with self._lock:
            if self._state == CircuitState.OPEN and time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(f"[CB:{self.name}] OPEN → HALF_OPEN (timeout elapsed)")
            return self._state

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute func through the circuit breaker.

        Raises CircuitBreakerOpenError if the circuit is OPEN.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            remaining = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
            raise CircuitBreakerOpenError(self.name, max(0, remaining))

        self._total_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions:
            self._on_failure()
            raise

    async def call_async(self, func, *args, **kwargs):
        """Execute async func through the circuit breaker."""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            remaining = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
            raise CircuitBreakerOpenError(self.name, max(0, remaining))

        self._total_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                # Single success in HALF_OPEN → close the circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(f"[CB:{self.name}] HALF_OPEN → CLOSED (recovered)")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset on success

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Failure in HALF_OPEN → back to OPEN
                self._state = CircuitState.OPEN
                logger.warning(f"[CB:{self.name}] HALF_OPEN → OPEN (recovery failed)")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"[CB:{self.name}] CLOSED → OPEN (failures={self._failure_count}/{self.failure_threshold})"
                )

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "failure_rate": (self._total_failures / self._total_calls if self._total_calls > 0 else 0.0),
        }

    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info(f"[CB:{self.name}] Manually reset to CLOSED")


# ─── Global circuit breaker instances for key services ──────────────

# LLM calls (most critical, highest threshold)
llm_breaker = CircuitBreaker(
    name="llm_service",
    failure_threshold=5,
    recovery_timeout=30.0,
)

# MCP tool calls (external protocol)
mcp_breaker = CircuitBreaker(
    name="mcp_service",
    failure_threshold=3,
    recovery_timeout=60.0,
)

# Web search (external, unreliable)
web_search_breaker = CircuitBreaker(
    name="web_search",
    failure_threshold=3,
    recovery_timeout=45.0,
)

# Image analysis (ML model)
image_breaker = CircuitBreaker(
    name="image_analysis",
    failure_threshold=3,
    recovery_timeout=30.0,
)


def get_all_breaker_stats() -> dict:
    """Get stats for all circuit breakers (for /health endpoint)."""
    return {
        "breakers": {
            name: breaker.get_stats()
            for name, breaker in [
                ("llm", llm_breaker),
                ("mcp", mcp_breaker),
                ("web_search", web_search_breaker),
                ("image", image_breaker),
            ]
        }
    }
