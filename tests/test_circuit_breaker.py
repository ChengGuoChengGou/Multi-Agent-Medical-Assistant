"""
Unit tests for circuit_breaker.py — CircuitBreaker state machine & protection logic.
Run: python -m pytest tests/test_circuit_breaker.py -v
"""
import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_breaker(threshold=3, recovery=1.0):
    return CircuitBreaker(
        name="test_service",
        failure_threshold=threshold,
        recovery_timeout=recovery,
    )


# ── CircuitState Enum ───────────────────────────────────────────────────────

class TestCircuitState:
    """Test CircuitState enum values."""

    def test_enum_values(self):
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"

    def test_enum_members(self):
        assert set(CircuitState) == {CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN}


# ── CircuitBreakerOpenError ─────────────────────────────────────────────────

class TestCircuitBreakerOpenError:
    """Test the exception raised when circuit is OPEN."""

    def test_attributes(self):
        err = CircuitBreakerOpenError("llm_service", 15.3)
        assert err.service_name == "llm_service"
        assert err.remaining_seconds == 15.3
        assert "llm_service" in str(err)
        assert "15.3s" in str(err)

    def test_zero_remaining(self):
        err = CircuitBreakerOpenError("svc", 0.0)
        assert err.remaining_seconds == 0.0


# ── CircuitBreaker: Initial State ───────────────────────────────────────────

class TestCircuitBreakerInit:
    """Test initial circuit breaker state."""

    def test_initial_state_closed(self):
        cb = _make_breaker()
        assert cb.state == CircuitState.CLOSED

    def test_defaults(self):
        cb = CircuitBreaker("svc")
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30.0
        assert cb.expected_exceptions == (Exception,)

    def test_custom_params(self):
        cb = CircuitBreaker("svc", failure_threshold=10, recovery_timeout=60.0)
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 60.0


# ── CircuitBreaker: CLOSED State ────────────────────────────────────────────

class TestCircuitBreakerClosed:
    """Test behavior in CLOSED (normal) state."""

    def test_success_passthrough(self):
        cb = _make_breaker()
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_failure_accumulates(self):
        cb = _make_breaker(threshold=3)
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(self._fail)
        # Still closed after 2 failures (threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_failure_opens_at_threshold(self):
        cb = _make_breaker(threshold=3)
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(self._fail)
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = _make_breaker(threshold=3)
        # 2 failures
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(self._fail)
        # 1 success resets counter
        cb.call(lambda: "ok")
        # 2 more failures should NOT open (counter was reset)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(self._fail)
        assert cb.state == CircuitState.CLOSED

    def test_unexpected_exception_not_counted(self):
        """Exceptions not in expected_exceptions should NOT count as failures."""
        cb = CircuitBreaker(
            "svc",
            failure_threshold=2,
            expected_exceptions=(ValueError,),
        )
        # TypeError is NOT in expected_exceptions → doesn't count
        with pytest.raises(TypeError):
            cb.call(lambda: (_ for _ in ()).throw(TypeError("wrong type")))
        assert cb.state == CircuitState.CLOSED

    def test_func_with_args(self):
        cb = _make_breaker()
        result = cb.call(lambda x, y: x + y, 10, 20)
        assert result == 30

    def test_func_with_kwargs(self):
        cb = _make_breaker()
        result = cb.call(lambda x=0, y=0: x * y, x=3, y=7)
        assert result == 21

    @staticmethod
    def _fail():
        raise ValueError("simulated failure")


# ── CircuitBreaker: OPEN State ──────────────────────────────────────────────

class TestCircuitBreakerOpen:
    """Test behavior in OPEN (fast-fail) state."""

    def test_open_raises_immediately(self):
        cb = _make_breaker(threshold=1, recovery=10.0)
        # Trigger open
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN
        # Next call should fast-fail
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            cb.call(lambda: 42)
        assert exc_info.value.service_name == "test_service"
        assert exc_info.value.remaining_seconds > 0

    def test_open_after_threshold(self):
        cb = _make_breaker(threshold=2, recovery=5.0)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        # State should be OPEN
        assert cb.state == CircuitState.OPEN
        # Next call fast-fails
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: 99)


# ── CircuitBreaker: HALF_OPEN → Recovery ────────────────────────────────────

class TestCircuitBreakerRecovery:
    """Test OPEN → HALF_OPEN → CLOSED transition."""

    def test_timeout_transitions_to_half_open(self):
        cb = _make_breaker(threshold=1, recovery=0.1)
        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        assert cb.state == CircuitState.OPEN
        # Wait for recovery timeout
        time.sleep(0.15)
        # State should auto-transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes(self):
        cb = _make_breaker(threshold=1, recovery=0.1)
        # Open
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        # Wait + trigger HALF_OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        # Success → CLOSED
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = _make_breaker(threshold=1, recovery=0.1)
        # Open
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        # Wait + trigger HALF_OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        # Failure → back to OPEN
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        assert cb.state == CircuitState.OPEN


# ── CircuitBreaker: Async ───────────────────────────────────────────────────

class TestCircuitBreakerAsync:
    """Test call_async method."""

    @staticmethod
    def _run(coro):
        """Run async coroutine on a new event loop (Python 3.12 compatible)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_async_success(self):
        cb = _make_breaker()

        async def async_func():
            return 42

        result = self._run(cb.call_async(async_func))
        assert result == 42

    def test_async_failure_opens(self):
        cb = _make_breaker(threshold=1, recovery=10.0)

        async def fail():
            raise ValueError("async fail")

        with pytest.raises(ValueError):
            self._run(cb.call_async(fail))
        assert cb.state == CircuitState.OPEN

    def test_async_open_fast_fail(self):
        cb = _make_breaker(threshold=1, recovery=10.0)

        async def fail():
            raise ValueError("open it")

        # Open the circuit
        with pytest.raises(ValueError):
            self._run(cb.call_async(fail))

        # Fast fail
        async def should_not_run():
            return 99

        with pytest.raises(CircuitBreakerOpenError):
            self._run(cb.call_async(should_not_run))


# ── CircuitBreaker: Stats & Reset ───────────────────────────────────────────

class TestCircuitBreakerStats:
    """Test get_stats() and reset() methods."""

    def test_initial_stats(self):
        cb = _make_breaker()
        stats = cb.get_stats()
        assert stats["name"] == "test_service"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["total_calls"] == 0
        assert stats["total_failures"] == 0
        assert stats["failure_rate"] == 0.0

    def test_stats_after_calls(self):
        cb = _make_breaker(threshold=5)
        # 3 successes + 2 failures = 5 total
        cb.call(lambda: 1)
        cb.call(lambda: 2)
        cb.call(lambda: 3)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        stats = cb.get_stats()
        assert stats["total_calls"] == 5
        assert stats["total_failures"] == 2
        assert stats["failure_rate"] == pytest.approx(0.4)
        assert stats["failure_count"] == 2

    def test_reset_closes_circuit(self):
        cb = _make_breaker(threshold=1, recovery=10.0)
        # Open
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        assert cb.state == CircuitState.OPEN
        # Reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
        # Can call again
        assert cb.call(lambda: 42) == 42


# ── CircuitBreaker: Global Instances ────────────────────────────────────────

class TestGlobalBreakers:
    """Test that global breaker instances exist and are configured."""

    def test_llm_breaker_exists(self):
        from circuit_breaker import llm_breaker
        assert llm_breaker.name == "llm_service"
        assert llm_breaker.failure_threshold == 5
        assert llm_breaker.recovery_timeout == 30.0

    def test_mcp_breaker_exists(self):
        from circuit_breaker import mcp_breaker
        assert mcp_breaker.name == "mcp_service"
        assert mcp_breaker.failure_threshold == 3
        assert mcp_breaker.recovery_timeout == 60.0

    def test_web_search_breaker_exists(self):
        from circuit_breaker import web_search_breaker
        assert web_search_breaker.name == "web_search"
        assert web_search_breaker.failure_threshold == 3

    def test_image_breaker_exists(self):
        from circuit_breaker import image_breaker
        assert image_breaker.name == "image_analysis"
