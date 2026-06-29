import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Dict


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitSnapshot:
    state: CircuitState
    consecutive_failures: int
    open_until: float
    half_open_in_flight: bool


class CircuitBreaker:
    """Three-state circuit breaker for model calls."""

    def __init__(
        self,
        failure_threshold: int = 3,
        open_duration_seconds: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.open_duration_seconds = open_duration_seconds
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._half_open_in_flight = False
        self._lock = Lock()

    def allow_call(self) -> bool:
        now = time.time()
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._open_until > now:
                    return False
                self._state = CircuitState.HALF_OPEN
                self._half_open_in_flight = True
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_in_flight:
                    return False
                self._half_open_in_flight = True
                return True

            return True

    def mark_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._open_until = 0.0
            self._half_open_in_flight = False

    def mark_failure(self) -> None:
        now = time.time()
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._open_until = now + self.open_duration_seconds
                self._consecutive_failures = 0
                self._half_open_in_flight = False
                return

            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._open_until = now + self.open_duration_seconds
                self._consecutive_failures = 0

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            return CircuitSnapshot(
                state=self._state,
                consecutive_failures=self._consecutive_failures,
                open_until=self._open_until,
                half_open_in_flight=self._half_open_in_flight,
            )


class CircuitBreakerRegistry:
    def __init__(
        self,
        failure_threshold: int = 3,
        open_duration_seconds: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.open_duration_seconds = open_duration_seconds
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()

    def get(self, model_id: str) -> CircuitBreaker:
        with self._lock:
            if model_id not in self._breakers:
                self._breakers[model_id] = CircuitBreaker(
                    failure_threshold=self.failure_threshold,
                    open_duration_seconds=self.open_duration_seconds,
                )
            return self._breakers[model_id]
