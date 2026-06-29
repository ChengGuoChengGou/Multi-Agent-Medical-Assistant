import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Deque, Dict, Iterator, Optional
from uuid import uuid4


class RateLimitTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class RateLimitPermit:
    request_id: str
    model_id: str
    queued_at: float
    acquired_at: float
    queue_wait_ms: int


@dataclass(frozen=True)
class RateLimiterSnapshot:
    model_id: str
    max_concurrent: int
    active: int
    queued: int


class FairRateLimiter:
    """Process-local FIFO limiter for model calls.

    The critical section is intentionally small: queue membership, head-of-line
    checks, active permit count, and dequeue happen under one condition lock.
    """

    def __init__(
        self,
        model_id: str,
        max_concurrent: int = 4,
        queue_timeout_seconds: float = 30.0,
    ):
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        self.model_id = model_id
        self.max_concurrent = max_concurrent
        self.queue_timeout_seconds = queue_timeout_seconds
        self._active = 0
        self._queue: Deque[str] = deque()
        self._condition = Condition(Lock())

    @contextmanager
    def acquire(
        self,
        request_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Iterator[RateLimitPermit]:
        permit = self._acquire(
            request_id=request_id or str(uuid4()),
            timeout_seconds=timeout_seconds,
        )
        try:
            yield permit
        finally:
            self.release()

    def _acquire(
        self,
        request_id: str,
        timeout_seconds: Optional[float] = None,
    ) -> RateLimitPermit:
        timeout = self.queue_timeout_seconds if timeout_seconds is None else timeout_seconds
        queued_at = time.time()
        deadline = queued_at + timeout

        with self._condition:
            self._queue.append(request_id)
            while True:
                is_head = self._queue and self._queue[0] == request_id
                has_capacity = self._active < self.max_concurrent

                if is_head and has_capacity:
                    self._queue.popleft()
                    self._active += 1
                    acquired_at = time.time()
                    return RateLimitPermit(
                        request_id=request_id,
                        model_id=self.model_id,
                        queued_at=queued_at,
                        acquired_at=acquired_at,
                        queue_wait_ms=int((acquired_at - queued_at) * 1000),
                    )

                remaining = deadline - time.time()
                if remaining <= 0:
                    try:
                        self._queue.remove(request_id)
                    except ValueError:
                        pass
                    self._condition.notify_all()
                    raise RateLimitTimeout(
                        f"Timed out waiting for model permit: model_id={self.model_id}"
                    )

                self._condition.wait(timeout=remaining)

    def release(self) -> None:
        with self._condition:
            if self._active > 0:
                self._active -= 1
            self._condition.notify_all()

    def snapshot(self) -> RateLimiterSnapshot:
        with self._condition:
            return RateLimiterSnapshot(
                model_id=self.model_id,
                max_concurrent=self.max_concurrent,
                active=self._active,
                queued=len(self._queue),
            )


class RateLimiterRegistry:
    def __init__(
        self,
        max_concurrent: int = 4,
        queue_timeout_seconds: float = 30.0,
    ):
        self.max_concurrent = max_concurrent
        self.queue_timeout_seconds = queue_timeout_seconds
        self._limiters: Dict[str, FairRateLimiter] = {}
        self._lock = Lock()

    def get(self, model_id: str) -> FairRateLimiter:
        with self._lock:
            if model_id not in self._limiters:
                self._limiters[model_id] = FairRateLimiter(
                    model_id=model_id,
                    max_concurrent=self.max_concurrent,
                    queue_timeout_seconds=self.queue_timeout_seconds,
                )
            return self._limiters[model_id]
