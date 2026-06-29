import time
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Type, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    base_delay_seconds: float = 0.25
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,)

    def run(self, func: Callable[[], T]) -> T:
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func()
            except self.retryable_exceptions as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.base_delay_seconds * attempt)
        raise last_error  # type: ignore[misc]
