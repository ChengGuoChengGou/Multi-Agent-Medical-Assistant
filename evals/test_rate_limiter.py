from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import ModelGateway, RateLimiterRegistry  # noqa: E402
from core.retry_policy import RetryPolicy  # noqa: E402


class SlowModel:
    def __init__(self, active_counter):
        self.active_counter = active_counter

    def invoke(self, prompt):
        with self.active_counter["lock"]:
            self.active_counter["active"] += 1
            self.active_counter["max_active"] = max(
                self.active_counter["max_active"],
                self.active_counter["active"],
            )
        time.sleep(0.05)
        with self.active_counter["lock"]:
            self.active_counter["active"] -= 1
        return f"done: {prompt}"


def main() -> int:
    active_counter = {
        "active": 0,
        "max_active": 0,
        "lock": threading.Lock(),
    }
    gateway = ModelGateway(
        retry_policy=RetryPolicy(max_attempts=1),
        rate_limiters=RateLimiterRegistry(max_concurrent=1, queue_timeout_seconds=2.0),
    )
    results = []

    def call(index):
        result = gateway.invoke(
            model=SlowModel(active_counter),
            prompt=f"request-{index}",
            model_id="limited-model",
        )
        results.append(result)

    threads = [threading.Thread(target=call, args=(idx,)) for idx in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert active_counter["max_active"] == 1
    assert all("queue_wait_ms" in result.metadata for result in results)

    print("Rate limiter checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
