from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.circuit_breaker import CircuitBreakerRegistry  # noqa: E402
from core.model_gateway import ModelGateway  # noqa: E402
from core.retry_policy import RetryPolicy  # noqa: E402


class EchoModel:
    def invoke(self, prompt):
        return f"echo: {prompt}"


class FailingModel:
    def invoke(self, prompt):
        raise RuntimeError("provider down")


def main() -> int:
    registry = CircuitBreakerRegistry(failure_threshold=1, open_duration_seconds=60)
    gateway = ModelGateway(
        retry_policy=RetryPolicy(max_attempts=1),
        circuit_breakers=registry,
    )

    success = gateway.invoke(
        model=EchoModel(),
        prompt="hello",
        model_id="test-echo",
    )
    assert success.output == "echo: hello"
    assert success.attempts == 1
    assert success.usage.total_tokens > 0

    try:
        gateway.invoke(
            model=FailingModel(),
            prompt="hello",
            model_id="test-failing",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Failing model should raise")

    try:
        gateway.invoke(
            model=EchoModel(),
            prompt="hello",
            model_id="test-failing",
        )
    except RuntimeError as exc:
        assert "circuit is open" in str(exc)
    else:
        raise AssertionError("Open circuit should block calls")

    print("Model gateway checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
