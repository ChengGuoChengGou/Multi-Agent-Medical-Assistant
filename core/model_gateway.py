import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .circuit_breaker import CircuitBreakerRegistry
from .rate_limiter import RateLimiterRegistry
from .retry_policy import RetryPolicy
from .token_tracker import HeuristicTokenCounter, TokenUsageEstimate


DEFAULT_CIRCUIT_BREAKERS = CircuitBreakerRegistry()
DEFAULT_RATE_LIMITERS = RateLimiterRegistry()


@dataclass
class ModelCallResult:
    output: Any
    model_id: str
    latency_ms: int
    usage: TokenUsageEstimate
    attempts: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelGateway:
    """Reliability wrapper around LangChain-style model calls.

    This is intentionally framework-light: any object exposing `.invoke(input)`
    can be routed through it. Later, multiple providers can be added by passing
    several candidates and selecting by priority.
    """

    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breakers: Optional[CircuitBreakerRegistry] = None,
        rate_limiters: Optional[RateLimiterRegistry] = None,
        token_counter: Optional[HeuristicTokenCounter] = None,
    ):
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breakers = circuit_breakers or DEFAULT_CIRCUIT_BREAKERS
        self.rate_limiters = rate_limiters or DEFAULT_RATE_LIMITERS
        self.token_counter = token_counter or HeuristicTokenCounter()
        self.logger = logging.getLogger(__name__)

    def invoke(
        self,
        *,
        model: Any,
        prompt: Any,
        model_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelCallResult:
        breaker = self.circuit_breakers.get(model_id)
        if not breaker.allow_call():
            snapshot = breaker.snapshot()
            raise RuntimeError(
                f"Model circuit is open for {model_id}; state={snapshot.state.value}"
            )

        attempts = 0
        start = time.time()
        limiter = self.rate_limiters.get(model_id)

        def call_model() -> Any:
            nonlocal attempts
            attempts += 1
            return model.invoke(prompt)

        try:
            with limiter.acquire() as permit:
                output = self.retry_policy.run(call_model)
            breaker.mark_success()
            latency_ms = int((time.time() - start) * 1000)
            usage = self.token_counter.estimate(prompt, output)
            call_metadata = {
                **(metadata or {}),
                "rate_limit_request_id": permit.request_id,
                "queue_wait_ms": permit.queue_wait_ms,
            }
            return ModelCallResult(
                output=output,
                model_id=model_id,
                latency_ms=latency_ms,
                usage=usage,
                attempts=attempts,
                metadata=call_metadata,
            )
        except Exception:
            breaker.mark_failure()
            self.logger.warning("Model call failed for %s", model_id)
            raise

    def invoke_content(
        self,
        *,
        model: Any,
        prompt: Any,
        model_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        result = self.invoke(
            model=model,
            prompt=prompt,
            model_id=model_id,
            metadata=metadata,
        )
        output = result.output
        return output.content if hasattr(output, "content") else output


def await_first_packet(
    producer: Callable[[Callable[[Any], None]], Any],
    timeout_seconds: float,
) -> Any:
    """Probe helper for future streaming integration.

    `producer` receives an `on_packet` callback. The first packet returned
    before timeout marks the stream as healthy. This helper is not wired into
    SSE yet, but documents the exact seam for first-token probing.
    """
    import queue

    packets: "queue.Queue[Any]" = queue.Queue(maxsize=1)

    def on_packet(packet: Any) -> None:
        if packets.empty():
            packets.put(packet)

    producer(on_packet)
    return packets.get(timeout=timeout_seconds)
