# Model Reliability Layer

This refactor adds a lightweight model gateway inspired by Ragent's model routing and failover layer.

## Implemented Modules

| Module | Responsibility |
| --- | --- |
| `core/circuit_breaker.py` | Three-state circuit breaker: `CLOSED`, `OPEN`, `HALF_OPEN`. |
| `core/retry_policy.py` | Retry wrapper with simple linear backoff. |
| `core/token_tracker.py` | Heuristic prompt/completion token estimation. |
| `core/model_gateway.py` | Unified wrapper around LangChain-style `.invoke()` calls. |

## Current Integration Points

The gateway is currently wired into:

- `agents/rag_agent/query_expander.py`
- `agents/rag_agent/response_generator.py`

This gives the RAG path retry, circuit-breaker protection, latency tracking, attempt count, and token usage estimates without changing external behavior.

Circuit breakers use a process-level shared registry, so failures for the same `model_id` are visible across gateway instances in the current application process.

## Circuit Breaker State Transitions

```text
CLOSED
  | consecutive failures >= threshold
  v
OPEN
  | cooldown elapsed
  v
HALF_OPEN
  | probe success       | probe failure
  v                     v
CLOSED               OPEN
```

Rules:

- `CLOSED`: calls are allowed normally.
- `OPEN`: calls are blocked until `open_duration_seconds` elapses.
- `HALF_OPEN`: only one probe call is allowed.
- Success in half-open closes the circuit.
- Failure in half-open reopens the circuit.

## Retry Behavior

`RetryPolicy` retries model calls before marking the circuit as failed. This handles transient network errors, 429s, and provider-side 5xx errors once exception classification is added.

Current default:

```text
max_attempts = 2
base_delay_seconds = 0.25
```

## Streaming First-Packet Probe

`await_first_packet()` documents the future SSE integration point:

1. Start the streaming producer.
2. Capture the first valid token/chunk.
3. If no chunk arrives before timeout, treat the call as failed.
4. Trigger retry or model fallback.

This answers the streaming success question: the model call is not considered healthy when the HTTP connection opens; it is considered healthy when the first valid content packet arrives.

## Token and Cost Tracking

The gateway estimates token usage when provider metadata is not available:

```text
prompt_tokens ~= len(prompt_text) / 4
completion_tokens ~= len(response_text) / 4
```

`ResponseGenerator` now includes:

```json
{
  "model_gateway": {
    "model_id": "rag-response-generator",
    "latency_ms": 1234,
    "attempts": 1,
    "usage": {
      "prompt_tokens": 2000,
      "completion_tokens": 300,
      "total_tokens": 2300
    }
  }
}
```

## Next Extensions

1. Add multiple provider candidates for chat, embedding, and rerank.
2. Classify retryable errors instead of retrying all exceptions.
3. Add queue/semaphore based concurrency limiting.
4. Add first-packet probing to FastAPI SSE streaming once streaming output is introduced.
5. Persist model-call metrics into trace storage.
