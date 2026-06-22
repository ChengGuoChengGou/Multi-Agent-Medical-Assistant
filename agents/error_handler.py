"""Application-level error classification and recovery for LLM calls.

Layer architecture:
  CircuitBreaker (service-level) → LLMFallbackChain (model-level) → this module (app-level)

Recovery strategies by error type:
  - context_length_exceeded: truncate input, retry
  - rate_limit / overload: exponential backoff, retry
  - max_tokens_exceeded: reduce max_tokens, retry
  - unknown: let CircuitBreaker / LLMFallbackChain handle
"""

from __future__ import annotations

import enum
import logging
import re
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMErrorType(enum.Enum):
    """Classified LLM error types."""

    CONTEXT_LENGTH = "context_length_exceeded"
    RATE_LIMIT = "rate_limit"
    OVERLOAD = "overload"
    MAX_TOKENS = "max_tokens_exceeded"
    AUTH = "auth_error"
    UNKNOWN = "unknown"


# ── Error classification ──────────────────────────────────────────────

_CONTEXT_PATTERNS = [
    re.compile(r"maximum context length", re.I),
    re.compile(r"context_length_exceeded", re.I),
    re.compile(r"max.*tokens.*exceeded", re.I),
    re.compile(r"too many tokens", re.I),
    re.compile(r"request too large", re.I),
]

_RATE_LIMIT_PATTERNS = [
    re.compile(r"rate.?limit", re.I),
    re.compile(r"429"),
    re.compile(r"too many requests", re.I),
]

_OVERLOAD_PATTERNS = [
    re.compile(r"overloaded", re.I),
    re.compile(r"503"),
    re.compile(r"service.?unavailable", re.I),
    re.compile(r"server.?busy", re.I),
]

_MAX_TOKENS_PATTERNS = [
    re.compile(r"max_tokens.*too large", re.I),
    re.compile(r"max_completion_tokens", re.I),
]

_AUTH_PATTERNS = [
    re.compile(r"401"),
    re.compile(r"unauthorized", re.I),
    re.compile(r"invalid.*api.?key", re.I),
    re.compile(r"authentication", re.I),
]


def classify_error(exc: Exception) -> LLMErrorType:
    """Classify an LLM/API exception into an error type."""
    msg = str(exc)
    err_type_name = type(exc).__name__

    # Check by error message content
    for pat in _CONTEXT_PATTERNS:
        if pat.search(msg):
            return LLMErrorType.CONTEXT_LENGTH
    for pat in _RATE_LIMIT_PATTERNS:
        if pat.search(msg):
            return LLMErrorType.RATE_LIMIT
    for pat in _OVERLOAD_PATTERNS:
        if pat.search(msg):
            return LLMErrorType.OVERLOAD
    for pat in _MAX_TOKENS_PATTERNS:
        if pat.search(msg):
            return LLMErrorType.MAX_TOKENS
    for pat in _AUTH_PATTERNS:
        if pat.search(msg):
            return LLMErrorType.AUTH

    # Check by exception type name
    if "RateLimitError" in err_type_name:
        return LLMErrorType.RATE_LIMIT
    if "AuthenticationError" in err_type_name:
        return LLMErrorType.AUTH
    if "BadRequestError" in err_type_name and "max_tokens" in msg.lower():
        return LLMErrorType.MAX_TOKENS

    return LLMErrorType.UNKNOWN


# ── Recovery strategies ───────────────────────────────────────────────


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, last_error: Exception, error_type: LLMErrorType, attempts: int):
        self.last_error = last_error
        self.error_type = error_type
        self.attempts = attempts
        super().__init__(f"Retry exhausted after {attempts} attempts (error_type={error_type.value}): {last_error}")


def _get_retry_delay(error_type: LLMErrorType, attempt: int) -> float:
    """Calculate backoff delay in seconds."""
    if error_type == LLMErrorType.RATE_LIMIT:
        # Rate limit: longer backoff (2s, 4s, 8s)
        return min(2 ** (attempt + 1), 30.0)
    if error_type == LLMErrorType.OVERLOAD:
        # Overload: moderate backoff (1s, 2s, 4s)
        return min(2**attempt, 15.0)
    if error_type == LLMErrorType.CONTEXT_LENGTH:
        # Context length: no delay, just retry after truncation
        return 0.0
    return 1.0


def _estimate_token_count(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars for English, ~2 for Chinese)."""
    # Simple heuristic: count CJK chars as 2 tokens, others as 0.25 tokens per char
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - cjk_count
    return cjk_count * 2 + int(other_count * 0.25)


def truncate_messages(messages: list, target_tokens: int = 8000) -> list:
    """Truncate message history to fit within token budget.

    Preserves system message (first) and last user message.
    Removes oldest non-system messages until within budget.
    """
    if not messages:
        return messages

    # Keep system message and last message
    system_msgs = []
    conversation_msgs = []

    for i, msg in enumerate(messages):
        if i == 0 and hasattr(msg, "type") and msg.type == "system":
            system_msgs.append(msg)
        else:
            conversation_msgs.append(msg)

    if len(conversation_msgs) <= 2:
        return messages  # Can't truncate further

    # Binary: keep last N messages that fit
    system_tokens = sum(_estimate_token_count(getattr(m, "content", "") or "") for m in system_msgs)
    last_msg_tokens = _estimate_token_count(getattr(conversation_msgs[-1], "content", "") or "")
    remaining_budget = target_tokens - system_tokens - last_msg_tokens

    kept = []
    current_tokens = 0
    # Keep messages from newest to oldest
    for msg in reversed(conversation_msgs[:-1]):
        msg_tokens = _estimate_token_count(getattr(msg, "content", "") or "")
        if current_tokens + msg_tokens > remaining_budget:
            break
        kept.append(msg)
        current_tokens += msg_tokens

    kept.reverse()
    result = system_msgs + kept + [conversation_msgs[-1]]

    if len(result) < len(messages):
        logger.info(
            f"[ERROR_HANDLER] Truncated messages: {len(messages)} → {len(result)} "
            f"(~{current_tokens + system_tokens + last_msg_tokens} tokens)"
        )

    return result


# ── Main retry wrapper ────────────────────────────────────────────────


def llm_call_with_recovery(
    func: Callable[..., T],
    *args,
    max_retries: int = 2,
    on_context_too_long: Callable[[], list] | None = None,
    max_tokens_override: int | None = None,
    **kwargs,
) -> T:
    """Call an LLM function with error classification and recovery.

    Args:
        func: The LLM call function (e.g., llm.invoke)
        *args: Positional args to func (first arg is usually messages)
        max_retries: Maximum retry attempts
        on_context_too_long: Callback to get truncated messages for retry.
            If None, uses default truncation on args[0].
        max_tokens_override: If set, reduce max_tokens to this value on MAX_TOKENS error.
        **kwargs: Keyword args to func

    Returns:
        Result of func call

    Raises:
        RetryExhausted: If all retries fail
        Exception: If error type is AUTH (no retry)
    """
    last_error: Exception | None = None
    messages = args[0] if args else None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            error_type = classify_error(e)
            last_error = e

            # Auth errors: don't retry
            if error_type == LLMErrorType.AUTH:
                logger.error(f"[ERROR_HANDLER] Auth error, not retrying: {e}")
                raise

            # Unknown errors: let caller handle (CircuitBreaker/FallbackChain)
            if error_type == LLMErrorType.UNKNOWN and attempt == 0:
                logger.warning(f"[ERROR_HANDLER] Unknown error on attempt {attempt + 1}: {e}")
                raise  # First unknown error → let upper layer handle

            # Known errors: apply recovery strategy
            if attempt < max_retries:
                delay = _get_retry_delay(error_type, attempt)

                if error_type == LLMErrorType.CONTEXT_LENGTH:
                    # Context too long → truncate messages
                    if on_context_too_long:
                        new_messages = on_context_too_long()
                        if args:
                            args = (new_messages,) + args[1:]
                        logger.info(
                            f"[ERROR_HANDLER] Context overflow, retruncated messages, retry {attempt + 1}/{max_retries}"
                        )
                    elif messages and isinstance(messages, list):
                        truncated = truncate_messages(messages)
                        args = (truncated,) + args[1:]
                        logger.info(
                            f"[ERROR_HANDLER] Context overflow, auto-truncated "
                            f"{len(messages)} → {len(truncated)} msgs, "
                            f"retry {attempt + 1}/{max_retries}"
                        )
                    else:
                        logger.warning("[ERROR_HANDLER] Context overflow but no truncation strategy")
                        raise

                elif error_type == LLMErrorType.MAX_TOKENS and max_tokens_override:
                    # Reduce max_tokens
                    kwargs["max_tokens"] = max_tokens_override
                    logger.info(
                        f"[ERROR_HANDLER] max_tokens reduced to {max_tokens_override}, "
                        f"retry {attempt + 1}/{max_retries}"
                    )

                elif error_type in (LLMErrorType.RATE_LIMIT, LLMErrorType.OVERLOAD):
                    logger.warning(
                        f"[ERROR_HANDLER] {error_type.value}, "
                        f"backing off {delay:.1f}s, "
                        f"retry {attempt + 1}/{max_retries}: {e}"
                    )
                    if delay > 0:
                        time.sleep(delay)

                else:
                    logger.warning(f"[ERROR_HANDLER] {error_type.value}, retry {attempt + 1}/{max_retries}: {e}")
                    if delay > 0:
                        time.sleep(delay)

            else:
                logger.error(f"[ERROR_HANDLER] All {max_retries + 1} attempts failed for {error_type.value}: {e}")

    assert last_error is not None, "last_error should be set after retry loop"
    raise RetryExhausted(last_error, error_type, max_retries + 1)


# ── StopHook: post-agent validation ──────────────────────────────────

import dataclasses


@dataclasses.dataclass
class HookResult:
    """Result of stopHook validation."""

    passed: bool
    reason: str = ""
    action: str = "pass"  # pass | retry | escalate | fallback
    confidence: float = 1.0


# Per-agent confidence thresholds (0-1).  Below threshold → escalate.
_AGENT_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "CONVERSATION_AGENT": 0.0,  # no confidence expected
    "RAG_AGENT": 0.3,  # retrieval quality gate
    "WEB_SEARCH_PROCESSOR_AGENT": 0.2,  # search results quality
    "BRAIN_TUMOR_AGENT": 0.7,  # medical image diagnosis
    "CHEST_XRAY_AGENT": 0.7,
    "SKIN_LESION_AGENT": 0.7,
}

# Dangerous medical advice patterns that MUST have a disclaimer
_MEDICAL_DANGER_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(你应该|you should|you must|你必须)\s*(停止|停用|stop|discontinue)\s*(吃|服用|taking|using)?\s*(药|medication|drug)",
        re.I,
    ),
    re.compile(r"(不需要|don'?t need|没必要)\s*(看|visit|consult|去)?\s*(医生|doctor|hospital|医院)", re.I),
    re.compile(r"(确诊|diagnosis|诊断).*?(就是|is|为)\s*(癌|cancer|恶性|malignant)", re.I),
]

# Disclaimer markers
_DISCLAIMER_PATTERNS: list[re.Pattern] = [
    re.compile(r"(建议|请|please)\s*(咨询|consult|就诊|visit)", re.I),
    re.compile(r"(仅供参考|for reference|not.*substitute|不能替代)", re.I),
    re.compile(r"(专业|professional)\s*(医疗|medical)\s*(建议|advice)", re.I),
]


class StopHookValidator:
    """Post-agent execution validator.

    Checks:
      1. Output completeness (empty / too short)
      2. Medical safety (dangerous advice without disclaimer)
      3. Confidence threshold (per agent)
    """

    MIN_OUTPUT_LENGTH = 20  # characters
    MIN_CONFIDENCE_DEFAULT = 0.3  # fallback threshold

    def validate(
        self,
        output_text: str,
        agent_name: str,
        confidence: float | None = None,
    ) -> HookResult:
        """Run all validation checks and return a HookResult."""
        # ── 1. Completeness ───────────────────────────────────────
        if not output_text or len(output_text.strip()) < self.MIN_OUTPUT_LENGTH:
            return HookResult(
                passed=False,
                reason=f"Output too short ({len(output_text or '')} chars)",
                action="fallback",
                confidence=0.0,
            )

        # ── 2. Medical safety ─────────────────────────────────────
        for pattern in _MEDICAL_DANGER_PATTERNS:
            if pattern.search(output_text):
                has_disclaimer = any(d.search(output_text) for d in _DISCLAIMER_PATTERNS)
                if not has_disclaimer:
                    return HookResult(
                        passed=False,
                        reason="Potentially dangerous medical advice without disclaimer",
                        action="escalate",
                        confidence=0.0,
                    )

        # ── 3. Confidence threshold ───────────────────────────────
        if confidence is not None:
            threshold = _AGENT_CONFIDENCE_THRESHOLDS.get(agent_name, self.MIN_CONFIDENCE_DEFAULT)
            if confidence < threshold:
                return HookResult(
                    passed=False,
                    reason=f"Confidence {confidence:.2f} < threshold {threshold:.2f} for {agent_name}",
                    action="escalate" if confidence > threshold * 0.5 else "fallback",
                    confidence=confidence,
                )

        # ── All checks passed ─────────────────────────────────────
        return HookResult(passed=True, confidence=confidence or 1.0)
