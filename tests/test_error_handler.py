"""Tests for agents/error_handler.py – error classification, retry, truncation, and StopHook."""

import pytest
from unittest.mock import MagicMock, patch, call

import agents.error_handler as eh
from agents.error_handler import (
    LLMErrorType,
    classify_error,
    RetryExhausted,
    _get_retry_delay,
    _estimate_token_count,
    truncate_messages,
    llm_call_with_recovery,
    HookResult,
    StopHookValidator,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _make_error(msg: str, cls_name: str = "Exception") -> Exception:
    """Create an exception with a specific class name using proper subclassing."""
    cls = type(cls_name, (Exception,), {"__module__": __name__})
    return cls(msg)


def _make_msg(role: str, content: str):
    msg = MagicMock()
    msg.type = role
    msg.content = content
    return msg


# ── LLMErrorType enum ───────────────────────────────────────────────

class TestLLMErrorType:
    def test_enum_values(self):
        assert LLMErrorType.CONTEXT_LENGTH.value == "context_length_exceeded"
        assert LLMErrorType.RATE_LIMIT.value == "rate_limit"
        assert LLMErrorType.OVERLOAD.value == "overload"
        assert LLMErrorType.MAX_TOKENS.value == "max_tokens_exceeded"
        assert LLMErrorType.AUTH.value == "auth_error"
        assert LLMErrorType.UNKNOWN.value == "unknown"

    def test_enum_membership(self):
        assert len(LLMErrorType) == 6
        assert LLMErrorType.CONTEXT_LENGTH in LLMErrorType


# ── classify_error ──────────────────────────────────────────────────

class TestClassifyError:
    # --- Context length ---
    def test_context_length_exceeded(self):
        err = Exception("context_length_exceeded")
        assert classify_error(err) == LLMErrorType.CONTEXT_LENGTH

    def test_maximum_context_length(self):
        err = Exception("maximum context length exceeded in request")
        assert classify_error(err) == LLMErrorType.CONTEXT_LENGTH

    def test_too_many_tokens(self):
        err = Exception("too many tokens for this model")
        assert classify_error(err) == LLMErrorType.CONTEXT_LENGTH

    def test_request_too_large(self):
        err = Exception("request too large for model context")
        assert classify_error(err) == LLMErrorType.CONTEXT_LENGTH

    # --- Rate limit ---
    def test_rate_limit(self):
        err = Exception("rate limit exceeded")
        assert classify_error(err) == LLMErrorType.RATE_LIMIT

    def test_429_status(self):
        err = Exception("HTTP 429 Too Many Requests")
        assert classify_error(err) == LLMErrorType.RATE_LIMIT

    def test_too_many_requests(self):
        err = Exception("too many requests, please retry")
        assert classify_error(err) == LLMErrorType.RATE_LIMIT

    def test_rate_limit_by_class_name(self):
        err = _make_error("please wait", "RateLimitError")
        assert classify_error(err) == LLMErrorType.RATE_LIMIT

    # --- Overload ---
    def test_overloaded(self):
        err = Exception("server overloaded")
        assert classify_error(err) == LLMErrorType.OVERLOAD

    def test_503_status(self):
        err = Exception("503 service unavailable")
        assert classify_error(err) == LLMErrorType.OVERLOAD

    def test_server_busy(self):
        err = Exception("server busy, try again later")
        assert classify_error(err) == LLMErrorType.OVERLOAD

    # --- Max tokens ---
    def test_max_tokens_too_large(self):
        err = Exception("max_tokens is too large for this model")
        assert classify_error(err) == LLMErrorType.MAX_TOKENS

    def test_max_completion_tokens(self):
        err = Exception("max_completion_tokens exceeds limit")
        assert classify_error(err) == LLMErrorType.MAX_TOKENS

    # --- Auth ---
    def test_401_status(self):
        err = Exception("HTTP 401 Unauthorized")
        assert classify_error(err) == LLMErrorType.AUTH

    def test_unauthorized(self):
        err = Exception("unauthorized access to API")
        assert classify_error(err) == LLMErrorType.AUTH

    def test_invalid_api_key(self):
        err = Exception("invalid api key provided")
        assert classify_error(err) == LLMErrorType.AUTH

    def test_auth_by_class_name(self):
        err = _make_error("bad credentials", "AuthenticationError")
        assert classify_error(err) == LLMErrorType.AUTH

    # --- Unknown ---
    def test_unknown_error(self):
        err = Exception("something completely different")
        assert classify_error(err) == LLMErrorType.UNKNOWN

    def test_generic_exception(self):
        err = RuntimeError("unexpected")
        assert classify_error(err) == LLMErrorType.UNKNOWN

    # --- Priority: message patterns before class name ---
    def test_message_pattern_takes_priority_over_class(self):
        """If message matches a pattern, class name check is skipped."""
        err = _make_error("429 rate limit hit", "GenericHTTPError")
        assert classify_error(err) == LLMErrorType.RATE_LIMIT


# ── _get_retry_delay ────────────────────────────────────────────────

class TestGetRetryDelay:
    def test_rate_limit_exponential_backoff(self):
        d0 = _get_retry_delay(LLMErrorType.RATE_LIMIT, 0)
        d1 = _get_retry_delay(LLMErrorType.RATE_LIMIT, 1)
        d2 = _get_retry_delay(LLMErrorType.RATE_LIMIT, 2)
        assert d0 > 0
        assert d1 > d0
        assert d2 > d1

    def test_overload_delay(self):
        d = _get_retry_delay(LLMErrorType.OVERLOAD, 0)
        assert d > 0

    def test_context_length_no_delay(self):
        d = _get_retry_delay(LLMErrorType.CONTEXT_LENGTH, 0)
        assert d == 0

    def test_max_tokens_no_delay(self):
        d = _get_retry_delay(LLMErrorType.MAX_TOKENS, 0)
        # MAX_TOKENS hits the else branch → 1.0
        assert d == 1.0

    def test_auth_no_delay(self):
        d = _get_retry_delay(LLMErrorType.AUTH, 0)
        # AUTH hits the else branch → 1.0
        assert d == 1.0

    def test_unknown_no_delay(self):
        d = _get_retry_delay(LLMErrorType.UNKNOWN, 0)
        # UNKNOWN hits the else branch → 1.0
        assert d == 1.0

    def test_rate_limit_exponential_backoff(self):
        """RATE_LIMIT uses min(2^(attempt+1), 30.0) — no jitter."""
        d0 = _get_retry_delay(LLMErrorType.RATE_LIMIT, 0)
        d1 = _get_retry_delay(LLMErrorType.RATE_LIMIT, 1)
        d2 = _get_retry_delay(LLMErrorType.RATE_LIMIT, 2)
        assert d0 == 2.0   # 2^1
        assert d1 == 4.0   # 2^2
        assert d2 == 8.0   # 2^3

    def test_rate_limit_capped_at_30(self):
        """RATE_LIMIT delay caps at 30.0."""
        d = _get_retry_delay(LLMErrorType.RATE_LIMIT, 10)
        assert d == 30.0


# ── _estimate_token_count ───────────────────────────────────────────

class TestEstimateTokenCount:
    def test_empty(self):
        assert _estimate_token_count("") == 0

    def test_pure_english(self):
        count = _estimate_token_count("hello world test")
        assert count > 0
        assert count == 4  # 15 chars / 4 ≈ 3.75 → ceil = 4

    def test_pure_chinese(self):
        count = _estimate_token_count("你好世界")
        # 4 CJK chars * 2 tokens each + int(0 * 0.25) = 8
        assert count == 8

    def test_mixed(self):
        text = "Hello你好World世界"
        count = _estimate_token_count(text)
        assert count > 0

    def test_long_english_text(self):
        text = "word " * 100  # 500 chars
        count = _estimate_token_count(text)
        assert count == 125  # 500 / 4 = 125


# ── truncate_messages ──────────────────────────────────────────────

class TestTruncateMessages:
    def test_empty_list(self):
        assert truncate_messages([]) == []

    def test_single_message(self):
        msgs = [_make_msg("user", "hello")]
        result = truncate_messages(msgs)
        assert len(result) == 1

    def test_two_messages_no_truncation(self):
        msgs = [_make_msg("system", "you are helpful"), _make_msg("user", "hi")]
        result = truncate_messages(msgs)
        assert len(result) == 2

    def test_three_messages_no_truncation_needed(self):
        msgs = [
            _make_msg("system", "system prompt"),
            _make_msg("user", "hi"),
            _make_msg("assistant", "hello"),
        ]
        result = truncate_messages(msgs)
        assert len(result) == 3

    def test_preserves_system_message(self):
        msgs = [
            _make_msg("system", "important system prompt"),
            _make_msg("user", "msg1"),
            _make_msg("assistant", "reply1"),
            _make_msg("user", "msg2"),
            _make_msg("assistant", "reply2"),
            _make_msg("user", "msg3"),
        ]
        result = truncate_messages(msgs, target_tokens=10)
        # System msg preserved
        assert result[0].content == "important system prompt"
        # Last msg preserved
        assert result[-1].content == "msg3"

    def test_truncation_drops_oldest(self):
        """With very small token budget, oldest non-system messages are dropped."""
        msgs = [
            _make_msg("system", "sys"),
            _make_msg("user", "a" * 200),  # old, ~50 tokens
            _make_msg("assistant", "b" * 200),  # old, ~50 tokens
            _make_msg("user", "c" * 200),  # recent
            _make_msg("assistant", "d" * 200),  # recent
            _make_msg("user", "last"),  # always kept
        ]
        result = truncate_messages(msgs, target_tokens=20)
        # Should have system + last user, with some middle dropped
        assert len(result) < len(msgs)
        assert result[0].content == "sys"
        assert result[-1].content == "last"

    def test_no_type_attribute(self):
        """Messages without .type are treated as non-system."""
        msg = MagicMock()
        msg.content = "hello"
        msg.type = None
        result = truncate_messages([msg, msg, msg])
        assert len(result) == 3


# ── RetryExhausted ─────────────────────────────────────────────────

class TestRetryExhausted:
    def test_attributes(self):
        original = Exception("fail")
        exc = RetryExhausted(original, LLMErrorType.RATE_LIMIT, 3)
        assert exc.last_error is original
        assert exc.error_type == LLMErrorType.RATE_LIMIT
        assert exc.attempts == 3

    def test_message(self):
        original = Exception("oops")
        exc = RetryExhausted(original, LLMErrorType.OVERLOAD, 5)
        msg = str(exc)
        assert "5 attempts" in msg
        assert "overload" in msg
        assert "oops" in msg


# ── HookResult ─────────────────────────────────────────────────────

class TestHookResult:
    def test_defaults(self):
        r = HookResult(passed=True)
        assert r.passed is True
        assert r.reason == ""
        assert r.action == "pass"
        assert r.confidence == 1.0

    def test_custom(self):
        r = HookResult(passed=False, reason="too short", action="fallback", confidence=0.0)
        assert r.passed is False
        assert r.action == "fallback"


# ── StopHookValidator ──────────────────────────────────────────────

class TestStopHookValidator:
    @pytest.fixture
    def v(self):
        return StopHookValidator()

    def test_passes_normal_output(self, v):
        result = v.validate("This is a normal medical response with enough content.", "CONVERSATION_AGENT")
        assert result.passed is True
        assert result.confidence == 1.0

    def test_empty_output_fails(self, v):
        result = v.validate("", "RAG_AGENT")
        assert result.passed is False
        assert result.action == "fallback"

    def test_short_output_fails(self, v):
        result = v.validate("hi", "RAG_AGENT")
        assert result.passed is False
        assert "too short" in result.reason.lower() or "Output too short" in result.reason

    def test_dangerous_advice_without_disclaimer(self, v):
        # Text must exceed MIN_OUTPUT_LENGTH=20 to pass completeness check
        result = v.validate(
            "根据检查结果分析，你应该停止服用这个药物，不需要看医生",
            "CONVERSATION_AGENT",
        )
        assert result.passed is False
        # Source code: dangerous advice without disclaimer → action="escalate"
        assert result.action == "escalate"

    def test_dangerous_advice_with_disclaimer_passes(self, v):
        result = v.validate(
            "你应该停止服用这个药物。建议咨询专业医生获得进一步指导。",
            "CONVERSATION_AGENT",
        )
        assert result.passed is True

    def test_confidence_below_threshold(self, v):
        result = v.validate(
            "Adequate length response for medical advice with enough chars.",
            "BRAIN_TUMOR_AGENT",  # threshold=0.7
            confidence=0.3,
        )
        assert result.passed is False
        assert "0.30" in result.reason
        # confidence=0.3 < threshold*0.5=0.35 → action="fallback"
        assert result.action == "fallback"

    def test_confidence_above_threshold_passes(self, v):
        result = v.validate(
            "Adequate length response for medical diagnosis advice.",
            "BRAIN_TUMOR_AGENT",
            confidence=0.8,
        )
        assert result.passed is True

    def test_confidence_none_skips_check(self, v):
        result = v.validate(
            "Adequate length response with no confidence provided.",
            "CONVERSATION_AGENT",
            confidence=None,
        )
        assert result.passed is True

    def test_unknown_agent_uses_default_threshold(self, v):
        result = v.validate(
            "Adequate length response for unknown agent type check.",
            "UNKNOWN_AGENT",
            confidence=0.1,
        )
        # Default threshold 0.3, 0.1 < 0.3 → fail
        assert result.passed is False

    def test_conversation_agent_zero_threshold(self, v):
        result = v.validate(
            "Any length response works for conversation agent.",
            "CONVERSATION_AGENT",
            confidence=0.0,
        )
        assert result.passed is True  # threshold=0.0, 0.0 >= 0.0

    def test_confidence_half_threshold_escalate(self, v):
        """When confidence > threshold*0.5 but < threshold → escalate (not fallback)."""
        result = v.validate(
            "Adequate length response for escalation test check.",
            "RAG_AGENT",  # threshold=0.3
            confidence=0.2,  # > 0.15 but < 0.3
        )
        assert result.passed is False
        assert result.action == "escalate"


# ── llm_call_with_recovery ─────────────────────────────────────────

class TestLLMCallWithRecovery:
    @patch("agents.error_handler.time.sleep")
    def test_success_first_try(self, mock_sleep):
        func = MagicMock(return_value="ok")
        result = llm_call_with_recovery(func, "msg")
        assert result == "ok"
        assert func.call_count == 1
        mock_sleep.assert_not_called()

    @patch("agents.error_handler.time.sleep")
    def test_rate_limit_retries_then_succeeds(self, mock_sleep):
        func = MagicMock(side_effect=[
            _make_error("429 rate limit", "RateLimitError"),
            "ok",
        ])
        result = llm_call_with_recovery(func, "msg", max_retries=2)
        assert result == "ok"
        assert func.call_count == 2
        mock_sleep.assert_called_once()

    @patch("agents.error_handler.time.sleep")
    def test_auth_error_no_retry(self, mock_sleep):
        err = _make_error("invalid api key", "AuthenticationError")
        func = MagicMock(side_effect=err)
        with pytest.raises(Exception, match="invalid api key"):
            llm_call_with_recovery(func, "msg", max_retries=2)
        assert func.call_count == 1
        mock_sleep.assert_not_called()

    @patch("agents.error_handler.time.sleep")
    def test_unknown_error_raises_immediately(self, mock_sleep):
        err = _make_error("something weird")
        func = MagicMock(side_effect=err)
        with pytest.raises(Exception):
            llm_call_with_recovery(func, "msg", max_retries=2)
        # First unknown error → immediate raise (attempt==0)
        assert func.call_count == 1

    @patch("agents.error_handler.time.sleep")
    def test_retry_exhausted(self, mock_sleep):
        err = _make_error("503 service unavailable")
        func = MagicMock(side_effect=err)
        with pytest.raises(RetryExhausted) as exc_info:
            llm_call_with_recovery(func, "msg", max_retries=2)
        assert exc_info.value.attempts == 3
        assert func.call_count == 3

    @patch("agents.error_handler.time.sleep")
    def test_context_length_uses_on_context_too_long(self, mock_sleep):
        err = Exception("maximum context length exceeded")
        truncated_msgs = [_make_msg("user", "short")]
        callback = MagicMock(return_value=truncated_msgs)
        func = MagicMock(side_effect=[err, "ok"])
        result = llm_call_with_recovery(
            func, [_make_msg("user", "long message")],
            max_retries=2,
            on_context_too_long=callback,
        )
        assert result == "ok"
        callback.assert_called_once()

    @patch("agents.error_handler.time.sleep")
    def test_context_length_auto_truncate(self, mock_sleep):
        """When no callback but messages are a list, uses truncate_messages."""
        err = Exception("context_length_exceeded")
        long_msg = _make_msg("user", "x" * 10000)
        func = MagicMock(side_effect=[err, "ok"])
        result = llm_call_with_recovery(
            func, [_make_msg("system", "sys"), long_msg, long_msg, long_msg],
            max_retries=2,
        )
        assert result == "ok"
        assert func.call_count == 2

    @patch("agents.error_handler.time.sleep")
    def test_max_tokens_reduces_max_tokens(self, mock_sleep):
        err = _make_error("max_tokens is too large", "BadRequestError")
        func = MagicMock(side_effect=[err, "ok"])
        result = llm_call_with_recovery(
            func, "msg",
            max_retries=2,
            max_tokens_override=512,
        )
        assert result == "ok"
        # Second call should have max_tokens=512
        _, kwargs = func.call_args
        assert kwargs.get("max_tokens") == 512

    @patch("agents.error_handler.time.sleep")
    def test_overload_retries_with_backoff(self, mock_sleep):
        err = _make_error("503 overloaded")
        func = MagicMock(side_effect=[err, err, "ok"])
        result = llm_call_with_recovery(func, "msg", max_retries=3)
        assert result == "ok"
        assert func.call_count == 3
        assert mock_sleep.call_count == 2  # slept twice before success

    @patch("agents.error_handler.time.sleep")
    def test_context_length_raises_if_no_strategy(self, mock_sleep):
        """Context overflow with no callback and non-list messages → raises immediately (not RetryExhausted)."""
        err = Exception("maximum context length exceeded")
        func = MagicMock(side_effect=err)
        # Source code: messages not a list → raises original exception immediately
        with pytest.raises(Exception, match="maximum context length exceeded"):
            llm_call_with_recovery(func, "not_a_list", max_retries=2)

    @patch("agents.error_handler.time.sleep")
    def test_zero_retries(self, mock_sleep):
        func = MagicMock(side_effect=_make_error("503 overloaded"))
        with pytest.raises(RetryExhausted) as exc_info:
            llm_call_with_recovery(func, "msg", max_retries=0)
        assert exc_info.value.attempts == 1
        assert func.call_count == 1
