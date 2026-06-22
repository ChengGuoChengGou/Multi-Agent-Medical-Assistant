"""
Unit tests for agents/error_handler.py — LLM error classification, retry, truncation, StopHook.
Run: python -m pytest tests/test_agents_error_handler.py -v
"""
import os
import sys
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.error_handler import (
    HookResult,
    LLMErrorType,
    RetryExhausted,
    StopHookValidator,
    _estimate_token_count,
    _get_retry_delay,
    classify_error,
    llm_call_with_recovery,
    truncate_messages,
)


# ═══════════════════════════════════════════════════════════════
# classify_error
# ═══════════════════════════════════════════════════════════════
class TestClassifyError:
    """classify_error() should map exception messages/types to LLMErrorType."""

    # ── Context length ──
    @pytest.mark.parametrize("msg", [
        "maximum context length is 4096",
        "context_length_exceeded",
        "max_tokens exceeded the limit",
        "too many tokens in the request",
        "request too large for the model",
    ])
    def test_context_length_patterns(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.CONTEXT_LENGTH

    # ── Rate limit ──
    @pytest.mark.parametrize("msg", [
        "rate limit exceeded",
        "429 Too Many Requests",
        "too many requests, please slow down",
        "RateLimitError occurred",
    ])
    def test_rate_limit_patterns(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.RATE_LIMIT

    # ── Overload ──
    @pytest.mark.parametrize("msg", [
        "server overloaded",
        "503 Service Unavailable",
        "service unavailable",
    ])
    def test_overload_patterns(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.OVERLOAD

    # ── Max tokens ──
    @pytest.mark.parametrize("msg", [
        "max_tokens value is too large",
    ])
    def test_max_tokens_patterns(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.MAX_TOKENS

    # ── Auth ──
    @pytest.mark.parametrize("msg", [
        "401 Unauthorized",
        "unauthorized access",
        "invalid api key provided",
        "authentication failed",
    ])
    def test_auth_patterns(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.AUTH

    # ── Exception type name matching ──
    def test_rate_limit_error_type_name(self):
        """Exception class name containing 'RateLimitError'."""
        class RateLimitError(Exception):
            pass
        exc = RateLimitError("something went wrong")
        assert classify_error(exc) == LLMErrorType.RATE_LIMIT

    def test_authentication_error_type_name(self):
        class AuthenticationError(Exception):
            pass
        exc = AuthenticationError("something went wrong")
        assert classify_error(exc) == LLMErrorType.AUTH

    def test_bad_request_max_tokens_type_name(self):
        class BadRequestError(Exception):
            pass
        exc = BadRequestError("max_tokens is too large for this model")
        assert classify_error(exc) == LLMErrorType.MAX_TOKENS

    # ── Unknown ──
    def test_unknown_error(self):
        assert classify_error(Exception("something random happened")) == LLMErrorType.UNKNOWN

    def test_empty_message(self):
        assert classify_error(Exception("")) == LLMErrorType.UNKNOWN

    # ── Priority: message patterns checked first ──
    def test_rate_limit_message_beats_type_name(self):
        """If message matches RATE_LIMIT pattern, it should be RATE_LIMIT even if type name is generic."""
        assert classify_error(Exception("rate limit hit")) == LLMErrorType.RATE_LIMIT


# ═══════════════════════════════════════════════════════════════
# _get_retry_delay
# ═══════════════════════════════════════════════════════════════
class TestGetRetryDelay:
    def test_rate_limit_backoff(self):
        # 2^(attempt+1): attempt 0 → 2, attempt 1 → 4, attempt 2 → 8
        assert _get_retry_delay(LLMErrorType.RATE_LIMIT, 0) == 2.0
        assert _get_retry_delay(LLMErrorType.RATE_LIMIT, 1) == 4.0
        assert _get_retry_delay(LLMErrorType.RATE_LIMIT, 2) == 8.0

    def test_rate_limit_capped_at_30(self):
        assert _get_retry_delay(LLMErrorType.RATE_LIMIT, 10) == 30.0

    def test_overload_backoff(self):
        # 2^attempt: attempt 0 → 1, attempt 1 → 2, attempt 2 → 4
        assert _get_retry_delay(LLMErrorType.OVERLOAD, 0) == 1.0
        assert _get_retry_delay(LLMErrorType.OVERLOAD, 1) == 2.0
        assert _get_retry_delay(LLMErrorType.OVERLOAD, 2) == 4.0

    def test_overload_capped_at_15(self):
        assert _get_retry_delay(LLMErrorType.OVERLOAD, 10) == 15.0

    def test_context_length_no_delay(self):
        assert _get_retry_delay(LLMErrorType.CONTEXT_LENGTH, 0) == 0.0
        assert _get_retry_delay(LLMErrorType.CONTEXT_LENGTH, 5) == 0.0

    def test_other_errors_constant_1s(self):
        assert _get_retry_delay(LLMErrorType.UNKNOWN, 0) == 1.0
        assert _get_retry_delay(LLMErrorType.MAX_TOKENS, 3) == 1.0
        assert _get_retry_delay(LLMErrorType.AUTH, 0) == 1.0


# ═══════════════════════════════════════════════════════════════
# _estimate_token_count
# ═══════════════════════════════════════════════════════════════
class TestEstimateTokenCount:
    def test_empty_string(self):
        assert _estimate_token_count("") == 0

    def test_english_text(self):
        # English: 0.25 tokens per char
        count = _estimate_token_count("hello")
        assert count == int(5 * 0.25)  # 1

    def test_chinese_text(self):
        # CJK: 2 tokens per char
        count = _estimate_token_count("你好")
        assert count == 4  # 2 chars * 2

    def test_mixed_text(self):
        # "hello你好" = 5 English chars + 2 CJK chars
        count = _estimate_token_count("hello你好")
        assert count == 2 * 2 + int(5 * 0.25)  # 4 + 1 = 5

    def test_long_text_positive(self):
        text = "This is a longer sentence with many words and characters to estimate tokens for."
        assert _estimate_token_count(text) > 0


# ═══════════════════════════════════════════════════════════════
# truncate_messages
# ═══════════════════════════════════════════════════════════════
class TestTruncateMessages:
    """Test truncate_messages with mock message objects."""

    def _msg(self, content, msg_type="human"):
        """Create a mock message object."""
        m = MagicMock()
        m.content = content
        m.type = msg_type
        return m

    def test_empty_messages(self):
        assert truncate_messages([]) == []

    def test_single_message_unchanged(self):
        msgs = [self._msg("hello")]
        assert truncate_messages(msgs) == msgs

    def test_two_messages_unchanged(self):
        msgs = [self._msg("system prompt", "system"), self._msg("user query")]
        assert truncate_messages(msgs) == msgs

    def test_three_messages_preserves_system_and_last(self):
        msgs = [
            self._msg("system prompt", "system"),
            self._msg("old message"),
            self._msg("latest query"),
        ]
        result = truncate_messages(msgs, target_tokens=10000)
        # Should keep system + last at minimum
        assert result[0].content == "system prompt"
        assert result[-1].content == "latest query"

    def test_truncates_oldest_when_budget_tight(self):
        msgs = [
            self._msg("system", "system"),
            self._msg("msg1"),
            self._msg("msg2"),
            self._msg("msg3"),
            self._msg("msg4"),
            self._msg("latest"),
        ]
        # Very tight budget: should drop middle messages
        result = truncate_messages(msgs, target_tokens=5)
        # System + latest must remain
        assert result[0].content == "system"
        assert result[-1].content == "latest"
        assert len(result) <= len(msgs)

    def test_no_truncation_when_budget_generous(self):
        msgs = [
            self._msg("system", "system"),
            self._msg("short"),
            self._msg("latest"),
        ]
        result = truncate_messages(msgs, target_tokens=100000)
        assert len(result) == len(msgs)


# ═══════════════════════════════════════════════════════════════
# RetryExhausted
# ═══════════════════════════════════════════════════════════════
class TestRetryExhausted:
    def test_attributes(self):
        orig = Exception("original")
        exc = RetryExhausted(orig, LLMErrorType.RATE_LIMIT, 3)
        assert exc.last_error is orig
        assert exc.error_type == LLMErrorType.RATE_LIMIT
        assert exc.attempts == 3
        assert "3 attempts" in str(exc)
        assert "rate_limit" in str(exc)


# ═══════════════════════════════════════════════════════════════
# llm_call_with_recovery
# ═══════════════════════════════════════════════════════════════
class TestLlmCallWithRecovery:
    @patch("agents.error_handler.time.sleep")
    def test_success_on_first_try(self, mock_sleep):
        func = MagicMock(return_value="ok")
        result = llm_call_with_recovery(func, "msg", max_retries=2)
        assert result == "ok"
        func.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("agents.error_handler.time.sleep")
    def test_auth_error_no_retry(self, mock_sleep):
        func = MagicMock(side_effect=Exception("401 Unauthorized"))
        with pytest.raises(Exception, match="401"):
            llm_call_with_recovery(func, "msg", max_retries=2)
        # Auth error → raised immediately, no retry
        assert func.call_count == 1
        mock_sleep.assert_not_called()

    @patch("agents.error_handler.time.sleep")
    def test_unknown_error_raises_immediately(self, mock_sleep):
        func = MagicMock(side_effect=Exception("something weird"))
        with pytest.raises(Exception, match="something weird"):
            llm_call_with_recovery(func, "msg", max_retries=2)
        # Unknown on attempt 0 → raise immediately
        assert func.call_count == 1

    @patch("agents.error_handler.time.sleep")
    def test_rate_limit_retries_then_exhausted(self, mock_sleep):
        func = MagicMock(side_effect=Exception("rate limit exceeded"))
        with pytest.raises(RetryExhausted) as exc_info:
            llm_call_with_recovery(func, "msg", max_retries=2)
        # 1 initial + 2 retries = 3 calls
        assert func.call_count == 3
        assert exc_info.value.error_type == LLMErrorType.RATE_LIMIT
        assert mock_sleep.call_count == 2  # 2 retry delays

    @patch("agents.error_handler.time.sleep")
    def test_rate_limit_succeeds_on_retry(self, mock_sleep):
        func = MagicMock(side_effect=[
            Exception("429 Too Many Requests"),
            "success",
        ])
        result = llm_call_with_recovery(func, "msg", max_retries=2)
        assert result == "success"
        assert func.call_count == 2

    @patch("agents.error_handler.time.sleep")
    def test_context_length_auto_truncates(self, mock_sleep):
        # Create mock messages that look like LangChain messages
        msg_system = MagicMock()
        msg_system.type = "system"
        msg_system.content = "system"
        msg_old = MagicMock()
        msg_old.type = "human"
        msg_old.content = "old message"
        msg_new = MagicMock()
        msg_new.type = "human"
        msg_new.content = "new message"
        messages = [msg_system, msg_old, msg_new]

        func = MagicMock(side_effect=[
            Exception("maximum context length exceeded"),
            "truncated_ok",
        ])
        result = llm_call_with_recovery(func, messages, max_retries=2)
        assert result == "truncated_ok"
        assert func.call_count == 2

    @patch("agents.error_handler.time.sleep")
    def test_max_tokens_reduces_kwargs(self, mock_sleep):
        func = MagicMock(side_effect=[
            Exception("max_tokens value is too large"),
            "ok",
        ])
        result = llm_call_with_recovery(
            func, "msg", max_retries=2, max_tokens_override=500
        )
        assert result == "ok"
        # Second call should have max_tokens=500
        _, kwargs = func.call_args
        assert kwargs.get("max_tokens") == 500

    @patch("agents.error_handler.time.sleep")
    def test_custom_on_context_too_long_callback(self, mock_sleep):
        new_msgs = [MagicMock(content="truncated", type="human")]
        callback = MagicMock(return_value=new_msgs)
        func = MagicMock(side_effect=[
            Exception("context_length_exceeded"),
            "ok",
        ])
        result = llm_call_with_recovery(
            func, [MagicMock(content="long", type="human")],
            max_retries=2, on_context_too_long=callback,
        )
        assert result == "ok"
        callback.assert_called_once()


# ═══════════════════════════════════════════════════════════════
# HookResult
# ═══════════════════════════════════════════════════════════════
class TestHookResult:
    def test_defaults(self):
        r = HookResult(passed=True)
        assert r.passed is True
        assert r.reason == ""
        assert r.action == "pass"
        assert r.confidence == 1.0

    def test_custom_values(self):
        r = HookResult(passed=False, reason="too short", action="fallback", confidence=0.1)
        assert r.passed is False
        assert r.action == "fallback"


# ═══════════════════════════════════════════════════════════════
# StopHookValidator
# ═══════════════════════════════════════════════════════════════
class TestStopHookValidator:
    def setup_method(self):
        self.v = StopHookValidator()

    # ── Completeness check ──
    def test_empty_output_fails(self):
        r = self.v.validate("", "CONVERSATION_AGENT")
        assert r.passed is False
        assert r.action == "fallback"

    def test_short_output_fails(self):
        r = self.v.validate("short", "CONVERSATION_AGENT")
        assert r.passed is False
        assert "too short" in r.reason.lower() or "20" in r.reason

    def test_sufficient_output_passes_completeness(self):
        text = "A" * 50  # > 20 chars
        r = self.v.validate(text, "CONVERSATION_AGENT")
        assert r.passed is True

    # ── Medical safety ──
    def test_dangerous_advice_without_disclaimer_fails(self):
        text = "你应该停止吃药，不需要看医生了。" + "x" * 20
        r = self.v.validate(text, "RAG_AGENT")
        assert r.passed is False
        assert r.action == "escalate"
        assert "dangerous" in r.reason.lower()

    def test_dangerous_advice_with_disclaimer_passes(self):
        text = "你应该停止吃药。建议咨询专业医生获取进一步指导。" + "x" * 20
        r = self.v.validate(text, "RAG_AGENT")
        # Should pass because disclaimer is present
        assert r.passed is True

    def test_safe_output_passes(self):
        text = "根据文献资料，该药物通常用于治疗高血压。建议咨询医生确认。" + "x" * 20
        r = self.v.validate(text, "RAG_AGENT")
        assert r.passed is True

    # ── Confidence threshold ──
    def test_low_confidence_escalates(self):
        text = "A" * 50
        # BRAIN_TUMOR_AGENT threshold=0.7; 0.5 < 0.7 → fail
        # 0.5 > 0.7*0.5=0.35 → escalate (not fallback)
        r = self.v.validate(text, "BRAIN_TUMOR_AGENT", confidence=0.5)
        assert r.passed is False
        assert r.action == "escalate"

    def test_high_confidence_passes(self):
        text = "A" * 50
        r = self.v.validate(text, "BRAIN_TUMOR_AGENT", confidence=0.9)
        assert r.passed is True

    def test_very_low_confidence_fallback(self):
        text = "A" * 50
        r = self.v.validate(text, "BRAIN_TUMOR_AGENT", confidence=0.1)
        # 0.1 < 0.7*0.5 = 0.35 → fallback
        assert r.passed is False
        assert r.action == "fallback"

    def test_no_confidence_skips_threshold(self):
        text = "A" * 50
        r = self.v.validate(text, "BRAIN_TUMOR_AGENT", confidence=None)
        assert r.passed is True

    def test_unknown_agent_uses_default_threshold(self):
        text = "A" * 50
        # Default threshold is 0.3
        r = self.v.validate(text, "UNKNOWN_AGENT", confidence=0.1)
        assert r.passed is False
