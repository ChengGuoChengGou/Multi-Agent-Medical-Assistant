"""Tests for agents/error_handler.py — LLM Error Classification & Recovery (Phase 58d)

Covers:
- classify_error: pattern-based error type detection
- _get_retry_delay: backoff delay per error type
- _estimate_token_count: CJK-aware token estimation
- truncate_messages: message history truncation preserving system + last msg
- llm_call_with_recovery: retry wrapper with recovery strategies
- StopHookValidator: completeness, medical safety, confidence checks
- RetryExhausted: exception details
- HookResult: dataclass fields
"""

from __future__ import annotations

import os
import re
import sys
import time

import pytest

# Bootstrap project root
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Remove stale cached agents modules so our patched versions take effect
for mod_key in list(sys.modules):
    if mod_key.startswith("agents.error_handler"):
        del sys.modules[mod_key]

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeMessage:
    """Lightweight message object mimicking langchain_core."""
    def __init__(self, role: str = "system", content: str = ""):
        self.type = role
        self.content = content


def _msg(content: str, role: str = "human") -> FakeMessage:
    return FakeMessage(role=role, content=content)


# =========================================================================
# classify_error
# =========================================================================

class TestClassifyError:
    """Pattern-based error classification."""

    # --- Context length ---
    @pytest.mark.parametrize("msg", [
        "maximum context length is 4096 tokens",
        "context_length_exceeded: request too large",
        "too many tokens in the request",
        "request too large for model gpt-4",
    ])
    def test_context_length(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.CONTEXT_LENGTH

    # --- Rate limit ---
    @pytest.mark.parametrize("msg", [
        "rate_limit_exceeded: too many requests",
        "429 Too Many Requests",
        "too many requests, please slow down",
    ])
    def test_rate_limit(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.RATE_LIMIT

    # --- Overload ---
    @pytest.mark.parametrize("msg", [
        "server is overloaded",
        "503 Service Unavailable",
        "service unavailable right now",
        "server busy, try again later",
    ])
    def test_overload(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.OVERLOAD

    # --- Max tokens ---
    @pytest.mark.parametrize("msg", [
        "max_tokens too large for this model",
    ])
    def test_max_tokens(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.MAX_TOKENS

    def test_max_completion_tokens_is_context_length(self):
        """max_completion_tokens matches CONTEXT_LENGTH pattern (checked first)."""
        assert classify_error(Exception("max_completion_tokens exceeded the limit")) == LLMErrorType.CONTEXT_LENGTH

    def test_max_tokens_must_be_less_is_unknown(self):
        """'max_tokens must be less' matches no pattern → UNKNOWN."""
        assert classify_error(Exception("max_tokens must be less than 16000")) == LLMErrorType.UNKNOWN

    # --- Auth ---
    @pytest.mark.parametrize("msg", [
        "401 Unauthorized",
        "unauthorized: invalid credentials",
        "invalid api key provided",
        "authentication failed",
    ])
    def test_auth(self, msg):
        assert classify_error(Exception(msg)) == LLMErrorType.AUTH

    # --- Exception type name detection ---
    def test_rate_limit_by_type_name(self):
        class RateLimitError(Exception):
            pass
        exc = RateLimitError("something happened")
        assert classify_error(exc) == LLMErrorType.RATE_LIMIT

    def test_auth_by_type_name(self):
        class AuthenticationError(Exception):
            pass
        exc = AuthenticationError("bad creds")
        assert classify_error(exc) == LLMErrorType.AUTH

    def test_max_tokens_by_type_name_and_msg(self):
        class BadRequestError(Exception):
            pass
        exc = BadRequestError("max_tokens value too large")
        assert classify_error(exc) == LLMErrorType.MAX_TOKENS

    # --- Unknown ---
    def test_unknown(self):
        assert classify_error(Exception("something random")) == LLMErrorType.UNKNOWN

    def test_unknown_bad_request_without_max_tokens(self):
        class BadRequestError(Exception):
            pass
        exc = BadRequestError("some other bad request")
        assert classify_error(exc) == LLMErrorType.UNKNOWN

    # --- Priority: message patterns first, then type name ---
    def test_message_pattern_takes_priority(self):
        class AuthenticationError(Exception):
            pass
        # Message says "rate_limit" → RATE_LIMIT, not AUTH
        exc = AuthenticationError("rate_limit exceeded")
        assert classify_error(exc) == LLMErrorType.RATE_LIMIT


# =========================================================================
# _get_retry_delay
# =========================================================================

class TestGetRetryDelay:
    """Backoff delay calculations per error type."""

    def test_rate_limit_exponential(self):
        # 2^(attempt+1): 4, 8, 16
        assert _get_retry_delay(LLMErrorType.RATE_LIMIT, 0) == 2.0
        assert _get_retry_delay(LLMErrorType.RATE_LIMIT, 1) == 4.0
        assert _get_retry_delay(LLMErrorType.RATE_LIMIT, 2) == 8.0

    def test_rate_limit_cap_at_30(self):
        # attempt=4 → 2^5=32 → capped at 30
        assert _get_retry_delay(LLMErrorType.RATE_LIMIT, 4) == 30.0

    def test_overload_moderate_backoff(self):
        # 2^attempt: 1, 2, 4
        assert _get_retry_delay(LLMErrorType.OVERLOAD, 0) == 1.0
        assert _get_retry_delay(LLMErrorType.OVERLOAD, 1) == 2.0
        assert _get_retry_delay(LLMErrorType.OVERLOAD, 2) == 4.0

    def test_overload_cap_at_15(self):
        # attempt=4 → 16 → capped at 15
        assert _get_retry_delay(LLMErrorType.OVERLOAD, 4) == 15.0

    def test_context_length_zero_delay(self):
        assert _get_retry_delay(LLMErrorType.CONTEXT_LENGTH, 0) == 0.0
        assert _get_retry_delay(LLMErrorType.CONTEXT_LENGTH, 3) == 0.0

    def test_other_types_one_second(self):
        assert _get_retry_delay(LLMErrorType.MAX_TOKENS, 0) == 1.0
        assert _get_retry_delay(LLMErrorType.AUTH, 0) == 1.0
        assert _get_retry_delay(LLMErrorType.UNKNOWN, 0) == 1.0


# =========================================================================
# _estimate_token_count
# =========================================================================

class TestEstimateTokenCount:
    """CJK-aware rough token estimation."""

    def test_english_text(self):
        # "hello" = 5 chars * 0.25 = 1.25 → int(1.25) = 1
        count = _estimate_token_count("hello")
        assert count == 1

    def test_chinese_text(self):
        # "你好" = 2 CJK chars * 2 = 4
        count = _estimate_token_count("你好")
        assert count == 4

    def test_mixed_text(self):
        # "你好hello" = 2*2 + 5*0.25 = 4 + 1.25 → 4+1 = 5
        count = _estimate_token_count("你好hello")
        assert count == 5

    def test_empty_string(self):
        assert _estimate_token_count("") == 0

    def test_long_english(self):
        # 1000 English chars ≈ 250 tokens
        count = _estimate_token_count("a" * 1000)
        assert count == 250

    def test_long_chinese(self):
        # 1000 Chinese chars = 2000 tokens
        count = _estimate_token_count("中" * 1000)
        assert count == 2000


# =========================================================================
# truncate_messages
# =========================================================================

class TestTruncateMessages:
    """Message history truncation preserving system + last message."""

    def test_empty_list(self):
        assert truncate_messages([]) == []

    def test_short_list_unchanged(self):
        msgs = [_msg("hi"), _msg("hello")]
        assert truncate_messages(msgs) == msgs

    def test_system_message_preserved(self):
        msgs = [
            _msg("You are a doctor", role="system"),
            _msg("msg1"),
            _msg("msg2"),
            _msg("msg3"),
            _msg("last question"),
        ]
        result = truncate_messages(msgs, target_tokens=100)
        # System msg should be first
        assert result[0].content == "You are a doctor"
        # Last msg should be preserved
        assert result[-1].content == "last question"

    def test_last_message_always_preserved(self):
        msgs = [_msg(f"msg{i}") for i in range(20)]
        result = truncate_messages(msgs, target_tokens=50)
        assert result[-1].content == "msg19"

    def test_truncation_reduces_count(self):
        msgs = [_msg(f"message number {i}" * 10) for i in range(20)]
        result = truncate_messages(msgs, target_tokens=100)
        assert len(result) < len(msgs)

    def test_large_budget_preserves_all(self):
        msgs = [_msg(f"msg{i}") for i in range(10)]
        result = truncate_messages(msgs, target_tokens=999999)
        assert len(result) == len(msgs)

    def test_first_msg_not_system_still_works(self):
        msgs = [_msg("not system", role="human"), _msg("second"), _msg("third")]
        result = truncate_messages(msgs, target_tokens=100)
        assert len(result) == 3  # Too short to truncate

    def test_only_system_and_one_other(self):
        msgs = [_msg("system prompt", role="system"), _msg("user query")]
        result = truncate_messages(msgs, target_tokens=100)
        assert len(result) == 2

    def test_two_conversation_msgs_unchanged(self):
        msgs = [_msg("system", role="system"), _msg("a"), _msg("b")]
        result = truncate_messages(msgs, target_tokens=100)
        assert len(result) == 3  # Can't truncate further (len<=2 for conversation)

    def test_no_content_attr_handled(self):
        """Messages without content attribute should not crash."""
        class NoContentMsg:
            type = "human"
        msgs = [NoContentMsg(), NoContentMsg(), _msg("last")]
        result = truncate_messages(msgs, target_tokens=100)
        assert result[-1].content == "last"


# =========================================================================
# llm_call_with_recovery
# =========================================================================

class TestLLMCallWithRecovery:
    """Retry wrapper with error classification and recovery."""

    def test_success_on_first_try(self):
        mock = lambda msgs: "answer"
        result = llm_call_with_recovery(mock, ["question"])
        assert result == "answer"

    def test_auth_error_no_retry(self):
        call_count = [0]
        def mock(msgs):
            call_count[0] += 1
            raise Exception("401 Unauthorized: invalid api key")
        with pytest.raises(Exception, match="Unauthorized"):
            llm_call_with_recovery(mock, ["q"], max_retries=3)
        assert call_count[0] == 1  # Not retried

    def test_unknown_error_raises_immediately(self):
        call_count = [0]
        def mock(msgs):
            call_count[0] += 1
            raise Exception("something weird")
        with pytest.raises(Exception, match="weird"):
            llm_call_with_recovery(mock, ["q"], max_retries=3)
        assert call_count[0] == 1  # First unknown → raise immediately

    def test_rate_limit_retries_with_backoff(self):
        call_count = [0]
        def mock(msgs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("429 Too Many Requests")
            return "success"
        # Patch time.sleep to avoid actual delay
        original_sleep = time.sleep
        time.sleep = lambda x: None
        try:
            result = llm_call_with_recovery(mock, ["q"], max_retries=3)
            assert result == "success"
            assert call_count[0] == 3
        finally:
            time.sleep = original_sleep

    def test_rate_limit_exhausted(self):
        def mock(msgs):
            raise Exception("rate_limit exceeded")
        original_sleep = time.sleep
        time.sleep = lambda x: None
        try:
            with pytest.raises(RetryExhausted) as exc_info:
                llm_call_with_recovery(mock, ["q"], max_retries=2)
            assert exc_info.value.attempts == 3
            assert exc_info.value.error_type == LLMErrorType.RATE_LIMIT
        finally:
            time.sleep = original_sleep

    def test_context_length_auto_truncation(self):
        call_count = [0]
        messages = [_msg(f"msg{i}") for i in range(10)]
        def mock(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("maximum context length exceeded")
            return "success"
        result = llm_call_with_recovery(mock, messages, max_retries=2)
        assert result == "success"
        assert call_count[0] == 2

    def test_context_length_custom_callback(self):
        call_count = [0]
        truncated_msgs = [_msg("truncated")]
        def mock(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("context_length_exceeded")
            # Second call should get truncated messages
            assert len(msgs) == 1
            return "ok"
        result = llm_call_with_recovery(
            mock, [_msg("orig")],
            max_retries=2,
            on_context_too_long=lambda: truncated_msgs,
        )
        assert result == "ok"

    def test_max_tokens_override(self):
        call_count = [0]
        def mock(msgs, max_tokens=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("max_tokens too large for this model")
            assert max_tokens == 500
            return "ok"
        result = llm_call_with_recovery(
            mock, ["q"],
            max_retries=2,
            max_tokens_override=500,
        )
        assert result == "ok"

    def test_overload_retries(self):
        call_count = [0]
        def mock(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("503 Service Unavailable")
            return "ok"
        original_sleep = time.sleep
        time.sleep = lambda x: None
        try:
            result = llm_call_with_recovery(mock, ["q"], max_retries=2)
            assert result == "ok"
        finally:
            time.sleep = original_sleep

    def test_no_args_calls_func(self):
        def mock():
            return "no-args"
        result = llm_call_with_recovery(mock)
        assert result == "no-args"


# =========================================================================
# RetryExhausted
# =========================================================================

class TestRetryExhausted:
    """Exception details when retries are exhausted."""

    def test_attributes(self):
        orig = Exception("fail")
        exc = RetryExhausted(orig, LLMErrorType.RATE_LIMIT, 3)
        assert exc.last_error is orig
        assert exc.error_type == LLMErrorType.RATE_LIMIT
        assert exc.attempts == 3

    def test_message(self):
        orig = Exception("too many")
        exc = RetryExhausted(orig, LLMErrorType.RATE_LIMIT, 2)
        msg = str(exc)
        assert "2 attempts" in msg
        assert "rate_limit" in msg
        assert "too many" in msg


# =========================================================================
# HookResult
# =========================================================================

class TestHookResult:
    """HookResult dataclass defaults and custom values."""

    def test_defaults(self):
        hr = HookResult(passed=True)
        assert hr.passed is True
        assert hr.reason == ""
        assert hr.action == "pass"
        assert hr.confidence == 1.0

    def test_custom_values(self):
        hr = HookResult(passed=False, reason="too short", action="fallback", confidence=0.0)
        assert hr.passed is False
        assert hr.reason == "too short"
        assert hr.action == "fallback"
        assert hr.confidence == 0.0


# =========================================================================
# StopHookValidator
# =========================================================================

class TestStopHookValidator:
    """Post-agent validation checks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.validator = StopHookValidator()

    # --- Completeness ---
    def test_empty_output_fails(self):
        result = self.validator.validate("", "CONVERSATION_AGENT")
        assert result.passed is False
        assert "too short" in result.reason.lower()
        assert result.action == "fallback"

    def test_short_output_fails(self):
        result = self.validator.validate("hi", "CONVERSATION_AGENT")
        assert result.passed is False

    def test_none_output_fails(self):
        result = self.validator.validate(None, "CONVERSATION_AGENT")
        assert result.passed is False

    def test_adequate_output_passes(self):
        text = "This is a long enough response for validation purposes."
        result = self.validator.validate(text, "CONVERSATION_AGENT")
        assert result.passed is True

    # --- Medical safety ---
    def test_dangerous_advice_without_disclaimer(self):
        text = "You should stop taking medication immediately."
        result = self.validator.validate(text, "CONVERSATION_AGENT")
        assert result.passed is False
        assert "dangerous" in result.reason.lower()
        assert result.action == "escalate"

    def test_dangerous_advice_with_disclaimer(self):
        text = "你应该停止服用这个药物。但建议咨询专业医生获取专业医疗建议。"
        result = self.validator.validate(text, "CONVERSATION_AGENT")
        assert result.passed is True

    def test_dont_need_doctor_without_disclaimer(self):
        text = "你不需要看医生，这个症状没什么大不了的，吃点药就行了。"
        result = self.validator.validate(text, "CONVERSATION_AGENT")
        assert result.passed is False
        assert result.action == "escalate"

    def test_cancer_diagnosis_without_disclaimer(self):
        text = "根据检查结果，诊断为恶性肿瘤，确诊就是癌症。"
        result = self.validator.validate(text, "CONVERSATION_AGENT")
        assert result.passed is False

    def test_safe_medical_advice_passes(self):
        text = "建议您保持规律作息，适当运动，如有不适请及时就医。"
        result = self.validator.validate(text, "CONVERSATION_AGENT")
        assert result.passed is True

    # --- Confidence threshold ---
    def test_low_confidence_rag_agent(self):
        text = "This is a valid medical response with enough characters."
        result = self.validator.validate(text, "RAG_AGENT", confidence=0.1)
        assert result.passed is False
        assert "confidence" in result.reason.lower()

    def test_high_confidence_rag_agent(self):
        text = "This is a valid medical response with enough characters."
        result = self.validator.validate(text, "RAG_AGENT", confidence=0.8)
        assert result.passed is True

    def test_no_confidence_skips_check(self):
        text = "This is a valid medical response with enough characters."
        result = self.validator.validate(text, "UNKNOWN_AGENT", confidence=None)
        assert result.passed is True

    def test_conversation_agent_zero_threshold(self):
        """CONVERSATION_AGENT has 0.0 threshold → any confidence passes."""
        text = "This is a valid medical response with enough characters."
        result = self.validator.validate(text, "CONVERSATION_AGENT", confidence=0.0)
        assert result.passed is True

    def test_brain_tumor_agent_high_threshold(self):
        text = "This is a valid medical response with enough characters."
        result = self.validator.validate(text, "BRAIN_TUMOR_AGENT", confidence=0.5)
        assert result.passed is False
        # Below 0.7 threshold, but above 0.35 (50% of 0.7) → escalate
        assert result.action == "escalate"

    def test_very_low_confidence_fallback(self):
        text = "This is a valid medical response with enough characters."
        # Below 50% of 0.7 = 0.35 → fallback
        result = self.validator.validate(text, "BRAIN_TUMOR_AGENT", confidence=0.1)
        assert result.passed is False
        assert result.action == "fallback"

    def test_unknown_agent_default_threshold(self):
        text = "This is a valid medical response with enough characters."
        # Default threshold is 0.3
        result = self.validator.validate(text, "CUSTOM_AGENT", confidence=0.1)
        assert result.passed is False
