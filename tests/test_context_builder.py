"""
Tests for agents/context_builder.py — Phase 63
Pure-logic tests for:
  1. compress_history_tags() — thinking/tool_result tag compression
  2. ContextSegments — to_prompt()
  3. ContextBuilder — fluent API, format_chat_history, format_vector_memory,
     summarize_and_truncate, format_conversation_summary
  4. MedicalSystemPrompt — constants existence
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.context_builder import (
    compress_history_tags,
    ContextSegments,
    ContextBuilder,
    MedicalSystemPrompt,
)


# ===== compress_history_tags =====


class TestCompressHistoryTags:
    """Tests for the compress_history_tags() pure function."""

    def test_short_thinking_kept(self):
        """Short <thinking> blocks are kept unchanged."""
        messages = [HumanMessage(content="<thinking>short</thinking>hello")]
        result = compress_history_tags(messages, max_tag_len=200)
        assert "<thinking>" in result[0].content
        assert "hello" in result[0].content

    def test_long_thinking_compressed(self):
        """Long <thinking> blocks are compressed."""
        long_think = "<thinking>" + "x" * 300 + "</thinking>"
        messages = [HumanMessage(content=long_think)]
        result = compress_history_tags(messages, max_tag_len=200, keep_recent=0)
        assert "reasoning omitted" in result[0].content

    def test_long_tool_result_compressed(self):
        """Long <tool_result> blocks are compressed."""
        long_tool = "<tool_result>" + "y" * 300 + "</tool_result>"
        messages = [AIMessage(content=long_tool)]
        result = compress_history_tags(messages, max_tag_len=200, keep_recent=0)
        assert "tool output omitted" in result[0].content

    def test_short_tags_preserved(self):
        """Short tags below max_tag_len are preserved."""
        msg = AIMessage(content="<tool_result>small</tool_result>")
        result = compress_history_tags([msg], max_tag_len=200)
        assert result[0].content == msg.content

    def test_keep_recent_messages_untouched(self):
        """Recent messages are never compressed."""
        old = HumanMessage(content="<thinking>" + "z" * 300 + "</thinking>")
        recent = HumanMessage(content="recent message")
        result = compress_history_tags([old, recent], max_tag_len=200, keep_recent=1)
        assert result[1].content == "recent message"

    def test_mixed_content_partial_compression(self):
        """Message with both thinking and regular text."""
        content = "<thinking>" + "a" * 300 + "</thinking>actual answer"
        msg = AIMessage(content=content)
        result = compress_history_tags([msg], max_tag_len=200, keep_recent=0)
        assert "actual answer" in result[0].content

    def test_empty_messages_list(self):
        result = compress_history_tags([])
        assert result == []

    def test_no_tags_no_change(self):
        msg = HumanMessage(content="plain text")
        result = compress_history_tags([msg])
        assert result[0].content == "plain text"

    def test_nested_tags(self):
        """Tags within tags should be handled gracefully."""
        content = "<thinking><inner>deep</inner></thinking>"
        msg = AIMessage(content=content + "x" * 300)
        result = compress_history_tags([msg], max_tag_len=50, keep_recent=0)
        # Should not crash
        assert len(result) == 1

    def test_non_string_content_skipped(self):
        """Messages with non-string content (e.g. list) are passed through."""
        msg = AIMessage(content=[{"type": "text", "text": "hi"}])
        result = compress_history_tags([msg])
        assert len(result) == 1


# ===== ContextSegments =====


class TestContextSegments:
    """Tests for the ContextSegments dataclass."""

    def test_to_prompt_all_segments(self):
        seg = ContextSegments(
            system_prompt="sys",
            vector_memory="mem",
            medical_kb="kb",
            chat_history="hist",
        )
        prompt = seg.to_prompt()
        assert "[System Instructions]" in prompt
        assert "[Relevant Medical Memory]" in prompt
        assert "[Medical Knowledge Base]" in prompt
        assert "[Conversation History]" in prompt
        assert "sys" in prompt

    def test_to_prompt_only_system(self):
        seg = ContextSegments(system_prompt="sys only")
        prompt = seg.to_prompt()
        assert "[System Instructions]" in prompt
        assert "[Relevant Medical Memory]" not in prompt

    def test_to_prompt_custom_separator(self):
        seg = ContextSegments(system_prompt="a", chat_history="b")
        prompt = seg.to_prompt(separator="|||")
        assert "|||" in prompt

    def test_to_prompt_empty(self):
        seg = ContextSegments()
        prompt = seg.to_prompt()
        # No parts → empty string
        assert prompt == ""

    def test_defaults_are_empty_strings(self):
        seg = ContextSegments()
        assert seg.system_prompt == ""
        assert seg.vector_memory == ""
        assert seg.medical_kb == ""
        assert seg.chat_history == ""


# ===== ContextBuilder =====


class TestContextBuilder:
    """Tests for the ContextBuilder class."""

    def _make_messages(self, n: int = 5) -> list:
        msgs = []
        for i in range(n):
            if i % 2 == 0:
                msgs.append(HumanMessage(content=f"question {i}"))
            else:
                msgs.append(AIMessage(content=f"answer {i}"))
        return msgs

    def test_init_defaults(self):
        cb = ContextBuilder()
        assert cb.max_history_chars == 3000
        assert cb.max_vector_results == 3
        assert cb.compress_old is True
        assert cb.keep_recent == 4

    def test_init_custom(self):
        cb = ContextBuilder(max_history_chars=5000, compress_old=False)
        assert cb.max_history_chars == 5000
        assert cb.compress_old is False

    # --- format_chat_history ---

    def test_format_chat_history_human_ai(self):
        cb = ContextBuilder(compress_old=False)
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        result = cb.format_chat_history(msgs)
        assert "User: hi" in result
        assert "Assistant: hello" in result

    def test_format_chat_history_skips_system(self):
        cb = ContextBuilder(compress_old=False)
        msgs = [SystemMessage(content="you are helpful"), HumanMessage(content="q")]
        result = cb.format_chat_history(msgs)
        assert "you are helpful" not in result
        assert "User: q" in result

    def test_format_chat_history_skips_empty(self):
        cb = ContextBuilder(compress_old=False)
        msgs = [HumanMessage(content=""), AIMessage(content="answer")]
        result = cb.format_chat_history(msgs)
        assert "answer" in result
        # empty message should not appear
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 1

    def test_format_chat_history_truncates(self):
        cb = ContextBuilder(max_history_chars=50, compress_old=False)
        msgs = [HumanMessage(content="x" * 100)]
        result = cb.format_chat_history(msgs)
        assert len(result) <= 50 + len("...\n")  # truncation adds prefix

    def test_format_chat_history_with_compression(self):
        """When compress_old=True, old messages get thinking tags compressed."""
        cb = ContextBuilder(compress_old=True, keep_recent=1)
        old = HumanMessage(content="<thinking>" + "a" * 300 + "</thinking>q1")
        recent = AIMessage(content="a1")
        result = cb.format_chat_history([old, recent])
        # Old thinking should be compressed
        assert "reasoning omitted" in result
        # Recent should be kept
        assert "a1" in result

    # --- format_vector_memory ---

    def test_format_vector_memory_dicts(self):
        cb = ContextBuilder()
        items = [
            {"text": "aspirin dosage", "score": 0.85},
            {"text": "pain relief", "score": 0.72},
        ]
        result = cb.format_vector_memory(items)
        assert "score=0.85" in result
        assert "aspirin dosage" in result
        assert "1." in result
        assert "2." in result

    def test_format_vector_memory_strings(self):
        cb = ContextBuilder()
        result = cb.format_vector_memory(["fact1", "fact2"])
        assert "1. fact1" in result
        assert "2. fact2" in result

    def test_format_vector_memory_empty(self):
        cb = ContextBuilder()
        assert cb.format_vector_memory([]) == ""

    def test_format_vector_memory_limits_results(self):
        cb = ContextBuilder(max_vector_results=2)
        items = [{"text": f"f{i}", "score": 0.5} for i in range(5)]
        result = cb.format_vector_memory(items)
        assert "3." not in result  # only 2 items shown

    def test_format_vector_memory_no_score(self):
        """Dict without 'score' key defaults to 0."""
        cb = ContextBuilder()
        items = [{"text": "item without score"}]
        result = cb.format_vector_memory(items)
        assert "score=0.00" in result

    # --- Fluent API (set_* pattern) ---

    def test_fluent_set_system(self):
        cb = ContextBuilder()
        ret = cb.set_system("sys prompt")
        assert ret is cb  # returns self for chaining
        assert cb._system_prompt == "sys prompt"

    def test_fluent_set_vector_memory_string(self):
        cb = ContextBuilder()
        ret = cb.set_vector_memory("memory text")
        assert ret is cb
        assert cb._vector_memory == "memory text"

    def test_fluent_set_vector_memory_list(self):
        cb = ContextBuilder()
        ret = cb.set_vector_memory([{"text": "fact", "score": 0.9}])
        assert ret is cb
        assert "fact" in cb._vector_memory

    def test_fluent_set_chat_history(self):
        cb = ContextBuilder()
        msgs = self._make_messages(3)
        ret = cb.set_chat_history(msgs)
        assert ret is cb

    def test_fluent_chaining(self):
        cb = ContextBuilder()
        result = (
            cb.set_system("sys")
              .set_vector_memory("mem")
              .set_chat_history([HumanMessage(content="q")])
        )
        assert result is cb

    # --- build() returns str ---

    def test_build_returns_str(self):
        cb = ContextBuilder()
        result = cb.build()
        assert isinstance(result, str)

    def test_build_empty(self):
        cb = ContextBuilder()
        result = cb.build()
        assert result == ""

    def test_build_with_system(self):
        cb = ContextBuilder()
        cb.set_system("you are a doctor")
        result = cb.build()
        assert "[SYSTEM]" in result
        assert "you are a doctor" in result

    def test_build_with_user_input(self):
        cb = ContextBuilder()
        result = cb.build(user_input="What is aspirin?")
        assert "[USER]" in result
        assert "What is aspirin?" in result

    def test_build_with_all_parts(self):
        cb = ContextBuilder()
        cb.set_system("sys").set_vector_memory("mem")
        cb.set_chat_history([HumanMessage(content="q1"), AIMessage(content="a1")])
        result = cb.build(user_input="new question")
        assert "[SYSTEM]" in result
        assert "[MEMORY]" in result
        assert "[HISTORY]" in result
        assert "[USER]" in result

    # --- build_decision_context / build_conversation_context / build_rag_context ---

    def test_build_decision_context_returns_segments(self):
        cb = ContextBuilder()
        seg = cb.build_decision_context([HumanMessage(content="q")], current_input="q")
        assert isinstance(seg, ContextSegments)
        assert seg.system_prompt == MedicalSystemPrompt.DECISION

    def test_build_conversation_context_returns_segments(self):
        cb = ContextBuilder()
        seg = cb.build_conversation_context([HumanMessage(content="q")])
        assert isinstance(seg, ContextSegments)
        assert seg.system_prompt == MedicalSystemPrompt.CONVERSATION

    def test_build_rag_context_returns_segments(self):
        cb = ContextBuilder()
        seg = cb.build_rag_context(
            [HumanMessage(content="q")],
            rag_results="rag data",
            vector_memory_text="mem",
        )
        assert isinstance(seg, ContextSegments)
        assert seg.system_prompt == MedicalSystemPrompt.RAG
        assert seg.medical_kb == "rag data"
        assert seg.vector_memory == "mem"

    # --- summarize_and_truncate ---

    def test_summarize_short_list(self):
        """When messages <= keep_recent, all go to recent."""
        cb = ContextBuilder()
        msgs = self._make_messages(3)
        old, recent = cb.summarize_and_truncate(msgs, keep_recent=6)
        assert old == []
        assert len(recent) == 3

    def test_summarize_exact_split(self):
        cb = ContextBuilder()
        msgs = self._make_messages(10)
        old, recent = cb.summarize_and_truncate(msgs, keep_recent=6)
        assert len(old) == 4
        assert len(recent) == 6

    def test_summarize_returns_tuple(self):
        cb = ContextBuilder()
        msgs = self._make_messages(5)
        result = cb.summarize_and_truncate(msgs, keep_recent=3)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # --- format_conversation_summary ---

    def test_format_summary_short(self):
        cb = ContextBuilder()
        result = cb.format_conversation_summary("User discussed headache symptoms.")
        assert "Previous conversation summary:" in result
        assert "headache" in result

    def test_format_summary_empty(self):
        cb = ContextBuilder()
        assert cb.format_conversation_summary("") == ""

    def test_format_summary_truncated(self):
        cb = ContextBuilder()
        long_summary = "x" * 1500
        result = cb.format_conversation_summary(long_summary)
        assert "[...]" in result
        assert len(result) < 1500


# ===== MedicalSystemPrompt =====


class TestMedicalSystemPrompt:
    """Verify that all expected prompt constants exist."""

    def test_decision_prompt_exists(self):
        assert len(MedicalSystemPrompt.DECISION) > 100

    def test_conversation_prompt_exists(self):
        assert len(MedicalSystemPrompt.CONVERSATION) > 100

    def test_rag_prompt_exists(self):
        assert len(MedicalSystemPrompt.RAG) > 50

    def test_prompts_are_strings(self):
        for attr in dir(MedicalSystemPrompt):
            if attr.isupper() and not attr.startswith("_"):
                val = getattr(MedicalSystemPrompt, attr)
                assert isinstance(val, str)
