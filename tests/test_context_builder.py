"""Tests for agents/context_builder.py — Medical Context Assembly (Phase 58d)

Covers:
- MedicalSystemPrompt: constants exist
- compress_history_tags: thinking/tool_result compression, keep_recent, truncation
- ContextSegments: to_prompt, empty segments, separator
- ContextBuilder: fluent API, format_chat_history, format_vector_memory,
  build_decision/conversation/rag_context, summarize_and_truncate,
  format_conversation_summary, max_history_chars limit
"""

import sys
import types

import pytest

# Remove conftest-injected mock if present
for mod_key in list(sys.modules.keys()):
    if mod_key in ("agents.context_builder",) or mod_key.startswith("agents.context_builder."):
        del sys.modules[mod_key]

from agents.context_builder import (
    ContextBuilder,
    ContextSegments,
    MedicalSystemPrompt,
    compress_history_tags,
)

# ---------------------------------------------------------------------------
# Helpers: lightweight message objects (avoid importing langchain_core directly)
# ---------------------------------------------------------------------------

class FakeMessage:
    """Mimics langchain_core message with type + content."""
    def __init__(self, role: str = "human", content: str = "", **kwargs):
        self.type = role
        self.content = content


def _human(text): return FakeMessage("human", text)
def _ai(text): return FakeMessage("ai", text)
def _system(text): return FakeMessage("system", text)


# =========================================================================
# MedicalSystemPrompt
# =========================================================================

class TestMedicalSystemPrompt:
    def test_decision_prompt_exists(self):
        assert isinstance(MedicalSystemPrompt.DECISION, str)
        assert "triage" in MedicalSystemPrompt.DECISION.lower()
        assert "JSON" in MedicalSystemPrompt.DECISION

    def test_conversation_prompt_exists(self):
        assert isinstance(MedicalSystemPrompt.CONVERSATION, str)
        assert "medical assistant" in MedicalSystemPrompt.CONVERSATION.lower()

    def test_rag_prompt_exists(self):
        assert isinstance(MedicalSystemPrompt.RAG, str)
        assert "context" in MedicalSystemPrompt.RAG.lower()


# =========================================================================
# compress_history_tags
# =========================================================================

class TestCompressHistoryTags:
    def test_empty_messages(self):
        assert compress_history_tags([]) == []

    def test_short_thinking_not_compressed(self):
        short_think = _ai("Hello <thinking>quick thought</thinking> world")
        result = compress_history_tags([short_think], max_tag_len=200, keep_recent=0)
        assert result[0].content == short_think.content

    def test_long_thinking_compressed(self):
        long_inner = "x" * 500
        msg = _ai(f"<thinking>{long_inner}</thinking> rest")
        result = compress_history_tags([msg], max_tag_len=200, keep_recent=0)
        assert "500 chars reasoning omitted" in result[0].content
        assert "rest" in result[0].content

    def test_long_tool_result_compressed(self):
        long_inner = "y" * 400
        msg = _ai(f"<tool_result>{long_inner}</tool_result> after")
        result = compress_history_tags([msg], max_tag_len=200, keep_recent=0)
        assert "400 chars tool output omitted" in result[0].content

    def test_keep_recent_preserves_last_n(self):
        msgs = [_human(f"msg{i}") for i in range(8)]
        result = compress_history_tags(msgs, keep_recent=4)
        # Last 4 messages should be identical objects (same reference)
        for i in range(4):
            assert result[4 + i] is msgs[4 + i]

    def test_old_messages_compressed(self):
        long_content = "A" * 1500
        old_msg = _human(long_content)
        result = compress_history_tags([old_msg, _human("recent1"), _human("recent2"), _human("recent3"), _human("recent4")], keep_recent=4)
        # Old message should be truncated (>1000 chars -> 950 + [...])
        assert "A" * 950 in result[0].content
        assert "[...]" in result[0].content

    def test_non_string_content_passthrough(self):
        msg = FakeMessage("human", None)
        result = compress_history_tags([msg], keep_recent=0)
        assert result[0] is msg

    def test_multiple_thinking_blocks(self):
        inner = "z" * 300
        msg = _ai(f"<thinking>{inner}</thinking> middle <thinking>{inner}</thinking>")
        result = compress_history_tags([msg], max_tag_len=200, keep_recent=0)
        # Both blocks compressed
        assert result[0].content.count("chars reasoning omitted") == 2

    def test_does_not_mutate_input(self):
        original = _ai("<thinking>" + "a" * 500 + "</thinking>")
        msgs = [original]
        compress_history_tags(msgs, keep_recent=0)
        assert original.content == "<thinking>" + "a" * 500 + "</thinking>"


# =========================================================================
# ContextSegments
# =========================================================================

class TestContextSegments:
    def test_defaults_empty(self):
        seg = ContextSegments()
        assert seg.system_prompt == ""
        assert seg.vector_memory == ""
        assert seg.medical_kb == ""
        assert seg.chat_history == ""
        assert seg.rag_results == ""

    def test_to_prompt_all_empty(self):
        seg = ContextSegments()
        assert seg.to_prompt() == ""

    def test_to_prompt_single_segment(self):
        seg = ContextSegments(system_prompt="You are a doctor.")
        result = seg.to_prompt()
        assert "[System Instructions]" in result
        assert "You are a doctor." in result

    def test_to_prompt_multiple_segments(self):
        seg = ContextSegments(
            system_prompt="sys",
            vector_memory="vmem",
            medical_kb="kb",
            chat_history="hist",
            rag_results="rag",
        )
        result = seg.to_prompt()
        assert "[System Instructions]" in result
        assert "[Relevant Medical Memory]" in result
        assert "[Medical Knowledge Base]" in result
        assert "[Conversation History]" in result
        assert "[Retrieved Information]" in result
        assert "---" in result  # default separator

    def test_to_prompt_custom_separator(self):
        seg = ContextSegments(system_prompt="s1", vector_memory="s2")
        result = seg.to_prompt(separator="|||")
        assert "|||" in result
        assert "---" not in result


# =========================================================================
# ContextBuilder — constructor + defaults
# =========================================================================

class TestContextBuilderInit:
    def test_defaults(self):
        builder = ContextBuilder()
        assert builder.max_history_chars == 3000
        assert builder.max_vector_results == 3
        assert builder.compress_old is True
        assert builder.keep_recent == 4

    def test_custom_params(self):
        builder = ContextBuilder(max_history_chars=500, max_vector_results=5, compress_old=False, keep_recent=2)
        assert builder.max_history_chars == 500
        assert builder.max_vector_results == 5
        assert builder.compress_old is False
        assert builder.keep_recent == 2


# =========================================================================
# ContextBuilder — Fluent API
# =========================================================================

class TestContextBuilderFluent:
    def test_set_system_returns_self(self):
        builder = ContextBuilder()
        result = builder.set_system("test")
        assert result is builder

    def test_set_vector_memory_string(self):
        builder = ContextBuilder()
        builder.set_vector_memory("raw memory text")
        assert builder._vector_memory == "raw memory text"

    def test_set_vector_memory_list(self):
        builder = ContextBuilder()
        builder.set_vector_memory([{"text": "fact1", "score": 0.9}, {"text": "fact2", "score": 0.7}])
        assert "fact1" in builder._vector_memory
        assert "0.90" in builder._vector_memory

    def test_set_chat_history_returns_self(self):
        builder = ContextBuilder()
        result = builder.set_chat_history([_human("hi")])
        assert result is builder

    def test_set_chat_history_max_messages(self):
        builder = ContextBuilder(compress_old=False)
        msgs = [_human(f"m{i}") for i in range(10)]
        builder.set_chat_history(msgs, max_messages=3)
        # Only last 3 messages should appear
        assert "m7" in builder._chat_history
        assert "m8" in builder._chat_history
        assert "m9" in builder._chat_history
        assert "m0" not in builder._chat_history


# =========================================================================
# ContextBuilder — build()
# =========================================================================

class TestContextBuilderBuild:
    def test_build_empty(self):
        builder = ContextBuilder()
        assert builder.build() == ""

    def test_build_with_system(self):
        builder = ContextBuilder()
        builder.set_system("doctor prompt")
        result = builder.build()
        assert "[SYSTEM]" in result
        assert "doctor prompt" in result

    def test_build_with_user_input(self):
        builder = ContextBuilder()
        result = builder.build(user_input="What is diabetes?")
        assert "[USER]" in result
        assert "What is diabetes?" in result

    def test_build_all_sections(self):
        builder = ContextBuilder()
        builder.set_system("sys").set_vector_memory("vmem").set_chat_history([_human("hello")])
        result = builder.build(user_input="query")
        assert "[SYSTEM]" in result
        assert "[MEMORY]" in result
        assert "[HISTORY]" in result
        assert "[USER]" in result


# =========================================================================
# ContextBuilder — format_chat_history
# =========================================================================

class TestFormatChatHistory:
    def test_human_ai_messages(self):
        builder = ContextBuilder(compress_old=False)
        msgs = [_human("What is X?"), _ai("X is a condition...")]
        result = builder.format_chat_history(msgs)
        assert "User: What is X?" in result
        assert "Assistant: X is a condition..." in result

    def test_system_messages_skipped(self):
        builder = ContextBuilder(compress_old=False)
        msgs = [_system("You are a doctor"), _human("hi")]
        result = builder.format_chat_history(msgs)
        assert "You are a doctor" not in result
        assert "User: hi" in result

    def test_max_history_chars_truncation(self):
        builder = ContextBuilder(max_history_chars=50, compress_old=False)
        msgs = [_human("A" * 100)]
        result = builder.format_chat_history(msgs)
        assert result.startswith("...")
        assert len(result) <= 50 + 4  # "...\n" prefix

    def test_empty_messages(self):
        builder = ContextBuilder()
        assert builder.format_chat_history([]) == ""


# =========================================================================
# ContextBuilder — format_vector_memory
# =========================================================================

class TestFormatVectorMemory:
    def test_empty(self):
        builder = ContextBuilder()
        assert builder.format_vector_memory([]) == ""

    def test_dict_items(self):
        builder = ContextBuilder(max_vector_results=5)
        items = [
            {"text": "Diabetes is...", "score": 0.85},
            {"text": "Hypertension...", "score": 0.72},
        ]
        result = builder.format_vector_memory(items)
        assert "1. [score=0.85] Diabetes is..." in result
        assert "2. [score=0.72] Hypertension..." in result

    def test_string_items(self):
        builder = ContextBuilder()
        result = builder.format_vector_memory(["fact A", "fact B"])
        assert "1. fact A" in result
        assert "2. fact B" in result

    def test_max_results_limit(self):
        builder = ContextBuilder(max_vector_results=2)
        items = [{"text": f"item{i}", "score": 0.5} for i in range(5)]
        result = builder.format_vector_memory(items)
        assert "item0" in result
        assert "item1" in result
        assert "item2" not in result  # capped at 2


# =========================================================================
# ContextBuilder — build_*_context methods
# =========================================================================

class TestBuildContextMethods:
    def test_build_decision_context(self):
        builder = ContextBuilder(compress_old=False)
        seg = builder.build_decision_context([_human("hi")])
        assert isinstance(seg, ContextSegments)
        assert "triage" in seg.system_prompt.lower()
        assert "User: hi" in seg.chat_history

    def test_build_conversation_context(self):
        builder = ContextBuilder(compress_old=False)
        seg = builder.build_conversation_context(
            [_human("hello"), _ai("hi there")],
            vector_memory_text="some memory",
        )
        assert "medical assistant" in seg.system_prompt.lower()
        assert "some memory" == seg.vector_memory
        assert "User: hello" in seg.chat_history

    def test_build_rag_context(self):
        builder = ContextBuilder(compress_old=False)
        seg = builder.build_rag_context(
            [_human("explain X")],
            rag_results="RAG output here",
        )
        assert "context" in seg.system_prompt.lower()
        assert seg.medical_kb == "RAG output here"


# =========================================================================
# ContextBuilder — summarize_and_truncate + format_conversation_summary
# =========================================================================

class TestSummarizeAndTruncate:
    def test_short_messages_returns_all_as_recent(self):
        builder = ContextBuilder()
        msgs = [_human("a"), _ai("b"), _human("c")]
        old, recent = builder.summarize_and_truncate(msgs, keep_recent=6)
        assert old == []
        assert recent == msgs

    def test_long_messages_split(self):
        builder = ContextBuilder()
        msgs = [_human(f"m{i}") for i in range(10)]
        old, recent = builder.summarize_and_truncate(msgs, keep_recent=4)
        assert len(old) == 6
        assert len(recent) == 4

    def test_format_conversation_summary_empty(self):
        builder = ContextBuilder()
        assert builder.format_conversation_summary("") == ""

    def test_format_conversation_summary_normal(self):
        builder = ContextBuilder()
        result = builder.format_conversation_summary("User asked about diabetes.")
        assert "Previous conversation summary:" in result
        assert "User asked about diabetes." in result

    def test_format_conversation_summary_truncated(self):
        builder = ContextBuilder()
        long_text = "S" * 1500
        result = builder.format_conversation_summary(long_text)
        assert len(result) <= 1000 + 40  # header + truncated text
        assert "[...]" in result
