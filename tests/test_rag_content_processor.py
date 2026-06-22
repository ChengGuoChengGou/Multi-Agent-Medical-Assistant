"""Tests for agents.rag_agent.content_processor - ContentProcessor pure logic."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# --- Undo conftest.py's global mock of agents.rag_agent ---
_MOCKED_KEYS = [k for k in sys.modules if k == "agents.rag_agent" or k.startswith("agents.rag_agent.")]
for k in _MOCKED_KEYS:
    del sys.modules[k]

# --- Patch heavy deps before import ---
for mod_name in [
    "docling", "docling.document_converter", "docling.datamodel",
    "docling.datamodel.base_models", "docling.datamodel.pipeline_options",
    "PyPDF2",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from agents.rag_agent.content_processor import ContentProcessor


@pytest.fixture
def processor():
    """Create ContentProcessor with mocked config."""
    config = MagicMock()
    config.rag.summarizer_model = MagicMock()
    config.rag.chunker_model = MagicMock()
    return ContentProcessor(config)


# ============================================================
# _replace_occurrences
# ============================================================
class TestReplaceOccurrences:
    """Tests for _replace_occurrences(text, target, replacements)."""

    def test_single_replacement(self, processor):
        text = "Before {img} After"
        result = processor._replace_occurrences(text, "{img}", ["photo of chest x-ray"])
        assert "photo of chest x-ray" in result
        assert "picture_counter_0" in result
        assert "{img}" not in result

    def test_multiple_replacements(self, processor):
        text = "{img} and {img}"
        result = processor._replace_occurrences(text, "{img}", ["first", "second"])
        assert "picture_counter_0 first" in result
        assert "picture_counter_1 second" in result
        assert "{img}" not in result

    def test_non_informative_removed(self, processor):
        """Non-informative replacements should be removed (empty string)."""
        text = "See {img} here"
        result = processor._replace_occurrences(text, "{img}", ["non-informative"])
        assert "picture_counter" not in result
        assert "{img}" not in result
        assert "See  here" in result

    def test_no_placeholder_text_unchanged(self, processor):
        text = "No placeholders here"
        result = processor._replace_occurrences(text, "{img}", ["replacement"])
        assert result == text

    def test_empty_replacements_returns_original(self, processor):
        text = "Has {img} placeholder"
        result = processor._replace_occurrences(text, "{img}", [])
        assert result == text

    def test_fewer_replacements_than_placeholders(self, processor):
        """If fewer replacements than placeholders, remaining placeholders stay."""
        text = "{img} and {img}"
        result = processor._replace_occurrences(text, "{img}", ["only_one"])
        assert "picture_counter_0 only_one" in result
        assert "{img}" in result  # second placeholder remains


# ============================================================
# _split_text_by_llm_suggestions
# ============================================================
class TestSplitTextByLLMSuggestions:
    """Tests for _split_text_by_llm_suggestions(chunked_text, llm_response)."""

    def test_no_split_response_returns_whole_text(self, processor):
        chunked_text = "<|start_chunk_0|>\nHello world\n<|end_chunk_0|>"
        # No "split_after:" in LLM response => return [chunked_text]
        result = processor._split_text_by_llm_suggestions(chunked_text, "no splits suggested")
        assert len(result) == 1
        assert result[0] == chunked_text

    def test_single_split_point(self, processor):
        chunked_text = (
            "<|start_chunk_0|>\nAAA\n<|end_chunk_0|>\n"
            "<|start_chunk_1|>\nBBB\n<|end_chunk_1|>\n"
            "<|start_chunk_2|>\nCCC\n<|end_chunk_2|>\n"
        )
        llm_response = "split_after: 1"
        result = processor._split_text_by_llm_suggestions(chunked_text, llm_response)
        # split_after: 1 means append after chunk 1 → section0=[chunk0,chunk1], section1=[chunk2]
        assert len(result) == 2
        assert "AAA" in result[0]
        assert "BBB" in result[0]
        assert "CCC" in result[1]

    def test_multiple_split_points(self, processor):
        chunked_text = (
            "<|start_chunk_0|>\nAAA\n<|end_chunk_0|>\n"
            "<|start_chunk_1|>\nBBB\n<|end_chunk_1|>\n"
            "<|start_chunk_2|>\nCCC\n<|end_chunk_2|>\n"
        )
        llm_response = "split_after: 0, 1"
        result = processor._split_text_by_llm_suggestions(chunked_text, llm_response)
        # split after 0 => [chunk0], split after 1 => [chunk1], remainder => [chunk2]
        assert len(result) == 3
        assert "AAA" in result[0]
        assert "BBB" in result[1]
        assert "CCC" in result[2]

    def test_empty_response_returns_whole(self, processor):
        chunked_text = "<|start_chunk_0|>\nText\n<|end_chunk_0|>"
        result = processor._split_text_by_llm_suggestions(chunked_text, "")
        assert len(result) == 1


