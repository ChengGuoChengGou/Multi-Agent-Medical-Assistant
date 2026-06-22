"""
Tests for agents.rag_agent.query_expander (Phase 71)

QueryExpander wraps an LLM call to expand medical queries with synonyms.
Testable: expand_query, _generate_expansions (mock LLM).
"""

import sys
from unittest.mock import MagicMock

# ── Unmock rag_agent from top-level conftest ────────────────────────
_MOCKED = [k for k in list(sys.modules) if k == "agents.rag_agent" or k.startswith("agents.rag_agent.")]
for _k in _MOCKED:
    del sys.modules[_k]

import pytest

from agents.rag_agent.query_expander import QueryExpander


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.rag.llm = MagicMock()
    return cfg


@pytest.fixture
def expander(mock_config):
    return QueryExpander(mock_config)


# ── Tests ───────────────────────────────────────────────────────────

class TestQueryExpanderInit:
    def test_stores_config(self, expander, mock_config):
        assert expander.config is mock_config

    def test_model_set_from_config(self, expander, mock_config):
        assert expander.model is mock_config.rag.llm


class TestExpandQuery:
    def test_returns_dict_with_expanded_query(self, expander):
        mock_resp = MagicMock()
        mock_resp.content = "headache migraine tension cephalgia"
        expander.model.invoke.return_value = mock_resp
        result = expander.expand_query("What causes headaches?")
        assert isinstance(result, dict)
        assert "headache" in result["expanded_query"].lower()

    def test_calls_llm_once(self, expander):
        mock_resp = MagicMock()
        mock_resp.content = "expanded"
        expander.model.invoke.return_value = mock_resp
        expander.expand_query("test query")
        expander.model.invoke.assert_called_once()

    def test_prompt_contains_original_query(self, expander):
        mock_resp = MagicMock()
        mock_resp.content = "ok"
        expander.model.invoke.return_value = mock_resp
        expander.expand_query("diabetes mellitus")
        prompt_arg = expander.model.invoke.call_args[0][0]
        assert "diabetes mellitus" in prompt_arg

    def test_stores_original_query(self, expander):
        mock_resp = MagicMock()
        mock_resp.content = "expanded"
        expander.model.invoke.return_value = mock_resp
        result = expander.expand_query("fever")
        assert result["original_query"] == "fever"

    def test_llm_exception_propagates(self, expander):
        expander.model.invoke.side_effect = RuntimeError("LLM down")
        with pytest.raises(RuntimeError, match="LLM down"):
            expander.expand_query("test")

    def test_chinese_query(self, expander):
        mock_resp = MagicMock()
        mock_resp.content = "头痛 偏头痛 紧张性头痛"
        expander.model.invoke.return_value = mock_resp
        result = expander.expand_query("头痛的原因是什么?")
        assert "头痛" in result["expanded_query"]

    def test_long_query(self, expander):
        long_q = "What are the symptoms, diagnosis, and treatment options for " * 20
        mock_resp = MagicMock()
        mock_resp.content = "expanded"
        expander.model.invoke.return_value = mock_resp
        result = expander.expand_query(long_q)
        assert isinstance(result, dict)


class TestGenerateExpansions:
    """Test the internal _generate_expansions method (alias for LLM call)."""

    def test_returns_llm_output(self, expander):
        mock_resp = MagicMock()
        mock_resp.content = "fever pyrexia hyperthermia"
        expander.model.invoke.return_value = mock_resp
        result = expander._generate_expansions("fever")
        assert result.content == "fever pyrexia hyperthermia"

    def test_prompt_structure(self, expander):
        mock_resp = MagicMock()
        mock_resp.content = "expanded"
        expander.model.invoke.return_value = mock_resp
        expander._generate_expansions("test")
        prompt = expander.model.invoke.call_args[0][0]
        # Should contain instructions about expanding
        assert isinstance(prompt, str)
        assert len(prompt) > 20
