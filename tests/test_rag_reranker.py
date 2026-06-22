"""Tests for agents.rag_agent.reranker - Reranker class."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Unmock rag_agent from top-level conftest ────────────────────────
_MOCKED = [k for k in list(sys.modules) if k == "agents.rag_agent" or k.startswith("agents.rag_agent.")]
for _k in _MOCKED:
    del sys.modules[_k]

# Patch sentence_transformers before import
_mock_st = MagicMock()
sys.modules.setdefault("sentence_transformers", _mock_st)

from agents.rag_agent.reranker import Reranker


def _make_config(reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_top_k=5):
    """Create a mock Config with rag.reranker_model and rag.reranker_top_k."""
    config = MagicMock()
    config.rag.reranker_model = reranker_model
    config.rag.reranker_top_k = reranker_top_k
    return config


@pytest.fixture
def reranker():
    """Create Reranker with mocked CrossEncoder."""
    config = _make_config()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.3, 0.7]
    with patch("agents.rag_agent.reranker.CrossEncoder", return_value=mock_model):
        r = Reranker(config)
        r.model = mock_model  # Ensure model is the mock directly
        r._mock_model = mock_model
        yield r


# ============================================================
# __init__
# ============================================================
class TestRerankerInit:
    """Tests for Reranker initialization."""

    def test_model_loaded(self, reranker):
        assert reranker.model is not None

    def test_top_k_from_config(self, reranker):
        assert reranker.top_k == 5

    def test_model_name_stored(self, reranker):
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# rerank
# ============================================================
class TestRerank:
    """Tests for Reranker.rerank(query, documents, parsed_content_dir)."""

    def test_empty_documents_returns_empty_list(self, reranker):
        result = reranker.rerank("test query", [], "/parsed")
        assert result == []

    def test_rerank_dict_documents(self, reranker):
        """Dict documents should be reranked and return (list[dict], list[str])."""
        docs = [
            {"content": "Diabetes treatment options", "score": 0.8, "source": "doc1.md"},
            {"content": "Hypertension management", "score": 0.6, "source": "doc2.md"},
            {"content": "Common cold symptoms", "score": 0.5, "source": "doc3.md"},
        ]
        result = reranker.rerank("diabetes treatment", docs, "/parsed")
        # Returns (reranked_docs, picture_paths) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        reranked_docs, picture_paths = result
        assert isinstance(reranked_docs, list)
        assert isinstance(picture_paths, list)
        # All docs should have rerank_score and combined_score
        for doc in reranked_docs:
            assert "rerank_score" in doc
            assert "combined_score" in doc

    def test_rerank_string_documents(self, reranker):
        """String documents should be converted to dicts and reranked."""
        docs = ["Diabetes info", "Hypertension info", "Flu symptoms"]
        result = reranker.rerank("diabetes", docs, "/parsed")
        assert isinstance(result, tuple)
        reranked_docs, picture_paths = result
        assert len(reranked_docs) == 3
        # String docs should be converted with default score 1.0
        for doc in reranked_docs:
            assert "content" in doc
            assert "rerank_score" in doc

    def test_rerank_preserves_content(self, reranker):
        """Original content should be preserved after reranking."""
        docs = [
            {"content": "Important medical info", "score": 0.9, "source": "doc.md"},
        ]
        reranker._mock_model.predict.return_value = [0.95]
        result = reranker.rerank("query", docs, "/parsed")
        reranked_docs = result[0]
        assert any("Important medical info" in d["content"] for d in reranked_docs)

    def test_rerank_sorted_by_combined_score(self, reranker):
        """Documents should be sorted by combined_score descending."""
        docs = [
            {"content": "Low relevance", "score": 0.2, "source": "a.md"},
            {"content": "High relevance", "score": 0.95, "source": "b.md"},
            {"content": "Medium relevance", "score": 0.5, "source": "c.md"},
        ]
        result = reranker.rerank("query", docs, "/parsed")
        reranked_docs = result[0]
        scores = [d["combined_score"] for d in reranked_docs]
        assert scores == sorted(scores, reverse=True)

    def test_picture_counter_extraction(self, reranker):
        """picture_counter_N references should be extracted into picture_paths."""
        docs = [
            {"content": "See picture_counter_0 for the X-ray", "score": 0.8, "source": "radiology.pdf"},
        ]
        reranker._mock_model.predict.return_value = [0.85]
        result = reranker.rerank("chest x-ray", docs, "/parsed")
        reranked_docs, picture_paths = result
        assert len(picture_paths) > 0
        assert "radiology" in picture_paths[0]
        assert "picture-0.png" in picture_paths[0]

    def test_top_k_limits_results(self, reranker):
        """Results should be limited to top_k (5 from fixture)."""
        docs = [
            {"content": f"Doc {i}", "score": 0.5, "source": f"d{i}.md"}
            for i in range(10)
        ]
        reranker._mock_model.predict.return_value = [float(i) for i in range(10)]
        result = reranker.rerank("query", docs, "/parsed")
        reranked_docs = result[0]
        assert len(reranked_docs) <= reranker.top_k

    def test_rerank_failure_returns_original(self, reranker):
        """On exception, should fallback to original documents."""
        reranker._mock_model.predict.side_effect = RuntimeError("Model error")
        docs = [
            {"content": "Test", "score": 0.5, "source": "s.md"},
        ]
        result = reranker.rerank("query", docs, "/parsed")
        # Fallback returns just the document list (not tuple)
        assert isinstance(result, list)
