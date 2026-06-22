"""Tests for agents.rag_agent.hybrid_search - BM25Index & HybridSearch."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ── Unmock rag_agent from top-level conftest ────────────────────────
_MOCKED = [k for k in list(sys.modules) if k == "agents.rag_agent" or k.startswith("agents.rag_agent.")]
for _k in _MOCKED:
    del sys.modules[_k]

# Patch heavy deps
for mod_name in [
    "qdrant_client", "qdrant_client.models", "qdrant_client.http",
    "langchain_qdrant", "sentence_transformers",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from agents.rag_agent.hybrid_search import BM25Index, HybridSearch


# ============================================================
# BM25Index._tokenize
# ============================================================
class TestTokenize:
    """Tests for BM25Index._tokenize(text)."""

    @pytest.fixture
    def bm25(self):
        return BM25Index()

    def test_basic_english(self, bm25):
        tokens = bm25._tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_medical_compound_terms(self, bm25):
        tokens = bm25._tokenize("COVID-19 is caused by SARS-CoV-2")
        assert "covid-19" in tokens
        assert "sars-cov-2" in tokens

    def test_single_chars_filtered(self, bm25):
        """Single-char tokens (len<=1) are filtered out."""
        tokens = bm25._tokenize("I am a test")
        # "i" len=1 filtered, "am" kept, "a" len=1 filtered, "test" kept
        assert "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "test" in tokens

    def test_empty_string(self, bm25):
        assert bm25._tokenize("") == []

    def test_numbers_kept_if_multichar(self, bm25):
        """Numbers like '123' (len>1) are kept; '2' (len=1) is filtered."""
        tokens = bm25._tokenize("patient 42 age 2")
        assert "42" in tokens
        assert "2" not in tokens  # len=1, filtered

    def test_punctuation_split(self, bm25):
        tokens = bm25._tokenize("hello, world! test-case")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test-case" in tokens

    def test_mixed_case_lowered(self, bm25):
        tokens = bm25._tokenize("COVID covid")
        assert all(t == t.lower() for t in tokens)
        assert tokens.count("covid") == 2

    def test_pure_chinese_no_latin(self, bm25):
        """Pure Chinese has no alphanumeric tokens => empty list."""
        assert bm25._tokenize("你好世界") == []

    def test_mixed_chinese_english(self, bm25):
        tokens = bm25._tokenize("高血压 hypertension")
        assert "hypertension" in tokens


# ============================================================
# BM25Index (corpus-level)
# ============================================================
class TestBM25Index:
    """Tests for BM25Index build/search/add/remove."""

    @pytest.fixture
    def bm25(self):
        return BM25Index()

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(page_content="Diabetes is a chronic condition", metadata={"source": "doc1.md"}),
            Document(page_content="Hypertension causes high blood pressure", metadata={"source": "doc2.md"}),
            Document(page_content="Aspirin is used for pain relief", metadata={"source": "doc3.md"}),
        ]

    def test_build_and_search(self, bm25, sample_docs):
        bm25.build_index(sample_docs)
        results = bm25.search("diabetes treatment", top_k=3)
        assert len(results) > 0
        assert len(results[0]) == 2  # (Document, score)
        # diabetes doc should be top result
        assert "Diabetes" in results[0][0].page_content

    def test_build_sets_built_flag(self, bm25, sample_docs):
        assert bm25._built is False
        bm25.build_index(sample_docs)
        assert bm25._built is True
        assert len(bm25._documents) == 3

    def test_add_documents_extends(self, bm25, sample_docs):
        bm25.build_index(sample_docs)
        new_doc = Document(page_content="Metformin treats diabetes", metadata={"source": "doc4.md"})
        bm25.add_documents([new_doc])
        assert len(bm25._documents) == 4

    def test_remove_documents_by_source(self, bm25, sample_docs):
        bm25.build_index(sample_docs)
        removed = bm25.remove_documents_by_source("doc1.md")
        assert removed == 1
        assert len(bm25._documents) == 2

    def test_remove_nonexistent_source(self, bm25, sample_docs):
        bm25.build_index(sample_docs)
        removed = bm25.remove_documents_by_source("nonexistent.md")
        assert removed == 0

    def test_search_empty_index(self, bm25):
        """Searching empty index should return empty list."""
        results = bm25.search("test", top_k=5)
        assert results == []


# ============================================================
# HybridSearch._rrf_fusion
# ============================================================
class TestRRFFusion:
    """Tests for HybridSearch._rrf_fusion."""

    @pytest.fixture
    def hybrid(self):
        return HybridSearch(rrf_k=60)

    def test_basic_fusion(self, hybrid):
        d1 = Document(page_content="Doc about diabetes", metadata={"source": "a"})
        d2 = Document(page_content="Doc about hypertension", metadata={"source": "b"})
        vector_results = [(d1, 0.9), (d2, 0.8)]
        bm25_results = [(d2, 1.5), (d1, 1.0)]

        results = hybrid._rrf_fusion(vector_results, bm25_results, top_k=2)
        # Returns list of (Document, rrf_score, debug_info)
        assert len(results) == 2
        assert len(results[0]) == 3  # (doc, score, debug)
        # Both docs appear in both lists => both get RRF scores
        doc_contents = [r[0].page_content for r in results]
        assert any("diabetes" in c for c in doc_contents)

    def test_single_list_dedup(self, hybrid):
        d1 = Document(page_content="Same doc", metadata={"source": "x"})
        # Same doc in both lists => should get combined RRF score
        vector_results = [(d1, 0.95)]
        bm25_results = [(d1, 2.0)]

        results = hybrid._rrf_fusion(vector_results, bm25_results, top_k=5)
        assert len(results) == 1
        # RRF score should be 1/(60+1) + 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(results[0][1] - expected) < 1e-6

    def test_empty_results(self, hybrid):
        results = hybrid._rrf_fusion([], [], top_k=10)
        assert results == []

    def test_top_k_limit(self, hybrid):
        docs = [
            Document(page_content=f"Doc {i}", metadata={"source": f"s{i}"})
            for i in range(20)
        ]
        vector_results = [(d, 0.5) for d in docs]
        bm25_results = []

        results = hybrid._rrf_fusion(vector_results, bm25_results, top_k=5)
        assert len(results) == 5

    def test_debug_info_structure(self, hybrid):
        d1 = Document(page_content="Test", metadata={"source": "s"})
        results = hybrid._rrf_fusion([(d1, 0.9)], [(d1, 1.0)], top_k=1)
        debug = results[0][2]
        assert "rrf_score" in debug
        assert "vector_rank" in debug
        assert "bm25_rank" in debug

    def test_rrf_k_attribute(self, hybrid):
        """rrf_k should be the instance attribute (default 60)."""
        assert hybrid.rrf_k == 60


# ============================================================
# HybridSearch.search (async)
# ============================================================
class TestHybridSearchSearch:
    """Tests for HybridSearch.search (async method)."""

    def test_search_is_coroutine_function(self):
        """search should be an async method."""
        import inspect
        assert inspect.iscoroutinefunction(HybridSearch.search)

    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        """search returns list of dicts."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = []

        hybrid = HybridSearch(vectorstore=mock_vectorstore, rrf_k=60)
        hybrid.bm25_index.search = MagicMock(return_value=[])

        with patch.object(hybrid, "build_bm25_from_docstore"):
            results = await hybrid.search("test query", top_k=5)
        assert isinstance(results, list)
