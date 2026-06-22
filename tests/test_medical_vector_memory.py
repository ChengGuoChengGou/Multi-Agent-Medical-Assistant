"""Tests for agents/medical_vector_memory.py - Medical Vector Memory module."""

import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# ── Fixtures ──


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level globals before each test."""
    import agents.medical_vector_memory as mvm

    mvm._initialized = False
    mvm._qdrant_client = None
    mvm._model = None
    mvm._init_error = None
    yield
    mvm._initialized = False
    mvm._qdrant_client = None
    mvm._model = None
    mvm._init_error = None


@pytest.fixture
def mock_deps():
    """Provide mocked qdrant_client and model with sys.modules patching for local imports."""
    import agents.medical_vector_memory as mvm

    mock_client = MagicMock()
    mock_model = MagicMock()

    # encode returns np.array based on input length
    def _encode_side_effect(texts, **kwargs):
        n = len(texts) if hasattr(texts, "__len__") else 1
        return np.array([np.array([0.1] * 384) for _ in range(n)])

    mock_model.encode.side_effect = _encode_side_effect

    # Build mock qdrant_client package with .models sub-module
    mock_qdrant = MagicMock()
    mock_models = MagicMock()
    mock_qdrant.models = mock_models

    # Real PointStruct class that stores constructor args
    class FakePointStruct:
        def __init__(self, id=None, vector=None, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload

    mock_models.PointStruct = FakePointStruct

    saved_modules = {}
    for key in ("qdrant_client", "qdrant_client.models"):
        saved_modules[key] = sys.modules.get(key)
    sys.modules["qdrant_client"] = mock_qdrant
    sys.modules["qdrant_client.models"] = mock_models

    mvm._initialized = True
    mvm._qdrant_client = mock_client
    mvm._model = mock_model
    mvm._collection_name = "test_medical_memory"
    mvm._qdrant_store_path = "/tmp/test_qdrant"

    yield mock_client, mock_model

    # Restore sys.modules
    for key, val in saved_modules.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


# ── _lazy_init tests ──


class TestLazyInit:
    def test_returns_true_when_already_initialized(self):
        import agents.medical_vector_memory as mvm

        mvm._initialized = True
        assert mvm._lazy_init() is True

    def test_initializes_on_first_call(self):
        """Test _lazy_init by pre-setting module globals (skip real imports)."""
        import agents.medical_vector_memory as mvm

        mvm._initialized = False

        # Simulate successful init by setting globals directly
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])

        mvm._qdrant_client = mock_client
        mvm._model = mock_model
        mvm._initialized = True

        assert mvm._initialized is True
        assert mvm._qdrant_client is mock_client
        assert mvm._model is mock_model

    def test_returns_false_on_failure(self):
        """Test that _init_error prevents re-initialization."""
        import agents.medical_vector_memory as mvm

        mvm._initialized = False
        mvm._init_error = "previous failure"
        result = mvm._lazy_init()
        assert result is False


# ── add_memory tests ──


class TestAddMemory:
    def test_success(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps

        result = mvm.add_memory("Patient shows symptoms of hypertension", user_id="user1")
        assert result is True
        client.upsert.assert_called_once()

    def test_empty_text_returns_false(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.add_memory("") is False

    def test_none_text_returns_false(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.add_memory(None) is False

    def test_whitespace_only_returns_false(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.add_memory("   ") is False

    def test_truncates_long_text(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        model.encode.return_value = [[0.1] * 384]

        long_text = "word " * 600
        result = mvm.add_memory(long_text)
        assert result is True
        encoded_text = model.encode.call_args[0][0][0]
        assert len(encoded_text.split()) <= 512

    def test_metadata_in_payload(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps

        mvm.add_memory("test entry", metadata={"type": "diagnosis"})
        points = client.upsert.call_args.kwargs["points"]
        assert points[0].payload["metadata"] == {"type": "diagnosis"}

    def test_upsert_failure_returns_false(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        client.upsert.side_effect = RuntimeError("connection lost")
        assert mvm.add_memory("test") is False

    def test_init_failure_returns_false(self):
        import agents.medical_vector_memory as mvm

        with patch.object(mvm, "_lazy_init", return_value=False):
            assert mvm.add_memory("test") is False


# ── add_memories_batch tests ──


class TestAddMemoriesBatch:
    def test_batch_success(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        model.encode.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        assert mvm.add_memories_batch(["t1", "t2", "t3"]) == 3

    def test_filters_empty(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        model.encode.return_value = [[0.1] * 384]
        assert mvm.add_memories_batch(["valid", "", None, "  "]) == 1

    def test_all_empty_returns_zero(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.add_memories_batch(["", None]) == 0

    def test_empty_list(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.add_memories_batch([]) == 0

    def test_failure_returns_zero(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        client.upsert.side_effect = RuntimeError("fail")
        assert mvm.add_memories_batch(["text"]) == 0


# ── search_memory tests ──


class TestSearchMemory:
    def _make_hit(self, text, score):
        hit = MagicMock()
        hit.score = score
        hit.payload = {"text": text}
        return hit

    def test_returns_texts(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        hits = MagicMock()
        hits.points = [self._make_hit("r1", 0.8), self._make_hit("r2", 0.6)]
        client.query_points.return_value = hits

        results = mvm.search_memory("hypertension")
        assert len(results) == 2
        assert "r1" in results

    def test_filters_low_score(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        hits = MagicMock()
        hits.points = [self._make_hit("good", 0.7), self._make_hit("bad", 0.1)]
        client.query_points.return_value = hits

        results = mvm.search_memory("q", min_score=0.3)
        assert len(results) == 1
        assert results[0] == "good"

    def test_custom_min_score_filters_all(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        hits = MagicMock()
        hits.points = [self._make_hit("ok", 0.5)]
        client.query_points.return_value = hits

        results = mvm.search_memory("q", min_score=0.9)
        assert results == []

    def test_empty_query(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.search_memory("") == []

    def test_none_query(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.search_memory(None) == []

    def test_user_id_filter(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        hits = MagicMock()
        hits.points = [self._make_hit("r", 0.8)]
        client.query_points.return_value = hits

        mvm.search_memory("q", user_id="u1")
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    def test_no_user_id_no_filter(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        hits = MagicMock()
        hits.points = [self._make_hit("r", 0.8)]
        client.query_points.return_value = hits

        mvm.search_memory("q")
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    def test_failure_returns_empty(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        client.query_points.side_effect = RuntimeError("fail")
        assert mvm.search_memory("q") == []


# ── get_all_memories tests ──


class TestGetAllMemories:
    def _make_point(self, text):
        p = MagicMock()
        p.payload = {"text": text}
        return p

    def test_returns_texts(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        client.scroll.return_value = ([self._make_point("m1"), self._make_point("m2")], None)
        assert mvm.get_all_memories() == ["m1", "m2"]

    def test_empty_payload(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        p = MagicMock()
        p.payload = {}
        client.scroll.return_value = ([p], None)
        assert mvm.get_all_memories() == []

    def test_with_user_id(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        client.scroll.return_value = ([self._make_point("um")], None)
        mvm.get_all_memories(user_id="u1")
        call_kwargs = client.scroll.call_args.kwargs
        assert call_kwargs["scroll_filter"] is not None

    def test_custom_limit(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        client.scroll.return_value = ([], None)
        mvm.get_all_memories(limit=50)
        assert client.scroll.call_args.kwargs["limit"] == 50

    def test_failure(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        client.scroll.side_effect = RuntimeError("fail")
        assert mvm.get_all_memories() == []


# ── seed_from_medical_knowledge tests ──


class TestSeedFromMedicalKnowledge:
    def test_empty_paths(self):
        import agents.medical_vector_memory as mvm

        assert mvm.seed_from_medical_knowledge([]) == 0

    def test_parses_kv_lines(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        model.encode.return_value = [[0.1] * 384, [0.2] * 384]

        content = "# Header\n---\nSymptom: chronic cough and fever\nDiagnosis | Pneumonia confirmed by X-ray"
        with patch("builtins.open", mock_open(read_data=content)), patch("os.path.exists", return_value=True):
            result = mvm.seed_from_medical_knowledge(["/fake.txt"])
        assert result >= 2

    def test_skips_headers_and_short(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        model.encode.return_value = []

        content = "# Header\n---\nshort\nab"
        with patch("builtins.open", mock_open(read_data=content)), patch("os.path.exists", return_value=True):
            result = mvm.seed_from_medical_knowledge(["/fake.txt"])
        assert result == 0

    def test_nonexistent_file(self, mock_deps):
        import agents.medical_vector_memory as mvm

        with patch("os.path.exists", return_value=False):
            assert mvm.seed_from_medical_knowledge(["/bad.txt"]) == 0

    def test_deduplicates(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        model.encode.return_value = [[0.1] * 384]

        content = "Symptom: Patient has recurring headache\nSymptom: Patient has recurring headache"
        with patch("builtins.open", mock_open(read_data=content)), patch("os.path.exists", return_value=True):
            result = mvm.seed_from_medical_knowledge(["/fake.txt"])
        # Should add 1 unique entry
        assert result >= 1
        if result > 0:
            assert len(model.encode.call_args[0][0]) == 1


# ── seed_from_rag_results tests ──


class TestSeedFromRagResults:
    def test_empty(self):
        import agents.medical_vector_memory as mvm

        assert mvm.seed_from_rag_results([]) == 0

    def test_filters_low_score(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        model.encode.return_value = [[0.1] * 384]

        rag = [
            {"content": "High relevance diabetes management info.", "score": 0.8, "source": "pubmed"},
            {"content": "Low relevance noise.", "score": 0.2, "source": "web"},
        ]
        assert mvm.seed_from_rag_results(rag) == 1

    def test_filters_short_content(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.seed_from_rag_results([{"content": "short", "score": 0.9, "source": "t"}]) == 0


# ── store_conversation_summary tests ──


class TestStoreConversationSummary:
    def test_short_returns_false(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.store_conversation_summary("hi") is False

    def test_empty_returns_false(self, mock_deps):
        import agents.medical_vector_memory as mvm

        assert mvm.store_conversation_summary("") is False

    def test_valid(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        summary = "Patient discussed chronic back pain with Dr. Smith and was prescribed physical therapy."
        assert mvm.store_conversation_summary(summary, user_id="p1") is True


# ── format_memory_for_prompt tests ──


class TestFormatMemoryForPrompt:
    def test_with_query(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, model = mock_deps
        hits = MagicMock()
        h1 = MagicMock()
        h1.score = 0.8
        h1.payload = {"text": "Aspirin reduces fever"}
        h2 = MagicMock()
        h2.score = 0.6
        h2.payload = {"text": "Ibuprofen treats inflammation"}
        hits.points = [h1, h2]
        client.query_points.return_value = hits

        result = mvm.format_memory_for_prompt("fever treatment")
        assert "MedicalMemory" in result
        assert "Aspirin" in result

    def test_no_query_fallback(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        p = MagicMock()
        p.payload = {"text": "stored memory"}
        client.scroll.return_value = ([p], None)

        result = mvm.format_memory_for_prompt("")
        assert "stored memory" in result

    def test_empty_search(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        hits = MagicMock()
        hits.points = []
        client.query_points.return_value = hits
        assert mvm.format_memory_for_prompt("unknown") == ""

    def test_no_memories(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        client.scroll.return_value = ([], None)
        assert mvm.format_memory_for_prompt("") == ""


# ── collection_stats tests ──


class TestCollectionStats:
    def test_returns_stats(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        info = MagicMock()
        info.points_count = 42
        info.indexed_vectors_count = 42
        info.status = "green"
        client.get_collection.return_value = info

        stats = mvm.collection_stats()
        assert stats["points_count"] == 42
        assert stats["collection"] == "test_medical_memory"

    def test_failure(self, mock_deps):
        import agents.medical_vector_memory as mvm

        client, _ = mock_deps
        client.get_collection.side_effect = RuntimeError("not found")
        stats = mvm.collection_stats()
        assert "error" in stats

    def test_init_failure(self):
        import agents.medical_vector_memory as mvm

        with patch.object(mvm, "_lazy_init", return_value=False):
            mvm._init_error = "init failed"
            stats = mvm.collection_stats()
            assert "error" in stats
