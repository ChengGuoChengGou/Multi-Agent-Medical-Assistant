"""Tests for agents/memory/vector_memory.py."""
from unittest.mock import MagicMock, patch
import pytest
import sys


@pytest.fixture(autouse=True)
def mock_medical_vector_memory():
    """Mock agents.medical_vector_memory before VectorMemory imports it."""
    mock_mod = MagicMock()
    mock_mod._lazy_init.return_value = True
    mock_mod.add_memory.return_value = True
    mock_mod.search_memory.return_value = [{"text": "headache symptom", "score": 0.95}]
    mock_mod.get_all_memories.return_value = [{"text": "all memory", "score": 1.0}]
    mock_mod.collection_stats.return_value = {"points_count": 42, "collection": "medical"}

    with patch.dict(sys.modules, {"agents.medical_vector_memory": mock_mod}):
        yield mock_mod


class TestVectorMemoryInit:
    """Test VectorMemory initialization."""

    def test_init_success(self, mock_medical_vector_memory):
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm.available is True

    def test_init_lazy_fail(self, mock_medical_vector_memory):
        mock_medical_vector_memory._lazy_init.return_value = False
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm.available is False

    def test_init_exception(self, mock_medical_vector_memory):
        mock_medical_vector_memory._lazy_init.side_effect = RuntimeError("no qdrant")
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm.available is False


class TestVectorMemoryAdd:
    """Test VectorMemory.add."""

    def test_add_success(self, mock_medical_vector_memory):
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        result = vm.add("patient has fever", {"user_id": "p1"})
        assert result is True
        mock_medical_vector_memory.add_memory.assert_called_with("patient has fever", {"user_id": "p1"})

    def test_add_not_available(self, mock_medical_vector_memory):
        mock_medical_vector_memory._lazy_init.return_value = False
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm.add("test") is False

    def test_add_exception(self, mock_medical_vector_memory):
        mock_medical_vector_memory.add_memory.side_effect = RuntimeError("fail")
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm.add("test") is False


class TestVectorMemorySearch:
    """Test VectorMemory.search — returns list[tuple[str, float]]."""

    def test_search_success(self, mock_medical_vector_memory):
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        results = vm.search("headache")
        assert len(results) == 1
        assert isinstance(results[0], tuple)
        assert results[0][0] == "headache symptom"
        assert results[0][1] == 0.95

    def test_search_threshold_filter(self, mock_medical_vector_memory):
        mock_medical_vector_memory.search_memory.return_value = [
            {"text": "high", "score": 0.9},
            {"text": "low", "score": 0.1},
        ]
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        results = vm.search("test", score_threshold=0.5)
        assert len(results) == 1
        assert results[0][0] == "high"

    def test_search_not_available(self, mock_medical_vector_memory):
        mock_medical_vector_memory._lazy_init.return_value = False
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm.search("test") == []

    def test_search_exception(self, mock_medical_vector_memory):
        mock_medical_vector_memory.search_memory.side_effect = RuntimeError("fail")
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm.search("test") == []


class TestVectorMemoryGetAll:
    """Test VectorMemory.get_all."""

    def test_get_all_success(self, mock_medical_vector_memory):
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        result = vm.get_all()
        assert len(result) == 1
        assert result[0]["text"] == "all memory"

    def test_get_all_not_available(self, mock_medical_vector_memory):
        mock_medical_vector_memory._lazy_init.return_value = False
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm.get_all() == []


class TestVectorMemoryStats:
    """Test VectorMemory.get_stats."""

    def test_get_stats_success(self, mock_medical_vector_memory):
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        stats = vm.get_stats()
        assert stats["available"] is True
        assert stats["points_count"] == 42

    def test_get_stats_not_available(self, mock_medical_vector_memory):
        mock_medical_vector_memory._lazy_init.return_value = False
        from agents.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        stats = vm.get_stats()
        assert stats["available"] is False


class TestVectorMemorySingleton:
    """Test get_vector_memory singleton."""

    def test_singleton(self, mock_medical_vector_memory):
        import agents.memory.vector_memory as mod
        mod._vector_memory = None
        from agents.memory.vector_memory import get_vector_memory
        m1 = get_vector_memory()
        m2 = get_vector_memory()
        assert m1 is m2
        mod._vector_memory = None
