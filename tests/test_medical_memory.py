"""Tests for agents/memory/medical_memory.py."""
from unittest.mock import MagicMock, patch
import pytest
from agents.memory.medical_memory import MedicalMemory, get_medical_memory


class TestMedicalMemoryInit:
    """Test MedicalMemory initialization."""

    @patch("agents.memory_module.get_memory_store")
    def test_init_success(self, mock_gms):
        mock_store = MagicMock()
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)
        assert mod.available is True
        assert mod._store is mock_store

    @patch("agents.memory_module.get_memory_store", side_effect=RuntimeError("no qdrant"))
    def test_init_failure(self, mock_gms):
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)
        assert mod.available is False
        assert mod._store is None


class TestMedicalMemoryRemember:
    """Test remember_medical."""

    @patch("agents.memory_module.get_memory_store")
    def test_remember_success(self, mock_gms):
        mock_store = MagicMock()
        mock_store.remember.return_value = True
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)

        result = mod.remember_medical("patient1", "has fever", category="symptom")
        assert result is True
        mock_store.remember.assert_called_once()
        call_args = mock_store.remember.call_args
        assert call_args[0][0] == "patient1"
        assert call_args[0][1] == "has fever"
        assert call_args[0][2]["category"] == "symptom"
        assert call_args[0][2]["medical"] is True

    def test_remember_not_available(self):
        mod = MedicalMemory.__new__(MedicalMemory)
        mod._store = None
        mod._available = False
        assert mod.remember_medical("u1", "test") is False

    @patch("agents.memory_module.get_memory_store")
    def test_remember_exception(self, mock_gms):
        mock_store = MagicMock()
        mock_store.remember.side_effect = RuntimeError("fail")
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)
        assert mod.remember_medical("u1", "test") is False


class TestMedicalMemoryRecall:
    """Test recall_medical."""

    @patch("agents.memory_module.get_memory_store")
    def test_recall_basic(self, mock_gms):
        mock_store = MagicMock()
        mock_store.recall.return_value = "patient has fever"
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)

        result = mod.recall_medical("patient1", "fever symptoms")
        assert result == "patient has fever"
        mock_store.recall.assert_called_once_with("patient1", "fever symptoms", limit=5)

    @patch("agents.memory_module.get_memory_store")
    def test_recall_with_category(self, mock_gms):
        mock_store = MagicMock()
        mock_store.recall.return_value = "allergic to penicillin"
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)

        result = mod.recall_medical("p1", "allergies", category="allergy")
        assert result == "allergic to penicillin"
        call_args = mock_store.recall.call_args
        assert call_args[0][1] == "[allergy] allergies"

    def test_recall_not_available(self):
        mod = MedicalMemory.__new__(MedicalMemory)
        mod._store = None
        mod._available = False
        assert mod.recall_medical("u1", "test") == ""


class TestMedicalMemoryHistory:
    """Test get_patient_history."""

    @patch("agents.memory_module.get_memory_store")
    def test_history_success(self, mock_gms):
        mock_store = MagicMock()
        mock_store.get_history.return_value = [{"text": "fever", "timestamp": "2025-01-01"}]
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)

        result = mod.get_patient_history("p1")
        assert len(result) == 1
        assert result[0]["text"] == "fever"

    def test_history_not_available(self):
        mod = MedicalMemory.__new__(MedicalMemory)
        mod._store = None
        mod._available = False
        assert mod.get_patient_history("u1") == []


class TestMedicalMemoryHelpers:
    """Test check_allergies, check_medications, forget_patient."""

    @patch("agents.memory_module.get_memory_store")
    def test_check_allergies(self, mock_gms):
        mock_store = MagicMock()
        mock_store.recall.return_value = "penicillin allergy"
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)

        result = mod.check_allergies("p1")
        assert "penicillin" in result

    @patch("agents.memory_module.get_memory_store")
    def test_check_medications(self, mock_gms):
        mock_store = MagicMock()
        mock_store.recall.return_value = "aspirin 100mg"
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)

        result = mod.check_medications("p1")
        assert "aspirin" in result

    @patch("agents.memory_module.get_memory_store")
    def test_forget_patient_success(self, mock_gms):
        mock_store = MagicMock()
        mock_store.forget.return_value = True
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)
        assert mod.forget_patient("p1") is True

    def test_forget_patient_not_available(self):
        mod = MedicalMemory.__new__(MedicalMemory)
        mod._store = None
        mod._available = False
        assert mod.forget_patient("p1") is False

    @patch("agents.memory_module.get_memory_store")
    def test_forget_patient_exception(self, mock_gms):
        mock_store = MagicMock()
        mock_store.forget.side_effect = RuntimeError("fail")
        mock_gms.return_value = mock_store
        mod = MedicalMemory.__new__(MedicalMemory)
        MedicalMemory.__init__(mod)
        assert mod.forget_patient("p1") is False


class TestMedicalMemorySingleton:
    """Test get_medical_memory singleton."""

    @patch("agents.memory_module.get_memory_store")
    def test_singleton(self, mock_gms):
        import agents.memory.medical_memory as mod
        mod._medical_memory = None
        mock_gms.return_value = MagicMock()
        m1 = get_medical_memory()
        m2 = get_medical_memory()
        assert m1 is m2
        mod._medical_memory = None
