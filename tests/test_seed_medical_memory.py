"""
Tests for agents/seed_medical_memory.py

seed_medical_facts() with mocked add_memory/collection_stats.
"""
import pytest
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.seed_medical_memory import seed_medical_facts


class TestSeedMedicalFacts:
    """Tests for the seed_medical_facts function."""

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_seeds_all_facts(self, mock_add, mock_stats):
        """Should call add_memory for each fact (15 medical + 3 interaction = 18)."""
        mock_add.return_value = True
        mock_stats.return_value = {"count": 18}
        success, fail = seed_medical_facts()
        assert success == 18
        assert fail == 0
        assert mock_add.call_count == 18

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_returns_tuple(self, mock_add, mock_stats):
        mock_add.return_value = True
        mock_stats.return_value = {}
        result = seed_medical_facts()
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_add_memory_failure_counts_as_fail(self, mock_add, mock_stats):
        """When add_memory returns falsy, it counts as fail."""
        mock_add.return_value = None
        mock_stats.return_value = {}
        success, fail = seed_medical_facts()
        assert success == 0
        assert fail == 18

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_add_memory_exception_counts_as_fail(self, mock_add, mock_stats):
        """When add_memory raises, it catches and counts as fail."""
        mock_add.side_effect = RuntimeError("vector store down")
        mock_stats.return_value = {}
        success, fail = seed_medical_facts()
        assert success == 0
        assert fail == 18

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_mixed_success_failure(self, mock_add, mock_stats):
        """Some succeed, some fail."""
        mock_add.side_effect = [True, None, RuntimeError("err")] + [True] * 15
        mock_stats.return_value = {"count": 16}
        success, fail = seed_medical_facts()
        assert success == 16
        assert fail == 2

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_calls_collection_stats(self, mock_add, mock_stats):
        """Should call collection_stats after seeding."""
        mock_add.return_value = True
        mock_stats.return_value = {"count": 18}
        seed_medical_facts()
        mock_stats.assert_called_once()

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_add_memory_receives_text_and_metadata(self, mock_add, mock_stats):
        """Each add_memory call should have text and metadata kwargs."""
        mock_add.return_value = True
        mock_stats.return_value = {}
        seed_medical_facts()
        for call in mock_add.call_args_list:
            assert "text" in call.kwargs
            assert "metadata" in call.kwargs
            assert isinstance(call.kwargs["text"], str)
            assert isinstance(call.kwargs["metadata"], dict)

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_medical_fact_metadata_has_type(self, mock_add, mock_stats):
        """Medical facts should have metadata.type == 'medical_fact'."""
        mock_add.return_value = True
        mock_stats.return_value = {}
        seed_medical_facts()
        medical_calls = [c for c in mock_add.call_args_list 
                         if c.kwargs["metadata"].get("type") == "medical_fact"]
        assert len(medical_calls) == 15

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_interaction_facts_metadata(self, mock_add, mock_stats):
        """Interaction facts should have type user_pref or project_fact."""
        mock_add.return_value = True
        mock_stats.return_value = {}
        seed_medical_facts()
        types = set()
        for c in mock_add.call_args_list:
            t = c.kwargs["metadata"].get("type")
            if t in ("user_pref", "project_fact"):
                types.add(t)
        assert "user_pref" in types
        assert "project_fact" in types

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_medical_fact_categories_exist(self, mock_add, mock_stats):
        """Medical facts should cover key categories."""
        mock_add.return_value = True
        mock_stats.return_value = {}
        seed_medical_facts()
        categories = set()
        for c in mock_add.call_args_list:
            cat = c.kwargs["metadata"].get("category")
            if cat:
                categories.add(cat)
        expected = {"cardiovascular", "endocrine", "infectious", "respiratory",
                    "medication", "emergency", "general", "psychiatric", "nutrition",
                    "communication", "architecture", "memory_management"}
        assert expected.issubset(categories)

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_no_duplicate_texts(self, mock_add, mock_stats):
        """All seeded texts should be unique."""
        mock_add.return_value = True
        mock_stats.return_value = {}
        seed_medical_facts()
        texts = [c.kwargs["text"] for c in mock_add.call_args_list]
        assert len(texts) == len(set(texts))

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_falsy_return_value_counts_as_fail(self, mock_add, mock_stats):
        """add_memory returning False counts as fail."""
        mock_add.return_value = False
        mock_stats.return_value = {}
        success, fail = seed_medical_facts()
        assert success == 0
        assert fail == 18

    @patch("agents.seed_medical_memory.collection_stats")
    @patch("agents.seed_medical_memory.add_memory")
    def test_empty_string_return_counts_as_fail(self, mock_add, mock_stats):
        """add_memory returning empty string counts as fail."""
        mock_add.return_value = ""
        mock_stats.return_value = {}
        success, fail = seed_medical_facts()
        assert success == 0
        assert fail == 18
