"""Tests for agents/memory/memory_cleanup.py."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pytest
from agents.memory.memory_cleanup import (
    MemoryCleanupPolicy,
    MemoryCleanup,
    get_cleanup_engine,
)


class TestMemoryCleanupPolicy:
    """Test MemoryCleanupPolicy defaults and custom values."""

    def test_defaults(self):
        p = MemoryCleanupPolicy()
        assert p.ttl_days == 365
        assert p.max_memories_per_user == 1000
        assert p.min_relevance_score == 0.1

    def test_custom(self):
        p = MemoryCleanupPolicy(ttl_days=30, max_memories_per_user=50, min_relevance_score=0.5)
        assert p.ttl_days == 30
        assert p.max_memories_per_user == 50
        assert p.min_relevance_score == 0.5


class TestShouldKeep:
    """Test MemoryCleanup.should_keep logic."""

    @pytest.fixture
    def engine(self):
        return MemoryCleanup(MemoryCleanupPolicy(ttl_days=30, min_relevance_score=0.1))

    def test_keep_no_timestamp(self, engine):
        mem = {"content": "test"}
        assert engine.should_keep(mem) is True

    def test_keep_recent(self, engine):
        recent = (datetime.now() - timedelta(days=5)).isoformat()
        mem = {"created_at": recent, "content": "recent"}
        assert engine.should_keep(mem) is True

    def test_remove_old(self, engine):
        old = (datetime.now() - timedelta(days=60)).isoformat()
        mem = {"created_at": old, "content": "old"}
        assert engine.should_keep(mem) is False

    def test_keep_old_allergy_extended_ttl(self, engine):
        old = (datetime.now() - timedelta(days=50)).isoformat()
        mem = {"created_at": old, "metadata": {"category": "allergy"}}
        # allergy gets 2x TTL (30*2=60), 50 < 60 → keep
        assert engine.should_keep(mem) is True

    def test_remove_old_allergy_past_extended(self, engine):
        old = (datetime.now() - timedelta(days=70)).isoformat()
        mem = {"created_at": old, "metadata": {"category": "allergy"}}
        assert engine.should_keep(mem) is False

    def test_keep_high_relevance(self, engine):
        mem = {"content": "important", "score": 0.8}
        assert engine.should_keep(mem) is True

    def test_remove_low_relevance(self, engine):
        mem = {"content": "noise", "score": 0.01}
        assert engine.should_keep(mem) is False

    def test_keep_default_relevance(self, engine):
        mem = {"content": "no score key"}
        # score defaults to 1.0 when missing
        assert engine.should_keep(mem) is True

    def test_keep_relevance_key(self, engine):
        mem = {"content": "test", "relevance": 0.5}
        assert engine.should_keep(mem) is True

    def test_keep_timestamp_int(self, engine):
        ts = int((datetime.now() - timedelta(days=5)).timestamp())
        mem = {"timestamp": ts}
        assert engine.should_keep(mem) is True

    def test_remove_bad_date_keeps(self, engine):
        mem = {"created_at": "not-a-date"}
        # bad date → exception caught → keep
        assert engine.should_keep(mem) is True


class TestCleanupUser:
    """Test cleanup_user with mocked store."""

    def test_cleanup_mixed(self):
        engine = MemoryCleanup(MemoryCleanupPolicy(ttl_days=30, min_relevance_score=0.1))
        recent = datetime.now().isoformat()
        old = (datetime.now() - timedelta(days=60)).isoformat()
        mock_store = MagicMock()
        mock_store.get_history.return_value = [
            {"content": "recent", "created_at": recent},
            {"content": "old", "created_at": old},
        ]

        with patch("agents.memory_module.get_memory_store", return_value=mock_store):
            result = engine.cleanup_user("u1")
        assert result["kept"] == 1
        assert result["removed"] == 1
        assert engine.stats["cleaned"] == 1

    def test_cleanup_exception(self):
        engine = MemoryCleanup()
        with patch("agents.memory_module.get_memory_store", side_effect=RuntimeError("fail")):
            result = engine.cleanup_user("u1")
        assert result["kept"] == 0
        assert result["removed"] == 0


class TestGetStaleMemories:
    """Test get_stale_memories."""

    def test_finds_stale(self):
        engine = MemoryCleanup(MemoryCleanupPolicy(ttl_days=30))
        old = (datetime.now() - timedelta(days=60)).isoformat()
        recent = datetime.now().isoformat()
        mock_store = MagicMock()
        mock_store.get_history.return_value = [
            {"content": "old", "created_at": old},
            {"content": "recent", "created_at": recent},
        ]
        with patch("agents.memory_module.get_memory_store", return_value=mock_store):
            stale = engine.get_stale_memories("u1")
        assert len(stale) == 1
        assert stale[0]["content"] == "old"

    def test_exception_returns_empty(self):
        engine = MemoryCleanup()
        with patch("agents.memory_module.get_memory_store", side_effect=RuntimeError("fail")):
            assert engine.get_stale_memories("u1") == []


class TestGetCleanupEngine:
    """Test singleton get_cleanup_engine."""

    def test_singleton(self):
        import agents.memory.memory_cleanup as mod
        mod._cleanup_instance = None
        e1 = get_cleanup_engine()
        e2 = get_cleanup_engine()
        assert e1 is e2
        mod._cleanup_instance = None

    def test_custom_policy(self):
        import agents.memory.memory_cleanup as mod
        mod._cleanup_instance = None
        p = MemoryCleanupPolicy(ttl_days=7)
        e = get_cleanup_engine(policy=p)
        assert e.policy.ttl_days == 7
        mod._cleanup_instance = None
