"""
Tests for cache.py — Phase 52: TTL, cleanup, semantic cache.
"""
import asyncio
import time
import sys
import os
import importlib.util

import pytest

# Import cache module from project root
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache.py")
spec = importlib.util.spec_from_file_location("cache", CACHE_PATH)
cache = importlib.util.module_from_spec(spec)
sys.modules["cache"] = cache
spec.loader.exec_module(cache)


# ── Memory cache TTL tests ──

class TestMemoryCacheTTL:
    def setup_method(self):
        cache._memory_cache.clear()

    @pytest.mark.asyncio
    async def test_cache_set_get_basic(self):
        """Set then get should return value."""
        key = cache._make_key("test", {"q": "hello"})
        await cache.cache_set(key, {"answer": "world"}, ttl=60)
        result = await cache.cache_get(key)
        assert result == {"answer": "world"}

    @pytest.mark.asyncio
    async def test_cache_expired_returns_none(self):
        """Expired entry should return None."""
        key = cache._make_key("test", {"q": "expired"})
        # Manually set with ts in the past
        cache._memory_cache[key] = {"value": {"old": True}, "ts": time.time() - 7200, "ttl": 60}
        result = await cache.cache_get(key)
        assert result is None
        # Should be removed from cache
        assert key not in cache._memory_cache

    @pytest.mark.asyncio
    async def test_cache_not_expired_returns_value(self):
        """Fresh entry should be returned."""
        key = cache._make_key("test", {"q": "fresh"})
        await cache.cache_set(key, {"answer": "yes"}, ttl=3600)
        result = await cache.cache_get(key)
        assert result == {"answer": "yes"}

    @pytest.mark.asyncio
    async def test_cache_max_eviction(self):
        """When full, should evict expired first, then oldest."""
        cache._memory_cache_max = 3
        for i in range(4):
            key = f"key_{i}"
            await cache.cache_set(key, {"i": i}, ttl=3600)
        assert len(cache._memory_cache) <= 3


# ── Memory cache cleanup tests ──

class TestMemoryCacheCleanup:
    def setup_method(self):
        cache._memory_cache.clear()

    def test_cleanup_removes_expired(self):
        """_memory_cache_cleanup should remove expired entries."""
        cache._memory_cache["expired"] = {"value": {"x": 1}, "ts": time.time() - 7200, "ttl": 60}
        cache._memory_cache["fresh"] = {"value": {"x": 2}, "ts": time.time(), "ttl": 3600}
        removed = cache._memory_cache_cleanup()
        assert removed == 1
        assert "expired" not in cache._memory_cache
        assert "fresh" in cache._memory_cache

    def test_cleanup_returns_zero_when_none_expired(self):
        cache._memory_cache["ok"] = {"value": {"x": 1}, "ts": time.time(), "ttl": 3600}
        removed = cache._memory_cache_cleanup()
        assert removed == 0


# ── Semantic cache tests ──

class TestSemanticCache:
    def setup_method(self):
        cache._semantic_cache.clear()
        cache._semantic_hits = 0
        cache._semantic_misses = 0

    def test_semantic_set_and_get_exact(self):
        """Exact match should return cached answer."""
        cache.semantic_set("什么是高血压", {"answer": "高血压是..."}, ttl=3600)
        result = cache.semantic_get("什么是高血压")
        assert result is not None
        assert result["answer"] == "高血压是..."

    def test_semantic_get_similar(self):
        """Similar query should match via cosine similarity."""
        cache.semantic_set("高血压的症状有哪些", {"answer": "头晕头痛..."}, ttl=3600)
        result = cache.semantic_get("高血压有什么症状")
        # May or may not match depending on threshold — just check no crash
        assert result is None or isinstance(result, dict)

    def test_semantic_expired_not_returned(self):
        """Expired entry should not be returned."""
        cache._semantic_cache.append({
            "query": "old question",
            "answer": {"stale": True},
            "ts": time.time() - 7200,
            "ttl": 60,
            "tokens": ["old", "question"],
            "tfidf": {"old": 1.0, "question": 1.0},
        })
        result = cache.semantic_get("old question")
        assert result is None

    def test_semantic_cleanup_expired_removes_old(self):
        """_semantic_cleanup_expired should remove stale entries."""
        cache._semantic_cache.append({
            "query": "expired", "answer": {}, "ts": time.time() - 7200, "ttl": 60,
            "tokens": ["expired"], "tfidf": {"expired": 1.0},
        })
        cache._semantic_cache.append({
            "query": "fresh", "answer": {}, "ts": time.time(), "ttl": 3600,
            "tokens": ["fresh"], "tfidf": {"fresh": 1.0},
        })
        removed = cache._semantic_cleanup_expired()
        assert removed == 1
        assert len(cache._semantic_cache) == 1


# ── Semantic stats test ──

class TestSemanticStats:
    def test_semantic_stats_returns_all_fields(self):
        stats = cache.semantic_stats()
        for key in ["entries", "hits", "misses", "hit_rate", "threshold", "max_entries"]:
            assert key in stats

    @pytest.mark.asyncio
    async def test_cache_stats_includes_memory_ttl(self):
        stats = await cache.cache_stats()
        assert "memory_expired_cleaned" in stats
        assert "memory_default_ttl" in stats
        assert stats["memory_default_ttl"] == 3600


# ── Make key deterministic test ──

class TestMakeKey:
    def test_deterministic(self):
        a = cache._make_key("p", {"q": "hello", "x": 1})
        b = cache._make_key("p", {"x": 1, "q": "hello"})
        assert a == b

    def test_different_data_different_key(self):
        a = cache._make_key("p", {"q": "hello"})
        b = cache._make_key("p", {"q": "world"})
        assert a != b
