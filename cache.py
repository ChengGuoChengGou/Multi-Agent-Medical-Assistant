"""
Redis-based response caching layer (Phase 26).
Falls back to in-memory LRU cache when Redis is unavailable.
"""
import hashlib
import json
import logging
import time
from typing import Optional, Any
from functools import lru_cache

logger = logging.getLogger("medical_chatbot.cache")

# --- Redis backend (optional) ---
_redis_client = None
REDIS_AVAILABLE = False

def init_redis(redis_url: str = "redis://localhost:6379/0") -> bool:
    """Initialize Redis connection. Returns True if successful."""
    global _redis_client, REDIS_AVAILABLE
    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(redis_url, decode_responses=True)
        REDIS_AVAILABLE = True
        logger.info("[Cache] Redis connected: %s", redis_url)
        return True
    except Exception as e:
        logger.warning("[Cache] Redis unavailable (%s), using in-memory fallback", e)
        REDIS_AVAILABLE = False
        return False

# --- In-memory fallback cache ---
_memory_cache: dict = {}
_memory_cache_max = 500

def _make_key(prefix: str, data: Any) -> str:
    """Generate deterministic cache key from input data."""
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


async def cache_get(key: str) -> Optional[dict]:
    """Retrieve cached response. Redis first, then memory fallback."""
    global _redis_client
    if REDIS_AVAILABLE and _redis_client:
        try:
            val = await _redis_client.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
    return _memory_cache.get(key)


async def cache_set(key: str, value: dict, ttl: int = 3600) -> None:
    """Store response in cache. Writes to both Redis and memory."""
    global _redis_client
    if REDIS_AVAILABLE and _redis_client:
        try:
            await _redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
        except Exception:
            pass
    # Always write to memory as fallback
    if len(_memory_cache) >= _memory_cache_max:
        # Evict oldest
        oldest = next(iter(_memory_cache))
        _memory_cache.pop(oldest, None)
    _memory_cache[key] = value


async def cache_delete(key: str) -> None:
    """Invalidate a cache entry."""
    global _redis_client
    _memory_cache.pop(key, None)
    if REDIS_AVAILABLE and _redis_client:
        try:
            await _redis_client.delete(key)
        except Exception:
            pass


async def cache_stats() -> dict:
    """Return cache statistics."""
    stats = {"backend": "redis" if REDIS_AVAILABLE else "memory", "memory_entries": len(_memory_cache)}
    if REDIS_AVAILABLE and _redis_client:
        try:
            info = await _redis_client.info("keyspace")
            stats["redis_keys"] = info.get("db0", {}).get("keys", 0)
        except Exception:
            stats["redis_keys"] = "unknown"
    return stats
