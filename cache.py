"""
Redis-based response caching layer (Phase 26).
Falls back to in-memory LRU cache when Redis is unavailable.

Phase 50: Added SemanticCache — TF-IDF cosine similarity cache
for finding similar (not identical) queries. Zero extra dependencies.
"""

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from typing import Any

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
_memory_cache: dict = {}  # key -> {"value": dict, "ts": float, "ttl": int}
_memory_cache_max = 500
_memory_cache_default_ttl = 3600  # 1 hour


def _make_key(prefix: str, data: Any) -> str:
    """Generate deterministic cache key from input data."""
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _memory_cache_cleanup() -> int:
    """Remove expired entries from memory cache. Returns count removed."""
    global _memory_cache
    now = time.time()
    expired = [k for k, v in _memory_cache.items() if now - v["ts"] > v.get("ttl", _memory_cache_default_ttl)]
    for k in expired:
        _memory_cache.pop(k, None)
    return len(expired)


async def cache_get(key: str) -> dict | None:
    """Retrieve cached response. Redis first, then memory fallback."""
    global _redis_client
    if REDIS_AVAILABLE and _redis_client:
        try:
            val = await _redis_client.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
    entry = _memory_cache.get(key)
    if entry is None:
        return None
    # Check TTL
    if time.time() - entry["ts"] > entry.get("ttl", _memory_cache_default_ttl):
        _memory_cache.pop(key, None)
        return None
    return entry["value"]


async def cache_set(key: str, value: dict, ttl: int = 3600) -> None:
    """Store response in cache. Writes to both Redis and memory."""
    global _redis_client
    if REDIS_AVAILABLE and _redis_client:
        try:
            await _redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
        except Exception:
            pass
    # Always write to memory as fallback (with TTL)
    if len(_memory_cache) >= _memory_cache_max:
        # Evict expired first, then oldest
        removed = _memory_cache_cleanup()
        if removed == 0 and len(_memory_cache) >= _memory_cache_max:
            oldest = next(iter(_memory_cache))
            _memory_cache.pop(oldest, None)
    _memory_cache[key] = {"value": value, "ts": time.time(), "ttl": ttl}


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
    expired_count = _memory_cache_cleanup()
    stats = {
        "backend": "redis" if REDIS_AVAILABLE else "memory",
        "memory_entries": len(_memory_cache),
        "memory_expired_cleaned": expired_count,
        "memory_max": _memory_cache_max,
        "memory_default_ttl": _memory_cache_default_ttl,
    }
    if REDIS_AVAILABLE and _redis_client:
        try:
            info = await _redis_client.info("keyspace")
            stats["redis_keys"] = info.get("db0", {}).get("keys", 0)
        except Exception:
            stats["redis_keys"] = "unknown"
    # Semantic cache stats
    _semantic_cleanup_expired()
    stats["semantic_cache_entries"] = len(_semantic_cache)
    stats["semantic_cache_hits"] = _semantic_hits
    stats["semantic_cache_misses"] = _semantic_misses
    return stats


# ============================================================
# Phase 50: Semantic Cache (TF-IDF cosine similarity)
# ============================================================
# Finds *similar* (not identical) queries and returns cached
# responses. Zero external dependencies — pure Python TF-IDF.
# ============================================================

_semantic_cache: list[dict] = []  # [{query, tfidf, response, ts, ttl}]
_semantic_max = 200
_semantic_threshold = 0.75  # cosine similarity threshold
_semantic_hits = 0
_semantic_misses = 0

# Chinese-aware stop words (minimal set)
_STOP_WORDS = frozenset(
    [
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "of",
        "in",
        "to",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
    ]
)


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: split on non-alphanumeric + single CJK chars."""
    # Keep CJK characters as individual tokens, split words on non-alpha
    text = text.lower().strip()
    tokens = []
    for char_group in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text):
        if not char_group:
            continue
        # For Chinese: split each character
        if any("\u4e00" <= c <= "\u9fff" for c in char_group):
            for c in char_group:
                if c not in _STOP_WORDS and len(c.strip()) > 0:
                    tokens.append(c)
        else:
            if char_group not in _STOP_WORDS and len(char_group) > 1:
                tokens.append(char_group)
    return tokens


def _compute_tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF vector for a token list given global IDF."""
    tf = Counter(tokens)
    total = len(tokens) or 1
    vec = {}
    for term, count in tf.items():
        vec[term] = (count / total) * idf.get(term, 1.0)
    return vec


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_idf() -> dict[str, float]:
    """Build IDF from all cached semantic entries."""
    doc_freq: Counter = Counter()
    n_docs = len(_semantic_cache)
    if n_docs == 0:
        return {}
    for entry in _semantic_cache:
        seen = set(entry["tokens"])
        for t in seen:
            doc_freq[t] += 1
    return {t: math.log((n_docs + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}


def _semantic_cleanup_expired() -> int:
    """Remove expired semantic cache entries. Returns count removed."""
    global _semantic_cache
    now = time.time()
    before = len(_semantic_cache)
    _semantic_cache = [e for e in _semantic_cache if now - e["ts"] <= e.get("ttl", 3600)]
    return before - len(_semantic_cache)


def semantic_get(query: str) -> dict | None:
    """Find a cached response for a semantically similar query."""
    global _semantic_hits, _semantic_misses
    if not _semantic_cache:
        _semantic_misses += 1
        return None

    tokens = _tokenize(query)
    if not tokens:
        _semantic_misses += 1
        return None

    idf = _build_idf()
    query_vec = _compute_tfidf(tokens, idf)

    best_sim = 0.0
    best_entry = None
    now = time.time()

    for entry in _semantic_cache:
        # Skip expired
        if now - entry["ts"] > entry.get("ttl", 3600):
            continue
        sim = _cosine_similarity(query_vec, entry["tfidf"])
        if sim > best_sim:
            best_sim = sim
            best_entry = entry

    if best_entry and best_sim >= _semantic_threshold:
        _semantic_hits += 1
        logger.info("[SemanticCache] HIT (sim=%.3f) for: %s", best_sim, query[:50])
        return best_entry["response"]

    _semantic_misses += 1
    return None


def semantic_set(query: str, response: dict, ttl: int = 3600) -> None:
    """Store a query-response pair in the semantic cache."""
    global _semantic_cache
    tokens = _tokenize(query)
    if not tokens:
        return

    # Evict oldest if at capacity
    if len(_semantic_cache) >= _semantic_max:
        _semantic_cache.pop(0)

    # Rebuild TF-IDF for all entries + new one
    idf = _build_idf()
    tfidf = _compute_tfidf(tokens, idf)

    _semantic_cache.append(
        {
            "query": query,
            "tokens": tokens,
            "tfidf": tfidf,
            "response": response,
            "ts": time.time(),
            "ttl": ttl,
        }
    )

    # Rebuild TF-IDF for existing entries (IDF changed)
    idf_new = _build_idf()
    for entry in _semantic_cache:
        entry["tfidf"] = _compute_tfidf(entry["tokens"], idf_new)

    logger.debug("[SemanticCache] Stored (total=%d): %s", len(_semantic_cache), query[:50])


def semantic_stats() -> dict:
    """Return semantic cache statistics."""
    return {
        "entries": len(_semantic_cache),
        "hits": _semantic_hits,
        "misses": _semantic_misses,
        "hit_rate": round(_semantic_hits / max(1, _semantic_hits + _semantic_misses), 3),
        "threshold": _semantic_threshold,
        "max_entries": _semantic_max,
    }
