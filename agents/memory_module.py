"""
Long-term memory module for the medical chatbot.
Three-tier architecture (from GenericAgent):
    1. Vector Memory (primary): Direct qdrant + sentence-transformers semantic search
    2. Mem0 (secondary): Cloud/local vector memory service
    3. In-memory dict (fallback): When nothing else is available

Usage:
    from agents.memory_module import get_memory_store
    memory = get_memory_store()

    # Save important medical context
    memory.remember(user_id, "Patient reports penicillin allergy")

    # Recall relevant memories for a query
    context = memory.recall(user_id, "What medications should I avoid?")
"""

import logging
import os

logger = logging.getLogger(__name__)


class MemoryStore:
    """Abstract interface for long-term memory operations."""

    def remember(self, user_id: str, content: str, metadata: dict | None = None) -> bool:
        raise NotImplementedError

    def recall(self, user_id: str, query: str, limit: int = 5) -> str:
        raise NotImplementedError

    def get_history(self, user_id: str, limit: int = 20) -> list:
        raise NotImplementedError

    def forget(self, user_id: str) -> bool:
        raise NotImplementedError


class VectorMemoryStore(MemoryStore):
    """Semantic vector memory backed by qdrant + sentence-transformers.

    This is the primary memory store, adapted from GenericAgent's vector_memory.
    Provides real semantic search (not keyword matching) with local persistence.
    """

    def __init__(self):
        try:
            from agents.medical_vector_memory import (
                _lazy_init,
                add_memory,
                collection_stats,
                get_all_memories,
                search_memory,
            )

            if not _lazy_init():
                raise RuntimeError("vector memory init failed")

            self._add_memory = add_memory
            self._search_memory = search_memory
            self._get_all_memories = get_all_memories
            self._stats = collection_stats
            self._available = True

            stats = self._stats()
            logger.info(
                f"VectorMemoryStore initialized: "
                f"{stats.get('points_count', 0)} vectors, "
                f"collection={stats.get('collection', '?')}"
            )

        except Exception as e:
            logger.warning(f"VectorMemoryStore init failed: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def remember(self, user_id: str, content: str, metadata: dict | None = None) -> bool:
        if not self._available:
            return False
        return self._add_memory(content, user_id=user_id, metadata=metadata)

    def recall(self, user_id: str, query: str, limit: int = 5) -> str:
        if not self._available:
            return ""
        results = self._search_memory(query, user_id=user_id, top_k=limit)
        if not results:
            return ""
        memories = [f"- {r}" for r in results]
        return "Previously known information about this user:\n" + "\n".join(memories)

    def get_history(self, user_id: str, limit: int = 20) -> list:
        if not self._available:
            return []
        return self._get_all_memories(limit=limit, user_id=user_id)

    def forget(self, user_id: str) -> bool:
        """Note: qdrant doesn't easily support per-user deletion without filter-based delete."""
        if not self._available:
            return False
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            from agents.medical_vector_memory import _collection_name, _qdrant_client

            _qdrant_client.delete(
                collection_name=_collection_name,
                points_selector=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete memories for user {user_id}: {e}")
            return False


class Mem0MemoryStore(MemoryStore):
    """Mem0-backed semantic memory store (secondary option).

    Mem0 provides:
    - Vector-based semantic search over memories
    - User-scoped memory isolation
    - Automatic deduplication
    - Cross-session persistence
    """

    def __init__(self):
        try:
            from mem0 import MemoryClient

            api_key = os.getenv("MEM0_API_KEY")
            host = os.getenv("MEM0_HOST")

            if api_key:
                self.client = MemoryClient(api_key=api_key, host=host)
                self.mode = "cloud"
            else:
                # Local mode with default config
                from mem0 import Memory

                self.client = Memory()
                self.mode = "local"

            logger.info(f"Mem0 initialized in {self.mode} mode")
            self._available = True

        except ImportError:
            logger.warning("mem0ai not installed. Run: pip install mem0ai")
            self._available = False
        except Exception as e:
            logger.warning(f"Mem0 init failed: {e}. Falling back to in-memory.")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def remember(self, user_id: str, content: str, metadata: dict | None = None) -> bool:
        """Store a memory for a user."""
        if not self._available:
            return False
        try:
            meta = metadata or {}
            meta["source"] = meta.get("source", "chat_interaction")
            if self.mode == "cloud":
                self.client.add([{"role": "user", "content": content}], user_id=user_id, metadata=meta)
            else:
                self.client.add(content, user_id=user_id, metadata=meta)
            logger.debug(f"Stored memory for user {user_id}: {content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    def recall(self, user_id: str, query: str, limit: int = 5) -> str:
        """Recall relevant memories for a query."""
        if not self._available:
            return ""
        try:
            results = self.client.search(query=query, user_id=user_id, limit=limit)
            if not results:
                return ""
            memories = []
            for r in results:
                if self.mode == "cloud":
                    mem_text = r.get("memory", r.get("text", ""))
                else:
                    mem_text = r.get("memory", str(r))
                if mem_text:
                    memories.append(f"- {mem_text}")
            if memories:
                return "Previously known information about this user:\n" + "\n".join(memories)
            return ""
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return ""

    def get_history(self, user_id: str, limit: int = 20) -> list:
        """Get all memories for a user."""
        if not self._available:
            return []
        try:
            results = self.client.get_all(user_id=user_id, limit=limit)
            if self.mode == "cloud":
                return [r.get("memory", r.get("text", "")) for r in results]
            return [r.get("memory", str(r)) for r in results]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

    def forget(self, user_id: str) -> bool:
        """Delete all memories for a user."""
        if not self._available:
            return False
        try:
            self.client.delete_all(user_id=user_id)
            logger.info(f"Deleted all memories for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memories: {e}")
            return False


class InMemoryStore(MemoryStore):
    """Simple in-memory fallback when nothing else is available.
    Uses a dict with basic keyword matching for recall.
    NOT suitable for production - no persistence, no semantic search.
    """

    def __init__(self):
        self._store: dict[str, list[dict]] = {}
        logger.info("Using in-memory fallback store (no persistence)")

    def remember(self, user_id: str, content: str, metadata: dict | None = None) -> bool:
        if user_id not in self._store:
            self._store[user_id] = []
        self._store[user_id].append({"content": content, "metadata": metadata or {}})
        return True

    def recall(self, user_id: str, query: str, limit: int = 5) -> str:
        memories = self._store.get(user_id, [])
        if not memories:
            return ""
        query_words = set(query.lower().split())
        scored = []
        for m in memories:
            content = m["content"].lower()
            score = sum(1 for w in query_words if w in content)
            if score > 0:
                scored.append((score, m["content"]))
        scored.sort(key=lambda x: -x[0])
        top = scored[:limit]
        if top:
            return "Previously known information:\n" + "\n".join(f"- {c}" for _, c in top)
        return ""

    def get_history(self, user_id: str, limit: int = 20) -> list:
        return [m["content"] for m in self._store.get(user_id, [])[-limit:]]

    def forget(self, user_id: str) -> bool:
        self._store.pop(user_id, None)
        return True


# === Singleton factory ===
_store_instance: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store instance.

    Priority order (from GenericAgent pattern):
    1. VectorMemoryStore (qdrant + sentence-transformers) - best semantic search
    2. Mem0MemoryStore (mem0 cloud/local) - if configured
    3. InMemoryStore (dict) - always works, no persistence

    The result is cached as a singleton.
    """
    global _store_instance
    if _store_instance is None:
        # Tier 1: Vector memory (local qdrant)
        store = VectorMemoryStore()
        if store.available:
            _store_instance = store
            logger.info("Memory store: VectorMemory (qdrant + sentence-transformers)")
            return _store_instance

        # Tier 2: Mem0
        store = Mem0MemoryStore()
        if store.available:
            _store_instance = store
            logger.info("Memory store: Mem0 (vector-based semantic memory)")
            return _store_instance

        # Tier 3: In-memory fallback
        _store_instance = InMemoryStore()
        logger.warning("Memory store: InMemory fallback (no vector memory available)")

    return _store_instance
