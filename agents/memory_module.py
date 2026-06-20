"""
Long-term memory module for the medical chatbot.
Uses Mem0 for vector-based semantic memory with graceful fallback.

Architecture:
    - Mem0 (primary): Vector-based semantic memory, supports user isolation
    - In-memory dict (fallback): When Mem0 is unavailable

Usage:
    from agents.memory_module import get_memory_store
    memory = get_memory_store()
    
    # Save important medical context
    memory.remember(user_id, "Patient reports penicillin allergy")
    
    # Recall relevant memories for a query
    context = memory.recall(user_id, "What medications should I avoid?")
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """Abstract interface for long-term memory operations."""
    
    def remember(self, user_id: str, content: str, metadata: dict = None) -> bool:
        raise NotImplementedError
    
    def recall(self, user_id: str, query: str, limit: int = 5) -> str:
        raise NotImplementedError
    
    def get_history(self, user_id: str, limit: int = 20) -> list:
        raise NotImplementedError
    
    def forget(self, user_id: str) -> bool:
        raise NotImplementedError


class Mem0MemoryStore(MemoryStore):
    """Mem0-backed semantic memory store.
    
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
    
    def remember(self, user_id: str, content: str, metadata: dict = None) -> bool:
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
            results = self.client.search(query, user_id=user_id, limit=limit)
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
    """Simple in-memory fallback when Mem0 is unavailable.
    Uses a dict with basic keyword matching for recall.
    NOT suitable for production - no persistence, no semantic search.
    """
    
    def __init__(self):
        self._store: dict[str, list[dict]] = {}
        logger.info("Using in-memory fallback store (no persistence)")
    
    def remember(self, user_id: str, content: str, metadata: dict = None) -> bool:
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
_store_instance: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store instance.
    Tries Mem0 first, falls back to in-memory dict.
    """
    global _store_instance
    if _store_instance is None:
        store = Mem0MemoryStore()
        if store.available:
            _store_instance = store
            logger.info("Memory store: Mem0 (vector-based semantic memory)")
        else:
            _store_instance = InMemoryStore()
            logger.warning("Memory store: InMemory fallback (install mem0ai for semantic memory)")
    return _store_instance
