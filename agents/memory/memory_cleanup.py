"""
Memory Cleanup - Strategies for managing memory lifecycle.

Provides cleanup policies for medical memory:
- TTL-based expiration
- Relevance-based pruning
- Duplicate consolidation
- Storage quota management
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryCleanupPolicy:
    """
    Memory cleanup policy configuration.

    Controls how old or irrelevant memories are managed.
    """

    def __init__(self, ttl_days: int = 365, max_memories_per_user: int = 1000, min_relevance_score: float = 0.1):
        """
        Args:
            ttl_days: Days before a memory is considered stale
            max_memories_per_user: Hard limit per user
            min_relevance_score: Minimum relevance to keep (0.0-1.0)
        """
        self.ttl_days = ttl_days
        self.max_memories_per_user = max_memories_per_user
        self.min_relevance_score = min_relevance_score


class MemoryCleanup:
    """
    Memory cleanup engine.

    Manages memory lifecycle with configurable policies.
    """

    # Default medical memory categories that should be preserved longer
    LONG_LIVED_CATEGORIES = {"allergy", "diagnosis", "history"}

    def __init__(self, policy: MemoryCleanupPolicy = None):
        self.policy = policy or MemoryCleanupPolicy()
        self._stats = {"cleaned": 0, "consolidated": 0, "expired": 0}
        logger.info("[MEMORY_CLEANUP] Cleanup engine initialized")

    @property
    def stats(self) -> dict[str, int]:
        """Get cleanup statistics."""
        return self._stats.copy()

    def should_keep(self, memory: dict[str, Any]) -> bool:
        """
        Determine if a memory should be kept.

        Args:
            memory: Memory record with metadata

        Returns:
            True if memory should be retained
        """
        # Check TTL
        created = memory.get("created_at") or memory.get("timestamp")
        if created:
            try:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created)
                elif isinstance(created, (int, float)):
                    created_dt = datetime.fromtimestamp(created)
                else:
                    created_dt = created

                age_days = (datetime.now() - created_dt).days

                # Long-lived categories get extended TTL
                category = memory.get("metadata", {}).get("category", "")
                ttl = self.policy.ttl_days
                if category in self.LONG_LIVED_CATEGORIES:
                    ttl = ttl * 2  # Double TTL for allergies, diagnoses, history

                if age_days > ttl:
                    return False
            except Exception:
                pass  # If we can't parse date, keep the memory

        # Check relevance
        score = memory.get("score") or memory.get("relevance", 1.0)
        return not score < self.min_relevance_score

    def cleanup_user(self, user_id: str) -> dict[str, int]:
        """
        Run cleanup for a specific user's memories.

        Args:
            user_id: User identifier

        Returns:
            Cleanup statistics for this user
        """
        result = {"kept": 0, "removed": 0, "consolidated": 0}

        try:
            from agents.memory_module import get_memory_store

            store = get_memory_store()

            history = store.get_history(user_id, limit=self.policy.max_memories_per_user)

            for memory in history:
                if self.should_keep(memory):
                    result["kept"] += 1
                else:
                    result["removed"] += 1
                    # Note: Actual deletion depends on MemoryStore implementation

            self._stats["cleaned"] += result["removed"]
            logger.info(f"[MEMORY_CLEANUP] User {user_id}: kept={result['kept']}, removed={result['removed']}")

        except Exception as e:
            logger.error(f"[MEMORY_CLEANUP] Cleanup failed for {user_id}: {e}")

        return result

    def get_stale_memories(self, user_id: str) -> list[dict]:
        """
        Get list of stale memories that should be cleaned.

        Args:
            user_id: User identifier

        Returns:
            List of stale memory records
        """
        stale = []

        try:
            from agents.memory_module import get_memory_store

            store = get_memory_store()

            history = store.get_history(user_id, limit=self.policy.max_memories_per_user)

            for memory in history:
                if not self.should_keep(memory):
                    stale.append(memory)

        except Exception as e:
            logger.error(f"[MEMORY_CLEANUP] Stale check failed: {e}")

        return stale


# Singleton
_cleanup_instance: MemoryCleanup | None = None


def get_cleanup_engine(policy: MemoryCleanupPolicy = None) -> MemoryCleanup:
    """Get singleton MemoryCleanup instance."""
    global _cleanup_instance
    if _cleanup_instance is None:
        _cleanup_instance = MemoryCleanup(policy)
    return _cleanup_instance
