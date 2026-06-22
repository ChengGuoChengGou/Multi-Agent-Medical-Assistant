"""
Vector Memory Wrapper - Unified interface for vector-based memory operations.

Wraps medical_vector_memory.py with a clean interface and error handling.
Provides semantic search capabilities for medical context retrieval.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorMemory:
    """
    Vector-based semantic memory interface.
    
    Provides semantic search over medical knowledge and conversation history
    using qdrant + sentence-transformers embeddings.
    """
    
    def __init__(self):
        self._available = False
        self._add_memory = None
        self._search_memory = None
        self._get_all_memories = None
        self._stats = None
        
        try:
            from agents.medical_vector_memory import (
                _lazy_init,
                add_memory,
                collection_stats,
                get_all_memories,
                search_memory,
            )
            
            if not _lazy_init():
                raise RuntimeError("Vector memory initialization failed")
            
            self._add_memory = add_memory
            self._search_memory = search_memory
            self._get_all_memories = get_all_memories
            self._stats = collection_stats
            self._available = True
            
            stats = self._stats()
            logger.info(
                f"[VECTOR_MEMORY] Initialized: "
                f"{stats.get('points_count', 0)} vectors, "
                f"collection={stats.get('collection', '?')}"
            )
            
        except Exception as e:
            logger.warning(f"[VECTOR_MEMORY] Init failed: {e}")
    
    @property
    def available(self) -> bool:
        """Check if vector memory is available."""
        return self._available
    
    def add(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a memory entry to the vector store.
        
        Args:
            text: Text content to embed and store
            metadata: Optional metadata (user_id, category, timestamp, etc.)
            
        Returns:
            True if added successfully
        """
        if not self.available:
            logger.warning("[VECTOR_MEMORY] Not available")
            return False
        
        try:
            return self._add_memory(text, metadata or {})
        except Exception as e:
            logger.error(f"[VECTOR_MEMORY] Add failed: {e}")
            return False
    
    def search(self, query: str, limit: int = 5, 
               score_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Semantic search over vector memory.
        
        Args:
            query: Search query
            limit: Maximum results
            score_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of (text, score) tuples sorted by relevance
        """
        if not self.available:
            return []
        
        try:
            results = self._search_memory(query, limit=limit)
            
            # Filter by threshold
            filtered = [
                (r.get("text", ""), r.get("score", 0.0))
                for r in results
                if r.get("score", 0.0) >= score_threshold
            ]
            
            return filtered
        except Exception as e:
            logger.error(f"[VECTOR_MEMORY] Search failed: {e}")
            return []
    
    def get_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all stored memories.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of memory records
        """
        if not self.available:
            return []
        
        try:
            return self._get_all_memories(limit=limit)
        except Exception as e:
            logger.error(f"[VECTOR_MEMORY] Get all failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector memory statistics.
        
        Returns:
            Dictionary with collection stats (points_count, collection name, etc.)
        """
        if not self.available:
            return {"available": False}
        
        try:
            stats = self._stats()
            stats["available"] = True
            return stats
        except Exception as e:
            logger.error(f"[VECTOR_MEMORY] Stats failed: {e}")
            return {"available": False, "error": str(e)}


# Singleton
_vector_memory: Optional[VectorMemory] = None


def get_vector_memory() -> VectorMemory:
    """Get singleton VectorMemory instance."""
    global _vector_memory
    if _vector_memory is None:
        _vector_memory = VectorMemory()
    return _vector_memory
