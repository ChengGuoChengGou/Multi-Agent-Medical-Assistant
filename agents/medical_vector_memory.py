"""
Medical Vector Memory for Multi-Agent Medical Assistant.
Adapted from GenericAgent's vector_memory.py with medical-domain enhancements.

Direct qdrant + sentence-transformers, no external memory service dependency.
Provides semantic search over medical knowledge and conversation memories.

Architecture:
    - sentence-transformers multilingual-MiniLM-L12-v2 (384-dim) for embeddings
    - qdrant (local persistent storage) for vector store
    - Lazy initialization on first use

Usage:
    from agents.medical_vector_memory import (
        add_memory, search_memory, seed_from_medical_knowledge,
        format_memory_for_prompt, collection_stats
    )

    # Store a medical fact
    add_memory("Patient has penicillin allergy", user_id="user_123")

    # Semantic search
    results = search_memory("What medications should be avoided?", user_id="user_123")

    # Seed from medical knowledge files
    seed_from_medical_knowledge(["path/to/medical_notes.txt"])
"""

import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Lazy-init globals ──
_qdrant_client = None
_model = None
_initialized = False
_init_error = None

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)  # D:\Code\Multi-Agent-Medical-Assistant

# ── Configurable paths ──
_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model_cache_folder = os.path.join(_project_root, "models")
_qdrant_store_path = os.path.join(_project_root, "data", "medical_memory")
_collection_name = "medical_memory"
_vector_size = 384  # MiniLM-L12-v2 output dimension


def _lazy_init():
    """Initialize model and qdrant client on first use."""
    global _qdrant_client, _model, _initialized, _init_error

    if _initialized:
        return True
    if _init_error:
        return False

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from sentence_transformers import SentenceTransformer

        os.makedirs(_qdrant_store_path, exist_ok=True)

        _model = SentenceTransformer(
            _model_name,
            cache_folder=_model_cache_folder,
        )

        _qdrant_client = QdrantClient(path=_qdrant_store_path)

        # Create collection if missing
        existing = [c.name for c in _qdrant_client.get_collections().collections]
        if _collection_name not in existing:
            _qdrant_client.create_collection(
                collection_name=_collection_name,
                vectors_config=VectorParams(
                    size=_vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created qdrant collection: {_collection_name}")

        _initialized = True
        logger.info(f"Medical vector memory initialized: qdrant={_qdrant_store_path}, model={_model_name}")
        return True

    except Exception as e:
        _init_error = str(e)
        logger.error(f"Medical vector memory init failed: {e}")
        return False


# ── Core API ──


def add_memory(text: str, user_id: str = "global", metadata: dict | None = None) -> bool:
    """Add a single memory entry."""
    if not _lazy_init():
        return False
    try:
        import uuid

        from qdrant_client.models import PointStruct

        text = (text or "").strip()
        if not text:
            return False

        # Truncate to 512 words to limit embedding time
        words = text.split()
        if len(words) > 512:
            text = " ".join(words[:512])

        embedding = _model.encode([text])[0].tolist()

        payload = {
            "text": text,
            "user_id": user_id,
            "added_at": datetime.now().isoformat(),
        }
        if metadata:
            payload["metadata"] = metadata

        _qdrant_client.upsert(
            collection_name=_collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        logger.debug(f"Added memory for user={user_id}: {text[:60]}...")
        return True
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return False


def add_memories_batch(texts: list, user_id: str = "global") -> int:
    """Batch-add memories. Returns number successfully added."""
    if not _lazy_init() or not texts:
        return 0
    try:
        import uuid

        from qdrant_client.models import PointStruct

        # Filter and truncate
        clean = []
        for t in texts:
            t = (t or "").strip()
            if t:
                words = t.split()
                if len(words) > 512:
                    t = " ".join(words[:512])
                clean.append(t)

        if not clean:
            return 0

        embeddings = _model.encode(clean, show_progress_bar=False)

        points = []
        for text, emb in zip(clean, embeddings, strict=False):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload={
                        "text": text,
                        "user_id": user_id,
                        "added_at": datetime.now().isoformat(),
                    },
                )
            )

        _qdrant_client.upsert(
            collection_name=_collection_name,
            points=points,
        )
        logger.info(f"Batch-added {len(points)} memories for user={user_id}")
        return len(points)

    except Exception as e:
        logger.error(f"Batch add failed: {e}")
        return 0


def search_memory(query: str, user_id: str | None = None, top_k: int = 5, min_score: float = 0.3) -> list:
    """Semantic search. Returns list of memory text strings."""
    if not _lazy_init():
        return []
    try:
        query = (query or "").strip()
        if not query:
            return []

        q_emb = _model.encode([query])[0].tolist()

        # Build filter for user_id if specified
        query_filter = None
        if user_id:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            )

        results = _qdrant_client.query_points(
            collection_name=_collection_name,
            query=q_emb,
            query_filter=query_filter,
            limit=top_k,
        )

        memories = []
        for hit in results.points:
            score = hit.score
            if score < min_score:
                continue
            text = hit.payload.get("text", "")
            if text:
                memories.append(text)

        return memories

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def get_all_memories(limit: int = 200, user_id: str | None = None) -> list:
    """Return all stored memory texts."""
    if not _lazy_init():
        return []
    try:
        query_filter = None
        if user_id:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            )

        results, _ = _qdrant_client.scroll(
            collection_name=_collection_name,
            limit=limit,
            scroll_filter=query_filter,
        )
        return [p.payload.get("text", "") for p in results if p.payload.get("text")]

    except Exception as e:
        logger.error(f"Get all failed: {e}")
        return []


# ── Medical-specific seeding ──


def seed_from_medical_knowledge(file_paths: list) -> int:
    """Seed vector memory from medical knowledge text files.

    Each line in the file is treated as a potential memory entry.
    Filters out headers, short lines, and metadata markers.
    """
    if not file_paths:
        return 0

    all_texts = []
    for filepath in file_paths:
        if not os.path.exists(filepath):
            logger.warning(f"Seed file not found: {filepath}")
            continue

        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                content = f.read()

            for line in content.split("\n"):
                line = line.strip()
                # Skip headers, separators, short lines
                if not line or line.startswith(("#", "---")):
                    continue
                if len(line) < 15:
                    continue
                # Try to extract value from "key: value" or "key | value" format
                for sep in ["|", ":"]:
                    if sep in line:
                        parts = line.split(sep, 1)
                        if len(parts) == 2 and len(parts[1].strip()) > 5:
                            all_texts.append(parts[1].strip())
                            break
                else:
                    # No separator found, use whole line if long enough
                    if len(line) > 20:
                        all_texts.append(line)

        except Exception as e:
            logger.error(f"Failed to read seed file {filepath}: {e}")

    # Deduplicate
    seen = set()
    unique = []
    for t in all_texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    if not unique:
        return 0

    return add_memories_batch(unique, user_id="medical_knowledge")


def seed_from_rag_results(rag_results: list, user_id: str = "rag_knowledge") -> int:
    """Extract key facts from RAG retrieval results and store in vector memory.

    Args:
        rag_results: List of dicts with 'content', 'score', 'source' keys
        user_id: Namespace for the stored memories

    Returns:
        Number of memories stored
    """
    if not rag_results:
        return 0

    texts = []
    for result in rag_results:
        content = result.get("content", "")
        source = result.get("source", "unknown")
        score = result.get("score", 0)

        if not content or len(content.strip()) < 30:
            continue

        # Only store high-relevance results
        if score < 0.5:
            continue

        # Extract key sentences (first 2-3 sentences as summary)
        sentences = re.split(r"[.。!！?？]", content)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        if key_sentences:
            summary = ". ".join(key_sentences)
            texts.append(f"[{source}] {summary}")

    if not texts:
        return 0

    return add_memories_batch(texts, user_id=user_id)


def store_conversation_summary(summary: str, user_id: str = "conversation") -> bool:
    """Store a conversation summary for future recall."""
    if not summary or len(summary.strip()) < 20:
        return False
    return add_memory(
        summary.strip(),
        user_id=user_id,
        metadata={"type": "conversation_summary", "stored_at": datetime.now().isoformat()},
    )


def format_memory_for_prompt(query: str = "", user_id: str | None = None, top_k: int = 8) -> str:
    """Format semantic search results for injection into system prompt.
    Falls back gracefully if vector memory unavailable.
    """
    if not query:
        # Fallback: return recent memories
        try:
            all_mems = get_all_memories(limit=top_k, user_id=user_id)
            if not all_mems:
                return ""
            lines = ["[MedicalMemory] Relevant stored knowledge:"]
            for i, mem in enumerate(all_mems, 1):
                lines.append(f"  {i}. {mem}")
            return "\n".join(lines)
        except Exception:
            return ""

    results = search_memory(query, user_id=user_id, top_k=top_k)
    if not results:
        return ""

    lines = ["[MedicalMemory] Semantically relevant memories:"]
    for i, mem in enumerate(results, 1):
        lines.append(f"  {i}. {mem}")
    return "\n".join(lines)


def collection_stats() -> dict:
    """Return basic stats about the memory collection."""
    if not _lazy_init():
        return {"error": _init_error}
    try:
        info = _qdrant_client.get_collection(_collection_name)
        return {
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count or 0,
            "status": str(info.status),
            "collection": _collection_name,
            "store_path": _qdrant_store_path,
        }
    except Exception as e:
        return {"error": str(e)}
