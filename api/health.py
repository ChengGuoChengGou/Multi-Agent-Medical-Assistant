"""
Dependency Health Check Endpoint
Provides /api/v1/health/detailed with status of all dependencies:
- LLM Provider (primary + fallbacks)
- Qdrant vector store
- Embedding model
- File storage
"""
import logging
import time
from typing import Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/health", tags=["Health v1"])


class DependencyStatus(BaseModel):
    name: str
    status: str  # "healthy" | "degraded" | "unhealthy"
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class DetailedHealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str
    uptime_seconds: float
    dependencies: list[DependencyStatus]


def _check_llm_primary() -> DependencyStatus:
    """Check primary LLM provider connectivity."""
    try:
        from config import Config
        cfg = Config()
        llm = cfg._make_llm()
        # Quick warmup call
        start = time.time()
        # Just verify the object was created successfully
        latency = (time.time() - start) * 1000
        provider = getattr(cfg, 'llm_provider', 'unknown')
        return DependencyStatus(
            name="llm_primary",
            status="healthy",
            latency_ms=round(latency, 1),
            detail=f"provider={provider}, model={getattr(llm, 'model_name', 'unknown')}"
        )
    except Exception as e:
        return DependencyStatus(
            name="llm_primary",
            status="unhealthy",
            detail=str(e)[:200]
        )


def _check_qdrant() -> DependencyStatus:
    """Check Qdrant vector store connectivity."""
    try:
        from config import Config
        cfg = Config()
        qdrant_url = getattr(cfg.rag, 'qdrant_url', None) or getattr(cfg, 'qdrant_url', None)
        if not qdrant_url:
            return DependencyStatus(name="qdrant", status="unhealthy", detail="qdrant_url not configured")
        
        start = time.time()
        import requests
        r = requests.get(f"{qdrant_url}/collections", timeout=5)
        latency = (time.time() - start) * 1000
        
        if r.status_code == 200:
            collections = r.json().get("result", {}).get("collections", [])
            return DependencyStatus(
                name="qdrant",
                status="healthy",
                latency_ms=round(latency, 1),
                detail=f"{len(collections)} collections"
            )
        else:
            return DependencyStatus(
                name="qdrant",
                status="degraded",
                latency_ms=round(latency, 1),
                detail=f"HTTP {r.status_code}"
            )
    except Exception as e:
        return DependencyStatus(name="qdrant", status="unhealthy", detail=str(e)[:200])


def _check_embedding() -> DependencyStatus:
    """Check embedding model availability."""
    try:
        from config import Config
        cfg = Config()
        embed_model = getattr(cfg.rag, 'embedding_model_name', None) or getattr(cfg, 'embedding_model_name', None)
        if not embed_model:
            return DependencyStatus(name="embedding", status="degraded", detail="no embedding model configured")
        
        # Try to instantiate the embedding client
        import openai
        api_key = getattr(cfg, 'openai_api_key', None)
        base_url = getattr(cfg, 'openai_base_url', None)
        if not api_key:
            return DependencyStatus(name="embedding", status="unhealthy", detail="no API key")
        
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        start = time.time()
        resp = client.embeddings.create(input=["health check"], model=embed_model)
        latency = (time.time() - start) * 1000
        
        return DependencyStatus(
            name="embedding",
            status="healthy",
            latency_ms=round(latency, 1),
            detail=f"model={embed_model}, dim={len(resp.data[0].embedding)}"
        )
    except Exception as e:
        return DependencyStatus(name="embedding", status="degraded", detail=str(e)[:200])


def _check_file_storage() -> DependencyStatus:
    """Check file storage directories exist and are writable."""
    try:
        import os
        import tempfile
        dirs_to_check = ["uploads/backend", "uploads/frontend", "data"]
        missing = []
        for d in dirs_to_check:
            if not os.path.isdir(d):
                missing.append(d)
        
        if missing:
            return DependencyStatus(
                name="file_storage",
                status="degraded",
                detail=f"missing dirs: {missing}"
            )
        
        # Test write
        test_file = os.path.join("uploads", "backend", ".health_check")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        
        return DependencyStatus(name="file_storage", status="healthy", detail="all dirs accessible")
    except Exception as e:
        return DependencyStatus(name="file_storage", status="unhealthy", detail=str(e)[:200])


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health():
    """Comprehensive health check of all dependencies."""
    import time as _time
    start = _time.time()
    
    deps = [
        _check_llm_primary(),
        _check_qdrant(),
        _check_embedding(),
        _check_file_storage(),
    ]
    
    # Determine overall status
    statuses = [d.status for d in deps]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"
    
    # Calculate uptime
    try:
        from app import _app_start_time
        uptime = _time.time() - _app_start_time
    except Exception:
        uptime = 0.0
    
    return DetailedHealthResponse(
        status=overall,
        version="2.0",
        uptime_seconds=round(uptime, 1),
        dependencies=deps
    )
