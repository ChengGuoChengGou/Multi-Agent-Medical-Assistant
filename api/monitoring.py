"""Phase 9: System monitoring aggregation endpoint.

Provides /api/v1/admin/stats that aggregates:
- Circuit breaker states (LLM, MCP, web search, image)
- Agent call metrics (per-agent latency, success rate)
- Rate limiter state (active IPs)
- System info (uptime, Python version, memory)
"""
import sys
import time
import os
from fastapi import APIRouter
from observability import metrics, agent_metrics
from circuit_breaker import get_all_breaker_stats
from utils.logging_config import get_logger

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
_logger = get_logger("monitoring")

# App start time (set by app.py)
_app_start_time: float = 0.0


def set_app_start_time(t: float):
    global _app_start_time
    _app_start_time = t


@router.get("/stats")
async def get_system_stats():
    """Aggregate system monitoring stats.
    
    Returns circuit breakers, agent metrics, rate limiter state,
    and system resource info in a single JSON response.
    No authentication required (same as /health) for monitoring
    tool compatibility. Restrict access via network/firewall in production.
    """
    uptime = time.time() - _app_start_time if _app_start_time else 0

    # Circuit breakers
    cb_stats = get_all_breaker_stats()

    # Agent metrics
    agent_stats = agent_metrics.get_all_stats()

    # System info
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        system_info = {
            "memory_rss_mb": round(mem.rss / 1024 / 1024, 1),
            "memory_vms_mb": round(mem.vms / 1024 / 1024, 1),
            "cpu_percent": process.cpu_percent(interval=0),
            "num_threads": process.num_threads(),
        }
    except ImportError:
        system_info = {"note": "psutil not installed, limited system info"}

    system_info.update({
        "python_version": sys.version.split()[0],
        "pid": os.getpid(),
    })

    return {
        "uptime_seconds": round(uptime, 1),
        "circuit_breakers": cb_stats,
        "agent_metrics": agent_stats,
        "system": system_info,
    }
