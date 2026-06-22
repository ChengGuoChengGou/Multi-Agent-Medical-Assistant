"""
conftest.py – Mock heavy dependencies before app.py import.

The app imports agents.rag_agent → langchain_qdrant → qdrant_client → etc.
which requires a full vector DB setup. For API-level tests we mock the entire
agent subsystem so only FastAPI routing / middleware logic is tested.
"""

import os
import sys
from unittest.mock import MagicMock

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Set test env vars BEFORE any project import ─────────────────────
os.environ.setdefault("XIAOMI_API_KEY", "test-key-not-real")
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("MILVUS_PORT", "19530")
os.environ.setdefault("ENABLED_AGENTS", "conversation_agent,report_agent")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")

# ── Mock heavy sub-modules before app.py is imported ─────────────────
# This avoids triggering the entire langchain / qdrant / MCP import chain.

_MOCK_PACKAGES = [
    # agent internals
    "agents.rag_agent",
    "agents.rag_agent.vectorstore_qdrant",
    "agents.agent_decision",
    "agents.mcp_client",
    # agents.memory_module removed from mock – pure dict, no heavy deps
    # langchain ecosystem (not needed for routing tests)
    "langchain_qdrant",
    "langchain.storage",
    "langchain_community.storage",
    # vector DB
    "qdrant_client",
    "qdrant_client.http",
    "qdrant_client.http.models",
    # MCP
    "mcp",
    "mcp.client",
    # edge-tts heavy deps
    "edge_tts",
]

for mod_name in _MOCK_PACKAGES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Provide process_query as an async-compatible mock so the /chat endpoint
# doesn't crash when called.
_process_query_mock = MagicMock()
_process_query_mock.return_value = {"status": "ok", "response": "mock reply"}
sys.modules["agents.agent_decision"].process_query = _process_query_mock
