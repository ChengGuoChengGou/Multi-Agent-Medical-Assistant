"""
Unit tests for dependencies.py — DI container.
Run: python -m pytest tests/test_dependencies.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dependencies


# ── get/set mcp_client ──
class TestMcpClient:
    def setup_method(self):
        dependencies._mcp_client = None

    def test_get_mcp_client_default_none(self):
        assert dependencies.get_mcp_client() is None

    def test_set_mcp_client(self):
        fake_client = {"type": "fake"}
        dependencies.set_mcp_client(fake_client)
        assert dependencies.get_mcp_client() == fake_client

    def test_set_mcp_client_overwrite(self):
        dependencies.set_mcp_client("first")
        dependencies.set_mcp_client("second")
        assert dependencies.get_mcp_client() == "second"


# ── get_cache_service ──
class TestCacheService:
    def setup_method(self):
        dependencies._cache_service = None

    def test_get_cache_service_returns_dict(self):
        result = dependencies.get_cache_service()
        assert isinstance(result, dict)
        assert "get" in result
        assert "set" in result
        assert callable(result["get"])
        assert callable(result["set"])
