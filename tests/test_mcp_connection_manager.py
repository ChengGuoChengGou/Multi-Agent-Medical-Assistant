"""Tests for agents/mcp_connection_manager.py — Resilient MCP Connection (Phase 58b)

Covers:
- RetryConfig: get_delay, backoff clamping
- TimeoutConfig: defaults
- ConnectionHealth: record_success, record_failure, failure_rate, health threshold
- ResilientMCPConnection: start, call_tool (retry/reconnect/timeout), health_check, stop
- ResilientMCPClientManager: add_server, start_all, stop_all, get_connection,
  get_all_tools, call_tool, call_on_server, health_check_all, get_health_summary
"""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove conftest-injected mock if present (conftest mocks agents.mcp_client)
for mod_key in list(sys.modules.keys()):
    if mod_key in ("agents.mcp_client", "agents.mcp_connection_manager") or mod_key.startswith(
        ("agents.mcp_client.", "agents.mcp_connection_manager.")
    ):
        del sys.modules[mod_key]

from agents.mcp_connection_manager import (
    ConnectionHealth,
    ResilientMCPClientManager,
    ResilientMCPConnection,
    RetryConfig,
    TimeoutConfig,
)


# ============================================================================
# RetryConfig
# ============================================================================
class TestRetryConfig:
    def test_default_values(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 30.0
        assert cfg.backoff_factor == 2.0
        assert cfg.retry_on_none is True

    def test_get_delay_attempt_0(self):
        """Attempt 0: 1.0 * 2^0 = 1.0"""
        cfg = RetryConfig(base_delay=1.0, backoff_factor=2.0)
        assert cfg.get_delay(0) == pytest.approx(1.0)

    def test_get_delay_attempt_1(self):
        """Attempt 1: 1.0 * 2^1 = 2.0"""
        cfg = RetryConfig(base_delay=1.0, backoff_factor=2.0)
        assert cfg.get_delay(1) == pytest.approx(2.0)

    def test_get_delay_attempt_2(self):
        """Attempt 2: 1.0 * 2^2 = 4.0"""
        cfg = RetryConfig(base_delay=1.0, backoff_factor=2.0)
        assert cfg.get_delay(2) == pytest.approx(4.0)

    def test_get_delay_clamped_at_max(self):
        """Large attempt clamped to max_delay."""
        cfg = RetryConfig(base_delay=1.0, backoff_factor=2.0, max_delay=5.0)
        assert cfg.get_delay(10) == pytest.approx(5.0)

    def test_custom_backoff(self):
        """Custom backoff_factor."""
        cfg = RetryConfig(base_delay=0.5, backoff_factor=3.0, max_delay=100.0)
        # attempt 2: 0.5 * 3^2 = 4.5
        assert cfg.get_delay(2) == pytest.approx(4.5)


# ============================================================================
# TimeoutConfig
# ============================================================================
class TestTimeoutConfig:
    def test_defaults(self):
        cfg = TimeoutConfig()
        assert cfg.init_timeout == 30.0
        assert cfg.tool_call_timeout == 60.0
        assert cfg.discovery_timeout == 15.0
        assert cfg.health_check_timeout == 5.0


# ============================================================================
# ConnectionHealth
# ============================================================================
class TestConnectionHealth:
    def test_initial_state(self):
        h = ConnectionHealth(server_name="test")
        assert h.is_healthy is False
        assert h.total_calls == 0
        assert h.failure_rate == 0.0
        assert h.consecutive_failures == 0

    def test_record_success(self):
        h = ConnectionHealth(server_name="test")
        h.record_success()
        assert h.is_healthy is True
        assert h.consecutive_failures == 0
        assert h.total_calls == 1
        assert h.last_success > 0

    def test_record_failure_marks_unhealthy_after_3(self):
        h = ConnectionHealth(server_name="test")
        h.is_healthy = True
        h.record_failure("e1")
        h.record_failure("e2")
        assert h.is_healthy is True  # still healthy after 2
        h.record_failure("e3")
        assert h.is_healthy is False  # unhealthy after 3 consecutive

    def test_record_success_resets_consecutive(self):
        h = ConnectionHealth(server_name="test")
        h.record_failure("e1")
        h.record_failure("e2")
        assert h.consecutive_failures == 2
        h.record_success()
        assert h.consecutive_failures == 0

    def test_failure_rate_calculation(self):
        h = ConnectionHealth(server_name="test")
        h.record_success()
        h.record_success()
        h.record_failure("e")
        h.record_failure("e")
        assert h.failure_rate == pytest.approx(0.5)

    def test_failure_rate_zero_calls(self):
        h = ConnectionHealth(server_name="test")
        assert h.failure_rate == 0.0

    def test_last_error_stored(self):
        h = ConnectionHealth(server_name="test")
        h.record_failure("boom")
        assert h.last_error == "boom"


# ============================================================================
# ResilientMCPConnection (async)
# ============================================================================
def _make_mock_connection(name="test_server", initialized=True, tools=None):
    """Create a mock MCPServerConnection."""
    conn = MagicMock()
    conn.name = name
    conn._initialized = initialized
    conn.tools = tools or []
    conn.start = AsyncMock(return_value=True)
    conn.stop = AsyncMock()
    conn.call_tool = AsyncMock(return_value=MagicMock(success=True, content="ok"))
    conn._send_request = AsyncMock(return_value={"result": "ok"})
    return conn


def _run(coro):
    """Run async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestResilientMCPConnection:
    def test_properties(self):
        conn = _make_mock_connection(name="bio")
        rc = ResilientMCPConnection(conn)
        assert rc.name == "bio"
        assert rc.is_initialized is True
        assert rc.health.server_name == "bio"

    def test_custom_configs(self):
        conn = _make_mock_connection()
        retry = RetryConfig(max_retries=5)
        timeout = TimeoutConfig(init_timeout=10.0)
        rc = ResilientMCPConnection(conn, retry_config=retry, timeout_config=timeout)
        assert rc._retry.max_retries == 5
        assert rc._timeouts.init_timeout == 10.0

    # --- start ---
    @pytest.mark.asyncio
    async def test_start_success(self):
        conn = _make_mock_connection()
        conn.start = AsyncMock(return_value=True)
        rc = ResilientMCPConnection(conn)
        result = await rc.start()
        assert result is True
        assert rc.health.is_healthy is True

    @pytest.mark.asyncio
    async def test_start_returns_false(self):
        conn = _make_mock_connection()
        conn.start = AsyncMock(return_value=False)
        rc = ResilientMCPConnection(conn)
        result = await rc.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_exception(self):
        conn = _make_mock_connection()
        conn.start = AsyncMock(side_effect=Exception("crash"))
        rc = ResilientMCPConnection(conn)
        result = await rc.start()
        assert result is False
        assert rc.health.total_failures >= 1

    # --- call_tool ---
    @pytest.mark.asyncio
    async def test_call_tool_success_first_try(self):
        """Successful call on first attempt, no retries."""
        mock_result = MagicMock(success=True, content="data")
        conn = _make_mock_connection()
        conn.call_tool = AsyncMock(return_value=mock_result)
        rc = ResilientMCPConnection(conn, retry_config=RetryConfig(max_retries=3))
        result = await rc.call_tool("search", {"q": "aspirin"})
        assert result.success is True
        assert conn.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_call_tool_retries_on_none(self):
        """Returns None twice then succeeds."""
        mock_result = MagicMock(success=True, content="ok")
        conn = _make_mock_connection()
        conn.call_tool = AsyncMock(side_effect=[None, None, mock_result])
        rc = ResilientMCPConnection(conn, retry_config=RetryConfig(max_retries=3, base_delay=0.01))
        result = await rc.call_tool("t", {})
        assert result.success is True
        assert conn.call_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_call_tool_all_retries_fail_returns_error_result(self):
        """All retries return None → MCPToolResult with error."""
        conn = _make_mock_connection()
        conn.call_tool = AsyncMock(return_value=None)
        conn.stop = AsyncMock()
        conn.start = AsyncMock(return_value=False)  # reconnect fails
        rc = ResilientMCPConnection(conn, retry_config=RetryConfig(max_retries=1, base_delay=0.01))
        result = await rc.call_tool("bad_tool", {})
        assert result.success is False
        assert "All attempts failed" in result.error

    @pytest.mark.asyncio
    async def test_call_tool_timeout_triggers_retry(self):
        """Timeout on first attempt, success on second."""
        mock_result = MagicMock(success=True, content="ok")
        conn = _make_mock_connection()
        conn.call_tool = AsyncMock(side_effect=[asyncio.TimeoutError, mock_result])
        rc = ResilientMCPConnection(conn, retry_config=RetryConfig(max_retries=2, base_delay=0.01))
        result = await rc.call_tool("t", {})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_tool_exception_triggers_retry(self):
        """Exception on first attempt, success on second."""
        mock_result = MagicMock(success=True, content="ok")
        conn = _make_mock_connection()
        conn.call_tool = AsyncMock(side_effect=[RuntimeError("oops"), mock_result])
        rc = ResilientMCPConnection(conn, retry_config=RetryConfig(max_retries=2, base_delay=0.01))
        result = await rc.call_tool("t", {})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_tool_records_failure_on_failed_result(self):
        """Result.success=False records failure in health."""
        mock_result = MagicMock(success=False, error="not found")
        mock_result.success = False
        conn = _make_mock_connection()
        conn.call_tool = AsyncMock(return_value=mock_result)
        rc = ResilientMCPConnection(conn, retry_config=RetryConfig(max_retries=0))
        result = await rc.call_tool("t", {})
        assert result.success is False
        assert rc.health.total_failures >= 1

    # --- health_check ---
    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        conn = _make_mock_connection(initialized=True)
        conn._send_request = AsyncMock(return_value={"result": []})
        rc = ResilientMCPConnection(conn)
        result = await rc.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        conn = _make_mock_connection(initialized=False)
        rc = ResilientMCPConnection(conn)
        result = await rc.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_none(self):
        conn = _make_mock_connection(initialized=True)
        conn._send_request = AsyncMock(return_value=None)
        rc = ResilientMCPConnection(conn)
        result = await rc.health_check()
        assert result is False

    # --- stop ---
    @pytest.mark.asyncio
    async def test_stop(self):
        conn = _make_mock_connection()
        conn.stop = AsyncMock()
        rc = ResilientMCPConnection(conn)
        await rc.stop()
        conn.stop.assert_called_once()
        assert rc.health.is_healthy is False


# ============================================================================
# ResilientMCPClientManager (async)
# ============================================================================
class TestResilientMCPClientManager:
    def test_add_server(self):
        mgr = ResilientMCPClientManager()
        conn = _make_mock_connection(name="bio")
        mgr.add_server(conn)
        assert "bio" in mgr._connections

    def test_add_server_custom_configs(self):
        mgr = ResilientMCPClientManager()
        conn = _make_mock_connection(name="icd")
        retry = RetryConfig(max_retries=5)
        mgr.add_server(conn, retry_config=retry)
        assert mgr._connections["icd"]._retry.max_retries == 5

    def test_get_connection(self):
        mgr = ResilientMCPClientManager()
        conn = _make_mock_connection(name="bio")
        mgr.add_server(conn)
        assert mgr.get_connection("bio") is not None
        assert mgr.get_connection("missing") is None

    @pytest.mark.asyncio
    async def test_start_all(self):
        mgr = ResilientMCPClientManager()
        c1 = _make_mock_connection(name="s1")
        c1.start = AsyncMock(return_value=True)
        c2 = _make_mock_connection(name="s2")
        c2.start = AsyncMock(return_value=True)
        mgr.add_server(c1)
        mgr.add_server(c2)
        results = await mgr.start_all()
        assert results["s1"] is True
        assert results["s2"] is True

    @pytest.mark.asyncio
    async def test_stop_all(self):
        mgr = ResilientMCPClientManager()
        c1 = _make_mock_connection(name="s1")
        c1.stop = AsyncMock()
        c2 = _make_mock_connection(name="s2")
        c2.stop = AsyncMock()
        mgr.add_server(c1)
        mgr.add_server(c2)
        await mgr.stop_all()
        c1.stop.assert_called()
        c2.stop.assert_called()

    def test_get_all_tools(self):
        mgr = ResilientMCPClientManager()
        tool1 = MagicMock(name="search")
        tool2 = MagicMock(name="lookup")
        c1 = _make_mock_connection(name="s1", initialized=True, tools=[tool1])
        c2 = _make_mock_connection(name="s2", initialized=True, tools=[tool2])
        mgr.add_server(c1)
        mgr.add_server(c2)
        all_tools = mgr.get_all_tools()
        assert len(all_tools) == 2

    def test_get_all_tools_skips_uninitialized(self):
        mgr = ResilientMCPClientManager()
        c1 = _make_mock_connection(name="s1", initialized=True, tools=[MagicMock()])
        c2 = _make_mock_connection(name="s2", initialized=False, tools=[MagicMock()])
        mgr.add_server(c1)
        mgr.add_server(c2)
        assert len(mgr.get_all_tools()) == 1

    @pytest.mark.asyncio
    async def test_call_tool_found(self):
        tool = MagicMock()
        tool.name = "search"
        mock_result = MagicMock(success=True, content="r")
        conn = _make_mock_connection(name="s1", tools=[tool])
        conn.call_tool = AsyncMock(return_value=mock_result)
        mgr = ResilientMCPClientManager()
        mgr.add_server(conn)
        result = await mgr.call_tool("search", {"q": "test"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        mgr = ResilientMCPClientManager()
        conn = _make_mock_connection(name="s1", tools=[])
        mgr.add_server(conn)
        result = await mgr.call_tool("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_on_server_found(self):
        mock_result = MagicMock(success=True, content="r")
        conn = _make_mock_connection(name="s1")
        conn.call_tool = AsyncMock(return_value=mock_result)
        mgr = ResilientMCPClientManager()
        mgr.add_server(conn)
        result = await mgr.call_on_server("s1", "tool", {})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_on_server_not_found(self):
        mgr = ResilientMCPClientManager()
        result = await mgr.call_on_server("missing", "tool", {})
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        mgr = ResilientMCPClientManager()
        c1 = _make_mock_connection(name="s1", initialized=True)
        c1._send_request = AsyncMock(return_value={"result": []})
        mgr.add_server(c1)
        results = await mgr.health_check_all()
        assert "s1" in results
        assert isinstance(results["s1"], bool)

    def test_get_health_summary(self):
        mgr = ResilientMCPClientManager()
        conn = _make_mock_connection(name="s1")
        mgr.add_server(conn)
        summary = mgr.get_health_summary()
        assert "s1" in summary
        assert "healthy" in summary["s1"]
        assert "failure_rate" in summary["s1"]
