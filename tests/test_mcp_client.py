"""Tests for agents/mcp_client.py — MCP Client Module (Phase 58b)

ROOT CAUSE FIX: conftest.py auto-mocks agents.mcp_client in sys.modules.
We MUST remove it before importing the real module.

Covers:
- MCPTool / MCPToolResult dataclasses
- MCPServerConnection: start, call_tool, stop, _next_id, _send_request
- MCPClientManager: register, start_all, stop_all, get_all_tools, call_tool, call_on_server, get_tools_summary
- create_default_mcp_client: server registration
- get_mcp_client / shutdown_mcp_client: singleton lifecycle
"""

import asyncio
import json
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Undo conftest's MagicMock for agents.mcp_client ──
# conftest.py inserts MagicMock() into sys.modules to avoid heavy imports.
# Since we WANT to test mcp_client.py, we must remove the mock first.
_project_root = r"D:\Code\Multi-Agent-Medical-Assistant"
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Remove conftest's mock so the real module loads
for _k in list(sys.modules):
    if _k == "agents.mcp_client" or _k.startswith("agents.mcp_client."):
        del sys.modules[_k]

# Now import the REAL module
from agents.mcp_client import (
    MCPTool,
    MCPToolResult,
    MCPServerConnection,
    MCPClientManager,
    create_default_mcp_client,
    get_project_mcp_dir,
)


# ============================================================================
# MCPTool / MCPToolResult dataclasses
# ============================================================================
class TestDataclasses:
    def test_mcp_tool_fields(self):
        t = MCPTool(name="search", description="Search tool", input_schema={"type": "object"}, server_name="biomcp")
        assert t.name == "search"
        assert t.server_name == "biomcp"
        assert t.description == "Search tool"

    def test_mcp_tool_result_success(self):
        r = MCPToolResult(success=True, content="data", tool_name="t", server_name="s")
        assert r.success is True
        assert r.error is None

    def test_mcp_tool_result_failure(self):
        r = MCPToolResult(success=False, content="", tool_name="t", server_name="s", error="timeout")
        assert r.success is False
        assert r.error == "timeout"

    def test_mcp_tool_result_defaults(self):
        r = MCPToolResult(success=True, content="", tool_name="t", server_name="s")
        assert r.error is None


# ============================================================================
# MCPServerConnection
# ============================================================================
class TestMCPServerConnection:
    def setup_method(self):
        self.conn = MCPServerConnection(
            name="test_server", command="python", args=["-m", "test"],
            env={"KEY": "val"}, cwd="/tmp"
        )

    def test_init_defaults(self):
        assert self.conn.name == "test_server"
        assert self.conn.process is None
        assert self.conn.tools == []
        assert self.conn._initialized is False
        assert self.conn._request_id == 0

    def test_server_connection_minimal_init(self):
        conn = MCPServerConnection(name="n", command="c", args=[])
        assert conn.name == "n"

    def test_next_id_incremental(self):
        assert self.conn._next_id() == 1
        assert self.conn._next_id() == 2
        assert self.conn._next_id() == 3

    @pytest.mark.asyncio
    async def test_send_request_no_process(self):
        result = await self.conn._send_request("test/method", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_send_notification_no_process(self):
        await self.conn._send_notification("test", {})  # should not raise

    @pytest.mark.asyncio
    async def test_send_request_empty_readline(self):
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = lambda: ""
        self.conn.process = mock_process

        result = await self.conn._send_request("test", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_start_success(self):
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()

        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}) + "\n"
        tools_resp = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {
            "tools": [
                {"name": "search", "description": "Search", "inputSchema": {"type": "object"}}
            ]
        }}) + "\n"

        call_count = 0
        def mock_readline():
            nonlocal call_count
            call_count += 1
            return init_resp if call_count == 1 else tools_resp

        mock_process.stdout.readline = mock_readline

        with patch("subprocess.Popen", return_value=mock_process):
            result = await self.conn.start()

        assert result is True
        assert self.conn._initialized is True
        assert len(self.conn.tools) == 1
        assert self.conn.tools[0].name == "search"

    @pytest.mark.asyncio
    async def test_start_init_failure(self):
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        err_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "fail"}}) + "\n"
        mock_process.stdout.readline = lambda: err_resp

        with patch("subprocess.Popen", return_value=mock_process):
            result = await self.conn.start()

        assert result is False
        assert self.conn._initialized is False

    @pytest.mark.asyncio
    async def test_start_exception(self):
        with patch("subprocess.Popen", side_effect=OSError("not found")):
            result = await self.conn.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_call_tool_not_initialized(self):
        result = await self.conn.call_tool("search", {"q": "test"})
        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        self.conn._initialized = True
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        tool_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
            "content": [{"type": "text", "text": "result data"}]
        }}) + "\n"
        mock_process.stdout.readline = lambda: tool_resp
        self.conn.process = mock_process

        result = await self.conn.call_tool("search", {"q": "test"})
        assert result.success is True
        assert "result data" in result.content

    @pytest.mark.asyncio
    async def test_call_tool_no_response(self):
        self.conn._initialized = True
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = lambda: ""
        self.conn.process = mock_process

        result = await self.conn.call_tool("search", {"q": "test"})
        assert result.success is False
        assert "no response" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_tool_multiple_content_parts(self):
        self.conn._initialized = True
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        tool_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
            "content": [
                {"type": "text", "text": "part1"},
                {"type": "text", "text": "part2"},
                {"type": "image", "data": "base64"},
            ]
        }}) + "\n"
        mock_process.stdout.readline = lambda: tool_resp
        self.conn.process = mock_process

        result = await self.conn.call_tool("tool", {})
        assert result.success is True
        assert "part1" in result.content
        assert "part2" in result.content

    @pytest.mark.asyncio
    async def test_call_tool_empty_content(self):
        self.conn._initialized = True
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": []}}) + "\n"
        mock_process.stdout.readline = lambda: resp
        self.conn.process = mock_process

        result = await self.conn.call_tool("t", {})
        assert result.success is True
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self):
        """_send_request returns None on error response, so call_tool reports 'No response'."""
        self.conn._initialized = True
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        err_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "tool error"}}) + "\n"
        mock_process.stdout.readline = lambda: err_resp
        self.conn.process = mock_process

        result = await self.conn.call_tool("bad_tool", {})
        assert result.success is False
        assert "no response" in result.error.lower()  # _send_request returns None on error

    @pytest.mark.asyncio
    async def test_call_tool_bad_json(self):
        self.conn._initialized = True
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = lambda: "not json\n"
        self.conn.process = mock_process

        result = await self.conn.call_tool("t", {})
        assert isinstance(result, MCPToolResult)

    @pytest.mark.asyncio
    async def test_stop_with_process(self):
        mock_process = MagicMock()
        mock_process.wait = MagicMock()
        self.conn.process = mock_process
        self.conn._initialized = True

        await self.conn.stop()
        mock_process.terminate.assert_called_once()
        assert self.conn.process is None
        assert self.conn._initialized is False

    @pytest.mark.asyncio
    async def test_stop_terminate_timeout_kills(self):
        mock_process = MagicMock()
        mock_process.wait.side_effect = Exception("timeout")
        self.conn.process = mock_process

        await self.conn.stop()
        mock_process.kill.assert_called_once()
        assert self.conn.process is None

    @pytest.mark.asyncio
    async def test_stop_no_process(self):
        self.conn.process = None
        await self.conn.stop()  # no-op


# ============================================================================
# MCPClientManager
# ============================================================================
class TestMCPClientManager:
    def test_register_server(self):
        mgr = MCPClientManager()
        mgr.register_server("s1", "cmd", ["a"])
        assert "s1" in mgr.servers
        assert isinstance(mgr.servers["s1"], MCPServerConnection)

    def test_register_server_with_env(self):
        mgr = MCPClientManager()
        mgr.register_server("s1", "cmd", [], env={"K": "V"})
        assert mgr.servers["s1"].env["K"] == "V"

    def test_register_overwrites(self):
        mgr = MCPClientManager()
        mgr.register_server("s", "cmd1", ["a"])
        mgr.register_server("s", "cmd2", ["b"])
        assert mgr.servers["s"].command == "cmd2"

    def test_get_all_tools_empty(self):
        mgr = MCPClientManager()
        assert mgr.get_all_tools() == []

    def test_get_all_tools_with_manual_add(self):
        mgr = MCPClientManager()
        t = MCPTool(name="t1", description="d", input_schema={}, server_name="s1")
        mgr._all_tools.append(t)
        assert len(mgr.get_all_tools()) == 1

    def test_get_tools_by_server(self):
        mgr = MCPClientManager()
        mgr._all_tools = [
            MCPTool(name="t1", description="d", input_schema={}, server_name="s1"),
            MCPTool(name="t2", description="d", input_schema={}, server_name="s2"),
            MCPTool(name="t3", description="d", input_schema={}, server_name="s1"),
        ]
        tools = mgr.get_tools_by_server("s1")
        assert len(tools) == 2
        assert all(t.server_name == "s1" for t in tools)

    def test_get_tools_by_server_empty(self):
        mgr = MCPClientManager()
        assert mgr.get_tools_by_server("missing") == []

    def test_get_tools_summary_format(self):
        mgr = MCPClientManager()
        mgr.register_server("s1", "cmd", [])
        mgr.servers["s1"]._initialized = True
        mgr.servers["s1"].tools = [
            MCPTool(name="search", description="Search medical data", input_schema={}, server_name="s1")
        ]
        mgr._all_tools = list(mgr.servers["s1"].tools)

        summary = mgr.get_tools_summary()
        assert "Available MCP Tools" in summary
        assert "\u2705" in summary
        assert "search" in summary

    def test_get_tools_summary_uninitialized(self):
        mgr = MCPClientManager()
        mgr.register_server("s1", "cmd", [])
        summary = mgr.get_tools_summary()
        assert "\u274c" in summary

    @pytest.mark.asyncio
    async def test_start_all(self):
        mgr = MCPClientManager()

        s1 = MCPServerConnection(name="s1", command="cmd", args=[])
        s1.start = AsyncMock(return_value=True)
        s1.tools = [MCPTool(name="t1", description="d", input_schema={}, server_name="s1")]

        s2 = MCPServerConnection(name="s2", command="cmd", args=[])
        s2.start = AsyncMock(return_value=False)
        s2.tools = []

        mgr.servers = {"s1": s1, "s2": s2}
        results = await mgr.start_all()

        assert results == {"s1": True, "s2": False}
        assert mgr._started is True
        assert len(mgr._all_tools) == 1

    @pytest.mark.asyncio
    async def test_stop_all(self):
        mgr = MCPClientManager()
        s1 = MCPServerConnection(name="s1", command="cmd", args=[])
        s1.stop = AsyncMock()
        mgr.servers = {"s1": s1}
        mgr._started = True

        await mgr.stop_all()
        s1.stop.assert_awaited_once()
        assert mgr._started is False

    @pytest.mark.asyncio
    async def test_call_tool_found(self):
        mgr = MCPClientManager()
        conn = MCPServerConnection(name="s1", command="cmd", args=[])
        conn.tools = [MCPTool(name="search", description="d", input_schema={}, server_name="s1")]
        conn.call_tool = AsyncMock(return_value=MCPToolResult(
            success=True, content="ok", tool_name="search", server_name="s1"
        ))
        mgr.servers = {"s1": conn}
        mgr._all_tools = list(conn.tools)

        result = await mgr.call_tool("search", {"q": "test"})
        assert result.success is True
        conn.call_tool.assert_awaited_once_with("search", {"q": "test"})

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        mgr = MCPClientManager()
        result = await mgr.call_tool("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_on_server_found(self):
        mgr = MCPClientManager()
        conn = MCPServerConnection(name="s1", command="cmd", args=[])
        conn.call_tool = AsyncMock(return_value=MCPToolResult(
            success=True, content="data", tool_name="t", server_name="s1"
        ))
        mgr.servers = {"s1": conn}

        result = await mgr.call_on_server("s1", "t", {"a": 1})
        assert result.success is True
        conn.call_tool.assert_awaited_once_with("t", {"a": 1})

    @pytest.mark.asyncio
    async def test_call_on_server_not_found(self):
        mgr = MCPClientManager()
        result = await mgr.call_on_server("missing", "t", {})
        assert result.success is False
        assert "not found" in result.error.lower()


# ============================================================================
# create_default_mcp_client
# ============================================================================
class TestCreateDefaultMCPClient:
    def test_registers_three_servers(self):
        client = create_default_mcp_client()
        assert len(client.servers) == 3
        assert "biomcp" in client.servers
        assert "autoicd" in client.servers
        assert "healthcare" in client.servers

    def test_biomcp_command(self):
        client = create_default_mcp_client()
        biomcp = client.servers["biomcp"]
        assert biomcp.args == ["serve"]

    def test_autoicd_command(self):
        client = create_default_mcp_client()
        autoicd = client.servers["autoicd"]
        assert "-y" in autoicd.args
        assert "autoicd-mcp" in autoicd.args

    def test_autoicd_env_has_api_key(self):
        client = create_default_mcp_client()
        autoicd = client.servers["autoicd"]
        assert "AUTOICD_API_KEY" in autoicd.env

    def test_healthcare_command(self):
        client = create_default_mcp_client()
        hc = client.servers["healthcare"]
        assert hc.command == "node"
        assert "index.js" in hc.args

    def test_project_mcp_dir(self):
        mcp_dir = get_project_mcp_dir()
        assert mcp_dir.endswith("mcp_servers")
        assert os.path.isabs(mcp_dir)


# ============================================================================
# Singleton: get_mcp_client / shutdown_mcp_client
# ============================================================================
class TestSingleton:
    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        import agents.mcp_client as mc_mod
        mc_mod._global_client = None
        yield
        mc_mod._global_client = None

    @pytest.mark.asyncio
    async def test_shutdown_mcp_client(self):
        import agents.mcp_client as mc_mod
        mock_client = MagicMock()
        mock_client.stop_all = AsyncMock()
        mc_mod._global_client = mock_client

        await mc_mod.shutdown_mcp_client()
        mock_client.stop_all.assert_awaited_once()
        assert mc_mod._global_client is None

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_none(self):
        import agents.mcp_client as mc_mod
        mc_mod._global_client = None
        await mc_mod.shutdown_mcp_client()


# ============================================================================
# Edge cases
# ============================================================================
class TestEdgeCases:
    def test_manager_register_overwrites(self):
        mgr = MCPClientManager()
        mgr.register_server("s", "cmd1", ["a"])
        mgr.register_server("s", "cmd2", ["b"])
        assert mgr.servers["s"].command == "cmd2"
