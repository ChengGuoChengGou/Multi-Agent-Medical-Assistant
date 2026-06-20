"""
MCP Client Module for Multi-Agent Medical Assistant

Provides a unified interface to communicate with multiple MCP servers:
- BioMCP: Biomedical data (gene, variant, article, trial, drug, disease, etc.)
- AutoICD: Medical coding (ICD-10-CM, ICD-11, ICF, LOINC, SNOMED CT)
- Healthcare MCP: FDA drugs, PubMed, clinical trials, health topics, medRxiv

Uses subprocess + stdio protocol (MCP standard) to communicate with servers.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents a tool provided by an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""
    success: bool
    content: str
    tool_name: str
    server_name: str
    error: Optional[str] = None


class MCPServerConnection:
    """Manages a single MCP server subprocess connection via stdio."""

    def __init__(self, name: str, command: str, args: List[str],
                 env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.cwd = cwd
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[MCPTool] = []
        self._initialized = False
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def start(self) -> bool:
        """Start the MCP server subprocess."""
        try:
            full_env = {**os.environ, **self.env}
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=full_env,
                cwd=self.cwd,
                text=True,
                bufsize=1,
            )
            # Send initialize request
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "medical-assistant", "version": "1.0.0"}
            })
            if init_result is not None:
                # Send initialized notification
                await self._send_notification("notifications/initialized", {})
                self._initialized = True
                # List available tools
                await self._discover_tools()
                logger.info(f"[MCP:{self.name}] Started with {len(self.tools)} tools")
                return True
            else:
                logger.error(f"[MCP:{self.name}] Failed to initialize")
                return False
        except Exception as e:
            logger.error(f"[MCP:{self.name}] Failed to start: {e}")
            return False

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Send a JSON-RPC request and wait for response."""
        if not self.process:
            return None

        req_id = self._next_id()
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        try:
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str)
            self.process.stdin.flush()

            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                asyncio.to_thread(self.process.stdout.readline),
                timeout=30.0
            )
            if response_line:
                response = json.loads(response_line.strip())
                if "result" in response:
                    return response["result"]
                elif "error" in response:
                    logger.error(f"[MCP:{self.name}] Error: {response['error']}")
                    return None
            return None
        except asyncio.TimeoutError:
            logger.error(f"[MCP:{self.name}] Request timeout for {method}")
            return None
        except Exception as e:
            logger.error(f"[MCP:{self.name}] Request error: {e}")
            return None

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            self.process.stdin.write(json.dumps(notification) + "\n")
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"[MCP:{self.name}] Notification error: {e}")

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        result = await self._send_request("tools/list", {})
        if result and "tools" in result:
            for tool_def in result["tools"]:
                self.tools.append(MCPTool(
                    name=tool_def.get("name", ""),
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                    server_name=self.name,
                ))

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call a tool on this server."""
        if not self._initialized:
            return MCPToolResult(
                success=False, content="", tool_name=tool_name,
                server_name=self.name, error="Server not initialized"
            )

        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        if result is None:
            return MCPToolResult(
                success=False, content="", tool_name=tool_name,
                server_name=self.name, error="No response from server"
            )

        # Extract content from MCP result
        content_parts = []
        if "content" in result:
            for item in result["content"]:
                if item.get("type") == "text":
                    content_parts.append(item.get("text", ""))

        return MCPToolResult(
            success=True,
            content="\n".join(content_parts),
            tool_name=tool_name,
            server_name=self.name,
        )

    async def stop(self) -> None:
        """Stop the MCP server subprocess."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
            self._initialized = False


class MCPClientManager:
    """
    Manages connections to multiple MCP servers and provides
    a unified interface for tool discovery and invocation.
    """

    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self._all_tools: List[MCPTool] = []
        self._started = False

    def register_server(self, name: str, command: str, args: List[str],
                       env: Optional[Dict[str, str]] = None,
                       cwd: Optional[str] = None) -> None:
        """Register an MCP server configuration."""
        self.servers[name] = MCPServerConnection(
            name=name, command=command, args=args, env=env, cwd=cwd
        )

    async def start_all(self) -> Dict[str, bool]:
        """Start all registered servers. Returns dict of server_name -> success."""
        results = {}
        for name, server in self.servers.items():
            success = await server.start()
            results[name] = success
            if success:
                self._all_tools.extend(server.tools)
        self._started = True
        return results

    async def stop_all(self) -> None:
        """Stop all servers."""
        for server in self.servers.values():
            await server.stop()
        self._started = False

    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools across all servers."""
        return self._all_tools

    def get_tools_by_server(self, server_name: str) -> List[MCPTool]:
        """Get tools from a specific server."""
        return [t for t in self._all_tools if t.server_name == server_name]

    def get_tools_summary(self) -> str:
        """Get a human-readable summary of all available tools."""
        lines = ["Available MCP Tools:"]
        for name, server in self.servers.items():
            status = "✅" if server._initialized else "❌"
            lines.append(f"\n{status} {name} ({len(server.tools)} tools):")
            for tool in server.tools:
                lines.append(f"  - {tool.name}: {tool.description[:100]}")
        return "\n".join(lines)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call a tool by name. Automatically finds the right server."""
        for server in self.servers.values():
            for tool in server.tools:
                if tool.name == tool_name:
                    return await server.call_tool(tool_name, arguments)

        return MCPToolResult(
            success=False, content="", tool_name=tool_name,
            server_name="unknown", error=f"Tool '{tool_name}' not found"
        )

    async def call_on_server(self, server_name: str, tool_name: str,
                            arguments: Dict[str, Any]) -> MCPToolResult:
        """Call a tool on a specific server."""
        server = self.servers.get(server_name)
        if not server:
            return MCPToolResult(
                success=False, content="", tool_name=tool_name,
                server_name=server_name, error=f"Server '{server_name}' not found"
            )
        return await server.call_tool(tool_name, arguments)


# ============================================================
# Default MCP Client Configuration
# ============================================================

def get_project_mcp_dir() -> str:
    """Get the path to the mcp_servers directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_servers")


def create_default_mcp_client() -> MCPClientManager:
    """
    Create and configure the default MCP client with all three servers.
    
    Server paths are relative to the project's mcp_servers/ directory.
    """
    mcp_dir = get_project_mcp_dir()
    client = MCPClientManager()

    # 1. BioMCP - Biomedical data (Python, via pip biomcp-cli)
    biomcp_dir = os.path.join(mcp_dir, "biomcp")
    client.register_server(
        name="biomcp",
        command="biomcp",
        args=["serve"],
        cwd=biomcp_dir,
    )

    # 2. AutoICD - Medical coding (Node.js, via npx)
    autoicd_dir = os.path.join(mcp_dir, "autoicd-mcp")
    autoicd_env = {
        "AUTOICD_API_KEY": os.environ.get("AUTOICD_API_KEY", ""),
    }
    client.register_server(
        name="autoicd",
        command="npx",
        args=["-y", "autoicd-mcp"],
        env=autoicd_env,
        cwd=autoicd_dir,
    )

    # 3. Healthcare MCP - FDA/PubMed/Trials (Node.js, from source)
    healthcare_dir = os.path.join(mcp_dir, "healthcare-mcp", "server")
    client.register_server(
        name="healthcare",
        command="node",
        args=["index.js"],
        cwd=healthcare_dir,
    )

    return client


# ============================================================
# Singleton for reuse across the application
# ============================================================

_global_client: Optional[MCPClientManager] = None


async def get_mcp_client() -> MCPClientManager:
    """Get or create the global MCP client instance."""
    global _global_client
    if _global_client is None:
        _global_client = create_default_mcp_client()
        results = await _global_client.start_all()
        logger.info(f"MCP servers started: {results}")
    return _global_client


async def shutdown_mcp_client() -> None:
    """Shutdown the global MCP client."""
    global _global_client
    if _global_client:
        await _global_client.stop_all()
        _global_client = None
