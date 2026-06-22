"""
MCP Connection Manager

Wraps MCPServerConnection with:
- Automatic retry with exponential backoff
- Connection health checking and auto-reconnect
- Configurable timeouts per operation type
- Connection pool management

Phase 4.4: MCP连接管理增强
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Retry behavior configuration."""
    max_retries: int = 3
    base_delay: float = 1.0       # seconds
    max_delay: float = 30.0       # seconds
    backoff_factor: float = 2.0   # exponential multiplier
    retry_on_none: bool = True    # retry when server returns None

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


@dataclass
class TimeoutConfig:
    """Timeout configuration per operation type."""
    init_timeout: float = 30.0      # server initialization
    tool_call_timeout: float = 60.0 # tool execution
    discovery_timeout: float = 15.0 # tool listing
    health_check_timeout: float = 5.0


@dataclass
class ConnectionHealth:
    """Tracks connection health metrics."""
    server_name: str
    is_healthy: bool = False
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    last_error: Optional[str] = None

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls

    def record_success(self) -> None:
        self.is_healthy = True
        self.last_success = time.monotonic()
        self.consecutive_failures = 0
        self.total_calls += 1

    def record_failure(self, error: str) -> None:
        self.last_failure = time.monotonic()
        self.consecutive_failures += 1
        self.total_calls += 1
        self.total_failures += 1
        self.last_error = error
        # Mark unhealthy after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_healthy = False


class ResilientMCPConnection:
    """
    Wraps MCPServerConnection with retry, reconnect, and health tracking.
    
    Usage:
        conn = ResilientMCPConnection(server_connection)
        result = await conn.call_tool("search", {"query": "aspirin"})
    """

    def __init__(
        self,
        connection: Any,  # MCPServerConnection
        retry_config: Optional[RetryConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        self._conn = connection
        self._retry = retry_config or RetryConfig()
        self._timeouts = timeout_config or TimeoutConfig()
        self._health = ConnectionHealth(server_name=connection.name)
        self._reconnecting = False
        self._reconnect_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._conn.name

    @property
    def health(self) -> ConnectionHealth:
        return self._health

    @property
    def is_initialized(self) -> bool:
        return self._conn._initialized

    @property
    def tools(self) -> List:
        return self._conn.tools

    async def start(self) -> bool:
        """Start with timeout and health tracking."""
        try:
            result = await asyncio.wait_for(
                self._conn.start(),
                timeout=self._timeouts.init_timeout,
            )
            if result:
                self._health.is_healthy = True
                self._health.record_success()
            else:
                self._health.record_failure("Initialization returned False")
            return result
        except asyncio.TimeoutError:
            self._health.record_failure(f"Init timeout ({self._timeouts.init_timeout}s)")
            logger.error(f"[MCP:{self.name}] Start timeout")
            return False
        except Exception as e:
            self._health.record_failure(str(e))
            logger.error(f"[MCP:{self.name}] Start error: {e}")
            return False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call tool with retry logic and health tracking.
        
        On failure:
        1. Retries up to max_retries with exponential backoff
        2. If all retries fail, attempts reconnect once
        3. If reconnect succeeds, retries the call
        """
        last_error = None
        
        for attempt in range(self._retry.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._conn.call_tool(tool_name, arguments),
                    timeout=self._timeouts.tool_call_timeout,
                )
                
                # Check for None response (server not responding)
                if result is None and self._retry.retry_on_none:
                    last_error = "Server returned None"
                    if attempt < self._retry.max_retries:
                        delay = self._retry.get_delay(attempt)
                        logger.warning(
                            f"[MCP:{self.name}] {tool_name} returned None, "
                            f"retry {attempt+1}/{self._retry.max_retries} in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                
                if result is not None:
                    # Check if result indicates failure
                    if hasattr(result, 'success') and not result.success:
                        self._health.record_failure(result.error or "Tool returned failure")
                    else:
                        self._health.record_success()
                    return result
                    
            except asyncio.TimeoutError:
                last_error = f"Timeout ({self._timeouts.tool_call_timeout}s)"
                logger.warning(
                    f"[MCP:{self.name}] {tool_name} timeout, "
                    f"attempt {attempt+1}/{self._retry.max_retries+1}"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[MCP:{self.name}] {tool_name} error: {e}, "
                    f"attempt {attempt+1}/{self._retry.max_retries+1}"
                )
            
            self._health.record_failure(last_error)
            
            if attempt < self._retry.max_retries:
                delay = self._retry.get_delay(attempt)
                await asyncio.sleep(delay)

        # All retries failed - attempt reconnect
        logger.warning(f"[MCP:{self.name}] All retries failed for {tool_name}, attempting reconnect...")
        reconnected = await self._reconnect()
        
        if reconnected:
            try:
                result = await asyncio.wait_for(
                    self._conn.call_tool(tool_name, arguments),
                    timeout=self._timeouts.tool_call_timeout,
                )
                if result is not None:
                    self._health.record_success()
                    return result
            except Exception as e:
                last_error = f"Post-reconnect call failed: {e}"
                logger.error(f"[MCP:{self.name}] {last_error}")

        # Final failure
        self._health.record_failure(last_error)
        from agents.mcp_client import MCPToolResult
        return MCPToolResult(
            success=False, content="", tool_name=tool_name,
            server_name=self.name,
            error=f"All attempts failed: {last_error}",
        )

    async def _reconnect(self) -> bool:
        """Attempt to restart the MCP server connection."""
        async with self._reconnect_lock:
            if self._reconnecting:
                return False
            self._reconnecting = True

        try:
            logger.info(f"[MCP:{self.name}] Reconnecting...")
            
            # Stop existing connection
            await self._conn.stop()
            await asyncio.sleep(1.0)  # Brief pause before restart
            
            # Restart
            result = await self.start()
            if result:
                logger.info(f"[MCP:{self.name}] Reconnect successful")
                return True
            else:
                logger.error(f"[MCP:{self.name}] Reconnect failed")
                return False
        finally:
            self._reconnecting = False

    async def health_check(self) -> bool:
        """
        Quick health check: try to list tools.
        Returns True if server is responsive.
        """
        if not self._conn._initialized:
            return False
        try:
            # Use tools/list as a lightweight health check
            result = await asyncio.wait_for(
                self._conn._send_request("tools/list", {}),
                timeout=self._timeouts.health_check_timeout,
            )
            healthy = result is not None
            if healthy:
                self._health.record_success()
            else:
                self._health.record_failure("Health check returned None")
            return healthy
        except Exception as e:
            self._health.record_failure(f"Health check error: {e}")
            return False

    async def stop(self) -> None:
        """Stop the underlying connection."""
        await self._conn.stop()
        self._health.is_healthy = False


class ResilientMCPClientManager:
    """
    Drop-in replacement for MCPClientManager with resilient connections.
    Wraps each MCPServerConnection in ResilientMCPConnection.
    """

    def __init__(self):
        self._connections: Dict[str, ResilientMCPConnection] = {}

    def add_server(self, connection: Any, 
                   retry_config: Optional[RetryConfig] = None,
                   timeout_config: Optional[TimeoutConfig] = None) -> None:
        """Add a server with resilient wrapping."""
        resilient = ResilientMCPConnection(
            connection=connection,
            retry_config=retry_config,
            timeout_config=timeout_config,
        )
        self._connections[connection.name] = resilient

    async def start_all(self) -> Dict[str, bool]:
        """Start all servers concurrently."""
        tasks = {
            name: asyncio.create_task(conn.start())
            for name, conn in self._connections.items()
        }
        results = {}
        for name, task in tasks.items():
            results[name] = await task
        return results

    async def stop_all(self) -> None:
        """Stop all servers."""
        await asyncio.gather(
            *[conn.stop() for conn in self._connections.values()],
            return_exceptions=True,
        )

    def get_connection(self, name: str) -> Optional[ResilientMCPConnection]:
        return self._connections.get(name)

    def get_all_tools(self) -> List:
        """Get all tools from all healthy connections."""
        tools = []
        for conn in self._connections.values():
            if conn.is_initialized:
                tools.extend(conn.tools)
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Find and call tool across all connections with retry."""
        for conn in self._connections.values():
            if any(t.name == tool_name for t in conn.tools):
                return await conn.call_tool(tool_name, arguments)
        
        from agents.mcp_client import MCPToolResult
        return MCPToolResult(
            success=False, content="", tool_name=tool_name,
            server_name="unknown", error=f"Tool '{tool_name}' not found on any server",
        )

    async def call_on_server(self, server_name: str, tool_name: str, 
                             arguments: Dict[str, Any]) -> Any:
        """Call tool on specific server with retry."""
        conn = self._connections.get(server_name)
        if conn:
            return await conn.call_tool(tool_name, arguments)
        
        from agents.mcp_client import MCPToolResult
        return MCPToolResult(
            success=False, content="", tool_name=tool_name,
            server_name=server_name, error=f"Server '{server_name}' not found",
        )

    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all connections."""
        results = {}
        for name, conn in self._connections.items():
            results[name] = await conn.health_check()
        return results

    def get_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get health metrics for all connections."""
        return {
            name: {
                "healthy": conn.health.is_healthy,
                "failure_rate": f"{conn.health.failure_rate:.1%}",
                "consecutive_failures": conn.health.consecutive_failures,
                "total_calls": conn.health.total_calls,
                "last_error": conn.health.last_error,
            }
            for name, conn in self._connections.items()
        }
