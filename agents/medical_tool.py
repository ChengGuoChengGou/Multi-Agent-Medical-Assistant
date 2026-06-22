"""
Medical Tool Abstraction Layer

Provides a unified MedicalTool interface (query/describe/validate/execute)
that wraps existing MCP tools with validation, error handling, and caching.

Inspired by CCS's buildTool pattern for consistent tool contracts.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MedicalToolResult:
    """Standardized result from any medical tool execution."""

    success: bool
    content: str
    tool_name: str
    source: str  # "mcp", "built_in", "fallback"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "tool_name": self.tool_name,
            "source": self.source,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }


class MedicalTool(ABC):
    """
    Abstract base class for all medical tools.

    Provides a consistent interface: query, describe, validate, execute.
    All medical tools (MCP-based, built-in, or fallback) should implement this.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name for routing and logging."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for LLM routing context."""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Tool category: 'research', 'coding', 'drug', 'diagnosis', 'utility'."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for input parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> MedicalToolResult:
        """Execute the tool with validated parameters."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return tool metadata for LLM routing context."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": self.input_schema,
        }

    def validate(self, **kwargs) -> tuple[bool, str | None]:
        """
        Validate input parameters against the schema.
        Returns (is_valid, error_message).
        """
        schema = self.input_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field_name in required:
            if field_name not in kwargs or kwargs[field_name] is None:
                return False, f"Missing required parameter: {field_name}"

        # Check types for provided fields
        for field_name, value in kwargs.items():
            if field_name in properties and value is not None:
                expected_type = properties[field_name].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"Parameter '{field_name}' must be string, got {type(value).__name__}"
                if expected_type == "integer" and not isinstance(value, int):
                    return False, f"Parameter '{field_name}' must be integer, got {type(value).__name__}"
                if expected_type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter '{field_name}' must be number, got {type(value).__name__}"

        return True, None

    async def query(self, **kwargs) -> MedicalToolResult:
        """
        High-level entry point: validate → execute → return result.
        Subclasses can override for pre/post processing.
        """
        import time

        start = time.monotonic()

        # Validate
        is_valid, error = self.validate(**kwargs)
        if not is_valid:
            elapsed = (time.monotonic() - start) * 1000
            return MedicalToolResult(
                success=False,
                content="",
                tool_name=self.name,
                source="validation",
                error=error,
                execution_time_ms=elapsed,
            )

        try:
            result = await self.execute(**kwargs)
            result.execution_time_ms = (time.monotonic() - start) * 1000
            return result
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(f"[MedicalTool:{self.name}] Execution failed: {e}")
            return MedicalToolResult(
                success=False,
                content="",
                tool_name=self.name,
                source="error",
                error=str(e),
                execution_time_ms=elapsed,
            )


class MCPMedicalTool(MedicalTool):
    """
    Wraps an existing MCPTool as a MedicalTool.
    Bridges the MCP client infrastructure with the MedicalTool interface.
    """

    def __init__(self, mcp_tool: Any, call_fn: Callable):
        """
        Args:
            mcp_tool: MCPTool instance from mcp_client.py
            call_fn: async callable(tool_name, arguments) -> MCPToolResult
        """
        self._mcp_tool = mcp_tool
        self._call_fn = call_fn
        self._name = mcp_tool.name
        self._description = mcp_tool.description
        self._input_schema = mcp_tool.input_schema
        self._server_name = mcp_tool.server_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> str:
        """Infer category from server name or tool name."""
        server = self._server_name.lower()
        tool = self._name.lower()
        if "drug" in tool or "interaction" in tool:
            return "drug"
        if "icd" in tool or "code" in tool or "snomed" in tool:
            return "coding"
        if "search" in tool or "article" in tool or "pubmed" in tool:
            return "research"
        if "diagnosis" in tool or "disease" in tool:
            return "diagnosis"
        if server == "biomcp":
            return "research"
        if server == "autoicd":
            return "coding"
        return "utility"

    @property
    def input_schema(self) -> dict[str, Any]:
        return self._input_schema

    @property
    def server_name(self) -> str:
        return self._server_name

    async def execute(self, **kwargs) -> MedicalToolResult:
        """Delegate execution to the underlying MCP tool."""
        result = await self._call_fn(self._name, kwargs)
        return MedicalToolResult(
            success=result.success,
            content=result.content,
            tool_name=self.name,
            source=f"mcp:{self._server_name}",
            error=result.error,
            metadata={"server": self._server_name},
        )


# ============================================================
# Tool Registry
# ============================================================


class MedicalToolRegistry:
    """
    Central registry for all medical tools.
    Provides lookup by name, category, and a summary for LLM context.
    """

    def __init__(self):
        self._tools: dict[str, MedicalTool] = {}

    def register(self, tool: MedicalTool) -> None:
        """Register a tool. Logs warning if overwriting."""
        if tool.name in self._tools:
            logger.warning(f"[ToolRegistry] Overwriting tool: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"[ToolRegistry] Registered: {tool.name} ({tool.category})")

    def get(self, name: str) -> MedicalTool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def get_by_category(self, category: str) -> list[MedicalTool]:
        """Get all tools in a category."""
        return [t for t in self._tools.values() if t.category == category]

    def get_all(self) -> list[MedicalTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_summary(self, max_desc_len: int = 100) -> str:
        """Human-readable summary for LLM routing context."""
        lines = ["Available Medical Tools:"]
        by_category: dict[str, list[MedicalTool]] = {}
        for tool in self._tools.values():
            by_category.setdefault(tool.category, []).append(tool)

        for category, tools in sorted(by_category.items()):
            lines.append(f"\n[{category.upper()}]")
            for tool in tools:
                desc = tool.description[:max_desc_len]
                lines.append(f"  - {tool.name}: {desc}")

        return "\n".join(lines)

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """Return tool descriptions in a format suitable for LLM context."""
        return [tool.describe() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ============================================================
# Singleton registry
# ============================================================

_global_registry: MedicalToolRegistry | None = None


def get_tool_registry() -> MedicalToolRegistry:
    """Get the global tool registry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MedicalToolRegistry()
    return _global_registry


async def init_tool_registry() -> MedicalToolRegistry:
    """
    Initialize the registry by wrapping all MCP tools as MedicalTools.
    Call once at application startup.
    """
    from agents.mcp_client import get_mcp_client

    registry = get_tool_registry()
    client = await get_mcp_client()

    # Wrap each MCP tool as a MedicalTool
    for mcp_tool in client.get_all_tools():
        medical_tool = MCPMedicalTool(
            mcp_tool=mcp_tool,
            call_fn=client.call_tool,
        )
        registry.register(medical_tool)

    logger.info(f"[ToolRegistry] Initialized with {len(registry)} tools")
    return registry
