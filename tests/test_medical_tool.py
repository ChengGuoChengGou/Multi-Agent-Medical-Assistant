"""
Tests for agents/medical_tool.py

MedicalToolResult, MedicalTool.validate, MedicalTool.query,
MCPMedicalTool.category inference, MedicalToolRegistry.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, Optional

# Add project root to path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.medical_tool import (
    MedicalToolResult,
    MedicalTool,
    MCPMedicalTool,
    MedicalToolRegistry,
    get_tool_registry,
)


# ── Fixtures ──────────────────────────────────────────────

class DummyTool(MedicalTool):
    """Minimal concrete MedicalTool for testing."""

    def __init__(
        self,
        name: str = "dummy_tool",
        description: str = "A dummy tool",
        category: str = "utility",
        schema: Optional[Dict[str, Any]] = None,
        execute_fn=None,
    ):
        self._name = name
        self._description = description
        self._category = category
        self._schema = schema or {"type": "object", "properties": {}, "required": []}
        self._execute_fn = execute_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> str:
        return self._category

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._schema

    async def execute(self, **kwargs) -> MedicalToolResult:
        if self._execute_fn:
            return await self._execute_fn(**kwargs)
        return MedicalToolResult(
            success=True, content="ok", tool_name=self.name, source="built_in"
        )


def make_mcp_tool(name, description="desc", server_name="biomcp", input_schema=None):
    """Create a mock MCPTool."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.server_name = server_name
    tool.input_schema = input_schema or {"type": "object", "properties": {}, "required": []}
    return tool


# ── MedicalToolResult ─────────────────────────────────────

class TestMedicalToolResult:

    def test_to_dict(self):
        r = MedicalToolResult(
            success=True, content="data", tool_name="t1", source="mcp",
            error=None, metadata={"k": "v"}, execution_time_ms=42.5,
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["content"] == "data"
        assert d["tool_name"] == "t1"
        assert d["source"] == "mcp"
        assert d["error"] is None
        assert d["metadata"] == {"k": "v"}
        assert d["execution_time_ms"] == 42.5

    def test_defaults(self):
        r = MedicalToolResult(success=False, content="", tool_name="t", source="error")
        assert r.error is None
        assert r.metadata == {}
        assert r.execution_time_ms == 0.0

    def test_to_dict_error(self):
        r = MedicalToolResult(
            success=False, content="", tool_name="t", source="error", error="boom"
        )
        d = r.to_dict()
        assert d["error"] == "boom"
        assert d["success"] is False


# ── MedicalTool.validate ──────────────────────────────────

class TestMedicalToolValidate:

    def test_valid_no_required(self):
        tool = DummyTool()
        ok, err = tool.validate()
        assert ok is True
        assert err is None

    def test_missing_required(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate()
        assert ok is False
        assert "query" in err

    def test_required_present(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate(query="aspirin")
        assert ok is True

    def test_required_none_value(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate(query=None)
        assert ok is False

    def test_type_string_wrong(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": [],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate(query=123)
        assert ok is False
        assert "string" in err

    def test_type_integer_wrong(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": [],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate(count="abc")
        assert ok is False
        assert "integer" in err

    def test_type_number_accepts_float(self):
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": [],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate(score=3.14)
        assert ok is True

    def test_type_number_accepts_int(self):
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": [],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate(score=42)
        assert ok is True

    def test_type_number_wrong(self):
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": [],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate(score="high")
        assert ok is False

    def test_extra_field_ignored(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": [],
        }
        tool = DummyTool(schema=schema)
        ok, err = tool.validate(query="ok", unknown="field")
        assert ok is True


# ── MedicalTool.query ─────────────────────────────────────

class TestMedicalToolQuery:

    @pytest.mark.asyncio
    async def test_query_success(self):
        tool = DummyTool()
        result = await tool.query()
        assert result.success is True
        assert result.execution_time_ms >= 0  # instant DummyTool may round to 0.0

    @pytest.mark.asyncio
    async def test_query_validation_failure(self):
        schema = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        tool = DummyTool(schema=schema)
        result = await tool.query()  # missing required 'q'
        assert result.success is False
        assert result.source == "validation"
        assert "q" in result.error

    @pytest.mark.asyncio
    async def test_query_execute_exception(self):
        async def boom(**kwargs):
            raise RuntimeError("tool broken")

        tool = DummyTool(execute_fn=boom)
        result = await tool.query()
        assert result.success is False
        assert result.source == "error"
        assert "tool broken" in result.error

    @pytest.mark.asyncio
    async def test_query_execution_time_recorded(self):
        import time

        async def slow(**kwargs):
            await asyncio.sleep(0.05)
            return MedicalToolResult(
                success=True, content="done", tool_name="t", source="built_in"
            )

        tool = DummyTool(execute_fn=slow)
        result = await tool.query()
        assert result.success is True
        assert result.execution_time_ms >= 40  # at least 40ms


# ── MedicalTool.describe ──────────────────────────────────

class TestMedicalToolDescribe:

    def test_describe(self):
        schema = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        tool = DummyTool(name="search", description="Search tool", category="research", schema=schema)
        d = tool.describe()
        assert d["name"] == "search"
        assert d["description"] == "Search tool"
        assert d["category"] == "research"
        assert d["input_schema"] == schema


# ── MCPMedicalTool ────────────────────────────────────────

class TestMCPMedicalToolCategory:

    def test_drug_in_tool_name(self):
        mcp = make_mcp_tool(name="drug_interaction_check")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "drug"

    def test_interaction_in_tool_name(self):
        mcp = make_mcp_tool(name="check_interactions")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "drug"

    def test_icd_in_name(self):
        mcp = make_mcp_tool(name="icd10_lookup")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "coding"

    def test_snomed_in_name(self):
        mcp = make_mcp_tool(name="snomed_search")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "coding"

    def test_search_in_name(self):
        mcp = make_mcp_tool(name="pubmed_search")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "research"

    def test_article_in_name(self):
        mcp = make_mcp_tool(name="fetch_article")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "research"

    def test_diagnosis_in_name(self):
        mcp = make_mcp_tool(name="diagnosis_helper")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "diagnosis"

    def test_disease_in_name(self):
        mcp = make_mcp_tool(name="disease_lookup")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "diagnosis"

    def test_biomcp_server_default_research(self):
        mcp = make_mcp_tool(name="unknown_tool", server_name="biomcp")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "research"

    def test_autoicd_server_default_coding(self):
        mcp = make_mcp_tool(name="unknown_tool", server_name="autoicd")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "coding"

    def test_unknown_server_default_utility(self):
        mcp = make_mcp_tool(name="unknown_tool", server_name="other")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.category == "utility"

    def test_properties(self):
        mcp = make_mcp_tool(name="test_tool", description="A test", server_name="srv")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=AsyncMock())
        assert t.name == "test_tool"
        assert t.description == "A test"
        assert t.server_name == "srv"


class TestMCPMedicalToolExecute:

    @pytest.mark.asyncio
    async def test_execute_delegates_to_call_fn(self):
        mcp_result = MedicalToolResult(
            success=True, content="result_data", tool_name="t", source="mcp"
        )
        call_fn = AsyncMock(return_value=mcp_result)
        mcp = make_mcp_tool(name="test_tool", server_name="biomcp")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=call_fn)

        result = await t.execute(query="aspirin")
        assert result.success is True
        assert result.content == "result_data"
        assert result.source == "mcp:biomcp"
        assert result.metadata == {"server": "biomcp"}
        call_fn.assert_called_once_with("test_tool", {"query": "aspirin"})

    @pytest.mark.asyncio
    async def test_execute_error(self):
        mcp_result = MedicalToolResult(
            success=False, content="", tool_name="t", source="mcp", error="connection failed"
        )
        call_fn = AsyncMock(return_value=mcp_result)
        mcp = make_mcp_tool(name="t", server_name="srv")
        t = MCPMedicalTool(mcp_tool=mcp, call_fn=call_fn)

        result = await t.execute()
        assert result.success is False
        assert result.error == "connection failed"


# ── MedicalToolRegistry ───────────────────────────────────

class TestMedicalToolRegistry:

    def test_register_and_get(self):
        reg = MedicalToolRegistry()
        tool = DummyTool(name="t1")
        reg.register(tool)
        assert reg.get("t1") is tool

    def test_get_missing(self):
        reg = MedicalToolRegistry()
        assert reg.get("nonexistent") is None

    def test_register_overwrite(self):
        reg = MedicalToolRegistry()
        t1 = DummyTool(name="t1", description="first")
        t2 = DummyTool(name="t1", description="second")
        reg.register(t1)
        reg.register(t2)
        assert reg.get("t1").description == "second"

    def test_get_by_category(self):
        reg = MedicalToolRegistry()
        reg.register(DummyTool(name="a", category="research"))
        reg.register(DummyTool(name="b", category="drug"))
        reg.register(DummyTool(name="c", category="research"))
        research = reg.get_by_category("research")
        assert len(research) == 2
        assert all(t.category == "research" for t in research)

    def test_get_all(self):
        reg = MedicalToolRegistry()
        reg.register(DummyTool(name="a"))
        reg.register(DummyTool(name="b"))
        assert len(reg.get_all()) == 2

    def test_len(self):
        reg = MedicalToolRegistry()
        assert len(reg) == 0
        reg.register(DummyTool(name="a"))
        assert len(reg) == 1

    def test_contains(self):
        reg = MedicalToolRegistry()
        reg.register(DummyTool(name="a"))
        assert "a" in reg
        assert "b" not in reg

    def test_get_summary(self):
        reg = MedicalToolRegistry()
        reg.register(DummyTool(name="search_pubmed", description="Search PubMed articles", category="research"))
        reg.register(DummyTool(name="drug_check", description="Check drug interactions", category="drug"))
        summary = reg.get_summary()
        assert "Available Medical Tools:" in summary
        assert "[RESEARCH]" in summary
        assert "[DRUG]" in summary
        assert "search_pubmed" in summary
        assert "drug_check" in summary

    def test_get_summary_max_desc_len(self):
        reg = MedicalToolRegistry()
        reg.register(DummyTool(name="t", description="A" * 200, category="c"))
        summary = reg.get_summary(max_desc_len=50)
        assert "A" * 50 in summary
        assert "A" * 51 not in summary

    def test_get_tools_for_llm(self):
        reg = MedicalToolRegistry()
        reg.register(DummyTool(name="t1", category="research"))
        reg.register(DummyTool(name="t2", category="drug"))
        tools = reg.get_tools_for_llm()
        assert len(tools) == 2
        assert all("name" in t for t in tools)
        assert all("input_schema" in t for t in tools)


# ── get_tool_registry (singleton) ─────────────────────────

class TestGetToolRegistry:

    def test_singleton_returns_same_instance(self):
        import agents.medical_tool as mt
        old = mt._global_registry
        mt._global_registry = None  # reset
        try:
            r1 = get_tool_registry()
            r2 = get_tool_registry()
            assert r1 is r2
            assert isinstance(r1, MedicalToolRegistry)
        finally:
            mt._global_registry = old
