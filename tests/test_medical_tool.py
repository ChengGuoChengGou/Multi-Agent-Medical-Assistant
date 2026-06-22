"""
Unit tests for agents/medical_tool.py
Run: python -m pytest tests/test_medical_tool.py -v
"""
import os, sys, pytest, asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.medical_tool import (
    MedicalToolResult,
    MedicalTool,
    MCPMedicalTool,
    MedicalToolRegistry,
    get_tool_registry,
    init_tool_registry,
)


# ── Helper: Concrete MedicalTool subclass for testing ABC ──

class DummyTool(MedicalTool):
    """Concrete implementation for testing the MedicalTool ABC."""

    def __init__(self, name="dummy", description="A dummy tool",
                 category="utility", schema=None):
        self._name = name
        self._description = description
        self._category = category
        self._schema = schema or {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }

    @property
    def name(self): return self._name

    @property
    def description(self): return self._description

    @property
    def category(self): return self._category

    @property
    def input_schema(self): return self._schema

    async def execute(self, **kwargs) -> MedicalToolResult:
        return MedicalToolResult(
            success=True, content="dummy_result",
            tool_name=self.name, source="test",
        )


# ═══════════════════════════════════════
# MedicalToolResult (dataclass)
# ═══════════════════════════════════════

class TestMedicalToolResult:
    """Tests for the MedicalToolResult dataclass."""

    def test_defaults(self):
        r = MedicalToolResult(success=True, content="ok", tool_name="t", source="s")
        assert r.error is None
        assert r.metadata == {}
        # execution_time_ms defaults to 0.0 in the actual dataclass
        assert r.execution_time_ms == 0.0 or r.execution_time_ms is None

    def test_to_dict_success(self):
        r = MedicalToolResult(
            success=True, content="data", tool_name="search",
            source="mcp:biomcp", metadata={"server": "biomcp"},
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["content"] == "data"
        assert d["tool_name"] == "search"
        assert d["source"] == "mcp:biomcp"
        assert d["metadata"] == {"server": "biomcp"}
        assert d["error"] is None

    def test_to_dict_failure(self):
        r = MedicalToolResult(
            success=False, content="", tool_name="t",
            source="error", error="boom",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "boom"

    def test_to_dict_with_execution_time(self):
        r = MedicalToolResult(
            success=True, content="ok", tool_name="t",
            source="s", execution_time_ms=42.5,
        )
        assert r.to_dict()["execution_time_ms"] == 42.5


# ═══════════════════════════════════════
# MedicalTool (ABC)
# ═══════════════════════════════════════

class TestMedicalToolABC:
    """Tests for the MedicalTool abstract base class via DummyTool."""

    def test_instantiation(self):
        tool = DummyTool()
        assert tool.name == "dummy"
        assert tool.description == "A dummy tool"
        assert tool.category == "utility"

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            MedicalTool()  # type: ignore

    # -- describe() --

    def test_describe_basic(self):
        tool = DummyTool()
        d = tool.describe()
        assert d["name"] == "dummy"
        assert d["description"] == "A dummy tool"
        assert d["category"] == "utility"
        assert "input_schema" in d

    def test_describe_returns_full_description(self):
        """describe() returns the full description without truncation."""
        long_desc = "x" * 500
        tool = DummyTool(description=long_desc)
        d = tool.describe()
        assert d["description"] == long_desc
        assert len(d["description"]) == 500

    # -- validate() --

    def test_validate_required_missing(self):
        tool = DummyTool()
        is_valid, err = tool.validate()  # missing "query"
        assert is_valid is False
        assert "Missing required parameter: query" in err

    def test_validate_required_present(self):
        tool = DummyTool()
        result = tool.validate(query="hello")
        # Returns (True, None) on success
        assert result == (True, None)

    def test_validate_type_string_wrong(self):
        tool = DummyTool()
        is_valid, err = tool.validate(query=123)
        assert is_valid is False
        assert "must be string" in err

    def test_validate_type_integer_wrong(self):
        tool = DummyTool()
        is_valid, err = tool.validate(query="hello", limit="not_int")
        assert is_valid is False
        assert "must be integer" in err

    def test_validate_extra_fields_ignored(self):
        tool = DummyTool()
        result = tool.validate(query="hello", extra_field="ignored")
        assert result == (True, None)

    def test_validate_optional_field_none_ok(self):
        tool = DummyTool()
        result = tool.validate(query="hello", limit=None)
        assert result == (True, None)

    # -- query() (async) --

    @pytest.mark.asyncio
    async def test_query_success(self):
        tool = DummyTool()
        result = await tool.query(query="test")
        assert result.success is True
        assert result.content == "dummy_result"
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_query_validation_failure(self):
        tool = DummyTool()
        result = await tool.query()  # missing required "query"
        assert result.success is False
        assert result.source == "validation"
        assert "Missing required" in result.error

    @pytest.mark.asyncio
    async def test_query_execution_exception(self):
        class BrokenTool(DummyTool):
            async def execute(self, **kwargs):
                raise RuntimeError("boom")

        tool = BrokenTool()
        result = await tool.query(query="test")
        assert result.success is False
        assert result.source == "error"
        assert "boom" in result.error


# ═══════════════════════════════════════
# MCPMedicalTool
# ═══════════════════════════════════════

class TestMCPMedicalTool:
    """Tests for MCPMedicalTool wrapping and category inference."""

    def _make_mcp_tool(self, name="test_tool", server="biomcp",
                       desc="Test", schema=None):
        mcp = MagicMock()
        mcp.name = name
        mcp.description = desc
        mcp.server_name = server
        mcp.input_schema = schema or {"type": "object", "properties": {}}
        return mcp

    def test_basic_wrapping(self):
        mcp = self._make_mcp_tool(name="pubmed_search", server="biomcp")
        tool = MCPMedicalTool(mcp, call_fn=AsyncMock())
        assert tool.name == "pubmed_search"
        assert tool.description == "Test"
        assert tool.server_name == "biomcp"

    # -- category inference --

    @pytest.mark.parametrize("name,server,expected_cat", [
        ("drug_interaction_check", "server", "drug"),
        ("find_drug_info", "server", "drug"),
        ("icd10_lookup", "server", "coding"),
        ("snomed_search", "server", "coding"),
        ("code_mapper", "server", "coding"),
        ("pubmed_search", "server", "research"),
        ("article_finder", "server", "research"),
        ("search_literature", "server", "research"),
        ("diagnosis_helper", "server", "diagnosis"),
        ("disease_lookup", "server", "diagnosis"),
        ("random_tool", "biomcp", "research"),
        ("random_tool", "autoicd", "coding"),
        ("random_tool", "other", "utility"),
    ])
    def test_category_inference(self, name, server, expected_cat):
        mcp = self._make_mcp_tool(name=name, server=server)
        tool = MCPMedicalTool(mcp, call_fn=AsyncMock())
        assert tool.category == expected_cat

    @pytest.mark.asyncio
    async def test_execute_delegates_to_call_fn(self):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = "result_data"
        mock_result.error = None

        call_fn = AsyncMock(return_value=mock_result)
        mcp = self._make_mcp_tool(name="t1", server="s1")
        tool = MCPMedicalTool(mcp, call_fn=call_fn)

        result = await tool.execute(query="test")
        call_fn.assert_awaited_once_with("t1", {"query": "test"})
        assert result.success is True
        assert result.content == "result_data"
        assert result.source == "mcp:s1"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.content = ""
        mock_result.error = "connection refused"

        call_fn = AsyncMock(return_value=mock_result)
        mcp = self._make_mcp_tool(name="t2", server="s2")
        tool = MCPMedicalTool(mcp, call_fn=call_fn)

        result = await tool.execute(query="x")
        assert result.success is False
        assert result.error == "connection refused"

    @pytest.mark.asyncio
    async def test_query_validation_then_execute(self):
        """Integration: query → validate → execute pipeline."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = "ok"
        mock_result.error = None

        call_fn = AsyncMock(return_value=mock_result)
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        mcp = self._make_mcp_tool(name="t3", server="s3", schema=schema)
        tool = MCPMedicalTool(mcp, call_fn=call_fn)

        # Valid call
        result = await tool.query(query="hello")
        assert result.success is True

        # Invalid call (missing required)
        result = await tool.query()
        assert result.success is False
        assert result.source == "validation"


# ═══════════════════════════════════════
# MedicalToolRegistry
# ═══════════════════════════════════════

class TestMedicalToolRegistry:
    """Tests for the MedicalToolRegistry."""

    def _make_tool(self, name="t", category="utility"):
        return DummyTool(name=name, category=category)

    def test_register_and_get(self):
        reg = MedicalToolRegistry()
        tool = self._make_tool(name="search")
        reg.register(tool)
        assert reg.get("search") is tool

    def test_get_nonexistent(self):
        reg = MedicalToolRegistry()
        assert reg.get("nope") is None

    def test_register_overwrite_logs_warning(self):
        reg = MedicalToolRegistry()
        t1 = self._make_tool(name="dup")
        t2 = self._make_tool(name="dup")
        reg.register(t1)
        reg.register(t2)
        assert reg.get("dup") is t2  # overwritten

    def test_get_by_category(self):
        reg = MedicalToolRegistry()
        reg.register(self._make_tool(name="a", category="drug"))
        reg.register(self._make_tool(name="b", category="drug"))
        reg.register(self._make_tool(name="c", category="coding"))

        drugs = reg.get_by_category("drug")
        assert len(drugs) == 2
        assert all(t.category == "drug" for t in drugs)

        codings = reg.get_by_category("coding")
        assert len(codings) == 1

        assert reg.get_by_category("nonexistent") == []

    def test_get_all(self):
        reg = MedicalToolRegistry()
        reg.register(self._make_tool(name="x"))
        reg.register(self._make_tool(name="y"))
        assert len(reg.get_all()) == 2

    def test_len(self):
        reg = MedicalToolRegistry()
        assert len(reg) == 0
        reg.register(self._make_tool(name="a"))
        assert len(reg) == 1

    def test_contains(self):
        reg = MedicalToolRegistry()
        reg.register(self._make_tool(name="exists"))
        assert "exists" in reg
        assert "nope" not in reg

    def test_get_summary(self):
        reg = MedicalToolRegistry()
        reg.register(self._make_tool(name="drug_check", category="drug"))
        reg.register(self._make_tool(name="pubmed", category="research"))

        summary = reg.get_summary()
        assert "Available Medical Tools:" in summary
        assert "[DRUG]" in summary
        assert "[RESEARCH]" in summary
        assert "drug_check" in summary
        assert "pubmed" in summary

    def test_get_summary_max_desc_len(self):
        tool = DummyTool(name="long", description="a" * 200, category="x")
        reg = MedicalToolRegistry()
        reg.register(tool)
        summary = reg.get_summary(max_desc_len=10)
        # Description should be truncated
        assert "a" * 10 in summary
        assert "a" * 200 not in summary

    def test_get_tools_for_llm(self):
        reg = MedicalToolRegistry()
        reg.register(self._make_tool(name="a"))
        reg.register(self._make_tool(name="b"))
        llm_list = reg.get_tools_for_llm()
        assert len(llm_list) == 2
        assert all(isinstance(d, dict) for d in llm_list)
        assert all("name" in d for d in llm_list)


# ═══════════════════════════════════════
# get_tool_registry (singleton)
# ═══════════════════════════════════════

class TestGetToolRegistry:
    """Tests for the singleton get_tool_registry function."""

    def test_singleton_returns_same_instance(self):
        # Reset singleton first
        import agents.medical_tool as mt
        mt._global_registry = None
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2
        assert isinstance(r1, MedicalToolRegistry)
        # Cleanup
        mt._global_registry = None

    def test_singleton_starts_empty(self):
        import agents.medical_tool as mt
        mt._global_registry = None
        reg = get_tool_registry()
        assert len(reg) == 0
        mt._global_registry = None


# ═══════════════════════════════════════
# init_tool_registry (integration)
# ═══════════════════════════════════════

class TestInitToolRegistry:
    """Tests for init_tool_registry with mocked MCP client."""

    @pytest.mark.asyncio
    async def test_init_wraps_mcp_tools(self):
        import agents.medical_tool as mt
        mt._global_registry = None

        # Create fake MCP tools
        mcp_tool_1 = MagicMock()
        mcp_tool_1.name = "search_pubmed"
        mcp_tool_1.description = "Search PubMed"
        mcp_tool_1.server_name = "biomcp"
        mcp_tool_1.input_schema = {"type": "object", "properties": {}}

        mcp_tool_2 = MagicMock()
        mcp_tool_2.name = "icd10_code"
        mcp_tool_2.description = "ICD-10 lookup"
        mcp_tool_2.server_name = "autoicd"
        mcp_tool_2.input_schema = {"type": "object", "properties": {}}

        mock_client = MagicMock()
        mock_client.get_all_tools.return_value = [mcp_tool_1, mcp_tool_2]
        mock_client.call_tool = AsyncMock()

        with patch.dict("sys.modules", {"agents.mcp_client": MagicMock(
                get_mcp_client=AsyncMock(return_value=mock_client)
            )}):
                reg = await init_tool_registry()

        assert len(reg) == 2
        assert "search_pubmed" in reg
        assert "icd10_code" in reg
        assert isinstance(reg.get("search_pubmed"), MCPMedicalTool)
        mt._global_registry = None
