"""Tests for agents/mcp_agent.py — MCP Agent Node (Phase 58)

Covers:
- _determine_mcp_tools_keyword: keyword-based routing (8 categories)
- _determine_mcp_tools_llm: LLM routing + fallback
- mcp_agent_node: async LangGraph node (happy path, error paths)
- _format_tool_results_raw: raw formatting
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from agents.mcp_agent import (
    MCP_ROUTING_PROMPT,
    _determine_mcp_tools_keyword,
    _determine_mcp_tools_llm,
    mcp_agent_node,
    _format_tool_results_raw,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm():
    """A mock LLM that returns a configurable response."""
    llm = MagicMock()
    return llm


@pytest.fixture
def sample_state():
    """Minimal LangGraph state with one HumanMessage."""
    return {
        "messages": [HumanMessage(content="What is the ICD code for diabetes?")]
    }


@pytest.fixture
def empty_state():
    """State with no messages."""
    return {"messages": []}


@pytest.fixture
def mock_mcp_client():
    """Mock MCP client with configurable call_on_server."""
    client = AsyncMock()
    return client


@pytest.fixture
def success_result():
    """Mock MCP call result — success."""
    r = MagicMock()
    r.success = True
    r.content = "Found 5 results for diabetes ICD codes."
    return r


@pytest.fixture
def error_result():
    """Mock MCP call result — failure."""
    r = MagicMock()
    r.success = False
    r.error = "Connection timeout"
    return r


@pytest.fixture
def mock_config():
    """Mock config object with optional llm attribute."""
    cfg = MagicMock()
    cfg.llm = None
    return cfg


@pytest.fixture
def mock_config_with_llm(mock_llm):
    """Mock config object that has an llm."""
    cfg = MagicMock()
    cfg.llm = mock_llm
    return cfg


# ============================================================
# _determine_mcp_tools_keyword
# ============================================================

class TestMCPKeywordRouting:
    """Keyword-based fallback routing — 8 categories."""

    def test_coding_keywords(self):
        """ICD/coding keywords → autoicd server."""
        for kw in ["ICD-10", "编码", "diagnosis code", "snomed", "loinc", "dsm"]:
            calls = _determine_mcp_tools_keyword(kw)
            assert len(calls) >= 1
            assert calls[0]["server"] == "autoicd"
            assert calls[0]["tool"] == "reference_search"
            assert "icd-10-cm" in calls[0]["args"]["code_type"]

    def test_drug_keywords(self):
        """Drug keywords → healthcare server."""
        for kw in ["drug interaction", "药物副作用", "medication", "fda批准"]:
            calls = _determine_mcp_tools_keyword(kw)
            assert len(calls) >= 1
            assert calls[0]["server"] == "healthcare"
            assert calls[0]["tool"] == "search_drugs"

    def test_trial_keywords(self):
        """Clinical trial keywords → biomcp server."""
        for kw in ["clinical trial", "临床试验", "randomized", "phase 3"]:
            calls = _determine_mcp_tools_keyword(kw)
            assert len(calls) >= 1
            assert calls[0]["server"] == "biomcp"
            assert calls[0]["tool"] == "trial_search"

    def test_gene_keywords(self):
        """Gene/variant keywords → biomcp variant_search + gene_search (2 calls)."""
        for kw in ["BRCA1 mutation", "EGFR突变", "genomic variant", "snp"]:
            calls = _determine_mcp_tools_keyword(kw)
            servers = [c["server"] for c in calls]
            tools = [c["tool"] for c in calls]
            assert all(s == "biomcp" for s in servers)
            assert "variant_search" in tools
            assert "gene_search" in tools

    def test_article_keywords(self):
        """Article/research keywords → biomcp article_search + healthcare search_pubmed."""
        for kw in ["PubMed研究", "meta-analysis", "systematic review", "预印本"]:
            calls = _determine_mcp_tools_keyword(kw)
            servers_tools = [(c["server"], c["tool"]) for c in calls]
            assert ("biomcp", "article_search") in servers_tools
            assert ("healthcare", "search_pubmed") in servers_tools

    def test_disease_keywords_only_when_no_prior_match(self):
        """Disease keywords alone → biomcp disease_search (only when no other match)."""
        calls = _determine_mcp_tools_keyword("What is the pathology of this syndrome?")
        assert len(calls) == 1
        assert calls[0]["server"] == "biomcp"
        assert calls[0]["tool"] == "disease_search"

    def test_disease_keywords_not_triggered_when_other_match(self):
        """Disease keyword should NOT override a prior drug match."""
        calls = _determine_mcp_tools_keyword("drug interaction disease syndrome")
        # drug keyword triggers first → healthcare
        servers = [c["server"] for c in calls]
        assert "healthcare" in servers
        # disease should not add (condition: `not calls` was false)
        disease_calls = [c for c in calls if c["tool"] == "disease_search"]
        assert len(disease_calls) == 0

    def test_no_match_defaults(self):
        """Query with no keyword match → healthcare search_health_topics + biomcp article_search."""
        calls = _determine_mcp_tools_keyword("hello world")
        servers_tools = [(c["server"], c["tool"]) for c in calls]
        assert ("healthcare", "search_health_topics") in servers_tools
        assert ("biomcp", "article_search") in servers_tools

    def test_chinese_keywords(self):
        """Chinese medical keywords are recognized."""
        calls = _determine_mcp_tools_keyword("这个药物的副作用")
        assert any(c["tool"] == "search_drugs" for c in calls)

    def test_multi_category_query(self):
        """Query with drug + gene keywords → multiple calls from different categories."""
        calls = _determine_mcp_tools_keyword("BRCA1 drug interaction mutation")
        tools = [c["tool"] for c in calls]
        assert "search_drugs" in tools  # drug keyword
        assert "variant_search" in tools  # gene keyword

    def test_returns_list_of_dicts(self):
        """Return type is List[Dict]."""
        calls = _determine_mcp_tools_keyword("test")
        assert isinstance(calls, list)
        for c in calls:
            assert "server" in c
            assert "tool" in c
            assert "args" in c


# ============================================================
# _determine_mcp_tools_llm
# ============================================================

class TestMCPLLMRouting:
    """LLM-based routing with keyword fallback."""

    def test_llm_returns_valid_json(self, mock_llm):
        """LLM returns valid JSON array → use it directly."""
        expected = [{"server": "biomcp", "tool": "article_search", "args": {"query": "test"}}]
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(expected))
        result = _determine_mcp_tools_llm("test query", mock_llm)
        assert result == expected

    def test_llm_returns_markdown_code_block(self, mock_llm):
        """LLM returns ```json ... ``` → parse correctly."""
        expected = [{"server": "healthcare", "tool": "search_drugs", "args": {"query": "aspirin"}}]
        content = "```json\n" + json.dumps(expected) + "\n```"
        mock_llm.invoke.return_value = MagicMock(content=content)
        result = _determine_mcp_tools_llm("aspirin", mock_llm)
        assert result == expected

    def test_llm_returns_code_block_without_json_tag(self, mock_llm):
        """LLM returns ``` ... ``` without 'json' prefix → still parse."""
        expected = [{"server": "autoicd", "tool": "reference_search", "args": {"query": "ICD"}}]
        content = "```\n" + json.dumps(expected) + "\n```"
        mock_llm.invoke.return_value = MagicMock(content=content)
        result = _determine_mcp_tools_llm("ICD codes", mock_llm)
        assert result == expected

    def test_llm_returns_non_list_falls_back(self, mock_llm):
        """LLM returns a dict (not list) → falls back to keyword routing."""
        mock_llm.invoke.return_value = MagicMock(content='{"not": "a list"}')
        result = _determine_mcp_tools_llm("drug query", mock_llm)
        # Should fall back to keyword routing
        assert isinstance(result, list)
        assert len(result) > 0

    def test_llm_raises_exception_falls_back(self, mock_llm):
        """LLM raises exception → falls back to keyword routing."""
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        result = _determine_mcp_tools_llm("clinical trial", mock_llm)
        assert isinstance(result, list)
        assert any(c["tool"] == "trial_search" for c in result)

    def test_llm_returns_invalid_json_falls_back(self, mock_llm):
        """LLM returns non-JSON text → falls back to keyword routing."""
        mock_llm.invoke.return_value = MagicMock(content="I don't understand the format.")
        result = _determine_mcp_tools_llm("diabetes", mock_llm)
        assert isinstance(result, list)
        assert len(result) > 0


# ============================================================
# _format_tool_results_raw
# ============================================================

class TestFormatToolResultsRaw:
    """Raw formatting when LLM synthesis is unavailable."""

    def test_single_success_result(self):
        results = [{"server": "biomcp", "tool": "article_search", "result": "5 articles found."}]
        output = _format_tool_results_raw("diabetes", results)
        assert "Medical Database Search Results" in output
        assert "BIOMCP" in output
        assert "5 articles found." in output
        assert "diabetes" in output

    def test_single_error_result(self):
        results = [{"server": "healthcare", "tool": "search_drugs", "error": "timeout"}]
        output = _format_tool_results_raw("aspirin", results)
        assert "HEALTHCARE" in output
        assert "Error: timeout" in output

    def test_mixed_results(self):
        results = [
            {"server": "biomcp", "tool": "article_search", "result": "Found 3 articles."},
            {"server": "autoicd", "tool": "reference_search", "error": "Server offline"},
        ]
        output = _format_tool_results_raw("ICD code", results)
        assert "BIOMCP" in output
        assert "Found 3 articles." in output
        assert "AUTOICD" in output
        assert "Error: Server offline" in output

    def test_empty_results(self):
        output = _format_tool_results_raw("test", [])
        assert "Medical Database Search Results" in output
        assert "test" in output
        assert "consult healthcare professionals" in output.lower()

    def test_disclaimer_present(self):
        results = [{"server": "biomcp", "tool": "test", "result": "ok"}]
        output = _format_tool_results_raw("q", results)
        assert "consult healthcare professionals" in output.lower()


# ============================================================
# mcp_agent_node (async)
# ============================================================

class TestMCPAgentNode:
    """Async LangGraph node — mcp_agent_node."""

    @pytest.mark.asyncio
    async def test_no_messages_returns_state(self, mock_config, empty_state):
        """Empty messages list → returns state with empty AI response."""
        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            # No messages → user_query="" → keyword routing → default calls
            mock_client = AsyncMock()
            r = MagicMock()
            r.success = True
            r.content = "default results"
            mock_client.call_on_server.return_value = r
            mock_get.return_value = mock_client

            result = await mcp_agent_node(empty_state, mock_config)
            assert "messages" in result
            # Should have at least one AIMessage appended
            assert len(result["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_keyword_routing_no_llm(self, sample_state, mock_config):
        """Config has no llm → uses keyword routing."""
        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            r = MagicMock()
            r.success = True
            r.content = "ICD results"
            mock_client.call_on_server.return_value = r
            mock_get.return_value = mock_client

            result = await mcp_agent_node(sample_state, mock_config)
            assert len(result["messages"]) == 2  # original + AI
            assert isinstance(result["messages"][-1], AIMessage)
            # "ICD" keyword → autoicd server
            mock_client.call_on_server.assert_called()
            call_args = mock_client.call_on_server.call_args[0]
            assert call_args[0] == "autoicd"

    @pytest.mark.asyncio
    async def test_llm_routing_with_synthesis(self, sample_state, mock_config_with_llm, mock_llm):
        """Config has llm → LLM routing + LLM synthesis."""
        # LLM routing returns a valid call
        routing_response = [{"server": "healthcare", "tool": "search_drugs", "args": {"query": "diabetes"}}]
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps(routing_response)),  # routing
            MagicMock(content="Diabetes is a metabolic disorder..."),  # synthesis
        ]

        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            r = MagicMock()
            r.success = True
            r.content = "Drug info for diabetes"
            mock_client.call_on_server.return_value = r
            mock_get.return_value = mock_client

            result = await mcp_agent_node(sample_state, mock_config_with_llm)
            assert isinstance(result["messages"][-1], AIMessage)
            assert "Diabetes" in result["messages"][-1].content

    @pytest.mark.asyncio
    async def test_mcp_client_failure(self, sample_state, mock_config):
        """MCP client connection fails → error message in state."""
        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = RuntimeError("Connection refused")

            result = await mcp_agent_node(sample_state, mock_config)
            ai_msg = result["messages"][-1]
            assert isinstance(ai_msg, AIMessage)
            assert "error" in ai_msg.content.lower()

    @pytest.mark.asyncio
    async def test_tool_call_error(self, sample_state, mock_config):
        """Tool call raises exception → error recorded in results."""
        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            mock_client.call_on_server.side_effect = RuntimeError("Tool crashed")
            mock_get.return_value = mock_client

            result = await mcp_agent_node(sample_state, mock_config)
            ai_msg = result["messages"][-1]
            assert isinstance(ai_msg, AIMessage)
            # Raw format should include error
            assert "Error" in ai_msg.content or "error" in ai_msg.content.lower()

    @pytest.mark.asyncio
    async def test_partial_tool_failure(self, sample_state, mock_config):
        """Some tool calls succeed, some fail → mixed results in output."""
        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            success_r = MagicMock()
            success_r.success = True
            success_r.content = "ICD-10 code found"
            error_r = MagicMock()
            error_r.success = False
            error_r.error = "Timeout"

            # keyword "ICD" → autoicd; default also adds healthcare+biomcp
            # Return success for autoicd, error for others
            async def side_effect(server, tool, args):
                if server == "autoicd":
                    return success_r
                return error_r

            mock_client.call_on_server.side_effect = side_effect
            mock_get.return_value = mock_client

            result = await mcp_agent_node(sample_state, mock_config)
            ai_msg = result["messages"][-1]
            assert isinstance(ai_msg, AIMessage)
            assert "ICD-10 code found" in ai_msg.content

    @pytest.mark.asyncio
    async def test_llm_synthesis_failure_falls_back_to_raw(self, sample_state, mock_config_with_llm, mock_llm):
        """LLM synthesis raises → _format_tool_results_raw used."""
        routing_response = [{"server": "biomcp", "tool": "article_search", "args": {"query": "diabetes"}}]
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps(routing_response)),  # routing OK
            RuntimeError("Synthesis LLM failed"),  # synthesis FAILS
        ]

        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            r = MagicMock()
            r.success = True
            r.content = "Article results"
            mock_client.call_on_server.return_value = r
            mock_get.return_value = mock_client

            result = await mcp_agent_node(sample_state, mock_config_with_llm)
            ai_msg = result["messages"][-1]
            assert isinstance(ai_msg, AIMessage)
            # Should fall back to raw format
            assert "Medical Database Search Results" in ai_msg.content

    @pytest.mark.asyncio
    async def test_no_successful_tool_results(self, sample_state, mock_config):
        """All tool calls fail → raw format with errors only."""
        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            r = MagicMock()
            r.success = False
            r.error = "Server down"
            mock_client.call_on_server.return_value = r
            mock_get.return_value = mock_client

            result = await mcp_agent_node(sample_state, mock_config)
            ai_msg = result["messages"][-1]
            assert isinstance(ai_msg, AIMessage)
            assert "Error" in ai_msg.content or "Server down" in ai_msg.content

    @pytest.mark.asyncio
    async def test_result_content_truncated(self, sample_state, mock_config):
        """Tool result content > 3000 chars → truncated to 3000."""
        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            r = MagicMock()
            r.success = True
            r.content = "x" * 5000  # 5000 chars
            mock_client.call_on_server.return_value = r
            mock_get.return_value = mock_client

            result = await mcp_agent_node(sample_state, mock_config)
            ai_msg = result["messages"][-1]
            assert isinstance(ai_msg, AIMessage)
            # The raw format should have truncated content (3000 chars max per result)
            assert len(ai_msg.content) < 5500  # header + truncated + footer

    @pytest.mark.asyncio
    async def test_non_human_message_last(self, mock_config):
        """Last message is AIMessage (not HumanMessage) → uses .content."""
        state = {
            "messages": [
                HumanMessage(content="original question"),
                AIMessage(content="some previous answer"),
            ]
        }
        with patch("agents.mcp_client.get_mcp_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            r = MagicMock()
            r.success = True
            r.content = "results"
            mock_client.call_on_server.return_value = r
            mock_get.return_value = mock_client

            result = await mcp_agent_node(state, mock_config)
            assert isinstance(result["messages"][-1], AIMessage)
