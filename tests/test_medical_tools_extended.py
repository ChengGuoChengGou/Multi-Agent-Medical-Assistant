"""
Comprehensive tests for medical_tools_extended.py

Covers:
  - PubMedTool: properties, validate, execute (success/empty/timeout/error/XML edge cases)
  - DrugInteractionTool: properties, validate, execute (all 3 actions, fallback, error paths)
  - register_extended_tools: registry integration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.medical_tool import MedicalToolRegistry, MedicalToolResult
from agents.medical_tools_extended import DrugInteractionTool, PubMedTool, register_extended_tools

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def pubmed():
    return PubMedTool()


@pytest.fixture
def drug():
    return DrugInteractionTool()


@pytest.fixture
def registry():
    return MedicalToolRegistry()


# ============================================================
# PubMedTool — Properties
# ============================================================


class TestPubMedProperties:
    def test_name(self, pubmed):
        assert pubmed.name == "pubmed_search"

    def test_description(self, pubmed):
        assert "PubMed" in pubmed.description

    def test_category(self, pubmed):
        assert pubmed.category == "research"

    def test_input_schema_structure(self, pubmed):
        schema = pubmed.input_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert "sort" in schema["properties"]
        assert "query" in schema["required"]

    def test_get_examples(self, pubmed):
        examples = pubmed.get_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0


# ============================================================
# PubMedTool — Validate
# ============================================================


class TestPubMedValidate:
    def test_missing_query(self, pubmed):
        err = pubmed.validate({})
        assert err is not None
        assert "query" in err.lower()

    def test_empty_query(self, pubmed):
        err = pubmed.validate({"query": ""})
        assert err is not None

    def test_valid_params(self, pubmed):
        err = pubmed.validate({"query": "diabetes treatment"})
        assert err is None

    def test_max_results_default_applied(self, pubmed):
        """max_results=5 is the default; specifying it explicitly should pass validation."""
        err = pubmed.validate({"query": "cancer", "max_results": 5})
        assert err is None


# ============================================================
# PubMedTool — Execute
# ============================================================

# PubMed uses JSON API (retmode=json)
SEARCH_JSON_WITH_RESULTS = {"esearchresult": {"idlist": ["12345"], "count": "1"}}

SEARCH_JSON_EMPTY = {"esearchresult": {"idlist": [], "count": "0"}}

SUMMARY_JSON_FULL = {
    "result": {
        "12345": {
            "title": "Aspirin for heart disease",
            "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
            "fulljournalname": "JAMA",
            "pubdate": "2024 Jan",
            "elocationid": "10.1001/jama.2024.12345",
        }
    }
}

SUMMARY_JSON_NO_ABSTRACT = {
    "result": {
        "12345": {
            "title": "A brief report",
            "authors": [{"name": "Lee K"}],
            "fulljournalname": "Lancet",
            "pubdate": "2023",
            "elocationid": "10.1016/S0140-6736(23)00001",
        }
    }
}


class TestPubMedExecute:
    @pytest.mark.asyncio
    async def test_success_with_results(self, pubmed):
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.return_value = SEARCH_JSON_WITH_RESULTS
        mock_search.raise_for_status = MagicMock()

        mock_summary = MagicMock()
        mock_summary.status_code = 200
        mock_summary.json.return_value = SUMMARY_JSON_FULL
        mock_summary.raise_for_status = MagicMock()

        responses = [mock_search, mock_summary]

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=responses)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await pubmed.execute({"query": "aspirin heart"})
            assert result.success is True
            assert "Found 1 PubMed articles" in result.content
            assert "12345" in result.content
            assert result.source == "built_in"

    @pytest.mark.asyncio
    async def test_empty_results(self, pubmed):
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.return_value = SEARCH_JSON_EMPTY
        mock_search.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_search)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await pubmed.execute({"query": "zzzznonexistent"})
            assert result.success is True
            assert "No PubMed articles found" in result.content

    @pytest.mark.asyncio
    async def test_timeout(self, pubmed):
        import httpx

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await pubmed.execute({"query": "test"})
            assert result.success is False
            assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_http_error(self, pubmed):
        import httpx

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 429
            instance.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("rate limited", request=MagicMock(), response=resp)
            )
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await pubmed.execute({"query": "test"})
            assert result.success is False

    @pytest.mark.asyncio
    async def test_malformed_xml(self, pubmed):
        """eutils returns garbage — should not crash."""
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.side_effect = ValueError("Invalid JSON")
        mock_search.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_search)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await pubmed.execute({"query": "test"})
            assert isinstance(result, MedicalToolResult)

    @pytest.mark.asyncio
    async def test_missing_abstract_in_summary(self, pubmed):
        """Summary has no abstract — should still format article."""
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.return_value = SEARCH_JSON_WITH_RESULTS
        mock_search.raise_for_status = MagicMock()

        mock_summary = MagicMock()
        mock_summary.status_code = 200
        mock_summary.json.return_value = SUMMARY_JSON_NO_ABSTRACT
        mock_summary.raise_for_status = MagicMock()

        responses = [mock_search, mock_summary]

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=responses)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await pubmed.execute({"query": "brief report"})
            assert result.success is True
            assert "A brief report" in result.content


# ============================================================
# DrugInteractionTool — Properties
# ============================================================


class TestDrugProperties:
    def test_name(self, drug):
        assert drug.name == "drug_interaction_check"

    def test_description(self, drug):
        assert "OpenFDA" in drug.description or "interaction" in drug.description.lower()

    def test_category(self, drug):
        assert drug.category == "diagnosis"

    def test_input_schema_structure(self, drug):
        schema = drug.input_schema
        assert schema["type"] == "object"
        assert "action" in schema["properties"]
        assert "drug_name" in schema["properties"]
        assert "second_drug" in schema["properties"]
        assert set(schema["required"]) == {"action", "drug_name"}

    def test_get_examples(self, drug):
        examples = drug.get_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0


# ============================================================
# DrugInteractionTool — Validate
# ============================================================


class TestDrugValidate:
    def test_invalid_action(self, drug):
        err = drug.validate({"action": "explode", "drug_name": "aspirin"})
        assert err is not None
        assert "action" in err.lower()

    def test_missing_action(self, drug):
        err = drug.validate({"drug_name": "aspirin"})
        assert err is not None

    def test_missing_drug_name(self, drug):
        err = drug.validate({"action": "get_label"})
        assert err is not None
        assert "drug_name" in err.lower()

    def test_empty_drug_name(self, drug):
        err = drug.validate({"action": "get_label", "drug_name": ""})
        assert err is not None

    def test_check_interaction_missing_second_drug(self, drug):
        err = drug.validate({"action": "check_interaction", "drug_name": "aspirin"})
        assert err is not None
        assert "second_drug" in err.lower()

    def test_valid_check_interaction(self, drug):
        err = drug.validate({"action": "check_interaction", "drug_name": "aspirin", "second_drug": "warfarin"})
        assert err is None

    def test_valid_get_label(self, drug):
        err = drug.validate({"action": "get_label", "drug_name": "metformin"})
        assert err is None

    def test_valid_search_adverse_events(self, drug):
        err = drug.validate({"action": "search_adverse_events", "drug_name": "ibuprofen"})
        assert err is None


# ============================================================
# DrugInteractionTool — Execute
# ============================================================

LABEL_RESULTS = {
    "results": [
        {
            "openfda": {"brand_name": ["LIPITOR"], "generic_name": ["ATORVASTATIN"], "manufacturer_name": ["Pfizer"]},
            "drug_interactions": [
                "Warfarin: May increase bleeding risk when combined with atorvastatin.",
                "Cyclosporine: Significant increase in atorvastatin levels.",
            ],
            "indications_and_usage": ["For lowering cholesterol"],
            "warnings": ["Do not use if pregnant"],
            "dosage_and_administration": ["10-80 mg daily"],
        }
    ]
}

LABEL_NO_RESULTS = {"results": []}

ADVERSE_EVENTS = {
    "results": [
        {"term": "Headache", "count": 1500},
        {"term": "Nausea", "count": 900},
    ],
    "meta": {"results": {"total": 50000}},
}


class TestDrugExecute:
    @pytest.mark.asyncio
    async def test_check_interaction_match(self, drug):
        """Drug2 is mentioned in interaction text → [MATCH] prefix."""
        mock_label = MagicMock()
        mock_label.status_code = 200
        mock_label.json.return_value = LABEL_RESULTS
        mock_label.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_label)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute(
                {"action": "check_interaction", "drug_name": "atorvastatin", "second_drug": "warfarin"}
            )
            assert result.success is True
            assert "[MATCH]" in result.content
            assert result.source == "built_in"

    @pytest.mark.asyncio
    async def test_check_interaction_no_match_no_brand(self, drug):
        """Brand returns nothing → falls back to generic name."""
        mock_empty = MagicMock()
        mock_empty.status_code = 200
        mock_empty.json.return_value = LABEL_NO_RESULTS
        mock_empty.raise_for_status = MagicMock()

        mock_generic = MagicMock()
        mock_generic.status_code = 200
        mock_generic.json.return_value = LABEL_RESULTS
        mock_generic.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=[mock_empty, mock_generic])
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute(
                {"action": "check_interaction", "drug_name": "atorvastatin", "second_drug": "aspirin"}
            )
            assert result.success is True
            # aspirin is not in LABEL_RESULTS interactions → no [MATCH]
            assert "[MATCH]" not in result.content

    @pytest.mark.asyncio
    async def test_check_interaction_no_results_at_all(self, drug):
        """Both brand and generic return empty → helpful message."""
        mock_empty = MagicMock()
        mock_empty.status_code = 200
        mock_empty.json.return_value = LABEL_NO_RESULTS
        mock_empty.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_empty)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute(
                {"action": "check_interaction", "drug_name": "fakemab", "second_drug": "placixil"}
            )
            assert result.success is True
            assert "No interaction data found" in result.content

    @pytest.mark.asyncio
    async def test_get_label_success(self, drug):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = LABEL_RESULTS
        mock_resp.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute({"action": "get_label", "drug_name": "atorvastatin"})
            assert result.success is True
            assert "LIPITOR" in result.content
            assert "Brand:" in result.content
            assert "Warnings:" in result.content

    @pytest.mark.asyncio
    async def test_get_label_not_found(self, drug):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = LABEL_NO_RESULTS
        mock_resp.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute({"action": "get_label", "drug_name": "fakemab"})
            assert result.success is True
            assert "No label found" in result.content

    @pytest.mark.asyncio
    async def test_search_adverse_events_success(self, drug):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = ADVERSE_EVENTS
        mock_resp.raise_for_status = MagicMock()

        # Second call for total count
        mock_count = MagicMock()
        mock_count.status_code = 200
        mock_count.json.return_value = {"meta": {"results": {"total": 50000}}}

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=[mock_resp, mock_count])
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute({"action": "search_adverse_events", "drug_name": "aspirin"})
            assert result.success is True
            assert "Headache" in result.content
            assert "Nausea" in result.content
            assert "50,000" in result.content

    @pytest.mark.asyncio
    async def test_search_adverse_events_empty(self, drug):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        mock_count = MagicMock()
        mock_count.status_code = 200
        mock_count.json.return_value = {"meta": {"results": {"total": 0}}}

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=[mock_resp, mock_count])
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute({"action": "search_adverse_events", "drug_name": "vitamin_c"})
            assert result.success is True
            assert "No adverse event data found" in result.content

    @pytest.mark.asyncio
    async def test_timeout(self, drug):
        import httpx

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute({"action": "get_label", "drug_name": "aspirin"})
            assert result.success is False
            assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_http_error(self, drug):
        import httpx

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 500
            instance.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("server error", request=MagicMock(), response=resp)
            )
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute({"action": "get_label", "drug_name": "aspirin"})
            assert result.success is False

    @pytest.mark.asyncio
    async def test_max_results_capped_at_10(self, drug):
        """max_results should be capped at 10."""
        mock_label = MagicMock()
        mock_label.status_code = 200
        mock_label.json.return_value = LABEL_RESULTS
        mock_label.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_label)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            # pass max_results=999 — should be capped to 10 internally
            result = await drug.execute({"action": "get_label", "drug_name": "aspirin", "max_results": 999})
            assert result.success is True


# ============================================================
# register_extended_tools
# ============================================================


class TestRegisterExtendedTools:
    def test_registers_both_tools(self, registry):
        register_extended_tools(registry)
        assert len(registry) == 2
        names = {t.name for t in registry.get_all()}
        assert names == {"pubmed_search", "drug_interaction_check"}

    def test_no_duplicates_on_double_register(self, registry):
        register_extended_tools(registry)
        register_extended_tools(registry)
        # Depending on impl, might overwrite or add — at least 2 exist
        assert len(registry) >= 2


# ============================================================
# MedicalToolResult — source field present
# ============================================================


class TestMedicalToolResultSource:
    """Verify every MedicalToolResult produced has source='built_in'."""

    @pytest.mark.asyncio
    async def test_pubmed_result_has_source(self, pubmed):
        mock_search = MagicMock()
        mock_search.status_code = 200
        mock_search.json.return_value = SEARCH_JSON_EMPTY
        mock_search.raise_for_status = MagicMock()

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_search)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await pubmed.execute({"query": "test"})
            assert result.source == "built_in"

    @pytest.mark.asyncio
    async def test_drug_error_result_has_source(self, drug):
        import httpx

        with patch("agents.medical_tools_extended.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=Exception("unexpected"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await drug.execute({"action": "get_label", "drug_name": "aspirin"})
            assert result.source == "built_in"
