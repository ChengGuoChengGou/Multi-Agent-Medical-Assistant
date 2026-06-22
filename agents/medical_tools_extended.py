"""
Extended Medical Tools

Standalone medical tools that implement the MedicalTool interface
but don't depend on MCP servers. Useful as fallbacks when MCP is down.

Phase 4.5: 新增工具 (PubMed API, 药物交互检查)
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from agents.medical_tool import MedicalTool, MedicalToolRegistry, MedicalToolResult

logger = logging.getLogger(__name__)


# ============================================================
# PubMed Tool (NCBI E-utilities)
# ============================================================


class PubMedTool(MedicalTool):
    """
    PubMed search via NCBI E-utilities API.

    Free API, no key required (but rate-limited to 3 req/sec without key).
    Provides direct PubMed access as fallback when MCP servers are down.
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self._name = "pubmed_search"
        self._description = (
            "Search PubMed for biomedical literature using NCBI E-utilities. "
            "Returns article titles, authors, journal, publication date, and PMID. "
            "Useful for finding recent research papers on medical topics."
        )
        self._category = "research"
        self._input_schema = self._build_schema()

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
        return self._input_schema

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (PubMed search syntax supported, e.g. 'diabetes AND metformin')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 20)",
                    "default": 5,
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order: 'relevance' or 'date'",
                    "enum": ["relevance", "date"],
                    "default": "relevance",
                },
            },
            "required": ["query"],
        }

    def validate(self, params: Dict[str, Any]) -> Optional[str]:
        query = params.get("query", "")
        if not query or not isinstance(query, str):
            return "query is required and must be a non-empty string"
        if len(query) > 500:
            return "query is too long (max 500 chars)"
        max_r = params.get("max_results", 5)
        if not isinstance(max_r, int) or max_r < 1 or max_r > 20:
            return "max_results must be an integer between 1 and 20"
        return None

    def get_examples(self) -> List[str]:
        return [
            "diabetes treatment guidelines 2024",
            "CRISPR gene therapy clinical trials",
            "machine learning radiology diagnosis",
        ]

    async def execute(self, params: Dict[str, Any]) -> MedicalToolResult:
        query = params["query"]
        max_results = min(params.get("max_results", 5), 20)
        sort = params.get("sort", "relevance")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Step 1: Search for PMIDs
                search_params = {
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "sort": sort,
                    "retmode": "json",
                }
                resp = await client.get(f"{self.BASE_URL}/esearch.fcgi", params=search_params)
                resp.raise_for_status()
                search_data = resp.json()

                id_list = search_data.get("esearchresult", {}).get("idlist", [])
                if not id_list:
                    return MedicalToolResult(
                        success=True,
                        content=f"No PubMed articles found for: {query}",
                        tool_name=self.name,
                        source="built_in",
                    )

                # Step 2: Fetch article details
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json",
                }
                resp = await client.get(f"{self.BASE_URL}/esummary.fcgi", params=fetch_params)
                resp.raise_for_status()
                summary_data = resp.json()

                # Parse results
                articles = []
                result_data = summary_data.get("result", {})
                for pmid in id_list:
                    article = result_data.get(pmid, {})
                    if not isinstance(article, dict):
                        continue
                    articles.append(
                        {
                            "pmid": pmid,
                            "title": article.get("title", "N/A"),
                            "authors": [a.get("name", "") for a in article.get("authors", [])[:5]],
                            "journal": article.get("fulljournalname", article.get("source", "N/A")),
                            "pub_date": article.get("pubdate", "N/A"),
                            "doi": article.get("elocationid", "N/A"),
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        }
                    )

                # Format output
                lines = [f"Found {len(articles)} PubMed articles for: {query}\n"]
                for i, art in enumerate(articles, 1):
                    authors_str = ", ".join(art["authors"][:3])
                    if len(art["authors"]) > 3:
                        authors_str += " et al."
                    lines.append(
                        f"{i}. [{art['pmid']}] {art['title']}\n"
                        f"   Authors: {authors_str}\n"
                        f"   Journal: {art['journal']} ({art['pub_date']})\n"
                        f"   URL: {art['url']}"
                    )

                return MedicalToolResult(
                    success=True,
                    content="\n".join(lines),
                    tool_name=self.name,
                    metadata={
                        "articles": articles,
                        "total_found": search_data.get("esearchresult", {}).get("count", len(articles)),
                    },
                    source="built_in",
                )

        except httpx.TimeoutException:
            return MedicalToolResult(
                success=False, content="PubMed API timeout", tool_name=self.name, source="built_in", error="timeout"
            )
        except httpx.HTTPStatusError as e:
            return MedicalToolResult(
                success=False,
                content=f"PubMed API error: {e.response.status_code}",
                tool_name=self.name,
                source="built_in",
                error=str(e),
            )
        except Exception as e:
            return MedicalToolResult(
                success=False,
                content=f"PubMed search failed: {e}",
                tool_name=self.name,
                source="built_in",
                error=str(e),
            )


# ============================================================
# Drug Interaction Tool (OpenFDA)
# ============================================================


class DrugInteractionTool(MedicalTool):
    """
    Drug interaction checker using OpenFDA API.

    Free API, no key required. Checks for known drug-drug interactions,
    adverse events, and drug labels.
    """

    OPENFDA_URL = "https://api.fda.gov/drug"

    def __init__(self):
        self._name = "drug_interaction_check"
        self._description = (
            "Check drug interactions, adverse events, and label information "
            "using the OpenFDA API. Supports checking interactions between "
            "two drugs, or retrieving drug label warnings."
        )
        self._category = "diagnosis"
        self._input_schema = self._build_schema()

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
        return self._input_schema

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["check_interaction", "get_label", "search_adverse_events"],
                },
                "drug_name": {
                    "type": "string",
                    "description": "Primary drug name (generic or brand name)",
                },
                "second_drug": {
                    "type": "string",
                    "description": "Second drug name for interaction check (optional)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default: 3)",
                    "default": 3,
                },
            },
            "required": ["action", "drug_name"],
        }

    def validate(self, params: Dict[str, Any]) -> Optional[str]:
        action = params.get("action", "")
        if action not in ("check_interaction", "get_label", "search_adverse_events"):
            return "action must be one of: check_interaction, get_label, search_adverse_events"
        drug = params.get("drug_name", "")
        if not drug or not isinstance(drug, str):
            return "drug_name is required and must be a non-empty string"
        if action == "check_interaction":
            second = params.get("second_drug", "")
            if not second:
                return "second_drug is required for interaction check"
        return None

    def get_examples(self) -> List[str]:
        return [
            "check interaction between warfarin and aspirin",
            "get label info for metformin",
            "search adverse events for ibuprofen",
        ]

    async def execute(self, params: Dict[str, Any]) -> MedicalToolResult:
        action = params["action"]
        drug_name = params["drug_name"]
        second_drug = params.get("second_drug", "")
        max_results = min(params.get("max_results", 3), 10)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                if action == "check_interaction":
                    return await self._check_interaction(client, drug_name, second_drug, max_results)
                elif action == "get_label":
                    return await self._get_label(client, drug_name)
                elif action == "search_adverse_events":
                    return await self._search_adverse_events(client, drug_name, max_results)
                else:
                    return MedicalToolResult(
                        success=False,
                        content=f"Unknown action: {action}",
                        tool_name=self.name,
                        source="built_in",
                        error="invalid_action",
                    )

        except httpx.TimeoutException:
            return MedicalToolResult(
                success=False, content="OpenFDA API timeout", tool_name=self.name, source="built_in", error="timeout"
            )
        except httpx.HTTPStatusError as e:
            return MedicalToolResult(
                success=False,
                content=f"OpenFDA API error: {e.response.status_code}",
                tool_name=self.name,
                source="built_in",
                error=str(e),
            )
        except Exception as e:
            return MedicalToolResult(
                success=False, content=f"Drug check failed: {e}", tool_name=self.name, source="built_in", error=str(e)
            )

    async def _check_interaction(
        self, client: httpx.AsyncClient, drug1: str, drug2: str, max_results: int
    ) -> MedicalToolResult:
        """Check drug interactions by searching drug labels for interaction warnings."""
        # Search for drug1's label mentioning drug2
        query = f'openfda.brand_name:"{drug1}" AND _exists_:"drug_interactions"'
        resp = await client.get(f"{self.OPENFDA_URL}/label.json", params={"search": query, "limit": max_results})
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            # Try generic name
            query = f'openfda.generic_name:"{drug1}" AND _exists_:"drug_interactions"'
            resp = await client.get(f"{self.OPENFDA_URL}/label.json", params={"search": query, "limit": max_results})
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

        if not results:
            return MedicalToolResult(
                success=True,
                content=f"No interaction data found for {drug1} in OpenFDA. Try checking with a pharmacist or clinical database.",
                tool_name=self.name,
                source="built_in",
            )

        lines = [f"Drug Interaction Information: {drug1} + {drug2}\n"]
        for i, result in enumerate(results, 1):
            interactions = result.get("drug_interactions", ["No interaction data available"])
            # Check if drug2 is mentioned
            interaction_text = " ".join(interactions) if isinstance(interactions, list) else str(interactions)

            drug2_lower = drug2.lower()
            if drug2_lower in interaction_text.lower():
                lines.append(f"[MATCH] Label #{i} mentions {drug2}:")
                for line in interactions[:10]:
                    if isinstance(line, str):
                        lines.append(f"  • {line[:300]}")
            else:
                lines.append(f"\nLabel #{i} drug interactions (may not mention {drug2} directly):")
                for line in interactions[:5]:
                    if isinstance(line, str):
                        lines.append(f"  • {line[:200]}")

        lines.append(
            "\n⚠️ This is reference information only. Always consult a healthcare provider for drug interaction advice."
        )

        return MedicalToolResult(
            success=True,
            content="\n".join(lines),
            tool_name=self.name,
            metadata={"drug1": drug1, "drug2": drug2, "labels_found": len(results)},
            source="built_in",
        )

    async def _get_label(self, client: httpx.AsyncClient, drug_name: str) -> MedicalToolResult:
        """Get drug label information."""
        query = f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"'
        resp = await client.get(f"{self.OPENFDA_URL}/label.json", params={"search": query, "limit": 1})
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return MedicalToolResult(
                success=True, content=f"No label found for: {drug_name}", tool_name=self.name, source="built_in"
            )

        label = results[0]
        openfda = label.get("openfda", {})

        lines = [f"Drug Label Information: {drug_name}\n"]
        if openfda.get("brand_name"):
            lines.append(f"Brand: {', '.join(openfda['brand_name'])}")
        if openfda.get("generic_name"):
            lines.append(f"Generic: {', '.join(openfda['generic_name'])}")
        if openfda.get("manufacturer_name"):
            lines.append(f"Manufacturer: {', '.join(openfda['manufacturer_name'])}")

        for field_name, display_name in [
            ("indications_and_usage", "Indications"),
            ("warnings", "Warnings"),
            ("dosage_and_administration", "Dosage"),
            ("contraindications", "Contraindications"),
            ("adverse_reactions", "Adverse Reactions"),
        ]:
            values = label.get(field_name)
            if values:
                text = values[0] if isinstance(values, list) else values
                lines.append(f"\n{display_name}:")
                lines.append(f"  {text[:500]}")

        return MedicalToolResult(
            success=True,
            content="\n".join(lines),
            tool_name=self.name,
            metadata={"openfda": openfda},
            source="built_in",
        )

    async def _search_adverse_events(
        self, client: httpx.AsyncClient, drug_name: str, max_results: int
    ) -> MedicalToolResult:
        """Search for adverse events reported for a drug."""
        query = f'patient.drug.openfda.brand_name:"{drug_name}" OR patient.drug.openfda.generic_name:"{drug_name}"'
        resp = await client.get(
            f"{self.OPENFDA_URL}/event.json",
            params={"search": query, "limit": max_results, "count": "patient.reaction.reactionmeddrapt.exact"},
        )
        resp.raise_for_status()
        data = resp.json()

        # Get top reactions count
        top_reactions = data.get("results", [])[:10]

        lines = [f"Top Adverse Events reported for: {drug_name}\n"]
        if top_reactions:
            for i, r in enumerate(top_reactions, 1):
                lines.append(f"  {i}. {r.get('term', 'N/A')} — {r.get('count', 0)} reports")
        else:
            lines.append("  No adverse event data found.")

        # Also get total count
        count_resp = await client.get(f"{self.OPENFDA_URL}/event.json", params={"search": query, "limit": 0})
        if count_resp.status_code == 200:
            total = count_resp.json().get("meta", {}).get("results", {}).get("total", 0)
            lines.append(f"\nTotal adverse event reports: {total:,}")

        lines.append("\n⚠️ Data from FDA Adverse Event Reporting System (FAERS). Reporting does not imply causation.")

        return MedicalToolResult(
            success=True,
            content="\n".join(lines),
            tool_name=self.name,
            source="built_in",
        )


# ============================================================
# Registration helper
# ============================================================


def register_extended_tools(registry: MedicalToolRegistry) -> None:
    """Register all extended medical tools into the given registry."""
    registry.register(PubMedTool())
    registry.register(DrugInteractionTool())
    logger.info("[ExtendedTools] Registered PubMed + DrugInteraction tools")
