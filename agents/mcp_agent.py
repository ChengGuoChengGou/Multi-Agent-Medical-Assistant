"""
MCP Agent Node for Multi-Agent Medical Assistant

This agent handles medical queries that require external tool access:
- BioMCP: Biomedical research data (genes, variants, articles, trials, drugs)
- AutoICD: Medical coding (ICD-10-CM, ICD-11, LOINC, SNOMED CT)
- Healthcare MCP: FDA drugs, PubMed, clinical trials, health topics

Integrated into the LangGraph as a new agent node alongside the existing 6 agents.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ============================================================
# MCP Tool Routing - decide which tools to call based on query
# ============================================================

MCP_ROUTING_PROMPT = """You are a medical query router. Based on the user's medical question, 
determine which MCP tools to call and with what arguments.

Available MCP tools:
1. **BioMCP** (biomedical research):
   - article_search: Search PubMed articles
   - variant_search: Search genetic variants (ClinVar, MyVariant)
   - gene_search: Search gene information
   - trial_search: Search clinical trials (ClinicalTrials.gov)
   - drug_search: Search drug information
   - disease_search: Search disease information
   - pathway_search: Search biological pathways
   - protein_search: Search protein structures
   - adverse_event_search: Search FDA adverse events

2. **AutoICD** (medical coding):
   - code_diagnosis: Extract diagnoses from clinical text and map to ICD-10-CM
   - code_translate: Translate codes between standards (ICD-10↔ICD-11)
   - reference_lookup: Look up medical code details
   - reference_search: Search medical codes
   - chart_audit: Audit clinical charts for coding accuracy

3. **Healthcare MCP** (general medical):
   - search_drugs: Search FDA drug database
   - search_pubmed: Search PubMed articles
   - search_trials: Search clinical trials
   - search_health_topics: Search health topics
   - search_icd_codes: Search ICD-10 codes
   - search_medrxiv: Search medRxiv preprints
   - get_drug_interactions: Check drug interactions
   - calculate_bmi: Calculate BMI

User query: {query}

Respond with a JSON array of tool calls to make (can be multiple for comprehensive answers):
[{{"server": "biomcp|autoicd|healthcare", "tool": "tool_name", "args": {{"param": "value"}}}}]

Only output the JSON array, nothing else."""


def _determine_mcp_tools_llm(query: str, llm) -> list[dict[str, Any]]:
    """
    Use the LLM to determine which MCP tools to call for a given query.
    Falls back to keyword-based routing if LLM fails.
    """
    try:
        response = llm.invoke(
            [
                SystemMessage(content="You are a medical tool router. Output only valid JSON."),
                HumanMessage(content=MCP_ROUTING_PROMPT.format(query=query)),
            ]
        )
        # Parse JSON from response
        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        tool_calls = json.loads(content)
        if isinstance(tool_calls, list):
            return tool_calls
    except Exception as e:
        logger.warning(f"LLM tool routing failed: {e}, falling back to keyword routing")

    # Fallback: keyword-based routing
    return _determine_mcp_tools_keyword(query)


def _determine_mcp_tools_keyword(query: str) -> list[dict[str, Any]]:
    """
    Keyword-based fallback routing for MCP tools.
    """
    query_lower = query.lower()
    calls = []

    # ICD coding / diagnosis mapping
    coding_keywords = [
        "icd",
        "编码",
        "诊断编码",
        "code",
        "diagnosis code",
        "billing code",
        "病历编码",
        "疾病编码",
        "dsm",
        "snomed",
        "loinc",
    ]
    if any(kw in query_lower for kw in coding_keywords):
        calls.append(
            {"server": "autoicd", "tool": "reference_search", "args": {"query": query, "code_type": "icd-10-cm"}}
        )

    # Drug queries
    drug_keywords = [
        "drug",
        "药物",
        "药品",
        "medication",
        "prescription",
        "副作用",
        "side effect",
        "interaction",
        "相互作用",
        "药代动力学",
        "fda批准",
    ]
    if any(kw in query_lower for kw in drug_keywords):
        calls.append({"server": "healthcare", "tool": "search_drugs", "args": {"query": query}})

    # Clinical trials
    trial_keywords = ["trial", "临床试验", "clinical study", "临床研究", "randomized", "rct", "phase", "入组", "招募"]
    if any(kw in query_lower for kw in trial_keywords):
        calls.append({"server": "biomcp", "tool": "trial_search", "args": {"query": query}})

    # Genetic / variant queries
    gene_keywords = [
        "gene",
        "基因",
        "mutation",
        "突变",
        "variant",
        "变异",
        "snp",
        "genomic",
        "genetic",
        "brca",
        "egfr",
        "her2",
        "alk",
    ]
    if any(kw in query_lower for kw in gene_keywords):
        calls.append({"server": "biomcp", "tool": "variant_search", "args": {"query": query}})
        calls.append({"server": "biomcp", "tool": "gene_search", "args": {"query": query}})

    # PubMed / research articles
    article_keywords = [
        "研究",
        "论文",
        "paper",
        "study",
        "research",
        "pubmed",
        "article",
        "文献",
        "meta分析",
        "meta-analysis",
        "systematic review",
        "preprint",
        "预印本",
    ]
    if any(kw in query_lower for kw in article_keywords):
        calls.append({"server": "biomcp", "tool": "article_search", "args": {"query": query}})
        calls.append({"server": "healthcare", "tool": "search_pubmed", "args": {"query": query}})

    # Disease / pathology
    disease_keywords = [
        "disease",
        "疾病",
        "disorder",
        "syndrome",
        "综合征",
        "pathology",
        "病理",
        "diagnosis",
        "诊断",
        "症状",
        "symptom",
    ]
    if any(kw in query_lower for kw in disease_keywords) and not calls:
        calls.append({"server": "biomcp", "tool": "disease_search", "args": {"query": query}})

    # If no specific match, try general search across multiple servers
    if not calls:
        calls.append({"server": "healthcare", "tool": "search_health_topics", "args": {"query": query}})
        calls.append({"server": "biomcp", "tool": "article_search", "args": {"query": query}})

    return calls


# ============================================================
# MCP Agent Node Function
# ============================================================


async def mcp_agent_node(state: dict[str, Any], config: Any) -> dict[str, Any]:
    """
    LangGraph node: MCP Agent.

    Handles queries that benefit from external medical databases:
    - Biomedical research (PubMed, ClinVar, ClinicalTrials.gov)
    - Medical coding (ICD-10, ICD-11, SNOMED CT)
    - Drug information (FDA, drug interactions)

    Uses MCP protocol to communicate with external tool servers.
    """
    from agents.mcp_client import get_mcp_client

    messages = state["messages"]
    last_message = messages[-1] if messages else None
    user_query = ""
    if isinstance(last_message, HumanMessage) or hasattr(last_message, "content"):
        user_query = last_message.content

    print(f"[MCP_AGENT] Processing: {user_query[:100]}...")

    # Get the LLM from config for tool routing
    llm = config.llm if hasattr(config, "llm") else None

    # Determine which MCP tools to call
    if llm:
        tool_calls = _determine_mcp_tools_llm(user_query, llm)
    else:
        tool_calls = _determine_mcp_tools_keyword(user_query)

    print(f"[MCP_AGENT] Planned {len(tool_calls)} tool calls: {[tc['server'] + '.' + tc['tool'] for tc in tool_calls]}")

    # Get MCP client and execute tool calls
    try:
        mcp_client = await get_mcp_client()
    except Exception as e:
        logger.error(f"[MCP_AGENT] Failed to get MCP client: {e}")
        error_msg = AIMessage(
            content=f"I encountered an error connecting to external medical databases: {e!s}. "
            f"Falling back to internal knowledge for your query."
        )
        state["messages"] = [*messages, error_msg]
        return state

    # Execute tool calls
    tool_results = []
    for call in tool_calls:
        server = call.get("server", "")
        tool = call.get("tool", "")
        args = call.get("args", {})

        try:
            result = await mcp_client.call_on_server(server, tool, args)
            if result.success:
                tool_results.append(
                    {
                        "server": server,
                        "tool": tool,
                        "result": result.content[:3000],  # Truncate long results
                    }
                )
                print(f"[MCP_AGENT] ✅ {server}.{tool}: {len(result.content)} chars")
            else:
                print(f"[MCP_AGENT] ❌ {server}.{tool}: {result.error}")
                tool_results.append(
                    {
                        "server": server,
                        "tool": tool,
                        "error": result.error,
                    }
                )
        except Exception as e:
            logger.error(f"[MCP_AGENT] Tool call error {server}.{tool}: {e}")
            tool_results.append(
                {
                    "server": server,
                    "tool": tool,
                    "error": str(e),
                }
            )

    # Generate response using LLM with tool results
    if llm and tool_results:
        context_parts = []
        for tr in tool_results:
            if "result" in tr:
                context_parts.append(f"### {tr['server'].upper()} - {tr['tool']}\n{tr['result']}")

        if context_parts:
            synthesis_prompt = f"""Based on the following external medical database results, 
provide a comprehensive answer to the user's question.

User question: {user_query}

External database results:
{chr(10).join(context_parts)}

Provide a clear, accurate, and well-structured medical response. 
Include source citations where available. If information conflicts, note the discrepancy.
Always remind the user to consult healthcare professionals for clinical decisions."""

            try:
                response = llm.invoke(
                    [
                        SystemMessage(
                            content="You are a medical AI assistant with access to external medical databases. Provide accurate, evidence-based responses."
                        ),
                        HumanMessage(content=synthesis_prompt),
                    ]
                )
                answer = response.content
            except Exception as e:
                logger.error(f"[MCP_AGENT] LLM synthesis failed: {e}")
                answer = _format_tool_results_raw(user_query, tool_results)
        else:
            answer = _format_tool_results_raw(user_query, tool_results)
    else:
        answer = _format_tool_results_raw(user_query, tool_results)

    ai_message = AIMessage(content=answer)
    state["messages"] = [*messages, ai_message]
    print(f"[MCP_AGENT] Response generated: {len(answer)} chars")
    return state


def _format_tool_results_raw(query: str, results: list[dict]) -> str:
    """Format tool results into a readable response when LLM synthesis is unavailable."""
    parts = [f"## Medical Database Search Results\n\n**Query:** {query}\n"]
    for tr in results:
        parts.append(f"\n### Source: {tr['server'].upper()} ({tr['tool']})")
        if "result" in tr:
            parts.append(tr["result"])
        elif "error" in tr:
            parts.append(f"❌ Error: {tr['error']}")
    parts.append("\n\n⚕️ *Please consult healthcare professionals for clinical decisions.*")
    return "\n".join(parts)
