"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
import logging
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, TypedDict, Union

logger = logging.getLogger(__name__)
import concurrent.futures
import getpass
import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel, Field, field_validator

from agents.context_builder import ContextBuilder, MedicalSystemPrompt, compress_history_tags
from agents.error_handler import StopHookValidator, llm_call_with_recovery
from agents.guardrails.local_guardrails import LocalGuardrails
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.mcp_agent import mcp_agent_node
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from request_context import request_id_var

# Vector memory integration (Phase 1: Memory System Upgrade)
try:
    from agents.medical_vector_memory import add_memory, collection_stats, search_memory

    VECTOR_MEMORY_AVAILABLE = True
    logger.info("Medical vector memory loaded successfully")
except Exception as e:
    VECTOR_MEMORY_AVAILABLE = False
    logger.warning(f"Medical vector memory unavailable: {e}")

# Memory Module (Three-tier: Vector → Mem0 → InMemory)
from agents.memory.medical_memory import get_medical_memory
from agents.memory_module import get_memory_store

# Prompt Manager (Centralized template management)
try:
    from prompts.manager import get_prompt_manager

    PROMPT_MANAGER_AVAILABLE = True
except Exception:
    PROMPT_MANAGER_AVAILABLE = False
    logger.warning("[PROMPT] PromptManager not available, using inline prompts")

# Medical Tool System (Phase 4: Unified Tool Interface)
try:
    from tools.registry import get_registry as get_unified_registry

    UNIFIED_REGISTRY_AVAILABLE = True
    logger.info("Unified tool registry loaded successfully")
except ImportError:
    UNIFIED_REGISTRY_AVAILABLE = False
    logger.warning("Unified tool registry not available")

# Optional: Langfuse observability (graceful degradation if not installed)
try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

    langfuse_handler = LangfuseCallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
except (ImportError, Exception):
    langfuse_handler = None
    LANGFUSE_ENABLED = False
    logger.warning("[Langfuse] Not installed or misconfigured. Install: pip install langfuse")

import cv2
import numpy as np

from agents.error_handler import LLMErrorType, RetryExhausted, classify_error
from circuit_breaker import get_all_breaker_stats, llm_breaker, mcp_breaker, web_search_breaker
from config import Config
from observability import agent_metrics

# Phase 5: Medical Planner & Diagnosis Reflection
try:
    from agents.medical_planner import (
        PLANNING_AVAILABLE,
        DiagnosisReflection,
        DiagnosticPlan,
        create_diagnostic_plan,
        get_plan_routing_hints,
        reflect_on_diagnosis,
    )
except ImportError:
    PLANNING_AVAILABLE = False
    logger.warning("[PLANNER] Medical planner module not available")

load_dotenv()

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()
stop_hook = StopHookValidator()  # Post-agent output validation


# Specify a thread
# Dynamic thread config - each session gets unique thread_id
def _make_thread_config(session_id: str | None = None) -> dict:
    """Generate thread config with unique session_id for multi-user support."""
    tid = session_id or str(uuid.uuid4())
    return {"configurable": {"thread_id": tid}}


# Default config for backward compatibility
thread_config = _make_thread_config("default")


# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""

    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer

    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"

    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.75

    # System instructions for the decision agent (loaded from prompts/decision_system.md via PromptManager)
    _decision_prompt_tmpl = get_prompt_manager().get("decision_system") if PROMPT_MANAGER_AVAILABLE else None
    DECISION_SYSTEM_PROMPT = (
        _decision_prompt_tmpl
        if _decision_prompt_tmpl
        else """You are an intelligent medical triage system that routes user queries to
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
    2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    4. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    5. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    6. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.
    7. MCP_AGENT - For queries requiring external biomedical databases and medical coding systems: gene/variant research (ClinVar, UniProt), clinical trials (ClinicalTrials.gov), drug information (FDA, OpenFDA), medical coding (ICD-10-CM, ICD-11, SNOMED CT, LOINC, RxNorm), PubMed literature search, and biomedical entity lookup. Use this when the user asks about specific drugs, genes, clinical trials, medical codes, or needs cross-referencing between medical ontologies.

    Make your decision based on these guidelines:
    - If the user has not uploaded any image, consider the query content: for general chat/greetings use CONVERSATION_AGENT; for medical knowledge questions use RAG_AGENT; for recent/current health situations use WEB_SEARCH_PROCESSOR_AGENT; for queries about drugs, genes, clinical trials, medical codes (ICD/SNOMED/LOINC), or biomedical database lookups use MCP_AGENT.
    - If the user uploads a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    - If the user asks about recent medical developments or current health situations, use the web search pocessor agent.
    - If the user asks specific medical knowledge questions, use the RAG agent.
    - For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.
    - For queries about specific drugs, genes/variants, clinical trials, medical coding systems (ICD-10/ICD-11/SNOMED CT/LOINC/RxNorm), or biomedical entity lookups, use MCP_AGENT.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """
    )

    image_analyzer = ImageAnalysisAgent(config=config)


class AgentState(MessagesState):
    """State maintained across the workflow."""

    # messages: List[BaseMessage]  # Conversation history
    agent_name: str | None  # Current active agent
    current_input: str | dict | None  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: str | None  # Type of medical image if present
    output: str | None  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information
    # Phase 5: Planning & Reflection
    diagnostic_plan: dict | None  # Serialized DiagnosticPlan (JSON-safe)
    plan_hints: dict | None  # Routing hints from planner for route_to_agent


class AgentDecision(BaseModel):
    """Output structure for the decision agent with validation."""

    agent: str = Field(description="Agent name to route to")
    reasoning: str = Field(default="", description="Step-by-step reasoning for selecting this agent")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

    VALID_AGENTS: ClassVar[set] = {
        "CONVERSATION_AGENT",
        "RAG_AGENT",
        "WEB_SEARCH_PROCESSOR_AGENT",
        "BRAIN_TUMOR_AGENT",
        "CHEST_XRAY_AGENT",
        "SKIN_LESION_AGENT",
        "MCP_AGENT",
    }

    @field_validator("agent")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        # Normalize: strip whitespace, uppercase
        v = v.strip().upper().replace(" ", "_")
        if v not in cls.VALID_AGENTS:
            raise ValueError(f"Unknown agent '{v}'. Valid: {cls.VALID_AGENTS}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))  # Clamp to [0, 1]


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)

    # LLM
    decision_model = config.agent_decision.llm

    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)

    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages(
        [("system", AgentConfig.DECISION_SYSTEM_PROMPT), ("human", "{input}")]
    )

    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser

    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None

        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Check input through guardrails if text is present
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                logger.info(f"Selected agent: INPUT GUARDRAILS, Message: {message}")
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True,  # flag to end flow
                }

        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response["image_type"]
            logger.info(f"ANALYZED IMAGE TYPE: {image_type}")

        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False,  # Explicitly set to False for normal flow
        }

    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "plan_diagnosis"

    def reflect_diagnosis(state: AgentState) -> AgentState:
        """
        Phase 5 node: Run diagnosis reflection after agent output.
        Lightweight LLM call to critique diagnosis and suggest follow-ups.
        """
        if not PLANNING_AVAILABLE:
            return state

        # Optimization: skip reflection when RAG already returned with high confidence
        # This saves ~12s of xiaomi-mimo API latency per request
        rag_confidence = state.get("rag_confidence", 0.0)
        if rag_confidence >= 0.75:
            logger.info(f"[REFLECT] Skipping reflection - RAG confidence {rag_confidence:.2f} >= 0.75")
            return state

        messages = state.get("messages", [])
        current_input = state.get("current_input", "")
        plan_data = state.get("diagnostic_plan")

        # Extract agent's output text
        output_text = ""
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                output_text = last_msg.content
            elif isinstance(last_msg, str):
                output_text = last_msg

        if not output_text or len(output_text) < 50:
            logger.debug("[REFLECT] Output too short, skipping reflection")
            return state

        # Get original query
        query = ""
        if isinstance(current_input, str):
            query = current_input
        elif isinstance(current_input, dict):
            query = current_input.get("text", "")

        # Reconstruct plan for context
        from agents.medical_planner import DiagnosticPlan

        try:
            plan_obj = DiagnosticPlan(**plan_data) if plan_data else None
        except Exception as e:
            logger.warning(f"[REFLECT] Failed to reconstruct plan: {e}")
            plan_obj = None

        # Run reflection (cheap LLM call, ~100 tokens)
        reflection = reflect_on_diagnosis(
            llm=config.rag.llm, original_query=query, diagnosis_output=output_text, plan=plan_obj
        )

        if reflection and reflection.needs_follow_up and reflection.follow_up_questions:
            followup_note = f"建议追问: {'; '.join(reflection.follow_up_questions[:2])}"
            logger.info(f"[REFLECT] {followup_note}")
            return {
                **state,
                "plan_hints": {
                    **(state.get("plan_hints") or {}),
                    "reflection_followup": reflection.follow_up_questions,
                    "reflection_notes": reflection.suggested_checks,
                    "confidence_adjustment": reflection.confidence_adjustment,
                },
            }

        logger.info("[REFLECT] Diagnosis looks solid, no follow-up needed")
        return state

    def plan_diagnosis(state: AgentState) -> AgentState:
        """
        Phase 5 node: Generate diagnostic plan for complex medical queries.
        Exploration → Plan → Verification pattern.
        For simple queries, passes through with hints for routing.
        """
        if not PLANNING_AVAILABLE:
            # Planner not loaded, pass through unchanged
            return {**state, "diagnostic_plan": None, "plan_hints": None}

        current_input = state["current_input"]
        if isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        else:
            input_text = str(current_input) if current_input else ""

        if not input_text.strip():
            return {**state, "diagnostic_plan": None, "plan_hints": None}

        try:
            plan = create_diagnostic_plan(input_text)
            hints = get_plan_routing_hints(plan)
            logger.info(
                f"[PLANNER] Plan created: complexity={plan.complexity}, "
                f"reasoning_depth={plan.reasoning_depth}, "
                f"recommended_agents={hints.get('recommended_agents', [])}"
            )
            return {**state, "diagnostic_plan": plan.model_dump(), "plan_hints": hints}
        except Exception as e:
            logger.warning(f"[PLANNER] Planning failed (non-fatal): {e}")
            return {**state, "diagnostic_plan": None, "plan_hints": None}

    def route_to_agent(state: AgentState) -> dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]

        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Phase 5: Inject plan hints into decision context
        plan_hints = state.get("plan_hints")
        plan_context = ""
        if plan_hints and plan_hints.get("recommended_agents"):
            recommended = plan_hints["recommended_agents"]
            plan_context = f"\n[PLANNER HINTS] Recommended agents: {', '.join(recommended)}\n"
            if plan_hints.get("priority_steps"):
                plan_context += f"[PLANNER] Priority: {', '.join(plan_hints['priority_steps'])}\n"
            logger.info(f"[ROUTE] Using planner hints: {recommended}")

        # Build context using ContextBuilder (Phase 2)
        ctx = ContextBuilder()
        ctx.set_system(MedicalSystemPrompt.DECISION)
        if VECTOR_MEMORY_AVAILABLE and input_text:
            try:
                mem_results = search_memory(input_text, top_k=3)
                if mem_results:
                    ctx.set_vector_memory(mem_results)
                    logger.debug(f"[VECTOR_MEMORY] Injected {len(mem_results)} memory results into decision")
            except Exception as e:
                logger.debug(f"[VECTOR_MEMORY] Decision search failed (non-fatal): {e}")
        ctx.set_chat_history(messages, max_messages=6)

        # Phase 4: Inject tool registry summary for better routing decisions
        tool_context = ""
        if UNIFIED_REGISTRY_AVAILABLE:
            try:
                registry = get_unified_registry()
                tool_names = registry.get_all_tool_names()
                if tool_names:
                    tool_context = f"\n\n[AVAILABLE TOOLS]\n{registry.get_summary(max_desc_len=80)}\n"
                    logger.debug(f"[TOOL_REGISTRY] Injected {len(tool_names)} tools into decision context")
            except Exception as e:
                logger.debug(f"[TOOL_REGISTRY] Tool summary failed (non-fatal): {e}")

        decision_input = ctx.build(
            f"User query: {input_text}\n\nHas image: {has_image}\nImage type: {image_type if has_image else 'None'}{plan_context}{tool_context}\n\nBased on this information, which agent should handle this query?"
        )

        # Make the decision with structured output validation + fallback
        # [Phase 3.4] Wrapped with llm_call_with_recovery for error classification
        try:
            with agent_metrics.track("DECISION_AGENT"):
                decision = llm_call_with_recovery(
                    lambda: llm_breaker.call(lambda: decision_chain.invoke({"input": decision_input})),
                    on_context_too_long=lambda: list(messages[-6:]),
                    max_retries=1,
                )
            # Validate through Pydantic model
            validated = AgentDecision(**decision)
            agent_name = validated.agent
            reasoning = validated.reasoning
            confidence = validated.confidence
        except (RetryExhausted, Exception) as e:
            logger.warning(f"Decision chain parse/validation failed: {e}. Falling back to CONVERSATION_AGENT")
            agent_name = "CONVERSATION_AGENT"
            reasoning = f"Fallback due to parse error: {str(e)[:200]}"
            confidence = 0.5

        # Decided agent
        logger.info(f"[{request_id_var.get()}] Decision: {agent_name} (confidence={confidence:.2f})")

        # Update state with decision
        updated_state = {
            **state,
            "agent_name": agent_name,
        }

        # Route based on agent name and confidence
        if confidence < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}

        return {"agent_state": updated_state, "next": agent_name}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        logger.info("Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]

        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Build context using ContextBuilder (Phase 2)
        ctx = ContextBuilder()
        ctx.set_system(MedicalSystemPrompt.CONVERSATION)

        # Phase 1: Vector memory (existing)
        if VECTOR_MEMORY_AVAILABLE:
            try:
                mem_results = search_memory(input_text, min_score=0.45, top_k=3)
                if mem_results:
                    ctx.set_vector_memory(mem_results)
                    logger.debug(f"[VECTOR_MEMORY] Injected {len(mem_results)} memories into conversation")
            except Exception as e:
                logger.debug(f"[VECTOR_MEMORY] Conversation search failed (non-fatal): {e}")

        # Memory Module recall (three-tier: Vector → Mem0 → InMemory)
        try:
            memory = get_memory_store()
            user_id = state.get("thread_id", "default")
            memory_context = memory.recall(user_id, input_text, limit=3)
            if memory_context:
                ctx.set_vector_memory([{"content": memory_context, "score": 0.5, "source": "memory_module"}])
                logger.debug(f"[MEMORY_MODULE] Injected memory context for user {user_id}")
        except Exception as e:
            logger.debug(f"[MEMORY_MODULE] Recall failed (non-fatal): {e}")

        # MedicalMemory: medical-specific recall (allergies, medications, history)
        try:
            med_mem = get_medical_memory()
            if med_mem.available:
                user_id = state.get("thread_id", "default")
                allergies = med_mem.check_allergies(user_id)
                medications = med_mem.check_medications(user_id)
                med_history = med_mem.recall_medical(user_id, input_text, limit=3)
                med_parts = []
                if allergies:
                    med_parts.append(f"已知过敏: {allergies}")
                if medications:
                    med_parts.append(f"当前用药: {medications}")
                if med_history:
                    med_parts.append(f"相关病史: {med_history}")
                if med_parts:
                    ctx.set_vector_memory([{"content": "\n".join(med_parts), "score": 0.7, "source": "medical_memory"}])
                    logger.debug(f"[MEDICAL_MEMORY] Injected medical context for user {user_id}")
        except Exception as e:
            logger.debug(f"[MEDICAL_MEMORY] Recall failed (non-fatal): {e}")

        ctx.set_chat_history(messages, max_messages=20)
        conversation_prompt = ctx.build(input_text)

        # print("Conversation Prompt:", conversation_prompt)

        # [Phase 3.4] Wrapped with llm_call_with_recovery for error classification
        try:
            with agent_metrics.track("CONVERSATION_AGENT"):
                response = llm_call_with_recovery(
                    lambda: llm_breaker.call(lambda: config.conversation.llm.invoke(conversation_prompt)),
                    on_context_too_long=lambda: list(messages[-20:]),
                    max_retries=2,
                )
        except (RetryExhausted, Exception) as e:
            logger.error(f"[CONVERSATION_AGENT] LLM invocation failed: {e}", exc_info=True)
            response = AIMessage(content="I apologize, but I'm experiencing technical difficulties. Please try again.")

        return {**state, "output": response, "agent_name": "CONVERSATION_AGENT"}

    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # Initialize the RAG agent

        logger.info("Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)

        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        recent_context = ""
        for msg in messages[-rag_context_limit:]:  # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        # Memory Module recall (three-tier: Vector → Mem0 → InMemory)
        try:
            memory = get_memory_store()
            user_id = state.get("thread_id", "default")
            memory_context = memory.recall(user_id, query if isinstance(query, str) else str(query), limit=3)
            if memory_context:
                recent_context = memory_context + "\n\n" + recent_context
                logger.debug(f"[MEMORY_MODULE] Injected memory context for user {user_id} in RAG")
        except Exception as e:
            logger.debug(f"[MEMORY_MODULE] RAG recall failed (non-fatal): {e}")

        # [Phase 3.4] Wrapped with llm_call_with_recovery for error classification
        try:
            with agent_metrics.track("RAG_AGENT"):
                response = llm_call_with_recovery(
                    lambda: llm_breaker.call(lambda: rag_agent.process_query(query, chat_history=recent_context)),
                )
        except (RetryExhausted, Exception) as e:
            logger.error(f"[RAG_AGENT] RAG query failed: {e}", exc_info=True)
            return {
                **state,
                "output": AIMessage(
                    content="I apologize, but the medical knowledge retrieval system encountered an error. Please try again."
                ),
                "agent_name": "RAG_AGENT",
                "retrieval_confidence": 0.0,
                "insufficient_info": True,
            }
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        logger.info(f"Retrieval Confidence: {retrieval_confidence}")
        logger.info(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        insufficient_info = False
        response_content = response["response"]

        # Extract the content properly based on type
        if isinstance(response_content, dict) and hasattr(response_content, "content"):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content

        logger.debug(f"Response text type: {type(response_text)}")
        logger.debug(f"Response text preview: {response_text[:100]}...")

        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text
            or "I don't have enough information" in response_text
            or "don't have enough information" in response_text.lower()
            or "not enough information" in response_text.lower()
            or "insufficient information" in response_text.lower()
            or "cannot answer" in response_text.lower()
            or "unable to answer" in response_text.lower()
        ):
            logger.warning("RAG response indicates insufficient information")
            logger.warning(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        logger.debug(f"Insufficient info flag set to: {insufficient_info}")

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)

            # Store high-confidence RAG results to vector memory
            if VECTOR_MEMORY_AVAILABLE:
                try:
                    add_memory(
                        text=f"Medical RAG Q&A: Q={query[:200]} A={response_text[:500]}",
                        metadata={
                            "type": "rag_qa",
                            "confidence": str(retrieval_confidence),
                            "sources": str(len(response.get("sources", []))),
                            "query_preview": query[:100],
                        },
                    )
                    logger.debug(f"[VECTOR_MEMORY] Stored RAG result (confidence={retrieval_confidence:.2f})")
                except Exception as e:
                    logger.debug(f"[VECTOR_MEMORY] RAG store failed (non-fatal): {e}")
        else:
            response_output = AIMessage(content="")

        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info,
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        logger.info("Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        logger.info("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")

        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]:  # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        # [Phase 3.4] Wrapped with llm_call_with_recovery for error classification
        try:
            with agent_metrics.track("WEB_SEARCH_PROCESSOR_AGENT"):
                processed_response = llm_call_with_recovery(
                    lambda: web_search_breaker.call(
                        lambda: web_search_processor.process_web_search_results(
                            query=state["current_input"], chat_history=recent_context
                        )
                    ),
                    max_retries=1,
                )
        except (RetryExhausted, Exception) as e:
            import traceback

            tb = traceback.format_exc()
            try:
                with open(r"D:\Code\Multi-Agent-Medical-Assistant\debug_web_search.log", "w") as df:
                    df.write(f"EXCEPTION: {e}\n\n{tb}")
            except Exception:
                pass
            logger.error(f"[WEB_SEARCH_PROCESSOR_AGENT] Web search processing failed: {e}", exc_info=True)
            processed_response = AIMessage(
                content=f"I apologize, but the web search system encountered an error: {e}. Please try again."
            )

        # print("######### DEBUG WEB SEARCH:", processed_response)

        if state["agent_name"] is not None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents,
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # Debug prints
        logger.debug(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        logger.debug(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")

        # Redirect if confidence is low or if response indicates insufficient info
        if state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or state.get(
            "insufficient_info", False
        ):
            logger.info("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient

    def run_parallel_retrieval(state: AgentState) -> AgentState:
        """Run RAG and WebSearch in parallel, merge best result (Phase 3.3)."""
        logger.info("Selected agent: PARALLEL_RETRIEVAL (RAG + WebSearch concurrent)")

        rag_state = None
        web_state = None

        # True parallel: submit both immediately, wait for RAG first
        rag_state = None
        web_state = None

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            # Submit both simultaneously
            rag_future = executor.submit(run_rag_agent, state)
            web_future = executor.submit(run_web_search_processor_agent, state)

            # Wait for RAG first with short timeout
            try:
                rag_state = rag_future.result(timeout=25)
                rag_conf = (rag_state or {}).get("retrieval_confidence", 0.0)
                if rag_conf >= config.rag.min_retrieval_confidence:
                    logger.info(f"[PARALLEL] RAG succeeded (conf={rag_conf:.2f}), canceling WebSearch")
                    web_future.cancel()
                    return {**rag_state, "agent_name": "RAG_AGENT"}
            except concurrent.futures.TimeoutError:
                logger.warning("RAG timed out (25s), waiting for WebSearch")
            except Exception as e:
                logger.warning(f"RAG failed: {e}")

            # Wait for WebSearch (already running)
            try:
                web_state = web_future.result(timeout=50)
            except concurrent.futures.TimeoutError:
                logger.warning("WebSearch timed out (50s)")
                web_future.cancel()
            except Exception as e:
                logger.warning(f"WebSearch failed: {e}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # Merge strategy: prefer RAG if high confidence, else WebSearch, else combine
        rag_conf = (rag_state or {}).get("retrieval_confidence", 0.0)
        web_conf = (web_state or {}).get("web_search_confidence", 0.0)

        merged_state = {**state}

        if rag_conf >= config.rag.min_retrieval_confidence:
            # RAG is good enough - use it as primary
            merged_state.update(
                {
                    "agent_name": "RAG_AGENT",
                    "output": rag_state.get("output", state.get("output")),
                    "retrieval_confidence": rag_conf,
                }
            )
            logger.info(f"[PARALLEL] Using RAG result (confidence={rag_conf:.2f})")
        elif web_conf >= 0.5:
            # Use web search result
            merged_state.update(
                {
                    "agent_name": "WEB_SEARCH_PROCESSOR_AGENT",
                    "output": web_state.get("output", state.get("output")),
                    "web_search_confidence": web_conf,
                }
            )
            logger.info(f"[PARALLEL] Using WebSearch result (confidence={web_conf:.2f})")
        else:
            # Both low - combine contexts for richer input to downstream
            combined_output = ""
            if rag_state and rag_state.get("output"):
                combined_output += f"[RAG Context]\n{rag_state['output']}\n\n"
            if web_state and web_state.get("output"):
                combined_output += f"[Web Context]\n{web_state['output']}"
            merged_state.update(
                {
                    "agent_name": "PARALLEL_RETRIEVAL",
                    "output": combined_output or state.get("output"),
                    "retrieval_confidence": max(rag_conf, web_conf),
                }
            )
            logger.info(f"[PARALLEL] Combined RAG({rag_conf:.2f}) + Web({web_conf:.2f})")

        return merged_state

    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis using EfficientNet-B0 classifier."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        logger.info("Selected agent: BRAIN_TUMOR_AGENT")

        # Classify brain MRI: glioma, meningioma, pituitary, no_tumor
        try:
            with agent_metrics.track("BRAIN_TUMOR_AGENT"):
                result = AgentConfig.image_analyzer.classify_brain_tumor(image_path)
        except Exception as e:
            logger.error(f"[BRAIN_TUMOR_AGENT] Image analysis failed: {e}", exc_info=True)
            return {
                **state,
                "output": AIMessage(
                    content="I apologize, but the brain tumor analysis encountered an error. Please ensure the image is a valid brain MRI scan and try again."
                ),
                "needs_human_validation": True,
                "agent_name": "BRAIN_TUMOR_AGENT",
            }

        if isinstance(result, dict):
            pred = result.get("prediction", "unknown")
            conf = result.get("confidence", 0.0)
            probs = result.get("all_probabilities", {})

            if pred == "no_tumor":
                response = AIMessage(
                    content=f"The analysis of the uploaded brain MRI image indicates **NO TUMOR** detected. "
                    f"Confidence: {conf:.1%}. This is a reassuring result, but please consult a neurologist for confirmation."
                )
            elif pred == "error":
                response = AIMessage(
                    content=f"Error analyzing the brain MRI image: {result.get('error', 'Unknown error')}"
                )
            else:
                # Tumor detected (glioma, meningioma, or pituitary)
                tumor_info = {
                    "glioma": "glioma tumor - a type that originates in the glial cells of the brain",
                    "meningioma": "meningioma - a tumor arising from the meninges (membranes surrounding the brain)",
                    "pituitary": "pituitary adenoma - a tumor in the pituitary gland at the base of the brain",
                }
                desc = tumor_info.get(pred, pred)

                # Build probability breakdown
                prob_str = ", ".join(f"{k}: {v:.1%}" for k, v in probs.items())

                response = AIMessage(
                    content=f"The analysis of the uploaded brain MRI image indicates a **POSITIVE** result for **{pred.upper()}**. "
                    f"\n\nDetails: The image shows characteristics consistent with {desc}. "
                    f"Confidence: {conf:.1%}. "
                    f"\n\nProbability breakdown: {prob_str}. "
                    f"\n\n⚠️ **Important**: This is an AI-assisted analysis and should NOT be used as a final diagnosis. "
                    f"Please consult a qualified neurologist or neurosurgeon for professional evaluation."
                )
        else:
            response = AIMessage(
                content="The uploaded image could not be processed. Please ensure it is a valid brain MRI scan."
            )

        return {**state, "output": response, "needs_human_validation": True, "agent_name": "BRAIN_TUMOR_AGENT"}

    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        logger.info("Selected agent: CHEST_XRAY_AGENT")

        # classify chest x-ray into covid or normal
        try:
            with agent_metrics.track("CHEST_XRAY_AGENT"):
                predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)
        except Exception as e:
            logger.error(f"[CHEST_XRAY_AGENT] Chest X-ray analysis failed: {e}", exc_info=True)
            return {
                **state,
                "output": AIMessage(
                    content="I apologize, but the chest X-ray analysis encountered an error. Please ensure the image is a valid chest X-ray and try again."
                ),
                "needs_human_validation": True,
                "agent_name": "CHEST_XRAY_AGENT",
            }

        if predicted_class == "covid19":
            response = AIMessage(
                content="The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**."
            )
        elif predicted_class == "normal":
            response = AIMessage(
                content="The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**."
            )
        else:
            response = AIMessage(
                content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image."
            )

        # response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT",
        }

    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        logger.info("Selected agent: SKIN_LESION_AGENT")

        # Segment skin lesion
        try:
            with agent_metrics.track("SKIN_LESION_AGENT"):
                predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)
        except Exception as e:
            logger.error(f"[SKIN_LESION_AGENT] Skin lesion analysis failed: {e}", exc_info=True)
            return {
                **state,
                "output": AIMessage(
                    content="I apologize, but the skin lesion analysis encountered an error. Please ensure the image is a valid dermoscopy image and try again."
                ),
                "needs_human_validation": True,
                "agent_name": "SKIN_LESION_AGENT",
            }

        if predicted_mask:
            response = AIMessage(
                content="Following is the analyzed **segmented** output of the uploaded skin lesion image:"
            )
        else:
            response = AIMessage(
                content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image."
            )

        # response = AIMessage(content="This would be handled by the skin lesion agent, analyzing the skin image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT",
        }

    def handle_human_validation(state: AgentState) -> dict:
        """Prepare for human validation if needed, with stopHook check."""
        # ── StopHook: post-agent validation (confidence + safety + completeness) ──
        agent_name = state.get("agent", "unknown")
        if agent_name not in ("unknown", "HUMAN_VALIDATION", "INTAKE_AGENT"):
            hook_result = stop_hook.validate_output(state, agent_name)
            if not hook_result.passed:
                logger.warning(
                    f"[STOP_HOOK] {agent_name} failed validation: {hook_result.reason} (action={hook_result.action})"
                )
                # Force human validation on hook failure
                state["needs_human_validation"] = True
                # Append hook failure info to messages for human reviewer
                hook_msg = (
                    f"[StopHook Alert] Agent '{agent_name}' output flagged: {hook_result.reason}. "
                    f"Suggested action: {hook_result.action}. Please review."
                )
                state.setdefault("messages", []).append(AIMessage(content=hook_msg))
        # ── End StopHook ──

        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}

    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        logger.info("Selected agent: HUMAN_VALIDATION")

        # Append validation request to the existing output
        validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {**state, "output": validation_message, "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"}

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content

        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")

            # If validation input exists
            if validation_input.lower().startswith(("yes", "no")):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")

                # If validation is 'No', modify the output
                if validation_input.lower().startswith("no"):
                    fallback_message = AIMessage(
                        content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies."
                    )
                    return {**state, "messages": [validation_response, fallback_message], "output": fallback_message}

                return {**state, "messages": validation_response}

        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Apply output sanitization
        sanitized_output = guardrails.check_output(output_text, input_text)
        # sanitized_output = output_text

        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output

        return {**state, "messages": sanitized_message, "output": sanitized_message}

    # Create the workflow graph
    workflow = StateGraph(AgentState)

    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("plan_diagnosis", plan_diagnosis)  # Phase 5: Planning node
    workflow.add_node("reflect_diagnosis", reflect_diagnosis)  # Phase 5: Reflection node
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("PARALLEL_RETRIEVAL", run_parallel_retrieval)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("MCP_AGENT", mcp_agent_node)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)

    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "plan_diagnosis": "route_to_agent",  # Phase 5: skip planner, go directly to router (planner always fails ~5s)
        },
    )

    # Phase 5: Plan → Route (planner feeds hints to router)
    workflow.add_edge("plan_diagnosis", "route_to_agent")

    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "PARALLEL_RETRIEVAL",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "MCP_AGENT": "MCP_AGENT",
            "needs_validation": "PARALLEL_RETRIEVAL",  # Default to parallel RAG+Web if confidence is low
        },
    )

    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_edge("PARALLEL_RETRIEVAL", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")
    workflow.add_edge("MCP_AGENT", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)

    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "reflect_diagnosis",  # Phase 5: Reflect before guardrails
        },
    )
    workflow.add_edge("reflect_diagnosis", "apply_guardrails")  # Phase 5: Reflect → Guardrails

    # workflow.add_edge("human_validation", END)

    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
    }


def process_query(query: str | dict, conversation_history: list[BaseMessage] | None = None) -> str:
    """
    Process a user query through the agent decision system.

    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages, NOT NEEDED ANYMORE since the state saves the conversation history now

    Returns:
        Response from the appropriate agent
    """
    # Initialize the graph
    graph = create_agent_graph()

    # Phase 4: Initialize tool registry (lazy, only on first call)
    if UNIFIED_REGISTRY_AVAILABLE:
        try:
            get_unified_registry().initialize()
        except Exception as e:
            logger.debug(f"[TOOL_REGISTRY] Init failed (non-fatal): {e}")

    # # Save Graph Flowchart
    # image_bytes = graph.get_graph().draw_mermaid_png()
    # decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # cv2.imwrite("./assets/graph.png", decoded)
    # print("Graph flowchart saved in assets.")

    # Initialize state
    state = init_agent_state()
    # if conversation_history:
    #     state["messages"] = conversation_history

    # Add the current query
    state["current_input"] = query

    # To handle image upload case
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."

    state["messages"] = [HumanMessage(content=query)]

    # Invoke with Langfuse callbacks if available
    runnable_config = {"configurable": {"thread_id": "default"}}
    invoke_kwargs = {"state": state, "config": runnable_config}
    if LANGFUSE_ENABLED and langfuse_handler:
        result = graph.invoke(state, runnable_config, callbacks=[langfuse_handler])
    else:
        result = graph.invoke(state, runnable_config)
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Phase 51: Compress history tags + summarize old conversation messages
    if len(result["messages"]) > config.max_conversation_history:
        try:
            messages = result["messages"]
            # Step 1: Pre-compress thinking and tool_result tags (Phase 2)
            messages = compress_history_tags(messages)

            if config.summarize_conversation_history and len(messages) > config.max_conversation_history:
                keep = config.summary_keep_recent
                old_messages = messages[:-keep]
                recent_messages = messages[-keep:]

                # Build summary text from old messages
                summary_parts = []
                for m in old_messages:
                    role = getattr(m, "type", "unknown")
                    content = getattr(m, "content", str(m))
                    if content:
                        summary_parts.append(f"{role}: {content[:200]}")

                if summary_parts:
                    summary_prompt = (
                        "Summarize the following medical conversation history concisely, "
                        "preserving key medical topics, symptoms discussed, and recommendations given:\n\n"
                        + "\n".join(summary_parts)
                    )
                    # [Phase 3.4] Wrapped with llm_call_with_recovery for error classification
                    summary_response = llm_call_with_recovery(
                        lambda: config.conversation.llm.invoke(summary_prompt),
                        max_retries=1,
                    )
                    summary_text = getattr(summary_response, "content", str(summary_response))

                    from langchain_core.messages import SystemMessage

                    summary_msg = SystemMessage(content=f"[Conversation Summary]\n{summary_text}")
                    result["messages"] = [summary_msg, *recent_messages]
                    logger.info(
                        f"[Phase51] Compressed + summarized {len(old_messages)} messages → summary + {keep} recent"
                    )

                    # Store conversation summary to vector memory
                    if VECTOR_MEMORY_AVAILABLE:
                        try:
                            add_memory(
                                text=f"Conversation summary: {summary_text}",
                                metadata={
                                    "type": "conversation_summary",
                                    "messages_summarized": str(len(old_messages)),
                                    "topic_preview": summary_text[:100],
                                },
                            )
                            logger.debug("[VECTOR_MEMORY] Stored Phase51 conversation summary")
                        except Exception as e:
                            logger.debug(f"[VECTOR_MEMORY] Phase51 store failed (non-fatal): {e}")
                else:
                    result["messages"] = recent_messages
            else:
                # Even if no summarization needed, store compressed messages back
                result["messages"] = messages
        except Exception as e:
            logger.warning(f"[Phase51] Summarization failed, falling back to truncation: {e}")
            result["messages"] = result["messages"][-config.max_conversation_history :]
    elif len(result["messages"]) > config.max_conversation_history:
        # Fallback: simple truncation (original behavior)
        result["messages"] = result["messages"][-config.max_conversation_history :]

    # visualize conversation history in console
    for m in result["messages"]:
        logger.debug(f"Graph:\n{m}")

    # Step 1.4: Save conversation to memory_module
    try:
        memory = get_memory_store()
        user_id = "default"
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            user_id = thread_id
        # Extract last Q&A for memory
        last_messages = result.get("messages", [])
        if len(last_messages) >= 2:
            user_msg = ""
            ai_msg = ""
            for m in reversed(last_messages):
                if isinstance(m, AIMessage) and not ai_msg:
                    ai_msg = m.content[:500]
                elif isinstance(m, HumanMessage) and not user_msg:
                    user_msg = m.content[:500]
                if user_msg and ai_msg:
                    break
            if user_msg and ai_msg:
                memory.remember(user_id, f"Q: {user_msg}\nA: {ai_msg}", metadata={"type": "conversation"})
                logger.debug(f"[MEMORY_MODULE] Stored conversation for user {user_id}")

                # MedicalMemory: store with medical category detection
                try:
                    med_mem = get_medical_memory()
                    if med_mem.available:
                        combined = (user_msg + " " + ai_msg).lower()
                        category = "general"
                        if any(kw in combined for kw in ["过敏", "allerg", "不良反应"]):
                            category = "allergy"
                        elif any(kw in combined for kw in ["药", "用药", "medication", "处方", "剂量"]):
                            category = "medication"
                        elif any(kw in combined for kw in ["诊断", "diagnos", "检查", "化验"]):
                            category = "diagnosis"
                        elif any(kw in combined for kw in ["症状", "symptom", "头痛", "发热", "咳嗽", "疼痛"]):
                            category = "symptom"
                        elif any(kw in combined for kw in ["病史", "history", "既往", "慢性"]):
                            category = "history"
                        med_mem.remember_medical(user_id, f"Q: {user_msg}\nA: {ai_msg}", category=category)
                        logger.debug(f"[MEDICAL_MEMORY] Stored (category={category}) for user {user_id}")
                except Exception as e:
                    logger.debug(f"[MEDICAL_MEMORY] Store failed (non-fatal): {e}")
    except Exception as e:
        logger.debug(f"[MEMORY_MODULE] Remember failed (non-fatal): {e}")

    # Add the response to conversation history
    return result


def process_query_streaming(query: str | dict, conversation_history: list[BaseMessage] | None = None):
    """
    Generator-based streaming version of process_query.
    Uses graph.stream() to yield intermediate node results as each agent completes,
    then yields the final result.

    Yields:
        dict: {"type": "node_start", "node": node_name} when a node starts
        dict: {"type": "node_end", "node": node_name, "output_preview": ...} when a node completes
        dict: {"type": "final", "result": result} when the full graph is done

    Usage:
        for event in process_query_streaming(query):
            if event["type"] == "final":
                return event["result"]
    """
    graph = create_agent_graph()
    langfuse_handler = None
    if LANGFUSE_ENABLED:
        try:
            from langfuse.callback import CallbackHandler

            langfuse_handler = CallbackHandler()
        except Exception:
            pass

    state = {
        "messages": [],
        "next": None,
        "current_agent": None,
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
    }

    if isinstance(query, dict):
        if "image_data" in query:
            image_msg = HumanMessage(
                content=[
                    {"type": "text", "text": query.get("text", "")},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query['image_data']}"}},
                ]
            )
            state["messages"].append(image_msg)
            state["has_image"] = True
            state["image_type"] = query.get("image_type", "unknown")
            state["current_input"] = query.get("text", "")
        else:
            state["messages"].append(HumanMessage(content=str(query)))
            state["current_input"] = str(query)
    else:
        state["messages"].append(HumanMessage(content=str(query)))
        state["current_input"] = str(query)

    runnable_config = {"configurable": {"thread_id": "default"}}
    callbacks = [langfuse_handler] if LANGFUSE_ENABLED and langfuse_handler else []

    # Stream node-by-node progress
    final_result = None
    try:
        for chunk in graph.stream(state, runnable_config, callbacks=callbacks):
            # graph.stream() returns dicts like {node_name: output} per step
            for node_name, node_output in chunk.items():
                # Yield node completion event
                output_preview = ""
                if node_output and "output" in node_output and node_output["output"]:
                    preview = str(node_output["output"])
                    output_preview = preview[:100] + "..." if len(preview) > 100 else preview
                yield {
                    "type": "node_end",
                    "node": node_name,
                    "output_preview": output_preview,
                    "needs_human_validation": node_output.get("needs_human_validation", False)
                    if node_output
                    else False,
                }
                final_result = node_output
    except Exception as e:
        logger.error(f"[STREAMING] Graph stream error: {e}")
        yield {"type": "error", "message": str(e)}
        return

    # Post-processing: compress + summarize (same as process_query)
    if final_result and "messages" in final_result:
        result = final_result
        if len(result["messages"]) > config.max_conversation_history:
            try:
                messages = result["messages"]
                messages = compress_history_tags(messages)
                if config.summarize_conversation_history and len(messages) > config.max_conversation_history:
                    keep = config.summary_keep_recent
                    old_messages = messages[:-keep]
                    recent_messages = messages[-keep:]
                    summary_parts = []
                    for m in old_messages:
                        role = getattr(m, "type", "unknown")
                        content = getattr(m, "content", str(m))
                        if content:
                            summary_parts.append(f"{role}: {content[:200]}")
                    if summary_parts:
                        summary_prompt = (
                            "Summarize the following medical conversation history concisely, "
                            "preserving key medical topics, symptoms discussed, and recommendations given:\n\n"
                            + "\n".join(summary_parts)
                        )
                        summary_response = llm_call_with_recovery(
                            lambda: config.conversation.llm.invoke(summary_prompt),
                            max_retries=1,
                        )
                        summary_text = getattr(summary_response, "content", str(summary_response))
                        from langchain_core.messages import SystemMessage

                        summary_msg = SystemMessage(content=f"[Conversation Summary]\n{summary_text}")
                        result["messages"] = [summary_msg, *recent_messages]
                        logger.info(f"[Phase51/Streaming] Compressed + summarized {len(old_messages)} messages")
                    else:
                        result["messages"] = recent_messages
                else:
                    result["messages"] = messages
            except Exception as e:
                logger.warning(f"[Phase51/Streaming] Summarization failed: {e}")
                result["messages"] = result["messages"][-config.max_conversation_history :]
        elif len(result["messages"]) > config.max_conversation_history:
            result["messages"] = result["messages"][-config.max_conversation_history :]

        yield {"type": "final", "result": result}
