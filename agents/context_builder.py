"""
Context Builder for Multi-Agent Medical Chatbot

Implements segmented context assembly, conversation history compression,
and unified medical system prompts. Phase 2 of the memory system refactoring.

Segments:
  1. system_prompt  - Role-specific medical system instructions
  2. vector_memory  - Semantic search results from vector DB
  3. medical_kb     - RAG-retrieved medical literature context
  4. chat_history   - Compressed conversation history
  5. rag_results    - Structured RAG output for downstream agents
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Medical System Prompts (unified, extracted from agent_decision + response_generator)
# ---------------------------------------------------------------------------


class MedicalSystemPrompt:
    """
    Centralized medical system prompts for all agent roles.
    Replaces inline prompt strings scattered across agent_decision.py and response_generator.py.
    """

    DECISION = (
        "You are an intelligent medical triage system that routes user queries to "
        "the appropriate specialized agent. Analyze the user's request and determine "
        "which agent is best suited based on query content, images, and conversation context.\n\n"
        "Available agents:\n"
        "1. CONVERSATION_AGENT - General chat, greetings, non-medical questions.\n"
        "2. RAG_AGENT - Specific medical knowledge questions from established literature "
        "(brain tumors, COVID-19 detection, deep learning diagnostics).\n"
        "3. WEB_SEARCH_PROCESSOR_AGENT - Recent medical developments, current outbreaks, "
        "time-sensitive medical information.\n"
        "4. BRAIN_TUMOR_AGENT - Brain MRI image analysis for tumor detection/segmentation.\n"
        "5. CHEST_XRAY_AGENT - Chest X-ray image analysis for abnormalities.\n"
        "6. SKIN_LESION_AGENT - Skin lesion image classification (benign/malignant).\n"
        "7. MCP_AGENT - External biomedical databases: ClinVar, UniProt, ClinicalTrials.gov, "
        "FDA/OpenFDA, ICD-10-CM, ICD-11, SNOMED CT, LOINC, RxNorm, PubMed.\n\n"
        "Routing guidelines:\n"
        "- No image: general chat→CONVERSATION; medical knowledge→RAG; recent/timely→WEB_SEARCH; "
        "drugs/genes/codes→MCP.\n"
        "- With image: route to appropriate vision agent first.\n"
        "- Recent developments: WEB_SEARCH.\n"
        "- Specific medical knowledge: RAG.\n"
        "- General conversation: CONVERSATION (but vision agent takes priority if image uploaded).\n\n"
        'Respond in JSON: {{"agent": "AGENT_NAME", "reasoning": "...", "confidence": 0.95}}'
    )

    CONVERSATION = (
        "You are a highly knowledgeable and professional medical assistant. "
        "Your goal is to provide accurate, helpful, and empathetic medical information.\n\n"
        "### Core Responsibilities:\n"
        "- **Medical Guidance**: Answer medical questions using verified knowledge. "
        "Always recommend consulting a healthcare professional for diagnosis/treatment.\n"
        "- **Complex Queries**: Route complex queries to RAG or web search if needed.\n"
        "- **Follow-ups**: Track conversation context for follow-up questions.\n"
        "- **Medical Images**: Redirect medical images to appropriate analysis agents.\n\n"
        "### Response Guidelines:\n"
        "1. **General Conversations**: Friendly, engaging, concise unless detail is needed.\n"
        "2. **Medical Questions**: Structured responses with cause, diagnosis, treatment.\n"
        "   Recommend consulting a healthcare professional.\n"
        "3. **Emotional Support**: Empathetic tone for health concerns.\n"
        "4. **Uncertainty**: If unsure, state clearly and suggest professional consultation.\n"
        "5. **Safety**: Never provide definitive diagnoses. Use qualified language "
        "(e.g., 'This may indicate...', 'It is advisable to consult...').\n"
        "6. **Formatting**: Use bullet points for readability. Bold key terms."
    )

    RAG = (
        "You are a medical assistant providing accurate information based on verified medical sources.\n\n"
        "### Table Handling:\n"
        "When using tabular data from context:\n"
        "1. Present with proper markdown table formatting (headers + separators).\n"
        "2. Re-format for readability; mention any new components introduced.\n"
        "3. Interpret tabular data explicitly.\n"
        "4. Replace reference numbers with actual values (paper titles, authors) if available.\n"
        "5. Summarize trends or patterns from tables.\n\n"
        "### Response Instructions:\n"
        "1. Answer based ONLY on provided context.\n"
        "2. If context lacks relevant info: \"I don't have enough information to answer this question "
        'based on the provided context."\n'
        "3. Do not use prior knowledge outside the context.\n"
        "4. Be concise and accurate.\n"
        "5. Use markdown headings/sub-headings for structure.\n"
        "6. Omit sections not meaningful for chatbot replies (e.g., explicit references).\n"
        "7. Use exact values from context; do not fabricate.\n"
        "8. Do not repeat the question in the answer."
    )


# ---------------------------------------------------------------------------
# History Compression
# ---------------------------------------------------------------------------


def compress_history_tags(
    messages: list,
    max_tag_len: int = 200,
    keep_recent: int = 4,
) -> list:
    """
    Compress conversation history for medical context.

    - Keeps recent `keep_recent` messages at full fidelity.
    - For older messages:
      - <thinking>...</thinking> → <thinking>[compressed: N chars]</thinking>
      - <tool_result>...</tool_result> → <tool_result>[compressed: N chars]</tool_result>
      - Very long assistant messages → truncated with [...]

    Returns a new list (does not mutate input).
    """
    if not messages:
        return []

    result = []
    n = len(messages)

    for i, msg in enumerate(messages):
        # Keep recent messages untouched
        if i >= n - keep_recent:
            result.append(msg)
            continue

        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            result.append(msg)
            continue

        compressed = content

        # Compress <thinking> blocks
        thinking_pattern = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)

        def _compress_thinking(m):
            inner = m.group(1).strip()
            if len(inner) > max_tag_len:
                return f"<thinking>[{len(inner)} chars reasoning omitted]</thinking>"
            return m.group(0)

        compressed = thinking_pattern.sub(_compress_thinking, compressed)

        # Compress <tool_result> blocks
        tool_pattern = re.compile(r"<tool_result>(.*?)</tool_result>", re.DOTALL)

        def _compress_tool(m):
            inner = m.group(1).strip()
            if len(inner) > max_tag_len:
                return f"<tool_result>[{len(inner)} chars tool output omitted]</tool_result>"
            return m.group(0)

        compressed = tool_pattern.sub(_compress_tool, compressed)

        # Truncate very long remaining content
        if len(compressed) > 1000:
            compressed = compressed[:950] + " [...]"

        # Create a copy of the message with compressed content
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

            if isinstance(msg, HumanMessage):
                new_msg = HumanMessage(content=compressed)
            elif isinstance(msg, AIMessage):
                new_msg = AIMessage(content=compressed)
            elif isinstance(msg, SystemMessage):
                new_msg = SystemMessage(content=compressed)
            elif isinstance(msg, ToolMessage):
                new_msg = ToolMessage(content=compressed, tool_call_id=getattr(msg, "tool_call_id", ""))
            else:
                new_msg = msg.__class__(content=compressed) if hasattr(msg, "__class__") else msg
        except Exception:
            new_msg = msg

        result.append(new_msg)

    return result


# ---------------------------------------------------------------------------
# Context Segments (data container)
# ---------------------------------------------------------------------------


@dataclass
class ContextSegments:
    """Container for assembled context segments."""

    system_prompt: str = ""
    vector_memory: str = ""
    medical_kb: str = ""
    chat_history: str = ""
    rag_results: str = ""

    def to_prompt(self, separator: str = "\n\n---\n\n") -> str:
        """Join non-empty segments into a single prompt string."""
        parts = []
        if self.system_prompt:
            parts.append(f"[System Instructions]\n{self.system_prompt}")
        if self.vector_memory:
            parts.append(f"[Relevant Medical Memory]\n{self.vector_memory}")
        if self.medical_kb:
            parts.append(f"[Medical Knowledge Base]\n{self.medical_kb}")
        if self.chat_history:
            parts.append(f"[Conversation History]\n{self.chat_history}")
        if self.rag_results:
            parts.append(f"[Retrieved Information]\n{self.rag_results}")
        return separator.join(parts)


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """
    Builds structured context from multiple sources for medical agents.

    Usage:
        builder = ContextBuilder(max_history_chars=3000, compress_old=True)
        segments = builder.build_decision_context(messages, vector_results="...")
        prompt = segments.to_prompt()
    """

    def __init__(
        self,
        max_history_chars: int = 3000,
        max_vector_results: int = 3,
        compress_old: bool = True,
        keep_recent: int = 4,
    ):
        self.max_history_chars = max_history_chars
        self.max_vector_results = max_vector_results
        self.compress_old = compress_old
        self.keep_recent = keep_recent
        # Fluent API state
        self._system_prompt = ""
        self._vector_memory = ""
        self._chat_history = ""
        self._extra_sections = []

    # ---- Fluent API (setter pattern) ----

    def set_system(self, prompt: str) -> "ContextBuilder":
        self._system_prompt = prompt
        return self

    def set_vector_memory(self, memory_results, min_score: float = 0.0) -> "ContextBuilder":
        if isinstance(memory_results, str):
            self._vector_memory = memory_results
        else:
            self._vector_memory = self.format_vector_memory(memory_results)
        return self

    def set_chat_history(self, messages: list, max_messages: int = 0) -> "ContextBuilder":
        if max_messages > 0:
            messages = messages[-max_messages:]
        self._chat_history = self.format_chat_history(messages)
        return self

    def build(self, user_input: str = "") -> str:
        parts = []
        if self._system_prompt:
            parts.append(f"[SYSTEM]\n{self._system_prompt}")
        if self._vector_memory:
            parts.append(f"[MEMORY]\n{self._vector_memory}")
        if self._chat_history:
            parts.append(f"[HISTORY]\n{self._chat_history}")
        for extra in self._extra_sections:
            parts.append(extra)
        if user_input:
            parts.append(f"[USER]\n{user_input}")
        return "\n\n".join(parts)

    # ---- Chat History ----

    def format_chat_history(self, messages: list) -> str:
        """Format conversation history into readable text with optional compression."""
        if self.compress_old:
            messages = compress_history_tags(messages, keep_recent=self.keep_recent)

        lines = []
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if not content:
                continue

            if role == "human":
                lines.append(f"User: {content}")
            elif role == "ai":
                lines.append(f"Assistant: {content}")
            elif role == "system":
                continue  # Don't include system messages in history
            else:
                lines.append(f"{role}: {content}")

        history = "\n".join(lines)

        # Hard limit on total history length
        if len(history) > self.max_history_chars:
            # Keep the tail (most recent)
            history = "...\n" + history[-self.max_history_chars :]

        return history

    # ---- Vector Memory ----

    def format_vector_memory(self, memory_results: list) -> str:
        """Format vector memory search results into readable text."""
        if not memory_results:
            return ""

        lines = []
        for i, item in enumerate(memory_results[: self.max_vector_results], 1):
            if isinstance(item, dict):
                text = item.get("text", str(item))
                score = item.get("score", 0)
                lines.append(f"{i}. [score={score:.2f}] {text}")
            elif isinstance(item, str):
                lines.append(f"{i}. {item}")
            else:
                lines.append(f"{i}. {item!s}")

        return "\n".join(lines)

    # ---- Context Building for Different Agents ----

    def build_decision_context(
        self,
        messages: list,
        current_input: str = "",
        vector_memory_text: str = "",
    ) -> ContextSegments:
        """Build context for the decision/routing agent."""
        return ContextSegments(
            system_prompt=MedicalSystemPrompt.DECISION,
            vector_memory=vector_memory_text,
            chat_history=self.format_chat_history(messages),
        )

    def build_conversation_context(
        self,
        messages: list,
        current_input: str = "",
        vector_memory_text: str = "",
    ) -> ContextSegments:
        """Build context for the conversation agent."""
        return ContextSegments(
            system_prompt=MedicalSystemPrompt.CONVERSATION,
            vector_memory=vector_memory_text,
            chat_history=self.format_chat_history(messages),
        )

    def build_rag_context(
        self,
        messages: list,
        current_input: str = "",
        rag_results: str = "",
        vector_memory_text: str = "",
    ) -> ContextSegments:
        """Build context for the RAG agent."""
        return ContextSegments(
            system_prompt=MedicalSystemPrompt.RAG,
            vector_memory=vector_memory_text,
            medical_kb=rag_results,
            chat_history=self.format_chat_history(messages),
        )

    # ---- Phase 51: Summarize + Truncate ----

    def summarize_and_truncate(
        self,
        messages: list,
        keep_recent: int = 6,
        max_summary_chars: int = 500,
    ) -> tuple:
        """
        For Phase 51: split messages into (old_to_summarize, recent_to_keep).
        Returns (old_messages, recent_messages) tuple.
        The caller is responsible for generating the summary of old_messages.
        """
        if len(messages) <= keep_recent:
            return [], messages

        old = messages[:-keep_recent]
        recent = messages[-keep_recent:]
        return old, recent

    def format_conversation_summary(self, summary_text: str) -> str:
        """Format a conversation summary for inclusion in context."""
        if not summary_text:
            return ""
        if len(summary_text) > 1000:
            summary_text = summary_text[:950] + " [...]"
        return f"Previous conversation summary:\n{summary_text}"
