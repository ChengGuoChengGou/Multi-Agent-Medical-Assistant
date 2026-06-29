from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    name: str
    kind: str
    description: str
    agent_name: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    is_external: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "agent_name": self.agent_name,
            "input_schema": self.input_schema,
            "is_external": self.is_external,
        }


@dataclass
class ToolResult:
    tool_id: str
    ok: bool
    content: Any = None
    error: Optional[str] = None
    latency_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class ToolRegistry:
    """MCP-like in-process tool registry.

    The registry intentionally mirrors Ragent's tool-id lookup model without
    requiring a running MCP server. Executors are optional, so tools can be used
    for explainable routing first and wired to concrete calls incrementally.
    """

    def __init__(self):
        self._specs: Dict[str, ToolSpec] = {}
        self._executors: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register(
        self,
        spec: ToolSpec,
        executor: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        if not spec.tool_id:
            raise ValueError("tool_id is required")
        self._specs[spec.tool_id] = spec
        if executor is not None:
            self._executors[spec.tool_id] = executor

    def contains(self, tool_id: Optional[str]) -> bool:
        return bool(tool_id and tool_id in self._specs)

    def get_spec(self, tool_id: Optional[str]) -> Optional[ToolSpec]:
        if not tool_id:
            return None
        return self._specs.get(tool_id)

    def list_specs(self) -> List[ToolSpec]:
        return list(self._specs.values())

    def execute(self, tool_id: str, parameters: Optional[Dict[str, Any]] = None) -> ToolResult:
        spec = self.get_spec(tool_id)
        if spec is None:
            return ToolResult(tool_id=tool_id, ok=False, error="tool_not_registered")
        executor = self._executors.get(tool_id)
        if executor is None:
            return ToolResult(tool_id=tool_id, ok=False, error="tool_executor_not_configured")

        started_at = time.time()
        try:
            content = executor(parameters or {})
            return ToolResult(
                tool_id=tool_id,
                ok=True,
                content=content,
                latency_ms=int((time.time() - started_at) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                tool_id=tool_id,
                ok=False,
                error=str(exc),
                latency_ms=int((time.time() - started_at) * 1000),
            )


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for spec in _default_tool_specs():
        registry.register(spec)
    return registry


def _default_tool_specs() -> List[ToolSpec]:
    return [
        ToolSpec(
            tool_id="knowledge_base",
            name="Local Medical Knowledge Base",
            kind="KB",
            description="Retrieve grounded chunks from local Qdrant, FAQ markdown, and parsed medical documents.",
            agent_name="RAG_AGENT",
            input_schema={"query": "string", "top_k": "integer"},
        ),
        ToolSpec(
            tool_id="web_search",
            name="Medical Web Search",
            kind="EXTERNAL_SEARCH",
            description="Search current web information for recent medical updates and time-sensitive questions.",
            agent_name="WEB_SEARCH_PROCESSOR_AGENT",
            input_schema={"query": "string"},
            is_external=True,
        ),
        ToolSpec(
            tool_id="chest_xray_classifier",
            name="Chest X-ray Classifier",
            kind="VISION_MODEL",
            description="Classify uploaded chest X-ray images for supported COVID-19/normal demo labels.",
            agent_name="CHEST_XRAY_AGENT",
            input_schema={"image_path": "string"},
        ),
        ToolSpec(
            tool_id="skin_lesion_segmenter",
            name="Skin Lesion Segmenter",
            kind="VISION_MODEL",
            description="Run skin lesion segmentation on uploaded dermoscopy/skin lesion images.",
            agent_name="SKIN_LESION_AGENT",
            input_schema={"image_path": "string"},
        ),
        ToolSpec(
            tool_id="brain_mri_analyzer",
            name="Brain MRI Analyzer",
            kind="VISION_MODEL",
            description="Route brain MRI tumor analysis tasks to the brain tumor image agent.",
            agent_name="BRAIN_TUMOR_AGENT",
            input_schema={"image_path": "string"},
        ),
        ToolSpec(
            tool_id="medical_safety_policy",
            name="Medical Safety Policy",
            kind="SAFETY",
            description="Apply guardrails for emergencies, unsafe dosage, diagnosis overclaim, and prompt injection.",
            agent_name="INPUT_GUARDRAILS",
            input_schema={"query": "string"},
        ),
    ]
