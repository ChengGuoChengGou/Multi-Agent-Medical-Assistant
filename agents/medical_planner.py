"""
Medical Planner & Diagnosis Reflection Module (Phase 5)

Implements:
  5.1 MedicalPlanner: Analyze patient query → Generate diagnostic plan
  5.2 Exploration → Plan → Verification three-stage workflow
  5.3 Diagnosis reflection: Auto-evaluate accuracy after each diagnosis
  5.4 Knowledge refresh scheduler: Periodic RAG re-index trigger

Inspired by GA's plan_sop (Exploration→Plan→Verification) and CCS ch15.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════


class PlanStage(str, Enum):
    """Three-stage diagnostic plan lifecycle."""

    EXPLORATION = "exploration"  # Gather info: what do we know? what's missing?
    PLANNING = "planning"  # Generate actionable plan
    VERIFICATION = "verification"  # Validate plan & results


class StepType(str, Enum):
    """Types of diagnostic steps."""

    RAG_SEARCH = "rag_search"  # Query medical knowledge base
    WEB_SEARCH = "web_search"  # Search recent medical literature
    MCP_LOOKUP = "mcp_lookup"  # External biomedical DB lookup
    IMAGE_ANALYSIS = "image_analysis"  # Medical image analysis
    CONVERSATION = "conversation"  # General conversation / clarification
    PARALLEL = "parallel"  # Run multiple steps in parallel
    REFLECTION = "reflection"  # Self-evaluation step


class StepPriority(str, Enum):
    """Priority levels for plan steps."""

    CRITICAL = "critical"  # Must be executed for safe diagnosis
    HIGH = "high"  # Important for accuracy
    MEDIUM = "medium"  # Nice to have
    LOW = "low"  # Optional enrichment


@dataclass
class PlanStep:
    """A single step in a diagnostic plan."""

    step_id: str
    step_type: StepType
    description: str
    priority: StepPriority = StepPriority.MEDIUM
    depends_on: list[str] = field(default_factory=list)  # step_ids this depends on
    agent_hint: str | None = None  # Suggested agent name
    query_hint: str | None = None  # Refined query for this step
    expected_output: str = ""  # What we expect to learn
    status: str = "pending"  # pending / running / done / failed / skipped
    result_summary: str = ""
    duration_ms: float = 0.0


@dataclass
class DiagnosticPlan:
    """Complete diagnostic plan for a patient query."""

    plan_id: str
    patient_query: str
    stage: PlanStage = PlanStage.EXPLORATION
    steps: list[PlanStep] = field(default_factory=list)
    exploration_findings: dict[str, Any] = field(default_factory=dict)
    verification_notes: list[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "patient_query": self.patient_query[:200],
            "stage": self.stage.value,
            "steps": [asdict(s) for s in self.steps],
            "overall_confidence": self.overall_confidence,
            "exploration_findings": self.exploration_findings,
            "verification_notes": self.verification_notes,
        }

    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "pending"]

    def is_complete(self) -> bool:
        return all(s.status in ("done", "skipped", "failed") for s in self.steps)


@dataclass
class DiagnosisReflection:
    """Post-diagnosis reflection and quality assessment."""

    plan_id: str
    query: str
    diagnosis_response: str
    # Quality dimensions (0-1)
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    safety_score: float = 0.0
    relevance_score: float = 0.0
    # Issues found
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    # Should we re-route or refine?
    needs_refinement: bool = False
    refinement_query: str | None = None


# ═══════════════════════════════════════════════════════════════
#  Medical Planner
# ═══════════════════════════════════════════════════════════════

# Patterns indicating complex queries that benefit from planning
COMPLEX_QUERY_INDICATORS = [
    "differential diagnosis",
    "differential",
    "what could cause",
    "multiple symptoms",
    "chronic",
    "comorbid",
    "drug interaction",
    "side effect",
    "treatment plan",
    "management",
    "workup",
    "investigation",
    "lab test",
    "imaging",
    "refer",
    "specialist",
    # Symptom combinations (2+ body systems)
    "chest pain",
    "shortness of breath",  # + neurological / GI etc.
    "fever",
    "rash",
    "headache",
    "vision",
    "abdominal pain",
    "nausea",
]

# Simple queries that don't need planning
SIMPLE_QUERY_PATTERNS = [
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "thank",
    "thanks",
    "bye",
    "goodbye",
    "what is",
    "define",
    "explain",
]


def _needs_planning(query: str) -> bool:
    """Determine if a query would benefit from diagnostic planning."""
    q = query.lower().strip()
    # Very short queries → no planning
    if len(q.split()) < 5:
        return False
    # Simple conversational → no planning
    if any(q.startswith(p) for p in SIMPLE_QUERY_PATTERNS):
        return False
    # Complex medical queries → plan
    if any(indicator in q for indicator in COMPLEX_QUERY_INDICATORS):
        return True
    # Multi-sentence queries with symptoms → likely complex
    return bool(len(q.split(".")) >= 3 and any(w in q for w in ["pain", "symptom", "feel", "diagnos"]))


def _generate_plan_id() -> str:
    import uuid

    return f"plan_{uuid.uuid4().hex[:8]}"


def _analyze_query_complexity(query: str) -> dict[str, Any]:
    """Extract features from query to inform plan generation."""
    q = query.lower()

    features: dict[str, Any] = {
        "has_image_ref": any(w in q for w in ["image", "x-ray", "xray", "mri", "ct scan", "scan", "photo"]),
        "has_symptoms": any(
            w in q for w in ["pain", "ache", "fever", "cough", "fatigue", "nausea", "rash", "swelling"]
        ),
        "has_drug_ref": any(w in q for w in ["medication", "drug", "dose", "prescription", "pill", "tablet"]),
        "has_lab_ref": any(w in q for w in ["lab", "blood test", "urine", "biopsy", "cbc", "glucose"]),
        "has_chronic": any(w in q for w in ["chronic", "long-term", "ongoing", "recurring", "persistent"]),
        "has_urgency": any(w in q for w in ["emergency", "urgent", "severe", "sudden", "acute", "critical"]),
        "body_systems": [],
        "query_length": len(q.split()),
    }

    # Detect body systems mentioned
    system_keywords = {
        "cardiovascular": ["heart", "chest pain", "blood pressure", "palpitation", "cardiac"],
        "respiratory": ["lung", "breath", "cough", "wheez", "asthma", "pneumonia"],
        "neurological": ["headache", "dizzy", "seizure", "numbness", "vision", "brain"],
        "gastrointestinal": ["stomach", "abdominal", "nausea", "vomit", "diarrhea", "bowel"],
        "dermatological": ["skin", "rash", "lesion", "mole", "itching", "dermat"],
        "musculoskeletal": ["joint", "bone", "muscle", "back pain", "arthritis"],
        "endocrine": ["diabetes", "thyroid", "hormone", "glucose", "insulin"],
    }

    for system, keywords in system_keywords.items():
        if any(kw in q for kw in keywords):
            features["body_systems"].append(system)

    features["multi_system"] = len(features["body_systems"]) >= 2
    return features


# ═══════════════════════════════════════════════════════════════
#  Exploration Stage
# ═══════════════════════════════════════════════════════════════


def exploration_stage(query: str, has_image: bool = False, image_type: str | None = None) -> dict[str, Any]:
    """
    Stage 1: Exploration — Analyze what we know and what we need.

    Returns findings dict that informs the planning stage.
    """
    features = _analyze_query_complexity(query)

    findings: dict[str, Any] = {
        "query_features": features,
        "has_image": has_image,
        "image_type": image_type,
        "information_gaps": [],
        "suggested_agents": [],
        "complexity": "simple",
    }

    # Determine information gaps
    if features["has_symptoms"] and not features["has_lab_ref"]:
        findings["information_gaps"].append("lab_results_missing")
    if features["has_drug_ref"]:
        findings["information_gaps"].append("drug_details_needed")
    if features["multi_system"]:
        findings["information_gaps"].append("multi_system_cross_ref")
    if features["has_chronic"]:
        findings["information_gaps"].append("history_context_needed")

    # Suggest agents based on features
    if has_image:
        findings["suggested_agents"].append(
            {
                "agent": f"{image_type.upper()}_AGENT" if image_type else "IMAGE_ANALYSIS_AGENT",
                "reason": "Image analysis required",
                "priority": "critical",
            }
        )

    if features["has_drug_ref"]:
        findings["suggested_agents"].append(
            {"agent": "MCP_AGENT", "reason": "Drug information / interaction lookup", "priority": "high"}
        )

    if features["has_symptoms"] or features["has_chronic"]:
        findings["suggested_agents"].append(
            {"agent": "RAG_AGENT", "reason": "Medical knowledge retrieval for symptom analysis", "priority": "high"}
        )
        findings["suggested_agents"].append(
            {
                "agent": "WEB_SEARCH_PROCESSOR_AGENT",
                "reason": "Recent literature for current guidelines",
                "priority": "medium",
            }
        )

    if not findings["suggested_agents"]:
        findings["suggested_agents"].append(
            {"agent": "CONVERSATION_AGENT", "reason": "General conversation", "priority": "medium"}
        )

    # Complexity assessment
    gap_count = len(findings["information_gaps"])
    agent_count = len(findings["suggested_agents"])
    if gap_count >= 3 or agent_count >= 3 or features["multi_system"]:
        findings["complexity"] = "complex"
    elif gap_count >= 1 or agent_count >= 2:
        findings["complexity"] = "moderate"

    return findings


# ═══════════════════════════════════════════════════════════════
#  Planning Stage
# ═══════════════════════════════════════════════════════════════


def planning_stage(
    query: str,
    findings: dict[str, Any],
    has_image: bool = False,
    image_type: str | None = None,
) -> DiagnosticPlan:
    """
    Stage 2: Generate an actionable diagnostic plan from exploration findings.
    """
    plan_id = _generate_plan_id()
    plan = DiagnosticPlan(
        plan_id=plan_id,
        patient_query=query,
        stage=PlanStage.PLANNING,
        exploration_findings=findings,
    )

    step_counter = 0
    suggested = findings.get("suggested_agents", [])

    # If image present → image analysis is first critical step
    if has_image and image_type:
        step_counter += 1
        plan.steps.append(
            PlanStep(
                step_id=f"s{step_counter}",
                step_type=StepType.IMAGE_ANALYSIS,
                description=f"Analyze {image_type} image for diagnostic findings",
                priority=StepPriority.CRITICAL,
                agent_hint=f"{image_type.upper()}_AGENT",
                query_hint=query,
                expected_output="Image analysis results with confidence scores",
            )
        )

    # Build steps from suggested agents
    prev_step_ids: list[str] = []
    for agent_info in suggested:
        agent = agent_info["agent"]
        reason = agent_info["reason"]
        priority_str = agent_info.get("priority", "medium")
        priority = (
            StepPriority(priority_str) if priority_str in [p.value for p in StepPriority] else StepPriority.MEDIUM
        )

        # Skip if already handled by image step
        if has_image and "IMAGE" in agent:
            continue

        step_counter += 1
        step_type = _agent_to_step_type(agent)

        plan.steps.append(
            PlanStep(
                step_id=f"s{step_counter}",
                step_type=step_type,
                description=reason,
                priority=priority,
                depends_on=prev_step_ids[:],  # Depend on all previous steps
                agent_hint=agent,
                query_hint=_refine_query_for_agent(query, agent, findings),
                expected_output=f"Results from {agent}",
            )
        )
        prev_step_ids.append(f"s{step_counter}")

    # Add reflection step for complex queries
    if findings.get("complexity") == "complex":
        step_counter += 1
        plan.steps.append(
            PlanStep(
                step_id=f"s{step_counter}",
                step_type=StepType.REFLECTION,
                description="Post-diagnosis quality reflection and cross-validation",
                priority=StepPriority.HIGH,
                depends_on=prev_step_ids[:],
                expected_output="Quality assessment and refinement suggestions",
            )
        )

    plan.stage = PlanStage.PLANNING
    return plan


def _agent_to_step_type(agent_name: str) -> StepType:
    """Map agent name to PlanStep type."""
    mapping = {
        "RAG_AGENT": StepType.RAG_SEARCH,
        "WEB_SEARCH_PROCESSOR_AGENT": StepType.WEB_SEARCH,
        "MCP_AGENT": StepType.MCP_LOOKUP,
        "CONVERSATION_AGENT": StepType.CONVERSATION,
        "PARALLEL_RETRIEVAL": StepType.PARALLEL,
    }
    for key, val in mapping.items():
        if key in agent_name:
            return val
    return StepType.CONVERSATION


def _refine_query_for_agent(query: str, agent: str, findings: dict) -> str | None:
    """Refine the query with context for a specific agent."""
    features = findings.get("query_features", {})

    if agent == "RAG_AGENT" and features.get("has_symptoms"):
        body_systems = features.get("body_systems", [])
        if body_systems:
            return f"{query}\n\nFocus areas: {', '.join(body_systems)}"

    if agent == "MCP_AGENT" and features.get("has_drug_ref"):
        return f"{query}\n\nPlease check drug interactions, contraindications, and dosing."

    if agent == "WEB_SEARCH_PROCESSOR_AGENT":
        return f"{query}\n\nSeek latest clinical guidelines and evidence."

    return None  # Use original query


# ═══════════════════════════════════════════════════════════════
#  Verification Stage
# ═══════════════════════════════════════════════════════════════


def verification_stage(
    plan: DiagnosticPlan,
    response: str,
    vector_memory_available: bool = False,
) -> DiagnosisReflection:
    """
    Stage 3: Verify the diagnosis quality and completeness.

    Uses heuristic checks (fast, no LLM call). For complex cases,
    the reflection step in the plan uses LLM for deeper analysis.
    """
    reflection = DiagnosisReflection(
        plan_id=plan.plan_id,
        query=plan.patient_query,
        diagnosis_response=response,
    )

    response_lower = response.lower()

    # ── Accuracy heuristics ──
    # Check for hedging language (good for medical responses)
    hedging_terms = ["may", "might", "could", "possible", "likely", "suggest", "consider", "recommend"]
    has_hedging = any(term in response_lower for term in hedging_terms)
    reflection.accuracy_score = 0.8 if has_hedging else 0.5

    # Check for definitive diagnosis (bad without disclaimer)
    definitive_terms = ["you have", "you are diagnosed", "you definitely", "this is definitely"]
    has_definitive = any(term in response_lower for term in definitive_terms)
    if has_definitive and "consult" not in response_lower:
        reflection.accuracy_score -= 0.2
        reflection.issues.append("Contains definitive diagnosis without professional consultation disclaimer")

    # ── Completeness checks ──
    completeness_indicators = []

    # Check if response addresses the query features
    features = plan.exploration_findings.get("query_features", {})
    if features.get("has_drug_ref"):
        if any(w in response_lower for w in ["drug", "medication", "dose", "interaction"]):
            completeness_indicators.append("drug_info_addressed")
        else:
            reflection.issues.append("Drug reference in query not addressed in response")

    if features.get("has_symptoms"):
        if any(w in response_lower for w in ["symptom", "cause", "condition", "diagnosis"]):
            completeness_indicators.append("symptoms_addressed")

    if features.get("has_lab_ref"):
        if any(w in response_lower for w in ["test", "lab", "result", "level"]):
            completeness_indicators.append("lab_info_addressed")

    expected = max(len(plan.steps), 1)
    reflection.completeness_score = min(1.0, len(completeness_indicators) / expected + 0.4)

    # ── Safety checks ──
    safety_disclaimers = [
        "consult",
        "healthcare",
        "doctor",
        "professional",
        "emergency",
        "seek medical",
        "not a substitute",
        "educational",
    ]
    safety_count = sum(1 for s in safety_disclaimers if s in response_lower)
    reflection.safety_score = min(1.0, safety_count * 0.2 + 0.3)

    # Check for dangerous advice
    dangerous_patterns = [
        "stop taking",
        "discontinue",
        "ignore your doctor",
        "don't need",
        "no need to see",
        "home remedy is enough",
    ]
    if any(p in response_lower for p in dangerous_patterns):
        reflection.safety_score -= 0.3
        reflection.issues.append("Potentially dangerous advice detected")

    # ── Relevance ──
    query_words = set(plan.patient_query.lower().split())
    response_words = set(response_lower.split())
    overlap = len(query_words & response_words) / max(len(query_words), 1)
    reflection.relevance_score = min(1.0, overlap * 1.5)

    # ── Overall assessment ──
    overall = (
        reflection.accuracy_score * 0.3
        + reflection.completeness_score * 0.3
        + reflection.safety_score * 0.25
        + reflection.relevance_score * 0.15
    )
    plan.overall_confidence = round(overall, 3)

    # ── Refinement suggestions ──
    if reflection.safety_score < 0.5:
        reflection.suggestions.append("Add stronger safety disclaimers")
    if reflection.completeness_score < 0.6:
        reflection.suggestions.append("Address missing information gaps")
        reflection.needs_refinement = True
    if has_definitive:
        reflection.suggestions.append("Replace definitive language with qualified suggestions")

    if features.get("multi_system") and "cross-validation" not in response_lower:
        reflection.suggestions.append("Consider cross-system differential diagnosis")

    # Store verification notes
    plan.verification_notes = reflection.suggestions
    plan.completed_at = time.time()
    plan.stage = PlanStage.VERIFICATION

    return reflection


# ═══════════════════════════════════════════════════════════════
#  Public API: Full Planning Pipeline
# ═══════════════════════════════════════════════════════════════


def create_diagnostic_plan(
    query: str,
    has_image: bool = False,
    image_type: str | None = None,
    skip_if_simple: bool = True,
) -> DiagnosticPlan | None:
    """
    Main entry point: Create a diagnostic plan for a patient query.

    Returns None if the query is simple and doesn't need planning
    (when skip_if_simple=True).
    """
    if skip_if_simple and not _needs_planning(query):
        logger.debug(f"[PLANNER] Query is simple, skipping planning: {query[:80]}")
        return None

    logger.info(f"[PLANNER] Creating diagnostic plan for: {query[:100]}")

    # Stage 1: Exploration
    findings = exploration_stage(query, has_image, image_type)
    logger.debug(
        f"[PLANNER] Exploration complete. Complexity: {findings['complexity']}, "
        f"Gaps: {len(findings['information_gaps'])}, Agents: {len(findings['suggested_agents'])}"
    )

    # Stage 2: Planning
    plan = planning_stage(query, findings, has_image, image_type)
    logger.info(f"[PLANNER] Plan {plan.plan_id} created with {len(plan.steps)} steps")

    return plan


def reflect_on_diagnosis(
    plan: DiagnosticPlan,
    response: str,
) -> DiagnosisReflection:
    """
    Post-diagnosis reflection. Call after getting agent response.
    """
    return verification_stage(plan, response)


def get_plan_routing_hints(plan: DiagnosticPlan) -> dict[str, Any]:
    """
    Extract routing hints from a plan to influence the agent graph.

    Returns dict that can be injected into AgentState.
    """
    hints = {
        "has_plan": True,
        "plan_id": plan.plan_id,
        "complexity": plan.exploration_findings.get("complexity", "simple"),
        "primary_agent": None,
        "secondary_agents": [],
        "needs_parallel": False,
        "information_gaps": plan.exploration_findings.get("information_gaps", []),
    }

    # Determine primary agent (highest priority step)
    priority_order = [StepPriority.CRITICAL, StepPriority.HIGH, StepPriority.MEDIUM, StepPriority.LOW]
    for priority in priority_order:
        for step in plan.steps:
            if step.priority == priority and step.agent_hint:
                hints["primary_agent"] = step.agent_hint
                break
        if hints["primary_agent"]:
            break

    # Check if parallel execution is beneficial
    rag_steps = [s for s in plan.steps if s.step_type == StepType.RAG_SEARCH]
    web_steps = [s for s in plan.steps if s.step_type == StepType.WEB_SEARCH]
    if rag_steps and web_steps:
        hints["needs_parallel"] = True
        hints["secondary_agents"].append("PARALLEL_RETRIEVAL")

    # Collect all agents
    for step in plan.steps:
        if step.agent_hint and step.agent_hint != hints["primary_agent"]:
            if step.agent_hint not in hints["secondary_agents"]:
                hints["secondary_agents"].append(step.agent_hint)

    return hints


# ═══════════════════════════════════════════════════════════════
#  Knowledge Refresh Scheduler (Phase 5.4)
# ═══════════════════════════════════════════════════════════════


@dataclass
class RefreshConfig:
    """Configuration for knowledge base refresh schedule."""

    rag_reindex_interval_hours: int = 24
    vector_seed_check_interval_hours: int = 12
    max_stale_hours: int = 72  # Alert if no update in 3 days
    last_rag_refresh: float = 0.0
    last_seed_check: float = 0.0


_refresh_config = RefreshConfig()


def check_refresh_needed() -> dict[str, bool]:
    """
    Check if knowledge base refresh is needed. Pure function, no side effects.
    Call from scheduler or periodically.

    Returns dict indicating what needs refreshing.
    """
    now = time.time()
    checks = {
        "rag_reindex_needed": (now - _refresh_config.last_rag_refresh)
        > (_refresh_config.rag_reindex_interval_hours * 3600),
        "seed_check_needed": (now - _refresh_config.last_seed_check)
        > (_refresh_config.vector_seed_check_interval_hours * 3600),
        "is_stale": (now - _refresh_config.last_rag_refresh) > (_refresh_config.max_stale_hours * 3600),
    }
    return checks


def mark_refreshed(refresh_type: str) -> None:
    """Mark a refresh as completed."""
    now = time.time()
    if refresh_type == "rag":
        _refresh_config.last_rag_refresh = now
    elif refresh_type == "seed":
        _refresh_config.last_seed_check = now
    logger.info(f"[SCHEDULER] Marked {refresh_type} refresh at {time.ctime(now)}")


# ═══════════════════════════════════════════════════════════════
#  AgentDecision Extension for Planning
# ═══════════════════════════════════════════════════════════════

PLANNING_AVAILABLE = True
logger.info("[PLANNER] Medical Planner module loaded successfully")
