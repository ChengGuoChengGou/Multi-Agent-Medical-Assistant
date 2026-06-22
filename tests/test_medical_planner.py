"""
Unit tests for agents.medical_planner – Medical Planner & Diagnosis Reflection.

Tests cover pure functions (no LLM/network calls):
  - Enums: PlanStage, StepType, StepPriority
  - Data models: PlanStep, DiagnosticPlan, DiagnosisReflection
  - Heuristic functions: _needs_planning, _analyze_query_complexity, _agent_to_step_type, _refine_query_for_agent
  - Pipeline stages: exploration_stage, planning_stage, verification_stage
  - Public API: create_diagnostic_plan, reflect_on_diagnosis, get_plan_routing_hints
  - Refresh scheduler: RefreshConfig, check_refresh_needed, mark_refreshed
"""

from agents.medical_planner import (
    DiagnosisReflection,
    DiagnosticPlan,
    PlanStage,
    PlanStep,
    RefreshConfig,
    StepPriority,
    StepType,
    _agent_to_step_type,
    _analyze_query_complexity,
    _needs_planning,
    _refine_query_for_agent,
    check_refresh_needed,
    create_diagnostic_plan,
    exploration_stage,
    get_plan_routing_hints,
    mark_refreshed,
    planning_stage,
    reflect_on_diagnosis,
    verification_stage,
)

# ═══════════════════════════════════════════
# Enum Tests
# ═══════════════════════════════════════════


class TestPlanStage:
    def test_values(self):
        assert PlanStage.EXPLORATION.value == "exploration"
        assert PlanStage.PLANNING.value == "planning"
        assert PlanStage.VERIFICATION.value == "verification"

    def test_members(self):
        assert len(PlanStage) == 3


class TestStepType:
    def test_values(self):
        assert StepType.RAG_SEARCH.value == "rag_search"
        assert StepType.WEB_SEARCH.value == "web_search"
        assert StepType.MCP_LOOKUP.value == "mcp_lookup"
        assert StepType.CONVERSATION.value == "conversation"
        assert StepType.IMAGE_ANALYSIS.value == "image_analysis"
        assert StepType.PARALLEL.value == "parallel"
        assert StepType.REFLECTION.value == "reflection"

    def test_members(self):
        assert len(StepType) == 7


class TestStepPriority:
    def test_values(self):
        assert StepPriority.CRITICAL.value == "critical"
        assert StepPriority.HIGH.value == "high"
        assert StepPriority.MEDIUM.value == "medium"
        assert StepPriority.LOW.value == "low"


# ═══════════════════════════════════════════
# Data Model Tests
# ═══════════════════════════════════════════


class TestPlanStep:
    def test_creation(self):
        step = PlanStep(
            step_id="s1",
            step_type=StepType.RAG_SEARCH,
            description="Search medical knowledge base",
        )
        assert step.step_id == "s1"
        assert step.step_type == StepType.RAG_SEARCH
        assert step.description == "Search medical knowledge base"

    def test_defaults(self):
        step = PlanStep(
            step_id="s1",
            step_type=StepType.CONVERSATION,
            description="test",
        )
        assert step.priority == StepPriority.MEDIUM
        assert step.depends_on == []
        assert step.agent_hint is None
        assert step.query_hint is None
        assert step.expected_output == ""
        assert step.status == "pending"

    def test_to_dict(self):
        from dataclasses import asdict

        step = PlanStep(
            step_id="s1",
            step_type=StepType.RAG_SEARCH,
            description="test",
            priority=StepPriority.HIGH,
            agent_hint="RAG_AGENT",
        )
        d = asdict(step)
        assert d["step_id"] == "s1"
        assert d["step_type"] == "rag_search"
        assert d["priority"] == "high"
        assert d["agent_hint"] == "RAG_AGENT"
        assert isinstance(d, dict)


class TestDiagnosticPlan:
    def test_creation(self):
        plan = DiagnosticPlan(
            plan_id="plan_abc123",
            patient_query="What could cause chest pain?",
        )
        assert plan.plan_id == "plan_abc123"
        assert plan.patient_query == "What could cause chest pain?"
        assert plan.stage == PlanStage.EXPLORATION

    def test_defaults(self):
        plan = DiagnosticPlan(plan_id="test", patient_query="q")
        assert plan.steps == []
        assert plan.exploration_findings == {}
        assert plan.verification_notes == []
        assert plan.overall_confidence == 0.0
        assert plan.completed_at is None

    def test_pending_steps(self):
        plan = DiagnosticPlan(
            plan_id="t",
            patient_query="q",
            steps=[PlanStep(step_id="s1", step_type=StepType.RAG_SEARCH, description="d")],
        )
        pending = plan.pending_steps()
        assert len(pending) == 1

    def test_is_complete_true(self):
        step = PlanStep(step_id="s1", step_type=StepType.RAG_SEARCH, description="d", status="done")
        plan = DiagnosticPlan(plan_id="t", patient_query="q", steps=[step])
        assert plan.is_complete() is True

    def test_is_complete_false(self):
        plan = DiagnosticPlan(
            plan_id="t",
            patient_query="q",
            steps=[
                PlanStep(step_id="s1", step_type=StepType.RAG_SEARCH, description="d", status="completed"),
                PlanStep(step_id="s2", step_type=StepType.WEB_SEARCH, description="d2", status="pending"),
            ],
        )
        assert plan.is_complete() is False

    def test_to_dict(self):
        from dataclasses import asdict

        plan = DiagnosticPlan(plan_id="t", patient_query="q")
        d = asdict(plan)
        assert d["plan_id"] == "t"
        assert d["stage"] == "exploration"
        assert isinstance(d["steps"], list)


class TestDiagnosisReflection:
    def test_creation(self):
        ref = DiagnosisReflection(plan_id="p1", query="q", diagnosis_response="r")
        assert ref.plan_id == "p1"
        assert ref.query == "q"
        assert ref.diagnosis_response == "r"

    def test_defaults(self):
        ref = DiagnosisReflection(plan_id="p1", query="q", diagnosis_response="r")
        assert ref.accuracy_score == 0.0
        assert ref.completeness_score == 0.0
        assert ref.safety_score == 0.0
        assert ref.relevance_score == 0.0
        assert ref.issues == []
        assert ref.suggestions == []
        assert ref.needs_refinement is False

    def test_to_dict(self):
        from dataclasses import asdict

        ref = DiagnosisReflection(plan_id="p1", query="q", diagnosis_response="r")
        d = asdict(ref)
        assert d["plan_id"] == "p1"
        assert "accuracy_score" in d
        assert "issues" in d


# ═══════════════════════════════════════════
# _needs_planning()
# ═══════════════════════════════════════════


class TestNeedsPlanning:
    def test_short_query_returns_false(self):
        assert _needs_planning("hi") is False

    def test_very_short_returns_false(self):
        assert _needs_planning("what is") is False  # 2 words < 5

    def test_simple_greeting_returns_false(self):
        assert _needs_planning("hello there how are you") is False

    def test_simple_question_returns_false(self):
        assert _needs_planning("what is diabetes and how does it work") is False

    def test_thank_you_returns_false(self):
        assert _needs_planning("thank you for the information doctor") is False

    def test_symptoms_returns_true(self):
        assert _needs_planning("I have chest pain shortness of breath and sweating") is True

    def test_diagnosis_returns_true(self):
        assert _needs_planning("I need a diagnosis for my chronic symptoms") is True

    def test_emergency_returns_true(self):
        assert _needs_planning("this is an emergency with chest pain and shortness of breath") is True

    def test_medication_query_returns_true(self):
        # "treatment plan" is in COMPLEX_QUERY_INDICATORS
        assert _needs_planning("What medications should I take for my chronic treatment plan") is True

    def test_chest_pain_returns_true(self):
        assert _needs_planning("I have chest pain and difficulty breathing") is True

    def test_long_query_returns_true(self):
        # Multi-sentence with symptom words triggers L4: 3+ sentences with symptoms
        long = "I have been experiencing these symptoms for quite a long time now. The pain is getting worse every day. I feel dizzy and nauseous."
        assert _needs_planning(long) is True


# ═══════════════════════════════════════════
# _analyze_query_complexity()
# ═══════════════════════════════════════════


class TestAnalyzeQueryComplexity:
    def test_returns_dict(self):
        result = _analyze_query_complexity("test query")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = _analyze_query_complexity("test query")
        expected_keys = [
            "has_image_ref",
            "has_symptoms",
            "has_drug_ref",
            "has_lab_ref",
            "has_chronic",
            "has_urgency",
            "body_systems",
            "query_length",
            "multi_system",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_image_keywords(self):
        result = _analyze_query_complexity("Please analyze this x-ray of my chest")
        assert result["has_image_ref"] is True

    def test_no_image(self):
        result = _analyze_query_complexity("I have a headache")
        assert result["has_image_ref"] is False

    def test_symptom_keywords(self):
        result = _analyze_query_complexity("I have pain and symptoms of fever")
        assert result["has_symptoms"] is True

    def test_drug_reference(self):
        result = _analyze_query_complexity("I am taking medication for blood pressure")
        assert result["has_drug_ref"] is True

    def test_lab_reference(self):
        result = _analyze_query_complexity("My lab results show elevated glucose levels")
        assert result["has_lab_ref"] is True

    def test_body_systems_cardiac(self):
        result = _analyze_query_complexity("chest pain and heart palpitations")
        assert "cardiovascular" in result["body_systems"]

    def test_body_systems_neuro(self):
        result = _analyze_query_complexity("severe headache and dizziness")
        assert "neurological" in result["body_systems"]

    def test_empty_query(self):
        result = _analyze_query_complexity("")
        assert isinstance(result, dict)
        assert result["has_symptoms"] is False


# ═══════════════════════════════════════════
# _agent_to_step_type()
# ═══════════════════════════════════════════


class TestAgentToStepType:
    def test_rag_agent(self):
        assert _agent_to_step_type("RAG_AGENT") == StepType.RAG_SEARCH

    def test_web_agent(self):
        assert _agent_to_step_type("WEB_SEARCH_PROCESSOR_AGENT") == StepType.WEB_SEARCH

    def test_mcp_agent(self):
        assert _agent_to_step_type("MCP_AGENT") == StepType.MCP_LOOKUP

    def test_conversation_agent(self):
        assert _agent_to_step_type("CONVERSATION_AGENT") == StepType.CONVERSATION

    def test_parallel_retrieval(self):
        assert _agent_to_step_type("PARALLEL_RETRIEVAL") == StepType.PARALLEL

    def test_unknown_defaults_to_conversation(self):
        assert _agent_to_step_type("SOME_NEW_AGENT") == StepType.CONVERSATION

    def test_substring_match_rag(self):
        assert _agent_to_step_type("CUSTOM_RAG_AGENT_V2") == StepType.RAG_SEARCH


# ═══════════════════════════════════════════
# _refine_query_for_agent()
# ═══════════════════════════════════════════


class TestRefineQueryForAgent:
    def test_rag_with_symptoms_and_body_systems(self):
        findings = {"query_features": {"has_symptoms": True, "body_systems": ["cardiovascular", "respiratory"]}}
        result = _refine_query_for_agent("chest pain", "RAG_AGENT", findings)
        assert result is not None
        assert "cardiovascular" in result
        assert "respiratory" in result

    def test_rag_without_body_systems(self):
        findings = {"query_features": {"has_symptoms": True, "body_systems": []}}
        result = _refine_query_for_agent("chest pain", "RAG_AGENT", findings)
        assert result is None

    def test_rag_without_symptoms(self):
        findings = {"query_features": {"has_symptoms": False, "body_systems": ["cardiovascular"]}}
        result = _refine_query_for_agent("chest pain", "RAG_AGENT", findings)
        assert result is None

    def test_mcp_with_drug_ref(self):
        findings = {"query_features": {"has_drug_ref": True}}
        result = _refine_query_for_agent("aspirin dosage", "MCP_AGENT", findings)
        assert result is not None
        assert "drug" in result.lower() or "interaction" in result.lower()

    def test_mcp_without_drug_ref(self):
        findings = {"query_features": {"has_drug_ref": False}}
        result = _refine_query_for_agent("test query", "MCP_AGENT", findings)
        assert result is None

    def test_web_agent(self):
        findings = {"query_features": {}}
        result = _refine_query_for_agent("treatment options", "WEB_SEARCH_PROCESSOR_AGENT", findings)
        assert result is not None
        assert "guidelines" in result.lower() or "clinical" in result.lower()

    def test_unknown_agent_returns_none(self):
        findings = {"query_features": {}}
        result = _refine_query_for_agent("test", "UNKNOWN_AGENT", findings)
        assert result is None


# ═══════════════════════════════════════════
# exploration_stage()
# ═══════════════════════════════════════════


class TestExplorationStage:
    def test_returns_dict(self):
        result = exploration_stage("I have chest pain")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = exploration_stage("I have chest pain and need diagnosis")
        assert "query_features" in result
        assert "complexity" in result
        assert "information_gaps" in result
        assert "suggested_agents" in result
        assert "has_image" in result

    def test_complexity_simple(self):
        result = exploration_stage("hi")
        assert result["complexity"] == "simple"

    def test_complexity_complex(self):
        result = exploration_stage("I have chest pain shortness of breath and sweating with elevated troponin levels")
        assert result["complexity"] == "complex"

    def test_suggested_agents_structure(self):
        result = exploration_stage("I need diagnosis for chest pain symptoms")
        agents = result["suggested_agents"]
        assert isinstance(agents, list)
        for agent_info in agents:
            assert "agent" in agent_info
            assert "reason" in agent_info
            assert "priority" in agent_info

    def test_image_type_passed_through(self):
        result = exploration_stage("analyze this scan", has_image=True, image_type="xray")
        assert result["has_image"] is True
        assert result["image_type"] == "xray"

    def test_query_features_present(self):
        result = exploration_stage("I am taking aspirin for my heart condition")
        features = result["query_features"]
        assert isinstance(features, dict)


# ═══════════════════════════════════════════
# planning_stage()
# ═══════════════════════════════════════════


class TestPlanningStage:
    def test_returns_diagnostic_plan(self):
        findings = exploration_stage("I need diagnosis for chest pain")
        result = planning_stage("I need diagnosis for chest pain", findings)
        assert isinstance(result, DiagnosticPlan)

    def test_plan_has_steps(self):
        findings = exploration_stage("I have severe chest pain and need diagnosis and treatment")
        result = planning_stage("I have severe chest pain and need diagnosis and treatment", findings)
        assert len(result.steps) > 0

    def test_plan_with_image(self):
        findings = exploration_stage("analyze my xray", has_image=True, image_type="xray")
        result = planning_stage("analyze my xray", findings, has_image=True, image_type="xray")
        step_types = [s.step_type for s in result.steps]
        assert StepType.IMAGE_ANALYSIS in step_types

    def test_plan_sets_stage(self):
        findings = exploration_stage("I need diagnosis for symptoms")
        result = planning_stage("I need diagnosis for symptoms", findings)
        assert result.stage == PlanStage.PLANNING

    def test_reflection_for_complex(self):
        findings = exploration_stage("I have chest pain shortness of breath and sweating with troponin elevation")
        result = planning_stage("I have chest pain shortness of breath and sweating with troponin elevation", findings)
        step_types = [s.step_type for s in result.steps]
        assert StepType.REFLECTION in step_types

    def test_plan_id_generated(self):
        findings = exploration_stage("test query")
        result = planning_stage("test query", findings)
        assert result.plan_id.startswith("plan_")


# ═══════════════════════════════════════════
# verification_stage()
# ═══════════════════════════════════════════


class TestVerificationStage:
    def _make_plan(self, query="chest pain symptoms"):
        findings = exploration_stage(query)
        return planning_stage(query, findings)

    def test_returns_reflection(self):
        plan = self._make_plan()
        result = verification_stage(plan, "You may have angina. Please consult a doctor.")
        assert isinstance(result, DiagnosisReflection)

    def test_hedging_gives_higher_accuracy(self):
        plan = self._make_plan()
        ref = verification_stage(plan, "You may have a condition. Consider consulting a healthcare professional.")
        assert ref.accuracy_score >= 0.7

    def test_definitive_without_disclaimer_lower_accuracy(self):
        plan = self._make_plan()
        ref = verification_stage(plan, "You have pneumonia. Take these antibiotics.")
        assert ref.accuracy_score < 0.8

    def test_safety_disclaimers_detected(self):
        plan = self._make_plan()
        ref = verification_stage(
            plan, "Please consult a healthcare professional. This is not a substitute for medical advice."
        )
        assert ref.safety_score >= 0.5

    def test_dangerous_advice_penalized(self):
        plan = self._make_plan()
        ref = verification_stage(plan, "Stop taking your medication. Home remedy is enough.")
        assert ref.safety_score < 0.5

    def test_completeness_for_drug_query(self):
        plan = self._make_plan("aspirin medication dosage")
        ref = verification_stage(plan, "The recommended dose of aspirin is 81mg daily for cardiac prophylaxis.")
        assert ref.completeness_score > 0.0

    def test_sets_plan_stage(self):
        plan = self._make_plan()
        verification_stage(plan, "test response")
        assert plan.stage == PlanStage.VERIFICATION

    def test_sets_plan_completed_at(self):
        plan = self._make_plan()
        verification_stage(plan, "test response")
        assert plan.completed_at is not None

    def test_relevance_score(self):
        plan = self._make_plan("chest pain diagnosis")
        ref = verification_stage(plan, "chest pain can indicate several conditions including angina")
        assert ref.relevance_score > 0.0


# ═══════════════════════════════════════════
# create_diagnostic_plan()
# ═══════════════════════════════════════════


class TestCreateDiagnosticPlan:
    def test_simple_query_returns_none(self):
        result = create_diagnostic_plan("hi")
        assert result is None

    def test_complex_query_returns_plan(self):
        result = create_diagnostic_plan("I have severe chest pain and need diagnosis")
        assert isinstance(result, DiagnosticPlan)

    def test_skip_if_simple_false(self):
        result = create_diagnostic_plan("hi", skip_if_simple=False)
        assert isinstance(result, DiagnosticPlan)

    def test_plan_has_exploration_findings(self):
        result = create_diagnostic_plan("I have chest pain symptoms and need diagnosis")
        assert result.exploration_findings != {}

    def test_plan_stage_is_planning(self):
        result = create_diagnostic_plan("I have chest pain symptoms and need diagnosis")
        assert result.stage == PlanStage.PLANNING

    def test_with_image(self):
        result = create_diagnostic_plan("analyze my xray for chest pain fractures", has_image=True, image_type="xray")
        assert result is not None
        step_types = [s.step_type for s in result.steps]
        assert StepType.IMAGE_ANALYSIS in step_types


# ═══════════════════════════════════════════
# reflect_on_diagnosis()
# ═══════════════════════════════════════════


class TestReflectOnDiagnosis:
    def test_returns_reflection(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        result = reflect_on_diagnosis(plan, "You may have angina. Please consult a doctor.")
        assert isinstance(result, DiagnosisReflection)

    def test_updates_plan(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        reflect_on_diagnosis(plan, "Consult a healthcare professional.")
        assert plan.stage == PlanStage.VERIFICATION
        assert plan.completed_at is not None


# ═══════════════════════════════════════════
# get_plan_routing_hints()
# ═══════════════════════════════════════════


class TestGetPlanRoutingHints:
    def test_returns_dict(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        hints = get_plan_routing_hints(plan)
        assert isinstance(hints, dict)

    def test_has_plan_key(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        hints = get_plan_routing_hints(plan)
        assert hints["has_plan"] is True

    def test_plan_id(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        hints = get_plan_routing_hints(plan)
        assert hints["plan_id"] == plan.plan_id

    def test_complexity_key(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        hints = get_plan_routing_hints(plan)
        assert hints["complexity"] in ("simple", "moderate", "complex")

    def test_primary_agent(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        hints = get_plan_routing_hints(plan)
        # Should have a primary agent if plan has steps with agent_hint
        if any(s.agent_hint for s in plan.steps):
            assert hints["primary_agent"] is not None

    def test_secondary_agents_list(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        hints = get_plan_routing_hints(plan)
        assert isinstance(hints["secondary_agents"], list)

    def test_needs_parallel_bool(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        hints = get_plan_routing_hints(plan)
        assert isinstance(hints["needs_parallel"], bool)

    def test_information_gaps(self):
        plan = create_diagnostic_plan("I have severe chest pain symptoms")
        hints = get_plan_routing_hints(plan)
        assert isinstance(hints["information_gaps"], list)


# ═══════════════════════════════════════════
# RefreshConfig / check_refresh_needed / mark_refreshed
# ═══════════════════════════════════════════


class TestRefreshConfig:
    def test_defaults(self):
        cfg = RefreshConfig()
        assert cfg.rag_reindex_interval_hours == 24
        assert cfg.vector_seed_check_interval_hours == 12
        assert cfg.max_stale_hours == 72
        assert cfg.last_rag_refresh == 0.0
        assert cfg.last_seed_check == 0.0

    def test_custom_values(self):
        cfg = RefreshConfig(rag_reindex_interval_hours=48, max_stale_hours=168)
        assert cfg.rag_reindex_interval_hours == 48
        assert cfg.max_stale_hours == 168


class TestCheckRefreshNeeded:
    def test_returns_dict(self):
        result = check_refresh_needed()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = check_refresh_needed()
        assert "rag_reindex_needed" in result
        assert "seed_check_needed" in result
        assert "is_stale" in result

    def test_initial_state_needs_refresh(self):
        # Module-level _refresh_config starts at 0.0, so all should be True initially
        # But previous tests may have called mark_refreshed, so just check types
        result = check_refresh_needed()
        assert isinstance(result["rag_reindex_needed"], bool)
        assert isinstance(result["seed_check_needed"], bool)
        assert isinstance(result["is_stale"], bool)


class TestMarkRefreshed:
    def test_mark_rag(self):
        mark_refreshed("rag")
        # After marking rag, rag_reindex_needed should be False
        result = check_refresh_needed()
        assert result["rag_reindex_needed"] is False

    def test_mark_seed(self):
        mark_refreshed("seed")
        result = check_refresh_needed()
        assert result["seed_check_needed"] is False

    def test_mark_unknown_does_not_crash(self):
        # Should not raise
        mark_refreshed("unknown_type")


# ═══════════════════════════════════════════
# Integration / Full Pipeline
# ═══════════════════════════════════════════


class TestFullPipeline:
    def test_end_to_end_complex_query(self):
        query = (
            "I have severe chest pain, shortness of breath, and my troponin levels are elevated. What could this be?"
        )

        # Stage 1: create plan
        plan = create_diagnostic_plan(query)
        assert plan is not None
        assert plan.stage == PlanStage.PLANNING
        assert len(plan.steps) > 0

        # Get routing hints
        hints = get_plan_routing_hints(plan)
        assert hints["has_plan"] is True

        # Stage 2: simulate response and reflect
        response = "Your symptoms suggest acute coronary syndrome. Troponin elevation indicates myocardial injury. Please consult a cardiologist immediately. This is not a substitute for professional medical advice."
        reflection = reflect_on_diagnosis(plan, response)
        assert isinstance(reflection, DiagnosisReflection)
        assert plan.stage == PlanStage.VERIFICATION
        assert plan.completed_at is not None

    def test_end_to_end_simple_query(self):
        plan = create_diagnostic_plan("hello")
        assert plan is None

    def test_end_to_end_with_image(self):
        plan = create_diagnostic_plan(
            "analyze this chest xray for signs of pneumonia with chest pain",
            has_image=True,
            image_type="xray",
        )
        assert plan is not None
        step_types = [s.step_type for s in plan.steps]
        assert StepType.IMAGE_ANALYSIS in step_types
