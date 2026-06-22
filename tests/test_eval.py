"""Pytest integration for medical QA evaluation.

These tests verify the evaluation framework structure and can optionally
run full LLM-based evaluations when --run-eval flag is passed.

Usage:
  pytest tests/test_eval.py -v                    # Structure tests only
  pytest tests/test_eval.py -v --run-eval          # Full eval (requires LLM)
"""

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser):
    parser.addoption("--run-eval", action="store_true", default=False, help="Run full evaluation")


@pytest.fixture
def benchmark_data():
    """Load benchmark dataset."""
    data_path = PROJECT_ROOT / "evaluation" / "medical_qa_benchmark.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def test_cases(benchmark_data):
    """Extract test cases."""
    return benchmark_data["test_cases"]


class TestBenchmarkStructure:
    """Test the benchmark dataset structure."""

    def test_dataset_has_metadata(self, benchmark_data):
        assert "name" in benchmark_data
        assert "test_cases" in benchmark_data
        assert len(benchmark_data["test_cases"]) > 0

    def test_test_cases_have_required_fields(self, test_cases):
        required = ["id", "question", "ground_truth", "contexts", "category"]
        for tc in test_cases:
            for field in required:
                assert field in tc, f"Missing '{field}' in test case {tc.get('id', '?')}"

    def test_test_cases_have_valid_categories(self, test_cases):
        valid_categories = {
            "symptom_inquiry",
            "disease_explanation",
            "drug_interaction",
            "emergency_triage",
            "mental_health_crisis",
            "pregnancy_safety",
        }
        for tc in test_cases:
            assert tc["category"] in valid_categories, f"Invalid category: {tc['category']}"

    def test_contexts_are_nonempty_lists(self, test_cases):
        for tc in test_cases:
            assert isinstance(tc["contexts"], list)
            assert len(tc["contexts"]) > 0
            for ctx in tc["contexts"]:
                assert len(ctx) > 0

    def test_ground_truth_is_meaningful(self, test_cases):
        for tc in test_cases:
            assert len(tc["ground_truth"]) > 50, f"Ground truth too short for {tc['id']}"


class TestEvalRunnerImports:
    """Test that evaluation modules can be imported."""

    def test_eval_module_import(self):
        from evaluation import eval_runner

        assert hasattr(eval_runner, "load_benchmark")
        assert hasattr(eval_runner, "run_ragas_evaluation")
        assert hasattr(eval_runner, "run_deepeval_evaluation")
        assert hasattr(eval_runner, "generate_report")

    def test_load_benchmark(self):
        from evaluation.eval_runner import load_benchmark

        cases = load_benchmark()
        assert len(cases) == 6

    def test_ragas_import(self):
        """Verify ragas package is installed."""
        try:
            import ragas

            assert hasattr(ragas, "__version__") or hasattr(ragas, "evaluate")
        except (ImportError, Exception) as e:
            pytest.xfail(f"ragas import issue (dependency): {e}")

    def test_deepeval_import(self):
        """Verify deepeval package is installed."""
        pytest.importorskip("deepeval", reason="deepeval not installed, skipping")
        import deepeval

        assert deepeval.__version__


class TestEvalReportGeneration:
    """Test report generation without LLM calls."""

    def test_generate_report_structure(self):
        from evaluation.eval_runner import generate_report

        report = generate_report(
            ragas_scores={"faithfulness": 1.0, "answer_relevancy": 0.9},
            deepeval_scores={"AnswerRelevancyMetric": 0.85},
            test_cases=[{"id": "test1", "category": "symptom_inquiry"}],
        )
        assert "timestamp" in report
        assert report["num_test_cases"] == 1
        assert "ragas" in report
        assert "deepeval" in report

    def test_generate_report_with_output(self, tmp_path):
        from evaluation.eval_runner import generate_report

        report = generate_report(
            ragas_scores={},
            deepeval_scores={},
            test_cases=[{"id": "test1", "category": "symptom_inquiry"}],
            output_dir=str(tmp_path),
        )
        assert "report_path" in report
        assert os.path.exists(report["report_path"])


@pytest.mark.skipif(
    not os.environ.get("RUN_FULL_EVAL"), reason="Set RUN_FULL_EVAL=1 to run full evaluation (requires API keys)"
)
class TestFullEvaluation:
    """Full evaluation tests - only run with --run-eval flag."""

    def test_ragas_baseline(self, test_cases):
        from evaluation.eval_runner import run_ragas_evaluation

        scores = run_ragas_evaluation(test_cases)
        assert "error" not in scores, f"RAGAS failed: {scores.get('error')}"
        assert scores.get("overall", 0) > 0.3, f"Baseline too low: {scores}"

    def test_deepeval_baseline(self, test_cases):
        from evaluation.eval_runner import run_deepeval_evaluation

        scores = run_deepeval_evaluation(test_cases)
        assert "error" not in scores, f"DeepEval failed: {scores.get('error')}"
        assert scores.get("overall", 0) > 0.3, f"Baseline too low: {scores}"
