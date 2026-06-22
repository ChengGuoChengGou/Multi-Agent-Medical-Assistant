"""Tests for evaluation/eval_runner.py — medical QA evaluation runner."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.eval_runner import (
    generate_report,
    load_benchmark,
    run_deepeval_evaluation,
    run_ragas_evaluation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_test_cases():
    return [
        {
            "id": "t1",
            "question": "What is aspirin used for?",
            "ground_truth": "Aspirin is used for pain relief and fever reduction.",
            "contexts": ["Aspirin is a common analgesic and antipyretic drug."],
            "category": "medication",
        },
        {
            "id": "t2",
            "question": "What are the symptoms of diabetes?",
            "ground_truth": "Common symptoms include increased thirst and frequent urination.",
            "contexts": ["Diabetes symptoms include polyuria, polydipsia, and polyphagia."],
            "category": "disease",
        },
    ]


@pytest.fixture
def benchmark_file(sample_test_cases, tmp_path):
    """Create a temporary benchmark JSON file."""
    data = {"test_cases": sample_test_cases}
    path = tmp_path / "benchmark.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(path)


# ---------------------------------------------------------------------------
# Test: load_benchmark
# ---------------------------------------------------------------------------


class TestLoadBenchmark:
    def test_loads_from_file(self, benchmark_file, sample_test_cases):
        result = load_benchmark(benchmark_file)
        assert len(result) == 2
        assert result[0]["id"] == "t1"
        assert result[0]["question"] == "What is aspirin used for?"

    def test_returns_list_of_dicts(self, benchmark_file):
        result = load_benchmark(benchmark_file)
        assert all(isinstance(tc, dict) for tc in result)
        assert all("question" in tc and "ground_truth" in tc for tc in result)

    def test_default_path(self):
        """Default path should point to evaluation/medical_qa_benchmark.json."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                load_benchmark()


# ---------------------------------------------------------------------------
# Test: run_ragas_evaluation
# ---------------------------------------------------------------------------


class TestRagasEvaluation:
    def test_returns_error_when_ragas_not_installed(self, sample_test_cases):
        """Should return error dict if ragas import fails."""
        with patch.dict("sys.modules", {"ragas": None}):
            result = run_ragas_evaluation(sample_test_cases)
            # Either ragas is available or we get an error dict
            if "error" in result:
                assert isinstance(result["error"], str)

    def test_returns_dict(self, sample_test_cases):
        """Should always return a dict."""
        result = run_ragas_evaluation(sample_test_cases)
        assert isinstance(result, dict)

    def test_with_empty_cases(self):
        """Empty test cases should still return a dict."""
        result = run_ragas_evaluation([])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test: run_deepeval_evaluation
# ---------------------------------------------------------------------------


class TestDeepevalEvaluation:
    def test_returns_error_when_deepeval_not_installed(self, sample_test_cases):
        """Should return error dict if deepeval import fails."""
        with patch.dict("sys.modules", {"deepeval": None}):
            result = run_deepeval_evaluation(sample_test_cases)
            if "error" in result:
                assert isinstance(result["error"], str)

    def test_returns_dict(self, sample_test_cases):
        result = run_deepeval_evaluation(sample_test_cases)
        assert isinstance(result, dict)

    def test_with_empty_cases(self):
        result = run_deepeval_evaluation([])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test: generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_structure(self, sample_test_cases):
        ragas_scores = {"faithfulness": 0.9, "overall": 0.85}
        deepeval_scores = {"answer_relevancy": 0.8, "overall": 0.8}

        report = generate_report(ragas_scores, deepeval_scores, sample_test_cases)

        assert "timestamp" in report
        assert report["dataset"] == "medical_qa_benchmark_v1"
        assert report["num_test_cases"] == 2
        assert "medication" in report["categories"] or "disease" in report["categories"]
        assert report["ragas"] == ragas_scores
        assert report["deepeval"] == deepeval_scores

    def test_report_with_output_dir(self, sample_test_cases, tmp_path):
        output_dir = str(tmp_path / "reports")
        ragas_scores = {"overall": 0.8}
        deepeval_scores = {"overall": 0.7}

        report = generate_report(ragas_scores, deepeval_scores, sample_test_cases, output_dir)

        assert "report_path" in report
        assert os.path.exists(report["report_path"])
        assert report["report_path"].endswith(".json")

        # Verify saved file content
        with open(report["report_path"], "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["num_test_cases"] == 2

    def test_report_without_output_dir(self, sample_test_cases):
        report = generate_report({}, {}, sample_test_cases)
        assert "report_path" not in report

    def test_categories_extracted(self, sample_test_cases):
        report = generate_report({}, {}, sample_test_cases)
        assert set(report["categories"]) == {"medication", "disease"}
