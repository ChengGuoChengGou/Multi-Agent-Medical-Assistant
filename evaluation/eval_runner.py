"""Medical QA Evaluation Runner.

Uses RAGAS for RAG quality metrics and DeepEval for LLM safety metrics.

Metrics:
  RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
  DeepEval: answer_relevancy, faithfulness, hallucination, bias, toxicity

Usage:
  python -m evaluation.eval_runner                    # Run all evaluations
  python -m evaluation.eval_runner --metrics ragas    # RAGAS only
  python -m evaluation.eval_runner --metrics deepeval # DeepEval only
  python -m evaluation.eval_runner --report results/  # Save report
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def load_benchmark(path: str = None) -> list[dict]:
    """Load medical QA benchmark dataset."""
    if path is None:
        path = Path(__file__).parent / "medical_qa_benchmark.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def run_ragas_evaluation(test_cases: list[dict], llm=None) -> dict:
    """Run RAGAS evaluation on test cases.
    
    Args:
        test_cases: List of {question, ground_truth, contexts, ...}
        llm: LLM instance for RAGAS metrics (optional, uses default)
    
    Returns:
        Dict of metric_name -> score
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError as e:
        logger.warning("RAGAS not available: %s", e)
        return {"error": str(e)}

    # Build dataset in RAGAS format
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for tc in test_cases:
        questions.append(tc["question"])
        answers.append(tc["ground_truth"])  # Using ground truth as "answer" for baseline
        contexts_list.append(tc["contexts"])
        ground_truths.append(tc["ground_truth"])

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    try:
        result = evaluate(
            dataset=ds,
            metrics=metrics,
        )
        scores = {m: result[m] for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
        scores["overall"] = sum(scores.values()) / len(scores)
        return scores
    except Exception as e:
        logger.warning("RAGAS evaluation failed: %s", e)
        return {"error": str(e)}


def run_deepeval_evaluation(test_cases: list[dict]) -> dict:
    """Run DeepEval evaluation on test cases.
    
    Returns:
        Dict of metric_name -> score
    """
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
            BiasMetric,
            ToxicityMetric,
        )
    except ImportError as e:
        logger.warning("DeepEval not available: %s", e)
        return {"error": str(e)}

    # Prepare test cases
    test_case_objects = []
    for tc in test_cases:
        try:
            test_case = LLMTestCase(
                input=tc["question"],
                actual_output=tc["ground_truth"],
                retrieval_context=tc["contexts"],
                expected_output=tc["ground_truth"],
            )
            test_case_objects.append(test_case)
        except Exception as e:
            logger.warning("Failed to create test case %s: %s", tc["id"], e)

    # Define metrics with relaxed thresholds for medical domain
    metrics = [
        AnswerRelevancyMetric(threshold=0.5),
        FaithfulnessMetric(threshold=0.5),
        HallucinationMetric(threshold=0.5),
        BiasMetric(threshold=0.7),     # Higher threshold - medical advice can seem "biased"
        ToxicityMetric(threshold=0.9), # Very high - medical terms aren't toxic
    ]

    results_summary = {}
    for metric in metrics:
        scores = []
        for tc in test_case_objects:
            try:
                metric.measure(tc)
                scores.append(metric.score)
            except Exception as e:
                logger.debug("Metric %s failed on test case: %s", metric.__name__, e)
        if scores:
            results_summary[metric.__name__] = sum(scores) / len(scores)

    if results_summary:
        results_summary["overall"] = sum(results_summary.values()) / len(results_summary)

    return results_summary


def generate_report(
    ragas_scores: dict,
    deepeval_scores: dict,
    test_cases: list[dict],
    output_dir: str = None,
) -> dict:
    """Generate evaluation report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "medical_qa_benchmark_v1",
        "num_test_cases": len(test_cases),
        "categories": sorted(set(tc["category"] for tc in test_cases)),
        "ragas": ragas_scores,
        "deepeval": deepeval_scores,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"eval_report_{ts}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["report_path"] = report_path

    return report


def main():
    parser = argparse.ArgumentParser(description="Medical QA Evaluation Runner")
    parser.add_argument("--metrics", choices=["ragas", "deepeval", "both"], default="both",
                       help="Which metrics to run")
    parser.add_argument("--report", type=str, default=None,
                       help="Output directory for reports")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to benchmark dataset JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    test_cases = load_benchmark(args.data)
    logger.info("Loaded %d test cases", len(test_cases))

    ragas_scores = {}
    deepeval_scores = {}

    if args.metrics in ("ragas", "both"):
        logger.info("Running RAGAS evaluation...")
        ragas_scores = run_ragas_evaluation(test_cases)
        logger.info("RAGAS scores: %s", ragas_scores)

    if args.metrics in ("deepeval", "both"):
        logger.info("Running DeepEval evaluation...")
        deepeval_scores = run_deepeval_evaluation(test_cases)
        logger.info("DeepEval scores: %s", deepeval_scores)

    report = generate_report(ragas_scores, deepeval_scores, test_cases, args.report)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Test cases: {report['num_test_cases']}")
    print(f"Categories: {', '.join(report['categories'])}")
    print()

    if ragas_scores and "error" not in ragas_scores:
        print("RAGAS Metrics:")
        for k, v in ragas_scores.items():
            print(f"  {k}: {v:.4f}")
    elif ragas_scores.get("error"):
        print(f"RAGAS: skipped ({ragas_scores['error']})")

    if deepeval_scores and "error" not in deepeval_scores:
        print("\nDeepEval Metrics:")
        for k, v in deepeval_scores.items():
            print(f"  {k}: {v:.4f}")
    elif deepeval_scores.get("error"):
        print(f"DeepEval: skipped ({deepeval_scores['error']})")

    if report.get("report_path"):
        print(f"\nReport saved: {report['report_path']}")

    print("=" * 60)

    return report


if __name__ == "__main__":
    main()
