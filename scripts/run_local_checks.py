from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


CHECKS = {
    "demo": ["python", "demo/offline_agentic_walkthrough.py"],
    "intent_eval": ["python", "evals/run_eval.py", "--task", "intent"],
    "tool_eval": ["python", "evals/run_eval.py", "--task", "tool"],
    "faq_retrieval_eval": ["python", "evals/run_faq_retrieval_eval.py", "--mode", "keyword"],
    "knowledge_inventory": ["python", "scripts/knowledge_inventory.py"],
    "dataset_report": ["python", "evals/dataset_report.py", "--dataset", "evals/datasets/intent_routing_eval.jsonl"],
    "metrics": ["python", "evals/test_metrics.py"],
    "model_gateway": ["python", "evals/test_model_gateway.py"],
    "prompt_registry": ["python", "evals/test_prompt_registry.py"],
    "rate_limiter": ["python", "evals/test_rate_limiter.py"],
    "react_controller": ["python", "evals/test_react_controller.py"],
    "query_planner": ["python", "evals/test_query_planner.py"],
    "conversation_memory": ["python", "evals/test_conversation_memory.py"],
    "tool_registry": ["python", "evals/test_tool_registry.py"],
    "safety_responses": ["python", "evals/test_safety_responses.py"],
    "compile": [
        "python",
        "-m",
        "compileall",
        "config.py",
        "core",
        "agents/rag_agent",
        "agents/agent_decision.py",
        "agents/router",
        "agents/tracing",
        "agents/tools",
        "evals",
        "demo",
        "scripts",
    ],
}


def run_check(name: str, command: List[str]) -> int:
    print(f"\n=== {name} ===", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local non-network project checks.")
    parser.add_argument(
        "checks",
        nargs="*",
        choices=sorted(CHECKS),
        help="Specific checks to run. Defaults to all checks.",
    )
    args = parser.parse_args()

    selected = args.checks or list(CHECKS)
    failures = []
    for name in selected:
        exit_code = run_check(name, CHECKS[name])
        if exit_code != 0:
            failures.append((name, exit_code))

    if failures:
        print("\nFailures:")
        for name, exit_code in failures:
            print(f"- {name}: exit_code={exit_code}")
        return 1

    print("\nAll selected checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
