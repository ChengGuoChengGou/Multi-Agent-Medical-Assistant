from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.router import IntentRouter  # noqa: E402
from evals.metrics import accuracy, group_by_difficulty  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_intents(path: Path) -> int:
    router = IntentRouter()
    rows = load_jsonl(path)
    evaluated = []

    for row in rows:
        decision = router.classify(
            row.get("query", ""),
            has_image=row.get("has_image", False),
            image_type=row.get("image_type"),
        )
        evaluated.append(
            {
                **row,
                "actual_intent": decision.intent.value,
                "actual_agent": decision.agent_name,
                "actual_tool": decision.tool_name,
                "confidence": decision.confidence,
                "reason": decision.reason,
            }
        )

    intent_result = accuracy(evaluated, "expected_intent", "actual_intent")
    agent_result = accuracy(evaluated, "expected_agent", "actual_agent")
    tool_result = accuracy(evaluated, "expected_tool", "actual_tool")

    print("Intent routing evaluation")
    print(f"- Dataset: {path}")
    print(f"- Samples: {intent_result.total}")
    print(f"- Intent accuracy: {intent_result.correct}/{intent_result.total} = {intent_result.accuracy:.2%}")
    print(f"- Agent accuracy: {agent_result.correct}/{agent_result.total} = {agent_result.accuracy:.2%}")
    print(f"- Tool accuracy: {tool_result.correct}/{tool_result.total} = {tool_result.accuracy:.2%}")

    print("\nBy difficulty")
    for difficulty, result in group_by_difficulty(evaluated, "expected_intent", "actual_intent").items():
        print(f"- {difficulty}: {result.correct}/{result.total} = {result.accuracy:.2%}")

    failures = [
        row
        for row in evaluated
        if row.get("expected_intent") != row.get("actual_intent")
        or row.get("expected_agent") != row.get("actual_agent")
        or row.get("expected_tool") != row.get("actual_tool")
    ]
    if failures:
        print("\nFailures")
        for row in failures:
            print(
                f"- {row['id']}: expected=({row.get('expected_intent')}, "
                f"{row.get('expected_agent')}, {row.get('expected_tool')}) actual=("
                f"{row.get('actual_intent')}, {row.get('actual_agent')}, {row.get('actual_tool')}) "
                f"confidence={row.get('confidence')} query={row.get('query')!r}"
            )

    return 0 if not failures else 1


def evaluate_tool_calls(path: Path) -> int:
    router = IntentRouter()
    rows = load_jsonl(path)
    evaluated = []

    for row in rows:
        decision = router.classify(
            row.get("query", ""),
            has_image=row.get("has_image", False),
            image_type=row.get("image_type"),
        )
        evaluated.append(
            {
                **row,
                "actual_agent": decision.agent_name,
                "actual_tool": decision.tool_name,
                "actual_confidence": decision.confidence,
                "actual_reason": decision.reason,
            }
        )

    agent_result = accuracy(evaluated, "expected_agent", "actual_agent")
    tool_result = accuracy(evaluated, "expected_tool", "actual_tool")

    print("Tool call evaluation")
    print(f"- Dataset: {path}")
    print(f"- Samples: {tool_result.total}")
    print(f"- Agent accuracy: {agent_result.correct}/{agent_result.total} = {agent_result.accuracy:.2%}")
    print(f"- Tool accuracy: {tool_result.correct}/{tool_result.total} = {tool_result.accuracy:.2%}")

    print("\nBy difficulty")
    for difficulty, result in group_by_difficulty(evaluated, "expected_tool", "actual_tool").items():
        print(f"- {difficulty}: {result.correct}/{result.total} = {result.accuracy:.2%}")

    failures = [
        row
        for row in evaluated
        if row.get("expected_agent") != row.get("actual_agent")
        or row.get("expected_tool") != row.get("actual_tool")
    ]
    if failures:
        print("\nFailures")
        for row in failures:
            print(
                f"- {row['id']}: expected=({row.get('expected_agent')}, {row.get('expected_tool')}) "
                f"actual=({row.get('actual_agent')}, {row.get('actual_tool')}) "
                f"confidence={row.get('actual_confidence')} query={row.get('query')!r}"
            )

    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local evaluation for the medical assistant.")
    parser.add_argument(
        "--task",
        choices=["intent", "tool"],
        default="intent",
        help="Which evaluation task to run.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a JSONL evaluation dataset.",
    )
    args = parser.parse_args()

    if args.task == "tool":
        dataset = args.dataset or "evals/datasets/tool_call_eval.jsonl"
        return evaluate_tool_calls(ROOT / dataset)

    dataset = args.dataset or "evals/datasets/intent_routing_eval.jsonl"
    return evaluate_intents(ROOT / dataset)


if __name__ == "__main__":
    raise SystemExit(main())
