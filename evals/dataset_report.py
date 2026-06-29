from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def print_counter(title: str, counter: Counter) -> None:
    total = sum(counter.values())
    print(title)
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        print(f"- {key}: {count} ({count / total:.1%})")


def report(path: Path) -> int:
    rows = load_jsonl(path)
    print(f"Dataset: {path}")
    print(f"Samples: {len(rows)}")

    difficulty = Counter(row.get("difficulty", "unknown") for row in rows)
    print_counter("\nDifficulty distribution", difficulty)

    if any("expected_intent" in row for row in rows):
        print_counter(
            "\nIntent distribution",
            Counter(row.get("expected_intent", "unknown") for row in rows),
        )

    if any("expected_agent" in row for row in rows):
        print_counter(
            "\nAgent distribution",
            Counter(row.get("expected_agent", "unknown") for row in rows),
        )

    if any("expected_tool" in row for row in rows):
        print_counter(
            "\nTool distribution",
            Counter(str(row.get("expected_tool")) for row in rows),
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Print evaluation dataset distribution.")
    parser.add_argument(
        "--dataset",
        default="evals/datasets/intent_routing_eval.jsonl",
        help="Path to a JSONL dataset.",
    )
    args = parser.parse_args()
    return report(ROOT / args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
