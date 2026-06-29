from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402


def main() -> int:
    retrieved = ["doc-a", "doc-b", "doc-c", "doc-d"]
    expected = ["doc-c", "doc-x"]

    assert recall_at_k(retrieved, expected, 1) == 0.0
    assert recall_at_k(retrieved, expected, 3) == 0.5
    assert reciprocal_rank(retrieved, expected) == 1 / 3
    assert 0.0 < ndcg_at_k(retrieved, expected, 3) <= 1.0

    print("Metric checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
