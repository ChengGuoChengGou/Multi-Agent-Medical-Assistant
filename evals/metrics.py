from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List


@dataclass
class AccuracyResult:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def accuracy(rows: Iterable[Dict[str, Any]], expected_key: str, actual_key: str) -> AccuracyResult:
    total = 0
    correct = 0
    for row in rows:
        total += 1
        if row.get(expected_key) == row.get(actual_key):
            correct += 1
    return AccuracyResult(total=total, correct=correct)


def group_by_difficulty(rows: Iterable[Dict[str, Any]], expected_key: str, actual_key: str) -> Dict[str, AccuracyResult]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("difficulty", "unknown"), []).append(row)
    return {
        difficulty: accuracy(group_rows, expected_key, actual_key)
        for difficulty, group_rows in groups.items()
    }


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    expected = set(expected_ids)
    return len(retrieved & expected) / len(expected)


def reciprocal_rank(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    expected = set(expected_ids)
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in expected:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    expected = set(expected_ids)
    dcg = 0.0
    for index, retrieved_id in enumerate(retrieved_ids[:k], start=1):
        relevance = 1.0 if retrieved_id in expected else 0.0
        dcg += relevance / math.log2(index + 1)

    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
