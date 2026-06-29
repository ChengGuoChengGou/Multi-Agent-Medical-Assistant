from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rag_agent.react_controller import RAGReActController  # noqa: E402


def main() -> int:
    calls = []

    def retrieve(query: str, top_k: int):
        calls.append((query, top_k))
        if len(calls) == 1:
            return [], [], {"channels": [], "final_chunk_count": 0}
        return [
            {
                "id": "chunk-1",
                "content": "Brain tumor MRI diagnosis context",
                "retrieval_confidence": 0.82,
            }
        ], [], {"channels": [{"channel_name": "qdrant_global_search"}], "final_chunk_count": 1}

    controller = RAGReActController(
        max_steps=2,
        timeout_seconds=2.0,
        min_chunks=1,
        min_confidence=0.4,
    )
    result = controller.run(
        original_query="Please answer in table format: brain tumor MRI diagnosis",
        initial_query="Please answer in table format: brain tumor MRI diagnosis oncology imaging",
        initial_top_k=5,
        retrieve=retrieve,
    )

    assert len(calls) == 2
    assert calls[1][1] == 10
    assert result.outcome == "sufficient_context"
    assert result.selected_attempt.confidence == 0.82
    assert result.steps[0].observation["sufficient"] is False
    assert result.steps[1].observation["sufficient"] is True

    print("ReAct controller checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
