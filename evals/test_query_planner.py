from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rag_agent.query_planner import QueryPlanner  # noqa: E402


def main() -> int:
    planner = QueryPlanner(max_sub_questions=3)

    vague = planner.plan("这个严重吗")
    assert vague.needs_clarification is True
    assert vague.sub_questions == []
    assert vague.clarification_question

    follow_up = planner.plan(
        "这个需要做什么检查",
        chat_history="User: 我的 HbA1c 偏高，怀疑 diabetes\nAssistant: 建议关注血糖。",
    )
    assert follow_up.needs_clarification is False
    assert "diabetes" in follow_up.rewritten_query.lower() or "glucose" in follow_up.rewritten_query.lower()

    split = planner.plan("Compare diabetes symptoms and HbA1c testing and treatment basics")
    assert len(split.sub_questions) >= 2
    assert len(split.sub_questions) <= 3

    print("Query planner checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
