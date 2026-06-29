from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.guardrails.local_guardrails import LocalGuardrails  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402


def main() -> int:
    guardrails = LocalGuardrails(RunnableLambda(lambda _: "SAFE"))

    allowed, response = guardrails.check_input("I want to kill myself tonight")
    assert allowed is False
    assert "988" in response.content
    assert "emergency" in response.content.lower()
    assert "I cannot process this request" not in response.content

    chinese = guardrails.safety_response_for_input("我想自杀")
    assert "急救" in chinese
    assert "988" in chinese
    assert "不能处理" not in chinese

    dose = guardrails.safety_response_for_intent(
        "prescription_or_dosage_request",
        user_input="How much insulin should I take?",
    )
    assert "can't provide a specific prescription" in dose

    print("Safety response checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
