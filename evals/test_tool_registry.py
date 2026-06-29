from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.tools import ToolSpec, build_default_tool_registry  # noqa: E402


def main() -> int:
    registry = build_default_tool_registry()

    kb = registry.get_spec("knowledge_base")
    assert kb is not None
    assert kb.kind == "KB"
    assert kb.agent_name == "RAG_AGENT"

    web = registry.get_spec("web_search")
    assert web is not None
    assert web.is_external is True

    missing = registry.execute("not-a-tool", {})
    assert missing.ok is False
    assert missing.error == "tool_not_registered"

    registry.register(
        ToolSpec(
            tool_id="echo",
            name="Echo",
            kind="TEST",
            description="Test executor",
            agent_name="TEST_AGENT",
        ),
        executor=lambda params: {"echo": params.get("text")},
    )
    result = registry.execute("echo", {"text": "hello"})
    assert result.ok is True
    assert result.content == {"echo": "hello"}

    print("Tool registry checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
