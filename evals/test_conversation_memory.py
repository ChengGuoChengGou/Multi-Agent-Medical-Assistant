from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.conversation_memory import ConversationMemoryService  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ConversationMemoryService(
            storage_dir=tmpdir,
            window_size=4,
            summary_trigger=5,
            max_summary_chars=600,
        )
        session_id = "eval-session"
        for index in range(4):
            memory.append(
                session_id,
                user_content=f"user turn {index} about diabetes",
                assistant_content=f"assistant turn {index} about glucose",
            )

        snapshot = memory.load(session_id)
        assert snapshot.summary
        assert "diabetes" in snapshot.summary
        assert len(snapshot.recent_messages) == 4
        assert "assistant turn 3" in snapshot.to_prompt_context()

    print("Conversation memory checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
