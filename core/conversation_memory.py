from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MemoryMessage:
    role: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "MemoryMessage":
        return cls(
            role=str(value.get("role", "message")),
            content=str(value.get("content", "")),
            created_at=str(value.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class ConversationMemorySnapshot:
    session_id: str
    summary: str
    recent_messages: List[MemoryMessage]
    total_messages: int

    def to_prompt_context(self) -> str:
        lines: List[str] = []
        if self.summary.strip():
            lines.append(f"Conversation summary: {self.summary.strip()}")
        for message in self.recent_messages:
            if message.content.strip():
                lines.append(f"{message.role}: {message.content.strip()}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "recent_messages": [message.to_dict() for message in self.recent_messages],
            "total_messages": self.total_messages,
        }


class ConversationMemoryService:
    """File-backed sliding-window conversation memory.

    It mirrors Ragent's memory shape in a lightweight local form: a summary for
    older turns plus a bounded recent window. The summary is extractive and
    deterministic so it works in offline/local validation.
    """

    def __init__(
        self,
        storage_dir: str = "data/session_memory",
        window_size: int = 8,
        summary_trigger: int = 10,
        max_summary_chars: int = 1200,
    ):
        self.storage_dir = Path(storage_dir)
        self.window_size = max(2, window_size)
        self.summary_trigger = max(self.window_size + 1, summary_trigger)
        self.max_summary_chars = max_summary_chars
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: Optional[str]) -> ConversationMemorySnapshot:
        safe_session_id = _safe_session_id(session_id)
        payload = self._read_payload(safe_session_id)
        messages = [
            MemoryMessage.from_dict(item)
            for item in payload.get("messages", [])
            if isinstance(item, dict)
        ]
        return ConversationMemorySnapshot(
            session_id=safe_session_id,
            summary=str(payload.get("summary", "")),
            recent_messages=messages[-self.window_size :],
            total_messages=len(messages),
        )

    def append(
        self,
        session_id: Optional[str],
        *,
        user_content: str,
        assistant_content: str,
    ) -> ConversationMemorySnapshot:
        safe_session_id = _safe_session_id(session_id)
        payload = self._read_payload(safe_session_id)
        messages = [
            MemoryMessage.from_dict(item)
            for item in payload.get("messages", [])
            if isinstance(item, dict)
        ]
        if user_content:
            messages.append(MemoryMessage(role="user", content=str(user_content)))
        if assistant_content:
            messages.append(MemoryMessage(role="assistant", content=str(assistant_content)))

        summary = str(payload.get("summary", ""))
        if len(messages) > self.summary_trigger:
            older = messages[: -self.window_size]
            messages = messages[-self.window_size :]
            summary = self._summarize(summary, older)

        self._write_payload(
            safe_session_id,
            {
                "session_id": safe_session_id,
                "summary": summary,
                "messages": [message.to_dict() for message in messages],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return ConversationMemorySnapshot(
            session_id=safe_session_id,
            summary=summary,
            recent_messages=messages[-self.window_size :],
            total_messages=len(messages),
        )

    def _read_payload(self, session_id: str) -> Dict[str, Any]:
        path = self._path(session_id)
        if not path.exists():
            return {"session_id": session_id, "summary": "", "messages": []}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {"session_id": session_id, "summary": "", "messages": []}
        return {"session_id": session_id, "summary": "", "messages": []}

    def _write_payload(self, session_id: str, payload: Dict[str, Any]) -> None:
        path = self._path(session_id)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.storage_dir),
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            temp_name = handle.name
        Path(temp_name).replace(path)

    def _path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def _summarize(self, existing_summary: str, older_messages: Iterable[MemoryMessage]) -> str:
        facts = _extract_memory_facts(older_messages)
        parts = []
        if existing_summary.strip():
            parts.append(existing_summary.strip())
        parts.extend(facts)
        summary = "；".join(_dedupe(parts))
        if len(summary) <= self.max_summary_chars:
            return summary
        return summary[-self.max_summary_chars :]


def _extract_memory_facts(messages: Iterable[MemoryMessage]) -> List[str]:
    facts: List[str] = []
    for message in messages:
        content = " ".join((message.content or "").split())
        if not content:
            continue
        content = _strip_markdown_sections(content)
        if len(content) > 180:
            content = content[:177].rstrip() + "..."
        facts.append(f"{message.role}: {content}")
    return facts


def _strip_markdown_sections(text: str) -> str:
    return re.sub(r"#+\s*", "", text).strip()


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


def _safe_session_id(session_id: Optional[str]) -> str:
    raw = str(session_id or "default")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return safe or "default"
