from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class TraceEvent:
    step: int
    event_type: str
    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentTrace:
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    events: List[TraceEvent] = field(default_factory=list)
    status: str = "RUNNING"
    error: Optional[str] = None

    def thought(self, name: str, content: str, **metadata: Any) -> TraceEvent:
        return self.add("thought", name, content, **metadata)

    def action(self, name: str, content: str, **metadata: Any) -> TraceEvent:
        return self.add("action", name, content, **metadata)

    def observation(self, name: str, content: str, **metadata: Any) -> TraceEvent:
        return self.add("observation", name, content, **metadata)

    def final(self, name: str, content: str, **metadata: Any) -> TraceEvent:
        self.status = "SUCCESS"
        return self.add("final", name, content, **metadata)

    def fail(self, name: str, content: str, **metadata: Any) -> TraceEvent:
        self.status = "ERROR"
        self.error = content
        return self.add("error", name, content, **metadata)

    def add(self, event_type: str, name: str, content: str, **metadata: Any) -> TraceEvent:
        event = TraceEvent(
            step=len(self.events) + 1,
            event_type=event_type,
            name=name,
            content=content,
            metadata=metadata,
        )
        self.events.append(event)
        return event

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "status": self.status,
            "error": self.error,
            "events": [asdict(event) for event in self.events],
        }
