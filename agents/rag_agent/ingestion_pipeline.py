import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from uuid import uuid4


T = TypeVar("T")


class IngestionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class IngestionNodeType(str, Enum):
    PARSE_DOCUMENT = "parse_document"
    PARSE_FAQ = "parse_faq"
    SUMMARIZE_IMAGES = "summarize_images"
    FORMAT_DOCUMENT = "format_document"
    CHUNK_DOCUMENT = "chunk_document"
    INDEX_VECTORSTORE = "index_vectorstore"


@dataclass
class IngestionNodeTrace:
    node_id: str
    node_type: IngestionNodeType
    status: IngestionStatus = IngestionStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    output_summary: Dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        self.status = IngestionStatus.RUNNING
        self.start_time = time.time()

    def complete(self, output_summary: Optional[Dict[str, Any]] = None) -> None:
        self.status = IngestionStatus.SUCCESS
        self.end_time = time.time()
        if self.start_time is not None:
            self.duration_seconds = self.end_time - self.start_time
        if output_summary:
            self.output_summary = output_summary

    def fail(self, error: Exception) -> None:
        self.status = IngestionStatus.FAILED
        self.end_time = time.time()
        if self.start_time is not None:
            self.duration_seconds = self.end_time - self.start_time
        self.error = str(error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "output_summary": self.output_summary,
        }


@dataclass
class IngestionPipelineTrace:
    task_id: str
    source: str
    status: IngestionStatus = IngestionStatus.PENDING
    nodes: List[IngestionNodeTrace] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None

    @classmethod
    def create(cls, source: str) -> "IngestionPipelineTrace":
        return cls(task_id=str(uuid4()), source=source)

    def start(self) -> None:
        self.status = IngestionStatus.RUNNING
        self.start_time = time.time()

    def complete(self) -> None:
        self.status = IngestionStatus.SUCCESS
        self.end_time = time.time()
        if self.start_time is not None:
            self.duration_seconds = self.end_time - self.start_time

    def fail(self, error: Exception) -> None:
        self.status = IngestionStatus.FAILED
        self.end_time = time.time()
        if self.start_time is not None:
            self.duration_seconds = self.end_time - self.start_time
        self.error = str(error)

    def run_node(
        self,
        node_type: IngestionNodeType,
        operation: Callable[[], T],
        output_summarizer: Optional[Callable[[T], Dict[str, Any]]] = None,
    ) -> T:
        node = IngestionNodeTrace(
            node_id=f"{len(self.nodes) + 1:02d}-{node_type.value}",
            node_type=node_type,
        )
        self.nodes.append(node)
        node.start()
        try:
            output = operation()
            summary = output_summarizer(output) if output_summarizer else {}
            node.complete(summary)
            return output
        except Exception as exc:
            node.fail(exc)
            self.fail(exc)
            raise

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "nodes": [node.to_dict() for node in self.nodes],
        }


def summarize_parse_output(output: Tuple[Any, List[str]]) -> Dict[str, Any]:
    _, images = output
    return {"images_extracted": len(images)}


def summarize_sequence(name: str) -> Callable[[List[Any]], Dict[str, Any]]:
    def summarizer(items: List[Any]) -> Dict[str, Any]:
        return {name: len(items)}

    return summarizer


def summarize_text(name: str) -> Callable[[str], Dict[str, Any]]:
    def summarizer(text: str) -> Dict[str, Any]:
        return {name: len(text)}

    return summarizer
