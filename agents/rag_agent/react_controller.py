import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


RetrieveFn = Callable[[str, int], Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]]


@dataclass
class RetrievalAttempt:
    query: str
    top_k: int
    documents: List[Dict[str, Any]]
    picture_paths: List[str]
    retrieval_trace: Dict[str, Any]
    confidence: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "document_count": len(self.documents),
            "picture_count": len(self.picture_paths),
            "confidence": self.confidence,
            "error": self.error,
            "retrieval_trace": self.retrieval_trace,
        }


@dataclass
class ReActStep:
    step_id: int
    thought: str
    action: Dict[str, Any]
    observation: Dict[str, Any]
    duration_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "duration_ms": self.duration_ms,
        }


@dataclass
class RAGReActResult:
    selected_attempt: RetrievalAttempt
    attempts: List[RetrievalAttempt] = field(default_factory=list)
    steps: List[ReActStep] = field(default_factory=list)
    outcome: str = "unknown"
    elapsed_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "elapsed_ms": self.elapsed_ms,
            "selected_attempt": self.selected_attempt.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "steps": [step.to_dict() for step in self.steps],
        }


class RAGReActController:
    """Small ReAct-style controller for retrieval self-correction.

    It keeps the RAG flow deterministic and bounded: retrieve once, observe the
    quality, and retry with a broader query/top_k only when the first retrieval
    is weak.
    """

    def __init__(
        self,
        max_steps: int = 2,
        timeout_seconds: float = 8.0,
        min_chunks: int = 1,
        min_confidence: float = 0.4,
        top_k_growth: int = 2,
    ):
        self.max_steps = max(1, max_steps)
        self.timeout_seconds = timeout_seconds
        self.min_chunks = min_chunks
        self.min_confidence = min_confidence
        self.top_k_growth = max(1, top_k_growth)

    def run(
        self,
        *,
        original_query: str,
        initial_query: str,
        initial_top_k: int,
        retrieve: RetrieveFn,
    ) -> RAGReActResult:
        started_at = time.time()
        current_query = initial_query
        current_top_k = initial_top_k
        attempts: List[RetrievalAttempt] = []
        steps: List[ReActStep] = []
        outcome = "max_steps_reached"

        for step_index in range(1, self.max_steps + 1):
            if time.time() - started_at >= self.timeout_seconds:
                outcome = "timeout"
                break

            step_started_at = time.time()
            thought = self._build_thought(step_index, current_query, attempts)
            action = {
                "tool": "knowledge_base_retrieval",
                "query": current_query,
                "top_k": current_top_k,
            }

            try:
                documents, picture_paths, retrieval_trace = retrieve(current_query, current_top_k)
                confidence = self._confidence(documents)
                attempt = RetrievalAttempt(
                    query=current_query,
                    top_k=current_top_k,
                    documents=documents,
                    picture_paths=picture_paths,
                    retrieval_trace=retrieval_trace,
                    confidence=confidence,
                )
            except Exception as exc:
                attempt = RetrievalAttempt(
                    query=current_query,
                    top_k=current_top_k,
                    documents=[],
                    picture_paths=[],
                    retrieval_trace={"error": str(exc)},
                    confidence=0.0,
                    error=str(exc),
                )

            attempts.append(attempt)
            observation = self._build_observation(attempt)
            steps.append(
                ReActStep(
                    step_id=step_index,
                    thought=thought,
                    action=action,
                    observation=observation,
                    duration_ms=int((time.time() - step_started_at) * 1000),
                )
            )

            if self._is_sufficient(attempt):
                outcome = "sufficient_context"
                break

            if step_index >= self.max_steps:
                break

            current_query = self._repair_query(original_query, current_query, step_index)
            current_top_k *= self.top_k_growth

        selected_attempt = self._select_best_attempt(attempts)
        return RAGReActResult(
            selected_attempt=selected_attempt,
            attempts=attempts,
            steps=steps,
            outcome=outcome,
            elapsed_ms=int((time.time() - started_at) * 1000),
        )

    def _is_sufficient(self, attempt: RetrievalAttempt) -> bool:
        return (
            attempt.error is None
            and len(attempt.documents) >= self.min_chunks
            and attempt.confidence >= self.min_confidence
        )

    def _select_best_attempt(self, attempts: List[RetrievalAttempt]) -> RetrievalAttempt:
        if not attempts:
            return RetrievalAttempt(
                query="",
                top_k=0,
                documents=[],
                picture_paths=[],
                retrieval_trace={},
                confidence=0.0,
                error="No retrieval attempts executed",
            )
        return max(attempts, key=lambda item: (item.confidence, len(item.documents)))

    def _build_thought(
        self,
        step_index: int,
        current_query: str,
        attempts: List[RetrievalAttempt],
    ) -> str:
        if step_index == 1:
            return "Retrieve medical knowledge using the expanded user query."
        previous = attempts[-1]
        if previous.error:
            return "The previous retrieval tool call failed, retry with a safer query and larger recall window."
        return "The retrieved context is weak, broaden the query and increase top_k before answering."

    def _build_observation(self, attempt: RetrievalAttempt) -> Dict[str, Any]:
        channel_errors = [
            channel
            for channel in attempt.retrieval_trace.get("channels", [])
            if channel.get("error")
        ]
        return {
            "document_count": len(attempt.documents),
            "picture_count": len(attempt.picture_paths),
            "confidence": attempt.confidence,
            "sufficient": self._is_sufficient(attempt),
            "channel_errors": channel_errors,
            "error": attempt.error,
        }

    def _repair_query(self, original_query: str, current_query: str, step_index: int) -> str:
        if _normalize(current_query) != _normalize(original_query):
            return original_query

        noisy_terms = {
            "answer",
            "compare",
            "format",
            "in",
            "list",
            "please",
            "show",
            "summarize",
            "table",
            "tabular",
        }
        tokens = re.findall(r"[A-Za-z0-9-]+", original_query)
        filtered = [token for token in tokens if token.lower() not in noisy_terms]
        repaired = " ".join(filtered).strip()
        return repaired or original_query

    def _confidence(self, documents: List[Dict[str, Any]]) -> float:
        if not documents:
            return 0.0
        scores = [_score(document) for document in documents[:3]]
        return sum(scores) / len(scores)


def _score(chunk: Dict[str, Any]) -> float:
    return float(
        chunk.get(
            "retrieval_confidence",
            chunk.get(
                "combined_score",
                chunk.get(
                    "rerank_score",
                    chunk.get("score", 0.0),
                ),
            ),
        )
    )


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())
