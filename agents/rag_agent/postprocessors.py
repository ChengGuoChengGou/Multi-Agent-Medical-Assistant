from typing import Any, Dict, List, Tuple


class DeduplicationPostProcessor:
    name = "deduplication"
    order = 1

    def process(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        channel_results: List[Any],
    ) -> List[Dict[str, Any]]:
        best_by_id: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = _dedupe_key(chunk)
            current_best = best_by_id.get(chunk_id)
            if current_best is None or _score(chunk) > _score(current_best):
                best_by_id[chunk_id] = chunk
        return sorted(best_by_id.values(), key=_score, reverse=True)


class FAQPriorityPostProcessor:
    name = "faq_priority_boost"
    order = 5

    PRIORITY_BOOST = {
        "P0": 0.12,
        "P1": 0.06,
        "P2": 0.02,
    }
    RISK_BOOST = {
        "high": 0.05,
        "medium": 0.02,
    }

    def process(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        channel_results: List[Any],
    ) -> List[Dict[str, Any]]:
        boosted_chunks = []
        for chunk in chunks:
            if not _is_patient_faq(chunk):
                boosted_chunks.append(chunk)
                continue

            boosted = dict(chunk)
            current_score = float(boosted.get("score", 0.0))
            priority = str(boosted.get("priority", "")).upper()
            risk_level = str(boosted.get("risk_level", "")).lower()
            priority_boost = self.PRIORITY_BOOST.get(priority, 0.0)
            risk_boost = self.RISK_BOOST.get(risk_level, 0.0)
            doctor_boost = 0.03 if _as_bool(boosted.get("need_doctor")) else 0.0
            total_boost = priority_boost + risk_boost + doctor_boost

            boosted["score_before_priority_boost"] = current_score
            boosted["priority_boost"] = total_boost
            boosted["score"] = current_score + total_boost
            boosted_chunks.append(boosted)

        return sorted(boosted_chunks, key=_score, reverse=True)


class CrossEncoderRerankPostProcessor:
    name = "cross_encoder_rerank"
    order = 10

    def __init__(self, reranker, parsed_content_dir: str):
        self.reranker = reranker
        self.parsed_content_dir = parsed_content_dir

    def process(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        channel_results: List[Any],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not self.reranker or len(chunks) <= 1:
            return chunks, []
        reranked = self.reranker.rerank(query, chunks, self.parsed_content_dir)
        if isinstance(reranked, tuple):
            return reranked
        return reranked, []


class ConfidencePostProcessor:
    name = "confidence_calculator"
    order = 20

    def process(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        channel_results: List[Any],
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return chunks
        for idx, chunk in enumerate(chunks):
            base_score = _score(chunk)
            rank_bonus = max(0.0, 0.05 - idx * 0.01)
            channel_bonus = 0.03 if chunk.get("retrieval_channel") == "intent_directed_medical_search" else 0.0
            chunk["retrieval_confidence"] = max(0.0, min(base_score + rank_bonus + channel_bonus, 1.0))
        return chunks


def _score(chunk: Dict[str, Any]) -> float:
    return float(
        chunk.get(
            "combined_score",
            chunk.get(
                "rerank_score",
                chunk.get("score", 0.0),
            ),
        )
    )


def _is_patient_faq(chunk: Dict[str, Any]) -> bool:
    return (
        chunk.get("content_type") == "patient_faq"
        or chunk.get("source_type") == "patient_faq"
        or bool(chunk.get("faq_id"))
    )


def _dedupe_key(chunk: Dict[str, Any]) -> str:
    if chunk.get("faq_id"):
        return f"faq:{chunk['faq_id']}"
    return str(chunk.get("id") or chunk.get("content", ""))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
