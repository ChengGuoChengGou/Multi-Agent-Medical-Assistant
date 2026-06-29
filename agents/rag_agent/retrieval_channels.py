import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .faq_loader import FAQEntry, load_faq_directory


@dataclass
class SearchContext:
    query: str
    vectorstore: Any
    docstore: Any
    top_k: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchChannelResult:
    channel_name: str
    channel_type: str
    chunks: List[Dict[str, Any]]
    latency_ms: int
    confidence: float = 0.0
    error: Optional[str] = None

    def to_trace(self) -> Dict[str, Any]:
        return {
            "channel_name": self.channel_name,
            "channel_type": self.channel_type,
            "chunk_count": len(self.chunks),
            "latency_ms": self.latency_ms,
            "confidence": self.confidence,
            "error": self.error,
        }


class SearchChannel(Protocol):
    name: str
    channel_type: str
    priority: int

    def is_enabled(self, context: SearchContext) -> bool:
        ...

    def search(self, context: SearchContext) -> SearchChannelResult:
        ...


@dataclass
class FAQKeywordCandidate:
    entry: FAQEntry
    question_tokens: set[str]
    keyword_tokens: set[str]
    answer_tokens: set[str]


class FAQKeywordSearchChannel:
    """Lexical FAQ retrieval for patient-style questions.

    The local demo can use hashing embeddings, which are intentionally simple.
    This channel gives curated FAQ entries an exact/near-exact keyword recall
    path before broader Qdrant retrieval runs.
    """

    name = "faq_keyword_search"
    channel_type = "FAQ_KEYWORD"
    priority = 0

    def __init__(self, faq_directory: str = "data/faq", top_k_multiplier: int = 2):
        self.faq_directory = faq_directory
        self.top_k_multiplier = top_k_multiplier
        self.logger = logging.getLogger(__name__)
        self.candidates = self._load_candidates()
        self.token_idf = self._build_token_idf(self.candidates)

    def is_enabled(self, context: SearchContext) -> bool:
        return bool(self.candidates and _lexical_tokens(context.query))

    def search(self, context: SearchContext) -> SearchChannelResult:
        start = time.time()
        try:
            query_tokens = set(_lexical_tokens(context.query))
            query_compact = _compact_text(context.query)
            scored = []
            for candidate in self.candidates:
                score = self._score_candidate(candidate, query_tokens, query_compact)
                if score <= 0:
                    continue
                entry = candidate.entry
                metadata = entry.to_metadata()
                scored.append(
                    {
                        "id": f"faq-keyword:{entry.faq_id}",
                        "content": entry.to_chunk(),
                        "score": score,
                        "source": Path(entry.source_file).name,
                        "source_path": entry.source_file,
                        "retrieval_channel": self.name,
                        **metadata,
                    }
                )

            chunks = sorted(scored, key=lambda item: item.get("score", 0.0), reverse=True)
            chunks = chunks[: context.top_k * self.top_k_multiplier]
            return SearchChannelResult(
                channel_name=self.name,
                channel_type=self.channel_type,
                chunks=chunks,
                latency_ms=int((time.time() - start) * 1000),
                confidence=_average_score(chunks),
            )
        except Exception as exc:
            return SearchChannelResult(
                channel_name=self.name,
                channel_type=self.channel_type,
                chunks=[],
                latency_ms=int((time.time() - start) * 1000),
                error=str(exc),
            )

    def _load_candidates(self) -> List[FAQKeywordCandidate]:
        faq_path = Path(self.faq_directory)
        if not faq_path.exists():
            return []
        try:
            return [
                FAQKeywordCandidate(
                    entry=entry,
                    question_tokens=set(_lexical_tokens(entry.question)),
                    keyword_tokens=set(_lexical_tokens(entry.fields.get("keywords", ""))),
                    answer_tokens=set(_lexical_tokens(entry.answer)),
                )
                for entry in load_faq_directory(str(faq_path))
            ]
        except Exception as exc:
            self.logger.warning("FAQ keyword channel disabled: %s", exc)
            return []

    def _score_candidate(
        self,
        candidate: FAQKeywordCandidate,
        query_tokens: set[str],
        query_compact: str,
    ) -> float:
        question_hits = query_tokens & candidate.question_tokens
        keyword_hits = query_tokens & candidate.keyword_tokens
        answer_hits = query_tokens & candidate.answer_tokens
        if not question_hits and not keyword_hits:
            return 0.0

        question_compact = _compact_text(candidate.entry.question)
        exact_bonus = 0.25 if query_compact and (
            query_compact in question_compact or question_compact in query_compact
        ) else 0.0
        query_weight = max(self._weight(query_tokens), 1.0)
        question_score = self._weight(question_hits) / query_weight
        keyword_score = self._weight(keyword_hits) / query_weight
        answer_score = self._weight(answer_hits) / query_weight
        abbreviation_bonus = 0.0
        for token in query_tokens:
            if not _is_abbreviation_token(token):
                continue
            if token in candidate.question_tokens:
                abbreviation_bonus += 0.35
            elif token in candidate.keyword_tokens:
                abbreviation_bonus += 0.12
        base_score = 0.08
        lexical_score = min(
            question_score * 0.65
            + keyword_score * 0.45
            + answer_score * 0.08,
            0.85,
        )
        priority_score = {
            "P0": 0.03,
            "P1": 0.02,
            "P2": 0.01,
        }.get(str(candidate.entry.fields.get("priority", "")).upper(), 0.0)
        return base_score + lexical_score + exact_bonus + priority_score + min(abbreviation_bonus, 0.45)

    def _build_token_idf(self, candidates: List[FAQKeywordCandidate]) -> Dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for candidate in candidates:
            tokens = candidate.question_tokens | candidate.keyword_tokens
            document_frequency.update(tokens)
        total = len(candidates)
        return {
            token: math.log((total + 1) / (count + 1)) + 1.0
            for token, count in document_frequency.items()
        }

    def _weight(self, tokens: set[str]) -> float:
        return sum(self.token_idf.get(token, 1.0) for token in tokens)


class QdrantGlobalSearchChannel:
    name = "qdrant_global_search"
    channel_type = "VECTOR_HYBRID_GLOBAL"
    priority = 10

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def is_enabled(self, context: SearchContext) -> bool:
        return True

    def search(self, context: SearchContext) -> SearchChannelResult:
        start = time.time()
        try:
            chunks = self.vector_store.retrieve_relevant_chunks(
                query=context.query,
                vectorstore=context.vectorstore,
                docstore=context.docstore,
                top_k=context.top_k,
            )
            for chunk in chunks:
                chunk["retrieval_channel"] = self.name
            return SearchChannelResult(
                channel_name=self.name,
                channel_type=self.channel_type,
                chunks=chunks,
                latency_ms=int((time.time() - start) * 1000),
                confidence=_average_score(chunks),
            )
        except Exception as exc:
            return SearchChannelResult(
                channel_name=self.name,
                channel_type=self.channel_type,
                chunks=[],
                latency_ms=int((time.time() - start) * 1000),
                error=str(exc),
            )


class IntentDirectedMedicalSearchChannel:
    """A lightweight intent-directed channel.

    The current data model has a single Qdrant collection, so this channel
    narrows the global results by medical domain keywords and boosts matching
    chunks. It gives the architecture a real extension point for future
    per-domain collections or metadata filters.
    """

    name = "intent_directed_medical_search"
    channel_type = "INTENT_DIRECTED"
    priority = 1

    DOMAIN_TERMS = {
        "brain_tumor": ("brain", "tumor", "glioma", "mri"),
        "chest_xray": ("chest", "x-ray", "xray", "covid", "pneumonia"),
        "skin_lesion": ("skin", "lesion", "melanoma", "dermoscopy"),
        "diabetes": ("diabetes", "insulin", "glucose", "a1c"),
        "cardiovascular": (
            "heart",
            "cardiac",
            "chest pain",
            "stroke",
            "blood pressure",
            "hypertension",
            "cholesterol",
            "palpitations",
            "afib",
        ),
        "medication_safety": (
            "medicine",
            "medication",
            "prescription",
            "dose",
            "overdose",
            "antibiotic",
            "acetaminophen",
            "drug interaction",
        ),
    }

    def __init__(self, vector_store, top_k_multiplier: int = 2):
        self.vector_store = vector_store
        self.top_k_multiplier = top_k_multiplier

    def is_enabled(self, context: SearchContext) -> bool:
        return bool(self._infer_domains(context.query))

    def search(self, context: SearchContext) -> SearchChannelResult:
        start = time.time()
        try:
            domains = self._infer_domains(context.query)
            if not domains:
                return SearchChannelResult(
                    channel_name=self.name,
                    channel_type=self.channel_type,
                    chunks=[],
                    latency_ms=int((time.time() - start) * 1000),
                )

            chunks = self.vector_store.retrieve_relevant_chunks(
                query=context.query,
                vectorstore=context.vectorstore,
                docstore=context.docstore,
                top_k=context.top_k * self.top_k_multiplier,
            )
            filtered = self._filter_and_boost(chunks, domains)
            for chunk in filtered:
                chunk["retrieval_channel"] = self.name
                chunk["matched_domains"] = domains
            return SearchChannelResult(
                channel_name=self.name,
                channel_type=self.channel_type,
                chunks=filtered[: context.top_k * self.top_k_multiplier],
                latency_ms=int((time.time() - start) * 1000),
                confidence=_average_score(filtered),
            )
        except Exception as exc:
            return SearchChannelResult(
                channel_name=self.name,
                channel_type=self.channel_type,
                chunks=[],
                latency_ms=int((time.time() - start) * 1000),
                error=str(exc),
            )

    def _infer_domains(self, query: str) -> List[str]:
        normalized = (query or "").lower()
        domains = []
        for domain, terms in self.DOMAIN_TERMS.items():
            if any(term in normalized for term in terms):
                domains.append(domain)
        return domains

    def _filter_and_boost(
        self,
        chunks: List[Dict[str, Any]],
        domains: List[str],
    ) -> List[Dict[str, Any]]:
        domain_terms = [
            term
            for domain in domains
            for term in self.DOMAIN_TERMS.get(domain, ())
        ]
        filtered = []
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            source = chunk.get("source", "").lower()
            match_count = sum(1 for term in domain_terms if term in content or term in source)
            if match_count == 0:
                continue
            boosted = dict(chunk)
            boosted["score"] = float(boosted.get("score", 0.0)) + min(match_count * 0.05, 0.2)
            boosted["intent_match_count"] = match_count
            filtered.append(boosted)
        return sorted(filtered, key=lambda item: item.get("score", 0.0), reverse=True)


class MultiChannelRetrievalEngine:
    def __init__(self, channels: List[SearchChannel], postprocessors: List[Any]):
        self.channels = sorted(channels, key=lambda channel: channel.priority)
        self.postprocessors = sorted(postprocessors, key=lambda processor: processor.order)
        self.logger = logging.getLogger(__name__)

    def retrieve(self, context: SearchContext) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        enabled_channels = [
            channel for channel in self.channels
            if channel.is_enabled(context)
        ]
        channel_results: List[SearchChannelResult] = []

        for channel in enabled_channels:
            result = channel.search(context)
            channel_results.append(result)
            if result.error:
                self.logger.warning("Retrieval channel %s failed: %s", channel.name, result.error)

        chunks = [
            chunk
            for result in channel_results
            for chunk in result.chunks
        ]

        trace = {
            "channels": [result.to_trace() for result in channel_results],
            "postprocessors": [],
            "initial_chunk_count": len(chunks),
        }
        picture_paths: List[str] = []

        for processor in self.postprocessors:
            before_count = len(chunks)
            processed = processor.process(
                query=context.query,
                chunks=chunks,
                channel_results=channel_results,
            )
            if isinstance(processed, tuple):
                chunks, processor_picture_paths = processed
                picture_paths.extend(processor_picture_paths)
            else:
                chunks = processed
            trace["postprocessors"].append(
                {
                    "name": processor.name,
                    "before_count": before_count,
                    "after_count": len(chunks),
                }
            )

        trace["final_chunk_count"] = len(chunks)
        trace["picture_reference_count"] = len(picture_paths)
        return chunks, _dedupe_preserve_order(picture_paths), trace


def _average_score(chunks: List[Dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    scores = [float(chunk.get("score", 0.0)) for chunk in chunks[:3]]
    return sum(scores) / len(scores)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _lexical_tokens(text: str) -> List[str]:
    normalized = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        tokens.extend(
            sequence[index : index + size]
            for size in (1, 2, 3)
            for index in range(0, len(sequence) - size + 1)
        )
        if len(sequence) <= 8:
            tokens.append(sequence)
    return [
        token
        for token in tokens
        if token.strip() and token not in CHINESE_STOP_TOKENS
    ]


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _is_abbreviation_token(token: str) -> bool:
    return bool(re.search(r"[a-z]", token) and re.search(r"[0-9]", token)) or token in {
        "ai",
        "ct",
        "mri",
        "covid",
        "hba1c",
    }


CHINESE_STOP_TOKENS = {
    "是",
    "什",
    "么",
    "什么",
    "是什",
    "是什么",
    "意思",
    "意",
    "思",
    "吗",
    "能",
    "不",
    "不能",
    "可以",
    "应该",
    "需要",
    "要",
    "不要",
    "怎么",
    "哪些",
    "哪个",
    "一定",
    "就是",
    "有没有",
    "有",
    "没",
    "没有",
    "么意",
    "么意思",
    "什么意",
    "是什么意思",
    "诊",
    "断",
    "诊断",
}
