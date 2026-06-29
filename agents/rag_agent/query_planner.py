from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class QueryPlan:
    original_query: str
    rewritten_query: str
    sub_questions: List[str]
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "sub_questions": self.sub_questions,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "trace": self.trace,
        }


class QueryPlanner:
    """Deterministic query rewrite, split, and clarification gate.

    Ragent uses an LLM-backed `rewriteWithSplit` stage. This project keeps the
    same shape but starts with bounded rules so local demos and tests do not
    depend on an external model call before retrieval.
    """

    def __init__(self, max_sub_questions: int = 3):
        self.max_sub_questions = max(1, max_sub_questions)

    def plan(
        self,
        query: str,
        chat_history: Optional[Any] = None,
    ) -> QueryPlan:
        original_query = (query or "").strip()
        history_text = _history_to_text(chat_history)
        trace: List[Dict[str, Any]] = []

        normalized_query = _normalize_medical_terms(original_query)
        if normalized_query != original_query:
            trace.append(
                {
                    "stage": "term_normalization",
                    "before": original_query,
                    "after": normalized_query,
                }
            )

        if _needs_clarification(normalized_query, history_text):
            clarification = (
                "请补充更具体的信息，例如相关疾病或检查名称、主要症状、持续时间、"
                "年龄段，以及你希望了解的是原因、风险、检查还是处理建议。"
            )
            trace.append(
                {
                    "stage": "clarification_gate",
                    "reason": "question_is_too_short_or_context_dependent",
                }
            )
            return QueryPlan(
                original_query=original_query,
                rewritten_query=normalized_query,
                sub_questions=[],
                needs_clarification=True,
                clarification_question=clarification,
                trace=trace,
            )

        rewritten_query = _complete_follow_up(normalized_query, history_text)
        if rewritten_query != normalized_query:
            trace.append(
                {
                    "stage": "context_completion",
                    "before": normalized_query,
                    "after": rewritten_query,
                }
            )

        sub_questions = _split_sub_questions(rewritten_query, self.max_sub_questions)
        if not sub_questions:
            sub_questions = [rewritten_query]
        trace.append(
            {
                "stage": "question_split",
                "sub_question_count": len(sub_questions),
                "sub_questions": sub_questions,
            }
        )

        return QueryPlan(
            original_query=original_query,
            rewritten_query=rewritten_query,
            sub_questions=sub_questions,
            trace=trace,
        )


def _history_to_text(chat_history: Optional[Any]) -> str:
    if not chat_history:
        return ""
    if isinstance(chat_history, str):
        return chat_history.strip()
    if isinstance(chat_history, Sequence):
        lines = []
        for item in chat_history:
            if isinstance(item, dict):
                role = item.get("role") or item.get("type") or "message"
                content = item.get("content") or item.get("text") or ""
                if content:
                    lines.append(f"{role}: {content}")
            elif hasattr(item, "content"):
                role = item.__class__.__name__.replace("Message", "").lower() or "message"
                lines.append(f"{role}: {item.content}")
            elif item:
                lines.append(str(item))
        return "\n".join(lines).strip()
    return str(chat_history).strip()


def _normalize_medical_terms(query: str) -> str:
    normalized = " ".join((query or "").split())
    replacements = {
        r"\bcovid19\b": "COVID-19",
        r"\bcovid 19\b": "COVID-19",
        r"\bxray\b": "X-ray",
        r"\bx ray\b": "X-ray",
        r"\bhba1c\b": "HbA1c",
        r"\bmri\b": "MRI",
        r"\bct\b": "CT",
    }
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def _needs_clarification(query: str, history_text: str) -> bool:
    compact = re.sub(r"\s+", "", query or "").lower()
    if not compact:
        return True
    if history_text.strip():
        return False

    vague_exact = {
        "怎么办",
        "咋办",
        "严重吗",
        "要吃药吗",
        "能好吗",
        "这个怎么办",
        "这个严重吗",
        "what should i do",
        "is it serious",
        "should i take medicine",
        "what about this",
    }
    if compact in {item.replace(" ", "") for item in vague_exact}:
        return True

    has_vague_pointer = any(
        pointer in compact
        for pointer in ("这个", "这", "它", "这样", "it", "this", "that")
    )
    has_medical_anchor = any(
        anchor in compact
        for anchor in (
            "tumor",
            "brain",
            "mri",
            "covid",
            "x-ray",
            "xray",
            "diabetes",
            "glucose",
            "skin",
            "lesion",
            "fever",
            "cough",
            "疼",
            "痛",
            "发烧",
            "咳嗽",
            "糖尿病",
            "肿瘤",
            "皮肤",
            "胸片",
        )
    )
    return has_vague_pointer and not has_medical_anchor and len(compact) <= 16


def _complete_follow_up(query: str, history_text: str) -> str:
    if not history_text.strip():
        return query
    if not _is_follow_up(query):
        return query

    topic = _extract_recent_topic(history_text)
    if not topic:
        return query
    return f"{topic}: {query}"


def _is_follow_up(query: str) -> bool:
    normalized = (query or "").strip().lower()
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) > 40:
        return False
    return any(
        marker in compact
        for marker in (
            "这个",
            "它",
            "这些",
            "这种",
            "刚才",
            "上面",
            "严重吗",
            "怎么办",
            "it",
            "this",
            "that",
            "them",
            "those",
        )
    )


def _extract_recent_topic(history_text: str) -> str:
    lowered = history_text.lower()
    topics = [
        ("brain tumor MRI", ("brain tumor", "glioma", "mri", "脑肿瘤", "胶质瘤")),
        ("COVID-19 chest X-ray", ("covid", "chest x-ray", "x-ray", "xray", "胸片")),
        ("diabetes and blood glucose", ("diabetes", "glucose", "hba1c", "insulin", "糖尿病", "血糖")),
        ("skin lesion", ("skin lesion", "melanoma", "dermoscopy", "皮肤", "黑色素瘤")),
        ("symptom triage", ("fever", "cough", "headache", "pain", "发烧", "咳嗽", "头痛")),
    ]
    matched = [label for label, terms in topics if any(term in lowered for term in terms)]
    return ", ".join(dict.fromkeys(matched[:2]))


def _split_sub_questions(query: str, limit: int) -> List[str]:
    candidates: List[str] = []
    for clause in re.split(r"[?？;；。！!\n]+", query):
        clause = clause.strip(" ,，")
        if clause:
            candidates.extend(_split_clause(clause))

    if len(candidates) == 1:
        candidates = _split_clause(candidates[0], aggressive=True)

    deduped = _dedupe_preserve_order(_ensure_question(item) for item in candidates if item.strip())
    return deduped[:limit]


def _split_clause(clause: str, aggressive: bool = False) -> List[str]:
    normalized = clause.strip()
    if not normalized:
        return []

    separators = (
        r"\s+(?:and|also|plus)\s+",
        r"\s*,\s+and\s+",
        r"(?:以及|并且|同时|另外|还有|再说|分别)",
    )
    should_split = aggressive or any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in separators)
    if not should_split:
        return [normalized]

    parts: List[str] = [normalized]
    for pattern in separators:
        next_parts: List[str] = []
        for part in parts:
            split_items = [
                item.strip(" ,，")
                for item in re.split(pattern, part, flags=re.IGNORECASE)
                if item.strip(" ,，")
            ]
            if len(split_items) > 1 and all(len(item) >= 3 for item in split_items):
                next_parts.extend(split_items)
            else:
                next_parts.append(part)
        parts = next_parts

    if len(parts) <= 1:
        return [normalized]

    anchor = _infer_anchor(normalized)
    anchored_parts = []
    for part in parts:
        if anchor and not _contains_anchor(part, anchor):
            anchored_parts.append(f"{anchor} {part}")
        else:
            anchored_parts.append(part)
    return anchored_parts


def _infer_anchor(text: str) -> str:
    anchors = (
        "diabetes",
        "COVID-19",
        "chest X-ray",
        "brain tumor",
        "MRI",
        "skin lesion",
        "糖尿病",
        "脑肿瘤",
        "胸片",
        "皮肤病灶",
    )
    lowered = text.lower()
    for anchor in anchors:
        if anchor.lower() in lowered:
            return anchor
    return ""


def _contains_anchor(text: str, anchor: str) -> bool:
    return anchor.lower() in (text or "").lower()


def _ensure_question(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped.endswith(("?", "？")):
        return stripped
    return stripped + "?"


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    results = []
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results
