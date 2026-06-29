from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .medical_intents import (
    INTENT_DEFINITIONS,
    IntentDefinition,
    IntentKind,
    MedicalIntent,
    RouteDecision,
)


IMAGE_TYPE_TO_INTENT = {
    "CHEST X-RAY": MedicalIntent.CHEST_XRAY_ANALYSIS,
    "BRAIN MRI SCAN": MedicalIntent.BRAIN_MRI_ANALYSIS,
    "SKIN LESION": MedicalIntent.SKIN_LESION_SEGMENTATION,
    "OTHER": MedicalIntent.UNSUPPORTED_OR_UNCLEAR_IMAGE,
    "NON-MEDICAL": MedicalIntent.UNSUPPORTED_OR_UNCLEAR_IMAGE,
}


class IntentRouter:
    """Rule-first medical intent router inspired by enterprise intent trees.

    The current implementation is deterministic so it can run in tests without
    a model API. A later LLM classifier can reuse the same RouteDecision schema.
    """

    def __init__(
        self,
        definitions: Optional[Dict[MedicalIntent, IntentDefinition]] = None,
        min_confidence: float = 0.35,
    ):
        self.definitions = definitions or INTENT_DEFINITIONS
        self.min_confidence = min_confidence

    def classify(
        self,
        text: str = "",
        *,
        has_image: bool = False,
        image_type: Optional[str] = None,
    ) -> RouteDecision:
        normalized = (text or "").strip().lower()

        if has_image:
            return self._classify_image(normalized, image_type)

        safety_decision = self._match_safety(normalized)
        if safety_decision:
            return safety_decision

        if not normalized:
            return self._decision(
                MedicalIntent.GENERAL_CHAT,
                0.72,
                "Empty or conversational input defaults to the conversation agent.",
            )

        system_decision = self._match_system(normalized)
        if system_decision:
            return system_decision

        scored = self._score_all(normalized)
        if not scored:
            return self._decision(
                MedicalIntent.GENERAL_CHAT,
                0.5,
                "No medical intent matched strongly enough; use the conversation agent.",
            )

        top_intent, top_score, matched = scored[0]

        # Time-sensitive wording should prefer web search even if it also matches literature.
        latest_keywords = self.definitions[MedicalIntent.LATEST_MEDICAL_RESEARCH].keywords
        latest_matches = self._matching_keywords(normalized, latest_keywords)
        if latest_matches:
            return self._decision(
                MedicalIntent.LATEST_MEDICAL_RESEARCH,
                max(0.78, top_score),
                "The question contains time-sensitive wording, so web search is safer than local-only RAG.",
                latest_matches,
            )

        if top_score < self.min_confidence:
            return self._decision(
                MedicalIntent.GENERAL_CHAT,
                top_score,
                "Intent score is below threshold; use conversation flow or ask for clarification.",
                matched,
                needs_clarification=True,
            )

        return self._decision(
            top_intent,
            top_score,
            f"Matched medical intent {top_intent.value} from user wording.",
            matched,
        )

    def _classify_image(self, normalized_text: str, image_type: Optional[str]) -> RouteDecision:
        image_type_key = (image_type or "").strip().upper()
        intent = IMAGE_TYPE_TO_INTENT.get(image_type_key)
        if not intent:
            scored = self._score_all(normalized_text)
            for candidate, score, matched in scored:
                if self.definitions[candidate].kind == IntentKind.VISION:
                    return self._decision(
                        candidate,
                        max(score, 0.64),
                        "Image is present and text points to a supported vision task.",
                        matched,
                    )
            intent = MedicalIntent.UNSUPPORTED_OR_UNCLEAR_IMAGE

        confidence = 0.9 if intent != MedicalIntent.UNSUPPORTED_OR_UNCLEAR_IMAGE else 0.55
        reason = (
            f"Uploaded image was classified as {image_type_key}."
            if image_type_key
            else "Image is present but type is unknown."
        )
        return self._decision(intent, confidence, reason)

    def _match_safety(self, normalized: str) -> Optional[RouteDecision]:
        for intent in (
            MedicalIntent.PROMPT_INJECTION_OR_DATA_EXFILTRATION,
            MedicalIntent.EMERGENCY_OR_SELF_HARM,
            MedicalIntent.PRESCRIPTION_OR_DOSAGE_REQUEST,
            MedicalIntent.DIAGNOSIS_OVERCLAIM,
        ):
            definition = self.definitions[intent]
            matched = self._matching_keywords(normalized, definition.keywords)
            if matched:
                return self._decision(
                    intent,
                    0.95,
                    f"Safety-sensitive wording matched {intent.value}.",
                    matched,
                )
        return None

    def _match_system(self, normalized: str) -> Optional[RouteDecision]:
        for intent in (
            MedicalIntent.CAPABILITY_QUESTION,
            MedicalIntent.OUT_OF_SCOPE,
            MedicalIntent.GENERAL_CHAT,
        ):
            definition = self.definitions[intent]
            matched = self._matching_keywords(normalized, definition.keywords)
            if matched:
                return self._decision(
                    intent,
                    0.78 if intent != MedicalIntent.GENERAL_CHAT else 0.72,
                    f"System-level wording matched {intent.value}.",
                    matched,
                )
        return None

    def _score_all(self, normalized: str) -> List[tuple[MedicalIntent, float, List[str]]]:
        scores: List[tuple[MedicalIntent, float, List[str]]] = []
        has_vision_request = self._has_vision_request(normalized)
        for intent, definition in self.definitions.items():
            if definition.kind == IntentKind.SAFETY:
                continue
            if definition.kind == IntentKind.VISION and not has_vision_request:
                continue
            matched = self._matching_keywords(normalized, definition.keywords)
            if not matched:
                continue
            base = 0.35 + min(len(matched) * 0.12, 0.45)
            if definition.kind == IntentKind.KB:
                base += 0.05
            if definition.kind == IntentKind.VISION:
                base += 0.06
            scores.append((intent, min(base, 0.94), matched))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def _has_vision_request(self, text: str) -> bool:
        negations = (
            "without an uploaded image",
            "without analyzing an uploaded image",
            "without uploaded image",
            "no uploaded image",
            "text only",
            "do not analyze an image",
            "don't analyze an image",
        )
        if any(negation in text for negation in negations):
            return False

        action_words = (
            "analyze",
            "detect",
            "segment",
            "classify",
            "upload",
            "image",
            "photo",
            "picture",
            "scan",
        )
        modality_words = (
            "x-ray",
            "xray",
            "mri",
            "lesion",
            "dermoscopy",
            "medical image",
        )
        has_action = any(word in text for word in action_words)
        has_modality = any(word in text for word in modality_words)
        has_image_reference = any(
            phrase in text
            for phrase in (
                "this image",
                "this photo",
                "this picture",
                "this scan",
                "uploaded image",
                "upload an image",
            )
        )
        return has_action and has_modality and has_image_reference

    def _matching_keywords(self, text: str, keywords: Iterable[str]) -> List[str]:
        return [keyword for keyword in keywords if keyword and keyword.lower() in text]

    def _decision(
        self,
        intent: MedicalIntent,
        confidence: float,
        reason: str,
        matched_keywords: Optional[List[str]] = None,
        *,
        needs_clarification: bool = False,
    ) -> RouteDecision:
        definition = self.definitions[intent]
        return RouteDecision(
            intent=intent,
            kind=definition.kind,
            agent_name=definition.agent_name,
            confidence=round(confidence, 3),
            reason=reason,
            tool_name=definition.tool_name,
            needs_clarification=needs_clarification,
            requires_human_validation=definition.requires_human_validation,
            matched_keywords=matched_keywords or [],
        )
