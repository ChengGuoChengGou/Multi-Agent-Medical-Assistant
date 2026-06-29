from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsageEstimate:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class HeuristicTokenCounter:
    """Cheap token estimator used when provider usage metadata is unavailable."""

    def count_text(self, value: Any) -> int:
        text = self._to_text(value)
        # A rough English-heavy estimate. Good enough for trend tracking.
        return max(1, len(text) // 4) if text else 0

    def estimate(self, prompt: Any, response: Any) -> TokenUsageEstimate:
        prompt_tokens = self.count_text(prompt)
        completion_tokens = self.count_text(response)
        return TokenUsageEstimate(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if hasattr(value, "content"):
            return str(value.content)
        return str(value)
