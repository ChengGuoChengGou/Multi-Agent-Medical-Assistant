import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings


class HashingEmbeddings(Embeddings):
    """Deterministic local embedding fallback.

    This is not a replacement for a production embedding model. It keeps local
    demos and ingestion runnable when an OpenAI-compatible gateway does not
    provide an embeddings endpoint. The vector dimension intentionally matches
    OpenAI's `text-embedding-3-small` default dimension used by this project.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
