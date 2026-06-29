"""Lightweight RAG package exports.

Importing this package should not load Docling, Qdrant, or model clients. The
`MedicalRAG` name below is a compatibility proxy for the original
`from agents.rag_agent import MedicalRAG` usage; the heavy implementation is
loaded only when the class is instantiated.
"""


class MedicalRAG:
    def __new__(cls, *args, **kwargs):
        from .medical_rag import MedicalRAG as MedicalRAGImpl

        return MedicalRAGImpl(*args, **kwargs)


__all__ = ["MedicalRAG"]
