"""
Hybrid Search Module - BM25 + Vector Search with RRF Fusion (Phase 6)

Combines:
1. BM25 keyword search (rank_bm25) for exact term matching
2. Qdrant dense vector search for semantic similarity
3. Reciprocal Rank Fusion (RRF) to merge results

This provides a search layer on top of existing vectorstore that adds
keyword-based retrieval capability, improving recall for medical terminology.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 keyword index for fast term-level document retrieval.
    Builds and maintains an in-memory BM25 index from document corpus.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        Lowercases, splits on non-alphanumeric, keeps medical terms intact.
        """
        text = text.lower()
        # Split on non-alphanumeric, keep hyphens in compound medical terms
        tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
        return [t for t in tokens if len(t) > 1]  # Filter single chars

    def build_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index from a list of LangChain Documents.
        
        Args:
            documents: List of Document objects with page_content
        """
        self._documents = documents
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._built = True
        self.logger.info(f"[BM25] Index built with {len(documents)} documents")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing index (rebuilds)."""
        self._documents.extend(documents)
        self._tokenized_corpus.extend(
            [self._tokenize(doc.page_content) for doc in documents]
        )
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self.logger.info(f"[BM25] Added {len(documents)} docs, total: {len(self._documents)}")

    def remove_documents_by_source(self, source: str) -> int:
        """Remove all documents from a given source file. Returns count removed."""
        before = len(self._documents)
        filtered = [
            (doc, tokens) for doc, tokens in zip(self._documents, self._tokenized_corpus)
            if doc.metadata.get("source") != source
        ]
        if filtered:
            self._documents, self._tokenized_corpus = zip(*filtered)
            self._documents = list(self._documents)
            self._tokenized_corpus = list(self._tokenized_corpus)
        else:
            self._documents, self._tokenized_corpus = [], []
        removed = before - len(self._documents)
        if removed > 0:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        self.logger.info(f"[BM25] Removed {removed} docs from source: {source}")
        return removed

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        Search index with BM25.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (Document, bm25_score) tuples, sorted by score desc
        """
        if not self._built or self._bm25 is None:
            self.logger.warning("[BM25] Index not built, returning empty results")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._documents[idx], float(scores[idx])))

        self.logger.info(f"[BM25] Query '{query[:50]}...' returned {len(results)} results")
        return results

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def doc_count(self) -> int:
        return len(self._documents)


class HybridSearch:
    """
    Hybrid search combining BM25 keyword search with Qdrant vector search,
    fused using Reciprocal Rank Fusion (RRF).
    
    Architecture:
    - BM25Index: In-memory keyword index (built from docstore)
    - QdrantVectorStore: Dense+Sparse vector search (existing)
    - RRF: score(d) = sum(1 / (k + rank_i(d))) for each retrieval method
    """

    def __init__(
        self,
        vectorstore=None,
        docstore=None,
        reranker=None,
        rrf_k: int = 60,
    ):
        """
        Args:
            vectorstore: QdrantVectorStore instance for dense vector search
            docstore: LocalFileStore for loading documents into BM25
            reranker: Optional Reranker instance for cross-encoder reranking
            rrf_k: RRF constant (default 60, standard value from literature)
        """
        self.logger = logging.getLogger(__name__)
        self.bm25_index = BM25Index()
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.reranker = reranker
        self.rrf_k = rrf_k

    def _load_docs_from_docstore(self) -> List[Document]:
        """Load all documents from docstore into BM25 index."""
        if self.docstore is None:
            self.logger.warning("[HYBRID] No docstore available for BM25 indexing")
            return []

        documents = []
        try:
            # LocalFileStore yields (key, value) pairs
            for key, value in self.docstore.yield_keys():
                content = self.docstore.mget([key])
                if content and content[0]:
                    text = content[0].decode("utf-8") if isinstance(content[0], bytes) else str(content[0])
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": key.decode("utf-8") if isinstance(key, bytes) else key}
                    ))
        except Exception as e:
            self.logger.warning(f"[HYBRID] Failed to load docs from docstore: {e}")

        return documents

    def build_bm25_from_docstore(self) -> None:
        """Build BM25 index from documents stored in docstore."""
        documents = self._load_docs_from_docstore()
        if documents:
            self.bm25_index.build_index(documents)
            self.logger.info(f"[HYBRID] BM25 index built from docstore: {len(documents)} docs")
        else:
            self.logger.warning("[HYBRID] No documents found in docstore for BM25")

    def build_bm25_from_documents(self, documents: List[Document]) -> None:
        """Build BM25 index from a provided list of documents."""
        self.bm25_index.build_index(documents)

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        top_k: int = 10,
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """
        Fuse vector and BM25 results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list where the doc appears.
        
        Args:
            vector_results: (Document, vector_score) tuples, sorted by score desc
            bm25_results: (Document, bm25_score) tuples, sorted by score desc
            rrf_k: RRF constant
            top_k: Final number of results
            
        Returns:
            List of (Document, rrf_score, debug_info) tuples
        """
        # Map documents by a content hash for deduplication
        doc_scores: Dict[str, Dict[str, Any]] = {}

        def _doc_key(doc: Document) -> str:
            """Generate a dedup key from document content and source."""
            source = doc.metadata.get("source", "")
            # Use first 100 chars + source as dedup key
            return f"{source}::{doc.page_content[:100]}"

        # Score vector results by rank
        for rank, (doc, v_score) in enumerate(vector_results):
            key = _doc_key(doc)
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None}
            doc_scores[key]["rrf_score"] += 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key]["vector_rank"] = rank + 1
            doc_scores[key]["vector_score"] = v_score

        # Score BM25 results by rank
        for rank, (doc, b_score) in enumerate(bm25_results):
            key = _doc_key(doc)
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None}
            doc_scores[key]["rrf_score"] += 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key]["bm25_rank"] = rank + 1
            doc_scores[key]["bm25_score"] = b_score

        # Sort by RRF score descending
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]

        results = []
        for item in sorted_results:
            debug = {
                "rrf_score": item["rrf_score"],
                "vector_rank": item.get("vector_rank"),
                "bm25_rank": item.get("bm25_rank"),
                "vector_score": item.get("vector_score"),
                "bm25_score": item.get("bm25_score"),
            }
            results.append((item["doc"], item["rrf_score"], debug))

        return results

    async def search(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        use_reranker: bool = True,
        parsed_content_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search: BM25 + Vector with RRF fusion.
        
        Args:
            query: Search query
            top_k: Final number of results after fusion
            vector_top_k: Number of vector results to fetch
            bm25_top_k: Number of BM25 results to fetch
            use_reranker: Whether to apply cross-encoder reranking after fusion
            parsed_content_dir: Path to parsed content for reranker
            
        Returns:
            List of result dicts with content, score, source, and debug info
        """
        vector_results = []
        bm25_results = []

        # Vector search
        if self.vectorstore is not None:
            try:
                docs = await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, k=vector_top_k
                )
                vector_results = docs
                self.logger.info(f"[HYBRID] Vector search returned {len(vector_results)} results")
            except Exception as e:
                self.logger.warning(f"[HYBRID] Vector search failed: {e}")

        # BM25 search
        if self.bm25_index.is_built:
            bm25_results = self.bm25_index.search(query, top_k=bm25_top_k)
            self.logger.info(f"[HYBRID] BM25 search returned {len(bm25_results)} results")

        # If only one method available, use its results directly
        if not vector_results and not bm25_results:
            self.logger.warning("[HYBRID] Both search methods returned empty")
            return []

        if not vector_results:
            self.logger.info("[HYBRID] Vector unavailable, using BM25 only")
            results = [
                {"content": doc.page_content, "score": score, "source": doc.metadata.get("source", ""),
                 "fusion": "bm25_only"}
                for doc, score in bm25_results[:top_k]
            ]
            return results

        if not bm25_results:
            self.logger.info("[HYBRID] BM25 unavailable, using vector only")
            results = [
                {"content": doc.page_content, "score": score, "source": doc.metadata.get("source", ""),
                 "fusion": "vector_only"}
                for doc, score in vector_results[:top_k]
            ]
            return results

        # RRF Fusion
        fused = self._rrf_fusion(vector_results, bm25_results, top_k=top_k)
        self.logger.info(f"[HYBRID] RRF fusion returned {len(fused)} results")

        # Optional reranking
        if use_reranker and self.reranker is not None and parsed_content_dir:
            try:
                docs_for_rerank = [doc for doc, _, _ in fused]
                reranked = self.reranker.rerank(query, docs_for_rerank, parsed_content_dir)
                if reranked:
                    return reranked
            except Exception as e:
                self.logger.warning(f"[HYBRID] Reranking failed, using RRF results: {e}")

        # Format results
        results = []
        for doc, rrf_score, debug in fused:
            results.append({
                "content": doc.page_content,
                "score": rrf_score,
                "source": doc.metadata.get("source", ""),
                "fusion": "rrf",
                "debug": debug,
            })

        return results

    def incremental_update(self, source: str, new_documents: List[Document]) -> None:
        """
        Incrementally update BM25 index for a specific source file.
        Removes old entries for the source and adds new ones.
        
        Args:
            source: Source file identifier (e.g., filename)
            new_documents: New Document objects to index
        """
        self.bm25_index.remove_documents_by_source(source)
        if new_documents:
            self.bm25_index.add_documents(new_documents)
        self.logger.info(f"[HYBRID] Incremental update for '{source}': {len(new_documents)} new docs")
