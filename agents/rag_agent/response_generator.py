import logging
import re
from collections import Counter, defaultdict
from typing import Any

from agents.context_builder import MedicalSystemPrompt

# Common English stopwords for keyword extraction
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "if",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "which",
        "who",
        "whom",
        "what",
        "where",
        "when",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "also",
        "just",
        "about",
        "up",
        "down",
        "here",
        "there",
        "very",
        "too",
        "any",
        "because",
        "while",
        "although",
        "however",
        "therefore",
        "thus",
        "e.g",
        "i.e",
        "fig",
        "figure",
        "table",
        "ref",
        "references",
        "et",
        "al",
        "pp",
        "vol",
        "method",
        "methods",
        "result",
        "results",
        "conclusion",
        "conclusions",
        "background",
        "objective",
        "objectives",
        "abstract",
        "discussion",
        "introduction",
        "materials",
        "supplementary",
        # Chinese common stopwords
        "的",
        "了",
        "在",
        "是",
        "和",
        "与",
        "及",
        "或",
        "等",
        "中",
        "对",
        "为",
        "以",
        "从",
        "到",
        "上",
        "下",
        "不",
        "有",
        "这",
        "那",
        "个",
        "被",
        "将",
        "把",
        "使",
        "让",
        "给",
        "用",
        "也",
        "都",
        "而",
        "但",
        "如",
        "所",
        "其",
        "该",
        "可",
        "会",
        "能",
        "已",
        "还",
        "更",
        "最",
        "较",
        "比",
        "并",
        "则",
        "即",
        "因",
        "由",
        "于",
        "之",
        "去",
        "来",
    }
)


class ResponseGenerator:
    """
    Generates responses based on retrieved context and user query.
    """

    def __init__(self, config):
        """
        Initialize the response generator.

        Args:
            config: Configuration object
            llm: Large language model for response generation
        """
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(self, query: str, context: str, chat_history: list[dict[str, str]] | None = None) -> str:
        """
        Build the prompt for the language model.

        Args:
            query: User query
            context: Formatted context from retrieved documents
            chat_history: Optional chat history

        Returns:
            Complete prompt string
        """

        # Use unified MedicalSystemPrompt.rag() for static system instructions (Phase 2)
        system_prompt = MedicalSystemPrompt.rag()

        # Build the prompt
        prompt = f"""{system_prompt}

        Here are the last few messages from our conversation:
        
        {chat_history}

        The user has asked the following question:
        {query}

        I've retrieved the following information to help answer this question:

        {context}

        Based on the provided information, please answer the user's question thoroughly but concisely.
        If the information doesn't contain the answer, acknowledge the limitations of the available information.

        Do not provide any source link that is not present in the context. Do not make up any source link.

        Medical Assistant Response:"""

        return prompt

    def generate_response(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        picture_paths: list[str],
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a response based on retrieved documents.

        Args:
            query: User query
            retrieved_docs: List of retrieved document dictionaries
            chat_history: Optional chat history

        Returns:
            Dict containing response text and source information
        """
        try:
            # Extract content from documents for context
            doc_texts = [doc["content"] for doc in retrieved_docs]

            # Combine retrieved documents into a single context
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)

            # Build the prompt
            prompt = self._build_prompt(query, context, chat_history)

            # Generate response
            response = self.response_generator_model.invoke(prompt)

            # Extract sources for citation
            sources = (
                self._extract_sources(retrieved_docs)
                if hasattr(self, "include_sources") and self.include_sources
                else []
            )

            # Calculate confidence
            confidence = self._calculate_confidence(retrieved_docs)

            # Add sources to response
            if hasattr(self, "include_sources") and self.include_sources:
                response_with_source = response.content + "\n\n##### Source documents:"
                for current_source in sources:
                    source_path = current_source["path"]
                    source_title = current_source["title"]
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = response.content

            # Add picture paths to response as inline images (Phase 6.3 UI optimization)
            if picture_paths:
                gallery_html = (
                    response_with_source
                    + "\n\n<details><summary>📷 Reference Images ("
                    + str(len(picture_paths))
                    + ')</summary>\n\n<div class="rag-ref-gallery">\n\n'
                )
                for picture_path in picture_paths:
                    img_name = picture_path.split("/")[-1]
                    # Use raw HTML <img> since marked.js won't parse markdown inside <div>
                    gallery_html += f'<img src="{picture_path}" alt="{img_name}" loading="lazy">\n\n'
                gallery_html += "</div>\n\n</details>"
                response_with_source_and_picture_paths = gallery_html
            else:
                response_with_source_and_picture_paths = response_with_source

            # Add cross-document references (Phase 6.4)
            cross_ref_html = self._build_cross_references(retrieved_docs)
            if cross_ref_html:
                response_with_source_and_picture_paths += cross_ref_html

            # Format final response
            result = {"response": response_with_source_and_picture_paths, "sources": sources, "confidence": confidence}

            return result

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0,
            }

    def _build_cross_references(self, retrieved_docs: list[dict[str, Any]]) -> str:
        """
        Analyze retrieved documents across different source papers and build
        cross-reference suggestions when multiple papers discuss related topics.

        Args:
            retrieved_docs: List of retrieved document dicts with 'source' and 'content'

        Returns:
            HTML string for cross-reference section, or empty string if <2 sources
        """
        # Group docs by source
        source_groups = defaultdict(list)
        for doc in retrieved_docs:
            source = doc.get("source", "unknown")
            source_groups[source].append(doc)

        # Need at least 2 different sources for cross-references
        if len(source_groups) < 2:
            return ""

        # Extract keywords per source
        source_keywords = {}
        for source, docs in source_groups.items():
            all_text = " ".join(d["content"] for d in docs)
            keywords = self._extract_keywords(all_text)
            source_keywords[source] = keywords

        # Find cross-reference pairs with shared keywords
        sources = list(source_groups.keys())
        cross_refs = []
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                src_a, src_b = sources[i], sources[j]
                kw_a = source_keywords[src_a]
                kw_b = source_keywords[src_b]
                shared = set(kw_a.keys()) & set(kw_b.keys())
                if shared:
                    # Rank by keyword importance (sum of frequencies)
                    shared_ranked = sorted(
                        shared,
                        key=lambda k: source_keywords[src_a].get(k, 0) + source_keywords[src_b].get(k, 0),
                        reverse=True,
                    )[:5]
                    cross_refs.append(
                        {
                            "src_a": src_a,
                            "src_b": src_b,
                            "shared_keywords": shared_ranked,
                            "count_a": len(source_groups[src_a]),
                            "count_b": len(source_groups[src_b]),
                            "strength": len(shared),
                        }
                    )

        if not cross_refs:
            return ""

        # Sort by strength (most shared keywords first)
        cross_refs.sort(key=lambda x: x["strength"], reverse=True)

        # Build HTML
        html = "\n\n<details><summary>🔗 Cross-References (" + str(len(cross_refs)) + ")</summary>\n\n"
        html += '<div class="rag-cross-ref">\n\n'
        for ref in cross_refs:
            title_a = ref["src_a"].replace(".pdf", "").replace(".md", "")
            title_b = ref["src_b"].replace(".pdf", "").replace(".md", "")
            keywords_str = ", ".join(ref["shared_keywords"])
            html += '<div class="rag-cross-ref-item">\n'
            html += f'  <span class="rag-cross-ref-paper">📄 {title_a}</span>\n'
            html += '  <span class="rag-cross-ref-arrow">⟷</span>\n'
            html += f'  <span class="rag-cross-ref-paper">📄 {title_b}</span>\n'
            html += f'  <span class="rag-cross-ref-kw">Shared: {keywords_str}</span>\n'
            html += "</div>\n\n"
        html += "</div>\n\n</details>"

        return html

    def _extract_keywords(self, text: str, top_n: int = 30) -> dict:
        """
        Extract top keywords from text using word frequency.
        Returns dict of {word: count} for keywords above threshold.

        Args:
            text: Input text
            top_n: Number of top keywords to return

        Returns:
            Dict mapping keyword to frequency count
        """
        # Tokenize: split on non-alpha chars, lowercase
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        # Filter stopwords and short words
        filtered = [w for w in words if w not in _STOPWORDS and len(w) >= 3]
        counts = Counter(filtered)
        # Return top N keywords
        return dict(counts.most_common(top_n))

    def _extract_sources(self, documents: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Extract source information from retrieved documents for citation.

        Args:
            documents: List of retrieved document dictionaries

        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()  # Track unique sources to avoid duplicates

        for doc in documents:
            # Extract source and source_path
            source = doc.get("source")
            source_path = doc.get("source_path")

            # Skip if no source information is available
            if not source:
                continue

            # Create a unique identifier for this source
            source_id = f"{source}|{source_path}"

            # Skip if we've already included this source
            if source_id in seen_sources:
                continue

            # Add to our sources list
            source_info = {
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0))),
            }

            sources.append(source_info)
            seen_sources.add(source_id)

        # Sort sources by score from highest to lowest
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Format the final sources list, removing the scores which were just used for sorting
        formatted_sources = []
        for source in sources:
            formatted_source = {"title": source["title"], "path": source["path"]}
            formatted_sources.append(formatted_source)

        return formatted_sources

    def _calculate_confidence(self, documents: list[dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved documents.

        Args:
            documents: Retrieved documents

        Returns:
            Confidence score between 0 and 1
        """
        if not documents:
            return 0.0

        # Use combined score (both reranker and cosine similarity) if available, otherwise use original score
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:3]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:3]]

        # Average of top 3 document scores or fewer if less than 3
        return sum(scores) / len(scores) if scores else 0.0
