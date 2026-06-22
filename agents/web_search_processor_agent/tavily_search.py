import os
import logging

logger = logging.getLogger(__name__)

class TavilySearchAgent:
    """
    Web search agent with Tavily (paid) + DuckDuckGo (free fallback).
    """

    def __init__(self):
        self._tavily_tool = None
        self._ddgs_available = False

        # Try Tavily if API key is present
        tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if tavily_key and not tavily_key.startswith("#"):
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults
                self._tavily_tool = TavilySearchResults(max_results=5)
                logger.info("[WebSearch] Tavily search initialized")
            except Exception as e:
                logger.warning(f"[WebSearch] Tavily init failed: {e}")

        # Check DuckDuckGo availability as fallback
        try:
            from duckduckgo_search import DDGS
            self._ddgs_available = True
            logger.info("[WebSearch] DuckDuckGo fallback available")
        except ImportError:
            logger.warning("[WebSearch] duckduckgo-search not installed, no fallback")

    def search_tavily(self, query: str) -> str:
        """Perform a web search. Tries Tavily first, falls back to DuckDuckGo."""
        query = query.strip('"\'').strip()

        # Try Tavily first
        if self._tavily_tool:
            try:
                search_docs = self._tavily_tool.invoke(query)
                if search_docs:
                    return "\n".join([
                        f"title: {res.get('title','')} - "
                        f"url: {res.get('url','')} - "
                        f"content: {res.get('content','')} - "
                        f"score: {res.get('score','')}"
                        for res in search_docs
                    ])
                return "No relevant results found."
            except Exception as e:
                logger.warning(f"[WebSearch] Tavily search failed, trying fallback: {e}")

        # Fallback: DuckDuckGo
        if self._ddgs_available:
            try:
                try:
                    from ddgs import DDGS
                except ImportError:
                    from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))
                if results:
                    return "\n".join([
                        f"title: {r.get('title','')} - "
                        f"url: {r.get('href','')} - "
                        f"content: {r.get('body','')}"
                        for r in results
                    ])
                return "No relevant results found."
            except Exception as e:
                logger.error(f"[WebSearch] DuckDuckGo search failed: {e}")
                return f"Error retrieving web search results: {e}"

        return "Error: No search backend available. Install duckduckgo-search or set TAVILY_API_KEY."
