"""Dependency Injection container for Multi-Agent Medical Chatbot.
Phase 30: Centralized service access for testability.
"""

# Forward references to avoid circular imports
_llm_service = None
_mcp_client = None
_cache_service = None


def get_llm_service():
    """Get LLM service instance (injected at startup)."""
    if _llm_service is None:
        from llm_factory import get_llm

        return get_llm()
    return _llm_service


def get_mcp_client():
    """Get MCP client instance (injected at startup)."""
    return _mcp_client


def set_mcp_client(client):
    """Set MCP client during app startup."""
    global _mcp_client
    _mcp_client = client


def get_cache_service():
    """Get cache service instance."""
    if _cache_service is None:
        from cache import cache_get, cache_set

        return {"get": cache_get, "set": cache_set}
    return _cache_service
