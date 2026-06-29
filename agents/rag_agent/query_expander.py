import logging
from typing import List, Dict, Any

from core import ModelGateway, PromptRegistry

class QueryExpander:
    """
    Expands user queries with medical terminology to improve retrieval.
    """
    def __init__(self, config):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm
        self.model_gateway = ModelGateway()
        self.prompt_registry = PromptRegistry()
        
    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        Expand the original query with relevant medical terms.
        
        Args:
            original_query: The user's original query
            
        Returns:
            Dictionary with original and expanded queries
        """
        self.logger.info(f"Expanding query: {original_query}")
        
        # Generate expansions - implement one of the strategies below
        expanded_query = self._generate_expansions(original_query)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }
    
    def _generate_expansions(self, query: str) -> str:
        """Use LLM to expand query with medical terminology."""
        prompt_template = self.prompt_registry.get("rag.query_expansion")
        prompt = prompt_template.render(query=query)
        expansion = self.model_gateway.invoke(
            model=self.model,
            prompt=prompt,
            model_id="rag-query-expander",
            metadata={
                "stage": "query_expansion",
                **prompt_template.metadata(),
            },
        ).output
        
        return expansion
