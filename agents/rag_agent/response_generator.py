import logging
from typing import List, Dict, Any, Optional, Union

from core import ModelGateway, PromptRegistry

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
        self.model_gateway = ModelGateway()
        self.prompt_registry = PromptRegistry()

    def _build_prompt(
            self,
            query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:
        """
        Build the prompt for the language model.
        
        Args:
            query: User query
            context: Formatted context from retrieved documents
            chat_history: Optional chat history
            
        Returns:
            Complete prompt string
        """

        prompt_template = self.prompt_registry.get("rag.response_generation")
        return prompt_template.render(
            query=query,
            context=context,
            chat_history=chat_history or "",
        )

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            picture_paths: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, Any]:
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
            prompt_template = self.prompt_registry.get("rag.response_generation")
            prompt = prompt_template.render(
                query=query,
                context=context,
                chat_history=chat_history or "",
            )
            
            # Generate response
            model_call = self.model_gateway.invoke(
                model=self.response_generator_model,
                prompt=prompt,
                model_id="rag-response-generator",
                metadata={
                    "stage": "rag_response_generation",
                    **prompt_template.metadata(),
                },
            )
            response = model_call.output
            
            # Extract sources for citation
            sources = self._extract_sources(retrieved_docs) if hasattr(self, 'include_sources') and self.include_sources else []
            
            # Calculate confidence
            confidence = self._calculate_confidence(retrieved_docs)

            # Add sources to response
            if hasattr(self, 'include_sources') and self.include_sources:
                response_with_source = response.content + "\n\n##### Source documents:"
                for current_source in sources:
                    source_path = current_source['path']
                    source_title = current_source['title']
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = response.content
            
            # Add picture paths to response
            response_with_source_and_picture_paths = response_with_source + "\n\n##### Reference images:"
            for picture_path in picture_paths:
                response_with_source_and_picture_paths += f"\n- [{picture_path.split('/')[-1]}]({picture_path})"
            
            # Format final response
            result = {
                "response": response_with_source_and_picture_paths,
                "sources": sources,
                "confidence": confidence,
                "model_gateway": {
                    "model_id": model_call.model_id,
                    "latency_ms": model_call.latency_ms,
                    "attempts": model_call.attempts,
                    "metadata": model_call.metadata,
                    "usage": {
                        "prompt_tokens": model_call.usage.prompt_tokens,
                        "completion_tokens": model_call.usage.completion_tokens,
                        "total_tokens": model_call.usage.total_tokens,
                    },
                },
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            }
            
            sources.append(source_info)
            seen_sources.add(source_id)
        
        # Sort sources by score from highest to lowest
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Format the final sources list, removing the scores which were just used for sorting
        formatted_sources = []
        for source in sources:
            formatted_source = {
                "title": source["title"],
                "path": source["path"]
            }
            formatted_sources.append(formatted_source)
            
        return formatted_sources

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
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
