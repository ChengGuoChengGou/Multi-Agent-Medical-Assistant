import os
import time
import logging
from typing import List, Optional, Dict, Any

from .doc_parser import MedicalDocParser
from .content_processor import ContentProcessor
from .vectorstore_qdrant import VectorStore
from .reranker import Reranker
from .query_expander import QueryExpander
from .response_generator import ResponseGenerator
from .retrieval_channels import (
    FAQKeywordSearchChannel,
    IntentDirectedMedicalSearchChannel,
    MultiChannelRetrievalEngine,
    QdrantGlobalSearchChannel,
    SearchContext,
)
from .postprocessors import (
    ConfidencePostProcessor,
    CrossEncoderRerankPostProcessor,
    DeduplicationPostProcessor,
    FAQPriorityPostProcessor,
)
from .react_controller import RAGReActController
from .query_planner import QueryPlanner
from .faq_loader import load_faq_directory, load_faq_file
from .ingestion_pipeline import (
    IngestionNodeType,
    IngestionPipelineTrace,
    summarize_parse_output,
    summarize_sequence,
    summarize_text,
)

class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation system that integrates all components.
    """
    def __init__(self, config):
        """
        Initialize the RAG Agent.
        
        Args:
            config: Configuration object with RAG settings
        """
        # Set up logging
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing Medical RAG system")
        self.config = config
        self.doc_parser = MedicalDocParser()
        self.content_processor = ContentProcessor(config)
        self.vector_store = VectorStore(config)
        self.reranker = Reranker(config)
        self.query_planner = QueryPlanner(
            max_sub_questions=getattr(self.config.rag, "max_sub_questions", 3)
        )
        self.query_expander = QueryExpander(config)
        self.response_generator = ResponseGenerator(config)
        self.react_controller = RAGReActController(
            max_steps=getattr(self.config.rag, "react_max_steps", 2),
            timeout_seconds=getattr(self.config.rag, "react_timeout_seconds", 8.0),
            min_confidence=getattr(self.config.rag, "min_retrieval_confidence", 0.4),
        )
        self.parsed_content_dir = self.config.rag.parsed_content_dir
        self.retrieval_engine = MultiChannelRetrievalEngine(
            channels=[
                FAQKeywordSearchChannel(),
                IntentDirectedMedicalSearchChannel(self.vector_store),
                QdrantGlobalSearchChannel(self.vector_store),
            ],
            postprocessors=[
                DeduplicationPostProcessor(),
                FAQPriorityPostProcessor(),
                CrossEncoderRerankPostProcessor(self.reranker, self.parsed_content_dir),
                ConfidencePostProcessor(),
            ],
        )
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all files in a directory into the RAG system.
        
        Args:
            directory_path: Path to the directory containing files to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting files from directory: {directory_path}")
        
        try:
            # Check if directory exists
            if not os.path.isdir(directory_path):
                raise ValueError(f"Directory not found: {directory_path}")
            
            # Get all files in the directory
            files = [os.path.join(directory_path + '/', f) for f in os.listdir(directory_path) 
                     if os.path.isfile(os.path.join(directory_path, f))]
            
            if not files:
                self.logger.warning(f"No files found in directory: {directory_path}")
                return {
                    "success": True,
                    "documents_ingested": 0,
                    "chunks_processed": 0,
                    "processing_time": time.time() - start_time,
                    "file_results": [],
                }
            
            # Track statistics
            total_chunks_processed = 0
            successful_ingestions = 0
            failed_ingestions = 0
            failed_files = []
            file_results = []
            
            # Process each file
            for file_path in files:
                self.logger.info(f"Processing file {successful_ingestions + failed_ingestions + 1}/{len(files)}: {file_path}")
                
                try:
                    result = self.ingest_file(file_path)
                    file_results.append(result)
                    if result["success"]:
                        successful_ingestions += 1
                        total_chunks_processed += result.get("chunks_processed", 0)
                    else:
                        failed_ingestions += 1
                        failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    failed_ingestions += 1
                    failed_files.append({"file": file_path, "error": str(e)})
                    file_results.append({
                        "success": False,
                        "file": file_path,
                        "error": str(e),
                    })
            
            return {
                "success": True,
                "documents_ingested": successful_ingestions,
                "failed_documents": failed_ingestions,
                "failed_files": failed_files,
                "chunks_processed": total_chunks_processed,
                "processing_time": time.time() - start_time,
                "file_results": file_results,
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def ingest_file(self, document_path: str) -> Dict[str, Any]:
        """
        Ingest a single file into the RAG system.
        
        Args:
            document_path: Path to the file to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting file: {document_path}")
        trace = IngestionPipelineTrace.create(source=document_path)
        trace.start()

        try:
            # Step 1: Parse document
            self.logger.info("1. Parsing document and extracting images...")
            parsed_document, images = trace.run_node(
                IngestionNodeType.PARSE_DOCUMENT,
                lambda: self.doc_parser.parse_document(document_path, self.parsed_content_dir),
                summarize_parse_output,
            )
            self.logger.info(f"   Parsed document and extracted {len(images)} images")

            # Step 2: Summarize images
            self.logger.info("2. Summarizing images...")
            image_summaries = trace.run_node(
                IngestionNodeType.SUMMARIZE_IMAGES,
                lambda: self.content_processor.summarize_images(images),
                summarize_sequence("image_summaries_generated"),
            )
            self.logger.info(f"   Generated {len(image_summaries)} image summaries")

            # Step 3: Format document with image summaries
            self.logger.info("3. Formatting document with image summaries...")
            formatted_document = trace.run_node(
                IngestionNodeType.FORMAT_DOCUMENT,
                lambda: self.content_processor.format_document_with_images(parsed_document, image_summaries),
                summarize_text("formatted_characters"),
            )

            # Step 4: Chunk document into semantic sections
            self.logger.info("4. Chunking document into semantic sections...")
            document_chunks = trace.run_node(
                IngestionNodeType.CHUNK_DOCUMENT,
                lambda: self.content_processor.chunk_document(formatted_document),
                summarize_sequence("chunks_created"),
            )
            self.logger.info(f"   Document split into {len(document_chunks)} chunks")

            # Step 5: Create vector store and document store
            self.logger.info("5. Creating vector store knowledge base...")
            trace.run_node(
                IngestionNodeType.INDEX_VECTORSTORE,
                lambda: self.vector_store.create_vectorstore(
                    document_chunks=document_chunks,
                    document_path=document_path
                ),
                lambda _: {"chunks_indexed": len(document_chunks)},
                )

            trace.complete()
            return {
                "success": True,
                "documents_ingested": 1,
                "chunks_processed": len(document_chunks),
                "processing_time": time.time() - start_time,
                "ingestion_trace": trace.to_dict(),
            }
        
        except Exception as e:
            self.logger.error(f"Error ingesting file: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "ingestion_trace": trace.to_dict(),
            }

    def ingest_faq_file(self, faq_path: str) -> Dict[str, Any]:
        """
        Ingest patient-facing FAQ markdown as one searchable chunk per FAQ entry.

        FAQ entries are already semantically small and source-tagged, so this
        path skips expensive PDF parsing, image summarization, and LLM chunking.
        """
        start_time = time.time()
        self.logger.info(f"Ingesting FAQ file: {faq_path}")
        trace = IngestionPipelineTrace.create(source=faq_path)
        trace.start()

        try:
            faq_entries = trace.run_node(
                IngestionNodeType.PARSE_FAQ,
                lambda: load_faq_file(faq_path),
                summarize_sequence("faq_entries_parsed"),
            )
            document_chunks = [entry.to_chunk() for entry in faq_entries]
            document_metadatas = [entry.to_metadata() for entry in faq_entries]

            trace.run_node(
                IngestionNodeType.INDEX_VECTORSTORE,
                lambda: self.vector_store.create_vectorstore(
                    document_chunks=document_chunks,
                    document_path=faq_path,
                    document_metadatas=document_metadatas,
                ),
                lambda _: {"chunks_indexed": len(document_chunks)},
            )

            trace.complete()
            return {
                "success": True,
                "documents_ingested": 1,
                "faq_entries_ingested": len(faq_entries),
                "chunks_processed": len(document_chunks),
                "processing_time": time.time() - start_time,
                "ingestion_trace": trace.to_dict(),
            }
        except Exception as e:
            self.logger.error(f"Error ingesting FAQ file: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "ingestion_trace": trace.to_dict(),
            }

    def ingest_faq_directory(self, faq_directory: str) -> Dict[str, Any]:
        """
        Ingest all FAQ markdown files in a directory.
        """
        start_time = time.time()
        self.logger.info(f"Ingesting FAQ directory: {faq_directory}")

        try:
            faq_entries = load_faq_directory(faq_directory)
            document_chunks = [entry.to_chunk() for entry in faq_entries]
            document_metadatas = [entry.to_metadata() for entry in faq_entries]

            self.vector_store.create_vectorstore(
                document_chunks=document_chunks,
                document_path=faq_directory,
                document_metadatas=document_metadatas,
            )

            domain_counts: Dict[str, int] = {}
            priority_counts: Dict[str, int] = {}
            for metadata in document_metadatas:
                domain = metadata.get("domain", "unknown")
                priority = metadata.get("priority", "unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

            return {
                "success": True,
                "documents_ingested": len({entry.source_file for entry in faq_entries}),
                "faq_entries_ingested": len(faq_entries),
                "chunks_processed": len(document_chunks),
                "domain_counts": domain_counts,
                "priority_counts": priority_counts,
                "processing_time": time.time() - start_time,
            }
        except Exception as e:
            self.logger.error(f"Error ingesting FAQ directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }
        
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a query with the RAG system.
        
        Args:
            query: The query string
            chat_history: Optional chat history for context
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        self.logger.info(f"RAG Agent processing query: {query}")
        
        # Process query and return result, passing chat_history
        try:
            # Step 1: Plan query rewrite, clarification, and sub-question split.
            self.logger.info(f"1. Planning query: '{query}'")
            query_plan = self.query_planner.plan(query, chat_history=chat_history)
            if query_plan.needs_clarification:
                return {
                    "response": query_plan.clarification_question,
                    "sources": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time,
                    "query_plan": query_plan.to_dict(),
                    "needs_clarification": True,
                }

            # Step 2: Expand query
            self.logger.info(f"2. Expanding query: '{query_plan.rewritten_query}'")
            expansion_result = self.query_expander.expand_query(query_plan.rewritten_query)
            expanded_query = expansion_result["expanded_query"]
            self.logger.info(f"   Original: '{query_plan.original_query}'")
            self.logger.info(f"   Rewritten: '{query_plan.rewritten_query}'")
            self.logger.info(f"   Expanded: '{expanded_query}'")

            planned_sub_questions = [
                sub_question
                for sub_question in query_plan.sub_questions
                if sub_question.strip()
            ] or [query_plan.rewritten_query]
            retrieval_queries = self._build_retrieval_queries(
                expanded_query=expanded_query,
                sub_questions=planned_sub_questions,
            )

            # Step 3: ReAct controlled multi-channel retrieval
            self.logger.info(f"3. Retrieving relevant documents for {len(retrieval_queries)} planned query path(s)")
            vectorstore, docstore = self.vector_store.load_vectorstore()

            def retrieve(candidate_query: str, top_k: int):
                retrieval_context = SearchContext(
                    query=candidate_query,
                    vectorstore=vectorstore,
                    docstore=docstore,
                    top_k=top_k,
                )
                return self.retrieval_engine.retrieve(retrieval_context)

            react_results = []
            for retrieval_query in retrieval_queries:
                react_results.append(
                    self.react_controller.run(
                        original_query=query_plan.original_query,
                        initial_query=retrieval_query,
                        initial_top_k=self.config.rag.top_k,
                        retrieve=retrieve,
                    )
                )

            retrieved_documents = self._merge_documents(
                result.selected_attempt.documents for result in react_results
            )
            picture_paths = self._merge_picture_paths(
                result.selected_attempt.picture_paths for result in react_results
            )
            selected_attempts = [
                result.selected_attempt.to_dict()
                for result in react_results
            ]
            retrieval_trace = {
                "query_paths": [
                    {
                        "query": retrieval_queries[index],
                        "retrieval_trace": react_results[index].selected_attempt.retrieval_trace,
                    }
                    for index in range(len(react_results))
                ],
                "merged_document_count": len(retrieved_documents),
                "merged_picture_count": len(picture_paths),
            }

            self.logger.info(f"   Retrieved {len(retrieved_documents)} relevant document chunks after post-processing")
            self.logger.info(f"   Found {len(picture_paths)} referenced images")

            # Step 4: Generate response
            self.logger.info("4. Generating response...")
            response = self.response_generator.generate_response(
                query=query_plan.rewritten_query,
                retrieved_docs=retrieved_documents,
                picture_paths=picture_paths,
                chat_history=chat_history
                )
            
            # Add timing information
            processing_time = time.time() - start_time
            response["processing_time"] = processing_time
            response["query_plan"] = query_plan.to_dict()
            response["retrieval_trace"] = retrieval_trace
            response["react_trace"] = {
                "query_count": len(react_results),
                "selected_attempts": selected_attempts,
                "results": [result.to_dict() for result in react_results],
            }
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return error response
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }

    def _build_retrieval_queries(
        self,
        *,
        expanded_query: str,
        sub_questions: List[str],
    ) -> List[str]:
        if len(sub_questions) <= 1:
            return [expanded_query]

        queries = []
        for sub_question in sub_questions:
            if sub_question.strip() and sub_question not in expanded_query:
                queries.append(f"{sub_question} {expanded_query}")
            else:
                queries.append(expanded_query)
        return queries

    def _merge_documents(self, document_groups) -> List[Dict[str, Any]]:
        best_by_key: Dict[str, Dict[str, Any]] = {}
        for documents in document_groups:
            for document in documents:
                key = str(document.get("faq_id") or document.get("id") or document.get("content", ""))
                current = best_by_key.get(key)
                if current is None or self._document_score(document) > self._document_score(current):
                    best_by_key[key] = document
        return sorted(best_by_key.values(), key=self._document_score, reverse=True)[: self.config.rag.top_k]

    def _merge_picture_paths(self, picture_path_groups) -> List[str]:
        seen = set()
        merged = []
        for picture_paths in picture_path_groups:
            for picture_path in picture_paths:
                if picture_path in seen:
                    continue
                seen.add(picture_path)
                merged.append(picture_path)
        return merged

    def _document_score(self, document: Dict[str, Any]) -> float:
        return float(
            document.get(
                "retrieval_confidence",
                document.get(
                    "combined_score",
                    document.get(
                        "rerank_score",
                        document.get("score", 0.0),
                    ),
                ),
            )
        )
