"""
Incremental Indexing Module - Watchdog-based File Monitor (Phase 6)

Watches the raw document directory for new/modified PDF files and
automatically triggers re-indexing pipeline:
  1. Parse new/changed PDF via ContentProcessor
  2. Chunk text into documents
  3. Upsert into Qdrant vectorstore (dense + sparse)
  4. Update BM25 index in HybridSearch

Features:
- Debounce to avoid repeated processing during file writes
- Configurable watch paths and file extensions
- Graceful start/stop lifecycle
- Callback-based integration with existing pipeline
"""

import logging
import os
import time
import threading
from pathlib import Path
from typing import Callable, Optional, Set, Dict, Any, List

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileDeletedEvent,
)

logger = logging.getLogger(__name__)


class IndexingEventHandler(FileSystemEventHandler):
    """
    Handles file system events in watched directories.
    Debounces rapid events and triggers indexing callback.
    """

    def __init__(
        self,
        callback: Callable[[str, str], None],
        extensions: Set[str] = None,
        debounce_seconds: float = 5.0,
    ):
        """
        Args:
            callback: Function to call with (file_path, event_type) when ready to index
            extensions: Set of file extensions to watch (e.g., {'.pdf'}). None = all
            debounce_seconds: Wait this long after last event before triggering
        """
        super().__init__()
        self.callback = callback
        self.extensions = extensions or {'.pdf'}
        self.debounce_seconds = debounce_seconds
        self._pending: Dict[str, float] = {}  # path -> last_event_time
        self._lock = threading.Lock()
        self._debounce_timer: Optional[threading.Timer] = None
        self.logger = logging.getLogger(__name__)

    def _should_process(self, path: str) -> bool:
        """Check if file matches watched extensions."""
        return Path(path).suffix.lower() in self.extensions

    def _schedule_debounce(self) -> None:
        """Schedule debounce check."""
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
        self._debounce_timer = threading.Timer(self.debounce_seconds, self._process_pending)
        self._debounce_timer.daemon = True
        self._debounce_timer.start()

    def _process_pending(self) -> None:
        """Process all pending events after debounce period."""
        with self._lock:
            now = time.time()
            ready = [
                (path, ts) for path, ts in self._pending.items()
                if now - ts >= self.debounce_seconds
            ]
            for path, _ in ready:
                del self._pending[path]

        for path, _ in ready:
            self.logger.info(f"[WATCHDOG] Processing indexed file: {path}")
            try:
                self.callback(path, "modified")
            except Exception as e:
                self.logger.error(f"[WATCHDOG] Indexing callback failed for {path}: {e}")

    def _queue_event(self, path: str) -> None:
        """Queue a file event for debounced processing."""
        with self._lock:
            self._pending[path] = time.time()
        self._schedule_debounce()

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.logger.info(f"[WATCHDOG] File created: {event.src_path}")
            self._queue_event(event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.logger.info(f"[WATCHDOG] File modified: {event.src_path}")
            self._queue_event(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            if self._should_process(event.dest_path):
                self.logger.info(f"[WATCHDOG] File moved to: {event.dest_path}")
                self._queue_event(event.dest_path)

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.logger.info(f"[WATCHDOG] File deleted: {event.src_path}")
            # For deletion, process immediately (no debounce needed)
            try:
                self.callback(event.src_path, "deleted")
            except Exception as e:
                self.logger.error(f"[WATCHDOG] Delete callback failed for {event.src_path}: {e}")


class IncrementalIndexer:
    """
    Manages incremental indexing of documents using watchdog file monitoring.
    
    Lifecycle:
        indexer = IncrementalIndexer(...)
        indexer.start()  # Begin watching
        ...
        indexer.stop()   # Stop watching and cleanup
    
    Integrates with ContentProcessor, QdrantVectorStore, and HybridSearch.
    """

    def __init__(
        self,
        watch_paths: List[str],
        content_processor=None,
        vectorstore_manager=None,
        hybrid_search=None,
        extensions: Set[str] = None,
        debounce_seconds: float = 5.0,
        recursive: bool = True,
        on_index_complete: Optional[Callable[[str, bool], None]] = None,
    ):
        """
        Args:
            watch_paths: List of directory paths to watch
            content_processor: ContentProcessor instance for parsing PDFs
            vectorstore_manager: QdrantVectorStoreManager for upserting docs
            hybrid_search: HybridSearch instance for updating BM25 index
            extensions: File extensions to watch (default: {'.pdf'})
            debounce_seconds: Debounce period for rapid events
            recursive: Watch subdirectories
            on_index_complete: Optional callback(path, success) after indexing
        """
        self.logger = logging.getLogger(__name__)
        self.watch_paths = [Path(p) for p in watch_paths]
        self.content_processor = content_processor
        self.vectorstore_manager = vectorstore_manager
        self.hybrid_search = hybrid_search
        self.extensions = extensions or {'.pdf'}
        self.debounce_seconds = debounce_seconds
        self.recursive = recursive
        self.on_index_complete = on_index_complete

        self._observer: Optional[Observer] = None
        self._running = False
        self._index_lock = threading.Lock()
        self._stats = {
            "files_processed": 0,
            "files_failed": 0,
            "last_processed": None,
        }

    def _index_file(self, file_path: str, event_type: str) -> None:
        """
        Index a single file. Called by the event handler.
        
        Steps:
        1. Parse PDF → pages
        2. Chunk pages → documents  
        3. Upsert into Qdrant vectorstore
        4. Update BM25 index
        """
        with self._index_lock:
            path = Path(file_path)
            self.logger.info(f"[INDEXER] Starting {'re-' if event_type == 'modified' else ''}indexing: {path.name}")

            try:
                # Step 1: Parse PDF
                if self.content_processor is None:
                    self.logger.warning("[INDEXER] No content_processor available, skipping parse")
                    return

                parsed_content = self.content_processor.parse_document(str(path))
                if not parsed_content:
                    self.logger.warning(f"[INDEXER] No content parsed from {path.name}")
                    return

                # Step 2: Chunk into documents
                documents = self.content_processor.chunk_content(parsed_content)
                if not documents:
                    self.logger.warning(f"[INDEXER] No chunks produced from {path.name}")
                    return

                self.logger.info(f"[INDEXER] Parsed {path.name}: {len(documents)} chunks")

                # Step 3: Upsert into Qdrant
                if self.vectorstore_manager is not None:
                    try:
                        vectorstore, docstore = self.vectorstore_manager.load_vectorstore()
                        # Delete existing entries for this source first
                        # Then upsert new documents
                        vectorstore.add_documents(documents)
                        self.logger.info(f"[INDEXER] Upserted {len(documents)} docs into Qdrant for {path.name}")
                    except Exception as e:
                        self.logger.error(f"[INDEXER] Qdrant upsert failed for {path.name}: {e}")

                # Step 4: Update BM25 index
                if self.hybrid_search is not None:
                    try:
                        self.hybrid_search.incremental_update(path.name, documents)
                        self.logger.info(f"[INDEXER] BM25 index updated for {path.name}")
                    except Exception as e:
                        self.logger.error(f"[INDEXER] BM25 update failed for {path.name}: {e}")

                self._stats["files_processed"] += 1
                self._stats["last_processed"] = str(path)

                if self.on_index_complete:
                    self.on_index_complete(file_path, True)

                self.logger.info(f"[INDEXER] Successfully indexed: {path.name}")

            except Exception as e:
                self._stats["files_failed"] += 1
                self.logger.error(f"[INDEXER] Failed to index {file_path}: {e}", exc_info=True)
                if self.on_index_complete:
                    self.on_index_complete(file_path, False)

    def start(self) -> None:
        """Start file monitoring."""
        if self._running:
            self.logger.warning("[INDEXER] Already running")
            return

        self._observer = Observer()
        handler = IndexingEventHandler(
            callback=self._index_file,
            extensions=self.extensions,
            debounce_seconds=self.debounce_seconds,
        )

        for watch_path in self.watch_paths:
            if not watch_path.exists():
                self.logger.warning(f"[INDEXER] Watch path does not exist: {watch_path}")
                continue
            self._observer.schedule(handler, str(watch_path), recursive=self.recursive)
            self.logger.info(f"[INDEXER] Watching: {watch_path}")

        self._observer.daemon = True
        self._observer.start()
        self._running = True
        self.logger.info("[INDEXER] File monitoring started")

    def stop(self) -> None:
        """Stop file monitoring."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=10)
            self._observer = None
        self._running = False
        self.logger.info("[INDEXER] File monitoring stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
