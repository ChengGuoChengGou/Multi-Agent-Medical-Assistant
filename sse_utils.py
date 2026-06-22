"""SSE (Server-Sent Events) streaming response utilities.

Provides streaming chat responses for better UX - users see tokens
as they're generated instead of waiting for the full response.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Dict, Optional

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


async def sse_generator(
    data: dict[str, Any],
    event: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a single SSE event.

    Args:
        data: Dictionary to send as JSON
        event: Optional SSE event name

    Yields:
        Formatted SSE string
    """
    if event:
        yield f"event: {event}\n"
    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def sse_stream_chat(
    query: str,
    session_id: str,
    process_fn,
    memory_context: str = "",
) -> StreamingResponse:
    """Create a streaming SSE response for chat.

    Sends events:
    - 'start': {session_id, query}
    - 'agent': {agent_name}: selected agent info
    - 'chunk': {text}: response text (sent as full text, not token-by-token
      since the underlying process_query is synchronous)
    - 'end': {total_length, agent}
    - 'error': {message}: if error occurs

    Args:
        query: User query text
        session_id: Session identifier
        process_fn: The process_query function
        memory_context: Optional memory context to prepend

    Returns:
        StreamingResponse with SSE content type
    """

    async def event_stream():
        try:
            # Start event
            async for chunk in sse_generator(
                {"session_id": session_id, "query": query[:100]},
                event="start",
            ):
                yield chunk

            # Process query in thread pool
            enhanced_query = query + memory_context if memory_context else query
            loop = asyncio.get_event_loop()
            response_data = await loop.run_in_executor(None, process_fn, enhanced_query, session_id)
            response_text = response_data["messages"][-1].content

            # Agent info event
            async for chunk in sse_generator(
                {"agent": response_data.get("agent_name", "unknown")},
                event="agent",
            ):
                yield chunk

            # Phase 52: Split response into streaming chunks for better UX
            chunk_size = 50  # Characters per chunk
            total = len(response_text)
            for i in range(0, total, chunk_size):
                text_chunk = response_text[i : i + chunk_size]
                async for chunk in sse_generator(
                    {
                        "text": text_chunk,
                        "done": False,
                        "progress": round((i + len(text_chunk)) / total, 2) if total > 0 else 1.0,
                    },
                    event="chunk",
                ):
                    yield chunk
                # Small delay to simulate streaming and prevent overwhelming the client
                if i + chunk_size < total:
                    await asyncio.sleep(0.02)

            # Final chunk marker
            async for chunk in sse_generator(
                {"text": "", "done": True, "progress": 1.0},
                event="chunk",
            ):
                yield chunk

            # Check for skin lesion output image
            import os

            result_extra = {}
            if response_data.get("agent_name") in ("SKIN_LESION_AGENT", "HUMAN_VALIDATION"):
                seg_path = os.path.join(".", "uploads", "skin_lesion_output", "segmentation_plot.png")
                if os.path.exists(seg_path):
                    result_extra["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"

            # End event
            async for chunk in sse_generator(
                {
                    "total_length": len(response_text),
                    "agent": response_data.get("agent_name", "unknown"),
                    **result_extra,
                },
                event="end",
            ):
                yield chunk

        except Exception as e:
            logger.error(f"[sse_stream_chat] Error: {e}")
            async for chunk in sse_generator(
                {"error": "An error occurred while processing your request."},
                event="error",
            ):
                yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


async def sse_stream_chat_streaming(
    query: str,
    session_id: str,
    streaming_fn,
    memory_context: str = "",
) -> StreamingResponse:
    """Create a streaming SSE response using process_query_streaming generator.

    Unlike sse_stream_chat which runs the full query then fakes streaming,
    this function receives real-time node progress events from graph.stream().

    Sends events:
    - 'start': {session_id, query}
    - 'progress': {node, status, output_preview} - real-time node progress
    - 'agent': {agent_name}: selected agent info
    - 'chunk': {text, done, progress}: response text chunks
    - 'end': {total_length, agent}
    - 'error': {message}: if error occurs

    Args:
        query: User query text
        session_id: Session identifier
        streaming_fn: The process_query_streaming generator function
        memory_context: Optional memory context to prepend

    Returns:
        StreamingResponse with SSE content type
    """

    async def event_stream():
        try:
            # Start event
            async for chunk in sse_generator(
                {"session_id": session_id, "query": query[:100]},
                event="start",
            ):
                yield chunk

            enhanced_query = query + memory_context if memory_context else query
            loop = asyncio.get_event_loop()

            # Run streaming generator in thread pool, yielding progress events
            import queue
            import threading

            q = queue.Queue()
            _sentinel = object()

            def _run_generator():
                try:
                    for event in streaming_fn(enhanced_query):
                        q.put(event)
                except Exception as e:
                    q.put({"type": "error", "message": str(e)})
                finally:
                    q.put(_sentinel)

            thread = threading.Thread(target=_run_generator, daemon=True)
            thread.start()

            final_result = None

            # Drain queue in async context
            while True:
                event = await loop.run_in_executor(None, q.get)
                if event is _sentinel:
                    break

                event_type = event.get("type", "")

                if event_type == "node_end":
                    # Real-time node progress event
                    async for chunk in sse_generator(
                        {
                            "node": event.get("node", "unknown"),
                            "output_preview": event.get("output_preview", ""),
                            "needs_human_validation": event.get("needs_human_validation", False),
                        },
                        event="progress",
                    ):
                        yield chunk

                elif event_type == "final":
                    final_result = event.get("result")

                elif event_type == "error":
                    async for chunk in sse_generator(
                        {"message": event.get("message", "Unknown error")},
                        event="error",
                    ):
                        yield chunk
                    return

            if not final_result:
                async for chunk in sse_generator(
                    {"message": "No result from graph"},
                    event="error",
                ):
                    yield chunk
                return

            response_text = final_result["messages"][-1].content
            agent_name = final_result.get("agent_name", "unknown")

            # Agent info event
            async for chunk in sse_generator(
                {"agent": agent_name},
                event="agent",
            ):
                yield chunk

            # Stream response text in chunks
            chunk_size = 50
            total = len(response_text)
            for i in range(0, total, chunk_size):
                text_chunk = response_text[i : i + chunk_size]
                async for chunk in sse_generator(
                    {
                        "text": text_chunk,
                        "done": False,
                        "progress": round((i + len(text_chunk)) / total, 2) if total > 0 else 1.0,
                    },
                    event="chunk",
                ):
                    yield chunk
                if i + chunk_size < total:
                    await asyncio.sleep(0.02)

            # Final chunk marker
            async for chunk in sse_generator(
                {"text": "", "done": True, "progress": 1.0},
                event="chunk",
            ):
                yield chunk

            # Skin lesion image check
            import os

            result_extra = {}
            if agent_name in ("SKIN_LESION_AGENT", "HUMAN_VALIDATION"):
                seg_path = os.path.join(".", "uploads", "skin_lesion_output", "segmentation_plot.png")
                if os.path.exists(seg_path):
                    result_extra["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"

            # End event
            async for chunk in sse_generator(
                {
                    "total_length": len(response_text),
                    "agent": agent_name,
                    **result_extra,
                },
                event="end",
            ):
                yield chunk

        except Exception as e:
            logger.error(f"[sse_stream_chat_streaming] Error: {e}")
            async for chunk in sse_generator(
                {"error": "An error occurred while processing your request."},
                event="error",
            ):
                yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
