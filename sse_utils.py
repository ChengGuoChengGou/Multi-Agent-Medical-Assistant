"""SSE (Server-Sent Events) streaming response utilities.

Provides streaming chat responses for better UX - users see tokens
as they're generated instead of waiting for the full response.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


async def sse_generator(
    data: Dict[str, Any],
    event: Optional[str] = None,
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
            response_data = await loop.run_in_executor(
                None, process_fn, enhanced_query, session_id
            )
            response_text = response_data['messages'][-1].content

            # Agent info event
            async for chunk in sse_generator(
                {"agent": response_data.get("agent_name", "unknown")},
                event="agent",
            ):
                yield chunk

            # Response chunk (full text - underlying model isn't streaming yet)
            async for chunk in sse_generator(
                {"text": response_text},
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
