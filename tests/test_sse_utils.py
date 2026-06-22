"""
Tests for sse_utils: SSE streaming utilities.

Covers:
- sse_generator: basic SSE event formatting
- sse_stream_chat: non-streaming SSE response
- sse_stream_chat_streaming: real streaming SSE with queue/threading

Run: python -m pytest tests/test_sse_utils.py -v
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sse_utils import sse_generator, sse_stream_chat, sse_stream_chat_streaming

# ── sse_generator ──────────────────────────────────────────────────────────


class TestSSEGenerator:
    """Tests for the basic sse_generator function."""

    @pytest.mark.asyncio
    async def test_basic_data_event(self):
        """Generator yields data: JSON + blank line."""
        chunks = []
        async for chunk in sse_generator({"message": "hello"}):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "data: " in chunks[0]
        parsed = json.loads(chunks[0].replace("data: ", "").strip())
        assert parsed == {"message": "hello"}

    @pytest.mark.asyncio
    async def test_named_event(self):
        """Generator yields event: name + data: JSON + blank line."""
        chunks = []
        async for chunk in sse_generator({"text": "hi"}, event="message"):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0] == "event: message" + "\n"
        assert "data: " in chunks[1]
        parsed = json.loads(chunks[1].replace("data: ", "").strip())
        assert parsed == {"text": "hi"}

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Generator handles unicode correctly."""
        chunks = []
        async for chunk in sse_generator({"diagnosis": "头痛"}):
            chunks.append(chunk)
        parsed = json.loads(chunks[0].replace("data: ", "").strip())
        assert parsed["diagnosis"] == "头痛"

    @pytest.mark.asyncio
    async def test_none_event(self):
        """None event behaves same as no event."""
        chunks = []
        async for chunk in sse_generator({"a": 1}, event=None):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "event:" not in chunks[0]

    @pytest.mark.asyncio
    async def test_empty_data(self):
        """Empty dict is valid."""
        chunks = []
        async for chunk in sse_generator({}):
            chunks.append(chunk)
        parsed = json.loads(chunks[0].replace("data: ", "").strip())
        assert parsed == {}


# ── sse_stream_chat ────────────────────────────────────────────────────────


class TestSSEStreamChat:
    """Tests for sse_stream_chat (non-streaming)."""

    @pytest.mark.asyncio
    async def test_full_sse_flow(self):
        """Sends start → agent → chunk(s) → done → end events in order."""

        def mock_process_fn(query, session_id):
            class FakeMsg:
                content = "This is a test response from the medical agent."

            return {"messages": [FakeMsg()], "agent_name": "CONVERSATION_AGENT"}

        response = await sse_stream_chat(
            query="What is diabetes?",
            session_id="test-session-1",
            process_fn=mock_process_fn,
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.media_type

        # Collect all SSE events
        body = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                body += chunk.encode()
            else:
                body += chunk

        text = body.decode("utf-8")

        # Should contain start, agent, chunk, end events
        assert "event: start" in text
        assert "event: agent" in text
        assert "event: chunk" in text
        assert "event: end" in text

        # Start event should contain session_id
        assert "test-session-1" in text

        # Agent event should contain agent name
        assert "CONVERSATION_AGENT" in text

        # End event should contain total_length
        assert "total_length" in text

    @pytest.mark.asyncio
    async def test_with_memory_context(self):
        """Memory context is prepended to query."""
        captured = {}

        def mock_process_fn(query, session_id):
            captured["query"] = query

            class FakeMsg:
                content = "answer"

            return {"messages": [FakeMsg()], "agent_name": "test"}

        response = await sse_stream_chat(
            query="What?",
            session_id="s1",
            process_fn=mock_process_fn,
            memory_context="Previous context. ",
        )
        # Must consume body to trigger the async generator (and thus process_fn)
        async for _ in response.body_iterator:
            pass

        assert captured["query"] == "What?Previous context. "

    @pytest.mark.asyncio
    async def test_no_memory_context(self):
        """Without memory_context, query is unchanged."""
        captured = {}

        def mock_process_fn(query, session_id):
            captured["query"] = query

            class FakeMsg:
                content = "ok"

            return {"messages": [FakeMsg()], "agent_name": "test"}

        resp = await sse_stream_chat(
            query="Hello",
            session_id="s2",
            process_fn=mock_process_fn,
            memory_context="",
        )
        # Must consume body to trigger the async generator (and thus process_fn)
        async for _ in resp.body_iterator:
            pass

        assert captured["query"] == "Hello"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Errors yield an error SSE event instead of crashing."""

        def failing_fn(query, session_id):
            raise RuntimeError("LLM unavailable")

        response = await sse_stream_chat(
            query="test",
            session_id="s3",
            process_fn=failing_fn,
        )

        body = b""
        async for chunk in response.body_iterator:
            body += chunk.encode() if isinstance(chunk, str) else chunk

        text = body.decode("utf-8")
        assert "event: error" in text
        assert "error occurred" in text.lower()

    @pytest.mark.asyncio
    async def test_response_headers(self):
        """Response includes SSE-required headers."""

        def mock_fn(q, s):
            class M:
                content = "r"

            return {"messages": [M()], "agent_name": "t"}

        response = await sse_stream_chat("q", "s", mock_fn)
        assert response.headers.get("Cache-Control") == "no-cache"
        assert response.headers.get("X-Accel-Buffering") == "no"

    @pytest.mark.asyncio
    async def test_chunk_streaming(self):
        """Long responses are split into multiple chunk events."""
        long_text = "A" * 200  # > 4 chunks of 50 chars

        def mock_fn(q, s):
            class M:
                content = long_text

            return {"messages": [M()], "agent_name": "t"}

        response = await sse_stream_chat("q", "s", mock_fn)

        body = b""
        async for chunk in response.body_iterator:
            body += chunk.encode() if isinstance(chunk, str) else chunk

        text = body.decode("utf-8")
        # Count chunk events (excluding the final "done" chunk)
        chunk_count = text.count("event: chunk")
        assert chunk_count >= 4  # 200/50 = 4 + 1 done marker = 5


# ── sse_stream_chat_streaming ──────────────────────────────────────────────


class TestSSEStreamChatStreaming:
    """Tests for sse_stream_chat_streaming (real streaming)."""

    @pytest.mark.asyncio
    async def test_streaming_flow(self):
        """Receives progress events from graph nodes then final result."""

        def mock_streaming_fn(query):
            # Simulate node progress events
            yield {"type": "node_end", "node": "retriever", "output_preview": "docs found"}
            yield {"type": "node_end", "node": "generator", "output_preview": "generating"}

            # Final result
            class FakeMsg:
                content = "Final streaming answer"

            yield {
                "type": "final",
                "result": {"messages": [FakeMsg()], "agent_name": "RAG_AGENT"},
            }

        response = await sse_stream_chat_streaming(
            query="What is hypertension?",
            session_id="stream-1",
            streaming_fn=mock_streaming_fn,
        )

        body = b""
        async for chunk in response.body_iterator:
            body += chunk.encode() if isinstance(chunk, str) else chunk

        text = body.decode("utf-8")

        # Should have start, progress, agent, chunk, end events
        assert "event: start" in text
        assert "event: progress" in text
        assert "event: agent" in text
        assert "event: chunk" in text
        assert "event: end" in text

        # Progress events should contain node names
        assert "retriever" in text
        assert "generator" in text

        # Agent name should appear
        assert "RAG_AGENT" in text

    @pytest.mark.asyncio
    async def test_streaming_error_in_fn(self):
        """Errors from streaming_fn yield error SSE event."""

        def failing_streaming_fn(query):
            raise RuntimeError("Graph crashed")
            yield  # make it a generator

        response = await sse_stream_chat_streaming(
            query="test",
            session_id="stream-err",
            streaming_fn=failing_streaming_fn,
        )

        body = b""
        async for chunk in response.body_iterator:
            body += chunk.encode() if isinstance(chunk, str) else chunk

        text = body.decode("utf-8")
        assert "event: error" in text

    @pytest.mark.asyncio
    async def test_streaming_no_final(self):
        """If streaming_fn yields no final event, error is emitted."""

        def incomplete_fn(query):
            yield {"type": "node_end", "node": "step1", "output_preview": "partial"}

        response = await sse_stream_chat_streaming(
            query="test",
            session_id="stream-nofinal",
            streaming_fn=incomplete_fn,
        )

        body = b""
        async for chunk in response.body_iterator:
            body += chunk.encode() if isinstance(chunk, str) else chunk

        text = body.decode("utf-8")
        assert "event: error" in text
        assert "No result" in text

    @pytest.mark.asyncio
    async def test_streaming_with_memory(self):
        """Memory context is prepended in streaming mode."""
        captured = {}

        def mock_streaming_fn(query):
            captured["query"] = query

            class FakeMsg:
                content = "ok"

            yield {"type": "final", "result": {"messages": [FakeMsg()], "agent_name": "x"}}

        resp = await sse_stream_chat_streaming(
            query="hello",
            session_id="s",
            streaming_fn=mock_streaming_fn,
            memory_context="ctx. ",
        )
        # Must consume body to trigger the async generator (and thus streaming_fn)
        async for _ in resp.body_iterator:
            pass

        assert captured["query"] == "helloctx. "

    @pytest.mark.asyncio
    async def test_streaming_headers(self):
        """Streaming response has correct SSE headers."""

        def mock_fn(q):
            class M:
                content = "r"

            yield {"type": "final", "result": {"messages": [M()], "agent_name": "x"}}

        response = await sse_stream_chat_streaming("q", "s", mock_fn)
        assert "text/event-stream" in response.media_type
        assert response.headers.get("Cache-Control") == "no-cache"
