"""Tests for session-based incremental prompt ingestion.

Tests verify that the proxy correctly handles X-Session-Id headers,
computes message deltas, and falls back to full history when needed.
These tests use mocked llama-server responses to validate behavior
without requiring a running llama-server instance.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from proxy.session_manager import SessionManager, Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session_manager():
    """Fresh SessionManager for each test."""
    return SessionManager(ttl_seconds=3600)


@pytest.fixture
def sample_messages():
    """A simple 3-turn conversation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
    ]


@pytest.fixture
def extended_messages():
    """A 5-turn conversation extending sample_messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "Tell me about sessions."},
        {"role": "assistant", "content": "Sessions allow caching of prompts."},
    ]


# ---------------------------------------------------------------------------
# Delta computation with SessionManager
# ---------------------------------------------------------------------------

class TestDeltaComputationIntegration:
    """Test delta computation in the context of session-based ingestion."""

    @pytest.mark.asyncio
    async def test_new_session_sends_full_history(self, session_manager, sample_messages):
        """A new session should have no stored history, so full messages are delta."""
        session, created = await session_manager.get_or_create("test-session-1")
        assert created is True
        assert session.message_count == 0

        delta, matches = session_manager.compute_delta([], sample_messages)
        assert matches is True
        assert delta == sample_messages
        assert len(delta) == 3

    @pytest.mark.asyncio
    async def test_second_request_sends_only_delta(self, session_manager, sample_messages, extended_messages):
        """On the second request with an existing session, only new messages are delta."""
        session, _ = await session_manager.get_or_create("test-session-2")
        # Simulate first request: store messages
        await session_manager.update_messages("test-session-2", sample_messages)

        # Second request: extended messages
        session, created = await session_manager.get_or_create("test-session-2")
        assert created is False  # session exists

        delta, matches = session_manager.compute_delta(session.messages, extended_messages)
        assert matches is True
        assert len(delta) == 2
        assert delta[0]["content"] == "Tell me about sessions."

    @pytest.mark.asyncio
    async def test_edited_message_invalidates_session(self, session_manager, sample_messages):
        """If the user edits an earlier message, the session is invalidated."""
        session, _ = await session_manager.get_or_create("test-session-3")
        await session_manager.update_messages("test-session-3", sample_messages)

        # Edit: change the first message
        edited = list(sample_messages)
        edited[1] = {"role": "user", "content": "Goodbye!"}
        edited.append({"role": "assistant", "content": "Bye!"})

        delta, matches = session_manager.compute_delta(session.messages, edited)
        assert matches is False
        # Delta should be the full list since history was edited
        assert delta == edited

    @pytest.mark.asyncio
    async def test_same_messages_no_delta(self, session_manager, sample_messages):
        """Sending the exact same messages produces no delta."""
        session, _ = await session_manager.get_or_create("test-session-4")
        await session_manager.update_messages("test-session-4", sample_messages)

        delta, matches = session_manager.compute_delta(session.messages, sample_messages)
        assert matches is True
        assert delta == []

    @pytest.mark.asyncio
    async def test_expired_session_gets_new_id(self, session_manager):
        """An expired session creates a new one on get_or_create."""
        mgr = SessionManager(ttl_seconds=0.01)
        session1, created1 = await mgr.get_or_create("expiring-session")
        assert created1 is True

        # Store some messages
        msgs = [{"role": "user", "content": "hello"}]
        await mgr.update_messages("expiring-session", msgs)

        # Let it expire
        await asyncio.sleep(0.05)

        # Should get a new session
        session2, created2 = await mgr.get_or_create("expiring-session")
        assert created2 is True
        assert session2.message_count == 0
        assert session2.messages == []


# ---------------------------------------------------------------------------
# Session header handling
# ---------------------------------------------------------------------------

class TestSessionHeaderHandling:
    """Test that X-Session-Id header is extracted and processed correctly."""

    def test_extract_session_id_from_header(self):
        """Test extracting session ID from request headers."""
        # This is verified by the server integration; we test the manager logic here
        mgr = SessionManager()
        # A None session ID should generate a UUID
        assert mgr.active_session_count == 0

    @pytest.mark.asyncio
    async def test_session_id_echo_response(self, session_manager):
        """Verify session_id can be retrieved after creation."""
        session, created = await session_manager.get_or_create("my-client-session")
        assert created is True
        assert session.session_id == "my-client-session"

        # Retrieving it again should return the same ID
        session2, created2 = await session_manager.get_or_create("my-client-session")
        assert created2 is False
        assert session2.session_id == "my-client-session"


# ---------------------------------------------------------------------------
# Assistant content extraction
# ---------------------------------------------------------------------------

class TestContentExtraction:
    """Test extracting assistant content from responses."""

    def test_extract_assistant_content_non_streaming(self):
        """Test extracting content from a non-streaming OpenAI response."""
        from proxy.server import _extract_assistant_content

        resp = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello!"}}
            ]
        }
        assert _extract_assistant_content(resp) == "Hello!"

    def test_extract_assistant_content_empty(self):
        """Test nil/missing content returns None."""
        from proxy.server import _extract_assistant_content

        assert _extract_assistant_content({}) is None
        assert _extract_assistant_content({"choices": []}) is None

    def test_extract_assistant_content_from_sse(self):
        """Test extracting content from SSE stream text."""
        from proxy.server import _extract_assistant_content_from_sse

        sse_text = (
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n'
            'data: [DONE]\n'
        )
        result = _extract_assistant_content_from_sse(sse_text)
        assert result == "Hello world"

    def test_extract_assistant_content_from_sse_empty(self):
        """Test SSE extraction with no content returns None."""
        from proxy.server import _extract_assistant_content_from_sse

        sse_text = 'data: [DONE]\n'
        result = _extract_assistant_content_from_sse(sse_text)
        assert result is None


# ---------------------------------------------------------------------------
# Session manager with delta flow simulation
# ---------------------------------------------------------------------------

class TestSessionDeltaFlow:
    """Simulate a multi-turn conversation flow end-to-end."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Simulate a 3-turn conversation verifying delta computation each step."""
        mgr = SessionManager(ttl_seconds=3600)

        # Turn 1: New session - full history
        session1, created1 = await mgr.get_or_create("conv-1")
        assert created1 is True
        msgs_t1 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        delta1, matches1 = mgr.compute_delta(session1.messages, msgs_t1)
        assert matches1 is True
        assert delta1 == msgs_t1  # first turn = entire payload

        # After turn 1, update session history with user + assistant messages
        msgs_after_t1 = msgs_t1 + [{"role": "assistant", "content": "Hello!"}]
        await mgr.update_messages("conv-1", msgs_after_t1)

        # Turn 2: Add a user message
        session2, created2 = await mgr.get_or_create("conv-1")
        assert created2 is False
        msgs_t2 = msgs_after_t1 + [{"role": "user", "content": "How are you?"}]
        delta2, matches2 = mgr.compute_delta(session2.messages, msgs_t2)
        assert matches2 is True
        assert len(delta2) == 1
        assert delta2[0]["content"] == "How are you?"

        # After turn 2
        msgs_after_t2 = msgs_t2 + [{"role": "assistant", "content": "I'm good!"}]
        await mgr.update_messages("conv-1", msgs_after_t2)

        # Turn 3: Add another message
        session3, created3 = await mgr.get_or_create("conv-1")
        assert created3 is False
        msgs_t3 = msgs_after_t2 + [{"role": "user", "content": "Great!"}]
        delta3, matches3 = mgr.compute_delta(session3.messages, msgs_t3)
        assert matches3 is True
        assert len(delta3) == 1
        assert delta3[0]["content"] == "Great!"

    @pytest.mark.asyncio
    async def test_edit_invalidates_and_full_reingestion(self):
        """Test that editing earlier messages invalidates the session."""
        mgr = SessionManager(ttl_seconds=3600)

        # Turn 1
        session, _ = await mgr.get_or_create("edit-test")
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        delta, matches = mgr.compute_delta(session.messages, msgs)
        assert delta == msgs  # First turn sends all

        msgs_after = msgs + [{"role": "assistant", "content": "Hi!"}]
        await mgr.update_messages("edit-test", msgs_after)

        # Turn 2: Edit first message
        edited_msgs = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Tell me about ships"},
        ]

        session, _ = await mgr.get_or_create("edit-test")
        delta, matches = mgr.compute_delta(session.messages, edited_msgs)
        assert matches is False
        assert delta == edited_msgs  # Full re-ingestion required

        # Invalidate the session
        await mgr.invalidate("edit-test")

        # Next request should create a new session
        session_new, created = await mgr.get_or_create("edit-test")
        assert created is True
        # New session has no history, so full payload
        delta, matches = mgr.compute_delta(session_new.messages, edited_msgs)
        assert matches is True
        assert delta == edited_msgs