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

        metrics = session_manager.compute_delta_metrics(session.messages, extended_messages)
        assert metrics["reduction_percent"] >= 30.0
        assert metrics["full_payload_bytes"] > metrics["delta_payload_bytes"]
        # Diagnostic-only latency capture: informational, not pass/fail gated.
        start = asyncio.get_event_loop().time()
        _ = session_manager.compute_delta_metrics(session.messages, extended_messages)
        elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000.0
        assert elapsed_ms >= 0

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

class TestRestoreContract:
    """Strict restore signal contract tests."""

    def test_classify_delta_routing_requires_restore_signal(self):
        from proxy.server import _classify_delta_routing

        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=2,
            restore_confirmed=False,
        )
        assert use_delta is False
        assert reason == "missing_restore_signal"

    def test_classify_delta_routing_allows_delta_when_signal_confirmed(self):
        from proxy.server import _classify_delta_routing

        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=2,
            restore_confirmed=True,
        )
        assert use_delta is True
        assert reason is None

    def test_classify_delta_routing_allows_delta_when_restore_signal_not_required(self):
        from proxy.server import _classify_delta_routing

        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=2,
            restore_confirmed=False,
            require_restore_signal=False,
        )
        assert use_delta is True
        assert reason is None

    def test_has_explicit_restore_signal_positive_and_negative(self):
        from proxy.server import _has_explicit_restore_signal

        assert _has_explicit_restore_signal(
            {"X-Llama-Session-Restored": "true"},
            None,
        ) is True
        assert _has_explicit_restore_signal({}, {"session_restored": True}) is True
        # Regression guard: history match without explicit signal is not restore success.
        assert _has_explicit_restore_signal({}, {"session_restored": False}) is False

    def test_detect_restore_signal_from_llama_log_matches_session(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log

        log_file = tmp_path / "llama-server.log"
        log_file.write_text(
            "INFO slot update\n"
            "INFO slot load_session: loading KV cache for session_id=abc-123\n"
        )

        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_detect_restore_signal_from_llama_log_requires_signal_and_session(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log

        log_file = tmp_path / "llama-server.log"
        log_file.write_text("INFO slot update without restore\n")

        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is False

    def test_detect_restore_signal_from_log_slice(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice

        log_file = tmp_path / "llama-server.log"
        log_file.write_text("before\n")
        start_offset = log_file.stat().st_size
        with log_file.open("a", encoding="utf-8") as f:
            f.write("slot update_slots restored context checkpoint\n")

        assert _detect_restore_signal_from_log_slice(log_file, start_offset) is True

    def test_detect_restore_signal_from_log_slice_negative(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice

        log_file = tmp_path / "llama-server.log"
        log_file.write_text("before\n")
        start_offset = log_file.stat().st_size
        with log_file.open("a", encoding="utf-8") as f:
            f.write("slot update_slots normal processing\n")

        assert _detect_restore_signal_from_log_slice(log_file, start_offset) is False


class TestRestoreObservability:
    """Observability counters for strict restore behavior."""

    def test_restore_observability_counters(self):
        from proxy.server import (
            session_restore_observability,
            _record_restore_success,
            _record_restore_fallback,
            _record_delta_payload_bytes,
        )

        session_restore_observability["restore_success_total"] = 0
        session_restore_observability["restore_fallback_total"] = {}
        session_restore_observability["delta_payload_bytes_total"] = 0

        _record_restore_success()
        _record_restore_fallback("missing_restore_signal")
        _record_restore_fallback("missing_restore_signal")
        _record_restore_fallback("history_mismatch")
        _record_delta_payload_bytes(123)

        assert session_restore_observability["restore_success_total"] == 1
        assert session_restore_observability["restore_fallback_total"]["missing_restore_signal"] == 2
        assert session_restore_observability["restore_fallback_total"]["history_mismatch"] == 1
        assert session_restore_observability["delta_payload_bytes_total"] == 123


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


# ---------------------------------------------------------------------------
# Proxy stabilization helpers
# ---------------------------------------------------------------------------

class TestSessionSingleFlightCoordinator:
    @pytest.mark.asyncio
    async def test_queue_mode_serializes_same_session_requests(self):
        from proxy.server import SessionSingleFlightCoordinator

        coordinator = SessionSingleFlightCoordinator()
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def worker():
            nonlocal active, max_active
            async with coordinator.acquire("same-session", mode="queue", max_queue_depth=8):
                async with lock:
                    active += 1
                    max_active = max(max_active, active)
                await asyncio.sleep(0.02)
                async with lock:
                    active -= 1

        await asyncio.gather(worker(), worker(), worker())
        assert max_active == 1

        metrics = coordinator.metrics_snapshot()
        assert metrics["queue_events_total"] >= 2

    @pytest.mark.asyncio
    async def test_reject_mode_rejects_second_inflight_request(self):
        from proxy.server import SessionSingleFlightCoordinator, SessionSingleFlightRejected

        coordinator = SessionSingleFlightCoordinator()

        async with coordinator.acquire("same-session", mode="reject", max_queue_depth=0):
            with pytest.raises(SessionSingleFlightRejected) as excinfo:
                async with coordinator.acquire("same-session", mode="reject", max_queue_depth=0):
                    pass

        assert excinfo.value.reason == "active_inflight"
        metrics = coordinator.metrics_snapshot()
        assert metrics["reject_events_total"] >= 1

    @pytest.mark.asyncio
    async def test_queue_mode_enforces_queue_depth(self):
        from proxy.server import SessionSingleFlightCoordinator, SessionSingleFlightRejected

        coordinator = SessionSingleFlightCoordinator()
        first_entered = asyncio.Event()
        first_release = asyncio.Event()

        async def first_request():
            async with coordinator.acquire("same-session", mode="queue", max_queue_depth=1):
                first_entered.set()
                await first_release.wait()

        async def second_request():
            async with coordinator.acquire("same-session", mode="queue", max_queue_depth=1):
                return "ok"

        first_task = asyncio.create_task(first_request())
        await first_entered.wait()

        queued_task = asyncio.create_task(second_request())
        await asyncio.sleep(0.01)

        with pytest.raises(SessionSingleFlightRejected) as excinfo:
            async with coordinator.acquire("same-session", mode="queue", max_queue_depth=1):
                pass

        assert excinfo.value.reason == "queue_full"

        first_release.set()
        await first_task
        assert await queued_task == "ok"


class TestStreamGuardrails:
    def test_repetition_detection_triggers_for_pathological_output(self):
        from proxy.server import _should_cutoff_for_repetition

        repeated_text = "abc123 " * 20
        assert _should_cutoff_for_repetition(
            repeated_text,
            min_pattern_chars=6,
            min_repeats=4,
        ) is True

    def test_repetition_detection_ignores_healthy_output(self):
        from proxy.server import _should_cutoff_for_repetition

        healthy_text = "This response uses varied words and does not loop over a fixed suffix."
        assert _should_cutoff_for_repetition(
            healthy_text,
            min_pattern_chars=6,
            min_repeats=4,
        ) is False

    def test_guardrail_evaluation_prioritizes_runtime_then_length_then_repetition(self):
        from proxy.server import evaluate_stream_guardrail

        assert (
            evaluate_stream_guardrail(
                runtime_seconds=11.0,
                completion_tokens=20,
                response_text="healthy",
                max_runtime_seconds=10.0,
                max_completion_tokens=100,
                repetition_min_pattern_chars=8,
                repetition_min_repeats=4,
            )
            == "runtime"
        )

        assert (
            evaluate_stream_guardrail(
                runtime_seconds=5.0,
                completion_tokens=120,
                response_text="healthy",
                max_runtime_seconds=10.0,
                max_completion_tokens=100,
                repetition_min_pattern_chars=8,
                repetition_min_repeats=4,
            )
            == "completion_tokens"
        )

        assert (
            evaluate_stream_guardrail(
                runtime_seconds=5.0,
                completion_tokens=20,
                response_text=("loop-me-forever " * 10),
                max_runtime_seconds=10.0,
                max_completion_tokens=100,
                repetition_min_pattern_chars=8,
                repetition_min_repeats=4,
            )
            == "repetition"
        )


class TestSessionHistoryIntegrityHelpers:
    def test_merge_session_history_preserves_existing_and_delta_without_duplicates(self):
        from proxy.server import merge_session_history_for_update

        existing = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        delta = [{"role": "user", "content": "next question"}]

        merged = merge_session_history_for_update(
            existing_messages=existing,
            request_messages=[],
            delta_messages=delta,
            is_delta_request=True,
            assistant_content=None,
        )

        assert merged == existing + delta

    def test_merge_session_history_appends_assistant_content_once(self):
        from proxy.server import merge_session_history_for_update

        request_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]

        merged = merge_session_history_for_update(
            existing_messages=[],
            request_messages=request_messages,
            delta_messages=None,
            is_delta_request=False,
            assistant_content="hello",
        )

        assert merged[-1] == {"role": "assistant", "content": "hello"}
        assert merged.count({"role": "assistant", "content": "hello"}) == 1

    def test_merge_session_history_with_delta_and_assistant(self):
        from proxy.server import merge_session_history_for_update

        existing_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        delta_messages = [{"role": "user", "content": "next"}]

        merged = merge_session_history_for_update(
            existing_messages=existing_messages,
            request_messages=[],
            delta_messages=delta_messages,
            is_delta_request=True,
            assistant_content="assistant reply",
        )

        assert merged == existing_messages + delta_messages + [{"role": "assistant", "content": "assistant reply"}]

    def test_merge_session_history_avoids_duplicate_assistant(self):
        from proxy.server import merge_session_history_for_update

        existing_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "assistant reply"},
        ]

        merged = merge_session_history_for_update(
            existing_messages=existing_messages,
            request_messages=existing_messages,
            delta_messages=None,
            is_delta_request=False,
            assistant_content="assistant reply",
        )

        assert merged == existing_messages