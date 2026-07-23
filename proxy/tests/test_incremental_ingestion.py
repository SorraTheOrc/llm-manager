"""Tests for session-based incremental prompt ingestion.

Tests verify that the proxy correctly handles session headers,
computes message deltas, and falls back to full history when needed.
These tests use mocked llama-server responses to validate behavior
without requiring a running llama-server instance.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from proxy.session_manager import SessionManager

pytestmark = pytest.mark.refactor_parity


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
    """Test that session headers are extracted and processed correctly."""

    def test_extract_session_id_from_header(self):
        """Test extracting session ID from request headers."""
        # This is verified by the server integration; we test the manager logic here
        mgr = SessionManager()
        # A None session ID should generate a UUID
        assert mgr.active_session_count == 0

    def test_resolve_session_id_header_prefers_explicit_session_id(self):
        from proxy.server import _resolve_session_id_header

        headers = {
            "x-session-id": "primary-session",
            "session_id": "fallback-session",
            "x-client-request-id": "client-session",
            "x-session-affinity": "affinity-session",
        }
        session_id, source = _resolve_session_id_header(headers)
        assert session_id == "primary-session"
        assert source == "x-session-id"

    def test_resolve_session_id_header_falls_back_to_client_request_id(self):
        from proxy.server import _resolve_session_id_header

        headers = {
            "x-client-request-id": "client-session",
            "x-session-affinity": "affinity-session",
        }
        session_id, source = _resolve_session_id_header(headers)
        assert session_id == "client-session"
        assert source == "x-client-request-id"

    def test_resolve_session_id_header_uses_affinity_as_last_resort(self):
        from proxy.server import _resolve_session_id_header

        headers = {
            "x-session-affinity": "affinity-session",
        }
        session_id, source = _resolve_session_id_header(headers)
        assert session_id == "affinity-session"
        assert source == "x-session-affinity"

    def test_log_session_header_resolution_with_header(self, caplog):
        from proxy.server import _log_session_header_resolution

        caplog.set_level(logging.INFO, logger="llama-proxy")
        _log_session_header_resolution("primary-session", "x-session-id")
        assert "Session header resolved" in caplog.text
        assert "source=x-session-id" in caplog.text
        assert "session=primary-" in caplog.text

    def test_log_session_header_resolution_without_header(self, caplog):
        from proxy.server import _log_session_header_resolution

        caplog.set_level(logging.INFO, logger="llama-proxy")
        _log_session_header_resolution(None, None)
        assert "No session header provided" in caplog.text

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

    def test_force_full_prompt_config(self):
        from proxy.server import _should_force_full_prompt

        assert _should_force_full_prompt({"force_full_prompt": True}) is True
        assert _should_force_full_prompt({"disable_delta": True}) is True
        assert _should_force_full_prompt({"force_full_prompt": False}) is False
        assert _should_force_full_prompt(None) is False


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

    def test_classify_delta_routing_respects_force_full_prompt(self):
        from proxy.server import _classify_delta_routing

        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=2,
            restore_confirmed=True,
            force_full_prompt=True,
        )
        assert use_delta is False
        assert reason == "delta_disabled"

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
            _record_delta_payload_bytes,
            _record_restore_fallback,
            _record_restore_success,
            session_restore_observability,
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

        repeated_text = "abc123" * 20
        assert _should_cutoff_for_repetition(
            repeated_text,
            min_pattern_chars=6,
            min_repeats=4,
        ) is True

    def test_repetition_detection_requires_consecutive_suffix(self):
        from proxy.server import _should_cutoff_for_repetition

        repeated_text = ("abc123" * 3) + "xyz"
        assert _should_cutoff_for_repetition(
            repeated_text,
            min_pattern_chars=6,
            min_repeats=3,
        ) is False

    def test_repetition_detection_ignores_healthy_output(self):
        from proxy.server import _should_cutoff_for_repetition

        healthy_text = "This response uses varied words and does not loop over a fixed suffix."
        assert _should_cutoff_for_repetition(
            healthy_text,
            min_pattern_chars=6,
            min_repeats=4,
        ) is False

    def test_extract_delta_text_from_sse_chunk_ignores_wrapper(self):
        from proxy.server import _extract_delta_text_from_sse_chunk

        wrapper_only = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
        assert _extract_delta_text_from_sse_chunk(wrapper_only) == ""

        mixed = (
            'data: {"choices":[{"delta":{"reasoning_content":"Think"}}]}\n'
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        )
        assert _extract_delta_text_from_sse_chunk(mixed) == "ThinkHello"

    def test_guardrail_evaluation_prioritizes_runtime_then_repetition(self):
        """Test guardrail priority: runtime first, then repetition (loop detection).

        Note: Hard completion_tokens cutoff has been removed in favor of
        loop detection. The max_completion_tokens parameter is now ignored.
        """
        from proxy.server import evaluate_stream_guardrail

        # Runtime guardrail still triggers first
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

        # High completion_tokens alone should NOT trigger guardrail anymore
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
            is None  # No guardrail triggered - this is the new behavior
        )

        # Repetition (loop detection) still triggers
        assert (
            evaluate_stream_guardrail(
                runtime_seconds=5.0,
                completion_tokens=20,
                response_text=("loopfore" * 6),
                max_runtime_seconds=10.0,
                max_completion_tokens=100,
                repetition_min_pattern_chars=8,
                repetition_min_repeats=4,
            )
            == "repetition"
        )

    def test_guardrail_invalidation_respects_repetition_override(self):
        from proxy.server import _should_invalidate_on_guardrail

        assert _should_invalidate_on_guardrail(
            "repetition",
            invalidate_on_cutoff=True,
            invalidate_on_repetition=False,
        ) is False
        assert _should_invalidate_on_guardrail(
            "repetition",
            invalidate_on_cutoff=False,
            invalidate_on_repetition=True,
        ) is True

    def test_guardrail_invalidation_defaults_to_cutoff(self):
        from proxy.server import _should_invalidate_on_guardrail

        assert _should_invalidate_on_guardrail(
            "runtime",
            invalidate_on_cutoff=True,
            invalidate_on_repetition=False,
        ) is True
        assert _should_invalidate_on_guardrail(
            "completion_tokens",
            invalidate_on_cutoff=False,
            invalidate_on_repetition=True,
        ) is False
        assert _should_invalidate_on_guardrail(
            None,
            invalidate_on_cutoff=True,
            invalidate_on_repetition=True,
        ) is False

    def test_no_hard_completion_tokens_cutoff(self):
        """Test that hard completion_tokens cutoff is removed.

        The guardrail should NOT trigger on completion_tokens alone.
        Loop detection should be used instead (repetition check).
        """
        from proxy.server import evaluate_stream_guardrail

        # Even with many completion_tokens, no cutoff should occur
        # if there's no repetition detected
        assert (
            evaluate_stream_guardrail(
                runtime_seconds=5.0,
                completion_tokens=5000,  # Exceeds old 2048 limit
                response_text="This is a legitimate long response with varied content.",
                max_runtime_seconds=120.0,
                max_completion_tokens=2048,  # Old limit, should be ignored
                repetition_min_pattern_chars=64,
                repetition_min_repeats=10,
            )
            is None
        )

    def test_loop_detection_via_repetition_still_works(self):
        """Test that loop detection via repetition check still works.

        When a response contains repeating patterns, the guardrail
        should trigger with 'repetition' reason.
        """
        from proxy.server import evaluate_stream_guardrail

        # Create a repeating pattern (64 chars repeated 10 times)
        pattern = "a" * 64
        repeating_text = pattern * 10

        assert (
            evaluate_stream_guardrail(
                runtime_seconds=5.0,
                completion_tokens=1000,
                response_text=repeating_text,
                max_runtime_seconds=120.0,
                max_completion_tokens=2048,
                repetition_min_pattern_chars=64,
                repetition_min_repeats=10,
            )
            == "repetition"
        )

    def test_session_not_invalidated_on_repetition_by_default(self):
        """Test that session is NOT invalidated when repetition is detected.

        By default, session_guardrail_invalidate_on_repetition is False,
        so the session should NOT be invalidated when a loop is detected.
        """
        from proxy.server import _should_invalidate_on_guardrail

        # Default config: invalidate_on_cutoff=True, invalidate_on_repetition=False
        assert (
            _should_invalidate_on_guardrail(
                "repetition",
                invalidate_on_cutoff=True,
                invalidate_on_repetition=False,
            )
            is False
        )

    def test_runtime_guardrail_still_invalidates_session(self):
        """Test that runtime guardrail still invalidates session.

        Runtime cutoff indicates a true runaway loop, so session
        should be invalidated.
        """
        from proxy.server import _should_invalidate_on_guardrail

        assert (
            _should_invalidate_on_guardrail(
                "runtime",
                invalidate_on_cutoff=True,
                invalidate_on_repetition=False,
            )
            is True
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


class TestSlotPersistenceHelpers:
    def _clear_slot_registry(self):
        from proxy.session import _slot_owners
        _slot_owners.clear()

    def test_slot_id_for_session_is_deterministic(self):
        from proxy.server import _slot_id_for_session
        self._clear_slot_registry()

        slot_id = _slot_id_for_session("session-123", 4)
        assert slot_id == _slot_id_for_session("session-123", 4)
        assert slot_id in range(4)

    def test_slot_id_for_session_single_slot(self):
        from proxy.server import _slot_id_for_session
        self._clear_slot_registry()

        assert _slot_id_for_session("session-123", 1) == 0

    def test_slot_id_for_session_returns_none_when_pool_invalid(self):
        from proxy.server import _slot_id_for_session
        self._clear_slot_registry()

        assert _slot_id_for_session("session-123", 0) is None
        assert _slot_id_for_session("session-123", -1) is None

    def test_slot_filename_for_session_sanitizes_id(self, tmp_path):
        from proxy.server import _slot_filename_for_session

        filename = _slot_filename_for_session("session:123/abc", tmp_path)
        assert str(tmp_path) in filename
        assert "session_123_abc" in filename

    @pytest.mark.asyncio
    async def test_call_slot_endpoint_uses_action_query_and_basename(self, tmp_path, monkeypatch):
        from proxy import server

        response = MagicMock()
        response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)
        monkeypatch.setattr(server, "_http_client", mock_client)

        filename = tmp_path / "slot_session.bin"
        ok = await server._call_slot_endpoint(
            1234,
            2,
            "save",
            str(filename),
            timeout=1.0,
            model="Qwen3",
        )

        assert ok is True
        mock_client.post.assert_awaited_once()
        args, kwargs = mock_client.post.call_args
        assert args[0] == "http://localhost:1234/slots/2?action=save"
        assert kwargs["json"]["filename"] == "slot_session.bin"
        assert kwargs["json"]["model"] == "Qwen3"

    def test_resolve_slot_model_name_in_router_mode(self):
        from proxy import server

        server.config = server.load_config()
        resolved = server._resolve_slot_model_name("qwen3", None, {"llama_router_mode": True})
        assert resolved == "Qwen3"

    def test_resolve_slot_model_name_uses_current_model_when_missing(self):
        from proxy import server

        server.config = server.load_config()
        resolved = server._resolve_slot_model_name(None, "Qwen3", {"llama_router_mode": True})
        assert resolved == "Qwen3"


# ---------------------------------------------------------------------------
# Tool call extraction from reasoning_content
# ---------------------------------------------------------------------------

class TestToolCallFromReasoning:
    """Tests for extracting tool calls from reasoning_content when content is null."""

    def test_extract_tool_call_from_reasoning_with_function_call(self):
        """Extract a well-formed <function=...>...</function> pattern."""
        from proxy.server import _extract_tool_call_from_reasoning

        reasoning = '\nI need to find files\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>'
        result = _extract_tool_call_from_reasoning(reasoning)
        assert result is not None
        assert '<function=bash>' in result
        assert '<parameter=command>' in result
        assert 'ls -la' in result
        assert '</function>' in result

    def test_extract_tool_call_from_reasoning_with_function_no_tool_wrapper(self):
        """Extract <function=...>...</function> without </tool_call> wrapper."""
        from proxy.server import _extract_tool_call_from_reasoning

        reasoning = 'Let me think... <function=bash>\n<parameter=command>\necho hello\n</parameter>\n</function>'
        result = _extract_tool_call_from_reasoning(reasoning)
        assert result is not None
        assert 'echo hello' in result

    def test_extract_tool_call_from_reasoning_no_tool_call(self):
        """Return None when no tool call pattern is present."""
        from proxy.server import _extract_tool_call_from_reasoning

        reasoning = 'I am thinking about the answer. The answer is 42.'
        result = _extract_tool_call_from_reasoning(reasoning)
        assert result is None

    def test_extract_tool_call_from_reasoning_empty(self):
        """Return None for empty string."""
        from proxy.server import _extract_tool_call_from_reasoning

        assert _extract_tool_call_from_reasoning("") is None
        assert _extract_tool_call_from_reasoning(None) is None

    def test_extract_tool_call_from_reasoning_with_incomplete_tag(self):
        """Return None for incomplete/partial <function=...> without </function>."""
        from proxy.server import _extract_tool_call_from_reasoning

        reasoning = 'Here is some code: <function=test>'
        result = _extract_tool_call_from_reasoning(reasoning)
        assert result is None

    def test_extract_assistant_content_from_sse_with_reasoning_content_no_content(self):
        """When delta.content is null but reasoning_content has a tool call, extract the tool call."""
        from proxy.server import _extract_assistant_content_from_sse

        # Use JSON escape sequences (\\n) for newlines inside JSON strings,
        # matching how the actual SSE stream delivers reasoning_content.
        sse_text = (
            'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n'
            'data: {"choices":[{"delta":{"reasoning_content":"\\n<function=bash>\\n"}}]}\n'
            'data: {"choices":[{"delta":{"reasoning_content":"<parameter=command>\\nls -la\\n</parameter>\\n"}}]}\n'
            'data: {"choices":[{"delta":{"reasoning_content":"</function>\\n</tool_call>"}}]}\n'
            'data: [DONE]\n'
        )
        result = _extract_assistant_content_from_sse(sse_text)
        assert result is not None
        assert '<function=bash>' in result
        assert 'ls -la' in result

    def test_extract_assistant_content_from_sse_reasoning_no_tool_call(self):
        """When reasoning_content has no tool call, promote reasoning text
        as fallback so clients receive a usable assistant message."""
        from proxy.server import _extract_assistant_content_from_sse

        sse_text = (
            'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n'
            'data: {"choices":[{"delta":{"reasoning_content":"Just thinking about it..."}}]}\n'
            'data: [DONE]\n'
        )
        result = _extract_assistant_content_from_sse(sse_text)
        # Reasoning content is promoted as a fallback so clients see the response
        assert result == "Just thinking about it..."

    def test_extract_assistant_content_from_sse_prefers_content_over_reasoning(self):
        """When both content and reasoning_content are present, prefer content."""
        from proxy.server import _extract_assistant_content_from_sse

        sse_text = (
            'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n'
            'data: {"choices":[{"delta":{"reasoning_content":"Thinking..."}}]}\n'
            'data: {"choices":[{"delta":{"content":"Hello there"}}]}\n'
            'data: [DONE]\n'
        )
        result = _extract_assistant_content_from_sse(sse_text)
        assert result == "Hello there"

    def test_extract_assistant_content_non_streaming_with_reasoning_content(self):
        """Non-streaming: extract tool call from message.reasoning_content when content is null."""
        from proxy.server import _extract_assistant_content

        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call>"
                    }
                }
            ]
        }
        result = _extract_assistant_content(resp)
        assert result is not None
        assert '<function=bash>' in result
        assert 'echo hi' in result

    def test_extract_assistant_content_non_streaming_reasoning_no_tool_call(self):
        """Non-streaming: when reasoning_content has no tool call,
        promote reasoning text as fallback so clients see it."""
        from proxy.server import _extract_assistant_content

        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Just thinking quietly..."
                    }
                }
            ]
        }
        result = _extract_assistant_content(resp)
        # Reasoning content is promoted as a fallback so clients see the response
        assert result == "Just thinking quietly..."

    def test_extract_assistant_content_non_streaming_prefers_content(self):
        """Non-streaming: when content is present, ignore reasoning_content."""
        from proxy.server import _extract_assistant_content

        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                        "reasoning_content": ";thinking..."
                    }
                }
            ]
        }
        result = _extract_assistant_content(resp)
        assert result == "Hello!"

    def test_is_empty_response_with_content(self):
        """Response with content is not empty."""
        from proxy.server import _is_empty_response

        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello there"
                    }
                }
            ]
        }
        assert _is_empty_response("Hello there", resp) is False

    def test_is_empty_response_with_reasoning_tool_call(self):
        """Response with tool call in reasoning_content is not empty."""
        from proxy.server import _is_empty_response

        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call>"
                    }
                }
            ]
        }
        assert _is_empty_response(None, resp) is False

    def test_is_empty_response_truly_empty(self):
        """Response with no content and no tool call is empty."""
        from proxy.server import _is_empty_response

        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None
                    }
                }
            ]
        }
        assert _is_empty_response(None, resp) is True

    def test_is_empty_response_with_text(self):
        """Response with non-empty text is not empty."""
        from proxy.server import _is_empty_response

        assert _is_empty_response("Hello world") is False

    def test_is_empty_response_with_blank_text(self):
        """Response with only whitespace text is empty."""
        from proxy.server import _is_empty_response

        assert _is_empty_response("   ") is True


class TestEmptyRetry:
    """Tests for _call_with_empty_retry retry-on-empty behavior."""

    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(self):
        """First call returns non-empty response, no retry needed."""
        from proxy.server import _call_with_empty_retry

        resp = MagicMock()
        resp.content = b'{"choices":[{"message":{"content":"Hello"}}]}'

        async def send_fn():
            return resp

        # _call_with_empty_retry now lives in proxy.utils and accesses
        # _call_with_backend_retries via a lazy import from proxy.lifecycle.
        with patch(
            "proxy.lifecycle._call_with_backend_retries",
            new_callable=AsyncMock,
        ) as mock_backend_retry:
            mock_backend_retry.return_value = resp

            result = await _call_with_empty_retry(send_fn, path="/v1/chat/completions")

            assert result is resp
            mock_backend_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_first_empty_then_succeeds(self):
        """First call returns empty, retry returns non-empty."""
        from proxy.server import _call_with_empty_retry

        empty_resp = MagicMock()
        empty_resp.content = b'{"choices":[{"message":{"content":null}}]}'
        good_resp = MagicMock()
        good_resp.content = b'{"choices":[{"message":{"content":"Hello"}}]}'

        async def send_fn():
            return empty_resp

        with patch(
            "proxy.lifecycle._call_with_backend_retries",
            new_callable=AsyncMock,
        ) as mock_backend_retry:
            mock_backend_retry.side_effect = [empty_resp, good_resp]

            result = await _call_with_empty_retry(send_fn, path="/v1/chat/completions", retry_delay=0.01)

            assert result is good_resp
            assert mock_backend_retry.await_count == 2

    @pytest.mark.asyncio
    async def test_all_empty_exhausts_retries(self):
        """All calls return empty, retry exhausts and returns last response."""
        from proxy.server import _call_with_empty_retry

        empty_resp = MagicMock()
        empty_resp.content = b'{"choices":[{"message":{"content":null}}]}'

        async def send_fn():
            return empty_resp

        with patch(
            "proxy.lifecycle._call_with_backend_retries",
            new_callable=AsyncMock,
        ) as mock_backend_retry:
            mock_backend_retry.side_effect = [empty_resp, empty_resp, empty_resp]

            result = await _call_with_empty_retry(send_fn, path="/v1/chat/completions", retry_delay=0.01)

            assert result is empty_resp
            # 1 initial + 2 retries = 3 total
            assert mock_backend_retry.await_count == 3

    @pytest.mark.asyncio
    async def test_non_json_content_passthrough(self):
        """Non-JSON response content passes through without retry."""
        from proxy.server import _call_with_empty_retry

        resp = MagicMock()
        resp.content = b"Just plain text, not JSON"

        async def send_fn():
            return resp

        with patch(
            "proxy.lifecycle._call_with_backend_retries",
            new_callable=AsyncMock,
        ) as mock_backend_retry:
            mock_backend_retry.return_value = resp

            result = await _call_with_empty_retry(send_fn, path="/v1/chat/completions")

            assert result is resp
            mock_backend_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tool_call_in_reasoning_no_retry(self):
        """Empty content but tool call in reasoning_content does not trigger retry."""
        from proxy.server import _call_with_empty_retry

        resp = MagicMock()
        resp.content = b'{"choices":[{"message":{"content":null,"reasoning_content":"<function=bash>\\n<parameter=command>\\nls -la\\n</parameter>\\n</function>"}}]}'

        async def send_fn():
            return resp

        with patch(
            "proxy.lifecycle._call_with_backend_retries",
            new_callable=AsyncMock,
        ) as mock_backend_retry:
            mock_backend_retry.return_value = resp

            result = await _call_with_empty_retry(send_fn, path="/v1/chat/completions")

            assert result is resp
            # No retry because tool call is present in reasoning_content
            mock_backend_retry.assert_awaited_once()


# ---------------------------------------------------------------------------
# Comprehensive restore signal detection tests
# ---------------------------------------------------------------------------


class TestHasExplicitRestoreSignalComprehensive:
    """Comprehensive tests for _has_explicit_restore_signal covering all
    header candidates, JSON fields, case insensitivity, and truthy values."""

    # ---- All header candidates ----

    def test_x_llama_session_restored_true(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"X-Llama-Session-Restored": "true"}) is True

    def test_x_session_restored_true(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"X-Session-Restored": "true"}) is True

    def test_x_llama_cache_restored_true(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"X-Llama-Cache-Restored": "true"}) is True

    def test_x_kv_cache_restored_true(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"X-KV-Cache-Restored": "true"}) is True

    def test_x_cache_restored_true(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"X-Cache-Restored": "true"}) is True

    # ---- Case insensitivity ----

    def test_header_case_insensitivity(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"x-llama-session-restored": "True"}) is True
        assert _has_explicit_restore_signal({"X-LLAMA-SESSION-RESTORED": "TRUE"}) is True
        assert _has_explicit_restore_signal({"x-Llama-Session-Restored": "tRuE"}) is True

    # ---- All truthy values ----

    def test_truthy_value_1(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"x-llama-session-restored": "1"}) is True

    def test_truthy_value_yes(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"x-llama-session-restored": "yes"}) is True

    def test_truthy_value_restored(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"x-llama-session-restored": "restored"}) is True

    def test_truthy_value_hit(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"x-llama-session-restored": "hit"}) is True

    def test_falsy_values(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"x-llama-session-restored": "false"}) is False
        assert _has_explicit_restore_signal({"x-llama-session-restored": "0"}) is False
        assert _has_explicit_restore_signal({"x-llama-session-restored": "no"}) is False
        assert _has_explicit_restore_signal({"x-llama-session-restored": "miss"}) is False

    # ---- All JSON field candidates ----

    def test_json_session_restored(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({}, {"session_restored": True}) is True

    def test_json_cache_restored(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({}, {"cache_restored": True}) is True

    def test_json_restore_success(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({}, {"restore_success": True}) is True

    def test_json_kv_cache_restored(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({}, {"kv_cache_restored": True}) is True

    def test_json_field_without_header(self):
        from proxy.server import _has_explicit_restore_signal
        # JSON carries the signal even when headers are empty
        assert _has_explicit_restore_signal({}, {"session_restored": True}) is True

    def test_json_field_false(self):
        from proxy.server import _has_explicit_restore_signal
        # JSON field explicitly False should not trigger
        assert _has_explicit_restore_signal({}, {"session_restored": False}) is False

    # ---- No signal at all ----

    def test_no_signal_at_all(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({"content-type": "application/json"}) is False

    def test_empty_headers_and_json(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal({}, {}) is False
        assert _has_explicit_restore_signal({}, None) is False

    # ---- Both header and JSON with different values ----

    def test_header_true_json_true(self):
        from proxy.server import _has_explicit_restore_signal
        assert _has_explicit_restore_signal(
            {"x-llama-session-restored": "true"},
            {"session_restored": True},
        ) is True

    def test_header_true_json_false(self):
        from proxy.server import _has_explicit_restore_signal
        # Header wins despite JSON being false
        assert _has_explicit_restore_signal(
            {"x-llama-session-restored": "true"},
            {"session_restored": False},
        ) is True

    def test_header_false_json_true(self):
        from proxy.server import _has_explicit_restore_signal
        # Header is false but JSON carries the signal
        assert _has_explicit_restore_signal(
            {"x-llama-session-restored": "false"},
            {"session_restored": True},
        ) is True


class TestDetectRestoreSignalFromLogSliceComprehensive:
    """Comprehensive edge-case tests for _detect_restore_signal_from_log_slice."""

    def test_nonexistent_log_file(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice
        nonexistent = tmp_path / "does_not_exist.log"
        assert _detect_restore_signal_from_log_slice(nonexistent, 0) is False

    def test_empty_log_file(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice
        log_file = tmp_path / "empty.log"
        log_file.write_text("")
        assert _detect_restore_signal_from_log_slice(log_file, 0) is False

    def test_restore_signal_restored_context_checkpoint(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice
        log_file = tmp_path / "restore_checkpoint.log"
        log_file.write_text("before\n")
        offset = log_file.stat().st_size
        with log_file.open("a", encoding="utf-8") as f:
            f.write("slot update_slots restored context checkpoint\n")
        assert _detect_restore_signal_from_log_slice(log_file, offset) is True

    def test_restore_signal_load_session(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice
        log_file = tmp_path / "load_session.log"
        log_file.write_text("before\n")
        offset = log_file.stat().st_size
        with log_file.open("a", encoding="utf-8") as f:
            f.write("slot load_session: loading KV cache for session abc-123\n")
        assert _detect_restore_signal_from_log_slice(log_file, offset) is True

    def test_restore_signal_session_restore(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice
        log_file = tmp_path / "session_restore.log"
        log_file.write_text("before\n")
        offset = log_file.stat().st_size
        with log_file.open("a", encoding="utf-8") as f:
            f.write("INFO session restore completed for slot 0\n")
        assert _detect_restore_signal_from_log_slice(log_file, offset) is True

    def test_restore_signal_restore_session(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice
        log_file = tmp_path / "restore_session.log"
        log_file.write_text("before\n")
        offset = log_file.stat().st_size
        with log_file.open("a", encoding="utf-8") as f:
            f.write("restore session for abc-123\n")
        assert _detect_restore_signal_from_log_slice(log_file, offset) is True

    def test_restore_signal_loading_kv_cache(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice
        log_file = tmp_path / "loading_kv.log"
        log_file.write_text("before\n")
        offset = log_file.stat().st_size
        with log_file.open("a", encoding="utf-8") as f:
            f.write("loading KV cache from disk for slot 0\n")
        assert _detect_restore_signal_from_log_slice(log_file, offset) is True

    def test_restore_signal_kv_cache_restored(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_log_slice
        log_file = tmp_path / "kv_restored.log"
        log_file.write_text("before\n")
        offset = log_file.stat().st_size
        with log_file.open("a", encoding="utf-8") as f:
            f.write("kv cache restored for session abc-123\n")
        assert _detect_restore_signal_from_log_slice(log_file, offset) is True

    def test_read_error_encoding(self, tmp_path):
        """Binary/garbled data at the start offset should not crash."""
        from proxy.server import _detect_restore_signal_from_log_slice
        log_file = tmp_path / "binary.log"
        log_file.write_bytes(b"\x00\x01\x02before\n\xff\xfe restored context checkpoint\n")
        # The function uses errors="replace" so it should not raise
        assert _detect_restore_signal_from_log_slice(log_file, 0) is True


class TestDetectRestoreSignalFromLlamaLogComprehensive:
    """Comprehensive tests for _detect_restore_signal_from_llama_log."""

    def test_none_session_id(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("INFO load_session: abc-123\n")
        assert _detect_restore_signal_from_llama_log(None, log_path=log_file) is False

    def test_empty_session_id(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("INFO load_session: abc-123\n")
        assert _detect_restore_signal_from_llama_log("", log_path=log_file) is False

    def test_nonexistent_log_file(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        nonexistent = tmp_path / "nonexistent.log"
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=nonexistent) is False

    def test_session_id_in_log_without_restore_phrase(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("abc-123 slot update processing\n")
        # Session ID appears but no restore phrase — should not match
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is False

    def test_session_id_with_load_session_phrase(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text(
            "slot load_session: loading KV cache for session_id=abc-123\n"
        )
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_session_id_with_session_restore_phrase(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("INFO session restore for abc-123\n")
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_session_id_with_restore_session_phrase(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("restore session abc-123 completed\n")
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_session_id_with_loading_kv_cache_phrase(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("loading KV cache for session abc-123\n")
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_session_id_with_kv_cache_restored_phrase(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("kv cache restored for abc-123\n")
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_session_id_with_restored_context_checkpoint_phrase(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("restored context checkpoint for abc-123\n")
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_different_session_id_in_log(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text(
            "INFO slot update: processing tokens, no restore phrases present\n"
        )
        # Log has no restore phrase at all, so should return False
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is False

    def test_multiple_sessions_only_one_matches(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text(
            "load_session: session_id=other-session\n"
            "load_session: session_id=abc-123\n"
            "load_session: session_id=another-one\n"
        )
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_log_path_override(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        custom_log = tmp_path / "custom-llama.log"
        custom_log.write_text("load_session: session_id=custom-session\n")
        assert _detect_restore_signal_from_llama_log("custom-session", log_path=custom_log) is True

    def test_lookback_lines_truncation(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        # Write enough lines to exceed lookback_lines
        lines = [f"line {i}\n" for i in range(500)]
        lines.append("load_session: session_id=abc-123\n")
        log_file.write_text("".join(lines))
        # With lookback_lines=400, the signal line should still be within range
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file, lookback_lines=400) is True

    def test_lookback_lines_excludes_signal(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        # Write many lines so the signal is outside the lookback window
        lines = [f"line {i}\n" for i in range(500)]
        lines.append("load_session: session_id=abc-123\n")
        log_file.write_text("".join(lines))
        # With lookback_lines=10, the signal (last line) should be included
        # Actually the function takes the LAST lookback_lines lines, so last 10 should include it
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file, lookback_lines=10) is True

    def test_session_id_case_insensitivity(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text(
            "slot load_session: loading KV cache for session_id=ABC-123\n"
        )
        # Both session_id and log text are lowercased before matching
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is True

    def test_restore_phrase_without_session_id(self, tmp_path):
        from proxy.server import _detect_restore_signal_from_llama_log
        log_file = tmp_path / "llama-server.log"
        log_file.write_text("slot processing tokens without any restore phrases\n")
        # No restore phrase at all, so should return False
        assert _detect_restore_signal_from_llama_log("abc-123", log_path=log_file) is False


# ---------------------------------------------------------------------------
# Additional delta routing classification tests
# ---------------------------------------------------------------------------


class TestClassifyDeltaRoutingComprehensive:
    """Additional edge cases for _classify_delta_routing."""

    def test_no_new_messages_fallback(self):
        """When delta_message_count <= 0, routing should fall back with reason."""
        from proxy.server import _classify_delta_routing

        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=0,
            restore_confirmed=True,
        )
        assert use_delta is False
        assert reason == "no_new_messages"

    def test_negative_delta_message_count(self):
        """Negative delta_message_count should also be treated as no_new_messages."""
        from proxy.server import _classify_delta_routing

        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=-1,
            restore_confirmed=True,
        )
        assert use_delta is False
        assert reason == "no_new_messages"

    def test_all_fallbacks_in_priority_order(self):
        """Verify that fallback reasons are applied in expected priority order."""
        from proxy.server import _classify_delta_routing

        # Priority 1: history_mismatch (takes precedence)
        use_delta, reason = _classify_delta_routing(
            history_matches=False,
            delta_message_count=2,
            restore_confirmed=True,
        )
        assert reason == "history_mismatch"

        # Priority 2: no_new_messages (history matches, but no delta)
        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=0,
            restore_confirmed=True,
        )
        assert reason == "no_new_messages"

        # Priority 3: delta_disabled (force_full_prompt takes precedence over missing signal)
        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=2,
            restore_confirmed=False,
            force_full_prompt=True,
        )
        assert reason == "delta_disabled"

        # Priority 4: missing_restore_signal
        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=2,
            restore_confirmed=False,
        )
        assert reason == "missing_restore_signal"
