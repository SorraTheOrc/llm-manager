"""Unit tests for SessionManager.

Covers: generation, acceptance, TTL expiry, eviction, delta computation,
invalidation, and fallback to full history for expired/invalid sessions.
"""

import asyncio
import time
import pytest

from proxy.session_manager import (
    Session,
    SessionManager,
    DEFAULT_SESSION_TTL_SECONDS,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_messages(n: int, prefix: str = "msg") -> list[dict]:
    """Return a list of n simple chat messages."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{prefix}-{i}"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

class TestSession:
    def test_default_fields(self):
        s = Session(session_id="abc")
        assert s.session_id == "abc"
        assert s.message_count == 0
        assert s.messages == []
        assert s.invalidated is False
        assert s.age_seconds >= 0
        assert s.idle_seconds >= 0

    def test_touch_updates_activity(self):
        s = Session(session_id="abc")
        # Set last_activity_at far in the past
        s.last_activity_at = time.monotonic() - 100  # 100 seconds ago
        idle_before = s.idle_seconds
        s.touch()
        # After touch, idle_seconds should be much smaller
        assert s.idle_seconds < idle_before

    def test_is_expired_within_ttl(self):
        s = Session(session_id="abc")
        assert s.is_expired(ttl=99999) is False

    def test_is_expired_beyond_ttl(self):
        s = Session(session_id="abc")
        assert s.is_expired(ttl=0) is True


# ---------------------------------------------------------------------------
# SessionManager – generation and acceptance
# ---------------------------------------------------------------------------

class TestSessionManagerGeneration:
    @pytest.mark.asyncio
    async def test_generate_uuid_when_no_header(self):
        mgr = SessionManager()
        session, created = await mgr.get_or_create(None)
        assert created is True
        assert session.session_id  # should be a UUID string
        # UUID v4 format check
        import uuid as _uuid

        assert _uuid.UUID(session.session_id).version == 4

    @pytest.mark.asyncio
    async def test_accept_client_session_id(self):
        mgr = SessionManager()
        session, created = await mgr.get_or_create("my-session-123")
        assert created is True
        assert session.session_id == "my-session-123"

    @pytest.mark.asyncio
    async def test_reuse_existing_session(self):
        mgr = SessionManager()
        session1, created1 = await mgr.get_or_create("abc")
        assert created1 is True
        session2, created2 = await mgr.get_or_create("abc")
        assert created2 is False
        assert session2.session_id == "abc"
        assert session2 is session1  # same object

    @pytest.mark.asyncio
    async def test_metrics_initialized(self):
        mgr = SessionManager()
        metrics = mgr.get_metrics()
        assert metrics["sessions_active"] == 0
        assert metrics["sessions_created_total"] == 0
        assert metrics["sessions_expired_total"] == 0

    @pytest.mark.asyncio
    async def test_metrics_after_creation(self):
        mgr = SessionManager()
        await mgr.get_or_create("s1")
        await mgr.get_or_create("s2")
        metrics = mgr.get_metrics()
        assert metrics["sessions_active"] == 2
        assert metrics["sessions_created_total"] == 2


# ---------------------------------------------------------------------------
# TTL expiry and eviction
# ---------------------------------------------------------------------------

class TestSessionManagerTTL:
    @pytest.mark.asyncio
    async def test_expired_session_is_evicted_on_get_or_create(self):
        mgr = SessionManager(ttl_seconds=0.01)  # 10ms TTL
        session, _ = await mgr.get_or_create("expiring")
        assert mgr.active_session_count == 1

        # Wait for expiry
        await asyncio.sleep(0.05)

        # The session should be evicted when we try to access it again
        session2, created = await mgr.get_or_create("expiring")
        assert created is True  # new session created
        assert session2 is not session  # different object
        assert mgr.total_sessions_expired == 1

    @pytest.mark.asyncio
    async def test_expired_session_returns_none_on_get(self):
        mgr = SessionManager(ttl_seconds=0.01)
        await mgr.get_or_create("expiring")
        await asyncio.sleep(0.05)

        result = await mgr.get("expiring")
        assert result is None
        assert mgr.total_sessions_expired == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_all_expired(self):
        mgr = SessionManager(ttl_seconds=0.01)
        await mgr.get_or_create("s1")
        await mgr.get_or_create("s2")
        await mgr.get_or_create("s3")
        await asyncio.sleep(0.05)

        evicted = await mgr.cleanup_expired()
        assert evicted == 3
        assert mgr.active_session_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_does_not_remove_active_sessions(self):
        mgr = SessionManager(ttl_seconds=3600)  # 1 hour
        await mgr.get_or_create("s1")
        evicted = await mgr.cleanup_expired()
        assert evicted == 0
        assert mgr.active_session_count == 1


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

class TestSessionManagerInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_marks_session(self):
        mgr = SessionManager()
        await mgr.get_or_create("abc")
        result = await mgr.invalidate("abc")
        assert result is True

    @pytest.mark.asyncio
    async def test_invalidated_session_creates_new_on_get_or_create(self):
        mgr = SessionManager()
        session1, created1 = await mgr.get_or_create("abc")
        assert created1 is True
        await mgr.invalidate("abc")

        session2, created2 = await mgr.get_or_create("abc")
        assert created2 is True  # new session created
        assert session2 is not session1

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_returns_false(self):
        mgr = SessionManager()
        result = await mgr.invalidate("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidated_session_returns_none_on_get(self):
        mgr = SessionManager()
        await mgr.get_or_create("abc")
        await mgr.invalidate("abc")
        result = await mgr.get("abc")
        assert result is None


# ---------------------------------------------------------------------------
# Message history and delta computation
# ---------------------------------------------------------------------------

class TestSessionManagerMessages:
    @pytest.mark.asyncio
    async def test_update_messages(self):
        mgr = SessionManager()
        session, _ = await mgr.get_or_create("s1")
        msgs = _make_messages(3)
        result = await mgr.update_messages("s1", msgs)
        assert result is True
        assert session.message_count == 3
        assert len(session.messages) == 3

    @pytest.mark.asyncio
    async def test_update_nonexistent_session(self):
        mgr = SessionManager()
        result = await mgr.update_messages("nonexistent", [])
        assert result is False

    @pytest.mark.asyncio
    async def test_append_messages(self):
        mgr = SessionManager()
        session, _ = await mgr.get_or_create("s1")
        await mgr.update_messages("s1", _make_messages(2))
        new_msgs = [{"role": "user", "content": "msg-2-extra"}]
        await mgr.append_messages("s1", new_msgs)
        assert session.message_count == 3
        assert session.messages[-1]["content"] == "msg-2-extra"


class TestDeltaComputation:
    def test_delta_empty_existing(self):
        mgr = SessionManager()
        incoming = _make_messages(3)
        delta, matches = mgr.compute_delta([], incoming)
        assert matches is True
        assert delta == incoming

    def test_delta_no_new_messages(self):
        mgr = SessionManager()
        messages = _make_messages(3)
        delta, matches = mgr.compute_delta(messages, messages)
        assert matches is True
        assert delta == []

    def test_delta_with_new_messages(self):
        mgr = SessionManager()
        existing = _make_messages(3)
        incoming = _make_messages(5)
        delta, matches = mgr.compute_delta(existing, incoming)
        assert matches is True
        assert len(delta) == 2
        assert delta[0]["content"] == "msg-3"
        assert delta[1]["content"] == "msg-4"

    def test_delta_edited_history_returns_full(self):
        mgr = SessionManager()
        existing = _make_messages(3)
        # Simulate an edit: change first message content
        incoming = list(existing)
        incoming[1] = {"role": "assistant", "content": "edited"}
        incoming.append({"role": "user", "content": "new-msg"})
        delta, matches = mgr.compute_delta(existing, incoming)
        assert matches is False
        # Delta should be the full incoming list since history was edited
        assert delta == incoming

    def test_delta_shorter_incoming_returns_full(self):
        mgr = SessionManager()
        existing = _make_messages(5)
        incoming = _make_messages(3)
        delta, matches = mgr.compute_delta(existing, incoming)
        # Incoming is shorter than existing – can't be a prefix match
        assert matches is False
        assert delta == incoming

    def test_delta_empty_incoming(self):
        mgr = SessionManager()
        existing = _make_messages(3)
        delta, matches = mgr.compute_delta(existing, [])
        assert matches is True
        assert delta == []

    def test_delta_both_empty(self):
        mgr = SessionManager()
        delta, matches = mgr.compute_delta([], [])
        assert matches is True
        assert delta == []

    def test_delta_role_mismatch(self):
        mgr = SessionManager()
        existing = [{"role": "user", "content": "hello"}]
        incoming = [{"role": "system", "content": "hello"}]
        delta, matches = mgr.compute_delta(existing, incoming)
        assert matches is False
        assert delta == incoming


# ---------------------------------------------------------------------------
# Session info and removal
# ---------------------------------------------------------------------------

class TestSessionManagerInfo:
    @pytest.mark.asyncio
    async def test_get_session_info(self):
        mgr = SessionManager()
        await mgr.get_or_create("test-s1")
        info = mgr.get_session_info("test-s1")
        assert info is not None
        assert info["session_id"] == "test-s1"
        assert info["message_count"] == 0
        assert info["invalidated"] is False
        assert isinstance(info["idle_seconds"], float)
        assert isinstance(info["age_seconds"], float)

    @pytest.mark.asyncio
    async def test_get_session_info_nonexistent(self):
        mgr = SessionManager()
        info = mgr.get_session_info("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_remove_session(self):
        mgr = SessionManager()
        await mgr.get_or_create("s1")
        removed = await mgr.remove("s1")
        assert removed is True
        assert mgr.active_session_count == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_session(self):
        mgr = SessionManager()
        removed = await mgr.remove("nonexistent")
        assert removed is False


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------

class TestSessionManagerCleanupTask:
    @pytest.mark.asyncio
    async def test_start_and_stop_cleanup_task(self):
        mgr = SessionManager(ttl_seconds=0.01)
        mgr.start_cleanup_task()
        assert mgr._cleanup_task is not None
        assert not mgr._cleanup_task.done()

        mgr.stop_cleanup_task()
        # Give the task a moment to cancel
        await asyncio.sleep(0.05)
        assert mgr._cleanup_task is None

    @pytest.mark.asyncio
    async def test_cleanup_task_evicts_expired(self):
        mgr = SessionManager(ttl_seconds=0.01, cleanup_interval_seconds=0.05)
        await mgr.get_or_create("s1")
        await mgr.get_or_create("s2")

        # Start cleanup task
        mgr.start_cleanup_task()

        # Wait for at least one cleanup cycle
        await asyncio.sleep(0.15)

        mgr.stop_cleanup_task()
        assert mgr.active_session_count == 0


# ---------------------------------------------------------------------------
# Fallback to full history for expired/invalid sessions
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    @pytest.mark.asyncio
    async def test_expired_session_falls_back_to_full_history(self):
        """When a session expires, get_or_create creates a new one.

        The proxy should detect this (created=True) and send full history.
        """
        mgr = SessionManager(ttl_seconds=0.01)
        session1, created1 = await mgr.get_or_create("s1")
        assert created1 is True

        # Store messages in session
        await mgr.update_messages("s1", _make_messages(3))

        # Wait for expiry
        await asyncio.sleep(0.05)

        # Accessing expired session creates a new one
        session2, created2 = await mgr.get_or_create("s1")
        assert created2 is True
        # New session has no messages – proxy must send full history
        assert session2.message_count == 0
        assert len(session2.messages) == 0

    @pytest.mark.asyncio
    async def test_invalidated_session_falls_back_to_full_history(self):
        """When a session is invalidated, get_or_create creates a new one."""
        mgr = SessionManager()
        session1, created1 = await mgr.get_or_create("s1")
        assert created1 is True

        await mgr.update_messages("s1", _make_messages(3))
        await mgr.invalidate("s1")

        session2, created2 = await mgr.get_or_create("s1")
        assert created2 is True
        assert session2.message_count == 0
        assert len(session2.messages) == 0