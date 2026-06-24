"""
Tests for prompt-cache / session reuse behavior.

Verifies that:
- Session continuity is preserved across multiple turns
- Session invalidation conditions are documented
- Full re-process is logged when cache invalidation occurs

Note: SessionManager methods are async; tests use @pytest.mark.asyncio.
"""

import pytest

from proxy.session_manager import Session, SessionManager
from proxy.session import _classify_delta_routing


# ===================================================================
# Session continuity tests
# ===================================================================


class TestSessionContinuity:
    """Tests that sessions are preserved across multiple turns."""

    @pytest.mark.asyncio
    async def test_session_preserved_across_turns(self):
        """A session persists across multiple turns."""
        mgr = SessionManager()
        session_id = "test-session-1"

        session, created = await mgr.get_or_create(session_id)
        assert created is True
        assert session.session_id == session_id

        session2, created = await mgr.get_or_create(session_id)
        assert created is False
        assert session2 is session

        session3, created = await mgr.get_or_create(session_id)
        assert created is False
        assert session3 is session

    @pytest.mark.asyncio
    async def test_different_sessions_independent(self):
        """Different sessions are independent."""
        mgr = SessionManager()

        s1, _ = await mgr.get_or_create("session-a")
        s2, _ = await mgr.get_or_create("session-b")

        assert s1 is not s2
        assert s1.session_id != s2.session_id

    @pytest.mark.asyncio
    async def test_session_invalidated_creates_new(self):
        """Invalidated sessions create new sessions on next request."""
        mgr = SessionManager()
        session_id = "test-session"

        session, _ = await mgr.get_or_create(session_id)
        original_created = session.created_at

        await mgr.invalidate(session_id)

        session2, created = await mgr.get_or_create(session_id)
        assert created is True
        assert session2.created_at > original_created

    @pytest.mark.asyncio
    async def test_session_expiry_ttl(self):
        """Expired sessions create new sessions."""
        mgr = SessionManager(ttl_seconds=0)
        session_id = "test-session"

        session, created = await mgr.get_or_create(session_id)
        assert created is True

        session2, created = await mgr.get_or_create(session_id)
        assert created is True
        assert session2 is not session


# ===================================================================
# Delta routing / cache invalidation tests
# ===================================================================


class TestDeltaRoutingConditions:
    """Tests for _classify_delta_routing cache invalidation conditions."""

    def test_history_mismatch_causes_full_reprocess(self):
        """History mismatch triggers full re-process."""
        use_delta, reason = _classify_delta_routing(
            history_matches=False,
            delta_message_count=5,
            restore_confirmed=True,
            require_restore_signal=False,
        )
        assert use_delta is False
        assert reason == "history_mismatch"

    def test_no_new_messages_causes_full_reprocess(self):
        """No new messages triggers full re-process."""
        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=0,
            restore_confirmed=True,
            require_restore_signal=False,
        )
        assert use_delta is False
        assert reason == "no_new_messages"

    def test_force_full_prompt_disables_delta(self):
        """force_full_prompt=True triggers full re-process."""
        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=5,
            restore_confirmed=True,
            require_restore_signal=False,
            force_full_prompt=True,
        )
        assert use_delta is False
        assert reason == "delta_disabled"

    def test_missing_restore_signal_causes_full_reprocess(self):
        """Missing restore signal triggers full re-process when required."""
        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=5,
            restore_confirmed=False,
            require_restore_signal=True,
        )
        assert use_delta is False
        assert reason == "missing_restore_signal"

    def test_all_conditions_met_allows_delta(self):
        """All conditions met allows delta routing."""
        use_delta, reason = _classify_delta_routing(
            history_matches=True,
            delta_message_count=5,
            restore_confirmed=True,
            require_restore_signal=False,
        )
        assert use_delta is True
        assert reason is None
