"""Tests for slot save/restore failure-path instrumentation (LP-0MSI1RWLM007N367 F1).

Verifies that every failed slot save/restore logs:
- elapsed time at failure (elapsed=...)
- the applied adaptive timeout (timeout=...)
- a proxy-side busy-state snapshot (busy={...}) used to correlate
  ReadTimeouts with concurrent local load.

Also verifies the success path logs elapsed time at DEBUG, and unit-tests
the ``_slot_busy_state_snapshot`` helper that defines the busy signal for
the F3 load gate.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.refactor_parity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_srv(logger: logging.Logger, **attrs):
    """Build a minimal mock server object with a logger and optional http_client."""
    srv = MagicMock()
    srv.logger = logger
    srv._http_client = None
    for key, value in attrs.items():
        setattr(srv, key, value)
    return srv


@pytest.fixture(autouse=True)
def _clear_slot_registry():
    from proxy.session import _slot_owners
    _slot_owners.clear()
    yield
    _slot_owners.clear()


# ---------------------------------------------------------------------------
# AC1: failure path logs elapsed, timeout, busy state
# ---------------------------------------------------------------------------

class TestFailurePathInstrumentation:
    @pytest.mark.asyncio
    async def test_exception_failure_logs_elapsed_timeout_busy(self, caplog):
        """AC1: exception failure WARNING includes elapsed=, timeout=, busy=."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        mock_srv = _make_mock_srv(
            logging.getLogger("test_logger"),
            active_queries=4,
            local_active_queries=3,
            local_dispatch_records={"session-a": {"active": True}},
        )
        from proxy.session import _slot_owners
        _slot_owners[1] = "session-a"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=TimeoutError("read timed out"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 1, "save", "/tmp/test.bin", timeout=5.0
            )

        assert result is False
        warning_record = caplog.records[0]
        assert warning_record.levelname == "WARNING"
        assert "elapsed=" in warning_record.message
        assert "timeout=5.0s" in warning_record.message
        assert "busy=" in warning_record.message
        # Busy snapshot reflects the injected load state (slot 1 owned by an
        # active session).
        busy_json = warning_record.message.split("busy=", 1)[1]
        busy = json.loads(busy_json)
        assert busy["active_queries"] == 4
        assert busy["local_active_queries"] == 3
        assert busy["active_sessions"] == 1
        assert busy["slot_busy"] is True

    @pytest.mark.asyncio
    async def test_non200_failure_logs_elapsed_timeout_busy(self, caplog):
        """AC1: non-200 failure WARNING includes elapsed=, timeout=, busy=."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        response = MagicMock()
        response.status_code = 500
        response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        mock_srv = _make_mock_srv(
            logging.getLogger("test_logger"),
            active_queries=0,
            local_active_queries=0,
            local_dispatch_records={},
        )
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 2, "restore", "/tmp/test.bin", timeout=3.0
            )

        assert result is False
        warning_record = caplog.records[0]
        assert "slot_restore failed" in warning_record.message
        assert "timeout=3.0s" in warning_record.message
        assert "elapsed=" in warning_record.message
        assert "busy=" in warning_record.message
        busy_json = warning_record.message.split("busy=", 1)[1]
        busy = json.loads(busy_json)
        assert busy["active_queries"] == 0
        assert busy["slot_busy"] is False

    @pytest.mark.asyncio
    async def test_failure_preserves_existing_error_fields(self, caplog):
        """AC1: the pre-existing error=Type/detail fields are preserved."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.WARNING)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ValueError("something broke"))
        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 0, "save", "/tmp/test.bin", timeout=1.0
            )

        assert result is False
        warning_record = caplog.records[0]
        assert "error=ValueError/something broke" in warning_record.message
        assert "slot_save failed" in warning_record.message
        assert "slot=0" in warning_record.message


# ---------------------------------------------------------------------------
# Success path: DEBUG-level elapsed log, no WARNING
# ---------------------------------------------------------------------------

class TestSuccessPathInstrumentation:
    @pytest.mark.asyncio
    async def test_success_logs_elapsed_at_debug(self, caplog):
        """Success path emits a DEBUG 'slot_save ok' with elapsed/timeout and
        no WARNING record (production log volume unchanged at INFO)."""
        from proxy.session import _call_slot_endpoint

        caplog.set_level(logging.DEBUG)

        response = MagicMock()
        response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        mock_srv = _make_mock_srv(logging.getLogger("test_logger"))
        mock_srv._http_client = mock_client

        with patch("proxy.session._srv", return_value=mock_srv):
            result = await _call_slot_endpoint(
                1234, 1, "save", "/tmp/test.bin", timeout=2.0
            )

        assert result is True
        debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert len(debug_records) >= 1
        ok_record = next(r for r in debug_records if "slot_save ok" in r.message)
        assert "elapsed=" in ok_record.message
        assert "timeout=2.0s" in ok_record.message
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 0


# ---------------------------------------------------------------------------
# _slot_busy_state_snapshot unit tests
# ---------------------------------------------------------------------------

class TestSlotBusyStateSnapshot:
    def test_empty_state_returns_zeroed_snapshot(self):
        """A server with zeroed counters/records yields zeros and slot_busy False."""
        from proxy.session import _slot_busy_state_snapshot

        # Real servers always have these attributes initialized (server.py);
        # an idle server has zeroed counters and an empty records dict.
        mock_srv = _make_mock_srv(
            logging.getLogger("test_logger"),
            active_queries=0,
            local_active_queries=0,
            local_dispatch_records={},
        )

        with patch("proxy.session._srv", return_value=mock_srv):
            state = _slot_busy_state_snapshot(slot_id=1)

        assert state == {
            "active_queries": 0,
            "local_active_queries": 0,
            "active_sessions": 0,
            "slot_busy": False,
        }

    def test_snapshot_counts_active_dispatch_records(self):
        """active_sessions counts only records with active=True."""
        from proxy.session import _slot_busy_state_snapshot

        mock_srv = _make_mock_srv(
            logging.getLogger("test_logger"),
            active_queries=7,
            local_active_queries=2,
            local_dispatch_records={
                "session-a": {"active": True},
                "session-b": {"active": False},
                "session-c": {"active": True},
            },
        )

        with patch("proxy.session._srv", return_value=mock_srv):
            state = _slot_busy_state_snapshot(slot_id=None)

        assert state["active_queries"] == 7
        assert state["local_active_queries"] == 2
        assert state["active_sessions"] == 2

    def test_slot_busy_true_when_owner_active(self):
        """slot_busy is True when the slot's owning session has an active lease."""
        from proxy.session import _slot_busy_state_snapshot, _slot_owners

        _slot_owners[3] = "session-a"
        mock_srv = _make_mock_srv(
            logging.getLogger("test_logger"),
            local_dispatch_records={"session-a": {"active": True}},
        )

        with patch("proxy.session._srv", return_value=mock_srv):
            state = _slot_busy_state_snapshot(slot_id=3)

        assert state["slot_busy"] is True

    def test_slot_busy_false_when_owner_inactive(self):
        """slot_busy is False when the slot's owning session has no active lease."""
        from proxy.session import _slot_busy_state_snapshot, _slot_owners

        _slot_owners[3] = "session-a"
        mock_srv = _make_mock_srv(
            logging.getLogger("test_logger"),
            local_dispatch_records={
                "session-a": {"active": False},
                "session-b": {"active": True},
            },
        )

        with patch("proxy.session._srv", return_value=mock_srv):
            state = _slot_busy_state_snapshot(slot_id=3)

        assert state["slot_busy"] is False
        # Other sessions' activity is still visible in the load counters.
        assert state["active_sessions"] == 1

    def test_slot_busy_false_for_unassigned_slot(self):
        """An unassigned slot reports slot_busy False even under load."""
        from proxy.session import _slot_busy_state_snapshot

        mock_srv = _make_mock_srv(
            logging.getLogger("test_logger"),
            local_dispatch_records={"session-a": {"active": True}},
        )

        with patch("proxy.session._srv", return_value=mock_srv):
            state = _slot_busy_state_snapshot(slot_id=9)

        assert state["slot_busy"] is False
        assert state["active_sessions"] == 1
