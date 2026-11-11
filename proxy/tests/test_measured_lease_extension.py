"""Measured lease extension and early abort (LP-0MW8VOKB9002W2C1).

Parent LP-0MW7V2DK400365ER AC2/AC4/AC14.  With measured prefill progress now
available (LP-0MW8VO2HT007Y1VQ), the dispatch lease extension can be driven by
observed tokens, and a genuinely failing prefill can be aborted *before* the
hard deadline with a distinct, diagnosable reason.

This module pins:

* ``lease_extended_on_measured_progress`` is logged (and the no-progress clock
  refreshed) when the measured progress advances;
* ``_watchdog_release_reason`` returns the distinct reason
  ``prefill_eta_exceeded`` / ``prefill_stalled`` / ``prefill_max_exceeded``;
* ``local_prefill_eta_factor`` (default 2x) bounds a runaway prefill relative
  to its admission ``eta_prefill_s``;
* the cleanup sweep records the reason and elapsed seconds (never a blind kill
  at the hard deadline first);
* the ``_watchdog_no_progress`` boolean contract is unchanged for existing
  callers.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_prefill_lease_extension import _info_log_lines, _make_srv


def _srv_with(records, **server_overrides):
    config = {
        "server": {
            "local_dispatch_no_progress_timeout_seconds": 180,
            "local_dispatch_max_prefill_seconds": 900,
            "local_prefill_eta_factor": 2.0,
            "local_dispatch_lease_prefill_poll_seconds": 0.05,
            "local_dispatch_lease_prefill_buffer_seconds": 0.5,
            "llama_server_port": 8080,
            "session_slot_pool_size": 2,
        }
    }
    config["server"].update(server_overrides)
    return _make_srv(records=records, config=config)


# ===========================================================================
# _watchdog_release_reason — distinct reasons
# ===========================================================================


class TestWatchdogReleaseReason:
    def test_eta_exceeded_is_earliest_evidence(self):
        """Elapsed > eta * factor releases with prefill_eta_exceeded even when
        the no-progress clock is fresh (the prefill is progressing, just far
        slower than priced)."""
        from proxy.router_helpers import _watchdog_release_reason

        now = time.monotonic()
        srv = _srv_with(
            {
                "s": {
                    "active": True,
                    "started_at": now - 100.0,
                    "last_progress_ts": now,  # progress is fresh
                    "eta_prefill_s": 30.0,  # 100s elapsed > 30 * 2
                }
            }
        )
        assert (
            _watchdog_release_reason(srv, srv.local_dispatch_records["s"])
            == "prefill_eta_exceeded"
        )

    def test_eta_bound_respects_factor(self):
        """A raised factor keeps the same record alive."""
        from proxy.router_helpers import _watchdog_release_reason

        now = time.monotonic()
        srv = _srv_with(
            {
                "s": {
                    "active": True,
                    "started_at": now - 100.0,
                    "last_progress_ts": now,
                    "eta_prefill_s": 30.0,
                }
            },
            local_prefill_eta_factor=10.0,  # 100s < 30 * 10
        )
        assert _watchdog_release_reason(srv, srv.local_dispatch_records["s"]) is None

    def test_stalled_reason_when_no_progress(self):
        """No progress for longer than the timeout releases as prefill_stalled."""
        from proxy.router_helpers import _watchdog_release_reason

        now = time.monotonic()
        srv = _srv_with(
            {
                "s": {
                    "active": True,
                    "started_at": now - 200.0,
                    "last_progress_ts": now - 200.0,
                }
            }
        )
        assert (
            _watchdog_release_reason(srv, srv.local_dispatch_records["s"])
            == "prefill_stalled"
        )

    def test_max_prefill_reason_is_the_hard_backstop(self):
        """A progressing record past the max-prefill ceiling is released as
        prefill_max_exceeded (the backstop)."""
        from proxy.router_helpers import _watchdog_release_reason

        now = time.monotonic()
        srv = _srv_with(
            {
                "s": {
                    "active": True,
                    "started_at": now - 1000.0,
                    "last_progress_ts": now,  # progressing
                }
            }
        )
        assert (
            _watchdog_release_reason(srv, srv.local_dispatch_records["s"])
            == "prefill_max_exceeded"
        )

    def test_healthy_record_has_no_reason(self):
        from proxy.router_helpers import _watchdog_release_reason

        now = time.monotonic()
        srv = _srv_with(
            {
                "s": {
                    "active": True,
                    "started_at": now - 5.0,
                    "last_progress_ts": now - 1.0,
                    "eta_prefill_s": 60.0,
                }
            }
        )
        assert _watchdog_release_reason(srv, srv.local_dispatch_records["s"]) is None

    def test_legacy_record_without_baseline_has_no_reason(self):
        from proxy.router_helpers import _watchdog_release_reason

        srv = _srv_with({"s": {"active": True, "started_at": time.monotonic()}})
        assert _watchdog_release_reason(srv, srv.local_dispatch_records["s"]) is None

    def test_boolean_wrapper_matches_reason(self):
        """_watchdog_no_progress stays a faithful boolean of the reason."""
        from proxy.router_helpers import (
            _watchdog_no_progress,
            _watchdog_release_reason,
        )

        now = time.monotonic()
        srv = _srv_with(
            {
                "stalled": {
                    "active": True,
                    "started_at": now - 200.0,
                    "last_progress_ts": now - 200.0,
                },
                "healthy": {
                    "active": True,
                    "started_at": now,
                    "last_progress_ts": now,
                },
            }
        )
        for key in ("stalled", "healthy"):
            record = srv.local_dispatch_records[key]
            assert _watchdog_no_progress(srv, key, record) is (
                _watchdog_release_reason(srv, record) is not None
            )


# ===========================================================================
# _extend_lease_during_prefill — measured extension
# ===========================================================================


class TestMeasuredExtension:
    @pytest.mark.asyncio
    async def test_advancing_progress_logs_measured_reason(self, monkeypatch):
        """An advancing measured progress logs lease_extended_on_measured_progress
        and a reason=measured_progress extension."""
        from proxy.router_helpers import _extend_lease_during_prefill

        seq = iter([(1000, True), (2500, True)])

        async def _fake(*a, **k):
            return next(seq)

        monkeypatch.setattr(
            "proxy.router_helpers._query_prefill_progress", _fake
        )
        now = time.monotonic()
        srv = _srv_with(
            {
                "sess-1": {
                    "backend": "local",
                    "started_at": now,
                    "active": True,
                    "expires_at": now + 0.05,
                    "last_progress": 0,
                    "last_progress_ts": now - 5.0,
                }
            }
        )

        last, extended = await _extend_lease_during_prefill(
            srv, "sess-1", llama_port=8080, slot_id=0, last_progress=0
        )
        assert extended is True and last == 1000
        last, extended = await _extend_lease_during_prefill(
            srv, "sess-1", llama_port=8080, slot_id=0, last_progress=last
        )
        assert extended is True and last == 2500

        lines = _info_log_lines(srv.logger)
        assert any("lease_extended_on_measured_progress" in line for line in lines)
        # The extension call carries reason='measured_progress' as an arg (logging
        # uses lazy %-formatting, so the reason is in call.args, not the template).
        assert any(
            "measured_progress" in call.args
            for call in srv.logger.info.call_args_list
        )
        # The no-progress clock advanced because measured progress was observed.
        record = srv.local_dispatch_records["sess-1"]
        assert record["last_progress"] == 2500
        assert record["last_progress_ts"] > now - 5.0

    @pytest.mark.asyncio
    async def test_progressing_prefill_not_released_by_watchdog(self, monkeypatch):
        """A slot that keeps advancing is never released by the watchdog even
        after a long total elapsed time (below the max ceiling)."""
        from proxy.router_helpers import (
            _extend_lease_during_prefill,
            _watchdog_no_progress,
        )

        seq = iter([(10000, True), (20000, True)])

        async def _fake(*a, **k):
            return next(seq)

        monkeypatch.setattr("proxy.router_helpers._query_prefill_progress", _fake)
        now = time.monotonic()
        srv = _srv_with(
            {
                "sess-1": {
                    "backend": "local",
                    "started_at": now - 200.0,
                    "active": True,
                    "expires_at": now + 1.0,
                    "last_progress": 5000,
                    "last_progress_ts": now - 200.0,
                    "eta_prefill_s": 10000.0,  # far in the future
                }
            }
        )
        # Before any measured progress the record is stalled.
        assert _watchdog_no_progress(srv, "sess-1", srv.local_dispatch_records["sess-1"])

        await _extend_lease_during_prefill(
            srv, "sess-1", llama_port=8080, slot_id=0, last_progress=5000
        )
        record = srv.local_dispatch_records["sess-1"]
        assert _watchdog_no_progress(srv, "sess-1", record) is False


# ===========================================================================
# _cleanup_stale_local_dispatch — recorded reason + elapsed
# ===========================================================================


class TestCleanupSweepRecordsReason:
    @pytest.mark.asyncio
    async def test_eta_exceeded_release_is_logged_with_reason_and_elapsed(self):
        from proxy.router_helpers import _cleanup_stale_local_dispatch

        now = time.monotonic()
        srv = _srv_with(
            {
                "sess-1": {
                    "backend": "local",
                    "endpoint": "local",
                    "active": True,
                    "started_at": now - 500.0,
                    "expires_at": now + 1000.0,
                    "last_progress": 9000,
                    "last_progress_ts": now,  # progressing
                    "eta_prefill_s": 100.0,  # 500s > 100 * 2
                }
            }
        )
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 1
        assert "sess-1" not in srv.local_dispatch_records
        warnings = [str(c) for c in srv.logger.warning.call_args_list]
        joined = "\n".join(warnings)
        # Reason is a lazy-format arg; assert it directly on the call args.
        assert any(
            "prefill_eta_exceeded" in call.args
            for call in srv.logger.warning.call_args_list
        )
        assert "elapsed_s=" in joined
        assert "stalled_for_s=" in joined

    @pytest.mark.asyncio
    async def test_stalled_release_is_logged_with_reason(self):
        from proxy.router_helpers import _cleanup_stale_local_dispatch

        now = time.monotonic()
        srv = _srv_with(
            {
                "sess-1": {
                    "backend": "local",
                    "endpoint": "local",
                    "active": True,
                    "started_at": now - 200.0,
                    "expires_at": now + 1000.0,
                    "last_progress": 100,
                    "last_progress_ts": now - 200.0,
                }
            }
        )
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 1
        joined = "\n".join(
            str(c) for c in srv.logger.warning.call_args_list
        )
        assert any(
            "prefill_stalled" in call.args
            for call in srv.logger.warning.call_args_list
        )
        assert "elapsed_s=" in joined
