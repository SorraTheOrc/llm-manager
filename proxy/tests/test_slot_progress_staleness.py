"""Regression tests for stale log-derived slot progress (LP-0MVECWYWK000U3E1).

Root cause observed 2026-10-18: the proxy reported slot 0 = ``35199`` and
slot 1 = ``16384`` while live llama-server showed ``794`` / ``214``.  Those
two numbers were byte-for-byte the last prefill-progress values in a frozen
``llama-server.log``.  Three compounding defects made a frozen log pin the
Web UI at stale token counts forever:

1. ``_update_slot_progress_from_log()`` re-parsed the same frozen tail each
   cycle and stamped every entry with the *read* time, so the values never
   aged past the 60 s prune window.
2. The periodic loop aliased ``_last_slot_details_cache`` to the list it then
   enriched in place, permanently baking the stale counts into the fallback
   cache.
3. ``_enrich_slot_details_with_progress()`` replaced ``n_decoded`` whenever
   the log value was larger — always true for stale data.

These tests cover the four paths required by AC6: fresh-log override,
stale-log rejection, fallback-cache non-mutation, and expiry.
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ======================================================================
# Helpers
# ======================================================================

_PROGRESS_LINE = (
    "slot update_slots: id  0 | task 1 | prompt processing progress, "
    "n_tokens = {n}, batch.n_tokens = {n}, progress = {p}\n"
)


class _FakeLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


def _write_log(tmp_path, n_tokens, progress, age_seconds):
    """Write a llama-server.log with one progress line and set its mtime.

    ``age_seconds`` is how long ago the file was last modified, so
    ``age_seconds > SLOT_PROGRESS_MAX_AGE_SECONDS`` produces a "frozen" log.
    """
    log_path = tmp_path / "llama-server.log"
    log_path.write_text(_PROGRESS_LINE.format(n=n_tokens, p=progress))
    mtime = time.time() - age_seconds
    os.utime(log_path, (mtime, mtime))
    return log_path, mtime


def _fake_srv(log_dir):
    srv = MagicMock()
    srv.log_dir = str(log_dir)
    return srv


# ======================================================================
# AC1 / AC6 — expiry: a frozen log yields no progress data
# ======================================================================


class TestLogDerivedProgressExpiry:
    """Log-derived entries must expire based on the log's real age."""

    def setup_method(self):
        from proxy.observability import _slot_progress_cache
        _slot_progress_cache.clear()

    def teardown_method(self):
        from proxy.observability import _slot_progress_cache
        _slot_progress_cache.clear()

    def test_frozen_log_yields_no_progress_after_prune_window(self, tmp_path):
        """AC1: a log older than the prune window produces no cached entries."""
        from proxy.observability import (
            SLOT_PROGRESS_MAX_AGE_SECONDS,
            _slot_progress_cache,
            _update_slot_progress_from_log,
        )

        _, mtime = _write_log(
            tmp_path,
            n_tokens=35199,
            progress=0.9,
            age_seconds=SLOT_PROGRESS_MAX_AGE_SECONDS + 60,
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            _update_slot_progress_from_log()

        assert _slot_progress_cache == {}, (
            "a frozen log must not leave progress data in the cache"
        )

    def test_fresh_log_populates_progress_cache(self, tmp_path):
        """AC4 support: a freshly written log populates the cache."""
        from proxy.observability import (
            _slot_progress_cache,
            _update_slot_progress_from_log,
        )

        _write_log(tmp_path, n_tokens=794, progress=0.5, age_seconds=5)
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            _update_slot_progress_from_log()

        assert _slot_progress_cache[0]["n_tokens"] == 794

    def test_timestamp_tracks_log_mtime_not_read_time(self, tmp_path):
        """Fix 2: entries are stamped with the log's mtime, not ``time.time()``.

        Reading a frozen log at a much later wall-clock time must not make its
        entries look fresh.
        """
        from proxy.observability import (
            _slot_progress_cache,
            _update_slot_progress_from_log,
        )

        _, mtime = _write_log(tmp_path, n_tokens=35199, progress=0.9, age_seconds=120)
        future_now = time.time() + 3600
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            with patch("proxy.observability.time.time", return_value=future_now):
                _update_slot_progress_from_log()

        # The stale entry must be pruned rather than re-stamped as "now".
        assert _slot_progress_cache == {}

    def test_future_mtime_is_clamped_to_now(self, tmp_path):
        """A future-dated log (clock skew) cannot keep entries alive forever."""
        from proxy.observability import (
            _slot_progress_cache,
            _update_slot_progress_from_log,
        )

        log_path = _write_log(tmp_path, n_tokens=4096, progress=0.5, age_seconds=0)[0]
        future = time.time() + 3600
        os.utime(log_path, (future, future))

        fixed_now = time.time() + 10
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            with patch("proxy.observability.time.time", return_value=fixed_now):
                _update_slot_progress_from_log()

        # Clamped to "now", so it survives the prune but is not future-dated.
        assert _slot_progress_cache[0]["timestamp"] <= fixed_now + 1e-6


# ======================================================================
# AC3 / AC4 — enrichment honours log freshness
# ======================================================================


class TestEnrichmentStaleness:
    """Stale log values must not override live ``/slots`` values."""

    def setup_method(self):
        from proxy.observability import (
            _slot_progress_cache,
            _slot_stable_tracker,
        )
        _slot_progress_cache.clear()
        _slot_stable_tracker.clear()

    def teardown_method(self):
        from proxy.observability import (
            _slot_progress_cache,
            _slot_stable_tracker,
        )
        _slot_progress_cache.clear()
        _slot_stable_tracker.clear()

    def test_stale_log_does_not_override_n_decoded(self):
        """AC3: a stale log entry leaves the live API ``n_decoded`` untouched."""
        from proxy.observability import (
            SLOT_PROGRESS_MAX_AGE_SECONDS,
            _enrich_slot_details_with_progress,
            _slot_progress_cache,
        )

        now = time.time()
        _slot_progress_cache[0] = {
            "n_tokens": 35199,
            "progress": 0.9,
            "timestamp": now - SLOT_PROGRESS_MAX_AGE_SECONDS - 60,
        }
        slot_details = [{"slot_id": 0, "is_processing": True, "n_decoded": 794}]

        with patch("proxy.observability.time.time", return_value=now):
            result = _enrich_slot_details_with_progress(slot_details)

        assert result[0]["n_decoded"] == 794
        # The stale log fields must not be injected either.
        assert "n_tokens" not in result[0]
        assert "progress" not in result[0]

    def test_fresh_log_still_overrides_n_decoded(self):
        """AC4: a fresh log still overrides a smaller ``n_decoded`` as designed."""
        from proxy.observability import (
            _enrich_slot_details_with_progress,
            _slot_progress_cache,
        )

        now = time.time()
        _slot_progress_cache[0] = {
            "n_tokens": 4096,
            "progress": 0.17,
            "timestamp": now - 5,
        }
        slot_details = [{"slot_id": 0, "is_processing": True, "n_decoded": 512}]

        with patch("proxy.observability.time.time", return_value=now):
            result = _enrich_slot_details_with_progress(slot_details)

        assert result[0]["n_decoded"] == 4096
        assert result[0]["n_tokens"] == 4096

    def test_stale_entry_excluded_from_processing_candidates(self):
        """A stale progress entry must not mark a slot as processing."""
        from proxy.observability import (
            SLOT_PROGRESS_MAX_AGE_SECONDS,
            _processing_slot_assignments,
            _slot_progress_cache,
            _update_processing_slot_assignments,
        )

        now = time.time()
        _slot_progress_cache[7] = {
            "n_tokens": 35199,
            "progress": 0.9,
            "timestamp": now - SLOT_PROGRESS_MAX_AGE_SECONDS - 60,
        }
        srv = MagicMock()
        srv.local_dispatch_records = {}

        with patch("proxy.observability.time.time", return_value=now):
            _update_processing_slot_assignments(srv, slot_details=None)

        assert 7 not in _processing_slot_assignments


# ======================================================================
# AC2 / AC5 — fallback cache must never be mutated by enrichment
# ======================================================================


class TestFallbackCacheNonMutation:
    """``_last_slot_details_cache`` must stay pristine across enrichment."""

    def setup_method(self):
        from proxy.observability import (
            _last_slot_details_cache,
            _slot_progress_cache,
            _slot_stable_tracker,
        )
        _last_slot_details_cache.clear()
        _slot_progress_cache.clear()
        _slot_stable_tracker.clear()

    def teardown_method(self):
        from proxy.observability import (
            _last_slot_details_cache,
            _slot_progress_cache,
            _slot_stable_tracker,
        )
        _last_slot_details_cache.clear()
        _slot_progress_cache.clear()
        _slot_stable_tracker.clear()

    def test_fresh_query_is_cached_pristine(self):
        """AC2: the cache holds the raw ``/slots`` values, not enriched ones."""
        from proxy.observability import (
            _enrich_slot_details_with_progress,
            _last_slot_details_cache,
            _resolve_slot_details_for_enrichment,
            _slot_progress_cache,
        )

        now = time.time()
        _slot_progress_cache[0] = {
            "n_tokens": 4096,
            "progress": 0.5,
            "timestamp": now,
        }
        fresh = [{"slot_id": 0, "is_processing": True, "n_decoded": 794}]

        working = _resolve_slot_details_for_enrichment(fresh)
        with patch("proxy.observability.time.time", return_value=now):
            _enrich_slot_details_with_progress(working)

        # The working copy was enriched ...
        assert working[0]["n_decoded"] == 4096
        # ... but the fallback cache keeps the live /slots value.
        assert _last_slot_details_cache[0]["n_decoded"] == 794
        assert "n_tokens" not in _last_slot_details_cache[0]
        # And the caller's original list is untouched too.
        assert fresh[0]["n_decoded"] == 794

    def test_fallback_path_returns_detached_copy(self):
        """The fallback path hands enrichment a copy, never the cache itself."""
        from proxy.observability import (
            _enrich_slot_details_with_progress,
            _last_slot_details_cache,
            _resolve_slot_details_for_enrichment,
            _slot_progress_cache,
        )

        _last_slot_details_cache[:] = [
            {"slot_id": 3, "is_processing": True, "n_decoded": 214},
        ]
        now = time.time()
        _slot_progress_cache[3] = {
            "n_tokens": 16384,
            "progress": 0.8,
            "timestamp": now,
        }

        working = _resolve_slot_details_for_enrichment([])
        assert working is not _last_slot_details_cache
        assert working[0] is not _last_slot_details_cache[0]

        with patch("proxy.observability.time.time", return_value=now):
            _enrich_slot_details_with_progress(working)

        assert working[0]["n_decoded"] == 16384
        assert _last_slot_details_cache[0]["n_decoded"] == 214


# ======================================================================
# AC5 — status endpoint never reports a stale n_decoded above live
# ======================================================================


class TestStatusEndpointStaleLog:
    """End-to-end guard for the observed 35199/16384 poisoning."""

    @pytest.mark.asyncio
    async def test_status_never_reports_n_decoded_above_live_after_stale_log(self):
        """AC5: with a stale log and ``slots_stale=true`` the served value is live.

        Simulates one broadcast cycle with a *frozen* log: the fresh ``/slots``
        query reports ``n_decoded=794`` and the stale log claims ``35199``.  A
        later status request whose fresh detail query fails then serves the
        fallback cache with ``slots_stale=true`` — and must report 794, never
        35199.
        """
        from proxy.observability import (
            _enrich_slot_details_with_progress,
            _last_slot_details_cache,
            _resolve_slot_details_for_enrichment,
            _slot_progress_cache,
            _slot_stable_tracker,
        )
        from proxy.server import app

        from proxy import server

        _last_slot_details_cache.clear()
        _slot_progress_cache.clear()
        _slot_stable_tracker.clear()

        # A frozen log line claiming a huge token count.
        now = time.time()
        _slot_progress_cache[0] = {
            "n_tokens": 35199,
            "progress": 0.9,
            "timestamp": now - 600,
        }

        # One broadcast cycle: fresh /slots reports the real live value.
        live_n_decoded = 794
        working = _resolve_slot_details_for_enrichment(
            [{"slot_id": 0, "is_processing": True, "n_decoded": live_n_decoded}]
        )
        with patch("proxy.observability.time.time", return_value=now):
            _enrich_slot_details_with_progress(working)

        # The cache must hold the live value, not the stale log value.
        assert _last_slot_details_cache[0]["n_decoded"] == live_n_decoded

        # Now a status request with a failing fresh detail query serves cache.
        async def fake_query():
            return {"llama_server_running": True}

        detail_mock = AsyncMock(return_value=[])
        counts_mock = AsyncMock(return_value=(2, 3))
        transport = httpx.ASGITransport(app=app)
        with patch("proxy.server.query_llama_status", side_effect=fake_query):
            with patch.object(server, "current_model", "test-model"):
                with patch.object(server, "model_switch_refcount", 0):
                    with patch.object(server, "model_switch_lock", _FakeLock()):
                        with patch.object(server, "background_loads", {}):
                            with patch.object(server, "local_dispatch_records", {}):
                                with patch.object(
                                    server, "local_dispatch_records_lock", _FakeLock()
                                ):
                                    with patch.object(
                                        server,
                                        "config",
                                        {"server": {"llama_server_port": 8080}},
                                    ):
                                        with patch(
                                            "proxy.observability._query_slots_detail",
                                            detail_mock,
                                        ):
                                            with patch(
                                                "proxy.observability._query_slots",
                                                counts_mock,
                                            ):
                                                async with httpx.AsyncClient(
                                                    transport=transport,
                                                    base_url="http://test",
                                                ) as ac:
                                                    resp = await ac.get(
                                                        "/llama/local/status"
                                                    )
        assert resp.status_code == 200
        body = resp.json()
        assert body["slots_stale"] is True
        reported = body["slots"][0]["n_decoded"]
        assert reported == live_n_decoded
        assert reported <= live_n_decoded
