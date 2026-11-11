"""Measured prefill progress from the llama-server log tail
(LP-0MW8VO2HT007Y1VQ, LP-0MW8VPXNZ007T0J2).

Root cause (certified 2026-11-08): the deployed llama.cpp build (b8782)
returns ``/slots`` without any numeric prefill field — no ``n_past``, no
``n_prompt_tokens_processed``, no ``context_tokens``.  The proxy's lease
extension therefore saw ``progress=None`` on every long prefill and logged
``prefill_progress_unobservable`` while extending blind.

The signal does exist: llama-server writes

    slot update_slots: id  0 | task 2833255 | prompt processing progress,
    n_tokens = 17690, batch.n_tokens = 1024, progress = 0.999774

to stderr, which is redirected into ``llama-server.log``.  The proxy already
parses that tail into ``_slot_progress_cache`` for the UI
(``_update_slot_progress_from_log``), but ``_query_prefill_progress`` never
read it.

These tests pin the new measured path:

* ``get_slot_prefill_progress`` returns the freshest parsed ``n_tokens`` for a
  known slot and ``None`` for a stale log or an unknown slot;
* ``_query_prefill_progress`` prefers the log-derived per-slot progress over
  the aggregate fallback when ``/slots`` has no numeric field;
* a stale/frozen log still yields the fail-safe liveness behaviour;
* the log parser handles the real multi-spacing / mixed-slot line shapes.
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PROGRESS_LINE = (
    "slot update_slots: id  {slot} | task {task} | prompt processing progress, "
    "n_tokens = {n}, batch.n_tokens = {batch}, progress = {p}\n"
)


def _write_log(tmp_path, lines, age_seconds=0.0):
    """Write ``lines`` into ``llama-server.log`` and set its mtime age."""
    log_path = tmp_path / "llama-server.log"
    log_path.write_text("".join(lines))
    if age_seconds:
        mtime = time.time() - age_seconds
        os.utime(log_path, (mtime, mtime))
    return log_path


def _fake_srv(log_dir):
    srv = MagicMock()
    srv.log_dir = str(log_dir)
    return srv


def _query_srv():
    """A server stub suitable for ``_query_prefill_progress``."""
    from test_prefill_lease_extension import _make_srv

    return _make_srv()


# ===========================================================================
# get_slot_prefill_progress — the accessor
# ===========================================================================


class TestGetSlotPrefillProgress:
    def test_returns_fresh_n_tokens_for_known_slot(self, tmp_path):
        """A fresh log line populates the cache and is returned by slot id."""
        from proxy.observability import get_slot_prefill_progress

        _write_log(
            tmp_path,
            [_PROGRESS_LINE.format(slot=0, task=1, n=17690, batch=1024, p=0.99)],
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            entry = get_slot_prefill_progress(0)
        assert entry is not None
        assert entry["n_tokens"] == 17690
        assert entry["progress"] == pytest.approx(0.99)

    def test_returns_none_for_unknown_slot(self, tmp_path):
        """A slot with no progress line yields None."""
        from proxy.observability import get_slot_prefill_progress

        _write_log(
            tmp_path,
            [_PROGRESS_LINE.format(slot=0, task=1, n=1000, batch=128, p=0.5)],
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            assert get_slot_prefill_progress(1) is None

    def test_returns_none_for_frozen_log(self, tmp_path):
        """A log older than the trust window is not served as live."""
        from proxy.observability import (
            SLOT_PROGRESS_MAX_AGE_SECONDS,
            get_slot_prefill_progress,
        )

        _write_log(
            tmp_path,
            [_PROGRESS_LINE.format(slot=0, task=1, n=35199, batch=1024, p=0.9)],
            age_seconds=SLOT_PROGRESS_MAX_AGE_SECONDS + 60,
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            assert get_slot_prefill_progress(0) is None

    def test_returns_none_when_slot_id_is_none(self, tmp_path):
        """Without a slot id progress cannot be attributed — no fabrication."""
        from proxy.observability import get_slot_prefill_progress

        _write_log(
            tmp_path,
            [_PROGRESS_LINE.format(slot=0, task=1, n=1000, batch=128, p=0.5)],
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            assert get_slot_prefill_progress(None) is None

    def test_returns_none_when_log_missing(self, tmp_path):
        """A missing log is a no-op (no exception, no entry)."""
        from proxy.observability import get_slot_prefill_progress

        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            assert get_slot_prefill_progress(0) is None


# ===========================================================================
# _query_prefill_progress — measured path integration
# ===========================================================================


class TestQueryPrefillProgressMeasuredPath:
    @pytest.mark.asyncio
    async def test_uses_log_progress_when_slots_has_no_numeric_field(
        self, tmp_path
    ):
        """On b8782 (/slots has no n_past), the log tail supplies progress."""
        from proxy.router_helpers import _query_prefill_progress

        _write_log(
            tmp_path,
            [_PROGRESS_LINE.format(slot=3, task=42, n=12345, batch=1024, p=0.7)],
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            with patch(
                "proxy.observability._query_slots_progress",
                AsyncMock(return_value={3: {"progress": None, "processing": True}}),
            ):
                progress, alive = await _query_prefill_progress(
                    _query_srv(), 8080, model_name="Qwen3", slot_id=3
                )
        assert progress == 12345
        assert alive is True

    @pytest.mark.asyncio
    async def test_log_progress_preferred_over_aggregate_fallback(self, tmp_path):
        """A per-slot log measurement wins over the aggregate /status value."""
        from proxy.router_helpers import _query_prefill_progress

        _write_log(
            tmp_path,
            [_PROGRESS_LINE.format(slot=0, task=7, n=9000, batch=512, p=0.8)],
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            with patch(
                "proxy.observability._query_slots_progress",
                AsyncMock(return_value={}),
            ):
                with patch(
                    "proxy.observability.query_llama_status",
                    AsyncMock(return_value={"kv_cache_tokens": 4321}),
                ):
                    progress, alive = await _query_prefill_progress(
                        _query_srv(), 8080, model_name="Qwen3", slot_id=0
                    )
        assert (progress, alive) == (9000, True)

    @pytest.mark.asyncio
    async def test_stale_log_falls_back_to_liveness_failsafe(self, tmp_path):
        """A frozen log leaves the fail-safe liveness-only behaviour intact."""
        from proxy.observability import SLOT_PROGRESS_MAX_AGE_SECONDS
        from proxy.router_helpers import _query_prefill_progress

        _write_log(
            tmp_path,
            [_PROGRESS_LINE.format(slot=3, task=1, n=9999, batch=1024, p=0.5)],
            age_seconds=SLOT_PROGRESS_MAX_AGE_SECONDS + 60,
        )
        srv = _query_srv()
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            with patch(
                "proxy.observability._query_slots_progress",
                AsyncMock(return_value={3: {"progress": None, "processing": True}}),
            ):
                with patch(
                    "proxy.observability.query_llama_status",
                    AsyncMock(return_value={"kv_cache_tokens": None}),
                ):
                    progress, alive = await _query_prefill_progress(
                        srv, 8080, model_name="Qwen3", slot_id=3
                    )
        assert progress is None
        assert alive is True

    @pytest.mark.asyncio
    async def test_per_slot_slots_progress_still_preferred(self, tmp_path):
        """When /slots does report n_past it is still the first preference."""
        from proxy.router_helpers import _query_prefill_progress

        _write_log(
            tmp_path,
            [_PROGRESS_LINE.format(slot=3, task=1, n=100, batch=64, p=0.1)],
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            with patch(
                "proxy.observability._query_slots_progress",
                AsyncMock(return_value={3: {"progress": 50000, "processing": True}}),
            ):
                progress, alive = await _query_prefill_progress(
                    _query_srv(), 8080, model_name="Qwen3", slot_id=3
                )
        assert (progress, alive) == (50000, True)


# ===========================================================================
# Log line parser shapes
# ===========================================================================


class TestLogLineParsing:
    @pytest.mark.parametrize(
        "line,expected",
        [
            # Real deployment shape: two spaces after ``id`` (router prefix).
            (
                "[43745] slot update_slots: id  0 | task 2833255 | prompt "
                "processing progress, n_tokens = 17690, batch.n_tokens = 1024, "
                "progress = 0.999774",
                (0, 17690, 0.999774),
            ),
            # Single space after ``id``.
            (
                "slot update_slots: id 1 | task 42 | prompt processing progress, "
                "n_tokens = 500, batch.n_tokens = 500, progress = 0.5",
                (1, 500, 0.5),
            ),
        ],
    )
    def test_parses_real_log_shapes(self, line, expected):
        from proxy.observability import _extract_progress_data_from_log

        assert _extract_progress_data_from_log(line) == expected

    def test_ignores_non_progress_lines(self):
        from proxy.observability import _extract_progress_data_from_log

        assert _extract_progress_data_from_log(
            "slot update_slots: id 0 | task 1 | prompt processing complete"
        ) is None
        assert _extract_progress_data_from_log("") is None

    def test_multi_slot_log_populates_both_slots(self, tmp_path):
        """Two concurrent prefills yield independent per-slot entries."""
        from proxy.observability import (
            _slot_progress_cache,
            _update_slot_progress_from_log,
        )

        _write_log(
            tmp_path,
            [
                _PROGRESS_LINE.format(slot=0, task=1, n=2000, batch=1024, p=0.4),
                _PROGRESS_LINE.format(slot=1, task=2, n=800, batch=512, p=0.2),
            ],
        )
        with patch("proxy.observability._srv", return_value=_fake_srv(tmp_path)):
            _update_slot_progress_from_log()
        assert _slot_progress_cache[0]["n_tokens"] == 2000
        assert _slot_progress_cache[1]["n_tokens"] == 800
