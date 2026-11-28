"""The post-acceptance eviction reason equals the watchdog release reason.

LP-0MWKQJHSM000CI6W AC1/AC3 and the reconciliation proposal recorded on the
item (2026-11-19 comment). Before the fix the deferred watchdog release path
emitted the literal umbrella ``no_progress_watchdog`` on
``local_eviction_after_acceptance`` even though the ``lease_released`` line
carried the distinct reason the sweep computed
(``prefill_eta_exceeded`` / ``prefill_stalled`` / ``prefill_max_exceeded``).
The residual analyser counts the eviction line, so a relabelling could be
mistaken for a fix.

These tests are hermetic: they build dispatch records on a fake server and run
the real :func:`proxy.router_helpers._cleanup_stale_local_dispatch` sweep, then
compare the two log lines the sweep emits for the same release.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC
from types import SimpleNamespace

import pytest

from proxy import admission_events, router_helpers

LOGGER_NAME = "llama-proxy"


class _Srv:
    """Minimal server stub for the cleanup sweep (no network)."""

    def __init__(self):
        self.logger = logging.getLogger(LOGGER_NAME)
        self.local_dispatch_records = {}
        self.local_dispatch_records_lock = asyncio.Lock()
        self.local_prefill_in_flight = {}
        self.local_active_queries = 1
        self.config = {"server": {}}


def _record_for(reason: str, **overrides) -> dict:
    """A dispatch record whose watchdog verdict is exactly *reason*.

    ``prefill_eta_exceeded`` — elapsed far past ``eta_prefill_s * factor``
    (factor 2.0), with recent progress so the stall bound does not fire first.
    ``prefill_stalled``     — no progress for longer than the 180 s timeout,
    with no ETA so the runaway bound is disabled.
    ``prefill_max_exceeded`` — past the 900 s ceiling, with recent progress and
    no ETA.
    """
    now = time.monotonic()
    base = {
        "backend": "local",
        "endpoint": "local",
        "active": True,
        "first_byte_seen": False,
        "estimated_tokens": 40_000,
        "new_tokens": 40_000,
        "slot_id": None,
        "llama_task_id": None,
        "last_progress": 0,
    }
    if reason == "prefill_eta_exceeded":
        base.update(
            started_at=now - 25.0,
            last_progress_ts=now,
            eta_prefill_s=10.0,
        )
    elif reason == "prefill_stalled":
        base.update(
            started_at=now - 200.0,
            last_progress_ts=now - 200.0,
            eta_prefill_s=None,
        )
    elif reason == "prefill_max_exceeded":
        base.update(
            started_at=now - 1_000.0,
            last_progress_ts=now,
            eta_prefill_s=None,
        )
    else:  # pragma: no cover - guard against typos in parametrisation
        raise ValueError(reason)
    base.update(overrides)
    return base


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def _eviction_lines(caplog) -> list[str]:
    return [m for m in _messages(caplog) if m.startswith("local_eviction_after_acceptance")]


def _lease_released_lines(caplog) -> list[str]:
    return [m for m in _messages(caplog) if m.startswith("lease_released")]


@pytest.fixture(autouse=True)
def _reset_counters():
    admission_events.reset_eviction_after_acceptance_counters()
    yield
    admission_events.reset_eviction_after_acceptance_counters()


@pytest.mark.parametrize(
    "reason",
    ["prefill_eta_exceeded", "prefill_stalled", "prefill_max_exceeded"],
)
def test_after_acceptance_reason_equals_lease_released_reason(reason, caplog):
    """AC1: one release, one distinct reason on both lines."""
    srv = _Srv()
    key = router_helpers._dispatch_lease_key(None, f"sess-{reason}")
    srv.local_dispatch_records[key] = _record_for(reason)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        removed = asyncio.run(router_helpers._cleanup_stale_local_dispatch(srv))

    assert removed == 1
    releases = _lease_released_lines(caplog)
    evictions = _eviction_lines(caplog)
    assert len(releases) == 1, releases
    assert len(evictions) == 1, evictions
    assert f"reason={reason}" in releases[0]
    assert f"reason={reason}" in evictions[0], (
        "the after-acceptance line must carry the same distinct reason the "
        f"lease_released line used; got {evictions[0]!r}"
    )
    assert "reason=no_progress_watchdog" not in evictions[0]


def test_mixed_window_reconciles_the_two_breakdowns(caplog):
    """Reconciliation: the two per-reason counts match in a mixed window.

    Before the fix a window could show 327 ``prefill_eta_exceeded`` releases
    against 0 evictions carrying that reason. Replaying a mixed window and
    asserting the two breakdowns agree is the regression guard the item's
    2026-11-19 comment proposed.
    """
    srv = _Srv()
    expected = {
        "prefill_eta_exceeded": 2,
        "prefill_stalled": 1,
        "prefill_max_exceeded": 1,
    }
    for reason, count in expected.items():
        for index in range(count):
            key = router_helpers._dispatch_lease_key(None, f"{reason}-{index}")
            srv.local_dispatch_records[key] = _record_for(reason)
            srv.local_active_queries += 1

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        asyncio.run(router_helpers._cleanup_stale_local_dispatch(srv))

    def breakdown(lines: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for line in lines:
            for part in line.split():
                if part.startswith("reason="):
                    value = part.split("=", 1)[1]
                    counts[value] = counts.get(value, 0) + 1
                    break
        return counts

    releases = breakdown(_lease_released_lines(caplog))
    evictions = breakdown(_eviction_lines(caplog))
    assert releases == expected
    assert evictions == expected


def test_release_context_fields_are_on_the_eviction_line(caplog):
    """The V1 window's classifier needs task_started/progress/queue_wait."""
    srv = _Srv()
    key = router_helpers._dispatch_lease_key(None, "sess-context")
    srv.local_dispatch_records[key] = _record_for(
        "prefill_eta_exceeded",
        last_progress=1_234,
        queue_wait_s=4.25,
    )

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        asyncio.run(router_helpers._cleanup_stale_local_dispatch(srv))

    line = _eviction_lines(caplog)[0]
    assert "task_started=no" in line
    assert "progress_observed=yes" in line
    assert "queue_wait_s=4.25" in line


def test_unknown_release_context_is_explicit_never_omitted(caplog):
    """A record without the evidence renders explicit no/unknown markers."""
    srv = _Srv()
    key = router_helpers._dispatch_lease_key(None, "sess-context-empty")
    srv.local_dispatch_records[key] = _record_for("prefill_stalled")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        asyncio.run(router_helpers._cleanup_stale_local_dispatch(srv))

    line = _eviction_lines(caplog)[0]
    assert "progress_observed=no" in line
    assert "queue_wait_s=unknown" in line


def test_existing_key_positions_are_unchanged(caplog):
    """AC3: every pre-existing key keeps its position; new keys are trailing.

    A deployed positional parser reads the base key order; the reason value is
    the only change to those keys. The release-context fields are appended.
    """
    srv = _Srv()
    key = router_helpers._dispatch_lease_key(None, "sess-order")
    srv.local_dispatch_records[key] = _record_for("prefill_eta_exceeded")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        asyncio.run(router_helpers._cleanup_stale_local_dispatch(srv))

    line = _eviction_lines(caplog)[0]
    base_order = [
        "session=",
        "reason=",
        "estimated_tokens=",
        "new_tokens=",
        "eta_prefill_s=",
        "elapsed_s=",
        "slot_id=",
        "llama_task_id=",
        "endpoint=",
    ]
    positions = [line.index(token) for token in base_order]
    assert positions == sorted(positions), (
        f"pre-existing key order changed: {line!r}"
    )
    assert line.index("task_started=") > positions[-1]
    assert line.index("progress_observed=") > positions[-1]
    assert line.index("queue_wait_s=") > positions[-1]


def test_analyser_reports_the_distinct_watchdog_reasons():
    """AC2: the per-reason split is reason-agnostic and reports the new ones."""
    import importlib
    import sys
    from datetime import datetime
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    analyser = importlib.import_module("analyse_prefill_deadline_residual")

    start = datetime(2026, 11, 20, 0, 0, 0)
    end = datetime(2026, 11, 20, 1, 0, 0)
    lines = [
        "2026-11-20 00:10:00,000 - ERROR - local_eviction_after_acceptance "
        "session=s-eta reason=prefill_eta_exceeded estimated_tokens=1000 "
        "new_tokens=1000 eta_prefill_s=10.0 elapsed_s=25.0 slot_id=1 "
        "llama_task_id=unknown endpoint=local task_started=no "
        "progress_observed=yes queue_wait_s=4.25",
        "2026-11-20 00:20:00,000 - ERROR - local_eviction_after_acceptance "
        "session=s-stall reason=prefill_stalled estimated_tokens=1000 "
        "new_tokens=1000 eta_prefill_s=unknown elapsed_s=200.0 slot_id=1 "
        "llama_task_id=unknown endpoint=local task_started=no "
        "progress_observed=no queue_wait_s=unknown",
    ]
    log = analyser.parse_window(lines, start, end)
    split = analyser.reason_split(log.evictions, 1.0)
    assert split["prefill_eta_exceeded"]["total"] == 1
    assert split["prefill_stalled"]["total"] == 1
    assert analyser.count_evictions_by_reason(log.evictions) == {
        "prefill_eta_exceeded": 1,
        "prefill_stalled": 1,
    }

    assert analyser.REASON_PREFILL_ETA_EXCEEDED == "prefill_eta_exceeded"
    assert analyser.REASON_PREFILL_STALLED == "prefill_stalled"
    assert analyser.REASON_PREFILL_MAX_EXCEEDED == "prefill_max_exceeded"
    assert set(admission_events.EVICTION_REASONS_AFTER_ACCEPTANCE) >= set(
        analyser.WATCHDOG_RELEASE_REASONS
    )
