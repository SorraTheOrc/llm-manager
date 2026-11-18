"""Deletion-path marker invariant rows (LP-0MWL5KJTN002X3DU).

Parent: **LP-0MWKWGLRT003AUG4** — *Orphaned ``local_prefill_in_flight`` marker
closes the local route for every session*, AC2 + AC3.

Invariant pinned here: **no dispatch-record deletion may leave a
``local_prefill_in_flight`` marker behind.** The marker is process-wide and the
admission gate prices the prefill cap with it, so a marker that outlives its
record denies every local dispatch while the GPU is idle — the 2026-11-17
incident shape ``records_total=0 ... prefill_in_flight=1 ...
llama_available_slots=2``.

One row per production record-deletion helper:

========================  ==================================================
row                       helper
========================  ==================================================
``prune``                 ``router_helpers._prune_stale_dispatch_records``
``explicit_release``      ``router_helpers._release_local_dispatch``
                          (``/v1/leases/release``)
``client_disconnect``     ``router_helpers._release_client_disconnect_dispatch``
``request_completed``     ``router_helpers._release_completed_dispatch``
``no_progress_watchdog``  ``router_helpers._cleanup_stale_local_dispatch``
``orphan_sweep``          ``router_helpers._cleanup_stale_local_dispatch``
``session_invalidation``  ``session._invalidate_session_and_slot``
========================  ==================================================

Each row seeds one record plus its marker — and a second, genuinely live
session whose record and marker must survive (the negative control that stops
the invariant being satisfied by over-popping).

Red-first encoding (the repository convention, see
``test_abandoned_local_dispatch.py``): the two paths that leak today —
``prune`` and ``session_invalidation`` — are marked
``pytest.mark.xfail(strict=True)`` so this test-only item lands green on
``dev`` with the defect present. ``[impl] Every record deletion pops the
marker`` (LP-0MWL5KLN1003E990) removes the markers as it turns the rows green;
``strict`` makes a leftover marker fail the suite, so it cannot be forgotten.

Every row is hermetic: a throwaway ``SimpleNamespace`` server, fresh
``asyncio.Lock``s, a frozen monotonic clock (no sleeps), and the slot registry
/ contention queue patched out so no shared state is touched.
"""

from __future__ import annotations

import asyncio
import copy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import proxy.contention_queue as contention_queue
import proxy.router_helpers as helpers
import proxy.server as server_module
import proxy.session as session_module
import pytest

#: Session whose record the row deletes (its marker must not outlive it).
SESSION = "deleted-session"
#: Session whose record and marker must survive every row (negative control).
LIVE_SESSION = "live-control-session"

#: Endpoints used by the per-endpoint (tuple-key) rows.
SERVED_ENDPOINT = "http://127.0.0.1:8080"
OTHER_ENDPOINT = "http://127.0.0.1:8081"

#: Frozen ``time.monotonic()`` value every row is built and driven against.
NOW = 10_000.0

XFAIL_REASON = (
    "this deletion path leaves the prefill marker behind until "
    "[impl] Every record deletion pops the marker (LP-0MWL5KLN1003E990) lands"
)

SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "session_slot_pool_size": 1,
        "local_dispatch_lease_timeout_seconds": 180,
        "local_dispatch_lease_per_token_seconds": 0.0,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 0.05,
        "local_dispatch_lease_prefill_buffer_seconds": 0.5,
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_max_prefill_seconds": 900,
        "local_prefill_deadline_seconds": 0,
        "local_dispatch_first_byte_timeout_seconds": 840,
        "local_dispatch_prefill_observability_warn_seconds": 0,
        "local_max_concurrent_prefills": 1,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 0.3,
    }
}


class _FrozenClock:
    """``time`` shim whose ``monotonic`` is pinned to :data:`NOW`.

    Only ``monotonic`` is injected; every other ``time`` attribute falls
    through to the real module, so the production code under test sees a
    stable clock without patching the global ``time`` module (no sleeps).
    """

    def monotonic(self) -> float:
        return NOW

    def __getattr__(self, name: str):
        return getattr(time, name)


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch):
    """Pin the clock and keep every row off the real registry/queue."""
    monkeypatch.setattr(helpers, "time", _FrozenClock())
    helpers._reset_abandoned_slots()

    freed: list[tuple] = []

    def _free_slot_assignment(session_id, endpoint=None):
        freed.append((session_id, endpoint))
        return True

    monkeypatch.setattr(session_module, "_free_slot_assignment", _free_slot_assignment)
    wake = AsyncMock(return_value=None)
    monkeypatch.setattr(contention_queue, "wake", wake)
    yield SimpleNamespace(freed=freed, wake=wake)
    helpers._reset_abandoned_slots()


def _record(
    *,
    active: bool = True,
    expires_in: float = 60.0,
    started_ago: float = 0.0,
    progress_ago: float | None = 0.0,
    model_name: str | None = "test-model",
    **overrides,
) -> dict:
    """A dispatch record bound to the frozen clock.

    ``progress_ago=None`` omits ``last_progress_ts`` entirely, the legacy
    record shape that falls through the watchdog to the orphan sweep.
    """
    record = {
        "backend": "local",
        "started_at": NOW - started_ago,
        "active": active,
        "expires_at": NOW + expires_in,
        "model_name": model_name,
        "last_progress": 0,
    }
    if progress_ago is not None:
        record["last_progress_ts"] = NOW - progress_ago
    record.update(overrides)
    return record


def _live_control_record() -> dict:
    """The negative control: a genuinely active, unexpired, progressing lease."""
    return _record(active=True, expires_in=600.0, progress_ago=0.0)


def _make_srv(records: dict, markers: dict) -> SimpleNamespace:
    """Throwaway server with fresh locks (no shared proxy state)."""
    return SimpleNamespace(
        config=copy.deepcopy(SERVER_CONFIG),
        local_active_queries=len(records),
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records=dict(records),
        local_dispatch_records_lock=asyncio.Lock(),
        local_prefill_in_flight=dict(markers),
        local_prefill_in_flight_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions=set(),
        logger=MagicMock(),
    )


def _live_record_session_ids(srv) -> set[str]:
    """Session ids of every dispatch record still present after the helper."""
    sessions: set[str] = set()
    for key in srv.local_dispatch_records:
        session_id = helpers._dispatch_key_session_id(key)
        if session_id is not None:
            sessions.add(session_id)
    return sessions


def _orphaned_markers(srv) -> set[str]:
    """Markers with no dispatch record left behind (the defect shape)."""
    return set(srv.local_prefill_in_flight) - _live_record_session_ids(srv)


# ======================================================================
# The seven deletion-path drivers
# ======================================================================


async def _drive_prune(srv, monkeypatch):
    removed = helpers._prune_stale_dispatch_records(srv, NOW)
    assert removed == [SESSION], f"the stale record must be pruned: {removed}"


async def _drive_explicit_release(srv, monkeypatch):
    released = await helpers._release_local_dispatch(srv, SESSION)
    assert released is True, "the explicit release must delete the record"


async def _drive_client_disconnect(srv, monkeypatch):
    released = helpers._release_client_disconnect_dispatch(srv, SESSION)
    assert released is True, "the client-disconnect release must delete the record"
    # The release schedules its detached stream close / contention wake; one
    # loop turn lets those complete inside the test's loop (no sleep).
    await asyncio.sleep(0)


async def _drive_request_completed(srv, monkeypatch):
    released = await helpers._release_completed_dispatch(
        srv, SESSION, served_endpoint=SERVED_ENDPOINT
    )
    assert released is True, "the completion release must delete the ghost lease"


async def _drive_no_progress_watchdog(srv, monkeypatch):
    removed = await helpers._cleanup_stale_local_dispatch(srv)
    assert removed == 1, f"the watchdog must release exactly one record: {removed}"


async def _drive_orphan_sweep(srv, monkeypatch):
    removed = await helpers._cleanup_stale_local_dispatch(srv)
    assert removed == 1, f"the orphan sweep must release exactly one record: {removed}"


async def _drive_session_invalidation(srv, monkeypatch):
    # ``_invalidate_session_and_slot`` reaches the server through
    # ``session._srv()`` (the ``proxy.server`` module), so point the module's
    # dispatch state at this row's throwaway server for the duration.
    monkeypatch.setattr(
        server_module, "local_dispatch_records", srv.local_dispatch_records
    )
    monkeypatch.setattr(
        server_module, "local_dispatch_records_lock", srv.local_dispatch_records_lock
    )
    monkeypatch.setattr(
        server_module, "local_prefill_in_flight", srv.local_prefill_in_flight
    )
    monkeypatch.setattr(server_module, "logger", srv.logger)

    await session_module._invalidate_session_and_slot(SESSION, "marker_invariant", None)
    assert SESSION not in _live_record_session_ids(srv), (
        "the invalidation must delete the session's dispatch record"
    )


#: One row per production deletion helper. ``records`` builds the seeded
#: registry (the target session plus the live negative control), ``defect``
#: marks the rows that leak today and therefore carry ``xfail(strict=True)``.
_DELETION_MATRIX: dict[str, dict] = {
    "prune": {
        # Inactive *and* past expiry: the incident's stale-lease shape.
        "drive": _drive_prune,
        "records": lambda: {SESSION: _record(active=False, expires_in=-5.0)},
        "defect": True,
    },
    "explicit_release": {
        "drive": _drive_explicit_release,
        "records": lambda: {SESSION: _record(active=True)},
        "defect": False,
    },
    "client_disconnect": {
        "drive": _drive_client_disconnect,
        "records": lambda: {SESSION: _record(active=True)},
        "defect": False,
    },
    "request_completed": {
        # The ghost lease: a record left active on an endpoint other than the
        # one the request completed on.
        "drive": _drive_request_completed,
        "records": lambda: {(OTHER_ENDPOINT, SESSION): _record(active=True)},
        "defect": False,
    },
    "no_progress_watchdog": {
        # Active with a far-out adaptive lease, but no progress for an hour.
        "drive": _drive_no_progress_watchdog,
        "records": lambda: {
            SESSION: _record(active=True, expires_in=600.0, progress_ago=3600.0)
        },
        "defect": False,
    },
    "orphan_sweep": {
        # Legacy shape (no ``last_progress_ts``) so the watchdog passes it to
        # the orphan sweep; ``model_name=None`` means no /slots probe is made.
        "drive": _drive_orphan_sweep,
        "records": lambda: {
            SESSION: _record(
                active=True, expires_in=-5.0, model_name=None, progress_ago=None
            )
        },
        "defect": False,
    },
    "session_invalidation": {
        "drive": _drive_session_invalidation,
        "records": lambda: {SESSION: _record(active=True)},
        "defect": True,
    },
}

_MATRIX_PARAMS = [
    pytest.param(
        name,
        marks=pytest.mark.xfail(strict=True, reason=XFAIL_REASON),
        id=name,
    )
    if row["defect"]
    else pytest.param(name, id=name)
    for name, row in _DELETION_MATRIX.items()
]


# ======================================================================
# AC2 (parent AC3): the deletion matrix
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _MATRIX_PARAMS)
async def test_deletion_path_leaves_no_orphaned_prefill_marker(
    path, _hermetic_environment, monkeypatch
):
    """LP-0MWKWGLRT003AUG4 AC3: a deleted record never orphans its marker.

    After the helper runs the marker set must be a **subset** of the live
    dispatch-record session ids — the deleted session's marker may not survive
    its record. The second, genuinely live session's record **and** marker must
    be untouched, so the invariant cannot be satisfied by over-popping.
    """
    row = _DELETION_MATRIX[path]
    records = dict(row["records"]())
    records[LIVE_SESSION] = _live_control_record()
    srv = _make_srv(records=records, markers={SESSION: True, LIVE_SESSION: True})

    await row["drive"](srv, monkeypatch)

    live = _live_record_session_ids(srv)
    assert SESSION not in live, (
        f"{path}: the row must actually delete the target dispatch record; "
        f"remaining records={sorted(map(str, srv.local_dispatch_records))}"
    )
    # Negative control (AC3): a marker backed by an ``active`` record is never
    # dropped by any helper in the matrix.
    assert LIVE_SESSION in live, (
        f"{path}: the live control record must survive; live={sorted(live)}"
    )
    assert srv.local_prefill_in_flight.get(LIVE_SESSION) is True, (
        f"{path}: the live control marker must survive; "
        f"markers={srv.local_prefill_in_flight}"
    )

    orphaned = _orphaned_markers(srv)
    assert not orphaned, (
        f"{path}: deleting a dispatch record left orphaned prefill marker(s) "
        f"{sorted(orphaned)}; markers={srv.local_prefill_in_flight} "
        f"live_records={sorted(live)}"
    )


# ======================================================================
# AC3 (parent AC3): the negative control on a non-deleting path
# ======================================================================


@pytest.mark.asyncio
async def test_gate_refusal_never_drops_a_live_sessions_marker():
    """LP-0MWKWGLRT003AUG4 AC3: a refusing gate pops nothing.

    The gate prunes and then refuses an unrelated session while the pool's
    prefill cap is spent. Refusing must not delete the live holder's record or
    pop its marker — the guard against "fixing" the invariant by over-popping.
    """
    srv = _make_srv(
        records={LIVE_SESSION: _live_control_record()},
        markers={LIVE_SESSION: True},
    )

    acquired, owner, _active, _retry = await helpers._try_acquire_local_dispatch(
        srv, 1, "newcomer-session", "local"
    )

    assert acquired is False, "a full pool must refuse the newcomer"
    assert owner == LIVE_SESSION, f"the refusal must name the live holder: {owner}"
    assert srv.local_dispatch_records.get(LIVE_SESSION) is not None, (
        "a refusal must not delete the live holder's record"
    )
    assert srv.local_prefill_in_flight == {LIVE_SESSION: True}, (
        "a refusal must not pop a live holder's prefill marker: "
        f"{srv.local_prefill_in_flight}"
    )
