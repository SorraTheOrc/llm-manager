"""Deterministic regression: the 2026-10-23 prefill-thrash wedge.

Child C1 (LP-0MVKUKEIQ003YOLA) of the wedged prefill lease loop investigation
(LP-0MVKN9IRP003XJH6). Reproduces the incident window (02:00-03:00) where
llama-server accepted 27 tasks (all 28k-124k prompt tokens), evaluated
~1.98M prompt tokens with **0 eval batches** and 0 tokens delivered, and the
local dispatch gate stayed wedged ~4 h.

The reproduction drives the *real* routing helpers — exactly the call sequence
in ``router.py``'s local dispatch path — with a stubbed llama-server (stale /
unobservable ``/slots``: ``progress=None, alive=False``) and an injected
monotonic clock. No real sleeps: the deadline monitor runs on sub-second event
loop timeouts and every timestamp the lease/watchdog logic reads comes from
the injected clock.

Mechanics (mirrors the incident evidence):

1. ``_try_acquire_local_dispatch`` grants the session a dispatch lease and
   marks ``local_prefill_in_flight[session]``.
2. Prefill produces no first byte; each ``_extend_lease_during_prefill`` poll
   refreshes ``expires_at``/``last_progress_ts`` from the *in-flight* marker
   (``lease_extended_during_prefill progress=None alive=False in_flight=True``)
   — the refresh that keeps the 180s no-progress watchdog from firing.
3. The prefill deadline fires (``prefill_deadline_exceeded``, 90s): the
   request detours and the router runs the dispatch cleanup
   (``_cleanup_after_request`` → ``_decrement_local_active_queries`` for the
   explicit path, immediate record deletion for the non-explicit path).
4. Pre-fix, that cleanup only marks the record inactive with a future
   ``expires_at`` (``lease_renewed timeout=60s``) and leaves
   ``local_prefill_in_flight`` untouched, so the same session re-acquires
   immediately (41 local POSTs in the actual incident) and every other
   session is denied (``local_dispatch_denied`` /
   ``contention_queue_fallback_after_queue reason=local_lease_active``).
5. The leases only cleared via ``lease_released reason=session_evicted``
   ~4 h later.

RED-phase convention
--------------------
The post-fix contract tests are marked ``pytest.mark.xfail(strict=True)`` so
``dev`` stays green while the wedge exists (see
``tests/test_events_first_frame_bound.py`` for the established pattern). With
the markers in place, ``dev`` is green and the strict-XPASS failure output —
record + ``local_prefill_in_flight`` retained, immediate re-dispatch — is the
documented bug reproduction. C4 (post-deadline thrash cooldown) and C5
(truthful lease and in-flight accounting) flip these to green and remove the
markers as part of the fix; a regression afterwards re-opens the defect
loudly.

Simulation model
----------------
- 1 simulated poll (10s cadence) == 20ms of wall time: the stub llama-server
  advances the injected clock by one poll interval per ``/slots`` probe, so
  the 90s prefill deadline fires after 9 probes and never blocks the event
  loop.
- The whole 5-minute window is 3 thrash cycles (90s each) plus a 30s tail of
  periodic ``_cleanup_stale_local_dispatch`` sweeps: the module runs in
  well under a second.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from proxy.router import _cleanup_after_request
from proxy.router_helpers import (
    _cleanup_stale_local_dispatch,
    _try_acquire_local_dispatch,
)

from proxy import contention_queue, server
from proxy import router_helpers as helpers

# ── Incident constants (evidence, 2026-10-23 02:00-03:00) ────────────────
PREFILL_DEADLINE_SECONDS = 90.0  # local_prefill_deadline_seconds
PREFILL_POLL_SECONDS = 10.0  # local_dispatch_lease_prefill_poll_seconds
NO_PROGRESS_TIMEOUT_SECONDS = 180.0  # local_dispatch_no_progress_timeout_seconds
LEASE_TIMEOUT_SECONDS = 60.0  # local_dispatch_lease_timeout_seconds
WINDOW_SECONDS = 300.0  # post-fix bound: release within <= 5 min simulated

# Wall-time compression: 20ms per simulated 10s poll.
_POLL_WALL_SECONDS = 0.02
_SIM_SECONDS_PER_POLL = 10.0

# Sessions from the incident shape (explicit owner + anonymous sessions).
WEDGE_SESSION = "01a16636-0000-0000-0000-000000000001"
ANON_SESSION_A = "anon-session-a"
ANON_SESSION_B = "anon-session-b"


# ---------------------------------------------------------------------------
# Injected clock + llama-server stub
# ---------------------------------------------------------------------------


class _InjectedClock:
    """Monotonic clock that only advances when the test drives it."""

    def __init__(self) -> None:
        self._now = 0.0

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _TimeShim:
    """Replace ``router_helpers.time`` so only ``monotonic`` is injected.

    Patching the module attribute (rather than the shared ``time.monotonic``)
    keeps the asyncio event-loop clock untouched while every lease/watchdog
    timestamp in the helpers reads the injected clock.
    """

    def __init__(self, clock: _InjectedClock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(time, name)


class _StaleSlotsStub:
    """Stubbed llama-server: stale/unobservable slots, alive=False.

    ``/slots`` is held for the whole prompt-processing batch under load, so
    the incident saw ``prefill_progress_unobservable`` / ``slots_stale=true``
    with ``alive=False``. Each probe advances the injected clock by one poll
    interval — one prefill poll elapsed — and reports no progress.
    """

    def __init__(self, clock: _InjectedClock) -> None:
        self._clock = clock
        self.probes = 0

    async def __call__(self, *args, **kwargs):
        self.probes += 1
        self._clock.advance(PREFILL_POLL_SECONDS)
        return None, False  # (progress, alive)


@pytest.fixture
def clock() -> _InjectedClock:
    return _InjectedClock()


@pytest.fixture
def wedge_srv(monkeypatch, clock: _InjectedClock):
    """A minimal ``proxy.server`` stand-in in the incident configuration."""
    for attr, value in {
        "local_dispatch_records": {},
        "local_dispatch_records_lock": asyncio.Lock(),
        "local_active_queries": 0,
        "local_active_queries_lock": asyncio.Lock(),
        "local_generating_queries": 0,
        "local_generating_queries_lock": asyncio.Lock(),
        "local_generating_sessions": set(),
        "local_prefill_in_flight": {},
        "local_prefill_in_flight_lock": asyncio.Lock(),
        "active_queries": 0,
        "active_queries_lock": asyncio.Lock(),
        "logger": MagicMock(),
    }.items():
        monkeypatch.setattr(server, attr, value, raising=False)
    monkeypatch.setattr(
        server,
        "config",
        {
            "server": {
                "local_dispatch_lease_timeout_seconds": LEASE_TIMEOUT_SECONDS,
                "session_slot_pool_size": 1,
                "local_dispatch_lease_prefill_poll_seconds": (
                    PREFILL_POLL_SECONDS
                ),
                "local_dispatch_lease_prefill_buffer_seconds": 30,
                "local_dispatch_no_progress_timeout_seconds": (
                    NO_PROGRESS_TIMEOUT_SECONDS
                ),
                "local_dispatch_max_prefill_seconds": 900,
                "local_prefill_deadline_seconds": PREFILL_DEADLINE_SECONDS,
                "local_max_concurrent_prefills": 1,
                "local_dispatch_lease_chunk_refresh_buffer_seconds": 30,
            }
        },
        raising=False,
    )
    # Inject the clock into every helper timestamp and the stale /slots stub.
    monkeypatch.setattr(helpers, "time", _TimeShim(clock))
    stub = _StaleSlotsStub(clock)
    monkeypatch.setattr(helpers, "_query_prefill_progress", stub)
    monkeypatch.setattr(helpers, "_query_slot_processing", _no_slot_processing)
    contention_queue.reset()
    yield server
    contention_queue.reset()


async def _no_slot_processing(*args, **kwargs) -> bool:
    """Stub slot-liveness probe: never processing (stale /slots)."""
    return False


# ---------------------------------------------------------------------------
# Simulation drivers — mirror router.py's dispatch path
# ---------------------------------------------------------------------------


async def _prefill_attempt(
    wedge_srv, session: str, *, deadline_seconds: float = PREFILL_DEADLINE_SECONDS
) -> bool:
    """Run one local dispatch that never produces a first byte.

    Mirrors ``proxy_to_local`` (router.py): acquire the dispatch lease, wait
    for a first byte under ``_await_first_byte_with_prefill_monitor`` (the
    prefill deadline this item hardens) and detour when it fires. Returns
    True when the dispatch was granted, False when admission refused it.
    """
    acquired, _owner, _active, _retry_after = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=session, backend=None
    )
    if not acquired:
        return False

    async def _no_first_byte():
        await asyncio.sleep(3600)  # cancelled by the monitor on deadline

    polls = int(deadline_seconds // PREFILL_POLL_SECONDS)
    try:
        await helpers._await_first_byte_with_prefill_monitor(
            wedge_srv,
            _no_first_byte(),
            timeout=polls * _POLL_WALL_SECONDS,
            poll_seconds=_POLL_WALL_SECONDS,
            warn_seconds=0.0,
            session_id=session,
            llama_port=8080,
            model_name="qwen3",
            slot_id=0,
        )
    except TimeoutError:
        pass  # prefill_deadline_exceeded -> detour (router.py:1507-1526)
    else:  # pragma: no cover - harness guard, not a wedge condition
        raise AssertionError(
            "the prefill deadline did not fire: the simulation harness is broken"
        )
    await asyncio.sleep(0)  # let the cancelled upstream task settle
    return True


async def _detour_cleanup(wedge_srv, session: str, *, session_explicit: bool) -> None:
    """Run the router's post-deadline cleanup (router.py:1524)."""
    await _cleanup_after_request(
        wedge_srv,
        session,
        decrement_local=True,
        session_explicit=session_explicit,
        backend=None,
    )


def _snapshot(wedge_srv, session: str, clock: _InjectedClock) -> dict:
    """Capture the lease/marker state for the failure documentation."""
    record = wedge_srv.local_dispatch_records.get(session)
    return {
        "record_present": record is not None,
        "record_active": bool(record.get("active")) if record else None,
        "expires_in_future": (
            bool(record and record.get("expires_at", 0) > clock.now)
        ),
        "marker_present": session in wedge_srv.local_prefill_in_flight,
        "now": clock.now,
    }


def _wedge_report(
    wedge_srv, session: str, clock: _InjectedClock, granted: int
) -> str:
    """Terse pre-fix-wedge diagnosis used in every post-fix contract assert."""
    snap = _snapshot(wedge_srv, session, clock)
    return (
        "LP-0MVKUKEIQ003YOLA pre-fix wedge reproduced: "
        f"dispatch record retained={snap['record_present']} "
        f"(active={snap['record_active']}, expires_in_future="
        f"{snap['expires_in_future']}), "
        f"local_prefill_in_flight retained={snap['marker_present']}, "
        f"local dispatch grants={granted} "
        f"(incident: 41 local POSTs in an hour, 2,056 other-session denials). "
        "Post-fix (C4 cooldown + C5 truthful lease/marker accounting) the "
        "detoured session's lease and marker are released on the spot and "
        "re-dispatch is suppressed until the cooldown expires."
    )


async def _simulate_wedge_window(
    wedge_srv, clock: _InjectedClock, session: str, *, session_explicit: bool
) -> tuple[int, list[dict]]:
    """Run the thrash loop for the full simulated window (<= 5 min).

    Each cycle mirrors the incident: acquire, prefill to the 90s deadline with
    no first byte (lease extended each poll), deadline fires, detour cleanup,
    periodic ``_cleanup_stale_local_dispatch`` sweep. A denied attempt (post-
    fix cooldown) still consumes the deadline duration without dispatching.
    """
    granted = 0
    snapshots: list[dict] = []
    end = clock.now + WINDOW_SECONDS
    while end - clock.now >= PREFILL_DEADLINE_SECONDS:
        acquired = await _prefill_attempt(wedge_srv, session)
        if not acquired:
            clock.advance(PREFILL_DEADLINE_SECONDS)
            await _cleanup_stale_local_dispatch(wedge_srv)
            continue
        granted += 1
        snapshots.append(_snapshot(wedge_srv, session, clock))
        await _detour_cleanup(
            wedge_srv, session, session_explicit=session_explicit
        )
        await _cleanup_stale_local_dispatch(wedge_srv)
    # Tail: keep the periodic sweep running to the window bound so the
    # no-progress watchdog / idle-timeout paths get every chance to release.
    while clock.now < end:
        clock.advance(min(PREFILL_POLL_SECONDS, end - clock.now))
        await _cleanup_stale_local_dispatch(wedge_srv)
    return granted, snapshots


_WEDGE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "LP-0MVKUKEIQ003YOLA pre-fix wedge: a dispatched session whose prefill "
        "never produces a first byte retains its dispatch record (lease_renewed) "
        "and local_prefill_in_flight marker and re-dispatches immediately; the "
        "lease/marker must be released within <=5 min simulated with no "
        "pre-cooldown re-dispatch. Flip to green and remove this marker when "
        "C4/C5 land."
    ),
)


# ---------------------------------------------------------------------------
# Explicit-session path (lease retention)
# ---------------------------------------------------------------------------


@_WEDGE_XFAIL
@pytest.mark.asyncio
async def test_explicit_wedge_released_lease_and_marker_within_five_minutes(
    wedge_srv, clock
):
    """AC2: after the thrash window (<=5 min simulated) the lease and
    in-flight marker are released and at most the first dispatch was granted.

    Pre-fix: after 3 thrash cycles + sweep tail the record survives
    (inactive, future ``expires_at``) and the marker survives — the ~4 h
    wedge. The failure output documents the retained state and the
    immediate re-dispatch count.
    """
    granted, _snapshots = await _simulate_wedge_window(
        wedge_srv, clock, WEDGE_SESSION, session_explicit=True
    )
    report = _wedge_report(wedge_srv, WEDGE_SESSION, clock, granted)

    assert WEDGE_SESSION not in wedge_srv.local_dispatch_records, report
    assert WEDGE_SESSION not in wedge_srv.local_prefill_in_flight, report
    assert granted == 1, report


@_WEDGE_XFAIL
@pytest.mark.asyncio
async def test_explicit_detour_releases_lease_before_any_retry(wedge_srv, clock):
    """Parent AC2: a dispatch that never produced a first byte must not
    refresh the lease — the record and marker are released the moment the
    prefill deadline detours, before any retry can occur.

    Pre-fix the detour only marks the record inactive with a future
    ``expires_at`` (``lease_renewed timeout=60s``) and leaves the marker in
    place, which is what keeps the retry loop alive.
    """
    granted = await _prefill_attempt(wedge_srv, WEDGE_SESSION)
    assert granted, "the first dispatch must be admitted"
    await _detour_cleanup(wedge_srv, WEDGE_SESSION, session_explicit=True)

    report = _wedge_report(wedge_srv, WEDGE_SESSION, clock, int(granted))
    assert WEDGE_SESSION not in wedge_srv.local_dispatch_records, report
    assert WEDGE_SESSION not in wedge_srv.local_prefill_in_flight, report
    assert wedge_srv.local_active_queries == 0, report


# ---------------------------------------------------------------------------
# Non-explicit (anonymous) path — the marker leak that denies other sessions
# ---------------------------------------------------------------------------


@_WEDGE_XFAIL
@pytest.mark.asyncio
async def test_anonymous_detour_releases_the_prefill_marker(wedge_srv, clock):
    """AC4 (non-explicit): a detoured anonymous dispatch releases its
    ``local_prefill_in_flight`` marker.

    Pre-fix the non-explicit cleanup deletes the dispatch record immediately
    (``lease_released reason=non_explicit``) but the marker leaks forever (no
    record ages it out), pinning ``in_flight_prefills=1`` for the rest of the
    process life.
    """
    granted = await _prefill_attempt(wedge_srv, ANON_SESSION_A)
    assert granted, "the first dispatch must be admitted"
    await _detour_cleanup(wedge_srv, ANON_SESSION_A, session_explicit=False)

    report = _wedge_report(wedge_srv, ANON_SESSION_A, clock, int(granted))
    assert ANON_SESSION_A not in wedge_srv.local_prefill_in_flight, report
    assert helpers._get_prefill_in_flight_count(wedge_srv) == 0, report


@_WEDGE_XFAIL
@pytest.mark.asyncio
async def test_anonymous_wedge_does_not_deny_the_next_session(wedge_srv, clock):
    """AC4 (non-explicit): after a wedged anonymous dispatch detours, a
    different session is admitted again.

    Pre-fix the leaked marker saturates the prefill cap
    (``local_max_concurrent_prefills: 1``), denying every other session —
    the incident's 2,056 ``local_dispatch_denied`` / 138
    ``contention_queue_fallback_after_queue reason=local_lease_active``.
    """
    granted_a = await _prefill_attempt(wedge_srv, ANON_SESSION_A)
    assert granted_a, "the first dispatch must be admitted"
    await _detour_cleanup(wedge_srv, ANON_SESSION_A, session_explicit=False)

    granted_b, owner, active_count, retry_after = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=ANON_SESSION_B, backend=None
    )
    report = _wedge_report(wedge_srv, ANON_SESSION_A, clock, int(granted_a))
    assert granted_b, (
        f"{report} Next session denied: owner={owner} "
        f"in_flight_prefills={active_count} retry_after={retry_after}"
    )


# ---------------------------------------------------------------------------
# Determinism contract (AC3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulation_determinism_two_runs_identical(wedge_srv, clock):
    """AC3: the harness is deterministic — running the full window twice with
    fresh state yields identical outcomes (same grants, same retained state
    at each detour), so the regression cannot pass/fail on wall-clock noise.
    """
    start = clock.now
    granted_1, snapshots_1 = await _simulate_wedge_window(
        wedge_srv, clock, WEDGE_SESSION, session_explicit=True
    )
    assert clock.now - start == WINDOW_SECONDS
    assert granted_1 >= 1  # the dispatch gate admitted the first attempt

    # Fresh state, shifted epoch: run it again and compare the trace.
    wedge_srv.local_dispatch_records = {}
    wedge_srv.local_prefill_in_flight = {}
    wedge_srv.local_active_queries = 0
    clock.advance(1.0)  # shift the epoch to catch hidden wall-clock reads
    start_2 = clock.now
    granted_2, snapshots_2 = await _simulate_wedge_window(
        wedge_srv, clock, WEDGE_SESSION, session_explicit=True
    )
    assert clock.now - start_2 == WINDOW_SECONDS
    assert granted_1 == granted_2
    assert [s["record_present"] for s in snapshots_1] == [
        s["record_present"] for s in snapshots_2
    ]
    assert [s["marker_present"] for s in snapshots_1] == [
        s["marker_present"] for s in snapshots_2
    ]
