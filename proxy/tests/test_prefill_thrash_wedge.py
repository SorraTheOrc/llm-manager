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
   extends ``expires_at`` from the *in-flight* marker
   (``lease_extended_during_prefill progress=None alive=False in_flight=True``).
   Pre-fix that poll also refreshed ``last_progress_ts``, which is what kept
   the 180s no-progress watchdog from ever firing; post-fix (C5 AC4) an open
   request is not progress, so the clock stays stale and the watchdog remains
   armed.
3. The prefill deadline fires (``prefill_deadline_exceeded``, 90s): the
   request detours. Pre-fix the router ran only the generic dispatch cleanup
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

Post-fix contract
-----------------
C4 (post-deadline thrash cooldown, LP-0MVKUKFHH005UDWL) and C5 (truthful lease
and in-flight accounting, LP-0MVKUKFSO008GI6O) landed, so the contract tests
below are green and unmarked: the harness mirrors the *post-fix* router
sequence and a regression re-opens the defect loudly.

- C5: on ``prefill_deadline_exceeded`` the router now releases the dispatch
  lease and pops the ``local_prefill_in_flight`` marker
  (``lease_released reason=prefill_deadline``) before the detour cleanup —
  ``_detour_cleanup`` below mirrors that order.
- C4: the session's post-deadline cooldown suppresses re-dispatch
  (``routing_skip_local reason=prefill_thrash_cooldown``) —
  ``_prefill_attempt`` consults ``_prefill_thrash_cooldown_remaining`` the
  same way provider.py does.

Simulation model
----------------
- 1 simulated poll (10s cadence) == 20ms of wall time: the stub llama-server
  advances the injected clock by one poll interval per ``/slots`` probe, so
  the 90s prefill deadline fires after 9 probes and never blocks the event
  loop.
- The whole 5-minute window is 3 thrash cycles (90s each) plus a 30s tail of
  periodic ``_cleanup_stale_local_dispatch`` sweeps: the module runs in
  well under a second.

Extending the same harness: the post-deadline **in-request** local
re-dispatch and the un-engaged thrash gate (2026-10-26 10:41-10:47,
LP-0MVSAM2AV0077UAE) reuse the injected clock, the stale-``/slots`` stub and
the lease helpers — see the closing section, which pins which gap that
incident exposed.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from proxy.router import _cleanup_after_request
from proxy.router_helpers import (
    _cleanup_stale_local_dispatch,
    _try_acquire_local_dispatch,
)
from proxy.session import _resolve_session_id_header

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
    helpers._reset_prefill_thrash_cooldowns()
    yield server
    helpers._reset_prefill_thrash_cooldowns()
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
    # C4: while the session's post-deadline cooldown is armed, local routing
    # skips the dispatch (``routing_skip_local reason=prefill_thrash_cooldown``)
    # — provider.py consults the same helper.
    if helpers._prefill_thrash_cooldown_remaining(session) > 0:
        return False

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


async def _detour_cleanup(
    wedge_srv, session: str, *, session_explicit: bool, request=None
) -> None:
    """Run the router's post-deadline detour path.

    Mirrors ``proxy_to_local``'s deadline branch: C5 releases the lease and
    pops the in-flight marker (``lease_released reason=prefill_deadline``)
    and the C4 cooldown is armed *before* the generic cleanup runs. C2
    (LP-0MVSAMN14001DMTY AC2) additionally marks the *request* terminal for
    local, so the same request cannot re-dispatch local after the release.
    """
    if request is not None:
        helpers._mark_local_dispatch_terminal(request)
    await helpers._release_prefill_deadline_dispatch(wedge_srv, session)
    if session:
        helpers._record_prefill_thrash_cooldown(
            session, wedge_srv.config
        )
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


# ---------------------------------------------------------------------------
# Explicit-session path (lease retention)
# ---------------------------------------------------------------------------


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
    helpers._reset_prefill_thrash_cooldowns()
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


# ═════════════════════════════════════════════════════════════════════════════
# Post-deadline in-request local re-dispatch (LP-0MVSAM2AV0077UAE)
#
# Child C1 of LP-0MVPUBQLQ008GJOV. The 2026-10-26 10:39-10:47 incident
# (``/var/log/llama-proxy/proxy.log.2026-10-26_131429``) is a *different*
# shape from the 2026-10-23 wedge above: the lease really is released, yet a
# fresh one is acquired ~150 ms later **inside the same request**, and the
# 600 s thrash cooldown never produces a single skip line for that session:
#
#   10:39:36,396  prefill_deadline_exceeded session=01a177c6… deadline=120s
#   10:39:36,396  prefill_thrash_cooldown_started session=01a177c6… cooldown_s=600
#   10:39:36,396  lease_released session=01a177c6… reason=prefill_deadline
#   10:39:36,546  [local] POST http://192.168.0.199:8000/… session=01a177c6…  (+150 ms)
#   10:41:36,629  lease_released session=01a177c6… reason=prefill_deadline  (same prompt)
#   10:41:36,776  [local] POST … session=01a177c6…                          (+147 ms)
#   10:44:25,480  [local] POST … session=01a177c6…                          (fresh hold)
#   10:47:33,163  lease_released session=01a177c6… reason=no_progress_watchdog
#                 (~308 s hold; other sessions denied 10:44:43-10:47:08)
#
# and **zero** ``routing_skip_local reason=prefill_thrash_cooldown`` lines for
# ``01a177c6…`` across four successive local dispatches, while ``herdr-…``
# sessions were consistently skipped.
#
# What each test pins, and the cause determination it yielded:
#
# | gap                                                         | test | cause |
# |-------------------------------------------------------------|------|-------|
# | the in-request re-dispatch bypasses the gate (hypothesis ii) | AC1  | **real** — the retry loops re-enter ``proxy_to_local`` after provider.py's chain gate, so only a gate at the dispatch choke point engages |
# | arm key == check key on the incident request shape           | AC2a | refuted for the incident session (the headers resolve to the same id) |
# | a header-less request arms under an id no gate can resolve   | AC2b | **real** for anonymous sessions — covered by the request-scoped terminal marker |
# | the delta-body arm fingerprint never matches the gate's      | AC2c | **real** (post-2026-10-29 gap of the same family) — the fingerprint is now turn-scoped |
# | a released marker cannot re-extend a dead dispatch           | AC3  | **real** pre-fix, fixed by refusing the re-acquisition itself |
# | a legitimate same-session re-acquisition stays ≤ 300 s       | AC4/AC7 | regression pin (no change needed) |
#
# C1 (LP-0MVSAM2AV0077UAE) encoded these red-first with
# ``pytest.mark.xfail(strict=True)``; C2 (LP-0MVSAMN14001DMTY) removed every
# marker as it closed each gap — the assertions now run unmarked, which is the
# only state that proves the fix. The gates C2 added are pinned below:
# ``_try_acquire_local_dispatch`` (the lease choke point, AC1) and the
# request-scoped post-deadline marker (AC2).
# ═════════════════════════════════════════════════════════════════════════════

#: The incident session id (explicit: the request carried
#: ``x-client-request-id`` + ``x-session-affinity``, log 10:35:21 and 10:37:30).
INCIDENT_SESSION = "01a177c6-a2ce-727d-90d0-8cb3d01bab96"

#: The incident request's session headers, verbatim from the log.
INCIDENT_HEADERS = {
    "x-client-request-id": INCIDENT_SESSION,
    "x-session-affinity": INCIDENT_SESSION,
}

#: Observed gap between ``lease_released … reason=prefill_deadline`` and the
#: next ``[local] POST`` for the same session: 147-150 ms. It matches the
#: configured retry delay (``local_slot_exhaustion_retry_delay_seconds: 0.1``
#: in ``proxy/config.yaml``) that the in-request local retry loops sleep for.
REDISPATCH_DELAY_SECONDS = 0.15

#: Parent AC3 / this item's AC4: no hold may exceed 300 s simulated.
REDISPATCH_BOUND_SECONDS = 300.0

#: A prompt big enough that the router's ``local_prefill_deadline_seconds``
#: (120 s) detour is the documented outcome, used as the cancelled payload.
_DOOMED_PROMPT = "x" * 40_000


def _arm_deadline_cooldown(wedge_srv, session, clock) -> float:
    """Arm the post-deadline window exactly as ``router.py`` does.

    ``router.py`` arms it from the *post-delta* ``body_json`` it is about to
    prefill (``_handle_session`` binds ``body_json`` to the delta before the
    dispatch), with the prompt size it estimated.
    """
    return helpers._record_prefill_thrash_cooldown(
        session,
        wedge_srv.config,
        now=clock.now,
        tokens=23_272,  # the incident's estimated_tokens
        fingerprint=helpers._prefill_thrash_fingerprint(
            {"messages": [{"role": "user", "content": _DOOMED_PROMPT}]}
        ),
    )


async def _in_request_redispatch(
    wedge_srv,
    clock,
    *,
    lease_session,
    gate_session,
    gate_consulted: bool,
) -> bool:
    """Mirror the in-request local re-dispatch that follows a deadline detour.

    *gate_consulted* selects which provider.py shape is simulated:

    - ``False`` — the ``local_slot_exhaustion_retry`` / ``local_http_retry``
      loops (provider.py, ``_proxy_with_fallback_cycle``), which call
      ``_dispatch_local`` → ``proxy_to_local`` directly. That is the bypass
      shape the 2026-10-26 log exposes: no ``routing_skip_local``, no
      provider-cooldown line, just a ``[local] POST`` 147-150 ms after the
      release.
    - ``True`` — the provider chain's local-branch entry, which consults
      ``_prefill_thrash_cooldown_remaining`` (and logs the skip) *before*
      dispatching. Used as the control that proves the gate itself works.

    *gate_session* is the session id the gate site resolves (provider.py:
    ``_resolve_session_id_header(request.headers)``); *lease_session* is the
    id the dispatch path keys the lease on (router.py: ``session.session_id``).
    They are passed separately because reproducing the arm-key/check-key
    mismatch needs them to differ.

    Returns True when the dispatch acquired a (fresh) active lease.
    """
    # The measured 147-150 ms gap: the retry delay the retry loops sleep for.
    clock.advance(REDISPATCH_DELAY_SECONDS)
    if gate_consulted:
        remaining = helpers._prefill_thrash_cooldown_remaining(
            gate_session, wedge_srv.config, now=clock.now
        )
        if remaining > 0:
            wedge_srv.logger.info(
                "routing_skip_local provider=local-qwen3 session=%s "
                "cooldown_remaining=%.0fs reason=prefill_thrash_cooldown "
                "→ skipping local, routing to next remote provider",
                gate_session or "unknown",
                remaining,
            )
            return False
    acquired, _owner, _active, _retry_after = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=lease_session, backend=None
    )
    return acquired


def _redispatch_report(wedge_srv, session, clock, *, reacquired: bool) -> str:
    """Diagnosis used by the post-deadline contract asserts."""
    record = wedge_srv.local_dispatch_records.get(session)
    return (
        "LP-0MVPUBQLQ008GJOV 2026-10-26 pre-fix behaviour reproduced: after "
        "``lease_released reason=prefill_deadline`` the same session's "
        f"in-request re-dispatch re-acquired a lease={reacquired} "
        f"(record_active={bool(record.get('active')) if record else None}), "
        f"local_prefill_in_flight={session in wedge_srv.local_prefill_in_flight}, "
        f"now={clock.now:.2f}s. The incident: 4 local POSTs 147-150 ms after "
        "each release, 3 wedge denials for other sessions, ~308 s unproductive "
        "hold released only by the no-progress watchdog."
    )


def _skip_lines(wedge_srv) -> list[str]:
    """Every ``routing_skip_local`` line the harness logged, rendered.

    The harness's logger is a ``MagicMock``, so the recorded call holds the
    format string plus its arguments — render them the way the logging module
    would, so the assertion reads the line an operator sees in
    ``proxy.log``.
    """
    lines: list[str] = []
    for call in wedge_srv.logger.info.call_args_list:
        args = getattr(call, "args", ())
        if not args or "routing_skip_local" not in str(args[0]):
            continue
        try:
            lines.append(str(args[0]) % tuple(args[1:]))
        except Exception:
            lines.append(str(call))
    return lines


def _in_flight_extension_count(wedge_srv) -> int:
    """How many ``lease_extended_during_prefill … in_flight=True`` lines were logged.

    A count, not a boolean: the *live* first dispatch legitimately extends
    itself on the in-flight signal while it prefills, so the assertion has to
    be that the cancelled dispatch adds none **after** its release.
    """
    count = 0
    for call in wedge_srv.logger.info.call_args_list:
        args = getattr(call, "args", ())
        if not args or "lease_extended_during_prefill" not in str(args[0]):
            continue
        try:
            rendered = str(args[0]) % tuple(args[1:])
        except Exception:
            rendered = str(call)
        if "in_flight=True" in rendered:
            count += 1
    return count


# ---------------------------------------------------------------------------
# AC1/AC6 — the in-request re-dispatch must not re-arm the lease
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_request_redispatch_after_the_deadline_is_not_admitted(
    wedge_srv, clock
):
    """AC1/AC6 + parent AC1: a dispatch cancelled by
    ``prefill_deadline_exceeded`` is not re-dispatched locally inside the same
    request.

    The harness runs the exact sequence the log shows: acquire → prefill to
    the injected-clock deadline → ``_release_prefill_deadline_dispatch`` →
    same-session local re-dispatch 150 ms later **through the retry shape that
    skips the gate**. Pre-fix a fresh active record is acquired, so other
    sessions are denied again (the incident's
    ``contention_queue_fallback_after_queue reason=local_lease_active``); the
    fix must leave no record to deny them with.
    """
    granted = await _prefill_attempt(wedge_srv, INCIDENT_SESSION)
    assert granted, "the first dispatch must be admitted"
    await _detour_cleanup(wedge_srv, INCIDENT_SESSION, session_explicit=True)

    # The release itself is truthful today (C5) — these are the preconditions
    # the re-dispatch test builds on, not the assertion under test.
    assert INCIDENT_SESSION not in wedge_srv.local_dispatch_records
    assert INCIDENT_SESSION not in wedge_srv.local_prefill_in_flight
    assert helpers._prefill_thrash_cooldown_remaining(
        INCIDENT_SESSION, wedge_srv.config, now=clock.now
    ) > 0, "the deadline must arm the session's thrash cooldown"

    reacquired = await _in_request_redispatch(
        wedge_srv,
        clock,
        lease_session=INCIDENT_SESSION,
        gate_session=INCIDENT_SESSION,
        gate_consulted=False,  # the retry loops' bypass shape
    )
    report = _redispatch_report(
        wedge_srv, INCIDENT_SESSION, clock, reacquired=reacquired
    )
    assert not reacquired, report
    assert INCIDENT_SESSION not in wedge_srv.local_dispatch_records, report

    # The wedge symptom itself: with no re-armed record, an unrelated session
    # is admitted instead of denied with ``local_lease_active``.
    other_admitted, owner, active_count, retry_after = (
        await _try_acquire_local_dispatch(
            wedge_srv, max_local=1, session_key=ANON_SESSION_B, backend=None
        )
    )
    assert other_admitted, (
        f"{report} Next session denied: owner={owner} "
        f"active={active_count} retry_after={retry_after}"
    )


# ---------------------------------------------------------------------------
# AC2 — gate-engagement isolation: arm key vs check key, and the bypass path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_gate_engages_when_the_incident_session_is_checked(
    wedge_srv, clock
):
    """AC2 (hypothesis (i), incident shape): the arming site and the gate site
    resolve the *same* session id, so the 2026-10-26 non-engagement was not a
    session-identity mismatch.

    The incident request carried ``x-client-request-id`` and
    ``x-session-affinity`` (log 10:35:21, 10:37:30) and the router logged
    ``Session header resolved: source=session_id``. ``_resolve_session_id_header``
    — the resolver provider.py's gate uses — therefore returns the very id
    router.py armed the window under, and the gate refuses the attempt and logs
    ``routing_skip_local reason=prefill_thrash_cooldown``.

    Together with AC1 this isolates the gap: identical state, identical keys —
    the gate engages when consulted, so the incident's missing skip proves the
    dispatch path never consulted it (bypass), not that the keys disagreed.
    """
    granted = await _prefill_attempt(wedge_srv, INCIDENT_SESSION)
    assert granted, "the first dispatch must be admitted"
    await _detour_cleanup(wedge_srv, INCIDENT_SESSION, session_explicit=True)

    check_key, source = _resolve_session_id_header(INCIDENT_HEADERS)
    assert check_key == INCIDENT_SESSION, (
        "the arm key (router.py ``session.session_id``) and the check key "
        f"(provider.py ``_session_id``, resolved from {source}) must agree: "
        f"{INCIDENT_SESSION!r} != {check_key!r}"
    )
    assert helpers._prefill_thrash_cooldown_remaining(
        check_key, wedge_srv.config, now=clock.now
    ) > 0

    admitted = await _in_request_redispatch(
        wedge_srv,
        clock,
        lease_session=INCIDENT_SESSION,
        gate_session=check_key,
        gate_consulted=True,
    )
    assert not admitted, _redispatch_report(
        wedge_srv, INCIDENT_SESSION, clock, reacquired=admitted
    )
    assert INCIDENT_SESSION not in wedge_srv.local_dispatch_records

    skip_lines = _skip_lines(wedge_srv)
    assert skip_lines, "the engaged skip must be attributable in the log"
    assert "reason=prefill_thrash_cooldown" in skip_lines[-1]
    assert f"session={INCIDENT_SESSION}" in skip_lines[-1]


@pytest.mark.asyncio
async def test_a_header_less_session_cannot_engage_the_gate(wedge_srv, clock):
    """AC2 (hypothesis (i), anonymous shape): the arm key and the check key
    genuinely disagree when the request carries no session header.

    ``_handle_session`` → ``session_manager.get_or_create(None)`` *generates* a
    session id, which router.py arms the window under, while provider.py's
    gate resolves the request's headers itself and finds nothing
    (``_prefill_thrash_cooldown_remaining(None)`` is always 0). An anonymous
    session's cooldown can therefore never engage — a second, independent gap
    of the same family as the bypass.

    Consequence pinned here: the post-deadline re-dispatch of a header-less
    request must still be refused (any mechanism: the in-request guard AC6
    allows, or an identity the gate can resolve), i.e. the fix cannot be
    cooldown-only.
    """
    generated_session = "01a177c6-0000-7000-8000-00000000anon"
    armed = _arm_deadline_cooldown(wedge_srv, generated_session, clock)
    assert armed > 0, "the router arms the window for a generated session id"

    # provider.py resolves nothing from a header-less request ...
    check_key, source = _resolve_session_id_header({})
    assert check_key is None and source is None
    assert (
        helpers._prefill_thrash_cooldown_remaining(
            check_key, wedge_srv.config, now=clock.now
        )
        == 0.0
    ), "the gate has no key to look the armed window up with"

    # ... so an attempt that consults the gate with its own key is admitted.
    admitted = await _in_request_redispatch(
        wedge_srv,
        clock,
        lease_session=generated_session,
        gate_session=check_key,
        gate_consulted=True,
    )
    assert not admitted, (
        "a header-less request re-dispatched locally after its prefill "
        f"deadline: the window armed under {generated_session!r} is "
        "unreachable for a gate that resolves no session id, so the "
        "in-request re-dispatch must be refused independently of it."
    )


def test_delta_arm_fingerprint_matches_the_gate_check_fingerprint(wedge_srv, clock):
    """AC2 (arm-key/check-key mismatch, fingerprint component).

    The window is keyed by ``(session, prompt fingerprint)`` since
    LP-0MVT2KSXY002O75C (2026-10-29, i.e. after this incident). The two sites
    fingerprint **different bodies**: router.py arms from the ``body_json``
    ``_handle_session`` has already rewritten to the *delta* messages it is
    about to prefill, while provider.py fingerprints the *original* request
    body it parsed before the chain started. For any continuing session the
    two digests differ, so the gate looks up a key nobody armed and never
    skips — the same non-engagement, one level down.
    """
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "ack"},
    ]
    turn = [{"role": "user", "content": f"the doomed prompt {_DOOMED_PROMPT}"}]
    full_messages = history + turn  # what provider.py fingerprints
    delta_messages = list(turn)  # what router.py fingerprints (and prefills)
    assert delta_messages != full_messages

    arm_fingerprint = helpers._prefill_thrash_fingerprint(
        {"messages": delta_messages}
    )
    check_fingerprint = helpers._prefill_thrash_fingerprint(
        {"messages": full_messages}
    )
    assert arm_fingerprint and check_fingerprint

    helpers._record_prefill_thrash_cooldown(
        INCIDENT_SESSION,
        wedge_srv.config,
        now=clock.now,
        tokens=23_272,
        fingerprint=arm_fingerprint,
    )
    remaining = helpers._prefill_thrash_cooldown_remaining(
        INCIDENT_SESSION,
        wedge_srv.config,
        now=clock.now,
        fingerprint=check_fingerprint,
    )
    assert remaining > 0, (
        "the gate cannot find the window the router armed: "
        f"arm fingerprint {arm_fingerprint!r} (delta body) != check "
        f"fingerprint {check_fingerprint!r} (original body), so a continuing "
        "session's post-deadline cooldown never engages."
    )


# ---------------------------------------------------------------------------
# AC3 — the cancelled dispatch's marker must not outlive it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_released_marker_cannot_re_extend_the_cancelled_dispatch(
    wedge_srv, clock
):
    """AC3 + parent AC2: after the cancellation the
    ``local_prefill_in_flight`` marker is cleared and
    ``_extend_lease_during_prefill`` can no longer re-extend from it
    (``lease_extended_during_prefill … in_flight=True`` must not appear for the
    cancelled dispatch).

    The incident's tell was exactly this: extensions with
    ``progress=None alive=False in_flight=True`` resumed 10 s after each
    release, which is only possible if something re-armed an active record and
    its marker.
    """
    granted = await _prefill_attempt(wedge_srv, INCIDENT_SESSION)
    assert granted, "the first dispatch must be admitted"
    await _detour_cleanup(wedge_srv, INCIDENT_SESSION, session_explicit=True)

    # The release clears the marker: there is nothing left to extend.
    assert INCIDENT_SESSION not in wedge_srv.local_prefill_in_flight
    _progress, extended = await helpers._extend_lease_during_prefill(
        wedge_srv,
        INCIDENT_SESSION,
        llama_port=8080,
        model_name="qwen3",
        slot_id=0,
        last_progress=0,
        prefill_in_flight=True,
    )
    assert extended is False, "no record exists, so nothing may be extended"

    # The in-request re-dispatch must not resurrect the marker/record, so the
    # extension path cannot resume for a dispatch that never produced a byte.
    extensions_before = _in_flight_extension_count(wedge_srv)
    reacquired = await _in_request_redispatch(
        wedge_srv,
        clock,
        lease_session=INCIDENT_SESSION,
        gate_session=INCIDENT_SESSION,
        gate_consulted=False,
    )
    report = _redispatch_report(
        wedge_srv, INCIDENT_SESSION, clock, reacquired=reacquired
    )
    assert INCIDENT_SESSION not in wedge_srv.local_prefill_in_flight, report
    _progress, extended_after = await helpers._extend_lease_during_prefill(
        wedge_srv,
        INCIDENT_SESSION,
        llama_port=8080,
        model_name="qwen3",
        slot_id=0,
        last_progress=0,
        prefill_in_flight=True,
    )
    assert extended_after is False, report
    assert _in_flight_extension_count(wedge_srv) == extensions_before, (
        f"{report} A dead dispatch must never produce a new "
        "``lease_extended_during_prefill … in_flight=True`` after its release."
    )


# ---------------------------------------------------------------------------
# AC4/AC7 — the remaining hold stays inside the epic's 5-minute bound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_session_reacquisition_hold_is_bounded_within_300s(
    wedge_srv, clock
):
    """AC4 + parent AC3: a legitimate same-session re-acquisition (no-preemption
    is load-bearing for genuine long prefills) is bounded at ≤300 s simulated.

    The incident's hold chain was released only by the no-progress watchdog at
    ≈308 s. This pins the bound without weakening either knob: the watchdog
    timeout (180 s) and the 10 s sweep cadence are unchanged, and the record is
    extended every poll by the in-flight signal — which must not refresh the
    no-progress clock (C5 AC4), so the watchdog still fires.
    """
    acquired, owner, active, retry_after = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=WEDGE_SESSION, backend=None
    )
    assert acquired, f"admission must grant the hold: owner={owner} active={active}"

    start = clock.now
    while clock.now - start <= WINDOW_SECONDS:
        # One prefill poll: the stale-/slots stub advances the injected clock
        # by the poll interval and reports no progress.
        await helpers._extend_lease_during_prefill(
            wedge_srv,
            WEDGE_SESSION,
            llama_port=8080,
            model_name="qwen3",
            slot_id=0,
            last_progress=0,
            prefill_in_flight=True,
        )
        await _cleanup_stale_local_dispatch(wedge_srv)
        if WEDGE_SESSION not in wedge_srv.local_dispatch_records:
            break

    held = clock.now - start
    report = (
        f"LP-0MVPUBQLQ008GJOV AC3: {held:.0f}s simulated hold "
        f"(record_present={WEDGE_SESSION in wedge_srv.local_dispatch_records}, "
        f"marker_present={WEDGE_SESSION in wedge_srv.local_prefill_in_flight}); "
        f"the 300 s bound must hold without weakening the "
        f"{NO_PROGRESS_TIMEOUT_SECONDS:.0f}s no-progress watchdog or the "
        f"{PREFILL_POLL_SECONDS:.0f}s sweep cadence."
    )
    assert WEDGE_SESSION not in wedge_srv.local_dispatch_records, report
    assert held <= REDISPATCH_BOUND_SECONDS, report
    assert WEDGE_SESSION not in wedge_srv.local_prefill_in_flight, report


# ---------------------------------------------------------------------------
# AC1 — the gate sits at the lease acquisition choke point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_choke_point_refuses_an_armed_session_and_logs_the_skip(
    wedge_srv, clock
):
    """AC1: ``_try_acquire_local_dispatch`` — the single choke point every
    local dispatch passes before a record exists — consults the session's
    post-deadline cooldown and refuses, logging the engaged skip.

    This is the mechanism the incident requires. provider.py's chain gate is
    skipped by the in-request retry loops (they re-enter ``proxy_to_local``
    *after* the chain decided), so a gate that lives only there never engages
    — the 2026-10-26 session logged zero skips across four dispatches while
    its 600 s window was armed. The same gate also covers the anonymous
    ``_increment_local_active_queries`` path and ``ui.py``'s direct dispatch.
    """
    helpers._record_prefill_thrash_cooldown(
        INCIDENT_SESSION, wedge_srv.config, now=clock.now
    )

    acquired, _owner, _active, retry_after = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=INCIDENT_SESSION, backend=None
    )
    assert not acquired, (
        "a session whose post-deadline cooldown is armed must not acquire a "
        "local dispatch lease at the choke point"
    )
    assert retry_after >= 1.0
    assert INCIDENT_SESSION not in wedge_srv.local_dispatch_records
    assert INCIDENT_SESSION not in wedge_srv.local_prefill_in_flight

    skip_lines = _skip_lines(wedge_srv)
    assert skip_lines, "the engaged skip must be attributable in the log"
    assert "reason=prefill_thrash_cooldown" in skip_lines[-1]
    assert f"session={INCIDENT_SESSION}" in skip_lines[-1]
    assert "entry=lease_acquire" in skip_lines[-1]

    # Cross-session isolation: another session is still admitted.
    other_admitted, owner, _active, _retry_after = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=ANON_SESSION_B, backend=None
    )
    assert other_admitted, f"another session must stay admissible: owner={owner}"


@pytest.mark.asyncio
async def test_lease_choke_point_still_refuses_without_a_fingerprint(
    wedge_srv, clock
):
    """AC1 (fail-open direction): a caller that cannot fingerprint its body
    (``body_json=None``) still has to assume it might be the failing prompt,
    so the choke point refuses on any of the session's windows — the gate
    keeps engaging, it can never silently stop refusing."""
    helpers._record_prefill_thrash_cooldown(
        INCIDENT_SESSION, wedge_srv.config, now=clock.now
    )
    acquired, *_ = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=INCIDENT_SESSION, backend=None
    )
    assert not acquired
    clock.advance(helpers._get_prefill_thrash_cooldown_seconds(wedge_srv.config) + 1)
    acquired_after_expiry, *_ = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=INCIDENT_SESSION, backend=None
    )
    assert acquired_after_expiry, "the gate must release when the window expires"


@pytest.mark.asyncio
async def test_lease_choke_point_gate_is_disabled_by_the_rollback_knob(
    wedge_srv, clock, monkeypatch
):
    """AC1 (rollback knob): ``local_prefill_thrash_cooldown_seconds: 0``
    disables the gate at the choke point too, so the documented rollback
    restores the pre-fix behaviour everywhere rather than only in provider.py.
    """
    helpers._record_prefill_thrash_cooldown(
        INCIDENT_SESSION, wedge_srv.config, now=clock.now
    )
    disabled = {"server": dict(wedge_srv.config["server"])}
    disabled["server"]["local_prefill_thrash_cooldown_seconds"] = 0
    monkeypatch.setattr(server, "config", disabled, raising=False)

    acquired, *_ = await _try_acquire_local_dispatch(
        wedge_srv, max_local=1, session_key=INCIDENT_SESSION, backend=None
    )
    assert acquired, "the rollback knob must disable the gate, not just arming"


# ---------------------------------------------------------------------------
# AC2 — the post-deadline detour is terminal for local inside the request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_deadline_detour_is_terminal_for_that_request(wedge_srv, clock):
    """AC2: after ``prefill_deadline_exceeded`` the *request* is marked, so
    local is terminal for it even when the session cooldown cannot be
    resolved — the header-less shape C1 pinned (a generated session id no gate
    site can re-derive from the headers).

    The cooldown is deliberately cleared before the assertions so only the
    request-scoped marker can refuse: the marker alone must be enough.
    """
    request = SimpleNamespace()
    granted = await _prefill_attempt(wedge_srv, INCIDENT_SESSION)
    assert granted, "the first dispatch must be admitted"
    await _detour_cleanup(
        wedge_srv, INCIDENT_SESSION, session_explicit=True, request=request
    )

    helpers._reset_prefill_thrash_cooldowns()
    assert (
        helpers._prefill_thrash_cooldown_remaining(
            INCIDENT_SESSION, wedge_srv.config, now=clock.now
        )
        == 0
    )
    assert helpers._local_dispatch_terminal_at(request) is not None

    refused, reason, remaining = helpers._local_dispatch_gate(
        wedge_srv,
        INCIDENT_SESSION,
        request=request,
        config=wedge_srv.config,
        entry="test",
    )
    assert refused and reason == "post_deadline_terminal" and remaining == 0.0

    acquired, *_ = await _try_acquire_local_dispatch(
        wedge_srv,
        max_local=1,
        session_key=INCIDENT_SESSION,
        backend=None,
        request=request,
    )
    assert not acquired, "local is terminal for the rest of the cancelled request"
    assert INCIDENT_SESSION not in wedge_srv.local_dispatch_records

    skip_lines = _skip_lines(wedge_srv)
    assert skip_lines and "reason=post_deadline_terminal" in skip_lines[-1]


@pytest.mark.asyncio
async def test_the_terminal_marker_does_not_block_a_concurrent_request(
    wedge_srv, clock
):
    """AC2 (risky-assumption guard): the marker is request-scoped, so a
    *concurrent* unrelated request from the same session is not blocked by it.

    Only the session-wide cooldown covers later requests; the marker must not
    leak across requests, or one cancelled dispatch would deny the session
    locally for as long as that ``Request`` object lived.
    """
    cancelled_request = SimpleNamespace()
    helpers._mark_local_dispatch_terminal(cancelled_request)

    concurrent_admitted, owner, _active, _retry_after = (
        await _try_acquire_local_dispatch(
            wedge_srv, max_local=1, session_key=INCIDENT_SESSION, backend=None
        )
    )
    assert concurrent_admitted, (
        "an unrelated request from the same session must stay admissible: "
        f"owner={owner}"
    )


def test_the_request_marker_survives_only_on_its_request():
    """AC2: the marker helpers are request-scoped, and fail soft for a
    request object that cannot carry attributes."""
    class _Slotted:
        __slots__ = ()

    marked = SimpleNamespace()
    assert helpers._mark_local_dispatch_terminal(marked, now=12.5) == 12.5
    assert helpers._local_dispatch_terminal_at(marked) == 12.5
    assert helpers._local_dispatch_terminal_at(SimpleNamespace()) is None
    assert helpers._local_dispatch_terminal_at(None) is None
    assert helpers._mark_local_dispatch_terminal(None) is None
    assert helpers._mark_local_dispatch_terminal(_Slotted()) is None


# ---------------------------------------------------------------------------
# AC2 — the arm and check fingerprints must agree for one request
# ---------------------------------------------------------------------------


def test_turn_scoped_fingerprint_still_distinguishes_different_turns():
    """AC2 (scope preserved): turn scoping must not collapse the prompt
    identity — two different turns (different content or order) still yield
    different digests, so the window keeps refusing only the prompt that
    failed."""
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ack"}]
    turn_p = [{"role": "user", "content": f"prompt P {_DOOMED_PROMPT}"}]
    turn_q = [{"role": "user", "content": "prompt Q"}]
    fp_p = helpers._prefill_thrash_fingerprint({"messages": history + turn_p})
    fp_q = helpers._prefill_thrash_fingerprint({"messages": history + turn_q})
    assert fp_p and fp_q and fp_p != fp_q

    # The same turn keeps the same digest no matter how much history precedes
    # it — that is what makes the router's delta body and the provider's
    # original body agree.
    longer_history = history + [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ack2"},
    ]
    assert (
        helpers._prefill_thrash_fingerprint({"messages": longer_history + turn_p})
        == fp_p
    )


# ---------------------------------------------------------------------------
# AC3 — the release clears the lease/marker before the stream close returns
# ---------------------------------------------------------------------------


class _RecordingStream:
    """Upstream stream stub that records the release state when closed."""

    def __init__(self, srv, session: str) -> None:
        self._srv = srv
        self._session = session
        self.closed = False
        self.record_present_at_close = None
        self.marker_present_at_close = None

    async def __aexit__(self, *_exc) -> bool:
        self.closed = True
        self.record_present_at_close = self._session in self._srv.local_dispatch_records
        self.marker_present_at_close = (
            self._session in self._srv.local_prefill_in_flight
        )
        return False


@pytest.mark.asyncio
async def test_the_deadline_release_clears_state_before_closing_the_stream(
    wedge_srv, clock
):
    """AC3: the dispatch record and ``local_prefill_in_flight`` marker are
    gone *before* the cancelled upstream stream is closed (and the close is
    awaited before the release returns), so a retry that lands immediately
    afterwards cannot re-extend the dispatch it just cancelled
    (``_extend_lease_during_prefill`` requires both the record and the
    session's own marker)."""
    granted = await _prefill_attempt(wedge_srv, INCIDENT_SESSION)
    assert granted, "the first dispatch must be admitted"
    stream = _RecordingStream(wedge_srv, INCIDENT_SESSION)

    removed = await helpers._release_prefill_deadline_dispatch(
        wedge_srv,
        INCIDENT_SESSION,
        upstream_stream_cm=stream,
    )
    assert removed
    assert stream.closed, "the release must close (and await) the upstream stream"
    assert stream.record_present_at_close is False
    assert stream.marker_present_at_close is False
    assert INCIDENT_SESSION not in wedge_srv.local_dispatch_records
    assert INCIDENT_SESSION not in wedge_srv.local_prefill_in_flight


def test_the_abandoned_stream_close_timeout_honours_the_configured_budget():
    """AC3: the close bound is the request's own
    ``disconnect_cleanup_timeout`` (default 5 s), not a hard-coded copy of it —
    and a missing/malformed/non-positive value keeps the bounded default
    rather than becoming unbounded."""
    default_srv = SimpleNamespace(config={"server": {}})
    assert helpers._abandoned_stream_close_timeout(default_srv) == 5.0

    configured_srv = SimpleNamespace(config={"server": {"disconnect_cleanup_timeout": 2.5}})
    assert helpers._abandoned_stream_close_timeout(configured_srv) == 2.5

    for bad in ("oops", None, 0, -1):
        broken_srv = SimpleNamespace(
            config={"server": {"disconnect_cleanup_timeout": bad}}
        )
        assert helpers._abandoned_stream_close_timeout(broken_srv) == 5.0
