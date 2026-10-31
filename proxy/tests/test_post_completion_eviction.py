"""A completed local dispatch must not be evicted after completion.

Parent work item: LP-0MVWEZCB0006WPBQ **AC8** — *no post-completion evictions*.
Red-first contract for LP-0MVWFKFFY002RU78 (this item, tests only); the paired
implementation item that turns the pinned assertion green is
LP-0MVWFKI8O001X694.

The defect (measured 2026-10-31, independent proxy-log review)
--------------------------------------------------------------

A local dispatch that has already been **fully served** — the request it
belonged to completed and the client got its response — can still be evicted
minutes later by the tenancy/lease sweep. Observed on an audit dispatch that
completed at 12:43:28 and was evicted at 12:45:55::

    12:45:55,577 - WARNING - lease_released session=audit-…-parent-1c9672b2 \
        reason=no_progress_watchdog endpoint=http://localhost:8080
    12:45:55,578 - ERROR - local_eviction session=audit-…-parent-1c9672b2 \
        reason=tenancy_lease_expired …
    12:45:55,578 - ERROR - local_eviction_after_acceptance session=audit-… \
        reason=no_progress_watchdog … elapsed_s=186.8

Mechanism (confirmed in the log, not inferred): the request acquired a local
dispatch lease, the local attempt ended without a first byte, the provider
chain re-routed the request to a remote provider, the request then completed
normally — and the local dispatch record was **never released**. Nothing marks
it terminal, so it stays ``active=True`` with a stale ``last_progress_ts`` and
the ``no_progress_watchdog`` branch of
``router_helpers._cleanup_stale_local_dispatch`` reclaims it
``local_dispatch_no_progress_timeout_seconds`` (180 s) later, emitting the
``tenancy_lease_expired`` eviction and the ``no_progress_watchdog``
after-acceptance ERROR. That eviction feeds the local entry-cooldown machinery
(the sibling defect tracked as parent AC1), so a request that *succeeded* ends
up quarantining the local route.

This is a second, independent path from the measured-cache-deficit abort: the
request did not abort, the **lease outlived it**.

What this file pins
-------------------

* ``test_completed_dispatch_is_not_evicted_after_completion`` — the post-fix
  expectation (AC8/AC8b): a completed dispatch's lease is released (or marked
  terminal) at completion time, so advancing the injected clock past the
  lease and watchdog horizons produces **no** ``local_eviction
  reason=tenancy_lease_expired``, **no**
  ``local_eviction_after_acceptance reason=no_progress_watchdog`` and **no**
  local entry cooldown. Red against today's behaviour, pinned with
  ``xfail(strict=True)`` so this test-only item can land on ``dev`` with a
  green suite; ``strict`` makes an unexpectedly passing test fail, so the
  marker cannot be forgotten by LP-0MVWFKI8O001X694.
* ``test_locally_served_dispatch_completes_without_eviction`` — control: the
  same harness driving a dispatch that is served *and* completed on the local
  endpoint produces no eviction today, so the assertion above is not vacuous.
* ``test_genuinely_stalled_dispatch_is_still_evicted`` — inverse control: a
  dispatch that never completes is still reclaimed by the watchdog with
  ``reason=no_progress_watchdog``, so the fix cannot pass by silencing the
  watchdog for real stalls.

Harness idioms are taken from ``test_abandoned_local_dispatch.py`` (injected
monotonic clock + fake server state) and
``test_measured_cache_abort_wiring.py`` (real lease bookkeeping rather than
hand-built records). Deterministic: no sleeps, no network, no real
llama-server.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import proxy.router as router
import proxy.router_helpers as helpers
import pytest

from proxy import admission_events, provider

#: The local provider entry whose cooldown the ghost eviction must not arm.
LOCAL_ENTRY = "local-qwen3"
#: A session id shaped like the audit sessions in the measured evidence.
SESSION = "audit-LP-0MVWEZCB0006WPBQ-parent-1c9672b2"
#: The llama-server endpoint the dispatch is leased against.
LOCAL_ENDPOINT = "http://localhost:8080"
#: The endpoint a re-routed request actually completes on.
REMOTE_ENDPOINT = "https://opencode.ai/zen/go"

#: The horizons the sweep must survive without evicting a completed dispatch:
#: the (adaptive) lease timeout, then the no-progress watchdog timeout.
LEASE_TIMEOUT_S = 30.0
NO_PROGRESS_TIMEOUT_S = 180.0

SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "session_slot_pool_size": 3,
        "local_dispatch_lease_timeout_seconds": int(LEASE_TIMEOUT_S),
        "local_dispatch_lease_per_token_seconds": 0.0,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 10,
        "local_dispatch_lease_prefill_buffer_seconds": 30,
        "local_dispatch_lease_chunk_refresh_buffer_seconds": 30,
        "local_dispatch_no_progress_timeout_seconds": int(NO_PROGRESS_TIMEOUT_S),
        "local_dispatch_max_prefill_seconds": 900,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 0.3,
        "session_guardrail_max_runtime_seconds": 3600,
        "session_guardrail_max_completion_tokens": 4096,
    }
}


# ---------------------------------------------------------------------------
# Injected clock (mirrors test_abandoned_local_dispatch.py)
# ---------------------------------------------------------------------------


class _InjectedClock:
    """Monotonic clock that only advances when the test drives it.

    Only the sweep reads this clock (``router_helpers.time`` is shimmed), so
    the ``elapsed_s`` field the eviction emitter renders — it measures against
    the *real* clock — is a harness artefact here and is deliberately not
    asserted; the dispatch age under test is driven by ``advance()``.
    """

    def __init__(self) -> None:
        self._now = 0.0

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _TimeShim:
    """Replace ``router_helpers.time`` so only ``monotonic`` is injected."""

    def __init__(self, clock: _InjectedClock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(time, name)


@pytest.fixture
def clock() -> _InjectedClock:
    return _InjectedClock()


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, clock: _InjectedClock):
    """Isolate the clock, the eviction counters, the cooldowns and the registry."""
    monkeypatch.setattr(helpers, "time", _TimeShim(clock))
    admission_events.reset_eviction_after_acceptance_counters()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    reset_abandoned = getattr(helpers, "_reset_abandoned_slots", None)
    reset_thrash = getattr(helpers, "_reset_prefill_thrash_cooldowns", None)
    if reset_abandoned is not None:
        reset_abandoned()
    if reset_thrash is not None:
        reset_thrash()
    yield
    admission_events.reset_eviction_after_acceptance_counters()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    if reset_abandoned is not None:
        reset_abandoned()
    if reset_thrash is not None:
        reset_thrash()


def _make_srv() -> SimpleNamespace:
    """Fake server state with a *real* logger so eviction lines are capturable."""
    return SimpleNamespace(
        config={"server": dict(SERVER_CONFIG["server"])},
        logger=logging.getLogger("llama-proxy"),
        active_queries=0,
        active_queries_lock=asyncio.Lock(),
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_sessions=set(),
        local_generating_queries_lock=asyncio.Lock(),
        local_prefill_in_flight={},
        local_prefill_in_flight_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        per_model_queries={},
        per_model_queries_lock=asyncio.Lock(),
    )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _eviction_lines(caplog, token: str) -> list[str]:
    """Every captured ``llama-proxy`` line containing *token*."""
    return [
        record.getMessage()
        for record in caplog.records
        if token in record.getMessage()
    ]


def _after_acceptance_by_reason() -> dict[str, int]:
    return dict(admission_events.eviction_after_acceptance_counters()["by_reason"])


async def _acquire(srv: SimpleNamespace) -> None:
    """Acquire a real local dispatch lease for :data:`SESSION`."""
    acquired, _owner, _active, _retry = await helpers._try_acquire_local_dispatch(
        srv,
        SERVER_CONFIG["server"]["max_concurrent_queries"],
        SESSION,
        LOCAL_ENDPOINT,
        {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
        "Qwen3",
    )
    assert acquired is True, "the harness must obtain the local dispatch lease"
    assert (LOCAL_ENDPOINT, SESSION) in srv.local_dispatch_records


async def _finish_request(srv: SimpleNamespace, *, endpoint: str | None) -> None:
    """Run the router's end-of-request cleanup for the endpoint it ended on."""
    await router._cleanup_after_request(
        srv,
        SESSION,
        decrement_local=True,
        session_explicit=True,
        model_name="Qwen3",
        backend=endpoint,
    )


async def _sweep_past_every_horizon(srv: SimpleNamespace, clock: _InjectedClock) -> None:
    """Advance past the lease and watchdog horizons, sweeping at each one.

    The lease sweep runs first (``idle_timeout`` territory), then the clock
    advances past ``local_dispatch_no_progress_timeout_seconds`` so the
    ``no_progress_watchdog`` branch — the one in the measured evidence — is
    reached.
    """
    clock.advance(LEASE_TIMEOUT_S + 1.0)
    await helpers._cleanup_stale_local_dispatch(srv)
    clock.advance(NO_PROGRESS_TIMEOUT_S + 1.0)
    await helpers._cleanup_stale_local_dispatch(srv)


# ---------------------------------------------------------------------------
# AC8 / AC8b — the red contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "AC8 red: a completed dispatch's local lease outlives the request, so "
        "the no_progress_watchdog sweep reclassifies it as "
        "local_eviction_after_acceptance reason=no_progress_watchdog and "
        "reclaims the tenancy (tenancy_lease_expired) ~180 s later. "
        "LP-0MVWFKI8O001X694 removes this marker."
    ),
)
async def test_completed_dispatch_is_not_evicted_after_completion(clock, caplog):
    """A request that completed must not leave a lease for the sweep to evict.

    Drives the measured 2026-10-31 sequence through the real lease
    bookkeeping: the local dispatch is admitted, the local attempt ends
    without a first byte, the request is re-routed and completes on the remote
    provider. The clock is then advanced past the lease timeout *and* the
    no-progress watchdog timeout while the real sweep runs.

    Post-fix expectation (AC8b), stated explicitly:

    * the completed dispatch's lease is released (or marked terminal) at
      completion time, so the sweep never emits
      ``local_eviction_after_acceptance reason=no_progress_watchdog``;
    * the tenancy is not reclaimed as ``tenancy_lease_expired`` either — that
      eviction is what feeds the local entry-cooldown machinery;
    * no cooldown is armed for the ``local-qwen3`` entry by this path.

    Red today: the local record stays ``active=True`` with a stale
    ``last_progress_ts`` because the end-of-request cleanup targets the
    endpoint the request actually completed on, and the watchdog reclaims it.
    """
    srv = _make_srv()
    await _acquire(srv)

    # The request completes on the re-routed (remote) provider — this is the
    # router's real end-of-request path, not a hand-built record mutation.
    await _finish_request(srv, endpoint=REMOTE_ENDPOINT)

    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        await _sweep_past_every_horizon(srv, clock)

    after_acceptance = _eviction_lines(caplog, "local_eviction_after_acceptance")
    assert after_acceptance == [], (
        "a completed dispatch was evicted after acceptance: "
        f"{after_acceptance} — the lease outlived the request"
    )

    tenancy_expiry = [
        line
        for line in _eviction_lines(caplog, "local_eviction ")
        if "reason=tenancy_lease_expired" in line
    ]
    assert tenancy_expiry == [], (
        "a completed dispatch's tenancy was reclaimed after completion: "
        f"{tenancy_expiry}"
    )

    assert _after_acceptance_by_reason() == {}, (
        "the ghost eviction was counted: "
        f"{_after_acceptance_by_reason()}"
    )
    assert provider._provider_cooldown_remaining(LOCAL_ENTRY) == 0, (
        "the post-completion eviction armed a local entry cooldown"
    )


# ---------------------------------------------------------------------------
# Control — the same harness on a dispatch that is served and completed locally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_locally_served_dispatch_completes_without_eviction(clock, caplog):
    """Control: a dispatch served *and* completed locally is not evicted.

    Proves the red assertion above is not vacuous: with the served marker set
    and the end-of-request cleanup targeting the local endpoint, today's code
    releases the lease and the sweep stays quiet.
    """
    srv = _make_srv()
    await _acquire(srv)
    helpers.mark_local_first_byte(srv, SESSION, endpoint=LOCAL_ENDPOINT)
    await _finish_request(srv, endpoint=LOCAL_ENDPOINT)

    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        await _sweep_past_every_horizon(srv, clock)

    assert _eviction_lines(caplog, "local_eviction_after_acceptance") == []
    assert [
        line
        for line in _eviction_lines(caplog, "local_eviction ")
        if "reason=tenancy_lease_expired" in line
    ] == []
    assert _after_acceptance_by_reason() == {}
    # The lease itself is gone — released, not merely reported quietly.
    assert (LOCAL_ENDPOINT, SESSION) not in srv.local_dispatch_records


# ---------------------------------------------------------------------------
# Inverse control — a genuine stall is still reclaimed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_genuinely_stalled_dispatch_is_still_evicted(clock, caplog):
    """Inverse control: an unserved dispatch with no completion is still evicted.

    The fix for AC8 must not silence the watchdog for real stalls: a dispatch
    that is never completed (no end-of-request cleanup) is reclaimed with
    ``reason=no_progress_watchdog`` and counted, exactly as the abandoned
    dispatch suite expects.
    """
    srv = _make_srv()
    await _acquire(srv)

    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        await _sweep_past_every_horizon(srv, clock)

    after_acceptance = _eviction_lines(caplog, "local_eviction_after_acceptance")
    assert len(after_acceptance) == 1, after_acceptance
    assert "reason=no_progress_watchdog" in after_acceptance[0]
    assert f"session={SESSION}" in after_acceptance[0]
    assert _after_acceptance_by_reason() == {"no_progress_watchdog": 1}
    assert (LOCAL_ENDPOINT, SESSION) not in srv.local_dispatch_records
