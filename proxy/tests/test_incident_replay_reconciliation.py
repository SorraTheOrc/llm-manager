"""Incident replay and reconciliation rows (LP-0MWL5KKEH004XA0Y).

Parent: **LP-0MWKWGLRT003AUG4** — *Orphaned ``local_prefill_in_flight`` marker
closes the local route for every session*, AC4 + AC6.

The incident this module replays is the 2026-11-17 wedge. One session
(``01a1e9e4-90be-70d7-a899-aa39b5c753ca``) held the only local dispatch: four
``lease_renewed`` lines (14:23:54 … 14:24:10) and **no** ``lease_released``.
Other sessions were denied while that lease was merely inactive
(``records_total=1 records_active=0``), and at **14:24:42** the stale-record
prune removed it — ``records_total=0`` — but its ``local_prefill_in_flight``
marker survived. From that moment every local dispatch was denied::

    local_dispatch_denied ... owner=none records_total=0 records_active=0
      prefill_in_flight=1 prefill_cap=1 llama_available_slots=2

with the GPU idle and llama-server reporting two free slots. The route stayed
closed until a hand-issued ``POST /v1/leases/release`` at 16:22:53 (the only
recovery available: no restart, no bounded sweep). Local serving resumed
immediately afterwards (``local_preference_resolved ... outcome=local`` →
``Stream started: provider=local``).

Two rows, one per parent AC:

* **AC4 incident replay** (:func:`test_incident_replay_*`) — seed the incident
  shape (session A's record inactive, past ``expires_at``, marker still
  present), then drive the **real admission gate**
  (:func:`proxy.router_helpers._try_acquire_local_dispatch`) for session B with
  ``X-Local-Preference: local-preferred``. B must be admitted locally, so no
  ``429 "Local admission refused"`` (``local_admission_reject``) can be built.
  Runs unmarked: the gate-pricing impl (LP-0MWL5KL0J006ZVDG) and the
  deletion-path pop impl (LP-0MWL5KLN1003E990) have both landed, and a
  regression on either turns this row red.

* **AC6 one-tick reconciliation** (:func:`test_orphaned_marker_*`) — an
  already-orphaned marker (no record at all, the 14:24:42 shape) must be
  dropped by the reconciliation pass the cleanup loop calls, within a single
  tick, logged exactly once with the session id, and the following tick must
  reclaim nothing. Runs unmarked: the sweep landed with
  ``[impl] Bounded orphan-marker reconciliation sweep``
  (LP-0MWL5KM8X00193B3), and a regression turns the row red.

Every row is hermetic: a throwaway ``SimpleNamespace`` server, fresh
``asyncio.Lock``s, a frozen monotonic clock (no sleeps), the real
``_dispatch_cleanup_loop`` driven for exactly one iteration via a sentinel, and
no network, lease-release endpoint or proxy restart. Recovery is asserted from
in-process state only (AC3).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import proxy.provider as provider
import proxy.router_helpers as helpers
import proxy.server as server_module
import pytest
from fastapi import Response

from proxy import contention_queue, prefill_eta

# ---------------------------------------------------------------------------
# Incident constants (evidence, 2026-11-17)
# ---------------------------------------------------------------------------

#: The session whose lease died without a clean release.
INCIDENT_LEAKED_SESSION = "01a1e9e4-90be-70d7-a899-aa39b5c753ca"
#: The different, locally-pinned session the incident starved.
INCIDENT_NEWCOMER_SESSION = "herdr-1794932459-3635932-21305"

#: The pin-local wire value the starved sessions carried.
LOCAL_PREFERRED = "local-preferred"

#: Frozen ``time.monotonic()`` value every row is built against.
NOW = 10_000.0

SERVER_CONFIG = {
    "server": {
        "session_slot_pool_size": 2,
        # Production value during the incident: one prefill at a time.
        "local_max_concurrent_prefills": 1,
        "local_dispatch_lease_timeout_seconds": 60,
        "local_dispatch_lease_per_token_seconds": 0.0,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 0.05,
        "local_dispatch_lease_prefill_buffer_seconds": 0.5,
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_max_prefill_seconds": 900,
        "local_prefill_deadline_seconds": 0,
        "session_single_flight_mode": "bypass",
    }
}


class _FrozenClock:
    """``time`` shim whose ``monotonic`` is pinned to :data:`NOW`.

    Only ``monotonic`` is injected; every other ``time`` attribute falls
    through to the real module, so the production code under test sees a
    stable clock without patching the global ``time`` module (no sleeps).
    """

    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str):
        return getattr(time, name)


def _stale_record(now: float = NOW) -> dict:
    """The incident's pruned lease: inactive and past ``expires_at``.

    This is exactly the record the 14:24:42 prune removed — no ``active``
    flag transition, a future-ish ``expires_at`` that has since elapsed.
    """
    return {
        "backend": "local",
        "started_at": now - 120.0,
        "active": False,
        "expires_at": now - 5.0,
        "model_name": "Qwen3",
        "last_progress_ts": now - 120.0,
    }


def _live_record(now: float = NOW) -> dict:
    """A genuinely live (active, unexpired, progressing) dispatch record."""
    return {
        "backend": "local",
        "started_at": now,
        "active": True,
        "expires_at": now + 600.0,
        "model_name": "Qwen3",
        "last_progress_ts": now,
    }


def _make_srv(records: dict, markers: dict) -> SimpleNamespace:
    """Throwaway server with fresh locks (no shared proxy state)."""
    return SimpleNamespace(
        config=copy.deepcopy(SERVER_CONFIG),
        local_active_queries=len(records),
        local_active_queries_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions=set(),
        local_dispatch_records=dict(records),
        local_dispatch_records_lock=asyncio.Lock(),
        local_prefill_in_flight=dict(markers),
        local_prefill_in_flight_lock=asyncio.Lock(),
        logger=MagicMock(),
    )


class _DummyRequest:
    """Minimal request stub (mirrors the contention-queue test helpers)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers=None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = SimpleNamespace(path="/v1/chat/completions")

    async def body(self) -> bytes:
        return self._body


def _pinned_request() -> _DummyRequest:
    """A request carrying the incident's ``X-Local-Preference`` header."""
    return _DummyRequest(headers={"x-local-preference": LOCAL_PREFERRED})


# ===========================================================================
# AC4 — incident replay: prune the stale record, then admit a *different*
# locally-pinned session (parent AC4)
# ===========================================================================


@pytest.mark.asyncio
async def test_incident_replay_pruned_stale_record_then_local_preferred_session_admitted(
    monkeypatch,
):
    """LP-0MWKWGLRT003AUG4 AC4: the incident shape no longer closes the route.

    Seed exactly the post-prune state of 2026-11-17 14:24:42: session A holds
    a record that is ``active=False`` and past ``expires_at``, and its
    ``local_prefill_in_flight`` marker is still present. Driving the real
    admission gate for session B (``local-preferred``) must:

    * admit B (``acquired=True``) — pre-fix the raw marker count priced the
      prefill cap and denied every dispatch permanently;
    * prune A's stale record **and pop its marker** — the deletion-path pop
      the incident's ``_prune_stale_dispatch_records`` lacked;
    * leave B holding a real dispatch record.

    Because the gate admits, the caller never builds the pinned refusal, so no
    ``local_admission_reject`` attempt / ``429 "Local admission refused"`` is
    produced (AC5).
    """
    monkeypatch.setattr(helpers, "time", _FrozenClock())
    srv = _make_srv(
        records={INCIDENT_LEAKED_SESSION: _stale_record()},
        markers={INCIDENT_LEAKED_SESSION: True},
    )
    request = _pinned_request()

    acquired, owner, active, retry_after = await helpers._try_acquire_local_dispatch(
        srv,
        max_local=2,
        session_key=INCIDENT_NEWCOMER_SESSION,
        backend="local",
        request=request,
    )

    assert acquired is True, (
        "LP-0MWKWGLRT003AUG4 AC4: a different locally-pinned session must be "
        "admitted after the stale record is pruned; "
        f"acquired={acquired} owner={owner!r} active={active} "
        f"retry_after={retry_after} records={srv.local_dispatch_records!r} "
        f"markers={srv.local_prefill_in_flight!r}"
    )
    # The prune removed the stale record and popped its marker (the incident's
    # 14:24:42 state is now fully reconciled).
    assert INCIDENT_LEAKED_SESSION not in srv.local_dispatch_records, (
        "the stale record must be pruned; "
        f"records={srv.local_dispatch_records!r}"
    )
    assert INCIDENT_LEAKED_SESSION not in srv.local_prefill_in_flight, (
        "the pruned record's prefill marker must be popped; "
        f"markers={srv.local_prefill_in_flight!r}"
    )
    # The newcomer holds a real dispatch record (it was served locally).
    assert INCIDENT_NEWCOMER_SESSION in srv.local_dispatch_records, (
        "the admitted session must hold a dispatch record; "
        f"records={srv.local_dispatch_records!r}"
    )

    # The request is genuinely pinned to local, so a refusal *would* be the
    # hard 429 rather than a remote fallback (LP-0MVSHWL47000EVJY). Admission
    # is therefore the proof that no `local_admission_reject` is produced; the
    # provider-level row below pins the same fact on the wire.
    assert provider._local_preference_pins_local(request) is True, (
        "a local-preferred request must stay pinned to the local route"
    )


@pytest.mark.asyncio
async def test_incident_replay_live_prefill_still_denies_the_newcomer(monkeypatch):
    """AC4 conservative guard: a *live* prefill still serialises the pool.

    The fix discounts orphaned markers only. The same shape with an ``active``
    record must still deny the newcomer, otherwise the liveness filter would
    have broken ``local_max_concurrent_prefills`` serialisation.
    """
    monkeypatch.setattr(helpers, "time", _FrozenClock())
    srv = _make_srv(
        records={INCIDENT_LEAKED_SESSION: _live_record()},
        markers={INCIDENT_LEAKED_SESSION: True},
    )

    acquired, owner, _active, _retry = await helpers._try_acquire_local_dispatch(
        srv,
        max_local=2,
        session_key=INCIDENT_NEWCOMER_SESSION,
        backend="local",
        request=_pinned_request(),
    )

    assert acquired is False, (
        "a live-backed prefill marker must still count against the cap: "
        f"acquired={acquired} owner={owner!r}"
    )
    assert owner == INCIDENT_LEAKED_SESSION, (
        f"the refusal must name the live prefill owner, got {owner!r}"
    )
    assert INCIDENT_LEAKED_SESSION in srv.local_prefill_in_flight, (
        "a refusal must not pop the live prefill's marker"
    )


@pytest.mark.asyncio
async def test_incident_replay_local_preferred_request_is_never_answered_with_429(
    monkeypatch,
):
    """AC4 (parent): the pinned request is served locally, never refused 429.

    This drives the *provider* cycle with a fake local dispatcher that calls
    the **real** admission gate, so the incident's decision chain is exercised
    end to end without a real llama-server: the gate admits (post-fix) and the
    request completes locally with ``200``. Pre-fix the orphaned marker made
    the gate deny, the local entry failed, and the pinned request ended on the
    ``429 "Local admission refused"`` refusal.
    """
    monkeypatch.setattr(helpers, "time", _FrozenClock())
    srv = _make_srv(
        records={INCIDENT_LEAKED_SESSION: _stale_record()},
        markers={INCIDENT_LEAKED_SESSION: True},
    )

    calls: list[str] = []

    async def _fake_local(request, path, endpoint=None):
        acquired, *_ = await helpers._try_acquire_local_dispatch(
            srv,
            max_local=2,
            session_key=INCIDENT_NEWCOMER_SESSION,
            backend="local",
            request=request,
        )
        calls.append("local" if acquired else "denied")
        if acquired:
            return Response(
                content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
                status_code=200,
                media_type="application/json",
            )
        return Response(status_code=503, content=b"local unavailable")

    async def _fake_remote(*_args, **_kwargs):  # pragma: no cover - must not run
        calls.append("remote")
        return Response(status_code=200, content=b"{}")

    model_config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }
    config = {
        "provider_cooldown_seconds": 60,
        "server": {
            "session_slot_pool_size": 2,
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 60,
            "contention_queue_max_depth": 4,
            "local_prefill_deadline_seconds": 0,
        },
    }

    with (
        patch("proxy.router.proxy_to_local", _fake_local),
        patch("proxy.server.proxy_to_remote", _fake_remote),
        patch("proxy.provider._get_local_concurrency_info", lambda *a, **k: (0, 2)),
        patch("proxy.provider.request_priority", lambda *a, **k: "medium"),
    ):
        contention_queue.reset()
        prefill_eta.reset()
        try:
            result = await provider.proxy_with_fallback(
                _pinned_request(),
                "v1/chat/completions",
                model_config,
                config,
            )
        finally:
            contention_queue.reset()
            prefill_eta.reset()

    assert calls == ["local"], (
        "a local-preferred request must be served by the local route and never "
        f"fall through; calls={calls}"
    )
    assert result.status_code == 200, (
        "the incident's pinned request must not be refused: "
        f"status={result.status_code}"
    )
    assert result.status_code != 429, (
        "429 'Local admission refused' must not be produced for an admitted "
        "local-preferred request"
    )


# ===========================================================================
# AC6 — one cleanup tick reclaims an already-orphaned marker (parent AC6)
# ===========================================================================


class _StopTick(BaseException):
    """Sentinel that terminates the cleanup loop after exactly one tick.

    Deliberately a ``BaseException`` that is neither ``CancelledError`` nor
    ``Exception``: the loop's own handlers must not swallow it, so it
    propagates out of ``while True`` and ends the task after one iteration.
    """


class _AsyncioShim:
    """``server_module.asyncio`` stand-in that stops the loop after one tick.

    Only ``sleep`` is replaced; every other attribute (``CancelledError`` …)
    falls through to the real module.
    """

    def __init__(self, sleep) -> None:
        self.sleep = sleep

    def __getattr__(self, name: str):
        return getattr(asyncio, name)


class _LineCollector(logging.Handler):
    """Collect the production ``llama-proxy`` log lines the loop emits."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
        except Exception:  # pragma: no cover - malformed record
            return


@contextlib.contextmanager
def _cleanup_loop_state(srv):
    """Point the real ``proxy.server`` module at *srv* for one loop tick.

    The loop reads module-level state (``proxy.server.local_dispatch_records``
    …) and logs through the module ``llama-proxy`` logger. Patching the module
    attributes at the throwaway server keeps the row hermetic while still
    driving the *real* loop body — no re-implemented simulation.

    The logger's level is lowered to ``DEBUG`` for the tick and restored
    afterwards: under pytest the ``llama-proxy`` logger inherits the root
    ``WARNING`` level, so an attached handler would never *see* the ``info``
    lines this row asserts on (handler level and logger level filter
    independently).
    """
    sleeps = {"n": 0}

    async def _sleep(_delay, *_args, **_kwargs):
        sleeps["n"] += 1
        if sleeps["n"] > 1:
            raise _StopTick()
        return None

    previous_level = server_module.logger.level
    server_module.logger.setLevel(logging.DEBUG)
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(server_module, "asyncio", _AsyncioShim(_sleep))
            )
            for attr in (
                "config",
                "local_dispatch_records",
                "local_dispatch_records_lock",
                "local_prefill_in_flight",
                "local_prefill_in_flight_lock",
                "local_active_queries",
                "local_active_queries_lock",
                "local_generating_queries",
                "local_generating_queries_lock",
                "local_generating_sessions",
            ):
                stack.enter_context(
                    patch.object(server_module, attr, getattr(srv, attr))
                )
            yield
    finally:
        server_module.logger.setLevel(previous_level)


async def _run_one_cleanup_tick(srv) -> None:
    """Run exactly one iteration of the real ``_dispatch_cleanup_loop``."""
    with _cleanup_loop_state(srv):
        task = asyncio.create_task(server_module._dispatch_cleanup_loop())
        try:
            with pytest.raises(_StopTick):
                await asyncio.wait_for(task, timeout=5.0)
        finally:
            if not task.done():  # pragma: no cover - defensive cancellation
                task.cancel()


def _reclaim_lines(lines: list[str]) -> list[str]:
    """The ``prefill_marker_reclaimed`` lines the sweep emitted."""
    return [line for line in lines if "prefill_marker_reclaimed" in line]


@pytest.mark.asyncio
async def test_orphaned_marker_reclaimed_within_one_cleanup_tick(monkeypatch):
    """LP-0MWKWGLRT003AUG4 AC6: one tick drops an already-orphaned marker.

    Seed the 14:24:42 post-prune shape — a ``local_prefill_in_flight`` marker
    with **no** dispatch record at all — and drive one real cleanup-loop tick.
    The reconciliation pass must drop the marker within that single tick and
    log exactly one ``prefill_marker_reclaimed session=<id> reason=orphaned``
    line naming the session. A second tick must reclaim nothing (idempotent):
    the marker set stays empty and no further reclaim line is emitted.

    Recovery is asserted from in-process state only — no lease-release
    endpoint, no proxy restart (AC3).
    """
    monkeypatch.setattr(helpers, "time", _FrozenClock())
    srv = _make_srv(records={}, markers={INCIDENT_LEAKED_SESSION: True})
    collector = _LineCollector()
    server_module.logger.addHandler(collector)
    try:
        await _run_one_cleanup_tick(srv)

        assert INCIDENT_LEAKED_SESSION not in srv.local_prefill_in_flight, (
            "an orphaned marker must be reclaimed within one cleanup tick; "
            f"markers={srv.local_prefill_in_flight!r} "
            f"lines={collector.lines!r}"
        )
        reclaim_lines = _reclaim_lines(collector.lines)
        assert len(reclaim_lines) == 1, (
            "exactly one prefill_marker_reclaimed line must name the orphaned "
            f"marker; lines={collector.lines!r}"
        )
        assert INCIDENT_LEAKED_SESSION in reclaim_lines[0], (
            "the reclaim line must name the reclaimed session; "
            f"line={reclaim_lines[0]!r}"
        )

        # Idempotent: the following tick reclaims nothing and logs nothing.
        before = len(collector.lines)
        await _run_one_cleanup_tick(srv)

        assert srv.local_prefill_in_flight == {}, (
            "the marker set must stay empty after the second tick; "
            f"markers={srv.local_prefill_in_flight!r}"
        )
        assert _reclaim_lines(collector.lines) == reclaim_lines, (
            "the following tick must reclaim nothing (idempotent); "
            f"lines={collector.lines!r}"
        )
        assert len(collector.lines) == before, (
            "a healthy tick must log nothing at all; "
            f"new_lines={collector.lines[before:]!r}"
        )
    finally:
        server_module.logger.removeHandler(collector)


@pytest.mark.asyncio
async def test_reconciliation_keeps_a_live_backed_marker(monkeypatch):
    """AC6 conservative guard: a marker backed by an ``active`` record survives.

    The sweep may only discount *phantom* markers. The same cleanup tick with a
    live record present must leave both the record and its marker untouched, so
    the reconciliation cannot "fix" the orphan by over-popping a running
    prefill.
    """
    monkeypatch.setattr(helpers, "time", _FrozenClock())
    srv = _make_srv(
        records={INCIDENT_LEAKED_SESSION: _live_record()},
        markers={INCIDENT_LEAKED_SESSION: True},
    )
    collector = _LineCollector()
    server_module.logger.addHandler(collector)
    try:
        await _run_one_cleanup_tick(srv)

        assert INCIDENT_LEAKED_SESSION in srv.local_dispatch_records, (
            "the live dispatch record must survive the cleanup tick; "
            f"records={srv.local_dispatch_records!r}"
        )
        assert srv.local_prefill_in_flight.get(INCIDENT_LEAKED_SESSION) is True, (
            "a live-backed prefill marker must survive the reconciliation; "
            f"markers={srv.local_prefill_in_flight!r}"
        )
        assert _reclaim_lines(collector.lines) == [], (
            "a live-backed marker must never be reclaimed; "
            f"lines={collector.lines!r}"
        )
    finally:
        server_module.logger.removeHandler(collector)


# ===========================================================================
# AC6 — the reconciliation pass itself (shape guards around the cleanup tick)
# ===========================================================================


def test_reclaim_orphans_handles_endpoint_and_legacy_dispatch_keys():
    """AC6: per-endpoint ``(endpoint, session)`` records back their markers.

    Production keys are ``(endpoint, session)`` tuples once per-endpoint
    records are in use (LP-0MRPILSMW004T4H8); legacy string keys are still
    accepted. The sweep must derive session ids through
    ``_dispatch_key_session_id`` for both, so a live dispatch is never
    misread as an orphan.
    """
    srv = _make_srv(
        records={
            ("http://gpu-a:8080", "session-a"): _live_record(),
            "session-b": _live_record(),
        },
        markers={"session-a": True, "session-b": True},
    )

    assert helpers._reclaim_orphaned_prefill_markers(srv) == []
    assert srv.local_prefill_in_flight == {
        "session-a": True,
        "session-b": True,
    }, "a marker backed by an endpoint or legacy record must survive"


def test_reclaim_orphans_drops_every_phantom_in_one_pass():
    """AC6: one pass reclaims all phantom markers and returns each session id."""
    srv = _make_srv(
        records={INCIDENT_NEWCOMER_SESSION: _live_record()},
        markers={
            INCIDENT_LEAKED_SESSION: True,
            "another-orphan": True,
            INCIDENT_NEWCOMER_SESSION: True,
        },
    )

    reclaimed = helpers._reclaim_orphaned_prefill_markers(srv)

    assert sorted(reclaimed) == sorted([INCIDENT_LEAKED_SESSION, "another-orphan"])
    assert srv.local_prefill_in_flight == {INCIDENT_NEWCOMER_SESSION: True}, (
        "only the backed marker may remain; "
        f"markers={srv.local_prefill_in_flight!r}"
    )
    assert helpers._reclaim_orphaned_prefill_markers(srv) == [], (
        "a second pass must reclaim nothing — the sweep is idempotent"
    )


def test_reclaim_orphans_keeps_a_marker_backed_by_an_inactive_record():
    """AC6 conservative guard: only *record-less* markers are phantom.

    A retained inactive lease is still a backing record; its marker is the
    per-deletion-path pops' responsibility, never the sweep's, so the
    reconciliation cannot discount a marker whose record still exists.
    """
    srv = _make_srv(
        records={INCIDENT_LEAKED_SESSION: _stale_record()},
        markers={INCIDENT_LEAKED_SESSION: True},
    )

    assert helpers._reclaim_orphaned_prefill_markers(srv) == []
    assert srv.local_prefill_in_flight.get(INCIDENT_LEAKED_SESSION) is True, (
        "a marker with a retained (inactive) record must survive; "
        f"markers={srv.local_prefill_in_flight!r}"
    )
