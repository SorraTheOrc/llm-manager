"""Per-release-path ``local_active_queries`` regression matrix.

Red-first matrix for parent LP-0MVWF78QK0060TZH **AC3**: for every path the
AC2 audit enumerated as ending a local request, a driven release must return
``local_active_queries`` to its pre-request value **within the request's own
lifecycle** — never merely "eventually", via the self-healing clamp
(``_recover_stuck_local_active_queries``).

Why a matrix and not one test
-----------------------------

The counter has exactly two increment sites and one decrement owner
(``_decrement_local_active_queries``, called only from
``router._cleanup_after_request``), so "exactly once" is a property of the
*pair* (one increment per acquisition × one cleanup per request). A single
end-to-end test cannot show that: a fix that trades one leak for a double
decrement would still pass. Each AC2-audit row therefore gets its own driven
release, so the matrix both pins the defects and guards the paths that are
already correct against a double decrement.

Rows are named by their AC2-audit table number (comment
``LP-C0MW31NHW8004PADI`` on the parent) so a verdict can be traced straight
back to the audit.

Red-first encoding
------------------

``xfail(strict=True)`` — the repository's red-first convention (see
``test_abandoned_local_dispatch.py`` and
``test_local_active_queries_invariant.py``) — marks the rows the audit
verdicts **missing**/leaking. This keeps the test-only item landable with a
green suite while the assertion stays pinned, and ``strict`` means the marker
cannot be forgotten: the implementation item removes it as it turns the row
green. Rows the audit verdicts *exactly once* (or *no decrement by design*)
run unmarked and must pass both before and after the fix.

Acceptance-criterion map
------------------------

- **AC1 — per-path regression matrix.**
  ``test_release_path_returns_counter_to_pre_request_value`` is parameterised
  over every release path in the AC2 table: normal completion (buffered and
  streaming share ``_cleanup_after_request``), pre-first-byte
  ``client_disconnect``, post-first-byte ``disconnect``, measured-cache-deficit
  abort, prefill deadline, no-progress watchdog, orphan cleanup,
  tenancy-lease/idle-timeout reclaim, slot-assignment free, eviction after
  acceptance and the generic exception path — plus the audit's defective rows
  (streaming single-flight duplicate/rejected, reaper cancellation with no
  disconnect verdict, cross-endpoint completion release, same-session
  re-acquisition, session eviction).
- **AC2 — the deficit-abort and eviction rows are present explicitly.**
  ``row-6-measured-cache-deficit-abort`` and
  ``row-8-eviction-after-acceptance`` (38 % / 36 % of the ante-nascent
  population) each assert the counter is not left elevated.
- **AC3 — RED on the current tree for the leaking rows.**
  ``row-4``, ``row-13``, ``row-16``, ``row-17``, ``row-19`` and ``row-20``
  carry the ``xfail(strict=True)`` marker; the non-leaking rows pass, which is
  what makes the matrix a check on the AC2 table's verdicts rather than a
  restatement of them.
- **AC4 — hermetic.** No network, no llama-server: the router-level rows reuse
  the in-process fake-upstream harness of
  ``test_local_active_queries_invariant.py`` and the helper-level rows drive
  the real release helpers against the real ``proxy.server`` state, with
  ``_query_slot_processing`` stubbed (no ``/slots`` probe). Interleaving is
  barrier-driven, not timed: the single-flight queue step waits on the
  coordinator's waiter counter under a bounded deadline rather than sleeping
  for a fixed interval.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import proxy.router as router
import proxy.router_helpers as helpers
import proxy.server as server
import proxy.session as session_mod
import pytest

# The router-level harness (fake upstreams, request stand-in, server install and
# the monkeypatch set that keeps a streaming local dispatch in-process) is
# shared with the AC4 contention-invariant test rather than copied: a second,
# drifting copy of that 400-line harness is exactly the duplication the test
# guidelines forbid. Both modules are collected from ``proxy/tests``, so this
# import resolves to the same module object pytest uses.
from test_local_active_queries_invariant import (
    _DONE,
    _FINISH_CHUNK,
    _BlockingUpstream,
    _drive,
    _install_server,
    _PacedUpstream,
    _patch_router,
    _Request,
    _wait,
    _warning_lines,
)

from proxy import prefill_eta

#: The implementation child that turns the pinned rows green.
IMPLEMENTATION_ITEM = "LP-0MVYWLBV70018ZZB"

MODEL = "Qwen3"

#: Two distinct local endpoints: the per-endpoint record keys the
#: cross-endpoint completion row needs (LP-0MRPILSMW004T4H8).
ENDPOINT_A = "http://127.0.0.1:8091"
ENDPOINT_B = "http://127.0.0.1:8092"

#: What the counter reads before a request takes its dispatch. Every row
#: acquires at least one lease, so this is a real observation, not a constant
#: the row is compared against without evidence.
PRE_REQUEST_VALUE = 0

_RED_FIRST = pytest.mark.xfail(
    strict=True,
    reason=(
        "RED on the current tree: the AC2 audit verdicts this release path as "
        "leaking the increment (or as leaving it without a decrement owner). "
        "Green once "
        f"{IMPLEMENTATION_ITEM} decrements exactly once on every release path."
    ),
)


# ---------------------------------------------------------------------------
# Context and shared observations
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """One row's freshly installed server state."""

    srv: Any
    session: str
    endpoint: str | None
    monkeypatch: Any
    plans: dict = field(default_factory=dict)
    sent: list = field(default_factory=list)


async def _context(monkeypatch, session: str, endpoint: str | None = None) -> _Ctx:
    """Install the real server state plus the in-process router harness."""
    helpers._reset_prefill_thrash_cooldowns()
    _install_server(monkeypatch)
    # The single-flight coordinator is shared, module-level state: a leaked
    # lock from a previous row would silently serialise this one.
    session_mod.session_single_flight_coordinator._states.clear()
    ctx = _Ctx(srv=server, session=session, endpoint=endpoint, monkeypatch=monkeypatch)
    _patch_router(monkeypatch, ctx.plans, ctx.sent)
    # Hermetic: never probe a real llama-server for slot liveness. Phase 2 of
    # the sweep treats an unverifiable probe as "not processing", which is the
    # dead-slot shape these rows describe.
    monkeypatch.setattr(
        helpers, "_query_slot_processing", AsyncMock(return_value=False)
    )
    assert await _read_counter(ctx.srv) == PRE_REQUEST_VALUE, (
        "harness bug: the counter must start at its pre-request value"
    )
    return ctx


def _set_server_config(monkeypatch, **overrides: Any) -> None:
    """Override ``server:`` config keys for one row."""
    config = json.loads(json.dumps(server.config))
    config.setdefault("server", {}).update(overrides)
    monkeypatch.setattr(server, "config", config)


async def _read_counter(srv) -> int:
    """Read ``local_active_queries`` under its own lock."""
    async with srv.local_active_queries_lock:
        return int(srv.local_active_queries)


def _active_records(srv) -> int:
    return int(helpers._count_active_local_dispatch_records(srv) or 0)


def _only_record(srv) -> dict:
    records = getattr(srv, "local_dispatch_records", {}) or {}
    assert len(records) == 1, (
        f"expected exactly one dispatch record, got {list(records)}"
    )
    return next(iter(records.values()))


def _status(result: Any) -> int | None:
    return getattr(result, "status_code", None)


async def _drain(coro: Awaitable) -> tuple[Any, bytes]:
    """Await a dispatch, tolerating a raised exception.

    The generic-exception row is about the cleanup that follows, not about the
    error surface, so the raised value is returned rather than propagated.
    """
    try:
        result, body = await coro
        return result, body
    except BaseException as exc:  # row 15 asserts cleanup only
        return exc, b""


async def _acquire(
    ctx: _Ctx,
    *,
    endpoint: str | None = None,
    max_local: int = 4,
    session: str | None = None,
) -> int:
    """Take a real local dispatch lease through the production choke point."""
    acquired, owner, active_count, _retry_after = (
        await helpers._try_acquire_local_dispatch(
            ctx.srv,
            max_local,
            session if session is not None else ctx.session,
            endpoint if endpoint is not None else ctx.endpoint,
            model_name=MODEL,
        )
    )
    assert acquired is True, (
        "the harness must acquire a real local dispatch lease, not be denied "
        f"(owner={owner})"
    )
    return int(active_count)


async def _assert_restored(ctx: _Ctx) -> None:
    """AC1: the counter is back at its pre-request value, without the clamp.

    "Within the request's lifecycle" is what is asserted: the counter must
    already be correct *before* the self-healing clamp is given a chance to
    repair it, so the clamp is run afterwards and must find nothing to repair.
    """
    counter = await _read_counter(ctx.srv)
    active = _active_records(ctx.srv)
    assert counter == PRE_REQUEST_VALUE, (
        f"local_active_queries did not return to its pre-request value: "
        f"counter={counter}, expected {PRE_REQUEST_VALUE}, "
        f"active dispatch records={active}, "
        f"records={list((ctx.srv.local_dispatch_records or {}).keys())}"
    )
    assert active == 0, (
        f"no active dispatch record may outlive the request: {active} left in "
        f"{list((ctx.srv.local_dispatch_records or {}).keys())}"
    )

    # AC1: "not merely eventually-via-clamp" — the real clamp must find
    # nothing to repair at this point.
    before = len(_warning_lines(ctx.srv.logger))
    await helpers._recover_stuck_local_active_queries(ctx.srv)
    recovered = [
        line
        for line in _warning_lines(ctx.srv.logger)[before:]
        if "counter recovered" in line
    ]
    assert not recovered, (
        "the self-healing clamp repaired the counter, so the release leaked an "
        f"increment instead of decrementing exactly once: {recovered}"
    )


# ---------------------------------------------------------------------------
# Fake upstreams used only by this module
# ---------------------------------------------------------------------------


class _CM:
    """Async context manager returning the wrapped fake response."""

    def __init__(self, response) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *_exc):
        return False


class _RaisingUpstream:
    """A fake SSE upstream that raises from ``aiter_bytes``.

    Models the generic exception path (AC2 row 15): the dispatch has been
    accepted when the upstream body blows up, so the request must still
    release its increment.
    """

    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.entered = asyncio.Event()

    def cm(self) -> _CM:
        return _CM(self)

    async def aiter_bytes(self):
        self.entered.set()
        raise self._exc

    async def aread(self) -> bytes:
        return b""


# ---------------------------------------------------------------------------
# Rows — each drives exactly one release path from the AC2 audit table
# ---------------------------------------------------------------------------


async def _row_normal_completion(ctx: _Ctx) -> None:
    """Row 1/2 — normal completion (buffered and streaming share this owner)."""
    await _acquire(ctx)
    assert await _read_counter(ctx.srv) == 1
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )


async def _row_client_disconnect_pre_first_byte(ctx: _Ctx) -> None:
    """Row 3 — client vanished while prefilling (``client_disconnect``)."""
    await _acquire(ctx)
    ctx.srv.local_prefill_in_flight[ctx.session] = True
    released = helpers._release_client_disconnect_dispatch(
        ctx.srv, ctx.session, endpoint=ctx.endpoint
    )
    assert released is True, (
        "the pre-first-byte disconnect release must pop the dispatch record"
    )
    assert ctx.session not in ctx.srv.local_prefill_in_flight, (
        "the disconnect release must drop the session's prefill marker"
    )
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        disconnected=True,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )


async def _row_client_disconnect_post_first_byte(ctx: _Ctx) -> None:
    """Row 5 — stream-end ``disconnect`` (client left after the first byte)."""
    await _acquire(ctx)
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        disconnected=True,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )


async def _row_measured_cache_deficit_abort(ctx: _Ctx) -> None:
    """Row 6 — measured-cache-deficit abort (AC2: 38 % of the population)."""
    await _acquire(ctx)
    ctx.srv.local_prefill_in_flight[ctx.session] = True
    released = await helpers._release_measured_cache_abort_dispatch(
        ctx.srv, ctx.session, endpoint=ctx.endpoint
    )
    assert released is True, "the measured-deficit abort must release the lease"
    # The release core removes the record but never the counter: the owning
    # request's cleanup is the single decrement owner.
    assert await _read_counter(ctx.srv) == 1, (
        "the deficit abort must not decrement the counter itself"
    )
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )


async def _row_prefill_deadline(ctx: _Ctx) -> None:
    """Row 7 — prefill-deadline release (the twin of row 6)."""
    await _acquire(ctx)
    released = await helpers._release_prefill_deadline_dispatch(
        ctx.srv, ctx.session, endpoint=ctx.endpoint
    )
    assert released is True, "the prefill deadline must release the lease"
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )


async def _row_no_progress_watchdog(ctx: _Ctx) -> None:
    """Row 9 — the no-progress watchdog releases a wedged active lease."""
    await _acquire(ctx)
    record = _only_record(ctx.srv)
    record["last_progress_ts"] = time.monotonic() - 10_000
    removed = await helpers._cleanup_stale_local_dispatch(ctx.srv)
    assert removed == 1, (
        "the watchdog must release the wedged record (it is the only record)"
    )


async def _row_orphan_cleanup(ctx: _Ctx) -> None:
    """Row 11 — expired *active* record whose slot is gone (orphan cleanup)."""
    # The watchdog must stay out of the way for this row (it runs first in the
    # same sweep); disabling it also disables phase 3's recent-progress
    # extension, which is what makes the dead-slot verdict reachable.
    _set_server_config(ctx.monkeypatch, local_dispatch_no_progress_timeout_seconds=0)
    await _acquire(ctx)
    record = _only_record(ctx.srv)
    now = time.monotonic()
    record["expires_at"] = now - 1
    record["last_progress_ts"] = now
    record["started_at"] = now
    removed = await helpers._cleanup_stale_local_dispatch(ctx.srv)
    assert removed == 1, "the orphan sweep must release the expired live record"


async def _row_generic_exception(ctx: _Ctx) -> None:
    """Row 15 — the request raised; its cleanup still owns the decrement."""
    ctx.plans["stream-boom"] = _RaisingUpstream(RuntimeError("upstream exploded"))
    result, _body = await _drain(_drive(_Request(ctx.session, "stream-boom")))
    assert result is not None, "the dispatch must have been attempted"


async def _row_eviction_after_acceptance(ctx: _Ctx) -> None:
    """Row 8 — the post-acceptance eviction emitters (AC2: 36 % of the pop.).

    Verdict: *n/a — never decrements, by design*. The emitters are
    observability only, so the row asserts two things: the emitters do not
    touch the counter, and the owning request's cleanup still restores it.
    """
    await _acquire(ctx)
    record = dict(_only_record(ctx.srv))
    released = helpers._record_local_eviction(
        ctx.srv, ctx.session, endpoint=ctx.endpoint, reason="lease_evicted"
    )
    helpers._emit_eviction_after_acceptance(
        ctx.srv, ctx.session, "measured_cache_deficit", record,
        endpoint=ctx.endpoint,
    )
    assert isinstance(released, bool)
    assert await _read_counter(ctx.srv) == 1, (
        "the eviction emitters must never decrement the counter themselves — "
        "the owning request's cleanup is the single decrement owner"
    )
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )


async def _row_tenancy_lease_idle_timeout(ctx: _Ctx) -> None:
    """Row 10 — tenancy-lease / idle-timeout reclaim of an inactive record.

    Verdict: *no decrement — correct*: the counter was already decremented when
    the record was marked inactive. The row therefore asserts the sweep does
    not decrement a second time (which would under-report and release another
    session's capacity).
    """
    await _acquire(ctx)
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )
    record = _only_record(ctx.srv)
    assert record["active"] is False, (
        "an explicit session keeps its lease inactive for the cooldown window"
    )
    record["expires_at"] = time.monotonic() - 1
    removed = await helpers._cleanup_stale_local_dispatch(ctx.srv)
    assert removed == 1, "the idle-timeout sweep must reclaim the inactive record"
    assert await _read_counter(ctx.srv) == PRE_REQUEST_VALUE, (
        "the idle-timeout sweep must not decrement a counter the completion "
        "path already decremented"
    )


async def _row_terminal_record_reap(ctx: _Ctx) -> None:
    """Row 12 — a completed record that somehow survived is reaped, un-counted."""
    await _acquire(ctx)
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )
    record = _only_record(ctx.srv)
    now = time.monotonic()
    record["terminal"] = True
    record["completed_at"] = now
    record["expires_at"] = now + 600
    removed = await helpers._cleanup_stale_local_dispatch(ctx.srv)
    assert removed == 1, "the terminal-record guard must reap the stale record"
    assert await _read_counter(ctx.srv) == PRE_REQUEST_VALUE, (
        "the terminal-record reap must not decrement (the completion path owns it)"
    )


async def _row_slot_assignment_free(ctx: _Ctx) -> None:
    """Row 14 — freeing the slot registry entry is not a counter path."""
    await _acquire(ctx)
    session_mod._free_slot_assignment(ctx.session, endpoint=ctx.endpoint)
    assert await _read_counter(ctx.srv) == 1, (
        "freeing a slot assignment must not move the counter"
    )
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ctx.endpoint,
    )


async def _row_streaming_duplicate(ctx: _Ctx) -> None:
    """Row 16 — streaming single-flight duplicate (``decrement_local=False``)."""
    first = _PacedUpstream()
    ctx.plans["stream-a"] = first
    original = asyncio.create_task(_drive(_Request(ctx.session, "stream-a")))
    await _wait(first.parked, "original stream parked")

    duplicate, _body = await _drain(_drive(_Request(ctx.session, "stream-a")))
    assert _status(duplicate) == 409, (
        "an identical in-flight prompt must take the duplicate detour (409), "
        f"got {_status(duplicate)}"
    )
    assert await _read_counter(ctx.srv) == 2, (
        "the duplicate re-acquired the lease before the single-flight guard "
        "ran, so the leak is visible as an extra increment"
    )

    first.push(_FINISH_CHUNK)
    first.push(_DONE)
    first.end()
    result, body = await asyncio.wait_for(original, timeout=10)
    assert _status(result) == 200, "the original stream must still complete"
    assert _DONE in body, "the original stream must reach its terminator"


async def _row_streaming_rejected(ctx: _Ctx) -> None:
    """Row 17 — streaming single-flight rejection (``queue_full``)."""
    first, second, third = _PacedUpstream(), _PacedUpstream(), _PacedUpstream()
    ctx.plans.update(
        {"stream-a": first, "stream-b": second, "stream-c": third}
    )
    task_first = asyncio.create_task(_drive(_Request(ctx.session, "stream-a")))
    await _wait(first.parked, "first stream parked")

    task_second = asyncio.create_task(_drive(_Request(ctx.session, "stream-b")))
    await _wait_for_single_flight_waiter(ctx.session)

    rejected, _body = await _drain(_drive(_Request(ctx.session, "stream-c")))
    assert _status(rejected) in (429, 503), (
        "a third concurrent stream on one session must be rejected by the "
        f"single-flight queue, got {_status(rejected)}"
    )
    assert await _read_counter(ctx.srv) == 3, (
        "all three streams took their increment before the guard ran, so the "
        "rejection's `decrement_local=False` detour leaks one"
    )

    # Let both admitted streams finish so the row settles.
    first.push(_FINISH_CHUNK)
    first.push(_DONE)
    first.end()
    second.push(_FINISH_CHUNK)
    second.push(_DONE)
    second.end()
    await asyncio.wait_for(task_first, timeout=10)
    await asyncio.wait_for(task_second, timeout=10)


async def _wait_for_single_flight_waiter(session_id: str, timeout: float = 10.0) -> None:
    """Barrier: wait until another request is queued behind *session_id*.

    Reads the coordinator's waiter counter rather than sleeping: the public
    ``queue_depth_current`` gauge is only recomputed when a guard is acquired
    or released, so it stays at 0 for the whole time a request is waiting —
    the counter itself is the state that has actually changed.
    """
    coordinator = session_mod.session_single_flight_coordinator
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = coordinator._states.get(session_id) or {}
        if int(state.get("waiters", 0)) >= 1:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(
        "the second streaming request never queued behind the first "
        f"(states={coordinator._states!r})"
    )


async def _row_cancelled_without_disconnect_verdict(ctx: _Ctx) -> None:
    """Row 4 — reaper cancellation with no client-disconnect verdict."""
    hanging = _BlockingUpstream()
    ctx.plans["stream-hang"] = hanging
    task = asyncio.create_task(_drive(_Request(ctx.session, "stream-hang")))
    await _wait(hanging.entered, "prefill entered")

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # The request never recorded a disconnect verdict, so the ``CancelledError``
    # branch released nothing: the record and the increment are both still here.
    assert await _read_counter(ctx.srv) == 1, (
        "harness bug: the cancelled dispatch should still hold its increment"
    )


async def _row_cross_endpoint_completion_release(ctx: _Ctx) -> None:
    """Row 19 — ``_release_completed_dispatch`` frees the other endpoint's lease."""
    await _acquire(ctx, endpoint=ENDPOINT_A)
    await _acquire(ctx, endpoint=ENDPOINT_B)
    assert await _read_counter(ctx.srv) == 2, (
        "one lease per local endpoint means two increments for one session"
    )
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ENDPOINT_B,
    )


async def _row_same_session_reacquisition(ctx: _Ctx) -> None:
    """Row 20 — a second acquisition for an already-counted lease."""
    await _acquire(ctx, endpoint=ENDPOINT_A)
    await _acquire(ctx, endpoint=ENDPOINT_A)
    assert await _read_counter(ctx.srv) == 2, (
        "a same-session re-acquisition increments again for a lease that is "
        "already counted"
    )
    assert len(ctx.srv.local_dispatch_records) == 1, (
        "the re-acquisition overwrites the single per-session record"
    )
    await router._cleanup_after_request(
        ctx.srv,
        ctx.session,
        decrement_local=True,
        session_explicit=True,
        model_name=MODEL,
        backend=ENDPOINT_A,
    )


async def _row_session_eviction(ctx: _Ctx) -> None:
    """Row 13 — session eviction releases the record without its increment."""
    await _acquire(ctx)
    await server._release_lease_on_session_eviction(ctx.session)
    assert not ctx.srv.local_dispatch_records, (
        "the eviction callback must release every record the session owns"
    )
    assert await _read_counter(ctx.srv) == 1, (
        "the record is gone but the increment stays: the AC2 audit's row 13 "
        "has no decrement owner"
    )


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One AC2-audit row: an id, the path's name, its driver and its verdict."""

    id: str
    summary: str
    drive: Callable[[_Ctx], Awaitable[None]]
    red_first: bool = False


MATRIX: tuple[_Row, ...] = (
    _Row(
        "row-1-2-normal-completion",
        "normal completion (buffered and streaming share _cleanup_after_request)",
        _row_normal_completion,
    ),
    _Row(
        "row-3-client-disconnect-pre-first-byte",
        "client cancel during prefill (reason=client_disconnect)",
        _row_client_disconnect_pre_first_byte,
    ),
    _Row(
        "row-5-client-disconnect-post-first-byte",
        "client disconnect after the first byte (reason=disconnect)",
        _row_client_disconnect_post_first_byte,
    ),
    _Row(
        "row-6-measured-cache-deficit-abort",
        "measured-cache-deficit abort (AC2 dominant population)",
        _row_measured_cache_deficit_abort,
    ),
    _Row(
        "row-7-prefill-deadline",
        "prefill-deadline release",
        _row_prefill_deadline,
    ),
    _Row(
        "row-8-eviction-after-acceptance",
        "post-acceptance eviction emitters (AC2 dominant population)",
        _row_eviction_after_acceptance,
    ),
    _Row(
        "row-9-no-progress-watchdog",
        "no-progress watchdog",
        _row_no_progress_watchdog,
    ),
    _Row(
        "row-10-tenancy-lease-idle-timeout",
        "tenancy-lease / idle-timeout reclaim",
        _row_tenancy_lease_idle_timeout,
    ),
    _Row(
        "row-11-orphan-cleanup",
        "orphan cleanup of an expired live record",
        _row_orphan_cleanup,
    ),
    _Row(
        "row-12-terminal-record-reap",
        "terminal-record reap",
        _row_terminal_record_reap,
    ),
    _Row(
        "row-14-slot-assignment-free",
        "slot-assignment free",
        _row_slot_assignment_free,
    ),
    _Row(
        "row-15-generic-exception",
        "generic exception path",
        _row_generic_exception,
    ),
    _Row(
        "row-4-cancelled-without-disconnect-verdict",
        "reaper cancellation with no client-disconnect verdict",
        _row_cancelled_without_disconnect_verdict,
        red_first=True,
    ),
    _Row(
        "row-13-session-eviction",
        "session eviction (record released, increment left behind)",
        _row_session_eviction,
        red_first=True,
    ),
    _Row(
        "row-16-streaming-single-flight-duplicate",
        "streaming single-flight duplicate (decrement_local=False)",
        _row_streaming_duplicate,
        red_first=True,
    ),
    _Row(
        "row-17-streaming-single-flight-rejected",
        "streaming single-flight rejection (queue_full)",
        _row_streaming_rejected,
        red_first=True,
    ),
    _Row(
        "row-19-cross-endpoint-completion-release",
        "cross-endpoint completion release",
        _row_cross_endpoint_completion_release,
        red_first=True,
    ),
    _Row(
        "row-20-same-session-reacquisition",
        "same-session re-acquisition of an already-counted lease",
        _row_same_session_reacquisition,
        red_first=True,
    ),
)


@pytest.fixture(autouse=True)
def _deterministic_prefill_eta():
    """Start every row from the uncalibrated shipped prefill-ETA fit.

    The measured-deficit row prices through the calibrated model, and every
    served dispatch records its own first-byte sample; a prior row could
    otherwise poison the ``(Qwen3, large-prompt)`` bucket.
    """
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _params() -> list:
    """Render the matrix as pytest parameters, keeping each row's verdict."""
    return [
        pytest.param(
            row,
            id=row.id,
            marks=(_RED_FIRST,) if row.red_first else (),
        )
        for row in MATRIX
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _params())
async def test_release_path_returns_counter_to_pre_request_value(row, monkeypatch):
    """AC1/AC2/AC3: every release path decrements exactly once."""
    ctx = await _context(monkeypatch, f"sess-{row.id}")
    await row.drive(ctx)
    await _assert_restored(ctx)


@pytest.mark.asyncio
async def test_every_ac2_release_path_has_a_matrix_row():
    """The matrix must cover each path the AC1 enum names, by name.

    Guards the matrix itself: deleting a row silently would otherwise shrink
    coverage without failing anything.
    """
    covered = {row.id for row in MATRIX}
    required = {
        "row-1-2-normal-completion",
        "row-3-client-disconnect-pre-first-byte",
        "row-5-client-disconnect-post-first-byte",
        "row-6-measured-cache-deficit-abort",
        "row-8-eviction-after-acceptance",
        "row-9-no-progress-watchdog",
        "row-10-tenancy-lease-idle-timeout",
        "row-14-slot-assignment-free",
        "row-15-generic-exception",
    }
    assert required <= covered, (
        "the matrix no longer covers every release path AC1 enumerates: "
        f"missing {sorted(required - covered)}"
    )
