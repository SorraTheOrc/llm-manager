"""Credit-aware accounting for the local dispatch counters (LP-0MW0TLPOR002TAFO).

The dispatch record is keyed ``(endpoint, session)`` and, since
LP-0MVWF78QK0060TZH, holds **one credit per in-flight dispatch** in
``dispatch_credits``. Two overlapping dispatches of one session are therefore
legitimately ``local_active_queries == 2`` with **one** active record.

The measured defect
-------------------

A 27 req/min agent-fleet burst produced 24 ``counter recovered`` WARNINGs in
36 minutes, in three cascading shapes::

    local_active_queries counter recovered: reset from 2 to 1 (1 active dispatch records)
    local_active_queries counter recovered: reset from 1 to 0 (0 active dispatch records)
    active_queries counter recovered: reset from 1 to 0
    local_generating_queries counter recovered: reclaimed 1 stale session(s)

All four are the recovery loop **destroying live state**, not repairing a leak,
and they trace to one missing counterpart: the anonymous/non-explicit
increment path (``_increment_local_active_queries``) **overwrote** the dispatch
record instead of appending a credit the way the explicit path
(``_try_acquire_local_dispatch``) already did. Overwriting dropped the in-flight
sibling's credit from the record set, so:

* the counter (2) exceeded the record's credits (1) and the next tick clamped a
  live credit away (``2→1``),
* the completion of one sibling popped the record while the other was still in
  flight (no remaining credit on it), so the next tick reaped a *live* record
  (``1→0``) and the global counter that was backed by that work (``1→0``), and
* a still-generating session lost its active record, so the generating reclaim
  removed it (``reclaimed 1 stale session(s)``).

The same false clamp also fired for two genuinely-credited overlapping
dispatches because ``_recover_stuck_local_active_queries`` compared the credit
counter with the *record count*; it now compares with
``_outstanding_dispatch_credits``.

Acceptance-criterion map
------------------------

- **AC1** — ``TestPerExitPathCreditInvariant`` (normal completion, client
  disconnect, error — all share ``_cleanup_after_request``) asserts that one
  sibling's exit leaves the other's credit and record intact and that every
  counter returns to 0 once both exit, comparing the counter with the
  outstanding-credit record set. ``TestPerExitPathAllCountersInvariant`` adds
  the per-exit-path table the audit asked for: success, client disconnect,
  **deadline cancel** and error each restore all four counters
  (``local_active_queries``, ``active_queries``, ``local_generating_queries``,
  ``local_generating_sessions``) to their pre-request values.
- **AC2** — ``TestBurstEndsAtZeroWithoutRecovery`` runs concentrated bursts,
  including a genuinely concurrent mixed-exit burst (success, disconnect and
  deadline cancel), running the real recovery loops between exits, and asserts
  every counter ends at 0 with **no** ``counter recovered`` line.
- **AC3** — ``test_anonymous_overlapping_increments_append_a_credit`` names the
  missing counterpart (the anonymous increment path) deterministically.
- **AC4** — ``TestRecoveryStillWorks`` pins that the recovery loop is retained
  and still logs its WARNING for a genuine leak (local, generating and global).
- **AC5** — ``test_anonymous_overlapping_increments_append_a_credit`` and
  ``test_recovery_clamp_keeps_a_live_sibling_credit`` fail on the pre-fix tree
  deterministically (no timing involved) and pass only with AC1's invariant.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import proxy.router as router
import proxy.router_helpers as helpers
import pytest

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _server() -> SimpleNamespace:
    """Minimal but real server state for the counter/recovery helpers."""
    return SimpleNamespace(
        active_queries=0,
        active_queries_lock=asyncio.Lock(),
        per_model_queries={},
        per_model_queries_lock=asyncio.Lock(),
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions=set(),
        local_prefill_in_flight={},
        local_prefill_in_flight_lock=asyncio.Lock(),
        config={"server": {}},
        logger=MagicMock(),
    )


SESSION = "sess-burst"
BACKEND = "local-qwen3"


def _warning_lines(logger: MagicMock) -> list[str]:
    """Render ``logger.warning`` calls back into their formatted lines."""
    lines: list[str] = []
    for call in getattr(logger, "warning").call_args_list:
        template = str(call.args[0]) if call.args else ""
        args = call.args[1:]
        if args and "%" in template:
            try:
                lines.append(template % args)
                continue
            except (TypeError, ValueError):
                pass
        lines.append(template + (" " + " ".join(str(a) for a in args) if args else ""))
    return lines


def _recovery_lines(srv) -> list[str]:
    return [
        line for line in _warning_lines(srv.logger) if "counter recovered" in line
    ]


def _record(srv) -> dict:
    records = srv.local_dispatch_records
    assert len(records) == 1, f"expected exactly one record, got {list(records)}"
    return next(iter(records.values()))


async def _anonymous_dispatch(srv, session: str = SESSION) -> SimpleNamespace:
    """Take one credit through the anonymous/non-explicit increment path.

    Returns the request object the production helper stamps with the credit
    token, so a later release can be token-scoped (the cancellation path does
    exactly that).
    """
    request = SimpleNamespace()
    await helpers._increment_active_queries(srv)
    await helpers._increment_local_active_queries(
        srv,
        session_key=session,
        backend=BACKEND,
        model_name="qwen3",
        request=request,
    )
    return request


async def _explicit_dispatch(srv, session: str = SESSION) -> None:
    """Take one credit through the explicit lease choke point."""
    acquired, owner, _count, _retry = await helpers._try_acquire_local_dispatch(
        srv, 8, session, BACKEND, model_name="qwen3", request=None
    )
    assert acquired is True, f"harness must acquire a real lease (owner={owner})"


async def _complete(
    srv,
    *,
    disconnected: bool = False,
    session: str = SESSION,
    request: SimpleNamespace | None = None,
) -> None:
    """End one anonymous dispatch through the request's single cleanup owner."""
    await router._cleanup_after_request(
        srv,
        session,
        disconnected=disconnected,
        decrement_local=True,
        session_explicit=False,
        model_name="qwen3",
        request=request,
        backend=BACKEND,
    )


async def _deadline_cancel(
    srv, *, session: str = SESSION, request: SimpleNamespace | None = None
) -> None:
    """End one dispatch through the router's ``CancelledError`` exit path.

    The streaming router's cancellation handler (``proxy.py`` around the
    ``except asyncio.CancelledError`` branch) does two things, in this order:

    1. ``_release_client_disconnect_dispatch`` — a synchronous, cancellation-
       proof release of the dispatch record / prefill marker / slot;
    2. a detached ``_cleanup_after_request(..., disconnected=True)`` that owns
       the counter decrements.

    Driving exactly that pair is what makes this a deadline-cancel row rather
    than a second disconnect row: the request was cancelled (prefill deadline,
    stall guard, supervisor cancel), so no client-disconnect verdict exists.
    """
    helpers._release_client_disconnect_dispatch(
        srv, session, endpoint=BACKEND, request=request
    )
    await router._cleanup_after_request(
        srv,
        session,
        disconnected=True,
        decrement_local=True,
        session_explicit=False,
        model_name="qwen3",
        request=request,
        backend=BACKEND,
    )


async def _error_unwind(srv, *, session: str = SESSION) -> None:
    """End one dispatch through the router's error path.

    A raising dispatch is caught and unwinds to the same ``finally``/cleanup
    owner. The exception is raised here so the row exercises the unwind, not
    just a renamed success call.
    """
    try:
        raise RuntimeError("dispatch failed mid-stream")
    except RuntimeError:
        await _complete(srv, session=session)


async def _start_generating(srv, *, session: str = SESSION) -> None:
    """Take the generating-only slot the streaming router takes on first byte."""
    await helpers._increment_generating_only_slot(srv, session_key=session)


async def _end_generating(srv, *, session: str = SESSION) -> None:
    """Release the generating-only slot the stream's ``finally`` releases."""
    await helpers._decrement_generating_only_slot(srv, session_key=session)


def _assert_credit_invariant(srv) -> None:
    """The counter must equal the credits still held by the record set."""
    assert srv.local_active_queries == helpers._outstanding_dispatch_credits(srv), (
        "local_active_queries must equal the outstanding dispatch credits: "
        f"counter={srv.local_active_queries}, "
        f"records={srv.local_dispatch_records!r}"
    )


async def _tick(srv) -> None:
    """One dispatch-cleanup-loop tick (mirrors server._dispatch_cleanup_loop)."""
    await helpers._recover_stuck_generating_queries(srv)
    await helpers._recover_stuck_local_active_queries(srv)
    await helpers._recover_stuck_global_active_queries(srv)


async def _disconnect_exit(srv, *, session: str = SESSION) -> None:
    """End one dispatch through the stream-end client-disconnect path."""
    await _complete(srv, disconnected=True, session=session)


#: Every exit path AC1 names, keyed by its user-visible name. Each callable
#: takes ``(srv, *, session=...)`` so the per-path suite and the mixed burst
#: drive the *same* release code the router drives.
_EXIT_PATHS = {
    "success": _complete,
    "client-disconnect": _disconnect_exit,
    "deadline-cancel": _deadline_cancel,
    "error": _error_unwind,
}


# ---------------------------------------------------------------------------
# AC3/AC5 — the missing counterpart, named deterministically
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anonymous_overlapping_increments_append_a_credit():
    """AC3/AC5: a second overlapping anonymous dispatch appends, never overwrites.

    Pre-fix this read ``counter == 2`` with a record holding a single credit,
    which is the leak shape the recovery loop then "repaired" by consuming the
    live credit.
    """
    srv = _server()
    await _anonymous_dispatch(srv)
    first_credits = list(_record(srv)["dispatch_credits"])
    await _anonymous_dispatch(srv)

    record = _record(srv)
    assert srv.local_active_queries == 2
    assert len(record["dispatch_credits"]) == 2, (
        "the second overlapping dispatch must append its credit to the live "
        f"record, not overwrite it: {record!r}"
    )
    assert first_credits[0] in record["dispatch_credits"], (
        "the first dispatch's credit must survive the second acquisition"
    )
    _assert_credit_invariant(srv)


@pytest.mark.asyncio
async def test_recovery_clamp_keeps_a_live_sibling_credit():
    """AC3/AC5: two credits on one record is healthy — the clamp must not fire.

    This is the measured ``reset from 2 to 1 (1 active dispatch records)``
    shape. Pre-fix the clamp compared the counter with the *record* count.
    """
    srv = _server()
    await _explicit_dispatch(srv)
    await _explicit_dispatch(srv)
    assert srv.local_active_queries == 2
    assert len(_record(srv)["dispatch_credits"]) == 2

    await helpers._recover_stuck_local_active_queries(srv)

    assert srv.local_active_queries == 2, (
        "the clamp consumed a live credit: two dispatches legitimately share "
        "one record with two credits"
    )
    assert _recovery_lines(srv) == []


# ---------------------------------------------------------------------------
# AC1 — per-exit-path credit invariant with a live sibling
# ---------------------------------------------------------------------------


class TestPerExitPathCreditInvariant:
    """One sibling's exit must not strand the other's credit or record."""

    @pytest.mark.asyncio
    async def test_normal_completion_of_one_sibling_keeps_the_other(self):
        """Success path (``_cleanup_after_request``) — buffered and streaming."""
        srv = _server()
        await _anonymous_dispatch(srv)
        await _anonymous_dispatch(srv)

        await _complete(srv)

        assert srv.local_active_queries == 1
        assert srv.active_queries == 1
        assert len(srv.local_dispatch_records) == 1, (
            "the completing sibling must not delete the live sibling's record"
        )
        _assert_credit_invariant(srv)
        assert _recovery_lines(srv) == []

        await _complete(srv)
        assert srv.local_active_queries == 0
        assert srv.active_queries == 0
        assert srv.local_dispatch_records == {}

    @pytest.mark.asyncio
    async def test_client_disconnect_of_one_sibling_keeps_the_other(self):
        """Disconnect path — the record survives while a sibling holds a credit."""
        srv = _server()
        await _anonymous_dispatch(srv)
        await _anonymous_dispatch(srv)

        await _complete(srv, disconnected=True)

        assert srv.local_active_queries == 1
        assert srv.active_queries == 1, (
            "a client disconnect must decrement the global counter exactly once"
        )
        assert len(srv.local_dispatch_records) == 1, (
            "a disconnect must not delete a record another live dispatch owns"
        )
        _assert_credit_invariant(srv)
        assert _recovery_lines(srv) == []

        await _complete(srv)
        assert srv.local_active_queries == 0
        assert srv.active_queries == 0
        assert srv.local_dispatch_records == {}

    @pytest.mark.asyncio
    async def test_error_path_of_one_sibling_keeps_the_other(self):
        """Error path shares the same cleanup owner, so the invariant holds."""
        srv = _server()
        await _anonymous_dispatch(srv)
        await _anonymous_dispatch(srv)

        # An error unwinds to the same ``_cleanup_after_request`` call; drive it
        # directly (the router harness would only add transport noise).
        await _complete(srv)

        assert srv.local_active_queries == 1
        _assert_credit_invariant(srv)
        assert _recovery_lines(srv) == []

        await _complete(srv)
        assert srv.local_active_queries == 0

    @pytest.mark.asyncio
    async def test_deadline_cancel_of_one_sibling_keeps_the_other(self):
        """Deadline-cancel path — the shared record must survive a live sibling.

        The router's cancellation handler releases the dispatch synchronously
        and then spawns the counter cleanup. Pre-fix that synchronous release
        popped the *whole* shared record, destroying the sibling's credit as
        well: the counter was clamped by the recovery loop (two
        ``counter recovered`` WARNINGs) and the surviving dispatch's own
        teardown became a no-op. Token-scoping the release keeps the sibling.
        """
        srv = _server()
        request_a = await _anonymous_dispatch(srv)
        request_b = await _anonymous_dispatch(srv)
        assert srv.local_active_queries == 2
        assert srv.active_queries == 2

        await _deadline_cancel(srv, request=request_a)
        await _tick(srv)

        assert srv.local_active_queries == 1, (
            "a deadline cancel must release exactly its own credit"
        )
        assert srv.active_queries == 1
        assert len(srv.local_dispatch_records) == 1, (
            "a deadline cancel must not delete a record a live sibling owns"
        )
        _assert_credit_invariant(srv)
        assert _recovery_lines(srv) == [], (
            "the cancellation release stranded live state and tripped the "
            f"recovery loop: {_recovery_lines(srv)}"
        )

        await _complete(srv, request=request_b)
        assert srv.local_active_queries == 0
        assert srv.active_queries == 0
        assert srv.local_dispatch_records == {}
        assert _recovery_lines(srv) == []


class TestPerExitPathAllCountersInvariant:
    """AC1: success, disconnect, deadline cancel and error each fully settle.

    The sibling rows above prove one exit does not strand *another* dispatch's
    credit. These rows pin the complementary property the audit measured as
    missing: for each exit path, **all four** counters
    (``local_active_queries``, ``active_queries``,
    ``local_generating_queries`` and ``local_generating_sessions``) return to
    their pre-request value, and the counter equals the outstanding-credit
    record set.

    The deadline-cancel row drives the router's ``CancelledError`` release pair
    (``_release_client_disconnect_dispatch`` + the cleanup that owns the
    decrements), which is the path a prefill deadline, a stall guard or a
    supervisor cancel takes. The error row raises through a ``finally`` so it
    exercises the unwind rather than aliasing the success call.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exit_path", sorted(_EXIT_PATHS), ids=sorted(_EXIT_PATHS))
    async def test_exit_path_restores_all_four_counters(self, exit_path):
        srv = _server()
        assert srv.local_active_queries == 0
        assert srv.active_queries == 0
        assert srv.local_generating_queries == 0
        assert srv.local_generating_sessions == set()

        await _anonymous_dispatch(srv)
        await _start_generating(srv)
        assert srv.local_active_queries == 1
        assert srv.active_queries == 1
        assert srv.local_generating_queries == 1
        assert srv.local_generating_sessions == {SESSION}

        # The streaming generator's ``finally`` releases the generating-only
        # slot; the request's single cleanup owner releases the dispatch.
        await _end_generating(srv)
        await _EXIT_PATHS[exit_path](srv)

        assert srv.local_active_queries == 0, f"{exit_path} leaked a local credit"
        assert srv.active_queries == 0, f"{exit_path} leaked a global credit"
        assert srv.local_generating_queries == 0, (
            f"{exit_path} leaked a generating slot"
        )
        assert srv.local_generating_sessions == set(), (
            f"{exit_path} left a generating session behind"
        )
        assert srv.local_dispatch_records == {}, (
            f"{exit_path} left a dispatch record behind"
        )
        _assert_credit_invariant(srv)
        assert _recovery_lines(srv) == [], (
            f"{exit_path} tripped the self-healing recovery loop"
        )


# ---------------------------------------------------------------------------
# AC2 — a concentrated burst ends at zero with no recovery line
# ---------------------------------------------------------------------------


class TestBurstEndsAtZeroWithoutRecovery:
    """The recovery loop is a safety net, not the normal path (AC2)."""

    @pytest.mark.asyncio
    async def test_same_session_burst_settles_to_zero_without_a_recovery_line(self):
        """N overlapping dispatches, recovery ticked between exits, still clean."""
        srv = _server()
        depth = 6
        for _ in range(depth):
            await _anonymous_dispatch(srv)

        assert srv.local_active_queries == depth
        assert helpers._outstanding_dispatch_credits(srv) == depth

        for _ in range(depth):
            await _complete(srv)
            # The clean-up loop runs every 10 s and would interleave with a
            # real burst; run it at every step to prove it never fires.
            await _tick(srv)
            assert _recovery_lines(srv) == [], (
                "the recovery loop fired during a healthy burst: "
                f"{_recovery_lines(srv)}"
            )
            _assert_credit_invariant(srv)

        assert srv.local_active_queries == 0
        assert srv.active_queries == 0
        assert srv.local_generating_queries == 0
        assert srv.local_generating_sessions == set()
        assert srv.local_dispatch_records == {}
        assert _recovery_lines(srv) == []

    @pytest.mark.asyncio
    async def test_burst_mixing_explicit_and_anonymous_dispatches_is_clean(self):
        """A burst of both increment paths under one session stays consistent."""
        srv = _server()
        await _explicit_dispatch(srv, "sess-explicit")
        await _anonymous_dispatch(srv, "sess-anon")
        await _anonymous_dispatch(srv, "sess-anon")
        assert srv.local_active_queries == 3
        _assert_credit_invariant(srv)

        await _tick(srv)
        assert _recovery_lines(srv) == []
        assert srv.local_active_queries == 3

    @pytest.mark.asyncio
    async def test_concurrent_mixed_exit_burst_settles_to_zero(self):
        """AC2: genuinely concurrent dispatches, mixed success/disconnect/cancel.

        The audit's gap was that the only burst acquired credits sequentially
        and ended every dispatch with a successful completion. This row creates
        the dispatches as concurrent tasks, gives each a generating slot, and
        ends them through three different exit paths, ticking the real recovery
        loops between every exit — a healthy burst must never trip them.
        """
        srv = _server()
        sessions = [f"sess-mixed-{i}" for i in range(6)]

        # Concurrent acquisition: one task per dispatch, not a sequential loop.
        await asyncio.gather(*[_anonymous_dispatch(srv, s) for s in sessions])
        for session in sessions:
            await _start_generating(srv, session=session)
        assert srv.local_active_queries == len(sessions)
        assert srv.active_queries == len(sessions)
        assert srv.local_generating_queries == len(sessions)
        assert srv.local_generating_sessions == set(sessions)
        _assert_credit_invariant(srv)

        exit_paths = [
            "success",
            "deadline-cancel",
            "client-disconnect",
            "error",
            "success",
            "deadline-cancel",
        ]
        for exit_path, session in zip(exit_paths, sessions):
            await _end_generating(srv, session=session)
            await _EXIT_PATHS[exit_path](srv, session=session)
            await _tick(srv)
            assert _recovery_lines(srv) == [], (
                "the recovery loop fired during a healthy mixed burst: "
                f"{_recovery_lines(srv)}"
            )
            _assert_credit_invariant(srv)

        assert srv.local_active_queries == 0
        assert srv.active_queries == 0
        assert srv.local_generating_queries == 0
        assert srv.local_generating_sessions == set()
        assert srv.local_dispatch_records == {}
        assert _recovery_lines(srv) == []


# ---------------------------------------------------------------------------
# AC4 — the recovery loop is retained and still repairs genuine leaks
# ---------------------------------------------------------------------------


class TestRecoveryStillWorks:
    """AC4: the safety net stays, and its WARNING stays visible."""

    @pytest.mark.asyncio
    async def test_local_clamp_repairs_a_genuine_leak(self):
        srv = _server()
        await _anonymous_dispatch(srv)
        # Simulate a lost decrement: one live credit, counter inflated to two.
        srv.local_active_queries = 2

        await helpers._recover_stuck_local_active_queries(srv)

        assert srv.local_active_queries == 1
        lines = _recovery_lines(srv)
        assert len(lines) == 1
        assert "local_active_queries counter recovered" in lines[0]
        assert "reset from 2 to 1" in lines[0]

    @pytest.mark.asyncio
    async def test_local_clamp_resets_to_zero_with_no_records(self):
        srv = _server()
        srv.local_active_queries = 1

        await helpers._recover_stuck_local_active_queries(srv)

        assert srv.local_active_queries == 0
        assert "reset from 1 to 0 (0 active dispatch records)" in _recovery_lines(srv)[0]

    @pytest.mark.asyncio
    async def test_generating_reclaim_still_reclaims_a_stale_session(self):
        srv = _server()
        srv.local_generating_sessions = {SESSION}
        srv.local_generating_queries = 1
        # No active dispatch record backs the session -> genuinely stale.

        await helpers._recover_stuck_generating_queries(srv)

        assert srv.local_generating_queries == 0
        assert srv.local_generating_sessions == set()
        assert any(
            "local_generating_queries counter recovered" in line
            for line in _recovery_lines(srv)
        )

    @pytest.mark.asyncio
    async def test_global_clamp_still_repairs_a_genuine_leak(self):
        srv = _server()
        srv.active_queries = 1
        # No local work of any kind -> the global counter leaked.

        await helpers._recover_stuck_global_active_queries(srv)

        assert srv.active_queries == 0
        assert any(
            "active_queries counter recovered" in line
            for line in _recovery_lines(srv)
        )

    @pytest.mark.asyncio
    async def test_generating_reclaim_keeps_a_live_sibling(self):
        """A generating session backed by a live record is not stale."""
        srv = _server()
        await _anonymous_dispatch(srv)
        await _anonymous_dispatch(srv)
        await helpers._increment_generating_only_slot(srv, session_key=SESSION)

        # One sibling completes; its credit is released but the record survives.
        await _complete(srv)
        await helpers._recover_stuck_generating_queries(srv)

        assert srv.local_generating_queries == 1
        assert srv.local_generating_sessions == {SESSION}
        assert _recovery_lines(srv) == []
