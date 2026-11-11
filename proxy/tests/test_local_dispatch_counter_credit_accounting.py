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
  outstanding-credit record set.
- **AC2** — ``TestBurstEndsAtZeroWithoutRecovery`` runs a concentrated
  same-session burst, running the real recovery loops between completions, and
  asserts every counter ends at 0 with **no** ``counter recovered`` line.
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


async def _anonymous_dispatch(srv, session: str = SESSION) -> None:
    """Take one credit through the anonymous/non-explicit increment path."""
    await helpers._increment_active_queries(srv)
    await helpers._increment_local_active_queries(
        srv, session_key=session, backend=BACKEND, model_name="qwen3"
    )


async def _explicit_dispatch(srv, session: str = SESSION) -> None:
    """Take one credit through the explicit lease choke point."""
    acquired, owner, _count, _retry = await helpers._try_acquire_local_dispatch(
        srv, 8, session, BACKEND, model_name="qwen3", request=None
    )
    assert acquired is True, f"harness must acquire a real lease (owner={owner})"


async def _complete(srv, *, disconnected: bool = False) -> None:
    """End one anonymous dispatch through the request's single cleanup owner."""
    await router._cleanup_after_request(
        srv,
        SESSION,
        disconnected=disconnected,
        decrement_local=True,
        session_explicit=False,
        model_name="qwen3",
        backend=BACKEND,
    )


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
        assert len(srv.local_dispatch_records) == 1, (
            "a disconnect must not delete a record another live dispatch owns"
        )
        _assert_credit_invariant(srv)
        assert _recovery_lines(srv) == []

        await _complete(srv)
        assert srv.local_active_queries == 0
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
