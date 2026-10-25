"""ETA-aware admission ordering in the contention queue.

LP-0MVIQ0G2M0006QF6 AC1 — ``local_max_concurrent_prefills: 1`` makes a long
prefill head-of-line blocking for every queued local request (baseline: 1 827
prefill-saturated ``contention_queue_wait`` samples vs 468 generating). Within
a retention class the queue now admits the **shortest projected prefill
first**, so a queue of 20-second turns is not parked behind one 7-minute
prefill.

Constraints covered here:

- retention still wins over size (a retained session is never starved by a
  shorter newcomer);
- ordering applies only within a retention class;
- a long-ETA waiter is promoted to the front of its class after
  ``contention_queue_eta_promotion_seconds`` so it cannot starve;
- ``contention_queue_eta_ordering: false`` restores plain FIFO (rollback).
"""

from __future__ import annotations

import asyncio

import pytest

from proxy import contention_queue


@pytest.fixture(autouse=True)
def _clean_queue():
    contention_queue.reset()
    yield
    contention_queue.reset()


async def _waiter(
    session_id: str,
    *,
    retained: bool,
    eta_seconds: float | None,
    eta_ordering: bool,
    promote_after_seconds: float = 0.0,
    slot_free: dict,
    dispatched: list,
):
    elapsed = await contention_queue.wait_for_local_slot(
        5.0,
        max_depth=8,
        slot_free_check=lambda: slot_free["value"],
        session_id=session_id,
        retained=retained,
        eta_seconds=eta_seconds,
        eta_ordering=eta_ordering,
        promote_after_seconds=promote_after_seconds,
    )
    if elapsed is not None:
        dispatched.append(session_id)
    return elapsed


@pytest.mark.asyncio
async def test_shortest_eta_is_admitted_first(caplog):
    """AC1: pending short and long prefills, one slot frees ⇒ the short one
    wins even though the long one queued first."""
    import logging

    caplog.set_level(logging.INFO, logger="llama-proxy")
    slot_free = {"value": False}
    dispatched: list[str] = []

    long_task = asyncio.create_task(
        _waiter(
            "long", retained=False, eta_seconds=400.0, eta_ordering=True,
            slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)
    short_task = asyncio.create_task(
        _waiter(
            "short", retained=False, eta_seconds=20.0, eta_ordering=True,
            slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)
    assert contention_queue.queue_depth() == 2

    slot_free["value"] = True
    await contention_queue.wake(2)
    await asyncio.gather(short_task, long_task)

    assert dispatched, "a slot freed; one waiter must be admitted"
    assert dispatched[0] == "short", (
        f"shortest projected prefill must be admitted first, got {dispatched}"
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "contention_queue_eta_order" in messages
    assert "reason=shortest_eta_first" in messages


@pytest.mark.asyncio
async def test_retention_still_wins_over_a_shorter_newcomer():
    """AC1: retention priority is the primary tie-break — a retained long
    prefill is not overtaken by a shorter non-retained newcomer."""
    slot_free = {"value": False}
    dispatched: list[str] = []

    retained_long = asyncio.create_task(
        _waiter(
            "retained-long", retained=True, eta_seconds=400.0,
            eta_ordering=True, slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)
    short_newcomer = asyncio.create_task(
        _waiter(
            "short-newcomer", retained=False, eta_seconds=20.0,
            eta_ordering=True, slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)

    slot_free["value"] = True
    await contention_queue.wake(2)
    await asyncio.gather(retained_long, short_newcomer)

    assert dispatched and dispatched[0] == "retained-long", (
        "retention must win over a shorter non-retained newcomer; "
        f"got {dispatched}"
    )


@pytest.mark.asyncio
async def test_ordering_applies_within_the_same_retention_class():
    """AC1: two newcomers order by ETA; the retained class is separate."""
    slot_free = {"value": False}
    dispatched: list[str] = []

    t1 = asyncio.create_task(
        _waiter(
            "newcomer-long", retained=False, eta_seconds=300.0,
            eta_ordering=True, slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(
        _waiter(
            "newcomer-short", retained=False, eta_seconds=15.0,
            eta_ordering=True, slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)

    slot_free["value"] = True
    await contention_queue.wake(2)
    await asyncio.gather(t1, t2)

    assert dispatched[0] == "newcomer-short"


@pytest.mark.asyncio
async def test_long_waiter_is_promoted_after_the_promotion_window():
    """AC1 starvation bound: a long-ETA waiter older than the promotion
    window is treated as highest priority within its class."""
    slot_free = {"value": False}
    dispatched: list[str] = []

    long_task = asyncio.create_task(
        _waiter(
            "starving-long", retained=False, eta_seconds=500.0,
            eta_ordering=True, promote_after_seconds=0.03,
            slot_free=slot_free, dispatched=dispatched,
        )
    )
    # Let the long waiter age past the promotion window.
    await asyncio.sleep(0.06)
    short_task = asyncio.create_task(
        _waiter(
            "fresh-short", retained=False, eta_seconds=10.0,
            eta_ordering=True, promote_after_seconds=0.03,
            slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)

    slot_free["value"] = True
    await contention_queue.wake(2)
    await asyncio.gather(long_task, short_task)

    assert dispatched[0] == "starving-long", (
        "an aged waiter must be promoted ahead of a fresh shorter one; "
        f"got {dispatched}"
    )


@pytest.mark.asyncio
async def test_eta_ordering_disabled_restores_fifo():
    """AC1 rollback: ``eta_ordering=False`` keeps plain FIFO order."""
    slot_free = {"value": False}
    dispatched: list[str] = []

    first = asyncio.create_task(
        _waiter(
            "fifo-first", retained=False, eta_seconds=400.0,
            eta_ordering=False, slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)
    second = asyncio.create_task(
        _waiter(
            "fifo-second", retained=False, eta_seconds=5.0,
            eta_ordering=False, slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)

    slot_free["value"] = True
    await contention_queue.wake(2)
    await asyncio.gather(first, second)

    assert dispatched[0] == "fifo-first", "ordering disabled must keep FIFO"


@pytest.mark.asyncio
async def test_missing_eta_never_jumps_ahead_of_a_known_eta():
    """A waiter with no projection is ordered last within its class (an
    unknown ETA must not displace a known short one)."""
    slot_free = {"value": False}
    dispatched: list[str] = []

    unknown_task = asyncio.create_task(
        _waiter(
            "unknown-eta", retained=False, eta_seconds=None,
            eta_ordering=True, slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)
    known_task = asyncio.create_task(
        _waiter(
            "known-eta", retained=False, eta_seconds=30.0,
            eta_ordering=True, slot_free=slot_free, dispatched=dispatched,
        )
    )
    await asyncio.sleep(0.01)

    slot_free["value"] = True
    await contention_queue.wake(2)
    await asyncio.gather(unknown_task, known_task)

    assert dispatched[0] == "known-eta"
