"""Lease-owner retention priority in the local contention queue.

LP-0MVI2VRT2000V1AP. A session that already holds — or has just released — a
local dispatch lease re-queues behind newcomers, waits the full
``contention_queue_max_wait_seconds`` and then detours to a paid remote
provider, only to come back and re-prefill. Observed on
``audit-WL-0MVFRRKKN00544PQ-phase2_child_5-6e22ef57`` (2026-10-21): seven
local↔remote transitions in twelve minutes, each remote hop preceded by a
120 s queue wait, none of them an ETA-guard or context-size decision.

These tests cover:

- AC1: a retained session (active lease, or lease granted within the grace
  window, with local cache history) is dispatched ahead of a newcomer, and
  the reorder is logged naming both sessions and the reason;
- AC2: the A→release→B→A scenario dispatches A first;
- AC3: a session whose prefix was evicted (no cache history) earns no
  retention and is not held ahead of a newcomer;
- AC5: local↔remote transitions are counted per session (``routing_hop``).
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from proxy.provider import (
    _cached_ratio_source,
    _last_cached_ratio,
    _last_routing_hop,
    clear_cached_ratio_for_session,
    note_routing_hop,
    update_cached_ratio,
)

from proxy import contention_queue

MODEL = "Qwen3"
GRACE = 300.0


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    _last_routing_hop.clear()
    yield
    contention_queue.reset()
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    _last_routing_hop.clear()


def _have_cache_history(session_id: str) -> None:
    """Give *session_id* a recorded local ratio (i.e. it served locally)."""
    update_cached_ratio(MODEL, session_id, cached_tokens=1000, prompt_tokens=4000)


class TestRetentionOrdering:
    """AC1/AC2/AC3: retained waiters are ordered ahead of newcomers."""

    @pytest.mark.asyncio
    async def test_retained_session_is_woken_ahead_of_a_queued_newcomer(self, caplog):
        """AC2: B queues first; retained A returns and is dispatched first."""
        caplog.set_level(logging.INFO, logger="llama-proxy")
        slot_free = {"value": False}
        dispatched: list[str] = []

        async def waiter(session_id: str, retained: bool):
            elapsed = await contention_queue.wait_for_local_slot(
                5.0,
                max_depth=8,
                slot_free_check=lambda: slot_free["value"],
                session_id=session_id,
                retained=retained,
            )
            if elapsed is not None:
                dispatched.append(session_id)
            return elapsed

        newcomer = asyncio.create_task(waiter("session-B-newcomer", retained=False))
        await asyncio.sleep(0.01)
        retained = asyncio.create_task(waiter("session-A-retained", retained=True))
        await asyncio.sleep(0.01)

        # One slot frees: the retained session must take it, not the newcomer.
        slot_free["value"] = True
        await contention_queue.wake(1)

        await retained
        assert dispatched == ["session-A-retained"]

        # The newcomer keeps waiting (spurious wake after A claimed the slot).
        slot_free["value"] = False
        await asyncio.sleep(0.01)
        newcomer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await newcomer

        messages = " ".join(r.message for r in caplog.records)
        assert "contention_queue_retention" in messages
        assert "session-A-retained" in messages
        assert "session-B-newcomer" in messages
        assert "lease_owner_retention" in messages

    @pytest.mark.asyncio
    async def test_fifo_preserved_between_retained_waiters(self):
        """Retention must not reorder two retained sessions (FIFO within class)."""
        slot_free = {"value": False}
        dispatched: list[str] = []

        async def waiter(session_id: str):
            elapsed = await contention_queue.wait_for_local_slot(
                5.0, 8, lambda: slot_free["value"], session_id=session_id,
                retained=True,
            )
            if elapsed is not None:
                dispatched.append(session_id)

        first = asyncio.create_task(waiter("retained-1"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(waiter("retained-2"))
        await asyncio.sleep(0.01)

        slot_free["value"] = True
        await contention_queue.wake(1)
        await first
        assert dispatched == ["retained-1"]

        slot_free["value"] = False
        await asyncio.sleep(0.01)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second

    @pytest.mark.asyncio
    async def test_returning_session_dispatched_before_queued_newcomer(self, caplog):
        """AC2: A leased local, released, B queued, A returns → A dispatches first.

        Exercises the full eligibility path (grant registry + cache history)
        rather than passing ``retained`` by hand.
        """
        caplog.set_level(logging.INFO, logger="llama-proxy")
        slot_free = {"value": False}
        dispatched: list[str] = []

        def retained_for(session_id: str) -> bool:
            return contention_queue.is_retained_local_owner(
                session_id, MODEL, grace_seconds=GRACE
            )

        async def waiter(session_id: str):
            elapsed = await contention_queue.wait_for_local_slot(
                5.0,
                8,
                lambda: slot_free["value"],
                session_id=session_id,
                retained=retained_for(session_id),
            )
            if elapsed is not None:
                dispatched.append(session_id)

        # A ran locally, then released its lease (the grace window keeps it a
        # returning session); its cache record survives.
        contention_queue.note_local_grant("session-A")
        _have_cache_history("session-A")

        newcomer = asyncio.create_task(waiter("session-B"))
        await asyncio.sleep(0.01)
        returner = asyncio.create_task(waiter("session-A"))
        await asyncio.sleep(0.01)

        slot_free["value"] = True
        await contention_queue.wake(1)
        await returner

        assert dispatched == ["session-A"]

        slot_free["value"] = False
        await asyncio.sleep(0.01)
        newcomer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await newcomer

        messages = " ".join(r.message for r in caplog.records)
        assert "contention_queue_retention" in messages
        assert "reason=lease_owner_retention" in messages

    @pytest.mark.asyncio
    async def test_evicted_session_queues_behind_the_newcomer(self):
        """AC3: no cache history means no priority — FIFO order is preserved."""
        slot_free = {"value": False}
        dispatched: list[str] = []

        def retained_for(session_id: str) -> bool:
            return contention_queue.is_retained_local_owner(
                session_id, MODEL, grace_seconds=GRACE
            )

        async def waiter(session_id: str):
            elapsed = await contention_queue.wait_for_local_slot(
                5.0,
                8,
                lambda: slot_free["value"],
                session_id=session_id,
                retained=retained_for(session_id),
            )
            if elapsed is not None:
                dispatched.append(session_id)

        contention_queue.note_local_grant("session-evicted")
        _have_cache_history("session-evicted")
        clear_cached_ratio_for_session("session-evicted")
        assert not retained_for("session-evicted")

        newcomer = asyncio.create_task(waiter("session-B"))
        await asyncio.sleep(0.01)
        evicted = asyncio.create_task(waiter("session-evicted"))
        await asyncio.sleep(0.01)

        slot_free["value"] = True
        await contention_queue.wake(1)
        await newcomer

        assert dispatched == ["session-B"], "evicted session must not jump the queue"

        slot_free["value"] = False
        await asyncio.sleep(0.01)
        evicted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await evicted

    @pytest.mark.asyncio
    async def test_newcomer_still_dispatches_when_it_is_alone(self):
        """Non-retained waiters keep today's behaviour (no regression)."""
        slot_free = {"value": False}

        task = asyncio.create_task(
            contention_queue.wait_for_local_slot(
                5.0, 8, lambda: slot_free["value"], session_id="solo", retained=False
            )
        )
        await asyncio.sleep(0.01)
        slot_free["value"] = True
        await contention_queue.wake(1)
        assert await task is not None


class TestRetentionEligibility:
    """The retained predicate: recent local owner AND surviving cache history."""

    def test_recent_local_owner_with_cache_history_is_retained(self):
        contention_queue.note_local_grant("sess-A")
        _have_cache_history("sess-A")
        assert contention_queue.is_retained_local_owner(
            "sess-A", MODEL, grace_seconds=GRACE
        )

    def test_evicted_prefix_is_not_retained(self):
        """AC3: no cache history (evicted/cleared) means no retention."""
        contention_queue.note_local_grant("sess-A")
        _have_cache_history("sess-A")
        clear_cached_ratio_for_session("sess-A")
        assert not contention_queue.is_retained_local_owner(
            "sess-A", MODEL, grace_seconds=GRACE
        )

    def test_grant_outside_the_grace_window_is_not_retained(self):
        contention_queue.note_local_grant("sess-A")
        _have_cache_history("sess-A")
        assert not contention_queue.is_retained_local_owner(
            "sess-A", MODEL, grace_seconds=0.0
        )

    def test_session_without_a_local_grant_is_not_retained(self):
        _have_cache_history("sess-new")
        assert not contention_queue.is_retained_local_owner(
            "sess-new", MODEL, grace_seconds=GRACE
        )

    def test_no_session_id_is_never_retained(self):
        assert not contention_queue.is_retained_local_owner(
            None, MODEL, grace_seconds=GRACE
        )


class TestThrashReplay:
    """AC6: replaying the recorded thrash pattern yields fewer transitions."""

    @staticmethod
    def _retained_for(session_id: str, grace: float) -> bool:
        return contention_queue.is_retained_local_owner(
            session_id, MODEL, grace_seconds=grace
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "grace,expected_local_wins", [(GRACE, 7), (0.0, 0)]
    )
    async def test_returning_session_wins_the_slot_every_round(
        self, grace, expected_local_wins
    ):
        """Seven rounds of the recorded pattern (newcomer already queued).

        With retention the returning session reclaims the slot every round, so
        it never hops to a remote provider (1 transition, local only). With
        retention disabled it loses every round — the observed thrash.
        """
        slot_free = {"value": False}
        local_wins = 0

        async def waiter(session_id: str):
            nonlocal local_wins
            elapsed = await contention_queue.wait_for_local_slot(
                0.05,
                8,
                lambda: slot_free["value"],
                session_id=session_id,
                retained=self._retained_for(session_id, grace),
            )
            return elapsed is not None

        contention_queue.note_local_grant("session-A")
        _have_cache_history("session-A")

        for _ in range(7):
            slot_free["value"] = False
            newcomer = asyncio.create_task(waiter("session-B"))
            await asyncio.sleep(0.005)
            returner = asyncio.create_task(waiter("session-A"))
            await asyncio.sleep(0.005)

            slot_free["value"] = True
            await contention_queue.wake(1)

            if await returner:
                local_wins += 1

            slot_free["value"] = False
            await asyncio.sleep(0.005)
            newcomer.cancel()
            await asyncio.gather(newcomer, return_exceptions=True)

        assert local_wins == expected_local_wins

    def test_hop_count_for_the_two_outcomes(self):
        """AC6: 7 local reclaims = 0 hops; 7 remote detours = 14 hops."""
        note_routing_hop("reclaimed", "local")
        assert all(note_routing_hop("reclaimed", "local") == 0 for _ in range(7))

        assert note_routing_hop("thrashed", "local") == 0
        for round_index in range(1, 8):
            assert note_routing_hop("thrashed", "opencode-go-3") == 2 * round_index - 1
            assert note_routing_hop("thrashed", "local") == 2 * round_index
        assert note_routing_hop("thrashed", "local") >= 3


class TestRoutingHopCount:
    """AC5: local↔remote transitions are counted per session."""

    def test_hops_count_only_when_the_provider_changes(self):
        assert note_routing_hop("s1", "local-qwen3") == 0
        assert note_routing_hop("s1", "local-qwen3") == 0
        assert note_routing_hop("s1", "opencode-go-3") == 1
        assert note_routing_hop("s1", "opencode-go-3") == 1
        assert note_routing_hop("s1", "local-qwen3") == 2
        assert note_routing_hop("s1", "munitoai-free") == 3

    def test_hop_counts_are_per_session(self):
        note_routing_hop("s1", "local-qwen3")
        note_routing_hop("s1", "opencode-go-3")
        assert note_routing_hop("s2", "local-qwen3") == 0
        assert note_routing_hop("s1", "opencode-go-3") == 1
