"""Integration tests for LP-0MTE9HAF8008909G F4 / LP-0MTIHWYGJ006YS6X.

LP-0MSB0RV72001KNRV fixed ghost slot registry entries for the periodic
cleanup (idle_timeout / orphan_cleanup). LP-0MSB0RP7F000U0WJ carried that
fix with `test_cleanup_stale_frees_slot_registry_entries` and
`test_expired_lease_frees_slot`. This module closes the remaining
coverage gap for AC 1/4 of LP-0MTIHWYGJ006YS6X:

- orphan-cleanup continuity is preserved (session-slot mapping does not
  fragment when the pool recovers)
- the ghost-slot failure mode stays closed under churn
- lease churn reduction / reliable affinity is verified end-to-end
"""

import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import proxy.session as sess
from proxy.router_helpers import _cleanup_stale_local_dispatch

pytestmark = pytest.mark.asyncio


class TestSessionSlotAffinityContinuity:

    async def test_churn_survivor_keeps_slot_and_freed_slots_are_reusable(self):
        """
        After a churn event (idle_timeout + orphan_cleanup), a new
        session can claim a freed slot, while a surviving valid session
        with an active lease keeps its slot affinity.
        """
        sess._slot_owners.clear()

        # Pre-assign: survivor owns slot 0, victims own slots 1 and 2
        survivor_slot = sess._slot_id_for_session("affinity-survivor", 3)
        idle_slot = sess._slot_id_for_session("affinity-idle-victim", 3)
        orphan_slot = sess._slot_id_for_session("affinity-orphan-victim", 3)
        assert survivor_slot == 0
        assert idle_slot == 1
        assert orphan_slot == 2

        srv = SimpleNamespace(
            config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
            local_active_queries=3,
            local_active_queries_lock=asyncio.Lock(),
            local_dispatch_records={
                "affinity-survivor": {
                    "backend": "local",
                    "started_at": time.monotonic(),
                    "active": True,
                    "expires_at": 10**12,
                },
                "affinity-idle-victim": {
                    "backend": "local",
                    "started_at": time.monotonic(),
                    "active": False,
                    "expires_at": 0.0,
                },
                "affinity-orphan-victim": {
                    "backend": "local",
                    "started_at": time.monotonic(),
                    "active": True,
                    "expires_at": 0.0,
                },
            },
            local_dispatch_records_lock=asyncio.Lock(),
            logger=MagicMock(),
        )

        # _cleanup_stale_local_dispatch may run a /slots liveness check;
        # stub it so the orphan branch falls through to cleanup (fail-open).
        async def fake_query(*args, **kwargs):
            return False

        monkey = pytest.importorskip("pytest")
        from unittest.mock import patch

        with patch("proxy.router_helpers._query_slot_processing", side_effect=fake_query):
            removed = await _cleanup_stale_local_dispatch(srv)

        assert removed >= 2
        assert "affinity-survivor" in srv.local_dispatch_records
        assert "affinity-survivor" in sess._slot_owners.values()
        assert "affinity-idle-victim" not in sess._slot_owners.values()
        assert "affinity-orphan-victim" not in sess._slot_owners.values()

        survivor_again = sess._slot_id_for_session("affinity-survivor", 3)
        assert survivor_again == survivor_slot

        fresh_slot = sess._slot_id_for_session("affinity-fresh", 3)
        assert fresh_slot in (1, 2)

        try:
            sess._slot_owners.clear()
        except Exception:
            pass

    async def test_no_ghost_slots_after_repeated_cleanup(self):
        """
        Expired dispatches must not leave ghost entries in _slot_owners
        after _cleanup_stale_local_dispatch runs.

        Simulate a pool of 2 slots that have accumulated ghost entries
        from repeated idle-timeout releases that never freed slots.
        """
        sess._slot_owners.clear()
        pool = 2

        # Poison _slot_owners: ghost entries from old sessions
        sess._slot_owners[0] = "ghost-session-0"
        sess._slot_owners[1] = "ghost-session-1"
        ghost_count_before = len(sess._slot_owners)

        srv = SimpleNamespace(
            config={"server": {"local_dispatch_lease_timeout_seconds": 60}},
            local_active_queries=ghost_count_before,
            local_active_queries_lock=asyncio.Lock(),
            local_dispatch_records={
                "ghost-session-0": {
                    "backend": "local",
                    "started_at": time.monotonic(),
                    "active": False,
                    "expires_at": 0.0,
                },
                "ghost-session-1": {
                    "backend": "local",
                    "started_at": time.monotonic(),
                    "active": False,
                    "expires_at": 0.0,
                },
            },
            local_dispatch_records_lock=asyncio.Lock(),
            logger=MagicMock(),
        )

        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == ghost_count_before
        assert len(sess._slot_owners) == 0
        assert len(srv.local_dispatch_records) == 0

        # Pool is recovered: new sessions should get clean slots
        fresh0 = sess._slot_id_for_session("pool-recovered-0", pool)
        fresh1 = sess._slot_id_for_session("pool-recovered-1", pool)
        assert fresh0 in (0, 1)
        assert fresh1 in (0, 1)
        assert fresh0 != fresh1
        sess._slot_owners.clear()
