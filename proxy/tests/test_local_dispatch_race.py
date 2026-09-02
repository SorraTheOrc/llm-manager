"""Hermetic tests for the TOCTOU race in ``_try_acquire_local_dispatch``.

See LP-0MS8ZM98R000M8AN.

The race: ``_try_acquire_local_dispatch`` counts ``occupied_by_others`` from
``local_dispatch_records`` while holding only ``local_dispatch_records_lock``,
then (nested) checks ``local_active_queries >= max_local`` under
``local_active_queries_lock``. A concurrent anonymous-session
``_increment_local_active_queries`` (which takes only
``local_active_queries_lock``) can land between the two checks, bumping
``local_active_queries`` and falsely denying an explicit session that
legitimately had a free slot (false 503 ``no_slots_available``).

All tests are hermetic: they exercise the helpers directly on a throwaway
fake server object (fresh ``asyncio.Lock()``s, fresh ``local_active_queries``
counter, fresh ``local_dispatch_records`` dict) inside the pytest process.
No live proxy, no real backend, no shared files or sockets.
"""

import asyncio
from types import SimpleNamespace

import pytest


def _make_server() -> SimpleNamespace:
    """Fresh fake server with 1 of N=2 slots occupied by ``sess-owner``.

    ``local_active_queries`` is 1 (the owner's query is active) and the
    owner holds an unexpired active dispatch record.
    """
    return SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=1,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={
            "sess-owner": {
                "backend": "local",
                "started_at": 1.0,
                "active": True,
                "expires_at": 10**12,
            }
        },
        local_dispatch_records_lock=asyncio.Lock(),
    )


@pytest.mark.asyncio
async def test_explicit_session_not_falsely_denied_when_anonymous_increment_lands_between_checks():
    """Deterministic TOCTOU repro (fails pre-fix, passes post-fix).

    Holds ``local_dispatch_records_lock`` so the explicit session blocks at
    its records-lock acquisition. While blocked:

    - Unfixed code: the explicit session does NOT yet hold
      ``local_active_queries_lock``, so an anonymous
      ``_increment_local_active_queries`` task acquires it and bumps
      ``local_active_queries`` 1 -> 2. When the explicit session finally
      runs, its records-count (1 of N=2 occupied, passes) then its
      counter-check (2 >= 2, denies) produce a false 503
      (``acquired=False``).

    - Fixed code: the explicit session acquires ``local_active_queries_lock``
      FIRST, so while it waits for the records lock it already blocks all
      counter mutators; the anonymous increment cannot land between the
      records-count and the counter-check, and the explicit session is
      granted.
    """
    from proxy.router_helpers import (
        _increment_local_active_queries,
        _try_acquire_local_dispatch,
    )

    srv = _make_server()

    # Hold the records lock so the explicit session blocks at its first
    # lock acquisition.
    await srv.local_dispatch_records_lock.acquire()

    explicit_task = asyncio.create_task(
        _try_acquire_local_dispatch(
            srv,
            max_local=2,
            session_key="sess-explicit",
            backend="local",
        )
    )
    # Let the explicit session run to its blocking point.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Anonymous-session increment (counter-only, no dispatch record).
    anon_task = asyncio.create_task(_increment_local_active_queries(srv))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Release the records lock; the explicit session proceeds.
    srv.local_dispatch_records_lock.release()

    acquired, owner, active_count, retry_after = await explicit_task
    await anon_task

    assert acquired is True, (
        "explicit session with a free slot was falsely denied "
        f"(owner={owner} active={active_count} retry={retry_after})"
    )


@pytest.mark.asyncio
async def test_stress_concurrent_anonymous_increments_no_false_503():
    """Stress-amplification loop: N=2, 1 slot occupied, concurrent
    anonymous-session increments -> zero false 503s.

    Each round builds a fresh fake server with 1 of 2 slots occupied,
    holds the records lock, launches the explicit acquisition alongside
    several anonymous ``_increment_local_active_queries`` tasks, releases
    the records lock, and asserts the explicit session is granted.
    Post-fix this holds over repeated runs; pre-fix the TOCTOU window
    produces false 503s (``acquired=False``).
    """
    from proxy.router_helpers import (
        _increment_local_active_queries,
        _try_acquire_local_dispatch,
    )

    rounds = 25
    anonymous_increments = 8

    for _ in range(rounds):
        srv = _make_server()

        await srv.local_dispatch_records_lock.acquire()

        explicit_task = asyncio.create_task(
            _try_acquire_local_dispatch(
                srv,
                max_local=2,
                session_key="sess-explicit",
                backend="local",
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        anon_tasks = [
            asyncio.create_task(_increment_local_active_queries(srv))
            for _ in range(anonymous_increments)
        ]
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        srv.local_dispatch_records_lock.release()

        acquired, owner, active_count, _ = await explicit_task
        await asyncio.gather(*anon_tasks)

        assert acquired is True, (
            "false 503 in round: explicit session denied "
            f"(owner={owner} active={active_count})"
        )


@pytest.mark.asyncio
async def test_counter_based_deny_still_applies_when_slots_full():
    """Post-fix sanity: when ``local_active_queries`` already reaches
    ``max_local`` (e.g. anonymous queries consume all capacity), a new
    explicit session is still denied — the counter gate is preserved.
    """
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 180}},
        local_active_queries=2,  # capacity reached (legacy)
        local_active_queries_lock=asyncio.Lock(),
        # Generating-only pool (LP-0MTH7JX82000YS5N): counter gate is
        # local_generating_queries, not local_active_queries.
        local_generating_queries=2,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions={"sess-a", "sess-b"},
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
    )

    acquired, owner, active_count, retry_after = await _try_acquire_local_dispatch(
        srv,
        max_local=2,
        session_key="sess-new",
        backend="local",
    )

    assert acquired is False
    assert active_count == 2
    assert retry_after >= 1
