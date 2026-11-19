"""Invariant guards: the own-record predicate must not reach adjacent surfaces.

Parent LP-0MWMSQR3O002YNPW **AC7** ("No behaviour change elsewhere"); guard item
LP-0MWNHCJDS006IDGN.

The parent's desired change is a **read-side** refinement of the routing
concurrency predicate (``_is_own_idle_lease`` / ``_get_local_concurrency_info``,
LP-0MWNHBG7D002NH5S): the gate stops counting the requesting session's own
finished-but-not-yet-released record. It must grant no capacity and it must not
perturb anything else in the local admission stack. This module pins the four
adjacent surfaces the AC names, and it pins them as **read-only** invariants —
not just "the output happens to be right today" — so a future predicate change
that starts *writing* one of them fails immediately:

- **Prefill serialisation (AC1).** The serialised prefill gate is
  ``server.local_prefill_in_flight`` priced by
  ``_get_live_prefill_in_flight_count``. The routing read must not add, drop or
  rewrite a marker, so the gate keeps serialising exactly as today. The
  behavioural coverage lives in ``test_prefill_serialisation.py`` (AC1/AC2/AC5)
  and remains the evidence that a prefill-in-flight scenario still serialises;
  the guard here adds the read-only pin around it.
- **Admission-guard verdicts (AC2).** ``local_admission_guard decision=admit`` /
  ``decision=reject`` for the representative (free slot, no free slot, own
  lease) matrix are pinned through the real
  ``provider._prefill_admission_guard``. The guard's occupancy read
  (``_local_gate_occupancy`` -> ``_local_concurrency_info``) is deliberately
  **anonymous** — it forwards no ``session_id`` — so the own-record predicate
  cannot reach it. The test asserts that anonymity explicitly: threading the
  session into the guard's occupancy read would change verdicts and is caught.
- **Abandoned-slot registry (AC3).** ``router_helpers._abandoned_slots`` is the
  published ghost-slot registry. No routing-gate read may write it (entry count
  and keys unchanged, entry contents unchanged); the registry stays a
  dispatch/release-owned surface.
- **Cooldown scope (AC4).** ``routing_skip_local reason=abandoned_slot_drain``
  is a **per-session** deferral (``_abandoned_drain_remaining_for_session``).
  One session's abandonment never widens to another session, and the gate read
  never arms or clears a window. The prefill-thrash cooldown is pinned the same
  way (``_prefill_thrash_cooldown_remaining`` is per ``(session, fingerprint)``).

Every assertion holds on the pre-change tree *and* after the predicate change,
so none is ``xfail``-marked: a regression is red immediately. Harness idioms
(dispatch-record shape, the ``proxy.server`` stand-in) are reused from
``test_local_concurrency_own_record_guards.py`` and
``test_local_active_queries_invariant.py``.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import proxy.provider as provider
import proxy.router_helpers as helpers
import proxy.server as server
import pytest

from proxy import prefill_eta

ENDPOINT = "http://localhost:8080"
POOL = 2
ME = "session-me"
OTHER = "session-other"

#: Per-dispatch credit tokens (LP-0MVYZEYZY003IQAH) held by the records below.
ME_TOKEN = 9101
OTHER_TOKEN = 9111


# ---------------------------------------------------------------------------
# Config, records and the server stand-in
# ---------------------------------------------------------------------------


def _config(**server_overrides) -> dict:
    server_cfg = {
        "llama_server_port": 8080,
        "session_slot_pool_size": POOL,
        # Two sessions must be able to hold a prefill marker at once, otherwise
        # a fixture could be denied by the serialised-prefill cap rather than
        # by the surface under test.
        "local_max_concurrent_prefills": POOL,
        "local_dispatch_lease_timeout_seconds": 30,
        "max_concurrent_queries": 8,
        "local_prefill_deadline_seconds": 120,
        "local_cold_prefill_budget_tokens": 25000,
        "contention_queue_max_wait_seconds": 30,
    }
    server_cfg.update(server_overrides)
    return {"server": server_cfg, "provider_cooldown_seconds": 60}


def _record(
    session: str,
    *,
    credits: list[int],
    streaming: list[int] | None = None,
    active: bool = True,
    # Long enough that a slow test session cannot let an idle lease lapse
    # between parametrisation and execution (the fixture would then read as a
    # free slot, not the occupancy the matrix pins).
    expires_in: float = 600.0,
) -> dict:
    """A per-endpoint dispatch record in the production post-acquire shape.

    ``dispatch_credits`` holds one credit per in-flight dispatch of this
    session; ``streaming_dispatch_tokens`` is the subset whose stream has not
    emitted its terminal chunk. ``streaming=[]`` is the finished-but-not-yet-
    released shape the parent's predicate change targets; leave it ``None``
    for a record that predates both keys.
    """
    now = time.monotonic()
    record = {
        "backend": "local",
        "started_at": now,
        "active": active,
        "expires_at": now + expires_in,
        "model_name": "Qwen3",
        "endpoint": ENDPOINT,
        "session": session,
        "dispatch_token": credits[0] if credits else None,
        "dispatch_credits": list(credits),
    }
    if streaming is not None:
        record["streaming_dispatch_tokens"] = list(streaming)
    return record


def _install(
    monkeypatch,
    records: dict | None = None,
    *,
    generating: int = 0,
    prefill: dict | None = None,
) -> dict:
    """Install the dispatch records and counters the routing read touches."""
    records = records if records is not None else {}
    active = sum(
        1
        for rec in records.values()
        if isinstance(rec, dict) and rec.get("active")
    )
    monkeypatch.setattr(server, "local_dispatch_records", records, raising=False)
    monkeypatch.setattr(
        server, "local_dispatch_records_lock", asyncio.Lock(), raising=False
    )
    monkeypatch.setattr(
        server, "local_generating_queries", generating, raising=False
    )
    monkeypatch.setattr(
        server, "local_generating_queries_lock", asyncio.Lock(), raising=False
    )
    monkeypatch.setattr(server, "local_active_queries", active, raising=False)
    monkeypatch.setattr(
        server, "local_active_queries_lock", asyncio.Lock(), raising=False
    )
    monkeypatch.setattr(
        server, "local_prefill_in_flight", dict(prefill or {}), raising=False
    )
    monkeypatch.setattr(
        server, "local_prefill_in_flight_lock", asyncio.Lock(), raising=False
    )
    monkeypatch.setattr(server, "active_queries", 0, raising=False)
    monkeypatch.setattr(server, "config", _config(), raising=False)
    return records


def _lease_server(**server_overrides) -> SimpleNamespace:
    """Minimal real server state for the ``_local_dispatch_gate`` choke point."""
    return SimpleNamespace(
        config=_config(**server_overrides),
        logger=MagicMock(),
    )


@pytest.fixture(autouse=True)
def _clean_registries():
    """Start every row from empty abandoned-slot and thrash registries."""
    helpers._reset_abandoned_slots()
    helpers._reset_prefill_thrash_cooldowns()
    yield
    helpers._reset_abandoned_slots()
    helpers._reset_prefill_thrash_cooldowns()


@pytest.fixture(autouse=True)
def _deterministic_prefill_eta():
    """Start every row from the uncalibrated shipped prefill-ETA fit."""
    prefill_eta.reset()
    yield
    prefill_eta.reset()


# ---------------------------------------------------------------------------
# Read-only snapshot helper
# ---------------------------------------------------------------------------


def _snapshot() -> dict:
    """Capture every adjacent surface, with identity, so a write is caught.

    Identity matters: a predicate that replaces a record dict (``record =
    dict(record)``) or rebinds the registry would pass a content-only check but
    is still a write the adjacent surfaces must not see. The deep copies catch
    an in-place edit of a nested list (e.g. appending to ``dispatch_credits``).
    """
    records = getattr(server, "local_dispatch_records", None)
    if records is None:
        records = {}
    prefill = getattr(server, "local_prefill_in_flight", None)
    if prefill is None:
        prefill = {}
    # NOTE: read the attribute without a truthiness fallback — ``{} or {}``
    # would hand back a fresh empty dict and hide a rebind of the real one.
    return {
        "records_id": id(records),
        "records": copy.deepcopy(records),
        "record_ids": {key: id(rec) for key, rec in records.items()},
        "prefill_id": id(prefill),
        "prefill": copy.deepcopy(prefill),
        "generating": getattr(server, "local_generating_queries", None),
        "active": getattr(server, "local_active_queries", None),
        "abandoned_id": id(helpers._abandoned_slots),
        "abandoned": copy.deepcopy(helpers._abandoned_slots),
        "thrash_id": id(helpers._prefill_thrash_cooldowns),
        "thrash": dict(helpers._prefill_thrash_cooldowns),
        "cancel_id": id(helpers._prefill_thrash_cancel_tokens),
        "cancel": dict(helpers._prefill_thrash_cancel_tokens),
    }


def _assert_unchanged(before: dict, after: dict, surfaces: str) -> None:
    assert after["records_id"] == before["records_id"], (
        f"{surfaces}: the dispatch-record registry was rebound"
    )
    assert after["records"] == before["records"], (
        f"{surfaces}: the dispatch-record registry was written: "
        f"{before['records']!r} -> {after['records']!r}"
    )
    assert after["record_ids"] == before["record_ids"], (
        f"{surfaces}: a dispatch record was replaced instead of read"
    )
    assert after["prefill_id"] == before["prefill_id"], (
        f"{surfaces}: the prefill-in-flight registry was rebound"
    )
    assert after["prefill"] == before["prefill"], (
        f"{surfaces}: the prefill-in-flight registry was written: "
        f"{before['prefill']!r} -> {after['prefill']!r}"
    )
    assert after["generating"] == before["generating"], (
        f"{surfaces}: local_generating_queries changed "
        f"({before['generating']} -> {after['generating']})"
    )
    assert after["active"] == before["active"], (
        f"{surfaces}: local_active_queries changed "
        f"({before['active']} -> {after['active']})"
    )
    assert after["abandoned_id"] == before["abandoned_id"], (
        f"{surfaces}: the abandoned-slot registry was rebound"
    )
    assert after["abandoned"] == before["abandoned"], (
        f"{surfaces}: the abandoned-slot registry was written: "
        f"{before['abandoned']!r} -> {after['abandoned']!r}"
    )
    assert after["thrash_id"] == before["thrash_id"], (
        f"{surfaces}: the prefill-thrash cooldown store was rebound"
    )
    assert after["thrash"] == before["thrash"], (
        f"{surfaces}: the prefill-thrash cooldown store was written: "
        f"{before['thrash']!r} -> {after['thrash']!r}"
    )
    assert after["cancel_id"] == before["cancel_id"], (
        f"{surfaces}: the prefill-thrash cancel-token store was rebound"
    )
    assert after["cancel"] == before["cancel"], (
        f"{surfaces}: the prefill-thrash cancel-token store was written: "
        f"{before['cancel']!r} -> {after['cancel']!r}"
    )


# ======================================================================
# AC3 + AC5 — every routing-gate read is read-only on all adjacent surfaces
# ======================================================================


def test_routing_gate_reads_are_read_only_on_every_adjacent_surface(monkeypatch):
    """AC3/AC5: the gate read leaves prefill, registry and cooldowns untouched.

    Every surface the parent's change must not disturb is populated first
    (a finished-but-unreleased own record, another streaming record, a live
    prefill marker, an abandoned-slot entry and an armed thrash cooldown), then
    the routing-gate read family is exercised. The snapshot compares identity
    *and* content: a rebind, a dict replacement or an in-place list edit all
    fail here — that is the "assert read-only behaviour" requirement.
    """
    records = _install(
        monkeypatch,
        {
            (ENDPOINT, ME): _record(ME, credits=[ME_TOKEN], streaming=[]),
            (ENDPOINT, OTHER): _record(
                OTHER, credits=[OTHER_TOKEN], streaming=[OTHER_TOKEN]
            ),
        },
        generating=1,
        prefill={OTHER: {"started_at": time.monotonic()}},
    )
    helpers._record_abandoned_slot(
        1,
        session_id=ME,
        endpoint=ENDPOINT,
        reason="no_progress_watchdog",
        ttl=600.0,
    )
    helpers._record_prefill_thrash_cooldown(
        ME, _config(), tokens=60_000, fingerprint="fp-own"
    )

    before = _snapshot()

    # The routing-gate read family, exactly as the gate and its callers use it.
    config = _config()
    provider._local_concurrency_info(config, endpoint=ENDPOINT, session_id=ME)
    provider._get_local_concurrency_info(
        config, endpoint=ENDPOINT, session_id=ME
    )
    provider._get_local_concurrency_info(config, endpoint=ENDPOINT)
    provider._is_own_idle_lease(
        (ENDPOINT, ME), records[(ENDPOINT, ME)], ME
    )
    provider._local_gate_occupancy(config, POOL, endpoint=ENDPOINT)
    # The abandoned-slot / cooldown readers are part of the adjacent stack; a
    # predicate change must not route a write through them either.
    helpers._abandoned_drain_remaining_for_session(ME)
    helpers._prefill_thrash_cooldown_remaining(ME, config)

    _assert_unchanged(before, _snapshot(), "routing-gate read family")


def test_routing_gate_read_does_not_mutate_another_sessions_records(monkeypatch):
    """AC3/AC5: the session-scoped read never writes a *different* session's record.

    The own-record exclusion is a session-scoped read. Reading as ``ME`` must
    not prune, replace or edit ``OTHER``'s record, and reading as an anonymous
    caller must not either — the registry is owned by acquire/release, not by
    routing.
    """
    _install(
        monkeypatch,
        {
            (ENDPOINT, ME): _record(ME, credits=[ME_TOKEN], streaming=[]),
            (ENDPOINT, OTHER): _record(
                OTHER, credits=[OTHER_TOKEN], streaming=[OTHER_TOKEN]
            ),
        },
        generating=1,
    )
    before = _snapshot()

    provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    )
    provider._get_local_concurrency_info(_config(), endpoint=ENDPOINT)

    _assert_unchanged(before, _snapshot(), "own-session read")


# ======================================================================
# AC1 — prefill serialisation state is untouched by the gate read
# ======================================================================


def test_prefill_in_flight_serialisation_is_untouched_by_the_gate_read(
    monkeypatch,
):
    """AC1: a live prefill marker keeps serialising exactly as today.

    The serialised prefill gate is ``local_prefill_in_flight`` priced by
    ``_get_live_prefill_in_flight_count``. A prefill-in-flight scenario is set
    up (another session's live marker backed by its active dispatch record),
    then the gate read runs. The marker set, the live count and the occupancy
    the admission guard prices must all be unchanged — behaviour coverage of
    the serialisation itself remains ``test_prefill_serialisation.py``.
    """
    _install(
        monkeypatch,
        {
            (ENDPOINT, ME): _record(ME, credits=[ME_TOKEN], streaming=[]),
            (ENDPOINT, OTHER): _record(
                OTHER, credits=[OTHER_TOKEN], streaming=[OTHER_TOKEN]
            ),
        },
        prefill={OTHER: {"started_at": time.monotonic()}},
    )
    config = _config()

    assert helpers._get_live_prefill_in_flight_count(server) == 1, (
        "fixture: the other session's prefill must count as live work"
    )
    busy_before, in_flight_before = provider._local_gate_occupancy(
        config, POOL, endpoint=ENDPOINT
    )
    assert in_flight_before == 1, (
        "fixture: the admission guard must price one in-flight prefill"
    )
    before = _snapshot()

    provider._get_local_concurrency_info(
        config, endpoint=ENDPOINT, session_id=ME
    )
    provider._local_gate_occupancy(config, POOL, endpoint=ENDPOINT)

    assert helpers._get_live_prefill_in_flight_count(server) == 1, (
        "the routing-gate read changed the live prefill count — prefill "
        "serialisation would no longer match today"
    )
    busy_after, in_flight_after = provider._local_gate_occupancy(
        config, POOL, endpoint=ENDPOINT
    )
    assert (busy_after, in_flight_after) == (busy_before, in_flight_before), (
        "the gate read changed the occupancy the admission guard prices"
    )
    _assert_unchanged(before, _snapshot(), "prefill serialisation")


# ======================================================================
# AC3 — the abandoned-slot registry is untouched
# ======================================================================


def test_abandoned_slot_registry_is_untouched_by_the_gate_read(monkeypatch):
    """AC3: the ghost-slot registry keeps its entry count and keys.

    The registry is written only by the abandonment paths (via
    ``_record_abandoned_slot``) and cleared by observability / release
    (``_clear_abandoned_slot``). A routing-gate read must be a pure read: the
    same entries, keyed identically, with identical contents, before and after.
    """
    _install(
        monkeypatch,
        {(ENDPOINT, ME): _record(ME, credits=[ME_TOKEN], streaming=[])},
    )
    helpers._record_abandoned_slot(
        1,
        session_id=ME,
        endpoint=ENDPOINT,
        reason="prefill_deadline",
        ttl=600.0,
    )
    helpers._record_abandoned_slot(
        2,
        session_id=OTHER,
        endpoint=ENDPOINT,
        reason="client_disconnect",
        ttl=600.0,
    )
    keys_before = sorted(helpers._abandoned_slots)
    entries_before = copy.deepcopy(helpers._abandoned_slots)
    ids_before = {key: id(entry) for key, entry in helpers._abandoned_slots.items()}
    registry_id_before = id(helpers._abandoned_slots)

    provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    )
    provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=OTHER
    )
    provider._local_gate_occupancy(_config(), POOL, endpoint=ENDPOINT)

    assert id(helpers._abandoned_slots) == registry_id_before, (
        "the routing-gate read rebound the abandoned-slot registry"
    )
    assert sorted(helpers._abandoned_slots) == keys_before, (
        "the routing-gate read changed the registry keys: "
        f"{keys_before} -> {sorted(helpers._abandoned_slots)}"
    )
    assert helpers._abandoned_slots == entries_before, (
        "the routing-gate read wrote a registry entry: "
        f"{entries_before!r} -> {helpers._abandoned_slots!r}"
    )
    assert {
        key: id(entry) for key, entry in helpers._abandoned_slots.items()
    } == ids_before, (
        "the routing-gate read replaced a registry entry instead of reading it"
    )


# ======================================================================
# AC4 — the abandoned-slot drain cooldown stays per-session
# ======================================================================


def test_abandoned_slot_drain_cooldown_is_per_session_and_never_widened():
    """AC4: ``routing_skip_local reason=abandoned_slot_drain`` is per-session.

    The gate helper ``_local_dispatch_gate`` is the source of the
    ``abandoned_slot_drain`` refusal. A live abandonment owned by ``ME`` must
    defer ``ME`` (bounded by the recorded window) and must *not* defer
    ``OTHER`` — the cooldown is scoped to the session that abandoned the work,
    never widened to the pool.
    """
    srv = _lease_server()
    helpers._record_abandoned_slot(
        1,
        session_id=ME,
        endpoint=ENDPOINT,
        reason="no_progress_watchdog",
        ttl=600.0,
    )
    registry_before = copy.deepcopy(helpers._abandoned_slots)

    refused_me, reason_me, remaining_me = helpers._local_dispatch_gate(
        srv, ME, config=_config(), entry="test"
    )
    assert refused_me is True, (
        "the session that abandoned a slot must be deferred by its own drain"
    )
    assert reason_me == helpers.LOCAL_GATE_REASON_ABANDONED_DRAIN
    assert remaining_me > 0.0, (
        "the deferral must carry the recorded remaining drain window"
    )

    refused_other, reason_other, remaining_other = helpers._local_dispatch_gate(
        srv, OTHER, config=_config(), entry="test"
    )
    assert refused_other is False, (
        "another session must never inherit the abandonment's cooldown: "
        f"refused={refused_other} reason={reason_other!r}"
    )
    assert reason_other == ""
    assert remaining_other == 0.0

    # The per-session reader agrees, and neither read wrote the registry.
    assert helpers._abandoned_drain_remaining_for_session(ME) > 0.0
    assert helpers._abandoned_drain_remaining_for_session(OTHER) == 0.0
    assert helpers._abandoned_drain_remaining_for_session(None) == 0.0
    assert helpers._abandoned_slots == registry_before, (
        "drain reads must not write the abandoned-slot registry: "
        f"{registry_before!r} -> {helpers._abandoned_slots!r}"
    )


def test_abandoned_drain_does_not_widen_with_an_endpoint_wide_entry():
    """AC4: an endpoint-wide ghost (no real slot) does not defer a session.

    An abandonment with an unresolvable slot identity is stored endpoint-wide.
    It must not defer any session — that would be an unbounded over-refusal,
    strictly wider than today's per-session drain.
    """
    srv = _lease_server()
    helpers._record_abandoned_slot(
        None,
        session_id=ME,
        endpoint=ENDPOINT,
        reason="client_disconnect",
        ttl=600.0,
    )
    assert helpers._abandoned_drain_remaining_for_session(ME) == 0.0

    refused, reason, remaining = helpers._local_dispatch_gate(
        srv, ME, config=_config(), entry="test"
    )
    assert (refused, reason, remaining) == (False, "", 0.0), (
        "an endpoint-wide ghost must not defer any session: "
        f"{(refused, reason, remaining)!r}"
    )


def test_prefill_thrash_cooldown_stays_per_session(monkeypatch):
    """AC4: the prefill-thrash cooldown is scoped, never widened by a gate read.

    The cooldown store is keyed by ``(session, fingerprint)``. An armed window
    for ``ME`` must not defer ``OTHER``, and a routing-gate read must not arm a
    window for anyone.
    """
    srv = _lease_server()
    config = _config()
    helpers._record_prefill_thrash_cooldown(
        ME, config, tokens=60_000, fingerprint="fp-own"
    )
    before = dict(helpers._prefill_thrash_cooldowns)

    # The gate read (routing) must not touch the cooldown store.
    _install(monkeypatch, {})
    provider._get_local_concurrency_info(config, endpoint=ENDPOINT, session_id=ME)
    provider._local_gate_occupancy(config, POOL, endpoint=ENDPOINT)
    assert helpers._prefill_thrash_cooldowns == before, (
        "the routing-gate read wrote the prefill-thrash cooldown store: "
        f"{before!r} -> {helpers._prefill_thrash_cooldowns!r}"
    )

    refused_me, reason_me, remaining_me = helpers._local_dispatch_gate(
        srv, ME, config=config, fingerprint="fp-own", entry="test"
    )
    assert refused_me is True
    assert reason_me == helpers.LOCAL_GATE_REASON_PREFILL_THRASH_COOLDOWN
    assert remaining_me > 0.0

    refused_other, reason_other, _remaining = helpers._local_dispatch_gate(
        srv, OTHER, config=config, fingerprint="fp-own", entry="test"
    )
    assert refused_other is False, (
        "another session must not inherit an armed thrash cooldown: "
        f"refused={refused_other} reason={reason_other!r}"
    )


# ======================================================================
# AC2 — admission-guard verdict matrix is unchanged
# ======================================================================


def _seed_calibration(tokens: int = 100_000, ms_per_token: float = 3.9) -> None:
    """Install the calibrated ``(Qwen3, ~tokens)`` prefill fit the guard uses."""
    for _ in range(6):
        prefill_eta.record_dispatch_sample(
            "Qwen3", tokens, 0.0, tokens * ms_per_token
        )
        prefill_eta.record_cold_measured_sample(
            "Qwen3", tokens, 0.0, tokens, tokens * ms_per_token
        )


async def _guard_decision(
    monkeypatch, records: dict, *, session_id: str | None = ME
):
    """Run the real ``_prefill_admission_guard`` for one matrix row.

    The occupancy the guard prices comes from ``_local_gate_occupancy``, i.e.
    the anonymous routing read. ``expected_cached_ratio`` and the occupant
    service are pinned so the only variable is the occupancy the matrix sets
    up.
    """
    _install(monkeypatch, records)
    config = _config()
    monkeypatch.setattr(
        provider, "expected_cached_ratio", lambda *a, **k: 0.0
    )
    monkeypatch.setattr(
        provider, "_local_occupant_prefill_seconds", lambda *a, **k: 0.0
    )
    return await provider._prefill_admission_guard(
        config,
        {"llama_model": "Qwen3"},
        6_000,
        session_id,
        "medium",
        messages=[{"role": "user", "content": "hi"}],
        max_local=POOL,
        endpoint=ENDPOINT,
    )


def _decision_verdict(decision) -> str:
    """The verdict ``record_guard_decision`` would log for *decision*."""
    if decision is None:
        return "inert"
    if not decision.active:
        return "admit"
    return "reject" if not decision.admit else "admit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario, records, expected_routing, expected_guard_occupancy, expected_verdict",
    [
        (
            "free_slot",
            {},
            0,
            0,
            "admit",
        ),
        (
            "no_free_slot",
            {
                (ENDPOINT, OTHER): _record(
                    OTHER, credits=[OTHER_TOKEN], streaming=[OTHER_TOKEN]
                )
            },
            1,
            1,
            "reject",
        ),
        (
            "own_lease",
            {
                (ENDPOINT, ME): _record(
                    ME, credits=[ME_TOKEN], active=False
                )
            },
            0,
            1,
            "reject",
        ),
    ],
)
async def test_admission_guard_verdicts_unchanged_across_the_matrix(
    monkeypatch,
    scenario,
    records,
    expected_routing,
    expected_guard_occupancy,
    expected_verdict,
    caplog,
):
    """AC2: the (free slot, no free slot, own lease) verdicts are pinned.

    The routing-gate view and the admission-guard view are deliberately
    different: the routing read excludes the requester's own idle lease, while
    the admission guard prices the gate **anonymously** (no ``session_id``) and
    therefore still sees it. That is today's behaviour and it must not move
    when the own-record predicate changes.

    The logged verdict (``local_admission_guard decision=...``) is asserted
    from the real guard so the AC's literal outcome is pinned, not just the
    boolean.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration(tokens=6_000, ms_per_token=3.9)

    # Install the scenario's records *before* reading occupancy; otherwise the
    # reads would describe the previous scenario's (or the real) registry.
    decision = await _guard_decision(monkeypatch, records)
    routing_occupancy, _max = provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    )
    busy_slots, in_flight = provider._local_gate_occupancy(
        _config(), POOL, endpoint=ENDPOINT
    )

    assert routing_occupancy == expected_routing, (
        f"[{scenario}] routing-gate occupancy moved: "
        f"{routing_occupancy} != {expected_routing}"
    )
    assert (busy_slots, in_flight) == (expected_guard_occupancy, 0), (
        f"[{scenario}] admission-guard occupancy moved: "
        f"{(busy_slots, in_flight)!r} != "
        f"{(expected_guard_occupancy, 0)!r}"
    )
    assert _decision_verdict(decision) == expected_verdict, (
        f"[{scenario}] admission-guard verdict moved: "
        f"{_decision_verdict(decision)!r} != {expected_verdict!r} "
        f"(decision={decision!r})"
    )

    guard_lines = [
        rec.getMessage()
        for rec in caplog.records
        if "local_admission_guard decision=" in rec.getMessage()
    ]
    assert guard_lines, (
        f"[{scenario}] the guard must emit its decision line"
    )
    assert guard_lines[-1].startswith(
        f"local_admission_guard decision={expected_verdict}"
    ), (
        f"[{scenario}] the logged verdict moved: {guard_lines[-1]!r}"
    )


def test_admission_guard_occupancy_read_is_anonymous(monkeypatch):
    """AC2: the guard's occupancy read forwards no ``session_id``.

    This is the mechanism that keeps admission-guard verdicts independent of
    the routing gate's own-record predicate. If a future change threads the
    session into the guard's occupancy read, the verdict matrix above would
    move — this row names the cause directly.
    """
    _install(monkeypatch, {})
    seen_kwargs: list[dict] = []
    real = provider._local_concurrency_info

    def _spy(config, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return real(config, **kwargs)

    monkeypatch.setattr(provider, "_local_concurrency_info", _spy)
    provider._local_gate_occupancy(_config(), POOL, endpoint=ENDPOINT)

    assert seen_kwargs, "the guard must read occupancy through the helper"
    assert all("session_id" not in kwargs for kwargs in seen_kwargs), (
        "the admission-guard occupancy read must stay anonymous so the "
        f"own-record predicate cannot change its verdicts: {seen_kwargs}"
    )
