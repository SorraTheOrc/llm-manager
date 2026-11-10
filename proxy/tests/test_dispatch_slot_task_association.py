"""Request ↔ llama slot/task association on dispatched local requests.

LP-0MW959RRO008AAH9 (parent LP-0MW7V29MA009MCPI AC10, merged from
LP-0MW0KZUUS002LICZ). A dispatched local request must be attributable to the
backend task it actually used: the 2026-11-03 eviction evidence carried
``slot_id=unknown llama_task_id=unknown``, so progress, cancellation and
eviction could not be mapped onto a llama slot/task at all.

Contract pinned here:

- **AC10a** — the dispatch record persists the slot id assigned at admission
  (the candidate, LP-0MW959P9S005DJY7) and the id appears on the admission
  decision line and on the eviction line.
- **AC10b** — the llama task id is read from the *already-cached* ``/slots``
  snapshot (``id_task``, which the deployed build publishes). No upstream call
  is made on this path: the cache is the only source.
- **AC10c** — a record with a known slot never logs a bare ``unknown/unknown``:
  an unavailable task id is reported with an explicit reason
  (``llama_task_id_reason=``) instead.
- **AC10d** — admit → dispatch → eviction carries the ids, and a refused
  admission records no association and no lease.
- **AC10e** — every pre-existing log key keeps its position; new values are
  appended.

Two measured facts shape the design and are pinned below. An *idle*
llama-server slot keeps the ``id_task`` of the last task it ran (measured
2026-10-31, see ``proxy/metrics.py``), so an idle slot's id must never be
attributed to a new dispatch; and at record-creation time (immediately before
the request is sent) the slot is usually not yet processing, so the id is
resolved again at release, by which time the slot is running this request.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import httpx
import proxy.observability as observability
import proxy.provider as provider
import proxy.router_helpers as router_helpers
import proxy.session as session_registry
import pytest

from proxy import admission_events

#: The slot pool used by the association fixtures (two slots, as deployed).
POOL_SIZE = 2
#: The llama task id the "deployed build" publishes for a processing slot.
ID_TASK = 42_424
SESSION = "sess-assoc"


@pytest.fixture(autouse=True)
def _isolated_state():
    """Reset the two documented sources between tests (cache + registry)."""
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    session_registry._slot_last_tenant.clear()
    admission_events.reset_eviction_after_acceptance_counters()
    yield
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    session_registry._slot_last_tenant.clear()
    admission_events.reset_eviction_after_acceptance_counters()


def _slot(
    slot_id: int,
    *,
    processing: bool = True,
    id_task: int | None = ID_TASK,
    context_tokens: int | None = 1_000,
) -> dict:
    """One cached ``/slots`` entry in the shape ``_query_slots_detail`` emits."""
    entry = {
        "slot_id": slot_id,
        "is_processing": processing,
        "n_decoded": None,
        "context_tokens": context_tokens,
    }
    entry["id_task"] = id_task
    return entry


def _snapshot(*slots: dict, age: float = 0.5) -> None:
    """Install a cached ``/slots`` snapshot (the only association source)."""
    observability._last_slot_details_cache[:] = [dict(s) for s in slots]
    observability._last_slot_details_at = time.monotonic() - age


class _RecordingClient:
    """Minimal httpx client stand-in for ``_query_slots_detail``."""

    def __init__(self, payload: list):
        self._payload = payload
        self.urls: list[str] = []

    async def get(self, url):
        self.urls.append(url)
        return httpx.Response(200, json=self._payload)


def _srv() -> SimpleNamespace:
    """Minimal server exposing the dispatch-lease state the paths touch."""
    return SimpleNamespace(
        config={"server": {"local_dispatch_lease_timeout_seconds": 10}},
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        local_prefill_in_flight={},
        logger=logging.getLogger("llama-proxy"),
    )


# ---------------------------------------------------------------------------
# AC10b — id_task is carried by the cached snapshot (source of the id)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slots_snapshot_carries_id_task():
    """``_query_slots_detail`` carries the published ``id_task`` additively."""
    payload = [
        {"id": 0, "is_processing": True, "id_task": ID_TASK},
        {"id": 1, "is_processing": False, "id_task": -1},
    ]

    result = await observability._query_slots_detail(
        8080, model="Qwen3", _client=_RecordingClient(payload)
    )

    assert [entry["id_task"] for entry in result] == [ID_TASK, -1]
    # AC10e: the pre-existing keys keep their positions.
    assert list(result[0])[:4] == [
        "slot_id",
        "is_processing",
        "n_decoded",
        "context_tokens",
    ]


@pytest.mark.asyncio
async def test_slots_snapshot_without_id_task_reports_none():
    """A build that publishes no ``id_task`` yields ``None``, never a guess."""
    result = await observability._query_slots_detail(
        8080, model="Qwen3", _client=_RecordingClient([{"id": 0, "is_processing": True}])
    )

    assert result[0]["id_task"] is None


@pytest.mark.asyncio
async def test_slot_snapshot_query_is_the_only_source(monkeypatch):
    """Resolving the id never issues an upstream ``/slots`` request (AC10b)."""

    async def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("association must not query /slots")

    monkeypatch.setattr(observability, "_fetch_slots_payload", _boom)
    _snapshot(_slot(0))

    task_id, reason = observability.slot_task_id_from_cache(0)

    assert (task_id, reason) == (ID_TASK, None)


# ---------------------------------------------------------------------------
# AC10b/AC10c — the resolver's closed vocabulary
# ---------------------------------------------------------------------------


def test_resolver_reads_the_processing_slots_task_id():
    _snapshot(_slot(0), _slot(1, id_task=ID_TASK + 1))

    assert observability.slot_task_id_from_cache(1) == (ID_TASK + 1, None)


def test_resolver_never_attributes_an_idle_slots_stale_task_id():
    """An idle slot keeps the previous task's id: it is not this dispatch's."""
    _snapshot(_slot(0, processing=False, id_task=19_999))

    task_id, reason = observability.slot_task_id_from_cache(0)

    assert task_id is None
    assert reason == observability.SLOT_TASK_REASON_NOT_STARTED


def test_resolver_reports_a_sentinel_id_as_not_started():
    _snapshot(_slot(0, id_task=-1))

    assert observability.slot_task_id_from_cache(0) == (
        None,
        observability.SLOT_TASK_REASON_NOT_STARTED,
    )


def test_resolver_reports_an_unpublished_id_task():
    _snapshot(_slot(0, id_task=None))

    assert observability.slot_task_id_from_cache(0) == (
        None,
        observability.SLOT_TASK_REASON_UNPUBLISHED,
    )


def test_resolver_reports_an_unsampled_slot():
    _snapshot(_slot(1))

    assert observability.slot_task_id_from_cache(0) == (
        None,
        observability.SLOT_TASK_REASON_NOT_SAMPLED,
    )


def test_resolver_reports_a_missing_snapshot_and_a_missing_slot():
    assert observability.slot_task_id_from_cache(0) == (
        None,
        observability.SLOT_TASK_REASON_NO_SNAPSHOT,
    )
    assert observability.slot_task_id_from_cache(None) == (
        None,
        observability.SLOT_TASK_REASON_NO_SLOT,
    )


# ---------------------------------------------------------------------------
# AC10a — the admitted dispatch record persists the candidate slot + task id
# ---------------------------------------------------------------------------


def test_dispatch_evidence_stores_the_admission_slot_and_resolved_task_id():
    """The candidate slot from the admission snapshot replaces the None marker."""
    _snapshot(_slot(1))
    snapshot = {
        "estimated_tokens": 41_682,
        "cached_ratio": 0.55,
        "new_tokens": 18_034,
        "eta_prefill_s": 86.0,
        "slot_id": 1,
    }

    evidence = router_helpers._dispatch_evidence(
        {"messages": [{"role": "user", "content": "hi"}]},
        "Qwen3",
        SESSION,
        admission_evidence=snapshot,
    )

    assert evidence["slot_id"] == 1
    assert evidence["llama_task_id"] == ID_TASK
    # Pre-existing pricing evidence is untouched (AC10e).
    assert evidence["new_tokens"] == 18_034
    assert evidence["eta_prefill_s"] == 86.0


def test_dispatch_evidence_keeps_none_markers_without_a_snapshot():
    """A record created outside the guarded path keeps its None fallback."""
    _snapshot(_slot(0))

    evidence = router_helpers._dispatch_evidence(
        {"messages": [{"role": "user", "content": "hi"}]}, "Qwen3", SESSION
    )

    assert evidence["slot_id"] is None
    assert evidence["llama_task_id"] is None


# ---------------------------------------------------------------------------
# AC10d — admit → dispatch → eviction carries the ids
# ---------------------------------------------------------------------------


def test_candidate_assignment_stamps_the_admission_evidence(tmp_path):
    """The candidate assigned at admission is what the snapshot carries."""
    session_registry._slot_owners[("", 0)] = SESSION
    server_config = {
        "session_slot_pool_size": POOL_SIZE,
        "session_slot_save_path": str(tmp_path),
    }
    candidate = provider._acquire_admission_candidate(
        SESSION, endpoint=None, server_config=server_config
    )
    request = SimpleNamespace()

    provider._stamp_local_admission_evidence(
        request,
        estimated_tokens=41_682,
        cached_ratio=0.55,
        new_tokens=18_034,
        slot_id=candidate.slot_id,
    )

    assert request._local_admission_evidence["slot_id"] == candidate.slot_id == 0


@pytest.mark.asyncio
async def test_admit_dispatch_eviction_carries_slot_and_task_ids(caplog):
    """End to end: the eviction of an admitted dispatch names its task (AC10d).

    The dispatch record is created while the slot is still idle (the request
    has not reached llama-server yet), so the task id is resolved at release
    time — the sequence a real request goes through.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")
    srv = _srv()
    session_registry._slot_owners[("", 0)] = SESSION
    # Admitted, not yet started: the slot is sampled idle with a stale id.
    _snapshot(_slot(0, processing=False, id_task=19_999))

    acquired, _owner, _active, _retry = await router_helpers._try_acquire_local_dispatch(
        srv,
        max_local=POOL_SIZE,
        session_key=SESSION,
        backend="local",
        admission_evidence={"estimated_tokens": 41_682, "slot_id": 0},
    )
    assert acquired is True
    record = srv.local_dispatch_records[SESSION]
    assert record["slot_id"] == 0
    assert record["llama_task_id"] is None
    assert record["llama_task_id_reason"] == observability.SLOT_TASK_REASON_NOT_STARTED

    # llama-server starts the task on slot 0 (the deployed build publishes it).
    _snapshot(_slot(0, processing=True, id_task=ID_TASK))

    emitted = admission_events.record_eviction_after_acceptance(
        srv, SESSION, admission_events.EVICTION_PREFILL_DEADLINE, record
    )
    assert emitted is True
    lines = [
        r.getMessage()
        for r in caplog.records
        if "local_eviction_after_acceptance" in r.getMessage()
    ]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "slot_id=0" in line
    assert f"llama_task_id={ID_TASK}" in line
    assert "llama_task_id_reason=" not in line
    assert "slot_id=unknown" not in line
    # The association the line reported is persisted on the record.
    assert record["llama_task_id"] == ID_TASK


@pytest.mark.asyncio
async def test_eviction_reports_the_slot_plus_a_reason_when_no_task_id(caplog):
    """AC10c: a known slot never logs a bare ``unknown/unknown``."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    srv = _srv()
    record = {
        "backend": "local",
        "endpoint": "local",
        "active": True,
        "started_at": time.monotonic(),
        "expires_at": time.monotonic() + 60.0,
        "estimated_tokens": 1_000,
        "new_tokens": 1_000,
        "eta_prefill_s": 10.0,
        "slot_id": 1,
        "llama_task_id": None,
        "first_byte_seen": False,
    }

    admission_events.record_eviction_after_acceptance(
        srv, SESSION, admission_events.EVICTION_PREFILL_DEADLINE, record
    )

    line = next(
        r.getMessage()
        for r in caplog.records
        if "local_eviction_after_acceptance" in r.getMessage()
    )
    assert "slot_id=1" in line
    assert f"llama_task_id_reason={observability.SLOT_TASK_REASON_NO_SNAPSHOT}" in line
    assert "slot_id=unknown llama_task_id=unknown" not in line


def test_eviction_line_is_append_only(caplog):
    """AC10e: every pre-existing eviction key keeps its position."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    srv = _srv()
    record = {
        "backend": "local",
        "endpoint": "local",
        "active": True,
        "started_at": time.monotonic(),
        "expires_at": time.monotonic() + 60.0,
        "estimated_tokens": 1_000,
        "new_tokens": 1_000,
        "eta_prefill_s": 10.0,
        "slot_id": 0,
        "llama_task_id": None,
        "first_byte_seen": False,
    }

    admission_events.record_eviction_after_acceptance(
        srv, SESSION, admission_events.EVICTION_PREFILL_DEADLINE, record
    )

    line = next(
        r.getMessage()
        for r in caplog.records
        if "local_eviction_after_acceptance" in r.getMessage()
    )
    assert line.startswith("local_eviction_after_acceptance session=")
    keys = [part.split("=", 1)[0] for part in line.split() if "=" in part]
    assert keys[:9] == [
        "session",
        "reason",
        "estimated_tokens",
        "new_tokens",
        "eta_prefill_s",
        "elapsed_s",
        "slot_id",
        "llama_task_id",
        "endpoint",
    ]
    # Everything the association adds comes after the pre-existing keys.
    assert keys[9:] == ["llama_task_id_reason"]


# ---------------------------------------------------------------------------
# AC10a — the admission decision line carries the slot candidate
# ---------------------------------------------------------------------------


def _admission_line(caplog) -> str:
    return next(
        r.getMessage()
        for r in caplog.records
        if "routing_resident_prefix" in r.getMessage()
    )


def _admit_through_the_ladder(caplog, tmp_path, *, owner: str, slot: dict):
    """Run the admission ladder with a candidate available (measured resident)."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    from proxy.residency import get_residency_view

    session_registry._slot_owners[("", slot["slot_id"])] = owner
    _snapshot(slot)
    server_config = {
        "session_slot_pool_size": POOL_SIZE,
        "session_slot_save_path": str(tmp_path),
    }
    skip, reason = provider._should_skip_local(
        "Qwen3",
        owner,
        {"messages": []},
        estimated_tokens=slot.get("context_tokens") or 1_000,
        cold_cache_threshold=1_000,
        resident_prefix_min_ratio=0.5,
        residency_view=get_residency_view(),
        server_config=server_config,
    )
    return skip, reason


def test_admission_line_carries_the_slot_candidate(caplog, tmp_path):
    """AC10a: the decision line names the candidate slot the verdict measured."""
    skip, reason = _admit_through_the_ladder(
        caplog, tmp_path, owner=SESSION, slot=_slot(0, context_tokens=50_000)
    )

    assert (skip, reason) == (False, None)
    line = _admission_line(caplog)
    assert "slot_candidate=0" in line


def test_admission_line_appends_slot_candidate_last(caplog, tmp_path):
    """AC10e: ``slot_candidate=`` is appended after every pre-existing key.

    The priced ``new_tokens=`` this group appends last
    (LP-0MW959SE30012HJJ AC6b) follows it, so ``slot_candidate`` remains a
    suffix field that no deployed parser reads positionally.
    """
    _admit_through_the_ladder(
        caplog, tmp_path, owner=SESSION, slot=_slot(0, context_tokens=50_000)
    )

    line = _admission_line(caplog)
    keys = [part.split("=", 1)[0] for part in line.split() if "=" in part]
    assert keys[-2] == "slot_candidate"
    assert line.startswith("routing_resident_prefix provider=local model=")
    assert keys[:3] == ["provider", "model", "decision"]
    assert "measurement" in keys and "residency_source" in keys
    # Every pre-existing key still precedes the appended fields.
    assert keys[-3] == "admission_overhead_ms"


def test_refused_admission_records_no_association_and_no_lease(caplog, tmp_path):
    """AC10d: a refused admission associates nothing and takes no lease.

    The candidate the refusal was measured against is released (no fabricated
    occupancy), no dispatch record is created, and the refusal line still names
    the candidate so the refusal is attributable without claiming an
    association the request never had.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")
    from proxy.residency import get_residency_view

    srv = _srv()
    # The session's own slot holds almost none of this prompt: the measured
    # resident-prefix rung refuses (the 2026-10-24 shape).
    session_registry._slot_owners[("", 0)] = SESSION
    _snapshot(_slot(0, context_tokens=738))
    server_config = {
        "session_slot_pool_size": POOL_SIZE,
        "session_slot_save_path": str(tmp_path),
    }

    skip, reason = provider._should_skip_local(
        "Qwen3",
        SESSION,
        {"messages": []},
        estimated_tokens=49_103,
        cold_cache_threshold=25_000,
        resident_prefix_min_ratio=0.5,
        residency_view=get_residency_view(),
        server_config=server_config,
    )

    assert skip is True
    assert reason == "resident_prefix_insufficient"
    # The refused admission's candidate was released: no fabricated occupancy.
    assert session_registry._slot_owners == {}
    # No dispatch record (and therefore no lease) exists for the refused request.
    assert srv.local_dispatch_records == {}
    line = _admission_line(caplog)
    assert "decision=resident_prefix_insufficient" in line
    assert "slot_candidate=0" in line
    assert "measurement=measured" in line
