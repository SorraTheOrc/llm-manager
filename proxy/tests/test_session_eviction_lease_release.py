"""Session-eviction lease release and post-acceptance reporting (LP-0MVO7IDTP005S92D).

Measured on 2026-10-25 (7.4 h window) 94 of 116 ``local_eviction_after_acceptance``
ERRORs were emitted on the ``Session ... invalidated, creating new`` /
``lease_released reason=session_evicted`` path with ``estimated_tokens=0
new_tokens=0 eta_prefill_s=unknown elapsed_s=0.0`` — the exact default shape of a
missing dispatch record. Root cause: the callback looked the record up under the
legacy string key only (``_dispatch_lease_key(None, session_id)``) while production
records are keyed by ``(endpoint, session_id)``, so the lookup returned ``None`` and
the emitter fired with no evidence; the same miss meant the per-endpoint lease was
never released either.

AC mapping: AC1 per-endpoint records released on session eviction; AC2 no event and
no counter increment when the session has no dispatch record; AC3 an accepted,
unserved record still yields exactly one ERROR with its evidence; AC4 a served
dispatch stays quiet; AC5 at most one event per record; AC6 existing INFO
``lease_released reason=session_evicted`` line is unchanged.
"""

import asyncio
import logging

import pytest

from proxy import admission_events
from proxy import server as proxy_server

_EP = "http://localhost:8080"


def _accepted_record(endpoint: str | None = _EP, **overrides) -> dict:
    """An accepted, unserved dispatch record with full evidence."""
    record = {
        "backend": endpoint,
        "endpoint": endpoint,
        "started_at": 1_000.0,
        "active": True,
        "expires_at": 1_000.0 + 60.0,
        "model_name": "Qwen3",
        "estimated_tokens": 40_000,
        "cached_ratio": 0.0,
        "new_tokens": 40_000,
        "eta_prefill_s": 148.1,
        "slot_id": 1,
        "llama_task_id": None,
        "first_byte_seen": False,
    }
    record.update(overrides)
    return record


@pytest.fixture(autouse=True)
def _reset_state():
    admission_events.reset_eviction_after_acceptance_counters()
    from proxy.session import _slot_owners

    _slot_owners.clear()
    yield
    admission_events.reset_eviction_after_acceptance_counters()
    _slot_owners.clear()


def _monkeypatch_server(monkeypatch, records: dict, session_id: str):
    """Point the real ``proxy.server`` module at a fresh dispatch-records dict."""
    monkeypatch.setattr(proxy_server, "local_dispatch_records", records)
    monkeypatch.setattr(proxy_server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(proxy_server, "local_prefill_in_flight", {})
    monkeypatch.setattr(proxy_server, "local_active_queries", 1)
    monkeypatch.setattr(proxy_server, "local_active_queries_lock", asyncio.Lock())
    # The callback logs through proxy.server.logger.
    monkeypatch.setattr(proxy_server, "logger", logging.getLogger("llama-proxy"))


def _eviction_events(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if "local_eviction_after_acceptance" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_tuple_keyed_unserved_record_is_released_and_reported(monkeypatch, caplog):
    """AC1/AC3: a per-endpoint accepted-unserved record is released and reported
    exactly once with the record's evidence (this was silently missed by the
    legacy string-key lookup)."""
    records = {(_EP, "sess-1"): _accepted_record()}
    _monkeypatch_server(monkeypatch, records, "sess-1")

    with caplog.at_level(logging.ERROR, logger="llama-proxy"):
        await proxy_server._release_lease_on_session_eviction("sess-1")

    assert records == {}, "per-endpoint dispatch record released by the callback"
    events = _eviction_events(caplog)
    assert len(events) == 1, "exactly one after-acceptance event"
    assert events[0].levelno == logging.ERROR
    message = events[0].getMessage()
    assert "session=sess-1" in message
    assert "reason=slot_assignment_freed" in message
    assert "estimated_tokens=40000" in message
    assert "new_tokens=40000" in message
    assert "eta_prefill_s=148.1" in message
    assert "elapsed_s=" in message
    total = admission_events.eviction_after_acceptance_counters()
    assert total["total"] == 1
    assert total["by_reason"] == {"slot_assignment_freed": 1}


@pytest.mark.asyncio
async def test_tuple_keyed_record_frees_slot_assignment(monkeypatch):
    """AC1: the per-endpoint slot assignment is freed, not just the record."""
    from proxy.session import _slot_owners

    records = {(_EP, "sess-1"): _accepted_record()}
    _slot_owners[(_EP, 0)] = "sess-1"
    _monkeypatch_server(monkeypatch, records, "sess-1")

    await proxy_server._release_lease_on_session_eviction("sess-1")

    assert records == {}
    assert "sess-1" not in list(_slot_owners.values())


@pytest.mark.asyncio
async def test_session_without_dispatch_record_emits_no_event(monkeypatch, caplog):
    """AC2: session churn without an accepted dispatch is not an error — the
    post-acceptance ERROR and the counter must stay silent even though the
    session is evicted (replays the measured 94 false positives)."""
    records: dict = {}
    _monkeypatch_server(monkeypatch, records, "sess-1")

    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        for i in range(3):
            await proxy_server._release_lease_on_session_eviction(f"sess-{i + 1}")

    assert _eviction_events(caplog) == [], "no ERROR for record-less evictions"
    total = admission_events.eviction_after_acceptance_counters()
    assert total["total"] == 0


@pytest.mark.asyncio
async def test_served_dispatch_stays_quiet_on_eviction(monkeypatch, caplog):
    """AC4: a dispatch that already produced a first byte is normal churn —
    released, but not reported and not counted."""
    records = {(_EP, "sess-1"): _accepted_record(first_byte_seen=True)}
    _monkeypatch_server(monkeypatch, records, "sess-1")

    with caplog.at_level(logging.ERROR, logger="llama-proxy"):
        await proxy_server._release_lease_on_session_eviction("sess-1")

    assert records == {}, "served record is still released"
    assert _eviction_events(caplog) == []
    assert admission_events.eviction_after_acceptance_counters()["total"] == 0


@pytest.mark.asyncio
async def test_legacy_string_keyed_record_still_released_and_reported(monkeypatch, caplog):
    """AC1 backward-compat: the legacy bare-session-id record path is unchanged."""
    legacy = _accepted_record(endpoint="local")
    records = {"sess-1": legacy}
    _monkeypatch_server(monkeypatch, records, "sess-1")

    with caplog.at_level(logging.ERROR, logger="llama-proxy"):
        await proxy_server._release_lease_on_session_eviction("sess-1")

    assert records == {}
    assert len(_eviction_events(caplog)) == 1
    assert admission_events.eviction_after_acceptance_counters()["total"] == 1


@pytest.mark.asyncio
async def test_release_info_line_unchanged(monkeypatch, caplog):
    """AC6: the INFO ``lease_released reason=session_evicted`` line is emitted
    for every eviction regardless of whether a dispatch record existed."""
    records = {(_EP, "sess-1"): _accepted_record()}
    _monkeypatch_server(monkeypatch, records, "sess-1")

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        await proxy_server._release_lease_on_session_eviction("sess-1")

    released = [
        r
        for r in caplog.records
        if "lease_released" in r.getMessage() and "session_evicted" in r.getMessage()
    ]
    assert len(released) == 1


@pytest.mark.asyncio
async def test_second_eviction_of_same_session_is_silent(monkeypatch, caplog):
    """AC5: a repeat eviction for an already-released session adds nothing."""
    records = {(_EP, "sess-1"): _accepted_record()}
    _monkeypatch_server(monkeypatch, records, "sess-1")

    with caplog.at_level(logging.ERROR, logger="llama-proxy"):
        await proxy_server._release_lease_on_session_eviction("sess-1")
        after_first = len(_eviction_events(caplog))
        await proxy_server._release_lease_on_session_eviction("sess-1")

    assert after_first == 1, "the first eviction is reported"
    assert len(_eviction_events(caplog)) == 1, "the repeat eviction adds nothing"
    assert admission_events.eviction_after_acceptance_counters()["total"] == 1
