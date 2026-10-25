"""Post-acceptance eviction is an ERROR (LP-0MVN4LCED00661MX, parent AC7).

An accepted local dispatch (admitted, dispatch record created) that is released
**before producing a first byte** is a defect signal — the admission/pricing
model admitted something the GPU could not serve — so it must be recorded at
ERROR with machine-parseable evidence and counted. A release *after* the first
byte is normal churn and must stay quiet; a refusal *before* acceptance is
normal operation and must not produce the event at all.

AC mapping: AC1 one ERROR event per eviction with the full evidence set and the
six-reason vocabulary; AC2 complete evidence (unknown → ``unknown``/``0``, never
omitted); AC3 classification by ``first_byte_seen`` (not by path); AC4 per-path
tests + inverse refusal test; AC5 counter with per-reason breakdown and
``/admin/metrics`` exposure; AC6 the legacy ``lease_released`` line is
untouched; AC7 triage-ready evidence.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from proxy import admission_events

AFTER_ACCEPTANCE_REASONS = (
    "prefill_deadline",
    "no_progress_watchdog",
    "idle_timeout",
    "orphan_cleanup",
    "lease_evicted",
    "slot_assignment_freed",
)


@pytest.fixture(autouse=True)
def _reset_counters():
    admission_events.reset_eviction_after_acceptance_counters()
    yield
    admission_events.reset_eviction_after_acceptance_counters()


class _Srv:
    """Minimal server stub exposing the attributes the paths touch."""

    def __init__(self):
        self.logger = logging.getLogger("llama-proxy")
        self.local_dispatch_records = {}
        self.local_dispatch_records_lock = asyncio.Lock()
        self.local_prefill_in_flight = {}
        self.local_active_queries = 1


def _accepted_record(**overrides) -> dict:
    """An accepted, unserved dispatch record with full evidence."""
    record = {
        "backend": "local",
        "endpoint": "local",
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


def _eviction_events(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if "local_eviction_after_acceptance" in r.getMessage()
    ]


# ---------------------------------------------------------------------------
# AC1 / AC2 / AC7 — one ERROR event with the full evidence set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reason", AFTER_ACCEPTANCE_REASONS)
def test_each_reason_emits_exactly_one_error_event(reason, caplog):
    srv = _Srv()
    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        emitted = admission_events.record_eviction_after_acceptance(
            srv, "session-1", reason, _accepted_record(), endpoint="local"
        )

    assert emitted is True
    events = _eviction_events(caplog)
    assert len(events) == 1, "exactly one event per eviction"
    record = events[0]
    assert record.levelno == logging.ERROR, "post-acceptance eviction is an ERROR"
    message = record.getMessage()
    for field in (
        "session=session-1",
        f"reason={reason}",
        "estimated_tokens=40000",
        "new_tokens=40000",
        "eta_prefill_s=148.1",
        "elapsed_s=",
        "slot_id=1",
        "llama_task_id=unknown",
    ):
        assert field in message, f"missing evidence field {field!r} in {message!r}"


def test_all_six_reasons_are_in_the_vocabulary():
    assert set(AFTER_ACCEPTANCE_REASONS) <= set(
        admission_events.EVICTION_REASONS_AFTER_ACCEPTANCE
    )
    assert set(AFTER_ACCEPTANCE_REASONS) <= set(admission_events.EVICTION_REASONS)


def test_unknown_evidence_is_emitted_explicitly_never_omitted(caplog):
    record = _accepted_record(
        estimated_tokens=None,
        new_tokens=None,
        eta_prefill_s=None,
        slot_id=None,
        llama_task_id=None,
    )
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        admission_events.record_eviction_after_acceptance(
            _Srv(), "session-x", "prefill_deadline", record
        )

    message = _eviction_events(caplog)[0].getMessage()
    assert "estimated_tokens=0" in message
    assert "new_tokens=0" in message
    assert "eta_prefill_s=unknown" in message
    assert "slot_id=unknown" in message
    assert "llama_task_id=unknown" in message


def test_event_is_not_emitted_at_warning_or_info(caplog):
    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        admission_events.record_eviction_after_acceptance(
            _Srv(), "session-2", "idle_timeout", _accepted_record()
        )

    non_error = [
        r
        for r in caplog.records
        if "local_eviction_after_acceptance" in r.getMessage()
        and r.levelno != logging.ERROR
    ]
    assert non_error == []


# ---------------------------------------------------------------------------
# AC3 — classification by first_byte_seen, and idempotence
# ---------------------------------------------------------------------------

def test_served_dispatch_is_not_an_eviction_error(caplog):
    """A lease released AFTER the first byte is normal churn."""
    record = _accepted_record(first_byte_seen=True)
    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        emitted = admission_events.record_eviction_after_acceptance(
            _Srv(), "session-3", "idle_timeout", record
        )

    assert emitted is False
    assert _eviction_events(caplog) == []
    assert admission_events.eviction_after_acceptance_counters()["total"] == 0


def test_the_event_is_idempotent_per_record(caplog):
    """Two release paths racing the same record emit one event."""
    record = _accepted_record()
    srv = _Srv()
    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        first = admission_events.record_eviction_after_acceptance(
            srv, "session-4", "prefill_deadline", record
        )
        second = admission_events.record_eviction_after_acceptance(
            srv, "session-4", "orphan_cleanup", record
        )

    assert (first, second) == (True, False)
    assert len(_eviction_events(caplog)) == 1
    assert admission_events.eviction_after_acceptance_counters()["total"] == 1


# ---------------------------------------------------------------------------
# AC4 inverse — a pre-acceptance refusal is NOT an error
# ---------------------------------------------------------------------------

def test_pre_acceptance_refusal_emits_nothing(caplog):
    """A budget/ETA refusal never created a record, so it cannot be evicted.

    The refusal path is exercised through the real admission guard; the event
    and the counter must both stay untouched (no false alarms).
    """
    from proxy import prefill_eta

    prefill_eta.reset()
    config = {
        "server": {
            "contention_queue_max_wait_seconds": 60,
            "local_prefill_deadline_seconds": 120,
            "local_cold_prefill_budget_tokens": 25_000,
            "local_admission_guard_policy": "reject",
        }
    }
    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        decision = prefill_eta.evaluate_admission(
            config, "Qwen3", 40_000, 0.0, priority="medium"
        )
        prefill_eta.record_guard_decision(decision, session_id="session-5")

    assert decision.admit is False
    assert decision.reason == "prefill_budget_exceeded"
    assert _eviction_events(caplog) == []
    assert admission_events.eviction_after_acceptance_counters()["total"] == 0


# ---------------------------------------------------------------------------
# AC5 — monotonic counter, per-reason breakdown, admin exposure
# ---------------------------------------------------------------------------

def test_counter_increments_once_per_eviction_with_a_reason_breakdown():
    srv = _Srv()
    for _ in range(3):
        admission_events.record_eviction_after_acceptance(
            srv, "s", "prefill_deadline", _accepted_record()
        )
    admission_events.record_eviction_after_acceptance(
        srv, "s", "orphan_cleanup", _accepted_record()
    )

    counters = admission_events.eviction_after_acceptance_counters()

    assert counters["total"] == 4
    assert counters["by_reason"] == {"prefill_deadline": 3, "orphan_cleanup": 1}


def test_prometheus_counter_is_incremented_per_reason(monkeypatch):
    """AC5: the emitter drives the Prometheus counter, labelled by reason.

    Asserted behaviourally against a spy rather than by scraping
    ``generate_latest()``: this repo ships a local ``prometheus_client`` test
    shim, so the rendered text is not the production registry.
    """
    from proxy import metrics

    calls: list[dict] = []

    class _Label:
        def inc(self):
            calls.append({"inc": 1})

    class _Counter:
        def labels(self, **kwargs):
            calls.append(kwargs)
            return _Label()

    monkeypatch.setattr(metrics, "_enabled", True, raising=False)
    monkeypatch.setattr(
        metrics,
        "llama_local_evictions_after_acceptance_total",
        _Counter(),
        raising=False,
    )

    admission_events.record_eviction_after_acceptance(
        _Srv(), "s", "no_progress_watchdog", _accepted_record()
    )

    assert {"reason": "no_progress_watchdog"} in calls
    assert {"inc": 1} in calls


def test_admin_metrics_exposes_the_counter():
    """AC5: the JSON payload carries the total and the per-reason breakdown."""
    from proxy import handlers

    admission_events.record_eviction_after_acceptance(
        _Srv(), "s", "idle_timeout", _accepted_record()
    )

    block = handlers._local_eviction_metrics()

    assert block["total"] == 1
    assert block["by_reason"] == {"idle_timeout": 1}


# ---------------------------------------------------------------------------
# AC6 — the legacy log line keeps its name and level
# ---------------------------------------------------------------------------

def test_legacy_lease_released_line_is_unchanged(caplog):
    """The pre-change parser contract: ``lease_released … reason=…`` still INFO."""
    from proxy import router_helpers

    srv = _Srv()
    srv.local_dispatch_records[router_helpers._dispatch_lease_key(None, "sess")] = (
        _accepted_record()
    )

    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        removed = asyncio.run(
            router_helpers._release_prefill_deadline_dispatch(srv, "sess")
        )

    assert removed is True
    messages = [r.getMessage() for r in caplog.records]
    legacy = [m for m in messages if m.startswith("lease_released")]
    assert any("reason=prefill_deadline" in m for m in legacy)
    legacy_records = [
        r for r in caplog.records if r.getMessage().startswith("lease_released")
    ]
    assert all(r.levelno <= logging.INFO for r in legacy_records), (
        "existing lease_released lines must keep their level for log consumers"
    )
    # The new structured ERROR is additive alongside it.
    evictions = _eviction_events(caplog)
    assert len(evictions) == 1
    assert evictions[0].levelno == logging.ERROR


# ---------------------------------------------------------------------------
# AC4 — per-path wiring (the real release functions, not the emitter)
# ---------------------------------------------------------------------------

def test_prefill_deadline_path_reports_the_eviction(caplog):
    from proxy import router_helpers

    srv = _Srv()
    srv.local_dispatch_records[router_helpers._dispatch_lease_key(None, "sess-pd")] = (
        _accepted_record()
    )

    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        asyncio.run(router_helpers._release_prefill_deadline_dispatch(srv, "sess-pd"))

    events = _eviction_events(caplog)
    assert len(events) == 1
    assert "reason=prefill_deadline" in events[0].getMessage()
    assert (
        admission_events.eviction_after_acceptance_counters()["by_reason"][
            "prefill_deadline"
        ]
        == 1
    )


def test_client_disconnect_path_reports_lease_evicted(caplog):
    from proxy import router_helpers

    srv = _Srv()
    srv.local_dispatch_records[router_helpers._dispatch_lease_key(None, "sess-dc")] = (
        _accepted_record()
    )

    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        router_helpers._release_client_disconnect_dispatch(srv, "sess-dc")

    events = _eviction_events(caplog)
    assert len(events) == 1
    assert "reason=lease_evicted" in events[0].getMessage()


def test_idle_timeout_path_reports_only_unserved_records(caplog):
    """The idle sweep classifies by the flag: served records stay quiet."""
    from proxy import router_helpers

    srv = _Srv()
    key = router_helpers._dispatch_lease_key("local", "sess-idle")
    srv.local_dispatch_records[key] = _accepted_record(
        active=False, expires_at=0.0, first_byte_seen=False
    )
    key_served = router_helpers._dispatch_lease_key("local", "sess-served")
    srv.local_dispatch_records[key_served] = _accepted_record(
        active=False, expires_at=0.0, first_byte_seen=True
    )

    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        asyncio.run(router_helpers._cleanup_stale_local_dispatch(srv))

    events = _eviction_events(caplog)
    assert len(events) == 1, "only the unserved record is an error"
    assert "reason=idle_timeout" in events[0].getMessage()
    assert "session=sess-idle" in events[0].getMessage()


def test_first_byte_flag_suppresses_a_later_release(caplog):
    """AC3 end-to-end: marking the first byte silences the eviction error."""
    from proxy import router_helpers

    srv = _Srv()
    key = router_helpers._dispatch_lease_key(null_endpoint := None, "sess-fb")
    srv.local_dispatch_records[router_helpers._dispatch_lease_key("local", "sess-fb")] = (
        _accepted_record()
    )
    router_helpers.mark_local_first_byte(srv, "sess-fb", endpoint="local")

    record = srv.local_dispatch_records[
        router_helpers._dispatch_lease_key("local", "sess-fb")
    ]
    assert record["first_byte_seen"] is True
    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        admission_events.record_eviction_after_acceptance(
            srv, "sess-fb", "prefill_deadline", record
        )
    assert _eviction_events(caplog) == []
    assert key is not None and null_endpoint is None


def test_dispatch_records_carry_the_evidence_at_creation():
    """AC2: the evidence is captured once, at acceptance."""
    from proxy import router_helpers

    evidence = router_helpers._dispatch_evidence(
        {"messages": [{"role": "user", "content": "hello world " * 100}]},
        "Qwen3",
        "sess-evidence",
    )

    assert evidence["estimated_tokens"] > 0
    assert 0.0 <= evidence["cached_ratio"] <= 1.0
    assert evidence["new_tokens"] <= evidence["estimated_tokens"]
    assert evidence["first_byte_seen"] is False
    assert "slot_id" in evidence and "llama_task_id" in evidence
