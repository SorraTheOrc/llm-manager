"""Price the diagnosed blocker class in the admission guard (LP-0MVV29YDQ00437N6).

Evidence: [`docs/dev/queue-wait-abandonment-diagnosis.md`] — the per-wait
diagnosis (`LP-0MVV29XPT003OXI5`). It measured the guard's occupancy term at
exactly the symmetric ``eta_prefill_s x busy_slots`` floor in **163 of 169**
abandoned waits that had an admission decision, with a median projection of
**28.4 s** against a **60 s** observed wait. The dominant blocker class was
``prefill_in_flight`` (**65.3 %**); ``lease_holder_generating`` was a **13.4 %**
minority whose fairness capping stays owned by ``LP-0MVOJ6KAW004EBV0``. The
worked case (2026-11-04 19:45, session ``herdr-1793821499-2528952-25402``) shows
the shape directly: the guard admitted at ``projected_wait_s=27.0`` against a
``budget_s=30.0`` while the slot's holder had ``expires_in=30.0`` on its live
dispatch lease, and the wait then expired at the full budget.

This file pins the behaviour fix:

- **AC1** — a blocker whose measured remaining hold exceeds the remaining queue
  budget refuses the queue admission instead of burning the full budget; a
  short-prefill blocker still admits with exactly today's decision.
- **AC2** — the term only *raises* the symmetric ``eta * busy_slots`` floor
  (never lowers it), prices only the diagnosed prefilling blocker class, and
  fails open to today's floor on missing/invalid occupant evidence (no guessed
  multiplier, no new configuration knob).
- **AC3** — ``high`` priority still never queues, even with a measured hold.
- **AC5** — the measured term is attributable on the ``local_admission_guard``
  log line.

The shipped guardrails (`contention_queue_max_wait_seconds`,
`local_prefill_deadline_seconds`, the ``< 840 < 900`` ordering) are not touched
by this change and stay pinned by
``test_prefill_budget_config.py::test_shipped_config_passes_the_ordering_validator``
and ``::test_ordering_holds_for_the_documented_values``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from proxy.router_helpers import _live_prefill_remaining_seconds

from proxy import prefill_eta

_MODEL = "Qwen3"
#: The newcomer's prompt: 4 000 uncached tokens at the seeded 2.5 ms/token is a
#: ~12 s own-prefill term at the estimator's conservatism factor.
_NEWCOMER_TOKENS = 4_000
_BUCKET0_SEED_TOKENS = 8_000
_MS_PER_TOKEN = 2.5


def _floor(busy_slots: int = 1):
    """The symmetric ``eta * busy_slots``-only projection (today's verdict)."""
    return prefill_eta.projected_wait_seconds(
        _MODEL, _NEWCOMER_TOKENS, 0.0, busy_slots, _cfg()
    )


def _cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 2,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 30,
        "contention_queue_max_depth": 2,
        "local_prefill_deadline_seconds": 180,
    }
    server.update(server_overrides)
    return {"provider_cooldown_seconds": 60, "server": server}


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _seed(tokens: int = _BUCKET0_SEED_TOKENS) -> None:
    """Seed a calibrated window for *tokens*' context bucket (solitary +
    contended), so both the own-prefill and the occupant terms are priced."""
    for _ in range(6):
        prefill_eta.record_dispatch_sample(
            _MODEL, tokens, 0.0, tokens * _MS_PER_TOKEN
        )
        prefill_eta.record_dispatch_sample(
            _MODEL, tokens, 0.0, tokens * _MS_PER_TOKEN, contended=True
        )


def _occupant(
    *,
    new_tokens=None,
    eta=100.0,
    started_ago: float = 0.0,
    lease_away: float | None = None,
    first_byte: bool = False,
    active: bool = True,
    endpoint: str | None = None,
    model: str | None = _MODEL,
) -> dict:
    """A live dispatch record shaped like the proxy's ``_dispatch_evidence``.

    *lease_away* is the record's remaining dispatch-lease hold in seconds
    (``expires_at`` is monotonic, like the proxy's own records).
    """
    record = {
        "active": active,
        "first_byte_seen": first_byte,
        "eta_prefill_s": eta,
        "started_at": time.monotonic() - started_ago,
        "new_tokens": new_tokens,
        "cached_ratio": 0.0,
        "model_name": model,
        "endpoint": endpoint,
        "backend": endpoint or "http://192.0.2.9:8080",
    }
    if lease_away is not None:
        record["expires_at"] = time.monotonic() + lease_away
    return record


def _srv(records: dict) -> SimpleNamespace:
    # No ``config`` attribute → the estimator's default guard settings.
    return SimpleNamespace(local_dispatch_records=records)


def _admit(occupant_hold: float, *, busy_slots: int = 1, **kwargs):
    return prefill_eta.evaluate_admission(
        _cfg(),
        _MODEL,
        _NEWCOMER_TOKENS,
        0.0,
        "low",
        busy_slots=busy_slots,
        in_flight_prefills=busy_slots,
        occupant_prefill_seconds=occupant_hold,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# AC1(a) — a blocker outlasting the remaining budget refuses, it does not queue
# ---------------------------------------------------------------------------

def test_lease_hold_beyond_the_remaining_budget_refuses_the_queue_admission():
    """AC1(a): the holder's prefill service is unpriceable (no ``new_tokens``)
    but its dispatch lease runs 45 s — longer than the whole 30 s queue budget,
    so the wait cannot be served and must be refused at admission.
    """
    _seed()
    floor = _floor()
    assert floor.projected_seconds < 30.0
    assert floor.measured_occupancy_seconds == 0.0

    hold = _live_prefill_remaining_seconds(
        _srv({"occ": _occupant(new_tokens=None, lease_away=45.0)})
    )
    assert hold == pytest.approx(45.0, abs=1.0)

    # Today's behaviour (the symmetric floor alone) still admits this wait.
    floor_only = _admit(0.0)
    assert floor_only.admit is True
    assert floor_only.projected_wait_seconds == pytest.approx(
        floor.projected_seconds
    )

    refused = _admit(hold)

    assert refused.admit is False
    assert refused.reason == "eta_exceeds_budget"
    # The measured hold is what the projection is priced with, not the floor.
    assert refused.projected_wait_seconds == pytest.approx(
        refused.eta_prefill_seconds + hold, abs=1.5
    )
    assert refused.projected_wait_seconds > 30.0
    assert refused.occupant_hold_seconds == pytest.approx(hold, abs=1.0)


def test_lease_hold_that_fits_the_budget_still_admits():
    """AC1(a) counter-case: a blocker that *can* be outlasted is not refused —
    the change refuses only what the budget genuinely cannot cover."""
    _seed()
    floor = _floor()
    hold = _live_prefill_remaining_seconds(
        _srv({"occ": _occupant(new_tokens=None, lease_away=5.0)})
    )
    assert hold == pytest.approx(5.0, abs=1.0)

    decision = _admit(hold)

    assert decision.admit is True
    assert decision.reason == "within_budget"
    # The measured hold is below the symmetric floor, so the floor keeps the
    # decision (and its projection) exactly as today.
    assert decision.projected_wait_seconds == pytest.approx(
        floor.projected_seconds
    )
    assert decision.eta_occupancy_seconds == pytest.approx(
        floor.occupancy_seconds
    )


def test_short_prefill_blocker_admits_with_the_same_decision_as_today():
    """AC1(b): a short measurable prefill whose lease is shorter still prices
    exactly as it does today (no over-refusal from the new term)."""
    _seed()
    # A ~2 000-token prefill still owes several seconds at the seeded rate, so
    # the 1 s lease is the smaller of the two terms.
    with_lease = _live_prefill_remaining_seconds(
        _srv({"occ": _occupant(new_tokens=2_000, lease_away=1.0)})
    )
    without_lease = _live_prefill_remaining_seconds(
        _srv({"occ": _occupant(new_tokens=2_000)})
    )

    assert without_lease > 1.0
    assert with_lease == pytest.approx(without_lease, abs=0.05)

    today = _admit(without_lease)
    after = _admit(with_lease)

    assert after.admit is True
    assert after.reason == today.reason
    assert after.projected_wait_seconds == today.projected_wait_seconds


# ---------------------------------------------------------------------------
# AC2 — only raises the floor; fail-open; only the diagnosed blocker class
# ---------------------------------------------------------------------------

def test_missing_or_invalid_occupant_evidence_fails_open_to_the_floor():
    """AC2: no usable occupant evidence ⇒ today's symmetric floor, byte-for-byte.

    Records that are inactive, already generating (a first byte was served), or
    whose lease has already expired contribute nothing — the ``prefill_in_flight``
    class is the one the diagnosis measured, and the generating minority stays
    with ``LP-0MVOJ6KAW004EBV0``.
    """
    _seed()
    records = {
        "no-evidence": _occupant(new_tokens=None),
        "bad-expiry": dict(_occupant(new_tokens=None), expires_at="soon"),
        "none-expiry": dict(_occupant(new_tokens=None), expires_at=None),
        "inactive": _occupant(new_tokens=None, lease_away=45.0, active=False),
        "generating": _occupant(new_tokens=None, lease_away=45.0, first_byte=True),
        "expired": _occupant(new_tokens=None, lease_away=-5.0),
        "junk": "not-a-record",
    }

    assert _live_prefill_remaining_seconds(_srv(records)) == 0.0

    floor = _floor()
    decision = _admit(0.0)
    assert decision.admit is True
    assert decision.eta_occupancy_seconds == pytest.approx(
        floor.occupancy_seconds
    )
    assert decision.occupant_hold_seconds == 0.0


@pytest.mark.parametrize("hold", [0.0, 2.0, 9.9, 10.0, 25.0, 400.0])
def test_measured_hold_only_raises_the_symmetric_floor(hold: float):
    """AC2: ``occupancy = max(eta * busy_slots, measured hold)`` — the floor is
    never lowered, whatever the measurement says."""
    _seed()
    floor = _floor()
    measured = prefill_eta.projected_wait_seconds(
        _MODEL, _NEWCOMER_TOKENS, 0.0, 1, _cfg(), occupant_prefill_seconds=hold
    )

    assert measured.occupancy_seconds >= floor.occupancy_seconds
    assert measured.measured_occupancy_seconds == pytest.approx(hold)
    assert measured.projected_seconds == pytest.approx(
        measured.own_prefill_seconds + max(floor.occupancy_seconds, hold)
    )


def test_no_usable_evidence_never_guesses_a_multiplier():
    """AC2: with neither a priceable prefill nor a lease, the term is 0.0 — the
    guard's founding rule is that it never prices from a guess."""
    # No calibration seeded at all, and a record with no lease either.
    assert _live_prefill_remaining_seconds(
        _srv({"occ": _occupant(new_tokens=60_000)})
    ) == 0.0
    assert _live_prefill_remaining_seconds(_srv({})) == 0.0


# ---------------------------------------------------------------------------
# AC3 — the new term cannot make a never-queueing priority queue
# ---------------------------------------------------------------------------

def test_high_priority_never_queues_even_with_a_measured_hold():
    """AC3: ``high`` has no queue budget, so the queue comparator is skipped —
    the measured hold can only move the deadline bound, never a queue wait."""
    _seed()
    decision = prefill_eta.evaluate_admission(
        _cfg(),
        _MODEL,
        _NEWCOMER_TOKENS,
        0.0,
        "high",
        busy_slots=1,
        in_flight_prefills=1,
        occupant_prefill_seconds=45.0,
    )

    assert decision.budget_seconds == 0.0
    # The bound actually enforced is the deadline, not the queue window.
    assert decision.bound_seconds == 180.0
    assert decision.admit is True


# ---------------------------------------------------------------------------
# AC5 — the measured term is attributable on the decision log line
# ---------------------------------------------------------------------------

def _guard_line(caplog) -> str:
    return next(
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("local_admission_guard")
    )


def test_decision_log_carries_the_measured_occupant_hold(caplog):
    """AC5: the post-fix window can attribute a refusal to the measured hold."""
    caplog.set_level("INFO", logger="llama-proxy")
    _seed()

    prefill_eta.record_guard_decision(
        _admit(45.0),
        provider="local-llama",
        session_id="s-1",
        model=_MODEL,
        priority="low",
    )

    line = _guard_line(caplog)
    assert "occupant_hold_s=45.0" in line
    # The pre-existing fields keep their meaning, so old parses still work.
    assert "eta_occupancy_s=" in line
    assert "projected_wait_s=" in line


def test_decision_log_reports_a_zero_hold_on_the_fail_open_path(caplog):
    """AC5: a fail-open decision stays attributable — ``occupant_hold_s=0.0``
    means "no usable occupant evidence", never "the occupant owed nothing"."""
    caplog.set_level("INFO", logger="llama-proxy")
    _seed()

    prefill_eta.record_guard_decision(
        _admit(0.0),
        provider="local-llama",
        session_id="s-2",
        model=_MODEL,
        priority="low",
    )

    assert "occupant_hold_s=0.0" in _guard_line(caplog)
