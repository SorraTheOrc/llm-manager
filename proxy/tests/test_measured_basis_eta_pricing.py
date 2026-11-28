"""Measured-basis prefill-ETA pricing (LP-0MWRQ6LSV009DVP4, parent AC1/AC6).

The admission guard historically priced its ETA from the **predicted** cache
ratio (``expected_cached_ratio``) even when a **measured** resident prefix was
available on the same admission path and discarded — the defect behind the
863.4 % ``prefill_eta_accuracy`` median error in the 2026-11-21 window.

The fix (parent AC1) prices ``new_tokens`` from
``min(n_past_actual, estimated_tokens)`` whenever a measured source exists
(``measured`` / ``measured_last_dispatch``); ``session_slot`` (size unknown)
and every unmeasured/absent/stale view keep the predicted ratio. The single
``server.local_eta_pricing_basis: predicted`` value restores the pre-change
numbering byte-for-byte (parent AC6).

These tests pin the guard's *priced ratio* only — the stamped dispatch record
is owned by ``test_local_admission_stamp_consistency.py``.
"""

from __future__ import annotations

import time

import proxy.measured_prefill as measured_prefill
import proxy.observability as observability
import proxy.provider as provider
import proxy.session as session_registry
import pytest
from proxy import prefill_eta
from proxy.measured_prefill import MeasuredPrefill

MODEL = "Qwen3"
SESSION = "sess-a"
# The recorded 2026-11-03 case: a 55 492-token prompt with 30 000 tokens of
# measured resident prefix. The prediction prices it as ~172 new tokens; the
# measurement is 25 492.
EST_TOKENS = 55_492
N_PAST = 30_000
PREDICTED_RATIO = 172 / EST_TOKENS  # ~0.9969
MEASURED_NEW = EST_TOKENS - N_PAST  # 25 492


# ---------------------------------------------------------------------------
# Residency fixtures — stub only the documented in-process sources
# ---------------------------------------------------------------------------

def _install_slot_state(
    *,
    slots: list[dict],
    owners: dict[tuple, str],
    age: float = 0.5,
) -> None:
    observability._last_slot_details_cache[:] = [dict(s) for s in slots]
    observability._last_slot_details_at = time.monotonic() - age
    session_registry._slot_owners.clear()
    session_registry._slot_owners.update(owners)


def _slot(slot_id: int, context_tokens: int | None, *, processing: bool = True) -> dict:
    return {
        "slot_id": slot_id,
        "is_processing": processing,
        "n_decoded": None,
        "context_tokens": context_tokens,
    }


@pytest.fixture(autouse=True)
def _reset_residency_state():
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    measured_prefill.reset_measured_prefill()
    prefill_eta.reset()
    yield
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    measured_prefill.reset_measured_prefill()
    prefill_eta.reset()


def _cfg(**overrides) -> dict:
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
        # Isolate the budget bound; the deadline bound is covered elsewhere.
        "local_prefill_deadline_seconds": 0,
    }
    server.update(overrides)
    return {"server": server}


def _measured_view(
    *,
    slots=None,
    owners=None,
    age: float = 0.5,
):
    from proxy.residency import get_residency_view

    _install_slot_state(
        slots=slots if slots is not None else [_slot(0, N_PAST)],
        owners=owners if owners is not None else {("", 0): SESSION},
        age=age,
    )
    return get_residency_view()


def _resolve(view, *, session_id=SESSION, estimated=EST_TOKENS):
    """Resolve facts through the provider helper under test."""
    return provider._resolve_eta_residency(
        MODEL, session_id, estimated, view=view, server_config=_cfg()["server"]
    )


def _seed_cold_window(tokens: int = EST_TOKENS, ms_per_token: float = 3.9) -> None:
    """Seed the genuinely-cold MEASURED window for *tokens*' bucket."""
    for _ in range(8):
        prefill_eta.record_cold_measured_sample(
            MODEL,
            tokens,
            measured_cached_ratio=0.0,
            processed_tokens=tokens,
            service_ms=tokens * ms_per_token,
        )


# ---------------------------------------------------------------------------
# AC1 — a measured resident prefix prices the ETA
# ---------------------------------------------------------------------------

def test_measured_warm_prices_from_measurement_not_prediction():
    """``source=measured`` prices new_tokens from min(n_past, estimated)."""
    view = _measured_view()
    measured_n_past, facts = _resolve(view)
    assert facts.source == provider.RESIDENCY_SOURCE_MEASURED
    assert measured_n_past == N_PAST

    measured = prefill_eta.evaluate_admission(
        _cfg(), MODEL, EST_TOKENS, PREDICTED_RATIO, "low",
        measured_n_past=measured_n_past,
    )
    predicted = prefill_eta.evaluate_admission(
        _cfg(), MODEL, EST_TOKENS, PREDICTED_RATIO, "low",
    )

    # Priced from the measurement (25 492), not the prediction (~172).
    assert measured.eta_pricing_basis == "measured"
    assert measured.new_tokens == MEASURED_NEW
    assert measured.eta_pricing_ratio == pytest.approx(N_PAST / EST_TOKENS)

    # The prediction-only decision is unchanged.
    assert predicted.eta_pricing_basis == "predicted"
    assert predicted.new_tokens == round(EST_TOKENS * (1 - PREDICTED_RATIO))


def test_measured_eta_matches_measured_estimate_when_calibrated():
    """The decision's ETA equals the estimate for the measured ratio."""
    _seed_cold_window()
    view = _measured_view()
    measured_n_past, _facts = _resolve(view)

    decision = prefill_eta.evaluate_admission(
        _cfg(), MODEL, EST_TOKENS, PREDICTED_RATIO, "low",
        measured_n_past=measured_n_past,
    )
    assert decision.calibrated is True
    assert decision.eta_pricing_basis == "measured"

    expected_eta, _coeffs = prefill_eta.estimate_prefill_seconds(
        MODEL, EST_TOKENS, N_PAST / EST_TOKENS, _cfg()
    )
    assert decision.eta_prefill_seconds == pytest.approx(expected_eta)


def test_measured_last_dispatch_prices_from_measurement():
    """``source=measured_last_dispatch`` (rung 2 store) prices measured too."""
    measured_prefill.set_measured_prefill(
        MODEL,
        SESSION,
        MeasuredPrefill(total=N_PAST, cache=0, processed=0, time_ms=1234.0),
    )
    # Slot owned, but context_tokens unknown -> rung 2 fires.
    view = _measured_view(slots=[_slot(0, None)], owners={("", 0): SESSION})
    measured_n_past, facts = _resolve(view)

    assert facts.source == provider.RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH
    assert measured_n_past == N_PAST

    decision = prefill_eta.evaluate_admission(
        _cfg(), MODEL, EST_TOKENS, PREDICTED_RATIO, "low",
        measured_n_past=measured_n_past,
    )
    assert decision.eta_pricing_basis == "measured"
    assert decision.new_tokens == MEASURED_NEW


# ---------------------------------------------------------------------------
# AC1 — no measurement is inferred from absent data
# ---------------------------------------------------------------------------

def test_session_slot_keeps_predicted():
    """``session_slot`` (size unknown, n_past=None) keeps the prediction."""
    # Slot owned, no context_tokens and no measured dispatch.
    view = _measured_view(slots=[_slot(0, None)], owners={("", 0): SESSION})
    measured_n_past, facts = _resolve(view)

    assert facts.source == provider.RESIDENCY_SOURCE_SESSION_SLOT
    assert facts.n_past_actual is None
    assert measured_n_past is None

    decision = prefill_eta.evaluate_admission(
        _cfg(), MODEL, EST_TOKENS, PREDICTED_RATIO, "low",
        measured_n_past=measured_n_past,
    )
    assert decision.eta_pricing_basis == "predicted"
    assert decision.new_tokens == round(EST_TOKENS * (1 - PREDICTED_RATIO))


def test_non_resident_keeps_predicted():
    """``non_resident`` (someone else's slot) keeps the predicted ratio."""
    view = _measured_view(
        slots=[_slot(0, N_PAST)], owners={("", 0): "sess-other"}
    )
    measured_n_past, facts = _resolve(view)

    assert facts.source == provider.RESIDENCY_SOURCE_NON_RESIDENT
    assert measured_n_past is None


def test_unknown_stale_view_keeps_predicted():
    """No fresh view and no ownership -> ``unknown`` -> predicted."""
    # Empty snapshot => get_residency_view() reports known=False.
    _install_slot_state(slots=[], owners={})
    view = _measured_view(slots=[], owners={})
    measured_n_past, facts = _resolve(view)

    assert facts.source == provider.RESIDENCY_SOURCE_UNKNOWN
    assert measured_n_past is None


def test_stale_view_keeps_predicted():
    """A stale snapshot with no ownership -> ``unknown`` -> predicted."""
    view = _measured_view(
        slots=[_slot(0, N_PAST)],
        owners={},
        age=999.0,
    )
    measured_n_past, facts = _resolve(view)

    assert facts.source == provider.RESIDENCY_SOURCE_UNKNOWN
    assert measured_n_past is None


# ---------------------------------------------------------------------------
# AC1 — cross-session isolation
# ---------------------------------------------------------------------------

def test_measurement_never_prices_another_session():
    """Session A's measurement never prices session B (isolation)."""
    view = _measured_view(
        slots=[_slot(0, N_PAST)], owners={("", 0): SESSION}
    )
    # Session A is measured...
    a_n_past, a_facts = _resolve(view, session_id=SESSION)
    assert a_facts.source == provider.RESIDENCY_SOURCE_MEASURED
    assert a_n_past == N_PAST
    # ...session B (owns no slot) is not.
    b_n_past, b_facts = _resolve(view, session_id="sess-b")
    assert b_facts.source == provider.RESIDENCY_SOURCE_NON_RESIDENT
    assert b_n_past is None


# ---------------------------------------------------------------------------
# AC6 — single-value rollback
# ---------------------------------------------------------------------------

def test_rollback_predicted_reproduces_pre_change_numbering():
    """``local_eta_pricing_basis: predicted`` ignores the measurement."""
    view = _measured_view()
    measured_n_past, _facts = _resolve(view)

    rollback = prefill_eta.evaluate_admission(
        _cfg(local_eta_pricing_basis="predicted"),
        MODEL, EST_TOKENS, PREDICTED_RATIO, "low",
        measured_n_past=measured_n_past,
    )
    # Byte-for-byte the prediction-only numbering.
    baseline = prefill_eta.evaluate_admission(
        _cfg(), MODEL, EST_TOKENS, PREDICTED_RATIO, "low",
    )
    assert rollback.eta_pricing_basis == "predicted"
    assert rollback.new_tokens == baseline.new_tokens
    assert rollback.eta_prefill_seconds == baseline.eta_prefill_seconds


def test_measured_is_the_default_basis():
    """The shipped default uses the measurement (AC6 target behaviour)."""
    assert prefill_eta.resolve_config({"server": {}}).eta_pricing_basis == "measured"
    assert prefill_eta.DEFAULT_ETA_PRICING_BASIS == "measured"
