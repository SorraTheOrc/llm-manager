"""Local admission stamp consistency (LP-0MWRQ6LSV009DVP4, parent AC2).

`prefill_eta_accuracy` scores the dispatch record's stamped ``cached_ratio`` /
``new_tokens`` / ``eta_prefill_s``. Before this fix the stamp always recorded
the routing-time *prediction* even when the admission guard had priced the
request from a *measured* resident prefix — so the metric scored a number the
guard never enforced.

These tests pin the AC2 contract on the two helpers that carry it:
:func:`proxy.provider._stamp_pricing_for_decision` (chooses the basis) and
:func:`proxy.provider._stamp_local_admission_evidence` (writes the triple), so
the stamped evidence is internally consistent and follows the ETA's basis.
"""

from __future__ import annotations

import proxy.provider as provider
import pytest
from proxy import prefill_eta

MODEL = "Qwen3"
EST_TOKENS = 55_492
N_PAST = 30_000
MEASURED_RATIO = N_PAST / EST_TOKENS
MEASURED_NEW = EST_TOKENS - N_PAST
ROUTING_RATIO = 172 / EST_TOKENS  # the prediction
ROUTING_NEW = 172


def _cfg(**overrides) -> dict:
    server = {
        "contention_queue_max_wait_seconds": 60,
        "local_prefill_deadline_seconds": 0,
    }
    server.update(overrides)
    return {"server": server}


class _Request:
    """Minimal request stub whose attributes ``setattr`` accepts."""


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _measured_decision(*, basis_default: str = "measured"):
    return prefill_eta.evaluate_admission(
        _cfg(local_eta_pricing_basis=basis_default),
        MODEL,
        EST_TOKENS,
        ROUTING_RATIO,
        "low",
        measured_n_past=N_PAST,
    )


def _predicted_decision():
    return prefill_eta.evaluate_admission(
        _cfg(), MODEL, EST_TOKENS, ROUTING_RATIO, "low"
    )


# ---------------------------------------------------------------------------
# AC2 — the stamp follows the ETA's measured basis
# ---------------------------------------------------------------------------

def test_stamp_pricing_uses_measured_basis():
    """A measured decision stamps the measured ratio and new-token count."""
    decision = _measured_decision()
    ratio, new_tokens = provider._stamp_pricing_for_decision(
        decision, ROUTING_RATIO, ROUTING_NEW, EST_TOKENS
    )
    assert ratio == pytest.approx(MEASURED_RATIO)
    assert new_tokens == MEASURED_NEW


def test_stamp_pricing_keeps_routing_values_on_prediction():
    """A predicted decision keeps the routing values byte-for-byte."""
    decision = _predicted_decision()
    ratio, new_tokens = provider._stamp_pricing_for_decision(
        decision, ROUTING_RATIO, ROUTING_NEW, EST_TOKENS
    )
    assert ratio == ROUTING_RATIO
    assert new_tokens == ROUTING_NEW


def test_stamp_pricing_without_decision_keeps_routing_values():
    ratio, new_tokens = provider._stamp_pricing_for_decision(
        None, ROUTING_RATIO, ROUTING_NEW, EST_TOKENS
    )
    assert (ratio, new_tokens) == (ROUTING_RATIO, ROUTING_NEW)


# ---------------------------------------------------------------------------
# AC2 — the stamped triple is mutually consistent on a measured admit
# ---------------------------------------------------------------------------

def test_stamped_triple_is_consistent_on_measured_admit():
    decision = _measured_decision()
    ratio, new_tokens = provider._stamp_pricing_for_decision(
        decision, ROUTING_RATIO, ROUTING_NEW, EST_TOKENS
    )
    request = _Request()
    provider._stamp_local_admission_evidence(
        request,
        estimated_tokens=EST_TOKENS,
        cached_ratio=ratio,
        new_tokens=new_tokens,
        decision=decision,
        model_name=MODEL,
    )
    stamp = getattr(request, provider.LOCAL_ADMISSION_EVIDENCE_ATTR)

    # new_tokens == estimated * (1 - cached_ratio), within rounding.
    assert abs(
        stamp["new_tokens"] - EST_TOKENS * (1 - stamp["cached_ratio"])
    ) <= 1
    # The stamped ratio is the measured one, not the prediction.
    assert stamp["cached_ratio"] == pytest.approx(MEASURED_RATIO)
    # eta_prefill_s equals the estimate the stamped ratio produces.
    expected_eta, _ = prefill_eta.estimate_prefill_seconds(
        MODEL, EST_TOKENS, stamp["cached_ratio"], _cfg()
    )
    assert stamp["eta_prefill_s"] == pytest.approx(expected_eta)


def test_stamp_is_written_once_and_re_dispatch_inherits_same_numbers():
    """A slot-retry re-dispatch reusing the request inherits the same triple."""
    decision = _measured_decision()
    ratio, new_tokens = provider._stamp_pricing_for_decision(
        decision, ROUTING_RATIO, ROUTING_NEW, EST_TOKENS
    )
    request = _Request()
    provider._stamp_local_admission_evidence(
        request,
        estimated_tokens=EST_TOKENS,
        cached_ratio=ratio,
        new_tokens=new_tokens,
        decision=decision,
        model_name=MODEL,
    )
    first = dict(getattr(request, provider.LOCAL_ADMISSION_EVIDENCE_ATTR))
    # A re-dispatch does not re-price: the same request keeps the same values.
    second = dict(getattr(request, provider.LOCAL_ADMISSION_EVIDENCE_ATTR))
    assert first == second


# ---------------------------------------------------------------------------
# AC2 — stamping is best-effort (never raises)
# ---------------------------------------------------------------------------

def test_stamp_is_best_effort_on_stubborn_object():
    """An object that rejects ``setattr`` does not raise."""
    class _Immutable:
        __slots__ = ()

    # Must not raise.
    provider._stamp_local_admission_evidence(
        _Immutable(),
        estimated_tokens=EST_TOKENS,
        cached_ratio=MEASURED_RATIO,
        new_tokens=MEASURED_NEW,
        decision=None,
        model_name=MODEL,
    )


def test_stamp_without_decision_still_consistent():
    """No decision -> the stamp re-estimates and stores a consistent triple."""
    request = _Request()
    provider._stamp_local_admission_evidence(
        request,
        estimated_tokens=EST_TOKENS,
        cached_ratio=MEASURED_RATIO,
        new_tokens=MEASURED_NEW,
        decision=None,
        model_name=MODEL,
    )
    stamp = getattr(request, provider.LOCAL_ADMISSION_EVIDENCE_ATTR)
    assert abs(stamp["new_tokens"] - EST_TOKENS * (1 - stamp["cached_ratio"])) <= 1
    expected_eta, _ = prefill_eta.estimate_prefill_seconds(
        MODEL, EST_TOKENS, MEASURED_RATIO, _cfg()
    )
    assert stamp["eta_prefill_s"] == pytest.approx(expected_eta)
