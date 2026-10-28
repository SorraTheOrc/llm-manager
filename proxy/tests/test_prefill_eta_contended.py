"""Co-residency-aware prefill calibration and pricing (LP-0MVS9Y76R004B5M8).

Prefill is serialised but prefill+decode batching is not: an occupant that is
*generating* halves a newcomer's prefill throughput (measured 2-4x lower t/s).
The mixed solitary-dominant calibration window therefore prices contended
admissions optimistically (a 6,659-token prompt priced at 13.9 s against a
~120 t/s prompt-eval median for the same day). This file's contract:

- AC1: ``record_dispatch_sample`` accepts a co-residency flag and stores each
  sample in the right window; the OR of admission-time ``busy_slots`` and a
  first-byte-time live check marks a sample contended
  (``dispatch_sample_contended``).
- AC2: ``coefficients`` / ``estimate_prefill_seconds`` accept a contended
  selector; a contended admission (``busy_slots >= 1``) prices from the
  contended window when calibrated and falls back to the solitary window when
  not (``rate_bucket=contended|solitary|solitary_fallback``).
- AC3: the ``local_admission_guard`` decision log carries ``rate_bucket`` and
  ``contended_samples``.
- AC4: a contended window seeded at the measured median (~120 t/s,
  ~8.3 ms/token) prices a cache-cold 6,659-token prompt at >= 55 s.
- AC5: warm prompts (``cached_ratio >= 0.9``) price identically to before the
  change (the contended selector is suppressed on the warm path).
"""

from __future__ import annotations

import logging

import pytest
from proxy.router_helpers import dispatch_sample_contended

from proxy import prefill_eta


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
        # Isolate the budget bound (the deadline bound is covered by
        # tests/test_prefill_eta_deadline.py).
        "local_prefill_deadline_seconds": 0,
    }
    server.update(server_overrides)
    return {"server": server}


def _seed(
    tokens: int = 100_000,
    ratio: float = 0.0,
    ms_per_token: float = 3.9,
    count: int = 6,
    contended: bool = False,
    model: str = "Qwen3",
) -> None:
    """Record *count* synthetic dispatches at *ms_per_token* into a window."""
    for _ in range(count):
        prefill_eta.record_dispatch_sample(
            model, tokens, ratio, tokens * ms_per_token, contended=contended
        )


def _fake_srv(records: dict | None = None):
    """Minimal server stand-in exposing only the live dispatch registry."""
    class _FakeSrv:
        local_dispatch_records = records or {}
    return _FakeSrv()


def _live_record(session_id: str, endpoint: str, active: bool = True) -> tuple:
    return (endpoint, session_id), {"active": active, "expires_at": 0.0}


# ---------------------------------------------------------------------------
# AC1: contended samples land in the contended window (incl. the OR)
# ---------------------------------------------------------------------------

def test_record_dispatch_sample_separates_windows_by_co_residency():
    """Each sample lands in the window matching its contended flag."""
    _seed(tokens=100_000, ms_per_token=3.9, count=6, contended=False)
    _seed(tokens=100_000, ms_per_token=8.3, count=6, contended=True)

    solitary = prefill_eta.coefficients("Qwen3", 100_000, _cfg(), contended=False)
    contended = prefill_eta.coefficients("Qwen3", 100_000, _cfg(), contended=True)

    assert solitary.samples == 6
    assert contended.samples == 6
    # The two windows carry different rates, so the fits differ.
    assert contended.prefill_ms_per_token > solitary.prefill_ms_per_token
    assert contended.rate_bucket == "contended"
    assert solitary.rate_bucket == "solitary"
    # The counts are attributed to the right windows, not mixed.
    assert contended.contended_samples == 6
    assert solitary.contended_samples == 6


def test_record_dispatch_sample_does_not_mix_a_short_contended_window():
    """Below ``min_samples`` the contended window is reported but never fit."""
    _seed(tokens=100_000, ms_per_token=3.9, count=6, contended=False)
    _seed(tokens=100_000, ms_per_token=8.3, count=4, contended=True)

    contended = prefill_eta.coefficients("Qwen3", 100_000, _cfg(), contended=True)

    # 4 contended samples < min_samples (5): the contended window is
    # reported for diagnosis but the solitary window priced the estimate.
    assert contended.contended_samples == 4
    assert contended.samples == 6
    assert contended.rate_bucket == "solitary_fallback"


def test_stamp_local_admission_evidence_carries_co_residency_snapshot():
    """The admission-time stamp feeds the contended OR at first-byte time."""
    from proxy import provider

    class _Request:
        pass

    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 40_000, 0.0, "low"
    )
    request = _Request()
    provider._stamp_local_admission_evidence(
        request,
        estimated_tokens=40_000,
        cached_ratio=0.0,
        new_tokens=40_000,
        decision=decision,
        model_name="Qwen3",
        busy_slots=3,
        in_flight_prefills=1,
    )

    snap = getattr(request, provider.LOCAL_ADMISSION_EVIDENCE_ATTR)
    assert snap["busy_slots"] == 3
    assert snap["in_flight_prefills"] == 1
    # The OR helper: an occupied gate at admission marks the sample contended
    # even with an otherwise-empty live registry at first-byte time.
    assert dispatch_sample_contended(
        _fake_srv(), "self", endpoint="ep", admission_busy_slots=snap["busy_slots"]
    ) is True


def test_dispatch_sample_contended_or_admission_time_snapshot_and_live_check():
    """Either signal marks a sample contended; neither leaves it solitary."""
    ep, self_id = "http://llama:8080", "session-self"
    other = _live_record("session-other", ep, active=True)

    # (b) live check alone: another session's dispatch is active.
    assert dispatch_sample_contended(
        _fake_srv(dict([other])), self_id, endpoint=ep, admission_busy_slots=0
    ) is True
    # (a) admission-time snapshot alone.
    assert dispatch_sample_contended(
        _fake_srv(), self_id, endpoint=ep, admission_busy_slots=1
    ) is True
    # Neither signal: solitary.
    assert dispatch_sample_contended(
        _fake_srv(), self_id, endpoint=ep, admission_busy_slots=0
    ) is False
    # Endpoint scope: another session on a DIFFERENT endpoint is not
    # co-residency for this gate.
    assert dispatch_sample_contended(
        _fake_srv(dict([_live_record("session-other", "http://other:8080")])),
        self_id,
        endpoint=ep,
        admission_busy_slots=0,
    ) is False
    # A session's own live dispatch never makes itself contended.
    assert dispatch_sample_contended(
        _fake_srv(dict([_live_record(self_id, ep)])),
        self_id,
        endpoint=ep,
        admission_busy_slots=0,
    ) is False
    # A released (inactive) record of another session is not co-residency.
    assert dispatch_sample_contended(
        _fake_srv(dict([_live_record("session-other", ep, active=False)])),
        self_id,
        endpoint=ep,
        admission_busy_slots=0,
    ) is False


# ---------------------------------------------------------------------------
# AC2: contended pricing selector + solitary fallback
# ---------------------------------------------------------------------------

def test_contended_admission_prices_from_contended_window_when_calibrated():
    """busy_slots >= 1 prices the own-prefill term from the contended window."""
    _seed(tokens=100_000, ms_per_token=3.9, count=6, contended=False)
    _seed(tokens=100_000, ms_per_token=8.3, count=6, contended=True)
    cfg = _cfg()

    decision = prefill_eta.evaluate_admission(
        cfg, "Qwen3", 100_000, 0.0, "low", busy_slots=1
    )

    assert decision.active is True
    assert decision.rate_bucket == "contended"
    assert decision.contended_samples == 6
    # The contended rate (8.3 ms * 1.2 conservatism) must be priced, not the
    # fast solitary rate — the whole point of the change.
    expected_eta = 100_000 * 8.3 * prefill_eta.CONSERVATISM_FACTOR / 1000.0
    assert decision.eta_prefill_seconds == pytest.approx(expected_eta, rel=1e-3)


def test_contended_admission_falls_back_to_solitary_when_uncalibrated():
    """No contended samples yet: price from the solitary window, say so."""
    _seed(tokens=100_000, ms_per_token=3.9, count=6, contended=False)
    cfg = _cfg()

    decision = prefill_eta.evaluate_admission(
        cfg, "Qwen3", 100_000, 0.0, "low", busy_slots=1
    )

    assert decision.rate_bucket == "solitary_fallback"
    assert decision.contended_samples == 0
    assert decision.eta_prefill_seconds == pytest.approx(
        100_000 * 3.9 * prefill_eta.CONSERVATISM_FACTOR / 1000.0, rel=1e-3
    )


def test_solitary_admission_prices_from_solitary_window():
    _seed(tokens=100_000, ms_per_token=3.9, count=6, contended=False)
    _seed(tokens=100_000, ms_per_token=8.3, count=6, contended=True)
    cfg = _cfg()

    decision = prefill_eta.evaluate_admission(
        cfg, "Qwen3", 100_000, 0.0, "low", busy_slots=0
    )

    assert decision.rate_bucket == "solitary"
    assert decision.rate_bucket != "contended"
    assert decision.eta_prefill_seconds == pytest.approx(
        100_000 * 3.9 * prefill_eta.CONSERVATISM_FACTOR / 1000.0, rel=1e-3
    )


def test_estimate_prefill_seconds_accepts_the_contended_selector():
    """The low-level estimator honours the selector directly."""
    _seed(tokens=6_659, ms_per_token=3.9, count=6, contended=False)
    _seed(tokens=6_659, ms_per_token=8.3, count=6, contended=True)
    cfg = _cfg()

    solitary_eta, solitary = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 6_659, 0.0, cfg, contended=False
    )
    contended_eta, contended = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 6_659, 0.0, cfg, contended=True
    )

    assert contended.rate_bucket == "contended"
    assert solitary.rate_bucket == "solitary"
    assert contended_eta > solitary_eta


# ---------------------------------------------------------------------------
# AC4: measured contended median prices the known case >= 55 s
# ---------------------------------------------------------------------------

def test_contended_window_at_measured_median_prices_6659_tokens_above_55s():
    """~120 t/s (~8.3 ms/token) * 6,659 tokens must price >= 55 s, not the
    recorded 13.9 s optimistic figure (parent AC2)."""
    _seed(tokens=6_659, ms_per_token=8.3, count=6, contended=True)
    cfg = _cfg()

    eta, coeffs = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 6_659, 0.0, cfg, contended=True
    )

    assert coeffs.rate_bucket == "contended"
    assert coeffs.calibrated is True
    assert eta >= 55.0
    # And a solitary-only calibration of the old shape prices the same prompt
    # well below the contended figure (the optimism this change fixes).
    prefill_eta.reset()
    _seed(tokens=6_659, ms_per_token=3.9, count=6, contended=False)
    pre_change_eta, _ = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 6_659, 0.0, cfg, contended=False
    )
    assert eta > pre_change_eta


# ---------------------------------------------------------------------------
# AC5: warm prompts are untouched
# ---------------------------------------------------------------------------

def test_warm_prompt_prices_identically_to_before_the_change():
    """cached_ratio >= 0.9 keeps the pre-change (solitary) computation even
    behind an occupied gate."""
    _seed(tokens=100_000, ms_per_token=8.3, count=6, contended=True)
    _seed(tokens=100_000, ms_per_token=3.9, count=6, contended=False)
    cfg = _cfg()

    pre_change_eta = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 100_000, 0.9, cfg, contended=False
    )[0]

    decision = prefill_eta.evaluate_admission(
        cfg, "Qwen3", 100_000, 0.9, "low", busy_slots=1
    )
    assert decision.active is True
    assert decision.rate_bucket == "solitary"
    assert decision.eta_prefill_seconds == pytest.approx(pre_change_eta, rel=1e-9)
    # A cache-miss prompt behind the same gate IS contended-priced (control).
    cold = prefill_eta.evaluate_admission(
        cfg, "Qwen3", 100_000, 0.0, "low", busy_slots=1
    )
    assert cold.rate_bucket == "contended"
    assert cold.eta_prefill_seconds > decision.eta_prefill_seconds


# ---------------------------------------------------------------------------
# AC3: decision log carries rate_bucket + contended_samples
# ---------------------------------------------------------------------------

def _decision_log_line(caplog, **kwargs) -> str:
    prefill_eta.record_guard_decision(kwargs.pop("decision"), **kwargs)
    return next(
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("local_admission_guard")
    )


def test_log_carries_rate_bucket_for_contended_decision(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed(tokens=100_000, ms_per_token=8.3, count=6, contended=True)
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 100_000, 0.0, "low", busy_slots=1
    )
    assert decision.active is True

    line = _decision_log_line(caplog, decision=decision)

    assert "rate_bucket=contended" in line
    assert "contended_samples=6" in line
    assert "busy_slots=1" in line


def test_log_carries_rate_bucket_for_solitary_decision(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed(tokens=100_000, ms_per_token=3.9, count=6, contended=False)
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 100_000, 0.0, "low", busy_slots=0
    )
    assert decision.active is True

    line = _decision_log_line(caplog, decision=decision)

    assert "rate_bucket=solitary" in line
    assert "contended_samples=0" in line


def test_log_carries_rate_bucket_for_uncalibrated_contended_fallback(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed(tokens=100_000, ms_per_token=3.9, count=6, contended=False)
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 100_000, 0.0, "low", busy_slots=1
    )
    assert decision.active is True
    assert decision.rate_bucket == "solitary_fallback"

    line = _decision_log_line(caplog, decision=decision)

    assert "rate_bucket=solitary_fallback" in line
    assert "contended_samples=0" in line


# ---------------------------------------------------------------------------
# Published status surface (scope: chosen bucket on /llama/local/status)
# ---------------------------------------------------------------------------

def test_status_estimate_exposes_the_chosen_bucket():
    _seed(tokens=32_000, ms_per_token=8.3, count=6, contended=True)
    _seed(tokens=32_000, ms_per_token=3.9, count=6, contended=False)
    cfg = _cfg()

    # No busy slots -> solitary bucket published.
    solitary = prefill_eta.status_estimate(cfg, "Qwen3", total_slots=2, available_slots=2)
    assert solitary["prefill_eta_rate_bucket"] == "solitary"
    # Busy gate -> contended bucket published when calibrated.
    contended = prefill_eta.status_estimate(cfg, "Qwen3", total_slots=2, available_slots=1)
    assert contended["prefill_eta_rate_bucket"] == "contended"
    assert contended["prefill_eta_contended_samples"] == 6


def test_status_estimate_fallback_bucket_when_uncalibrated():
    _seed(tokens=32_000, ms_per_token=3.9, count=6, contended=False)
    cfg = _cfg()

    contended = prefill_eta.status_estimate(cfg, "Qwen3", total_slots=2, available_slots=1)
    assert contended["prefill_eta_rate_bucket"] == "solitary_fallback"
    assert contended["prefill_eta_contended_samples"] == 0
