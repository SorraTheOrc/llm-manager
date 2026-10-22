"""Prefill-ETA calibration + admission decisions (LP-0MVGDUV5E005UWBQ).

Covers the guard's own contract:

- AC1: per-model / per-context-bucket coefficients fitted from observed
  ``dispatch_first_byte_ms`` samples, sane defaults, and inert behaviour when
  there are too few samples.
- AC3: ETA above budget ⇒ refuse; ETA within budget ⇒ admit; ``high``
  priority has no budget (never queues); missing calibration ⇒ admit.
- AC5: a cache-warm prompt (90 % of 100k tokens cached) is admitted even when
  the same bucket has been calibrated on cache-cold dispatches.
"""

from __future__ import annotations

import logging
import math

import pytest

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
        "contention_queue_max_wait_seconds": 120,
        "contention_queue_max_depth": 8,
    }
    server.update(server_overrides)
    return {"server": server}


def _seed_samples(
    model: str = "Qwen3",
    tokens: int = 100_000,
    ratio: float = 0.0,
    ms_per_token: float = 3.9,
    count: int = 6,
) -> None:
    """Record *count* synthetic dispatches at *ms_per_token*."""
    for _ in range(count):
        prefill_eta.record_dispatch_sample(
            model, tokens, ratio, tokens * ms_per_token
        )


# ---------------------------------------------------------------------------
# AC1: calibration
# ---------------------------------------------------------------------------

def test_bucket_for_assigns_each_context_interval():
    assert prefill_eta.bucket_for(0) == 0
    assert prefill_eta.bucket_for(16_383) == 0
    assert prefill_eta.bucket_for(16_384) == 1
    assert prefill_eta.bucket_for(65_536) == 2
    assert prefill_eta.bucket_for(131_072) == 3
    assert prefill_eta.bucket_for(500_000) == 3


def test_too_few_samples_keeps_the_defaults_and_the_guard_inert():
    """AC1/AC3: below ``min_samples`` the guard makes no decision, so today's
    queueing behaviour is unchanged."""
    _seed_samples(count=2)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is True
    assert decision.active is False
    assert decision.reason == "insufficient_samples"
    assert decision.calibrated is False
    coeffs = prefill_eta.coefficients("Qwen3", 100_000, _cfg())
    assert coeffs.calibrated is False
    assert coeffs.coded_ms_per_token == prefill_eta.DEFAULT_CODED_MS_PER_TOKEN
    assert coeffs.prefill_ms_per_token == prefill_eta.DEFAULT_PREFILL_MS_PER_TOKEN


def test_fitted_coefficients_track_the_observed_cost():
    """AC1: coefficients are derived from the samples, not a fixed constant."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=5.0, count=6)
    coeffs = prefill_eta.coefficients("Qwen3", 100_000, _cfg())

    assert coeffs.calibrated is True
    assert coeffs.samples == 6
    # 5.0 ms/token observed × the conservatism factor.
    assert coeffs.prefill_ms_per_token == pytest.approx(
        5.0 * prefill_eta.CONSERVATISM_FACTOR, rel=1e-6
    )
    # No warm samples ⇒ the coded coefficient keeps its default.
    assert coeffs.coded_ms_per_token == pytest.approx(
        prefill_eta.DEFAULT_CODED_MS_PER_TOKEN
    )


def test_calibration_is_per_model_and_per_context_bucket():
    """AC1: a hot bucket does not calibrate an unrelated model or bucket."""
    _seed_samples(model="Qwen3", tokens=100_000, ms_per_token=5.0, count=6)

    other_model = prefill_eta.coefficients("Other", 100_000, _cfg())
    other_bucket = prefill_eta.coefficients("Qwen3", 8_000, _cfg())

    assert other_model.calibrated is False
    assert other_bucket.calibrated is False
    assert prefill_eta.coefficients("Qwen3", 100_000, _cfg()).calibrated is True


def test_fitted_coefficients_are_clamped_to_plausible_bounds():
    """AC1: an absurd outlier cannot make the guard reject everything."""
    _seed_samples(tokens=100_000, ms_per_token=1_000.0, count=6)
    coeffs = prefill_eta.coefficients("Qwen3", 100_000, _cfg())

    assert coeffs.calibrated is True
    assert coeffs.prefill_ms_per_token <= prefill_eta.MAX_PREFILL_MS_PER_TOKEN
    assert coeffs.coded_ms_per_token >= prefill_eta.MIN_CODED_MS_PER_TOKEN


def test_invalid_samples_are_ignored():
    prefill_eta.record_dispatch_sample("Qwen3", 0, 0.0, 1000.0)
    prefill_eta.record_dispatch_sample("", 10_000, 0.0, 1000.0)
    prefill_eta.record_dispatch_sample("Qwen3", 10_000, 0.0, 0.0)
    prefill_eta.record_dispatch_sample("Qwen3", 10_000, 0.0, math.inf)

    coeffs = prefill_eta.coefficients("Qwen3", 10_000, _cfg())
    assert coeffs.samples == 0
    assert coeffs.calibrated is False


def test_window_is_bounded():
    _seed_samples(count=prefill_eta.WINDOW_SAMPLES + 30)
    coeffs = prefill_eta.coefficients("Qwen3", 100_000, _cfg())
    assert coeffs.samples == prefill_eta.WINDOW_SAMPLES


# ---------------------------------------------------------------------------
# AC3: admission decisions
# ---------------------------------------------------------------------------

def test_eta_above_budget_rejects_immediately():
    """AC3: a cache-cold 100k prompt cannot prefill in 120 s ⇒ refuse."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "low")

    assert decision.admit is False
    assert decision.active is True
    assert decision.reason == "eta_exceeds_budget"
    assert decision.eta_prefill_seconds > decision.budget_seconds
    assert decision.budget_seconds == pytest.approx(120.0)


def test_eta_within_budget_admits():
    """AC3: a small cache-cold prompt prefills well inside 120 s ⇒ admit."""
    _seed_samples(tokens=2_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 2_000, 0.0, "low")

    assert decision.admit is True
    assert decision.active is True
    assert decision.reason == "within_budget"
    assert decision.eta_prefill_seconds <= decision.budget_seconds


def test_high_priority_never_queues_and_so_is_never_refused():
    """AC3 regression guard: ``high`` priority never queues, so it has no
    budget and the guard cannot refuse it."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "high")

    assert decision.admit is True
    assert decision.active is False
    assert decision.reason == "no_budget"


def test_observe_policy_logs_the_verdict_without_refusing():
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_policy="observe"), "Qwen3", 100_000, 0.0, "low"
    )

    assert decision.admit is True
    assert decision.active is True
    assert decision.reason == "eta_exceeds_budget"
    assert decision.policy == "observe"


def test_off_policy_does_not_evaluate_the_guard():
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_policy="off"), "Qwen3", 100_000, 0.0, "low"
    )

    assert decision.admit is True
    assert decision.active is False
    assert decision.reason == "policy_off"


def test_invalid_policy_falls_back_to_reject():
    guard_cfg = prefill_eta.resolve_config(
        _cfg(local_admission_guard_policy="banana")
    )
    assert guard_cfg.policy == prefill_eta.POLICY_REJECT


def test_clamped_wait_cap_still_yields_a_budget():
    """A zero wait cap is clamped to the 1 s floor by the queue config, so the
    guard still has a (tiny) budget and refuses the huge cold prompt."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(
        _cfg(contention_queue_max_wait_seconds=0),
        "Qwen3",
        100_000,
        0.0,
        "low",
    )
    # The queue config clamps the wait to >= 1 s, so the guard still decides.
    assert decision.admit is False


# ---------------------------------------------------------------------------
# AC5: cheap (cache-warm) prefills stay admissible
# ---------------------------------------------------------------------------

def test_warm_cache_prompt_is_admitted_even_when_bucket_is_cold_calibrated():
    """AC5: 90 %-cached 100k prompt ⇒ the expensive ``new_tokens`` term is only
    10k tokens, so the projected prefill fits the 120 s budget."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.9, "low")

    assert decision.admit is True, decision
    assert decision.reason == "within_budget"
    # 90k cached × 0.5 ms + 10k new × fitted-4.68 ms ≈ 91.8 s < 120 s.
    assert decision.eta_prefill_seconds < decision.budget_seconds


def test_cold_prompt_of_the_same_size_is_refused():
    """Contrast for AC5: the identical prompt with a cold cache is refused."""
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "low")
    assert decision.admit is False


# ---------------------------------------------------------------------------
# Estimation + published status fields
# ---------------------------------------------------------------------------

def test_estimate_uses_both_coefficient_terms():
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=4.0, count=6)
    eta, coeffs = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 100_000, 0.25, _cfg()
    )
    expected_ms = (
        25_000 * coeffs.coded_ms_per_token + 75_000 * coeffs.prefill_ms_per_token
    )
    assert coeffs.calibrated is True
    assert eta == pytest.approx(expected_ms / 1000.0)


def test_estimate_without_samples_uses_defaults():
    eta, coeffs = prefill_eta.estimate_prefill_seconds("Qwen3", 10_000, 0.0, _cfg())
    assert coeffs.calibrated is False
    assert eta == pytest.approx(
        10_000 * prefill_eta.DEFAULT_PREFILL_MS_PER_TOKEN / 1000.0
    )


def test_status_estimate_publishes_prefill_and_queue_eta():
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    fields = prefill_eta.status_estimate(
        _cfg(), "Qwen3", total_slots=2, available_slots=0, active_queries=2
    )

    assert fields["prefill_eta_seconds"] > 0
    # Two occupied slots ⇒ newcomer waits for both prefills plus its own.
    assert fields["queue_eta_seconds"] == pytest.approx(
        fields["prefill_eta_seconds"] * 3, rel=1e-2
    )
    assert fields["prefill_eta_reference_tokens"] == 100_000
    assert fields["prefill_eta_samples"] == 6
    assert fields["prefill_eta_calibrated"] is True


def test_status_estimate_uses_configured_reference_before_any_dispatch():
    fields = prefill_eta.status_estimate(
        _cfg(local_admission_guard_reference_tokens=8_000),
        "Qwen3",
        total_slots=1,
        available_slots=1,
        active_queries=0,
    )
    assert fields["prefill_eta_reference_tokens"] == 8_000
    # An idle slot ⇒ the newcomer only waits for its own prefill.
    assert fields["queue_eta_seconds"] == pytest.approx(
        fields["prefill_eta_seconds"], rel=1e-2
    )
    assert fields["prefill_eta_calibrated"] is False


def test_status_estimate_falls_back_to_active_queries_without_slot_data():
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    fields = prefill_eta.status_estimate(
        _cfg(), "Qwen3", total_slots=0, available_slots=0, active_queries=1
    )
    assert fields["queue_eta_seconds"] == pytest.approx(
        fields["prefill_eta_seconds"] * 2, rel=1e-2
    )


def test_per_slot_prefill_eta_is_null_when_context_unknown():
    """LP-0MVGDUVGE0024YAI AC3: an unobservable prompt size yields ``None``
    (never a misleading ``0``); a known size costs from the same estimator."""
    assert prefill_eta.per_slot_prefill_eta_seconds("Qwen3", None, _cfg()) is None
    assert prefill_eta.per_slot_prefill_eta_seconds("Qwen3", 0, _cfg()) is None

    expected, _coeffs = prefill_eta.estimate_prefill_seconds("Qwen3", 8_000, 0.0, _cfg())
    assert prefill_eta.per_slot_prefill_eta_seconds(
        "Qwen3", 8_000, _cfg()
    ) == pytest.approx(round(expected, 1))


def test_status_estimate_publishes_admission_budget():
    """LP-0MVGDUVGE0024YAI AC3: the published budget is the guard's own wait
    budget (one source of truth), and ``None`` when none is resolvable."""
    fields = prefill_eta.status_estimate(_cfg(), "Qwen3", total_slots=1, available_slots=1)
    assert fields["prefill_budget_seconds"] == 120.0
    # The published budget is exactly wait_budget_seconds (null when the
    # guard resolves no budget), never an independent number.
    resolved = prefill_eta.wait_budget_seconds(
        {"server": {"contention_queue_policy": "off"}}
    )
    expected = round(float(resolved), 1) if float(resolved) > 0 else None
    assert (
        prefill_eta.status_estimate(
            {"server": {"contention_queue_policy": "off"}}, "Qwen3"
        )["prefill_budget_seconds"]
        == expected
    )


# ---------------------------------------------------------------------------
# Retry hint + log line (AC2)
# ---------------------------------------------------------------------------

def test_suggested_retry_after_is_the_overshoot_in_whole_seconds():
    assert prefill_eta.suggested_retry_after_seconds(390.0, 120.0) == 60  # capped
    assert prefill_eta.suggested_retry_after_seconds(150.4, 120.0) == 31
    assert prefill_eta.suggested_retry_after_seconds(100.0, 120.0) == 1
    assert prefill_eta.suggested_retry_after_seconds(None, None) == 1


def test_logged_decision_line_is_parsable(caplog):
    """AC2: the decision line carries the decision + ETA + budget fields the
    usage report parses."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_samples(tokens=100_000, ratio=0.0, ms_per_token=3.9, count=6)
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "low")

    prefill_eta.record_guard_decision(
        decision, provider="local-llama", session_id="s-1", model="Qwen3",
        priority="low",
    )

    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("local_admission_guard")
    )
    assert line.startswith("local_admission_guard decision=reject")
    assert "reason=eta_exceeds_budget" in line
    assert "enforced=true" in line
    assert "eta_prefill_s=" in line
    assert "budget_s=" in line
    assert "provider=local-llama" in line
    assert "session=s-1" in line


def test_inert_decisions_are_not_logged(caplog):
    """An inert guard made no decision — logging it would flood the log."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    decision = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 100_000, 0.0, "low")
    prefill_eta.record_guard_decision(decision, provider="local-llama")

    assert decision.active is False
    assert not [r for r in caplog.records if r.getMessage().startswith("local_admission_guard")]
