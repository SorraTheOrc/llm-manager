"""Conservative prefill ETA for large prompts (LP-0MVKUKF65008D5UN).

The point least-squares estimate is optimistic for large prompts (a 19,940-token
prompt was modelled at 19.9 s while measured first-byte times for this workload
span p50 16 s … p90 189 s … max 591 s). This module verifies:

- AC1: for a synthetic sample window the estimator prices a large prompt at
  >= the configured percentile of observed first-byte times (default p75) for
  its (model, bucket) window — never below it.
- AC2: large-prompt buckets (>= 32k tokens) are never priced below the window's
  measured p50, and the conservative ETA combined with the C2 deadline bound
  (``local_prefill_deadline_seconds``, LP-0MVKUKEU80022C3U) refuses local
  dispatch instead of starting a prefill the deadline will cancel.
- AC3: small-prompt (< 8k) estimates and admissions are unchanged whether the
  knob is on or off.
- AC4: the ``local_admission_guard_eta_percentile`` knob is read, clamped,
  has a documented default (75) and rollback (0 disables), and the shipped
  ``config.yaml`` declares it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from proxy import prefill_eta

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"


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
    }
    server.update(server_overrides)
    return {"server": server}


def _seed_varied(
    model: str = "Qwen3",
    tokens: int = 100_000,
    ratio: float = 0.0,
    service_ms: tuple[float, ...] = (),
) -> None:
    """Record one synthetic cold dispatch per *service_ms* value (spread
    window — unlike the identical-sample fixtures used elsewhere, so the
    conservative percentile actually has something to be conservative about)."""
    for ms in service_ms:
        prefill_eta.record_dispatch_sample(model, tokens, ratio, float(ms))


# ---------------------------------------------------------------------------
# AC4: the config knob
# ---------------------------------------------------------------------------

def test_resolve_config_defaults_percentile_to_p75():
    assert prefill_eta.DEFAULT_ETA_PERCENTILE == 75.0
    guard_cfg = prefill_eta.resolve_config(_cfg())
    assert guard_cfg.eta_percentile == pytest.approx(75.0)


def test_resolve_config_reads_the_percentile_knob():
    guard_cfg = prefill_eta.resolve_config(
        _cfg(local_admission_guard_eta_percentile=90)
    )
    assert guard_cfg.eta_percentile == pytest.approx(90.0)


def test_resolve_config_rejects_out_of_range_percentiles_to_default():
    for bad in (101, -1, "banana"):
        guard_cfg = prefill_eta.resolve_config(
            _cfg(local_admission_guard_eta_percentile=bad)
        )
        assert guard_cfg.eta_percentile == pytest.approx(
            prefill_eta.DEFAULT_ETA_PERCENTILE
        )


def test_shipped_config_declares_the_percentile_knob_with_default_and_rollback():
    """AC4: the shipped config carries the knob at its default with a comment
    documenting the rollback."""
    with open(CONFIG_PATH) as handle:
        server = (yaml.safe_load(handle) or {})["server"]
    assert "local_admission_guard_eta_percentile" in server
    assert server["local_admission_guard_eta_percentile"] == 75

    text = CONFIG_PATH.read_text()
    key_index = text.index("local_admission_guard_eta_percentile:")
    guard_block = text[max(0, key_index - 2_000):key_index]
    # The knob's documentation states both the default and the rollback (0).
    assert "default" in guard_block
    assert "rollback" in guard_block


# ---------------------------------------------------------------------------
# AC1: large prompts priced at >= the configured percentile
# ---------------------------------------------------------------------------

def test_large_prompt_estimate_is_at_least_the_configured_percentile():
    """AC1: with a window whose p75 exceeds the point estimate, the estimator
    prices the 100k cold prompt at the p75 of the observed first-byte times.

    Window: ms = [150, 150, 150, 150, 600, 900]k. mean = 350k ⇒ point estimate
    = 1.2 × 350k = 420k ms; p75 (linear interp at index 3.75 of 6) =
    150k + 0.75 × (600k − 150k) = 487.5k ms ⇒ the conservative estimate is
    exactly 487.5 s (≥ p75, never the optimistic 420 s).
    """
    _seed_varied(service_ms=(150_000, 150_000, 150_000, 150_000, 600_000, 900_000))
    eta, coeffs = prefill_eta.estimate_prefill_seconds("Qwen3", 100_000, 0.0, _cfg())

    assert eta * 1000.0 == pytest.approx(487_500.0, rel=1e-6)
    assert coeffs.eta_percentile == pytest.approx(75.0)
    assert coeffs.percentile_ms == pytest.approx(487_500.0, rel=1e-6)
    assert coeffs.uplift > 1.0
    # The conservative estimate is the observed p75, not the 1.2 × mean point
    # estimate (420 s) — the point estimate would have admitted this prompt.
    assert eta * 1000.0 > 420_000.0


def test_percentile_pricing_starts_exactly_at_32k_tokens():
    """The conservative pricing (and the p50 floor) applies to large prompts
    >= 32k tokens and not one token below (AC2 boundary)."""
    _seed_varied(
        tokens=40_000,
        service_ms=(80_000, 80_000, 80_000, 80_000, 240_000, 320_000),
    )
    above, coeffs_above = prefill_eta.estimate_prefill_seconds(
        "Qwen3", prefill_eta.CONSERVATIVE_MIN_TOKENS, 0.0, _cfg()
    )
    below, coeffs_below = prefill_eta.estimate_prefill_seconds(
        "Qwen3", prefill_eta.CONSERVATIVE_MIN_TOKENS - 1, 0.0, _cfg()
    )

    # Same (model, bucket) window, so the only difference is the threshold.
    assert coeffs_above.samples == coeffs_below.samples == 6
    assert coeffs_above.uplift > 1.0
    assert coeffs_below.uplift == pytest.approx(1.0)
    assert coeffs_above.percentile_ms is not None
    assert coeffs_below.percentile_ms is None
    # Above the threshold the 40 768-token… 32 768-token prompt is priced at
    # p75 (200 000 ms / 40k = 5.0 ms/token × 32 768 / 1000 = 163.84 s); below
    # it the point estimate (4.4 ms/token) applies.
    assert above * 1000.0 == pytest.approx(
        prefill_eta.CONSERVATIVE_MIN_TOKENS * 5.0, rel=1e-6
    )
    assert below * 1000.0 == pytest.approx(
        (prefill_eta.CONSERVATIVE_MIN_TOKENS - 1) * 4.4, rel=1e-6
    )


# ---------------------------------------------------------------------------
# AC2: the p50 floor + the C2 deadline refusal
# ---------------------------------------------------------------------------

def test_large_bucket_never_prices_below_the_measured_p50():
    """AC2 floor: even with a below-median percentile configured, a large
    prompt is priced at >= the window's measured p50.

    Window: ms = [10, 500, 1000, 1000, 1000, 1000]k. p30 = 750k < p50 = 1000k,
    and the 1.2 × mean point estimate (902k) is itself below p50 — so only the
    floor keeps the price at the measured median.
    """
    _seed_varied(
        service_ms=(10_000, 500_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000)
    )
    eta, coeffs = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 100_000, 0.0, _cfg(local_admission_guard_eta_percentile=30)
    )

    assert eta * 1000.0 == pytest.approx(1_000_000.0, rel=1e-6)
    assert coeffs.percentile_ms == pytest.approx(750_000.0, rel=1e-6)
    assert coeffs.floor_ms == pytest.approx(1_000_000.0, rel=1e-6)
    assert coeffs.uplift > 1.0


def test_conservative_large_prompt_is_refused_under_the_deadline():
    """AC2 + C2: a large cold prompt whose conservative ETA exceeds the 90 s
    deadline is refused at admission (``prefill_eta_exceeds_deadline``), where
    the optimistic point estimate would have admitted it.

    Window: 50k-token cold dispatch, ms = [4, 4, 4, 4, 136, 136]k. Point
    estimate = 1.2 × 48k ms = 57.6 s (within the 60 s budget); p75 =
    4k + 0.75 × 132k = 103k ms = 103 s >= the 90 s deadline ⇒ refused.
    """
    _seed_varied(
        tokens=50_000,
        service_ms=(4_000, 4_000, 4_000, 4_000, 136_000, 136_000),
    )

    conservative = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 50_000, 0.0, "low"
    )
    rollback = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_eta_percentile=0),
        "Qwen3",
        50_000,
        0.0,
        "low",
    )

    assert conservative.admit is False
    assert conservative.reason == "prefill_eta_exceeds_deadline"
    assert conservative.eta_prefill_seconds == pytest.approx(103.0, rel=1e-6)

    assert rollback.admit is True
    assert rollback.reason == "within_budget"
    assert rollback.eta_prefill_seconds == pytest.approx(57.6, rel=1e-6)


# ---------------------------------------------------------------------------
# AC3 / AC4: rollback to the point estimate + small prompts unchanged
# ---------------------------------------------------------------------------

def test_zero_percentile_is_the_rollback_to_the_point_estimate():
    """AC4 rollback: ``local_admission_guard_eta_percentile: 0`` restores the
    exact point-estimate behaviour (1.2 × the fitted mean) for every prompt."""
    _seed_varied(service_ms=(150_000, 150_000, 150_000, 150_000, 600_000, 900_000))
    eta, coeffs = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 100_000, 0.0, _cfg(local_admission_guard_eta_percentile=0)
    )

    # 1.2 × mean(350k ms) = 420 k ms — the pre-change estimator's answer.
    assert eta * 1000.0 == pytest.approx(420_000.0, rel=1e-6)
    assert coeffs.uplift == pytest.approx(1.0)
    assert coeffs.percentile_ms is None
    assert coeffs.floor_ms is None


def test_small_prompt_estimates_and_admissions_are_unchanged():
    """AC3: a < 8k prompt is priced and admitted exactly as before whether the
    knob is on (default 75) or rolled back (0) — even with a spread window."""
    _seed_varied(
        tokens=8_000,
        service_ms=(8_000, 12_000, 16_000, 20_000, 24_000, 28_000),
    )

    eta_on, coeffs_on = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 8_000, 0.0, _cfg()
    )
    eta_off, coeffs_off = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 8_000, 0.0, _cfg(local_admission_guard_eta_percentile=0)
    )
    decision_on = prefill_eta.evaluate_admission(_cfg(), "Qwen3", 8_000, 0.0, "low")
    decision_off = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_eta_percentile=0), "Qwen3", 8_000, 0.0, "low"
    )

    # 1.2 × mean(18k ms) for 8k tokens = 21.6 s in both modes.
    assert eta_on == pytest.approx(eta_off)
    assert eta_on * 1000.0 == pytest.approx(21_600.0, rel=1e-6)
    assert coeffs_on.uplift == coeffs_off.uplift == pytest.approx(1.0)
    assert coeffs_on.percentile_ms is None
    assert coeffs_on.prefill_ms_per_token == coeffs_off.prefill_ms_per_token
    assert decision_on.admit is decision_off.admit is True
    assert decision_on.reason == decision_off.reason == "within_budget"
    assert decision_on.eta_prefill_seconds == pytest.approx(
        decision_off.eta_prefill_seconds
    )
