"""Cold-regime calibration rows (LP-0MW957PGJ007MCDH, Q410 test half).

Pins the cold-only prefill fit (parent AC1 and AC5 of LP-0MW7V2D13004ZO65):
the genuinely-cold MEASURED window must be the ONLY rate basis for large
prompts, warm/cache-hit samples must never price the cold rate, and the p75
conservative uplift / rollbacks must behave exactly as documented.

Rows land AFTER the implementation (Q1 producer decision 2026-11-13, option
a): the impl item LP-0MW957Q41002EXE2 ships the cold window and re-bases the
previously-invalidated conservative rows; this module adds the dedicated
cold-fit rows (warm exclusion, predicate boundaries, rollback, no-regression).

AC mapping:
- AC1  warm/cache-hit samples never price the cold rate;
- AC2  cold-sample predicate boundaries (ratio 0.0999/0.10, 31 999/32 000);
- AC3  percentile 0 rollback (byte-for-byte `==`) + observe/off never refuse;
- AC4  small-prompt (< 32k) calibration unchanged (mixed window as today);
- AC5  conservative rows re-based in Q410; a regression guard asserts the
       cold fit differs from a polluted mixed fit.
"""

from __future__ import annotations

import pytest

from proxy import prefill_eta


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _cfg(**server_overrides) -> dict:
    """Isolate the budget bound (deadline covered in the deadline file)."""
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
        "local_prefill_deadline_seconds": 0,
    }
    server.update(server_overrides)
    return {"server": server}


def _seed_cold_varied(
    model: str = "Qwen3",
    tokens: int = 100_000,
    ratio: float = 0.0,
    processed: int | None = None,
    service_ms: tuple[float, ...] = (),
) -> None:
    """Record one genuinely-cold MEASURED sample per *service_ms* value.

    *processed* defaults to *tokens* (a fully-processed cold prefill), so the
    samples satisfy the AC1 predicate (cached ratio < 0.10, processed
    >= 32 000). Callers that need boundary rows pass an explicit *processed*.
    """
    processed = tokens if processed is None else processed
    for ms in service_ms:
        prefill_eta.record_cold_measured_sample(
            model, tokens, ratio, processed, float(ms)
        )


# ---------------------------------------------------------------------------
# AC1: warm/cache-hit samples never price the cold rate
# ---------------------------------------------------------------------------

def test_warm_cache_hits_never_enter_the_cold_window():
    """Instant cache hits (ratio >= 0.90, service < 1 s, processed ~ 0) are
    dropped by the predicate: the cold window keeps only the genuinely-cold
    large samples, so the cold fit cannot be polluted by cache-hit
    optimism."""
    _seed_cold_varied(
        tokens=100_000,
        ratio=0.95,
        processed=5,
        service_ms=(600.0, 700.0, 800.0, 900.0),
    )  # warm cache hits: all below the predicate
    _seed_cold_varied(
        tokens=100_000,
        ratio=0.0,
        service_ms=(100_000 * 4.0,) * 6,
    )  # genuinely-cold large samples

    coeffs = prefill_eta.coefficients("Qwen3", 100_000, _cfg())

    # Only the genuinely-cold samples entered the window.
    assert coeffs.cold_samples == 6
    assert coeffs.calibrated is True
    assert coeffs.rate_bucket == "cold"
    # Fitted from the cold distribution (6.0 ms/token x conservatism), never
    # the instant-hit rate.
    assert coeffs.prefill_ms_per_token == pytest.approx(
        4.0 * prefill_eta.CONSERVATISM_FACTOR, rel=1e-6
    )


def test_cold_fit_is_materially_slower_than_the_mixed_window_hit():
    """The whole point of the change: a genuinely-cold fit is materially more
    pessimistic than the (cache-hit-polluted) mixed window's optimistic rate,
    so a regression to a mixed fit fails loudly."""
    # Mixed window: dominated by instant cache hits (85 751 tokens in 2.6 s).
    for _ in range(6):
        prefill_eta.record_dispatch_sample(
            "Qwen3", 100_000, 0.0, 2_600.0
        )
    # Cold window: measured cold large prefills at 6.0 ms/token.
    _seed_cold_varied(
        tokens=100_000,
        service_ms=(100_000 * 6.0,) * 6,
    )
    mixed = prefill_eta.coefficients("Qwen3", 100_000, _cfg())
    assert mixed.rate_bucket == "cold"  # large cold prompts price cold
    assert mixed.prefill_ms_per_token > 5.0  # 6.0*1.2 = 7.2: slow, honest
    # A fit that secretly used the instant-hit window would report ~2.6*1.2
    # ms/token — materially optimistic. Assert the cold rate is well above it.
    assert mixed.prefill_ms_per_token > 2.6 * prefill_eta.CONSERVATISM_FACTOR * 1.5


# ---------------------------------------------------------------------------
# AC2: cold-sample predicate boundaries
# ---------------------------------------------------------------------------

def test_predicate_ratio_boundary_in_even_at_0999_excludes_010():
    _seed_cold_varied(tokens=100_000, ratio=0.0999, processed=100_000,
                      service_ms=(100_000 * 4.0,))
    _seed_cold_varied(tokens=100_000, ratio=0.10, processed=100_000,
                      service_ms=(100_000 * 4.0,))
    coeffs = prefill_eta.coefficients("Qwen3", 100_000, _cfg())
    # 0.0999 admitted, 0.10 excluded.
    assert coeffs.cold_samples == 1


def test_predicate_processed_boundary_31999_out_32000_in():
    _seed_cold_varied(tokens=100_000, ratio=0.0, processed=31_999,
                      service_ms=(100_000 * 4.0,))
    _seed_cold_varied(tokens=100_000, ratio=0.0, processed=32_000,
                      service_ms=(100_000 * 4.0,))
    coeffs = prefill_eta.coefficients("Qwen3", 100_000, _cfg())
    assert coeffs.cold_samples == 1


def test_predicate_excludes_sub_second_first_byte():
    _seed_cold_varied(tokens=100_000, ratio=0.0, processed=100_000,
                      service_ms=(999.0,))
    coeffs = prefill_eta.coefficients("Qwen3", 100_000, _cfg())
    assert coeffs.cold_samples == 0
    assert coeffs.rate_bucket == "cold"  # inert, never a mixed fallback


# ---------------------------------------------------------------------------
# AC3: rollback rows
# ---------------------------------------------------------------------------

def test_percentile_zero_restores_the_cold_point_estimate_byte_for_byte():
    """``eta_percentile: 0`` returns the cold point estimate exactly (`==`,
    not approximately), byte-for-byte the conservative-pricing rollback."""
    _seed_cold_varied(
        tokens=100_000,
        service_ms=(150_000, 150_000, 150_000, 150_000, 600_000, 900_000),
    )
    coeffs = prefill_eta.coefficients(
        "Qwen3", 100_000, _cfg(local_admission_guard_eta_percentile=0)
    )
    assert coeffs.uplift == 1.0
    assert coeffs.percentile_ms is None
    assert coeffs.floor_ms is None
    # The point estimate: the two-coefficient least-squares on the cold
    # samples (coded ~ 0 default, prefill carries the full cold rate),
    # x the conservatism factor — byte-for-byte the pre-uplift fit.
    assert coeffs.coded_ms_per_token == prefill_eta.DEFAULT_CODED_MS_PER_TOKEN
    assert coeffs.prefill_ms_per_token == 4.2
    # Contrast: the p75 path uplifts the same window (the conservative-ETA
    # change still applies, now to the cold distribution).
    uplifted = prefill_eta.coefficients("Qwen3", 100_000, _cfg())
    assert uplifted.uplift > 1.0
    assert uplifted.prefill_ms_per_token > coeffs.prefill_ms_per_token


def test_observe_policy_never_refuses_a_cold_large_prompt():
    cfg = _cfg(local_prefill_deadline_seconds=90, local_admission_guard_policy="observe")
    _seed_cold_varied(tokens=60_000, service_ms=(60_000 * 6.0,) * 6)
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 60_000, 0.0, "low")
    assert decision.admit is True  # observe never refuses
    assert decision.policy == "observe"
    # The verdict is still active and recorded (a would-be refusal).
    assert decision.active is True


def test_off_policy_skips_the_guard_for_a_cold_large_prompt():
    cfg = _cfg(local_prefill_deadline_seconds=90, local_admission_guard_policy="off")
    _seed_cold_varied(tokens=60_000, service_ms=(60_000 * 6.0,) * 6)
    decision = prefill_eta.evaluate_admission(cfg, "Qwen3", 60_000, 0.0, "low")
    assert decision.admit is True
    assert decision.reason == "policy_off"
    assert decision.active is False


# ---------------------------------------------------------------------------
# AC4: small-prompt calibration unchanged
# ---------------------------------------------------------------------------

def test_small_prompt_below_32000_keeps_the_mixed_window_calibration():
    """Prompts below the percentile floor keep the existing mixed-window
    calibration byte-for-byte (parent AC1: 'small-prompt calibration must not
    be degraded')."""
    for _ in range(6):
        prefill_eta.record_dispatch_sample("Qwen3", 8_000, 0.0, 8_000 * 4.0)
    coeffs = prefill_eta.coefficients("Qwen3", 8_000, _cfg())
    assert coeffs.rate_bucket == "solitary"
    assert coeffs.calibrated is True
    assert coeffs.prefill_ms_per_token == pytest.approx(
        4.0 * prefill_eta.CONSERVATISM_FACTOR, rel=1e-6
    )
    # The cold window is not involved for a small prompt.
    assert coeffs.cold_samples == 0


# ---------------------------------------------------------------------------
# AC5: no-regression guard — the decision reports the honest split
# ---------------------------------------------------------------------------

def test_admission_decision_carries_cold_samples_and_bucket():
    """The reason an inert cold admission (or a calibrated cold refusal)
    happened is legible: ``rate_bucket=cold`` + ``cold_samples`` on the
    decision (the row that made the honest split observable)."""
    # Uncalibrated cold window: large cold prompt is inert (admit), and the
    # reason is insufficient cold samples — never a mixed-window price.
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 100_000, 0.0, "low"
    )
    assert decision.admit is True
    assert decision.active is False
    assert decision.reason == "insufficient_samples"
    assert decision.rate_bucket == "cold"
    assert decision.cold_samples == 0

    # Calibrated cold window: refuses on the budget bound, cold bucket.
    _seed_cold_varied(tokens=100_000, service_ms=(100_000 * 3.9,) * 6)
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 100_000, 0.0, "low"
    )
    assert decision.admit is False
    assert decision.reason == "eta_exceeds_budget"
    assert decision.rate_bucket == "cold"
    assert decision.cold_samples == 6
