"""Deadline token capacity: compute, publish, enforce (LP-0MW957RB3000ER79).

Pins parent AC2/AC6 (implementation half): the prefill deadline is exposed as
an explicit number of **cold tokens** (``local_prefill_deadline_seconds * 1000
/ cold_ms_per_token``), published on ``/llama/local/status`` and on the
``local_admission_guard`` decision log line, and enforced at admission with
its own distinct reason (``prefill_eta_exceeds_capacity``).

AC mapping:
- AC1  capacity derivation + publication (payload and log line);
- AC2  distinct refusal reason and the at-capacity / capacity+1 boundary;
- AC3  cold whole-prompt basis when residency is unmeasured, measured new
       tokens when a measured prefill record exists (warm-prompt guardrail);
- AC4  rollbacks: ``observe`` logs-only, ``off`` inert, percentile 0 derives
       from the cold point estimate;
- AC5  fail-open: an uncalibrated cold window or a disabled deadline never
       manufactures a refusal.

Rows assert against the *derivation* (the cold rate the guard prices with),
never against a hard-coded capacity number.
"""

from __future__ import annotations

import logging

import pytest

from proxy import admission_reserve, prefill_eta

_MODEL = "Qwen3"
#: A prompt size in the second context bucket (>= the cold-fit percentile
#: floor of 32 768), so the cold window prices it.
_TOKENS = 40_000
#: Raw cold rate fed to the measured-prefill window. The fitted guard rate
#: includes the conservatism factor (3.9 * 1.2 = 4.68 ms/token).
_RAW_MS_PER_TOKEN = 3.9


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _cfg(**server_overrides) -> dict:
    """Isolate the capacity bound from the queue/deadline-fit rows."""
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
        "local_prefill_deadline_seconds": 90,
    }
    server.update(server_overrides)
    return {"server": server}


def _seed_cold(
    tokens: int = _TOKENS,
    ms_per_token: float = _RAW_MS_PER_TOKEN,
    count: int = 6,
) -> None:
    """Record *count* genuinely-cold MEASURED samples at *ms_per_token*."""
    for _ in range(count):
        prefill_eta.record_cold_measured_sample(
            _MODEL, tokens, 0.0, tokens, tokens * ms_per_token
        )


def _seed_solitary(tokens: int = _TOKENS, ms_per_token: float = _RAW_MS_PER_TOKEN) -> None:
    """Calibrate the (mixed) solitary window so a warm request prices."""
    for _ in range(6):
        prefill_eta.record_dispatch_sample(
            _MODEL, tokens, 0.0, tokens * ms_per_token
        )


def _capacity(cfg: dict, tokens: int = _TOKENS):
    return prefill_eta.deadline_capacity(_MODEL, tokens, cfg)


def _expected_tokens(cfg: dict, tokens: int = _TOKENS) -> int:
    capacity = _capacity(cfg, tokens)
    assert capacity.cold_ms_per_token is not None
    return int(capacity.deadline_seconds * 1000.0 / capacity.cold_ms_per_token)


# ---------------------------------------------------------------------------
# AC1: derivation and publication
# ---------------------------------------------------------------------------

def test_capacity_is_derived_from_the_priced_cold_rate():
    """``deadline_capacity_tokens`` is exactly ``deadline * 1000 / rate`` with
    ``rate`` the cold coefficient the guard prices a cold large prompt with."""
    _seed_cold()
    cfg = _cfg()
    coeffs = prefill_eta.coefficients(_MODEL, _TOKENS, cfg, cached_ratio=0.0)
    assert coeffs.rate_bucket == "cold"
    assert coeffs.calibrated is True
    capacity = _capacity(cfg)
    assert capacity.calibrated is True
    assert capacity.cold_ms_per_token == pytest.approx(
        coeffs.prefill_ms_per_token, rel=1e-9
    )
    assert capacity.tokens == int(
        cfg["server"]["local_prefill_deadline_seconds"]
        * 1000.0
        / coeffs.prefill_ms_per_token
    )


def test_capacity_is_published_on_the_status_estimate():
    """The ``/llama/local/status`` estimate payload carries the capacity and
    the rate it was derived from; the value tracks the derivation, not a
    hard-coded number."""
    _seed_cold()
    cfg = _cfg()
    status = prefill_eta.status_estimate(cfg, _MODEL)
    assert status["deadline_capacity_tokens"] == _expected_tokens(cfg)
    assert status["deadline_capacity_calibrated"] is True
    assert status["deadline_capacity_cold_ms_per_token"] == pytest.approx(
        _capacity(cfg).cold_ms_per_token, rel=1e-6
    )


def test_capacity_appears_on_the_guard_log_line(caplog):
    """The decision log line carries the capacity, the basis and the cold
    rate, so the published payload and the enforced decision cannot drift."""
    _seed_cold()
    cfg = _cfg()
    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, 25_000, 0.95, "low", residency_measured=False
    )
    capacity = _capacity(cfg)
    assert decision.reason == prefill_eta.REASON_DEADLINE_CAPACITY
    assert decision.deadline_capacity_tokens == capacity.tokens

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        prefill_eta.record_guard_decision(
            decision, model=_MODEL, provider="local-llama"
        )
    line = next(r for r in caplog.records if "local_admission_guard" in r.message)
    message = line.getMessage()
    assert f"deadline_capacity_tokens={capacity.tokens}" in message
    assert "capacity_basis=cold" in message
    assert f"cold_ms_per_token={capacity.cold_ms_per_token:.4f}" in message
    assert "reason=prefill_eta_exceeds_capacity" in message
    assert "decision=reject" in message


# ---------------------------------------------------------------------------
# AC2: distinct reason and the capacity boundary
# ---------------------------------------------------------------------------

def test_request_above_capacity_is_refused_with_its_own_reason():
    """A prompt whose cold basis exceeds the capacity is refused with the
    distinct capacity reason (never relabelled as a deadline miss)."""
    _seed_cold()
    _seed_solitary()
    cfg = _cfg()
    capacity_tokens = _expected_tokens(cfg)

    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, capacity_tokens + 1, 0.95, "low", residency_measured=False
    )

    assert decision.admit is False
    assert decision.reason == prefill_eta.REASON_DEADLINE_CAPACITY
    assert decision.reason != "prefill_eta_exceeds_deadline"
    assert decision.deadline_capacity_tokens == capacity_tokens
    assert decision.capacity_priced_tokens == capacity_tokens + 1
    assert decision.active is True


def test_request_at_capacity_is_admitted():
    """Boundary: exactly the capacity tokens is admitted (the comparison is
    ``> capacity``), so an off-by-one cannot start refusing feasible prompts."""
    _seed_cold()
    _seed_solitary()
    cfg = _cfg()
    capacity_tokens = _expected_tokens(cfg)

    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, capacity_tokens, 0.95, "low", residency_measured=False
    )

    assert decision.admit is True
    assert decision.reason != prefill_eta.REASON_DEADLINE_CAPACITY


def test_capacity_reason_is_a_too_large_capacity_verdict():
    """The new reason is classified ``too_large`` (waiting cannot help) and is
    never advertised with a retry timer."""
    from proxy import provider

    assert prefill_eta.REASON_DEADLINE_CAPACITY in (
        admission_reserve.CAPACITY_TOO_LARGE_REASONS
    )
    assert prefill_eta.REASON_DEADLINE_CAPACITY in (
        provider.NOT_TIME_RETRYABLE_REFUSAL_REASONS
    )
    verdict = admission_reserve.local_capacity_verdict(
        reason=prefill_eta.REASON_DEADLINE_CAPACITY
    )
    assert verdict.verdict == admission_reserve.CAPACITY_TOO_LARGE


# ---------------------------------------------------------------------------
# AC3: cold basis when unmeasured, measured new tokens when measured
# ---------------------------------------------------------------------------

def test_unmeasured_residency_prices_the_whole_prompt_cold():
    """No measured residency ⇒ the whole prompt is priced cold, so a prompt
    larger than the capacity is refused even though its ratio claims warmth."""
    _seed_cold()
    _seed_solitary()
    cfg = _cfg()
    capacity_tokens = _expected_tokens(cfg)
    prompt = capacity_tokens * 2

    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, prompt, 0.95, "low", residency_measured=False
    )

    assert decision.admit is False
    assert decision.reason == prefill_eta.REASON_DEADLINE_CAPACITY
    assert decision.capacity_basis == "cold"
    assert decision.capacity_priced_tokens == prompt


def test_measured_residency_prices_the_measured_new_tokens():
    """A measured prefill record prices the measured new tokens, so a large
    measured-warm prompt whose new tokens fit the capacity stays admissible
    (the parent's warm-prompt guardrail)."""
    _seed_cold()
    _seed_solitary()
    cfg = _cfg()
    capacity_tokens = _expected_tokens(cfg)
    prompt = capacity_tokens * 2
    ratio = 0.95
    new_tokens = int(round(prompt * (1.0 - ratio)))
    assert new_tokens < capacity_tokens

    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, prompt, ratio, "low", residency_measured=True
    )

    assert decision.admit is True
    assert decision.reason != prefill_eta.REASON_DEADLINE_CAPACITY
    assert decision.capacity_basis == "measured"
    assert decision.capacity_priced_tokens == new_tokens


def test_no_residency_signal_keeps_the_capacity_bound_inert():
    """``residency_measured=None`` is the legacy path: the bound is not
    enforced, so callers without a residency signal behave byte-for-byte as
    before."""
    _seed_cold()
    _seed_solitary()
    cfg = _cfg()
    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, 100_000, 0.95, "low"
    )
    assert decision.admit is True
    assert decision.reason != prefill_eta.REASON_DEADLINE_CAPACITY
    assert decision.capacity_basis == "legacy"


# ---------------------------------------------------------------------------
# AC4: rollbacks
# ---------------------------------------------------------------------------

def test_observe_policy_logs_and_admits():
    """``local_admission_guard_policy: observe`` records the would-be capacity
    refusal but never refuses."""
    _seed_cold()
    _seed_solitary()
    cfg = _cfg(local_admission_guard_policy="observe")
    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, 25_000, 0.95, "low", residency_measured=False
    )
    assert decision.admit is True
    assert decision.reason == prefill_eta.REASON_DEADLINE_CAPACITY
    assert decision.policy == "observe"
    assert decision.active is True


def test_off_policy_skips_the_capacity_bound():
    """``local_admission_guard_policy: off`` is inert: the capacity is not
    evaluated and no refusal can be manufactured."""
    _seed_cold()
    cfg = _cfg(local_admission_guard_policy="off")
    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, 100_000, 0.95, "low", residency_measured=False
    )
    assert decision.admit is True
    assert decision.reason == "policy_off"
    assert decision.active is False
    assert decision.capacity_basis == "cold"
    assert decision.capacity_priced_tokens == 0


def test_percentile_zero_derives_the_capacity_from_the_point_estimate():
    """``local_admission_guard_eta_percentile: 0`` disables the conservative
    uplift, so the capacity is derived from the (smaller) cold point estimate
    and is therefore larger, byte-for-byte the documented rollback."""
    # A varied window so the p75 uplift is material.
    for ms in (100_000, 150_000, 200_000, 250_000, 300_000, 350_000):
        prefill_eta.record_cold_measured_sample(
            _MODEL, _TOKENS, 0.0, _TOKENS, float(ms)
        )
    uplifted = _capacity(_cfg())
    rolled_back = _capacity(_cfg(local_admission_guard_eta_percentile=0))

    point = prefill_eta.coefficients(
        _MODEL,
        _TOKENS,
        _cfg(local_admission_guard_eta_percentile=0),
        cached_ratio=0.0,
    )
    assert point.uplift == 1.0
    assert rolled_back.cold_ms_per_token == pytest.approx(
        point.prefill_ms_per_token, rel=1e-9
    )
    assert rolled_back.tokens == int(90 * 1000.0 / point.prefill_ms_per_token)
    assert rolled_back.tokens >= uplifted.tokens


# ---------------------------------------------------------------------------
# AC5: fail-open
# ---------------------------------------------------------------------------

def test_uncalibrated_cold_window_never_refuses():
    """Without cold samples the capacity is informational only; the guard is
    inert and admits exactly as an uncalibrated bucket does today."""
    cfg = _cfg()
    capacity = _capacity(cfg)
    assert capacity.calibrated is False
    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, 100_000, 0.95, "low", residency_measured=False
    )
    assert decision.admit is True
    assert decision.reason != prefill_eta.REASON_DEADLINE_CAPACITY
    assert decision.deadline_capacity_calibrated is False


def test_disabled_deadline_yields_no_capacity():
    """``local_prefill_deadline_seconds: 0`` disables the deadline, so there is
    no capacity to enforce or publish."""
    _seed_cold()
    cfg = _cfg(local_prefill_deadline_seconds=0)
    capacity = _capacity(cfg)
    assert capacity.tokens is None
    status = prefill_eta.status_estimate(cfg, _MODEL)
    assert status["deadline_capacity_tokens"] is None
    decision = prefill_eta.evaluate_admission(
        cfg, _MODEL, 100_000, 0.95, "low", residency_measured=False
    )
    assert decision.admit is True
    assert decision.reason != prefill_eta.REASON_DEADLINE_CAPACITY
    assert decision.deadline_capacity_tokens is None
