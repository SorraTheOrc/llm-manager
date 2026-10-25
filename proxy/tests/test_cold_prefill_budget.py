"""Cold-prefill budget enforced on every admission path (LP-0MVN4LBZ500281Z7).

The parent item (LP-0MVN3KMGI009BZ0D) requires that a local prefill which
cannot finish before the prefill deadline is refused **at admission**, on
**every** path, instead of being dispatched and then cancelled — the measured
window (fixture ``prefill_thrash_2026_10_24.jsonl``) showed 125 deadline
cancellations and ~779k tokens computed then discarded, most of them prompts
well over 25k new tokens.

AC mapping: AC1 flat budget + boundary, AC2 refusal precedes dispatch, AC3 all
four admission paths, AC4 deadline-relative p75 pricing, AC5 warm path
unchanged, AC6 fixture replay proves the fix, AC7 observability, AC8 observe
policy.
"""

from __future__ import annotations

import json
import logging

import prefill_thrash_replay_harness as harness
import pytest

from proxy import prefill_eta

BUDGET = 25_000
DEADLINE = 120.0
SAFETY = 0.95
SAFETY_LIMIT = SAFETY * DEADLINE  # 114 s


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _cfg(**server_overrides) -> dict:
    """The shipped post-change admission configuration."""
    server = {
        "session_slot_pool_size": 2,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
        "local_prefill_deadline_seconds": int(DEADLINE),
        "local_cold_prefill_budget_tokens": BUDGET,
        "local_admission_guard_eta_percentile_min_tokens": 20_000,
        "local_admission_guard_budget_safety_fraction": SAFETY,
        "local_admission_guard_policy": "reject",
        "local_admission_guard_min_samples": 5,
        "local_admission_guard_window_samples": 200,
        "local_admission_guard_eta_percentile": 75,
    }
    server.update(server_overrides)
    return {"server": server}


def _seed_cache_cold_dispatches(
    service_ms: float, tokens: int = 40_000, count: int = 6, model: str = "Qwen3"
) -> None:
    """Seed an identical-sample window so the bucket is calibrated."""
    for _ in range(count):
        prefill_eta.record_dispatch_sample(model, tokens, 0.0, service_ms)


# ---------------------------------------------------------------------------
# AC1 — the flat budget, calibration- and budget-independent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("new_tokens", [24_999, 25_000])
def test_budget_boundary_is_inclusive(new_tokens):
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", new_tokens, 0.0, priority="medium"
    )

    assert decision.admit is True
    assert decision.reason != "prefill_budget_exceeded"


def test_one_token_over_the_budget_is_refused():
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 25_001, 0.0, priority="medium"
    )

    assert decision.admit is False
    assert decision.reason == "prefill_budget_exceeded"
    assert decision.new_tokens == 25_001
    assert decision.budget_tokens == BUDGET


def test_budget_prices_new_tokens_not_prompt_size():
    """A 200k prompt whose cached prefix is large is within budget."""
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 200_000, 0.90, priority="medium"
    )

    assert decision.new_tokens == 20_000
    assert decision.admit is True
    assert decision.reason != "prefill_budget_exceeded"


def test_budget_binds_for_high_priority_which_has_no_wait_budget():
    """``high`` never queues (budget 0) but must still be capped (AC1)."""
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 40_000, 0.0, priority="high"
    )

    assert decision.admit is False
    assert decision.reason == "prefill_budget_exceeded"


def test_budget_binds_for_an_uncalibrated_bucket():
    """No samples at all: the old guard was inert, the cap must still fire."""
    assert prefill_eta.resolve_config(_cfg()).cold_prefill_budget_tokens == BUDGET

    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 63_000, 0.0, priority="medium"
    )

    assert decision.calibrated is False
    assert decision.admit is False
    assert decision.reason == "prefill_budget_exceeded"


def test_budget_zero_disables_the_cap():
    """Rollback: budget 0 restores today's behaviour byte-for-byte."""
    config = _cfg(local_cold_prefill_budget_tokens=0)

    decision = prefill_eta.evaluate_admission(
        config, "Qwen3", 93_000, 0.0, priority="medium"
    )

    assert decision.admit is True
    assert decision.reason != "prefill_budget_exceeded"


# ---------------------------------------------------------------------------
# AC4 — deadline-relative pricing (p75 from 20k, safety fraction)
# ---------------------------------------------------------------------------

def test_safety_fraction_limit_is_114s_at_a_120s_deadline():
    guard = prefill_eta.resolve_config(_cfg())

    assert prefill_eta._deadline_budget_seconds(guard) == pytest.approx(SAFETY_LIMIT)


def test_25k_in_the_fast_band_is_admitted():
    """25k at the 20-30k p75 rate (~270 t/s) prices ~93 s < 114 s.

    The queue wait budget is widened here (300 s) so the *separate* wait bound
    cannot mask the deadline-relative bound this criterion is about: with the
    shipped 60 s budget a 25k prompt is already refused by the wait bound, and
    AC4 is specifically the safety-fraction band decision.
    """
    harness.seed_pinned_profile()

    decision = prefill_eta.evaluate_admission(
        _cfg(contention_queue_max_wait_seconds=300),
        harness.MODEL,
        25_000,
        0.0,
        priority="medium",
    )

    assert decision.eta_prefill_seconds < SAFETY_LIMIT
    assert decision.admit is True
    assert decision.reason == "within_budget"


def test_band_slow_25k_is_refused_by_the_safety_fraction():
    """A 25k prompt priced at the slow (30-40k band) rate exceeds 114 s.

    The same prompt size the flat cap admits on a fast band must be refused
    when the measured band is slow — that is what the deadline-relative term
    exists for (parent AC3).
    """
    # ~190 t/s band: 25k / 190 = 132 s.
    _seed_cache_cold_dispatches(service_ms=25_000 / 190.0 * 1000.0, tokens=25_000)

    decision = prefill_eta.evaluate_admission(
        _cfg(contention_queue_max_wait_seconds=300),
        "Qwen3",
        25_000,
        0.0,
        priority="medium",
    )

    assert decision.eta_prefill_seconds > SAFETY_LIMIT
    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_safety_budget"


def test_safety_fraction_zero_disables_the_deadline_relative_bound():
    _seed_cache_cold_dispatches(service_ms=25_000 / 190.0 * 1000.0, tokens=25_000)

    decision = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_budget_safety_fraction=0.0),
        "Qwen3",
        25_000,
        0.0,
        priority="medium",
    )

    assert decision.reason != "prefill_eta_exceeds_safety_budget"


def test_percentile_pricing_engages_from_the_configured_floor():
    """The 20k floor (not the old hard-coded 32 768) prices a 25k prompt.

    A varied window makes p75 differ from the point estimate; with the floor
    at 20 000 the 25k prompt is uplifted, with the legacy 32 768 floor it is
    not.
    """
    for service_ms in (80_000.0, 84_000.0, 88_000.0, 132_000.0, 140_000.0):
        prefill_eta.record_dispatch_sample("Qwen3", 25_000, 0.0, service_ms)

    new_floor, _ = prefill_eta.estimate_prefill_seconds(
        "Qwen3", 25_000, 0.0, _cfg()
    )
    legacy_floor, _ = prefill_eta.estimate_prefill_seconds(
        "Qwen3",
        25_000,
        0.0,
        _cfg(local_admission_guard_eta_percentile_min_tokens=32_768),
    )

    assert new_floor > legacy_floor


# ---------------------------------------------------------------------------
# AC5 — the warm path is unchanged
# ---------------------------------------------------------------------------

def test_warm_prompt_above_the_budget_stays_local():
    """47k prompt / 549 new tokens: the cap counts new tokens only (parent AC5)."""
    harness.seed_pinned_profile()

    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 47_000, 1.0 - (549 / 47_000), priority="medium"
    )

    assert decision.new_tokens == pytest.approx(549, abs=2)
    assert decision.admit is True
    assert decision.reason != "prefill_budget_exceeded"


# ---------------------------------------------------------------------------
# AC3 — every admission path uses the same implementation
# ---------------------------------------------------------------------------

def test_queue_entry_guard_refuses_an_over_budget_request():
    """Path (a): the contention-queue entry guard refuses before queueing."""
    server = pytest.importorskip("proxy.server")
    import proxy.provider as provider

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            provider,
            "_local_gate_occupancy",
            lambda *a, **k: (1, 0),
            raising=False,
        )
        decision = _run_guard(
            provider,
            _cfg(),
            estimated_tokens=40_000,
            free_slot=False,
        )

    assert decision is not None
    assert decision.admit is False
    assert decision.reason == "prefill_budget_exceeded"
    assert server is not None


def test_direct_free_slot_guard_refuses_an_over_budget_request():
    """Path (b): the free-slot (direct) guard refuses the same way."""
    import proxy.provider as provider

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            provider,
            "_local_gate_occupancy",
            lambda *a, **k: (0, 0),
            raising=False,
        )
        decision = _run_guard(
            provider,
            _cfg(),
            estimated_tokens=40_000,
            free_slot=True,
        )

    assert decision is not None
    assert decision.admit is False
    assert decision.reason == "prefill_budget_exceeded"


def test_idle_economic_bypass_does_not_admit_an_over_budget_prompt():
    """Path (c): an idle GPU is not permission to start a doomed prefill."""
    import proxy.provider as provider

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(provider, "_idle_admit_config", lambda cfg: (True, 30.0))
        mp.setattr(provider, "_local_idle_blocker", lambda *a, **k: None)
        import proxy.local_activity as activity

        mp.setattr(activity, "local_idle_seconds", lambda: 500.0)

        verdict = provider._evaluate_idle_admission_override(
            _cfg(), {"llama_model": "Qwen3", "name": "local-qwen3"}, 63_000, 0.0
        )

    assert verdict is not None, "idle capacity applies, so a verdict is returned"
    assert verdict["admitted"] is False
    assert verdict["guard_reason"] == "prefill_budget_exceeded"


def test_idle_economic_bypass_still_admits_a_within_budget_prompt():
    import proxy.provider as provider

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(provider, "_idle_admit_config", lambda cfg: (True, 30.0))
        mp.setattr(provider, "_local_idle_blocker", lambda *a, **k: None)
        import proxy.local_activity as activity

        mp.setattr(activity, "local_idle_seconds", lambda: 500.0)

        verdict = provider._evaluate_idle_admission_override(
            _cfg(), {"llama_model": "Qwen3", "name": "local-qwen3"}, 20_000, 0.0
        )

    assert verdict is not None
    assert verdict["admitted"] is True


def test_reserve_endpoint_refuses_an_over_budget_request():
    """Path (d): ``POST /llama/local/reserve`` shares the same decision."""
    from proxy import admission_reserve

    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 40_000, 0.0, priority="medium"
    )
    # The reserve endpoint calls exactly this function (admission_reserve
    # imports it rather than re-deriving a budget), so the same refusal applies.
    assert admission_reserve.evaluate_admission is prefill_eta.evaluate_admission
    assert decision.admit is False
    assert decision.reason == "prefill_budget_exceeded"


def _run_guard(provider, config, *, estimated_tokens, free_slot):
    import asyncio

    return asyncio.run(
        provider._prefill_admission_guard(
            config,
            {"llama_model": "Qwen3", "name": "local-qwen3"},
            estimated_tokens,
            "session-budget-test",
            "medium",
            None,
            1,
            None,
            None,
            free_slot,
        )
    )


# ---------------------------------------------------------------------------
# AC2 / AC7 — refusal precedes dispatch and is logged with evidence
# ---------------------------------------------------------------------------

def test_budget_refusal_is_logged_with_evidence(caplog):
    """AC7: the refusal rides the existing ``local_admission_guard`` line."""
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 40_000, 0.0, priority="medium"
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        prefill_eta.record_guard_decision(
            decision, provider="local-qwen3", session_id="s-1", model="Qwen3"
        )

    line = next(
        r.getMessage() for r in caplog.records if "local_admission_guard" in r.getMessage()
    )
    assert "reason=prefill_budget_exceeded" in line
    assert "decision=reject" in line
    assert "new_tokens=40000" in line
    assert "budget_tokens=25000" in line
    assert "deadline_s=120.0" in line
    assert "deadline_budget_s=114.0" in line


def test_pre_acceptance_refusal_creates_no_dispatch_record(caplog):
    """AC2: the refusal is a routing decision, not a dispatch-then-cancel.

    The guard returns before any local dispatch record exists, so nothing can
    later be evicted at the deadline (that would log
    ``prefill_deadline_exceeded``).
    """
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", 63_000, 0.0, priority="medium"
    )

    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        prefill_eta.record_guard_decision(
            decision, provider="local-qwen3", session_id="s-2", model="Qwen3"
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any("reason=prefill_budget_exceeded" in m for m in messages)
    assert not any("prefill_deadline_exceeded" in m for m in messages)
    assert not any("lease_released" in m for m in messages)


# ---------------------------------------------------------------------------
# AC8 — ``observe`` logs the verdict without refusing
# ---------------------------------------------------------------------------

def test_observe_policy_logs_the_refusal_without_refusing(caplog):
    config = _cfg(local_admission_guard_policy="observe")

    decision = prefill_eta.evaluate_admission(
        config, "Qwen3", 40_000, 0.0, priority="medium"
    )

    assert decision.admit is True, "observe must not refuse"
    assert decision.reason == "prefill_budget_exceeded"
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        prefill_eta.record_guard_decision(decision, session_id="s-3")
    line = next(
        r.getMessage() for r in caplog.records if "local_admission_guard" in r.getMessage()
    )
    assert "decision=reject" in line
    assert "enforced=false" in line


def test_off_policy_keeps_the_guard_inert():
    decision = prefill_eta.evaluate_admission(
        _cfg(local_admission_guard_policy="off"),
        "Qwen3",
        93_000,
        0.0,
        priority="medium",
    )

    assert decision.admit is True
    assert decision.reason == "policy_off"


# ---------------------------------------------------------------------------
# AC6 — fixture replay proves the fix
# ---------------------------------------------------------------------------

def test_replay_refuses_every_measured_over_budget_prompt():
    """Replaying the measured thrash window admits ZERO >25k cold prompts.

    The pre-change baseline (the sibling fixture test) still passes unchanged:
    this is the same oracle read after the fix.
    """
    meta, records = harness.load_fixture()
    harness.seed_pinned_profile()

    result = harness.replay(records, config=_cfg())
    over_budget = [a for a in result.attempts if a.new_tokens > BUDGET]
    admitted_over_budget = [a for a in over_budget if a.guard_admit]

    assert len(over_budget) > 500, "the window really is dominated by >25k prompts"
    assert admitted_over_budget == [], (
        "every over-budget attempt must be refused with prefill_budget_exceeded"
    )
    assert all(
        a.guard_reason == "prefill_budget_exceeded" for a in over_budget
    )
    # The measured deadline cancellations were dominated by over-budget cold
    # prompts; the residue are within-budget prompts that genuinely lost the
    # GPU-speed race (the accepted residual in the parent item's trade-off).
    cancelled_over_budget = [
        a for a in result.deadline_cancelled if a.new_tokens > BUDGET
    ]
    assert len(cancelled_over_budget) >= 0.75 * len(result.deadline_cancelled)
    assert result.totals["admitted_and_doomed"] == 0
    assert meta["totals"]["deadline_cancelled_attempts"] > 0


def test_replay_keeps_warm_prompts_admissible():
    """The same replay still admits the warm traffic (AC5)."""
    _, records = harness.load_fixture()
    harness.seed_pinned_profile()

    result = harness.replay(records, config=_cfg())
    warm = [a for a in result.attempts if a.cached_ratio >= 0.9]

    assert warm, "the window contains warm traffic"
    assert any(a.guard_admit for a in warm)


def test_replay_json_carries_the_budget_decision_for_audits():
    _, records = harness.load_fixture()
    harness.seed_pinned_profile()

    payload = harness.replay_result_to_json(harness.replay(records[:100], config=_cfg()))

    assert payload["budget_tokens"] == BUDGET
    assert payload["config_deadline_seconds"] == DEADLINE
    json.dumps(payload)
