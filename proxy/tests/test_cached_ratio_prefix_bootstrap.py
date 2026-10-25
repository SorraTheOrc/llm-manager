"""Second-turn cache credit: derive the expected prefix reuse from the previous turn.

LP-0MVI2XOTY008Q4UV. The routing ``cached_ratio`` is written from the
*previous* local response's measured ``cached_tokens``. A session's first
request is genuinely cache-cold (nothing to reuse), so ``cached_ratio`` is
recorded as ``0.0`` — and every later request inherited that zero until some
*later* response happened to report a hit. The second turn of a growing
session was therefore always costed as fully cache-cold even though the
first turn's tokens were sitting in the KV slot.

These tests cover the fix:

- AC1: a second turn whose prompt extends the first turn's message list gets
  a non-zero expected ratio (prefix credit from the previous prompt size);
- AC2: a brand-new session still routes cold;
- AC3: a changed head (different system prompt / truncated history) earns no
  credit;
- AC4: the prefill-ETA admission guard's ETA reflects the corrected ratio;
- AC5: measured/restored ratios are preserved, and the recorded credit is
  invalidated when the session's ratio is cleared.
"""

from __future__ import annotations

import pytest
from proxy.provider import (
    _cached_ratio_source,
    _get_cached_ratio,
    _last_cached_ratio,
    _last_prompt_prefix,
    _prefill_admission_guard,
    clear_cached_ratio_for_session,
    expected_cached_ratio,
    record_prompt_prefix,
    set_cached_ratio,
    update_cached_ratio,
)

from proxy import prefill_eta

MODEL = "Qwen3"
SESSION = "sess-bootstrap-1"
SAFETY_MARGIN = 0.9  # slot_cache_ratio.DEFAULT_SAFETY_MARGIN


@pytest.fixture(autouse=True)
def _clean_state():
    """Isolate the process-global ratio/prefix/calibration state between tests."""
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    _last_prompt_prefix.clear()
    prefill_eta.reset()
    yield
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    _last_prompt_prefix.clear()
    prefill_eta.reset()


def _msgs(*texts: str) -> list[dict]:
    return [{"role": "user", "content": t} for t in texts]


class TestPrefixCredit:
    """AC1/AC2/AC3: the expected ratio follows the recorded prompt prefix."""

    def test_second_turn_after_cold_first_turn_is_not_zero(self):
        """AC1: turn 1 measured 0.0 cached; turn 2 still earns prefix credit."""
        first_prompt = _msgs("read the file", "tool result")
        # Turn 1 was the session's first request: genuinely 0 cached tokens.
        update_cached_ratio(MODEL, SESSION, cached_tokens=0, prompt_tokens=2067)
        record_prompt_prefix(MODEL, SESSION, 2067, first_prompt)
        assert _get_cached_ratio(MODEL, SESSION) == 0.0

        second_prompt = _msgs("read the file", "tool result", "next question")
        ratio = expected_cached_ratio(MODEL, SESSION, 5221, second_prompt)

        assert ratio == pytest.approx(SAFETY_MARGIN * 2067 / 5221, abs=1e-6)
        assert ratio > 0.0

    def test_brand_new_session_routes_cold(self):
        """AC2: no local dispatch history means no credit."""
        assert (
            expected_cached_ratio(MODEL, SESSION, 5221, _msgs("hello")) == 0.0
        )

    def test_changed_head_earns_no_credit(self):
        """AC3: a different system prompt must not be credited."""
        record_prompt_prefix(MODEL, SESSION, 2067, _msgs("system A", "turn 1"))
        update_cached_ratio(MODEL, SESSION, cached_tokens=0, prompt_tokens=2067)

        ratio = expected_cached_ratio(
            MODEL, SESSION, 5221, _msgs("system B", "turn 1", "turn 2")
        )

        assert ratio == 0.0

    def test_truncated_history_earns_no_credit(self):
        """AC3: history truncated from the front fails the prefix check."""
        record_prompt_prefix(MODEL, SESSION, 2067, _msgs("turn 1", "turn 2"))
        update_cached_ratio(MODEL, SESSION, cached_tokens=0, prompt_tokens=2067)

        ratio = expected_cached_ratio(
            MODEL, SESSION, 5221, _msgs("turn 2 truncated", "turn 3")
        )

        assert ratio == 0.0

    def test_credit_never_exceeds_the_estimate(self):
        """The recorded prompt cannot credit more than the incoming prompt."""
        record_prompt_prefix(MODEL, SESSION, 5000, _msgs("a", "b"))
        ratio = expected_cached_ratio(MODEL, SESSION, 100, _msgs("a", "b", "c"))
        assert 0.0 <= ratio <= 1.0


class TestMeasuredRatioPreserved:
    """AC5: the measured/restored ratio is a floor, never lowered."""

    def test_measured_warm_ratio_is_kept_without_a_record(self):
        update_cached_ratio(MODEL, SESSION, cached_tokens=68708, prompt_tokens=68708)
        assert expected_cached_ratio(MODEL, SESSION, 69000, _msgs("x")) == 1.0

    def test_measured_ratio_wins_when_higher_than_prefix_credit(self):
        update_cached_ratio(MODEL, SESSION, cached_tokens=5200, prompt_tokens=5221)
        set_cached_ratio(MODEL, SESSION, 0.99, source="restored")
        record_prompt_prefix(MODEL, SESSION, 2067, _msgs("a", "b"))

        ratio = expected_cached_ratio(MODEL, SESSION, 5221, _msgs("a", "b", "c"))

        assert ratio == pytest.approx(0.99)

    def test_clearing_the_session_forgets_the_prefix(self):
        record_prompt_prefix(MODEL, SESSION, 2067, _msgs("a", "b"))
        update_cached_ratio(MODEL, SESSION, cached_tokens=0, prompt_tokens=2067)

        clear_cached_ratio_for_session(SESSION)

        assert expected_cached_ratio(MODEL, SESSION, 5221, _msgs("a", "b", "c")) == 0.0

    def test_zero_estimate_falls_back_to_measured(self):
        update_cached_ratio(MODEL, SESSION, cached_tokens=50, prompt_tokens=100)
        record_prompt_prefix(MODEL, SESSION, 2067, _msgs("a"))
        assert expected_cached_ratio(MODEL, SESSION, 0, _msgs("a", "b")) == 0.5

    def test_configured_margin_scales_the_credit(self):
        record_prompt_prefix(MODEL, SESSION, 2067, _msgs("a", "b"))
        extended = _msgs("a", "b", "c")
        quarter = expected_cached_ratio(
            MODEL, SESSION, 5221, extended,
            server_config={"cached_prefix_credit_margin": 0.25},
        )
        assert quarter == pytest.approx(0.25 * 2067 / 5221, abs=1e-6)

    def test_zero_margin_disables_the_credit(self):
        """Rollback: margin 0.0 restores the pre-fix behaviour exactly."""
        update_cached_ratio(MODEL, SESSION, cached_tokens=0, prompt_tokens=2067)
        record_prompt_prefix(MODEL, SESSION, 2067, _msgs("a", "b"))
        assert (
            expected_cached_ratio(
                MODEL, SESSION, 5221, _msgs("a", "b", "c"),
                server_config={"cached_prefix_credit_margin": 0.0},
            )
            == 0.0
        )

    def test_invalid_records_are_ignored(self):
        """Fail-open: a metric path never raises and never credits garbage."""
        record_prompt_prefix(MODEL, SESSION, 0, _msgs("a"))
        record_prompt_prefix(MODEL, SESSION, 100, [])
        record_prompt_prefix(MODEL, None, 100, _msgs("a"))
        assert _last_prompt_prefix == {}
        assert expected_cached_ratio(MODEL, SESSION, 5221, _msgs("a")) == 0.0


class TestPrefillEtaGuard:
    """AC4: the guard's ETA uses the corrected ratio."""

    @staticmethod
    def _calibrate():
        # A handful of cold dispatches identify the prefill coefficient.
        for _ in range(6):
            prefill_eta.record_dispatch_sample(MODEL, 20000, 0.0, 78000.0)

    @staticmethod
    def _guard_config():
        return {
            "server": {
                "local_admission_guard_policy": "reject",
                "local_admission_guard_min_samples": 1,
                "contention_queue_max_wait_seconds": 60,
                # Isolate the BUDGET bound (the deadline bound is covered by
                # tests/test_prefill_eta_deadline.py, LP-0MVKUKEU80022C3U).
                "local_prefill_deadline_seconds": 0,
            }
        }

    @pytest.mark.asyncio
    async def test_guard_eta_reflects_prefix_credit(self):
        """The wired guard costs the corrected new-token count."""
        self._calibrate()
        config = self._guard_config()
        provider_cfg = {"name": "local-qwen3", "llama_model": MODEL}

        # Baseline: the guard costed as fully cache-cold (today's behaviour).
        cold = prefill_eta.evaluate_admission(config, MODEL, 45028, 0.0, "medium")

        record_prompt_prefix(
            MODEL, SESSION, 41000, _msgs("read the file", "tool result")
        )
        update_cached_ratio(MODEL, SESSION, cached_tokens=0, prompt_tokens=41000)

        # Wired path: turn 2 extends turn 1's message list, so the recorded
        # prefix earns credit and the ETA drops below the budget.
        corrected = await _prefill_admission_guard(
            config,
            provider_cfg,
            45028,
            SESSION,
            "medium",
            messages=_msgs("read the file", "tool result", "next"),
        )

        assert cold.eta_prefill_seconds is not None
        assert cold.reason == "eta_exceeds_budget"
        assert corrected is not None and corrected.eta_prefill_seconds is not None
        assert corrected.eta_prefill_seconds < cold.eta_prefill_seconds
        assert corrected.reason == "within_budget"
