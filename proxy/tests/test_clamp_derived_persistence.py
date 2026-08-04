"""
Tests for clamp-derived persistence cap and token_estimate_multiplier consistency.

Regression for LP-0MSEGPO77005CYCQ (ctx-size evaluation follow-up):
- session_slot_max_prompt_tokens must be derived dynamically from the
  effective per-slot clamp (local_model_ctx_size // active_slots - headroom)
  so it auto-adapts to slot-count/ctx-size changes.
- token_estimate_multiplier must be applied consistently in BOTH routing
  (provider.py) and persistence (session.py) estimate paths.

These tests are expected to FAIL (red) against the current code and pass
after the production changes land (LP-0MSEQR4IU008IE0D + LP-0MSEQR964003C30S).
"""

import pytest
from proxy.provider import (
    effective_per_slot_threshold,
    _estimate_prompt_tokens_for_routing,
)

# cl100k (tiktoken) ≈ 1 token per 8 chars of dense prose
# 80K chars ≈ 10000 tokens


def _body_for_tokens(n_tokens: int) -> dict:
    """Build a request body whose tiktoken estimate is ≈ *n_tokens*."""
    return {"messages": [{"role": "user", "content": "x" * (n_tokens * 8)}]}


# ===================================================================
# AC1: effective_per_slot_threshold live-schedule variants
# ===================================================================


class TestEffectivePerSlotThresholdLiveSchedule:
    """Verify effective_per_slot_threshold values for the live schedule."""

    def test_option_c_day_3_slot(self):
        """(131072, 3) → 39594 (Option C day 3-slot)."""
        assert effective_per_slot_threshold(131072, 3) == 39594

    def test_night_8_slot(self):
        """(131072, 8) → 12288 (night 8-slot)."""
        assert effective_per_slot_threshold(131072, 8) == 12288

    def test_night_8_slot_262k_ctx(self):
        """(262144, 8) → 28672 (night 8-slot, 262K ctx variant)."""
        assert effective_per_slot_threshold(262144, 8) == 28672


# ===================================================================
# AC2: 60K fixture exceeds 4x65.5K clamp with multiplier
# ===================================================================


class TestSixtyKFixtureRoutingWithMultiplier:
    """Prove the 60K fixture (tiktoken est 47680 * 1.69 ≈ 80579) exceeds
    the 4x65.5K-class clamp (61440), so routing bypasses local to remote."""

    def test_60k_fixture_exceeds_4x_clamp_with_multiplier(self):
        """tiktoken estimate for the 60K fixture × 1.69 > 4×65.5K clamp.

        The 60K fixture has tiktoken est ~47680 tokens. With multiplier 1.69:
          47680 * 1.69 = 80579 > 61440 (the 4x65.5K clamp)
        so routing should bypass local.

        Without the multiplier the tiktoken est 47680 < 61440 would pass
        the clamp, causing the HTTP 400 that triggered LP-0MSAOQTJS000FFVM.
        """
        tiktoken_est = 47680
        multiplier = 1.69
        clamp_4x = effective_per_slot_threshold(262144, 4)  # 61440

        multiplied = int(tiktoken_est * multiplier)
        assert multiplied > clamp_4x, (
            f"60K fixture with multiplier ({multiplied}) should exceed "
            f"4x65.5K clamp ({clamp_4x}) to avoid HTTP 400"
        )

    def test_60k_fixture_without_multiplier_passes_clamp(self):
        """Without multiplier the tiktoken est passes the 4x clamp — this is
        the bug that causes HTTP 400 (proves the need for the multiplier)."""
        tiktoken_est = 47680
        clamp_4x = effective_per_slot_threshold(262144, 4)  # 61440
        assert tiktoken_est < clamp_4x, (
            "Without multiplier, tiktoken estimate should pass the 4x clamp"
        )


# ===================================================================
# AC3: Persistence estimate path applies the same multiplier
# ===================================================================


class TestPersistenceEstimateMultiplierConsistency:
    """Prove the persistence estimate path applies the same multiplier
    as routing: N * multiplier vs the clamp-derived cap."""

    def test_multiplied_estimate_exceeds_cap(self):
        """A body with tiktoken estimate N is compared against the
        clamp-derived cap as N * multiplier (e.g. 20000 * 1.69 = 33800)."""
        tiktoken_est = 20000
        multiplier = 1.69
        multiplied = int(tiktoken_est * multiplier)

        # With multiplier: 20000 * 1.69 = 33800
        # Clamp-derived cap for Option C (131072, 3) = 39594
        # 33800 < 39594 → should be admitted
        assert multiplied == 33800

        # Without multiplier: 20000 < 39594 → also admitted
        # But a larger body would be misclassified without multiplier:
        # 25000 * 1.69 = 42250 > 39594 → should be rejected
        # 25000 without multiplier → 25000 < 39594 → admitted (WRONG)
        larger_est = 25000
        larger_multiplied = int(larger_est * multiplier)
        cap = effective_per_slot_threshold(131072, 3)  # 39594
        assert larger_multiplied > cap, (
            f"25000 * 1.69 = {larger_multiplied} should exceed cap {cap}"
        )
        assert larger_est < cap, (
            "25000 without multiplier should pass the clamp (proves need)"
        )

    def test_estimate_slot_prompt_tokens_applies_server_multiplier(self):
        """_estimate_slot_prompt_tokens applies the server-level
        token_estimate_multiplier when passed in server_config.

        A ~20000-token body (160K chars) with multiplier 1.69 must return
        ≈ 33800 (20000 * 1.69), matching the routing path.

        RED test: current _estimate_slot_prompt_tokens(body_json) takes no
        config and applies no multiplier.
        """
        from proxy.session import _estimate_slot_prompt_tokens

        body = _body_for_tokens(20000)
        server_config = {"token_estimate_multiplier": 1.69}

        result = _estimate_slot_prompt_tokens(body, server_config)

        raw = _estimate_prompt_tokens_for_routing(body)
        assert result == int(raw * 1.69), (
            f"Expected {int(raw * 1.69)} (raw {raw} * 1.69), got {result}"
        )

    def test_estimate_slot_prompt_tokens_default_no_multiplier(self):
        """Without server_config, no multiplier is applied (raw estimate)."""
        from proxy.session import _estimate_slot_prompt_tokens

        body = _body_for_tokens(5000)
        result = _estimate_slot_prompt_tokens(body)
        raw = _estimate_prompt_tokens_for_routing(body)
        assert result == raw


# ===================================================================
# AC4: Static 12288 config value no longer gates persistence
#         when a clamp-derived cap is higher
# ===================================================================


class TestDynamicCapOverridesStaticConfig:
    """Tests assert that the static 12288 config value no longer gates
    persistence when a clamp-derived cap is higher.

    These tests exercise _build_slot_context with session_slot_max_prompt_tokens: 0
    (opt into dynamic derivation) and verify that contexts between 12288 and
    the clamp-derived cap ARE persisted.

    EXPECTED TO FAIL against current code because _build_slot_context reads
    the static config value 12288 directly and treats 0 as "check disabled".
    """

    @pytest.fixture(autouse=True)
    def _clear_slot_registry(self):
        from proxy.session import _slot_owners
        _slot_owners.clear()
        yield
        _slot_owners.clear()

    def _make_config(self, ctx_size=131072, pool_size=3, max_prompt_tokens=0):
        """Build a config dict (slot_schedule slot counts are derived from
        pool_size in these unit tests)."""
        return {
            "session_slot_save_path": "/tmp/slot-cache",
            "session_slot_pool_size": pool_size,
            "local_model_ctx_size": ctx_size,
            "session_slot_max_prompt_tokens": max_prompt_tokens,
            "session_slot_timeout_seconds": 3.0,
        }

    def test_context_between_static_and_dynamic_cap_is_admitted(self):
        """A 20K-token context should be persisted when the clamp-derived
        cap (39594 for Option C 3-slot) is higher than the static 12288.

        With dynamic derivation (session_slot_max_prompt_tokens=0):
          cap = effective_per_slot_threshold(131072, 3) = 39594
          20000 estimated tokens < 39594 → admitted (above old 12288 static).
        """
        from proxy.session import _build_slot_context

        config = self._make_config(ctx_size=131072, pool_size=3, max_prompt_tokens=0)
        body = _body_for_tokens(20000)
        slot_id, filename, _ = _build_slot_context(config, "test-session", body)

        assert slot_id is not None, (
            "Context of ~20K tokens should be persisted when clamp-derived cap "
            f"(39594) > estimate, but got None (persistence skipped)"
        )

    def test_context_above_dynamic_cap_is_rejected(self):
        """A 50K-token context should be rejected when the clamp-derived
        cap (39594) is lower.

        RED test: with session_slot_max_prompt_tokens=0 the current code
        treats the cap as disabled and admits the 50K context.
        """
        from proxy.session import _build_slot_context

        config = self._make_config(ctx_size=131072, pool_size=3, max_prompt_tokens=0)
        body = _body_for_tokens(50000)
        slot_id, filename, _ = _build_slot_context(config, "test-session-2", body)

        assert slot_id is None, (
            "Context of ~50K tokens should be rejected when it exceeds "
            "the clamp-derived cap (39594)"
        )

    def test_static_config_value_still_works_when_set(self):
        """When session_slot_max_prompt_tokens is set to a non-zero value,
        it should still gate persistence (explicit override preserved)."""
        from proxy.session import _build_slot_context

        config = self._make_config(ctx_size=131072, pool_size=3, max_prompt_tokens=12288)
        body = _body_for_tokens(20000)  # 20000 > 12288 → rejected
        slot_id, filename, _ = _build_slot_context(config, "test-session-3", body)

        assert slot_id is None, (
            "When session_slot_max_prompt_tokens is explicitly set to 12288, "
            "a 20K context should be rejected"
        )

    def test_night_8_slot_derived_cap(self):
        """Night schedule: (131072, 8) → cap = 12288. Contexts above
        12288 should be rejected, matching the static value.

        RED test: with session_slot_max_prompt_tokens=0 the current code
        admits the 20K context (check disabled).
        """
        from proxy.session import _build_slot_context

        config = self._make_config(ctx_size=131072, pool_size=8, max_prompt_tokens=0)
        # ~5K tokens → should be admitted (5000 < 12288)
        slot_id, filename, _ = _build_slot_context(
            config, "night-session", _body_for_tokens(5000)
        )
        assert slot_id is not None, (
            "5K tokens should be admitted under night 8-slot cap (12288)"
        )

        # ~20K tokens → should be rejected (20000 > 12288)
        slot_id2, _, _ = _build_slot_context(
            config, "night-session-2", _body_for_tokens(20000)
        )
        assert slot_id2 is None, (
            "20K tokens should be rejected under night 8-slot cap (12288)"
        )

    def test_262k_8_slot_derived_cap(self):
        """(262144, 8) → cap = 28672. A 15K-token context (above old static
        12288) should be persisted under the derived cap."""
        from proxy.session import _build_slot_context

        config = self._make_config(ctx_size=262144, pool_size=8, max_prompt_tokens=0)
        body = _body_for_tokens(15000)  # 15000 < 28672, but > 12288
        slot_id, filename, _ = _build_slot_context(config, "large-ctx-session", body)
        assert slot_id is not None, (
            "15K tokens should be admitted under 262K/8-slot cap (28672), "
            "even though it exceeds the old static 12288 cap"
        )


# ===================================================================
# Multiplier integration: _build_slot_context gate uses multiplied estimate
# ===================================================================


class TestBuildSlotContextAppliesMultiplier:
    """The persistence gate must compare the MULTIPLIED estimate against the
    cap, so a body whose raw tiktoken estimate is under the cap but whose
    Qwen3-native token count exceeds it is correctly rejected."""

    @pytest.fixture(autouse=True)
    def _clear_slot_registry(self):
        from proxy.session import _slot_owners
        _slot_owners.clear()
        yield
        _slot_owners.clear()

    def _make_config(self, ctx_size=131072, pool_size=3, max_prompt_tokens=0):
        return {
            "session_slot_save_path": "/tmp/slot-cache",
            "session_slot_pool_size": pool_size,
            "local_model_ctx_size": ctx_size,
            "session_slot_max_prompt_tokens": max_prompt_tokens,
            "session_slot_timeout_seconds": 3.0,
        }

    def test_multiplied_estimate_rejected_above_cap(self):
        """~25000 raw tokens × 1.69 = 42250 > 39594 (Option C cap) →
        persistence rejected.

        RED test: current _build_slot_context compares the RAW estimate
        (25000 < 39594 → admitted).
        """
        from proxy.session import _build_slot_context

        config = self._make_config_with_multiplier(multiplier=1.69)
        body = _body_for_tokens(25000)
        slot_id, filename, _ = _build_slot_context(config, "mult-session", body)

        assert slot_id is None, (
            "~25000 raw tokens * 1.69 = 42250 > cap 39594 → persistence "
            "should be rejected"
        )

    def _make_config_with_multiplier(self, multiplier: float):
        config = self._make_config(ctx_size=131072, pool_size=3, max_prompt_tokens=0)
        config["token_estimate_multiplier"] = multiplier
        return config
