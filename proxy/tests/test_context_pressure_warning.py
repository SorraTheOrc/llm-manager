"""
Context-pressure warning for session compaction.

Regression for LP-0MSDCLQ2W001LGWC: sessions averaging ~57K tokens (up to
437K) push the local Qwen3 decode speed to 0.5-22 tok/s because KV reads
scale linearly with context (20 KB/token at f16). Compaction is performed
by the agents, not the proxy, so the proxy's job is to *signal* when a
session's context approaches the per-slot limit, giving operators and
agents a data-backed prompt to compact before decode degrades.

This module verifies the warning computation used at routing time.
"""
import pytest
from proxy.provider import (
    _LOCAL_ROUTING_OUTPUT_HEADROOM,
    context_pressure_ratio,
    should_warn_context_pressure,
)


class TestContextPressureRatio:
    """Fraction of the effective per-slot context consumed by a session."""

    def test_normal_session(self):
        """30K of 61.4K effective per-slot -> 0.49, below 0.8 warn ratio."""
        ctx_size, slots = 262144, 4
        assert context_pressure_ratio(30000, ctx_size, slots) == pytest.approx(
            30000 / (262144 // 4 - _LOCAL_ROUTING_OUTPUT_HEADROOM)
        )

    def test_zero_ctx_disabled(self):
        """ctx_size=0 disables the clamp -> ratio 0 (no warning)."""
        assert context_pressure_ratio(30000, 0, 4) == 0.0

    def test_zero_slots_safe_default(self):
        """slots<=0 makes the computation meaningless -> 0.0, no crash."""
        assert context_pressure_ratio(30000, 262144, 0) == 0.0

    def test_oversized_session(self):
        """70K of 61.4K per-slot -> ratio > 1 (way over the slot)."""
        ratio = context_pressure_ratio(70000, 262144, 4)
        assert ratio > 1.0


class TestShouldWarnContextPressure:
    """Threshold logic for the compaction advisory.

    LP-0MTVXP7DG00613ZB AC3: the advisory is unified with the single
    ``compaction_trigger_ratio`` detection knob (default 0.70); the legacy
    ``context_pressure_warn_ratio`` key is ignored.
    """

    def test_below_trigger_no_warning(self):
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
        }}
        assert should_warn_context_pressure(30000, config) is False

    def test_at_trigger_warns(self):
        """Effective per-slot = 61440; trigger 0.70 * 61440 = 43008."""
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
        }}
        # Strictly above the trigger, matching ``should_compact_session``.
        assert should_warn_context_pressure(43008, config) is False
        assert should_warn_context_pressure(43009, config) is True
        assert should_warn_context_pressure(50000, config) is True

    def test_configured_trigger_lowers_threshold(self):
        """Operator can lower ``compaction_trigger_ratio`` for earlier signal."""
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
            "compaction_trigger_ratio": 0.6,
        }}
        # 0.6 * 61440 = 36864
        assert should_warn_context_pressure(36864, config) is False
        assert should_warn_context_pressure(36865, config) is True
        assert should_warn_context_pressure(30000, config) is False

    def test_legacy_warn_ratio_key_is_ignored(self):
        """The retired ``context_pressure_warn_ratio`` key no longer lowers it."""
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
            "context_pressure_warn_ratio": 0.1,
        }}
        # 0.1 * 61440 = 6144 would warn, but the trigger (43008) governs.
        assert should_warn_context_pressure(10000, config) is False
        assert should_warn_context_pressure(43009, config) is True

    def test_compaction_disabled_disables_warning(self):
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
            "compaction_trigger_ratio": 0,
        }}
        assert should_warn_context_pressure(100000, config) is False

    def test_no_ctx_size_disables_warning(self):
        assert should_warn_context_pressure(50000, {"server": {}}) is False

    def test_default_trigger_is_070(self):
        """Default compaction trigger 0.70 -> session above 70% of per-slot warns."""
        config = {"server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 2,
        }}
        # per-slot 65536 - 4096 headroom = 61440; 0.70 * 61440 = 43008
        assert should_warn_context_pressure(43009, config) is True
        assert should_warn_context_pressure(43008, config) is False
