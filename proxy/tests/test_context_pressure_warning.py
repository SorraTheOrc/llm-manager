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
    context_pressure_ratio,
    should_warn_context_pressure,
    _LOCAL_ROUTING_OUTPUT_HEADROOM,
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
    """Threshold logic for the compaction warning."""

    def test_below_warn_ratio_no_warning(self):
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
        }}
        assert should_warn_context_pressure(30000, config) is False

    def test_at_warn_ratio_warns(self):
        """Effective per-slot = 61440; 0.8 * 61440 = 49152."""
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
        }}
        assert should_warn_context_pressure(49152, config) is True
        assert should_warn_context_pressure(50000, config) is True

    def test_configured_warn_ratio_lowered(self):
        """Operator can lower the warn ratio (e.g. 0.6) for earlier signal."""
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
            "context_pressure_warn_ratio": 0.6,
        }}
        # 0.6 * 61440 = 36864
        assert should_warn_context_pressure(36864, config) is True
        assert should_warn_context_pressure(30000, config) is False

    def test_ratio_zero_disables_warning(self):
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
            "context_pressure_warn_ratio": 0,
        }}
        assert should_warn_context_pressure(100000, config) is False

    def test_no_ctx_size_disables_warning(self):
        assert should_warn_context_pressure(50000, {"server": {}}) is False

    def test_default_ratio_is_08(self):
        """Default warn ratio 0.8 -> session at 80% of per-slot warns."""
        config = {"server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 2,
        }}
        # per-slot 65536 - 4096 headroom = 61440; 0.8 * 61440 = 49152
        assert should_warn_context_pressure(49152, config) is True
        assert should_warn_context_pressure(49151, config) is False
