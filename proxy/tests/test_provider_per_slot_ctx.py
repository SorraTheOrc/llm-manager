"""
Tests for per-slot context-aware routing threshold clamping.

Regression for the misleading "maximum output token limit" truncation bug
(LP-0MSAZXXDY005AWA1): models.ini [global] ctx-size is ignored by llama.cpp
router (only [*] sections are the global preset), so the Qwen3 child ran
without --ctx-size and auto-sized per-slot context to 262144/slots (32-44K),
while routing thresholds (60000 cold / 100000 warm) assumed 65K per slot.
Oversized prompts were routed local and hit context exhaustion
(finish_reason=length), which pi surfaces as the misleading output-limit
error.

The fix clamps the effective routing thresholds to the actual per-slot
context so oversized prompts fall through to remote BEFORE hitting context
exhaustion.
"""
import pytest
from proxy.provider import (
    _effective_large_context_thresholds,
    _get_active_local_slots,
    _should_skip_local,
)


class TestGetActiveLocalSlots:
    def test_returns_pool_size_when_no_schedule(self):
        config = {"server": {"session_slot_pool_size": 6}}
        assert _get_active_local_slots(config) == 6

    def test_defaults_to_one(self):
        assert _get_active_local_slots({"server": {}}) == 1

    def test_flat_config(self):
        config = {"session_slot_pool_size": 4}
        assert _get_active_local_slots(config) == 4

    def test_schedule_overrides_pool_size(self, monkeypatch):
        """When a slot schedule is active, its current slot count wins."""
        config = {"server": {"session_slot_pool_size": 6}}
        sched = type("S", (), {"get_active_slot": lambda self, now=None: 8})()
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "slot_scheduler", sched)
        assert _get_active_local_slots(config) == 8


class TestEffectiveLargeContextThresholds:
    """Effective cold/warm thresholds must clamp to the actual per-slot
    context so oversized prompts fall through to remote before hitting
    context exhaustion."""

    def test_no_ctx_size_keeps_configured_thresholds(self):
        """Without local_model_ctx_size, thresholds pass through unchanged."""
        config = {"server": {
            "local_large_context_cold_cache_threshold": 60000,
            "local_large_context_warm_cache_threshold": 100000,
        }}
        cold, warm = _effective_large_context_thresholds(config)
        assert cold == 60000
        assert warm == 100000

    def test_ctx_size_8_slots_clamps_warm_to_per_slot(self):
        """With total ctx 262144 and 8 slots, per-slot = 32768. The warm
        threshold must be clamped below that (with output headroom) so a
        43K-token prompt is skipped instead of routed local."""
        config = {"server": {
            "local_large_context_cold_cache_threshold": 60000,
            "local_large_context_warm_cache_threshold": 100000,
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 8,
        }}
        cold, warm = _effective_large_context_thresholds(config)
        assert warm < 32768
        assert warm >= 28000  # headroom ~4K

    def test_ctx_size_6_slots_clamps_higher(self):
        """6 slots -> per-slot = 43690, so warm clamps higher than 8-slot."""
        config = {"server": {
            "local_large_context_warm_cache_threshold": 100000,
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 6,
        }}
        _, warm = _effective_large_context_thresholds(config)
        assert 32768 < warm <= 43690

    def test_skips_oversized_prompt_that_previously_truncated(self):
        """Repro of the observed failure: prompt 43,737 tokens, warm cache
        ratio 0.96. Previously (warm=100000) it was routed local and
        truncated at the 43,776-token slot. With clamped thresholds the
        oversized prompt must be skipped."""
        config = {"server": {
            "local_large_context_cold_cache_threshold": 60000,
            "local_large_context_warm_cache_threshold": 100000,
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 8,
        }}
        cold, warm = _effective_large_context_thresholds(config)
        from proxy.provider import update_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=96, prompt_tokens=100)
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7300}]}  # ~43.7K tokens
        assert _should_skip_local(
            "Qwen3", "sess_a", body, cold, warm_cache_threshold=warm
        ) is True
