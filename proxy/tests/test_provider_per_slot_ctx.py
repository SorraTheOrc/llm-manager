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
            "local_large_context_cold_cache_threshold": 20000,
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
            "local_large_context_cold_cache_threshold": 20000,
            "local_large_context_warm_cache_threshold": 100000,
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 6,
        }}
        cold, warm = _effective_large_context_thresholds(config)
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


class TestColdWarmBandReachable:
    """The cached_ratio routing check (Check 2 in _should_skip_local) must be
    reachable: cold must stay BELOW the clamped warm threshold so the
    (cold, warm] band has non-zero width (LP-0MSI2M5BT004BCDP).

    LP-0MSAZXXDY005AWA1 clamped BOTH thresholds to the per-slot cap, so
    cold == warm collapsed the band and Check 2 became unreachable dead
    code. Only WARM (the hard capacity limit) is clamped now; COLD is the
    economic new-token threshold."""

    def _production_config(self):
        """Mirror the live proxy/config.yaml routing settings (fast-mode
        default profile; cold raised 30000→38000, LP-0MSOMVOPH004ATAK;
        ctx 262144 per operator supersede LP-0MSY0SDAS0031Y7F)."""
        return {"server": {
            "local_large_context_cold_cache_threshold": 38000,
            "local_large_context_warm_cache_threshold": 100000,
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
        }}

    def test_production_config_returns_cold_below_warm(self):
        """AC1: under the production config, cold < warm so the band exists.

        262144 ctx / 3 slots = 87381 per-slot, minus 4096 headroom =
        83285 cap. Cold (38000) must stay below the clamped warm (83285).
        """
        cold, warm = _effective_large_context_thresholds(self._production_config())
        assert cold == 38000
        assert warm == 83285
        assert cold < warm

    def test_only_warm_is_clamped_cold_passes_through(self):
        """COLD is an economic threshold and must NOT be clamped to the
        per-slot cap; only WARM (the hard capacity limit) is clamped."""
        config = {"server": {
            "local_large_context_cold_cache_threshold": 15000,
            "local_large_context_warm_cache_threshold": 100000,
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 3,
        }}
        cold, warm = _effective_large_context_thresholds(config)
        assert cold == 15000  # untouched
        assert warm == 39594  # clamped to per-slot cap
        assert cold < warm

    def test_check2_warm_ratio_routes_local_in_band(self):
        """AC2: a request in the (cold, warm] band with a warm cache (high
        ratio) stays local — Check 2 computes new_tokens = estimated *
        (1 - ratio) <= cold."""
        config = self._production_config()
        cold, warm = _effective_large_context_thresholds(config)
        from proxy.provider import update_cached_ratio

        # In-band prompt (39K est, band (38000, 39594]) with 60% cached ->
        # 15.6K new tokens <= cold 38000, so routes local.
        update_cached_ratio("Qwen3", "band_sess", cached_tokens=60, prompt_tokens=100)
        body = {"messages": [{"role": "user", "content": "x " * 39000}]}  # ~39K est
        assert _should_skip_local(
            "Qwen3", "band_sess", body, cold, warm_cache_threshold=warm
        ) is False

    def test_check2_cold_ratio_bypasses_in_band(self):
        """AC2: the same band request with a cold cache (ratio 0 -> all 39K
        tokens are new) bypasses local: new_tokens > cold."""
        config = self._production_config()
        cold, warm = _effective_large_context_thresholds(config)
        from proxy.provider import update_cached_ratio

        # Cold cache (ratio 0.0): unknown session defaults to conservative.
        body = {"messages": [{"role": "user", "content": "x " * 39000}]}  # ~39K est
        assert _should_skip_local(
            "Qwen3", "never_seen", body, cold, warm_cache_threshold=warm
        ) is True

    def test_check2_warm_ratio_high_new_tokens_bypasses(self):
        """AC2: even with a warm cache, if the uncached (new) token count
        still exceeds cold, local is bypassed inside the band."""
        config = self._production_config()
        cold, warm = _effective_large_context_thresholds(config)
        from proxy.provider import update_cached_ratio

        # 39.5K estimate (in band), ratio 0.03 -> ~38.3K new tokens still
        # > cold 38000, so local is bypassed despite a warm-ish cache.
        update_cached_ratio("Qwen3", "band_sess2", cached_tokens=3, prompt_tokens=100)
        body = {"messages": [{"role": "user", "content": "x " * 39500}]}  # ~39.5K est
        assert _should_skip_local(
            "Qwen3", "band_sess2", body, cold, warm_cache_threshold=warm
        ) is True

    def test_35k_cold_cache_now_routes_local(self):
        """AC3 (fast mode): a cold-cache prompt at ~35K tokens now routes
        local — below the raised cold threshold 38000 (was bypassed at
        30000, LP-0MSOMVOPH004ATAK)."""
        config = self._production_config()
        cold, warm = _effective_large_context_thresholds(config)
        # Cold cache: unknown session, ratio defaults to 0.0 (conservative).
        body = {"messages": [{"role": "user", "content": "x " * 35000}]}  # ~35K est
        assert _should_skip_local(
            "Qwen3", "never_seen", body, cold, warm_cache_threshold=warm
        ) is False

    def test_45k_cold_cache_still_bypasses(self):
        """AC3 (fast mode): a cold-cache prompt at ~45K tokens still
        bypasses local — above the per-slot warm clamp 39594
        (context_too_large, physical capacity; AC2 of
        LP-0MSOMVOPH004ATAK)."""
        config = self._production_config()
        cold, warm = _effective_large_context_thresholds(config)
        body = {"messages": [{"role": "user", "content": "x " * 45000}]}  # ~45K est
        assert _should_skip_local(
            "Qwen3", "never_seen", body, cold, warm_cache_threshold=warm
        ) is True


class TestCheapModeColdThreshold:
    """Cold-cache threshold for the CHEAP profile (LP-0MSOMVOPH004ATAK,
    reverted, then re-raised).

    Cheap mode runs 2 slots x 262144 (per-period ctx override), so the
    effective warm threshold resolves to min(100000, 262144//2 - 4096 =
    126976) = 100000. Post-deploy validation (LP-0MSRM54YO007YG0K AC6/AC7)
    showed the raised cheap cold 60000 breached the cheap queue guardrails on
    two consecutive nights, so it was reverted to 30000, then re-raised to
    38000 (LP-0MSY0V4ZO002ANPL). The safe-level investigation
    (LP-0MT50SMU1005ZAD6) pins cheap cold at 42000 (LP-0MT50WCCP000DU00):
    42K < warm 100000, < boot-transient clamp 61440, and < the do-not-raise
    boundary 46000 — the (42000, 100000] band never collapses (dead-code
    guard LP-0MSI2M5BT004BCDP). Fast/default stay at 38000 (cheap-only
    change).
    """

    def _cheap_config(self):
        """Mirror the live proxy/config-cheap.yaml routing settings (cold
        raised to 42000 per LP-0MT50WCCP000DU00)."""
        return {"server": {
            "local_large_context_cold_cache_threshold": 42000,
            "local_large_context_warm_cache_threshold": 100000,
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 2,
        }}

    def test_cheap_cold_below_effective_warm(self):
        """AC1 (cheap mode): cold 42000 < effective warm.

        The cheap schedule entries carry ctx_size 262144, so the warm
        clamp resolves to min(100000, 262144//2 - 4096) = 100000. Cold
        42000 stays below it; the (42000, 100000] band is non-empty.
        """
        from proxy.provider import _effective_large_context_thresholds

        config = self._cheap_config()
        cold, warm = _effective_large_context_thresholds(config)
        # Static pool 2 @ 131072 → per-slot cap 61440; warm clamps to it.
        assert warm == 61440
        assert cold == 42000
        assert cold < warm

    def test_cheap_cold_below_scheduled_effective_warm(self, monkeypatch):
        """AC1 (cheap mode, scheduled): with the live schedule active
        (2 slots @ 262144), the effective warm resolves to 100000 and cold
        42000 stays below it — the (42000, 100000] band is non-empty."""
        import proxy.server as srv_mod
        from proxy.provider import _effective_large_context_thresholds

        sched = type(
            "S",
            (),
            {
                "get_active_ctx_size": lambda self, now=None: 262144,
                "get_active_slot": lambda self, now=None: 2,
            },
        )()
        monkeypatch.setattr(srv_mod, "slot_scheduler", sched)
        cold, warm = _effective_large_context_thresholds(self._cheap_config())
        assert warm == 100000  # min(100000, 262144//2 - 4096)
        assert cold == 42000
        assert cold < warm

    def test_cheap_cold_below_boot_transient_clamp(self):
        """AC1 (cheap mode): cold 42000 stays below the boot-transient
        clamp 61440 (2x65536 before the catch-up restart,
        LP-0MSMZOAJW002UR2A) so the band never collapses even then."""
        from proxy.provider import effective_per_slot_threshold

        boot_transient_clamp = effective_per_slot_threshold(65536 * 2, 2)
        assert boot_transient_clamp == 61440
        assert 42000 < boot_transient_clamp

    def test_cheap_35k_cold_cache_now_routes_local(self):
        """AC3 (cheap mode): a cold-cache prompt at ~35K tokens routes
        local — below the raised cold 42000, 35K new tokens are under the
        threshold (recaptured band (42K, 100K]; 35K < 42K so it routes
        local, LP-0MT50WCCP000DU00)."""
        from proxy.provider import _effective_large_context_thresholds, _should_skip_local

        config = self._cheap_config()
        cold, warm = _effective_large_context_thresholds(config)
        body = {"messages": [{"role": "user", "content": "x " * 35000}]}  # ~35K est
        assert _should_skip_local(
            "Qwen3", "never_seen", body, cold, warm_cache_threshold=warm
        ) is False

    def test_cheap_45k_cold_cache_still_bypasses(self):
        """AC3 (cheap mode): a cold-cache prompt at ~45K tokens still
        bypasses — above the recaptured cold 42000, so all ~45K tokens are
        new and exceed the threshold (recaptured band (42K, 100K];
        LP-0MT50WCCP000DU00)."""
        from proxy.provider import _effective_large_context_thresholds, _should_skip_local

        config = self._cheap_config()
        cold, warm = _effective_large_context_thresholds(config)
        body = {"messages": [{"role": "user", "content": "x " * 45000}]}  # ~45K est
        assert _should_skip_local(
            "Qwen3", "never_seen", body, cold, warm_cache_threshold=warm
        ) is True

    def test_cheap_110k_cold_cache_still_bypasses(self):
        """AC3 (cheap mode): a cold-cache prompt at ~110K tokens still
        bypasses — above the warm 100000 (context_too_large, physical
        capacity)."""
        from proxy.provider import _effective_large_context_thresholds, _should_skip_local

        config = self._cheap_config()
        cold, warm = _effective_large_context_thresholds(config)
        body = {"messages": [{"role": "user", "content": "x " * 110000}]}  # ~110K est
        assert _should_skip_local(
            "Qwen3", "never_seen", body, cold, warm_cache_threshold=warm
        ) is True

    def test_cheap_cold_cache_bypasses_only_above_cold_in_band(self):
        """AC3 (cheap mode): inside the (42000, 100000] band with a cold
        cache, a prompt whose new tokens exceed cold 42000 still bypasses —
        cold-cache recapture only applies up to the cold threshold."""
        from proxy.provider import _effective_large_context_thresholds, _should_skip_local

        config = self._cheap_config()
        cold, warm = _effective_large_context_thresholds(config)
        # 43K is in (42000, 100000]; cold cache -> 43K new tokens > cold 42000
        body = {"messages": [{"role": "user", "content": "x " * 43000}]}  # ~43K est
        assert _should_skip_local(
            "Qwen3", "never_seen", body, cold, warm_cache_threshold=warm
        ) is True


class TestGetActiveLocalCtxSize:
    """Per-period ctx_size resolution (LP-0MSLNK96T0018W4D)."""

    def test_falls_back_to_config(self):
        """No live scheduler → static local_model_ctx_size from config."""
        from proxy.provider import _get_active_local_ctx_size

        config = {"server": {"local_model_ctx_size": 131072}}
        assert _get_active_local_ctx_size(config) == 131072

    def test_flat_config(self):
        from proxy.provider import _get_active_local_ctx_size

        config = {"local_model_ctx_size": 262144}
        assert _get_active_local_ctx_size(config) == 262144

    def test_zero_when_unset(self):
        from proxy.provider import _get_active_local_ctx_size

        assert _get_active_local_ctx_size({"server": {}}) == 0

    def test_scheduler_override_wins(self, monkeypatch):
        """When the live scheduler exposes a per-period ctx_size it wins."""
        from proxy.provider import _get_active_local_ctx_size

        config = {"server": {"local_model_ctx_size": 131072}}
        sched = type("S", (), {"get_active_ctx_size": lambda self, now=None: 262144})()
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "slot_scheduler", sched)
        assert _get_active_local_ctx_size(config) == 262144

    def test_scheduler_none_falls_back(self, monkeypatch):
        """A scheduler with no per-period ctx falls back to config."""
        from proxy.provider import _get_active_local_ctx_size

        config = {"server": {"local_model_ctx_size": 131072}}
        sched = type("S", (), {"get_active_ctx_size": lambda self, now=None: None})()
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "slot_scheduler", sched)
        assert _get_active_local_ctx_size(config) == 131072


class TestEffectiveLargeContextThresholdsPerPeriod:
    """Thresholds must use the ACTIVE period's (ctx_size, slots)
    (LP-0MSLNK96T0018W4D)."""

    def test_night_period_2slots_262144(self, monkeypatch):
        """Night: 2 slots @ 262144 → per-slot cap 126,976."""
        from proxy.provider import _effective_large_context_thresholds

        config = {"server": {
            "local_large_context_cold_cache_threshold": 60000,
            "local_large_context_warm_cache_threshold": 200000,
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 3,
        }}
        sched = type(
            "S",
            (),
            {
                "get_active_ctx_size": lambda self, now=None: 262144,
                "get_active_slot": lambda self, now=None: 2,
            },
        )()
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "slot_scheduler", sched)
        cold, warm = _effective_large_context_thresholds(config)
        # 262144 // 2 - 4096 = 126976 → warm clamped down to the per-slot cap.
        assert warm == 126976
        assert cold == 60000  # cold stays as the economic threshold

    def test_day_period_3slots_131072(self, monkeypatch):
        """Day: 3 slots @ 131072 → per-slot cap 39,594."""
        from proxy.provider import _effective_large_context_thresholds

        config = {"server": {
            "local_large_context_cold_cache_threshold": 60000,
            "local_large_context_warm_cache_threshold": 200000,
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 3,
        }}
        sched = type(
            "S",
            (),
            {
                "get_active_ctx_size": lambda self, now=None: None,
                "get_active_slot": lambda self, now=None: 3,
            },
        )()
        import proxy.server as srv_mod

        monkeypatch.setattr(srv_mod, "slot_scheduler", sched)
        cold, warm = _effective_large_context_thresholds(config)
        # 131072 // 3 - 4096 = 39594
        assert warm == 39594
