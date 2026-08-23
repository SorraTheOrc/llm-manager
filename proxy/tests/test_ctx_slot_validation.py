"""
Config-consistency validation for ctx-size / slot-count routing clamp.

Regression for LP-0MSCABA5K0010LW2 (and the original failure LP-0MSAOQTJS000FFVM):
setting [Qwen3] ctx-size = 65536 while the slot schedule remained at 6 slots
silently collapsed the large-context routing threshold to 6826 tokens (65536//6
- 4096 headroom). Every agent session (17K-62K estimated tokens) exceeded the
clamp, so ALL traffic was bypassed to remote providers.

This test module verifies the startup config validation that catches this
misconfiguration before it reaches production.
"""
import pytest
from proxy.provider import (
    _LOCAL_ROUTING_OUTPUT_HEADROOM,
    effective_per_slot_threshold,
    validate_local_routing_config,
)


class TestEffectivePerSlotThreshold:
    """Core computation reused by both _effective_large_context_thresholds and
    the validation function."""

    def test_normal_case(self):
        """ctx=262144, slots=4 → per_slot=65536, threshold=61440."""
        assert effective_per_slot_threshold(262144, 4) == 61440

    def test_zero_slots(self):
        assert effective_per_slot_threshold(131072, 0) == 0

    def test_headroom_exceeded(self):
        """ctx=65536, slots=6 → per_slot=10922, threshold=6826."""
        assert effective_per_slot_threshold(65536, 6) == 6826

    def test_headroom_not_met(self):
        """ctx=4096, slots=1 → per_slot=4096 == headroom → 0 (no room for
        output tokens, clamp is not meaningful)."""
        assert effective_per_slot_threshold(4096, 1) == 0
        assert effective_per_slot_threshold(4000, 1) == 0

    def test_exact_headroom(self):
        """ctx=8192, slots=1 → per_slot=8192, threshold=8192 - 4096 = 4096."""
        assert effective_per_slot_threshold(8192, 1) == 4096


class TestValidateLocalRoutingConfig:
    """Validate the ctx-size / slot-count routing clamp configuration.

    Each test exercises the validation function against a config that mirrors
    what the proxy reads at startup.
    """

    def test_no_ctx_size_skips_validation(self):
        """When local_model_ctx_size is 0 or absent, nothing is validated."""
        config = {"server": {}}
        assert validate_local_routing_config(config) == []

    def test_ctx_65536_slots_6_rejected(self):
        """AC1 / AC3: ctx 65536 with 6 slots → effective = 6826 < 10000."""
        config = {"server": {
            "local_model_ctx_size": 65536,
            "session_slot_pool_size": 6,
        }}
        problems = validate_local_routing_config(config)
        assert len(problems) >= 1
        assert "65536" in problems[0]
        assert "6" in problems[0]
        assert "6826" in problems[0]
        assert "10000" in problems[0]

    def test_ctx_131072_slots_3_passes_option_c(self):
        """Live config (Option C restored per LP-0MSEGPO77005CYCQ F4):
        ctx 131072 with 3 slots → effective = 39594 ≥ 10000."""
        config = {"server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 3,
        }}
        assert validate_local_routing_config(config) == []

    def test_ctx_262144_slots_4_passes(self):
        """Rejected candidate 4x65.5K (no longer live, but still a valid
        combination per validation): ctx 262144 with 4 slots → 61440."""
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
        }}
        assert validate_local_routing_config(config) == []

    def test_schedule_aware_all_entries_checked(self):
        """AC2: ALL slot_schedule entries are checked, not just static pool."""
        config = {"server": {
            "local_model_ctx_size": 65536,
            "session_slot_pool_size": 3,  # ok: 65536//3 - 4096 = 21832
            "slot_schedule": {
                "enabled": True,
                "entries": [
                    {"time": "10:00", "slots": 6},  # bad: 6826
                    {"time": "23:59", "slots": 4},  # bad: 12288
                ],
            },
        }}
        problems = validate_local_routing_config(config)
        # 4 slots → 16384 - 4096 = 12288 ≥ 10000 → ok
        # 6 slots → 10922 - 4096 = 6826 < 10000 → bad
        # So one problem for the 6-slot entry
        assert len(problems) == 1
        assert "6" in problems[0]

    def test_schedule_aware_best_case_ok(self):
        """When all schedule entries pass, no problems reported."""
        config = {"server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 6,  # ok
            "slot_schedule": {
                "enabled": True,
                "entries": [
                    {"time": "10:00", "slots": 6},  # ok: 39594
                    {"time": "23:59", "slots": 4},  # ok: 57832
                ],
            },
        }}
        assert validate_local_routing_config(config) == []

    def test_custom_minimum_threshold(self):
        """AC1: minimum is configurable via min_local_routing_threshold."""
        config = {"server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 3,  # effective = 39594
            "min_local_routing_threshold": 50000,  # higher than 39594
        }}
        problems = validate_local_routing_config(config)
        assert len(problems) == 1
        assert "50000" in problems[0]

    def test_fatal_threshold_configured(self):
        """When min_local_routing_threshold_fatal is true, validation should
        indicate that startup must fail (return flag)."""
        config = {"server": {
            "local_model_ctx_size": 65536,
            "session_slot_pool_size": 6,
            "min_local_routing_threshold": 10000,
            "min_local_routing_threshold_fatal": True,
        }}
        result = validate_local_routing_config(config)
        assert len(result) >= 1
        assert result[0].endswith("FATAL") or "fatal" in result[0].lower()

    def test_min_threshold_zero_disables_check(self):
        """min_local_routing_threshold: 0 disables the minimum check
        (consistent with local_model_ctx_size: 0 disabling the clamp)."""
        config = {"server": {
            "local_model_ctx_size": 65536,
            "session_slot_pool_size": 6,
            "min_local_routing_threshold": 0,
        }}
        assert validate_local_routing_config(config) == []

    def test_no_schedule_only_static_checked(self):
        """Without slot_schedule, only static pool size is validated."""
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
            # no slot_schedule key at all
        }}
        assert validate_local_routing_config(config) == []

    def test_disabled_schedule_only_static_checked(self):
        """Disabled slot_schedule: only static pool size is validated."""
        config = {"server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 4,
            "slot_schedule": {
                "enabled": False,
                "entries": [],
            },
        }}
        assert validate_local_routing_config(config) == []

    def test_flat_config_keys(self):
        """Validation also works with flat config (test compat)."""
        config = {
            "local_model_ctx_size": 65536,
            "session_slot_pool_size": 6,
            "min_local_routing_threshold": 10000,
        }
        problems = validate_local_routing_config(config)
        assert len(problems) >= 1
        assert "6826" in problems[0]

    def test_problem_message_is_helpful(self):
        """Error messages must be actionable: include ctx, slots, effective
        threshold, minimum, and explain the consequence."""
        config = {"server": {
            "local_model_ctx_size": 65536,
            "session_slot_pool_size": 6,
        }}
        problems = validate_local_routing_config(config)
        msg = problems[0]
        assert "ctx-size" in msg.lower() or "65536" in msg
        assert "slot" in msg.lower() or "6" in msg
        assert "threshold" in msg.lower() or "effective" in msg
        assert "remote" in msg.lower() or "bypass" in msg or "bypassed" in msg


class TestValidateLocalRoutingConfigPerPeriodCtx:
    """Per-entry ctx_size validation (LP-0MSLNK96T0018W4D)."""

    def test_entry_ctx_size_checked_independently(self):
        """Each entry's (ctx_size, slots) is validated with ITS ctx_size."""
        config = {"server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 3,  # ok: 131072//3 - 4096 = 39594
            "slot_schedule": {
                "enabled": True,
                "entries": [
                    {"time": "10:00", "slots": 3},
                    # Bad: 65536//6 - 4096 = 6826 < 10000
                    {"time": "23:59", "slots": 6, "ctx_size": 65536},
                ],
            },
        }}
        problems = validate_local_routing_config(config)
        assert len(problems) == 1
        assert "65536" in problems[0]
        assert "6" in problems[0]

    def test_entry_ctx_size_high_ok(self):
        """Night entry 2 slots @ 262144 → 126,976 ≥ minimum → no problem."""
        config = {"server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 3,  # ok
            "slot_schedule": {
                "enabled": True,
                "entries": [
                    {"time": "10:00", "slots": 3},
                    {"time": "23:59", "slots": 2, "ctx_size": 262144},
                ],
            },
        }}
        assert validate_local_routing_config(config) == []

    def test_entries_without_ctx_use_global(self):
        """Entries without ctx_size fall back to the global value."""
        config = {"server": {
            "local_model_ctx_size": 65536,
            "session_slot_pool_size": 3,  # ok: 65536//3 - 4096 = 21832
            "slot_schedule": {
                "enabled": True,
                "entries": [
                    {"time": "10:00", "slots": 6},  # bad: 6826 (global ctx)
                ],
            },
        }}
        problems = validate_local_routing_config(config)
        assert len(problems) == 1
        assert "65536" in problems[0]

    def test_global_zero_but_entry_ctx_enables_clamp(self):
        """A global ctx of 0 (clamp disabled) still validates entries that
        carry their own ctx_size."""
        config = {"server": {
            "local_model_ctx_size": 0,
            "session_slot_pool_size": 3,
            "slot_schedule": {
                "enabled": True,
                "entries": [
                    {"time": "10:00", "slots": 3},
                    # Bad despite global 0: 65536//6 - 4096 = 6826 < 10000
                    {"time": "23:59", "slots": 6, "ctx_size": 65536},
                ],
            },
        }}
        problems = validate_local_routing_config(config)
        assert len(problems) == 1
        assert "65536" in problems[0]


class TestCtxSlotConsistency:
    """AC7: the routing clamp must never admit prompts larger than the real
    per-slot context after llama.cpp's n_ctx rounding (LP-0MSLNK96T0018W4D).

    llama.cpp rounds n_ctx_seq (per-slot context) UP to a multiple of 256:
        n_ctx = GGML_PAD(n_ctx, 256)
        n_ctx_seq = GGML_PAD(n_ctx // n_seq_max, 256)
        n_ctx = n_ctx_seq * n_seq_max
    The proxy's clamp (ctx_size // slots - headroom) must be ≤ the rounded
    per-slot context so a prompt admitted local fits the slot.
    """

    @staticmethod
    def _llama_rounded_per_slot(ctx_size: int, slots: int) -> int:
        """Mimic llama.cpp n_ctx rounding for a given total ctx and slot count."""
        def pad(x, n):
            return ((x + n - 1) // n) * n

        n_ctx = pad(ctx_size, 256)
        n_ctx_seq = pad(n_ctx // slots, 256)
        return n_ctx_seq

    def test_clamp_never_exceeds_rounded_per_slot(self):

        cases = [
            (131072, 3),   # day: real per-slot 43776, clamp 39594
            (262144, 2),   # night: real per-slot 131072, clamp 126976
            (131072, 2),   # 2 slots @ 128K
            (262144, 1),   # 1 slot @ 256K
            (262144, 4),   # 4 slots @ 256K
            (65536, 6),    # degenerate case
        ]
        for ctx_size, slots in cases:
            real_per_slot = self._llama_rounded_per_slot(ctx_size, slots)
            clamp = ctx_size // slots - _LOCAL_ROUTING_OUTPUT_HEADROOM
            assert clamp <= real_per_slot, (
                f"ctx={ctx_size} slots={slots}: clamp {clamp} > real per-slot "
                f"{real_per_slot} — prompts at the clamp would truncate"
            )

    def test_night_config_clamp_leaves_headroom(self):
        """2 slots @ 262144: clamp 126,976 × 2 = 253,952 ≤ 262,144."""
        from proxy.provider import effective_per_slot_threshold

        clamp = effective_per_slot_threshold(262144, 2)
        assert clamp == 126976
        assert clamp * 2 <= 262144
        # A prompt admitted at the clamp + headroom fits the real per-slot ctx.
        assert clamp + 4096 <= self._llama_rounded_per_slot(262144, 2)

    def test_day_config_clamp_leaves_headroom(self):
        """3 slots @ 131072: clamp 39,594 ≤ real per-slot 43,776."""
        from proxy.provider import effective_per_slot_threshold

        clamp = effective_per_slot_threshold(131072, 3)
        assert clamp == 39594
        assert clamp + 4096 <= self._llama_rounded_per_slot(131072, 3)


class TestLiveConfigsValidate:
    """The live mode configs must pass startup validation with the
    mode-aware cold-cache thresholds (LP-0MSOMVOPH004ATAK AC1).

    Fast/default cold is 38000; cheap mode was reverted to 30000 after the
    60000 raise breached guardrails (LP-0MSOMVOPH004ATAK AC6 /
    LP-0MSRM54YO007YG0K AC7), re-raised to 38000 (LP-0MSY0V4ZO002ANPL),
    then raised to 42000 for cheap only (LP-0MT50SMU1005ZAD6 /
    LP-0MT50WCCP000DU00). The warm clamp must still resolve above cold so
    the (cold, warm] band never collapses (LP-0MSI2M5BT004BCDP), and
    validate_local_routing_config must report no problems for either
    profile.
    """

    def _load(self, name: str) -> dict:
        import yaml
        from proxy.mode import proxy_dir

        with open(proxy_dir() / name) as fh:
            return yaml.safe_load(fh)

    @pytest.mark.parametrize("config_file", ["config.yaml", "config-fast.yaml", "config-cheap.yaml"])
    def test_live_configs_pass_validation(self, config_file):
        """Every live profile passes validate_local_routing_config."""
        cfg = self._load(config_file)
        problems = validate_local_routing_config(cfg)
        assert problems == [], f"{config_file}: {problems}"

    def test_fast_mode_cold_below_warm(self):
        """Fast mode: cold 38000 < effective warm clamp 83285 (3x262144,
        LP-0MSY0SDAS0031Y7F)."""
        from proxy.provider import _effective_large_context_thresholds

        cold, warm = _effective_large_context_thresholds(self._load("config-fast.yaml"))
        assert cold == 38000
        assert warm == 83285
        assert cold < warm

    def test_default_mode_cold_below_warm(self):
        """Default profile (config.yaml) mirrors fast: cold 38000 < 83285."""
        from proxy.provider import _effective_large_context_thresholds

        cold, warm = _effective_large_context_thresholds(self._load("config.yaml"))
        assert cold == 38000
        assert warm == 83285
        assert cold < warm

    def test_cheap_mode_cold_below_warm(self):
        """Cheap mode: cold 42000 < effective warm (static pool 2 @ 131072
        → clamp 61440; the scheduled 2×262144 period resolves higher).
        Raised to 42000 per LP-0MT50SMU1005ZAD6 / LP-0MT50WCCP000DU00
        (was 38000 after LP-0MSY0V4ZO002ANPL, 30000 after the 60000 revert)."""
        from proxy.provider import _effective_large_context_thresholds

        cold, warm = _effective_large_context_thresholds(self._load("config-cheap.yaml"))
        assert cold == 42000
        assert warm >= 61440  # min(100000, 262144//2 - 4096) under schedule
        assert cold < warm

    def test_cheap_mode_scheduled_warm_resolves_100000(self, monkeypatch):
        """Cheap mode with the live schedule active (2 slots × 262144):
        warm resolves to 100000 (min(100000, 126976)) — band (42000, 100000]
        non-empty after the raise to 42000 (LP-0MT50SMU1005ZAD6 /
        LP-0MT50WCCP000DU00)."""
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
        cold, warm = _effective_large_context_thresholds(self._load("config-cheap.yaml"))
        assert cold == 42000
        assert warm == 100000
        assert cold < warm
