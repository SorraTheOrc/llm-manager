"""Tests for the auto-derived large-context warm routing threshold.

LP-0MU466L1X003RTHN: the effective warm threshold is resolved through a
3-term model — ``min(physical per-slot clamp, latency-derived economic
ceiling)`` — so a mode/profile change (model context size or slot count)
needs no manual threshold recalculation.

Economic ceiling = ``round(ratio × local_model_ctx_size)``, capped by the
absolute ``local_large_context_warm_cache_threshold`` when > 0. The physical
term is the existing ``local_model_ctx_size // slots - 4096`` clamp
(LP-0MSAZXXDY005AWA1). COLD stays a fixed per-mode economic value so the
(cold, warm] band used by the cached_ratio check never collapses
(LP-0MSI2M5BT004BCDP).

Evidence / design review: LP-0MU466L1X003RTHN (2026-09-30 answers).
"""

import pytest
from proxy.provider import (
    _effective_large_context_thresholds,
    _get_economic_ceiling_ratio,
    effective_large_context_economic_ceiling,
    effective_per_slot_threshold,
)

SHIPPED_RATIO = 0.3815
SHIPPED_CTX = 262144


def _cfg(ctx=SHIPPED_CTX, slots=1, ratio=SHIPPED_RATIO, warm=100000, cold=38000):
    """Build a nested-config test fixture mirroring the production surface."""
    server = {
        "local_model_ctx_size": ctx,
        "session_slot_pool_size": slots,
        "local_large_context_warm_cache_threshold": warm,
        "local_large_context_cold_cache_threshold": cold,
    }
    if ratio is not None:
        server["local_large_context_economic_ceiling_ratio"] = ratio
    return {"server": server}


class TestEconomicCeilingHelper:
    """AC1/AC3: the economic ceiling scales with model ctx and is capped by
    the absolute warm value."""

    def test_ratio_scales_with_ctx(self):
        """0.3815 × 131072 = 50003.968 → 50004 (linear scaling, no warm cap)."""
        cfg = _cfg(ctx=131072, ratio=SHIPPED_RATIO, warm=0)
        assert effective_large_context_economic_ceiling(cfg) == 50004

    def test_capped_by_absolute_warm_override(self):
        """0.3815 × 262144 = 100007.936 → 100008, capped to the absolute 100000."""
        assert effective_large_context_economic_ceiling(_cfg()) == 100000

    def test_ratio_zero_falls_back_to_absolute(self):
        """A ratio of 0 uses the absolute warm value verbatim (legacy)."""
        assert effective_large_context_economic_ceiling(
            _cfg(ratio=0, warm=90000)
        ) == 90000

    def test_both_zero_disables(self):
        """Ratio 0 AND absolute 0 → warm cap disabled (returns 0)."""
        assert effective_large_context_economic_ceiling(_cfg(ratio=0, warm=0)) == 0

    def test_flat_config_keys_supported(self):
        """Flat (non-nested) config keys resolve identically."""
        cfg = {
            "local_model_ctx_size": SHIPPED_CTX,
            "local_large_context_warm_cache_threshold": 100000,
            "local_large_context_economic_ceiling_ratio": SHIPPED_RATIO,
        }
        assert effective_large_context_economic_ceiling(cfg) == 100000

    def test_no_ctx_falls_back_to_absolute(self):
        """No ctx to scale with → the absolute warm value is used."""
        assert effective_large_context_economic_ceiling(
            _cfg(ctx=0, warm=90000)
        ) == 90000

    def test_non_numeric_ratio_treated_as_disabled(self):
        cfg = _cfg(ratio=0, warm=90000)
        cfg["server"]["local_large_context_economic_ceiling_ratio"] = "nonsense"
        assert _get_economic_ceiling_ratio(cfg) == 0.0
        assert effective_large_context_economic_ceiling(cfg) == 90000

    def test_negative_ratio_treated_as_disabled(self):
        assert _get_economic_ceiling_ratio(_cfg(ratio=-1.0)) == 0.0


class TestEffectiveThresholdsThreeTermModel:
    """AC2/AC6: ``_effective_large_context_thresholds`` resolves warm as
    ``min(physical clamp, economic ceiling)``."""

    @pytest.mark.parametrize(
        "slots,expected_warm",
        [(1, 100000), (2, 100000), (3, 83285)],
    )
    def test_shipped_ratio_slot_scaling(self, slots, expected_warm):
        """Shipped config: fast 1-slot = 100000, 2-slot = 100000,
        cheap 3-slot = 83285 (physical clamp binds at 3 slots)."""
        _, warm = _effective_large_context_thresholds(_cfg(slots=slots))
        assert warm == expected_warm

    def test_fast_warm_never_exceeds_economic_ceiling(self):
        """AC6: on the 1-slot profile the physical clamp is 258048, but the
        economic ceiling (100000) must bind — warm may not exceed it."""
        cfg = _cfg(slots=1)
        _, warm = _effective_large_context_thresholds(cfg)
        assert warm == 100000
        assert warm <= effective_large_context_economic_ceiling(cfg)

    def test_ceiling_scales_with_model_ctx(self):
        """AC3: ctx 131072 / 1 slot → warm ≈ 50000 (physical 126976 does not
        bind; the economic ceiling does)."""
        _, warm = _effective_large_context_thresholds(_cfg(ctx=131072, slots=1))
        assert warm == 50004
        assert 49500 <= warm <= 50500

    def test_physical_clamp_binds_when_smaller_than_ceiling(self):
        """The per-slot clamp dominates the economic ceiling at 3 slots."""
        _, warm = _effective_large_context_thresholds(_cfg(slots=3))
        assert warm == effective_per_slot_threshold(SHIPPED_CTX, 3)
        assert warm == 83285

    def test_ratio_zero_uses_absolute_warm_then_clamp(self):
        """Ratio 0 → the absolute warm (200000) still clamps to the physical."""
        _, warm = _effective_large_context_thresholds(
            _cfg(ratio=0, warm=200000, ctx=131072, slots=3)
        )
        assert warm == effective_per_slot_threshold(131072, 3) == 39594

    def test_both_zero_disables_warm(self):
        _, warm = _effective_large_context_thresholds(_cfg(ratio=0, warm=0))
        assert warm == 0

    def test_disabled_clamp_returns_warm_ceiling(self):
        """No local_model_ctx_size → the per-slot clamp is disabled; warm is
        still the economic ceiling (which may be capped by the absolute)."""
        _, warm = _effective_large_context_thresholds(_cfg(ctx=0, slots=0))
        assert warm == 100000

    @pytest.mark.parametrize(
        "slots,cold",
        [(1, 38000), (2, 42000), (3, 42000)],
    )
    def test_band_non_collapse(self, slots, cold):
        """AC4: the (cold, warm] band is non-empty for every shipped profile."""
        c, w = _effective_large_context_thresholds(_cfg(slots=slots, cold=cold))
        assert c == cold
        assert c < w

    def test_cold_is_never_scaled_or_clamped(self):
        """AC4: COLD is a fixed economic value, untouched by ctx/slots."""
        cfg = _cfg(ctx=131072, slots=3, cold=42000)
        cold, _ = _effective_large_context_thresholds(cfg)
        assert cold == 42000


class TestShippedConfigDerivesExpectedValues:
    """AC2/AC7: the shipped config files carry the ratio and resolve to the
    documented effective thresholds (via the production merge path)."""

    @staticmethod
    def _load(name):
        from tests.config_test_utils import load_profile

        return load_profile(name)

    def test_base_config_declares_ratio(self):
        cfg = self._load("config.yaml")
        assert cfg["server"]["local_large_context_economic_ceiling_ratio"] == 0.3815

    def test_fast_profile_effective_warm(self):
        cold, warm = _effective_large_context_thresholds(
            self._load("config-fast.yaml")
        )
        assert cold == 38000
        assert warm == 100000

    def test_cheap_profile_effective_warm(self):
        cold, warm = _effective_large_context_thresholds(
            self._load("config-cheap.yaml")
        )
        assert cold == 42000
        assert warm == 83285


class TestStartupThresholdLogging:
    """AC5: startup logs the computed physical clamp, economic ceiling and
    effective warm/cold thresholds."""

    def test_logs_clamp_ceiling_and_thresholds(self, monkeypatch):
        import proxy.server as srv

        captured: dict[str, str] = {}

        class _FakeLogger:
            def info(self, msg, *args):
                captured["msg"] = msg % args

            def warning(self, *args, **kwargs):  # pragma: no cover - unused
                pass

            def error(self, *args, **kwargs):  # pragma: no cover - unused
                pass

        monkeypatch.setattr(srv, "logger", _FakeLogger(), raising=False)
        srv._log_effective_routing_thresholds(_cfg(slots=1))

        msg = captured.get("msg", "")
        assert "cold=38000" in msg
        assert "warm=100000" in msg
        assert "per_slot_cap=258048" in msg
        assert "economic_ceiling=100000" in msg

    def test_log_is_best_effort_never_raises(self, monkeypatch):
        """A broken logger must not break startup (exceptions swallowed)."""
        import proxy.server as srv

        class _BoomLogger:
            def info(self, *args, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr(srv, "logger", _BoomLogger(), raising=False)
        srv._log_effective_routing_thresholds(_cfg(slots=1))  # must not raise
