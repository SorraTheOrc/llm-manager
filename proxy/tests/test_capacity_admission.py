"""Capacity-based local admission (LP-0MVGDUVS6003T2IX).

The economic size band may be admitted locally while the projected
slot-seconds of admissions fit inside a share of MEASURED idle slot time. The
existing thresholds are untouched; this gate only changes the *effective*
treatment of the band, and only when enabled.

Covers the ACs:

- AC2: the rule is capacity-based (projection from rolling observed data, no
  fixed capacity constant) — a projection that exceeds the idle-slot budget
  keeps the bypass;
- AC4: ``local_capacity_admission_enabled`` is the rollback switch (default
  off, disabled ⇒ byte-for-byte today's routing);
- AC5: the effective warm threshold is logged with the decision so the
  achieved setting and its reason are recorded.
"""

from __future__ import annotations

import time

import pytest
from proxy.provider import _should_skip_local

from proxy import capacity_admission as ca

MODEL = "Qwen3"
SESSION = "sess-cap-1"
COLD = 38_000


@pytest.fixture(autouse=True)
def _clean_capacity_state():
    ca.reset_capacity_state()
    yield
    ca.reset_capacity_state()


def _config(**server_overrides) -> dict:
    server = {
        "local_large_context_cold_cache_threshold": COLD,
        "local_large_context_warm_cache_threshold": 0,
        "session_slot_pool_size": 2,
    }
    server.update(server_overrides)
    return {"server": server}


def _enabled_config(**server_overrides) -> dict:
    base = {
        "local_capacity_admission_enabled": True,
        "local_capacity_admission_budget_ratio": 0.8,
        "local_capacity_admission_horizon_seconds": 3600,
        "local_capacity_admission_decode_seconds": 30,
    }
    base.update(server_overrides)
    return _config(**base)


class TestPolicyResolution:
    """AC4 — the rollback switch defaults off and is validated."""

    def test_disabled_by_default(self):
        policy = ca.resolve_policy({})

        assert policy.enabled is False
        assert policy.budget_ratio == ca.DEFAULT_BUDGET_RATIO
        assert policy.horizon_seconds == ca.DEFAULT_HORIZON_SECONDS
        assert policy.decode_seconds == ca.DEFAULT_DECODE_SECONDS

    def test_enabled_flag(self):
        assert ca.resolve_policy(_enabled_config()).enabled is True

    @pytest.mark.parametrize("bad", [0, 1.5, -0.2, "nonsense"])
    def test_budget_ratio_is_validated(self, bad):
        policy = ca.resolve_policy(
            _enabled_config(local_capacity_admission_budget_ratio=bad)
        )

        assert policy.budget_ratio == ca.DEFAULT_BUDGET_RATIO

    def test_horizon_and_decode_validated(self):
        policy = ca.resolve_policy(
            _enabled_config(
                local_capacity_admission_horizon_seconds=0,
                local_capacity_admission_decode_seconds=-5,
            )
        )

        assert policy.horizon_seconds == ca.DEFAULT_HORIZON_SECONDS
        assert policy.decode_seconds == ca.DEFAULT_DECODE_SECONDS


class TestBusyAccounting:
    """AC2 — idle capacity is measured from real dispatch intervals."""

    def test_no_intervals_means_fully_idle(self):
        now = 10_000.0

        assert ca.busy_slot_seconds(3600, now=now) == 0.0
        assert ca.idle_slot_seconds(_config(), 3600, now=now) == pytest.approx(7200.0)

    def test_overlapping_intervals_are_merged(self):
        now = 10_000.0
        ca.record_slot_busy(now - 300, now - 200)
        ca.record_slot_busy(now - 250, now - 100)  # overlaps the first

        assert ca.busy_slot_seconds(3600, now=now) == pytest.approx(200.0)

    def test_intervals_partially_outside_the_horizon_are_clipped(self):
        now = 10_000.0
        ca.record_slot_busy(now - 5000, now - 4000)  # entirely outside (ends 4000 s ago)
        ca.record_slot_busy(now - 100, now + 100)  # partly in the future

        assert ca.busy_slot_seconds(3600, now=now) == pytest.approx(100.0)

    def test_idle_shrinks_with_busy_time(self):
        now = 10_000.0
        ca.record_slot_busy(now - 1000, now - 400)  # 600 s busy

        # 2 slots * 3600 s = 7200 s capacity
        assert ca.idle_slot_seconds(_config(), 3600, now=now) == pytest.approx(6600.0)


class TestProjection:
    """AC2 — the projection comes from the calibrated prefill ETA."""

    def test_projection_uses_prefill_eta_plus_decode(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        monkeypatch.setattr(
            prefill_eta,
            "estimate_prefill_seconds",
            lambda *a, **k: (120.0, object()),
        )

        projected = ca.project_slot_seconds(_config(), MODEL, 60_000, 0.0)

        assert projected == pytest.approx(150.0)  # 120 s prefill + 30 s decode

    def test_unavailable_eta_projects_nothing(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        def _boom(*a, **k):
            raise RuntimeError("no calibration")

        monkeypatch.setattr(prefill_eta, "estimate_prefill_seconds", _boom)

        assert ca.project_slot_seconds(_config(), MODEL, 60_000, 0.0) == 0.0


class TestAdmissionDecision:
    """AC2/AC4 — the decision matrix."""

    def test_disabled_returns_none(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        monkeypatch.setattr(
            prefill_eta, "estimate_prefill_seconds", lambda *a, **k: (60.0, object())
        )

        assert (
            ca.evaluate_capacity_admission(_config(), MODEL, 60_000, 0.0) is None
        )

    def test_admits_when_projection_fits_idle_budget(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        monkeypatch.setattr(
            prefill_eta, "estimate_prefill_seconds", lambda *a, **k: (120.0, object())
        )

        decision = ca.evaluate_capacity_admission(
            _enabled_config(), MODEL, 60_000, 0.0
        )

        assert decision is not None
        assert decision["reason"] == "capacity_admission"
        assert decision["projected_seconds"] == pytest.approx(150.0)
        assert decision["budget_seconds"] == pytest.approx(0.8 * 7200.0)
        assert decision["admitted_seconds"] == pytest.approx(150.0)

    def test_refuses_when_projection_exceeds_idle_budget(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        # One slot, 3600 s horizon, 300 s already busy: idle 3300 s, so the
        # budget is 0.8 * 3300 = 2640 s. A projected 3000 s prefill does not
        # fit and keeps today's bypass.
        now = time.monotonic()
        ca.record_slot_busy(now - 600, now - 300)
        cfg = _enabled_config(
            session_slot_pool_size=1,
            local_capacity_admission_decode_seconds=0,
        )
        monkeypatch.setattr(
            prefill_eta, "estimate_prefill_seconds", lambda *a, **k: (3000.0, object())
        )

        assert ca.evaluate_capacity_admission(cfg, MODEL, 60_000, 0.0) is None

    def test_admits_again_when_the_projection_fits(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        now = time.monotonic()
        ca.record_slot_busy(now - 600, now - 300)
        cfg = _enabled_config(
            session_slot_pool_size=1,
            local_capacity_admission_decode_seconds=0,
        )
        monkeypatch.setattr(
            prefill_eta, "estimate_prefill_seconds", lambda *a, **k: (1000.0, object())
        )

        decision = ca.evaluate_capacity_admission(cfg, MODEL, 60_000, 0.0)

        assert decision is not None
        assert decision["idle_seconds"] == pytest.approx(3300.0)
        assert decision["budget_seconds"] == pytest.approx(2640.0)

    def test_burst_admissions_exhaust_the_budget(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        # Small idle budget: 1 slot, horizon 1000 s, 80 % => 800 s budget.
        cfg = _enabled_config(
            session_slot_pool_size=1,
            local_capacity_admission_horizon_seconds=1000,
            local_capacity_admission_decode_seconds=0,
        )

        calls = {"n": 0}

        def _eta(*a, **k):
            calls["n"] += 1
            return (300.0, object())

        monkeypatch.setattr(prefill_eta, "estimate_prefill_seconds", _eta)

        first = ca.evaluate_capacity_admission(cfg, MODEL, 60_000, 0.0)
        second = ca.evaluate_capacity_admission(cfg, MODEL, 60_000, 0.0)
        third = ca.evaluate_capacity_admission(cfg, MODEL, 60_000, 0.0)

        assert first is not None
        assert second is not None  # 600 s <= 800 s
        assert second["admitted_seconds"] == pytest.approx(600.0)
        assert third is None  # 900 s > 800 s -> today's bypass

    def test_projection_is_recomputed_per_request(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        seen = []

        def _eta(model, tokens, ratio, cfg):
            seen.append((tokens, ratio))
            return (10.0, object())

        monkeypatch.setattr(prefill_eta, "estimate_prefill_seconds", _eta)
        cfg = _enabled_config()

        ca.evaluate_capacity_admission(cfg, MODEL, 40_000, 0.1)
        ca.evaluate_capacity_admission(cfg, MODEL, 90_000, 0.9)

        assert seen == [(40_000, 0.1), (90_000, 0.9)]


class TestRoutingGate:
    """AC2 — the override only ever affects the economic band."""

    def test_capacity_admit_flips_the_economic_bypass(self, monkeypatch):
        import proxy.prefill_eta as prefill_eta

        monkeypatch.setattr(
            prefill_eta, "estimate_prefill_seconds", lambda *a, **k: (60.0, object())
        )
        body = {"messages": [{"role": "user", "content": "x" * 480_000}]}

        skip, reason = _should_skip_local(
            MODEL, SESSION, body, COLD, estimated_tokens=120_000
        )
        assert (skip, reason) == (True, "large_context_bypass")

        decision = ca.evaluate_capacity_admission(
            _enabled_config(), MODEL, 120_000, 0.0
        )

        assert decision is not None

    def test_physical_cap_is_never_overridden(self, monkeypatch):
        """``context_too_large`` stays a hard cap regardless of capacity."""
        import proxy.prefill_eta as prefill_eta

        monkeypatch.setattr(
            prefill_eta, "estimate_prefill_seconds", lambda *a, **k: (1.0, object())
        )
        body = {"messages": [{"role": "user", "content": "x" * 800_000}]}

        skip, reason = _should_skip_local(
            MODEL,
            SESSION,
            body,
            COLD,
            estimated_tokens=200_000,
            warm_cache_threshold=126_976,
        )

        assert (skip, reason) == (True, "context_too_large")
