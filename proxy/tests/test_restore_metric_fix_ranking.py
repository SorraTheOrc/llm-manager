"""Tests for the restore-rate metric and fix ranking (F4, LP-0MTCMG18A008ZKVT).

The metric module computes the current restore rate (with Wilson CI) from
the F1 corpus, derives per-mode targets, and ranks fix options by expected
prefill-token savings and GPU-wedge risk. Tests assert on observable math —
rates, CI bounds, savings arithmetic, ranking order — never internals.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import restore_metric_fix_ranking as metric


class TestWilsonInterval:
    def test_ci_bounds_reasonable(self):
        c, lo, hi = metric.wilson_ci(831, 926)
        # Wilson center ≈ plain proportion (small shrinkage at n=926)
        assert abs(c - 831 / 926) < 0.005
        assert 0 <= lo < c < hi <= 1
        # 95% CI is ~±2 points at this sample
        assert hi - lo == pytest.approx(0.039, abs=0.005)

    def test_zero_denominator(self):
        assert metric.wilson_ci(0, 0) == (0.0, 0.0, 0.0)

    def test_full_rate(self):
        c, lo, hi = metric.wilson_ci(100, 100)
        # Wilson shrinkage: center < 1 at finite n, but upper bound is 1
        assert 0.9 <= c < 1.0
        assert hi == pytest.approx(1.0)


class TestRateFromBaseline:
    def test_current_restore_rates(self):
        """Both mechanisms' current rates from the baseline JSON match the
        published 2026-08-26 numbers."""
        baseline = {
            "baseline_metrics": {
                "slot_restore_success": 831,
                "total_slot_saves": 926,
                "llama_checkpoints_restored": 154,
                "llama_checkpoints_created": 3191,
            }
        }
        res = metric.compute_current_rates(baseline["baseline_metrics"])
        # proxy slot persistence
        assert res["proxy"]["k"] == 831
        assert res["proxy"]["n"] == 926
        assert res["proxy"]["rate_pct"] == pytest.approx(89.58, abs=0.1)
        # llama native checkpoints — the ~5% incident number
        assert res["llama_native"]["k"] == 154
        assert res["llama_native"]["n"] == 3191
        assert 4.0 <= res["llama_native"]["rate_pct"] <= 6.0
        assert res["llama_native"]["ci_pct"]["lo"] > 0

    def test_prefill_tokens_for_savings(self):
        """Incident-day prefill totals feed expected-savings math."""
        baseline = {
            "baseline_metrics": {
                "slot_restore_success": 831,
                "total_slot_saves": 926,
                "llama_checkpoints_restored": 154,
                "llama_checkpoints_created": 3191,
                "prompt_done_tokens_total": 46_075_973,
            }
        }
        res = metric.compute_current_rates(baseline["baseline_metrics"])
        assert res["prefill_done_tokens"] == 46_075_973


class TestPerModeTargets:
    def test_targets_defined_per_mode(self):
        """Fast and cheap modes have distinct targets with rationale."""
        res = metric.define_targets()
        assert "fast" in res and "cheap" in res
        f = res["fast"]
        c = res["cheap"]
        assert f["priority"] == "ttft_p95"
        assert c["priority"] == "cost_local_utilization"
        assert f["target_rate_pct"] >= 0
        assert c["target_rate_pct"] >= 0
        # rationale strings present
        assert f["rationale"]
        assert c["rationale"]

    def test_config_values(self):
        """Mode config values (caps, slots) are captured for the document."""
        res = metric.define_targets()
        assert res["fast"]["persistence_cap_tokens"] == 83285
        assert res["fast"]["slots"] == 3
        assert res["cheap"]["persistence_cap_tokens"] == 126976
        assert res["cheap"]["slots"] == 2


class TestFixRanking:
    FIXES = {
        "raise_cap": {
            "name": "raise cap with timeout+cooldown",
            "rate_gain": 0.65,
            "recovered_fraction": 0.8,
            "gpu_wedge_risk": "medium",
        },
        "affinity_fix": {
            "name": "affinity/slot-ownership fix",
            "rate_gain": 0.15,
            "recovered_fraction": 0.5,
            "gpu_wedge_risk": "low",
        },
        "restore_before_save": {
            "name": "restore-before-save ordering",
            "rate_gain": 0.05,
            "recovered_fraction": 0.2,
            "gpu_wedge_risk": "none",
        },
    }

    def test_savings_are_ordered(self):
        """Fix options rank by expected prefill-token savings desc; ties
        broken by lower GPU-wedge risk."""
        pool = 1_000_000
        rate = 0.05  # current native restore rate
        ranked = metric.rank_fixes(self.FIXES, pool, rate)
        assert ranked[0]["name"].startswith("raise cap")
        gap = 1.0 - rate
        expected = pool * gap * 0.65 * 0.8
        assert ranked[0]["expected_savings_tokens"] == int(expected)
        assert ranked[-1]["name"].startswith("restore-before-save")

    def test_zero_gain_yields_zero_savings(self):
        fixes = {"x": {"name": "x", "rate_gain": 0, "recovered_fraction": 0,
                       "gpu_wedge_risk": "none"}}
        ranked = metric.rank_fixes(fixes, 1_000_000, 0.05)
        assert ranked[0]["expected_savings_tokens"] == 0

    def test_at_least_three_options(self):
        """The module ships the required ≥3 fix options."""
        options = metric.DEFAULT_FIX_OPTIONS
        assert len(options) >= 3
        names = " ".join(o["name"] for o in options.values())
        # the four mandated option classes are present
        assert "cap" in names
        assert "affinity" in names
        assert "ordering" in names
        assert "busy" in names or "skip-when-busy" in names


class TestSaveRateProjection:
    def test_project_restore_rate_gap(self):
        """Projection: raising native restore rate from ~5% toward the proxy
        slot-restore rate (89.6%) quantifies the addressable gap."""
        res = metric.compute_current_rates({
            "slot_restore_success": 831,
            "total_slot_saves": 926,
            "llama_checkpoints_restored": 154,
            "llama_checkpoints_created": 3191,
            "prompt_done_tokens_total": 46_075_973,
        })
        gap = res["llama_native"]["rate_pct"]
        assert gap < 10.0
        # prefill tokens that re-prefill daily because native restores fail
        assert res["prefill_done_tokens"] > 0
