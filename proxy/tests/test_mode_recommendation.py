"""Tests for the mode-specific recommendation (F5, LP-0MTCMGJBM007AQ55).

The recommendation module consolidates the F4 fix ranking into per-mode
fix sets with concrete config values (cap, timeout, cooldown), estimates
fallback-rate / TTFT-P95 impact, and emits the follow-up implementation
brief. Tests assert on the concrete per-mode numbers and the
GPU-wedge-mitigation contract (cap raise ⇒ timeout/cooldown plan) — never
on private details.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import mode_recommendation as rec


class TestPerModeRecommendation:
    def test_fast_and_cheap_have_concrete_caps(self):
        res = rec.build_recommendation()
        assert res["fast"]["selected_fixes"]
        assert res["cheap"]["selected_fixes"]
        assert res["fast"]["persistence_cap_tokens"] == 83285
        assert res["cheap"]["persistence_cap_tokens"] == 126976
        assert res["fast"]["slots"] == 3
        assert res["cheap"]["slots"] == 2

    def test_cap_raise_has_timeout_cooldown_plan(self):
        """GPU-wedge constraint (LP-0MS91DHPZ001VWQO): any cap raise must
        carry an explicit timeout/cooldown plan."""
        res = rec.build_recommendation()
        for mode in ("fast", "cheap"):
            m = res[mode]
            plan = m.get("gpu_wedge_plan", {})
            assert plan.get("max_timeout_seconds") == 60
            assert plan.get("failure_cooldown_seconds") == 300
            assert plan.get("max_consecutive_failures") == 3
            assert plan.get("skip_when_busy") is True

    def test_fixes_ranked_with_savings(self):
        res = rec.build_recommendation()
        for mode in ("fast", "cheap"):
            fixes = res[mode]["selected_fixes"]
            assert fixes[0]["rank"] == 1
            for f in fixes:
                assert f["expected_savings_tokens"] >= 0

    def test_impact_estimates_present(self):
        res = rec.build_recommendation()
        for mode in ("fast", "cheap"):
            imp = res[mode]["expected_impact"]
            assert "fallback_rate" in imp
            assert "ttft_p95" in imp
            assert imp["fallback_rate"]["direction"] in ("down", "unchanged")
            assert imp["ttft_p95"]["direction"] in ("down", "unchanged")

    def test_follow_up_brief_created(self):
        res = rec.build_recommendation()
        brief = res["follow_up_brief"]
        assert "title" in brief
        assert "cap_fast" in brief
        assert "cap_cheap" in brief
        assert brief["no_code_change_now"] is True


class TestConvergenceNote:
    def test_modes_may_converge_but_stay_separate(self):
        """Parent risk: fast/cheap thresholds may converge; document if so,
        but keep separate recommendations."""
        res = rec.build_recommendation()
        if res["fast"]["persistence_cap_tokens"] == res["cheap"]["persistence_cap_tokens"]:
            assert res["notes"].get("converged") is True
        else:
            assert res["notes"].get("converged") in (None, False)


class TestFullPlan:
    def test_plan_is_complete(self):
        res = rec.build_recommendation()
        assert set(res.keys()) == {"fast", "cheap", "follow_up_brief", "notes"}
        assert res["follow_up_brief"]["title"]
        # concrete config values for both modes
        for mode in ("fast", "cheap"):
            assert res[mode]["persistence_cap_tokens"] > 0
            assert res[mode]["gpu_wedge_plan"]["max_timeout_seconds"] >= 30
