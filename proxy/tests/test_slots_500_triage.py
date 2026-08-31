"""Tests for the /slots 500-storm triage analysis (F3, LP-0MTCMEV1G0022A35).

These tests run the triage module against synthetic llama-server access-log
fixtures that mirror real shapes, asserting observable classifications
(responder, proximate cause, window correlation, fix ranking presence) —
never private implementation details.
"""

from __future__ import annotations

import gzip as gz
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import slots_500_triage as triage


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    opener = gz.open if name.endswith(".gz") else open
    with opener(p, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return p


ROUTER_200 = "srv  log_server_r: done request: GET /slots 127.0.0.1 200"
ROUTER_500 = "srv  log_server_r: done request: GET /slots 127.0.0.1 500"
ROUTER_400 = "srv  log_server_r: done request: GET /slots 127.0.0.1 400"
MODEL_200 = "[59455] srv  log_server_r: done request: GET /slots 127.0.0.1 200"
PROXY_LINE = "srv  proxy_reques: proxying request to model Qwen3 on port 59455"
CANCEL_LINE = "srv  operator(): http client error: Connection handling canceled"
PREFILL_LINE = "[59455] slot update_slots: id  1 | task 1994 | prompt processing progress, n_tokens = 10240, batch.n_tokens = 2048, progress = 0.462052"
CHECKPOINT_LINE = "[59455] slot update_slots: id  2 | task 1 | created context checkpoint 1 of 32 (pos_min = 906, pos_max = 906, n_tokens = 907, size = 62.813 MiB)"


class TestClassification:
    def test_router_vs_model_responses(self, tmp_path):
        """Router returns 500/400; model instances only 200."""
        _write(tmp_path, "llama-server.log", [
            ROUTER_200, ROUTER_500, ROUTER_400, MODEL_200,
        ])
        res = triage.analyze_slots_500(tmp_path)
        cl = res["classification"]["responder_status"]
        assert cl["router:500"] == 1
        assert cl["router:400"] == 1
        assert cl["router:200"] == 1
        assert cl["59455:200"] == 1
        assert "59455:500" not in cl

    def test_proximate_causes(self, tmp_path):
        """500 near proxying/cancel/prefill is classified as those causes."""
        _write(tmp_path, "llama-server.log", [
            PROXY_LINE, CANCEL_LINE, PREFILL_LINE, ROUTER_500,
        ])
        res = triage.analyze_slots_500(tmp_path)
        causes = res["classification"]["proximate_causes"]
        assert causes.get("router_proxy_cancel", 0) >= 1
        assert causes.get("connection_canceled", 0) >= 1
        assert causes.get("concurrent_prefill", 0) >= 1
        stats = res["classification"]["stats"]
        assert stats["slots_500"] == 1

    def test_restart_race_classification(self, tmp_path):
        """500 near model-load lines is classified as restart race."""
        _write(tmp_path, "llama-server.log", [
            "srv  load: spawning server instance with name=Qwen3 on port 51873",
            ROUTER_500,
        ])
        res = triage.analyze_slots_500(tmp_path)
        causes = res["classification"]["proximate_causes"]
        assert causes.get("restart_race", 0) >= 1

    def test_stats_counts(self, tmp_path):
        """stat counters match the synthetic lines."""
        _write(tmp_path, "llama-server.log", [
            ROUTER_200, ROUTER_500, ROUTER_200, ROUTER_500,
            PROXY_LINE, CANCEL_LINE,
        ])
        res = triage.analyze_slots_500(tmp_path)
        s = res["classification"]["stats"]
        assert s["slots_total"] == 4
        assert s["slots_500"] == 2
        assert s["slots_200"] == 2
        assert s["proxying_events"] == 1
        assert s["cancel_events"] == 1


class TestCorrelation:
    def test_slab_high_rate_windows(self, tmp_path):
        """Slabs dominated by 500s show up as high-rate windows."""
        lines = []
        # fill slab 0 (1000 lines) with a 500-heavy burst + prefill
        for _ in range(900):
            lines.append(ROUTER_500)
        for _ in range(99):
            lines.append(ROUTER_200)
        lines.append(PREFILL_LINE)
        # slab 1 healthy
        for _ in range(1000):
            lines.append(ROUTER_200)
        _write(tmp_path, "llama-server.log", lines)
        res = triage.analyze_slots_500(tmp_path)
        corr = res["correlation"]
        assert len(corr["high_rate_windows"]) >= 1
        rates = {r["five_hundred_pct"] for r in corr["high_rate_windows"]}
        assert max(rates) >= 50.0
        # busy windows require both prefill activity and 500s
        assert corr["busy_windows_count"] >= 1

    def test_healthy_slab_no_high_rate(self, tmp_path):
        """Slabs without 500s never produce high-rate windows."""
        _write(tmp_path, "llama-server.log", [ROUTER_200] * 100)
        res = triage.analyze_slots_500(tmp_path)
        assert res["correlation"]["high_rate_windows"] == []
        assert res["correlation"]["busy_windows_count"] == 0

    def test_fix_options_ranked_with_tracked_refs(self, tmp_path):
        """Fix options carry rank, impact, and tracked-elsewhere flags."""
        _write(tmp_path, "llama-server.log", [ROUTER_500])
        res = triage.analyze_slots_500(tmp_path)
        fixes = res["fix_options"]
        assert fixes[0]["rank"] == 1
        for f in fixes:
            assert f["expected_impact"]
            assert "tracked_elsewhere" in f
        refs = " ".join(f["tracked_elsewhere"] for f in fixes)
        # the parent's related-work items are flagged
        assert "LP-0MSVP7XJ6008QPKX" in refs
        assert "LP-0MSB0RV72001KNRV" in refs


class TestIncidentReproduction:
    """Guards the live 2026-08-26 /slots storm numbers (skips when logs are
    unavailable so CI without /var/log/llama-proxy does not break)."""

    def test_incident_numbers(self):
        import subprocess

        log_dir = Path("/var/log/llama-proxy")
        target = log_dir / "llama-server.log-2026-08-27.gz"
        if not target.exists():
            pytest.skip("incident-day llama log not available")

        out = subprocess.run(
            [sys.executable, str(Path(triage.__file__)), "--log-dir", str(log_dir),
             "--llama-file", "*2026-08-27*", "--summary", "--compact"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        res = json.loads(out.stdout)

        # incident claims: 6,459/69.6K 500s (~9.3%) + 527 HTTP 400s
        cl = res["classification"]
        responder_status = cl["responder_status"]
        assert responder_status["router:500"] >= 6000, cl
        assert responder_status["router:400"] == 527, cl
        # all GET /slots polls (router + model instances) form the denominator
        total = sum(v for k, v in responder_status.items() if k.endswith(":200") or k.endswith(":400") or k.endswith(":500"))
        rate = 100 * responder_status["router:500"] / total
        assert 5.0 <= rate <= 15.0, cl
        # router-proxy-cancel explains essentially all 500s
        causes = cl["proximate_causes"]
        assert causes.get("router_proxy_cancel", 0) >= 6000, causes
