"""Tests for the save/restore reuse-gap analysis (F2, LP-0MTCMEOHB002X1JN).

The reuse-gap analysis correlates save vs restore events and gating factors
to explain why ~95% of context checkpoints are never restored. These tests
run the analysis module against synthetic fixture logs that mirror real
proxy.log / llama-server.log shapes, asserting observable per-factor counts,
timeline buckets and cross-references — never private implementation details.
"""

from __future__ import annotations

import gzip as gz
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import slot_reuse_gap_analysis as an


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    opener = gz.open if name.endswith(".gz") else open
    with opener(p, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return p


# --- Fixture lines (real shapes) -------------------------------------------

SAVE_SUCCESS = "2026-08-26 16:01:33,000 - INFO - slot_save success session=herdr-17 slot=2"
RESTORE_SUCCESS = "2026-08-26 16:28:02,887 - INFO - slot_restore success session=herdr-17 slot=2"
SAVE_FAILURE = (
    "2026-08-26 21:02:06,576 - WARNING - slot_save failed slot=0 "
    "error=PoolTimeout/PoolTimeout elapsed=9.3s timeout=9.3s "
    'busy={"active_queries": 1, "local_active_queries": 1, '
    '"active_sessions": 1, "slot_busy": true}'
)
SKIP_TOO_LARGE = (
    "2026-08-26 22:02:00,000 - INFO - routing_skip_local provider=local-qwen3-next "
    "model=Qwen3 estimated_tokens=83494 cold_threshold=38000 warm_threshold=83285 "
    "new_tokens=40 cached_ratio=1.00 reason=context_too_large \u2192 skipping local, "
    "routing to next remote provider session=herdr-17"
)
SKIP_BYPASS = (
    "2026-08-26 22:03:00,000 - INFO - routing_skip_local provider=local-qwen3-next "
    "model=Qwen3 estimated_tokens=424128 cold_threshold=38000 warm_threshold=83285 "
    "new_tokens=424128 cached_ratio=0.00 reason=large_context_bypass \u2192 skipping "
    "local, routing to next remote provider session=herdr-17"
)
STATUS_STALE = (
    "2026-08-26 20:33:06,038 - INFO - status_request active_query=false "
    "available_slots=0 client_ip=192.168.0.191 client_port=56110 "
    "current_model=None latency_ms=1000 llama_server_running=false "
    "local_active_query=false local_owner_lease_remaining_seconds=None "
    "local_owner_session_id=None model_switch_in_progress=false "
    "slots_stale=true total_slots=3"
)
LEASE_ORPHAN = (
    "2026-08-26 16:21:18,388 - WARNING - lease_released "
    "session=herdr-17 reason=orphan_cleanup stream_abandoned=True"
)
LEASE_RELEASED = (
    "2026-08-26 09:28:53,041 - INFO - lease_released session=audit-s1 reason=session_evicted"
)
LEASE_RENEWED = (
    "2026-08-29 00:00:45,402 - INFO - lease_renewed session=herdr-17 timeout=30s"
)
LLAMA_SLOTS_500 = "[51873] srv  log_server_r: done request: GET /slots 127.0.0.1 500"
LLAMA_SLOTS_200 = "[51873] srv  log_server_r: done request: GET /slots 127.0.0.1 200"
CHECKPOINT_CREATE = (
    "[59455] slot update_slots: id  2 | task 1 | created context checkpoint "
    "1 of 32 (pos_min = 906, pos_max = 906, n_tokens = 907, size = 62.813 MiB)"
)
CHECKPOINT_RESTORE = (
    "[59455] slot update_slots: id  1 | task 2547 | restored context checkpoint "
    "(pos_min = 22801, pos_max = 22801, n_tokens = 22802, n_past = 22802, "
    "size = 62.813 MiB)"
)


class TestFactorCounts:
    def test_restore_rate_reuse_gap(self, tmp_path):
        """Restore-rate gap between llama-server native checkpoints and proxy
        slot persistence, per the incident's ~95%-unrestored claim."""
        _write(tmp_path, "llama-server.log-2026-08-27.gz", [
            CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE,
            CHECKPOINT_RESTORE,
            CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE,
            CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE,
            CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE,
            CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE, CHECKPOINT_CREATE,
            CHECKPOINT_RESTORE,
        ])
        _write(tmp_path, "proxy.log", [SAVE_SUCCESS, RESTORE_SUCCESS])
        res = an.analyze_reuse_gap(tmp_path)
        factors = res["factor_breakdown"]
        # llama checkpoint restore rate: 2 restores / 20 created = 10.0%
        assert factors["llama_checkpoint_restore_rate_pct"] == pytest.approx(10.0, abs=0.01)
        # proxy slot persistence restore rate is high (100%)
        assert factors["proxy_slot_restore_rate_pct"] == 100.0
        # the reuse gap: native checkpoints 90% unreused
        assert factors["llama_checkpoints_unrestored_pct"] == pytest.approx(90.0, abs=0.01)

    def test_context_gating_factor(self, tmp_path):
        """context_too_large skips are counted as the size-gating factor."""
        _write(tmp_path, "proxy.log", [SKIP_TOO_LARGE, SKIP_BYPASS, SAVE_SUCCESS])
        res = an.analyze_reuse_gap(tmp_path)
        # the factor key is context_gating
        assert res["factor_breakdown"]["context_gating"]["context_too_large"] == 1
        assert res["factor_breakdown"]["context_gating"]["large_context_bypass"] == 1
        assert res["factor_breakdown"]["context_gating"]["total"] == 2

    def test_slots_stale_factor(self, tmp_path):
        """slots_stale polls count toward the staleness factor."""
        _write(tmp_path, "proxy.log", [
            STATUS_STALE,
            STATUS_STALE.replace("slots_stale=true", "slots_stale=false"),
            STATUS_STALE.replace("20:33:06", "20:34:06"),
        ])
        res = an.analyze_reuse_gap(tmp_path)
        f = res["factor_breakdown"]["slots_stale"]
        assert f["stale_polls"] == 2
        assert f["total_polls"] == 3
        assert f["stale_pct"] == pytest.approx(66.67, abs=0.01)

    def test_lease_churn_factor(self, tmp_path):
        """Lease churn (orphan cleanup releases = affinity breaks) is counted."""
        _write(tmp_path, "proxy.log", [LEASE_ORPHAN, LEASE_RELEASED, LEASE_RENEWED])
        res = an.analyze_reuse_gap(tmp_path)
        f = res["factor_breakdown"]["lease_churn"]
        assert f["orphan_releases"] == 1
        assert f["evicted_releases"] == 1
        assert f["total_lease_events"] == 3

    def test_slots_500_factor(self, tmp_path):
        """GET /slots 500s from llama-server access log count as a factor."""
        _write(tmp_path, "llama-server.log", [
            LLAMA_SLOTS_500, LLAMA_SLOTS_500, LLAMA_SLOTS_200,
        ])
        res = an.analyze_reuse_gap(tmp_path)
        f = res["factor_breakdown"]["slots_500"]
        assert f["five_hundreds"] == 2
        assert f["total_polls"] == 3
        assert f["five_hundred_pct"] == pytest.approx(66.67, abs=0.01)


class TestTimeline:
    def test_timeline_buckets_proxy_events(self, tmp_path):
        """Events bucket into half-hour slots for incident correlation."""
        lines = [
            SAVE_SUCCESS,                          # 16:01
            RESTORE_SUCCESS,                       # 16:28
            SKIP_TOO_LARGE,                        # 22:02
            STATUS_STALE.replace("20:33:06", "22:31:00"),
            LEASE_ORPHAN.replace("16:21:18", "22:45:00"),
            SAVE_SUCCESS.replace("16:01:33", "22:58:00"),
        ]
        _write(tmp_path, "proxy.log", lines)
        res = an.analyze_reuse_gap(tmp_path)
        tl = res["hourly_timeline"]
        assert tl["16:00"]["saves"] == 1
        assert tl["16:00"]["restores"] == 1
        assert tl["22:00"]["skips"] == 1
        assert tl["22:00"]["stale_polls"] == 1
        assert tl["22:00"]["orphan_releases"] == 1
        assert tl["22:00"]["saves"] == 1
        # the 22:02-23:09 snapshot-write window is the incident focus
        assert isinstance(tl["16:00"], dict) and "saves" in tl["16:00"] and "restores" in tl["16:00"]

    def test_llama_events_attributed_globally(self, tmp_path):
        """llama-server events (no timestamps) roll into a global bucket and
        day-attribution keys off the file name."""
        _write(tmp_path, "llama-server.log-2026-08-27.gz", [CHECKPOINT_CREATE, CHECKPOINT_RESTORE])
        _write(tmp_path, "proxy.log", [SAVE_SUCCESS])
        res = an.analyze_reuse_gap(tmp_path)
        assert res["llama_per_file"]["llama-server.log-2026-08-27.gz"]["created"] == 1
        assert res["llama_per_file"]["llama-server.log-2026-08-27.gz"]["restored"] == 1


class TestCrossReference:
    def test_sessions_with_persistence_vs_gated_out(self, tmp_path):
        """Cross-reference: sessions gated out by context_too_large never get
        a restore; sessions with saves get restores."""
        _write(tmp_path, "proxy.log", [
            SKIP_TOO_LARGE,          # herdr-17 gated out (context_too_large)
            SAVE_SUCCESS,            # herdr-17 save (later, smaller ctx)
            RESTORE_SUCCESS,         # herdr-17 restore
        ])
        res = an.analyze_reuse_gap(tmp_path)
        gaps = res["session_cross_reference"]
        # herdr-17 had both a gating skip and a successful restore
        assert gaps["herdr-17"]["skips"] == 1
        assert gaps["herdr-17"]["saves"] == 1
        assert gaps["herdr-17"]["restores"] == 1

    def test_session_gated_never_restored(self, tmp_path):
        """A session whose ONLY persistence event is a context_too_large skip
        is counted in the never-restored gap (the incident's oversized
        sessions)."""
        _write(tmp_path, "proxy.log", [
            SKIP_TOO_LARGE.replace("session=herdr-17", "session=giant-1"),
        ])
        res = an.analyze_reuse_gap(tmp_path)
        gaps = res["session_cross_reference"]
        assert gaps["giant-1"]["skips"] == 1
        assert gaps["giant-1"]["saves"] == 0
        assert not any(
            s["restores"] > 0
            for s in res["session_cross_reference"].values()
        )


class TestIncidentRootCause:
    """Guards the published root-cause findings against the live incident
    logs (skips when logs are unavailable so CI without /var/log/llama-proxy
    does not break)."""

    def test_root_cause_findings(self):
        import subprocess

        log_dir = Path("/var/log/llama-proxy")
        if not log_dir.exists() or not list(log_dir.glob("llama-server.log-*")):
            pytest.skip("live incident logs not available")

        out = subprocess.run(
            [sys.executable, str(Path(an.__file__)), "--log-dir", str(log_dir),
             "--start", "2026-08-26", "--end", "2026-08-27",
             "--llama-file", "*2026-08-27*", "--summary", "--compact"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        f = json.loads(out.stdout)["factor_breakdown"]

        # 1. Size gating dominates: thousands of context_too_large skips
        assert f["context_gating"]["context_too_large"] >= 1000, f
        # 2. Proxy slot persistence restores at high rate when it runs
        assert f["proxy_slot_restore_rate_pct"] >= 80.0, f
        # 3. llama native checkpoints restore at ~5% (the incident's number)
        assert 2.0 <= f["llama_checkpoint_restore_rate_pct"] <= 8.0, f
        # 4. slots_stale present (~47% on incident day)
        assert f["slots_stale"]["stale_pct"] >= 20.0, f
        # 5. lease churn with orphan releases (affinity breaks)
        assert f["lease_churn"]["orphan_releases"] >= 50, f
