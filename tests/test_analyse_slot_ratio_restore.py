#!/usr/bin/env python3
"""Tests for scripts/analyse_slot_ratio_restore.py (LP-0MVGGPSVM003W35K).

The script is the reproducible before/after measurement of restored-slot-prefix
warmness (LP-0MVGD9YXH008O304's AC5). These tests feed it synthetic proxy-log
lines covering every event class it parses and assert the observable metric
contract: routing-decision ratio-0 share, local prefill percentiles split by
cache_state, local-skip/bypass counts, slot save/restore failure rates,
restored-prefix engagement counters, window boundaries and .gz handling.
"""

import gzip
import json
import os
import subprocess
import sys

SCRIPT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "scripts", "analyse_slot_ratio_restore.py"))


def _run(args, tmp_path):
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )


# Synthetic window: two hours; inside-window lines at 10:00/10:01, outliers at
# 08:59 and 11:01 that must be excluded by the window filter.
#   * 3 routing_check inside (two ratio=0.00, one 0.50) -> 2/3 = 66.7%
#   * 2 skip_local (one large_context_bypass) + 1 economic_bypass (no reason)
#   * 3 local dispatches: warm 5s, warm 10s, cold 20s -> p50=10s, max=20s
#   * 2 Stream started: one local, one remote
#   * 1 slot_save success + 1 slot_save failed (+1 failure outside window)
#   * 1 slot_ratio_restored line (must count but not imply warm credit is true)
WIN = {
    "D0": "2026-05-01 09:00:00",  # before the window (excluded)
    "D1": "2026-05-01 10:00:00",  # inside the window
    "D2": "2026-05-01 10:30:00",
    "D3": "2026-05-01 11:05:00",  # after the window (excluded)
    "D4": "2026-05-01 10:00:01,900 - INFO",
    "D5": "2026-05-01 10:00:03,900 - INFO",
}

LINES = {
    "rc_zero": (
        "{D1},900 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
        "estimated_tokens=40000 cold_threshold=38000 warm_threshold=120000 "
        "new_tokens=40000 cached_ratio=0.00 cache_state=cold messages=5 session=s1 "
        "chain_hold_cycle=0"
    ),
    "rc_half": (
        "{D1},901 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
        "estimated_tokens=40000 cold_threshold=38000 warm_threshold=120000 "
        "new_tokens=20000 cached_ratio=0.50 cache_state=warm messages=5 session=s1 "
        "chain_hold_cycle=0"
    ),
    "rc_out_early": (
        "{D0},901 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
        "estimated_tokens=40000 cold_threshold=38000 warm_threshold=120000 "
        "new_tokens=40000 cached_ratio=0.00 cache_state=cold messages=5 session=s9 "
        "chain_hold_cycle=0"
    ),
    "skip_big": (
        "{D1},902 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
        "estimated_tokens=200000 cold_threshold=38000 warm_threshold=120000 "
        "new_tokens=200000 cached_ratio=0.00 cache_state=cold "
        "reason=large_context_bypass → skipping local, routing "
        "session=s1 chain_hold_cycle=0"
    ),
    "skip_ctx": (
        "{D1},903 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
        "estimated_tokens=99000 cold_threshold=38000 warm_threshold=120000 "
        "new_tokens=99000 cached_ratio=0.00 cache_state=cold reason=context_too_large "
        "→ skipping local, routing session=s2 chain_hold_cycle=0"
    ),
    "eco": (
        "{D1},904 - INFO - routing_economic_bypass_local provider=local-qwen3 model=Qwen3 "
        "estimated_tokens=39900 cold_threshold=38000 warm_threshold=120000 "
        "new_tokens=39900 cached_ratio=0.12 cache_state=cold reason=economic_ceiling "
        "session=s3 chain_hold_cycle=0"
    ),
    "dispatch_warm5": (
        "{D1},905 - INFO - dispatch_first_byte_ms=5000.0 dispatch_to_first_byte_ms=4999.0 "
        "cache_state=warm cached_ratio=0.500 queue_wait_ms=1.2 session=s1 model=Qwen3"
    ),
    "dispatch_warm10": (
        "{D1},906 - INFO - dispatch_first_byte_ms=10000.0 dispatch_to_first_byte_ms=9999.0 "
        "cache_state=warm cached_ratio=0.500 queue_wait_ms=1.3 session=s1 model=Qwen3"
    ),
    "dispatch_cold20": (
        "{D1},907 - INFO - dispatch_first_byte_ms=20000.0 dispatch_to_first_byte_ms=19999.0 "
        "cache_state=cold cached_ratio=0.00 queue_wait_ms=1.4 session=s4 model=Qwen3"
    ),
    "dispatch_restored": (
        "{D5} - INFO - dispatch_first_byte_ms=7000.0 dispatch_to_first_byte_ms=6999.0 "
        "cache_state=restored cached_ratio=0.900 queue_wait_ms=1.5 session=s5 model=Qwen3"
    ),
    "stream_local": ("{D1},909 - INFO - Stream started: provider=local model=Qwen3 session=s1"),
    "stream_remote": ("{D1},910 - INFO - Stream started: provider=deepseek-v4.1-flash model=plan session=s6"),
    "save_ok": "{D1},911 - INFO - slot_save success session=s1 slot=1",
    "save_fail": "{D1},912 - INFO - slot_save failed slot=2 error=ReadTimeout/ReadTimeout",
    "save_fail_out": "{D3} - INFO - slot_save failed slot=3 error=ReadTimeout/ReadTimeout",
    "ratio_restored": (
        "{D1},913 - INFO - slot_ratio_restored session=s5 model=Qwen3 cached_tokens=18000 "
        "estimated_tokens=20000 cached_ratio=0.810"
    ),
    "restored_line": (
        "{D1},914 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
        "estimated_tokens=20000 cold_threshold=38000 warm_threshold=120000 "
        "new_tokens=3800 cached_ratio=0.81 cache_state=restored messages=4 session=s5 "
        "chain_hold_cycle=0"
    ),
}


def _render(template):
    for key, value in WIN.items():
        template = template.replace("{" + key + "}", value)
    return template


def _write_corpus(tmp_path, gzipped=False):
    """Write the synthetic corpus; one line is in a separate .gz file."""

    def lines():
        for name in (
            "rc_zero",
            "rc_half",
            "rc_out_early",
            "skip_big",
            "skip_ctx",
            "eco",
            "dispatch_warm5",
            "dispatch_warm10",
            "dispatch_cold20",
            "dispatch_restored",
            "stream_local",
            "stream_remote",
            "save_ok",
            "save_fail",
            "save_fail_out",
            "ratio_restored",
            "restored_line",
        ):
            yield _render(LINES[name]), name

    data = "\n".join(line for line, _ in lines()) + "\n"
    proxy = tmp_path / "proxy.log"
    proxy.write_text(data, encoding="utf-8")
    if gzipped:
        gz = tmp_path / "proxy.log.2026-05-01_00.gz"
        with gzip.open(gz, "wt", encoding="utf-8") as fh:
            fh.write(data)
        proxy.unlink()
    return proxy


def _summary(proc):
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload["windows"]) == 1
    return payload["windows"][0]


def test_window_metrics(tmp_path):
    _write_corpus(tmp_path)
    proc = _run(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "w=2026-05-01T09:05:00,2026-05-01T10:50:00",
            "--json",
        ],
        tmp_path,
    )
    s = _summary(proc)

    # routing decisions: 3 inside (rc_zero, rc_half, restored_line)
    assert s["routing"]["decisions"] == 3
    assert s["routing"]["cached_ratio_zero"] == 1
    assert s["routing"]["cached_ratio_zero_pct"] == round(1 / 3 * 100, 2)
    assert s["routing"]["mean_cached_ratio"] == round((0.0 + 0.5 + 0.81) / 3, 4)

    # local skips / bypasses
    assert s["local_skip"]["routing_skip_local"] == 2
    assert s["local_skip"]["economic_bypass"] == 1
    assert s["local_skip"]["large_context_bypass"] == 1
    assert s["local_skip"]["reasons"]["large_context_bypass"] == 1
    assert s["local_skip"]["reasons"]["context_too_large"] == 1
    assert "economic_ceiling" in s["local_skip"]["reasons"]

    # dispatch percentiles (5, 7, 10, 20 seconds; warm 5/10 + restored 7, cold 20)
    prefill = s["local_prefill_ms"]["all"]
    assert prefill["count"] == 4
    assert prefill["p50"] == 8.5  # (7+10)/2
    assert prefill["max"] == 20.0
    by_state = s["local_prefill_ms"]["by_cache_state"]
    assert by_state["warm"]["count"] == 2
    assert by_state["warm"]["p50"] == 7.5
    assert by_state["cold"]["count"] == 1
    assert by_state["cold"]["p50"] == 20.0
    assert by_state["restored"]["count"] == 1
    assert by_state["restored"]["p50"] == 7.0

    # streams
    assert s["streams"]["total"] == 2
    assert s["streams"]["local"] == 1
    assert s["streams"]["local_share_pct"] == 50.0

    # slots + restored engagement
    assert s["slots"]["save_ok"] == 1
    assert s["slots"]["save_failed"] == 1
    assert s["slots"]["save_failure_pct"] == 50.0
    assert s["slots"]["restore_failed"] == 0
    assert s["slots"]["ratio_restored_events"] == 1
    assert s["slots"]["cache_state_restored_lines"] == 1  # restored routing_check

    # window boundary: the 08:59 routing_check is excluded
    assert s["routing"]["by_cache_state"] == {"cold": 1, "restored": 1, "warm": 1}

    # engagement flag
    assert payload_engagement(proc) == {
        "ratio_restored_events": 1,
        "cache_state_restored_lines": 1,
        "sidecars_on_disk": 0,
        "engaged": True,
    }


def payload_engagement(proc):
    payload = json.loads(proc.stdout)
    return payload["restored_engagement"]


def test_gz_input(tmp_path):
    _write_corpus(tmp_path, gzipped=True)
    proc = _run(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "w=2026-05-01T09:05:00,2026-05-01T10:50:00",
            "--json",
        ],
        tmp_path,
    )
    s = _summary(proc)
    assert s["routing"]["decisions"] == 3
    assert s["local_prefill_ms"]["all"]["count"] == 4
    assert s["slots"]["save_failed"] == 1


def test_out_of_window_lines_ignored(tmp_path):
    _write_corpus(tmp_path)
    # A window that covers only the second half: dispatch lines at 10:00:02+
    proc = _run(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "late=2026-05-01T10:00:02,2026-05-01T10:50:00",
            "--json",
        ],
        tmp_path,
    )
    s = _summary(proc)
    assert s["routing"]["decisions"] == 0
    assert s["slots"]["save_ok"] == 0  # 10:00:00 line excluded
    # 10:00:02 line = dispatch_restored only
    prefill = s["local_prefill_ms"]["all"]
    assert prefill["count"] == 1
    assert prefill["p50"] == 7.0


def test_reports_no_restored_engagement(tmp_path):
    # corpus without the restored/seed lines
    data = "\n".join(_render(LINES[name]) for name in ("rc_zero", "dispatch_warm5", "save_ok")) + "\n"
    (tmp_path / "proxy.log").write_text(data, encoding="utf-8")
    proc = _run(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "w=2026-05-01T09:05:00,2026-05-01T10:50:00",
            "--json",
        ],
        tmp_path,
    )
    payload = json.loads(proc.stdout)
    assert payload["restored_engagement"]["engaged"] is False
    assert payload["restored_engagement"]["ratio_restored_events"] == 0


def test_invalid_window_rejected(tmp_path):
    (tmp_path / "proxy.log").write_text(_render(LINES["rc_zero"]) + "\n", encoding="utf-8")
    proc = _run(["--log-dir", str(tmp_path), "--window", "bad=2026-05-01T10:00"], tmp_path)
    assert proc.returncode == 2


def test_markdown_renders(tmp_path):
    _write_corpus(tmp_path)
    proc = _run(
        ["--log-dir", str(tmp_path), "--window", "w=2026-05-01T09:05:00,2026-05-01T10:50:00"],
        tmp_path,
    )
    assert proc.returncode == 0
    text = proc.stdout
    assert "Routing decisions" in text
    assert "large_context_bypass" in text
    assert "restored" in text


def test_overlapping_windows_each_counted(tmp_path):
    """The full span and the last-24h window overlap; both must be measured."""
    _write_corpus(tmp_path)
    proc = _run(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "full=2026-05-01T09:05:00,2026-05-01T10:50:00",
            "--window",
            "inner=2026-05-01T10:00:00,2026-05-01T10:50:00",
            "--json",
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    full, inner = payload["windows"]
    assert full["routing"]["decisions"] == 3
    # the inner window starts at 10:00:00 so it sees the same three decisions
    assert inner["routing"]["decisions"] == 3
    assert inner["local_prefill_ms"]["all"]["count"] == 4


def test_context_bucket_pairing(tmp_path):
    """Each dispatch is bucketed by its session's preceding routing estimate."""
    _write_corpus(tmp_path)
    proc = _run(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "w=2026-05-01T09:05:00,2026-05-01T10:50:00",
            "--json",
        ],
        tmp_path,
    )
    s = _summary(proc)
    buckets = s["local_prefill_ms"]["by_context_bucket"]["by_bucket"]
    # s1 (est 40k) authorised two warm dispatches; s5 (est 20k) one restored.
    assert buckets["38,000-120,000"]["warm"]["count"] == 2
    assert buckets["38,000-120,000"]["warm"]["p50"] == 7.5
    assert buckets["<= 38,000"]["restored"]["count"] == 1
    # s4's cold dispatch has no preceding routing_check, so it is not paired.
    assert "cold" not in buckets.get("<= 38,000", {})
