#!/usr/bin/env python3
"""Tests for scripts/slot_geometry_ab_report.py (LP-0MVNCMDW50043RSP AC2/AC3).

The reporter is the measurement half of the slot-geometry A/B protocol
(docs/dev/slot-geometry-ab.md): it captures one arm's metrics for one window
and compares two captured arms through the pre-registered numeric gate.

These tests are fixture-driven so no production log or GPU is needed:

  * a synthetic proxy log carrying every line shape the log scan reads
    (``routing_skip_local`` reasons including the legacy ``warm_cache_bypass``
    alias, ``dispatch_to_first_byte_ms`` samples, HTTP status lines, an
    out-of-window line, and malformed lines);
  * a canned ``analyze_proxy_usage.py --json`` summary injected with
    ``--analyzer-json`` (the same artifact the protocol archives), so the
    canonic analysis tooling is not re-implemented here.

Every metric parent AC3 names has one green path with a distinctive non-zero
fixture value, so a metric that silently reads zero fails the suite.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "slot_geometry_ab_report.py",
    )
)

WINDOW_START = "2026-10-28 22:00:00"
WINDOW_END = "2026-10-29 06:00:00"

# Real proxy.log line shapes (see /var/log/llama-proxy/proxy.log). Timestamps
# are anchored to the test window; one line sits deliberately OUTSIDE it.
SYNTH = """\
2026-10-28 21:00:00,000 - INFO - dispatch_first_byte_ms=9999.0 dispatch_to_first_byte_ms=9999.0 cache_state=cold cached_ratio=0.000 queue_wait_ms=1.0 session=pre model=Qwen3
2026-10-28 22:00:01,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 estimated_tokens=130500 cold_threshold=38000 warm_threshold=120000 new_tokens=130500 cached_ratio=0.00 cache_state=cold reason=context_too_large \u2192 skipping local, routing to next remote provider session=s1 chain_hold_cycle=0
2026-10-28 22:01:00,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 estimated_tokens=130500 cold_threshold=38000 warm_threshold=120000 new_tokens=130500 cached_ratio=0.00 cache_state=cold reason=large_context_bypass \u2192 skipping local, routing to next remote provider session=s1 chain_hold_cycle=0
2026-10-28 22:02:00,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 estimated_tokens=61000 cold_threshold=38000 warm_threshold=120000 new_tokens=61000 cached_ratio=0.00 cache_state=cold reason=large_context_bypass \u2192 skipping local, routing to next remote provider session=s2 chain_hold_cycle=0
2026-10-28 22:02:30,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 estimated_tokens=90000 cold_threshold=38000 warm_threshold=120000 new_tokens=90000 cached_ratio=0.00 cache_state=cold reason=warm_cache_bypass \u2192 skipping local, routing to next remote provider session=s3 chain_hold_cycle=0
2026-10-28 22:03:00,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 estimated_tokens=9000 cold_threshold=38000 warm_threshold=120000 new_tokens=9000 cached_ratio=0.00 cache_state=cold reason=rate_limited \u2192 skipping local, routing to next remote provider session=s1 chain_hold_cycle=0
2026-10-28 22:04:00,000 - INFO - dispatch_first_byte_ms=1420.5 dispatch_to_first_byte_ms=1419.0 cache_state=warm cached_ratio=0.990 queue_wait_ms=1.5 session=s1 model=Qwen3
2026-10-28 22:05:00,000 - INFO - dispatch_first_byte_ms=2001.0 dispatch_to_first_byte_ms=2000.0 cache_state=cold cached_ratio=0.000 queue_wait_ms=1.0 session=s2 model=Qwen3
2026-10-28 22:06:00,000 - INFO - dispatch_first_byte_ms=5002.0 dispatch_to_first_byte_ms=5000.0 cache_state=cold cached_ratio=0.000 queue_wait_ms=2.0 session=s3 model=Qwen3
2026-10-28 22:07:00,000 - WARNING - slot_save failed slot=1 error=ReadTimeout/ReadTimeout elapsed=3.0s timeout=3.0s busy={"active_queries": 1}
2026-10-28 22:08:00,000 - WARNING - [remote] upstream error status=503 url=https://api.example.com/v1/chat/completions body={"error":{}}
2026-10-28 22:09:00,000 - INFO - Fallback triggered for model=v1/chat/completions, from=munitoai-free, to=opencode-go, reason=HTTP 400 session=s1 attempt=1
not a proxy log line at all
2026-10-28 22:10:00,000 - INFO - dispatch_to_first_byte_ms=not-a-number session=s4
"""

# Canned ``analyze_proxy_usage.py --json`` summary. Values are distinctive so
# a mis-mapped key cannot coincidentally equal the expectation.
ANALYZER = {
    "window_start": WINDOW_START,
    "window_end": WINDOW_END,
    "sessions": 42,
    "total_requests": 900,
    "local_requests": 321,
    "remote_requests": 579,
    "contention_dispatch": 7,
    "contention_fallback_after_queue": 2,
    "contention_queued_duration_seconds": 33.5,
    "errors": 11,
    "errors_by_type": {"slot_save_error": 3, "upstream_http_error": 8},
    "upstream_status_503_errors": 1,
    "local_busy": {
        "definition": "dispatch_to_finish",
        "busy_pct": 61.5,
        "busy_seconds": 17712.0,
        "window_seconds": 28800.0,
        "streams": 321,
        "peak_concurrency": 2,
        "avg_concurrency": 1.4,
        "prefill_busy_seconds": 8000.0,
        "decode_busy_seconds": 9712.0,
        "prefill_vs_decode": {
            "unit": "slot-seconds",
            "prefill": {"slot_seconds": 8300.0, "samples": 321},
            "decode": {"slot_seconds": 9400.0, "samples": 321},
            "combined_slot_seconds": 17700.0,
        },
    },
    "ttft": {
        "total": {"median": 900.0, "p10": 100.0, "p90": 1500.0, "samples": 900},
        "fast": {"median": 900.0, "p10": 100.0, "p90": 1500.0, "samples": 900},
        "cheap": {"median": 0.0, "p10": 0.0, "p90": 0.0, "samples": 0},
    },
}

CONFIG = """\
server:
  llama_server_port: 8080
  local_model_ctx_size: 262144
  local_large_context_cold_cache_threshold: 38000
  local_large_context_warm_cache_threshold: 120000
  local_large_context_economic_ceiling_ratio: 0.4578
  session_slot_pool_size: 2
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_logs(tmp_path, text=SYNTH, name="proxy.log"):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / name).write_text(text)
    return log_dir


def _write_analyzer(tmp_path, payload=None):
    path = tmp_path / "analyzer.json"
    path.write_text(json.dumps(ANALYZER if payload is None else payload))
    return path


def _write_config(tmp_path, text=CONFIG):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def _capture(tmp_path, *extra, log_dir=None, analyzer=None, out=None, expect=None):
    log_dir = log_dir if log_dir is not None else _write_logs(tmp_path)
    analyzer = analyzer if analyzer is not None else _write_analyzer(tmp_path)
    out = out if out is not None else tmp_path / "out"
    cmd = [
        "python3",
        SCRIPT,
        "capture",
        "--label",
        "arm-test",
        "--start",
        WINDOW_START,
        "--end",
        WINDOW_END,
        "--log-dir",
        str(log_dir),
        "--analyzer-json",
        str(analyzer),
        "--out-dir",
        str(out),
        "--config",
        str(_write_config(tmp_path)),
        "--no-live-probe",
        "--json",
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == (expect if expect is not None else 0), proc.stderr
    if expect is not None and expect != 0:
        return None, proc
    return json.loads(proc.stdout), proc


def _dig(payload, dotted):
    node = payload
    for part in dotted.split("."):
        assert part in node, f"missing key {dotted!r} (at {part!r}): {sorted(node)}"
        node = node[part]
    return node


# ===================================================================
# AC2 — every parent-AC3 metric is emitted, from a real (fixture) signal
# ===================================================================

AC3_METRICS = [
    ("metrics.local_busy_pct", 61.5),
    ("metrics.turns_served_locally", 321),
    ("metrics.remote_requests", 579),
    ("metrics.divert_total", 5),
    ("metrics.diverts_by_reason.context_too_large", 2),
    ("metrics.diverts_by_reason.large_context_bypass", 2),
    ("metrics.diverts_by_reason.rate_limited", 1),
    ("metrics.context_ceiling_diverts", 4),
    ("metrics.queue_waits", 7),
    ("metrics.fallback_after_queue", 2),
    ("metrics.queue_wait_seconds", 33.5),
    ("metrics.local_dispatch_first_byte.samples", 3),
    ("metrics.local_dispatch_first_byte.p50_ms", 2000.0),
    ("metrics.local_dispatch_first_byte.p90_ms", 4400.0),
    ("metrics.ttft_all_streams.p90_ms", 1500.0),
    ("metrics.prefill_slot_seconds", 8300.0),
    ("metrics.decode_slot_seconds", 9400.0),
    ("metrics.peak_concurrency", 2),
    ("metrics.errors.total", 11),
    ("metrics.errors.slot_save_failed", 3),
    ("metrics.errors.http_400_lines", 1),
    ("metrics.errors.http_503_lines", 1),
    ("metrics.errors.upstream_503", 1),
]


@pytest.mark.parametrize("path,expected", AC3_METRICS, ids=[p for p, _ in AC3_METRICS])
def test_capture_emits_each_ac3_metric(tmp_path, path, expected):
    data, _ = _capture(tmp_path)
    assert _dig(data, path) == expected


def test_capture_records_window_and_source_files(tmp_path):
    data, _ = _capture(tmp_path)
    assert data["window"] == {
        "start": WINDOW_START,
        "end": WINDOW_END,
        "hours": 8.0,
    }
    assert data["metrics"]["log_files"] == ["proxy.log"]
    # Malformed and unparseable lines are counted, never fatal (AC3).
    # every malformed line shape is tolerated, never fatal (AC3)
    assert data["metrics"]["scan"]["unparsed_lines"] >= 1


def test_capture_records_geometry_arm_and_label(tmp_path):
    data, _ = _capture(tmp_path, "--arm", "A")
    assert data["arm"] == "A"
    assert data["label"] == "arm-test"
    geometry = data["geometry"]
    assert geometry["slots"] == 2
    assert geometry["ctx_size"] == 262144
    assert geometry["per_slot_ctx"] == 131072
    assert geometry["per_slot_route_clamp"] == 126976
    assert geometry["warm_threshold"] == 120000
    assert geometry["cold_threshold"] == 38000
    # The non-collapsed routing band is a precondition of a valid arm.
    assert geometry["band_width"] == 82000


def test_capture_writes_arm_json_archive(tmp_path):
    data, _ = _capture(tmp_path)
    archive = tmp_path / "out" / "arm.json"
    assert archive.is_file()
    assert json.loads(archive.read_text())["metrics"]["local_busy_pct"] == 61.5


def test_capture_archives_the_analysis_input_it_measured(tmp_path):
    """The archived arm must carry the analysis summary it was built from."""
    data, _ = _capture(tmp_path)
    archived = tmp_path / "out" / "analyzer" / "summary.json"
    assert archived.is_file()
    assert json.loads(archived.read_text()) == ANALYZER


def test_capture_fails_when_no_log_files_are_found(tmp_path):
    empty = tmp_path / "empty-logs"
    empty.mkdir()
    data, proc = _capture(tmp_path, log_dir=empty, expect=1)
    assert "no proxy log files" in proc.stderr.lower()


def test_capture_fails_when_analyzer_summary_is_malformed(tmp_path):
    bad = tmp_path / "analyzer.json"
    bad.write_text("{not json")
    data, proc = _capture(tmp_path, analyzer=bad, expect=1)
    assert "analyzer" in proc.stderr.lower()


# ===================================================================
# AC3 — comparison through the pre-registered numeric gate
# ===================================================================

GATE_MET_ARM_A = {
    "label": "arm-a",
    "arm": "A",
    "geometry": {"slots": 2, "per_slot_ctx": 131072},
    "metrics": {
        "local_busy_pct": 60.0,
        "turns_served_locally": 300,
        "context_ceiling_diverts": 100,
        "queue_waits": 10,
        "fallback_after_queue": 2,
        "local_dispatch_first_byte": {"samples": 300, "p50_ms": 1000.0, "p90_ms": 4000.0},
        "ttft_all_streams": {"samples": 900, "p50_ms": 900.0, "p90_ms": 1500.0},
    },
}

GATE_MET_ARM_B = {
    "label": "arm-b",
    "arm": "B",
    "geometry": {"slots": 1, "per_slot_ctx": 262144},
    "metrics": {
        "local_busy_pct": 62.0,
        "turns_served_locally": 320,
        "context_ceiling_diverts": 20,
        "queue_waits": 12,
        "fallback_after_queue": 3,
        "local_dispatch_first_byte": {"samples": 320, "p50_ms": 1100.0, "p90_ms": 4500.0},
        "ttft_all_streams": {"samples": 900, "p50_ms": 950.0, "p90_ms": 1600.0},
    },
}


def _write_arm(tmp_path, name, payload):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def _compare(tmp_path, baseline, candidate, expect=0):
    cmd = [
        "python3",
        SCRIPT,
        "compare",
        "--baseline",
        str(_write_arm(tmp_path, "baseline", baseline)),
        "--candidate",
        str(_write_arm(tmp_path, "candidate", candidate)),
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == expect, proc.stderr
    if not proc.stdout.strip():
        return None, proc
    return json.loads(proc.stdout), proc


def _with(arm, **metrics):
    copy = json.loads(json.dumps(arm))
    copy["metrics"].update(metrics)
    return copy


def test_compare_adopts_single_slot_when_the_gate_passes(tmp_path):
    data, _ = _compare(tmp_path, GATE_MET_ARM_A, GATE_MET_ARM_B)
    assert data["evidence_gate"]["met"] is True
    assert data["decision"] == "adopt_1x262k"
    criteria = {c["name"]: c for c in data["criteria"]}
    assert set(criteria) == {
        "turns_served_locally",
        "context_ceiling_diverts",
        "first_byte_p90",
        "local_busy_pct",
    }
    assert all(c["met"] for c in criteria.values())


def test_compare_keeps_two_slots_when_diverts_do_not_fall_by_half(tmp_path):
    # 100 -> 60 diverts is only a 40 % cut, short of the 50 % gate.
    data, _ = _compare(
        tmp_path,
        GATE_MET_ARM_A,
        _with(GATE_MET_ARM_B, context_ceiling_diverts=60),
    )
    assert data["decision"] == "keep_2_slots"
    criteria = {c["name"]: c for c in data["criteria"]}
    assert criteria["context_ceiling_diverts"]["met"] is False
    assert criteria["context_ceiling_diverts"]["reduction_pct"] == pytest.approx(40.0)


def test_compare_keeps_two_slots_when_local_turns_drop(tmp_path):
    data, _ = _compare(
        tmp_path,
        GATE_MET_ARM_A,
        _with(GATE_MET_ARM_B, turns_served_locally=299),
    )
    assert data["decision"] == "keep_2_slots"
    criteria = {c["name"]: c for c in data["criteria"]}
    assert criteria["turns_served_locally"]["met"] is False


def test_compare_keeps_two_slots_when_first_byte_p90_worsens_past_30pct(tmp_path):
    # 4000 ms -> 5300 ms is a 32.5 % regression.
    data, _ = _compare(
        tmp_path,
        GATE_MET_ARM_A,
        _with(
            GATE_MET_ARM_B,
            local_dispatch_first_byte={"samples": 320, "p50_ms": 1100.0, "p90_ms": 5300.0},
        ),
    )
    assert data["decision"] == "keep_2_slots"
    criteria = {c["name"]: c for c in data["criteria"]}
    assert criteria["first_byte_p90"]["met"] is False


def test_compare_keeps_two_slots_on_any_busy_regression(tmp_path):
    """Criterion (d) is zero-tolerance (producer decision 2026-10-28, answer 2):
    "local busy must not regress at all" — even a 0.1-point dip fails."""
    data, _ = _compare(
        tmp_path,
        GATE_MET_ARM_A,
        _with(GATE_MET_ARM_B, local_busy_pct=59.9),
    )
    assert data["decision"] == "keep_2_slots"
    criteria = {c["name"]: c for c in data["criteria"]}
    assert criteria["local_busy_pct"]["met"] is False
    assert "0-point tolerance" in criteria["local_busy_pct"]["detail"]


def test_compare_adopts_when_busy_is_merely_unchanged(tmp_path):
    """Zero tolerance must not mean "must improve": equal busy passes (d)."""
    data, _ = _compare(
        tmp_path,
        GATE_MET_ARM_A,
        _with(GATE_MET_ARM_B, local_busy_pct=GATE_MET_ARM_A["metrics"]["local_busy_pct"]),
    )
    criteria = {c["name"]: c for c in data["criteria"]}
    assert criteria["local_busy_pct"]["met"] is True
    assert data["decision"] == "adopt_1x262k"


def test_compare_is_inconclusive_when_the_busy_evidence_gate_is_unmet(tmp_path):
    """Parent AC0: an arm measured on an idle GPU cannot decide the geometry."""
    data, proc = _compare(
        tmp_path,
        _with(GATE_MET_ARM_A, local_busy_pct=6.1),
        GATE_MET_ARM_B,
        expect=2,
    )
    assert data["evidence_gate"]["met"] is False
    assert data["evidence_gate"]["required_busy_pct"] == 60.0
    assert data["decision"] == "inconclusive"
    assert "busy" in json.dumps(data["evidence_gate"]).lower()


def test_compare_falls_back_to_stream_ttft_without_local_first_byte_samples(tmp_path):
    data, _ = _compare(
        tmp_path,
        _with(
            GATE_MET_ARM_A,
            local_dispatch_first_byte={"samples": 0, "p50_ms": 0.0, "p90_ms": 0.0},
        ),
        _with(
            GATE_MET_ARM_B,
            local_dispatch_first_byte={"samples": 0, "p50_ms": 0.0, "p90_ms": 0.0},
        ),
    )
    criteria = {c["name"]: c for c in data["criteria"]}
    assert criteria["first_byte_p90"]["source"] == "ttft_all_streams"
    assert data["decision"] == "adopt_1x262k"


def test_compare_rejects_an_arm_compared_with_itself(tmp_path):
    data, proc = _compare(tmp_path, GATE_MET_ARM_A, GATE_MET_ARM_A, expect=1)
    assert "same arm" in proc.stderr.lower()


def test_compare_reports_missing_input(tmp_path):
    proc = subprocess.run(
        [
            "python3",
            SCRIPT,
            "compare",
            "--baseline",
            str(tmp_path / "nope.json"),
            "--candidate",
            str(tmp_path / "nope2.json"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1
    assert "not found" in proc.stderr.lower()


def test_compare_text_summary_names_the_decision(tmp_path):
    baseline = _write_arm(tmp_path, "a", GATE_MET_ARM_A)
    candidate = _write_arm(tmp_path, "b", GATE_MET_ARM_B)
    proc = subprocess.run(
        [
            "python3",
            SCRIPT,
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "adopt_1x262k" in proc.stdout
