#!/usr/bin/env python3
"""Tests for scripts/measure_warm_path_window.py (LP-0MVW8S8OZ00449YR).

The tool supplies the two AC5 no-regression metrics the existing instruments do
not: local completions per hour and **large-prompt warm** service latency
p50/p90. These tests run it against synthetic proxy logs and assert the
observable metric contract:

  * a dispatch is banded ``large`` only from its own ``estimated_tokens`` and
    only from the ``cache_state`` the proxy logged;
  * service latency is ``dispatch_to_first_byte_ms`` (queue wait excluded), not
    ``dispatch_first_byte_ms``;
  * a remote stream whose request payload merely *quotes* ``provider=local`` is
    not counted as a local completion, and ``tokens=missing`` lines are excluded
    rather than counted as zero-token completions;
  * window bounds are authoritative, and a window whose logs have rotated away
    is an explicit failure (exit 1), not a silently empty report.
"""

import json
import os
import subprocess

SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "measure_warm_path_window.py",
    )
)

# Local dispatch lines carry the prompt estimate and the cache label; the
# 'Stream finished' lines carry measured usage. Both shapes are copied from the
# live grammar (2026-11-06 window).
SYNTH = """\
2026-11-06 14:16:00,000 - INFO - dispatch_first_byte_ms=3000.0 dispatch_to_first_byte_ms=2999.0 cache_state=warm cached_ratio=0.990 queue_wait_ms=1.0 session=A model=Qwen3 priority=low estimated_tokens=25000 turn=aaaaaaaa
2026-11-06 14:17:00,000 - INFO - dispatch_first_byte_ms=400.0 dispatch_to_first_byte_ms=399.0 cache_state=warm cached_ratio=0.995 queue_wait_ms=1.0 session=B model=Qwen3 priority=low estimated_tokens=5000 turn=bbbbbbbb
2026-11-06 14:18:00,000 - INFO - dispatch_first_byte_ms=70000.0 dispatch_to_first_byte_ms=60000.0 cache_state=cold cached_ratio=0.000 queue_wait_ms=10000.0 session=C model=Qwen3 priority=low estimated_tokens=30000 turn=cccccccc
2026-11-06 14:19:00,000 - INFO - dispatch_first_byte_ms=2000.0 dispatch_to_first_byte_ms=1900.0 cache_state=restored cached_ratio=0.879 queue_wait_ms=100.0 session=D model=Qwen3 priority=low estimated_tokens=40000 turn=dddddddd
2026-11-06 14:20:00,000 - INFO - dispatch_first_byte_ms=1200.0 dispatch_to_first_byte_ms=1100.0 cache_state=warm cached_ratio=0.900 queue_wait_ms=100.0 session=E model=Qwen3 priority=low turn=eeeeeeee
2026-11-06 14:21:00,000 - INFO - Stream finished: reason=tool_calls tokens=25000/40/25040 session=A provider=local model=Qwen3 request=[{'type': 'text', 'text': 'x'}]
2026-11-06 14:22:00,000 - INFO - Stream finished: reason=tool_calls tokens=5000/10/5010 session=B provider=local model=Qwen3 request=[{'type': 'text', 'text': 'x'}]
2026-11-06 14:23:00,000 - INFO - Stream finished: reason=tool_calls tokens=30000/50/30050 session=C provider=OpenCode Go 2 model=deepseek-v4.1-flash entry=opencode-go-2 request=[{'type': 'text', 'text': 'retry with provider=local as a fallback'}]
2026-11-06 14:24:00,000 - INFO - Stream finished: reason=stop tokens=missing session=E provider=local model=Qwen3 request=[]
2026-11-06 14:25:00,000 - INFO - Stream finished: reason=tool_calls tokens=41000/60/41060 session=D provider=local model=Qwen3 request=[{'type': 'text', 'text': 'provider=local is quoted here'}]
"""


def _write_logs(tmp_path, text=SYNTH, name="proxy.log"):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / name).write_text(text, encoding="utf-8")
    return log_dir


def _run(log_dir, *extra):
    return subprocess.run(
        [
            "python3",
            SCRIPT,
            "--log-dir",
            str(log_dir),
            "--since",
            "2026-11-06 14:15:56",
            "--until",
            "2026-11-07 14:15:56",
            "--json",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _result(log_dir, *extra):
    proc = _run(log_dir, *extra)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# ===================================================================
# Band and cache-state banding
# ===================================================================


def test_large_band_and_warm_split(tmp_path):
    data = _result(_write_logs(tmp_path))
    # warm dispatches: A (large), B (small), E (unknown size)
    assert data["warm"]["total"] == 3
    # only A stays in the large warm population
    assert data["large_warm"]["total"] == 1
    # C is large and cold; D is large with the distinct 'restored' path
    assert data["large_cold"]["total"] == 1
    assert data["large_other_cache_state"]["total"] == 1
    assert data["dispatches"]["unknown_prompt_estimate"] == 1
    assert data["dispatches"]["cache_state_breakdown"] == {
        "cold": 1,
        "restored": 1,
        "warm": 3,
    }


def test_service_latency_excludes_queue_wait(tmp_path):
    data = _result(_write_logs(tmp_path))
    # dispatch_to_first_byte_ms (2999.0), not dispatch_first_byte_ms (3000.0)
    assert data["large_warm"]["service_ms"]["n"] == 1
    assert data["large_warm"]["service_ms"]["p50"] == 2999.0
    # the cold line's 10 s queue wait is excluded from its service time
    assert data["large_cold"]["service_ms"]["p50"] == 60000.0


def test_first_byte_fast_share(tmp_path):
    data = _result(_write_logs(tmp_path))
    share = data["large_warm"]["first_byte_under_threshold"]
    assert share["threshold_ms"] == 10000.0
    assert share["count"] == 1
    assert share["share"] == 1.0


def test_large_threshold_is_configurable(tmp_path):
    data = _result(_write_logs(tmp_path), "--large-tokens", "40000")
    assert data["large_warm"]["total"] == 0
    assert data["large_cold"]["total"] == 0
    # D is the only 40k+ dispatch and it is 'restored', not warm
    assert data["large_other_cache_state"]["total"] == 1


# ===================================================================
# Local completions
# ===================================================================


def test_local_completions_exclude_remote_and_missing_usage(tmp_path):
    data = _result(_write_logs(tmp_path))
    completions = data["local_completions"]
    # A, B, D are local with usage; C is remote (payload only quotes
    # 'provider=local'); E reports tokens=missing.
    assert completions["total"] == 3
    assert completions["usage_missing"] == 1
    assert completions["large"]["total"] == 2  # A (25k) and D (41k)
    assert completions["prompt_tokens"]["max"] == 41000.0


def test_per_hour_rates_use_the_requested_window(tmp_path):
    data = _result(_write_logs(tmp_path))
    assert data["window"]["hours"] == 24.0
    assert data["local_completions"]["per_hour"] == round(3 / 24.0, 3)
    assert data["large_warm"]["per_hour"] == round(1 / 24.0, 3)


# ===================================================================
# Window bounds and coverage
# ===================================================================


def test_window_bounds_are_authoritative(tmp_path):
    text = SYNTH + (
        "2026-11-07 14:15:56,000 - INFO - dispatch_first_byte_ms=1.0 "
        "dispatch_to_first_byte_ms=1.0 cache_state=warm cached_ratio=1.0 "
        "queue_wait_ms=0.0 session=Z model=Qwen3 priority=low "
        "estimated_tokens=99999 turn=zzzzzzzz\n"
    )
    data = _result(_write_logs(tmp_path, text))
    # the exclusive upper bound drops the last line
    assert data["large_warm"]["total"] == 1


def test_rotated_away_window_fails_loudly(tmp_path):
    log_dir = _write_logs(
        tmp_path,
        "2026-10-01 10:00:00,000 - INFO - Stream finished: reason=stop "
        "tokens=10/1/11 session=A provider=local model=Qwen3 request=[]\n",
    )
    proc = _run(log_dir)
    assert proc.returncode == 1
    assert "no log lines" in proc.stderr
    assert json.loads(proc.stdout)["window"]["lines_in_window"] == 0


def test_coverage_reports_a_covered_window(tmp_path):
    data = _result(_write_logs(tmp_path))
    assert data["coverage"]["covered"] is True
    assert data["coverage"]["gaps"] == []
    assert data["coverage"]["files_with_lines"] == 1
