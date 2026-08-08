#!/usr/bin/env python3
"""Tests for scripts/slot-persistence-correlate.py (LP-0MSI1RWLM007N367 F1).

The correlation script maps proxy slot_save/slot_restore failures to
concurrent local load and the adaptive-timeout cadence. These tests run the
script against a synthetic log directory so no production logs are needed.
"""

import json
import os
import subprocess

SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'scripts', 'slot-persistence-correlate.py')
)

# A synthetic day of proxy logs: 3 failures at the adaptive-timeout cadence
# (31s and 25s gaps) with 2 concurrent local streams, plus successes and one
# llama-server prompt_save line (no timestamps).
SYNTH_PROXY = """\
2026-08-06 01:55:00,000 - INFO - Stream started: provider=local model=Qwen3 session=019fd490-aaaa request=[...]
2026-08-06 01:55:01,000 - INFO - Stream started: provider=local model=Qwen3 session=019fd491-bbbb request=[...]
2026-08-06 01:55:10,000 - INFO - Stream finished: reason=tool_calls session=019fd490-aaaa provider=local model=Qwen3 request=[...]
2026-08-06 01:57:19,894 - WARNING - slot_save failed slot=2 error=ReadTimeout/ReadTimeout
2026-08-06 01:57:50,503 - WARNING - slot_save failed slot=2 error=ReadTimeout/ReadTimeout
2026-08-06 01:58:15,748 - WARNING - slot_save failed slot=2 error=ReadTimeout/ReadTimeout
2026-08-06 01:58:20,000 - INFO - Stream finished: reason=tool_calls session=019fd491-bbbb provider=local model=Qwen3 request=[...]
2026-08-06 01:59:00,000 - INFO - slot_save success session=019fd490 slot=0
2026-08-06 01:59:01,000 - INFO - slot_restore success session=019fd490 slot=0
2026-08-06 02:00:00,000 - INFO - Fallback triggered for model=v1/chat/completions, from=local-qwen3, to=opencode-go, reason=local_concurrency_limit
"""

SYNTH_LLAMA = """\
[36051] srv   prompt_save:  - saving prompt with length 26234, total state size = 335.517 MiB
"""


def _write_synth_logs(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "proxy.log").write_text(SYNTH_PROXY)
    (log_dir / "llama-server.log").write_text(SYNTH_LLAMA)
    return log_dir


def _run(log_dir, *extra):
    return subprocess.run(
        [SCRIPT, "--log-dir", str(log_dir), "--json", *extra],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_script_counts_failures_successes_and_cadence(tmp_path):
    """AC2: the script reports failure/success counts and the cadence gaps."""
    log_dir = _write_synth_logs(tmp_path)
    proc = _run(log_dir)
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    report = json.loads(proc.stdout)

    assert report["totals"]["slot_save_failed"] == 3
    assert report["totals"]["slot_save_success"] == 1
    assert report["totals"]["slot_restore_success"] == 1
    assert report["totals"]["slot_restore_failed"] == 0
    # llama-server prompt_save line is counted (timing not possible — no ts).
    assert report["totals"]["llama_prompt_save"] == 1
    # Failure-rate sanity: 3/4 = 75%.
    assert report["failure_rate_pct"]["save"] == 75.0

    # Cadence: 01:57:19 -> 01:57:50 = 31s, 01:57:50 -> 01:58:15 = 25s.
    assert report["cadence_seconds_by_slot"]["slot_2"] == [31, 25]


def test_script_detects_concurrent_load_at_failure(tmp_path):
    """AC2: failures are flagged as occurring under concurrent local load."""
    log_dir = _write_synth_logs(tmp_path)
    proc = _run(log_dir)
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    report = json.loads(proc.stdout)

    # Two streams started at 01:55:00/01:55:01; the first finished before the
    # failures, the second finished at 01:58:20 — so at each failure exactly
    # one session (019fd491) still had an active local stream.
    for failure in report["failures"]:
        assert failure["active_local_streams_approx"] == 1, (
            f"expected 1 concurrent local stream at {failure['ts']}, "
            f"got {failure['active_local_streams_approx']}"
        )
    assert report["load_context"]["failures_with_local_streams"] == 3
    assert report["load_context"]["failures_without_local_streams"] == 0
    assert report["load_context"]["max_active_local_streams_at_failure"] == 1


def test_script_window_filter(tmp_path):
    """--start/--end filters restrict the reported failure window."""
    log_dir = _write_synth_logs(tmp_path)
    proc = _run(log_dir, "--start", "2026-08-06 01:58:00", "--end", "2026-08-06 01:58:30")
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    report = json.loads(proc.stdout)

    # Only 01:58:15 falls inside the window.
    assert report["totals"]["slot_save_failed"] == 1
    assert report["failures"][0]["ts"] == "2026-08-06 01:58:15"
    # Load context is still computed from the full history: at 01:58:15 the
    # second stream (019fd491) is still active.
    assert report["failures"][0]["active_local_streams_approx"] == 1


def test_script_empty_logs(tmp_path):
    """An empty log directory yields zeroed totals, exit 0."""
    log_dir = tmp_path / "empty"
    log_dir.mkdir()
    (log_dir / "proxy.log").write_text("")
    (log_dir / "llama-server.log").write_text("")
    proc = _run(log_dir)
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    report = json.loads(proc.stdout)
    assert report["totals"]["slot_save_failed"] == 0
    assert report["failures"] == []
    assert report["load_context"]["failures_with_local_streams"] == 0
