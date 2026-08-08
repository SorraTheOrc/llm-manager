#!/usr/bin/env python3
"""Tests for scripts/slot-persistence-failures.sh (LP-0MSI1RWLM007N367 F4).

The observation wrapper reuses scripts/slot-persistence-correlate.py to emit
a compact daily failure-rate row. These tests run it in --dry-run mode against
a synthetic log directory so no production logs are touched.
"""

import os
import subprocess

SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), '..', 'scripts', 'slot-persistence-failures.sh'
    )
)
CORRELATE = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), '..', 'scripts', 'slot-persistence-correlate.py'
    )
)

SYNTH_PROXY = """\
2026-08-06 01:55:00,000 - INFO - Stream started: provider=local model=Qwen3 session=019fd490-aaaa request=[...]
2026-08-06 01:57:19,894 - WARNING - slot_save failed slot=2 error=ReadTimeout/ReadTimeout
2026-08-06 01:58:00,000 - INFO - slot_save success session=019fd490 slot=0
2026-08-06 01:58:01,000 - INFO - slot_restore success session=019fd490 slot=0
2026-08-06 01:59:00,000 - INFO - Stream finished: reason=tool_calls session=019fd490-aaaa provider=local model=Qwen3 request=[...]
"""


def _write_synth_logs(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "proxy.log").write_text(SYNTH_PROXY)
    (log_dir / "llama-server.log").write_text("")
    return log_dir


def test_correlation_script_hint():
    """The wrapper depends on the correlation script it wraps; both exist."""
    assert os.path.exists(SCRIPT), f"script not found: {SCRIPT}"
    assert os.path.exists(CORRELATE), f"missing dependency: {CORRELATE}"


def test_dry_run_emits_daily_row(tmp_path):
    """--dry-run prints a compact daily row without touching the log file."""
    log_dir = _write_synth_logs(tmp_path)
    proc = subprocess.run(
        [SCRIPT, "--date", "2026-08-06", "--log-dir", str(log_dir), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    stdout = proc.stdout.strip()
    # Row shows 1 failed / 1 success save (50%) and 1 restore success (0%).
    assert stdout.startswith("| 2026-08-06 |")
    assert "save 1/2 (50.0%)" in stdout, f"unexpected save row: {stdout}"
    assert "restore 0/1 (0.0%)" in stdout, f"unexpected restore row: {stdout}"


def test_append_writes_rows_and_header(tmp_path):
    """Append mode writes a header on first run and appends the daily row."""
    log_dir = _write_synth_logs(tmp_path)
    out = tmp_path / "observation-log.md"
    proc = subprocess.run(
        [SCRIPT, "--date", "2026-08-06", "--log-dir", str(log_dir),
         "--log-file", str(out)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    content = out.read_text()
    assert "Slot-persistence failure observation log" in content
    assert "| 2026-08-06 | save 1/2 (50.0%)" in content
