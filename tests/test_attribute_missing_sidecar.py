#!/usr/bin/env python3
"""Tests for scripts/attribute_missing_sidecar.py (LP-0MVVI8TEU006KTOZ).

The script is the reproducible AC1 evidence for the writer-side sidecar gap:
it pins which save path (router vs eviction) produced each ``missing_sidecar``
snapshot in a window, with counts. These tests feed synthetic proxy-log lines
covering router saves, eviction saves (including the new unpaired-save skip),
window boundaries, ``.gz`` input and the empty-window case.
"""

import gzip
import json
import os
import subprocess
import sys

SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "attribute_missing_sidecar.py")
)

WINDOW = "2026-10-29T12:58:32,2026-10-30T12:58:32"

# Inside the window unless noted. Session prefixes are 8 chars, as logged.
LINES = [
    # A router save for session 01a187be, then a seed after it -> router.
    "2026-10-29 13:00:00,000 - INFO - slot_save success session=01a187be slot=1",
    "2026-10-29 13:00:05,000 - INFO - slot_ratio_seed outcome=missing_sidecar session=01a187be model=Qwen3 stored_messages=0 incoming_messages=42 age_s=0 cached_ratio=0.00",
    # An eviction save for session audit-AH, then a seed after it -> eviction.
    "2026-10-29 14:00:00,000 - INFO - slot_save session=audit-AH slot=0 reason=tenancy_lease_expired saved=True source=eviction",
    "2026-10-29 14:00:09,000 - INFO - slot_ratio_seed outcome=missing_sidecar session=audit-AH model=Qwen3 stored_messages=0 incoming_messages=55 age_s=0 cached_ratio=0.00",
    # A seed with no save in the window -> no-save-in-window.
    "2026-10-29 15:00:00,000 - INFO - slot_ratio_seed outcome=missing_sidecar session=orphan22 model=Qwen3 stored_messages=0 incoming_messages=10 age_s=0 cached_ratio=0.00",
    # A pre-fix eviction save *failure* is counted separately from a success.
    "2026-10-30 09:30:00,000 - WARNING - slot_save session=audit-MS slot=1 reason=tenancy_lease_expired saved=False source=eviction",
    # Other outcomes must not be attributed.
    "2026-10-30 10:00:00,000 - INFO - slot_ratio_seed outcome=missing_snapshot session=01a187be model=Qwen3 stored_messages=4 incoming_messages=42 age_s=8 cached_ratio=0.00",
    # A slot_busy persistence skip.
    "2026-10-30 11:00:00,000 - INFO - slot persistence skipped session=01a187be slot=0 reason=slot_busy active_sessions=2",
    # Outside the window on both sides.
    "2026-10-29 12:00:00,000 - INFO - slot_ratio_seed outcome=missing_sidecar session=before00 model=Qwen3 stored_messages=0 incoming_messages=1 age_s=0 cached_ratio=0.00",
    "2026-10-30 13:00:00,000 - INFO - slot_ratio_seed outcome=missing_sidecar session=after000 model=Qwen3 stored_messages=0 incoming_messages=1 age_s=0 cached_ratio=0.00",
]


def _write(tmp_path, lines, name="proxy.log.2026-10-29_13"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(args, tmp_path):
    return subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )


def _json_report(tmp_path):
    result = _run(["--log-dir", str(tmp_path), "--window", WINDOW, "--json"], tmp_path)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_attributes_each_missing_sidecar_to_its_write_path(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    assert report["missing_sidecar_seeds"] == 3
    assert report["missing_sidecar_attribution"] == {
        "eviction": 1,
        "no-save-in-window": 1,
        "router": 1,
    }
    assert report["sessions"] == {
        "01a187be": {"router": 1},
        "audit-AH": {"eviction": 1},
        "orphan22": {"no-save-in-window": 1},
    }


def test_tallies_outcomes_saves_and_skips(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    assert report["seed_outcomes"] == {"missing_sidecar": 3, "missing_snapshot": 1}
    assert report["successful_saves"] == {"eviction": 1, "router": 1}
    assert report["persistence_skips"] == {
        "eviction_save_failed": 1,
        "slot_busy": 1,
    }


def test_out_of_window_lines_are_excluded(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    assert "before00" not in report["sessions"]
    assert "after000" not in report["sessions"]


def test_most_recent_save_wins(tmp_path):
    _write(
        tmp_path,
        [
            "2026-10-29 13:00:00,000 - INFO - slot_save success session=01a187be slot=1",
            "2026-10-29 13:10:00,000 - INFO - slot_save session=01a187be slot=1 reason=eviction saved=True source=eviction",
            "2026-10-29 13:20:00,000 - INFO - slot_ratio_seed outcome=missing_sidecar session=01a187be model=Qwen3 stored_messages=0 incoming_messages=4 age_s=0 cached_ratio=0.00",
        ],
    )

    report = _json_report(tmp_path)

    assert report["missing_sidecar_attribution"] == {"eviction": 1}


def test_gz_logs_are_read(tmp_path):
    path = tmp_path / "proxy.log.2026-10-29_12.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(LINES) + "\n")

    report = _json_report(tmp_path)

    assert report["missing_sidecar_seeds"] == 3


def test_empty_window_is_a_zero_report(tmp_path):
    _write(tmp_path, [])

    result = _run(["--log-dir", str(tmp_path), "--window", WINDOW], tmp_path)
    assert result.returncode == 0
    assert "missing_sidecar attribution: {} (total 0 across 0 sessions)" in result.stdout


def test_text_output_summarises_the_attribution(tmp_path):
    _write(tmp_path, LINES)

    result = _run(["--log-dir", str(tmp_path), "--window", WINDOW], tmp_path)

    assert result.returncode == 0
    assert "'eviction': 1" in result.stdout
    assert "'router': 1" in result.stdout


def test_invalid_window_is_rejected(tmp_path):
    result = _run(["--log-dir", str(tmp_path), "--window", "not-a-window"], tmp_path)

    assert result.returncode == 2
    assert "window" in result.stderr.lower()


def test_missing_log_dir_fails(tmp_path):
    result = _run(
        ["--log-dir", str(tmp_path / "nope"), "--window", WINDOW], tmp_path
    )

    assert result.returncode == 1
    assert "log directory not found" in result.stderr
