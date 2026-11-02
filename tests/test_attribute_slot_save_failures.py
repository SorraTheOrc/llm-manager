#!/usr/bin/env python3
"""Tests for scripts/attribute_slot_save_failures.py (LP-0MVVX6RM3001P4EY).

The script is the AC1/AC2 evidence tool for the slot-save failure cliff: it
counts save attempts per path (router vs eviction), ok vs failed, failure
reasons (ReadTimeout etc.), persistence skips, and — because the proxy only
logs save duration on the failure path — states explicitly whether a success
duration distribution exists. It also attributes ``slot_ratio_seed`` outcomes
to the *most recent save attempt*, including failed attempts, so the
``missing_sidecar``/``missing_snapshot`` seeds caused by a failed save are
quantified (AC4).

The key mechanism the script pins: ``slot_save failed slot=N …`` lines do not
carry ``session=``/``source=``, but every eviction attempt logs a paired
``slot_save session=… saved=False source=eviction`` line at the same
timestamp — so a failure is attributed to the eviction path when a same-slot
paired line exists within 2 s, and to the router path otherwise.
"""

import gzip
import json
import os
import subprocess
import sys

SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "attribute_slot_save_failures.py")
)

WINDOW = "2026-10-29T12:58:32,2026-10-30T12:58:32"

# Inside the window unless noted. Session prefixes are 8 chars, as logged.
LINES = [
    # Router save success -> router attempt ok.
    "2026-10-29 13:00:00,000 - INFO - slot_save success session=01a187be slot=1",
    "2026-10-29 13:00:05,000 - INFO - slot_ratio_seed outcome=missing_sidecar session=01a187be model=Qwen3 stored_messages=0 incoming_messages=42 age_s=0 cached_ratio=0.00",
    # Eviction save success -> eviction attempt ok.
    "2026-10-29 14:00:00,000 - INFO - slot_save session=audit-AH slot=0 reason=tenancy_lease_expired saved=True source=eviction",
    "2026-10-29 14:00:09,000 - INFO - slot_ratio_seed outcome=missing_snapshot session=audit-AH model=Qwen3 stored_messages=0 incoming_messages=55 age_s=0 cached_ratio=0.00",
    # Eviction save FAILURE: the failure warning (no session=) is paired with
    # the identical-timestamp saved=False eviction line.
    "2026-10-29 15:00:00,000 - WARNING - slot_save failed slot=0 error=ReadTimeout/ReadTimeout elapsed=3.0s timeout=3.0s busy={\"active_queries\": 1, \"local_active_queries\": 1, \"active_sessions\": 1, \"slot_busy\": false}",
    "2026-10-29 15:00:00,001 - INFO - slot_save session=herdr-17 slot=0 reason=lease_evicted saved=False source=eviction",
    "2026-10-29 15:00:04,000 - INFO - slot_ratio_seed outcome=missing_snapshot session=herdr-17 model=Qwen3 stored_messages=0 incoming_messages=9 age_s=0 cached_ratio=0.00",
    # Router save FAILURE: no paired eviction line -> router path.
    "2026-10-29 16:00:00,000 - WARNING - slot_save failed slot=1 error=ReadTimeout/ReadTimeout elapsed=35.6s timeout=35.6s busy={\"active_queries\": 0, \"local_active_queries\": 0, \"active_sessions\": 0, \"slot_busy\": false}",
    # A slot_busy persistence skip (not a failure, but part of the cliff).
    "2026-10-29 17:00:00,000 - INFO - slot persistence skipped session=01a187be slot=0 reason=slot_busy active_sessions=2",
    # No DEBUG success-duration line present in the deployed logs: the report
    # must state success durations are not logged (INFO level deployment).
    "2026-10-29 18:00:00,000 - INFO - slot_save success session=01a187be slot=0",
    # Outside the window on both sides.
    "2026-10-29 12:00:00,000 - INFO - slot_save success session=before00 slot=0",
    "2026-10-30 13:00:00,000 - INFO - slot_save success session=after000 slot=0",
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


def test_per_path_attempts_and_failures(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    assert report["per_path"] == {
        "router": {"attempts": 3, "ok": 2, "failed": 1},
        "eviction": {"attempts": 2, "ok": 1, "failed": 1},
    }


def test_failure_reasons_elapsed_and_timeout(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    assert report["failure_reasons"] == {"ReadTimeout/ReadTimeout": 2}
    assert report["failure_elapsed_seconds"] == {"3.0": 1, "35.6": 1}
    assert report["timeout_used_seconds"] == {"3.0": 1, "35.6": 1}


def test_failures_carry_path_and_paired_session(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    by_path = {f["path"]: f for f in report["failures"]}
    assert by_path["eviction"]["session"] == "herdr-17"
    assert by_path["eviction"]["elapsed_s"] == 3.0
    assert by_path["eviction"]["busy"]["active_queries"] == 1
    assert by_path["router"]["session"] is None


def test_success_duration_is_stated_as_not_logged(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    assert report["success_duration_logged"] is False


def test_persistence_skips_are_tallied(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    assert report["persistence_skips"] == {"slot_busy": 1}


def test_seeds_attributed_including_failed_saves(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    assert report["seed_outcomes"] == {"missing_sidecar": 1, "missing_snapshot": 2}
    # 01a187be -> router-ok; audit-AH -> eviction-ok; herdr-17 -> eviction-failed.
    assert report["seeds_by_last_attempt"]["missing_sidecar"] == {"router": 1}
    assert report["seeds_by_last_attempt"]["missing_snapshot"] == {"eviction": 1, "eviction-failed": 1}
    # AC4: who is attributable to a failed save?
    assert report["failed_save_attribution"]["missing_sidecar"] == 0
    assert report["failed_save_attribution"]["missing_snapshot"] == 1


def test_source_field_is_preferred_over_pairing_inference(tmp_path):
    """Post-fix logs carry source= on the failure line: trust it."""
    _write(
        tmp_path,
        [
            "2026-10-29 15:00:00,000 - WARNING - slot_save failed slot=0 error=ReadTimeout/ReadTimeout elapsed=3.0s timeout=3.0s busy={\"active_queries\": 1, \"local_active_queries\": 1, \"active_sessions\": 1, \"slot_busy\": false} source=eviction",
            "2026-10-29 15:00:00,001 - INFO - slot_save session=herdr-17 slot=0 reason=lease_evicted saved=False source=eviction",
            "2026-10-29 16:00:00,000 - WARNING - slot_save failed slot=1 error=ReadTimeout/ReadTimeout elapsed=35.6s timeout=35.6s busy={\"active_queries\": 0, \"local_active_queries\": 0, \"active_sessions\": 0, \"slot_busy\": false} source=router",
        ],
    )

    report = _json_report(tmp_path)

    by_path = {f["path"]: f for f in report["failures"]}
    assert set(by_path) == {"eviction", "router"}
    assert by_path["eviction"]["session"] == "herdr-17"
    assert report["per_path"]["eviction"]["failed"] == 1
    assert report["per_path"]["router"]["failed"] == 1


def test_gz_logs_are_read(tmp_path):
    path = tmp_path / "proxy.log.2026-10-29_12.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(LINES) + "\n")

    report = _json_report(tmp_path)

    assert report["per_path"]["router"]["attempts"] == 3


def test_out_of_window_lines_are_excluded(tmp_path):
    _write(tmp_path, LINES)

    report = _json_report(tmp_path)

    sessions = {f.get("session") for f in report["failures"]}
    assert "before00" not in sessions
    assert "after000" not in [s for s in sessions if s]


def test_empty_window_is_a_zero_report(tmp_path):
    _write(tmp_path, [])

    result = _run(["--log-dir", str(tmp_path), "--window", WINDOW], tmp_path)
    assert result.returncode == 0
    assert "router: 0 attempts" in result.stdout


def test_text_output_summarises_paths(tmp_path):
    _write(tmp_path, LINES)

    result = _run(["--log-dir", str(tmp_path), "--window", WINDOW], tmp_path)

    assert result.returncode == 0
    assert "router: 3 attempts (2 ok / 1 failed)" in result.stdout
    assert "eviction: 2 attempts (1 ok / 1 failed)" in result.stdout


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
