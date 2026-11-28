#!/usr/bin/env python3
"""Tests for scripts/slots_snapshot_age_distribution.py (LP-0MWLZ68SF0052E7Z).

The script is the reproducible "tightened reading" of the `slots_stale`
window (F6 / LP-0MVVCO1N7004TNGY AC2, plan question Q3): the as-configured
verdict is judged at the delivered ceiling (`slots_max_snapshot_age_seconds`,
300 s by default), but the plan proposed a 2 s ceiling, so the same polls must
also be reported at that tighter value together with the snapshot-age
distribution.

These tests run the CLI against a synthetic proxy log, so no production log is
needed, and pin the observable contract:

  * window filtering is per-line and half-open (`since <= ts < until`), the
    same rule ``scripts/lib/proxy_logs.py`` gives every other window tool;
  * a poll that carries ``slots_snapshot_age_seconds=None`` is **undatable**
    (counted, never silently dropped) and a poll with no age field at all is a
    pre-F1 poll, counted separately;
  * percentiles use the shared nearest-rank definition, so they are
    reproducible from the sample;
  * a threshold reading reports the count and the rate over both the datable
    and the whole status-poll population, plus the ``capacity_hint`` split that
    answers "how much of it sits in the prefill-blocked regime";
  * ``over_ceiling`` + ``undatable`` reconciles with the tool's contract
    ``slots_stale`` count (age above the ceiling **or** no datable snapshot).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "slots_snapshot_age_distribution.py",
    )
)

FIELD = (
    "capacity_hint={hint} status_query_ms=5 slots_counts_ms=0 "
    "slots_detail_ms=0 slots_max_snapshot_age_seconds=300.0 "
    "slots_snapshot_age_seconds={age} slots_stale=false"
)

SYNTH = (
    "\n".join(
        [
            # outside the window (upper bound is exclusive)
            f"2026-11-18 10:24:59,000 - INFO - status_request {FIELD.format(hint='idle', age='0.1')}",
            # inside: fresh, over the 2 s threshold, undatable, over the ceiling
            f"2026-11-18 10:25:01,000 - INFO - status_request {FIELD.format(hint='idle', age='0.5')}",
            f"2026-11-18 10:25:02,000 - INFO - status_request {FIELD.format(hint='prefill_bound', age='3.0')}",
            f"2026-11-18 10:25:03,000 - INFO - status_request {FIELD.format(hint='idle', age='None')}",
            f"2026-11-18 10:25:04,000 - INFO - status_request {FIELD.format(hint='saturated', age='301.5')}",
            # inside: a pre-F1 poll (no age field at all)
            "2026-11-18 10:25:05,000 - INFO - status_request capacity_hint=idle status_query_ms=5",
            # not a status poll
            "2026-11-18 10:25:06,000 - INFO - some_other_event slots_snapshot_age_seconds=999.0",
            # inside the half-open window's upper bound (excluded)
            f"2026-11-19 10:25:00,000 - INFO - status_request {FIELD.format(hint='idle', age='40.0')}",
        ]
    )
    + "\n"
)

SINCE = "2026-11-18 10:25:00"
UNTIL = "2026-11-19 10:25:00"


def run_cli(log_dir, *extra):
    proc = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--log-dir",
            str(log_dir),
            "--since",
            SINCE,
            "--until",
            UNTIL,
            "--json",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    return proc


def _write_log(tmp_path):
    (tmp_path / "proxy.log").write_text(SYNTH, encoding="utf-8")
    return tmp_path


class TestWindowAndPopulation:
    def test_exit_zero_and_json(self, tmp_path):
        proc = run_cli(_write_log(tmp_path))
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)
        assert report["tool"] == "scripts/slots_snapshot_age_distribution.py"
        assert report["window"] == {"since": SINCE, "until": UNTIL, "half_open": True}

    def test_population_is_split_into_datable_undatable_and_poll_age_field(self, tmp_path):
        report = json.loads(run_cli(_write_log(tmp_path)).stdout)
        assert report["polls"] == {
            "status_polls": 5,
            "polls_with_age_field": 4,
            "datable": 3,
            "undatable": 1,
            "no_age_field": 1,
        }

    def test_window_bounds_are_per_line_and_half_open(self, tmp_path):
        # The 10:24:59 and 10:25:00 lines must be excluded by the bounds.
        report = json.loads(run_cli(_write_log(tmp_path)).stdout)
        assert report["age_seconds"]["max"] == 301.5


class TestDistribution:
    def test_nearest_rank_percentiles(self, tmp_path):
        report = json.loads(run_cli(_write_log(tmp_path)).stdout)
        assert report["age_seconds"] == {
            "count": 3,
            "min": 0.5,
            "p50": 3.0,
            "p90": 301.5,
            "p99": 301.5,
            "max": 301.5,
        }

    def test_ceiling_reconciliation_counts_undatable_as_stale(self, tmp_path):
        report = json.loads(run_cli(_write_log(tmp_path)).stdout)
        assert report["ceiling_seconds"] == 300.0
        assert report["contract_stale"] == {"count": 2, "rate_of_status_polls": 0.4}


class TestThresholdReading:
    def test_default_threshold_is_two_seconds(self, tmp_path):
        report = json.loads(run_cli(_write_log(tmp_path)).stdout)
        assert report["thresholds"][0]["threshold_seconds"] == 2.0

    def test_threshold_reports_both_denominators_and_hint_split(self, tmp_path):
        reading = json.loads(run_cli(_write_log(tmp_path)).stdout)["thresholds"][0]
        assert reading["over"] == 2
        # rates use the shared ``share()`` (four decimals), like every other
        # measurement surface in ``scripts/lib/capacity_signals.py``
        assert reading["rate_of_datable"] == 0.6667
        assert reading["rate_of_status_polls"] == 0.4
        assert reading["by_capacity_hint"] == {"prefill_bound": 1, "saturated": 1}

    def test_threshold_is_repeatable_and_sorted(self, tmp_path):
        report = json.loads(
            run_cli(_write_log(tmp_path), "--threshold-seconds", "300", "--threshold-seconds", "2").stdout
        )
        assert [r["threshold_seconds"] for r in report["thresholds"]] == [2.0, 300.0]
        assert [r["over"] for r in report["thresholds"]] == [2, 1]

    def test_threshold_with_no_value_over_is_zero_not_none(self, tmp_path):
        report = json.loads(run_cli(_write_log(tmp_path), "--threshold-seconds", "1000").stdout)
        assert report["thresholds"][0]["over"] == 0
        assert report["thresholds"][0]["by_capacity_hint"] == {}


class TestErrors:
    def test_missing_log_dir_exits_one(self, tmp_path):
        proc = run_cli(tmp_path / "nope")
        assert proc.returncode == 1
        assert "log directory not found" in proc.stderr

    def test_empty_log_file_reports_zero_polls_without_fabricated_numbers(self, tmp_path):
        (tmp_path / "proxy.log").write_text("", encoding="utf-8")
        proc = run_cli(tmp_path)
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)
        assert report["polls"]["status_polls"] == 0
        assert report["age_seconds"] == {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
        assert report["contract_stale"] == {"count": 0, "rate_of_status_polls": 0.0}
