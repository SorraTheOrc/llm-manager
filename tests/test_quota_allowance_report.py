#!/usr/bin/env python3
"""Tests for scripts/quota_allowance_report.py (LP-0MVF7KTMI005B667).

The script is the epic's post-deploy allowance-utilisation measurement
(parent AC13): per account, the monthly percentage at the moment of reset.
These tests drive it against synthetic proxy logs so no production corpus is
needed, and assert the observable measurement contract end to end:

- periods are derived from the poller's own ``resetsAt`` anchor (so a monthly
  rollover opens a new period and never mixes two periods);
- the reading at reset is the poll closest to *and immediately before* the
  reset, with the residual (100 - percent) and coverage gap reported;
- completed periods are classified against the >= 90% target, the hard-stop
  threshold (overspend) and missing data;
- hard-stop dwell, quota-driven skips (applied vs dry-run) and poll failures
  are attributed to the period that contains them;
- surge intervals are paired from the enter/leave transitions and clipped to
  the period's final-week window;
- the CLI records its exact invocation for reproducibility (AC13d) and writes
  both artifacts (AC13b).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "quota_allowance_report.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import quota_allowance_report as qar  # noqa: E402

UTC = dt.UTC
ACCOUNT = "OPENCODE_API_KEY"
RESET_A = dt.datetime(2026, 10, 23, 8, 0, tzinfo=UTC)
RESET_B = dt.datetime(2026, 11, 23, 8, 0, tzinfo=UTC)


def poll_line(
    ts: dt.datetime,
    monthly: float,
    *,
    account: str = ACCOUNT,
    resets_at: dt.datetime = RESET_A,
    rolling: float = 0.0,
    weekly: float = 0.0,
) -> str:
    """One ``quota poll ok`` log line, as proxy/quota_state.py writes it."""
    seconds = int((resets_at - ts).total_seconds())
    return (
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - quota poll ok "
        f"account={account} rolling={rolling:.1f}% weekly={weekly:.1f}% "
        f"monthly={monthly:.1f}% seconds_to_reset={seconds}s "
        f"effective_headroom={100.0 - monthly:.1f}\n"
    )


def surge_line(ts: dt.datetime, state: str, *, account: str = ACCOUNT) -> str:
    return (
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - quota_surge_bypass "
        f"{state} provider=opencode-go account={account} "
        f"resets_at=2026-10-23T08:00:00Z seconds_to_reset=1s "
        f"reason=quota_surge\n"
    )


def skip_line(ts: dt.datetime, monthly: float, *, account: str = ACCOUNT) -> str:
    return (
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - Skipping "
        f"provider=opencode-go: reason=quota_exhausted (account={account}, "
        f"monthly_percent={monthly:.1f}%, effective_headroom=0.0%, "
        f"threshold=99.5%)\n"
    )


def dry_run_skip_line(ts: dt.datetime, *, account: str = ACCOUNT) -> str:
    return (
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - quota_skip dry_run "
        f"would_skip=true provider=opencode-go account={account} "
        f"effective_headroom=0.0% resets_in=60s reason=quota_surge_dry_run\n"
    )


def poll_failure_line(ts: dt.datetime, *, account: str = ACCOUNT) -> str:
    return (
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},123 - WARNING - quota poll failed "
        f"account={account} error=ConnectError keeping_last_known=true\n"
    )


def write_log(tmp_path: Path, lines: list[str]) -> Path:
    (tmp_path / "proxy.log").write_text("".join(lines))
    return tmp_path


def run(log_dir: Path, **overrides):
    """Run the measurement over *log_dir* with a fixed, completed window."""
    kwargs = {
        "start": dt.datetime(2026, 10, 19, 0, 0, tzinfo=UTC),
        "end": dt.datetime(2026, 11, 25, 0, 0, tzinfo=UTC),
        "threshold": qar.DEFAULT_THRESHOLD,
        "surge_days": qar.DEFAULT_SURGE_DAYS,
        "max_poll_gap_hours": qar.DEFAULT_MAX_POLL_GAP_HOURS,
    }
    kwargs.update(overrides)
    return qar.run_measurement(log_dir, **kwargs)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_events_keeps_poll_samples_and_ignores_noise(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 20, 0, 0, tzinfo=UTC), 40.0),
            "2026-10-20 00:01:00,000 - INFO - unrelated dispatch line\n",
            # Outside the window: must not contribute a reading.
            poll_line(dt.datetime(2026, 10, 1, 0, 0, tzinfo=UTC), 99.0),
            # Malformed timestamp: skipped, never fatal.
            "not-a-timestamp - INFO - quota poll ok account=X monthly=99.0%\n",
        ],
    )
    events = qar.parse_events(
        log,
        dt.datetime(2026, 10, 19, tzinfo=UTC),
        dt.datetime(2026, 11, 25, tzinfo=UTC),
    )
    assert list(events) == [ACCOUNT]
    assert [s.monthly for s in events[ACCOUNT].samples] == [40.0]
    assert events[ACCOUNT].samples[0].resets_at == RESET_A


def test_resets_at_is_derived_from_the_poll_countdown(tmp_path):
    ts = dt.datetime(2026, 10, 22, 8, 0, tzinfo=UTC)
    events = qar.parse_events(
        write_log(tmp_path, [poll_line(ts, 50.0)]),
        dt.datetime(2026, 10, 19, tzinfo=UTC),
        dt.datetime(2026, 11, 25, tzinfo=UTC),
    )
    sample = events[ACCOUNT].samples[0]
    assert sample.resets_at == RESET_A
    assert sample.effective_headroom == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Period derivation
# ---------------------------------------------------------------------------


def test_rollover_opens_a_new_period_and_never_mixes_readings(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 23, 6, 0, tzinfo=UTC), 93.0),
            poll_line(dt.datetime(2026, 10, 23, 7, 0, tzinfo=UTC), 93.0),
            # Rollover: the poller now reports the next reset and a fresh period.
            poll_line(
                dt.datetime(2026, 10, 23, 9, 0, tzinfo=UTC),
                2.0,
                resets_at=RESET_B,
            ),
        ],
    )
    result = run(log)
    periods = result["accounts"][ACCOUNT]
    assert [p["resets_at"] for p in periods] == [
        "2026-10-23T08:00:00Z",
        "2026-11-23T08:00:00Z",
    ]
    assert periods[0]["completed"] is True
    assert periods[0]["poll_count"] == 2
    assert periods[0]["monthly_percent_at_reset"] == pytest.approx(93.0)
    assert periods[1]["completed"] is True
    assert periods[1]["poll_count"] == 1


def test_period_anchors_cluster_seconds_of_drift_into_one_period():
    samples = [
        qar.PollSample(
            ts=dt.datetime(2026, 10, 22, tzinfo=UTC),
            account=ACCOUNT,
            rolling=0.0,
            weekly=0.0,
            monthly=10.0,
            seconds_to_reset=86400 + drift,
            effective_headroom=90.0,
        )
        for drift in (0, 30, -30)
    ]
    anchors = qar.period_anchors(samples)
    assert len(anchors) == 1


def test_period_starting_after_the_window_end_is_not_completed(tmp_path):
    log = write_log(
        tmp_path,
        [poll_line(dt.datetime(2026, 11, 22, 6, 0, tzinfo=UTC), 96.0, resets_at=RESET_B)],
    )
    result = run(log, end=dt.datetime(2026, 11, 23, 0, 0, tzinfo=UTC))
    period = result["accounts"][ACCOUNT][0]
    assert period["completed"] is False
    assert period["outcome"] == "in_progress"
    # A projected reading is never reported as a moment-of-reset measurement.
    assert period["monthly_percent_at_reset"] is None
    assert period["residual_percent"] is None
    assert period["monthly_percent_latest"] == pytest.approx(96.0)
    # An in-progress period never counts toward the completed-period summary.
    assert result["summary"]["completed_periods"] == 0


# ---------------------------------------------------------------------------
# Reading at reset and outcome classification
# ---------------------------------------------------------------------------


def test_percent_at_reset_is_the_last_poll_before_the_reset(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 23, 5, 0, tzinfo=UTC), 91.0),
            poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 93.0),
            # A later reading inside the *next* period must not be used.
            poll_line(dt.datetime(2026, 10, 23, 9, 0, tzinfo=UTC), 4.0, resets_at=RESET_B),
        ],
    )
    period = run(log)["accounts"][ACCOUNT][0]
    assert period["monthly_percent_at_reset"] == pytest.approx(93.0)
    assert period["residual_percent"] == pytest.approx(7.0)
    assert period["coverage_gap_hours"] == pytest.approx(0.25)
    assert period["reading_approximate"] is False


@pytest.mark.parametrize(
    ("percent_at_reset", "expected"),
    [
        (93.0, "target_met"),
        (90.0, "target_met"),
        (89.9, "below_target"),
        (99.5, "overspend"),
        (100.0, "overspend"),
    ],
)
def test_outcome_classification_against_target_and_hard_stop(
    tmp_path, percent_at_reset, expected
):
    log = write_log(
        tmp_path,
        [poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), percent_at_reset)],
    )
    period = run(log)["accounts"][ACCOUNT][0]
    assert period["outcome"] == expected


def test_two_accounts_are_reported_independently(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 93.0),
            poll_line(
                dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC),
                55.0,
                account="OPENCODE_2_API_KEY",
            ),
        ],
    )
    result = run(log)
    assert set(result["accounts"]) == {ACCOUNT, "OPENCODE_2_API_KEY"}
    assert result["accounts"][ACCOUNT][0]["monthly_percent_at_reset"] == pytest.approx(93.0)
    assert (
        result["accounts"]["OPENCODE_2_API_KEY"][0]["monthly_percent_at_reset"]
        == pytest.approx(55.0)
    )


def test_large_cost_gap_marks_the_reading_approximate(tmp_path):
    log = write_log(
        tmp_path,
        [poll_line(dt.datetime(2026, 10, 22, 0, 0, tzinfo=UTC), 88.0)],
    )
    period = run(log, max_poll_gap_hours=1.0)["accounts"][ACCOUNT][0]
    assert period["coverage_gap_hours"] == pytest.approx(32.0)
    assert period["reading_approximate"] is True


# ---------------------------------------------------------------------------
# Hard stop, skips, failures
# ---------------------------------------------------------------------------


def test_hard_stop_dwell_is_measured_from_consecutive_samples(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 23, 4, 0, tzinfo=UTC), 99.0),
            poll_line(dt.datetime(2026, 10, 23, 4, 15, tzinfo=UTC), 99.6),
            poll_line(dt.datetime(2026, 10, 23, 4, 30, tzinfo=UTC), 99.7),
            poll_line(dt.datetime(2026, 10, 23, 4, 45, tzinfo=UTC), 99.8),
        ],
    )
    period = run(log)["accounts"][ACCOUNT][0]
    # 04:15 → 04:45 dwell above the 99.5% threshold.
    assert period["hard_stop_hours"] == pytest.approx(0.5)
    assert period["outcome"] == "overspend"


def test_quota_skips_dry_run_skips_and_poll_failures_are_attributed(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 23, 6, 0, tzinfo=UTC), 93.0),
            skip_line(dt.datetime(2026, 10, 23, 6, 5, tzinfo=UTC), 99.6),
            skip_line(dt.datetime(2026, 10, 23, 6, 10, tzinfo=UTC), 99.6),
            dry_run_skip_line(dt.datetime(2026, 10, 23, 6, 15, tzinfo=UTC)),
            poll_failure_line(dt.datetime(2026, 10, 23, 6, 20, tzinfo=UTC)),
        ],
    )
    period = run(log)["accounts"][ACCOUNT][0]
    assert period["quota_skip_count"] == 2
    assert period["quota_skip_dry_run_count"] == 1
    assert period["poll_failures"] == 1


# ---------------------------------------------------------------------------
# Surge activity
# ---------------------------------------------------------------------------


def test_surge_intervals_are_paired_and_clipped_to_the_final_window(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 10, 0, 0, tzinfo=UTC), 20.0),
            poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 93.0),
            # 10 days of surge; only the part inside the final 7-day window
            # (from 2026-10-16T08:00, the reset minus 7 days) is counted there.
            surge_line(dt.datetime(2026, 10, 10, 0, 0, tzinfo=UTC), "entered"),
            surge_line(dt.datetime(2026, 10, 20, 0, 0, tzinfo=UTC), "left"),
        ],
    )
    period = run(log, start=dt.datetime(2026, 10, 9, tzinfo=UTC))["accounts"][
        ACCOUNT
    ][0]
    assert period["surge_active_hours"] == pytest.approx(240.0)
    assert period["surge_final_window_hours"] == pytest.approx(88.0)


def test_unclosed_surge_entry_is_reported_as_presence_only(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 10, 0, 0, tzinfo=UTC), 20.0),
            poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 93.0),
            surge_line(dt.datetime(2026, 10, 22, 0, 0, tzinfo=UTC), "entered"),
        ],
    )
    period = run(log, start=dt.datetime(2026, 10, 9, tzinfo=UTC))["accounts"][
        ACCOUNT
    ][0]
    # No matching `left`: zero-length interval, never a fabricated duration.
    assert period["surge_active_hours"] == 0.0


def test_surge_before_the_first_observed_sample_is_clipped(tmp_path):
    # The first period's lower bound is its first sample (measurement starts at
    # deployment), so a surge interval entirely before it is not observable.
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 40.0),
            surge_line(dt.datetime(2026, 10, 19, 9, 0, tzinfo=UTC), "entered"),
            surge_line(dt.datetime(2026, 10, 19, 10, 0, tzinfo=UTC), "left"),
        ],
    )
    period = run(log)["accounts"][ACCOUNT][0]
    assert period["surge_active_hours"] == 0.0
    assert period["surge_final_window_hours"] == 0.0


def test_surge_inside_the_period_final_week_is_counted(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 22, 2, 0, tzinfo=UTC), 40.0),
            poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 93.0),
            surge_line(dt.datetime(2026, 10, 22, 0, 0, tzinfo=UTC), "entered"),
            surge_line(dt.datetime(2026, 10, 22, 12, 0, tzinfo=UTC), "left"),
        ],
    )
    period = run(log)["accounts"][ACCOUNT][0]
    # The interval clipped to [first sample 02:00, reset 08:00] = 10 h, all of
    # it inside the final 7-day window.
    assert period["surge_active_hours"] == pytest.approx(10.0)
    assert period["surge_final_window_hours"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# CLI: artifacts, invocation record, exit codes
# ---------------------------------------------------------------------------


def test_cli_writes_markdown_and_json_with_the_exact_invocation(tmp_path):
    log = write_log(
        tmp_path,
        [
            poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 93.0),
            poll_line(dt.datetime(2026, 10, 23, 9, 0, tzinfo=UTC), 2.0, resets_at=RESET_B),
        ],
    )
    md = tmp_path / "out" / "report.md"
    js = tmp_path / "out" / "report.json"
    argv = [
        "--log-dir", str(log),
        "--start", "2026-10-19T00:00:00",
        "--end", "2026-10-25T00:00:00",
        "--output-md", str(md),
        "--output-json", str(js),
    ]
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    artifact = json.loads(js.read_text())
    invocation = artifact["invocation"]
    assert invocation["argv"].startswith("scripts/quota_allowance_report.py")
    assert "--start 2026-10-19T00:00:00" in invocation["argv"]
    assert invocation["window"] == {
        "start": "2026-10-19T00:00:00Z",
        "end": "2026-10-25T00:00:00Z",
    }
    assert invocation["threshold_percent"] == qar.DEFAULT_THRESHOLD
    assert invocation["log_dir"] == str(log)

    accounts = artifact["accounts"][ACCOUNT]
    assert accounts[0]["monthly_percent_at_reset"] == pytest.approx(93.0)
    assert accounts[0]["outcome"] == "target_met"
    assert artifact["target_percent"] == 90.0
    assert artifact["summary"]["target_met"] == 1

    report = md.read_text()
    assert "Target:** ≥ 90% monthly consumption at reset" in report
    assert "OPENCODE_API_KEY" in report
    assert "target met" in report.lower()


def test_production_log_line_formats_are_parsed(tmp_path):
    # The exact message formats emitted by proxy/quota_state.py and
    # proxy/provider.py — copied from the source so the measurement cannot
    # silently drift from the production corpus it reads.
    poll_ts = dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC)
    seconds = int((RESET_A - poll_ts).total_seconds())
    lines = [
        # proxy/quota_state.py _poll_account INFO line.
        f"{poll_ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - quota poll ok "
        f"account={ACCOUNT} rolling=0.0% weekly=0.0% monthly=93.0% "
        f"seconds_to_reset={seconds}s effective_headroom=7.0\n",
        # proxy/quota_state.py _warn_stale_suspension WARN line.
        f"{poll_ts.strftime('%Y-%m-%d %H:%M:%S')},123 - WARNING - quota surge "
        f"suspended account={ACCOUNT} last_poll_age=7200s max_age=3600s "
        f"reason=quota_surge_suspended fail_safe=true effective_headroom=7.0% "
        f"resets_in=900s\n",
        # proxy/quota_state.py _resume_suspension INFO line.
        f"{poll_ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - quota surge "
        f"resumed account={ACCOUNT} reason=quota_surge_resumed\n",
        # proxy/provider.py _quota_exhausted reason (inside the 'Skipping'
        # INFO line written by resolve_provider).
        f"{poll_ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - Skipping "
        f"provider=opencode-go: reason=quota_exhausted (account={ACCOUNT}, "
        f"monthly_percent=99.6%, effective_headroom=0.4%, threshold=99.5%)\n",
        # proxy/provider.py _log_surge_bypass_transition INFO line.
        f"{poll_ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - quota_surge_bypass "
        f"entered provider=opencode-go account={ACCOUNT} "
        f"resets_at={RESET_A.isoformat().replace('+00:00', 'Z')} "
        f"seconds_to_reset={seconds}s reason=quota_surge\n",
        # proxy/provider.py _log_quota_skip_decision dry-run INFO line.
        f"{poll_ts.strftime('%Y-%m-%d %H:%M:%S')},123 - INFO - quota_skip "
        f"dry_run would_skip=true provider=opencode-go account={ACCOUNT} "
        f"effective_headroom=0.4% resets_in=900s reason=quota_surge_dry_run\n",
    ]
    log = write_log(tmp_path, lines)
    events = qar.parse_events(
        log,
        dt.datetime(2026, 10, 19, tzinfo=UTC),
        dt.datetime(2026, 11, 25, tzinfo=UTC),
    )
    entry = events[ACCOUNT]
    assert len(entry.samples) == 1
    assert entry.samples[0].resets_at == RESET_A
    assert entry.suspensions == [poll_ts]
    assert entry.skips == [poll_ts]
    assert entry.surge_entered == [poll_ts]
    assert entry.dry_run_skips == [poll_ts]


def test_cli_is_reproducible_for_the_same_inputs(tmp_path):
    log = write_log(
        tmp_path,
        [poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 78.0)],
    )
    outputs = []
    for name in ("a.json", "b.json"):
        js = tmp_path / name
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--log-dir", str(log),
                "--start", "2026-10-19T00:00:00",
                "--end", "2026-10-25T00:00:00",
                "--output-json", str(js),
                "--quiet",
            ],
            check=True,
        )
        payload = json.loads(js.read_text())
        payload.pop("generated_at")
        outputs.append(payload)
    assert outputs[0]["accounts"] == outputs[1]["accounts"]
    assert outputs[0]["accounts"][ACCOUNT][0]["outcome"] == "below_target"


def test_cli_fails_cleanly_without_quota_data(tmp_path):
    log = write_log(tmp_path, ["2026-10-20 00:00:00,000 - INFO - nothing here\n"])
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log-dir", str(log),
            "--start", "2026-10-19T00:00:00",
            "--end", "2026-10-25T00:00:00",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "no quota poll data" in completed.stderr


def test_cli_json_flag_prints_the_artifact(tmp_path):
    log = write_log(
        tmp_path,
        [poll_line(dt.datetime(2026, 10, 23, 7, 45, tzinfo=UTC), 93.0)],
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log-dir", str(log),
            "--start", "2026-10-19T00:00:00",
            "--end", "2026-10-25T00:00:00",
            "--json",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(completed.stdout)
    assert artifact["accounts"][ACCOUNT][0]["outcome"] == "target_met"
