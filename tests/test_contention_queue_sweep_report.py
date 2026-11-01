#!/usr/bin/env python3
"""Tests for ``scripts/contention_queue_sweep_report.py`` (LP-0MVT110PJ003XYLP).

The script is the residual-queue telemetry behind the 60 s → 30 s contention
sweep (parent LP-0MVIKPRMD000Y97R). It is a pure log analyser: these tests drive
the CLI against synthetic proxy-log fixtures, so no production log and no
running proxy is required.

The contract asserted here is the one the acceptance criteria are written
against:

* (a) ticket-backed dispatches — ``local_admission_reserve_claimed`` count;
* (b) residual queue wins — ``contention_queue_dispatch`` count, the share whose
  ``queued_duration`` is at or below the sweep budget, and the wait
  distribution (p50/p75/p90/max);
* (c) residual queue losses — ``contention_queue_fallback_after_queue`` count,
  aggregate wait, wait-per-hour and the ``reason`` split, where a missing
  ``reason=`` is counted as ``none`` and never dropped;
* exit codes mirror ``scripts/idle_admit_impact.py``: 0 complete, 2 incomplete
  (no logs / thin window / coverage gap — never a silent pass), 1 invalid input.

Rotation handling (plain and ``.gz``), per-line window filtering and the
message-body anchor that stops a proxied request body quoting a marker from
inflating the counts are asserted too.
"""

import gzip
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "contention_queue_sweep_report.py")
)

SINCE = "2026-10-29 00:00"
UNTIL = "2026-10-30 06:00"  # 30 h — a closed, >= 24 h baseline window
SWEEP_BUDGET_S = "30"


# ---------------------------------------------------------------------------
# Synthetic log helpers
# ---------------------------------------------------------------------------


def _ts(base: str, seconds: int) -> str:
    """A proxy-log timestamp (``YYYY-MM-DD HH:MM:SS,mmm``) at *base* + seconds."""
    moment = datetime.strptime(base, "%Y-%m-%d %H:%M") + timedelta(seconds=seconds)
    return moment.strftime("%Y-%m-%d %H:%M:%S,000")


def _line(ts: str, body: str) -> str:
    return f"{ts} - INFO - {body}\n"


def _win(ts: str, session: str, queued: float, *, reason: str | None = None, depth: int = 0) -> str:
    """A ``contention_queue_dispatch`` line (a residual queue win)."""
    body = (
        f"contention_queue_dispatch provider=local-qwen3 session={session} "
        f"queued_duration={queued:.2f}s policy=queue depth={depth}"
    )
    if reason is not None:
        body += f" reason={reason}"
    return _line(ts, body)


def _loss(
    ts: str, session: str, queued: float, *, reason: str | None = None
) -> str:
    """A ``contention_queue_fallback_after_queue`` line (a residual loss)."""
    body = (
        f"contention_queue_fallback_after_queue provider=local-qwen3 "
        f"session={session} queued_duration={queued:.2f}s preference=prefer-local"
    )
    if reason is not None:
        body += f" reason={reason}"
    return _line(ts, body)


def _ticket(ts: str, session: str, ticket: str, tokens: int) -> str:
    """A ``local_admission_reserve_claimed`` line (a ticket-backed dispatch)."""
    return _line(
        ts,
        f"local_admission_reserve_claimed ticket={ticket} session={session} tokens={tokens}",
    )


def _baseline_lines() -> list[str]:
    """The reference fixture: 4 wins, 4 losses, 2 claimed tickets in-window.

    Wins carry 4.5 s / 12 s (lease site) / 45 s (over the 30 s budget) / 30 s
    (exactly at the budget) waits; losses carry one reason-less line (``none``),
    two ``local_lease_active`` lines and one other reason. Two events sit
    outside the window, and one in-window line only *mentions* a loss marker
    inside a proxied request body.
    """
    return [
        # (b) residual queue wins
        _win(_ts(SINCE, 3600), "win-a", 4.5),
        _win(_ts(SINCE, 7200), "win-b", 12.0, reason="local_lease_active", depth=1),
        _win(_ts(SINCE, 10_800), "win-c", 45.0, depth=2),
        _win(_ts(SINCE, 14_400), "win-d", 30.0),
        # (c) residual queue losses
        _loss(_ts(SINCE, 18_000), "loss-a", 60.0),
        _loss(_ts(SINCE, 21_600), "loss-b", 60.0, reason="local_lease_active"),
        _loss(_ts(SINCE, 25_200), "loss-c", 60.0, reason="local_lease_active"),
        _loss(_ts(SINCE, 28_800), "loss-d", 55.5, reason="prefill_deadline"),
        # (a) ticket-backed dispatches
        _ticket(_ts(SINCE, 32_400), "ticket-a", "tk-1", 41_000),
        _ticket(_ts(SINCE, 33_000), "ticket-b", "tk-2", 52_000),
        # out of window (before since / after until)
        _win("2026-10-28 23:00:00,000", "too-early", 5.0),
        _loss("2026-10-30 07:00:00,000", "too-late", 60.0),
        # a proxied request body that merely quotes a loss marker
        _line(
            _ts(SINCE, 36_000),
            '[remote] POST /v1/chat/completions status=200 body={"note": '
            '"2026-10-29 10:00:00,000 - INFO - '
            'contention_queue_fallback_after_queue provider=x session=echo '
            'queued_duration=60.00s"}',
        ),
    ]


def _write_logs(directory, lines, *, name="proxy.log-2026-10-29_00", compress=False):
    path = directory / (name + ".gz" if compress else name)
    payload = "".join(lines)
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _run(args, **kwargs):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True,
        text=True,
        **kwargs,
    )


def _report(tmp_path, extra=()):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    _write_logs(log_dir, _baseline_lines())
    result = _run(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            SINCE,
            "--until",
            UNTIL,
            "--sweep-budget-seconds",
            SWEEP_BUDGET_S,
            *extra,
        ]
    )
    return result, log_dir


# ---------------------------------------------------------------------------
# Read-only reuse of the canonical log parser
# ---------------------------------------------------------------------------


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "contention_queue_sweep_report_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_skill_log_parser_field_extractors_are_reused():
    """The queue-line fields come from the proxy-usage-analysis parser.

    The numeric tests cannot tell an imported parser from a hand-copied regex,
    so the wiring is pinned here: the skill module is loaded read-only and its
    extractors back the report. A silent downgrade to the builtin patterns would
    let the two definitions drift apart.
    """
    module = _load_script_module()
    if not module._SKILL_LOG_PARSER_PATH.is_file():
        pytest.skip("this checkout has no .pi/ skill tree")

    assert module._SKILL_LOG_PARSER is not None
    assert module.PARSER_SOURCE == "proxy-usage-analysis/log_parser"
    assert module._QUEUED_RE is module._SKILL_LOG_PARSER.RE_CONTENTION_DURATION
    assert module._REASON_RE is module._SKILL_LOG_PARSER.RE_REASON_KV
    assert module._SESSION_RE is module._SKILL_LOG_PARSER.RE_SESSION


# ---------------------------------------------------------------------------
# AC1/AC2 — the three sections, machine-readable
# ---------------------------------------------------------------------------


def test_json_report_emits_all_three_sections(tmp_path):
    result, _ = _report(tmp_path, ("--json",))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["meta"]["sweep_budget_seconds"] == 30.0
    assert report["meta"]["window"]["hours"] == 30.0
    assert report["status"]["complete"] is True
    assert report["status"]["reasons"] == []

    # (a) ticket-backed dispatches
    assert report["sections"]["ticket_backed"]["count"] == 2

    # (b) residual queue wins
    wins = report["sections"]["residual_wins"]
    assert wins["count"] == 4
    assert wins["at_or_below_budget"] == 3
    assert wins["over_budget"] == 1
    assert wins["share_at_or_below_budget_pct"] == 75.0
    assert wins["wait_seconds"]["total"] == 91.5
    assert wins["wait_seconds"]["p50"] == 21.0
    assert wins["wait_seconds"]["p75"] == 33.75
    assert wins["wait_seconds"]["p90"] == 40.5
    assert wins["wait_seconds"]["max"] == 45.0

    # (c) residual queue losses
    losses = report["sections"]["residual_losses"]
    assert losses["count"] == 4
    assert abs(losses["aggregate_wait_seconds"] - 235.5) < 1e-9
    # 235.5 s of dead wait across a 30 h window
    assert abs(losses["wait_per_hour_seconds"] - 235.5 / 30) < 1e-6

    # reason split: lease-active vs none vs other, nothing dropped
    by_reason = losses["by_reason"]
    assert set(by_reason) == {"local_lease_active", "none", "prefill_deadline"}
    assert by_reason["local_lease_active"]["count"] == 2
    assert abs(by_reason["local_lease_active"]["aggregate_wait_seconds"] - 120.0) < 1e-9
    assert by_reason["none"]["count"] == 1
    assert abs(by_reason["none"]["aggregate_wait_seconds"] - 60.0) < 1e-9
    assert by_reason["prefill_deadline"]["count"] == 1
    assert losses["reason_groups"]["local_lease_active"] == 2
    assert losses["reason_groups"]["none"] == 1
    assert losses["reason_groups"]["other"] == 1
    # every loss is accounted for in the split
    assert sum(entry["count"] for entry in by_reason.values()) == losses["count"]

    # queue-level denominator the flip criteria are phrased against
    queue = report["sections"]["queue_events"]
    assert queue["count"] == 8
    assert queue["win_share_pct"] == 50.0


def test_window_filtering_excludes_events_outside_the_stated_window(tmp_path):
    result, _ = _report(tmp_path, ("--json",))
    assert result.returncode == 0, result.stderr
    sections = json.loads(result.stdout)["sections"]

    # the pre-window win and post-window loss are excluded, and the request-body
    # echo of a loss marker is never counted
    assert sections["residual_wins"]["count"] == 4
    assert sections["residual_losses"]["count"] == 4
    assert sections["residual_wins"]["at_or_below_budget"] + sections["residual_wins"][
        "over_budget"
    ] == sections["residual_wins"]["count"]


def test_sweep_budget_changes_the_at_or_below_share(tmp_path):
    result, _ = _report(tmp_path, ("--json", "--sweep-budget-seconds", "10"))
    assert result.returncode == 0, result.stderr
    wins = json.loads(result.stdout)["sections"]["residual_wins"]
    assert json.loads(result.stdout)["meta"]["sweep_budget_seconds"] == 10.0
    assert wins["at_or_below_budget"] == 1  # only the 4.5 s win
    assert wins["share_at_or_below_budget_pct"] == 25.0


def test_gzipped_rotated_log_is_discovered(tmp_path):
    log_dir = tmp_path / "logs-gz"
    log_dir.mkdir()
    _write_logs(log_dir, _baseline_lines(), compress=True)
    result = _run(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            SINCE,
            "--until",
            UNTIL,
            "--json",
        ]
    )
    assert result.returncode == 0, result.stderr
    sections = json.loads(result.stdout)["sections"]
    assert sections["residual_wins"]["count"] == 4
    assert sections["residual_losses"]["count"] == 4
    assert json.loads(result.stdout)["meta"]["coverage"]["files"] == 1


# ---------------------------------------------------------------------------
# AC2 — human and markdown renders carry the same three sections
# ---------------------------------------------------------------------------


def test_text_render_contains_all_three_sections(tmp_path):
    result, _ = _report(tmp_path)
    assert result.returncode == 0, result.stderr
    out = result.stdout

    assert "Ticket-backed dispatches" in out
    assert "Residual queue wins" in out
    assert "Residual queue losses" in out
    # win count / share / distribution
    assert "75.0%" in out
    assert "p50" in out and "p75" in out and "p90" in out
    # loss aggregate wait and the reason split
    assert "235.5" in out
    assert "local_lease_active" in out
    assert "none" in out
    assert "COMPLETE" in out


def test_markdown_render_contains_the_reason_split(tmp_path):
    result, _ = _report(tmp_path, ("--markdown",))
    assert result.returncode == 0, result.stderr
    out = result.stdout

    assert "### (a) Ticket-backed dispatches" in out
    assert "### (b) Residual queue wins" in out
    assert "### (c) Residual queue losses" in out
    assert "`local_lease_active`" in out
    assert "| `none` |" in out


# ---------------------------------------------------------------------------
# Exit codes — 2 = incomplete, 1 = invalid input
# ---------------------------------------------------------------------------


def test_thin_window_is_incomplete(tmp_path):
    log_dir = tmp_path / "logs-thin"
    log_dir.mkdir()
    _write_logs(log_dir, _baseline_lines())
    result = _run(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            SINCE,
            "--until",
            "2026-10-29 02:00",  # 2 h — shorter than the 24 h baseline
            "--json",
        ]
    )
    assert result.returncode == 2, result.stderr
    status = json.loads(result.stdout)["status"]
    assert status["complete"] is False
    assert any("24" in reason or "thin" in reason.lower() for reason in status["reasons"])


def test_empty_window_is_incomplete(tmp_path):
    _, log_dir = _report(tmp_path)
    result = _run(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            "2026-09-01 00:00",
            "--until",
            "2026-09-02 00:00",
            "--json",
        ]
    )
    assert result.returncode == 2, result.stderr
    status = json.loads(result.stdout)["status"]
    assert status["complete"] is False
    assert any("data" in reason.lower() for reason in status["reasons"])


def test_coverage_gap_is_incomplete(tmp_path):
    log_dir = tmp_path / "logs-gap"
    log_dir.mkdir()
    # two adjacent files with a 4 h hole between them: a missing interval must
    # fail the verdict instead of reading as a partial success
    _write_logs(
        log_dir,
        [_win("2026-10-29 00:30:00,000", "gap-a", 4.0)],
        name="proxy.log-2026-10-29_00",
    )
    _write_logs(
        log_dir,
        [_win("2026-10-29 05:30:00,000", "gap-b", 6.0)],
        name="proxy.log-2026-10-29_05",
    )
    result = _run(
        [
            "--log-dir",
            str(log_dir),
            "--since",
            SINCE,
            "--until",
            UNTIL,
            "--json",
        ]
    )
    assert result.returncode == 2, result.stderr
    status = json.loads(result.stdout)["status"]
    assert status["complete"] is False
    assert any("gap" in reason.lower() for reason in status["reasons"])


def test_no_logs_in_directory_is_incomplete(tmp_path):
    log_dir = tmp_path / "empty"
    log_dir.mkdir()
    result = _run(["--log-dir", str(log_dir), "--json"])
    assert result.returncode == 2, result.stderr
    status = json.loads(result.stdout)["status"]
    assert status["complete"] is False
    assert any("no proxy logs" in reason.lower() for reason in status["reasons"])


def test_missing_log_directory_is_invalid_input(tmp_path):
    result = _run(["--log-dir", str(tmp_path / "does-not-exist"), "--json"])
    assert result.returncode == 1
    assert "log directory not found" in result.stderr


def test_invalid_timestamp_is_invalid_input(tmp_path):
    _, log_dir = _report(tmp_path)
    result = _run(["--log-dir", str(log_dir), "--since", "not-a-timestamp"])
    assert result.returncode == 1
    assert "invalid timestamp" in result.stderr


def test_zero_sweep_budget_is_invalid_input(tmp_path):
    _, log_dir = _report(tmp_path)
    result = _run(["--log-dir", str(log_dir), "--sweep-budget-seconds", "0"])
    assert result.returncode == 1
    assert "sweep-budget-seconds" in result.stderr


def test_non_numeric_hours_is_invalid_input(tmp_path):
    _, log_dir = _report(tmp_path)
    result = _run(["--log-dir", str(log_dir), "--hours", "many"])
    # argparse rejects the value; the CLI must still report invalid input (1),
    # never the "incomplete measurement" code (2)
    assert result.returncode == 1


# ---------------------------------------------------------------------------
# --hours default window resolution
# ---------------------------------------------------------------------------


def test_hours_window_is_relative_to_until(tmp_path):
    _, log_dir = _report(tmp_path)
    result = _run(
        [
            "--log-dir",
            str(log_dir),
            "--until",
            UNTIL,
            "--hours",
            "28",
            "--json",
        ]
    )
    assert result.returncode == 0, result.stderr
    window = json.loads(result.stdout)["meta"]["window"]
    assert window["hours"] == 28.0
    assert window["since"] == "2026-10-29 02:00:00"
    # the 01:00 win now falls before the window
    assert json.loads(result.stdout)["sections"]["residual_wins"]["count"] == 3
