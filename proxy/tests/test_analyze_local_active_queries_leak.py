#!/usr/bin/env python3
"""Tests for ``scripts/analyze_local_active_queries_leak.py`` (LP-0MVYWKG0T0081EGC).

The tool is the deterministic replacement for the hand-measured AC1 baseline of
parent LP-0MVWF78QK0060TZH, and the number the post-fix AC6 window compares
against. These tests drive it with synthetic logs in a temp directory (never the
live ``/var/log/llama-proxy``) and assert the observable measurement contract:

* the recovery-warning count, the ``reset from N to M`` delta distribution, the
  local ``Stream finished`` denominator and the resulting leak ratio;
* window filtering, including rotated files that hold data past their name;
* the explicit per-file row (first/last timestamp, per-file counts) that makes a
  double-counted rotated file visible;
* cross-file duplicate suppression (a ``copytruncate`` tail copied into the next
  file is counted once);
* both rotation naming schemes (dot and dash) and gzip compression;
* the coverage verdict: an adjacent-file hole, a head hole, and a tail hole that
  is only hard for a closed window — a partial read must never look like a
  measured leak rate;
* the CLI contract: exit 0 for a covered window (even with zero warnings),
  1 for a missing log directory, 2 for invalid arguments, 3 for a coverage gap.
"""

from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_local_active_queries_leak.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_local_active_queries_leak", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def leak():
    """The analyser module under test (imported once per module)."""
    return _load_module()


# ---------------------------------------------------------------------------
# Synthetic log builders (line shapes copied from the deployed proxy log)
# ---------------------------------------------------------------------------


def recovery(ts: str, frm: int, to: int, *, extra: str = "") -> str:
    """One ``local_active_queries counter recovered`` WARNING."""
    return (
        f"{ts} - WARNING - local_active_queries counter recovered: "
        f"reset from {frm} to {to} ({to} active dispatch records){extra}\n"
    )


def stream(ts: str, *, session: str = "s1", provider: str = "local") -> str:
    """One ``Stream finished`` INFO line."""
    return (
        f"{ts} - INFO - Stream finished: reason=stop tokens=100/20/120 "
        f"session={session} provider={provider} model=Qwen3 "
        f"entry=local-qwen3 request=[{{'type': 'text', 'text': 'x'... turn=abc\n"
    )


def write_log(directory: Path, name: str, lines: list[str]) -> Path:
    """Write *lines* to *directory/name*, gzip-compressing a ``.gz`` name."""
    path = directory / name
    payload = "".join(lines)
    if name.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def run(leak, directory: Path, window: str, **kwargs) -> dict:
    start, end = leak.parse_window(window)
    return leak.analyse(directory, start, end, **kwargs)


# ---------------------------------------------------------------------------
# AC1 (a)(b)(c): counts, deltas and leak ratio
# ---------------------------------------------------------------------------


def test_counts_warnings_delta_distribution_and_local_streams(leak, tmp_path):
    write_log(
        tmp_path,
        "proxy.log.2026-10-31_00",
        [
            recovery("2026-10-31 00:42:39,487", 3, 2),
            stream("2026-10-31 00:42:05,127", session="a"),
            stream("2026-10-31 00:42:06,943", session="b"),
            # A remote stream must not enter the local denominator.
            stream("2026-10-31 00:43:00,000", session="c", provider="OpenCode"),
        ],
    )
    write_log(
        tmp_path,
        "proxy.log.2026-10-31_06.gz",
        [
            recovery("2026-10-31 06:46:20,287", 1, 0),
            recovery("2026-10-31 06:46:30,287", 2, 1),
        ],
    )

    # The two fixture files are 6 h apart; a generous gap bound keeps this
    # test about the counts, with coverage covered by its own tests below.
    report = run(
        leak,
        tmp_path,
        "2026-10-31T00:00:00,2026-10-31T07:00:00",
        window_closed=True,
        max_gap_seconds=8 * 3600,
    )

    assert report["recovery_warnings"]["total"] == 3
    assert report["delta_distribution"] == {
        "1\u21920": 1,
        "2\u21921": 1,
        "3\u21922": 1,
    }
    assert report["local_streams"]["total"] == 2
    assert report["leak_ratio"]["warnings_per_local_stream"] == 1.5
    assert report["leak_ratio"]["local_streams_per_warning"] == 0.67
    assert report["coverage"]["covered"] is True


def test_zero_warnings_reports_no_ratio_but_is_a_result(leak, tmp_path):
    write_log(
        tmp_path,
        "proxy.log.2026-11-02_03",
        [stream("2026-11-02 03:00:00,000")],
    )

    report = run(
        leak,
        tmp_path,
        "2026-11-02T02:30:00,2026-11-02T03:30:00",
        window_closed=True,
    )

    assert report["recovery_warnings"]["total"] == 0
    assert report["delta_distribution"] == {}
    assert report["leak_ratio"] == {
        "warnings_per_local_stream": 0.0,
        "local_streams_per_warning": None,
    }
    assert report["coverage"]["covered"] is True


def test_local_provider_variants_count_and_remote_does_not(leak, tmp_path):
    write_log(
        tmp_path,
        "proxy.log.2026-11-02_03",
        [
            stream("2026-11-02 03:00:00,000", session="a", provider="local"),
            stream("2026-11-02 03:00:01,000", session="b", provider="local-qwen3"),
            stream("2026-11-02 03:00:02,000", session="c", provider="OpenCode"),
        ],
    )

    report = run(
        leak,
        tmp_path,
        "2026-11-02T00:00:00,2026-11-02T06:00:00",
        window_closed=True,
    )

    assert report["local_streams"]["total"] == 2


def test_recovery_warning_with_identifying_fields_is_still_matched(leak, tmp_path):
    """AC5 appends identifying fields after the parenthesis; the match must hold."""
    write_log(
        tmp_path,
        "proxy.log.2026-11-02_03",
        [
            recovery(
                "2026-11-02 03:00:00,000",
                3,
                2,
                extra=" session=abc turn=def record=17 reason=client_disconnect",
            )
        ],
    )

    report = run(
        leak,
        tmp_path,
        "2026-11-02T00:00:00,2026-11-02T06:00:00",
        window_closed=True,
    )

    assert report["recovery_warnings"]["total"] == 1
    assert report["delta_distribution"] == {"3\u21922": 1}


# ---------------------------------------------------------------------------
# AC1 (d): explicit timestamped file set
# ---------------------------------------------------------------------------


def test_per_file_rows_carry_spans_and_counts(leak, tmp_path):
    write_log(
        tmp_path,
        "proxy.log.2026-10-31_00",
        [
            recovery("2026-10-31 00:42:39,487", 3, 2),
            stream("2026-10-31 01:00:00,000"),
        ],
    )
    write_log(
        tmp_path,
        "proxy.log-2026-10-31_06.gz",
        [recovery("2026-10-31 06:46:20,287", 1, 0)],
    )

    report = run(
        leak,
        tmp_path,
        "2026-10-31T00:00:00,2026-10-31T13:14:00",
        window_closed=True,
    )

    rows = {row["file"]: row for row in report["files"]}
    assert set(rows) == {"proxy.log.2026-10-31_00", "proxy.log-2026-10-31_06.gz"}
    assert rows["proxy.log.2026-10-31_00"]["first"] == "2026-10-31 00:42:39"
    assert rows["proxy.log.2026-10-31_00"]["last"] == "2026-10-31 01:00:00"
    assert rows["proxy.log.2026-10-31_00"]["recovery_warnings"] == 1
    assert rows["proxy.log.2026-10-31_00"]["local_streams"] == 1
    assert rows["proxy.log-2026-10-31_06.gz"]["recovery_warnings"] == 1
    assert rows["proxy.log-2026-10-31_06.gz"]["local_streams"] == 0
    # The per-file counts must add up to the headline figures.
    assert sum(row["recovery_warnings"] for row in rows.values()) == 2
    assert sum(row["local_streams"] for row in rows.values()) == 1
    assert report["files_scanned"] == 2


def test_window_filter_excludes_lines_outside_the_window(leak, tmp_path):
    write_log(
        tmp_path,
        "proxy.log.2026-10-31_00",
        [
            recovery("2026-10-30 23:59:59,999", 2, 1),  # before
            recovery("2026-10-31 01:00:00,000", 2, 1),  # inside
            recovery("2026-10-31 14:00:00,000", 2, 1),  # after
            stream("2026-10-31 01:00:00,000", session="in"),
            stream("2026-10-31 14:00:00,000", session="out"),
        ],
    )

    report = run(
        leak,
        tmp_path,
        "2026-10-31T00:00:00,2026-10-31T13:14:00",
        window_closed=True,
    )

    assert report["recovery_warnings"]["total"] == 1
    assert report["local_streams"]["total"] == 1
    assert report["files"][0]["lines"] == 2


def test_cross_file_duplicate_is_counted_once(leak, tmp_path):
    """A logrotate ``copytruncate`` tail is copied, not re-logged."""
    duplicated_warning = recovery("2026-10-31 00:42:39,487", 3, 2)
    duplicated_stream = stream("2026-10-31 00:42:05,127")
    write_log(
        tmp_path,
        "proxy.log.2026-10-31_00",
        [duplicated_warning, duplicated_stream],
    )
    write_log(
        tmp_path,
        "proxy.log.2026-10-31_06",
        [duplicated_warning, duplicated_stream, stream("2026-10-31 06:50:00,000")],
    )

    report = run(
        leak,
        tmp_path,
        "2026-10-31T00:00:00,2026-10-31T13:14:00",
        window_closed=True,
    )

    assert report["recovery_warnings"]["total"] == 1
    assert report["local_streams"]["total"] == 2
    assert report["duplicates_dropped"] == 2
    rows = {row["file"]: row for row in report["files"]}
    assert rows["proxy.log.2026-10-31_00"]["duplicates_dropped"] == 0
    assert rows["proxy.log.2026-10-31_06"]["duplicates_dropped"] == 2


def test_identical_lines_within_one_file_are_not_deduplicated(leak, tmp_path):
    """Suppression is cross-file only: a repeated event in one file is real."""
    repeated = recovery("2026-10-31 00:42:39,487", 3, 2)
    write_log(tmp_path, "proxy.log.2026-10-31_00", [repeated, repeated])

    report = run(
        leak,
        tmp_path,
        "2026-10-31T00:00:00,2026-10-31T13:14:00",
        window_closed=True,
    )

    assert report["recovery_warnings"]["total"] == 2
    assert report["duplicates_dropped"] == 0


def test_file_starting_after_the_window_is_skipped_not_read(leak, tmp_path):
    write_log(
        tmp_path,
        "proxy.log.2026-11-02_12",
        [recovery("2026-11-02 12:00:00,000", 2, 1)],
    )

    report = run(
        leak,
        tmp_path,
        "2026-11-02T00:00:00,2026-11-02T06:49:00",
        window_closed=True,
    )

    assert report["files"] == []
    assert report["files_skipped_after_window"] == ["proxy.log.2026-11-02_12"]


def test_explicit_files_restrict_the_scan(leak, tmp_path):
    write_log(tmp_path, "proxy.log.2026-10-31_00", [recovery("2026-10-31 01:00:00,000", 2, 1)])
    write_log(tmp_path, "proxy.log.2026-10-31_06", [recovery("2026-10-31 07:00:00,000", 2, 1)])

    report = run(
        leak,
        tmp_path,
        "2026-10-31T00:00:00,2026-10-31T13:14:00",
        files=["proxy.log.2026-10-31_00"],
        window_closed=True,
    )

    assert [row["file"] for row in report["files"]] == ["proxy.log.2026-10-31_00"]
    assert report["recovery_warnings"]["total"] == 1
    assert report["files_scanned"] == 1


# ---------------------------------------------------------------------------
# Coverage verdict
# ---------------------------------------------------------------------------


def test_between_file_gap_fails_the_coverage_verdict(leak, tmp_path):
    write_log(tmp_path, "proxy.log.2026-10-31_00", [stream("2026-10-31 01:00:00,000")])
    write_log(tmp_path, "proxy.log.2026-10-31_06", [stream("2026-10-31 06:00:00,000")])

    report = run(
        leak,
        tmp_path,
        "2026-10-31T00:00:00,2026-10-31T13:14:00",
        window_closed=True,
    )

    assert report["coverage"]["covered"] is False
    gaps = report["coverage"]["gaps"]
    assert gaps[0]["kind"] == "between_files"
    assert gaps[0]["before"] == "proxy.log.2026-10-31_00"
    assert gaps[0]["after"] == "proxy.log.2026-10-31_06"
    assert gaps[0]["gap_seconds"] == pytest.approx(5 * 3600)


def test_head_gap_fails_even_when_the_tail_is_open(leak, tmp_path):
    write_log(tmp_path, "proxy.log.2026-10-31_00", [stream("2026-10-31 03:00:00,000")])

    report = run(leak, tmp_path, "2026-10-31T00:00:00,2026-10-31T13:14:00")

    assert report["coverage"]["covered"] is False
    assert report["coverage"]["gaps"][0]["kind"] == "window_head"
    assert report["coverage"]["gaps"][0]["gap_seconds"] == pytest.approx(3 * 3600)


def test_tail_gap_is_informational_until_the_window_is_closed(leak, tmp_path):
    write_log(tmp_path, "proxy.log.2026-10-31_00", [stream("2026-10-31 01:00:00,000")])
    window = "2026-10-31T00:00:00,2026-10-31T13:14:00"

    open_window = run(leak, tmp_path, window)
    closed_window = run(leak, tmp_path, window, window_closed=True)

    assert open_window["coverage"]["covered"] is True
    assert open_window["coverage"]["window_end_gap_seconds"] == pytest.approx(
        12 * 3600 + 14 * 60
    )
    assert closed_window["coverage"]["covered"] is False
    assert closed_window["coverage"]["gaps"][-1]["kind"] == "window_tail"


def test_window_outside_retention_is_a_gap_not_an_empty_result(leak, tmp_path):
    write_log(tmp_path, "proxy.log.2026-10-31_00", [stream("2026-10-31 01:00:00,000")])

    report = run(
        leak,
        tmp_path,
        "2026-11-01T00:00:00,2026-11-01T06:00:00",
        window_closed=True,
    )

    assert report["coverage"]["covered"] is False
    assert report["coverage"]["gaps"][0]["kind"] == "outside_retention"
    assert report["files"] == []


def test_empty_but_covered_window_reports_zero_not_a_gap(leak, tmp_path):
    """Retention spans the window; the proxy simply wrote no matched lines."""
    write_log(
        tmp_path,
        "proxy.log.2026-10-31_00",
        [
            stream("2026-10-31 00:30:00,000", provider="OpenCode"),
            stream("2026-10-31 05:00:00,000", provider="OpenCode"),
        ],
    )

    report = run(
        leak,
        tmp_path,
        "2026-10-31T01:00:00,2026-10-31T04:00:00",
        window_closed=True,
    )

    assert report["coverage"]["covered"] is True
    assert report["recovery_warnings"]["total"] == 0
    assert report["local_streams"]["total"] == 0


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_cli_json_output_and_exit_zero_for_covered_window(leak, tmp_path, capsys):
    write_log(
        tmp_path,
        "proxy.log.2026-10-31_00",
        [recovery("2026-10-31 01:00:00,000", 2, 1), stream("2026-10-31 01:00:00,000")],
    )

    code = leak.main(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "2026-10-31T00:30:00,2026-10-31T01:30:00",
            "--window-closed",
            "--json",
        ]
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["recovery_warnings"]["total"] == 1
    assert report["files"][0]["file"] == "proxy.log.2026-10-31_00"
    assert report["coverage"]["covered"] is True


def test_cli_exit_three_on_coverage_gap(leak, tmp_path, capsys):
    write_log(tmp_path, "proxy.log.2026-10-31_00", [stream("2026-10-31 01:00:00,000")])
    write_log(tmp_path, "proxy.log.2026-10-31_06", [stream("2026-10-31 06:00:00,000")])

    code = leak.main(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "2026-10-31T00:00:00,2026-10-31T13:14:00",
            "--window-closed",
        ]
    )

    captured = capsys.readouterr()
    assert code == 3
    assert "COVERAGE GAP" in captured.out
    assert "coverage gap" in captured.err


def test_cli_exit_one_for_missing_log_directory(leak, tmp_path, capsys):
    code = leak.main(
        [
            "--log-dir",
            str(tmp_path / "nope"),
            "--window",
            "2026-10-31T00:00:00,2026-10-31T13:14:00",
        ]
    )

    assert code == 1
    assert "log directory not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "window",
    [
        "2026-10-31T00:00:00",  # no separator
        "2026-10-31T13:14:00,2026-10-31T00:00:00",  # start after end
        "not-a-date,2026-10-31T00:00:00",  # unparseable
        "2026-10-31T00:00:00Z,2026-10-31T13:14:00Z",  # UTC offset is ambiguous
    ],
)
def test_cli_exit_two_for_invalid_window(leak, tmp_path, capsys, window):
    write_log(tmp_path, "proxy.log.2026-10-31_00", [stream("2026-10-31 01:00:00,000")])

    code = leak.main(["--log-dir", str(tmp_path), "--window", window])

    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_exit_two_for_unknown_explicit_file(leak, tmp_path, capsys):
    code = leak.main(
        [
            "--log-dir",
            str(tmp_path),
            "--window",
            "2026-10-31T00:00:00,2026-10-31T13:14:00",
            "--files",
            "missing.log",
        ]
    )

    assert code == 2
    assert "explicit log file not found" in capsys.readouterr().err


def test_render_states_the_gap_and_the_file_set(leak, tmp_path):
    write_log(tmp_path, "proxy.log.2026-10-31_00", [stream("2026-10-31 01:00:00,000")])
    write_log(tmp_path, "proxy.log.2026-10-31_06", [stream("2026-10-31 06:00:00,000")])

    report = run(
        leak,
        tmp_path,
        "2026-10-31T00:00:00,2026-10-31T13:14:00",
        window_closed=True,
    )
    text = leak.render(report)

    assert "COVERAGE GAP" in text
    assert "GAP between_files" in text
    assert "proxy.log.2026-10-31_00" in text
    assert "recovery warnings: 0" in text


def test_parse_window_accepts_space_separated_iso8601(leak):
    start, end = leak.parse_window(
        "2026-10-31 00:00:00,2026-10-31 13:14:00"
    )

    assert start == dt.datetime(2026, 10, 31, 0, 0, 0)
    assert end == dt.datetime(2026, 10, 31, 13, 14, 0)
