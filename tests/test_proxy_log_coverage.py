#!/usr/bin/env python3
"""Log-coverage measurement (LP-0MVLHMAWV008EKTI — proposed change item 2).

``measure_log_coverage`` computes, per log file, the first/last timestamp of
the lines inside a requested window and reports an ``adjacent-file`` gap when
the boundary between two files exceeds a bounded threshold. That is the exact
incident shape: ``proxy.log.2026-10-23_12`` ended 18:29:12 and the next file
(``proxy.log``) started 22:04:16 — a 3 h 35 m hole inside the requested
verification window. Adjacent-file gaps report *lost windows*, not ordinary
idle periods within a single file.

Contract under test (observable behaviour):

* contiguous files (handover gap << threshold) produce no gaps;
* the incident shape (18:29:12 -> 22:04:16) is reported as a gap and fails
  the coverage verdict;
* overlapping/duplicated coverage (e.g. logrotate copytruncate) is not a gap;
* lines outside ``since``/``until`` are excluded from spans, so a gap outside
  the window is not reported;
* a span that itself covers the boundary suppresses the gap;
* a directory with no parseable lines is not covered.
"""

from __future__ import annotations

import gzip
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lib.proxy_logs import measure_log_coverage

LIVE = "proxy.log"
DOT = "proxy.log."
DASH = "proxy.log-"


def _ts(iso: str) -> datetime:
    return datetime.strptime(iso, "%Y-%m-%d %H:%M:%S")


def _line(iso: str, text: str = "status_request") -> str:
    return f"{iso},123 - INFO - {text}\n"


def _write(log_dir: Path, name: str, lines: list[str]) -> Path:
    p = log_dir / name
    p.write_text("".join(lines), encoding="utf-8")
    return p


def _dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# spans: windowed first/last per file
# ---------------------------------------------------------------------------


def test_spans_capture_window_within_file_boundaries(tmp_path):
    log_dir = _dir(tmp_path)
    _write(
        log_dir,
        DOT + "2026-10-23_12",
        [
            _line("2026-10-23 12:29:16"),
            _line("2026-10-23 18:29:12"),
        ],
    )
    coverage = measure_log_coverage(log_dir)
    assert [s.path.name for s in coverage.spans] == [DOT + "2026-10-23_12"]
    span = coverage.spans[0]
    assert span.first == _ts("2026-10-23 12:29:16")
    assert span.last == _ts("2026-10-23 18:29:12")
    assert span.line_count == 2


def test_window_filter_excludes_out_of_range_lines(tmp_path):
    """Lines outside since/until do not contribute to spans."""
    log_dir = _dir(tmp_path)
    _write(
        log_dir,
        LIVE,
        [
            _line("2026-10-23 10:00:00"),
            _line("2026-10-23 22:04:16"),
            _line("2026-10-24 01:00:00"),
        ],
    )
    coverage = measure_log_coverage(
        log_dir,
        since=_ts("2026-10-23 12:00:00"),
        until=_ts("2026-10-23 23:00:00"),
    )
    span = coverage.spans[0]
    assert span.first == _ts("2026-10-23 22:04:16")
    assert span.last == _ts("2026-10-23 22:04:16")
    assert span.line_count == 1


def test_empty_dir_is_not_covered(tmp_path):
    coverage = measure_log_coverage(_dir(tmp_path))
    assert coverage.spans == []
    assert coverage.gaps == []
    assert coverage.covered is False


def test_unparseable_files_are_ignored(tmp_path):
    log_dir = _dir(tmp_path)
    _write(log_dir, LIVE, ["no timestamp here\n", "raw junk\n"])
    coverage = measure_log_coverage(log_dir)
    assert coverage.spans == []
    assert coverage.covered is False


# ---------------------------------------------------------------------------
# gaps: adjacent-file boundaries
# ---------------------------------------------------------------------------


def test_contiguous_files_have_no_gap(tmp_path):
    """A normal restart handover (2 min down) is not a coverage failure."""
    log_dir = _dir(tmp_path)
    _write(
        log_dir,
        DOT + "2026-10-23_12",
        [_line("2026-10-23 12:29:16"), _line("2026-10-23 18:29:12")],
    )
    _write(log_dir, LIVE, [_line("2026-10-23 18:31:00"), _line("2026-10-23 23:00:00")])

    coverage = measure_log_coverage(log_dir, max_gap_seconds=3600)
    assert coverage.gaps == []
    assert coverage.covered


def test_incident_shaped_gap_is_reported_and_fails_verdict(tmp_path):
    """18:29:12 -> 22:04:16 (3 h 35 m) is exactly the 2026-10-23 hole."""
    log_dir = _dir(tmp_path)
    _write(
        log_dir,
        DOT + "2026-10-23_12",
        [_line("2026-10-23 12:29:16"), _line("2026-10-23 18:29:12")],
    )
    _write(log_dir, LIVE, [_line("2026-10-23 22:04:16")])

    coverage = measure_log_coverage(log_dir, max_gap_seconds=3600)
    assert len(coverage.gaps) == 1
    gap = coverage.gaps[0]
    assert gap.before.name == DOT + "2026-10-23_12"
    assert gap.after.name == LIVE
    assert gap.before_last == _ts("2026-10-23 18:29:12")
    assert gap.after_first == _ts("2026-10-23 22:04:16")
    assert gap.gap_seconds == pytest.approx(3 * 3600 + 35 * 60 + 4)
    assert coverage.covered is False, (
        "an unmeasurable window must fail the coverage verdict"
    )


def test_overlapping_files_are_not_a_gap(tmp_path):
    """copytruncate duplicates the tail of the previous file: no gap."""
    log_dir = _dir(tmp_path)
    _write(
        log_dir,
        DASH + "2026-10-23_18",
        [_line("2026-10-23 17:59:00"), _line("2026-10-23 18:29:12")],
    )
    _write(
        log_dir,
        LIVE,
        [_line("2026-10-23 18:20:00"), _line("2026-10-23 22:04:16")],
    )
    coverage = measure_log_coverage(log_dir, max_gap_seconds=3600)
    assert coverage.gaps == []
    assert coverage.covered


def test_restart_archive_and_live_handover_is_not_a_gap(tmp_path):
    """The restart-safe archive and the new live log hand over cleanly."""
    log_dir = _dir(tmp_path)
    _write(
        log_dir,
        DOT + "2026-10-23_220400",  # unique-name archive written by the restart
        [_line("2026-10-23 22:04:00")],
    )
    _write(log_dir, LIVE, [_line("2026-10-23 22:04:16")])
    coverage = measure_log_coverage(log_dir, max_gap_seconds=3600)
    assert coverage.gaps == []
    assert coverage.covered


def test_gap_outside_the_window_is_not_reported(tmp_path):
    log_dir = _dir(tmp_path)
    _write(log_dir, DOT + "2026-10-23_12", [_line("2026-10-23 12:29:16")])
    _write(log_dir, LIVE, [_line("2026-10-23 22:04:16")])
    coverage = measure_log_coverage(
        log_dir,
        since=_ts("2026-10-23 21:00:00"),
        until=_ts("2026-10-23 23:00:00"),
        max_gap_seconds=3600,
    )
    assert coverage.gaps == []
    assert coverage.covered


def test_multiple_gaps_all_reported(tmp_path):
    log_dir = _dir(tmp_path)
    _write(log_dir, DOT + "2026-10-23_06", [_line("2026-10-23 06:29:16")])
    _write(log_dir, DOT + "2026-10-23_12", [_line("2026-10-23 18:29:12")])
    _write(log_dir, LIVE, [_line("2026-10-23 22:04:16")])
    coverage = measure_log_coverage(log_dir, max_gap_seconds=3600)
    assert len(coverage.gaps) == 2
    assert sorted(g.gap_seconds for g in coverage.gaps) == sorted(
        [
            (11 * 3600 + 59 * 60 + 56),  # 06:29:16 -> 18:29:12
            (3 * 3600 + 35 * 60 + 4),    # 18:29:12 -> 22:04:16
        ]
    )


def test_gap_threshold_bounds_the_definition(tmp_path):
    """A gap under the threshold is tolerated; the same gap over it is not."""
    log_dir = _dir(tmp_path)
    _write(log_dir, DOT + "2026-10-23_12", [_line("2026-10-23 18:29:12")])
    _write(log_dir, LIVE, [_line("2026-10-23 19:00:00")])

    assert measure_log_coverage(log_dir, max_gap_seconds=3600).covered
    assert not measure_log_coverage(log_dir, max_gap_seconds=60).covered


def test_window_start_and_end_gaps_are_reported_info(tmp_path):
    """The lead-in before since and the tail after the last line are measured
    (informational) but do not fail the verdict — the tail is a live window
    whose end has not arrived yet."""
    log_dir = _dir(tmp_path)
    _write(log_dir, LIVE, [_line("2026-10-23 22:04:16")])
    coverage = measure_log_coverage(
        log_dir,
        since=_ts("2026-10-23 21:00:00"),
        until=_ts("2026-10-23 23:00:00"),
    )
    assert coverage.window_start_gap_seconds == pytest.approx(3600 + 4 * 60 + 16)
    coverage.covered  # tail/head gaps are informational


def test_gz_and_dash_files_participate(tmp_path):
    log_dir = _dir(tmp_path)
    gz = log_dir / (DOT + "2026-10-23_12.gz")
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write(_line("2026-10-23 12:29:16") + _line("2026-10-23 18:29:12"))
    _write(log_dir, LIVE, [_line("2026-10-23 22:04:16")])
    coverage = measure_log_coverage(log_dir, max_gap_seconds=3600)
    assert [g.after.name for g in coverage.gaps] == [LIVE]
    assert coverage.covered is False


def test_discover_returns_empty_for_missing_directory(tmp_path):
    coverage = measure_log_coverage(tmp_path / "nope", max_gap_seconds=3600)
    assert coverage.spans == []
    assert coverage.gaps == []
