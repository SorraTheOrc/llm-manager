"""Unit tests for the llama-server eval-timing parser (decode/generation speed).

Covers: eval-timing line parsing (decode + prompt eval), Qwen3 child-port
discovery, log-file discovery, streaming iteration, day/night speed
aggregation, and the end-to-end report/CSV additions.

Fixtures are derived from real lines in /var/log/llama-proxy/llama-server.log
(see tests/fixtures.py).
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bucketing
import llama_log_parser
import reporting

from tests import fixtures

WINDOW_START = datetime(2026, 8, 2, 14, 0, 0)
WINDOW_END = datetime(2026, 8, 2, 15, 0, 0)

MOCK_MTIME = datetime(2026, 8, 2, 14, 30, 0)


def _schedule() -> bucketing.SlotSchedule:
    return bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])


def _set_mtime(path: Path, ts: datetime) -> None:
    os.utime(path, (ts.timestamp(), ts.timestamp()))


# ---------------------------------------------------------------------------
# Eval-timing line parsing
# ---------------------------------------------------------------------------


class TestEvalLineParsing:
    def test_decode_line(self):
        ev = llama_log_parser.parse_eval_line(fixtures.DECODE_EVAL_REAL)
        assert ev is not None
        assert ev.kind == "decode"
        assert ev.port == 32999
        assert ev.ms == pytest.approx(3776.71)
        assert ev.tokens == 153
        assert ev.tok_s == pytest.approx(40.51)

    def test_prompt_eval_line(self):
        ev = llama_log_parser.parse_eval_line(fixtures.PROMPT_EVAL_REAL)
        assert ev is not None
        assert ev.kind == "prompt_eval"
        assert ev.port == 32999
        assert ev.ms == pytest.approx(29504.01)
        assert ev.tokens == 11449
        assert ev.tok_s == pytest.approx(388.05)

    def test_decode_line_second_sample(self):
        ev = llama_log_parser.parse_eval_line(fixtures.DECODE_EVAL_REAL2)
        assert ev is not None
        assert ev.kind == "decode"
        assert ev.tok_s == pytest.approx(41.21)

    def test_prompt_eval_line_without_port(self):
        ev = llama_log_parser.parse_eval_line(fixtures.PROMPT_EVAL_NO_PORT)
        assert ev is not None
        assert ev.kind == "prompt_eval"
        assert ev.port is None
        assert ev.tok_s == pytest.approx(99.53)

    @pytest.mark.parametrize(
        "line",
        [
            fixtures.TOTAL_TIME_LINE,       # no tok/s in the line
            fixtures.SLOT_PRINT_TIMING_IGNORED,
            fixtures.SLOT_RELEASE_IGNORED,
            fixtures.SRV_LOAD_IGNORED,
            fixtures.MALFORMED_EVAL_LINE,
            "srv          load:   --port",
            "",
            None,
        ],
    )
    def test_ignored_lines(self, line):
        assert llama_log_parser.parse_eval_line(line) is None

    def test_other_port_line_parses_but_is_filterable(self):
        # Embeddings child (mxbai-embed) on a different port: the line parses,
        # but the Qwen3-port filter in iter_eval_timings drops it.
        line = (
            "[51973]        eval time =      10.00 ms /     1 tokens "
            "(   10.00 ms per token,   100.00 tokens per second)"
        )
        ev = llama_log_parser.parse_eval_line(line)
        assert ev is not None
        assert ev.port == 51973
        assert ev.kind == "decode"


# ---------------------------------------------------------------------------
# Qwen3 child-port discovery
# ---------------------------------------------------------------------------


class TestQwen3Port:
    def test_port_found(self, tmp_path):
        log = tmp_path / "llama-server.log"
        log.write_text(f"{fixtures.QWEN3_SPAWN_LINE}\n[32999] slot print_timing: id 1 | task 2 |\n")
        assert llama_log_parser.qwen3_port(log) == 32999

    def test_port_missing_returns_none(self, tmp_path):
        log = tmp_path / "llama-server.log"
        log.write_text("srv load: spawning server instance with name=mxbai-embed on port 51973\n")
        assert llama_log_parser.qwen3_port(log) is None

    def test_empty_file_returns_none(self, tmp_path):
        log = tmp_path / "llama-server.log"
        log.write_text("")
        assert llama_log_parser.qwen3_port(log) is None


# ---------------------------------------------------------------------------
# Log-file discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def _write_files(self, tmp_path: Path) -> Path:
        log_dir = tmp_path / "llama_logs"
        log_dir.mkdir()
        (log_dir / "llama-server.log").write_text("live\n")
        (log_dir / "llama-server.1.log").write_text("rotated\n")
        (log_dir / "llama-server.2.log").write_text("old\n")
        (log_dir / "proxy.log").write_text("not a llama log\n")
        (log_dir / "unrelated.txt").write_text("no\n")
        _set_mtime(log_dir / "llama-server.1.log", datetime(2026, 8, 2, 16, 0))
        _set_mtime(log_dir / "llama-server.2.log", datetime(2026, 8, 2, 12, 0))
        return log_dir

    def test_discovery_includes_live_and_in_window_rotated(self, tmp_path):
        log_dir = self._write_files(tmp_path)
        files = llama_log_parser.discover_llama_logs(log_dir, WINDOW_START)
        names = [p.name for p in files]
        assert "llama-server.log" in names
        assert "llama-server.1.log" in names
        assert "llama-server.2.log" not in names  # mtime < window start
        assert "proxy.log" not in names
        assert "unrelated.txt" not in names

    def test_discovery_missing_dir_returns_empty(self, tmp_path):
        assert llama_log_parser.discover_llama_logs(tmp_path / "nope", WINDOW_START) == []

    def test_discovery_empty_dir_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert llama_log_parser.discover_llama_logs(empty, WINDOW_START) == []


# ---------------------------------------------------------------------------
# Streaming iteration
# ---------------------------------------------------------------------------


class TestIterEvalTimings:
    def _write_log(self, tmp_path: Path) -> Path:
        log = tmp_path / "llama-server.log"
        log.write_text(
            "\n".join(
                [
                    fixtures.QWEN3_SPAWN_LINE,
                    "[51973] srv  log_server_r: done request: GET /slots 127.0.0.1 200",
                    fixtures.PROMPT_EVAL_REAL,
                    fixtures.DECODE_EVAL_REAL,
                    fixtures.TOTAL_TIME_LINE,
                    fixtures.MALFORMED_EVAL_LINE,
                    fixtures.PROMPT_EVAL_REAL2,
                    fixtures.DECODE_EVAL_REAL2,
                    "",
                ]
            )
            + "\n"
        )
        _set_mtime(log, MOCK_MTIME)
        return log

    def test_streams_qwen3_port_lines_only(self, tmp_path):
        log = self._write_log(tmp_path)
        timings = list(
            llama_log_parser.iter_eval_timings(log, 32999, WINDOW_START, WINDOW_END)
        )
        kinds = [t.kind for t in timings]
        assert kinds == ["prompt_eval", "decode", "prompt_eval", "decode"]
        assert all(t.port == 32999 for t in timings)
        assert all(t.model == "Qwen3" for t in timings)
        # ts is approximated from the file's last-write time.
        assert all(t.ts == MOCK_MTIME for t in timings)

    def test_wrong_port_yields_nothing(self, tmp_path):
        log = self._write_log(tmp_path)
        timings = list(
            llama_log_parser.iter_eval_timings(log, 51973, WINDOW_START, WINDOW_END)
        )
        assert timings == []

    def test_out_of_window_rotated_file_yields_nothing(self, tmp_path):
        # Rotated files (llama-server.N.log) are filtered by mtime; a file
        # whose last write is after the window end yields nothing.
        log = tmp_path / "llama-server.1.log"
        log.write_text(
            fixtures.QWEN3_SPAWN_LINE + "\n" + fixtures.DECODE_EVAL_REAL + "\n"
        )
        _set_mtime(log, datetime(2026, 8, 2, 17, 0))  # after window end
        timings = list(
            llama_log_parser.iter_eval_timings(log, 32999, WINDOW_START, WINDOW_END)
        )
        assert timings == []

    def test_live_file_written_after_window_end_still_included(self, tmp_path):
        # The live llama-server.log is appended to continuously, so its mtime
        # can exceed the window end; samples are clamped to window_end.
        log = tmp_path / "llama-server.log"
        log.write_text(
            fixtures.QWEN3_SPAWN_LINE + "\n" + fixtures.DECODE_EVAL_REAL + "\n"
        )
        _set_mtime(log, datetime(2026, 8, 2, 16, 0))  # mtime after window end 15:00
        timings = list(
            llama_log_parser.iter_eval_timings(log, 32999, WINDOW_START, WINDOW_END)
        )
        assert len(timings) == 1
        assert timings[0].ts == WINDOW_END

    def test_live_file_skipped_when_window_ended_before_its_span(self, tmp_path):
        # The live file's content begins at the previous rotation (live_span_start);
        # a window that ended before that rotation must not read it.
        log = tmp_path / "llama-server.log"
        log.write_text(
            fixtures.QWEN3_SPAWN_LINE + "\n" + fixtures.DECODE_EVAL_REAL + "\n"
        )
        _set_mtime(log, datetime(2026, 8, 2, 16, 0))
        live_span_start = datetime(2026, 8, 2, 17, 0)  # rotation after window end
        timings = list(
            llama_log_parser.iter_eval_timings(
                log, 32999, WINDOW_START, WINDOW_END, live_span_start=live_span_start
            )
        )
        assert timings == []

    def test_missing_file_yields_nothing(self, tmp_path):
        timings = list(
            llama_log_parser.iter_eval_timings(tmp_path / "nope.log", 32999, WINDOW_START, WINDOW_END)
        )
        assert timings == []


# ---------------------------------------------------------------------------
# Speed aggregation
# ---------------------------------------------------------------------------


class TestSpeedStats:
    def _log_dir(self, tmp_path: Path, lines: list[str]) -> Path:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "llama-server.log").write_text(
            fixtures.QWEN3_SPAWN_LINE + "\n" + "\n".join(lines) + "\n"
        )
        _set_mtime(log_dir / "llama-server.log", MOCK_MTIME)
        return log_dir

    def test_percentiles_and_buckets(self, tmp_path):
        lines = [
            # 14:30 (mtime) is day under the test schedule.
            "[32999]        eval time =    100.00 ms /    10 tokens (   10.00 ms per token,    40.00 tokens per second)",
            "[32999]        eval time =    100.00 ms /    10 tokens (   10.00 ms per token,    50.00 tokens per second)",
            "[32999]        eval time =    100.00 ms /    10 tokens (   10.00 ms per token,    60.00 tokens per second)",
            "[32999]        eval time =    100.00 ms /    10 tokens (   10.00 ms per token,    70.00 tokens per second)",
            "[32999] prompt eval time =     10.00 ms /    10 tokens (    1.00 ms per token,   200.00 tokens per second)",
            "[32999] prompt eval time =     10.00 ms /    10 tokens (    1.00 ms per token,   400.00 tokens per second)",
        ]
        log_dir = self._log_dir(tmp_path, lines)
        files = llama_log_parser.discover_llama_logs(log_dir, WINDOW_START)
        stats = llama_log_parser.build_speed_stats(files, WINDOW_START, WINDOW_END, _schedule())

        dec = stats.decode
        assert dec["total"].count == 4
        # linear-interpolation percentiles over [40, 50, 60, 70]
        assert dec["total"].median == 55.0
        assert dec["total"].p90 == 67.0
        assert dec["total"].p10 == 43.0
        assert dec["day"].count == 4
        assert dec["night"].count == 0
        assert dec["night"].median is None

        pe = stats.prompt_eval
        assert pe["total"].count == 2
        # linear interpolation over [200, 400]
        assert pe["total"].median == 300.0
        assert pe["total"].p90 == 380.0
        assert pe["total"].p10 == 220.0
        assert stats.files_parsed == 1
        assert stats.files_skipped == 0

    def test_files_without_qwen3_port_are_counted_and_skipped(self, tmp_path):
        log_dir = tmp_path / "logs2"
        log_dir.mkdir()
        (log_dir / "llama-server.log").write_text(
            "srv load: spawning server instance with name=mxbai-embed on port 51973\n"
            "[51973]        eval time =    10.00 ms /     1 tokens (   10.00 ms per token,   100.00 tokens per second)\n"
        )
        _set_mtime(log_dir / "llama-server.log", MOCK_MTIME)
        files = llama_log_parser.discover_llama_logs(log_dir, WINDOW_START)
        stats = llama_log_parser.build_speed_stats(files, WINDOW_START, WINDOW_END, _schedule())
        assert stats.files_parsed == 0
        assert stats.files_skipped == 1
        assert stats.decode["total"].count == 0
        assert stats.prompt_eval["total"].count == 0

    def test_empty_files_no_stats(self, tmp_path):
        log_dir = tmp_path / "logs3"
        log_dir.mkdir()
        (log_dir / "llama-server.log").write_text(fixtures.QWEN3_SPAWN_LINE + "\n")
        _set_mtime(log_dir / "llama-server.log", MOCK_MTIME)
        files = llama_log_parser.discover_llama_logs(log_dir, WINDOW_START)
        stats = llama_log_parser.build_speed_stats(files, WINDOW_START, WINDOW_END, _schedule())
        assert stats.decode["total"].count == 0
        assert stats.prompt_eval["total"].count == 0

    def test_missing_dir_no_stats(self, tmp_path):
        stats = llama_log_parser.build_speed_stats([], WINDOW_START, WINDOW_END, _schedule())
        assert stats.decode["total"].count == 0
        assert stats.files_parsed == 0

    def test_night_bucket(self, tmp_path):
        log_dir = tmp_path / "logs4"
        log_dir.mkdir()
        (log_dir / "llama-server.log").write_text(
            fixtures.QWEN3_SPAWN_LINE
            + "\n[32999]        eval time =    100.00 ms /    10 tokens (   10.00 ms per token,    42.00 tokens per second)\n"
        )
        # mtime at 23:59+ → night bucket.
        _set_mtime(log_dir / "llama-server.log", datetime(2026, 8, 2, 23, 59, 30))
        files = llama_log_parser.discover_llama_logs(log_dir, WINDOW_START)
        stats = llama_log_parser.build_speed_stats(
            files,
            datetime(2026, 8, 2, 0, 0),
            datetime(2026, 8, 3, 0, 0),
            _schedule(),
        )
        assert stats.decode["day"].count == 0
        assert stats.decode["night"].count == 1
        assert stats.decode["night"].median == 42.0


# ---------------------------------------------------------------------------
# End-to-end: report section + CSV column
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _write_logs(self, tmp_path: Path) -> Path:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES) + "\n")
        (log_dir / "llama-server.log").write_text(
            fixtures.QWEN3_SPAWN_LINE
            + "\n"
            + fixtures.PROMPT_EVAL_REAL
            + "\n"
            + fixtures.DECODE_EVAL_REAL
            + "\n"
            + fixtures.PROMPT_EVAL_REAL2
            + "\n"
            + fixtures.DECODE_EVAL_REAL2
            + "\n"
        )
        _set_mtime(log_dir / "llama-server.log", MOCK_MTIME)
        return log_dir

    def test_report_and_csv_include_speed_data(self, tmp_path):
        log_dir = self._write_logs(tmp_path)
        out_dir = tmp_path / "out"

        _ = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out_dir,
            config=None,
        )

        md = (out_dir / "report.md").read_text()

        # Decode speed section with the real decode samples (40.51, 41.21).
        decode_section = md.split("## Decode speed", 1)[1].split("## ", 1)[0]
        assert "| Model | Bucket | Samples | Median (tok/s) | p90 (tok/s) | p10 (tok/s) |" in decode_section
        assert "| Qwen3 | Total | 2 | 40.9 | 41.1 | 40.6 |" in decode_section
        assert "| Qwen3 | Day | 2 |" in decode_section

        # Prompt eval speed section with the real prompt-eval samples (99.53, 388.05).
        pe_section = md.split("## Prompt eval speed", 1)[1].split("## ", 1)[0]
        assert "| Qwen3 | Total | 2 | 243.8 | 359.2 | 128.4 |" in pe_section

        # CSV gains the decode tok/s column.
        with (out_dir / "daytime_sessions.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert "decode_tok_s" in rows[0].keys()

    def test_no_llama_logs_renders_empty_section(self, tmp_path):
        log_dir = tmp_path / "logs_none"
        log_dir.mkdir()
        (log_dir / "proxy.log").write_text("\n".join(fixtures.E2E_LINES) + "\n")
        out_dir = tmp_path / "out_none"

        _ = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=out_dir,
            config=None,
        )
        md = (out_dir / "report.md").read_text()
        assert "## Decode speed" in md
        assert "No llama-server eval timing samples" in md
        assert "## Prompt eval speed" in md

    def test_summary_to_json_includes_speed(self, tmp_path):
        log_dir = self._write_logs(tmp_path)
        result = reporting.run_analysis(
            log_dir=log_dir,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            output_dir=tmp_path / "out_json",
            config=None,
        )
        data = reporting.summary_to_json(result.summary)
        assert data["decode_speed"]["samples"] == 2
        assert data["decode_speed"]["median_tok_s"] == 40.9
        assert data["prompt_eval_speed"]["samples"] == 2
