"""Tests for the reproducible Aug 3 error-extraction harness.

Verifies the F1 deliverable (LP-0MSDP2P3E0053WOD): the harness parses
llama-proxy logs for a window, extracts every error event into a structured
dataset, aggregates counts by error type/provider/model, runs a tolerance-
based assertion pass over the headline counts, and writes CSV/JSON evidence
artifacts plus a summary table.

Tests use fixture log lines (derived from real /var/log/llama-proxy lines)
written to a temp log file; they never touch live logs.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

# Fixture log lines (real lines from the Aug 3 window; kept in this module
# so the harness tests are self-contained).
STREAM_FINISHED_ERROR = (
    "2026-08-03 10:13:14,159 - INFO - Stream finished: reason=error "
    "session=019fc52e-05a0-78d5-b59d-bcb91055b787 provider=opencode-go "
    "model=deepseek-v4-flash entry=opencode-go-2-deepseek "
    "request=[{'type': 'text', 'text': 'The conversation history before this point was compac..."
)
STREAM_FINISHED_ERROR_LOCAL = (
    "2026-08-03 11:00:00,000 - INFO - Stream finished: reason=error "
    "session=019fc52e-05a0-78d5-b59d-bcb91055b787 provider=local model=Qwen3 "
    "entry=local-qwen3 request=[...]"
)
STREAM_ERROR_LINE = (
    "2026-08-03 12:47:13,378 - WARNING - Stream error: "
    "session=019fc754-d847-75af-86ea-991480e799d0 provider=local model=Qwen3 error=NameError"
)
SLOT_SAVE_FAILED = (
    "2026-08-03 13:39:43,255 - WARNING - slot_save failed slot=2 error=ReadTimeout/ReadTimeout"
)
BACKEND_RETRY_TIMEOUT = (
    "2026-08-03 12:37:15,723 - WARNING - backend_retry path=v1/chat/completions stream=True "
    "attempt=1/8 delay=0.216s signal=connect_failures error=ConnectTimeout"
)
UPSTREAM_429 = (
    "2026-08-03 13:58:04,053 - WARNING - [remote] upstream error status=429 "
    "url=https://opencode.ai/zen/v1/chat/completions "
    "body={\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\","
    "\"message\":\"Rate limit exceeded. Please try again later.\"},\"metadata\":{}}"
)
# A line the harness must ignore (outside window / not an error).
ROUTING_CHECK = (
    "2026-08-03 10:00:00,000 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=12217 messages=2 session=019fc28d-051b-75fd-9f88-1c336f6779e0"
)

AUG3_START = datetime(2026, 8, 3, 0, 0, 0)
AUG3_END = datetime(2026, 8, 4, 0, 0, 0)

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import extract_aug3_errors as harness  # noqa: E402


def _write_fixture_log(tmp_path: Path, lines: list[str]) -> Path:
    """Write fixture lines into a fake proxy.log; returns its path."""
    log = tmp_path / "proxy.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


class TestCollectErrors:
    """The harness extracts exactly the error events inside the window."""

    def test_extracts_all_error_kinds(self, tmp_path):
        log = _write_fixture_log(
            tmp_path,
            [
                STREAM_FINISHED_ERROR,
                STREAM_FINISHED_ERROR_LOCAL,
                STREAM_ERROR_LINE,
                SLOT_SAVE_FAILED,
                BACKEND_RETRY_TIMEOUT,
                UPSTREAM_429,
            ],
        )
        events = harness.collect_errors(log.parent, AUG3_START, AUG3_END)
        kinds = sorted(e.kind for e in events)
        assert kinds == [
            "backend_retry",
            "slot_save_error",
            "stream_error",
            "stream_finish_error",
            "stream_finish_error",
            "upstream_http_error",
        ]

    def test_ignores_non_error_lines(self, tmp_path):
        log = _write_fixture_log(tmp_path, [ROUTING_CHECK, STREAM_ERROR_LINE])
        events = harness.collect_errors(log.parent, AUG3_START, AUG3_END)
        assert [e.kind for e in events] == ["stream_error"]

    def test_window_filtering(self, tmp_path):
        """Events outside the window are excluded."""
        outside = (
            "2026-08-02 23:59:59,999 - WARNING - Stream error: "
            "session=019fc754-d847-75af-86ea-991480e799d0 provider=local model=Qwen3 error=NameError"
        )
        log = _write_fixture_log(tmp_path, [outside, STREAM_ERROR_LINE])
        events = harness.collect_errors(log.parent, AUG3_START, AUG3_END)
        assert [e.kind for e in events] == ["stream_error"]

    def test_records_evidence_and_source(self, tmp_path):
        log = _write_fixture_log(tmp_path, [STREAM_FINISHED_ERROR])
        events = harness.collect_errors(log.parent, AUG3_START, AUG3_END)
        assert len(events) == 1
        e = events[0]
        assert e.provider == "opencode-go"
        assert e.model == "deepseek-v4-flash"
        assert e.session == "019fc52e-05a0-78d5-b59d-bcb91055b787"
        assert e.src_file == "proxy.log"
        assert e.raw and "Stream finished: reason=error" in e.raw


class TestAggregateCounts:
    def test_counts_by_type(self, tmp_path):
        log = _write_fixture_log(
            tmp_path,
            [STREAM_FINISHED_ERROR, STREAM_FINISHED_ERROR_LOCAL, SLOT_SAVE_FAILED],
        )
        events = harness.collect_errors(log.parent, AUG3_START, AUG3_END)
        counts = harness.aggregate_counts(events)
        assert counts["stream_finish_error"] == 2
        assert counts["slot_save_error"] == 1

    def test_split_by_provider_model(self, tmp_path):
        log = _write_fixture_log(
            tmp_path,
            [STREAM_FINISHED_ERROR, STREAM_FINISHED_ERROR, STREAM_FINISHED_ERROR_LOCAL],
        )
        events = harness.collect_errors(log.parent, AUG3_START, AUG3_END)
        split = harness.split_counts(events, "stream_finish_error")
        assert split[("opencode-go", "deepseek-v4-flash")] == 2
        assert split[("local", "Qwen3")] == 1


class TestAssertions:
    """Tolerance-based headline-count assertions (AC: "within tolerance")."""

    def test_passes_with_headline_counts(self):
        counts = {
            "stream_finish_error": 127,
            "stream_error": 6,
            "slot_save_error": 17,
            "backend_retry": 93,
            "upstream_http_error": 112,
        }
        split = {
            ("opencode-go", "deepseek-v4-flash"): 93,
            ("opencode", "deepseek-v4-flash-free"): 28,
            ("local", "Qwen3"): 6,
        }
        result = harness.run_assertions(counts, split, free_usage_429=4)
        assert result["passed"] is True, result
        assert result["failures"] == []

    def test_fails_below_floor(self):
        counts = {
            "stream_finish_error": 5,  # far below expected 98
            "stream_error": 0,
            "slot_save_error": 0,
            "backend_retry": 0,
            "upstream_http_error": 0,
        }
        split = {}
        result = harness.run_assertions(counts, split, free_usage_429=4)
        assert result["passed"] is False
        assert len(result["failures"]) >= 2  # stream errors + several floors

    def test_failure_messages_are_actionable(self):
        counts = {
            "stream_finish_error": 5,
            "stream_error": 0,
            "slot_save_error": 0,
            "backend_retry": 0,
            "upstream_http_error": 0,
        }
        result = harness.run_assertions(counts, {}, free_usage_429=0)
        msg = " ".join(result["failures"])
        assert "Stream finished" in msg or "stream_finish_error" in msg


class TestOutputArtifacts:
    """CSV/JSON evidence extracts + summary table (repo artifacts)."""

    def test_writes_all_artifacts(self, tmp_path):
        log = _write_fixture_log(
            tmp_path,
            [STREAM_FINISHED_ERROR, STREAM_ERROR_LINE, SLOT_SAVE_FAILED, UPSTREAM_429],
        )
        out = tmp_path / "out"
        report = harness.main(
            ["--log-dir", str(log.parent), "--output-dir", str(out),
             "--start", "2026-08-03 00:00:00", "--end", "2026-08-04 00:00:00",
             "--no-assert"]
        )
        assert report["passed"] is True
        assert (out / "errors.csv").exists()
        assert (out / "counts.csv").exists()
        assert (out / "counts.json").exists()
        assert (out / "evidence.txt").exists()
        assert (out / "summary.md").exists()

    def test_errors_csv_shape(self, tmp_path):
        log = _write_fixture_log(
            tmp_path,
            [STREAM_FINISHED_ERROR, STREAM_ERROR_LINE, SLOT_SAVE_FAILED, UPSTREAM_429],
        )
        out = tmp_path / "out"
        harness.main(
            ["--log-dir", str(log.parent), "--output-dir", str(out),
             "--start", "2026-08-03 00:00:00", "--end", "2026-08-04 00:00:00",
             "--no-assert"]
        )
        rows = list(csv.DictReader((out / "errors.csv").open()))
        assert len(rows) == 4
        cols = {"error_type", "timestamp", "provider", "model", "session",
                "evidence", "source_file"}
        assert cols.issubset(rows[0].keys())

    def test_counts_json_matches_csv(self, tmp_path):
        log = _write_fixture_log(
            tmp_path,
            [STREAM_FINISHED_ERROR, STREAM_ERROR_LINE, SLOT_SAVE_FAILED, UPSTREAM_429],
        )
        out = tmp_path / "out"
        harness.main(
            ["--log-dir", str(log.parent), "--output-dir", str(out),
             "--start", "2026-08-03 00:00:00", "--end", "2026-08-04 00:00:00",
             "--no-assert"]
        )
        data = json.loads((out / "counts.json").read_text())
        assert data["by_type"]["stream_finish_error"] == 1
        assert data["by_type"]["upstream_http_error"] == 1
        assert data["window_start"] == "2026-08-03 00:00:00"

    def test_missing_log_dir_is_actionable(self, tmp_path):
        out = tmp_path / "out"
        report = harness.main(
            ["--log-dir", str(tmp_path / "nope"), "--output-dir", str(out),
             "--start", "2026-08-03 00:00:00", "--end", "2026-08-04 00:00:00"]
        )
        assert report["passed"] is False
        assert "error" in report["message"].lower() or "no log" in report["message"].lower()
