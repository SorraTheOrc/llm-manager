#!/usr/bin/env python3
"""Behavioural tests for scripts/prefill_throughput_report.py (LP-0MWCEHDRV000B7HD).

The report re-measures prompt-eval (prefill) throughput from llama-server's own
``prompt eval time`` lines, which is the only way to re-measure the shipped
config's throughput without occupying the GPU. The tests pin the properties the
re-measurement's conclusions depend on, without a live llama-server:

- ``parse_prompt_eval_lines`` — only well-formed *measured* lines are counted;
  a truncated/short line is skipped rather than poisoning the aggregate.
- ``aggregate_tokens_per_second`` — aggregate is tokens/engine-milliseconds over
  the whole window, which is **not** the mean of the per-request rates. That
  distinction is the item's headline (per-request vs aggregate throughput).
- ``bucket_for`` / ``summarise`` — the 40–80k regime the item targets is
  bucketed on its own, with its own aggregate.
- ``log_coverage`` — the measurement window is derived from the log files
  themselves (rotated name → that UTC day; live file → ctime..mtime), so every
  quoted figure carries a window.
"""

from __future__ import annotations

import gzip
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prefill_throughput_report import (
    aggregate_tokens_per_second,
    bucket_for,
    log_coverage,
    parse_prompt_eval_lines,
    summarise,
)

MEASURED_LINES = [
    "[43745] prompt eval time =   57125.15 ms / 18362 tokens (    3.11 ms per token,   321.43 tokens per second)",
    "[43745] prompt eval time =    8737.28 ms /  1510 tokens (    5.79 ms per token,   172.82 tokens per second)",
    "[43745] prompt eval time =  240000.00 ms / 60000 tokens (    4.00 ms per token,   250.00 tokens per second)",
]


# ---------------------------------------------------------------------------
# Parsing: measured lines only
# ---------------------------------------------------------------------------


def test_parse_extracts_tokens_milliseconds_and_rate() -> None:
    rows = parse_prompt_eval_lines(MEASURED_LINES)
    assert len(rows) == 3
    assert rows[0]["prompt_tokens"] == 18_362
    assert rows[0]["prompt_ms"] == 57_125.15
    assert rows[0]["tokens_per_second"] == 321.43


def test_parse_skips_unmeasured_and_malformed_lines() -> None:
    noisy = [
        "slot update_slots: id  1 | task 1 | n_past = 5265 (not a timing line)",
        # llama-server prints a zero-token line for a fully-cached prompt; it is
        # a cache hit, not a prefill measurement, and would flatten the median.
        "[43745] prompt eval time =     0.00 ms /     0 tokens (   12.00 ms per token,    83.33 tokens per second)",
        "prompt eval time = truncated",
        "PREFILL_CONCURRENCY_DONE",
    ]
    assert parse_prompt_eval_lines(noisy) == []


def test_parse_reads_gzip_and_plain_sources(tmp_path: Path) -> None:
    plain = tmp_path / "llama-server.log"
    plain.write_text("\n".join(MEASURED_LINES) + "\n")
    packed = tmp_path / "llama-server.log-2026-11-10.gz"
    with gzip.open(packed, "wt") as handle:
        handle.write("\n".join(MEASURED_LINES) + "\n")
    assert len(parse_prompt_eval_lines([plain])) == 3
    assert len(parse_prompt_eval_lines([packed])) == 3


# ---------------------------------------------------------------------------
# Aggregate: tokens / engine time, never the mean of per-request rates
# ---------------------------------------------------------------------------


def test_aggregate_is_tokens_over_engine_milliseconds() -> None:
    rows = [
        {"prompt_tokens": 20_000, "prompt_ms": 100_000.0},
        {"prompt_tokens": 10_000, "prompt_ms": 40_000.0},
    ]
    # 30,000 tokens / 140 s = 214.3 tok/s, whereas the mean per-request rate is
    # (200 + 250) / 2 = 225 tok/s: the aggregate is the honest number.
    assert aggregate_tokens_per_second(rows) == 214.3


def test_aggregate_empty_is_zero() -> None:
    assert aggregate_tokens_per_second([]) == 0.0


# ---------------------------------------------------------------------------
# Bucketing and summary shape the report quotes
# ---------------------------------------------------------------------------


def test_bucket_boundaries_split_the_targeted_regime() -> None:
    assert bucket_for(500) == "lt8k"
    assert bucket_for(8_000) == "8k-20k"
    assert bucket_for(19_999) == "8k-20k"
    assert bucket_for(20_000) == "20k-40k"
    assert bucket_for(40_960) == "40k-80k"
    assert bucket_for(80_000) == "ge80k"


def test_summarise_reports_aggregate_beside_per_request_percentiles() -> None:
    rows = [
        {"prompt_tokens": 40_000, "prompt_ms": 160_000.0},
        {"prompt_tokens": 40_000, "prompt_ms": 200_000.0},
        {"prompt_tokens": 8_000, "prompt_ms": 16_000.0},
    ]
    summary = summarise(rows)
    assert summary["requests"] == 3
    assert summary["prompt_tokens"] == 88_000
    assert summary["aggregate_tokens_per_second"] == 234.0  # 88k / 376 s
    # Per-request rates: 250, 200, 500 -> p50 250, p90 450.
    assert summary["per_request"]["p50"] == 250.0
    assert summary["per_request"]["p90"] == 450.0
    target = summary["buckets"]["40k-80k"]
    assert target["requests"] == 2
    assert target["aggregate_tokens_per_second"] == 222.2  # 80k / 360 s


# ---------------------------------------------------------------------------
# Window provenance
# ---------------------------------------------------------------------------


def test_log_coverage_of_a_rotated_file_is_the_day_it_holds(tmp_path: Path) -> None:
    # dateext names the file with the *rotation* instant, so a file rotated at
    # 2026-11-11T00:00 holds 2026-11-10's content.
    rotated = tmp_path / "llama-server.log-2026-11-11.gz"
    rotated.write_bytes(b"")
    start, end = log_coverage(rotated)
    assert start == datetime(2026, 11, 10, tzinfo=UTC)
    assert end == datetime(2026, 11, 11, tzinfo=UTC)


def test_log_coverage_of_the_live_file_ends_at_its_mtime(tmp_path: Path) -> None:
    live = tmp_path / "llama-server.log"
    live.write_text("x\n")
    start, end = log_coverage(live)
    assert start <= end
    assert abs((end - datetime.now(UTC)).total_seconds()) < 300


def test_log_coverage_of_the_live_file_starts_at_the_last_rotation(
    tmp_path: Path,
) -> None:
    live = tmp_path / "llama-server.log"
    live.write_text("x\n")
    rotation = datetime(2026, 11, 11, tzinfo=UTC)
    start, end = log_coverage(live, live_start=rotation)
    assert start == rotation
    assert end >= start
