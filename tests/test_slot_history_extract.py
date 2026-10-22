#!/usr/bin/env python3
"""Tests for the slot-count history / concurrency extractor (LP-0MVGCQ3QN009HRE8).

These pin the parsing contract so the numbers in
``docs/slot-count-reevaluation-LP-0MVGCQ3QN009HRE8.md`` stay traceable to code:
the historical report table (AC1) and the concurrency-vs-capacity statement
(AC2) must be reproducible from the same inputs.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from slot_history_extract import (
    ConcurrencyResult,
    _duration_to_seconds,
    collect_reports,
    parse_report,
    reconstruct_concurrency,
    render_concurrency_markdown,
    render_history_markdown,
    select_daily_rows,
    slot_config_for,
)

REPORT_TEMPLATE = """# Proxy Usage Analysis Report

- Generated: 2026-10-20 05:00:09
- Window: 2026-10-19 05:00:01 → 2026-10-20 05:00:01 (24.0h)
- Sessions in window: **348** | Requests: **17580** (local 523 / remote 17057)
- Fallback events: **16400** (93.3% of requests) | dispatch denied: 2641

## Summary by hour

| Hour | Busy | % | Requests started | Local attempts | Local served | Fallbacks | Local-skip |
|---|---|---|---|---|---|---|---|
| Totals | 7h 47m | 29.3% | 17580 | 523 | 520 | 17057 | 120 |

## Session summary

| Metric | Total | Fast | Cheap |
|---|---|---|---|
| Queued → dispatched local | 855 | - | 855 |
| Fallback after queue | 4947 | - | 4947 |
| Dispatch denied | 2641 | 1389 (52.6%) | 1252 (47.4%) |

## Decode speed

| Model | Bucket | Samples | Median (tok/s) | p90 (tok/s) | p10 (tok/s) |
|---|---|---|---|---|---|
| Qwen3 | Total | 463 | 21.2 | 36.0 | 9.4 |

## Prompt eval speed

| Model | Bucket | Samples | Median (tok/s) | p90 (tok/s) | p10 (tok/s) |
|---|---|---|---|---|---|
| Qwen3 | Total | 463 | 57.3 | 179.2 | 10.3 |
"""


def _write_report(root: Path, snapshot: str, text: str) -> Path:
    path = root / snapshot / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_duration_to_seconds_handles_report_formats():
    assert _duration_to_seconds("7h 47m") == 7 * 3600 + 47 * 60
    assert _duration_to_seconds("1h 00m") == 3600
    assert _duration_to_seconds("30m 30s") == 1830
    assert _duration_to_seconds("0m") is None
    assert _duration_to_seconds("--") is None


def test_parse_report_extracts_all_metrics():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_report(Path(tmp), "2026-10-20", REPORT_TEMPLATE)
        row = parse_report(path)

    assert row is not None
    assert row["window_start"] == "2026-10-19T05:00:01"
    assert row["sessions"] == 348
    assert row["requests"] == 17580
    assert row["local_requests"] == 523
    assert row["local_share"] == 523 / 17580
    assert row["dispatch_denied"] == 2641
    assert row["queued_dispatched"] == 855
    assert row["fallback_after_queue"] == 4947
    assert row["busy_seconds"] == 7 * 3600 + 47 * 60
    assert row["busy_pct"] == 29.3
    assert row["decode_median_tps"] == 21.2
    assert row["prompt_eval_median_tps"] == 57.3
    # Window start 2026-10-19 is in the priority-routing era (N=2 both modes).
    assert row["fast_slots"] == "2"
    assert row["cheap_slots"] == "2"


def test_parse_report_rejects_non_report():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.md"
        path.write_text("just some notes\n")
        assert parse_report(path) is None


def test_slot_config_for_maps_every_era_boundary():
    assert slot_config_for(datetime(2026, 7, 20))[:2] == ("2", "2")
    assert slot_config_for(datetime(2026, 8, 2))[:2] == ("3", "3")
    assert slot_config_for(datetime(2026, 8, 3))[:2] == ("4", "4")
    assert slot_config_for(datetime(2026, 9, 14))[:2] == ("1", "3")
    assert slot_config_for(datetime(2026, 10, 17))[:2] == ("2", "3")
    assert slot_config_for(datetime(2026, 10, 19))[:2] == ("2", "2")


def test_collect_reports_sorted_and_skips_junk():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_report(root, "2026-10-20", REPORT_TEMPLATE)
        _write_report(
            root,
            "2026-10-18",
            REPORT_TEMPLATE.replace("2026-10-19 05:00:01", "2026-10-17 05:00:01").replace(
                "2026-10-20 05:00:01", "2026-10-18 05:00:01"
            ),
        )
        (root / "junk").mkdir()
        (root / "junk" / "report.md").write_text("not a report")
        rows = collect_reports(root)

    assert [r["window_start"][:10] for r in rows] == ["2026-10-17", "2026-10-19"]


def test_select_daily_rows_prefers_date_named_and_complete():
    rows = [
        {"window_start": "2026-10-19T05:00:01", "snapshot": "measure-run", "requests": 50},
        {"window_start": "2026-10-19T05:00:01", "snapshot": "2026-10-20", "requests": 900},
        {"window_start": "2026-10-19T05:00:01", "snapshot": "2026-10-20_2", "requests": 1200},
        {"window_start": "2026-10-18T05:00:01", "snapshot": "2026-10-19", "requests": 0},
        {"window_start": "2026-10-17T05:00:01", "snapshot": "2026-10-18", "requests": 800},
    ]
    selected = select_daily_rows(rows)
    assert [r["window_start"][:10] for r in selected] == ["2026-10-17", "2026-10-19"]
    # The most complete date-named snapshot for 2026-10-19 wins; the empty day is dropped.
    assert selected[1]["snapshot"] == "2026-10-20_2"
    assert selected[1]["requests"] == 1200


# ---------------------------------------------------------------------------
# Concurrency reconstruction
# ---------------------------------------------------------------------------

T0 = datetime(2026, 10, 20, 10, 0, 0)


def _line(ts: datetime, text: str) -> str:
    return f"{ts.strftime('%Y-%m-%d %H:%M:%S')},000 - INFO - {text}\n"


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines))


def test_reconstruct_concurrency_peak_and_distribution(tmp_path):
    log = tmp_path / "proxy.log"
    lines = [
        # Session A: 10:00:00 -> 10:02:30
        _line(datetime(2026, 10, 20, 10, 0, 0), "Stream started: provider=local model=Qwen3 session=A"),
        _line(datetime(2026, 10, 20, 10, 2, 30), "Stream finished: reason=stop session=A provider=local"),
        # Session B: 10:01:00 -> 10:03:30
        _line(datetime(2026, 10, 20, 10, 1, 0), "Stream started: provider=local model=Qwen3 session=B"),
        _line(datetime(2026, 10, 20, 10, 3, 30), "Stream finished: reason=stop session=B provider=local"),
        # Session C: 10:01:30 -> 10:01:45
        _line(datetime(2026, 10, 20, 10, 1, 30), "Stream started: provider=local model=Qwen3 session=C"),
        _line(datetime(2026, 10, 20, 10, 1, 45), "Stream finished: reason=stop session=C provider=local"),
    ]
    _write_log(log, lines)

    result = reconstruct_concurrency(
        tmp_path, datetime(2026, 10, 20, 10, 0, 0), datetime(2026, 10, 20, 10, 5, 0)
    )

    # Samples at 10:01, 10:02, 10:03, 10:04, 10:05 (complete minute boundaries).
    assert result.minute_concurrency == [2, 2, 1, 0, 0]
    # A, B and C are all in flight at 10:01:30.
    assert result.peak == 3
    assert result.intervals == 3
    assert result.share_above(1) == 0.4  # 10:01 and 10:02 above 1
    # session-minutes: 10:01=2, 10:02=2, 10:03=1,04=0,05=0 -> total 5,
    # denied at N=1 = (1+1+0+0+0)=2 -> 2/5
    assert result.denied_share(1) == 0.4


def test_reconstruct_concurrency_pairs_across_rotated_files(tmp_path):
    older = tmp_path / "proxy.log-2026-10-20_00"
    newer = tmp_path / "proxy.log.2026-10-20_00"
    _write_log(
        older,
        [
            _line(datetime(2026, 10, 20, 9, 59, 0), "Stream started: provider=local session=X"),
        ],
    )
    _write_log(
        newer,
        [
            _line(datetime(2026, 10, 20, 10, 1, 30), "Stream finished: reason=stop session=X provider=local"),
        ],
    )

    result = reconstruct_concurrency(
        tmp_path, datetime(2026, 10, 20, 10, 0, 0), datetime(2026, 10, 20, 10, 2, 0),
        files=[newer, older],  # deliberately out of order
    )
    # The stream is in flight at the first sample (10:01) and finished by the
    # second (10:02); pairing across the rotated files is what is asserted.
    assert result.intervals == 1
    assert result.minute_concurrency == [1, 0]
    assert result.peak == 1


def test_reconstruct_concurrency_drops_stale_and_inverted_pairs(tmp_path):
    log = tmp_path / "proxy.log"
    lines = [
        # Stale open more than MAX_STREAM_SECONDS before the window, finished
        # 10 min into the window: dropped at the scan pre-filter, never paired.
        _line(datetime(2026, 10, 20, 6, 0, 0), "Stream started: provider=local session=stale"),
        _line(datetime(2026, 10, 20, 10, 10, 0), "Stream finished: reason=stop session=stale provider=local"),
        # A normal stream.
        _line(datetime(2026, 10, 20, 10, 1, 0), "Stream started: provider=local session=ok"),
        _line(datetime(2026, 10, 20, 10, 1, 5), "Stream finished: reason=stop session=ok provider=local"),
        # A finish with no matching start is ignored.
        _line(datetime(2026, 10, 20, 10, 2, 0), "Stream finished: reason=stop session=orphan provider=local"),
    ]
    _write_log(log, lines)
    result = reconstruct_concurrency(
        tmp_path, datetime(2026, 10, 20, 10, 0, 0), datetime(2026, 10, 20, 10, 5, 0)
    )
    assert result.intervals == 1
    assert result.peak == 1


def test_reconstruct_concurrency_counts_denials_and_contention(tmp_path):
    log = tmp_path / "proxy.log"
    _write_log(
        log,
        [
            _line(datetime(2026, 10, 20, 10, 0, 10), "local_dispatch_denied session=A owner=B active=2 cold_start=False"),
            _line(datetime(2026, 10, 20, 10, 0, 20), "local_dispatch_denied session=C owner=B active=1 cold_start=False"),
            _line(datetime(2026, 10, 20, 10, 0, 30), "contention_queue_wait max_wait=120.0s max_depth=8 saturation=generating"),
            _line(datetime(2026, 10, 20, 10, 0, 40), "contention_queue_dispatch provider=local session=A queued_duration=1.0s"),
            _line(datetime(2026, 10, 20, 10, 0, 50), "contention_queue_fallback_after_queue provider=local session=C queued_duration=120.0s reason=local_lease_active"),
            # Outside the window: must not be counted.
            _line(datetime(2026, 10, 20, 9, 0, 0), "local_dispatch_denied session=Z owner=Y active=2 cold_start=False"),
        ],
    )
    result = reconstruct_concurrency(
        tmp_path, datetime(2026, 10, 20, 10, 0, 0), datetime(2026, 10, 20, 10, 5, 0)
    )
    assert result.dispatch_denied_total == 2
    assert result.dispatch_denied_by_active == {2: 1, 1: 1}
    assert result.contention_dispatch == 1
    assert result.contention_fallback_after_queue == 1


def test_reconstruct_concurrency_captures_prefill_and_finish_reasons(tmp_path):
    log = tmp_path / "proxy.log"
    _write_log(
        log,
        [
            _line(
                datetime(2026, 10, 20, 10, 0, 5),
                "dispatch_first_byte_ms=61800.0 dispatch_to_first_byte_ms=61799.9 "
                "cache_state=warm cached_ratio=0.9 queue_wait_ms=1.0 session=A model=Qwen3",
            ),
            _line(
                datetime(2026, 10, 20, 10, 0, 6),
                "dispatch_first_byte_ms=817000.0 dispatch_to_first_byte_ms=816999.0 "
                "cache_state=cold cached_ratio=0.0 queue_wait_ms=0.9 session=B model=Qwen3",
            ),
            _line(
                datetime(2026, 10, 20, 10, 0, 30),
                "Stream finished: reason=tool_calls tokens=1200/50/1250 session=A provider=local model=Qwen3",
            ),
            _line(
                datetime(2026, 10, 20, 10, 0, 40),
                "Stream finished: reason=length tokens=5000/18000/23000 session=B provider=local model=Qwen3",
            ),
            _line(
                datetime(2026, 10, 20, 10, 0, 45),
                "Stream finished: reason=stop tokens=10/10/20 session=C provider=MunitoAI Free model=deepseek",
            ),
        ],
    )
    result = reconstruct_concurrency(
        tmp_path, datetime(2026, 10, 20, 10, 0, 0), datetime(2026, 10, 20, 10, 5, 0)
    )
    prefill = result.prefill_summary()
    assert prefill is not None
    assert prefill["count"] == 2
    assert prefill["median_s"] == 817.0  # sorted [61.8, 817.0] -> middle index 1
    assert prefill["max_s"] == 817.0
    assert result.finish_reason_counts == {"tool_calls": 1, "length": 1, "stop": 1}
    assert result.local_finish_count == 2


def test_concurrency_result_capacity_model():
    result = ConcurrencyResult(minute_concurrency=[0, 1, 2, 3, 4])
    assert result.distribution() == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    assert result.share_above(2) == 0.4  # only the 3 and 4 minutes
    # total session-minutes = 10; denied N=2 = 1+2 = 3 -> 0.3
    assert result.denied_share(2) == 0.3
    assert ConcurrencyResult(minute_concurrency=[]).denied_share(2) is None


def test_render_concurrency_markdown_contains_capacity_table():
    result = ConcurrencyResult(minute_concurrency=[1, 2, 3, 4], peak=4)
    md = render_concurrency_markdown(result)
    assert "| 2 |" in md and "| 3 |" in md and "| 4 |" in md
    assert "session-minutes denied" in md


def test_render_history_markdown_contains_row_values():
    rows = [
        {
            "window_start": "2026-10-19T05:00:01",
            "snapshot": "2026-10-20",
            "fast_slots": "2",
            "cheap_slots": "2",
            "requests": 17580,
            "local_share": 0.0298,
            "busy_pct": 29.3,
            "busy_seconds": 7 * 3600 + 47 * 60,
            "decode_median_tps": 21.2,
            "prompt_eval_median_tps": 57.3,
            "queued_dispatched": 855,
            "fallback_after_queue": 4947,
            "dispatch_denied": 2641,
        }
    ]
    md = render_history_markdown(rows)
    assert "2026-10-19" in md
    assert "17,580" in md
    assert "2/2" in md
    assert "2,641" in md
    assert "7h 47m" in md
