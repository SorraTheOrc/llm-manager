"""Tests for the oversized-session correlation analysis.

F2 deliverable (LP-0MTC8A2UB0040NKQ): correlate oversized sessions (routing
checks with ``estimated_tokens`` beyond the per-slot clamp) with the
fallback-storm/decode-collapse signals on the 2026-08-26 incident day, and
quantify wasted prefill work (checks where the session context ratio > 1.0 —
a context that can never be resident in one local slot).

Fixture lines mirror real proxy.log / llama-server.log shapes. Tests write
them into a temp log dir; they never touch live logs. Assertions are on
observable behaviour (parsed counts, per-session aggregates, hourly timeline,
wasted-work math, decode evidence).
"""

from __future__ import annotations

import gzip as gz
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import correlate_oversized_sessions as analyzer  # noqa: E402


# --- Fixture log lines (real shapes) ------------------------------------

ROUTING_FAST_SMALL = (
    "2026-08-26 10:00:00,000 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=49912 cold_threshold=38000 warm_threshold=83285 new_tokens=2251 "
    "cached_ratio=0.95 messages=68 session=herdr-1787669217-851654-15207"
)
ROUTING_FAST_OVER = (
    "2026-08-26 10:01:00,000 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=100000 cold_threshold=38000 warm_threshold=83285 new_tokens=2251 "
    "cached_ratio=0.00 messages=253 session=herdr-1787669217-851654-15207"
)
ROUTING_CHEAP_OVER = (
    "2026-08-26 10:02:00,000 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=70000 cold_threshold=38000 warm_threshold=100000 new_tokens=2251 "
    "cached_ratio=0.00 messages=200 session=herdr-1787669217-851654-15208"
)
PRESSURE_LINE = (
    "2026-08-26 22:01:50,864 - WARNING - context_pressure "
    "session=herdr-1787669217-851654-15207 estimated_tokens=75204 per_slot_ctx=83285 "
    "ratio=0.90 >= 0.80; consider compacting the session history"
)
SKIP_TOO_LARGE = (
    "2026-08-26 22:02:00,000 - INFO - routing_skip_local provider=local-qwen3 "
    "model=Qwen3 estimated_tokens=424128 cold_threshold=38000 warm_threshold=83285 "
    "new_tokens=424128 cached_ratio=0.00 reason=context_too_large \u2192 skipping "
    "local, routing to next remote provider session=herdr-1787669217-851654-15207"
)
SKIP_BYPASS = (
    "2026-08-26 22:03:00,000 - INFO - routing_skip_local provider=local-qwen3 "
    "model=Qwen3 estimated_tokens=424128 cold_threshold=38000 warm_threshold=83285 "
    "new_tokens=424128 cached_ratio=0.00 reason=large_context_bypass \u2192 skipping "
    "local, routing to next remote provider session=herdr-1787669217-851654-15207"
)
DENIED_LINE = (
    "2026-08-26 22:04:00,000 - INFO - local_dispatch_denied "
    "session=herdr-1787669217-851654-15207 owner=herdr-1787669217-851654-15207 active=2"
)
UPSTREAM_503 = (
    "2026-08-26 22:05:00,000 - WARNING - [remote] upstream error status=503 "
    "url=https://opencode.ai/zen/go elapsed=12.3"
)
UPSTREAM_402 = (
    "2026-08-26 22:06:00,000 - WARNING - [remote] upstream error status=402 "
    "url=https://api.deepseek.com/v1/chat/completions"
)
LLAMA_EVAL_SLOW = (
    "[53169]        eval time =   61474.19 ms /  58 tokens (   1059.90 ms per token,    0.94 tokens per second)"
)
LLAMA_EVAL_FAST = (
    "[53169]        eval time =   4783.12 ms /  179 tokens (   26.72 ms per token,    37.42 tokens per second)"
)
LLAMA_PREFILL = (
    "[53169] prompt eval time =  147659.36 ms / 30860 tokens (    4.78 ms per token,   208.99 tokens per second)"
)
LLAMA_WARMUP = "srv          load: spawning server instance with name=Qwen3 on port 8000"


def _write_log(tmp_path: Path, lines: list[str], name: str = "proxy.log") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_llama(tmp_path: Path, lines: list[str], name: str, mtime: datetime) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.utime(p, (mtime.timestamp(), mtime.timestamp()))
    return p


DAY = datetime(2026, 8, 26)
CAPS = {"fast": 83285, "cheap": 61440}


class TestRegexes:
    def test_pressure_regex(self):
        m = analyzer.PRESSURE_LINE_RE.search(PRESSURE_LINE)
        assert m is not None
        assert m.group(2) == "herdr-1787669217-851654-15207"
        assert int(m.group(3)) == 75204
        assert float(m.group(5)) == 0.90

    def test_skip_regex(self):
        m = analyzer.SKIP_LINE_RE.search(SKIP_TOO_LARGE)
        assert m is not None
        assert m.group(2) == "context_too_large"
        assert analyzer.SKIP_LINE_RE.search(SKIP_BYPASS).group(2) == "large_context_bypass"

    def test_denied_and_upstream(self):
        assert analyzer.DENIED_RE.search(DENIED_LINE) is not None
        assert analyzer.DENIED_RE.search(ROUTING_FAST_SMALL) is None
        assert analyzer.UPSTREAM_5XX_RE.search(UPSTREAM_503).group(2) == "503"
        assert analyzer.UPSTREAM_5XX_RE.search(UPSTREAM_402) is None

    def test_llama_regexes_split_families(self):
        d = analyzer.LLAMA_EVAL_RE.search(LLAMA_EVAL_SLOW)
        assert d is not None
        assert round(float(d.group(4)), 2) == 0.94  # slow decode
        p = analyzer.LLAMA_PREFILL_RE.search(LLAMA_PREFILL)
        assert p is not None
        assert p.group(2) == "30860"
        # prefill lines must not be counted as decode lines
        assert analyzer.LLAMA_EVAL_RE.search(LLAMA_PREFILL) is None
        assert analyzer.LLAMA_PREFILL_RE.search(LLAMA_WARMUP) is None


class TestAnalyzeDay:
    def test_event_rollup_and_wasted_work(self, tmp_path):
        _write_log(
            tmp_path,
            [
                ROUTING_FAST_SMALL,   # est 49912, ratio < 1 -> not wasted
                ROUTING_FAST_OVER,    # est 100000 > 83285 -> wasted
                ROUTING_CHEAP_OVER,   # est 70000 > 61440 (cheap cap) -> wasted
                PRESSURE_LINE,
                SKIP_TOO_LARGE,
                SKIP_BYPASS,
                DENIED_LINE,
                UPSTREAM_503,
                UPSTREAM_402,
            ],
        )
        res = analyzer.analyze_day(tmp_path, DAY, CAPS)

        assert res.total_pressure == 1
        assert res.total_skips == 2
        agg = res.sessions["herdr-1787669217-851654-15207"]
        assert agg.mode == "fast"
        assert agg.checks == 2
        assert agg.peak_est == 100000
        # prefill work: 49912 + 100000; wasted: 100000 only
        assert agg.prefill_work == 149912
        assert agg.wasted_work == 100000
        assert agg.ratio_gt_one == 1
        assert agg.skip_count == 2
        assert agg.pressure_count == 1

        cheap = res.sessions["herdr-1787669217-851654-15208"]
        assert cheap.mode == "cheap"
        assert cheap.wasted_work == 70000

        assert res.total_prefill_work == 149912 + 70000
        assert res.total_wasted_work == 100000 + 70000

    def test_hourly_timeline_buckets(self, tmp_path):
        _write_log(
            tmp_path,
            [
                ROUTING_FAST_SMALL,  # 10:00
                PRESSURE_LINE,       # 22:01
                UPSTREAM_503,        # 22:05
                SKIP_TOO_LARGE,      # 22:02
                DENIED_LINE,         # 22:04
                ROUTING_CHEAP_OVER,  # 10:02
            ],
        )
        res = analyzer.analyze_day(tmp_path, DAY, CAPS)
        by_hour = {h: row for h, *row in res.hours}
        assert by_hour["10:00"][4] == 2          # routing checks
        assert by_hour["22:00"][0] == 1          # pressure
        assert by_hour["22:00"][1] == 1          # skips
        assert by_hour["22:00"][2] == 1          # denied
        assert by_hour["22:00"][3] == 1          # upstream 5xx
        assert by_hour["00:00"][4] == 0          # empty hours stay zero

    def test_window_filter(self, tmp_path):
        outside = ROUTING_FAST_OVER.replace("2026-08-26", "2026-08-25")
        _write_log(tmp_path, [outside, ROUTING_FAST_SMALL])
        res = analyzer.analyze_day(tmp_path, DAY, CAPS)
        assert res.total_routing_checks == 1


class TestLlamaDecodeWindow:
    def test_slow_decodes_attributed_by_mtime(self, tmp_path):
        _write_llama(
            tmp_path,
            [LLAMA_EVAL_SLOW, LLAMA_EVAL_FAST, LLAMA_PREFILL, LLAMA_WARMUP],
            name="llama-server.1.log",
            mtime=datetime(2026, 8, 26, 1, 0),  # inside incident window
        )
        _write_llama(
            tmp_path,
            [LLAMA_EVAL_FAST],
            name="llama-server.2.log",
            mtime=datetime(2026, 8, 27, 23, 0),  # outside
        )
        windows = analyzer.llama_log_windows(tmp_path)
        stats = analyzer.decode_stats_for_window(
            tmp_path, windows, datetime(2026, 8, 25, 22, 0), datetime(2026, 8, 27, 0, 0)
        )
        assert stats["decode_obs"] == 2
        assert stats["slow_decodes_lt_1tps"] == 1
        assert stats["slow_decodes_examples"][0][0] == 0.94
        assert stats["prefill_total_tokens"] == 30860
        assert stats["windows"] == ["llama-server.1.log"]

    def test_binary_garbage_does_not_crash(self, tmp_path):
        p = tmp_path / "llama-server.1.log"
        p.write_bytes(LLAMA_EVAL_SLOW.encode() + b"\xff\xfe garbage\n" + LLAMA_EVAL_FAST.encode())
        os.utime(p, (datetime(2026, 8, 26, 1, 0).timestamp(),) * 2)
        windows = analyzer.llama_log_windows(tmp_path)
        stats = analyzer.decode_stats_for_window(
            tmp_path, windows, datetime(2026, 8, 25, 22, 0), datetime(2026, 8, 27, 0, 0)
        )
        assert stats["decode_obs"] == 2


class TestReportAndArtifacts:
    def test_build_report_math(self, tmp_path):
        _write_log(
            tmp_path,
            [ROUTING_FAST_SMALL, ROUTING_FAST_OVER, PRESSURE_LINE, SKIP_TOO_LARGE],
        )
        _write_llama(
            tmp_path,
            [LLAMA_EVAL_SLOW, LLAMA_PREFILL],
            name="llama-server.1.log",
            mtime=datetime(2026, 8, 26, 1, 0),
        )
        report = analyzer.build_report(tmp_path, DAY, CAPS)
        t = report["totals"]
        assert t["routing_checks"] == 2
        assert t["prefill_work_tokens"] == 149912
        assert t["wasted_work_tokens"] == 100000
        assert t["wasted_pct_of_prefill"] == pytest.approx(66.7, abs=0.1)
        assert report["llama_server_decode"]["slow_decodes_lt_1tps"] == 1
        top = report["top_sessions"][0]
        assert top["session"] == "herdr-1787669217-851654-15207"
        assert top["checks_ratio_gt_1"] == 1
        json.dumps(report)  # JSON-serialisable

    def test_writes_artifacts(self, tmp_path):
        report = {
            "day": "2026-08-26",
            "caps": {},
            "totals": {
                "routing_checks": 0,
                "prefill_work_tokens": 0,
                "wasted_work_tokens": 0,
                "wasted_pct_of_prefill": 0.0,
                "context_pressure_warnings": 0,
                "routing_skips": 0,
                "sessions": 0,
            },
            "hourly_timeline": [],
            "top_sessions": [],
            "llama_server_decode": {
                "windows": [],
                "decode_obs": 0,
                "decode_median_tps": None,
                "decode_min_tps": None,
                "slow_decodes_lt_1tps": 0,
                "slow_decodes_examples": [],
                "prefill_events": 0,
                "prefill_total_tokens": 0,
                "max_prefill_tokens": 0,
            },
        }
        out = tmp_path / "out"
        jp, mp = analyzer.write_artifacts(report, out)
        assert jp.exists() and mp.exists()
        assert json.loads(jp.read_text(encoding="utf-8")) == report
        txt = mp.read_text(encoding="utf-8")
        assert "Oversized-session correlation" in txt


class TestLLamaWindows:
    def test_windows_exclude_current_and_non_llama(self, tmp_path):
        _write_llama(tmp_path, [LLAMA_EVAL_FAST], "llama-server.1.log", datetime(2026, 8, 26, 5, 0))
        _write_log(tmp_path, [ROUTING_FAST_SMALL], name="proxy.log")
        _write_log(tmp_path, [ROUTING_FAST_SMALL], name="proxy.log.2026-08-26_01")
        _write_llama(tmp_path, [LLAMA_EVAL_FAST], "llama-server.log", datetime(2026, 8, 28, 9, 0))
        names = [n for n, _ in analyzer.llama_log_windows(tmp_path)]
        assert "llama-server.1.log" in names
        assert "llama-server.log" not in names
        assert all("proxy.log" not in n for n in names)