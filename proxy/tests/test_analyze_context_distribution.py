"""Tests for the session estimated-context distribution analysis.

F1 deliverable (LP-0MTC87GBV0031F4B): the script derives the distribution of
session estimated-context sizes (routing-time ``estimated_tokens`` from
``routing_check`` log lines) and counts sessions breaching the per-mode caps
(fast 83285 / cheap 61440) for the 2026-08-24..26 window. Evaluation only —
the script never changes proxy behavior; it reads logs and writes artifacts.

Tests use fixture log lines written to a temp log dir; they never touch live
logs. They assert observable behaviour (parsed values, aggregated stats,
breach counts, artifact output), not implementation details.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_context_distribution as analyzer

# --- Fixture log lines (real shapes from /var/log/llama-proxy) ----------

ROUTING_FAST_SMALL = (
    "2026-08-25 12:00:00,000 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=49912 cold_threshold=38000 warm_threshold=83285 new_tokens=2251 "
    "cached_ratio=0.95 messages=68 session=herdr-1787669217-851654-15207"
)
ROUTING_FAST_LARGE = (
    "2026-08-25 12:01:00,000 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=100000 cold_threshold=38000 warm_threshold=83285 new_tokens=2251 "
    "cached_ratio=0.95 messages=68 session=herdr-1787669217-851654-15207"
)
ROUTING_CHEAP = (
    "2026-08-25 12:02:00,000 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=45000 cold_threshold=38000 warm_threshold=100000 new_tokens=2251 "
    "cached_ratio=0.95 messages=68 session=herdr-1787669217-851654-15208"
)
ROUTING_UNKNOWN_THRESHOLD = (
    "2026-08-25 12:03:00,000 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=12345 cold_threshold=38000 warm_threshold=77777 new_tokens=2251 "
    "cached_ratio=0.95 messages=68 session=herdr-1787669217-851654-15209"
)
CONTEXT_PRESSURE = (
    "2026-08-26 22:01:50,864 - WARNING - context_pressure "
    "session=audit-LP-0MSC95WEI007BNX7-child_LP-0MSC95WEI007BNX7-af12a124 "
    "estimated_tokens=75204 per_slot_ctx=83285 ratio=0.90 >= 0.80; consider "
    "compacting the session history to reduce local decode cost "
    "(KV read scales with context)"
)
ROUTING_SKIP_TOO_LARGE = (
    "2026-08-26 22:02:00,000 - INFO - routing_skip_local provider=local-qwen3 "
    "model=Qwen3 estimated_tokens=424128 cold_threshold=38000 warm_threshold=83285 "
    "new_tokens=424128 cached_ratio=0.00 reason=context_too_large \u2192 skipping "
    "local, routing to next remote provider session=herdr-1787588474-220578-15272"
)
ROUTING_SKIP_BYPASS = (
    "2026-08-26 22:03:00,000 - INFO - routing_skip_local provider=local-qwen3 "
    "model=Qwen3 estimated_tokens=424128 cold_threshold=38000 warm_threshold=83285 "
    "new_tokens=424128 cached_ratio=0.00 reason=large_context_bypass \u2192 skipping "
    "local, routing to next remote provider session=herdr-1787588474-220578-15272"
)
UNRELATED_LINE = (
    "2026-08-26 22:04:00,000 - INFO - Stream finished: reason=tool_calls "
    "session=herdr-1787669217-851654-15207 provider=local model=Qwen3"
)
OUTSIDE_WINDOW = (
    "2026-08-23 23:59:59,999 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=90000 cold_threshold=38000 warm_threshold=83285 new_tokens=2251 "
    "cached_ratio=0.95 messages=68 session=herdr-1787669217-851654-15207"
)


def _write_log(tmp_path: Path, lines: list[str], name: str = "proxy.log") -> Path:
    log = tmp_path / name
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


DAY = datetime(2026, 8, 25)


class TestParseRoutingSample:
    """routing_check lines become the session context-size signal."""

    def test_parses_estimated_tokens_and_mode(self):
        s = analyzer.parse_routing_sample(ROUTING_FAST_SMALL)
        assert s is not None
        assert s.session == "herdr-1787669217-851654-15207"
        assert s.estimated_tokens == 49912
        assert s.mode == "fast"

    def test_mode_classification_by_warm_threshold(self):
        assert analyzer.parse_routing_sample(ROUTING_FAST_SMALL).mode == "fast"
        assert analyzer.parse_routing_sample(ROUTING_CHEAP).mode == "cheap"
        assert (
            analyzer.parse_routing_sample(ROUTING_UNKNOWN_THRESHOLD).mode == "other"
        )

    def test_non_routing_lines_return_none(self):
        assert analyzer.parse_routing_sample(UNRELATED_LINE) is None
        assert analyzer.parse_routing_sample("total garbage") is None


class TestParsePressureAndSkip:
    def test_parses_context_pressure(self):
        p = analyzer.parse_pressure(CONTEXT_PRESSURE)
        assert p is not None
        session, est, per_slot, ratio = p
        assert session.startswith("audit-")
        assert est == 75204
        assert per_slot == 83285
        assert ratio == 0.90

    def test_parses_skip_reason(self):
        ts, reason, session = analyzer.parse_skip(ROUTING_SKIP_TOO_LARGE)
        assert isinstance(ts, datetime)
        assert reason == "context_too_large"
        assert session == "herdr-1787588474-220578-15272"
        ts2, reason2, _ = analyzer.parse_skip(ROUTING_SKIP_BYPASS)
        assert reason2 == "large_context_bypass"

    def test_context_pressure_line_ignored_by_routing_parser(self):
        assert analyzer.parse_routing_sample(CONTEXT_PRESSURE) is None


class TestSessionAggregation:
    def test_session_mode_is_dominant(self, tmp_path):
        _write_log(tmp_path, [ROUTING_FAST_SMALL, ROUTING_FAST_LARGE])
        res = analyzer.analyze_day(tmp_path, DAY)
        agg = res.sessions["herdr-1787669217-851654-15207"]
        assert agg.count == 2
        assert agg.max_tokens == 100000
        assert agg.avg_tokens == pytest.approx((49912 + 100000) / 2)
        assert agg.mode == "fast"

    def test_window_filter_excludes_outside_lines(self, tmp_path):
        _write_log(tmp_path, [OUTSIDE_WINDOW, ROUTING_FAST_SMALL])
        res = analyzer.analyze_day(tmp_path, DAY)
        assert len(res.sessions) == 1
        assert res.sessions["herdr-1787669217-851654-15207"].count == 1

    def test_multiple_sessions_and_modes(self, tmp_path):
        _write_log(
            tmp_path, [ROUTING_FAST_SMALL, ROUTING_CHEAP, ROUTING_UNKNOWN_THRESHOLD]
        )
        res = analyzer.analyze_day(tmp_path, DAY)
        assert len(res.sessions) == 3
        modes = {agg.mode for agg in res.sessions.values()}
        assert modes == {"fast", "cheap", "other"}


class TestDistributionStats:
    def test_percentile_nearest_rank(self):
        sv = sorted([100.0, 200.0, 300.0, 400.0])
        assert analyzer._percentile(sv, 50) == 200.0
        assert analyzer._percentile(sv, 90) == 400.0

    def test_empty_values(self):
        stats = analyzer.distribution_stats([])
        assert stats["count"] == 0
        assert stats["median"] is None

    def test_known_distribution(self):
        stats = analyzer.distribution_stats([10, 20, 30, 40, 50])
        assert stats["median"] == 30
        assert stats["mean"] == 30
        assert stats["p90"] == 50
        assert stats["max"] == 50


class TestBreachCounts:
    def test_fast_and_cheap_breach_sessions(self, tmp_path):
        # One fast session at 100000 (> 83285), one cheap at 45000 (< 61440).
        _write_log(tmp_path, [ROUTING_FAST_LARGE, ROUTING_CHEAP])
        res = analyzer.analyze_day(tmp_path, DAY)
        caps = {"fast": analyzer.FAST_CAP, "cheap": analyzer.CHEAP_CAP}
        breaches = analyzer.breach_summary(res, caps)
        assert breaches["fast"]["sessions"] == 1
        assert breaches["fast"]["breach"] == 1
        assert breaches["fast"]["breach_pct"] == 100.0
        assert breaches["cheap"]["sessions"] == 1
        assert breaches["cheap"]["breach"] == 0

    def test_unknown_mode_excluded_from_buckets(self, tmp_path):
        _write_log(tmp_path, [ROUTING_UNKNOWN_THRESHOLD])
        res = analyzer.analyze_day(tmp_path, DAY)
        caps = {"fast": 83285, "cheap": 61440}
        breaches = analyzer.breach_summary(res, caps)
        assert "other" in breaches
        assert breaches["other"]["sessions"] == 1


class TestWarningsAndSkips:
    def test_pressure_and_skip_counts(self, tmp_path):
        _write_log(
            tmp_path,
            [
                CONTEXT_PRESSURE,
                ROUTING_SKIP_TOO_LARGE,
                ROUTING_SKIP_BYPASS,
                ROUTING_FAST_SMALL,
            ],
        )
        res = analyzer.analyze_day(tmp_path, DAY)
        assert res.pressure_count == 0  # pressure line is on Aug 26, out of day
        assert res.skip_counts == {}

    def test_pressure_counted_in_its_own_day(self, tmp_path):
        _write_log(tmp_path, [CONTEXT_PRESSURE])
        res = analyzer.analyze_day(tmp_path, datetime(2026, 8, 26))
        assert res.pressure_count == 1
        assert len(res.pressure_sessions) == 1

    def test_skip_reasons_tallied(self, tmp_path):
        _write_log(tmp_path, [ROUTING_SKIP_TOO_LARGE, ROUTING_SKIP_BYPASS])
        res = analyzer.analyze_day(tmp_path, datetime(2026, 8, 26))
        assert res.skip_counts["context_too_large"] == 1
        assert res.skip_counts["large_context_bypass"] == 1


class TestReportAndArtifacts:
    def test_build_report_shape(self, tmp_path):
        _write_log(
            tmp_path,
            [ROUTING_FAST_SMALL, ROUTING_FAST_LARGE, ROUTING_CHEAP,
             CONTEXT_PRESSURE],
        )
        caps = {"fast": 83285, "cheap": 61440}
        report = analyzer.build_report(
            tmp_path, [datetime(2026, 8, 25), datetime(2026, 8, 26)], caps
        )
        assert report["window"] == "2026-08-25..2026-08-26"
        assert set(report["days"]) == {"2026-08-25", "2026-08-26"}
        day25 = report["days"]["2026-08-25"]
        assert day25["sessions"] == 2
        assert day25["breaches"]["fast"]["breach"] == 1
        day26 = report["days"]["2026-08-26"]
        assert day26["context_pressure_warnings"] == 1
        # JSON-serialisable (no dates/objects).
        json.dumps(report)

    def test_writes_artifacts(self, tmp_path):
        report = {"window": "t", "caps": {}, "days": {}}
        out = tmp_path / "out"
        jp, mp = analyzer.write_artifacts(report, out)
        assert jp.exists() and mp.exists()
        assert json.loads(jp.read_text(encoding="utf-8")) == report
        assert "Session estimated-context distribution" in mp.read_text(encoding="utf-8")


class TestDiscoverLogFiles:
    def test_includes_gz_and_plain(self, tmp_path):
        _write_log(tmp_path, [ROUTING_FAST_SMALL], name="proxy.log")
        _write_log(tmp_path, [ROUTING_CHEAP], name="proxy.log.2026-08-24_01")
        gz = tmp_path / "proxy.log.2026-08-24_10.gz"
        gz.write_text(ROUTING_CHEAP + "\n", encoding="utf-8")
        names = [p.name for p in analyzer.discover_log_files(tmp_path)]
        assert "proxy.log" in names
        assert "proxy.log.2026-08-24_01" in names
        assert "proxy.log.2026-08-24_10.gz" in names

    def test_gzip_content_is_read(self, tmp_path):
        import gzip as gz_mod
        gz = tmp_path / "proxy.log.2026-08-24_10.gz"
        with gz_mod.open(gz, "wt", encoding="utf-8") as fh:
            fh.write(ROUTING_CHEAP + "\n")
        res = analyzer.analyze_day(tmp_path, DAY)
        assert any(agg.mode == "cheap" for agg in res.sessions.values())
