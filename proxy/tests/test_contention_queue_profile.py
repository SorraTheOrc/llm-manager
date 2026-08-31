"""
Unit tests for the contention-queue burst profile quantifier
(proxy/benchmarks/contention_queue_profile.py, T2 of LP-0MTED3OFP006I7NO).

Covers the four AC metrics:
1. Concurrent local requests during bursts (denied active inside bursts,
   available_slots on cheap-tier snapshots).
2. Queue depth observed (status snapshots + depth-after-pop on dispatch).
3. Queue-wait durations (dispatch vs fallback-after-queue queued_duration).
4. Fallbacks while the queue was non-empty (three-way split vs nearest
   status snapshot: gt0 / eq0 / no_snapshot).

Plugin behavior (inter_arrival_gaps, detect_bursts) is tested separately,
with the burst arrival set = denied + dispatch + fallback_after_queue
(full-occupancy contention events).
"""

import gzip
import json
import tempfile
from pathlib import Path

import pytest


def _import_module():
    try:
        import proxy.benchmarks.contention_queue_profile as m
        return m
    except ImportError:
        pass
    try:
        from benchmarks import contention_queue_profile as m
        return m
    except ImportError:
        pass
    return None


M = _import_module()
pytestmark = pytest.mark.skipif(
    M is None, reason="contention_queue_profile module not importable"
)

m = M

# ---------------------------------------------------------------------------
# Realistic log line fixtures (formats mirror /var/log/llama-proxy/proxy.log*)
# ---------------------------------------------------------------------------

BASE_TS = "2026-08-23 01:18:32,000"


def tline(offset_s: float, body: str) -> str:
    """Prepend a timestamp at BASE_TS + offset_s to a log body."""
    base = 1 * 3600 + 18 * 60 + 32  # 01:18:32 within the hour
    total_ms = base * 1000 + int(offset_s * 1000)
    secs, ms = divmod(total_ms, 1000)
    h, rem = divmod(secs, 3600)
    mi, s = divmod(rem, 60)
    return f"2026-08-23 {h:02d}:{mi:02d}:{s:02d},{ms:03d} {body}"


def dispatch_line(ts: str, wait: float, session: str = "sess-1", depth: int = 0) -> str:
    return (
        f"{ts} INFO router.routing: contention_queue_dispatch "
        f"session={session} queued_duration={wait:.2f}s policy=queue depth={depth}"
    )


def fallback_line(ts: str, wait: float, session: str = "sess-1") -> str:
    return (
        f"{ts} INFO router.routing: contention_queue_fallback_after_queue "
        f"session={session} queued_duration={wait:.2f}s"
    )


def status_line(ts: str, depth: int = 0, slots: int = 2, policy: str = "queue") -> str:
    return (
        f"{ts} INFO router.routing: status_request path=/v1/chat/completions "
        f"available_slots={slots} total_slots=2 "
        f"contention_queue_policy={policy} contention_queue_depth={depth} "
        f"contention_queued_count=1 contention_fallback_after_queue_count=2"
    )


def status_line_no_depth(ts: str, slots: int = 3) -> str:
    """Other-tier status line without queue fields (e.g. embedding pool)."""
    return (
        f"{ts} INFO router.routing: status_request current_model=mxbai-embed "
        f"available_slots={slots} total_slots=3"
    )


def denied_line(ts: str, session: str, active: int) -> str:
    return f"{ts} INFO router.routing: local_dispatch_denied session={session} active={active}"


def stream_finished_line(ts: str, session: str) -> str:
    return f"{ts} INFO router.routing: Stream finished request_id=r-1 " \
           f"session_id={session} provider=local status=200"


def fallback_triggered_line(ts: str, reason: str = "local_concurrency_limit") -> str:
    return (
        f"{ts} INFO router.routing: Fallback triggered for model=local-qwen3, "
        f"from=local-qwen3, to=qwen3:32b, reason={reason}"
    )


# ---------------------------------------------------------------------------
# Raw-line parsing
# ---------------------------------------------------------------------------


class TestParseFallbackTriggered:
    def test_reason_parsed(self):
        f = m.parse_fallback_triggered(
            fallback_triggered_line(f"{BASE_TS} INFO router.routing:"))
        assert f is not None
        assert f["reason"] == "local_concurrency_limit"
        assert f["ts"] == pytest.approx(
            m.cs._parse_ts(f"{BASE_TS} INFO router.routing:"))

    def test_other_reasons_and_unrelated_lines(self):
        f = m.parse_fallback_triggered(
            fallback_triggered_line(f"{BASE_TS} INFO router.routing:",
                                    reason="model_switch_in_progress"))
        assert f is not None and f["reason"] == "model_switch_in_progress"
        assert m.parse_fallback_triggered(
            f"{BASE_TS} INFO router.routing: something else") is None


# ---------------------------------------------------------------------------
# Burst / gap / bucket helpers
# ---------------------------------------------------------------------------


class TestBurstHelpers:
    def test_gaps_use_arrival_for_queue_path_events_and_ts_for_denied(self):
        d = m.cs.LogEvent(type="dispatch", ts=20.0, wait=5.0, arrival=15.0)
        n = m.cs.LogEvent(type="denied", ts=18.0)
        f = m.cs.LogEvent(type="fallback_after_queue", ts=30.0, wait=10.0, arrival=20.0)
        gaps = m.inter_arrival_gaps([d, n, f])
        assert gaps == pytest.approx([3.0, 2.0])  # 18-15, 20-18 (not 30-20)

    def test_detect_bursts_splits_on_gap_threshold(self):
        evs = [
            m.cs.LogEvent(type="denied", ts=10.0),
            m.cs.LogEvent(type="denied", ts=12.0),
            m.cs.LogEvent(type="denied", ts=45.0),  # gap 33 > 30 -> split
            m.cs.LogEvent(type="dispatch", ts=50.0, wait=1.0, arrival=49.0),
        ]
        bursts = m.detect_bursts(evs, gap_seconds=30.0, min_arrivals=1)
        assert bursts == [(10.0, 12.0, 2), (45.0, 49.0, 2)]

    def test_burst_requires_min_arrivals(self):
        evs = [m.cs.LogEvent(type="denied", ts=10.0)]
        assert m.detect_bursts(evs, min_arrivals=2) == []

    def test_nearest_depth_bucket(self):
        snapshots = {1.0: 0, 2.0: 3, 10.0: 0}
        assert m.nearest_depth_bucket(snapshots, 2.5, delta=1.0) == "gt0"
        assert m.nearest_depth_bucket(snapshots, 1.4, delta=1.0) == "eq0"
        assert m.nearest_depth_bucket(snapshots, 8.0, delta=1.0) is None


# ---------------------------------------------------------------------------
# Profile metrics (AC1-AC4)
# ---------------------------------------------------------------------------


class TestComputeProfile:
    def test_burst_metrics_and_concurrency(self):
        """Bursts derive from the combined full-occupancy stream; denied
        active counts inside bursts; queue-path waits and depths."""
        evs = [
            # burst 1: denieds at 10, 12 (inside) and a dispatch at 15
            m.cs.LogEvent(type="denied", ts=10.0, active=2),
            m.cs.LogEvent(type="denied", ts=12.0, active=1),
            m.cs.LogEvent(type="dispatch", ts=15.0, wait=2.0, arrival=13.0,
                          depth_after=3),
            # outside any burst: denied at 60.0 (gap 45 > 30)
            m.cs.LogEvent(type="denied", ts=60.0, active=0),
        ]
        sd = {14.5: 3, 16.0: 0, 61.0: 0}
        res = m.compute_profile(evs, sd, available_slots=[2, 2, 1],
                                fallback_triggered=[], files=["f.log"])
        assert res.arrivals == 4
        assert res.bursts == [(10.0, 13.0, 3)]
        # only the two denieds inside the burst are sampled
        assert sorted(res.denied_active) == [1, 2]
        assert res.available_slots == [2, 2, 1]
        assert res.dispatch_waits == [2.0]
        assert res.dispatch_depth_after == [3]
        assert res.status_depth == [3, 0, 0]

    def test_metric4_three_way_split(self):
        """fallback_after_queue and Fallback-triggered events classified by
        nearest snapshot depth within the delta."""
        evs = [
            # fb at 12 -> nearest snapshot 13: depth 0  -> eq0
            m.cs.LogEvent(type="fallback_after_queue", ts=12.0, wait=10.0,
                          arrival=2.0),
            # fb at 30 -> nearest snapshot 29: depth 4  -> gt0
            m.cs.LogEvent(type="fallback_after_queue", ts=30.0, wait=10.0,
                          arrival=20.0),
            # fb at 50 -> no snapshot within 1s        -> no_snapshot
            m.cs.LogEvent(type="fallback_after_queue", ts=50.0, wait=10.0,
                          arrival=40.0),
        ]
        ftg = [
            {"ts": 28.8, "reason": "local_concurrency_limit"},   # gt0 (29.0:4)
            {"ts": 55.0, "reason": "local_concurrency_limit"},   # no_snapshot
            {"ts": 12.5, "reason": "model_switch_in_progress"},  # not cheap
        ]
        sd = {13.0: 0, 29.0: 4}
        res = m.compute_profile(evs, sd, [], ftg, snapshot_delta=1.0)
        assert res.fallback_after_queue_total == 3
        assert res.fallback_after_queue_split == {"gt0": 1, "eq0": 1,
                                                  "no_snapshot": 1}
        assert res.fallback_triggered_total == 2  # cheap reason only
        assert res.fallback_triggered_split == {"gt0": 1, "eq0": 0,
                                                "no_snapshot": 1}


class TestLoadProfile:
    def _write_log(self, tmp: Path, lines: list[str], name: str = "proxy.log",
                   gz: bool = False) -> Path:
        p = tmp / name if not gz else tmp / f"{name}.gz"
        if gz:
            with gzip.open(p, "wt") as fh:
                fh.write("\n".join(lines) + "\n")
        else:
            p.write_text("\n".join(lines) + "\n")
        return p

    def test_load_profile_end_to_end(self):
        lines = [
            dispatch_line(tline(5.0, ""), 2.0, depth=1),
            fallback_line(tline(8.0, ""), 3.0),
            denied_line(tline(6.0, ""), "s-1", 2),
            denied_line(tline(7.0, ""), "s-2", 1),
            status_line(tline(4.0, ""), depth=2, slots=0),
            status_line(tline(9.0, ""), depth=0, slots=2),
            status_line_no_depth(tline(9.5, ""), slots=3),  # excluded
            fallback_triggered_line(tline(10.0, ""), "local_concurrency_limit"),
            fallback_triggered_line(tline(11.0, ""), "model_switch_in_progress"),
            stream_finished_line(tline(6.5, ""), "s-1"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_log(Path(tmp), lines)
            res = m.load_profile([str(Path(tmp) / "*.log")],
                                 snapshot_delta=2.0)
        # window span from queue-path events only (dispatch at 5s .. fb at 8s)
        assert res.window_start == pytest.approx(
            m.cs._parse_ts(tline(5.0, "")))
        assert res.files == [str(p)]
        assert res.arrivals == 4  # dispatch/fb (2) + denied (2)
        assert res.bursts[0][2] == 4  # one burst containing all four events
        assert sorted(res.denied_active) == [1, 2]
        assert res.available_slots == [0, 2]  # embedding line excluded
        assert res.status_depth == [2, 0]
        assert res.dispatch_waits == [2.0]
        assert res.dispatch_depth_after == [1]
        assert res.fallback_waits == [3.0]
        assert res.fallback_triggered_cheap == [
            {"ts": m.cs._parse_ts(tline(10.0, "")),
             "reason": "local_concurrency_limit"}]
        # fb at 8s: nearest snapshot 9s depth 0 -> eq0
        assert res.fallback_after_queue_total == 1
        assert res.fallback_after_queue_split == {"gt0": 0, "eq0": 1,
                                                  "no_snapshot": 0}
        # cheap ftg at 10s: nearest snapshot 9s depth 0 -> eq0
        assert res.fallback_triggered_split == {"gt0": 0, "eq0": 1,
                                                "no_snapshot": 0}

    def test_window_filter_limits_snapshots(self):
        """Snapshots outside [start, end) are excluded from matching."""
        lines = [
            status_line(tline(1.0, ""), depth=3),
            status_line(tline(85.0, ""), depth=0),  # 01:19:57 < end
            fallback_line(tline(70.0, ""), 5.0),  # in-window; >10s from only
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_log(Path(tmp), lines)
            start = m.cs.parse_iso("2026-08-23T01:18:40")
            end = m.cs.parse_iso("2026-08-23T01:20:00")
            res = m.load_profile([str(p)], start=start, end=end,
                                 snapshot_delta=10.0)
        assert res.status_depth == [0]  # only the 85s snapshot is in-window
        assert res.fallback_after_queue_split["no_snapshot"] == 1


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------


class TestReport:
    def test_markdown_renders_all_four_sections(self):
        evs = [
            m.cs.LogEvent(type="denied", ts=10.0, active=2),
            m.cs.LogEvent(type="dispatch", ts=15.0, wait=2.0, arrival=13.0,
                          depth_after=1),
            m.cs.LogEvent(type="fallback_after_queue", ts=20.0, wait=10.0,
                          arrival=10.0),
        ]
        res = m.compute_profile(evs, {14.0: 1}, [0, 1],
                                [{"ts": 12.0, "reason": "local_concurrency_limit"}],
                                window_start=10.0, window_end=20.0,
                                files=["a.log"], snapshot_delta=5.0)
        md = m.render_markdown(res)
        for heading in ["## 1. Concurrent local requests during bursts",
                        "## 2. Queue depth observed",
                        "## 3. Queue-wait durations",
                        "## 4. Fallbacks while the queue was non-empty"]:
            assert heading in md
        # metrics embedded in tables/text
        assert "local_concurrency_limit" in md
        assert "contention_queue_fallback_after_queue" in md


class TestCli:
    def test_json_report(self, capsys):
        lines = [
            dispatch_line(tline(5.0, ""), 2.0, depth=0),
            denied_line(tline(6.0, ""), "s-1", 2),
            status_line(tline(4.0, ""), depth=1, slots=1),
            fallback_triggered_line(tline(10.0, ""), "local_concurrency_limit"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "proxy.log"
            p.write_text("\n".join(lines) + "\n")
            report_path = Path(tmp) / "report.md"
            rc = m.main(["--log-files", str(p), "--json", "--report",
                         str(report_path)])
            assert rc == 0
            assert report_path.exists()
        out = capsys.readouterr().out
        assert "report written:" in out
        report = json.loads(out[out.index('{\n  "window"'):])
        assert report["arrivals"] == 2
        assert report["bursts"]["count"] == 1
        assert report["concurrency"]["denied_active_inside_bursts"]["histogram"] == {"2": 1}
        assert report["queue_depth"]["dispatch_after_pop"]["histogram"] == {"0": 1}
        assert report["queue_wait_durations_seconds"]["dispatched"]["max"] == 2.0
