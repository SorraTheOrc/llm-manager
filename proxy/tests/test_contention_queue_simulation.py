"""
Unit tests for the contention-queue simulation harness
(proxy/benchmarks/contention_queue_simulation.py, T1 of LP-0MTED3OFP006I7NO).

Covers:
- AC1: event parsing / coverage across the four queue event types
  (dispatch, fallback_after_queue, status_request queue snapshot,
  local_dispatch_denied) plus the slot-free wake source.
- AC2: default-caps replay reproduces observed dispatch/fallback counts for a
  synthetic cheap-mode window within tolerance.
- The FIFO + deadline queue model (timeout fallback, depth-cap fallback,
  same-instant race dispatch, FIFO ordering, end-of-window flush).
"""

import gzip
import json
import sys
import tempfile
from pathlib import Path

import pytest


def _import_module():
    try:
        import proxy.benchmarks.contention_queue_simulation as m
        return m
    except ImportError:
        pass
    try:
        from benchmarks import contention_queue_simulation as m
        return m
    except ImportError:
        pass
    return None


M = _import_module()
pytestmark = pytest.mark.skipif(
    M is None, reason="contention_queue_simulation module not importable"
)

m = M

# ---------------------------------------------------------------------------
# Realistic log line fixtures (format mirrors /var/log/llama-proxy/proxy.log*)
# ---------------------------------------------------------------------------

TS = "%(ts)s INFO router.routing:"


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


def status_line(ts: str, depth: int = 0, policy: str = "queue") -> str:
    return (
        f"{ts} INFO router.routing: status_request path=/v1/chat/completions "
        f"contention_queue_policy={policy} contention_queue_depth={depth} "
        f"contention_queued_count=1 contention_fallback_after_queue_count=2"
    )


def denied_line(ts: str, session: str, active: int) -> str:
    return f"{ts} INFO router.routing: local_dispatch_denied session={session} active={active}"


def stream_finished_line(ts: str, session: str, provider: str = "local") -> str:
    return (
        f"{ts} INFO router.routing: Stream finished request_id=r-1 "
        f"session_id={session} provider={provider} status=200"
    )


BASE_TS = "2026-08-23 01:18:32,000"


def tline(offset_s: float, body: str) -> str:
    """Prepend a timestamp at BASE_TS + offset_s to a log body."""
    base = 1 * 3600 + 18 * 60 + 32  # 01:18:32 within the hour
    total_ms = base * 1000 + int(offset_s * 1000)
    secs, ms = divmod(total_ms, 1000)
    h, rem = divmod(secs, 3600)
    mi, s = divmod(rem, 60)
    return f"2026-08-23 {h:02d}:{mi:02d}:{s:02d},{ms:03d} {body}"


# ---------------------------------------------------------------------------
# Parsing (AC1)
# ---------------------------------------------------------------------------


class TestParseLine:
    def test_dispatch_event(self):
        ev = m.parse_line(dispatch_line(f"{BASE_TS} INFO router.routing:", 15.84, depth=1))
        assert ev is not None
        assert ev.type == "dispatch"
        assert ev.wait == pytest.approx(15.84)
        assert ev.arrival == pytest.approx(ev.ts - 15.84)
        assert ev.depth_after == 1
        assert ev.session == "sess-1"

    def test_fallback_after_queue_event(self):
        ev = m.parse_line(fallback_line(f"{BASE_TS} INFO router.routing:", 60.00))
        assert ev is not None
        assert ev.type == "fallback_after_queue"
        assert ev.wait == pytest.approx(60.00)
        assert ev.arrival == pytest.approx(ev.ts - 60.00)

    def test_status_queue_snapshot_only_when_policy_queue(self):
        ev = m.parse_line(
            status_line(f"{BASE_TS} INFO router.routing:", depth=3, policy="queue")
        )
        assert ev is not None and ev.type == "status"
        # Non-queue policy status lines (e.g. fast mode) are not queue events.
        assert m.parse_line(status_line(f"{BASE_TS} INFO router.routing:", policy="fallback")) is None

    def test_local_dispatch_denied_event(self):
        ev = m.parse_line(denied_line(f"{BASE_TS} INFO router.routing:", "sess-x", 2))
        assert ev is not None
        assert ev.type == "denied"
        assert ev.active == 2
        assert ev.session == "sess-x"

    def test_slot_free_only_for_local_streams(self):
        ev = m.parse_line(stream_finished_line(f"{BASE_TS} INFO router.routing:", "sess-1"))
        assert ev is not None and ev.type == "slot_free"
        # Remote streams do not free a local slot.
        assert m.parse_line(
            stream_finished_line(f"{BASE_TS} INFO router.routing:", "sess-1", provider="remote")
        ) is None

    def test_unrelated_lines_ignored(self):
        assert m.parse_line(f"{BASE_TS} INFO router.routing: something_else x=1") is None
        # Status snapshot field name must not be confused with the event marker.
        assert m.parse_line(
            f"{BASE_TS} INFO router.routing: status_request contention_fallback_after_queue_count=2"
        ) is None


class TestCoverage:
    def test_full_coverage_on_synthetic_log(self):
        lines = [
            dispatch_line(tline(1.0, ""), 0.5),
            fallback_line(tline(2.0, ""), 60.0),
            status_line(tline(3.0, ""), depth=1),
            denied_line(tline(4.0, ""), "sess-1", 2),
            stream_finished_line(tline(5.0, ""), "sess-1"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "proxy.log"
            p.write_text("\n".join(lines) + "\n")
            loaded = m.load_events([str(p)])
        assert loaded.coverage == {
            "dispatch": {"parsed": 1, "candidate": 1},
            "fallback_after_queue": {"parsed": 1, "candidate": 1},
            "status": {"parsed": 1, "candidate": 1},
            "denied": {"parsed": 1, "candidate": 1},
            "slot_free": {"parsed": 1, "candidate": 1},
        }
        assert {e.type for e in loaded.events} == {
            "dispatch", "fallback_after_queue", "status", "denied", "slot_free",
        }

    def test_gzip_and_plain_files_both_loaded(self):
        lines = [dispatch_line(tline(1.0, ""), 0.5), denied_line(tline(2.0, ""), "s", 1)]
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "a.log"
            plain.write_text(lines[0] + "\n")
            gz = Path(tmp) / "b.log.gz"
            with gzip.open(gz, "wt") as fh:
                fh.write(lines[1] + "\n")
            loaded = m.load_events([str(Path(tmp) / "*.log*")])
        assert len(loaded.events) == 2
        assert loaded.files == sorted([str(plain), str(gz)])

    def test_window_filter(self):
        lines = [
            dispatch_line(tline(1.0, ""), 0.5),
            dispatch_line(tline(20.0, ""), 0.5),
            dispatch_line(tline(40.0, ""), 0.5),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "proxy.log"
            p.write_text("\n".join(lines) + "\n")
            start = m.parse_iso("2026-08-23T01:18:35")  # excludes the first event
            end = m.parse_iso("2026-08-23T01:19:10")  # excludes the last
            loaded = m.load_events([str(p)], start=start, end=end)
        assert len(loaded.events) == 1  # only tline(20.0) falls inside
        # Coverage still counts the full file (window filters events, not files).
        assert loaded.coverage["dispatch"] == {"parsed": 3, "candidate": 3}

    def test_missing_files_raise(self):
        with pytest.raises(FileNotFoundError):
            m.load_events(["/nonexistent/proxy.log*"])


# ---------------------------------------------------------------------------
# Queue model
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_same_instant_race_dispatch(self):
        """An arrival at the same instant a local stream ends dispatches with
        the observed queued_duration=0.00s race."""
        arrival = m.LogEvent(type="dispatch", ts=10.0, wait=0.0, arrival=10.0)
        free = m.LogEvent(type="slot_free", ts=10.0)
        result = m.simulate([arrival, free], max_wait_seconds=60.0, max_depth=4)
        assert result.dispatched == 1
        assert result.fallback_after_queue == 0
        assert result.waits == [0.0]

    def test_timeout_fallback_at_deadline(self):
        """A waiter never served falls back at exactly arrival+max_wait."""
        arrival = m.LogEvent(type="dispatch", ts=70.0, wait=60.0, arrival=10.0)
        result = m.simulate([arrival], max_wait_seconds=60.0)
        assert result.dispatched == 0
        assert result.fallback_after_queue == 1
        assert result.timeout_fallbacks == 1
        assert result.waits == [60.0]

    def test_fifo_ordering(self):
        """The FIFO head is served before later arrivals."""
        a1 = m.LogEvent(type="dispatch", ts=25.0, wait=15.0, arrival=10.0)
        a2 = m.LogEvent(type="dispatch", ts=40.0, wait=20.0, arrival=20.0)
        f1 = m.LogEvent(type="slot_free", ts=25.0)
        f2 = m.LogEvent(type="slot_free", ts=35.0)
        result = m.simulate([a1, a2, f1, f2], max_wait_seconds=60.0)
        assert result.dispatched == 2
        assert result.fallback_after_queue == 0
        assert result.waits == [15.0, 15.0]  # a1 at 25 (wait 15), a2 at 35 (wait 15)

    def test_depth_cap_immediate_fallback(self):
        """An arrival when the queue is at max_depth falls back immediately."""
        a1 = m.LogEvent(type="dispatch", ts=50.0, wait=40.0, arrival=10.0)
        a2 = m.LogEvent(type="dispatch", ts=50.0, wait=30.0, arrival=20.0)
        a3 = m.LogEvent(type="dispatch", ts=50.0, wait=20.0, arrival=30.0)
        f = m.LogEvent(type="slot_free", ts=50.0)
        result = m.simulate([a1, a2, a3, f], max_wait_seconds=60.0, max_depth=1)
        assert result.dispatched == 1
        assert result.fallback_after_queue == 2
        assert result.depth_capped_fallbacks == 2
        assert result.waits == [0.0, 0.0, 40.0]
        assert result.max_queue_depth == 1

    def test_end_of_window_flush_counts_served_waiter(self):
        """Waiters still queued at window end fall back at their deadline."""
        a = m.LogEvent(type="dispatch", ts=70.0, wait=60.0, arrival=10.0)
        f = m.LogEvent(type="slot_free", ts=65.0)
        result = m.simulate([a, f], max_wait_seconds=60.0, end_time=100.0)
        # Slot frees at 65; the waiter's deadline (70) has not passed -> dispatched.
        assert result.dispatched == 1
        assert result.fallback_after_queue == 0
        assert result.waits == [55.0]

    def test_wait_stats_percentiles(self):
        a1 = m.LogEvent(type="dispatch", ts=25.0, wait=15.0, arrival=10.0)
        a2 = m.LogEvent(type="dispatch", ts=35.0, wait=20.0, arrival=15.0)
        f1 = m.LogEvent(type="slot_free", ts=25.0)
        f2 = m.LogEvent(type="slot_free", ts=35.0)
        result = m.simulate([a1, a2, f1, f2], max_wait_seconds=60.0)
        stats = result.wait_stats()
        assert result.waits == [15.0, 20.0]  # a1 serves at 25, a2 at 35
        assert stats["median"] == pytest.approx(17.5)
        assert stats["p50"] == pytest.approx(17.5)
        assert stats["p90"] == pytest.approx(19.5)

    def test_empty_input(self):
        result = m.simulate([])
        assert result.dispatched == 0
        assert result.fallback_after_queue == 0
        assert result.wait_stats()["median"] == 0.0


# ---------------------------------------------------------------------------
# Replication validation (AC2)
# ---------------------------------------------------------------------------


class TestReplication:
    def test_synthetic_window_replicated_within_tolerance(self):
        """Build a synthetic cheap-mode window with known observed counts and
        verify the default-caps replay reproduces them (d=2, f=1)."""
        lines = [
            # arrival at 95s, dispatched at 100s (wait 5s)
            dispatch_line(tline(100.0, ""), 5.0, depth=0),
            # arrival at 100s, dispatched at 105s (wait 5s)
            dispatch_line(tline(105.0, ""), 5.0, depth=0),
            # arrival at 100s, falls back at 160s (wait 60s)
            fallback_line(tline(160.0, ""), 60.0),
            # two local stream ends freeing slots
            stream_finished_line(tline(98.0, ""), "s-a"),
            stream_finished_line(tline(104.0, ""), "s-b"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "proxy.log"
            p.write_text("\n".join(lines) + "\n")
            loaded = m.load_events([str(p)])

        observed = m.observed_counts(loaded.events)
        assert observed == {"dispatched": 2, "fallback_after_queue": 1}

        validation = m.replication_validation(loaded.events, tolerance_pct=10.0)
        assert validation["within_tolerance"] is True
        assert validation["simulated"]["dispatched"] == 2
        assert validation["simulated"]["fallback_after_queue"] == 1
        assert validation["deviation_pct"] == {"dispatched": 0.0, "fallback_after_queue": 0.0}

    def test_validation_fails_outside_tolerance(self):
        """A model that diverges from observation must be reported as out of
        tolerance (guards the gate from silently passing)."""
        a = m.LogEvent(type="dispatch", ts=60.0, wait=60.0, arrival=0.0)
        f = m.LogEvent(type="slot_free", ts=10.0)
        validation = m.replication_validation([a, f], tolerance_pct=10.0)
        # observed d=1 f=0; sim: arrival queues, slot_free dispatches it -> d=1 f=0
        assert validation["within_tolerance"] is True
        # Now diverge: a single observed fallback that the sim dispatches.
        fb = m.LogEvent(type="fallback_after_queue", ts=60.0, wait=60.0, arrival=0.0)
        validation = m.replication_validation([fb, f], tolerance_pct=10.0)
        # observed d=0 f=1; sim dispatches the waiter (slot free at 10) -> f is
        # -100% off, well outside tolerance.
        assert validation["within_tolerance"] is False
        assert validation["observed"] == {"dispatched": 0, "fallback_after_queue": 1}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_json_report(self, capsys):
        lines = [
            dispatch_line(tline(100.0, ""), 5.0),
            fallback_line(tline(160.0, ""), 60.0),
            stream_finished_line(tline(98.0, ""), "s-a"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "proxy.log"
            p.write_text("\n".join(lines) + "\n")
            rc = m.main(["--log-files", str(p), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        report = json.loads(out[out.index("{"):])
        assert report["coverage"]["dispatch"]["coverage_pct"] == 100.0
        assert report["coverage_gate_pct"] == 95.0
        assert report["simulation"]["dispatched"] == 1
        assert report["simulation"]["fallback_after_queue"] == 1
        assert report["validation"]["within_tolerance"] is True
        assert report["window"]["files"] == [str(p)]

    def test_missing_logs_exit_2(self, capsys):
        rc = m.main(["--log-files", "/nonexistent/*.gz", "--json"])
        assert rc == 2
        assert "ERROR" in capsys.readouterr().err
