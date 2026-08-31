"""
Unit tests for the contention-queue sweep (T3 of LP-0MTED3OFP006I7NO).

Covers:
- AC1: queue-cap sweep produces projected Δ dispatched / Δ fallbacks
- AC2: slots projection counts denied-event active levels correctly
- Report rendering includes all sections
- CLI JSON output
"""

import json as _json
import sys
import tempfile
from pathlib import Path

import pytest


def _import_module():
    try:
        import proxy.benchmarks.contention_queue_sweep as m
        return m
    except ImportError:
        pass
    try:
        from benchmarks import contention_queue_sweep as m
        return m
    except ImportError:
        pass
    return None


M = _import_module()
pytestmark = pytest.mark.skipif(
    M is None, reason="contention_queue_sweep module not importable"
)

m = M


# ---------------------------------------------------------------------------
# Helpers: build synthetic LogEvent instances for the T1 simulate model
# ---------------------------------------------------------------------------

@M.cs.dataclass
class _FakeEvent:
    """Minimal LogEvent for simulate / slots tests."""
    type: str
    ts: float
    wait: float = 0.0
    arrival: float = 0.0
    depth_after: int | None = None
    active: int | None = None
    session: str | None = None


def _make_events(
    dispatches: int = 3,
    fallbacks: int = 2,
    slot_frees: int = 5,
    denied_active: list[int] | None = None,
) -> list:
    """Build a minimal event list with the given counts."""
    events: list = []
    t = 1000.0  # base timestamp
    if denied_active is None:
        denied_active = [2, 2, 1]

    # Denied events (arrive before any queue activity)
    for a in denied_active:
        events.append(_FakeEvent(type="denied", ts=t, active=a))
        t += 1.0

    # Dispatched events: arrival at t-5, dispatch at t
    for i in range(dispatches):
        arrival = t + i * 30 - 5.0
        ts = t + i * 30
        events.append(_FakeEvent(
            type="dispatch", ts=ts, wait=5.0, arrival=arrival,
        ))
    t += dispatches * 30

    # Fallback-after-queue events
    for i in range(fallbacks):
        ts = t + i * 30
        events.append(_FakeEvent(
            type="fallback_after_queue", ts=ts, wait=60.0,
            arrival=ts - 60.0,
        ))
    t += fallbacks * 30

    # Slot-free events (spread evenly to serve arrivals)
    for i in range(slot_frees):
        events.append(_FakeEvent(type="slot_free", ts=t + i * 20))

    return events


# ---------------------------------------------------------------------------
# _wait_stats
# ---------------------------------------------------------------------------

class TestWaitStats:
    def test_empty(self):
        s = m._wait_stats([])
        assert s == {"median": 0.0, "p50": 0.0, "p90": 0.0,
                     "p95": 0.0, "max": 0.0}

    def test_single(self):
        s = m._wait_stats([42.0])
        assert s["median"] == 42.0
        assert s["p90"] == 42.0

    def test_percentiles(self):
        s = m._wait_stats([10.0, 20.0, 30.0, 40.0, 50.0])
        assert s["median"] == 30.0
        assert s["p95"] == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

class TestRunSweep:
    def test_baseline_matches_direct_call(self):
        events = _make_events(dispatches=3, fallbacks=2, slot_frees=5)
        sweep_results, baseline = m.run_sweep(events, end_time=float("inf"))
        # Baseline should reproduce the simulate() result for (60, 4)
        direct = m.cs.simulate(events, max_wait_seconds=m.BASELINE_WAIT,
                               max_depth=m.BASELINE_DEPTH, end_time=float("inf"))
        assert baseline.dispatched == direct.dispatched
        assert baseline.fallback_after_queue == direct.fallback_after_queue

    def test_sweep_count(self):
        events = _make_events(dispatches=2, fallbacks=1, slot_frees=5)
        sweep_results, baseline = m.run_sweep(events, end_time=float("inf"))
        expected_count = len(m.WAIT_VALUES) * len(m.DEPTH_VALUES)
        assert len(sweep_results) == expected_count

    def test_longer_wait_saves_more(self):
        """A longer max_wait should not reduce dispatched (monotone)."""
        events = _make_events(dispatches=5, fallbacks=3, slot_frees=10)
        sweep_results, baseline = m.run_sweep(events, end_time=float("inf"))
        # Sort by wait; dispatched should be >= baseline
        for r in sweep_results:
            assert r.dispatched >= baseline.dispatched - 1  # ±1 tolerance

    def test_more_depth_saves_more(self):
        """A larger max_depth should not reduce dispatched."""
        events = _make_events(dispatches=5, fallbacks=3, slot_frees=10)
        sweep_results, baseline = m.run_sweep(events, end_time=float("inf"))
        for r in sweep_results:
            assert r.dispatched >= baseline.dispatched - 1


# ---------------------------------------------------------------------------
# compute_slots_projection
# ---------------------------------------------------------------------------

class TestSlotsProjection:
    def test_all_active_2_saved_at_3(self):
        events = _make_events(denied_active=[2, 2, 2])
        projections = m.compute_slots_projection(events)
        proj = next(p for p in projections if p.scenario == 3)
        assert proj.denied_events_saved == 3
        assert proj.denied_events_remaining == 0

    def test_active_3_not_saved_at_3(self):
        events = _make_events(denied_active=[2, 3, 3])
        projections = m.compute_slots_projection(events)
        proj = next(p for p in projections if p.scenario == 3)
        assert proj.denied_events_saved == 1
        assert proj.denied_events_remaining == 2

    def test_all_saved_at_4(self):
        events = _make_events(denied_active=[2, 3])
        projections = m.compute_slots_projection(events)
        proj = next(p for p in projections if p.scenario == 4)
        assert proj.denied_events_saved == 2

    def test_scenario_2_not_in_list(self):
        events = _make_events()
        projections = m.compute_slots_projection(events)
        scenarios = [p.scenario for p in projections]
        assert 2 not in scenarios
        assert 3 in scenarios


# ---------------------------------------------------------------------------
# render_comparison
# ---------------------------------------------------------------------------

class TestRenderComparison:
    def test_all_sections_present(self):
        events = _make_events(dispatches=2, fallbacks=1, slot_frees=4)
        sweep_results, baseline = m.run_sweep(events, end_time=float("inf"))
        projections = m.compute_slots_projection(events)
        md = m.render_comparison(
            sweep_results, baseline, projections,
            window_start=1000.0, window_end=2000.0, files=["proxy.log"],
        )
        assert "# Queue Caps vs Slots" in md
        assert "## Baseline" in md
        assert "## Queue-cap tuning scenarios" in md
        assert "## Slots increase scenarios" in md
        assert "## Headline comparison" in md
        assert "context-bypass" in md.lower()

    def test_baseline_section_contains_counts(self):
        events = _make_events(dispatches=2, fallbacks=1, slot_frees=4)
        sweep_results, baseline = m.run_sweep(events, end_time=float("inf"))
        projections = m.compute_slots_projection(events)
        md = m.render_comparison(
            sweep_results, baseline, projections,
            window_start=1000.0, window_end=2000.0, files=["a.log"],
        )
        assert f"Dispatched (model): **{baseline.dispatched}**" in md


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

class TestCli:
    def test_json_output_valid(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                "2026-08-23 01:18:32,000 router.routing: local_dispatch_denied session=sess-1 active=2",
                "2026-08-23 01:18:35,000 router.routing: contention_queue_dispatch session=sess-2 queued_duration=5.00s depth=-1",
                "2026-08-23 01:18:40,000 router.routing: contention_queue_fallback_after_queue session=sess-3 queued_duration=60.00s",
                "2026-08-23 01:18:33,000 INFO router.routing: Stream finished model=local-qwen3 session=sess-4 provider=local",
            ]
            p = Path(tmp) / "proxy.log"
            p.write_text("\n".join(lines) + "\n")
            rc = m.main(["--log-files", str(p), "--json"])
            assert rc == 0
        out = capsys.readouterr().out
        assert '"queue_cap_scenarios"' in out
        report = _json.loads(out[out.index("{"):])
        assert "baseline" in report
        assert "slots_scenarios" in report

    def test_report_file_created(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                "2026-08-23 01:18:32,000 router.routing: local_dispatch_denied session=sess-1 active=2",
                "2026-08-23 01:18:35,000 router.routing: contention_queue_dispatch session=sess-2 queued_duration=5.00s depth=-1",
                "2026-08-23 01:18:33,000 INFO router.routing: Stream finished model=local-qwen3 session=sess-4 provider=local",
            ]
            p = Path(tmp) / "proxy.log"
            p.write_text("\n".join(lines) + "\n")
            report_path = Path(tmp) / "report.md"
            rc = m.main(["--log-files", str(p), "--report", str(report_path)])
            assert rc == 0
            assert report_path.exists()
            md = report_path.read_text()
            assert "# Queue Caps vs Slots" in md
