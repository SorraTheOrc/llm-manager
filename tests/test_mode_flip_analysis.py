#!/usr/bin/env python3
"""Unit tests for the mode-flip classifier (LP-0MV9UNE75004903J).

The classifier quantifies fast<->cheap proxy mode flips and attributes each
flip to a trigger source (schedule vs ContextHub fast-on-command vs idle-cheap)
from the llama-proxy logs. These tests pin the parsing and classification
contract so the analysis is reproducible and the numbers in
``docs/mode-flip-analysis-LP-0MV9UNE75004903J.md`` stay traceable to code.

Parent: LP-0MV9UNE75004903J.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_mode_flips import (
    COOLDOWN_BOUNDARY_MAX_MINUTES,
    KIND_PRIORITY,
    attach_schedule_markers,
    build_restarts,
    classify_trigger,
    collect_events,
    compute_flips,
    parse_event,
    parse_timestamp,
    summarise,
)

T = "2026-10-14 14:02:00,000 - INFO - "


def test_parse_timestamp_accepts_proxy_line():
    assert parse_timestamp("2026-10-14 14:02:03,123 - INFO - x") == datetime(
        2026, 10, 14, 14, 2, 3
    )
    assert parse_timestamp("no timestamp here") is None


def test_parse_event_classifies_every_marker():
    assert parse_event(T + "Starting LLama Proxy Server") == ("start", None, None)
    assert parse_event(T + "Mode-switch restart spawned: /x/start-proxy.sh --restart") == (
        "spawn",
        None,
        None,
    )
    assert parse_event(T + "Mode scheduler: applied scheduled mode fast") == (
        "sched",
        "fast",
        None,
    )
    assert (
        parse_event(
            T + "Grandfathering: enabled; other-mode config config-fast.yaml (current=cheap)"
        )
        == ("mode", "cheap", None)
    )
    # The frequent pruning lines must not be misclassified as startup mode.
    assert parse_event(T + "Grandfathering: pruned 1 expired binding(s)") is None
    assert parse_event(T + "status_request active_query=true available_slots=0") is None


def test_collect_events_sorts_chronologically(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # Two files, out of name order: the earlier-in-time file sorts last by name.
    (log_dir / "proxy.log").write_text(
        T.replace("14:02:00", "15:00:00") + "Starting LLama Proxy Server\n",
        encoding="utf-8",
    )
    (log_dir / "proxy.log-2026-10-14_14").write_text(
        T.replace("14:02:00", "14:00:00") + "Mode scheduler: applied scheduled mode fast\n",
        encoding="utf-8",
    )
    events = collect_events(log_dir)
    assert [e[0] for e in events] == [
        datetime(2026, 10, 14, 14, 0),
        datetime(2026, 10, 14, 15, 0),
    ]
    assert events[0][1] == "sched"


def test_build_restarts_attaches_startup_mode_within_window():
    events = [
        (datetime(2026, 10, 14, 12, 50, 0), "start", None, None),
        (datetime(2026, 10, 14, 12, 50, 1), "mode", "fast", None),  # 1 s later -> attached
        (datetime(2026, 10, 14, 14, 0, 0), "start", None, None),
        (datetime(2026, 10, 14, 14, 0, 40), "mode", "cheap", None),  # 40 s later -> NOT attached
    ]
    restarts = build_restarts(events)
    assert restarts[0]["mode"] == "fast"
    assert restarts[1]["mode"] is None  # outside the 30 s window


def test_attach_schedule_markers_one_to_one_nearest():
    restarts = [
        {"ts": datetime(2026, 10, 14, 10, 0, 30), "mode": "fast", "sched_marker": None},
        {"ts": datetime(2026, 10, 14, 10, 30, 0), "mode": "cheap", "sched_marker": None},
    ]
    scheds = [(datetime(2026, 10, 14, 10, 0, 25), "fast")]
    attach_schedule_markers(restarts, scheds)
    assert restarts[0]["sched_marker"] == "fast"
    assert restarts[1]["sched_marker"] is None  # marker consumed, not reused


def test_compute_flips_only_on_mode_change():
    restarts = [
        {"ts": datetime(2026, 10, 14, 10, 0), "mode": "fast", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 10, 30), "mode": "fast", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 11, 0), "mode": "cheap", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 11, 2), "mode": "fast", "sched_marker": None, "source": None},
    ]
    flips = compute_flips(restarts)
    assert len(flips) == 2
    assert flips[0]["to"] == "cheap" and flips[0]["prev_from"] == "fast"
    assert flips[1]["to"] == "fast"


def test_classify_trigger_schedule_vs_direction():
    assert classify_trigger({"to": "fast", "sched_marker": "fast"}) == "schedule"
    assert classify_trigger({"to": "cheap", "sched_marker": None}) == "to-cheap"
    assert classify_trigger({"to": "fast", "sched_marker": None}) == "to-fast"


def test_summarise_per_day_trigger_totals_and_window():
    restarts = [
        {"ts": datetime(2026, 10, 14, 9, 0), "mode": "cheap", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 10, 0), "mode": "fast", "sched_marker": "fast", "source": None},
        {"ts": datetime(2026, 10, 14, 10, 35), "mode": "cheap", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 10, 36), "mode": "fast", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 12, 0), "mode": "cheap", "sched_marker": None, "source": None},
    ]
    flips = compute_flips(restarts)
    summary = summarise(restarts, flips)
    assert summary["flips"] == 4
    assert summary["trigger_totals"] == {"schedule": 1, "to-fast": 1, "to-cheap": 2}
    assert summary["per_day"]["2026-10-14"]["to-cheap"] == 2


def test_summarise_studied_window_restricts_counts():
    restarts = [
        {"ts": datetime(2026, 10, 14, 13, 0), "mode": "cheap", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 14, 30), "mode": "fast", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 15, 0), "mode": "cheap", "sched_marker": None, "source": None},
    ]
    flips = compute_flips(restarts)
    summary = summarise(
        restarts,
        flips,
        window=(datetime(2026, 10, 14, 14, 2), datetime(2026, 10, 14, 15, 30)),
    )
    assert summary["flips"] == 2
    assert summary["studied_window"]["flips"] == 2  # both flips inside
    summary2 = summarise(
        restarts,
        flips,
        window=(datetime(2026, 10, 14, 14, 2), datetime(2026, 10, 14, 14, 45)),
    )
    assert summary2["studied_window"]["flips"] == 1  # only the 14:30 flip


def test_summarise_counts_cooldown_boundary_and_wasted_dwells():
    # A cheap->fast flip, then 29.5 min later a fast->cheap flip (boundary
    # band), and a sub-5-min cheap dwell right after.
    restarts = [
        {"ts": datetime(2026, 10, 14, 9, 30), "mode": "cheap", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 10, 0), "mode": "fast", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 10, 29, 30), "mode": "cheap", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 10, 30, 0), "mode": "fast", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 11, 30), "mode": "cheap", "sched_marker": None, "source": None},
    ]
    flips = compute_flips(restarts)
    summary = summarise(restarts, flips)
    assert summary["cooldown_breaches"] == []  # 29.5 min is not a breach
    assert summary["cooldown_boundary_hits"] == 1
    assert summary["cheap_dwells"]["wasted_under_5m"] == 1  # 30 s dwell


def test_classify_trigger_source_scheduler():
    """source=scheduler always maps to schedule regardless of direction."""
    assert classify_trigger({"to": "fast", "source": "scheduler"}) == "schedule"
    assert classify_trigger({"to": "cheap", "source": "scheduler"}) == "schedule"


def test_classify_trigger_source_worker():
    """source=worker maps to to-fast/to-cheap by direction."""
    assert classify_trigger({"to": "fast", "source": "worker"}) == "to-fast"
    assert classify_trigger({"to": "cheap", "source": "worker"}) == "to-cheap"


def test_classify_trigger_source_api():
    """source=api maps to to-fast/to-cheap by direction."""
    assert classify_trigger({"to": "fast", "source": "api"}) == "to-fast"
    assert classify_trigger({"to": "cheap", "source": "api"}) == "to-cheap"


def test_classify_trigger_source_manual():
    """source=manual maps to to-fast/to-cheap by direction."""
    assert classify_trigger({"to": "fast", "source": "manual"}) == "to-fast"
    assert classify_trigger({"to": "cheap", "source": "manual"}) == "to-cheap"


def test_classify_trigger_unknown_source_falls_back_to_heuristic():
    """An unrecognised ``source`` value is not a canonical attribution label, so
    classification degrades to the heuristic instead of guessing a category.

    Edge case pinned by LP-0MVC145DE006SAY5: only ``scheduler``, ``worker``,
    ``api`` and ``manual`` are deterministic inputs; anything else (a future or
    mistyped token) must behave exactly like an absent source.
    """
    # Unknown source without a schedule marker -> direction-based classification.
    assert classify_trigger({"to": "fast", "source": "mystery"}) == "to-fast"
    assert classify_trigger({"to": "cheap", "source": "mystery"}) == "to-cheap"
    # Unknown source with a schedule marker -> the heuristic marker still wins.
    assert (
        classify_trigger({"to": "fast", "source": "mystery", "sched_marker": "fast"})
        == "schedule"
    )


def test_classify_trigger_source_absent_falls_back_to_heuristic():
    """When source is absent, fall back to direction-based heuristic."""
    assert classify_trigger({"to": "fast", "source": None}) == "to-fast"
    assert classify_trigger({"to": "cheap", "source": None}) == "to-cheap"
    assert classify_trigger({"to": "fast", "source": None, "sched_marker": "fast"}) == "schedule"


def test_classify_trigger_source_overrides_heuristic():
    """Source attribution takes precedence over sched_marker heuristic."""
    # source=worker should classify as to-fast even when sched_marker would say schedule
    assert classify_trigger({"to": "fast", "source": "worker", "sched_marker": "fast"}) == "to-fast"
    # source=api should classify as to-cheap even when sched_marker would say schedule
    assert classify_trigger({"to": "cheap", "source": "api", "sched_marker": "fast"}) == "to-cheap"


def test_build_restarts_captures_source_from_mode_event():
    """Source attribution on the mode event flows into the restart record."""
    events = [
        (datetime(2026, 10, 14, 12, 50, 0), "start", None, None),
        (datetime(2026, 10, 14, 12, 50, 1), "mode", "fast", "worker"),
        (datetime(2026, 10, 14, 14, 0, 0), "start", None, None),
        (datetime(2026, 10, 14, 14, 0, 1), "mode", "cheap", "scheduler"),
    ]
    restarts = build_restarts(events)
    assert restarts[0]["source"] == "worker"
    assert restarts[1]["source"] == "scheduler"


def test_parse_event_extracts_source_marker():
    """parse_event returns source= value when present on spawn/sched/mode lines."""
    # Spawn with source
    result = parse_event(
        T + 'Mode-switch restart spawned: /x/start-proxy.sh --restart source=worker actor=mode-switch-worker'
    )
    assert result is not None
    assert result[0] == "spawn"
    assert result[2] == "worker"

    # Sched with source
    result = parse_event(T + "Mode scheduler: applied scheduled mode fast source=scheduler actor=mode-scheduler")
    assert result is not None
    assert result[0] == "sched"
    assert result[1] == "fast"
    assert result[2] == "scheduler"

    # Mode with source
    result = parse_event(
        T + "Grandfathering: enabled; other-mode config config-fast.yaml (current=cheap) source=worker actor=mode-switch-worker reason=idle"
    )
    assert result is not None
    assert result[0] == "mode"
    assert result[1] == "cheap"
    assert result[2] == "worker"

    # No source marker -> None
    result = parse_event(T + "Mode-switch restart spawned: /x/start-proxy.sh --restart")
    assert result is not None
    assert result[2] is None


def test_attach_schedule_markers_sets_source_scheduler():
    """When a schedule marker's mode is 'scheduler', set source=scheduler on the restart."""
    restarts = [
        {"ts": datetime(2026, 10, 14, 10, 0, 30), "mode": "fast", "sched_marker": None, "source": None},
        {"ts": datetime(2026, 10, 14, 10, 30, 0), "mode": "cheap", "sched_marker": None, "source": "worker"},
    ]
    scheds = [(datetime(2026, 10, 14, 10, 0, 25), "scheduler")]
    attach_schedule_markers(restarts, scheds)
    assert restarts[0]["sched_marker"] == "scheduler"
    assert restarts[0]["source"] == "scheduler"
    # The second restart already has source=worker, should NOT be overwritten
    assert restarts[1]["source"] == "worker"


def test_compute_flips_includes_source():
    """Flips carry the source field from the *current* (target) restart."""
    restarts = [
        {"ts": datetime(2026, 10, 14, 10, 0), "mode": "fast", "sched_marker": None, "source": "worker"},
        {"ts": datetime(2026, 10, 14, 10, 30), "mode": "cheap", "sched_marker": None, "source": "scheduler"},
    ]
    flips = compute_flips(restarts)
    assert len(flips) == 1
    # source comes from the current (target) restart, not the previous one
    assert flips[0]["source"] == "scheduler"


def test_kind_priority_orders_spawn_before_start_before_mode():
    assert KIND_PRIORITY["spawn"] < KIND_PRIORITY["start"] < KIND_PRIORITY["mode"]
