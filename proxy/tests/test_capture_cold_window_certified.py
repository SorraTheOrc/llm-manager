"""Fixture builder contract for the certified-window cold evidence.

LP-0MWGAEBWJ00334W2. ``proxy/scripts/capture_cold_window_certified.py`` turns
the certified window's logs into the replayable fixture the cold-window tests
consume. Its parsers need production logs, but :func:`build_fixture` is pure —
so the one *inferential* step (matching a timestamp-less llama-server task to
the proxy eviction that abandoned it) is pinned here with synthetic input, plus
the record-kind selection rules and the fixture's determinism.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "capture_cold_window_certified.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("capture_cold_window", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


capture = _load_script()


def _task(
    *,
    total: int,
    cache: int,
    release: int,
    done: bool = False,
    eval_ms: float | None = None,
    eval_tokens: int | None = None,
) -> dict:
    return {
        "total": total,
        "cache": cache,
        "last_progress": cache,
        "done": done,
        "release": release,
        "eval_ms": eval_ms,
        "eval_tokens": eval_tokens,
    }


def _eviction(
    *,
    total: int,
    new_tokens: int = 1_000,
    reason: str = "prefill_deadline",
    elapsed: float = 180.1,
) -> dict:
    return {
        "ts": "2026-11-07 04:59:38",
        "session": "sess-1",
        "reason": reason,
        "estimated_tokens": total,
        "new_tokens": new_tokens,
        "eta_prefill_s": 120.0,
        "elapsed_s": elapsed,
    }


def _kinds(records: list[dict], kind: str) -> list[dict]:
    return [record for record in records if record["kind"] == kind]


# ---------------------------------------------------------------------------
# The join (the only inferential step)
# ---------------------------------------------------------------------------


def test_abandoned_task_is_joined_to_the_closest_eviction_proxy_estimate():
    """The proxy's estimate is not the llama-server total, so a delta is normal."""
    tasks = {
        (1, 10): _task(total=95_337, cache=35_222, release=43_414),
    }
    fixture = capture.build_fixture([_eviction(total=95_500)], tasks)
    rows = _kinds(fixture["records"], "abandoned")
    assert len(rows) == 1
    row = rows[0]
    assert (row["total"], row["cache"], row["processed"]) == (95_337, 35_222, 43_414)
    # The dispatch's own measured elapsed, in milliseconds — the field the
    # abandonment sample prices.
    assert row["time_ms"] == pytest.approx(180_100.0)
    assert row["evidence"]["delta_tokens"] == 163
    assert row["evidence"]["source"] == "llama_server.slot_release"


def test_abandoned_task_without_a_matching_eviction_is_dropped():
    """No eviction inside tolerance ⇒ unknown elapsed ⇒ the row is not guessed at."""
    tasks = {(1, 11): _task(total=95_337, cache=0, release=43_414)}
    fixture = capture.build_fixture([_eviction(total=60_000)], tasks)
    assert _kinds(fixture["records"], "abandoned") == []


def test_measured_cache_deficit_abort_is_not_a_sample_source():
    """Only the deadline / ceiling reasons abandon a prefill for sampling."""
    tasks = {(1, 12): _task(total=95_337, cache=0, release=43_414)}
    fixture = capture.build_fixture(
        [_eviction(total=95_337, reason="measured_cache_deficit")], tasks
    )
    assert _kinds(fixture["records"], "abandoned") == []


def test_completed_tasks_only_contribute_when_genuinely_cold():
    cold = _task(total=40_000, cache=100, release=40_000, done=True, eval_ms=160_000.0, eval_tokens=39_900)
    warm = _task(total=40_000, cache=39_000, release=40_000, done=True, eval_ms=1_000.0, eval_tokens=1_000)
    fixture = capture.build_fixture([], {(1, 20): cold, (1, 21): warm})
    rows = _kinds(fixture["records"], "completed")
    assert len(rows) == 1
    assert rows[0]["total"] == 40_000
    assert rows[0]["time_ms"] == pytest.approx(160_000.0)


def test_completed_cold_task_below_the_processed_floor_is_dropped():
    small = _task(total=25_000, cache=10, release=25_000, done=True, eval_ms=100_000.0, eval_tokens=24_990)
    fixture = capture.build_fixture([], {(1, 22): small})
    assert _kinds(fixture["records"], "completed") == []


# ---------------------------------------------------------------------------
# Replay population
# ---------------------------------------------------------------------------


def test_replay_keeps_only_predicted_warm_evictions():
    evictions = [
        _eviction(total=90_000, new_tokens=819),  # reuse 0.9909 → predicted warm
        _eviction(total=60_000, new_tokens=30_000),  # reuse 0.5 → not warm
    ]
    fixture = capture.build_fixture(evictions, {})
    rows = _kinds(fixture["records"], "replay")
    assert [row["total"] for row in rows] == [90_000]
    assert rows[0]["priced_new_tokens"] == 819
    assert rows[0]["predicted_reuse_ratio"] == pytest.approx(1 - 819 / 90_000, rel=1e-4)


# ---------------------------------------------------------------------------
# Header and determinism
# ---------------------------------------------------------------------------


def test_fixture_header_records_the_window_method_and_counts():
    tasks = {
        (1, 30): _task(total=95_337, cache=0, release=43_414),
        (1, 31): _task(total=40_000, cache=0, release=40_000, done=True, eval_ms=160_000.0, eval_tokens=39_900),
    }
    fixture = capture.build_fixture([_eviction(total=95_337)], tasks)
    meta = fixture["header"]["_meta"]
    assert meta["work_item"] == "LP-0MWGAEBWJ00334W2"
    assert meta["window"] == {"start": capture.WINDOW_START, "end": capture.WINDOW_END}
    assert meta["counts"] == {
        "evictions": 1,
        "abandonable_evictions": 1,
        "abandoned": 1,
        "completed": 1,
        "replay": 1,
    }
    assert "closest prompt size" in meta["matching_note"]


def test_write_fixture_is_deterministic_and_round_trips(tmp_path: Path):
    tasks = {
        (1, 40): _task(total=95_337, cache=0, release=43_414),
        (1, 41): _task(total=80_000, cache=0, release=40_000),
    }
    evictions = [_eviction(total=95_337), _eviction(total=80_000)]
    fixture = capture.build_fixture(evictions, tasks)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    capture.write_fixture(fixture, first)
    capture.write_fixture(capture.build_fixture(evictions, tasks), second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert "_meta" in lines[0]
    # Records are grouped by kind and sorted by prompt size, so a re-run of an
    # unchanged log set produces a byte-identical fixture.
    assert [row["kind"] for row in lines[1:]] == [
        "abandoned",
        "abandoned",
        "replay",
        "replay",
    ]
    assert [row["total"] for row in lines[1:3]] == [80_000, 95_337]


def test_checked_in_fixture_carries_the_certified_window_evidence():
    """The committed fixture is the AC1/AC2 evidence — assert its shape."""
    fixture_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "cold_window_certified_2026_11_06_08.jsonl"
    )
    records = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = records[0]["_meta"]
    assert meta["window"] == {"start": "2026-11-06 14:15:50", "end": "2026-11-08 00:00:00"}
    abandoned = [row for row in records[1:] if row["kind"] == "abandoned"]
    replay = [row for row in records[1:] if row["kind"] == "replay"]
    assert meta["counts"]["abandoned"] == len(abandoned)
    assert meta["counts"]["replay"] == len(replay)
    # Bucket 2 (65k+) evidence — the population the shipped source can never see.
    assert [row for row in abandoned if row["total"] >= 65_536]
    # Every abandoned row carries the four prompt_progress fields plus provenance.
    for row in abandoned:
        assert row["processed"] <= row["total"]
        assert row["cache"] <= row["total"]
        assert row["time_ms"] > 0
        assert row["evidence"]["eviction_reason"] in capture.ABANDON_REASONS
