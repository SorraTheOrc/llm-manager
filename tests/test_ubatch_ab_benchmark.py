#!/usr/bin/env python3
"""Behavioural tests for scripts/ubatch_ab_benchmark.py (LP-0MVJXC9YZ000QMH8).

Covers the script's pure helpers so the benchmark harness (a manual CLI tool)
stays verifiable without a live llama-server endpoint:
- ``percentile`` — percentile summary semantics (sorted, interpolation-free
  nearest-rank), including the empty and degenerate inputs the benchmark's
  per-round aggregation feeds it.
- ``build_prompt`` — deterministic, salt-unique prompt body approximating the
  requested token count (the load-bearing A/B cache-defeat property).
- ``aggregate_round`` — the aggregate-throughput arithmetic that separates
  aggregate throughput from per-request throughput (LP-0MWCEHDRV000B7HD).
- ``run_round`` — the concurrent round runner: N requests really in flight at
  once, each carrying its window offsets for the aggregate.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ubatch_ab_benchmark import aggregate_round, build_prompt, percentile, run_round


def test_percentile_midpoints_and_extremes() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    # Linear interpolation: p0 = min, p50 = 25.0, p100 = max.
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 50.0) == 25.0
    assert percentile(values, 100.0) == 40.0


def test_percentile_is_insensitive_to_input_order() -> None:
    assert percentile([40.0, 10.0, 30.0, 20.0], 50.0) == percentile([10.0, 20.0, 30.0, 40.0], 50.0)


def test_percentile_single_value_and_empty() -> None:
    assert percentile([7.0], 90.0) == 7.0
    assert percentile([], 50.0) == 0.0


def test_build_prompt_grows_with_token_request_and_is_salt_unique() -> None:
    small = build_prompt(approx_tokens=50)
    large = build_prompt(approx_tokens=400)
    assert len(large) > len(small), "a larger token request must produce a longer prompt"
    # The cache-defeat guarantee: two builds at the same size share no line.
    assert build_prompt(approx_tokens=50, salt=1) != build_prompt(approx_tokens=50, salt=2)
    # ~4 chars/token estimate, with a floor.
    assert len(small) >= 200
    assert 0.5 <= len(large) / 400 / 4 <= 2.0


def test_build_prompt_is_deterministic_for_a_given_salt() -> None:
    assert build_prompt(approx_tokens=250, salt=7) == build_prompt(approx_tokens=250, salt=7)


# ---------------------------------------------------------------------------
# Aggregate throughput (LP-0MWCEHDRV000B7HD): a concurrent round's throughput is
# the tokens it processed divided by the wall window in which it processed them,
# NOT the sum of the per-request rates. The two disagree as soon as requests
# overlap, which is exactly the quantity the parallel-prefill hypothesis is about.
# ---------------------------------------------------------------------------


def test_aggregate_round_uses_the_overlapped_window_not_the_sum_of_rates() -> None:
    rows = [
        # Two requests started together, each 20,000 tokens in 100 s of a shared
        # 100 s window: per-request 200 tok/s each, aggregate 400 tok/s.
        {"prompt_tokens": 20_000, "started_offset_s": 0.0, "ended_offset_s": 100.0},
        {"prompt_tokens": 20_000, "started_offset_s": 0.0, "ended_offset_s": 100.0},
    ]
    agg = aggregate_round(rows)
    assert agg["prompt_tokens"] == 40_000
    assert agg["window_seconds"] == 100.0
    assert agg["tokens_per_second"] == 400.0


def test_aggregate_round_ignores_failed_requests_in_the_rate_but_keeps_the_window() -> None:
    rows = [
        {"prompt_tokens": 10_000, "started_offset_s": 0.0, "ended_offset_s": 50.0},
        # A failed request contributed no tokens but did extend the window with
        # its (empty) token count; the window is measured from all requests that
        # actually ran, so the aggregate is the good work over the elapsed time.
        {
            "prompt_tokens": 0,
            "started_offset_s": 0.0,
            "ended_offset_s": 60.0,
            "error": "timeout",
        },
    ]
    agg = aggregate_round(rows)
    assert agg["prompt_tokens"] == 10_000
    assert agg["window_seconds"] == 60.0
    assert agg["tokens_per_second"] == pytest.approx(166.7)


def test_aggregate_round_empty_round_is_zero_throughput() -> None:
    assert aggregate_round([])["tokens_per_second"] == 0.0


def test_aggregate_round_serial_round_equals_the_single_request_rate() -> None:
    rows = [
        {"prompt_tokens": 8_000, "started_offset_s": 0.0, "ended_offset_s": 32.0},
    ]
    assert aggregate_round(rows)["tokens_per_second"] == 250.0


# ---------------------------------------------------------------------------
# Concurrent round runner: the cache-defeat and measurement guards must survive
# concurrency — every request in a round gets unique text, requests really start
# together, and each row records its own window for the aggregate.
# ---------------------------------------------------------------------------


def test_run_round_serial_is_sequential() -> None:
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def fake_measure(prompt: str) -> dict:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return {"prompt_tokens": len(prompt), "prompt_per_second": 1.0}

    rows = run_round([build_prompt(64, salt=s) for s in range(3)], fake_measure, 1)
    assert peak == 1, "concurrency=1 must never have two requests in flight"
    assert len(rows) == 3
    assert all("started_offset_s" in r and "ended_offset_s" in r for r in rows)
    # Serial rows do not overlap: each starts after the previous ended.
    ordered = sorted(rows, key=lambda r: r["started_offset_s"])
    for earlier, later in zip(ordered, ordered[1:]):
        assert later["started_offset_s"] >= earlier["ended_offset_s"] - 1e-6


def test_run_round_concurrent_really_overlaps_requests() -> None:
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def fake_measure(prompt: str) -> dict:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.15)
        with lock:
            in_flight -= 1
        return {"prompt_tokens": 5_000, "prompt_per_second": 200.0}

    rows = run_round(["a", "b"], fake_measure, 2)
    assert peak == 2, "concurrency=2 must hold two requests in flight at once"
    assert len(rows) == 2
    # Overlapped window is ~one request duration, not two.
    window = max(r["ended_offset_s"] for r in rows) - min(r["started_offset_s"] for r in rows)
    assert 0.15 <= window < 0.28, f"window {window} indicates serialised requests"
    assert aggregate_round(rows)["prompt_tokens"] == 10_000


def test_run_round_isolates_a_failing_request() -> None:
    def fake_measure(prompt: str) -> dict:
        if prompt == "boom":
            raise TimeoutError("timed out")
        return {"prompt_tokens": 100, "prompt_per_second": 10.0}

    rows = run_round(["ok", "boom"], fake_measure, 2)
    by_index = {r["request_index"]: r for r in rows}
    assert by_index[0]["prompt_tokens"] == 100
    assert by_index[1]["prompt_tokens"] == 0
    assert "timed out" in by_index[1]["error"]
