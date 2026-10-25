#!/usr/bin/env python3
"""Behavioural tests for scripts/ubatch_ab_benchmark.py (LP-0MVJXC9YZ000QMH8).

Covers the script's pure helpers so the benchmark harness (a manual CLI tool)
stays verifiable without a live llama-server endpoint:
- ``percentile`` — percentile summary semantics (sorted, interpolation-free
  nearest-rank), including the empty and degenerate inputs the benchmark's
  per-round aggregation feeds it.
- ``build_prompt`` — deterministic, salt-unique prompt body approximating the
  requested token count (the load-bearing A/B cache-defeat property).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ubatch_ab_benchmark import build_prompt, percentile


def test_percentile_midpoints_and_extremes() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    # Linear interpolation: p0 = min, p50 = 25.0, p100 = max.
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 50.0) == 25.0
    assert percentile(values, 100.0) == 40.0


def test_percentile_is_insensitive_to_input_order() -> None:
    assert percentile([40.0, 10.0, 30.0, 20.0], 50.0) == percentile(
        [10.0, 20.0, 30.0, 40.0], 50.0
    )


def test_percentile_single_value_and_empty() -> None:
    assert percentile([7.0], 90.0) == 7.0
    assert percentile([], 50.0) == 0.0


def test_build_prompt_grows_with_token_request_and_is_salt_unique() -> None:
    small = build_prompt(approx_tokens=50)
    large = build_prompt(approx_tokens=400)
    assert len(large) > len(small), "a larger token request must produce a longer prompt"
    # The cache-defeat guarantee: two builds at the same size share no line.
    assert build_prompt(approx_tokens=50, salt=1) != build_prompt(
        approx_tokens=50, salt=2
    )
    # ~4 chars/token estimate, with a floor.
    assert len(small) >= 200
    assert 0.5 <= len(large) / 400 / 4 <= 2.0


def test_build_prompt_is_deterministic_for_a_given_salt() -> None:
    assert build_prompt(approx_tokens=250, salt=7) == build_prompt(
        approx_tokens=250, salt=7
    )
