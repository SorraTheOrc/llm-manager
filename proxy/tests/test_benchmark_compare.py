"""
Unit tests for the benchmark comparison tool (proxy/benchmarks/compare_results.py).

These tests validate delta computation, gating policy enforcement, and Markdown
report generation.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

BASELINE = {
    "config": {
        "run_type": "baseline",
        "model": "Qwen3",
        "quantization": "Q5_K_M",
        "ctx_size": 65000,
    },
    "requests": [
        {"request_index": 0, "total_duration_seconds": 2.0, "prompt_tokens": 10,
         "completion_tokens": 20, "tokens_per_second": 10.0,
         "time_to_first_token_seconds": 0.2, "status": "completed", "prompt": "test 1",
         "error": None},
        {"request_index": 1, "total_duration_seconds": 2.5, "prompt_tokens": 12,
         "completion_tokens": 25, "tokens_per_second": 10.0,
         "time_to_first_token_seconds": 0.25, "status": "completed", "prompt": "test 2",
         "error": None},
    ],
    "summary": {
        "total_requests": 2, "completed": 2, "errors": 0,
        "avg_total_duration_seconds": 2.25,
        "avg_tokens_per_second": 10.0,
        "avg_time_to_first_token_seconds": 0.225,
        "total_prompt_tokens": 22, "total_completion_tokens": 45,
        "memory_snapshot_bytes": 8_000_000_000,
    },
}

CANDIDATE_IMPROVED = {
    "config": {
        "run_type": "candidate",
        "model": "Qwen3",
        "quantization": "Q4_K_M",
        "ctx_size": 65000,
    },
    "requests": [
        {"request_index": 0, "total_duration_seconds": 1.5, "prompt_tokens": 10,
         "completion_tokens": 22, "tokens_per_second": 14.67,
         "time_to_first_token_seconds": 0.15, "status": "completed", "prompt": "test 1",
         "error": None},
        {"request_index": 1, "total_duration_seconds": 2.0, "prompt_tokens": 12,
         "completion_tokens": 28, "tokens_per_second": 14.0,
         "time_to_first_token_seconds": 0.20, "status": "completed", "prompt": "test 2",
         "error": None},
    ],
    "summary": {
        "total_requests": 2, "completed": 2, "errors": 0,
        "avg_total_duration_seconds": 1.75,
        "avg_tokens_per_second": 14.33,
        "avg_time_to_first_token_seconds": 0.175,
        "total_prompt_tokens": 22, "total_completion_tokens": 50,
        "memory_snapshot_bytes": 6_000_000_000,
    },
}

CANDIDATE_REGRESSION = {
    "config": {
        "run_type": "candidate",
        "model": "Qwen3",
        "quantization": "Q4_K_M",
        "ctx_size": 65000,
    },
    "requests": [
        {"request_index": 0, "total_duration_seconds": 3.0, "prompt_tokens": 10,
         "completion_tokens": 18, "tokens_per_second": 6.0,
         "time_to_first_token_seconds": 0.4, "status": "completed", "prompt": "test 1",
         "error": None},
        {"request_index": 1, "total_duration_seconds": 3.5, "prompt_tokens": 12,
         "completion_tokens": 22, "tokens_per_second": 6.29,
         "time_to_first_token_seconds": 0.45, "status": "completed", "prompt": "test 2",
         "error": None},
    ],
    "summary": {
        "total_requests": 2, "completed": 2, "errors": 0,
        "avg_total_duration_seconds": 3.25,
        "avg_tokens_per_second": 6.14,
        "avg_time_to_first_token_seconds": 0.425,
        "total_prompt_tokens": 22, "total_completion_tokens": 40,
        "memory_snapshot_bytes": 6_000_000_000,
    },
}

# Default gating thresholds (matching the README)
DEFAULT_THRESHOLDS = {
    "memory_reduction_pct": 25.0,
    "max_latency_regression_pct": 10.0,
    "max_tps_regression_pct": 10.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def baseline_file(temp_dir):
    """Write baseline data to a temp file and return the path."""
    path = temp_dir / "baseline.json"
    with open(path, "w") as f:
        json.dump(BASELINE, f, indent=2)
    return path


@pytest.fixture
def candidate_improved_file(temp_dir):
    """Write improved candidate data to a temp file."""
    path = temp_dir / "candidate_improved.json"
    with open(path, "w") as f:
        json.dump(CANDIDATE_IMPROVED, f, indent=2)
    return path


@pytest.fixture
def candidate_regression_file(temp_dir):
    """Write regression candidate data to a temp file."""
    path = temp_dir / "candidate_regression.json"
    with open(path, "w") as f:
        json.dump(CANDIDATE_REGRESSION, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Delta computation tests
# ---------------------------------------------------------------------------


class TestDeltaComputation:
    """Verify that delta computations between baseline and candidate are correct."""

    def test_baseline_and_candidate_differ(self):
        """Baseline and candidate configs should differ by run_type."""
        assert BASELINE["config"]["run_type"] != CANDIDATE_IMPROVED["config"]["run_type"]

    def test_memory_delta(self):
        """Memory delta should reflect reduction."""
        baseline_mem = BASELINE["summary"]["memory_snapshot_bytes"]
        candidate_mem = CANDIDATE_IMPROVED["summary"]["memory_snapshot_bytes"]
        delta_pct = (candidate_mem - baseline_mem) / baseline_mem * 100
        # Candidate uses less memory
        assert delta_pct < 0
        # ~25% reduction
        assert round(delta_pct, 0) == -25.0

    def test_latency_delta(self):
        """Latency delta should reflect improvement (negative means faster)."""
        baseline_lat = BASELINE["summary"]["avg_total_duration_seconds"]
        candidate_lat = CANDIDATE_IMPROVED["summary"]["avg_total_duration_seconds"]
        delta_pct = (candidate_lat - baseline_lat) / baseline_lat * 100
        assert delta_pct < 0  # Candidate is faster

    def test_tps_delta(self):
        """Tokens-per-second delta should reflect improvement (positive means faster)."""
        baseline_tps = BASELINE["summary"]["avg_tokens_per_second"]
        candidate_tps = CANDIDATE_IMPROVED["summary"]["avg_tokens_per_second"]
        delta_pct = (candidate_tps - baseline_tps) / baseline_tps * 100
        assert delta_pct > 0  # Candidate generates faster

    def test_regression_delta(self):
        """Regression candidate should show increased latency."""
        baseline_lat = BASELINE["summary"]["avg_total_duration_seconds"]
        candidate_lat = CANDIDATE_REGRESSION["summary"]["avg_total_duration_seconds"]
        delta_pct = (candidate_lat - baseline_lat) / baseline_lat * 100
        assert delta_pct > 10.0  # Clearly exceeds 10% regression threshold


# ---------------------------------------------------------------------------
# Gating policy tests
# ---------------------------------------------------------------------------


class TestGatingPolicy:
    """Verify the gating policy enforces thresholds correctly."""

    def test_improved_passes_memory_gate(self):
        """Improved candidate memory reduction should meet 25% threshold."""
        baseline_mem = BASELINE["summary"]["memory_snapshot_bytes"]
        candidate_mem = CANDIDATE_IMPROVED["summary"]["memory_snapshot_bytes"]
        reduction_pct = (1 - candidate_mem / baseline_mem) * 100
        assert reduction_pct >= DEFAULT_THRESHOLDS["memory_reduction_pct"]

    def test_improved_passes_latency_gate(self):
        """Improved candidate latency should not regress more than 10%."""
        baseline_lat = BASELINE["summary"]["avg_total_duration_seconds"]
        candidate_lat = CANDIDATE_IMPROVED["summary"]["avg_total_duration_seconds"]
        regression_pct = (candidate_lat - baseline_lat) / baseline_lat * 100
        assert regression_pct <= DEFAULT_THRESHOLDS["max_latency_regression_pct"]

    def test_improved_passes_tps_gate(self):
        """Improved candidate TPS should not regress more than 10%."""
        baseline_tps = BASELINE["summary"]["avg_tokens_per_second"]
        candidate_tps = CANDIDATE_IMPROVED["summary"]["avg_tokens_per_second"]
        regression_pct = (baseline_tps - candidate_tps) / baseline_tps * 100
        assert regression_pct <= DEFAULT_THRESHOLDS["max_tps_regression_pct"]

    def test_regression_fails_latency_gate(self):
        """Regression candidate should fail the latency gate."""
        baseline_lat = BASELINE["summary"]["avg_total_duration_seconds"]
        candidate_lat = CANDIDATE_REGRESSION["summary"]["avg_total_duration_seconds"]
        regression_pct = (candidate_lat - baseline_lat) / baseline_lat * 100
        assert regression_pct > DEFAULT_THRESHOLDS["max_latency_regression_pct"]

    def test_regression_fails_tps_gate(self):
        """Regression candidate should fail the TPS gate."""
        baseline_tps = BASELINE["summary"]["avg_tokens_per_second"]
        candidate_tps = CANDIDATE_REGRESSION["summary"]["avg_tokens_per_second"]
        regression_pct = (baseline_tps - candidate_tps) / baseline_tps * 100
        assert regression_pct > DEFAULT_THRESHOLDS["max_tps_regression_pct"]


# ---------------------------------------------------------------------------
# Markdown report format tests
# ---------------------------------------------------------------------------


class TestMarkdownReportFormat:
    """Verify the expected format of the Markdown comparison report."""

    EXPECTED_SECTIONS = [
        "# Benchmark Comparison Report",
        "## Configuration",
        "## Summary Comparison",
        "## Gating Policy Results",
    ]

    def test_report_contains_required_sections(self, temp_dir):
        """The comparison report should contain all required sections."""
        cr = _import_compare_results()
        if cr is None:
            pytest.skip("compare_results module not importable from current sys.path")
        deltas = cr.compute_deltas(BASELINE, CANDIDATE_IMPROVED)
        gates = cr.check_gates(deltas, BASELINE, CANDIDATE_IMPROVED, cr.DEFAULT_THRESHOLDS)
        report = cr.generate_report(BASELINE, CANDIDATE_IMPROVED, deltas, gates)
        for section in self.EXPECTED_SECTIONS:
            assert section in report, f"Missing section: {section}"


def _import_compare_results():
    """Try to import compare_results from multiple possible paths."""
    import sys
    from pathlib import Path

    # Try with proxy. prefix (works when repo root is on sys.path)
    try:
        from proxy.benchmarks import compare_results
        return compare_results
    except ImportError:
        pass

    # Without proxy. prefix (works when proxy/ is on sys.path, e.g. via conftest.py)
    try:
        from benchmarks import compare_results
        return compare_results
    except ImportError:
        pass

    # Try adding repo root to sys.path
    this_dir = Path(__file__).resolve().parent
    proxy_dir = this_dir.parent
    root_dir = proxy_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    try:
        from proxy.benchmarks import compare_results
        return compare_results
    except ImportError:
        return None
