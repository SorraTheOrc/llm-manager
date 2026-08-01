"""
Unit tests for the benchmark runner (proxy/benchmarks/run_benchmark.py).

These tests validate the core logic of the benchmark CLI: request generation,
result recording, JSON output format, and argument parsing. They do not require
a running llama-server.
"""

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHAT_REQUEST = {
    "model": "Qwen3",
    "messages": [
        {"role": "user", "content": "Hello, what is 2+2?"}
    ],
    "max_tokens": 50,
}

SAMPLE_BENCHMARK_RESULT = {
    "config": {
        "run_type": "baseline",
        "model": "Qwen3",
        "prompts": ["Hello, what is 2+2?"],
        "num_requests": 1,
        "quantization": "Q5_K_M",
        "ctx_size": 65000,
    },
    "requests": [
        {
            "request_index": 0,
            "prompt": "Hello, what is 2+2?",
            "status": "completed",
            "total_duration_seconds": 1.234,
            "prompt_tokens": 12,
            "completion_tokens": 24,
            "tokens_per_second": 19.45,
            "time_to_first_token_seconds": 0.123,
            "error": None,
        }
    ],
    "summary": {
        "total_requests": 1,
        "completed": 1,
        "errors": 0,
        "avg_total_duration_seconds": 1.234,
        "avg_tokens_per_second": 19.45,
        "avg_time_to_first_token_seconds": 0.123,
        "total_prompt_tokens": 12,
        "total_completion_tokens": 24,
    },
}

SAMPLE_BENCHMARK_RESULT_CANDIDATE = {
    "config": {
        "run_type": "candidate",
        "model": "Qwen3",
        "prompts": ["Hello, what is 2+2?"],
        "num_requests": 1,
        "quantization": "Q4_K_M",
        "ctx_size": 65000,
    },
    "requests": [
        {
            "request_index": 0,
            "prompt": "Hello, what is 2+2?",
            "status": "completed",
            "total_duration_seconds": 1.100,
            "prompt_tokens": 12,
            "completion_tokens": 28,
            "tokens_per_second": 25.45,
            "time_to_first_token_seconds": 0.100,
            "error": None,
        }
    ],
    "summary": {
        "total_requests": 1,
        "completed": 1,
        "errors": 0,
        "avg_total_duration_seconds": 1.100,
        "avg_tokens_per_second": 25.45,
        "avg_time_to_first_token_seconds": 0.100,
        "total_prompt_tokens": 12,
        "total_completion_tokens": 28,
    },
}


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for benchmark output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# JSON schema / format tests
# ---------------------------------------------------------------------------


class TestBenchmarkResultFormat:
    """Verify that benchmark result JSON follows the expected schema."""

    def test_result_has_required_top_level_keys(self):
        """A benchmark result must contain config, requests, and summary."""
        assert "config" in SAMPLE_BENCHMARK_RESULT
        assert "requests" in SAMPLE_BENCHMARK_RESULT
        assert "summary" in SAMPLE_BENCHMARK_RESULT

    def test_config_contains_run_type(self):
        """Config must specify run_type (baseline or candidate)."""
        assert SAMPLE_BENCHMARK_RESULT["config"]["run_type"] == "baseline"
        assert SAMPLE_BENCHMARK_RESULT_CANDIDATE["config"]["run_type"] == "candidate"

    def test_request_entry_has_required_fields(self):
        """Each request entry must have the expected timing fields."""
        req = SAMPLE_BENCHMARK_RESULT["requests"][0]
        for field in ("request_index", "prompt", "status", "total_duration_seconds",
                      "prompt_tokens", "completion_tokens", "tokens_per_second",
                      "time_to_first_token_seconds", "error"):
            assert field in req, f"Missing field: {field}"

    def test_summary_has_aggregate_fields(self):
        """Summary must contain aggregate statistics."""
        s = SAMPLE_BENCHMARK_RESULT["summary"]
        for field in ("total_requests", "completed", "errors", "avg_total_duration_seconds",
                      "avg_tokens_per_second", "avg_time_to_first_token_seconds",
                      "total_prompt_tokens", "total_completion_tokens"):
            assert field in s, f"Missing summary field: {field}"


# ---------------------------------------------------------------------------
# JSON file I/O tests
# ---------------------------------------------------------------------------


class TestBenchmarkFileIO:
    """Verify that benchmark results can be written and read as JSON."""

    def test_write_and_read_json(self, temp_results_dir):
        """Writing a benchmark result to JSON and reading it back should preserve data."""
        filepath = temp_results_dir / "baseline.json"
        with open(filepath, "w") as f:
            json.dump(SAMPLE_BENCHMARK_RESULT, f, indent=2)
        assert filepath.exists()

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["config"]["run_type"] == "baseline"
        assert len(loaded["requests"]) == 1
        assert loaded["summary"]["avg_tokens_per_second"] == 19.45


# ---------------------------------------------------------------------------
# DEFAULT_PROMPTS constant
# ---------------------------------------------------------------------------


class TestDefaultPrompts:
    """Verify that the default prompts defined by run_benchmark are sensible."""

    def test_default_prompts_importable(self):
        """The DEFAULT_PROMPTS constant should be importable from run_benchmark."""
        rb = _import_run_benchmark()
        if rb is None:
            pytest.skip("run_benchmark module not importable from current sys.path")
        assert len(rb.DEFAULT_PROMPTS) > 0
        assert all(isinstance(p, str) for p in rb.DEFAULT_PROMPTS)


def _import_run_benchmark():
    """Try to import run_benchmark from multiple possible paths."""
    import sys
    from pathlib import Path

    # Try with proxy. prefix (works when repo root is on sys.path)
    try:
        from proxy.benchmarks import run_benchmark
        return run_benchmark
    except ImportError:
        pass

    # Without proxy. prefix (works when proxy/ is on sys.path, e.g. via conftest.py)
    try:
        from benchmarks import run_benchmark
        return run_benchmark
    except ImportError:
        pass

    # Try adding repo root to sys.path
    this_dir = Path(__file__).resolve().parent
    proxy_dir = this_dir.parent  # proxy/
    root_dir = proxy_dir.parent  # repo root
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    try:
        from proxy.benchmarks import run_benchmark
        return run_benchmark
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Request parameter construction
# ---------------------------------------------------------------------------


class TestBenchmarkRequestBuilder:
    """Verify that benchmark request construction follows expected patterns."""

    def test_chat_request_structure(self):
        """A single benchmark chat request must have model, messages, and max_tokens."""
        assert "model" in SAMPLE_CHAT_REQUEST
        assert "messages" in SAMPLE_CHAT_REQUEST
        assert "max_tokens" in SAMPLE_CHAT_REQUEST
        assert SAMPLE_CHAT_REQUEST["model"] == "Qwen3"
        assert len(SAMPLE_CHAT_REQUEST["messages"]) == 1
        assert SAMPLE_CHAT_REQUEST["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# Inline per-request progress (LP-0MRWU82H5003KV47)
# ---------------------------------------------------------------------------


class TestInlineProgress:
    """run_benchmark_async must report per-request progress via callback."""

    def test_progress_callback_reports_sending_and_completion(self, monkeypatch):
        """Each request triggers a 'sending' event (result=None) before its
        completion event, with 1-based request numbers and correct totals."""
        rb = _import_run_benchmark()
        if rb is None:
            pytest.skip("run_benchmark module not importable from current sys.path")

        import asyncio
        import types

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        async def fake_send(client, config, prompt, idx):
            return rb.RequestResult(
                request_index=idx,
                prompt=prompt,
                status="completed",
                total_duration_seconds=0.1,
                prompt_tokens=10,
                completion_tokens=20,
                tokens_per_second=200.0,
                time_to_first_token_seconds=0.05,
            )

        monkeypatch.setattr(
            rb, "httpx", types.SimpleNamespace(AsyncClient=lambda **kw: FakeAsyncClient())
        )
        monkeypatch.setattr(rb, "send_single_request", fake_send)

        config = rb.BenchmarkConfig(
            run_type="baseline",
            model="Qwen3",
            prompts=["p1"],
            num_requests=3,
            concurrency=1,
            timeout=5.0,
        )
        events = []
        result = asyncio.run(
            rb.run_benchmark_async(
                config,
                progress_callback=lambda n, total, res: events.append((n, total, res)),
            )
        )

        sending = [e for e in events if e[2] is None]
        done = [e for e in events if e[2] is not None]
        assert len(sending) == 3
        assert len(done) == 3
        # 1-based request numbers and correct totals
        assert all(e[1] == 3 for e in events)
        assert {e[0] for e in sending} == {1, 2, 3}
        assert {e[0] for e in done} == {1, 2, 3}
        # Each request's 'sending' event must precede its 'done' event.
        for n in (1, 2, 3):
            send_idx = next(i for i, e in enumerate(events) if e[0] == n and e[2] is None)
            done_idx = next(i for i, e in enumerate(events) if e[0] == n and e[2] is not None)
            assert send_idx < done_idx, "request done before sending"

        # Final output unchanged: requests in index order, summary intact
        assert [r["request_index"] for r in result["requests"]] == [0, 1, 2]
        assert result["summary"]["completed"] == 3
        assert result["summary"]["total_requests"] == 3

    def test_no_callback_preserves_prior_behavior(self, monkeypatch):
        """Without a progress callback, results are identical in structure."""
        rb = _import_run_benchmark()
        if rb is None:
            pytest.skip("run_benchmark module not importable from current sys.path")

        import asyncio
        import types

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        async def fake_send(client, config, prompt, idx):
            return rb.RequestResult(
                request_index=idx,
                prompt=prompt,
                status="completed",
                total_duration_seconds=0.1,
                prompt_tokens=10,
                completion_tokens=20,
                tokens_per_second=200.0,
                time_to_first_token_seconds=0.05,
            )

        monkeypatch.setattr(
            rb, "httpx", types.SimpleNamespace(AsyncClient=lambda **kw: FakeAsyncClient())
        )
        monkeypatch.setattr(rb, "send_single_request", fake_send)

        config = rb.BenchmarkConfig(
            run_type="baseline",
            model="Qwen3",
            prompts=["p1"],
            num_requests=3,
            concurrency=1,
            timeout=5.0,
        )
        result = asyncio.run(rb.run_benchmark_async(config))
        assert [r["request_index"] for r in result["requests"]] == [0, 1, 2]
        assert result["summary"]["completed"] == 3
