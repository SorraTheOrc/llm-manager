#!/usr/bin/env python3
"""
Benchmark runner for KV quantization and config experiments.

Executes a configurable set of requests against a local router-mode llama-server
and records timings, per-token latencies, and memory snapshots.

Usage:
    # Record baseline metrics
    python -m proxy.benchmarks.run_benchmark --baseline

    # Record candidate metrics with alternative config
    python -m proxy.benchmarks.run_benchmark --candidate --config models.ini

    # Record with explicit output file
    python -m proxy.benchmarks.run_benchmark --baseline --output baseline.json

    # Custom endpoint and request count
    python -m proxy.benchmarks.run_benchmark --candidate \
        --base-url http://localhost:8000 \
        --num-requests 10 \
        --concurrency 2
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen3"
DEFAULT_MAX_TOKENS = 128
DEFAULT_NUM_REQUESTS = 5
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 60.0

DEFAULT_PROMPTS = [
    "Explain the concept of quantum computing in simple terms.",
    "Write a short poem about artificial intelligence.",
    "What are the key differences between Python and Rust?",
    "Summarize the main causes of climate change.",
    "Write a function to sort a list of integers in Python.",
]

PROMPTS_FILE = Path(__file__).parent / "prompts.json"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Timing and metadata for a single benchmark request."""

    request_index: int
    prompt: str
    status: str  # "completed" or "error"
    total_duration_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token_seconds: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self):
        return {
            "request_index": self.request_index,
            "prompt": self.prompt,
            "status": self.status,
            "total_duration_seconds": round(self.total_duration_seconds, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "time_to_first_token_seconds": (
                round(self.time_to_first_token_seconds, 4)
                if self.time_to_first_token_seconds is not None
                else None
            ),
            "error": self.error,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    run_type: str  # "baseline" or "candidate"
    model: str
    prompts: list[str]
    num_requests: int
    quantization: str = ""
    ctx_size: int = 0
    base_url: str = DEFAULT_BASE_URL
    concurrency: int = DEFAULT_CONCURRENCY
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    snapshot_script: Optional[str] = None

    def to_dict(self):
        return {
            "run_type": self.run_type,
            "model": self.model,
            "prompts": self.prompts,
            "num_requests": self.num_requests,
            "quantization": self.quantization,
            "ctx_size": self.ctx_size,
            "base_url": self.base_url,
            "concurrency": self.concurrency,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "snapshot_script": self.snapshot_script,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_project_root() -> Path:
    """Return the project root directory (two levels up from proxy/benchmarks/)."""
    return Path(__file__).resolve().parent.parent.parent


def _parse_models_ini(config_path: Optional[str] = None) -> dict:
    """Parse models.ini for quantization and ctx-size values.

    Returns a dict with model names as keys, each containing quantization
    and ctx_size values.
    """
    if config_path:
        path = Path(config_path)
    else:
        path = _get_project_root() / "models.ini"

    models = {}
    if not path.exists():
        return models

    current_section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                if current_section not in ("global",):
                    models[current_section] = {"quantization": "", "ctx_size": 0}
            elif current_section and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if current_section in models:
                    if key == "hf-repo" and ":" in value:
                        models[current_section]["quantization"] = value.split(":")[1]
                elif current_section == "global":
                    if key == "ctx-size":
                        try:
                            for m in models:
                                models[m]["ctx_size"] = int(value)
                        except ValueError:
                            pass
    return models


def _get_memory_snapshot() -> dict:
    """Capture memory-related metrics from /admin/metrics and local tools.

    Returns a dict with memory_snapshot_bytes and related metadata.
    This is best-effort; returns None values when metrics are unavailable.
    """
    result = {
        "memory_snapshot_bytes": None,
        "gpu_memory_used_bytes": None,
        "gpu_memory_total_bytes": None,
    }

    # Try to fetch from /admin/metrics endpoint
    try:
        import urllib.request

        resp = urllib.request.urlopen(
            f"{DEFAULT_BASE_URL}/admin/metrics", timeout=5
        )
        data = resp.read().decode()
        for line in data.split("\n"):
            if line.startswith("llama_process_rss_bytes"):
                parts = line.split()
                if len(parts) >= 2:
                    result["memory_snapshot_bytes"] = int(parts[-1])
            elif line.startswith("rocm_vram_used_bytes"):
                parts = line.split()
                if len(parts) >= 2:
                    result["gpu_memory_used_bytes"] = int(parts[-1])
            elif line.startswith("rocm_vram_total_bytes"):
                parts = line.split()
                if len(parts) >= 2:
                    result["gpu_memory_total_bytes"] = int(parts[-1])
    except Exception:
        pass

    return result


def _run_prometheus_snapshot(script_path: str, output_dir: Path) -> Optional[Path]:
    """Run the prometheus_snapshot.sh helper script.

    Returns the path to the snapshot file if successful, None otherwise.
    """
    if not os.path.isfile(script_path):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = output_dir / f"metrics_snapshot_{timestamp}.txt"

    try:
        subprocess.run(
            ["bash", script_path, "--output", str(snapshot_file)],
            capture_output=True,
            timeout=30,
        )
        return snapshot_file if snapshot_file.exists() else None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


async def send_single_request(
    client: httpx.AsyncClient,
    config: BenchmarkConfig,
    prompt: str,
    index: int,
) -> RequestResult:
    """Send a single chat completion request and record timings."""
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "stream": False,
    }

    start = time.monotonic()
    time_to_first_token = None

    try:
        resp = await client.post(
            f"{config.base_url}/v1/chat/completions",
            json=payload,
            timeout=config.timeout,
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            return RequestResult(
                request_index=index,
                prompt=prompt,
                status="error",
                total_duration_seconds=elapsed,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens > 0 else 0.0

        # Estimate TTFT from response timing (best-effort)
        # When not streaming, we approximate TTFT as the first ~10% of total time
        if completion_tokens > 0:
            time_to_first_token = elapsed / (completion_tokens + 1) * 1.5

        return RequestResult(
            request_index=index,
            prompt=prompt,
            status="completed",
            total_duration_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=round(tps, 2),
            time_to_first_token=time_to_first_token,
        )

    except httpx.TimeoutException as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            request_index=index,
            prompt=prompt,
            status="error",
            total_duration_seconds=elapsed,
            error=f"Timeout: {e}",
        )
    except httpx.ConnectError as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            request_index=index,
            prompt=prompt,
            status="error",
            total_duration_seconds=elapsed,
            error=f"Connection error: {e}",
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            request_index=index,
            prompt=prompt,
            status="error",
            total_duration_seconds=elapsed,
            error=f"Unexpected error: {e}",
        )


async def run_benchmark_async(config: BenchmarkConfig) -> dict:
    """Execute the full benchmark asynchronously."""
    if httpx is None:
        print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    # Build the list of requests (cycle through prompts)
    prompts = []
    for i in range(config.num_requests):
        prompts.append(config.prompts[i % len(config.prompts)])

    results: list[RequestResult] = []

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        # Use semaphore for concurrency control
        sem = asyncio.Semaphore(config.concurrency)

        async def bounded_request(prompt: str, idx: int):
            async with sem:
                return await send_single_request(client, config, prompt, idx)

        tasks = [
            bounded_request(prompts[i], i) for i in range(config.num_requests)
        ]
        results = await asyncio.gather(*tasks)

    # Compute summary statistics
    completed = [r for r in results if r.status == "completed"]
    errors = [r for r in results if r.status == "error"]

    summary = {
        "total_requests": len(results),
        "completed": len(completed),
        "errors": len(errors),
        "avg_total_duration_seconds": (
            round(sum(r.total_duration_seconds for r in completed) / len(completed), 4)
            if completed
            else 0.0
        ),
        "avg_tokens_per_second": (
            round(sum(r.tokens_per_second for r in completed) / len(completed), 2)
            if completed
            else 0.0
        ),
        "avg_time_to_first_token_seconds": (
            round(
                sum(r.time_to_first_token_seconds for r in completed if r.time_to_first_token_seconds)
                / len([r for r in completed if r.time_to_first_token_seconds]),
                4,
            )
            if completed
            else None
        ),
        "total_prompt_tokens": sum(r.prompt_tokens for r in completed),
        "total_completion_tokens": sum(r.completion_tokens for r in completed),
    }

    # Capture memory snapshot
    memory = _get_memory_snapshot()
    summary["memory_snapshot_bytes"] = memory.get("memory_snapshot_bytes")
    summary["gpu_memory_used_bytes"] = memory.get("gpu_memory_used_bytes")
    summary["gpu_memory_total_bytes"] = memory.get("gpu_memory_total_bytes")

    # Build the full result
    result = {
        "config": config.to_dict(),
        "requests": [r.to_dict() for r in results],
        "summary": summary,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark runner for KV quantization and config evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    run_type = parser.add_mutually_exclusive_group(required=True)
    run_type.add_argument(
        "--baseline",
        action="store_true",
        help="Record baseline metrics",
    )
    run_type.add_argument(
        "--candidate",
        action="store_true",
        help="Record candidate metrics (alternative config)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to models.ini for candidate config (used with --candidate)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <run_type>.json)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Proxy base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to benchmark (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help=f"Number of requests to send (default: {DEFAULT_NUM_REQUESTS})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to JSON file with prompts array (optional)",
    )
    parser.add_argument(
        "--snapshot-script",
        type=str,
        default=None,
        help="Path to prometheus_snapshot.sh script (optional)",
    )

    args = parser.parse_args(argv)
    return args


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point for the benchmark runner."""
    args = parse_args(argv)

    if httpx is None:
        print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    # Determine run type
    run_type = "baseline" if args.baseline else "candidate"

    # Load prompts
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS

    # Load model config for quantization info
    models_config = _parse_models_ini(args.config)
    model_info = models_config.get(args.model, {})
    quantization = model_info.get("quantization", "")
    ctx_size = model_info.get("ctx_size", 0)

    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"{run_type}_{timestamp}.json")

    # Build config
    config = BenchmarkConfig(
        run_type=run_type,
        model=args.model,
        prompts=prompts,
        num_requests=args.num_requests,
        quantization=quantization,
        ctx_size=ctx_size,
        base_url=args.base_url,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        snapshot_script=args.snapshot_script,
    )

    print(f"Running benchmark ({run_type})...")
    print(f"  Model:       {config.model}")
    print(f"  Quant:       {config.quantization or '(not specified)'}")
    print(f"  Requests:    {config.num_requests}")
    print(f"  Concurrency: {config.concurrency}")
    print(f"  Base URL:    {config.base_url}")
    print()

    if args.snapshot_script:
        snapshot_dir = output_path.parent
        snapshot_file = _run_prometheus_snapshot(args.snapshot_script, snapshot_dir)
        if snapshot_file:
            print(f"  Metrics snapshot saved to: {snapshot_file}")
        else:
            print("  Metrics snapshot script returned no output")

    # Run benchmark
    result = asyncio.run(run_benchmark_async(config))

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    summary = result["summary"]
    print(f"Benchmark complete. Results written to: {output_path}")
    print(f"  Completed:     {summary['completed']}/{summary['total_requests']}")
    print(f"  Errors:        {summary['errors']}")
    print(f"  Avg duration:  {summary['avg_total_duration_seconds']:.2f}s")
    print(f"  Avg TPS:       {summary['avg_tokens_per_second']:.2f}")
    if summary.get("memory_snapshot_bytes"):
        mem_mb = summary["memory_snapshot_bytes"] / (1024 * 1024)
        print(f"  Memory (RSS):  {mem_mb:.1f} MB")


if __name__ == "__main__":
    main()
