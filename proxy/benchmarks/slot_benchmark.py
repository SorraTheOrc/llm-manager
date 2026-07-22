#!/usr/bin/env python3
"""
Slot-count benchmarking for local Qwen3.

Varies ``session_slot_pool_size`` (and thus llama-server ``--parallel``) and
measures throughput, latency, and GPU memory for long-running queries with
~16k token system prompts that take ~10 minutes to process.

Usage:
    # Run all slot counts (1,2,4,6,8,12) sequentially
    sudo ./slot_benchmark.py --all

    # Run a specific slot count only
    ./slot_benchmark.py --slots 6

    # Dry run — just validate prompt length, don't send requests
    ./slot_benchmark.py --slots 6 --dry-run

    # Resume from a specific slot count (skip earlier ones)
    ./slot_benchmark.py --all --resume 6

    # Custom output directory
    ./slot_benchmark.py --all --output-dir ./benchmark-results
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # /home/rgardler/projects/llm
CONFIG_YAML = PROJECT_ROOT / "proxy" / "config.yaml"
SLOT_CACHE_DIR = Path("/home/rgardler/projects/llm/slot-cache")
LLAMA_SERVER_PORT = 8080
PROXY_PORT = 8000

# ---------------------------------------------------------------------------
# Target prompt / output characteristics
# ---------------------------------------------------------------------------
TARGET_PROMPT_TOKENS = 16_000      # ~16k token system prompt
TARGET_RESPONSE_TIME_S = 600       # ~10 minutes per query
TARGET_OUTPUT_TOKENS = 18_000      # ~30 tok/s × 600s
TOTAL_REQUESTS_PER_RUN = 6         # 6 concurrent requests per slot-count run
WARMUP_REQUESTS = 1                # 1 warmup request before measurements
INTER_REQUEST_GAP_S = 90.0          # 1.5 min stagger between request starts to simulate realistic workloads

# ---------------------------------------------------------------------------
# Slot counts to test
# ---------------------------------------------------------------------------
SLOT_COUNTS = [1, 2, 4, 6, 8, 12]

# ---------------------------------------------------------------------------
# Long padding document (~500 words, repeated to hit token target)
# ---------------------------------------------------------------------------
_PROMPT_PADDING = """
Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore
eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt
in culpa qui officia deserunt mollit anim id est laborum Sed ut perspiciatis
unde omnis iste natus error sit voluptatem accusantium doloremque laudantium
totam rem aperiam eaque ipsa quae ab illo inventore veritatis et quasi
architecto beatae vitae dicta sunt explicabo Nemo enim ipsam voluptatem quia
voluptas sit aspernatur aut odit aut fugit sed quia consequuntur magni dolores
eos qui ratione voluptatem sequi nesciunt Neque porro quisquam est qui dolorem
ipsum quia dolor sit amet consectetur adipisci velit sed quia non numquam eius
modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem
Ut enim ad minima veniam quis nostrum exercitationem ullam corporis suscipit
laboriosam nisi ut aliquid ex ea commodi consequatur Quis autem vel eum iure
reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur
vel illum qui dolorem eum fugiat quo voluptas nulla pariatur
"""


def build_system_prompt(target_tokens: int = TARGET_PROMPT_TOKENS) -> str:
    """Build a system prompt containing approximately ``target_tokens`` tokens.

    Uses a long repetitive document as padding so the model's attention pattern
    resembles a real large-context scenario. Rough heuristic: 1 token ≈ 0.75 words
    for English prose, so we repeat the padding to reach target word count.
    """
    # Rough: 1 token ≈ 0.75 words for English
    target_words = int(target_tokens * 0.75)
    words_per_block = len(_PROMPT_PADDING.split())
    repeats = max(1, target_words // words_per_block)

    instruction = (
        "You are analyzing a long financial report. Carefully read the entire document "
        "and provide a detailed, step-by-step analysis covering: "
        "1) Executive summary of key findings, "
        "2) Revenue breakdown by business segment, "
        "3) Cost analysis and margin trends, "
        "4) Risk factors and mitigation strategies, "
        "5) Forward-looking projections with assumptions. "
        "Be thorough and specific in your analysis.\n\n"
        "---DOCUMENT START---\n"
    )
    closing = "\n---DOCUMENT END---\n\nBased on the above document, provide a comprehensive analysis."

    padding = (_PROMPT_PADDING * repeats).strip()
    return instruction + padding + closing


def build_user_prompt() -> str:
    """Build a user prompt that triggers a long chain-of-thought response."""
    return (
        "Synthesize a comprehensive investment thesis for the company described in the document. "
        "Include detailed financial analysis, competitive positioning, growth catalysts, "
        "risk assessment, and a 12-month price target with supporting rationale. "
        "Write at least 2000 words in your response. Be thorough and specific."
    )


# ---------------------------------------------------------------------------
# Config manipulation
# ---------------------------------------------------------------------------

def _read_config() -> str:
    return CONFIG_YAML.read_text()


def _write_config(content: str) -> None:
    CONFIG_YAML.write_text(content)


def _replace_slot_pool_size(content: str, new_value: int) -> str:
    """Replace session_slot_pool_size in config.yaml content."""
    return re.sub(
        r'^\s*session_slot_pool_size:\s*\d+',
        f'session_slot_pool_size: {new_value}',
        content,
        flags=re.MULTILINE,
    )


def set_slot_count(slot_count: int) -> None:
    """Update config.yaml with the given slot count and validate the change."""
    content = _read_config()
    updated = _replace_slot_pool_size(content, slot_count)

    # Verify the change took effect
    new_val = re.search(r'^\s*session_slot_pool_size:\s*(\d+)', updated, re.MULTILINE)
    assert new_val and int(new_val.group(1)) == slot_count, (
        f"Failed to set session_slot_pool_size to {slot_count}"
    )
    _write_config(updated)
    print(f"  config.yaml: session_slot_pool_size → {slot_count}")


# ---------------------------------------------------------------------------
# Service management
# ---------------------------------------------------------------------------

def restart_services(slot_count: int) -> None:
    """Restart proxy (which triggers llama-server restart with new slot count).

    In this dev environment, the proxy runs as a uvicorn process managed by
    the user's terminal session.  We:
      1. Update config.yaml with the new slot count
      2. Kill the current uvicorn process
      3. Restart it in the background
      4. Wait for proxy and llama-server to become ready
    """
    print("  Restarting proxy and llama-server...")

    # 1. Find current proxy PID
    proxy_pid = None
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'cmdline']):
            cmdline = proc.info.get('cmdline') or []
            if 'uvicorn' in str(cmdline).lower() and 'proxy.server' in str(cmdline):
                proxy_pid = proc.info['pid']
                break
    except ImportError:
        # fallback: pgrep
        import subprocess
        try:
            result = subprocess.run(
                ["pgrep", "-f", "uvicorn.*proxy.server"],
                capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                proxy_pid = int(result.stdout.strip().split('\n')[0])
        except Exception:
            pass

    if not proxy_pid:
        print("  WARNING: Could not find running proxy process. Restart manually.")
        print("  Config updated. Run manually:")
        print(f"    cd {PROJECT_ROOT} && source .venv/bin/activate")
        print("    python3 -m uvicorn proxy.server:app --host 0.0.0.0 --port 8000")
        return

    # 2. Kill proxy gracefully (SIGTERM)
    import signal
    try:
        os.kill(proxy_pid, signal.SIGTERM)
        print(f"  Sent SIGTERM to proxy PID {proxy_pid}")
    except ProcessLookupError:
        print("  Proxy process already exited")
    except PermissionError:
        print(f"  WARNING: No permission to kill PID {proxy_pid}. Restart manually.")
        return

    # 3. Wait for cleanup
    try:
        import psutil
        psutil_proc = psutil.Process(proxy_pid)
        psutil_proc.wait(timeout=30)
        print("  Proxy process exited gracefully")
    except (ImportError, psutil.NoSuchProcess):
        # Give it a moment
        time.sleep(3)
    except psutil.TimeoutExpired:
        print("  Proxy did not exit in 30s, force-killing...")
        try:
            os.kill(proxy_pid, signal.SIGKILL)
        except Exception:
            pass

    # 4. Start new proxy
    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python3")
    log_file = open(str(PROJECT_ROOT / "logs" / "proxy-benchmark.log"), "a")
    new_proc = subprocess.Popen(
        [
            venv_python, "-m", "uvicorn", "proxy.server:app",
            "--host", "0.0.0.0", "--port", str(PROXY_PORT),
            "--log-level", "info",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # Don't daemonize — keep as child so we can kill it later
    )
    print(f"  New proxy started (PID {new_proc.pid})")

    # 5. Wait for proxy to be ready
    _wait_for_proxy()

    # 6. Wait for llama-server to be ready
    _wait_for_llama_server()


def _wait_for_llama_server(timeout: int = 300) -> None:
    """Wait until llama-server responds on its health endpoint."""
    from urllib.request import urlopen
    from urllib.error import URLError

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urlopen(f"http://localhost:{LLAMA_SERVER_PORT}/health", timeout=5)
            if resp.status == 200:
                print("  llama-server ready")
                return
        except (URLError, OSError):
            pass
        time.sleep(2)
    print("  WARNING: llama-server did not become ready within timeout")


def _wait_for_proxy(timeout: int = 120) -> None:
    """Wait until the proxy responds to a health check."""
    from urllib.request import urlopen, URLError

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urlopen(f"http://localhost:{PROXY_PORT}/health", timeout=5)
            if resp.status == 200:
                print("  Proxy ready")
                return
        except URLError:
            pass
        time.sleep(2)
    print("  WARNING: proxy did not become ready within timeout")


def clear_slot_cache() -> None:
    """Delete cached slot files to ensure clean-slate measurements."""
    """Delete cached slot files to ensure clean-slate measurements."""
    if not SLOT_CACHE_DIR.exists():
        return
    count = 0
    for f in SLOT_CACHE_DIR.iterdir():
        if f.suffix == ".bin":
            f.unlink()
            count += 1
    if count > 0:
        print(f"  Cleared {count} cached slot files")


# ---------------------------------------------------------------------------
# Benchmark request
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_index: int
    status: str  # "completed" or "error"
    total_duration_seconds: float
    time_to_first_token_seconds: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    error: str | None = None

    def to_dict(self):
        return dict(
            request_index=self.request_index,
            status=self.status,
            total_duration_seconds=round(self.total_duration_seconds, 3),
            time_to_first_token_seconds=(
                round(self.time_to_first_token_seconds, 3)
                if self.time_to_first_token_seconds is not None
                else None
            ),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            tokens_per_second=round(self.tokens_per_second, 2),
            error=self.error,
        )


@dataclass
class SlotRunResult:
    slot_count: int
    results: list[RequestResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    gpu_memory_bytes: int | None = None
    error: str | None = None

    def summary(self) -> dict:
        completed = [r for r in self.results if r.status == "completed"]
        errors = [r for r in self.results if r.status == "error"]

        durations = [r.total_duration_seconds for r in completed]
        ttfts = [r.time_to_first_token_seconds for r in completed if r.time_to_first_token_seconds is not None]
        tpss = [r.tokens_per_second for r in completed]

        return dict(
            slot_count=self.slot_count,
            total_requests=len(self.results),
            completed=len(completed),
            errors=len(errors),
            total_duration_seconds=sum(durations) if durations else 0.0,
            avg_duration=round(sum(durations) / len(durations), 1) if durations else None,
            min_duration=round(min(durations), 1) if durations else None,
            max_duration=round(max(durations), 1) if durations else None,
            median_duration=round(sorted(durations)[len(durations) // 2], 1) if durations else None,
            avg_ttft=round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
            avg_tps=round(sum(tpss) / len(tpss), 1) if tpss else None,
            total_prompt_tokens=sum(r.prompt_tokens for r in completed),
            total_completion_tokens=sum(r.completion_tokens for r in completed),
            gpu_memory_bytes=self.gpu_memory_bytes,
            start_time=self.start_time,
            end_time=self.end_time,
            error=self.error,
        )

    def to_dict(self) -> dict:
        return dict(
            config=dict(
                slot_count=self.slot_count,
                total_requests=len(self.results),
                prompt_tokens_target=TARGET_PROMPT_TOKENS,
                output_tokens_target=TARGET_OUTPUT_TOKENS,
            ),
            requests=[r.to_dict() for r in self.results],
            summary=self.summary(),
        )


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def send_request(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_prompt: str,
    index: int,
    base_url: str = f"http://localhost:{PROXY_PORT}",
    max_tokens: int = TARGET_OUTPUT_TOKENS,
    timeout: float = TARGET_RESPONSE_TIME_S * 2,  # 2× timeout safety margin
) -> RequestResult:
    """Send a single request and measure timing."""
    payload = {
        "model": "Qwen3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    start = time.monotonic()
    try:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            body = resp.text[:500]
            return RequestResult(
                request_index=index,
                status="error",
                total_duration_seconds=elapsed,
                error=f"HTTP {resp.status_code}: {body}",
            )

        data = resp.json()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens > 0 else 0.0

        # TTFT estimate from non-streaming response — not perfectly accurate,
        # but gives a rough idea (models first-token latency as ~1.5 decode steps)
        time_to_first_token = None
        if completion_tokens > 0:
            time_to_first_token = elapsed / completion_tokens * 1.5

        return RequestResult(
            request_index=index,
            status="completed",
            total_duration_seconds=elapsed,
            time_to_first_token=time_to_first_token,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tps,
        )

    except httpx.TimeoutException as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            request_index=index,
            status="error",
            total_duration_seconds=elapsed,
            error=f"Timeout after {elapsed:.0f}s: {e}",
        )
    except httpx.ConnectError as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            request_index=index,
            status="error",
            total_duration_seconds=elapsed,
            error=f"Connection error: {e}",
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            request_index=index,
            status="error",
            total_duration_seconds=elapsed,
            error=f"Exception: {e}",
        )


async def run_slot_benchmark(
    slot_count: int,
    system_prompt: str,
    user_prompt: str,
    num_requests: int = TOTAL_REQUESTS_PER_RUN,
    warmup: int = WARMUP_REQUESTS,
    base_url: str = f"http://localhost:{PROXY_PORT}",
) -> SlotRunResult:
    """Execute a benchmark run for a given slot count."""
    result = SlotRunResult(slot_count=slot_count)
    result.start_time = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}")
    print(f"  Slot count: {slot_count}")
    print(f"  Requests: {num_requests}")
    print(f"  {'='*54}")

    if httpx is None:
        result.error = "httpx not installed"
        return result

    # Clear slot cache for clean measurements
    clear_slot_cache()

    total = num_requests + warmup

    async with httpx.AsyncClient(timeout=TARGET_RESPONSE_TIME_S * 2) as client:
        for i in range(total):
            idx = i - warmup  # warmup requests have negative index
            is_warmup = idx < 0

            # Stagger start times to simulate realistic interleaving
            if i > 0:
                await asyncio.sleep(INTER_REQUEST_GAP_S)

            req_result = await send_request(
                client, system_prompt, user_prompt,
                index=idx if not is_warmup else -1,
                base_url=base_url,
            )

            if is_warmup:
                status_char = "✓" if req_result.status == "completed" else "✗"
                print(f"  Warmup {i+1}/{warmup}: {status_char} ({req_result.total_duration_seconds:.0f}s)")
            else:
                result.results.append(req_result)
                status_char = "✓" if req_result.status == "completed" else "✗"
                print(f"  Request {idx+1}/{num_requests}: {status_char}  "
                      f"dur={req_result.total_duration_seconds:.0f}s  "
                      f"tokens={req_result.completion_tokens}  "
                      f"error={req_result.error or ''}")

    result.end_time = datetime.now(timezone.utc).isoformat()

    # Capture GPU memory snapshot
    result.gpu_memory_bytes = _capture_gpu_memory()

    return result


def _capture_gpu_memory() -> int | None:
    """Try to capture GPU memory usage from /admin/metrics."""
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            f"http://localhost:{PROXY_PORT}/admin/metrics", timeout=5
        )
        for line in resp.read().decode().split("\n"):
            if "rocm_vram_used_bytes" in line:
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[-1])
            if "cuda_vram_used_bytes" in line:
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[-1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_results: list[SlotRunResult], output_dir: Path) -> Path:
    """Generate a full benchmark report as JSON and a human-readable summary."""
    report_path = output_dir / "slot_benchmark_report.json"
    summary_path = output_dir / "slot_benchmark_summary.txt"

    report = {
        "benchmark": "Slot-count optimization for local Qwen3",
        "description": (
            "Measures throughput, latency, and GPU memory across varying "
            "session_slot_pool_size values for ~10-min queries with ~16k token "
            "system prompts."
        ),
        "prompt_tokens_target": TARGET_PROMPT_TOKENS,
        "output_tokens_target": TARGET_OUTPUT_TOKENS,
        "target_response_time_seconds": TARGET_RESPONSE_TIME_S,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": [
            dict(
                slot_count=r.slot_count,
                summary=r.summary(),
                requests=[req.to_dict() for req in r.results],
                start_time=r.start_time,
                end_time=r.end_time,
            )
            for r in all_results
        ],
    }

    # Write JSON report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    # Write human-readable summary
    lines = [
        "=" * 72,
        "  SLOT-COUNT BENCHMARK REPORT",
        "=" * 72,
        f"  Target: {TARGET_PROMPT_TOKENS:,} prompt tokens, ~{TARGET_RESPONSE_TIME_S}s response",
        f"  Requests per run: {TOTAL_REQUESTS_PER_RUN}  |  Warmup: {WARMUP_REQUESTS}",
        f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"{'Slots':>6}  {'Completed':>10}  {'Errors':>7}  {'Avg Dur':>8}  {'Min Dur':>8}  {'Max Dur':>8}  {'Avg TPS':>8}  {'Avg TTFT':>9}  {'GPU Mem':>10}",
        "-" * 80,
    ]

    for r in all_results:
        s = r.summary()
        gpu_str = ""
        if s.get("gpu_memory_bytes"):
            gb = s["gpu_memory_bytes"] / (1024**3)
            gpu_str = f"{gb:.1f}GB"

        avg_dur = '--' if s['avg_duration'] is None else f"{s['avg_duration']:>7.0f}s"
        min_dur = '--' if s['min_duration'] is None else f"{s['min_duration']:>7.0f}s"
        max_dur = '--' if s['max_duration'] is None else f"{s['max_duration']:>7.0f}s"
        avg_tps_val = '--' if s['avg_tps'] is None else f"{s['avg_tps']:>7.1f}"
        avg_ttft_val = '--' if s['avg_ttft'] is None else f"{s['avg_ttft']:>8.2f}s"

        lines.append(
            f"{s['slot_count']:>6}  "
            f"{s['completed']:>5}/{s['total_requests']:>3}  "
            f"{s['errors']:>7}  "
            f"{avg_dur}  "
            f"{min_dur}  "
            f"{max_dur}  "
            f"{avg_tps_val}  "
            f"{avg_ttft_val}  "
            f"{gpu_str:>10}"
        )

    lines += [
        "-" * 80,
        "",
        "  Throughput scaling (completed requests x avg_tps):",
    ]

    for r in all_results:
        s = r.summary()
        t = s.get("total_completion_tokens", 0)
        t_min = s.get("total_duration_seconds", 1) / 60 if s.get("total_duration_seconds", 0) > 0 else 0
        lines.append(
            f"    {s['slot_count']} slots -> {s['completed']} req in {t_min:.1f}min total, "
            f"{t} tokens generated"
        )

    summary_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines[5:]))

    return report_path
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slot-count benchmark for local Qwen3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true",
        help=f"Run all slot counts: {SLOT_COUNTS}",
    )
    group.add_argument(
        "--slots", type=int, choices=SLOT_COUNTS,
        help=f"Run a single slot count: {SLOT_COUNTS}",
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate prompt length and config, do not send requests",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./benchmark-results",
        help="Output directory for results (default: ./benchmark-results)",
    )
    parser.add_argument(
        "--resume", type=int, choices=SLOT_COUNTS, default=None,
        help="Resume from given slot count (skip earlier ones)",
    )
    parser.add_argument(
        "--skip-restart", action="store_true",
        help="Skip proxy/llama-server restart between runs (dangerous)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if httpx is None:
        print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt materials
    print("Building system prompt...")
    system_prompt = build_system_prompt(TARGET_PROMPT_TOKENS)
    user_prompt = build_user_prompt()

    # Rough token check
    prompt_word_count = len(system_prompt.split())
    estimated_tokens = prompt_word_count // 0.75
    print(f"  System prompt: {len(system_prompt)} chars, ~{prompt_word_count} words, ~{int(estimated_tokens):,} tokens (target: {TARGET_PROMPT_TOKENS:,})")
    print(f"  User prompt: {len(user_prompt)} chars")

    if args.dry_run:
        print("\nDry run — config looks valid. Run without --dry-run to execute.")
        return

    # Determine slot counts to run
    slot_counts = SLOT_COUNTS.copy()
    if args.slots is not None:
        slot_counts = [args.slots]
    if args.resume:
        slot_counts = [s for s in slot_counts if s >= args.resume]
        if not slot_counts:
            print(f"No slot counts >= {args.resume} in {SLOT_COUNTS}")
            sys.exit(1)

    all_results: list[SlotRunResult] = []

    for slot_count in slot_counts:
        print(f"\n{'#'*60}")
        print(f"#  Slot count: {slot_count}")
        print(f"{'#'*60}")

        # 1. Update config
        set_slot_count(slot_count)

        # 2. Restart services
        if not args.skip_restart:
            clear_slot_cache()
            restart_llama_server()
            restart_proxy(slot_count)
        else:
            print("  Skipping restart (--skip-restart)")

        # 3. Run benchmark
        run_result = asyncio.run(run_slot_benchmark(
            slot_count=slot_count,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ))
        all_results.append(run_result)

        # 4. Quick inline summary
        s = run_result.summary()
        print(f"\n  --- Slot count {slot_count} summary ---")
        print(f"  Completed: {s['completed']}/{s['total_requests']}  "
              f"Errors: {s['errors']}")
        if s['avg_duration']:
            print(f"  Avg duration: {s['avg_duration']:.0f}s  "
                  f"Avg TTFT: {s['avg_ttft']:.2f}s  "
                  f"Avg TPS: {s['avg_tps']:.1f}")
            gpu_str = ""
            if s.get("gpu_memory_bytes"):
                gb = s["gpu_memory_bytes"] / (1024**3)
                gpu_str = f"  GPU memory: {gb:.1f}GB"
            print(f"  Total tokens generated: {s['total_completion_tokens']}{gpu_str}")

        # 5. Write intermediate result
        run_file = output_dir / f"slot_{slot_count}.json"
        run_file.write_text(json.dumps(run_result.to_dict(), indent=2))
        print(f"  Intermediate result saved to: {run_file}")

        # Brief pause between runs for GPU to idle
        if slot_count != slot_counts[-1]:
            print("  Cooling down (30s)...")
            time.sleep(30)

    # Generate final report
    report_path = generate_report(all_results, output_dir)

    # Restore original slot count (6 = current production value)
    print(f"\nRestoring slot count to 6...")
    set_slot_count(6)
    if not args.skip_restart:
        restart_llama_server()
        restart_proxy(6)
    else:
        print("  WARNING: slot count left at last tested value. Restart manually.")

    print(f"\n{'='*72}")
    print("  BENCHMARK COMPLETE")
    print(f"  Report: {report_path}")
    print(f"  Summary: {output_dir / 'slot_benchmark_summary.txt'}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
