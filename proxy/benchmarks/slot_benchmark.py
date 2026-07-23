#!/usr/bin/env python3
"""
Slot-count benchmarking for local Qwen3.

Varies ``session_slot_pool_size`` (and thus llama-server ``--parallel``) and
measures throughput, latency, and GPU memory for long-running queries with
~16k token system prompts that take ~10 minutes to process.

Usage:
    # Run all slot counts (1,2,4,6,8,12) sequentially
    python3 -m proxy.benchmarks.slot_benchmark --all

    # Run a specific slot count only
    python3 -m proxy.benchmarks.slot_benchmark --slots 6

    # Dry run — just validate prompt length, don't send requests
    python3 -m proxy.benchmarks.slot_benchmark --slots 6 --dry-run

    # Resume from a specific slot count (skip earlier ones)
    python3 -m proxy.benchmarks.slot_benchmark --all --resume 6

    # Custom output directory
    python3 -m proxy.benchmarks.slot_benchmark --all --output-dir ./benchmark-results
"""

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
import math
import uuid
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
TARGET_PROMPT_TOKENS = 6_000       # ~6k total: 3k shared + 3k varied
TARGET_SHARED_TOKENS = 3_000       # ~3k fixed system prompt
TARGET_VARIED_TOKENS = 3_000       # ~3k varying data (random numbers)
TARGET_RESPONSE_TIME_S = 600       # ~10 minutes per query
TARGET_OUTPUT_TOKENS = 18_000      # ~30 tok/s × 600s
TOTAL_REQUESTS_PER_RUN = 6         # 6 concurrent requests per slot-count run
WARMUP_REQUESTS = 1                # 1 warmup request before measurements
INTER_REQUEST_GAP_S = 60.0          # 1 min stagger between request starts to simulate realistic workloads
HEARTBEAT_INTERVAL_S = 15           # seconds between live progress updates during a request
PROGRESS_POLL_URL = f"http://localhost:{PROXY_PORT}/health"  # polled for kv_cache_tokens

TOTAL_SESSION_TURNS = 4              # turns per session (initial + 3 follow-ups)
EXTRA_NUMBERS_PER_TURN = 100        # random numbers added each follow-up turn


def _generate_extra_numbers(idx: int, turn: int) -> str:
    """Generate comma-separated random numbers for a follow-up turn.

    Seeded per (idx, turn) so results are reproducible across runs.
    """
    rng = random.Random(42 + idx * 100 + turn)
    numbers = [str(rng.randint(1, 999999)) for _ in range(EXTRA_NUMBERS_PER_TURN)]
    return ",".join(numbers)

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


def _build_padding(target_tokens: int) -> str:
    """Build repeated lorem ipsum padding to reach ``target_tokens`` tokens."""
    # Conservative: ~1 token ≈ 0.5 words for English prose to avoid overshooting
    target_words = int(target_tokens * 0.5)
    words_per_block = len(_PROMPT_PADDING.split())
    repeats = max(1, target_words // words_per_block)
    return (_PROMPT_PADDING * repeats).strip()


def build_shared_content() -> str:
    """Build the fixed ~8k token preamble shared across all requests.

    Returns text with a preamble plus ~8k tokens of padding, followed by
    a marker where the varied content will be appended.
    """
    preamble = (
        "You are a data processing assistant. Below are two parts.\n\n"
        "PART 1 (fixed reference): A background document for context.\n"
        "---PART 1 START---\n"
    )
    padding = _build_padding(TARGET_SHARED_TOKENS - 20)  # leave room for preamble
    marker = "\n---PART 1 END---\n\nPART 2 (data): Below is a list of comma-separated numbers.\n---PART 2 START---\n"
    return preamble + padding + marker


def build_varied_content(index: int) -> str:
    """Build ~8k tokens of comma-separated random numbers, unique per request.

    The output is a single line of comma-separated integers.  The model
    must count how many numbers appear and report the total.
    """
    import random
    rng = random.Random(42 + index)  # seeded per index for reproducibility

    # ~3k tokens; each number can be multiple subword tokens, so use ~1000 numbers
    count = 1000
    numbers = [str(rng.randint(1, 999999)) for _ in range(count)]
    closing = "\n---PART 2 END---"
    return ",".join(numbers) + closing


def build_user_prompt(index: int = 0) -> str:
    """Build a user prompt asking to count the numbers in Part 2.

    The response requires the model to attend to all the varied content,
    ensuring real processing work rather than a trivial answer.
    """
    return (
        f"How many numbers appear in Part 2? Count them carefully and provide "
        f"the exact total. Explain your reasoning step by step, then state "
        f"the final answer as 'Total: <number>'. "
        f"If you do not see Part 2 data, say 'NO DATA'."
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
    def _replace_match(m: re.Match) -> str:
        indent = m.group(0)[:m.group(0).index('session_slot_pool_size')]
        return f'{indent}session_slot_pool_size: {new_value}'
    return re.sub(
        r'^\s*session_slot_pool_size:\s*\d+',
        _replace_match,
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
    """Restart proxy via start-proxy.sh (which triggers llama-server restart).

    Delegates to the project's canonical start-proxy.sh which handles:
      - Python/venv resolution
      - API key resolution from config.yaml and auth.json
      - Uvicorn invocation with correct module path and working directory

    Steps:
      1. Kill the current proxy process (children, including llama-server, die with it)
      2. Run start-proxy.sh in the background
      3. Wait for proxy and llama-server to become ready
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
        try:
            result = subprocess.run(
                ["pgrep", "-f", "uvicorn.*proxy.server"],
                capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                proxy_pid = int(result.stdout.strip().split('\n')[0])
        except Exception:
            pass

    if proxy_pid:
        # 2. Kill proxy gracefully (SIGTERM) — children (llama-server) die with it
        import signal
        try:
            os.kill(proxy_pid, signal.SIGTERM)
            print(f"  Sent SIGTERM to proxy PID {proxy_pid}")
        except ProcessLookupError:
            print("  Proxy process already exited")
        except PermissionError:
            print(f"  WARNING: No permission to kill PID {proxy_pid}.")
            return

        # 3. Wait for cleanup
        try:
            import psutil as _psutil
            _psutil_proc = _psutil.Process(proxy_pid)
            _psutil_proc.wait(timeout=30)
            print("  Proxy process exited gracefully")
        except ImportError:
            # psutil not available, just wait a moment
            time.sleep(3)
        except Exception as _e:
            if 'TimeoutExpired' in type(_e).__name__:
                print("  Proxy did not exit in 30s, force-killing...")
                try:
                    os.kill(proxy_pid, signal.SIGKILL)
                except Exception:
                    pass
            else:
                time.sleep(3)
    else:
        print("  No running proxy found — starting fresh.")

    # 4. Ensure no stale llama-server is holding port 8080
    _kill_llama_server()

    # 5. Start new proxy via start-proxy.sh
    start_script = str(PROJECT_ROOT / "proxy" / "scripts" / "start-proxy.sh")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(str(log_dir / "proxy-benchmark.log"), "a")

    new_proc = subprocess.Popen(
        [start_script],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"  New proxy started via start-proxy.sh (PID {new_proc.pid})")

    # 6. Wait for proxy to be ready
    _wait_for_proxy()

    # 7. Wait for llama-server to be ready
    _wait_for_llama_server()


def _kill_llama_server() -> None:
    """Kill any llama-server process holding port 8080.

    The proxy spawns llama-server as a child, but killing the proxy does not
    always cascade to the child processes.  This ensures port 8080 is free
    before the new proxy spawns a fresh llama-server.
    """
    import signal
    killed = 0
    try:
        result = subprocess.run(
            ["pgrep", "-f", "llama-server.*--port 8080"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            pid = int(line.split()[0]) if ' ' in line else int(line)
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass
    except Exception:
        pass

    if killed:
        print(f"  Killed {killed} stale llama-server process(es) on port {LLAMA_SERVER_PORT}")
        time.sleep(2)
    else:
        # fuser fallback: kills any process on the port
        try:
            result = subprocess.run(
                ["fuser", "-k", f"{LLAMA_SERVER_PORT}/tcp"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"  Freed port {LLAMA_SERVER_PORT} via fuser")
                time.sleep(2)
        except Exception:
            pass


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
            # Use the index page (/) instead of /health because /health
            # queries llama-server which can hang during model loading.
            resp = urlopen(f"http://localhost:{PROXY_PORT}/", timeout=5)
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
    resolved_model: str | None = None
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
            resolved_model=self.resolved_model,
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
# Progress helpers
# ---------------------------------------------------------------------------

_heartbeat_lock = asyncio.Lock()
"""Prevent overlapping heartbeat prints when tasks finish near-simultaneously."""


async def _poll_health() -> dict:
    """Best-effort poll of the proxy health endpoint for diagnostics.

    Returns a dict with keys:
    - ``model``: current_model
    - ``running``: llama_server_running bool
    - ``ready``: backend_ready bool
    - ``slots_processing``: number of slots actively processing
    - ``sessions_active``: active session count
    """
    result = {
        "model": None,
        "running": None,
        "ready": None,
        "slots_processing": None,
        "sessions_active": None,
    }
    try:
        import urllib.request
        resp = urllib.request.urlopen(PROGRESS_POLL_URL, timeout=3)
        data = resp.read().decode()
        parsed = json.loads(data)
        result["model"] = parsed.get("current_model")
        result["running"] = parsed.get("llama_server_running", False)
        result["ready"] = parsed.get("backend_reachable", False)
    except Exception:
        result["ready"] = False

    # Poll llama-server /slots for per-slot status
    try:
        import urllib.request
        slots_resp = urllib.request.urlopen(
            f"http://localhost:{LLAMA_SERVER_PORT}/slots", timeout=5
        )
        slots_data = json.loads(slots_resp.read().decode())
        if isinstance(slots_data, list):
            processing = [s for s in slots_data if s.get("is_processing", False)]
            result["slots_processing"] = len(processing)
    except Exception:
        pass

    # Poll /admin/metrics for session count
    try:
        import urllib.request
        metrics_resp = urllib.request.urlopen(
            f"http://localhost:{PROXY_PORT}/admin/metrics", timeout=3
        )
        metrics = json.loads(metrics_resp.read().decode())
        sess = metrics.get("session_metrics", {})
        result["sessions_active"] = sess.get("sessions_active", 0)
    except Exception:
        pass

    return result


async def _heartbeat_progress(
    label: str,
    start: float,
    request_task: asyncio.Task,
):
    """Print diagnostic heartbeats every HEARTBEAT_INTERVAL_S while a request runs."""
    no_activity_warning = False
    while not request_task.done():
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        if request_task.done():
            break
        elapsed = time.monotonic() - start
        health = await _poll_health()

        # Build status string from available diagnostics
        parts = [f"({elapsed:.0f}s)"]
        if health["slots_processing"] is not None:
            if health["slots_processing"] > 0:
                parts.append(f"slots={health['slots_processing']} active")
                no_activity_warning = False
            else:
                parts.append("no slot activity")
                if elapsed > 60 and not no_activity_warning:
                    parts.append("WARNING: request may be stuck")
                    no_activity_warning = True
        elif health.get("running") is True and health.get("ready") is False:
            # llama-server is running but too busy to respond to health checks
            parts.append("backend busy (health check timeout)")
        elif health.get("running") is None:
            parts.append("health check timed out")

        if health["sessions_active"] is not None and health["sessions_active"] > 0:
            parts.append(f"sessions={health['sessions_active']}")
        if health["model"]:
            parts.append(f"model={health['model']}")
        if health.get("running") is False:
            parts.append("LLAMA NOT RUNNING")
        elif health.get("ready") is False and health.get("running") is not False:
            # Only show BACKEND NOT READY when we're sure the backend isn't busy
            # (handled above: busy gets a different message)
            pass

        async with _heartbeat_lock:
            print(f"    {label}: {'  '.join(parts)}", flush=True)


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def send_request(
    client: httpx.AsyncClient,
    messages: list[dict],
    index: int,
    base_url: str = f"http://localhost:{PROXY_PORT}",
    max_tokens: int = TARGET_OUTPUT_TOKENS,
    timeout: float = TARGET_RESPONSE_TIME_S * 2,  # 2× timeout safety margin
    progress_label: str = "",
    extra_headers: dict | None = None,
) -> RequestResult:
    """Send a chat completion request and measure timing.

    While the request runs, a background heartbeat prints progress every
    ``HEARTBEAT_INTERVAL_S`` seconds showing elapsed time and diagnostics.

    Args:
        messages: The ``messages`` list for the chat completion payload.
        extra_headers: Optional HTTP headers (e.g. ``{"X-Session-Id": "..."}``).
        progress_label: Human-readable label for heartbeat output.
            Empty string = no heartbeats.
    """
    payload = {
        "model": "plan",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    start = time.monotonic()

    async def _do_post():
        kwargs = {"json": payload, "timeout": timeout}
        if extra_headers:
            kwargs["headers"] = extra_headers
        return await client.post(
            f"{base_url}/v1/chat/completions",
            **kwargs,
        )

    request_task = asyncio.create_task(_do_post())

    # Start heartbeat background task if we have a label
    if progress_label:
        heartbeat = asyncio.create_task(
            _heartbeat_progress(progress_label, start, request_task),
        )

    try:
        resp = await request_task
        elapsed = time.monotonic() - start

        # Cancel the heartbeat task now that the request is done
        if progress_label:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

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

        # Read X-Resolved-Model header to see which provider served this request
        resolved_model = resp.headers.get("X-Resolved-Model") or None

        # TTFT estimate from non-streaming response — not perfectly accurate,
        # but gives a rough idea (models first-token latency as ~1.5 decode steps)
        time_to_first_token = None
        if completion_tokens > 0:
            time_to_first_token = elapsed / completion_tokens * 1.5

        return RequestResult(
            request_index=index,
            status="completed",
            total_duration_seconds=elapsed,
            time_to_first_token_seconds=time_to_first_token,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tps,
            resolved_model=resolved_model,
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
    shared_content: str,
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
        # Create all tasks with staggered start delays.
        # Each task independently sleeps for stagger * index seconds before
        # sending, so requests start 60s apart but run concurrently.
        async def _run_session(idx):
            """Run a 4-turn session for one benchmark request.

            Turn 1: Send initial system prompt (shared + varied ~8k numbers)
                    + user prompt asking to count.
            Turns 2-4: Send 100 new random numbers each, ask updated count.
            """
            is_warmup = idx < 0
            session_id = f"benchmark-{slot_count}-{idx}-{uuid.uuid4().hex[:8]}"
            label = "Warmup" if is_warmup else f"Request {idx+1}/{num_requests}"

            # Stagger: delay start by INTER_REQUEST_GAP_S * position
            if idx > 0 and not is_warmup:
                await asyncio.sleep(INTER_REQUEST_GAP_S * idx)

            # Build initial system prompt with varied content
            varied = build_varied_content(idx)
            full_system = shared_content + varied

            headers = {"X-Session-Id": session_id}
            session_base = max(idx, 0) * TOTAL_SESSION_TURNS
            turn_results: list[RequestResult] = []

            for turn in range(1, TOTAL_SESSION_TURNS + 1):
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                turn_label = f"{label} turn {turn}/{TOTAL_SESSION_TURNS}"

                # Build messages for this turn
                if turn == 1:
                    messages = [
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_prompt},
                    ]
                else:
                    extra = _generate_extra_numbers(idx, turn)
                    follow_up = (
                        f"Here are {EXTRA_NUMBERS_PER_TURN} more numbers: {extra}. "
                        f"Count them too and tell me the total so far."
                    )
                    messages = [{"role": "user", "content": follow_up}]

                print(f"  [{now_str}] Sending {turn_label}...")

                req = await send_request(
                    client, messages,
                    index=session_base + turn,
                    base_url=base_url,
                    progress_label=turn_label,
                    extra_headers=headers,
                )
                turn_results.append(req)

                status_char = "✓" if req.status == "completed" else "✗"
                model_info = f"  via={req.resolved_model}" if req.resolved_model else ""
                print(f"    {turn_label}: {status_char}  "
                      f"dur={req.total_duration_seconds:.0f}s  "
                      f"tokens={req.completion_tokens}{model_info}"
                      + (f"  error={req.error}" if req.error else ""))

            # Summarize the session
            if is_warmup:
                total_dur = sum(r.total_duration_seconds for r in turn_results)
                print(f"  Warmup: ✓ ({total_dur:.0f}s total, {TOTAL_SESSION_TURNS} turns)")
            else:
                for r in turn_results:
                    result.results.append(r)
                total_dur = sum(r.total_duration_seconds for r in turn_results)
                total_tok = sum(r.completion_tokens for r in turn_results if r.status == "completed")
                print(f"  {label}: ✓  "
                      f"dur={total_dur:.0f}s  "
                      f"tokens={total_tok}  "
                      f"({TOTAL_SESSION_TURNS} turns)")

        # Build task list
        tasks = []
        if warmup > 0:
            tasks.append(_run_session(-1))
        for i in range(num_requests):
            tasks.append(_run_session(i))

        # Run all tasks concurrently, processing results as they complete
        await asyncio.gather(*tasks)

        # Sort results back to request_index order
        result.results.sort(key=lambda r: r.request_index)

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
        f"  Requests per run: {TOTAL_REQUESTS_PER_RUN}  |  Warmup: {WARMUP_REQUESTS if WARMUP_REQUESTS > 0 else 'none'}",
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


async def run_test_mode(
    slot_count: int = 6,
    base_url: str = f"http://localhost:{PROXY_PORT}",
    skip_restart: bool = False,
) -> None:
    """Send a single small 'Say Hi' request to verify proxy/llama-server connectivity.

    This is a lightweight connectivity check.  It does not update config.yaml or
    restart services unless ``skip_restart`` is False (the default).

    Args:
        slot_count: Slot count to configure (only used when restarting).
        base_url: Proxy base URL.
        skip_restart: If True, skip config update and service restart.
    """
    print(f"\n{'='*60}")
    print(f"  TEST MODE: Sending a single small prompt")
    print(f"{'='*60}")

    if not skip_restart:
        print("  Updating config to slot count {slot_count}...")
        set_slot_count(slot_count)
        clear_slot_cache()
        restart_services(slot_count)
    else:
        print("  Skipping config update and restart (--skip-restart)")

    if httpx is None:
        print("Error: httpx is not installed. Install with: pip install httpx", file=sys.stderr)
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say Hi"},
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"  Sending: 'Say Hi'")
        req = await send_request(
            client, messages, index=0, base_url=base_url,
            max_tokens=20, timeout=120.0,
            progress_label="Test",
        )

    if req.status == "completed":
        print(f"  ✓ Test passed!")
        print(f"      Duration: {req.total_duration_seconds:.1f}s")
        print(f"      Prompt tokens: {req.prompt_tokens}")
        print(f"      Completion tokens: {req.completion_tokens}")
        print(f"      Tokens/sec: {req.tokens_per_second:.1f}")
        if req.time_to_first_token_seconds is not None:
            print(f"      TTFT: {req.time_to_first_token_seconds:.3f}s")
    else:
        print(f"  ✗ Test FAILED: {req.error}")

    print(f"\n{'='*60}")
    print(f"  TEST MODE COMPLETE")
    print(f"{'='*60}")


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
    group.add_argument(
        "--test", action="store_true",
        help="Send a single 'Say Hi' request to verify proxy/llama-server connectivity",
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
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip the warmup request before measured requests (default: 1 warmup)",
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
    print("Building prompt materials...")
    shared = build_shared_content()
    user_prompt = build_user_prompt()

    # Rough token estimates using character heuristics
    # English prose: ~4 chars/token. Numbers+commas: ~5 chars/token.
    # Actual tokenization is more expensive (numbers are subword tokens).
    # Use conservative estimates: ~3 chars/token for prose, ~4 for numbers.
    sample_varied = build_varied_content(0)
    shared_est = len(shared) // 3
    varied_est = len(sample_varied) // 4
    print(f"  Shared content: ~{shared_est:,} tokens (target: {TARGET_SHARED_TOKENS:,})  [{len(shared):,} chars]")
    print(f"  Varied content: ~{varied_est:,} tokens (target: {TARGET_VARIED_TOKENS:,})  [{len(sample_varied):,} chars]  [generated per request]")
    print(f"  Total: ~{shared_est + varied_est:,} tokens (target: {TARGET_PROMPT_TOKENS:,})")

    if args.dry_run:
        print("\nDry run — config looks valid. Run without --dry-run to execute.")
        return

    # Handle --test mode: send a single small request to verify connectivity
    if args.test:
        asyncio.run(run_test_mode(
            slot_count=args.slots or SLOT_COUNTS[0],
            base_url=f"http://localhost:{PROXY_PORT}",
            skip_restart=args.skip_restart,
        ))
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
            restart_services(slot_count)
        else:
            print("  Skipping restart (--skip-restart)")

        # Determine warmup count
        warmup_count = 0 if args.no_warmup else WARMUP_REQUESTS

        # 3. Run benchmark
        run_result = asyncio.run(run_slot_benchmark(
            slot_count=slot_count,
            shared_content=shared,
            user_prompt=user_prompt,
            warmup=warmup_count,
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
        restart_services(6)
    else:
        print("  WARNING: slot count left at last tested value. Restart manually.")

    print(f"\n{'='*72}")
    print("  BENCHMARK COMPLETE")
    print(f"  Report: {report_path}")
    print(f"  Summary: {output_dir / 'slot_benchmark_summary.txt'}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
