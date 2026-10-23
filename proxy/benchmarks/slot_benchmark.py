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
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

# Reuse the log-parsing helpers from the contention-queue simulation harness
# (same proxy log markers: contention_queue_dispatch /
# contention_queue_fallback_after_queue).
try:
    from proxy.benchmarks import contention_queue_simulation as _cqs
except ImportError:  # pragma: no cover - exercised when run as a script
    try:
        from benchmarks import contention_queue_simulation as _cqs
    except ImportError:
        _cqs = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The repo root whose live proxy/config is benchmarked. Defaults to the repo
# this harness lives in; ``LLAMA_PROXY_BENCHMARK_ROOT`` overrides it so the
# harness code can live in a git worktree while it operates on the live
# checkout (the running proxy reads that checkout's config.yaml and
# start-proxy.sh).
PROJECT_ROOT = Path(
    os.environ.get("LLAMA_PROXY_BENCHMARK_ROOT") or Path(__file__).resolve().parent.parent.parent
)
CONFIG_YAML = PROJECT_ROOT / "proxy" / "config.yaml"
LOG_DIR = Path("/var/log/llama-proxy")
BENCHMARK_LOG = PROJECT_ROOT / "logs" / "proxy-benchmark.log"
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

#: Default health-snapshot cadence during a run (used for local quarantine
#: episode counting and local stream availability).
HEALTH_SAMPLE_INTERVAL_S = 30


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

#: The 1/2/3-slot sweep mandated by LP-0MVAZPQUM006EAC7 (AC1) — run via
#: ``--sweep``. Distinct from ``--all`` (historical full set: 1,2,4,6,8,12).
SWEEP_SLOT_COUNTS = [1, 2, 3]

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


def build_shared_content(prompt_tokens: int = TARGET_PROMPT_TOKENS) -> str:
    """Build the fixed preamble shared across all requests.

    The shared portion is half the target prompt size (defaults preserve the
    historical ~3k shared tokens when ``prompt_tokens`` is the 6k default).
    Returns text with a preamble plus padding, followed by a marker where the
    varied content will be appended.
    """
    preamble = (
        "You are a data processing assistant. Below are two parts.\n\n"
        "PART 1 (fixed reference): A background document for context.\n"
        "---PART 1 START---\n"
    )
    shared_tokens = max(20, prompt_tokens // 2)
    padding = _build_padding(shared_tokens - 20)  # leave room for preamble
    marker = "\n---PART 1 END---\n\nPART 2 (data): Below is a list of comma-separated numbers.\n---PART 2 START---\n"
    return preamble + padding + marker


def build_varied_content(index: int, prompt_tokens: int = TARGET_PROMPT_TOKENS) -> str:
    """Build comma-separated random numbers, unique per request.

    The output is a single line of comma-separated integers.  The varied
    portion is half the target prompt size (~3 tokens per number); defaults
    preserve the historical ~1000-number varied block when ``prompt_tokens``
    is the 6k default.
    """
    import random
    rng = random.Random(42 + index)  # seeded per index for reproducibility

    varied_tokens = max(20, prompt_tokens - prompt_tokens // 2)
    count = max(1, varied_tokens // 3)
    numbers = [str(rng.randint(1, 999999)) for _ in range(count)]
    closing = "\n---PART 2 END---"
    return ",".join(numbers) + closing


def build_user_prompt(index: int = 0) -> str:
    """Build a user prompt asking to count the numbers in Part 2.

    The response requires the model to attend to all the varied content,
    ensuring real processing work rather than a trivial answer.
    """
    return (
        "How many numbers appear in Part 2? Count them carefully and provide "
        "the exact total. Explain your reasoning step by step, then state "
        "the final answer as 'Total: <number>'. "
        "If you do not see Part 2 data, say 'NO DATA'."
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

def clear_proxy_logs() -> None:
    """Truncate proxy and benchmark logs so each run starts with clean output."""
    for log_path in [LOG_DIR / "proxy.log", BENCHMARK_LOG]:
        try:
            log_path.write_text("")
            print(f"  Cleared: {log_path}")
        except Exception as e:
            print(f"  WARNING: could not clear {log_path}: {e}")


def restart_services(slot_count: int) -> None:
    """Restart proxy via start-proxy.sh (which triggers llama-server restart).

    Delegates to the project's canonical start-proxy.sh which handles:
      - Python/venv resolution
      - API key resolution from config.yaml and auth.json
      - Uvicorn invocation with correct module path and working directory

    Steps:
      1. Clear proxy logs for a fresh start
      2. Kill the current proxy process (children, including llama-server, die with it)
      3. Run start-proxy.sh in the background
      4. Wait for proxy and llama-server to become ready
    """
    print("  Restarting proxy and llama-server...")
    clear_proxy_logs()

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
    proxy_ok = _wait_for_proxy()

    # 7. Wait for llama-server to be ready
    llama_ok = _wait_for_llama_server()
    return llama_ok and proxy_ok


def _read_slot_pool_size_from_config() -> int:
    """Return the current ``session_slot_pool_size`` in config.yaml (base)."""
    try:
        m = re.search(r'^\s*session_slot_pool_size:\s*(\d+)', _read_config(), re.MULTILINE)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 1


#: Output headroom subtracted from per-slot ctx (mirrors provider.py routing
#: clamp: ``physical = local_model_ctx_size // slots - 4096``).
PHYSICAL_CTX_HEADROOM = 4096


def _read_ctx_size_from_config() -> int | None:
    """Return ``local_model_ctx_size`` from config.yaml, else None."""
    try:
        m = re.search(r'^\s*local_model_ctx_size:\s*(\d+)', _read_config(), re.MULTILINE)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _capture_system_memory() -> int | None:
    """Return available system memory (MemAvailable) in bytes, else None."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                return int(parts[1]) * 1024
    except Exception:
        pass
    return None


def _observe_llama_parallel() -> int | None:
    """Observe llama-server --parallel via /slots (count of slots), else None."""
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            f"http://localhost:{LLAMA_SERVER_PORT}/slots?model=Qwen3", timeout=5
        )
        data = json.loads(resp.read().decode())
        if isinstance(data, list):
            return len(data)
    except Exception:
        pass
    return None


def _format_viability_gb(value: int | None) -> str:
    """Render a byte count as GiB text, or '--' when None."""
    if value is None:
        return "--"
    return f"{value / (1024**3):.1f}GB"


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


def _wait_for_llama_server(timeout: int = 300) -> bool:
    """Wait until llama-server responds on its health endpoint.

    Returns True when ready, False on timeout (caller records it as a
    viability-failure signal rather than silently continuing; AC4 of
    LP-0MVAZPQUM006EAC7).
    """
    from urllib.error import URLError
    from urllib.request import urlopen

    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            resp = urlopen(f"http://localhost:{LLAMA_SERVER_PORT}/health", timeout=5)
            if resp.status == 200:
                print("  llama-server ready")
                return True
        except (URLError, OSError) as e:
            last_err = e
        time.sleep(2)
    print(f"  WARNING: llama-server did not become ready within {timeout}s (last error: {last_err})")
    return False


def _wait_for_proxy(timeout: int = 120) -> bool:
    """Wait until the proxy responds to a health check."""
    from urllib.request import URLError, urlopen

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # Use the index page (/) instead of /health because /health
            # queries llama-server which can hang during model loading.
            resp = urlopen(f"http://localhost:{PROXY_PORT}/", timeout=5)
            if resp.status == 200:
                print("  Proxy ready")
                return True
        except URLError:
            pass
        time.sleep(2)
    print("  WARNING: proxy did not become ready within timeout")
    return False


def _slot_cache_dir() -> Path:
    """Return the slot-cache directory from config (session_slot_save_path).

    Falls back to <repo>/slot-cache when the config value is missing or the
    key is absent, so clear_slot_cache() never raises on malformed config.
    """
    try:
        content = _read_config()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("session_slot_save_path:") or line.startswith(
                "  session_slot_save_path:"
            ):
                value = line.split(":", 1)[1].strip().strip('"').strip("'")
                if value:
                    return Path(value).expanduser()
    except Exception:
        pass
    return PROJECT_ROOT / "slot-cache"


def clear_slot_cache() -> None:
    """Delete cached slot files to ensure clean-slate measurements.

    Opt-in via --clean-cache (default OFF) so production's warm slot-cache
    persistence is preserved for representative warm-phase measurements.
    """
    cache_dir = _slot_cache_dir()
    if not cache_dir.exists():
        print(f"  Slot cache dir not found: {cache_dir}")
        return
    count = 0
    for f in cache_dir.iterdir():
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


# ---------------------------------------------------------------------------

@dataclass
class ContentionMetrics:
    """Contention metrics derived from proxy observability for one slot run.

    Every metric is nullable — when a metric cannot be derived (no logs, no
    health snapshots, no resolved models) the field stays ``None`` and the
    reason is recorded in :attr:`unavailable`, so absent metrics are reported
    explicitly rather than silently omitted (parent AC1).

    Sources (constraint: no new production instrumentation):

    - ``queue_timeout_*``  — proxy log parsing of the
      ``contention_queue_fallback_after_queue`` / ``contention_queue_dispatch``
      markers (reusing ``contention_queue_simulation``).
    - ``local_stream_*``   — per-request ``resolved_model`` from the proxy's
      ``X-Resolved-Model`` response header (local provider prefix).
    - ``quarantine_*``     — ``/health`` endpoint snapshots taken during the
      run (transitions into local-unavailable states).
    """

    queue_timeout_count: int | None = None
    queue_dispatch_count: int | None = None
    queue_timeout_rate: float | None = None
    local_stream_count: int | None = None
    observed_stream_count: int | None = None
    local_stream_share: float | None = None
    quarantine_episode_count: int | None = None
    health_samples: int = 0
    log_files: list[str] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(
            queue_timeout_count=self.queue_timeout_count,
            queue_timeout_rate=(
                round(self.queue_timeout_rate, 4)
                if self.queue_timeout_rate is not None else None
            ),
            local_stream_share=(
                round(self.local_stream_share, 4)
                if self.local_stream_share is not None else None
            ),
            local_stream_count=self.local_stream_count,
            observed_stream_count=self.observed_stream_count,
            quarantine_episode_count=self.quarantine_episode_count,
            unavailable=dict(self.unavailable),
        )

    def to_dict(self) -> dict:
        return self.summary()


@dataclass
class SlotRunResult:
    slot_count: int
    results: list[RequestResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    gpu_memory_bytes: int | None = None
    error: str | None = None
    phase: str = "cold"  # 'cold' or 'warm' measurement phase
    proxy_restart_time: str | None = None  # ISO timestamp of proxy restart
    llama_ready_time: str | None = None  # ISO timestamp of llama-server ready
    concurrency: int | None = None  # concurrent requests configured for the run
    prompt_tokens_target: int | None = None  # prompt-token target for the run
    start_unix: float | None = None  # monotonic-ish window start (log filtering)
    end_unix: float | None = None  # window end (log filtering)
    contention: ContentionMetrics | None = None  # contention metrics for the run
    # --- Reproducibility / viability metadata (LP-0MVAZPQUM006EAC7) ---
    command: str = ""  # exact CLI invocation for this run
    config_diff: dict = field(default_factory=dict)  # session_slot_pool_size before/after
    model: str = ""  # model name benchmarked (e.g. "plan" -> local Qwen3)
    context_size: int | None = None  # local_model_ctx_size (total, across slots)
    per_slot_ctx: int | None = None  # physical per-slot ctx: ctx // slots - headroom
    observed_slots: int | None = None  # llama-server --parallel actually observed
    system_memory_bytes: int | None = None  # MemAvailable at run end
    llama_started: bool | None = None  # None = n/a (skip-restart), True/False = restart
    viability_warnings: list[str] = field(default_factory=list)  # OOM/start-failure/ctx-reduction

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
            contention=self.contention.summary() if self.contention else None,
        )

    def to_dict(self) -> dict:
        return dict(
            config=dict(
                slot_count=self.slot_count,
                total_requests=len(self.results),
                concurrency=self.concurrency,
                prompt_tokens_target=(
                    self.prompt_tokens_target if self.prompt_tokens_target is not None
                    else TARGET_PROMPT_TOKENS
                ),
                output_tokens_target=TARGET_OUTPUT_TOKENS,
                phase=self.phase,
                proxy_restart_time=self.proxy_restart_time,
                llama_ready_time=self.llama_ready_time,
            ),
            reproducibility=self.reproducibility_dict(),
            viability=self.viability_dict(),
            requests=[r.to_dict() for r in self.results],
            summary=self.summary(),
            contention=self.contention.to_dict() if self.contention else {},
        )

    def reproducibility_dict(self) -> dict:
        """Exact command / config diff / model / ctx for this run (AC1)."""
        return dict(
            command=self.command,
            config_diff=dict(self.config_diff),
            model=self.model,
            context_size=self.context_size,
            per_slot_ctx=self.per_slot_ctx,
        )

    def viability_dict(self) -> dict:
        """Viability-failure signals for this run: OOM/start-failure/ctx (AC4)."""
        return dict(
            observed_slots=self.observed_slots,
            system_memory_bytes=self.system_memory_bytes,
            llama_started=self.llama_started,
            warnings=list(self.viability_warnings),
        )


# ---------------------------------------------------------------------------
# Contention metric derivation helpers
# ---------------------------------------------------------------------------

#: Provider-name prefixes that denote the local model in ``X-Resolved-Model``
#: header values (e.g. ``local/Qwen3``, ``local-qwen3/Qwen3``).
_LOCAL_PROVIDER_PREFIXES = ("local", "local-")


def is_local_resolved_model(resolved_model: str | None) -> bool:
    """True when an X-Resolved-Model header value denotes the local model."""
    if not resolved_model:
        return False
    provider = resolved_model.split("/", 1)[0].strip()
    return provider.startswith(_LOCAL_PROVIDER_PREFIXES)


def derive_local_stream_share(
    results: Sequence[RequestResult],
) -> tuple[int, int, float | None, str | None]:
    """Derive local stream share from per-request resolved models.

    Returns ``(local_count, observed_count, share, unavailable_reason)``.
    ``observed`` counts completed requests that reported a resolved model;
    ``share`` is ``local_count / observed``, or ``None`` with a reason when
    no request reported a resolved model.
    """
    observed = [r for r in results if r.status == "completed" and r.resolved_model]
    if not observed:
        return 0, 0, None, "no completed requests reported a resolved model"
    local = sum(1 for r in observed if is_local_resolved_model(r.resolved_model))
    return local, len(observed), local / len(observed), None


def _local_backend_available(snapshot: dict) -> bool | None:
    """Interpret one /health snapshot as local-backend availability.

    Returns True when the local backend is up, False when it is quarantined /
    unavailable (self-healing, GPU wedge, not running, or unreachable), and
    None when the snapshot carries no availability signals (unknown).
    """
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get("self_healing_in_progress"):
        return False
    if snapshot.get("gpu_wedge_detected"):
        return False
    running = snapshot.get("llama_server_running")
    reachable = snapshot.get("backend_reachable")
    if running is None and reachable is None:
        return None
    return bool(running) and bool(reachable)


def count_quarantine_episodes(
    health_snapshots: Sequence[dict],
) -> tuple[int | None, str | None]:
    """Count local quarantine episodes from /health snapshots.

    An episode is a transition from a known-available state to a known-
    unavailable state between consecutive snapshots (a contiguous period of
    local unavailability). Returns ``(count, unavailable_reason)``; the count
    is ``None`` (with a reason) when no snapshots were collected.
    """
    if not health_snapshots:
        return None, "no health snapshots collected"
    episodes = 0
    prev_known: bool | None = None
    usable = 0
    for snap in health_snapshots:
        available = _local_backend_available(snap)
        if available is None:
            continue  # unknown state — neither starts nor ends an episode
        usable += 1
        if prev_known is True and available is False:
            episodes += 1
        prev_known = available
    if usable == 0:
        return None, "no usable health snapshots collected"
    return episodes, None


def derive_queue_metrics(
    log_files: Sequence[str],
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> tuple[int | None, int | None, float | None, list[str], str | None]:
    """Derive queue-timeout metrics from proxy logs in a run window.

    Returns ``(timeout_count, dispatch_count, timeout_rate, matched_files,
    unavailable_reason)``. The rate is ``timeout / (dispatch + timeout)`` — the
    fraction of queue-path arrivals that exceeded the contention-queue caps and
    fell back. When no queue-path events occurred in the window the counts are
    0 but the rate is ``None`` (a rate is meaningless with no arrivals).
    """
    if _cqs is None:
        return None, None, None, [], "contention_queue_simulation import unavailable"
    resolved_files = [str(p) for p in log_files]
    try:
        loaded = _cqs.load_events(resolved_files, start=start_ts, end=end_ts)
    except FileNotFoundError:
        return None, None, None, [], "no proxy log files matched in run window"
    except Exception:
        return None, None, None, [], "proxy log parsing failed"
    timeouts = sum(1 for e in loaded.events if e.type == "fallback_after_queue")
    dispatches = sum(1 for e in loaded.events if e.type == "dispatch")
    arrivals = timeouts + dispatches
    if arrivals == 0:
        return timeouts, dispatches, None, loaded.files, (
            "no queue-path events in run window"
        )
    return timeouts, dispatches, timeouts / arrivals, loaded.files, None


def build_contention_metrics(
    results: Sequence[RequestResult],
    *,
    log_files: Sequence[str] | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
    health_snapshots: Sequence[dict] | None = None,
) -> ContentionMetrics:
    """Assemble contention metrics for one slot run.

    Each metric derives from its own observability source independently so a
    failure in one (e.g. missing logs) does not mask the others; every absent
    metric carries an explicit reason in ``unavailable``.
    """
    metrics = ContentionMetrics()
    metrics.log_files = [str(p) for p in (log_files or [])]

    timeouts, dispatches, rate, matched, reason = derive_queue_metrics(
        metrics.log_files, start_ts=start_ts, end_ts=end_ts
    )
    metrics.queue_timeout_count = timeouts
    metrics.queue_dispatch_count = dispatches
    metrics.queue_timeout_rate = rate
    metrics.log_files = matched or metrics.log_files
    if reason:
        metrics.unavailable["queue_timeout"] = reason

    local, observed, share, reason = derive_local_stream_share(results)
    metrics.local_stream_count = local
    metrics.observed_stream_count = observed
    metrics.local_stream_share = share
    if reason:
        metrics.unavailable["local_stream_share"] = reason

    snapshots = list(health_snapshots or [])
    metrics.health_samples = len(snapshots)
    episodes, reason = count_quarantine_episodes(snapshots)
    metrics.quarantine_episode_count = episodes
    if reason:
        metrics.unavailable["quarantine_episodes"] = reason

    return metrics


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


def _fetch_health_snapshot() -> dict | None:
    """Blocking fetch of the raw /health JSON, or None on any failure."""
    try:
        import urllib.request
        resp = urllib.request.urlopen(PROGRESS_POLL_URL, timeout=3)
        return json.loads(resp.read().decode())
    except Exception:
        return None


async def _sample_health_loop(
    snapshots: list,
    interval_s: float = HEALTH_SAMPLE_INTERVAL_S,
) -> None:
    """Append /health snapshots to *snapshots* every *interval_s* seconds.

    Runs until cancelled; failures are appended as ``None`` sentinels so the
    quarantine-episode derivation can tell a genuine quarantine from a
    sampling gap (sentinels are skipped as "unknown" states).
    """
    while True:
        snapshots.append(await asyncio.to_thread(_fetch_health_snapshot))
        await asyncio.sleep(interval_s)


async def _heartbeat_progress(
    label: str,
    start: float,
    request_task: asyncio.Task,
    resolved: dict,
    model_name: str = "",
):
    """Print diagnostic heartbeats every HEARTBEAT_INTERVAL_S while a request runs.

    Uses a shared mutable ``resolved`` dict so that once the response headers
    arrive (with ``X-Resolved-Model``), the heartbeat can switch from showing
    ``req_model=<name>`` to ``via=<provider/model>``.

    Args:
        resolved: Shared dict with key ``"model"`` — populated by the caller
            after the response is received.
        model_name: The model name being requested (e.g. "plan").
    """
    no_activity_warning = False
    try:
        while not request_task.done():
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            except asyncio.CancelledError:
                # Task was cancelled — print final line with resolved model if available
                if resolved.get("model"):
                    async with _heartbeat_lock:
                        print(f"    {label}: done  via={resolved['model']}", flush=True)
                raise

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
                parts.append("backend busy (health check timeout)")
            elif health.get("running") is None:
                parts.append("health check timed out")

            if health["sessions_active"] is not None and health["sessions_active"] > 0:
                parts.append(f"sessions={health['sessions_active']}")
            if resolved.get("model"):
                # X-Resolved-Model header available from completed response
                parts.append(f"via={resolved['model']}")
            elif model_name:
                parts.append(f"req_model={model_name}")
            elif health["model"]:
                parts.append(f"model={health['model']}")
            if health.get("running") is False:
                parts.append("LLAMA NOT RUNNING")
            elif health.get("ready") is False and health.get("running") is not False:
                pass

            async with _heartbeat_lock:
                print(f"    {label}: {'  '.join(parts)}", flush=True)
    except asyncio.CancelledError:
        # Re-raise if we didn't handle it in the inner sleep handler
        raise


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

    # Shared mutable container so the heartbeat can show the actual
    # resolved provider once response headers arrive.
    _resolved = {"model": None}

    # Start heartbeat background task if we have a label
    if progress_label:
        heartbeat = asyncio.create_task(
            _heartbeat_progress(
                progress_label, start, request_task, _resolved,
                model_name=payload.get("model", ""),
            ),
        )

    try:
        resp = await request_task
        elapsed = time.monotonic() - start

        # Populate resolved model from response header for the heartbeat
        if resp.status_code == 200:
            _resolved["model"] = resp.headers.get("X-Resolved-Model") or None

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
    prompt_tokens: int = TARGET_PROMPT_TOKENS,
    health_samples: list | None = None,
    health_interval_s: float = HEALTH_SAMPLE_INTERVAL_S,
    clean_cache: bool = False,
) -> SlotRunResult:
    """Execute a benchmark run for a given slot count.

    When *health_samples* is a list, /health snapshots are appended to it for
    the duration of the run (used for quarantine-episode counting by
    :func:`build_contention_metrics`).

    Slot cache is cleared ONLY when *clean_cache* is True (explicit opt-in via
    ``--clean-cache``); a live production proxy's warm KV cache must never be
    dropped by a benchmark default (AC5 config-drift guard, LP-0MVAZPQUM006EAC7).
    """
    result = SlotRunResult(slot_count=slot_count, concurrency=num_requests,
                           prompt_tokens_target=prompt_tokens)
    result.start_time = datetime.now(UTC).isoformat()
    result.start_unix = time.time()

    print(f"\n{'='*60}")
    print(f"  Slot count: {slot_count}")
    print(f"  Requests: {num_requests}")
    print(f"  {'='*54}")

    if httpx is None:
        result.error = "httpx not installed"
        return result

    # Clear slot cache ONLY on explicit opt-in (--clean-cache); the default
    # preserves the live proxy's warm KV cache (AC5, LP-0MVAZPQUM006EAC7).
    if clean_cache:
        clear_slot_cache()

    health_task = None
    if health_samples is not None:
        health_task = asyncio.create_task(
            _sample_health_loop(health_samples, health_interval_s)
        )

    try:
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
                varied = build_varied_content(idx, prompt_tokens)
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
    finally:
        if health_task is not None:
            health_task.cancel()
            try:
                await health_task
            except (asyncio.CancelledError, Exception):
                pass

    result.end_time = datetime.now(UTC).isoformat()
    result.end_unix = time.time()

    # Capture GPU memory snapshot
    result.gpu_memory_bytes = _capture_gpu_memory()

    return result


def _capture_gpu_memory() -> int | None:
    """Try to capture GPU memory usage.

    Sources, in order:
    1. ``/admin/metrics`` — JSON keys ``rocm_vram_used_bytes`` /
       ``cuda_vram_used_bytes`` (modern proxy emits JSON rather than the
       historic Prometheus text lines the original parser scanned).
    2. ``rocm-smi --showmeminfo vram --json`` — used bytes on ROCm hosts
       (APU/GTT platforms); ``LLAMA_PROXY_ROCSMI_BIN`` overrides the binary.
    3. ``nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits``.

    Returns None when nothing is available (recorded as an explicit
    non-available metric rather than a hidden failure; LP-0MVAZPQUM006EAC7
    records memory where available).
    """
    # 1. Proxy /admin/metrics (JSON).
    try:
        import json as _json
        import urllib.request
        resp = urllib.request.urlopen(
            f"http://localhost:{PROXY_PORT}/admin/metrics", timeout=5
        )
        text = resp.read().decode()
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("rocm_vram_used_bytes", "cuda_vram_used_bytes"):
                val = data.get(key) or data.get("gpu", {}).get(key)
                if isinstance(val, (int, float)) and val > 0:
                    return int(val)
        else:
            for line in text.split("\n"):
                parts = line.split()
                if len(parts) >= 2 and (
                    "rocm_vram_used_bytes" in parts[0]
                    or "cuda_vram_used_bytes" in parts[0]
                ):
                    try:
                        return int(parts[-1])
                    except ValueError:
                        pass
    except Exception:
        pass

    # 2. rocm-smi (ROCm GPUs / APUs).
    rocsmi = os.environ.get("LLAMA_PROXY_ROCSMI_BIN", "rocm-smi")
    try:
        result = subprocess.run(
            [rocsmi, "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            import json as _json
            data = _json.loads(result.stdout)
            for card in data.values() or []:
                used = card.get("VRAM Total Used Memory (B)")
                if isinstance(used, str) and used.strip().isdigit():
                    return int(used)
    except Exception:
        pass

    # 3. nvidia-smi.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            value = result.stdout.strip().splitlines()[0].strip()
            if value.isdigit():
                return int(value) * 1024 * 1024
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
            "Measures throughput, latency, GPU memory, and contention metrics "
            "(queue-timeout rate, local stream share, local quarantine "
            "episodes) across varying session_slot_pool_size values for "
            "~10-min queries."
        ),
        "prompt_tokens_target": TARGET_PROMPT_TOKENS,
        "output_tokens_target": TARGET_OUTPUT_TOKENS,
        "target_response_time_seconds": TARGET_RESPONSE_TIME_S,
        "timestamp": datetime.now(UTC).isoformat(),
        # Reproducibility / environmental context for each run is recorded
        # per-run (command, config diff, model, ctx size, per-slot ctx,
        # observed slots, system memory, viability warnings) so a third party
        # can re-derive the comparison without re-running (LP-0MVAZPQUM006EAC7).
        "runs": [
            dict(
                slot_count=r.slot_count,
                concurrency=r.concurrency,
                prompt_tokens_target=(r.prompt_tokens_target or TARGET_PROMPT_TOKENS),
                summary=r.summary(),
                requests=[req.to_dict() for req in r.results],
                contention=r.contention.to_dict() if r.contention else {},
                reproducibility=r.reproducibility_dict(),
                viability=r.viability_dict(),
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
    concurrency = all_results[0].concurrency if all_results else TOTAL_REQUESTS_PER_RUN
    lines = [
        "=" * 72,
        "  SLOT-COUNT BENCHMARK REPORT",
        "=" * 72,
        f"  Target: {TARGET_PROMPT_TOKENS:,} prompt tokens, ~{TARGET_RESPONSE_TIME_S}s response",
        f"  Concurrency: {concurrency}  |  Requests per run: {TOTAL_REQUESTS_PER_RUN}  |  Warmup: 0 (use --warmup to enable)",
        f"  Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
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

    # Contention metrics section (queue-timeout, local stream share,
    # quarantine episodes) — parent AC1. Every metric is reported; absent
    # metrics show the explicit reason instead of being silently omitted.
    lines += [
        "-" * 80,
        "",
        "  CONTENTION METRICS (per slot count)",
        "    Queue-timeout = contention_queue_fallback_after_queue count",
        "    Queue rate    = queue-timeouts / queue-path arrivals (dispatch+timeout)",
        "    Local share   = local stream share (local / observed resolved streams)",
        "    Quarantine    = local quarantine episodes (from /health snapshots)",
        "",
        f"{'Slots':>6}  {'Queue t/o':>10}  {'Queue rate':>10}  {'Local share':>12}  {'Local streams':>14}  {'Quarant. episodes':>18}",
        "-" * 80,
    ]

    for r in all_results:
        c = r.contention.summary() if r.contention else {}
        q_count = c.get("queue_timeout_count")
        q_rate = c.get("queue_timeout_rate")
        l_share = c.get("local_stream_share")
        l_count = c.get("local_stream_count")
        l_observed = c.get("observed_stream_count")
        q_epi = c.get("quarantine_episode_count")

        q_count_str = '--' if q_count is None else f"{q_count}"
        q_rate_str = '--' if q_rate is None else f"{q_rate:.1%}"
        l_share_str = '--' if l_share is None else f"{l_share:.1%}"
        l_count_str = '--' if l_count is None else (
            f"{l_count}/{l_observed or 0}"
        )
        q_epi_str = '--' if q_epi is None else f"{q_epi}"

        lines.append(
            f"{r.slot_count:>6}  "
            f"{q_count_str:>10}  "
            f"{q_rate_str:>10}  "
            f"{l_share_str:>12}  "
            f"{l_count_str:>14}  "
            f"{q_epi_str:>18}"
        )

    # Explicit absent-metric reasons (rather than silent omission)
    reasons_seen: list[str] = []
    for r in all_results:
        if not r.contention:
            continue
        for metric, reason in r.contention.unavailable.items():
            entry = f"    slots={r.slot_count}: {metric} — {reason}"
            if entry not in reasons_seen:
                reasons_seen.append(entry)
    if reasons_seen:
        lines += ["-" * 80, "", "  Metrics not measurable (explicit reasons):"]
        lines += reasons_seen

    # Repro context + viability signals (LP-0MVAZPQUM006EAC7 AC1/AC4)
    lines += ["-" * 80, "", "  REPRODUCIBILITY + VIABILITY (per slot count)"]
    for r in all_results:
        lines.append(f"    slots={r.slot_count}:")
        lines.append(f"      command: {r.command or '--'}")
        diff = r.config_diff.get("session_slot_pool_size", {})
        lines.append(
            f"      config: session_slot_pool_size {diff.get('before', '?')} -> "
            f"{diff.get('after', '?')}  |  model: {r.model or '--'}"
        )
        lines.append(
            f"      ctx: total={r.context_size or '--'}  "
            f"per-slot(physical)={r.per_slot_ctx or '--'}  "
            f"observed slots={r.observed_slots or '--'}"
        )
        mem = _format_viability_gb(r.system_memory_bytes)
        lines.append(f"      system MemAvailable: {mem}  llama_started: {r.llama_started}")
        if r.viability_warnings:
            for w in r.viability_warnings:
                lines.append(f"      ! VIABILITY: {w}")
        else:
            lines.append("      viability: no warnings")

    summary_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines[5:]))

    return report_path


async def run_test_mode(
    slot_count: int | None = None,
    base_url: str = f"http://localhost:{PROXY_PORT}",
    skip_restart: bool = False,
    clean_cache: bool = False,
    phase: str = "cold",
) -> None:
    """Send a single small 'Say Hi' request to verify proxy/llama-server connectivity.

    This is a lightweight connectivity check.  It does not update config.yaml or
    restart services unless ``skip_restart`` is False (the default).

    Args:
        slot_count: Slot count to configure (only used when restarting). When
            None, the current config value is used and restored afterwards, so
            the connectivity check never changes the running config.
        base_url: Proxy base URL.
        skip_restart: If True, skip config update and service restart.
        clean_cache: If True, clear slot cache before restart. Default: False.
        phase: Measurement phase ('cold' or 'warm'). Default: 'cold'.
    """
    print(f"\n{'='*60}")
    print(f"  TEST MODE: Sending a single small prompt (phase={phase})")
    print(f"{'='*60}")

    original_slot = None
    original_config = None
    if not skip_restart:
        # Never write a hard-coded value; snapshot what is there and restore it
        # afterwards so the proxy is left untouched (LP-0MVAZPQUM006EAC7 AC5).
        original_config = _read_config()
        original_slot = _read_slot_pool_size_from_config()
        target = slot_count if slot_count is not None else original_slot
        print(f"  Updating config to slot count {target}...")
        set_slot_count(target)
        # Only clear slot cache when --clean-cache is explicitly set
        if clean_cache:
            clear_slot_cache()
        else:
            print("  Slot cache preserved (--clean-cache not set)")
        restart_services(target)
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
        print("  Sending: 'Say Hi'")
        req = await send_request(
            client, messages, index=0, base_url=base_url,
            max_tokens=20, timeout=120.0,
            progress_label="Test",
        )

    if req.status == "completed":
        print("  ✓ Test passed!")
        print(f"      Duration: {req.total_duration_seconds:.1f}s")
        print(f"      Prompt tokens: {req.prompt_tokens}")
        print(f"      Completion tokens: {req.completion_tokens}")
        print(f"      Tokens/sec: {req.tokens_per_second:.1f}")
        if req.time_to_first_token_seconds is not None:
            print(f"      TTFT: {req.time_to_first_token_seconds:.3f}s")
    else:
        print(f"  ✗ Test FAILED: {req.error}")

    # Restore the config we changed (test mode must never leave drift)
    if not skip_restart and original_config is not None:
        print(f"  Restoring original config (session_slot_pool_size: {original_slot})...")
        _write_config(original_config)
        restart_services(original_slot or 1)

    print(f"\n{'='*60}")
    print("  TEST MODE COMPLETE")
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
        "--sweep", action="store_true",
        help=f"Run the 1/2/3-slot sweep only: {SWEEP_SLOT_COUNTS}",
    )
    group.add_argument(
        "--slots", type=int, choices=sorted(set(SLOT_COUNTS) | set(SWEEP_SLOT_COUNTS)),
        help=f"Run a single slot count: {sorted(set(SLOT_COUNTS) | set(SWEEP_SLOT_COUNTS))}",
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
        "--resume", type=int,
        choices=sorted(set(SLOT_COUNTS) | set(SWEEP_SLOT_COUNTS)), default=None,
        help="Resume from given slot count (skip earlier ones)",
    )
    parser.add_argument(
        "--skip-restart", action="store_true",
        help="Skip proxy/llama-server restart between runs (dangerous)",
    )
    parser.add_argument(
        "--warmup", action="store_true",
        help="Include 1 warmup request before measured requests (default: no warmup)",
    )
    parser.add_argument(
        "--clean-cache", action="store_true",
        help="Clear slot cache before run (default: OFF — preserves warm cache)",
    )
    parser.add_argument(
        "--phase", type=str, choices=["cold", "warm"], default="cold",
        help="Measurement phase: 'cold' (first runs post-restart) or 'warm' (steady-state). Default: cold",
    )
    parser.add_argument(
        "--concurrency", type=int, default=TOTAL_REQUESTS_PER_RUN,
        help=(
            f"Number of concurrent benchmark requests per slot-count run "
            f"(default: {TOTAL_REQUESTS_PER_RUN}, preserving existing behaviour)"
        ),
    )
    parser.add_argument(
        "--prompt-tokens", type=int, default=TARGET_PROMPT_TOKENS,
        help=(
            f"Target total prompt size in tokens (default: {TARGET_PROMPT_TOKENS}, "
            f"preserving existing behaviour; split ~50/50 shared/varied)"
        ),
    )
    parser.add_argument(
        "--log-dir", type=str, default=str(LOG_DIR),
        help=f"Proxy log directory scanned for contention metrics (default: {LOG_DIR})",
    )
    parser.add_argument(
        "--no-health-samples", action="store_true",
        help=(
            "Disable /health sampling. By default /health is sampled during "
            "each run to derive local quarantine episodes"
        ),
    )

    return parser.parse_args(argv)


def resolve_slot_counts(args: argparse.Namespace) -> list[int]:
    """Resolve the ordered slot counts a run will execute.

    ``--sweep`` runs ``SWEEP_SLOT_COUNTS``, ``--slots N`` a single count, and
    ``--resume`` filters the sweep list to counts >= the resume point. An empty
    list after filtering is a fatal configuration error (preserving the
    historical inline behaviour). Shared by the live run and the ``--dry-run``
    run-list printout.
    """
    if args.sweep:
        slot_counts = SWEEP_SLOT_COUNTS.copy()
    else:
        slot_counts = SLOT_COUNTS.copy()
    if args.slots is not None:
        slot_counts = [args.slots]
    if args.resume:
        slot_counts = [s for s in slot_counts if s >= args.resume]
        if not slot_counts:
            print(f"No slot counts >= {args.resume} in {SLOT_COUNTS}")
            sys.exit(1)
    return slot_counts


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if httpx is None:
        print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.concurrency < 1:
        print("Error: --concurrency must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.prompt_tokens < 1:
        print("Error: --prompt-tokens must be >= 1", file=sys.stderr)
        sys.exit(2)

    # Build prompt materials
    print("Building prompt materials...")
    shared = build_shared_content(args.prompt_tokens)
    user_prompt = build_user_prompt()

    # Rough token estimates using character heuristics
    # English prose: ~4 chars/token. Numbers+commas: ~5 chars/token.
    # Actual tokenization is more expensive (numbers are subword tokens).
    # Use conservative estimates: ~3 chars/token for prose, ~4 for numbers.
    sample_varied = build_varied_content(0, args.prompt_tokens)
    shared_est = len(shared) // 3
    varied_est = len(sample_varied) // 4
    shared_target = max(20, args.prompt_tokens // 2)
    varied_target = max(20, args.prompt_tokens - args.prompt_tokens // 2)
    print(f"  Shared content: ~{shared_est:,} tokens (target: {shared_target:,})  [{len(shared):,} chars]")
    print(f"  Varied content: ~{varied_est:,} tokens (target: {varied_target:,})  [{len(sample_varied):,} chars]  [generated per request]")
    print(f"  Total: ~{shared_est + varied_est:,} tokens (target: {args.prompt_tokens:,})")
    print(f"  Concurrency: {args.concurrency} concurrent requests")

    if args.dry_run:
        # Print the resolved run list so a dry run proves *which* slot counts
        # (and in which order) the real invocation would execute — e.g.
        # ``--sweep --resume 2`` must show 2, 3 and no N=1
        # (LP-0MVK0969J004MTG3 AC4).
        if not args.test:
            planned = resolve_slot_counts(args)
            print(f"  Slot counts: {', '.join(str(s) for s in planned)}")
        print("\nDry run — config looks valid. Run without --dry-run to execute.")
        return

    # Handle --test mode: send a single small request to verify connectivity
    if args.test:
        asyncio.run(run_test_mode(
            slot_count=None,
            base_url=f"http://localhost:{PROXY_PORT}",
            skip_restart=args.skip_restart,
            clean_cache=args.clean_cache,
            phase=args.phase,
        ))
        return

    # Determine slot counts to run
    slot_counts = resolve_slot_counts(args)

    # Snapshot the ORIGINAL slot count + full config so the sweep restores
    # exactly what it found (production value here is 1, not the historical 6;
    # never hard-code the restore value).
    original_slot = _read_slot_pool_size_from_config()
    original_config = _read_config()
    command_line = " ".join(sys.argv)

    all_results: list[SlotRunResult] = []
    report_path: Path | None = None
    error: Exception | None = None
    try:
        for slot_count in slot_counts:
            print(f"\n{'#'*60}")
            print(f"#  Slot count: {slot_count}")
            print(f"{'#'*60}")

            # 1. Update config
            set_slot_count(slot_count)

            # 2. Restart services
            restart_ok: bool | None = None
            if not args.skip_restart:
                # Only clear slot cache when --clean-cache is explicitly set
                # Default is OFF to preserve warm cache for realistic measurements
                if args.clean_cache:
                    clear_slot_cache()
                else:
                    print("  Slot cache preserved (--clean-cache not set)")

                # Record restart timestamps for reproducible measurements
                from datetime import datetime
                proxy_restart_time = datetime.now(UTC).isoformat()
                print(f"  Proxy restart recorded at: {proxy_restart_time}")

                restart_ok = restart_services(slot_count)

                # Record llama-server ready timestamp
                llama_ready_time = datetime.now(UTC).isoformat()
                print(f"  Llama-server ready at: {llama_ready_time}")
            else:
                print("  Skipping restart (--skip-restart)")
                proxy_restart_time = None
                llama_ready_time = None

            # Determine warmup count
            warmup_count = WARMUP_REQUESTS if args.warmup else 0

            # 3. Run benchmark
            health_samples: list | None = None if args.no_health_samples else []
            run_result = asyncio.run(run_slot_benchmark(
                slot_count=slot_count,
                shared_content=shared,
                user_prompt=user_prompt,
                warmup=warmup_count,
                num_requests=args.concurrency,
                prompt_tokens=args.prompt_tokens,
                health_samples=health_samples,
                clean_cache=args.clean_cache,
            ))
            # Attach metadata for JSON output
            run_result.phase = args.phase
            run_result.proxy_restart_time = proxy_restart_time
            run_result.llama_ready_time = llama_ready_time
            # --- Reproducibility + viability metadata (LP-0MVAZPQUM006EAC7) ---
            run_result.command = command_line
            run_result.config_diff = {
                "session_slot_pool_size": {"before": original_slot, "after": slot_count},
            }
            run_result.model = "plan -> local-qwen3/Qwen3"
            run_result.context_size = _read_ctx_size_from_config()
            # Physical per-slot ctx mirrors provider.py: ctx // slots - output headroom.
            if run_result.context_size:
                run_result.per_slot_ctx = max(
                    1, run_result.context_size // slot_count - PHYSICAL_CTX_HEADROOM
                )
            run_result.observed_slots = _observe_llama_parallel()
            run_result.system_memory_bytes = _capture_system_memory()
            if args.skip_restart:
                run_result.llama_started = None
            else:
                run_result.llama_started = bool(restart_ok)
                if restart_ok is False:
                    run_result.viability_warnings.append(
                        f"llama-server failed to become ready at {slot_count} slots "
                        f"(OOM / start failure) — requests may all error"
                    )
            if (run_result.observed_slots is not None
                    and run_result.observed_slots != slot_count):
                run_result.viability_warnings.append(
                    f"observed llama-server slot count {run_result.observed_slots} != "
                    f"requested {slot_count}"
                )
            if (run_result.per_slot_ctx and run_result.context_size
                    and slot_count > 1
                    and run_result.per_slot_ctx < run_result.context_size // 4):
                run_result.viability_warnings.append(
                    f"per-slot context {run_result.per_slot_ctx} < 25% of model ctx "
                    f"{run_result.context_size} — may change routing behaviour"
                )
            if run_result.viability_warnings:
                for w in run_result.viability_warnings:
                    print(f"  ! VIABILITY: {w}")
            # Derive contention metrics from proxy observability: queue-timeout
            # and local stream share always derive (logs + response headers);
            # quarantine episodes derive from /health snapshots when the
            # --health-samples flag collected them. Absent metrics are reported
            # explicitly with a reason rather than silently omitted.
            run_result.contention = build_contention_metrics(
                run_result.results,
                log_files=[str(Path(args.log_dir) / "proxy.log*")],
                start_ts=run_result.start_unix,
                end_ts=run_result.end_unix,
                health_snapshots=health_samples or [],
            )
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
    except KeyboardInterrupt:
        print("\nInterrupted — restoring original config before exit.")
    except Exception as e:  # restore must run on ANY failure (AC5)
        error = e
        print(f"\nERROR during sweep: {e}")
        print("Restoring original config before exit.")
    finally:
        # Restore the ORIGINAL slot count / config (production value here is 1;
        # never hard-code it). Config must be left exactly as found so the proxy
        # runs the correct scheduled mode afterwards (AC5). This must run even
        # if the sweep failed or was interrupted mid-way.
        print(f"\nRestoring original config (session_slot_pool_size: {original_slot})...")
        try:
            _write_config(original_config)
        except Exception:  # restore best-effort — never mask the primary failure
            print("  WARNING: could not restore config file")
        if not args.skip_restart:
            try:
                restart_services(original_slot)
            except Exception as re_:  # restart best-effort — never mask the primary failure
                print(f"  WARNING: restart with original slot failed: {re_}")
        else:
            print("  WARNING: slot count left at last tested value. Restart manually.")

    if error is not None:
        raise error

    print(f"\n{'='*72}")
    print("  BENCHMARK COMPLETE")
    print(f"  Report: {report_path}")
    print(f"  Restored config: {CONFIG_YAML} -> session_slot_pool_size: {original_slot}")
    print(f"  Summary: {output_dir / 'slot_benchmark_summary.txt'}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Large-prompt fixture generation
# ---------------------------------------------------------------------------

def generate_large_prompt_fixture(token_target: int, seed: int = 42) -> str:
    """Generate a large prompt fixture targeting approximately token_target tokens.

    Uses deterministic content generation (seeded RNG) for reproducibility.
    English prose at ~3 chars/token, with paragraph breaks for realism.

    Args:
        token_target: Target token count (e.g., 30000, 60000, 90000, 120000).
        seed: Random seed for reproducibility.

    Returns:
        A string of approximately token_target tokens of English prose.
    """
    import random
    random.seed(seed)

    # Paragraph templates for realistic content
    paragraphs = [
        "The quick brown fox jumps over the lazy dog. This classic pangram contains every letter of the English alphabet and has been used for decades in typing practice and font displays.",
        "Machine learning models require substantial computational resources to train. The attention mechanism in transformer architectures allows parallel processing of sequence data, enabling efficient training on large datasets.",
        "Data centers consume significant amounts of electricity for both computation and cooling. Modern facilities use advanced power management systems and renewable energy sources to reduce their environmental impact.",
        "Natural language processing has evolved significantly with the development of large language models. These models can understand and generate human-like text across diverse domains and languages.",
        "Cloud computing services provide scalable infrastructure for applications of all sizes. Organizations can leverage distributed computing resources without maintaining physical hardware in-house.",
        "The study of algorithms and data structures remains fundamental to computer science education. Efficient algorithms enable solutions to complex problems while minimizing resource consumption.",
        "Software engineering practices such as version control, code review, and continuous integration help maintain code quality in collaborative development environments.",
        "Network security is critical for protecting digital assets and user privacy. Firewalls, encryption, and access controls form the foundation of secure network architecture.",
        "Database systems manage large volumes of structured and unstructured data. Modern databases support distributed architectures for high availability and fault tolerance.",
        "Artificial intelligence research continues to advance rapidly, with breakthroughs in areas such as computer vision, speech recognition, and autonomous systems.",
        "The internet of things connects billions of devices worldwide, creating new opportunities for automation and data collection. Edge computing brings processing closer to data sources.",
        "Cybersecurity threats evolve continuously, requiring adaptive defense strategies. Organizations must implement comprehensive security frameworks including threat detection, incident response, and recovery planning.",
        "Quantum computing promises to solve certain problems exponentially faster than classical computers. While still in early stages, quantum algorithms for optimization and cryptography show significant potential.",
        "DevOps practices bridge the gap between development and operations teams. Automation of deployment pipelines and infrastructure management improves reliability and reduces time to market.",
        "Mobile application development requires consideration of diverse devices, operating systems, and network conditions. Responsive design and offline-first architecture improve user experience.",
    ]

    # Calculate approximate chars needed (~3 chars per token)
    chars_target = token_target * 3
    current_length = 0
    result_parts = []
    paragraph_idx = 0

    while current_length < chars_target:
        paragraph = paragraphs[paragraph_idx % len(paragraphs)]
        # Add some variation with random numbers
        num_numbers = random.randint(5, 20)
        numbers = ", ".join(str(random.randint(1, 999999)) for _ in range(num_numbers))
        content = f"{paragraph} Additional data: {numbers}."
        result_parts.append(content)
        current_length += len(content)
        paragraph_idx += 1

    return "\n\n".join(result_parts)


def load_large_prompt_fixture(fixture_path: str | Path, key: str | None = None) -> str | dict:
    """Load large-prompt fixtures from a JSON file.

    Supports two formats:
    - Top-level keys: {"30k": "...", "60k": "..."} -> returns dict when key=None
    - Nested under 'prompts': {"prompts": {"30k": "...", "60k": "..."}}

    Args:
        fixture_path: Path to the JSON fixture file.
        key: Specific key to load. If None, returns the entire prompts dict.

    Returns:
        A single prompt string (if key specified) or dict of all prompts.

    Raises:
        FileNotFoundError: If fixture_path does not exist.
        KeyError: If key is not found in the fixture.
    """
    fixture_path = Path(fixture_path)
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    data = json.loads(fixture_path.read_text())

    # Check for nested 'prompts' key first
    if "prompts" in data:
        data = data["prompts"]

    if key is None:
        return data

    if key in data:
        return data[key]
    raise KeyError(f"Key '{key}' not found in fixture file. Available: {list(data.keys())}")


def save_large_prompt_fixture(fixture_path: str | Path, prompts: dict) -> None:
    """Save large-prompt fixtures to a JSON file.

    Args:
        fixture_path: Path to save the JSON fixture file.
        prompts: Dict of prompt_name -> prompt_text.
    """
    fixture_path = Path(fixture_path)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(prompts, indent=2))


if __name__ == "__main__":
    main()
