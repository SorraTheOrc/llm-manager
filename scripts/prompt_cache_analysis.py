#!/usr/bin/env python3
"""Reproducible llama-server prompt-cache analysis (LP-0MUPPAC7R0053J8L).

Measures the metrics that define the 8 GiB prompt-cache saturation problem and
the value that fixes it:

  * ``--cache-ram`` (MiB) — llama-server's in-memory prompt cache cap
    (default 8192). When the cap is hit, ``server_prompt_cache::update()``
    evicts the oldest cached prompt; the evicted session's next turn is then a
    full prefill of its whole context.
  * ``--ctx-checkpoints`` / ``--checkpoint-every-n-tokens`` — hybrid
    (attention + SSM/recurrent) models keep per-slot context checkpoints; with
    the default 32 checkpoints at ~62.8 MiB each that is ~2 GiB of state per
    cached prompt, the dominant term in per-prompt size.

The metric set mirrors the AC1 baseline so a before/after comparison is
apples-to-apples:

  requests, prompt tokens requested, tokens reused from cache, tokens
  prefilled, full-prefill requests split into small (< --large-tokens, new
  session — expected) and large (>= --large-tokens — lost prefix), prefill
  wall time, eviction count + sizes, cache-saturation samples and checkpoint
  overhead.

AC6 (LP-0MUYFBVHX004Z0UE) adds the throughput / latency metrics that must
not regress:

  * prompt-eval and decode (generation) throughput, from the llama-server
    ``prompt eval time`` / ``eval time`` lines (tok/s); and
  * client-visible first-byte latency, from the proxy's
    ``dispatch_first_byte_ms=`` lines (``--proxy-log-dir``, default the same
    directory as the llama-server logs).

Usage:
  ./scripts/prompt_cache_analysis.py                          # summary
  ./scripts/prompt_cache_analysis.py --json                   # machine readable
  ./scripts/prompt_cache_analysis.py --glob 'llama-server.log-2026-09-30.gz'
  ./scripts/prompt_cache_analysis.py --glob 'llama-server.log,llama-server.*.log'
  ./scripts/prompt_cache_analysis.py --cache-ram-filter 23552 # post-deploy corpus
  ./scripts/prompt_cache_analysis.py --recommend              # sizing proposal
  ./scripts/prompt_cache_analysis.py --assume-cache-ram-mib 24576
  ./scripts/prompt_cache_analysis.py --json > after.json      # baseline artifact

Exit codes:
  0 - success
  1 - no llama-server logs found / unexpected error
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# The llama-server log iterator (live + dot/dash rotated, plain/.gz) and the
# shared prompt-eval/checkpoint regexes live in the F1 harness
# (LP-0MTCMEJX2008W85X); reuse them so the corpus stays consistent.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import slot_persistence_harness as harness  # noqa: E402
from lib.proxy_logs import (  # noqa: E402
    discover_proxy_log_files,
    open_proxy_log_text,
)

MIB = 1024 * 1024

# --- llama-server prompts -------------------------------------------------
_NEW_PROMPT_RE = re.compile(
    r"slot update_slots: id\s+(?P<slot>\d+) \| task (?P<task>\d+) \| "
    r"new prompt, n_ctx_slot = (?P<ctx>\d+), n_keep = (?P<keep>\d+), "
    r"task\.n_tokens = (?P<tokens>\d+)"
)
_PROGRESS_RE = re.compile(
    r"slot update_slots: id\s+(?P<slot>\d+) \| task (?P<task>\d+) \| "
    r"prompt processing progress, n_tokens = (?P<n_tokens>\d+), "
    r"batch\.n_tokens = (?P<batch>\d+)"
)
_DONE_RE = re.compile(
    r"slot update_slots: id\s+(?P<slot>\d+) \| task (?P<task>\d+) \| "
    r"prompt processing done, n_tokens = (?P<n_tokens>\d+), "
    r"batch\.n_tokens = (?P<batch>\d+)"
)
_FORCE_RE = re.compile(
    r"slot update_slots: id\s+(?P<slot>\d+) \| task (?P<task>\d+) \| "
    r"forcing full prompt re-processing"
)
# llama-server eval-timing lines, e.g.
#   [32999] prompt eval time = 29504.01 ms / 11449 tokens ( 2.58 ms per token, 388.05 tokens per second)
#   [32999]        eval time =  3776.71 ms /   153 tokens (24.68 ms per token,  40.51 tokens per second)
# The alternation is ordered so ``prompt eval time`` wins over ``eval time``
# at its own (earlier) position.
_EVAL_RE = re.compile(
    r"(?P<kind>prompt eval time|eval time)\s*=\s+"
    r"(?P<ms>[0-9.]+) ms /\s+(?P<tokens>\d+) tokens \("
    r"\s*(?P<ms_per_tok>[0-9.]+) ms per token,\s*(?P<tok_s>[0-9.]+) tokens per second\)"
)
# proxy.log: client-visible first-byte latency (LP-0MTJET616005S7PN).
_FIRST_BYTE_RE = re.compile(r"dispatch_first_byte_ms=(?P<ms>[0-9.]+)")
_CKPT_RE = harness._LLAMA_CHECKPOINT_RE

# --- cache bookkeeping ----------------------------------------------------
# [37623] srv        update:  - cache state: 6 prompts, 6644.301 MiB (limits: 8192.000 MiB, 262656 tokens, 393209 est)
_CACHE_STATE_RE = re.compile(
    r"cache state: (?P<prompts>\d+) prompts, (?P<size>[0-9.]+) MiB "
    r"\(limits: (?P<limit>[0-9.]+) MiB, (?P<tokens>\d+) tokens, (?P<est>\d+) est\)"
)
# [37623] srv        update:  - cache size limit reached, removing oldest entry (size = 314.916 MiB)
_EVICT_RE = re.compile(
    r"cache size limit reached, removing oldest entry \(size = (?P<size>[0-9.]+) (?P<unit>[MGT])iB\)"
)
_UNIT_MIB = {"M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}

_DEFAULT_LARGE_TOKENS = 10_000
_DEFAULT_LIMIT_MIB = 8192.0


@dataclass
class _Task:
    """Prompt processing state for one (pid, slot, task)."""

    total: int
    prefilled: int = 0
    forced: bool = False
    finalised: bool = False


@dataclass
class _Metrics:
    requests: int = 0
    prompt_tokens_requested: int = 0
    tokens_prefilled: int = 0
    full_prefill_requests: int = 0
    small_full_prefill_requests: int = 0
    small_full_prefill_tokens: int = 0
    large_full_prefill_requests: int = 0
    large_full_prefill_tokens: int = 0
    forced_full_prefills: int = 0
    prefill_wall_ms: float = 0.0
    prefill_eval_lines: int = 0
    prompt_eval_tok_s: list[float] = field(default_factory=list)
    decode_tok_s: list[float] = field(default_factory=list)
    eviction_sizes_mib: list[float] = field(default_factory=list)
    cache_states: int = 0
    cache_limit_mib: float = 0.0
    cache_peak_mib: float = 0.0
    cache_saturated: int = 0
    cache_prompts_max: int = 0
    checkpoint_counts: Counter = field(default_factory=Counter)
    checkpoint_sizes_mib: list[float] = field(default_factory=list)
    checkpoint_per_prompt: list[int] = field(default_factory=list)
    files: int = 0


@dataclass
class _ProxyMetrics:
    """Proxy-side metrics (AC6): client-visible first-byte latency."""

    first_byte_ms: list[float] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def _finalise(metrics: _Metrics, task: _Task, large_tokens: int) -> None:
    if task.finalised:
        return
    task.finalised = True
    metrics.requests += 1
    metrics.prompt_tokens_requested += task.total
    metrics.tokens_prefilled += task.prefilled
    # "0 reused" — the whole prompt was recomputed.
    if task.prefilled >= task.total:
        metrics.full_prefill_requests += 1
        if task.forced:
            metrics.forced_full_prefills += 1
        if task.total < large_tokens:
            metrics.small_full_prefill_requests += 1
            metrics.small_full_prefill_tokens += task.total
        else:
            metrics.large_full_prefill_requests += 1
            metrics.large_full_prefill_tokens += task.total


def _read_llama_lines(path: Path):
    """Yield lines from one llama-server log, gzip-aware."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        yield from fh


def _file_declares_cache_limit(path: Path, limit_mib: float) -> bool:
    """True when *path* has a cache-state line with the given cap (MiB)."""
    needle = f"limits: {limit_mib:.3f} MiB"
    try:
        for line in _read_llama_lines(path):
            if needle in line:
                return True
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
    return False


def _select_llama_files(
    log_dir: Path,
    patterns: list[str] | None,
    cache_ram_filter: float | None,
) -> list[Path]:
    """Resolve the llama-server corpus for the run.

    *patterns* keeps files matching any glob (empty = all). *cache_ram_filter*
    keeps only files whose cache-state lines declare that ``--cache-ram`` limit
    — llama-server logs carry no timestamps, so the cap is the reproducible way
    to isolate a post-deploy corpus.
    """
    import fnmatch

    files = harness._iter_llama_files(log_dir)
    if patterns:
        files = [p for p in files if any(fnmatch.fnmatch(p.name, pat) for pat in patterns)]
    if cache_ram_filter is not None:
        files = [p for p in files if _file_declares_cache_limit(p, cache_ram_filter)]
    return files


def collect(
    log_dir: Path,
    patterns: list[str] | None = None,
    large_tokens: int = _DEFAULT_LARGE_TOKENS,
    cache_ram_filter: float | None = None,
) -> _Metrics:
    """Parse llama-server logs and return the aggregate metric set."""
    metrics = _Metrics()
    tasks: dict[tuple[str, str, str], _Task] = {}
    ckpt_per_task: Counter = Counter()

    files = _select_llama_files(log_dir, patterns, cache_ram_filter)
    metrics.files = len(files)

    for path in files:
        for line in _read_llama_lines(path):
            m = _NEW_PROMPT_RE.search(line)
            if m:
                key = _pid_key(line, m)
                # Finalise a stale task on the same slot before overwriting.
                _finalise_slot(metrics, tasks, line, m.group("slot"), large_tokens)
                tasks[key] = _Task(total=int(m.group("tokens")))
                continue

            m = _PROGRESS_RE.search(line) or _DONE_RE.search(line)
            if m:
                key = _pid_key(line, m)
                task = tasks.get(key)
                if task is None:
                    # Processing without a preceding "new prompt" (e.g. a
                    # continuation or a log rotation seam): track it anyway.
                    task = _Task(total=int(m.group("n_tokens")))
                    tasks[key] = task
                task.prefilled += int(m.group("batch"))
                continue

            m = _FORCE_RE.search(line)
            if m:
                task = tasks.get(_pid_key(line, m))
                if task is not None:
                    task.forced = True
                continue

            m = _EVAL_RE.search(line)
            if m:
                if m.group("kind") == "prompt eval time":
                    metrics.prefill_wall_ms += float(m.group("ms"))
                    metrics.prefill_eval_lines += 1
                    metrics.prompt_eval_tok_s.append(float(m.group("tok_s")))
                else:
                    metrics.decode_tok_s.append(float(m.group("tok_s")))
                continue

            m = _EVICT_RE.search(line)
            if m:
                metrics.eviction_sizes_mib.append(float(m.group("size")) * _UNIT_MIB[m.group("unit")])
                continue

            m = _CACHE_STATE_RE.search(line)
            if m:
                metrics.cache_states += 1
                size = float(m.group("size"))
                limit = float(m.group("limit"))
                metrics.cache_limit_mib = limit or metrics.cache_limit_mib
                metrics.cache_peak_mib = max(metrics.cache_peak_mib, size)
                metrics.cache_prompts_max = max(metrics.cache_prompts_max, int(m.group("prompts")))
                if limit > 0 and size >= 0.9 * limit:
                    metrics.cache_saturated += 1
                continue

            m = _CKPT_RE.search(line)
            if m:
                metrics.checkpoint_sizes_mib.append(float(m.group("size")))
                metrics.checkpoint_counts[int(m.group("total"))] += 1
                ckpt_per_task[_pid_key(line, m)] += 1
                continue

    # Finalise everything still open at end of corpus.
    for task in tasks.values():
        _finalise(metrics, task, large_tokens)
    metrics.checkpoint_per_prompt = list(ckpt_per_task.values())
    return metrics


def collect_proxy(proxy_log_dir: Path, patterns: list[str] | None = None) -> _ProxyMetrics:
    """Parse proxy logs for client-visible first-byte latency (AC6).

    A missing/unreadable directory yields empty stats rather than an error, so
    a llama-only corpus still runs. *patterns* restricts the proxy files by
    glob (any match) so the proxy window can be aligned with the llama corpus
    — proxy logs are rotated far more often, so the default full-history span
    would otherwise be much wider than a filtered llama-server corpus.
    """
    import fnmatch

    metrics = _ProxyMetrics()
    if not proxy_log_dir.is_dir():
        return metrics
    files = discover_proxy_log_files(proxy_log_dir)
    if patterns:
        files = [p for p in files if any(fnmatch.fnmatch(p.name, pat) for pat in patterns)]
    seen: set[str] = set()
    for path in files:
        seen.add(str(path))
        try:
            with open_proxy_log_text(path) as fh:
                for line in fh:
                    m = _FIRST_BYTE_RE.search(line)
                    if m:
                        metrics.first_byte_ms.append(float(m.group("ms")))
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
    metrics.files = sorted(seen)
    return metrics


def _pid_key(line: str, match: re.Match) -> tuple[str, str, str]:
    pid_m = re.match(r"^\[(?P<pid>\d+)\]", line)
    pid = pid_m.group("pid") if pid_m else "?"
    return (pid, match.group("slot"), match.group("task"))


def _finalise_slot(metrics: _Metrics, tasks: dict, line: str, slot: str, large_tokens: int) -> None:
    pid_m = re.match(r"^\[(?P<pid>\d+)\]", line)
    pid = pid_m.group("pid") if pid_m else "?"
    for key, task in tasks.items():
        if key[0] == pid and key[1] == slot and not task.finalised:
            _finalise(metrics, task, large_tokens)


# ---------------------------------------------------------------------------
# Derived statistics + recommendation
# ---------------------------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stats(values: list[float]) -> dict:
    """Median/p10/p90 summary for a sample list (empty -> zeros)."""
    return {
        "samples": len(values),
        "median": round(_percentile(values, 50), 1) if values else 0.0,
        "p10": round(_percentile(values, 10), 1) if values else 0.0,
        "p90": round(_percentile(values, 90), 1) if values else 0.0,
    }


def summarise(
    metrics: _Metrics,
    large_tokens: int,
    proxy_metrics: _ProxyMetrics | None = None,
) -> dict:
    """Turn raw parse output into the comparable metric summary."""
    proxy_metrics = proxy_metrics or _ProxyMetrics()
    requested = metrics.prompt_tokens_requested
    prefilled = metrics.tokens_prefilled
    reused = max(0, requested - prefilled)
    reuse_pct = (100.0 * reused / requested) if requested else 0.0
    ev = metrics.eviction_sizes_mib
    ckpt = metrics.checkpoint_sizes_mib
    ckpt_total = max(metrics.checkpoint_counts) if metrics.checkpoint_counts else 0
    return {
        "requests": metrics.requests,
        "prompt_tokens_requested": requested,
        "tokens_reused": reused,
        "reuse_pct": round(reuse_pct, 2),
        "tokens_prefilled": prefilled,
        "full_prefill_requests": metrics.full_prefill_requests,
        "forced_full_prefills": metrics.forced_full_prefills,
        "small_full_prefill_requests": metrics.small_full_prefill_requests,
        "small_full_prefill_tokens": metrics.small_full_prefill_tokens,
        "large_full_prefill_requests": metrics.large_full_prefill_requests,
        "large_full_prefill_tokens": metrics.large_full_prefill_tokens,
        "large_full_prefill_share_of_prefill_pct": round(
            100.0 * metrics.large_full_prefill_tokens / prefilled, 2
        ) if prefilled else 0.0,
        "large_tokens_threshold": large_tokens,
        "prefill_wall_seconds": round(metrics.prefill_wall_ms / 1000.0, 1),
        "prefill_eval_lines": metrics.prefill_eval_lines,
        "evictions": len(ev),
        "eviction_mean_mib": round(_mean(ev), 1),
        "eviction_max_mib": round(max(ev), 1) if ev else 0.0,
        "cache_states": metrics.cache_states,
        "cache_limit_mib": metrics.cache_limit_mib or _DEFAULT_LIMIT_MIB,
        "cache_peak_mib": round(metrics.cache_peak_mib, 1),
        "cache_saturated_samples": metrics.cache_saturated,
        "cache_saturated_pct": round(
            100.0 * metrics.cache_saturated / metrics.cache_states, 2
        ) if metrics.cache_states else 0.0,
        "cache_prompts_max": metrics.cache_prompts_max,
        "checkpoint_size_mib": round(_mean(ckpt), 3) if ckpt else 0.0,
        "checkpoints_per_slot_max": ckpt_total,
        "checkpoint_overhead_per_prompt_mib": round(ckpt_total * _mean(ckpt), 1) if ckpt else 0.0,
        "checkpoints_observed": len(ckpt),
        "prompt_eval_tok_s": _stats(metrics.prompt_eval_tok_s),
        "decode_tok_s": _stats(metrics.decode_tok_s),
        "first_byte_ms": _stats(proxy_metrics.first_byte_ms),
        "files": metrics.files,
    }


def _meminfo() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
                    out[parts[0].rstrip(":")] = float(parts[1]) / 1024.0  # KiB -> MiB
    except OSError:
        pass
    return out


def recommend(
    summary: dict,
    *,
    target_prompts: int | None = None,
    host_total_mib: float | None = None,
    host_available_mib: float | None = None,
    reserve_mib: float | None = None,
    margin: float = 1.25,
) -> dict:
    """Derive a ``--cache-ram`` (and checkpoint) proposal from the measurement.

    AC2 — the value is derived from measured per-prompt state size and the
    number of prompts that must be covered, not from a round number.
    AC3 — the checkpoint contribution is measured and the upstream
    ``~100 MB per checkpoint`` rule is cross-checked; a ``--ctx-checkpoints``
    value is proposed with its trade-off stated.
    AC4 — host headroom is checked with an explicit reserve margin and the
    OOM fallback path (``limit_size = 0.4 * size()`` after a failed alloc) is
    called out as the failure mode to avoid.
    """
    mem = _meminfo()
    total = host_total_mib if host_total_mib is not None else mem.get("MemTotal", 0.0)
    avail = host_available_mib if host_available_mib is not None else mem.get("MemAvailable", 0.0)
    # Safety margin that must remain after the cache is allocated. MemAvailable
    # already excludes the currently-loaded model + KV + other backends, so the
    # margin covers other backends loading later and OS page-cache pressure.
    # Defaults to ~10% of total RAM (>= 8 GiB).
    if reserve_mib is None:
        reserve_mib = max(8192.0, total * 0.1) if total else 8192.0
    if target_prompts is None:
        target_prompts = max(summary.get("cache_prompts_max", 0), 8)

    ev = summary["eviction_mean_mib"]
    ckpt_overhead = summary["checkpoint_overhead_per_prompt_mib"]
    # The evicted entries are exactly the large prompts we want to retain, so
    # their measured mean is the best per-prompt estimate; fall back to the
    # checkpoint overhead (an upper bound on the largest contexts).
    if ev > 0:
        per_prompt, source = ev, "eviction_size"
    elif ckpt_overhead > 0:
        per_prompt, source = ckpt_overhead, "checkpoint_overhead"
    else:
        per_prompt, source = 1024.0, "fallback_1gib"

    raw_cache = per_prompt * target_prompts * margin
    # Round up to the next whole GiB so the value is stable across reruns.
    cache_mib = math.ceil(raw_cache / 1024.0) * 1024

    # Upstream guidance: size the cache as ~ctx_checkpoints x ~100 MB.
    checkpoints_per_slot = summary["checkpoints_per_slot_max"] or 32
    upstream_per_prompt = checkpoints_per_slot * 100.0

    projected_free = (avail - cache_mib) if avail else 0.0
    safe = (projected_free >= reserve_mib) if avail else None

    return {
        "per_prompt_state_mib": round(per_prompt, 1),
        "per_prompt_source": source,
        "target_prompts": target_prompts,
        "margin": margin,
        "observed_max_prompts": summary.get("cache_prompts_max", 0),
        "holdable_prompts": int(cache_mib // per_prompt) if per_prompt else 0,
        "raw_cache_mib": round(raw_cache, 1),
        "recommended_cache_ram_mib": int(cache_mib),
        "recommended_cache_ram_gib": round(cache_mib / 1024.0, 2),
        "checkpoints_per_slot": checkpoints_per_slot,
        "checkpoint_size_mib": summary["checkpoint_size_mib"],
        "checkpoint_overhead_per_prompt_mib": ckpt_overhead,
        "upstream_rule_per_prompt_mib": upstream_per_prompt,
        "host_total_mib": round(total, 1),
        "host_available_mib": round(avail, 1),
        "reserve_mib": round(reserve_mib, 1),
        "projected_free_after_cache_mib": round(projected_free, 1),
        "headroom_safe": safe,
        "oom_fallback": (
            "server_prompt_cache::alloc() sets limit_size = 0.4 * size() when "
            "the allocation fails; entering that path silently caps the cache "
            "below the configured value — size so it is never reached."
        ),
        "note": (
            "Per-prompt size is dominated by context checkpoints "
            f"({checkpoints_per_slot} x {summary['checkpoint_size_mib']} MiB "
            f"= {ckpt_overhead} MiB). Reducing --ctx-checkpoints saves memory "
            "but coarsens rollback granularity; on a hybrid/recurrent model "
            "fewer checkpoints can itself increase full prefills, so the "
            "primary lever is --cache-ram and the checkpoint count is kept "
            "unless the host cannot afford the derived cache."
        ),
    }


def project_capacity(summary: dict, assume_cache_mib: float) -> dict:
    """Project avoided evictions / full prefills at a larger cache cap.

    The projection is a capacity model: the extra room admits ``holdable``
    prompts (``assume / per_prompt``); evictions avoided scale with how close
    that is to the observed maximum number of concurrently-retained prompts.
    Assumption (stated in the output): each avoided eviction preserves one
    session prefix and saves one lost-prefix full prefill.
    """
    limit = summary["cache_limit_mib"] or _DEFAULT_LIMIT_MIB
    extra = max(0.0, assume_cache_mib - limit)
    ev = summary["evictions"]
    ev_mean = summary["eviction_mean_mib"] or 0.0
    observed_max = summary.get("cache_prompts_max", 0)
    base = {
        "assumed_cache_ram_mib": assume_cache_mib,
        "extra_capacity_mib": round(extra, 1),
        "holdable_prompts": int(assume_cache_mib // ev_mean) if ev_mean else 0,
        "observed_max_prompts": observed_max,
        "capacity_adequate": bool(ev_mean and observed_max and assume_cache_mib // ev_mean >= observed_max),
    }
    if ev == 0 or ev_mean <= 0:
        return {**base, "projected_evictions_avoided": 0,
                "projected_evictions_avoided_pct": 0.0,
                "projected_large_full_prefills_avoided": 0,
                "projected_prefill_tokens_saved": 0}
    if base["capacity_adequate"]:
        avoided = ev
    else:
        ratio = min(1.0, base["holdable_prompts"] / observed_max) if observed_max else 0.0
        avoided = int(round(ev * ratio))
    large = summary["large_full_prefill_requests"]
    large_avoided = min(large, avoided)
    mean_large = (summary["large_full_prefill_tokens"] / large) if large else 0
    return {
        **base,
        "projected_evictions_avoided": avoided,
        "projected_evictions_avoided_pct": round(100.0 * avoided / ev, 1),
        "projected_large_full_prefills_avoided": large_avoided,
        "projected_prefill_tokens_saved": int(large_avoided * mean_large),
        "assumption": "one avoided eviction = one avoided lost-prefix full prefill",
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_stats(stats: dict) -> str:
    if not stats.get("samples"):
        return "n/a"
    return (
        f"{stats['median']:.1f} (p10 {stats['p10']:.1f} / p90 {stats['p90']:.1f}, "
        f"n={stats['samples']})"
    )


def render_summary(summary: dict, recommendation: dict | None, projection: dict | None) -> str:
    lines = [
        "llama-server prompt-cache analysis",
        "=" * 34,
        f"log files                     : {summary['files']}",
        f"requests                      : {summary['requests']:,}",
        f"prompt tokens requested       : {summary['prompt_tokens_requested']:,}",
        f"tokens reused from cache      : {summary['tokens_reused']:,} ({summary['reuse_pct']}%)",
        f"tokens prefilled              : {summary['tokens_prefilled']:,}",
        f"full-prefill requests (0 reuse): {summary['full_prefill_requests']:,}",
        f"  small (<{summary['large_tokens_threshold']:,})       : "
        f"{summary['small_full_prefill_requests']:,} reqs / {summary['small_full_prefill_tokens']:,} tokens",
        f"  large (>={summary['large_tokens_threshold']:,})      : "
        f"{summary['large_full_prefill_requests']:,} reqs / {summary['large_full_prefill_tokens']:,} tokens",
        f"  forced by llama.cpp         : {summary['forced_full_prefills']:,}",
        f"large share of prefill work   : {summary['large_full_prefill_share_of_prefill_pct']}%",
        f"prompt-eval wall time         : {summary['prefill_wall_seconds']:,} s",
        f"cache limit / peak            : {summary['cache_limit_mib']:.0f} / {summary['cache_peak_mib']:.0f} MiB",
        f"cache-saturated samples       : {summary['cache_saturated_samples']:,} / "
        f"{summary['cache_states']:,} ({summary['cache_saturated_pct']}%)",
        f"max prompts held              : {summary['cache_prompts_max']}",
        f"evictions                     : {summary['evictions']:,} "
        f"(mean {summary['eviction_mean_mib']:.0f} MiB, max {summary['eviction_max_mib']:.0f} MiB)",
        f"checkpoints                   : {summary['checkpoints_per_slot_max']} x "
        f"{summary['checkpoint_size_mib']} MiB = "
        f"{summary['checkpoint_overhead_per_prompt_mib']:.0f} MiB/prompt",
        f"prompt-eval throughput        : {_fmt_stats(summary['prompt_eval_tok_s'])} tok/s",
        f"decode throughput             : {_fmt_stats(summary['decode_tok_s'])} tok/s",
        f"first-byte latency (proxy)    : {_fmt_stats(summary['first_byte_ms'])} ms",
    ]
    if recommendation:
        r = recommendation
        lines += [
            "",
            "sizing recommendation (derived)",
            "-" * 34,
            f"per-prompt state              : {r['per_prompt_state_mib']:.0f} MiB ({r['per_prompt_source']})",
            f"target prompts to hold        : {r['target_prompts']} "
            f"(observed max {r['observed_max_prompts']}, margin x{r['margin']})",
            f"recommended --cache-ram       : {r['recommended_cache_ram_mib']} MiB "
            f"({r['recommended_cache_ram_gib']} GiB, holds ~{r['holdable_prompts']} prompts)",
            f"checkpoint overhead           : {r['checkpoint_overhead_per_prompt_mib']:.0f} MiB "
            f"(upstream rule {r['upstream_rule_per_prompt_mib']:.0f} MiB)",
            f"host available / reserve      : {r['host_available_mib']:.0f} / {r['reserve_mib']:.0f} MiB",
            f"headroom safe                 : {r['headroom_safe']}",
        ]
    if projection:
        p = projection
        lines += [
            "",
            f"projection @ {p['assumed_cache_ram_mib']:.0f} MiB",
            "-" * 34,
            f"extra capacity                : {p['extra_capacity_mib']:.0f} MiB",
            f"holdable prompts              : {p['holdable_prompts']} "
            f"(observed max {p['observed_max_prompts']}, adequate={p['capacity_adequate']})",
            f"evictions avoided             : {p['projected_evictions_avoided']} "
            f"({p['projected_evictions_avoided_pct']}%)",
            f"large full prefills avoided   : {p['projected_large_full_prefills_avoided']}",
            f"prefill tokens saved (proj.)  : {p['projected_prefill_tokens_saved']:,}",
        ]
    return "\n".join(lines)


def _split_patterns(raw: str | None) -> list[str]:
    """Split a comma-separated ``--glob`` value into non-empty patterns."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _file_list(log_dir: Path, patterns: list[str], cache_ram_filter: float | None) -> list[str]:
    """Names of the llama-server files the run actually measured."""
    return [p.name for p in _select_llama_files(log_dir, patterns, cache_ram_filter)]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", default="/var/log/llama-proxy", type=Path)
    p.add_argument("--glob", default=None,
                   help="filename glob restricting llama-server files; comma-separated = any match")
    p.add_argument("--cache-ram-filter", type=float, default=None,
                   help="keep only files whose cache-state lines declare this --cache-ram cap (MiB)")
    p.add_argument("--proxy-log-dir", type=Path, default=None,
                   help="directory with proxy.log* for first-byte latency (default: --log-dir)")
    p.add_argument("--proxy-glob", default=None,
                   help="filename glob(s) restricting proxy.log* files; comma-separated = any match")
    p.add_argument("--no-proxy", action="store_true",
                   help="skip proxy-log parsing (first-byte stats stay empty)")
    p.add_argument("--large-tokens", type=int, default=_DEFAULT_LARGE_TOKENS,
                   help="prompt-length threshold separating new-session from lost-prefix prefills")
    p.add_argument("--json", action="store_true")
    p.add_argument("--recommend", action="store_true", help="derive a sizing proposal")
    p.add_argument("--target-prompts", type=int, default=None,
                   help="prompts to hold (default: observed max, floor 8)")
    p.add_argument("--assume-cache-ram-mib", type=float, default=None,
                   help="project avoided evictions/prefills at this cap")
    p.add_argument("--host-total-mib", type=float, default=None, help="override MemTotal (tests)")
    p.add_argument("--host-available-mib", type=float, default=None, help="override MemAvailable (tests)")
    p.add_argument("--reserve-mib", type=float, default=None, help="host RAM reserved for model+KV+other")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.log_dir.is_dir():
        print(f"error: log directory not found: {args.log_dir}", file=sys.stderr)
        return 1
    patterns = _split_patterns(args.glob)
    metrics = collect(args.log_dir, patterns, args.large_tokens, args.cache_ram_filter)
    if metrics.files == 0:
        print(f"error: no llama-server logs found in {args.log_dir}", file=sys.stderr)
        return 1
    proxy_dir = args.proxy_log_dir or args.log_dir
    proxy_patterns = _split_patterns(args.proxy_glob)
    proxy_metrics = (
        _ProxyMetrics() if args.no_proxy else collect_proxy(proxy_dir, proxy_patterns)
    )
    summary = summarise(metrics, args.large_tokens, proxy_metrics)

    recommendation = None
    if args.recommend:
        recommendation = recommend(
            summary,
            target_prompts=args.target_prompts,
            host_total_mib=args.host_total_mib,
            host_available_mib=args.host_available_mib,
            reserve_mib=args.reserve_mib,
        )
    projection = None
    if args.assume_cache_ram_mib is not None:
        projection = project_capacity(summary, args.assume_cache_ram_mib)

    if args.json:
        payload = {
            "meta": {
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "log_dir": str(args.log_dir),
                "glob": args.glob,
                "globs": patterns,
                "cache_ram_filter_mib": args.cache_ram_filter,
                "proxy_log_dir": str(proxy_dir),
                "proxy_glob": args.proxy_glob,
                "proxy_globs": proxy_patterns,
                "proxy_files": [Path(p).name for p in proxy_metrics.files],
                "large_tokens_threshold": args.large_tokens,
                "files": _file_list(args.log_dir, patterns, args.cache_ram_filter),
            },
            "summary": summary,
        }
        if recommendation:
            payload["recommendation"] = recommendation
        if projection:
            payload["projection"] = projection
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print(render_summary(summary, recommendation, projection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
