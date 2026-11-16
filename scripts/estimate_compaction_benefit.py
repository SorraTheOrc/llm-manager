#!/usr/bin/env python3
"""Cost the session-compaction lever against the 2026-11-01 measured basis.

Read-only analysis for **LP-0MWCEGXEK000U6G6** (spike parent
LP-0MVXY8ZBZ0003042, question AC1a). It answers three questions with
arithmetic another engineer can re-run:

1. **Prefill-vs-size curve** — per-request prefill seconds at the window's
   median and top-decile prompt, under the guard's own coefficients and under
   the engine rates measured by LP-0MWCEHDRV000B7HD.
2. **Threshold costing** — for candidate prompt caps (the intake's ~57k median
   cap, the shipped compaction targets 38k/30k, and the 25k cold-prefill
   budget), how many of the window's size-refused prompts become admissible,
   and what prompt-token / stream share follows once the idle prefill gate is
   accounted for.
3. **Throughput and per-slot cost** — the rate this lever can and cannot add,
   the idle gate-seconds it converts, and the summariser slot's VRAM and GPU
   cost.

Why this shape
--------------
Compaction does not make the gate faster. It rewrites a session's prompt to a
smaller one, so the guard's *size* tests — the flat cold-prefill token cap
(``local_cold_prefill_budget_tokens: 25000``), the residency ladder above
``local_large_context_cold_cache_threshold: 38000``, and the economic warm cap
(``0.4578 x 262144 = 120000``) — are what a cap can change. A cap can only
convert requests those tests currently refuse; the resulting share is then
bounded twice over, by the gate's idle seconds in the window and by the
serialised-capacity ceiling the baseline child pinned (1.62 % token share /
5.96 % stream share).

The model is deterministic and reads only committed or measured inputs:

* the per-request **size distribution** of the window of record, reconstructed
  from ``standard_sessions.csv`` (``messages x avg_context_size`` per session,
  which reproduces the report's offered prompt tokens to ~1 %);
* the **cache-density** evidence for the refused population, measured on the
  retained 2026-11-10/11 log window (the 2026-11-01 raw logs have rotated out);
* the guard/routing **constants** read from ``proxy/config.yaml`` and
  ``proxy/proxy/prefill_eta.py``;
* the **engine rates** measured by the sibling item
  (``docs/dev/prefill-concurrency-2026-11-11.md``);
* the **measured basis** (offered/local tokens, busy seconds, capacity
  ceiling) from ``docs/dev/local-share-baseline-2026-11-01.json``.

No proxy behaviour change; no live slot is read; the proxy need not be running.

Usage::

    # regenerate the committed register from the window CSV
    python3 scripts/estimate_compaction_benefit.py \\
        --sessions-csv ~/proxy-usage-reports/postfix-thrash-gate/standard_sessions.csv \\
        --json > docs/dev/compaction-benefit-estimate-2026-11-01.json

    # re-verify the committed register (self-contained: it embeds its inputs)
    python3 scripts/estimate_compaction_benefit.py \\
        --register docs/dev/compaction-benefit-estimate-2026-11-01.json --check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

WORK_ITEM = "LP-0MWCEGXEK000U6G6"
PARENT_ITEM = "LP-0MVXY8ZBZ0003042"

# ---------------------------------------------------------------------------
# Measured basis of the window of record (2026-11-01 13:12:48 -> 14:00:03).
# Defaults mirror docs/dev/local-share-baseline-2026-11-01.json; --basis
# overrides them so the artefact and this module cannot drift.
# ---------------------------------------------------------------------------
DEFAULT_BASIS = {
    "window_seconds": 2835.0,
    "offered_prompt_tokens": 125_577_780.0,
    "offered_requests": 1788,
    "local_prompt_tokens": 877_908.0,
    "local_streams": 46,
    "local_busy_seconds": 1223.0,
    "capacity_tokens": 2_035_057.0,
    "source_report": "~/proxy-usage-reports/postfix-thrash-gate/report.md",
}

DEFAULT_SESSIONS_CSV = (
    "~/proxy-usage-reports/postfix-thrash-gate/standard_sessions.csv"
)

# ---------------------------------------------------------------------------
# Guard constants (proxy/config.yaml + proxy/proxy/prefill_eta.py).
#
#   local_cold_prefill_budget_tokens:                25000
#   local_prefill_deadline_seconds:                    180
#   local_admission_guard_budget_safety_fraction:     0.95  -> 171 s
#   contention_queue_max_wait_seconds:                  30
#   local_admission_guard_free_gate_deadline_budget:  true
#   local_admission_guard_policy:                   reject
#   DEFAULT_PREFILL_MS_PER_TOKEN:                      3.9
#   DEFAULT_CODED_MS_PER_TOKEN:                        0.5
# ---------------------------------------------------------------------------
GUARD = {
    "flat_cap_tokens": 25_000,
    "deadline_seconds": 180.0,
    "safety_budget_seconds": 171.0,  # 0.95 x 180
    "queue_budget_seconds": 30.0,
    "free_gate_rung": True,
    "ms_per_new_token": 3.9,
    "ms_per_coded_token": 0.5,
}

# ---------------------------------------------------------------------------
# Routing size tests (proxy/proxy/provider.py).
#
#   local_large_context_cold_cache_threshold:  38000  (always-local band)
#   local_large_context_warm_cache_threshold: 120000  (economic warm cap)
#   economic ceiling = round(0.4578 x 262144) = 120000
# ---------------------------------------------------------------------------
ROUTING = {
    "always_local_tokens": 38_000,
    "warm_cap_tokens": 120_000,
    "warm_cap_ratio": 0.4578,
    "local_model_ctx_size": 262_144,
    "per_slot_ctx_tokens": 131_072,
}

# ---------------------------------------------------------------------------
# Compaction constants (proxy/config.yaml, proxy/proxy/compaction.py).
# ---------------------------------------------------------------------------
COMPACTION = {
    "trigger_ratio": 0.0,  # shipped: assembly-time compaction DISABLED
    "dry_run": False,  # enforcement was already live when it was switched off
    "resolved_trigger_tokens": 91_750,  # 0.70 x 131072 if re-enabled
    "fast_target_tokens": 38_000,
    "cheap_target_tokens": 30_000,
    "remote_boundary_compaction": False,
    "remote_boundary_min_tokens": 20_000,
    "summarizer_model": "Qwen2.5-7B",
    "summarizer_quant": "Q4_K_M",
    "summarizer_parallel": 1,
    "summarizer_ctx_size": 32_768,
    "summarizer_prompt_tokens": 8_192,  # config summariser_ctx_size
    "summarizer_max_tokens": 512,
    "summarizer_weights_gb": 4.5,  # models.ini [Qwen2.5-7B] rationale
    # Qwen2.5-7B: 28 layers, 4 KV heads, head_dim 128, fp16 ->
    # 2 x 28 x 4 x 128 x 2 B = 57,344 B/token.
    "summarizer_kv_bytes_per_token": 2 * 28 * 4 * 128 * 2,
    "gpu_vram_gb": 120.0,
}

# ---------------------------------------------------------------------------
# Engine rates measured by LP-0MWCEHDRV000B7HD
# (docs/dev/prefill-concurrency-2026-11-11.md, section 2).
# ---------------------------------------------------------------------------
ENGINE = {
    "w3_solo_tok_s": 253.8,
    "bucket_20k_40k_aggregate_tok_s": 242.3,  # 241.0 / 243.5 mean
    "bucket_40k_80k_tok_s": 156.2,
    "aggregate_all_sizes_tok_s": 160.0,  # 152.8 / 166.9 mean
    # Regime mapping used to price compacted demand: the measured long-prompt
    # rate falls once a prefill exceeds ~40k tokens (memory-bandwidth bound).
    "small_regime_max_tokens": 40_000,
    "large_regime_max_tokens": 80_000,
    "small_regime_tok_s": 253.8,
    "large_regime_tok_s": 156.2,
    "huge_regime_tok_s": 160.0,
}

# ---------------------------------------------------------------------------
# Cache density of the LOCAL-REFUSED population, measured on the retained
# 2026-11-10 20:00 -> 2026-11-11 20:00 log window (see the report's provenance
# section; the 2026-11-01 raw logs have rotated out). ``reuse`` = 1 -
# cached_ratio, so reuse 1.0 means the whole prompt must be prefilled.
# ---------------------------------------------------------------------------
CACHE_DENSITY = {
    "measured_window": "2026-11-10 20:00:00 -> 2026-11-11 20:00:00 (UTC)",
    "provenance": (
        "retained /var/log/llama-proxy segments; routing_skip_local lines "
        "carry estimated_tokens + cached_ratio + reason"
    ),
    "skip_decisions": 47_722,
    "cold_share": 41_000 / 47_722,
    "cache_state_counts": {"cold": 41_000, "warm": 5_938, "restored": 784},
    "reuse_percentiles_all_skips": {
        "p10": 0.17,
        "p25": 1.0,
        "p50": 1.0,
        "p75": 1.0,
        "p90": 1.0,
    },
    # 2026-11-01 window: 413 of 417 residency denials had no local slot at all
    # (report section "Residency-aware admission").
    "window_residency_denials_without_slot": 413,
    "window_residency_denials": 417,
    "notes": (
        "The refused population is essentially cache-cold by construction: "
        "a session with no resident slot is denied before any residency "
        "measurement, and context_too_large is decided before the ladder."
    ),
}

#: Share of guard decisions taken with a free prefill gate (measured on the
#: same retained window: 0.716 of 31,958 decisions). Applies to admission
#: *attempts*, independent of prompt size.
GATE_MIX = {
    "free": 0.716,
    "busy": 1.0 - 0.716,
    "provenance": (
        "retained 2026-11-10/11 local_admission_guard lines, "
        "gate_occupancy=0 share of 31,958 decisions"
    ),
}

#: Candidate caps. ``None`` is the shipped, uncapped baseline.
CAPS: tuple[tuple[int | None, str], ...] = (
    (None, "uncapped (shipped)"),
    (57_000, "intake median cap (~57k)"),
    (38_000, "shipped fast compaction target / always-local threshold"),
    (30_000, "shipped cheap compaction target"),
    (25_000, "cold-prefill budget (local_cold_prefill_budget_tokens)"),
)

#: Refusal classes whose binding cause is prompt SIZE (policy or residency),
#: i.e. the only classes a prompt-size cap can convert. Capacity/lifecycle
#: classes (``local_concurrency_limit``, ``local_lease_active``) and the
#: unattributed bucket are excluded and keep the measured baseline.
SIZE_CLASSES = (
    "context_too_large",
    "large_context_bypass",
    "resident_prefix_insufficient",
)

#: Warm-tolerance marker: a reuse of 0.10 is the measured median of the
#: *dispatched* (admitted) population, retained as a sensitivity only.
WARM_REUSE = 0.10

REL_TOL = 1e-9


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def quantile(values: Iterable[float], pct: float) -> float:
    """Linear-interpolated *pct* (0-100) of *values*, like the proxy's own."""
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        raise ValueError("quantile of an empty sample")
    if n == 1:
        return ordered[0]
    idx = (pct / 100.0) * (n - 1)
    lower = min(n - 1, int(idx))
    upper = min(n - 1, lower + 1)
    frac = max(0.0, min(1.0, idx - lower))
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def decile_grid(values: Iterable[float]) -> list[float]:
    """Ten equal-weight representatives (p5..p95) of a size distribution.

    A grid keeps the artefact self-contained and small while preserving the
    distribution's shape and mean closely enough for threshold arithmetic:
    every cap test is a comparison against a token count, so the fraction of a
    grid above a threshold approximates the fraction of the sample above it.
    """
    sample = [float(v) for v in values]
    if not sample:
        raise ValueError("decile_grid of an empty sample")
    return [round(quantile(sample, 5 + 10 * i), 1) for i in range(10)]


def grid_mean(grid: list[float]) -> float:
    """Mean of a decile grid (equal-weight representative points)."""
    if not grid:
        raise ValueError("grid_mean of an empty grid")
    return sum(float(v) for v in grid) / len(grid)


# ---------------------------------------------------------------------------
# The pricing model
# ---------------------------------------------------------------------------
def prefill_seconds(
    tokens: float,
    reuse: float,
    ms_per_new_token: float = GUARD["ms_per_new_token"],
    ms_per_coded_token: float = GUARD["ms_per_coded_token"],
) -> float:
    """The guard's own prefill estimate for *tokens* at *reuse*.

    ``reuse`` is ``1 - cached_ratio``: the share of the prompt that must
    actually be prefilled. Mirrors
    ``proxy.proxy.prefill_eta.estimate_prefill_seconds``.
    """
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    if not 0.0 <= reuse <= 1.0:
        raise ValueError("reuse must be within [0, 1]")
    coded = tokens * (1.0 - reuse)
    new = tokens * reuse
    return (coded * ms_per_coded_token + new * ms_per_new_token) / 1000.0


def routing_verdict(tokens: float, reuse: float) -> str:
    """``admit`` / ``context_too_large`` / ``residency_required``.

    Mirrors ``_should_skip_local``: the economic warm cap is a hard size test;
    prompts at or below the always-local band bypass the residency ladder;
    above it a COLD prompt (no reusable prefix) is denied by the ladder.
    """
    if tokens > ROUTING["warm_cap_tokens"]:
        return "context_too_large"
    if tokens <= ROUTING["always_local_tokens"]:
        return "admit"
    # Residency ladder: a prompt with no reusable prefix has no slot to price
    # against, so it is denied (``resident_prefix_insufficient``).
    if reuse >= 0.999:
        return "residency_required"
    return "admit"


def guard_verdict(tokens: float, reuse: float, gate: str) -> str:
    """The admission guard's verdict for one request.

    Order mirrors ``evaluate_admission``: the FLAT cold-prefill token cap is
    checked first and is calibration-independent; then the queue-wait bound
    (a busy serialised gate projects ``eta x (busy + 1)``), then the deadline
    and its deadline-relative safety budget.
    """
    new = tokens * reuse
    if new > GUARD["flat_cap_tokens"]:
        return "prefill_budget_exceeded"
    eta = prefill_seconds(tokens, reuse)
    if gate == "free":
        # Free gate + the shipped free-gate rung: the bound is the
        # deadline-relative safety budget, not the queue budget.
        bound = (
            max(GUARD["queue_budget_seconds"], GUARD["safety_budget_seconds"])
            if GUARD["free_gate_rung"]
            else GUARD["queue_budget_seconds"]
        )
        if eta > bound:
            return "prefill_eta_exceeds_safety_budget"
        if eta >= GUARD["deadline_seconds"]:
            return "prefill_eta_exceeds_deadline"
        return "within_budget"
    # Busy serialised gate: projected pre-first-byte wait is eta x (1+1) and
    # the bound collapses to the queue budget.
    if 2.0 * eta > GUARD["queue_budget_seconds"]:
        return "eta_exceeds_budget"
    return "within_budget"


def admissible(tokens: float, reuse: float, gate: str) -> bool:
    """True when the routing ladder and the admission guard both admit."""
    if routing_verdict(tokens, reuse) != "admit":
        return False
    return guard_verdict(tokens, reuse, gate) == "within_budget"


def admission_probability(tokens: float, reuse: float, gate_mix: dict) -> float:
    """Probability of admission over the measured free/busy gate mix."""
    free = 1.0 if admissible(tokens, reuse, "free") else 0.0
    busy = 1.0 if admissible(tokens, reuse, "busy") else 0.0
    return gate_mix["free"] * free + gate_mix["busy"] * busy


def engine_rate(tokens: float) -> float:
    """Measured engine prefill rate for a prompt of *tokens* (tok/s)."""
    if tokens <= ENGINE["small_regime_max_tokens"]:
        return ENGINE["small_regime_tok_s"]
    if tokens <= ENGINE["large_regime_max_tokens"]:
        return ENGINE["large_regime_tok_s"]
    return ENGINE["huge_regime_tok_s"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_basis(path: Path | None) -> dict[str, Any]:
    """Measured-basis inputs, defaulting to the values the baseline records."""
    basis = dict(DEFAULT_BASIS)
    if path is None:
        return basis
    data = json.loads(path.read_text(encoding="utf-8"))
    measured = data.get("measured") or {}
    derived = data.get("derived") or {}
    window = data.get("window") or {}
    mapping = {
        "window_seconds": (window, "seconds"),
        "offered_prompt_tokens": (measured, "offered_prompt_tokens"),
        "offered_requests": (measured, "requests"),
        "local_prompt_tokens": (measured, "local_prompt_tokens"),
        "local_streams": (measured, "local_streams"),
        "local_busy_seconds": (measured, "local_busy_seconds"),
        "capacity_tokens": (derived, "ceiling_tokens"),
    }
    for key, (source, source_key) in mapping.items():
        if source_key in source:
            basis[key] = source[source_key]
    basis["source_report"] = data.get("report_path", basis["source_report"])
    return basis


def load_sessions(path: Path) -> list[dict[str, Any]]:
    """Read a proxy-usage-analysis ``standard_sessions.csv``."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reconstruct_sizes(
    sessions: Iterable[dict[str, Any]], reason: str | None = None
) -> list[float]:
    """Per-request prompt sizes reconstructed from session aggregates.

    Each session contributes ``messages`` requests at its ``avg_context_size``;
    the sum over the window reproduces the report's offered prompt tokens to
    about 1 %. Sessions without token evidence contribute nothing (they are
    counted, never silently treated as zero-sized).
    """
    sizes: list[float] = []
    for session in sessions:
        if reason is not None and (session.get("fallback_reason") or "") != reason:
            continue
        messages = _float(session.get("messages"))
        average = _float(session.get("avg_context_size"))
        if not messages or average is None or average <= 0:
            continue
        sizes.extend([average] * int(round(messages)))
    return sizes


def class_records(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per refusal class: session/request counts and a decile size grid."""
    records: list[dict[str, Any]] = []
    for reason in SIZE_CLASSES:
        members = [
            s for s in sessions if (s.get("fallback_reason") or "") == reason
        ]
        sizes = reconstruct_sizes(members)
        records.append(
            {
                "reason": reason,
                "sessions": len(members),
                "requests": int(round(sum(_float(s.get("messages")) or 0 for s in members))),
                "requests_with_size_evidence": len(sizes),
                "mean_tokens": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
                "size_grid": decile_grid(sizes) if sizes else [],
            }
        )
    return records


def size_distribution(sessions: list[dict[str, Any]], basis: dict[str, Any]) -> dict[str, Any]:
    """The window's reconstructed prompt-size distribution and its validation."""
    sizes = reconstruct_sizes(sessions)
    if not sizes:
        raise ValueError("no request sizes could be reconstructed")
    total = sum(sizes)
    offered = basis["offered_prompt_tokens"]
    return {
        "source": "standard_sessions.csv (messages x avg_context_size)",
        "sessions_with_size_evidence": len(
            {s.get("session_id") for s in sessions if reconstruct_sizes([s])}
        ),
        "requests_with_size_evidence": len(sizes),
        "total_tokens": round(total, 1),
        "mean_tokens": round(total / len(sizes), 1),
        "percentiles": {
            f"p{int(p)}": round(quantile(sizes, p), 1)
            for p in (10, 25, 50, 75, 90, 95, 99)
        },
        "offered_prompt_tokens": offered,
        "reconstruction_vs_offered_pct": round((total / offered - 1.0) * 100.0, 2)
        if offered
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Threshold costing
# ---------------------------------------------------------------------------
def _capped(tokens: float, cap: int | None) -> float:
    return tokens if cap is None else min(tokens, cap)


def cost_cap(
    classes: list[dict[str, Any]],
    cap: int | None,
    reuse: float,
    gate_mix: dict,
    basis: dict[str, Any],
) -> dict[str, Any]:
    """Cost one candidate cap: admission gain and the share it produces."""
    added_streams = 0.0
    added_demand_tokens = 0.0
    baseline_streams = 0.0
    population = 0.0
    for record in classes:
        grid = record["size_grid"]
        if not grid:
            continue
        weight = record["requests_with_size_evidence"] / len(grid)
        population += record["requests_with_size_evidence"]
        for tokens in grid:
            baseline = admission_probability(tokens, reuse, gate_mix)
            compacted = _capped(tokens, cap)
            with_cap = admission_probability(compacted, reuse, gate_mix)
            baseline_streams += weight * baseline
            gain = max(0.0, with_cap - baseline)
            added_streams += weight * gain
            added_demand_tokens += weight * gain * compacted
    mean_compacted = added_demand_tokens / added_streams if added_streams else 0.0
    idle_seconds = basis["window_seconds"] - basis["local_busy_seconds"]
    rate = (
        engine_rate(mean_compacted)
        if mean_compacted
        else ENGINE["small_regime_tok_s"]
    )
    demand_gate_seconds = (
        added_demand_tokens / rate if rate else 0.0
    )
    # The gate has only its idle seconds to spend on the new work: whatever the
    # demand, served volume cannot exceed idle x measured rate.
    served_streams = min(
        added_streams,
        (idle_seconds * rate) / mean_compacted if mean_compacted else 0.0,
    )
    added_tokens = served_streams * mean_compacted
    local_tokens = min(
        basis["local_prompt_tokens"] + added_tokens, basis["capacity_tokens"]
    )
    local_streams = basis["local_streams"] + served_streams
    mean_local_tokens = local_tokens / local_streams if local_streams else 0.0
    stream_ceiling = (
        basis["capacity_tokens"] / mean_local_tokens if mean_local_tokens else 0.0
    )
    local_streams = min(local_streams, stream_ceiling)
    return {
        "tokens": cap,
        "label": dict(CAPS).get(cap, str(cap)),
        "size_refused_population_requests": round(population, 1),
        "newly_admissible_share_pct": round(
            100.0 * added_streams / population, 2
        )
        if population
        else 0.0,
        "already_admissible_streams": round(baseline_streams, 1),
        "newly_admitted_streams": round(added_streams, 1),
        "mean_compacted_tokens": round(mean_compacted, 1),
        "demand_tokens": round(added_demand_tokens, 1),
        "demand_gate_seconds": round(demand_gate_seconds, 1),
        "idle_gate_seconds": round(idle_seconds, 1),
        "engine_rate_tok_s": rate,
        "served_streams": round(served_streams, 1),
        "added_tokens": round(added_tokens, 1),
        "local_prompt_tokens": round(local_tokens, 1),
        "local_token_share_pct": round(
            100.0 * local_tokens / basis["offered_prompt_tokens"], 3
        )
        if basis["offered_prompt_tokens"]
        else 0.0,
        "local_streams": round(local_streams, 1),
        "local_stream_share_pct": round(
            100.0 * local_streams / basis["offered_requests"], 3
        )
        if basis["offered_requests"]
        else 0.0,
        "capacity_ceiling_pct": round(
            100.0 * basis["capacity_tokens"] / basis["offered_prompt_tokens"], 3
        )
        if basis["offered_prompt_tokens"]
        else 0.0,
        "binding_constraint": (
            "gate idle seconds (demand clipped)"
            if added_streams > served_streams + 1e-9
            else "demand"
        ),
    }


def prefill_curve(sizes: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-request prefill seconds at the median and top-decile prompt."""
    points = (
        ("median", sizes["percentiles"]["p50"]),
        ("top decile (p90)", sizes["percentiles"]["p90"]),
    )
    curve: list[dict[str, Any]] = []
    for label, tokens in points:
        curve.append(
            {
                "point": label,
                "tokens": tokens,
                "cold_guard_default": round(prefill_seconds(tokens, 1.0), 1),
                "cold_engine_253_8_tok_s": round(tokens / ENGINE["w3_solo_tok_s"], 1),
                "cold_engine_156_2_tok_s": round(
                    tokens / ENGINE["bucket_40k_80k_tok_s"], 1
                ),
                "warm_reuse_0p10_guard_default": round(
                    prefill_seconds(tokens, WARM_REUSE), 1
                ),
                "capped_25000_cold": round(prefill_seconds(min(tokens, 25_000), 1.0), 1),
                "capped_38000_cold": round(prefill_seconds(min(tokens, 38_000), 1.0), 1),
                "capped_57000_cold": round(prefill_seconds(min(tokens, 57_000), 1.0), 1),
            }
        )
    return curve


def per_slot_cost() -> dict[str, Any]:
    """Summariser slot resources and the GPU time one compaction costs."""
    kv_bytes = COMPACTION["summarizer_kv_bytes_per_token"] * COMPACTION["summarizer_ctx_size"]
    kv_gib = kv_bytes / (1024 ** 3)
    vram = COMPACTION["summarizer_weights_gb"] + kv_gib
    return {
        "summarizer_model": COMPACTION["summarizer_model"],
        "quantisation": COMPACTION["summarizer_quant"],
        "parallel": COMPACTION["summarizer_parallel"],
        "ctx_size": COMPACTION["summarizer_ctx_size"],
        "max_output_tokens": COMPACTION["summarizer_max_tokens"],
        "weights_gb": COMPACTION["summarizer_weights_gb"],
        "kv_cache_bytes_per_token": COMPACTION["summarizer_kv_bytes_per_token"],
        "kv_cache_gib": round(kv_gib, 2),
        "total_vram_gb_estimate": round(vram, 2),
        "vram_share_pct": round(100.0 * vram / COMPACTION["gpu_vram_gb"], 1),
        "prompt_prefill_seconds_estimate": [
            round(COMPACTION["summarizer_prompt_tokens"] / 500.0, 1),
            round(COMPACTION["summarizer_prompt_tokens"] / 250.0, 1),
        ],
        "generation_seconds_estimate": [
            round(COMPACTION["summarizer_max_tokens"] / 60.0, 1),
            round(COMPACTION["summarizer_max_tokens"] / 40.0, 1),
        ],
        "gpu_seconds_per_compaction_estimate": [
            round(COMPACTION["summarizer_prompt_tokens"] / 500.0
                  + COMPACTION["summarizer_max_tokens"] / 60.0, 1),
            round(COMPACTION["summarizer_prompt_tokens"] / 250.0
                  + COMPACTION["summarizer_max_tokens"] / 40.0, 1),
        ],
        "documented_typical_latency": "typically < 2 s (models.ini [Qwen2.5-7B])",
        "notes": (
            "The summariser runs on the same GPU as the main model, so its "
            "GPU-seconds compete with the serialised prefill gate; VRAM is "
            "held for the lifetime of the server (parallel=1, ctx 32768)."
        ),
    }


# ---------------------------------------------------------------------------
# Register assembly
# ---------------------------------------------------------------------------
def build_register(
    sessions: list[dict[str, Any]],
    basis: dict[str, Any],
    gate_mix: dict = GATE_MIX,
) -> dict[str, Any]:
    """Assemble the full machine-readable register."""
    sizes = size_distribution(sessions, basis)
    classes = class_records(sessions)
    thresholds = [
        cost_cap(classes, cap, 1.0, gate_mix, basis) for cap, _ in CAPS
    ]
    cold_thresholds = [dict(entry) for entry in thresholds]
    warm_thresholds = [
        cost_cap(classes, cap, WARM_REUSE, gate_mix, basis) for cap, _ in CAPS
    ]
    baseline = cold_thresholds[0]
    for entry in cold_thresholds + warm_thresholds:
        entry["token_share_delta_pp"] = round(
            entry["local_token_share_pct"] - baseline["local_token_share_pct"], 3
        )
        entry["stream_share_delta_pp"] = round(
            entry["local_stream_share_pct"] - baseline["local_stream_share_pct"], 3
        )
    idle_seconds = basis["window_seconds"] - basis["local_busy_seconds"]
    register = {
        "schema_version": SCHEMA_VERSION,
        "work_item": WORK_ITEM,
        "parent_item": PARENT_ITEM,
        "purpose": (
            "Cost the proxy-side session-compaction lever against the "
            "2026-11-01 measured basis, for the spike's AC1a option table."
        ),
        "read_only": True,
        "window": {
            "start": "2026-11-01 13:12:48",
            "end": "2026-11-01 14:00:03",
            "seconds": basis["window_seconds"],
            "source_report": basis["source_report"],
        },
        "basis": {
            key: basis[key]
            for key in (
                "window_seconds",
                "offered_prompt_tokens",
                "offered_requests",
                "local_prompt_tokens",
                "local_streams",
                "local_busy_seconds",
                "capacity_tokens",
            )
        },
        "inputs": {
            "size_distribution": sizes,
            "cache_density": CACHE_DENSITY,
            "gate_mix": gate_mix,
            "guard": GUARD,
            "routing": ROUTING,
            "compaction": COMPACTION,
            "engine": ENGINE,
        },
        "classes": classes,
        "prefill_curve": prefill_curve(sizes),
        "thresholds_cold": cold_thresholds,
        "thresholds_warm_sensitivity": warm_thresholds,
        "throughput": {
            "shipped_gate_rate_tok_s": ENGINE["w3_solo_tok_s"],
            "blended_delivered_tok_s": round(
                basis["local_prompt_tokens"] / basis["local_busy_seconds"], 1
            ),
            "cold_prefill_rate_ratio": round(
                (basis["local_prompt_tokens"] / basis["local_busy_seconds"])
                / ENGINE["w3_solo_tok_s"],
                2,
            ),
            "rate_added_by_lever_tok_s": 0.0,
            "idle_gate_seconds": round(idle_seconds, 1),
            "idle_gate_share_pct": round(
                100.0 * idle_seconds / basis["window_seconds"], 1
            ),
            "attributable_delivered_tok_s_cold_25k": round(
                cold_thresholds[-1]["added_tokens"] / basis["window_seconds"], 1
            ),
            "notes": (
                "The lever adds no prefill rate: it converts idle gate-seconds "
                "into served tokens by making large cold prompts admissible. "
                "Delivered-rate attribution is the converted volume divided by "
                "the window. Displacement caveat: a cold token costs ~"
                f"{(basis['local_prompt_tokens'] / basis['local_busy_seconds']) / ENGINE['w3_solo_tok_s']:.2f}x "
                "the gate time of a blended delivered token, so the gain is "
                "real only while gate time is idle; in a saturated window the "
                "same substitution would lower the delivered token share."
            ),
        },
        "per_slot_cost": per_slot_cost(),
        "compaction_state": {
            "assembly_time_trigger_ratio": COMPACTION["trigger_ratio"],
            "assembly_time_enabled": COMPACTION["trigger_ratio"] > 0,
            "dry_run": COMPACTION["dry_run"],
            "resolved_trigger_tokens_if_enabled": COMPACTION["resolved_trigger_tokens"],
            "targets_tokens": {
                "fast": COMPACTION["fast_target_tokens"],
                "cheap": COMPACTION["cheap_target_tokens"],
            },
            "remote_boundary_compaction": COMPACTION["remote_boundary_compaction"],
            "remote_boundary_min_tokens": COMPACTION["remote_boundary_min_tokens"],
            "summarizer_model": COMPACTION["summarizer_model"],
            "summarizer_parallel": COMPACTION["summarizer_parallel"],
            "notes": (
                "Two independent triggers. Assembly-time compaction "
                "(compaction_trigger_ratio) is the one that shrinks a LOCAL "
                "prompt and is therefore the lever costed here; it ships "
                "disabled (0). Boundary compaction only shrinks the history "
                "sent to a REMOTE provider and ships false, so it does not "
                "change local prefill."
            ),
        },
        "verdicts": verdicts(cold_thresholds, warm_thresholds, sizes, classes, basis),
    }
    register["checks"] = checks(register)
    return register


def _cold_cap(entries: list[dict[str, Any]], tokens: int) -> dict[str, Any]:
    for entry in entries:
        if entry["tokens"] == tokens:
            return entry
    raise KeyError(tokens)


def verdicts(
    cold: list[dict[str, Any]],
    warm: list[dict[str, Any]],
    sizes: dict[str, Any],
    classes: list[dict[str, Any]],
    basis: dict[str, Any],
) -> dict[str, Any]:
    """The arithmetic verdicts the spike's comparison table can quote."""
    cap_25 = _cold_cap(cold, 25_000)
    cap_30 = _cold_cap(cold, 30_000)
    cap_38 = _cold_cap(cold, 38_000)
    cap_57 = _cold_cap(cold, 57_000)
    busy_ceiling = GUARD["queue_budget_seconds"] / 2.0 * 1000.0 / GUARD["ms_per_new_token"]
    size_refused = sum(r["requests"] for r in classes)
    return {
        "flat_cap_is_the_binding_size_test": (
            "The cold refused population is cache-cold, so new_tokens equals "
            "the prompt size. A cap therefore clears the flat "
            "local_cold_prefill_budget_tokens=25000 test only when the cap "
            "itself is <= 25000: caps at 30k/38k/57k change prefill cost but "
            "admit no cold prompt."
        ),
        "large_prompt_refusal_is_size_bound": (
            f"The three size-caused classes (context_too_large, "
            f"large_context_bypass, resident_prefix_insufficient) are "
            f"{size_refused} of {basis['offered_requests']} requests in the "
            f"window (sessions: "
            + ", ".join(f"{r['reason']} {r['sessions']}" for r in classes)
            + "); capacity/lease classes are not convertible by a size cap."
        ),
        "token_share_upside": {
            "uncapped_pct": _cold_cap(cold, None)["local_token_share_pct"],
            "cap_25000_pct": cap_25["local_token_share_pct"],
            "cap_25000_delta_pp": cap_25["token_share_delta_pp"],
            "cap_57000_pct": cap_57["local_token_share_pct"],
            "capacity_ceiling_pct": cap_25["capacity_ceiling_pct"],
            "binding_constraint": cap_25["binding_constraint"],
            "added_tokens": cap_25["added_tokens"],
            "demand_gate_seconds": cap_25["demand_gate_seconds"],
            "idle_gate_seconds": cap_25["idle_gate_seconds"],
        },
        "shipped_targets_do_not_convert": {
            "cap_57000_newly_admitted": cap_57["newly_admitted_streams"],
            "cap_38000_newly_admitted": cap_38["newly_admitted_streams"],
            "cap_30000_newly_admitted": cap_30["newly_admitted_streams"],
            "cap_25000_newly_admitted": cap_25["newly_admitted_streams"],
            "cap_25000_served_streams_gate_bound": cap_25["served_streams"],
            "note": (
                "The shipped fast/cheap targets (38k/30k) sit above the flat "
                "cold-prefill budget, so enabling assembly-time compaction "
                "without lowering the target would pay the summariser cost "
                "without admitting a cold prompt. Even at 25k the gain is "
                "gate-bound: the demand exceeds the window's idle gate "
                "seconds, so only a fraction of the newly admissible "
                "requests can be served."
            ),
        },
        "busy_gate_ceiling_tokens": round(busy_ceiling, 0),
        "warm_population_gain": {
            "cap_25000_pct": _cold_cap(warm, 25_000)["local_token_share_pct"],
            "cap_57000_pct": _cold_cap(warm, 57_000)["local_token_share_pct"],
            "note": (
                "Sensitivity only: the admitted (dispatched) population is "
                "90 % cache-warm, and warm prompts are already admissible at "
                "sizes far above any candidate cap, so a cap adds little "
                "there."
            ),
        },
        "size_distribution": {
            "median_tokens": sizes["percentiles"]["p50"],
            "top_decile_tokens": sizes["percentiles"]["p90"],
            "mean_tokens": sizes["mean_tokens"],
            "intake_figures": "median ~57k, top decile >100k",
        },
    }


def _close(recorded: float, recomputed: float, rel: float = REL_TOL, absolute: float = 0.0) -> bool:
    """True when two numbers agree inside ``rel`` (or ``absolute``)."""
    difference = abs(float(recorded) - float(recomputed))
    return difference <= absolute + rel * max(1.0, abs(float(recorded)))


def checks(register: dict[str, Any]) -> list[dict[str, Any]]:
    """Self-consistency claims re-verified by ``--check``.

    Every entry records the number the register claims next to the number
    recomputed from its own inputs, so a failure names the drifted quantity.
    """
    basis = register["basis"]
    sizes = register["inputs"]["size_distribution"]
    cold = register["thresholds_cold"]
    curve = register["prefill_curve"]
    results: list[dict[str, Any]] = []

    def add(
        claim: str,
        recorded: Any,
        recomputed: Any,
        ok: bool,
        tolerance: str,
    ) -> None:
        results.append(
            {
                "claim": claim,
                "recorded": recorded,
                "recomputed": recomputed,
                "tolerance": tolerance,
                "ok": bool(ok),
            }
        )

    # 1. The reconstruction reproduces the report's offered prompt tokens.
    recomputed_pct = round(
        (sizes["total_tokens"] / basis["offered_prompt_tokens"] - 1.0) * 100.0, 2
    )
    add(
        "reconstructed size total within 3 % of offered prompt tokens",
        sizes["reconstruction_vs_offered_pct"],
        recomputed_pct,
        abs(recomputed_pct) <= 3.0,
        "+/-3 % relative",
    )
    # 2. The measured basis reproduces the baseline shares the spike quotes.
    add(
        "baseline local token share reproduces 0.699 %",
        0.699,
        round(100.0 * basis["local_prompt_tokens"] / basis["offered_prompt_tokens"], 3),
        _close(0.699, 100.0 * basis["local_prompt_tokens"] / basis["offered_prompt_tokens"], absolute=0.002),
        "+/-0.002 pp",
    )
    add(
        "baseline local stream share reproduces 2.573 %",
        2.573,
        round(100.0 * basis["local_streams"] / basis["offered_requests"], 3),
        _close(2.573, 100.0 * basis["local_streams"] / basis["offered_requests"], absolute=0.002),
        "+/-0.002 pp",
    )
    # 3. No threshold may claim more than the measured capacity ceiling.
    for entry in cold:
        add(
            f"token share at cap {entry['tokens']} never exceeds the capacity ceiling",
            entry["local_token_share_pct"],
            entry["capacity_ceiling_pct"],
            entry["local_token_share_pct"] <= entry["capacity_ceiling_pct"] + 1e-9,
            "clamped at basis.ceiling_tokens",
        )
    # 4. The guard's flat-cap boundary: 25 000 new tokens is admissible,
    #    25 001 is not (the guard refuses on >, not >=).
    add(
        "flat cap admits exactly 25 000 cold new tokens",
        "within_budget",
        guard_verdict(GUARD["flat_cap_tokens"], 1.0, "free"),
        guard_verdict(GUARD["flat_cap_tokens"], 1.0, "free") == "within_budget",
        "guard refuses on >, not >=",
    )
    add(
        "flat cap refuses 25 001 cold new tokens",
        "prefill_budget_exceeded",
        guard_verdict(GUARD["flat_cap_tokens"] + 1, 1.0, "free"),
        guard_verdict(GUARD["flat_cap_tokens"] + 1, 1.0, "free")
        == "prefill_budget_exceeded",
        "guard refuses on >, not >=",
    )
    # 5. The busy-gate admission ceiling follows the queue-budget arithmetic.
    recomputed_ceiling = round(
        GUARD["queue_budget_seconds"] / 2.0 * 1000.0 / GUARD["ms_per_new_token"], 0
    )
    add(
        "busy-gate ceiling = queue budget / 2 at the measured coefficients",
        register["verdicts"]["busy_gate_ceiling_tokens"],
        recomputed_ceiling,
        _close(
            register["verdicts"]["busy_gate_ceiling_tokens"], recomputed_ceiling
        ),
        "exact",
    )
    # 6. The prefill curve is linear in tokens at a fixed cache state.
    point = curve[0]
    slope = point["cold_guard_default"] / point["tokens"]
    add(
        "cold prefill seconds scale linearly with prompt size",
        round(slope, 6),
        round(GUARD["ms_per_new_token"] / 1000.0, 6),
        _close(slope, GUARD["ms_per_new_token"] / 1000.0, absolute=1e-6),
        "exact",
    )
    # 7. A cap can never reduce local served volume.
    volumes = [entry["local_prompt_tokens"] for entry in cold]
    add(
        "capping never reduces served local prompt tokens",
        volumes[0],
        min(volumes),
        min(volumes) >= volumes[0] - 1e-6,
        "monotone",
    )
    return results


def all_ok(register: dict[str, Any]) -> bool:
    return all(entry["ok"] for entry in register.get("checks") or [])


# ---------------------------------------------------------------------------
# Re-verification of a committed register (self-contained)
# ---------------------------------------------------------------------------
def recheck_register(register: dict[str, Any]) -> dict[str, Any]:
    """Re-derive every headline figure from the register's own inputs.

    The register embeds the per-class decile size grids and every constant, so
    a committed artefact can be re-verified after the source CSV (which lives
    outside the repo) has rotated away. Any drift between the recorded figures
    and the recomputation is reported as a failed check.
    """
    rebuilt = build_register_from_register(register)
    drift: list[dict[str, Any]] = []
    for section in ("thresholds_cold", "prefill_curve"):
        recorded = register.get(section)
        recomputed = rebuilt.get(section)
        if recorded != recomputed:
            drift.append(
                {
                    "section": section,
                    "recorded": recorded,
                    "recomputed": recomputed,
                }
            )
    return {
        "ok": not drift,
        "drift": drift,
        "recorded_checks_ok": all_ok(register),
    }


def build_register_from_register(register: dict[str, Any]) -> dict[str, Any]:
    """Recompute the costed sections of *register* from its embedded inputs."""
    basis = register["basis"]
    gate_mix = register["inputs"]["gate_mix"]
    classes = register["classes"]
    cold = [cost_cap(classes, cap, 1.0, gate_mix, basis) for cap, _ in CAPS]
    warm = [cost_cap(classes, cap, WARM_REUSE, gate_mix, basis) for cap, _ in CAPS]
    baseline = cold[0]
    for entry in cold + warm:
        entry["token_share_delta_pp"] = round(
            entry["local_token_share_pct"] - baseline["local_token_share_pct"], 3
        )
        entry["stream_share_delta_pp"] = round(
            entry["local_stream_share_pct"] - baseline["local_stream_share_pct"], 3
        )
    return {
        "thresholds_cold": cold,
        "thresholds_warm_sensitivity": warm,
        "prefill_curve": prefill_curve(register["inputs"]["size_distribution"]),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _f(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render(register: dict[str, Any]) -> str:
    """Human-readable summary of the register."""
    lines: list[str] = []
    sizes = register["inputs"]["size_distribution"]
    basis = register["basis"]
    lines.append("Compaction-benefit estimate — 2026-11-01 measured basis")
    lines.append("")
    lines.append(
        f"Window: {sizes['requests_with_size_evidence']} sized requests "
        f"(mean {_f(sizes['mean_tokens'])} tokens, p50 "
        f"{_f(sizes['percentiles']['p50'])}, p90 "
        f"{_f(sizes['percentiles']['p90'])}); offered "
        f"{_f(basis['offered_prompt_tokens'])} prompt tokens."
    )
    lines.append("")
    lines.append("Prefill curve (per request, guard coefficients):")
    for point in register["prefill_curve"]:
        lines.append(
            f"  {point['point']:<18} {_f(point['tokens']):>10} tok  cold "
            f"{_f(point['cold_guard_default']):>9} s   warm(reuse 0.10) "
            f"{_f(point['warm_reuse_0p10_guard_default']):>8} s   "
            f"capped@25k cold {_f(point['capped_25000_cold'])} s"
        )
    lines.append("")
    lines.append(
        "Candidate caps (cold refused population; gate mix "
        f"free {register['inputs']['gate_mix']['free']:.1%}):"
    )
    lines.append(
        "  cap        new-admissible  served   added tokens   token share   stream share"
    )
    for entry in register["thresholds_cold"]:
        cap = entry["tokens"] if entry["tokens"] is not None else "none"
        lines.append(
            f"  {str(cap):<9} {entry['newly_admitted_streams']:>13.1f} "
            f"{entry['served_streams']:>7.1f} {_f(entry['added_tokens']):>14} "
            f"{entry['local_token_share_pct']:>11.3f} % "
            f"{entry['local_stream_share_pct']:>11.3f} %"
        )
    lines.append("")
    verdict = register["verdicts"]["shipped_targets_do_not_convert"]
    lines.append(
        "Verdict: only a cap at or below the 25 000-token cold-prefill budget "
        f"converts cold prompts (25k -> {verdict['cap_25000_newly_admitted']:.1f} "
        f"newly admissible, {verdict['cap_25000_served_streams_gate_bound']:.1f} "
        f"served; 30k -> {verdict['cap_30000_newly_admitted']:.1f}; "
        f"38k -> {verdict['cap_38000_newly_admitted']:.1f}; "
        f"57k -> {verdict['cap_57000_newly_admitted']:.1f})."
    )
    busy = register["verdicts"]["busy_gate_ceiling_tokens"]
    lines.append(
        f"Behind a busy serialised gate the admission ceiling is "
        f"{_f(busy)} cold tokens (queue budget 30 s / 2)."
    )
    upside = register["verdicts"]["token_share_upside"]
    lines.append(
        f"Gate-bound upside at 25k: +{_f(upside['added_tokens'])} tokens "
        f"({upside['demand_gate_seconds']:,.0f} s of demand against "
        f"{_f(upside['idle_gate_seconds'])} idle gate-seconds); "
        f"token share {upside['uncapped_pct']:.3f} % -> "
        f"{upside['cap_25000_pct']:.3f} %."
    )
    if not all_ok(register):
        lines.append("")
        lines.append("CHECKS FAILED:")
        for entry in register["checks"]:
            if not entry["ok"]:
                lines.append(f"  - {entry['claim']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cost the session-compaction lever against the 2026-11-01 "
            "measured basis (read-only analysis)."
        )
    )
    parser.add_argument(
        "--sessions-csv",
        type=Path,
        help="proxy-usage-analysis standard_sessions.csv for the window",
    )
    parser.add_argument(
        "--basis",
        type=Path,
        help="local-share basis JSON artefact (defaults to its recorded values)",
    )
    parser.add_argument(
        "--register",
        type=Path,
        help="committed register to re-verify (self-contained)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when a recorded figure no longer reproduces",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.register is not None:
        try:
            register = json.loads(args.register.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"error: cannot read register {args.register}: {exc}", file=sys.stderr)
            return 2
        recheck = recheck_register(register)
        if args.json:
            print(json.dumps(recheck, indent=2))
        else:
            print(
                f"{args.register}: "
                + ("reproduces" if recheck["ok"] else "DRIFT")
                + f" ({len(recheck['drift'])} drifted section(s); "
                f"recorded checks {'ok' if recheck['recorded_checks_ok'] else 'FAILED'})"
            )
            for entry in recheck["drift"]:
                print(f"  drift: {entry['section']}")
        return 0 if recheck["ok"] and recheck["recorded_checks_ok"] else 1

    csv_path = args.sessions_csv or Path(DEFAULT_SESSIONS_CSV).expanduser()
    try:
        sessions = load_sessions(csv_path)
    except OSError as exc:
        print(f"error: cannot read sessions CSV {csv_path}: {exc}", file=sys.stderr)
        return 2
    try:
        basis = load_basis(args.basis)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read basis {args.basis}: {exc}", file=sys.stderr)
        return 2
    try:
        register = build_register(sessions, basis)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(register, indent=2))
    else:
        print(render(register))

    if args.check and not all_ok(register):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
