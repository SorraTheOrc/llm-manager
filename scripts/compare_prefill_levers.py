#!/usr/bin/env python3
"""Assemble the costed four-family lever comparison and the share target
(LP-0MWCEII0V003V76A).

Spike parent LP-0MVXY8ZBZ0003042, AC1 + AC2 (synthesis child). This is a *pure
assembly* analyser: it reads the five committed registers produced by the
spike's verification and lever children, puts their rows side by side on the
same two denominators, and computes what a candidate stream-share target costs
in gate time. It never changes proxy behaviour, never reads a live slot, and
never takes a new measurement — every figure it prints is traceable to one of
the source registers named in ``SOURCE_PATHS``.

Why the tool exists
-------------------
The levers were measured in four separate children, each reporting in its own
units: one in converted idle gate-seconds, one in aggregate tok/s, one in
GiB and decode latency, one in a ceiling band. A decision needs them in *one*
unit set:

- **throughput (tok/s)** the lever actually adds to the prefill gate;
- **per-slot resource cost** it occupies for that throughput;
- **estimated impact on local share**, quoted on the denominator the lever
  actually moves (a tok/s lever multiplies tokens; a stream target divides the
  same capacity by the current local prompt size — see below).

The two denominators are never mixed. LP-0MWCEGJDP009WURD pinned them:

- **token share** = local prompt tokens / offered prompt tokens
  (actual 0.699 %, point ceiling 1.621 %);
- **stream share** = locally served streams / offered streams
  (actual 2.573 %, point ceiling 5.964 %).

They differ by the prompt-size mix (offered streams average 70 234 tokens,
local streams 19 085), so one stream-share point is worth ~3.68 token-share
points. A comparison that swaps one for the other is wrong by that factor.

The share target
----------------
Parent LP-0MVWEZCB0006WPBQ AC9 asks for a local share "materially above" its
2.8 % baseline without a number. The target block below converts a candidate
stream share into: the delivered prompt tokens it needs, its multiple of the
point capacity ceiling, the cache-reuse fraction it would require at the point
engine rate, and the prefill gate-seconds it costs — both under the window's
own cache mix and in the pessimistic case where every newly served token is
cold. A target that fits inside the window's *idle* gate seconds in the
pessimistic case needs no lever at all; one that does not is a throughput
target in disguise.

Usage::

    python3 scripts/compare_prefill_levers.py --json
    python3 scripts/compare_prefill_levers.py
    python3 scripts/compare_prefill_levers.py \\
        --register docs/dev/local-prefill-lever-comparison-2026-11-11.json --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

WORK_ITEM = "LP-0MWCEII0V003V76A"
SPIKE_WORK_ITEM = "LP-0MVXY8ZBZ0003042"
TARGET_ITEM = "LP-0MVWEZCB0006WPBQ"

#: Committed registers this tool reads, relative to the repository root.
SOURCE_PATHS = {
    "local_share_baseline": "docs/dev/local-share-baseline-2026-11-01.json",
    "single_slot_throughput": "docs/dev/single-slot-throughput-baseline-2026-10-31.json",
    "prefill_concurrency": "docs/dev/prefill-concurrency-baseline-2026-11-11.json",
    "compaction": "docs/dev/compaction-benefit-estimate-2026-11-01.json",
    "quantisation": "docs/dev/quantisation-cost-2026-11-11.json",
    "accept_ceiling": "docs/dev/accept-ceiling-arithmetic-2026-11-01.json",
}

#: The quantisation candidates carried into the comparison, low then high.
QUANT_CANDIDATES = ("ud_q4_k_m_estimated", "ud_q4_k_s_measured")
#: The prompt-size bucket the local stream mix occupies (mean 19 085 tokens).
QUANT_BUCKET = "20k-40k"

#: The share target this item recommends for the parent's AC9, as a percentage
#: of offered streams. Chosen from the arithmetic below, not by preference:
#: 1.25x the AC9 baseline, 0.587x the point stream ceiling, and reachable
#: inside the window's idle prefill gate even if every added token is cold.
RECOMMENDED_STREAM_SHARE_PCT = 3.5

#: Candidate stream shares shown in the target table, with their labels.
TARGET_CANDIDATES: tuple[tuple[float, str], ...] = (
    (2.5727, "measured (2026-11-01 window)"),
    (2.8, "parent AC9 baseline"),
    (3.0, "candidate"),
    (RECOMMENDED_STREAM_SHARE_PCT, "recommended"),
    (3.7, "prior producer-facing recommendation (LP-0MVXWQQ20002HAX4)"),
    (4.0, "candidate"),
    (5.0, "candidate"),
    (5.9637, "point stream ceiling"),
    (10.0, "prior >=10 % proposal"),
)

#: Tolerances. Chosen from the precision the source registers carry, never from
#: what makes a check pass.
TOL_RATIO = 0.05          # 5 % relative on measured ratios
TOL_PP = 0.05             # 0.05 pp on share deltas
TOL_RECONCILE_REL = 0.02  # 2 % on cross-register reconciliation


class RegisterError(RuntimeError):
    """A source register is missing a block this tool derives from."""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _read(root: Path, key: str) -> dict[str, Any]:
    path = root / SOURCE_PATHS[key]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise RegisterError(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise RegisterError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RegisterError(f"{path} does not contain a JSON object")
    return data


def _need(block: Any, key: str, where: str) -> Any:
    if not isinstance(block, dict) or key not in block:
        raise RegisterError(f"{where} is missing {key!r}")
    return block[key]


def _figure(register: dict[str, Any], key: str, where: str) -> float:
    """Read a ``figures``/``derived`` entry's numeric ``value``."""
    entry = _need(register, key, where)
    value = _need(entry, "value", f"{where}.{key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RegisterError(f"{where}.{key}.value is not numeric: {value!r}") from exc


def load_inputs(root: Path) -> dict[str, Any]:
    """Read every figure the comparison quotes, with its provenance block.

    Only the blocks this tool derives from are carried, so the register is
    self-contained for ``--register … --check`` and cannot silently depend on a
    field that a later edit of a source register removes.
    """
    baseline = _read(root, "local_share_baseline")
    single = _read(root, "single_slot_throughput")
    concurrency = _read(root, "prefill_concurrency")
    compaction = _read(root, "compaction")
    quantisation = _read(root, "quantisation")
    accept = _read(root, "accept_ceiling")

    measured = _need(baseline, "measured", "local_share_baseline")
    baseline_derived = _need(baseline, "derived", "local_share_baseline")
    window = _need(baseline, "window", "local_share_baseline")

    accept_derived = _need(accept, "derived", "accept_ceiling")
    ceiling = _need(accept_derived, "ceiling", "accept_ceiling.derived")
    reuse = _need(accept_derived, "reuse_fractions", "accept_ceiling.derived")
    targets = _need(accept_derived, "targets", "accept_ceiling.derived")
    levers_bound = _need(accept, "levers_bound", "accept_ceiling")
    accept_option = _need(accept, "accept_option", "accept_ceiling")

    compaction_throughput = _need(compaction, "throughput", "compaction")
    compaction_cost = _need(compaction, "per_slot_cost", "compaction")
    compaction_state = _need(compaction, "compaction_state", "compaction")
    cold_rows = _need(compaction, "thresholds_cold", "compaction")
    if not isinstance(cold_rows, list) or not cold_rows:
        raise RegisterError("compaction.thresholds_cold is not a non-empty list")
    cold_by_cap = {row.get("tokens"): row for row in cold_rows if isinstance(row, dict)}

    quant_candidates = _need(quantisation, "candidates", "quantisation")
    quant_capacity_inputs = _need(
        _need(quantisation, "inputs", "quantisation"), "capacity", "quantisation.inputs"
    )
    quant_slot_inputs = _need(
        _need(quantisation, "inputs", "quantisation"),
        "per_slot_resources",
        "quantisation.inputs",
    )

    shipped_config = _need(concurrency, "shipped_config", "prefill_concurrency")
    concurrency_figures_block = _need(concurrency, "figures", "prefill_concurrency")
    concurrency_derived_block = _need(concurrency, "derived", "prefill_concurrency")
    concurrency_figures = {
        key: _figure(concurrency_figures_block, key, "prefill_concurrency.figures")
        for key in (
            "E1.prompt_tokens",
            "E1.c1_aggregate_tps_median",
            "E1.c1_aggregate_tps_max",
            "E1.c2_aggregate_tps_median",
            "E1.c2_aggregate_tps_min",
            "E1.c1_per_request_p50_median",
            "E1.c2_per_request_min_median",
            "E1.per_checkpoint_mib",
            "E1.scheduling_cost_live_local_first_byte_s",
            "P1.bucket_20_40k_aggregate_tps",
            "P2.bucket_20_40k_aggregate_tps",
            "P2.aggregate_tps",
        )
    }
    concurrency_derived = {
        key: _figure(concurrency_derived_block, key, "prefill_concurrency.derived")
        for key in (
            "D1.aggregate_ratio_median",
            "D2.window_factor_median",
            "D3.per_request_ratio_median",
            "D4.single_gate_daily_tokens_at_40k",
            "D5.parallel_daily_tokens_measured_at_40k",
            "D6.daily_tokens_at_20_40k",
            "D7.per_slot_checkpoint_gib",
        )
    }

    single_figures = _need(single, "figures", "single_slot")
    single_derived = _need(single, "derived", "single_slot")
    geometry = {
        "steal_pct_pool1_two_sessions": _figure(
            single_figures, "W7.steal_pct_pool1_two_sessions", "single_slot.figures"
        ),
        "worst_case_extra_reprefill_tokens": _figure(
            single_derived, "D9.worst_case_extra_reprefill_tokens", "single_slot.derived"
        ),
        "reprefill_pool_increase_pct": _figure(
            single_derived, "D10.reprefill_pool_increase_pct", "single_slot.derived"
        ),
        "warm_ceiling_one_slot_with_cochange": _figure(
            single_derived, "D16.warm_ceiling_arm_b_with_cochange", "single_slot.derived"
        ),
        "warm_ceiling_one_slot_without_cochange": _figure(
            single_derived, "D17.warm_ceiling_arm_b_without_cochange", "single_slot.derived"
        ),
        "pool_size": _need(
            _need(single, "geometry", "single_slot"), "arm_a", "single_slot.geometry"
        ).get("session_slot_pool_size"),
    }

    def candidate(name: str) -> dict[str, Any]:
        block = _need(quant_candidates, name, "quantisation.candidates")
        bucket = _need(_need(block, "prefill", f"candidates.{name}"), QUANT_BUCKET, name)
        capacity = _need(block, "capacity", f"candidates.{name}")
        decode = _need(block, "decode", f"candidates.{name}")
        return {
            "label": name,
            "weights_gib": _need(block, "gib", name),
            "delta_vs_shipped_gib": _need(block, "delta_vs_shipped_gib", name),
            "provenance": _need(block, "provenance", name),
            "prefill_gain_ceiling_pct": _need(bucket, "prefill_gain_ceiling_pct", name),
            "prefill_gain_measured_share_pct": _need(
                bucket, "prefill_gain_measured_share_pct", name
            ),
            "prefill_tokens_per_second_measured": _need(
                bucket, "prefill_tokens_per_second_measured", name
            ),
            "prefill_tokens_per_second_after": _need(
                bucket, "prefill_tokens_per_second_measured_share", name
            ),
            "weight_share_pct_of_pass": _need(bucket, "weight_share_pct", name),
            "decode_byte_ratio_bound_pct": _need(decode, "byte_ratio_bound_pct", name),
            "decode_tokens_per_second_ceiling": _need(
                decode, "tokens_per_second_ceiling", name
            ),
            "decode_slot_seconds_freed": _need(decode, "slot_seconds_freed", name),
            "local_token_share_before_pct": _need(capacity, "local_token_share_before_pct", name),
            "local_token_share_after_pct": _need(capacity, "local_token_share_after_pct", name),
            "local_stream_share_before_pct": _need(capacity, "local_stream_share_before_pct", name),
            "local_stream_share_after_pct": _need(capacity, "local_stream_share_after_pct", name),
            "free_slot_seconds_gain_pct": _need(capacity, "free_slot_seconds_gain_pct", name),
        }

    return {
        "window": {
            "start": _need(window, "start", "local_share_baseline.window"),
            "end": _need(window, "end", "local_share_baseline.window"),
            "seconds": _need(window, "seconds", "local_share_baseline.window"),
            "source": SOURCE_PATHS["local_share_baseline"],
        },
        "measured": {
            "offered_prompt_tokens": _need(measured, "offered_prompt_tokens", "measured"),
            "offered_streams": _need(measured, "requests", "measured"),
            "local_streams": _need(measured, "local_streams", "measured"),
            "local_prompt_tokens": _need(measured, "local_prompt_tokens", "measured"),
            "local_busy_seconds": _need(measured, "local_busy_seconds", "measured"),
            "local_completion_tokens": _need(measured, "local_completion_tokens", "measured"),
            "local_prompt_tokens_per_stream": _need(
                baseline_derived, "local_prompt_tokens_per_stream", "derived"
            ),
            "offered_prompt_tokens_per_stream": _need(
                baseline_derived, "offered_prompt_tokens_per_stream", "derived"
            ),
            "local_token_share_pct": _need(baseline_derived, "local_token_share_pct", "derived"),
            "local_stream_share_pct": _need(
                baseline_derived, "local_stream_share_pct", "derived"
            ),
            "canonical_ceiling_tokens": _need(baseline_derived, "ceiling_tokens", "derived"),
            "canonical_token_share_pct": _need(
                baseline_derived, "ceiling_token_share_pct", "derived"
            ),
            "canonical_stream_share_pct": _need(
                baseline_derived, "ceiling_stream_share_pct", "derived"
            ),
            "source": SOURCE_PATHS["local_share_baseline"],
        },
        "ceiling": {
            name: {
                "engine_rate_tok_s": _need(ceiling[name], "engine_rate_tok_s", name),
                "reuse_fraction": _need(ceiling[name], "reuse_fraction", name),
                "prefill_capacity_tokens": _need(ceiling[name], "prefill_capacity_tokens", name),
                "delivered_tokens": _need(ceiling[name], "delivered_tokens", name),
                "token_share_pct": _need(ceiling[name], "token_share_pct", name),
                "stream_share_pct": _need(ceiling[name], "stream_share_pct", name),
            }
            for name in ("min", "point", "max")
        },
        "reuse_fractions": {
            name: _need(reuse, name, "reuse_fractions") for name in ("min", "point", "max")
        },
        "prefill_seconds_available": _need(
            accept_derived, "prefill_seconds_available", "accept_ceiling.derived"
        ),
        "parent_targets": {
            "ac9_stream_pct": _need(
                targets["parent_ac9_stream"], "target_pct", "targets.parent_ac9_stream"
            ),
            "ac9_factor_vs_point": _need(
                targets["parent_ac9_stream"],
                "factor_vs_point_capacity",
                "targets.parent_ac9_stream",
            ),
            "ac1_stream_pct": _need(
                targets["spike_ac1_stream"], "target_pct", "targets.spike_ac1_stream"
            ),
            "ac1_factor_vs_point": _need(
                targets["spike_ac1_stream"],
                "factor_vs_point_capacity",
                "targets.spike_ac1_stream",
            ),
        },
        "compaction": {
            "work_item": "LP-0MWCEGXEK000U6G6",
            "gate_rate_tok_s": _need(compaction_throughput, "shipped_gate_rate_tok_s", "compaction"),
            "rate_added_by_lever_tok_s": _need(
                compaction_throughput, "rate_added_by_lever_tok_s", "compaction"
            ),
            "attributable_delivered_tok_s": _need(
                compaction_throughput, "attributable_delivered_tok_s_cold_25k", "compaction"
            ),
            "idle_gate_seconds": _need(compaction_throughput, "idle_gate_seconds", "compaction"),
            "idle_gate_share_pct": _need(
                compaction_throughput, "idle_gate_share_pct", "compaction"
            ),
            "cold_vs_blended_gate_cost_ratio": _need(
                compaction_throughput, "cold_prefill_rate_ratio", "compaction"
            ),
            "summariser_model": _need(compaction_cost, "summarizer_model", "per_slot_cost"),
            "summariser_quantisation": _need(compaction_cost, "quantisation", "per_slot_cost"),
            "summariser_parallel": _need(compaction_cost, "parallel", "per_slot_cost"),
            "summariser_ctx_size": _need(compaction_cost, "ctx_size", "per_slot_cost"),
            "summariser_weights_gb": _need(compaction_cost, "weights_gb", "per_slot_cost"),
            "summariser_kv_cache_gib": _need(compaction_cost, "kv_cache_gib", "per_slot_cost"),
            "summariser_total_vram_gb": _need(
                compaction_cost, "total_vram_gb_estimate", "per_slot_cost"
            ),
            "summariser_vram_share_pct": _need(compaction_cost, "vram_share_pct", "per_slot_cost"),
            "gpu_seconds_per_compaction": _need(
                compaction_cost, "gpu_seconds_per_compaction_estimate", "per_slot_cost"
            ),
            "trigger_ratio": _need(compaction_state, "assembly_time_trigger_ratio", "compaction_state"),
            "fast_target_tokens": _need(compaction_state, "targets_tokens", "compaction_state").get(
                "fast"
            ),
            "cheap_target_tokens": _need(compaction_state, "targets_tokens", "compaction_state").get(
                "cheap"
            ),
            "cold_rows": {
                str(cap): {
                    "label": cold_by_cap[cap].get("label"),
                    "added_tokens": cold_by_cap[cap].get("added_tokens"),
                    "served_streams": cold_by_cap[cap].get("served_streams"),
                    "local_token_share_pct": cold_by_cap[cap].get("local_token_share_pct"),
                    "local_stream_share_pct": cold_by_cap[cap].get("local_stream_share_pct"),
                    "token_share_delta_pp": cold_by_cap[cap].get("token_share_delta_pp"),
                    "stream_share_delta_pp": cold_by_cap[cap].get("stream_share_delta_pp"),
                    "demand_gate_seconds": cold_by_cap[cap].get("demand_gate_seconds"),
                    "idle_gate_seconds": cold_by_cap[cap].get("idle_gate_seconds"),
                    "binding_constraint": cold_by_cap[cap].get("binding_constraint"),
                }
                for cap in sorted(cold_by_cap, key=lambda c: (c is None, c))
                if cap is not None
            },
            "source": SOURCE_PATHS["compaction"],
        },
        "parallel_prefill": {
            "work_item": "LP-0MWCEHDRV000B7HD",
            "shipped_max_concurrent_prefills": _need(
                shipped_config, "local_max_concurrent_prefills", "shipped_config"
            ),
            "per_slot_ctx": _need(shipped_config, "per_slot_ctx", "shipped_config"),
            "figures": concurrency_figures,
            "derived": concurrency_derived,
            "verdict": _need(
                _need(concurrency, "verdict", "prefill_concurrency"), "decision", "verdict"
            ),
            "source": SOURCE_PATHS["prefill_concurrency"],
        },
        "quantisation": {
            "work_item": "LP-0MWCEHYOO002TRGD",
            "candidates": {name: candidate(name) for name in QUANT_CANDIDATES},
            "bucket": QUANT_BUCKET,
            "shipped_preset": _need(
                _need(quantisation, "engine", "quantisation"), "shipped_preset", "engine"
            ),
            "shipped_weights_gib": _need(
                _need(_need(quantisation, "layouts", "quantisation"), "shipped_ud_q5_k_m", "layouts"),
                "gib",
                "layouts.shipped_ud_q5_k_m",
            ),
            "expert_bytes_share_pct": round(
                sum(
                    _need(role, "share_pct", f"role_bytes.{name}")
                    for name, role in _need(
                        _need(_need(quantisation, "layouts", "quantisation"), "shipped_ud_q5_k_m", "layouts"),
                        "role_bytes",
                        "layouts.shipped_ud_q5_k_m",
                    ).items()
                    if name.startswith("expert_")
                ),
                1,
            ),
            "per_slot_ctx": _need(quant_slot_inputs, "per_slot_ctx", "per_slot_resources"),
            "pool_size": _need(quant_slot_inputs, "session_slot_pool_size", "per_slot_resources"),
            "checkpoint_mib": _need(quant_slot_inputs, "checkpoint_mib", "per_slot_resources"),
            "ctx_checkpoints": _need(quant_slot_inputs, "ctx_checkpoints", "per_slot_resources"),
            "source": SOURCE_PATHS["quantisation"],
            "geometry_source": SOURCE_PATHS["single_slot_throughput"],
        },
        "geometry": geometry,
        "accept_ceiling": {
            "work_item": "LP-0MWCEI8QV007NIC0",
            "rate_added_by_option_tok_s": _need(
                _need(accept_option, "throughput_tok_s", "accept_option"),
                "rate_added_by_option_tok_s",
                "accept_option.throughput_tok_s",
            ),
            "gpu_memory_gib": _need(
                _need(accept_option, "per_slot_resource_cost", "accept_option"),
                "gpu_memory_gib",
                "accept_option.per_slot_resource_cost",
            ),
            "config_changes": _need(
                _need(accept_option, "per_slot_resource_cost", "accept_option"),
                "config_changes",
                "accept_option.per_slot_resource_cost",
            ),
            "token_point_pct": _need(
                _need(accept_option, "local_share_pct", "accept_option"),
                "token_point",
                "accept_option.local_share_pct",
            ),
            "stream_point_pct": _need(
                _need(accept_option, "local_share_pct", "accept_option"),
                "stream_point",
                "accept_option.local_share_pct",
            ),
            "levers_bound": levers_bound,
            "source": SOURCE_PATHS["accept_ceiling"],
        },
        "decode": {
            "local_busy_seconds": _need(quant_capacity_inputs, "local_busy_seconds", "capacity"),
            "decode_slot_seconds": _need(quant_capacity_inputs, "decode_slot_seconds", "capacity"),
            "decode_slot_share_pct": _need(
                quant_capacity_inputs, "decode_slot_share_pct", "capacity"
            ),
        },
    }


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole


def _target(inputs: dict[str, Any], target_pct: float, label: str) -> dict[str, Any]:
    """Convert a stream-share target into capacity, reuse and gate-time terms."""
    measured = inputs["measured"]
    point = inputs["ceiling"]["point"]
    reuse_point = inputs["reuse_fractions"]["point"]
    prefill_seconds = inputs["prefill_seconds_available"]
    tokens_per_stream = measured["local_prompt_tokens_per_stream"]
    point_rate = point["engine_rate_tok_s"]
    idle_gate = inputs["compaction"]["idle_gate_seconds"]

    required_streams = target_pct / 100.0 * measured["offered_streams"]
    required_tokens = required_streams * tokens_per_stream
    delta_tokens = max(0.0, required_tokens - measured["local_prompt_tokens"])
    delta_streams = max(0.0, required_streams - measured["local_streams"])
    cold_at_point_reuse = delta_tokens * (1.0 - reuse_point)
    gate_seconds_at_point_reuse = cold_at_point_reuse / point_rate
    gate_seconds_if_cold = delta_tokens / point_rate
    reuse_needed = 1.0 - point["prefill_capacity_tokens"] / required_tokens
    rate_needed = required_tokens * (1.0 - reuse_point) / prefill_seconds
    return {
        "label": label,
        "target_stream_share_pct": target_pct,
        "equivalent_token_share_pct": round(_pct(required_tokens, measured["offered_prompt_tokens"]), 3),
        "required_streams": round(required_streams, 2),
        "required_delivered_tokens": round(required_tokens, 1),
        "delta_streams_vs_measured": round(delta_streams, 2),
        "delta_delivered_tokens_vs_measured": round(delta_tokens, 1),
        "factor_vs_point_capacity": round(required_tokens / point["delivered_tokens"], 3),
        "headroom_factor_vs_point_ceiling": (
            round(point["delivered_tokens"] / required_tokens, 2) if required_tokens else None
        ),
        "reuse_needed_at_point_rate": round(max(0.0, reuse_needed), 4),
        "rate_needed_at_point_reuse_tok_s": round(rate_needed, 1),
        "reachable_at_point": required_tokens <= point["delivered_tokens"],
        "reachable_at_zero_reuse": required_tokens <= point["prefill_capacity_tokens"],
        "delta_cold_tokens_at_point_reuse": round(cold_at_point_reuse, 1),
        "gate_seconds_at_point_reuse": round(gate_seconds_at_point_reuse, 1),
        "gate_seconds_if_marginal_cold": round(gate_seconds_if_cold, 1),
        "idle_gate_seconds": idle_gate,
        "idle_gate_used_pct_at_point_reuse": round(_pct(gate_seconds_at_point_reuse, idle_gate), 1),
        "idle_gate_used_pct_if_marginal_cold": round(_pct(gate_seconds_if_cold, idle_gate), 1),
        "above_ac9_baseline_factor": round(
            target_pct / inputs["parent_targets"]["ac9_stream_pct"], 3
        ),
    }


def derive(inputs: dict[str, Any]) -> dict[str, Any]:
    measured = inputs["measured"]
    ceiling = inputs["ceiling"]
    point = ceiling["point"]
    offered_tokens = measured["offered_prompt_tokens"]
    offered_streams = measured["offered_streams"]
    tokens_per_stream = measured["local_prompt_tokens_per_stream"]
    size_ratio = inputs["measured"]["offered_prompt_tokens_per_stream"] / tokens_per_stream

    denominators = {
        "token": {
            "definition": "local prompt tokens / offered prompt tokens",
            "offered_tokens": offered_tokens,
            "actual_pct": round(measured["local_token_share_pct"], 4),
            "ceiling_min_pct": ceiling["min"]["token_share_pct"],
            "ceiling_point_pct": point["token_share_pct"],
            "ceiling_max_pct": ceiling["max"]["token_share_pct"],
            "canonical_ceiling_tokens": round(measured["canonical_ceiling_tokens"], 1),
            "moved_by": "a tok/s lever (throughput, compaction conversion, quantisation)",
        },
        "stream": {
            "definition": "locally served streams / offered streams",
            "offered_streams": offered_streams,
            "actual_pct": round(measured["local_stream_share_pct"], 4),
            "ceiling_min_pct": ceiling["min"]["stream_share_pct"],
            "ceiling_point_pct": point["stream_share_pct"],
            "ceiling_max_pct": ceiling["max"]["stream_share_pct"],
            "moved_by": "admission/eligibility and capacity, at the current local prompt-size mix",
        },
        "prompt_size_mix": {
            "local_prompt_tokens_per_stream": round(tokens_per_stream, 2),
            "offered_prompt_tokens_per_stream": round(
                measured["offered_prompt_tokens_per_stream"], 2
            ),
            "offered_over_local_ratio": round(size_ratio, 3),
            "note": (
                "Stream share and token share are the same capacity divided by "
                "different per-stream sizes: offered streams average "
                f"{size_ratio:.2f}x the local stream, so one stream-share point is "
                "worth that many token-share points. Never mix the two in one comparison."
            ),
        },
    }

    compaction = inputs["compaction"]
    compaction_rows = compaction["cold_rows"]
    cap25 = str(25000)
    if cap25 not in compaction_rows:
        raise RegisterError("compaction.cold_rows has no 25000-token row")
    row25 = compaction_rows[cap25]
    shipped_rows = {
        cap: row
        for cap, row in compaction_rows.items()
        if int(cap) in {30000, 38000, 57000}
    }
    shipped_converts_nothing = all(
        float(row.get("added_tokens") or 0.0) == 0.0 for row in shipped_rows.values()
    )

    parallel = inputs["parallel_prefill"]
    parallel_figures = parallel["figures"]
    parallel_derived = parallel["derived"]

    quantisation = inputs["quantisation"]
    quant_rows = []
    for name in QUANT_CANDIDATES:
        candidate = quantisation["candidates"][name]
        quant_rows.append(
            {
                "candidate": name,
                "provenance": candidate["provenance"],
                "weights_gib": candidate["weights_gib"],
                "delta_vs_shipped_gib": candidate["delta_vs_shipped_gib"],
                "weights_per_slot_gib_at_pool": round(
                    candidate["weights_gib"] / quantisation["pool_size"], 2
                ),
                "prefill_tok_s_before": candidate["prefill_tokens_per_second_measured"],
                "prefill_tok_s_after": candidate["prefill_tokens_per_second_after"],
                "prefill_gain_measured_share_pct": candidate["prefill_gain_measured_share_pct"],
                "prefill_gain_ceiling_pct": candidate["prefill_gain_ceiling_pct"],
                "weight_share_pct_of_prefill_pass": candidate["weight_share_pct_of_pass"],
                "decode_byte_ratio_bound_pct": candidate["decode_byte_ratio_bound_pct"],
                "decode_tok_s_ceiling": candidate["decode_tokens_per_second_ceiling"],
                "decode_slot_seconds_freed": candidate["decode_slot_seconds_freed"],
                "token_share_before_pct": candidate["local_token_share_before_pct"],
                "token_share_after_pct": candidate["local_token_share_after_pct"],
                "token_share_delta_pp": round(
                    candidate["local_token_share_after_pct"]
                    - candidate["local_token_share_before_pct"],
                    4,
                ),
                "stream_share_before_pct": candidate["local_stream_share_before_pct"],
                "stream_share_after_pct": candidate["local_stream_share_after_pct"],
                "stream_share_delta_pp": round(
                    candidate["local_stream_share_after_pct"]
                    - candidate["local_stream_share_before_pct"],
                    4,
                ),
                "free_slot_seconds_gain_pct": candidate["free_slot_seconds_gain_pct"],
            }
        )

    geometry = inputs["geometry"]
    accept = inputs["accept_ceiling"]
    bound = accept["levers_bound"]

    families = [
        {
            "key": "a_session_compaction",
            "label": "(a) Session size / compaction",
            "question": (
                "AC1a — median request 57k new tokens, top decile >100k: what does "
                "compaction buy?"
            ),
            "work_item": compaction["work_item"],
            "artefact": "docs/dev/compaction-benefit-estimate-2026-11-01.md",
            "register": compaction["source"],
            "throughput": {
                "gate_rate_tok_s": compaction["gate_rate_tok_s"],
                "rate_added_tok_s": compaction["rate_added_by_lever_tok_s"],
                "attributable_delivered_tok_s": compaction["attributable_delivered_tok_s"],
                "idle_gate_seconds": compaction["idle_gate_seconds"],
                "idle_gate_share_pct": compaction["idle_gate_share_pct"],
                "cold_vs_blended_gate_cost_ratio": compaction["cold_vs_blended_gate_cost_ratio"],
                "demand_gate_seconds_at_25k": row25.get("demand_gate_seconds"),
                "note": (
                    "Adds 0 tok/s of prefill rate: it converts idle prefill-gate "
                    "seconds into served cold tokens, which is a demand-side lever, "
                    "not a capacity one."
                ),
            },
            "per_slot_resource_cost": {
                "summariser_model": compaction["summariser_model"],
                "quantisation": compaction["summariser_quantisation"],
                "parallel": compaction["summariser_parallel"],
                "ctx_size": compaction["summariser_ctx_size"],
                "weights_gb": compaction["summariser_weights_gb"],
                "kv_cache_gib": compaction["summariser_kv_cache_gib"],
                "total_vram_gb": compaction["summariser_total_vram_gb"],
                "vram_share_pct": compaction["summariser_vram_share_pct"],
                "gpu_seconds_per_compaction": compaction["gpu_seconds_per_compaction"],
                "note": (
                    "One dedicated summariser slot held for the server's lifetime, "
                    "sharing the GPU with the prefill gate it is meant to relieve."
                ),
            },
            "local_share": {
                "binding_cap_tokens": 25000,
                "added_delivered_tokens": row25.get("added_tokens"),
                "served_streams": row25.get("served_streams"),
                "token_share_before_pct": measured["local_token_share_pct"],
                "token_share_after_pct": row25.get("local_token_share_pct"),
                "token_share_delta_pp": row25.get("token_share_delta_pp"),
                "stream_share_before_pct": measured["local_stream_share_pct"],
                "stream_share_after_pct": row25.get("local_stream_share_pct"),
                "stream_share_delta_pp": row25.get("stream_share_delta_pp"),
                "binding_constraint": row25.get("binding_constraint"),
                "shipped_targets_convert_nothing": shipped_converts_nothing,
                "shipped_targets_pp": {
                    cap: {
                        "token_share_delta_pp": row.get("token_share_delta_pp"),
                        "stream_share_delta_pp": row.get("stream_share_delta_pp"),
                    }
                    for cap, row in shipped_rows.items()
                },
            },
            "provenance": (
                "estimated — guard coefficients x measured gate mix; the summariser's "
                "GPU-seconds are a plausible 7B Q4 range, not a benchmark"
            ),
            "verdict": (
                "conditional — the only family with a positive share delta, but only at a "
                "target at or below the 25k cold-prefill budget; at the shipped 30k/38k "
                "targets (and at the intake's 57k) it converts nothing"
            ),
            "conditions": [
                "the compaction target must be co-lowered to 25 000 tokens "
                "(compaction_trigger_ratio and *target* are orthogonal)",
                "the gain rides on 1 612 s of idle gate time (56.9 % of the window) and "
                "would displace delivery in a saturated window",
                "the summariser's 24.9-45.6 s of GPU time per compaction competes with "
                "the gate it relieves",
                "~25k-token contexts are the product cost: this is a quality trade, not a "
                "free capacity win",
            ],
        },
        {
            "key": "b_parallel_prefill",
            "label": "(b) Prefill serialisation / parallel prefill",
            "question": (
                "AC1b — re-measure with the current config and test the single-gate "
                "assumption (ubatch/concurrency > 1)"
            ),
            "work_item": parallel["work_item"],
            "artefact": "docs/dev/prefill-concurrency-2026-11-11.md",
            "register": parallel["source"],
            "throughput": {
                "serial_aggregate_tok_s_at_44_770": parallel_figures["E1.c1_aggregate_tps_median"],
                "parallel_aggregate_tok_s_at_44_770": parallel_figures["E1.c2_aggregate_tps_median"],
                "aggregate_ratio": parallel_derived["D1.aggregate_ratio_median"],
                "wall_time_factor": parallel_derived["D2.window_factor_median"],
                "per_request_ratio": parallel_derived["D3.per_request_ratio_median"],
                "production_bucket_20_40k_tok_s": [
                    parallel_figures["P1.bucket_20_40k_aggregate_tps"],
                    parallel_figures["P2.bucket_20_40k_aggregate_tps"],
                ],
                "rate_added_tok_s": 0.0,
                "single_gate_daily_tokens": parallel_derived["D4.single_gate_daily_tokens_at_40k"],
                "parallel_daily_tokens": parallel_derived["D5.parallel_daily_tokens_measured_at_40k"],
                "note": (
                    "Aggregate prefill throughput does not rise with concurrency: the GPU "
                    "conserves memory bandwidth, so two prefills interleave instead of "
                    "running in parallel. Serialisation is not the binding constraint."
                ),
            },
            "per_slot_resource_cost": {
                "prefilling_slots_required": 2,
                "per_slot_ctx_tokens": parallel["per_slot_ctx"],
                "context_checkpoints_per_slot": 32,
                "checkpoint_mib_each": parallel_figures["E1.per_checkpoint_mib"],
                "checkpoint_state_gib_per_slot": parallel_derived["D7.per_slot_checkpoint_gib"],
                "checkpoint_state_gib_total": round(
                    2 * parallel_derived["D7.per_slot_checkpoint_gib"], 1
                ),
                "scheduling_cost_first_byte_s": parallel_figures[
                    "E1.scheduling_cost_live_local_first_byte_s"
                ],
                "note": (
                    "Two slots' worth of context-checkpoint state (2 x 2.0 GiB at "
                    "131 072 tokens/slot) is occupied for zero additional throughput, and "
                    "a live request queued behind the prefill waited 183 s to first byte."
                ),
            },
            "local_share": {
                "token_share_delta_pp": 0.0,
                "stream_share_delta_pp": 0.0,
                "binding_constraint": "engine prefill rate (156.2 tok/s at 44 770 tokens)",
                "note": "No rate change, so no share change on either denominator.",
            },
            "provenance": "measured — controlled engine probe, three interleaved rounds, plus two production windows",
            "verdict": "reject — 1.00x aggregate for 2.00x wall time and 0.81x per-request latency",
            "conditions": [
                "rollback value is the shipped local_max_concurrent_prefills: 1",
                "re-testing costs a one-key config edit, but there is no measured throughput to buy",
            ],
        },
        {
            "key": "c_quantisation_geometry",
            "label": "(c) Model / quantisation / ctx-slot geometry",
            "question": (
                "AC1c — inferencing-mode quantisation, ubatch size, slot-count and "
                "ctx-slot geometry"
            ),
            "work_item": quantisation["work_item"],
            "artefact": "docs/dev/quantisation-cost-2026-11-11.md",
            "register": quantisation["source"],
            "throughput": {
                "shipped_preset": quantisation["shipped_preset"],
                "gate_rate_before_tok_s": quant_rows[-1]["prefill_tok_s_before"],
                "candidates": [
                    {
                        "candidate": row["candidate"],
                        "prefill_tok_s_after": row["prefill_tok_s_after"],
                        "prefill_gain_measured_share_pct": row["prefill_gain_measured_share_pct"],
                        "prefill_gain_ceiling_pct": row["prefill_gain_ceiling_pct"],
                        "weight_share_pct_of_prefill_pass": row[
                            "weight_share_pct_of_prefill_pass"
                        ],
                        "decode_byte_ratio_bound_pct": row["decode_byte_ratio_bound_pct"],
                        "decode_tok_s_ceiling": row["decode_tok_s_ceiling"],
                    }
                    for row in quant_rows
                ],
                "largest_prefill_gain_pct": max(
                    row["prefill_gain_measured_share_pct"] for row in quant_rows
                ),
                "geometry_gate_rate_added_tok_s": 0.0,
                "note": (
                    "Prefill is not weight-bound (a pass streams 6.3 GB/s of weights "
                    "against the 88 GB/s batch-1 decode already sustains), so a lower "
                    "quant buys decode latency, not prefill throughput. Slot count "
                    "redistributes concurrency and prefix reuse; it adds no aggregate rate."
                ),
            },
            "per_slot_resource_cost": {
                "weights_gib_before": quantisation["shipped_weights_gib"],
                "expert_bytes_share_pct": quantisation["expert_bytes_share_pct"],
                "per_candidate": [
                    {
                        "candidate": row["candidate"],
                        "weights_gib": row["weights_gib"],
                        "delta_vs_shipped_gib": row["delta_vs_shipped_gib"],
                        "weights_per_slot_gib": row["weights_per_slot_gib_at_pool"],
                        "decode_slot_seconds_freed": row["decode_slot_seconds_freed"],
                        "free_slot_seconds_gain_pct": row["free_slot_seconds_gain_pct"],
                    }
                    for row in quant_rows
                ],
                "checkpoint_mib_each": quantisation["checkpoint_mib"],
                "ctx_checkpoints": quantisation["ctx_checkpoints"],
                "geometry_steal_pct_pool1_two_sessions": geometry["steal_pct_pool1_two_sessions"],
                "geometry_worst_case_extra_reprefill_tokens": geometry[
                    "worst_case_extra_reprefill_tokens"
                ],
                "geometry_reprefill_pool_increase_pct": geometry["reprefill_pool_increase_pct"],
                "geometry_warm_ceiling_with_cochange": geometry[
                    "warm_ceiling_one_slot_with_cochange"
                ],
                "geometry_warm_ceiling_without_cochange": geometry[
                    "warm_ceiling_one_slot_without_cochange"
                ],
                "note": (
                    "Weight bytes are shared across slots and are weight-independent of "
                    "ctx, KV (q8_0) and the ~2.0 GiB/prompt checkpoint term. One slot buys "
                    "a 258 048-token clamp only with three co-changed knobs, at a 95 % "
                    "steal rate and up to 11.5 M tokens/day of extra re-prefill."
                ),
            },
            "local_share": {
                "token_share_delta_pp": round(
                    max(row["token_share_delta_pp"] for row in quant_rows), 4
                ),
                "stream_share_delta_pp": round(
                    max(row["stream_share_delta_pp"] for row in quant_rows), 4
                ),
                "per_candidate": [
                    {
                        "candidate": row["candidate"],
                        "token_share_after_pct": row["token_share_after_pct"],
                        "token_share_delta_pp": row["token_share_delta_pp"],
                        "stream_share_after_pct": row["stream_share_after_pct"],
                        "stream_share_delta_pp": row["stream_share_delta_pp"],
                    }
                    for row in quant_rows
                ],
                "geometry_token_share_delta_pp": 0.0,
                "geometry_stream_share_delta_pp": 0.0,
                "binding_constraint": "decode slot-seconds (13.4 % of local busy), not prefill",
                "note": (
                    "The share delta is decode capacity freed, not prefill capacity gained: "
                    "it moves the ceiling by hundredths of a point."
                ),
            },
            "provenance": (
                "quantisation: measured GGUF tensor tables + measured production logs, with "
                "the prefill effect bounded above by the measured weight-time share; geometry: "
                "desk analysis — the arm A/B is unmeasured (needs a >=60 % busy window)"
            ),
            "verdict": (
                "not a share lever — quantisation buys decode latency for an unmeasured "
                "quality risk (+0.02/+0.03 pp token share); slot geometry adds no aggregate "
                "throughput and its A/B is unmeasured"
            ),
            "conditions": [
                "no local quality delta exists for either lower-quant candidate; 89.9 % of "
                "the quantised bytes are MoE expert weights",
                "only the UD-Q4_K_S MTP GGUF is present locally and it cannot be loaded "
                "non-MTP, so a measured Q4 prefill/decode needs a ~21 GiB download first",
                "ubatch and thread/checkpoint axes stay with LP-0MVNBJXR6002OSPD / "
                "LP-0MVNBJY2R004EZ51 (both blocked on a 40-80k measurement band)",
            ],
        },
        {
            "key": "d_accept_ceiling",
            "label": "(d) Accept the ceiling",
            "question": "AC1d — re-scope the target to the hardware limit, with arithmetic",
            "work_item": accept["work_item"],
            "artefact": "docs/dev/accept-ceiling-arithmetic-2026-11-01.md",
            "register": accept["source"],
            "throughput": {
                "delivered_blend_tok_s": round(
                    measured["local_prompt_tokens"] / measured["local_busy_seconds"], 2
                ),
                "prefill_ceiling_tok_s": [
                    ceiling["min"]["engine_rate_tok_s"],
                    point["engine_rate_tok_s"],
                    ceiling["max"]["engine_rate_tok_s"],
                ],
                "rate_added_tok_s": accept["rate_added_by_option_tok_s"],
                "note": (
                    "Buys no capacity: it re-scopes the target to what the measured engine "
                    "already sustains, and it is the baseline every other family is measured "
                    "against."
                ),
            },
            "per_slot_resource_cost": {
                "gpu_memory_gib": accept["gpu_memory_gib"],
                "config_changes": accept["config_changes"],
                "note": "No summariser slot, no model download, no config edit, no risk added.",
            },
            "local_share": {
                "token_point_pct": accept["token_point_pct"],
                "stream_point_pct": accept["stream_point_pct"],
                "token_share_delta_pp": 0.0,
                "stream_share_delta_pp": 0.0,
                "token_band_pct": [ceiling["min"]["token_share_pct"], ceiling["max"]["token_share_pct"]],
                "stream_band_pct": [ceiling["min"]["stream_share_pct"], ceiling["max"]["stream_share_pct"]],
                "ac9_baseline_pct": inputs["parent_targets"]["ac9_stream_pct"],
                "ac9_factor_vs_point": inputs["parent_targets"]["ac9_factor_vs_point"],
                "ac9_reachable": inputs["parent_targets"]["ac9_factor_vs_point"] <= 1.0,
                "ac1_10pct_factor_vs_point": inputs["parent_targets"]["ac1_factor_vs_point"],
                "ac1_10pct_reachable": inputs["parent_targets"]["ac1_factor_vs_point"] <= 1.0,
                "binding_constraint": "one prefill gate at 156-254 tok/s aggregate, times cache reuse",
                "note": (
                    "The point ceiling reproduces the committed 1.621 % / 5.964 % exactly; "
                    "the band's width is cache reuse, not engine rate."
                ),
            },
            "provenance": "derived — arithmetic over the measured registers; no new measurement",
            "verdict": (
                "endorsed as the comparison baseline: AC9's 2.8 % is 0.470x the point "
                "capacity (reachable), while >=10 % of streams is 1.677x it (not reachable "
                "at the point capacity)"
            ),
            "conditions": [
                "the denominator must be stated with every figure (0.41-1.62 % token vs "
                "1.52-5.96 % stream at the measured band)",
                "the ceiling assumes the window's 70.8 % cache reuse scales with volume; the "
                "marginal stream is cold (reuse ~ 1.0), which is why the band is wide",
            ],
        },
    ]

    recommended = _target(
        inputs, RECOMMENDED_STREAM_SHARE_PCT, "recommended for parent AC9"
    )
    candidates = [
        _target(inputs, pct, label) for pct, label in TARGET_CANDIDATES
    ]
    prior_recommendation = next(
        (row for row in candidates if abs(row["target_stream_share_pct"] - 3.7) < 1e-9),
        None,
    )

    share_target = {
        "parent_item": TARGET_ITEM,
        "parent_criterion": (
            "AC9 — measured aggregate recovery: local share materially above the 2.8 % "
            "baseline (165 local vs 5 707 remote streams) over a >=30 min window with "
            ">=3 prefer-local sessions"
        ),
        "denominator": "stream — local streams / offered streams, at the current local prompt-size mix",
        "recommended_stream_share_pct": RECOMMENDED_STREAM_SHARE_PCT,
        "recommended_token_share_companion_pct": recommended["equivalent_token_share_pct"],
        "arithmetic": recommended,
        "candidates": candidates,
        "rationale": [
            (
                f"{RECOMMENDED_STREAM_SHARE_PCT} % of streams is "
                f"{recommended['above_ac9_baseline_factor']:.2f}x the AC9 baseline (2.8 %) and "
                f"{recommended['factor_vs_point_capacity']:.3f}x the point capacity ceiling — "
                f"{recommended['headroom_factor_vs_point_ceiling']:.2f}x inside it."
            ),
            (
                "It needs "
                f"{recommended['delta_delivered_tokens_vs_measured']:,.0f} more delivered prompt "
                f"tokens than the measured window ({recommended['delta_streams_vs_measured']:.1f} "
                f"streams = {recommended['required_delivered_tokens']:,.0f} tokens)."
            ),
            (
                "At the window's own cache reuse that is "
                f"{recommended['gate_seconds_at_point_reuse']:,.0f} prefill gate-seconds "
                f"({recommended['idle_gate_used_pct_at_point_reuse']:.1f} % of the "
                f"{recommended['idle_gate_seconds']:,.0f} idle seconds); if every added token is "
                f"cold it is {recommended['gate_seconds_if_marginal_cold']:,.0f} s "
                f"({recommended['idle_gate_used_pct_if_marginal_cold']:.1f} %) — inside the idle "
                "gate either way, so the target needs no throughput lever."
            ),
            (
                "It sits below the combined all-lever bound "
                f"({bound['combined_high']['stream_share_after_pct']:.2f} % streams), i.e. it is "
                "reachable by admission/routing work rather than by funding a lever."
            ),
            (
                f"The same volume is {recommended['equivalent_token_share_pct']:.3f} % of offered "
                "prompt tokens — quote both, because the two denominators differ by the "
                "3.68x prompt-size mix."
            ),
            (
                "The sibling's prior 3.7 % recommendation would consume "
                f"{prior_recommendation['idle_gate_used_pct_if_marginal_cold']:.1f} % of the "
                "window's idle prefill gate if every added token is cold — no margin for a "
                "cold expansion; 3.5 % leaves "
                f"{100.0 - recommended['idle_gate_used_pct_if_marginal_cold']:.0f} % of it unused."
                if prior_recommendation
                else ""
            ),
        ],
        "rejected_targets": {
            "10pct_stream": {
                "reason": (
                    "1.677x the point capacity; reachable only in the optimistic corner "
                    "(90 % cache reuse at the best rate), which admission cannot sustain as "
                    "volume grows"
                ),
                "rate_needed_tok_s": 406.2,
                "reuse_needed": 0.8257,
            },
            "10pct_token": {
                "reason": (
                    "6.171x the point capacity — a different machine, not a target: 1 495 tok/s "
                    "or 95 % cache reuse"
                ),
            },
        },
    }

    recommendation = {
        "primary_option": "accept-and-rescope",
        "primary": (
            "Accept the measured ceiling and re-scope parent AC9's share clause to a specific "
            f"number: >= {RECOMMENDED_STREAM_SHARE_PCT} % of locally served streams (equivalent "
            f"to {recommended['equivalent_token_share_pct']:.3f} % of offered prompt tokens) over "
            "a >=30 min window with >=3 prefer-local sessions, with the 0 deficit-attributable "
            "local_provider_cooldown skips clause unchanged."
        ),
        "why": (
            "The four families leave the aggregate prefill rate where it is: parallel prefill "
            "1.00x, quantisation <=1.9 % prefill (decode-only benefit), geometry 0 tok/s, and "
            "compaction converts idle gate time without adding capacity. The combined all-lever "
            f"bound is {bound['combined_high']['stream_share_after_pct']:.2f} % of streams and "
            f"{bound['combined_high']['token_share_after_pct']:.3f} % of tokens, so no affordable "
            "lever set reaches a >=10 % stream target."
        ),
        "secondary_option": {
            "option": "implement-one-lever",
            "lever": "a_session_compaction, co-tuned (compaction_trigger_ratio 0.70 + target 25 000)",
            "would_create": [
                (
                    "[impl] Co-tune compaction trigger and target to the 25 000-token "
                    "cold-prefill budget — the only configuration that converts cold prompts, "
                    "worth +0.92 pp of stream share"
                ),
                (
                    "[verify] Post-change stream-share measurement over a >=30 min window, "
                    "checking the converted volume against the 1 612 idle gate-seconds it rides on"
                ),
            ],
            "condition": (
                "only if the producer wants the extra +0.92 pp; it is not a path to >=10 % and "
                "it trades ~25k-token contexts and a summariser slot for the gain"
            ),
        },
        "not_recommended": {
            "option": "follow-up-benchmark",
            "reason": (
                "no open measurement question gates the share decision: the four families are "
                "measured, and the remaining gaps (slot-geometry arm A/B needs a >=60 % busy "
                "window; a Q4 prefill/decode measurement needs a ~21 GiB GGUF download; the "
                "ubatch/thread axes have their own blocked children) each serve a lever this "
                "comparison does not recommend"
            ),
            "exception": (
                "if quantisation is ever pursued for decode latency, gate it on a quality "
                "benchmark first — 89.9 % of the quantised bytes are MoE expert weights and "
                "there is no local quality delta"
            ),
        },
        "priority_note": (
            "Followed the intake risk note: (b) prefill serialisation and (c) model/geometry "
            "were pursued first as the time-constrained priority, and both returned negative — "
            "so the comparison's only positive family is (a), and the recommended action is (d)."
        ),
    }

    return {
        "denominators": denominators,
        "families": families,
        "combined_bound": {
            "rate_factors": bound["rate_factors"],
            "token_share_after_pct": bound["combined_high"]["token_share_after_pct"],
            "stream_share_after_pct": bound["combined_high"]["stream_share_after_pct"],
            "token_share_delta_pp": bound["combined_high"]["token_share_delta_pp"],
            "stream_share_delta_pp": bound["combined_high"]["stream_share_delta_pp"],
            "gap_closed_pct_of_gap": {
                "token": bound["gap_to_target"]["token"]["gap_closed_pct_of_gap"],
                "stream": bound["gap_to_target"]["stream"]["gap_closed_pct_of_gap"],
            },
            "remaining_factor_to_10pct": {
                "token": bound["gap_to_target"]["token"]["remaining_factor"],
                "stream": bound["gap_to_target"]["stream"]["remaining_factor"],
            },
            "caveats": bound["caveats"],
            "note": (
                "The combined bound is the all-levers-at-their-best-measured-end case, applied "
                "to the point ceiling; the recommended target sits below it."
            ),
        },
        "share_target": share_target,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _rel_gap(actual: float, expected: float) -> float:
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs(actual - expected) / abs(expected)


def check(inputs: dict[str, Any], derived: dict[str, Any]) -> list[dict[str, Any]]:
    """Assert the properties the comparison leans on, with their evidence."""
    entries: list[dict[str, Any]] = []

    def add(claim: str, ok: bool, evidence: Any, note: str = "") -> None:
        entries.append(
            {
                "claim": claim,
                "ok": bool(ok),
                "verdict": "confirms" if ok else "refuted",
                "evidence": evidence,
                "note": note,
            }
        )

    families = {family["key"]: family for family in derived["families"]}
    parallel = families["b_parallel_prefill"]
    compaction = families["a_session_compaction"]
    quant = families["c_quantisation_geometry"]
    accept = families["d_accept_ceiling"]
    target = derived["share_target"]["arithmetic"]
    bound = derived["combined_bound"]

    add(
        "parallel prefill adds no aggregate prefill throughput",
        abs(parallel["throughput"]["aggregate_ratio"] - 1.0) <= TOL_RATIO,
        {
            "aggregate_ratio": parallel["throughput"]["aggregate_ratio"],
            "serial_tok_s": parallel["throughput"]["serial_aggregate_tok_s_at_44_770"],
            "parallel_tok_s": parallel["throughput"]["parallel_aggregate_tok_s_at_44_770"],
        },
        "the parallel arm's median aggregate sits inside the serial arm's own spread",
    )
    add(
        "parallel prefill costs two slots' checkpoint state for that zero",
        parallel["per_slot_resource_cost"]["checkpoint_state_gib_total"]
        >= 2 * parallel["per_slot_resource_cost"]["checkpoint_state_gib_per_slot"] - 0.1,
        {
            "checkpoint_gib_per_slot": parallel["per_slot_resource_cost"][
                "checkpoint_state_gib_per_slot"
            ],
            "total": parallel["per_slot_resource_cost"]["checkpoint_state_gib_total"],
            "aggregate_gain": parallel["throughput"]["aggregate_ratio"],
        },
    )
    add(
        "compaction adds no prefill rate but converts idle gate time",
        compaction["throughput"]["rate_added_tok_s"] == 0.0
        and (compaction["local_share"]["added_delivered_tokens"] or 0.0) > 0.0,
        {
            "rate_added_tok_s": compaction["throughput"]["rate_added_tok_s"],
            "added_delivered_tokens": compaction["local_share"]["added_delivered_tokens"],
            "idle_gate_seconds": compaction["throughput"]["idle_gate_seconds"],
        },
    )
    add(
        "compaction converts nothing unless the cap is at or below the cold-prefill budget",
        compaction["local_share"]["shipped_targets_convert_nothing"]
        and float(compaction["local_share"]["added_delivered_tokens"] or 0.0) > 0.0,
        {
            "binding_cap_tokens": compaction["local_share"]["binding_cap_tokens"],
            "shipped_targets_pp": compaction["local_share"]["shipped_targets_pp"],
            "at_25k_added_tokens": compaction["local_share"]["added_delivered_tokens"],
        },
    )
    add(
        "quantisation moves prefill throughput by less than 2 %",
        quant["throughput"]["largest_prefill_gain_pct"] < 2.0,
        {
            "largest_prefill_gain_pct": quant["throughput"]["largest_prefill_gain_pct"],
            "per_candidate": quant["throughput"]["candidates"],
        },
        "the byte-ratio ceiling is unreachable because a prefill pass streams weights "
        "at 6.3 GB/s against the 88 GB/s batch-1 decode already sustains",
    )
    add(
        "quantisation's share uplift is below 0.05 pp of tokens and 0.10 pp of streams",
        quant["local_share"]["token_share_delta_pp"] < 0.05
        and quant["local_share"]["stream_share_delta_pp"] < 0.10,
        {
            "token_share_delta_pp": quant["local_share"]["token_share_delta_pp"],
            "stream_share_delta_pp": quant["local_share"]["stream_share_delta_pp"],
        },
        "the differing bounds are the denominator ratio: a stream point is worth ~3.7 "
        "token points at the current mix",
    )
    add(
        "the parent AC9 2.8 % stream baseline is inside the point ceiling",
        accept["local_share"]["ac9_reachable"],
        {
            "factor_vs_point": accept["local_share"]["ac9_factor_vs_point"],
            "baseline_pct": accept["local_share"]["ac9_baseline_pct"],
        },
    )
    add(
        "a >=10 % stream target stays outside the point ceiling",
        not accept["local_share"]["ac1_10pct_reachable"],
        {
            "factor_vs_point": accept["local_share"]["ac1_10pct_factor_vs_point"],
            "target_pct": inputs["parent_targets"]["ac1_stream_pct"],
        },
    )
    add(
        "no affordable lever set closes the gap to >=10 % of streams",
        bound["stream_share_after_pct"] <= inputs["ceiling"]["point"]["stream_share_pct"]
        and bound["remaining_factor_to_10pct"]["stream"] > 1.0,
        {
            "stream_share_after_pct": bound["stream_share_after_pct"],
            "point_ceiling_pct": inputs["ceiling"]["point"]["stream_share_pct"],
            "remaining_factor": bound["remaining_factor_to_10pct"]["stream"],
        },
    )
    add(
        "the recommended target is materially above the AC9 baseline",
        target["above_ac9_baseline_factor"] >= 1.2,
        {
            "target_pct": target["target_stream_share_pct"],
            "baseline_pct": inputs["parent_targets"]["ac9_stream_pct"],
            "factor": target["above_ac9_baseline_factor"],
        },
        ">=1.2x is the smallest factor that is honestly 'materially above' the "
        "measured 2.573 % and the 2.8 % baseline",
    )
    add(
        "the recommended target keeps half the point ceiling in headroom",
        target["factor_vs_point_capacity"] <= 0.75,
        {
            "factor_vs_point_capacity": target["factor_vs_point_capacity"],
            "headroom_factor": target["headroom_factor_vs_point_ceiling"],
        },
    )
    add(
        "the recommended target is reachable inside the window's idle prefill gate",
        target["gate_seconds_if_marginal_cold"] <= target["idle_gate_seconds"],
        {
            "gate_seconds_if_marginal_cold": target["gate_seconds_if_marginal_cold"],
            "gate_seconds_at_point_reuse": target["gate_seconds_at_point_reuse"],
            "idle_gate_seconds": target["idle_gate_seconds"],
        },
        "so the target needs no throughput lever: it is a routing/admission goal",
    )
    add(
        "the recommended stream target and its token companion describe the same volume",
        abs(
            target["equivalent_token_share_pct"]
            - derived["share_target"]["recommended_token_share_companion_pct"]
        )
        <= TOL_PP,
        {
            "stream_pct": target["target_stream_share_pct"],
            "token_pct": target["equivalent_token_share_pct"],
        },
    )
    added_streams = float(compaction["local_share"]["served_streams"] or 0.0)
    added_tokens = float(compaction["local_share"]["added_delivered_tokens"] or 0.0)
    delta_token_pp = float(compaction["local_share"]["token_share_delta_pp"] or 0.0)
    delta_stream_pp = float(compaction["local_share"]["stream_share_delta_pp"] or 0.0)
    implied_size = (
        (delta_token_pp / 100.0 * inputs["measured"]["offered_prompt_tokens"])
        / (delta_stream_pp / 100.0 * inputs["measured"]["offered_streams"])
        if delta_stream_pp
        else 0.0
    )
    registered_size = added_tokens / added_streams if added_streams else 0.0
    add(
        "a family's two share deltas are related by the per-stream size of its own volume",
        _rel_gap(implied_size, registered_size) <= 0.01
        and _rel_gap(
            registered_size, float(compaction["local_share"]["binding_cap_tokens"])
        )
        <= TOL_RATIO,
        {
            "implied_tokens_per_added_stream": round(implied_size, 1),
            "registered_tokens_per_added_stream": round(registered_size, 1),
            "binding_cap_tokens": compaction["local_share"]["binding_cap_tokens"],
            "current_local_tokens_per_stream": round(
                inputs["measured"]["local_prompt_tokens_per_stream"], 1
            ),
        },
        "the compaction conversion's added stream carries the 25k cap, not the current "
        "local mean (19 085) — so its stream and token deltas are not related by the "
        "baseline's 3.68x mix, and mixing the two baselines would be wrong",
    )
    add(
        "the point ceiling reproduces the committed canonical ceilings",
        _rel_gap(
            inputs["ceiling"]["point"]["token_share_pct"], inputs["measured"]["canonical_token_share_pct"]
        )
        <= TOL_RECONCILE_REL
        and _rel_gap(
            inputs["ceiling"]["point"]["stream_share_pct"],
            inputs["measured"]["canonical_stream_share_pct"],
        )
        <= TOL_RECONCILE_REL,
        {
            "point_token_pct": inputs["ceiling"]["point"]["token_share_pct"],
            "canonical_token_pct": inputs["measured"]["canonical_token_share_pct"],
            "point_stream_pct": inputs["ceiling"]["point"]["stream_share_pct"],
            "canonical_stream_pct": inputs["measured"]["canonical_stream_share_pct"],
        },
        "wiring guard across registers, not an independent validation",
    )
    add(
        "every family reports a throughput, a cost and both share denominators",
        all(
            family["throughput"]
            and family["per_slot_resource_cost"]
            and "token_share_delta_pp" in family["local_share"]
            and "stream_share_delta_pp" in family["local_share"]
            for family in derived["families"]
        ),
        {"families": [family["key"] for family in derived["families"]]},
    )
    return entries


# ---------------------------------------------------------------------------
# register assembly
# ---------------------------------------------------------------------------


def build_register(root: Path) -> dict[str, Any]:
    inputs = load_inputs(root)
    derived = derive(inputs)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_item": WORK_ITEM,
        "spike_work_item": SPIKE_WORK_ITEM,
        "target_item": TARGET_ITEM,
        "title": "Costed four-family lever comparison and local-share target (spike AC1 + AC2)",
        "generated_at": "2026-11-11",
        "purpose": (
            "Put the four lever families on one unit set (prefill tok/s, per-slot resource "
            "cost, estimated impact on local share) and convert a candidate stream-share "
            "target into capacity, cache-reuse and gate-time terms for the parent's AC9."
        ),
        "read_only": True,
        "provenance_classes": {
            "measured": "Copied verbatim from a named source register; the tool never measures.",
            "derived": "Arithmetic over the registered figures; the formula and inputs are recorded here.",
            "estimated": "A source register's own explicitly-estimated figure, carried with that label.",
        },
        "sources": {key: {"path": path} for key, path in SOURCE_PATHS.items()},
        "model": {
            "target_required_tokens": (
                "target_pct / 100 x offered_streams x local_prompt_tokens_per_stream"
            ),
            "target_factor_vs_capacity": "required_tokens / point delivered_tokens",
            "target_reuse_needed": "1 - point prefill_capacity_tokens / required_tokens",
            "target_gate_seconds_at_point_reuse": (
                "(required_tokens - local_prompt_tokens) x (1 - reuse_point) / rate_point"
            ),
            "target_gate_seconds_if_cold": (
                "(required_tokens - local_prompt_tokens) / rate_point"
            ),
            "notes": [
                "Stream share and token share are one capacity over two per-stream sizes; "
                "the ratio between them is the prompt-size mix, never 1.",
                "The gate-time test is what separates a routing target from a throughput "
                "target: a target that fits inside the measured idle gate seems reachable "
                "without adding capacity, one that does not is a capacity demand.",
                "The pessimistic (cold) gate-time term is the honest bound for expansion "
                "into the refused population, which the refusal decomposition shows is "
                "cache-cold.",
            ],
        },
        "inputs": inputs,
        "derived": derived,
        "checks": check(inputs, derived),
    }


def all_ok(register: dict[str, Any]) -> bool:
    return all(entry["verdict"] != "refuted" for entry in register["checks"])


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= max(1e-6, 1e-6 * abs(float(b)))
    return a == b


def _walk_drift(recorded: Any, recomputed: Any, path: str, drift: list[dict[str, Any]]) -> None:
    if isinstance(recomputed, dict):
        if not isinstance(recorded, dict):
            drift.append({"claim": path, "recorded": recorded, "recomputed": recomputed})
            return
        for key, value in recomputed.items():
            _walk_drift(recorded.get(key), value, f"{path}.{key}" if path else key, drift)
        for key in recorded:
            if key not in recomputed:
                drift.append(
                    {
                        "claim": f"{path}.{key}" if path else key,
                        "recorded": recorded[key],
                        "recomputed": None,
                    }
                )
        return
    if isinstance(recomputed, list):
        if not isinstance(recorded, list) or len(recorded) != len(recomputed):
            drift.append({"claim": path, "recorded": recorded, "recomputed": recomputed})
            return
        for index, value in enumerate(recomputed):
            _walk_drift(recorded[index], value, f"{path}[{index}]", drift)
        return
    if not _close(recorded, recomputed):
        drift.append({"claim": path, "recorded": recorded, "recomputed": recomputed})


def verify_register(register: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-derive from the register's own inputs — a self-contained drift guard."""
    try:
        derived = derive(register["inputs"])
    except (KeyError, RegisterError, TypeError) as exc:
        return [{"claim": "register inputs are complete", "recorded": None, "recomputed": str(exc)}]
    drift: list[dict[str, Any]] = []
    _walk_drift(register.get("derived") or {}, derived, "", drift)
    return drift


# ---------------------------------------------------------------------------
# rendering / CLI
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def render(register: dict[str, Any]) -> str:
    derived = register["derived"]
    families = derived["families"]
    target = derived["share_target"]["arithmetic"]
    lines = [
        f"Costed lever comparison and share target — {register['work_item']}",
        f"  spike {register['spike_work_item']} / target {register['target_item']}",
        "",
        "Denominators (never mixed)",
    ]
    for name in ("token", "stream"):
        entry = derived["denominators"][name]
        lines.append(
            f"  {name:<7} actual {_fmt(entry['actual_pct'], 4)} %"
            f"  ceiling {_fmt(entry['ceiling_min_pct'], 4)} / {_fmt(entry['ceiling_point_pct'], 4)}"
            f" / {_fmt(entry['ceiling_max_pct'], 4)} % (min / point / max)"
        )
    lines.append(
        "  mix     offered stream is "
        f"{_fmt(derived['denominators']['prompt_size_mix']['offered_over_local_ratio'], 3)}x the "
        "local stream"
    )
    lines += ["", "Families"]
    for family in families:
        lines.append(f"  {family['label']} — {family['verdict']}")
        throughput = family["throughput"]
        if family["key"] == "b_parallel_prefill":
            lines.append(
                f"    throughput   {_fmt(throughput['serial_aggregate_tok_s_at_44_770'], 1)} serial vs "
                f"{_fmt(throughput['parallel_aggregate_tok_s_at_44_770'], 1)} parallel tok/s at "
                f"{_fmt(throughput['production_bucket_20_40k_tok_s'][0], 1)}-"
                f"{_fmt(throughput['production_bucket_20_40k_tok_s'][1], 1)} tok/s in the 20-40k bucket"
            )
        elif family["key"] == "c_quantisation_geometry":
            lines.append(
                f"    throughput   prefill +{_fmt(throughput['largest_prefill_gain_pct'], 3)} % "
                f"({_fmt(throughput['gate_rate_before_tok_s'], 1)} tok/s before); geometry +0"
            )
        elif family["key"] == "a_session_compaction":
            lines.append(
                f"    throughput   +{_fmt(throughput['rate_added_tok_s'], 1)} tok/s rate, "
                f"{_fmt(throughput['attributable_delivered_tok_s'], 1)} tok/s attributed volume, "
                f"{_fmt(throughput['idle_gate_seconds'], 0)} idle gate-seconds"
            )
        else:
            lines.append(
                f"    throughput   ceiling {_fmt(throughput['prefill_ceiling_tok_s'][0], 1)} / "
                f"{_fmt(throughput['prefill_ceiling_tok_s'][1], 1)} / "
                f"{_fmt(throughput['prefill_ceiling_tok_s'][2], 1)} tok/s, "
                f"+{_fmt(throughput['rate_added_tok_s'], 1)} added"
            )
        share = family["local_share"]
        lines.append(
            f"    share        token {_fmt(share['token_share_delta_pp'], 3)} pp, "
            f"stream {_fmt(share['stream_share_delta_pp'], 3)} pp"
        )
    lines += [
        "",
        "Combined lever bound",
        f"  rate factor {_fmt(derived['combined_bound']['rate_factors']['combined'], 4)}x ->"
        f" token {_fmt(derived['combined_bound']['token_share_after_pct'], 3)} %,"
        f" stream {_fmt(derived['combined_bound']['stream_share_after_pct'], 3)} %"
        f" (remaining {_fmt(derived['combined_bound']['remaining_factor_to_10pct']['stream'], 2)}x"
        " to 10 % of streams)",
        "",
        "Share target candidates (stream denominator)",
        "  target %  required streams  delivered tokens  x capacity  gate s (point/cold)"
        "  idle used %",
    ]
    for candidate in derived["share_target"]["candidates"]:
        lines.append(
            f"  {_fmt(candidate['target_stream_share_pct'], 4):>8}"
            f"  {_fmt(candidate['required_streams'], 2):>16}"
            f"  {_fmt(candidate['required_delivered_tokens'], 0):>16}"
            f"  {_fmt(candidate['factor_vs_point_capacity'], 3):>10}"
            f"  {_fmt(candidate['gate_seconds_at_point_reuse'], 0):>8}"
            f"/{_fmt(candidate['gate_seconds_if_marginal_cold'], 0):<7}"
            f"  {_fmt(candidate['idle_gate_used_pct_if_marginal_cold'], 1):>8}"
            f"  {candidate['label']}"
        )
    lines += [
        "",
        f"Recommended: >= {_fmt(target['target_stream_share_pct'], 1)} % of streams"
        f" (= {_fmt(target['equivalent_token_share_pct'], 3)} % of offered prompt tokens)",
    ]
    lines += [f"  - {item}" for item in derived["share_target"]["rationale"]]
    lines += [
        "",
        "Recommendation",
        f"  primary    {derived['recommendation']['primary_option']}",
        f"  secondary  {derived['recommendation']['secondary_option']['option']}"
        f" ({derived['recommendation']['secondary_option']['lever']})",
        f"  not        {derived['recommendation']['not_recommended']['option']}",
        "",
        "Checks",
    ]
    for entry in register["checks"]:
        mark = "ok  " if entry["ok"] else "FAIL"
        lines.append(f"  {mark} {entry['claim']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the register as JSON")
    parser.add_argument(
        "--register",
        type=Path,
        default=None,
        help="verify a committed register against its own inputs (drift guard)",
    )
    parser.add_argument(
        "--check", action="store_true", help="with --register: exit non-zero on drift"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        if args.register is not None:
            register = json.loads(args.register.read_text(encoding="utf-8"))
            drift = verify_register(register)
            if args.json:
                print(json.dumps({"drift": drift, "ok": not drift}, indent=2))
            else:
                if drift:
                    print(f"DRIFT in {args.register}: {len(drift)} field(s)")
                    for entry in drift[:20]:
                        print(
                            f"  {entry['claim']}: recorded={entry['recorded']!r} "
                            f"recomputed={entry['recomputed']!r}"
                        )
                else:
                    print(f"{args.register}: no drift ({len(register['checks'])} checks)")
                    if not all_ok(register):
                        print("WARNING: recorded checks report a refuted claim")
                        return 2
            return 0 if not drift else 1
        register = build_register(root)
    except RegisterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(register, indent=2))
    else:
        print(render(register))
    return 0 if all_ok(register) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
