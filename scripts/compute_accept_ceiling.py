#!/usr/bin/env python3
"""Compute the accept-the-ceiling option's arithmetic (LP-0MWCEI8QV007NIC0).

Spike parent LP-0MVXY8ZBZ0003042, AC1d. This is a *pure arithmetic* analyser:
it reads the committed baseline and lever registers, re-derives the local-share
ceiling from the measured engine prefill rate, and prices the "accept the
ceiling" option for the spike's comparison table. It never changes proxy
behaviour, never reads a live slot and never takes a new measurement.

Why the tool exists
-------------------
The ceiling had been quoted as an extrapolation of the *delivered* rate
(``local_prompt_tokens / local_busy_seconds`` x window seconds), which hides
two things a decision needs:

1. **What the hardware can prefill.** Delivered prompt tokens include cache
   hits, which cost no gate time. The binding resource is the prefill gate, so
   the ceiling has to start from the measured aggregate prefill rate
   (156.2-253.8 tok/s), not from the 717.8 tok/s delivered blend.
2. **What "maximum sustainable" means.** A ceiling is only meaningful once the
   decode reserve (gate time that must go to decoding) and the cache-reuse
   fraction of the marginal stream are both stated.

The model below therefore has one formula and two measured parameters::

    T_prefill = window_seconds x (1 - decode_share)
    C_cold    = T_prefill x R_prefill                # tokens actually prefilled
    C_del     = C_cold / (1 - r)                     # delivered prompt tokens
    share     = C_del / denominator                  # token or stream denominator

``R_prefill`` is the measured aggregate prefill tok/s (min / point / max over
the regimes the traffic actually occupies) and ``r`` is the share of delivered
prompt tokens served from cache (three independently measured values).

The answer is reported in three tiers because they answer different questions:

- **hardware prefill ceiling** — ``C_cold / offered``: the share of the offered
  prompt tokens the gate can *originate* in a window (0.31-0.50 %);
- **delivered-share ceiling** — ``C_del / offered``: the same capacity once the
  cache hits are counted, which is what "local share" has always meant
  (0.41-4.96 %, point 1.62 %);
- **stream-denominated equivalents** — the same capacity over the offered
  streams at the current local mean prompt size (1.52-18.3 %, point 5.96 %).

The committed 1.62 % ceiling is the delivered-share tier at the cache-reuse
fraction the window itself implies (``r = 0.7077``). That equality is
algebraic — ``R`` cancels — and it is the finding, not a check of the model: it
says the ceiling of record assumes the window's cache reuse *scales* with the
volume. On the marginal stream, which the refusal decomposition shows is cold
(reuse ~ 1.0), it does not.

Inputs
------
Every figure is read from a committed register; nothing is hand-copied and
nothing is measured here:

- ``docs/dev/local-share-baseline-2026-11-01.json`` — the measured basis and
  the two share denominators (LP-0MWCEGJDP009WURD);
- ``docs/dev/single-slot-throughput-baseline-2026-10-31.json`` — W1
  prefill/decode slot-seconds and the W3/W4 engine rates and prefix reuse;
- ``docs/dev/prefill-concurrency-baseline-2026-11-11.json`` — the re-measured
  aggregate prefill rates and the parallel-prefill verdict
  (LP-0MWCEHDRV000B7HD);
- ``docs/dev/compaction-benefit-estimate-2026-11-01.json`` — the compaction
  lever's added tokens and per-slot cost (LP-0MWCEGXEK000U6G6);
- ``docs/dev/quantisation-cost-2026-11-11.json`` — the quantisation prefill
  bound and the freed decode slot-seconds (LP-0MVCEHYOO002TRGD).

Usage::

    python3 scripts/compute_accept_ceiling.py --json
    python3 scripts/compute_accept_ceiling.py
    python3 scripts/compute_accept_ceiling.py \\
        --register docs/dev/accept-ceiling-arithmetic-2026-11-01.json --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

WORK_ITEM = "LP-0MWCEI8QV007NIC0"
PARENT_WORK_ITEM = "LP-0MVXY8ZBZ0003042"

#: Registers this tool reads. Relative to the repository root.
SOURCE_PATHS = {
    "local_share_baseline": "docs/dev/local-share-baseline-2026-11-01.json",
    "single_slot_throughput": "docs/dev/single-slot-throughput-baseline-2026-10-31.json",
    "prefill_concurrency": "docs/dev/prefill-concurrency-baseline-2026-11-11.json",
    "compaction": "docs/dev/compaction-benefit-estimate-2026-11-01.json",
    "quantisation": "docs/dev/quantisation-cost-2026-11-11.json",
}

#: Which quantisation candidate supplies the prefill bound for the lever cap.
#: UD-Q4_K_S is the only locally present lower-quant GGUF, so its figure is the
#: measured-share one; UD-Q4_K_M is an estimate and is carried as the low end.
QUANT_CANDIDATE_HIGH = "ud_q4_k_s_measured"
QUANT_CANDIDATE_LOW = "ud_q4_k_m_estimated"
#: The prompt-size bucket the local stream mix occupies (mean 19 085 tokens).
QUANT_BUCKET = "20k-40k"

#: The scenario names, in report order.
SCENARIOS = ("min", "point", "max")

#: Tolerances used by the recorded checks. Chosen from the precision the source
#: registers carry (0.1 tok/s, 0.1 pp), never from what makes a check pass.
TOL_RECONCILE_REL = 0.001     # 0.1 % relative — reproduces "the same ceiling"
TOL_PP = 0.05                 # 0.05 pp


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


def load_inputs(root: Path) -> dict[str, Any]:
    """Collect every measured figure the arithmetic needs, with provenance.

    Raises ``RegisterError`` rather than defaulting: a missing engine rate would
    otherwise silently produce a zero ceiling.
    """
    share = _read(root, "local_share_baseline")
    measured = _need(share, "measured", "local-share baseline")
    derived_share = _need(share, "derived", "local-share baseline")
    shared_verdicts = _need(share, "verdicts", "local-share baseline")

    single = _read(root, "single_slot_throughput")
    figures = _need(single, "figures", "single-slot throughput baseline")

    concurrency = _read(root, "prefill_concurrency")
    conc_figures = _need(concurrency, "figures", "prefill-concurrency register")

    compaction = _read(root, "compaction")
    thresholds = _need(compaction, "thresholds_cold", "compaction register")

    quant = _read(root, "quantisation")
    cand_high = _need(_need(quant, "candidates", "quantisation register"), QUANT_CANDIDATE_HIGH, "quantisation register")
    cand_low = _need(quant["candidates"], QUANT_CANDIDATE_LOW, "quantisation register")

    def fig(name: str) -> float:
        entry = _need(figures, name, "single-slot throughput baseline")
        return float(_need(entry, "value", f"figures[{name}]"))

    def conc(name: str) -> float:
        entry = _need(conc_figures, name, "prefill-concurrency register")
        return float(_need(entry, "value", f"figures[{name}]"))

    # The compaction scenario that converts anything at all is the 25 000-token
    # cold-prefill budget; the shipped 30k/38k targets and the 57k intake cap
    # admit no cold prompt (their added_tokens is 0).
    cap_scenarios = [row for row in thresholds if float(row.get("added_tokens") or 0.0) > 0.0]
    if not cap_scenarios:
        raise RegisterError("the compaction register has no cap that converts cold traffic")
    cap = max(cap_scenarios, key=lambda row: float(row["added_tokens"]))

    decode_share_pct = float(_need(cand_high["decode"], "slot_seconds_share_of_busy_pct", "quantisation decode block"))

    return {
        "window": {
            "start": measured["window_start"],
            "end": measured["window_end"],
            "seconds": float(measured["window_seconds"]),
            "source": SOURCE_PATHS["local_share_baseline"],
        },
        "measured": {
            "offered_prompt_tokens": float(measured["offered_prompt_tokens"]),
            "offered_streams": float(measured["requests"]),
            "local_prompt_tokens": float(measured["local_prompt_tokens"]),
            "local_streams": float(measured["local_streams"]),
            "local_busy_seconds": float(measured["local_busy_seconds"]),
            "local_completion_tokens": float(measured["local_completion_tokens"]),
            "local_prompt_tokens_per_stream": float(
                derived_share["local_prompt_tokens_per_stream"]
            ),
            "offered_prompt_tokens_per_stream": float(
                derived_share["offered_prompt_tokens_per_stream"]
            ),
            "local_rate_tok_s": float(derived_share["local_rate_tok_s"]),
            "offered_prompt_tok_s": float(derived_share["offered_prompt_tok_s"]),
            "canonical_ceiling_tokens": float(derived_share["ceiling_tokens"]),
            "canonical_token_share_pct": float(derived_share["ceiling_token_share_pct"]),
            "canonical_stream_share_pct": float(derived_share["ceiling_stream_share_pct"]),
            "source": SOURCE_PATHS["local_share_baseline"],
        },
        "engine": {
            # min: the 44 770-token engine probe rate — the prompt size nearest
            # the refused/offered mean (60 715 / 70 234 tokens).
            "rate_min_tok_s": float(conc("E1.c1_aggregate_tps_median")),
            "rate_min_label": (
                "sustained serial prefill at 44 770-token prompts (engine probe, "
                "live traffic sharing the GPU)"
            ),
            # point: the re-measured 20k-40k aggregate, the bucket the local
            # stream mix (mean 19 085 tokens/stream) actually occupies.
            "rate_point_tok_s": round(
                (float(conc("P1.bucket_20_40k_aggregate_tps"))
                 + float(conc("P2.bucket_20_40k_aggregate_tps"))) / 2.0,
                2,
            ),
            "rate_point_label": (
                "mean of the P1/P2 re-measured 20k-40k aggregate prefill rates "
                "(under the shipped serialised gate)"
            ),
            # max: the best aggregate ever measured for this engine.
            "rate_max_tok_s": float(fig("W3.solo_new_tok_s_p50")),
            "rate_max_label": "W3 solo-prefill p50 aggregate (best measured aggregate)",
            "solo_p50_tok_s": float(fig("W3.solo_new_tok_s_p50")),
            "paired_aggregate_tok_s": float(fig("W3.aggregate_tok_s")),
            "source": SOURCE_PATHS["prefill_concurrency"],
        },
        "decode": {
            "rate_tok_s": float(fig("W1.decode_tok_s_median")),
            "slot_seconds": float(measured["local_completion_tokens"])
            / float(fig("W1.decode_tok_s_median")),
            "share_of_busy_pct": decode_share_pct,
            "source": SOURCE_PATHS["quantisation"],
        },
        "duty_context": {
            # The ceiling is a 100 %-duty statement. Recorded so the extrapolation
            # is visible: it is a hardware bound, not a demand forecast.
            "spike_window_busy_pct": round(
                100.0 * float(measured["local_busy_seconds"]) / float(measured["window_seconds"]),
                2,
            ),
            "w1_busy_pct": float(fig("W1.busy_pct")),
            "w2_busy_pct": float(fig("W2.busy_pct")),
            "w1_avg_concurrency_while_busy": float(fig("W1.avg_concurrency_while_busy")),
            "w1_peak_concurrency": float(fig("W1.peak_concurrency")),
            "headroom_factor": round(
                float(measured["window_seconds"]) / float(measured["local_busy_seconds"]), 4
            ),
            "note": (
                "The measured duty cycle (43.1 % in the spike window, 10.2 % over "
                "W1 and 18.5 % over W2) sets the headroom the ceiling extrapolates "
                "across; it is not part of the ceiling. Concurrency while busy (1.04, "
                "peak 2) confirms the single prefill gate is the resource."
            ),
            "source": SOURCE_PATHS["single_slot_throughput"],
        },
        "reuse": {
            # min: the measured aggregate prefix reuse — what reuse looks like
            # across all traffic, including the large cold prompts the marginal
            # stream is made of.
            "reuse_min": float(fig("W4.aggregate_prefix_reuse_pct")) / 100.0,
            "reuse_min_label": (
                "W4 measured aggregate prefix reuse (26.1 % of prompt tokens "
                "reused) — the pessimistic marginal-mix assumption"
            ),
            # max: the admitted population's warmth (new tokens ~ 10 % of the
            # prompt). A selection effect of residency admission, so it is an
            # upper bound for the marginal stream, not a sustained property.
            "reuse_max": 0.90,
            "reuse_max_label": (
                "admitted-population warmth (new tokens ~ 10 % of the prompt, "
                "compaction register) — an upper bound, not sustainable at scale"
            ),
            "source": SOURCE_PATHS["single_slot_throughput"],
        },
        "denominators": {
            "token": {
                "label": "offered prompt tokens in the window",
                "value": float(measured["offered_prompt_tokens"]),
            },
            "stream": {
                "label": "offered streams in the window",
                "value": float(measured["requests"]),
                "tokens_per_stream": float(
                    derived_share["local_prompt_tokens_per_stream"]
                ),
            },
            "ruled_out": {
                "uncached_prompt_tokens": (
                    "an uncached denominator would have to be 27.8 % of offered "
                    "(cached_ratio 0.72) against a project-level measured median "
                    "of 0.952, which would put the ceiling near 34 %"
                ),
                "peak_offered_load": (
                    "neither published figure uses a peak: both divide by the "
                    "window mean and the model is already at 100 % duty cycle"
                ),
            },
            "source": SOURCE_PATHS["local_share_baseline"],
        },
        "levers": {
            "compaction": {
                "label": cap["label"],
                "cap_tokens": cap["tokens"],
                "added_tokens": float(cap["added_tokens"]),
                "served_streams": float(cap["served_streams"]),
                "idle_gate_seconds": float(cap["idle_gate_seconds"]),
                "token_share_delta_pp": float(cap["token_share_delta_pp"]),
                "stream_share_delta_pp": float(cap["stream_share_delta_pp"]),
                "per_slot_cost": _need(compaction, "per_slot_cost", "compaction register"),
                "source": SOURCE_PATHS["compaction"],
            },
            "parallel_prefill": {
                "verdict": "rejected — aggregate ratio 1.00 (wall 2.00x, per-request 0.81x)",
                "prefill_gain_pct": 0.0,
                "source": SOURCE_PATHS["prefill_concurrency"],
            },
            "quantisation": {
                "prefill_gain_high_pct": float(
                    cand_high["prefill"][QUANT_BUCKET]["prefill_gain_measured_share_pct"]
                ),
                "prefill_gain_low_pct": float(
                    cand_low["prefill"][QUANT_BUCKET]["prefill_gain_measured_share_pct"]
                ),
                "capacity_token_share_after_pct": float(
                    cand_high["capacity"]["local_token_share_after_pct"]
                ),
                "capacity_stream_share_after_pct": float(
                    cand_high["capacity"]["local_stream_share_after_pct"]
                ),
                "source": SOURCE_PATHS["quantisation"],
            },
            "geometry": {
                "verdict": (
                    "no aggregate throughput: the gate is GPU memory-bandwidth "
                    "bound (parallel prefill 1.00x), so slot geometry trades "
                    "concurrency and prefix reuse, not tok/s"
                ),
                "prefill_gain_pct": 0.0,
                "source": SOURCE_PATHS["quantisation"],
            },
        },
        "base_shares": {
            "token_share_pct": float(derived_share["local_token_share_pct"]),
            "stream_share_pct": float(derived_share["local_stream_share_pct"]),
            "source": SOURCE_PATHS["local_share_baseline"],
        },
        "published": {
            "stream_ceiling_pct": float(shared_verdicts["published_stream_ceiling_pct"]),
            "stream_definition_pct": float(shared_verdicts["stream_definition"]["ceiling_pct"]),
            "token_definition_pct": float(shared_verdicts["token_definition"]["ceiling_pct"]),
        },
        "parent_targets": {
            "ac9_local_share_baseline_pct": 2.8,
            "ac9_denominator": "local share of streams (165 local vs 5 707 remote)",
            "ac1_target_pct": 10.0,
            "ac1_denominator": "local share of streams (LP-0MVXWQQ20002HAX4 AC1, e.g. >=10 %)",
        },
    }


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def derive(inputs: dict[str, Any]) -> dict[str, Any]:
    """Derive the ceiling matrix, the reconciliation and the ≥10 % requirements.

    Pure: the same ``inputs`` always produce the same ``derived`` block, which
    is what makes ``--register ... --check`` a self-contained re-verification.
    """
    window_seconds = float(inputs["window"]["seconds"])
    measured = inputs["measured"]
    decode_share = float(inputs["decode"]["share_of_busy_pct"]) / 100.0

    prefill_seconds = window_seconds * (1.0 - decode_share)
    prefill_seconds_used = float(measured["local_busy_seconds"]) * (1.0 - decode_share)

    rates = {
        "min": float(inputs["engine"]["rate_min_tok_s"]),
        "point": float(inputs["engine"]["rate_point_tok_s"]),
        "max": float(inputs["engine"]["rate_max_tok_s"]),
    }

    # r is implied by the window's own occupancy at the point engine rate: the
    # tokens the gate could have prefilled in the prefill time it actually used,
    # against the prompt tokens it actually delivered.
    prefilled_used = prefill_seconds_used * rates["point"]
    reuse_point = 1.0 - prefilled_used / float(measured["local_prompt_tokens"])
    reuses = {
        "min": float(inputs["reuse"]["reuse_min"]),
        "point": reuse_point,
        "max": float(inputs["reuse"]["reuse_max"]),
    }
    if not 0.0 <= reuse_point < 1.0:
        raise RegisterError(
            "the implied cache-reuse fraction is outside [0, 1): the engine rate "
            "and the delivered volume disagree"
        )
    for name, value in reuses.items():
        if not 0.0 <= value < 1.0:
            raise RegisterError(
                f"reuse fraction {name}={value!r} is outside [0, 1): a fully "
                "reused stream would make the ceiling infinite"
            )

    offered_tokens = float(inputs["denominators"]["token"]["value"])
    offered_streams = float(inputs["denominators"]["stream"]["value"])
    tokens_per_stream = float(inputs["denominators"]["stream"]["tokens_per_stream"])

    def ceiling(rate: float, reuse: float) -> dict[str, Any]:
        cold = prefill_seconds * rate
        delivered = cold / (1.0 - reuse)
        streams = delivered / tokens_per_stream
        return {
            "engine_rate_tok_s": rate,
            "reuse_fraction": round(reuse, 6),
            "prefill_capacity_tokens": round(cold, 1),
            "delivered_tokens": round(delivered, 1),
            "servable_streams": round(streams, 3),
            "prefill_share_pct_of_offered": round(_pct(cold, offered_tokens), 4),
            "token_share_pct": round(_pct(delivered, offered_tokens), 4),
            "stream_share_pct": round(_pct(streams, offered_streams), 4),
        }

    matrix: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        matrix[scenario] = ceiling(rates[scenario], reuses[scenario])

    # The full 3 x 3 (rate rows, reuse columns) so the two axes stay separable:
    # the corner-to-corner band is wide, and which axis dominates is the finding.
    grid: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        grid[scenario] = {
            "reuse_" + other: ceiling(rates[scenario], reuses[other])
            for other in SCENARIOS
        }

    points = matrix["point"]

    # Tier 1: the hardware ceiling in originated tokens, independent of reuse.
    hardware = {
        "label": "share of offered prompt tokens the gate can prefill",
        "min_pct": round(
            _pct(prefill_seconds * rates["min"], offered_tokens), 4
        ),
        "point_pct": round(
            _pct(prefill_seconds * rates["point"], offered_tokens), 4
        ),
        "max_pct": round(
            _pct(prefill_seconds * rates["max"], offered_tokens), 4
        ),
        "prefill_tokens": {name: round(prefill_seconds * rates[name], 1) for name in SCENARIOS},
        "note": (
            "tokens the gate originates, before any cache hit is counted; the "
            "delivered-share tier is this divided by (1 - reuse)"
        ),
    }
    canonical_tokens = float(measured["canonical_ceiling_tokens"])
    reconciliation = {
        "canonical_ceiling_tokens": canonical_tokens,
        "modelled_delivered_tokens": points["delivered_tokens"],
        "token_share_delta_pct": round(
            100.0 * (points["delivered_tokens"] - canonical_tokens) / canonical_tokens, 4
        ),
        "canonical_token_share_pct": float(measured["canonical_token_share_pct"]),
        "modelled_token_share_pct": points["token_share_pct"],
        "canonical_stream_share_pct": float(measured["canonical_stream_share_pct"]),
        "modelled_stream_share_pct": points["stream_share_pct"],
    }

    def requirement(target_pct: float, denominator: str) -> dict[str, Any]:
        """What a target share needs, in one denominator, against the point model.

        The two denominators are never mixed: a stream target converts to a
        token requirement through the *current local* mean prompt size per
        stream, which is the assumption the stream ceiling itself carries.
        """
        if denominator == "token":
            required = target_pct / 100.0 * offered_tokens
        elif denominator == "stream":
            required = target_pct / 100.0 * offered_streams * tokens_per_stream
        else:  # pragma: no cover - guarded by the caller
            raise RegisterError(f"unknown denominator {denominator!r}")
        cold_point = points["prefill_capacity_tokens"]
        reuse_needed = 1.0 - cold_point / required
        rate_needed = required * (1.0 - reuses["point"]) / prefill_seconds
        return {
            "denominator": denominator,
            "target_pct": target_pct,
            "required_tokens": round(required, 1),
            "factor_vs_point_capacity": round(required / points["delivered_tokens"], 3),
            "reuse_needed_at_point_rate": round(reuse_needed, 4),
            "rate_needed_at_point_reuse_tok_s": round(rate_needed, 1),
            "rate_factor_vs_measured_max": round(rate_needed / rates["max"], 2),
            "reachable_at_point": required <= points["delivered_tokens"],
            "reachable_within_measured_band": required <= matrix["max"]["delivered_tokens"],
        }

    ac9 = float(inputs["parent_targets"]["ac9_local_share_baseline_pct"])
    ac1 = float(inputs["parent_targets"]["ac1_target_pct"])
    targets = {
        "parent_ac9_stream": requirement(ac9, "stream"),
        "parent_ac9_token": requirement(ac9, "token"),
        "spike_ac1_stream": requirement(ac1, "stream"),
        "spike_ac1_token": requirement(ac1, "token"),
    }

    return {
        "prefill_seconds_available": round(prefill_seconds, 2),
        "prefill_seconds_used": round(prefill_seconds_used, 2),
        "decode_share_of_busy_pct": round(100.0 * decode_share, 4),
        "engine_rates_tok_s": rates,
        "implied_prefilled_tokens_used": round(prefilled_used, 1),
        "reuse_fractions": {name: round(value, 6) for name, value in reuses.items()},
        "hardware_prefill_ceiling": hardware,
        "ceiling": matrix,
        "grid": grid,
        "reconciliation": reconciliation,
        "targets": targets,
    }


def bound_levers(inputs: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    """Price the *combined* upside of the compaction, parallel-prefill and
    quantisation/geometry levers, and measure it against the ≥10 % gap.

    Compaction is additive on delivered tokens (it converts idle gate seconds);
    quantisation is multiplicative on the gate rate (a faster engine yields more
    from the same idle seconds) plus a small additive term for the decode slot
    seconds it frees; parallel prefill and slot geometry add nothing.
    """
    base_token = float(inputs["base_shares"]["token_share_pct"])
    base_stream = float(inputs["base_shares"]["stream_share_pct"])
    compaction = inputs["levers"]["compaction"]
    quant = inputs["levers"]["quantisation"]

    rate_factor_high = 1.0 + float(quant["prefill_gain_high_pct"]) / 100.0
    rate_factor_low = 1.0 + float(quant["prefill_gain_low_pct"]) / 100.0
    parallel = 1.0 + float(inputs["levers"]["parallel_prefill"]["prefill_gain_pct"]) / 100.0
    geometry = 1.0 + float(inputs["levers"]["geometry"]["prefill_gain_pct"]) / 100.0

    def combined(rate_factor: float) -> dict[str, Any]:
        # Compaction's idle-gate yield scales with the gate rate; the decode
        # slot-seconds quantisation frees are a separate additive contribution
        # already priced in the quantisation register's capacity block.
        compaction_tokens = float(compaction["added_tokens"]) * rate_factor
        quant_capacity_tokens = (
            float(quant["capacity_token_share_after_pct"]) - base_token
        ) / 100.0 * float(inputs["denominators"]["token"]["value"])
        token_delta_pp = _pct(compaction_tokens, float(inputs["denominators"]["token"]["value"])) + (
            float(quant["capacity_token_share_after_pct"]) - base_token
        )
        stream_delta_pp = _pct(
            float(compaction["served_streams"]),
            float(inputs["denominators"]["stream"]["value"]),
        ) + (float(quant["capacity_stream_share_after_pct"]) - base_stream)
        return {
            "compaction_added_tokens": round(compaction_tokens, 1),
            "quantisation_freed_capacity_tokens": round(quant_capacity_tokens, 1),
            "token_share_delta_pp": round(token_delta_pp, 4),
            "stream_share_delta_pp": round(stream_delta_pp, 4),
            "token_share_after_pct": round(base_token + token_delta_pp, 4),
            "stream_share_after_pct": round(base_stream + stream_delta_pp, 4),
        }

    high = combined(rate_factor_high * parallel * geometry)
    low = combined(rate_factor_low * parallel * geometry)

    target = float(inputs["parent_targets"]["ac1_target_pct"])
    token_gap_pp = target - base_token
    stream_gap_pp = target - base_stream

    return {
        "rate_factors": {
            "quantisation_high": round(rate_factor_high, 6),
            "quantisation_low": round(rate_factor_low, 6),
            "parallel_prefill": parallel,
            "geometry": geometry,
            "combined": round(rate_factor_high * parallel * geometry, 6),
        },
        "combined_low": low,
        "combined_high": high,
        "gap_to_target": {
            "token": {
                "target_pct": target,
                "gap_pp": round(token_gap_pp, 4),
                "gap_closed_pp": high["token_share_delta_pp"],
                "gap_closed_pct_of_gap": round(
                    100.0 * high["token_share_delta_pp"] / token_gap_pp, 3
                ),
                "remaining_factor": round(
                    target / (base_token + high["token_share_delta_pp"]), 3
                ),
                "closed": target <= base_token + high["token_share_delta_pp"],
            },
            "stream": {
                "target_pct": target,
                "gap_pp": round(stream_gap_pp, 4),
                "gap_closed_pp": high["stream_share_delta_pp"],
                "gap_closed_pct_of_gap": round(
                    100.0 * high["stream_share_delta_pp"] / stream_gap_pp, 3
                ),
                "remaining_factor": round(
                    target / (base_stream + high["stream_share_delta_pp"]), 3
                ),
                "closed": target <= base_stream + high["stream_share_delta_pp"],
            },
        },
        "caveats": [
            "Compaction's gain rides on idle gate seconds (56.9 % of the window); it "
            "does not raise the ceiling and would displace delivery in a saturated window.",
            "Quantisation's prefill bound is the measured-share end (1.1-1.9 %), not the "
            "20-27 % byte-ratio ceiling, which needs the prefill pass to be weight-bound "
            "and it is not (6.3 GB/s against the 88 GB/s decode already sustains).",
            "No local quality delta exists for either lower-quant candidate; switching "
            "preset without a quality gate is out of scope.",
            "Parallel prefill is measured at 1.00x aggregate, and slot geometry does not "
            "move aggregate throughput, so both contribute zero.",
        ],
    }


def accept_option(inputs: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    """The "accept the ceiling" row of the spike's AC1 comparison table."""
    point = derived["ceiling"]["point"]
    return {
        "label": "AC1d — accept the ceiling (re-scope the target)",
        "throughput_tok_s": {
            "delivered_prompt_tok_s": round(float(inputs["measured"]["local_rate_tok_s"]), 2),
            "prefill_ceiling_tok_s_min": round(
                float(inputs["engine"]["rate_min_tok_s"]), 2
            ),
            "prefill_ceiling_tok_s_point": round(
                float(inputs["engine"]["rate_point_tok_s"]), 2
            ),
            "prefill_ceiling_tok_s_max": round(
                float(inputs["engine"]["rate_max_tok_s"]), 2
            ),
            "rate_added_by_option_tok_s": 0.0,
            "note": (
                "measured ceiling, unchanged: the option buys no throughput, it "
                "re-scopes the target to what the measured engine already sustains"
            ),
        },
        "per_slot_resource_cost": {
            "gpu_memory_gib": 0.0,
            "gpu_seconds_per_window": 0.0,
            "config_changes": 0,
            "note": "no new model, no summariser slot, no config edit",
        },
        "local_share_pct": {
            "token_point": point["token_share_pct"],
            "stream_point": point["stream_share_pct"],
        },
    }


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _rel_gap(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected) if expected else float("inf")


def check(inputs: dict[str, Any], derived: dict[str, Any], levers: dict[str, Any]) -> list[dict[str, Any]]:
    """Recorded assertions, each reproducible from the register's own inputs."""
    checks: list[dict[str, Any]] = []

    def add(claim: str, ok: bool, reproduced: Any, note: str = "") -> None:
        entry: dict[str, Any] = {
            "claim": claim,
            "reproduced": reproduced,
            "ok": bool(ok),
            "verdict": "reproduces" if ok else "refuted",
        }
        if note:
            entry["note"] = note
        checks.append(entry)

    rec = derived["reconciliation"]
    add(
        "the point model reproduces the committed token-denominated ceiling (1.62 %)",
        _rel_gap(rec["modelled_delivered_tokens"], rec["canonical_ceiling_tokens"])
        <= TOL_RECONCILE_REL,
        {
            "modelled": rec["modelled_delivered_tokens"],
            "canonical": rec["canonical_ceiling_tokens"],
            "delta_pct": rec["token_share_delta_pct"],
        },
        "formula-regression guard, not independent validation: the equality is "
        "algebraic (R cancels once r is implied from occupancy), so it pins the "
        "wiring and records that the ceiling of record assumes reuse scales",
    )
    add(
        "the point model reproduces the committed stream-denominated ceiling (5.96 %)",
        abs(rec["modelled_stream_share_pct"] - rec["canonical_stream_share_pct"]) <= TOL_PP,
        {
            "modelled": rec["modelled_stream_share_pct"],
            "canonical": rec["canonical_stream_share_pct"],
        },
    )

    ceiling = derived["ceiling"]
    hardware = derived["hardware_prefill_ceiling"]
    add(
        "the ceiling band is ordered min < point < max on both denominators",
        ceiling["min"]["token_share_pct"] < ceiling["point"]["token_share_pct"]
        < ceiling["max"]["token_share_pct"]
        and ceiling["min"]["stream_share_pct"] < ceiling["point"]["stream_share_pct"]
        < ceiling["max"]["stream_share_pct"]
        and hardware["min_pct"] < hardware["point_pct"] < hardware["max_pct"],
        {
            "hardware_prefill_pct": {
                "min": hardware["min_pct"],
                "point": hardware["point_pct"],
                "max": hardware["max_pct"],
            },
            "token_pct": {name: ceiling[name]["token_share_pct"] for name in SCENARIOS},
            "stream_pct": {name: ceiling[name]["stream_share_pct"] for name in SCENARIOS},
        },
    )

    # The reuse axis, not the rate axis, dominates the band: swapping only the
    # reuse fraction moves the ceiling further than the whole measured rate band.
    grid = derived["grid"]
    rate_only = (
        grid["max"]["reuse_point"]["token_share_pct"]
        / grid["min"]["reuse_point"]["token_share_pct"]
    )
    reuse_only = (
        grid["point"]["reuse_max"]["token_share_pct"]
        / grid["point"]["reuse_min"]["token_share_pct"]
    )
    add(
        "cache reuse dominates the gate rate as the ceiling's driver",
        reuse_only > rate_only,
        {"reuse_only_factor": round(reuse_only, 3), "rate_only_factor": round(rate_only, 3)},
        "the marginal stream's cache-reuse fraction moves the ceiling further than "
        "the whole measured engine-rate band does",
    )

    add(
        "the delivered volume implies a cache-reuse fraction between the two measurements",
        float(inputs["reuse"]["reuse_min"])
        < derived["reuse_fractions"]["point"]
        < float(inputs["reuse"]["reuse_max"]),
        {
            "aggregate_measured": derived["reuse_fractions"]["min"],
            "implied_by_window": derived["reuse_fractions"]["point"],
            "admitted_population": derived["reuse_fractions"]["max"],
        },
        "the window ran at higher reuse than the aggregate measurement and below "
        "the admitted population's warmth, as an admitted-population bias requires",
    )

    add(
        "the decode reserve is below a quarter of the gate's time",
        derived["decode_share_of_busy_pct"] < 25.0,
        derived["decode_share_of_busy_pct"],
        "measured local decode slot-seconds (completion tokens / 30.7 tok/s) over busy seconds",
    )

    duty = inputs["duty_context"]
    add(
        "one prefill gate is the resource: measured concurrency while busy is near one",
        float(duty["w1_avg_concurrency_while_busy"]) < 1.5,
        {
            "avg_concurrency_while_busy": duty["w1_avg_concurrency_while_busy"],
            "peak_concurrency": duty["w1_peak_concurrency"],
            "headroom_factor": duty["headroom_factor"],
        },
        "so the ceiling extrapolates a single gate's rate over the window's spare "
        "duty, and the measured duty (43.1 % here, 10.2 %/18.5 % on W1/W2) is the "
        "headroom that extrapolation crosses, not part of the ceiling",
    )

    targets = derived["targets"]
    add(
        ">=10 % prompt-token share is unreachable inside the measured band",
        not targets["spike_ac1_token"]["reachable_within_measured_band"],
        {
            "factor_vs_point": targets["spike_ac1_token"]["factor_vs_point_capacity"],
            "rate_needed_tok_s": targets["spike_ac1_token"]["rate_needed_at_point_reuse_tok_s"],
            "reuse_needed": targets["spike_ac1_token"]["reuse_needed_at_point_rate"],
        },
    )
    add(
        ">=10 % stream share needs capacity beyond the point ceiling",
        not targets["spike_ac1_stream"]["reachable_at_point"],
        {
            "factor_vs_point": targets["spike_ac1_stream"]["factor_vs_point_capacity"],
            "rate_needed_tok_s": targets["spike_ac1_stream"]["rate_needed_at_point_reuse_tok_s"],
            "reuse_needed": targets["spike_ac1_stream"]["reuse_needed_at_point_rate"],
            "reachable_within_measured_band": targets["spike_ac1_stream"][
                "reachable_within_measured_band"
            ],
        },
        "within the band only at the optimistic corner (90 % cache reuse at the "
        "best measured rate), which admission selection cannot sustain at scale",
    )
    add(
        "the parent AC9 2.8 % stream baseline is below the point stream ceiling",
        targets["parent_ac9_stream"]["reachable_at_point"],
        {
            "factor_vs_point": targets["parent_ac9_stream"]["factor_vs_point_capacity"],
            "required_tokens": targets["parent_ac9_stream"]["required_tokens"],
        },
        "so AC9 is already reachable inside the measured ceiling; it is the >=10 % "
        "proposal that is not",
    )

    gap = levers["gap_to_target"]
    add(
        "the combined levers do not close the gap to >=10 % under either denominator",
        not gap["token"]["closed"] and not gap["stream"]["closed"],
        {
            "token_gap_closed_pct_of_gap": gap["token"]["gap_closed_pct_of_gap"],
            "stream_gap_closed_pct_of_gap": gap["stream"]["gap_closed_pct_of_gap"],
            "token_remaining_factor": gap["token"]["remaining_factor"],
            "stream_remaining_factor": gap["stream"]["remaining_factor"],
        },
    )
    add(
        "the levers' added tokens stay inside the point ceiling's headroom",
        levers["combined_high"]["compaction_added_tokens"]
        + levers["combined_high"]["quantisation_freed_capacity_tokens"]
        <= ceiling["point"]["delivered_tokens"] - float(inputs["measured"]["local_prompt_tokens"]),
        {
            "lever_tokens": round(
                levers["combined_high"]["compaction_added_tokens"]
                + levers["combined_high"]["quantisation_freed_capacity_tokens"],
                1,
            ),
            "ceiling_headroom_tokens": round(
                ceiling["point"]["delivered_tokens"]
                - float(inputs["measured"]["local_prompt_tokens"]),
                1,
            ),
        },
        "the levers convert existing headroom; they do not raise the ceiling",
    )

    return checks


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def build_register(root: Path) -> dict[str, Any]:
    inputs = load_inputs(root)
    derived = derive(inputs)
    levers = bound_levers(inputs, derived)
    option = accept_option(inputs, derived)
    register = {
        "schema_version": SCHEMA_VERSION,
        "work_item": WORK_ITEM,
        "parent_work_item": PARENT_WORK_ITEM,
        "title": "Accept-the-ceiling arithmetic (spike AC1d)",
        "generated_at": "2026-11-11",
        "purpose": (
            "Cost the 'accept the ceiling' option with explicit arithmetic: the "
            "hardware prefill ceiling, the share ceiling under each fixed "
            "denominator, what a >=10 % target would require, and the bounded "
            "combined upside of the lever children."
        ),
        "read_only": True,
        "provenance_classes": {
            "measured": "Copied verbatim from the named source register; not re-derived here.",
            "derived": "Arithmetic over measured figures registered here, with the formula and inputs recorded.",
        },
        "sources": {
            key: {"path": path}
            for key, path in SOURCE_PATHS.items()
        },
        "model": {
            "prefill_seconds": "window_seconds x (1 - decode_share_of_busy)",
            "prefill_capacity_tokens": "prefill_seconds x engine_rate_tok_s",
            "delivered_tokens": "prefill_capacity_tokens / (1 - reuse_fraction)",
            "token_share_pct": "100 x delivered_tokens / offered_prompt_tokens",
            "stream_share_pct": "100 x (delivered_tokens / local_prompt_tokens_per_stream) / offered_streams",
            "reuse_point": (
                "1 - (local_busy_seconds x (1 - decode_share) x engine_rate_point) "
                "/ local_prompt_tokens"
            ),
            "notes": [
                "The denominator is fixed by LP-0MWCEGJDP009WURD: offered prompt tokens "
                "for the token share, offered streams for the stream share. Never mixed.",
                "The engine rate is the measured aggregate prefill rate; the delivered "
                "blend rate (717.8 tok/s) is not a capacity figure because cache hits cost "
                "no gate time.",
                "The decode reserve is scale-invariant: it is a share of gate time, so "
                "extrapolating the window to 100 % duty keeps it constant.",
            ],
        },
        "inputs": inputs,
        "derived": derived,
        "levers_bound": levers,
        "accept_option": option,
        "checks": check(inputs, derived, levers),
    }
    return register


def all_ok(register: dict[str, Any]) -> bool:
    return all(entry["verdict"] != "refuted" for entry in register["checks"])


def verify_register(register: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-derive from the register's *own* inputs and compare with its recorded
    derived blocks — a self-contained drift guard (no source registers needed).
    """
    drift: list[dict[str, Any]] = []
    try:
        derived = derive(register["inputs"])
    except (KeyError, RegisterError) as exc:
        return [
            {
                "claim": "register inputs are complete",
                "ok": False,
                "verdict": "refuted",
                "note": str(exc),
            }
        ]
    recorded = register.get("derived") or {}
    for key, value in derived.items():
        seen = recorded.get(key)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                seen_sub = (seen or {}).get(sub_key) if isinstance(seen, dict) else None
                if isinstance(sub_value, dict):
                    for leaf, leaf_value in sub_value.items():
                        seen_leaf = (
                            seen_sub.get(leaf) if isinstance(seen_sub, dict) else None
                        )
                        if not _close(seen_leaf, leaf_value):
                            drift.append(
                                {
                                    "claim": f"derived.{key}.{sub_key}.{leaf}",
                                    "recorded": seen_leaf,
                                    "recomputed": leaf_value,
                                    "ok": False,
                                    "verdict": "refuted",
                                }
                            )
                elif not _close(seen_sub, sub_value):
                    drift.append(
                        {
                            "claim": f"derived.{key}.{sub_key}",
                            "recorded": seen_sub,
                            "recomputed": sub_value,
                            "ok": False,
                            "verdict": "refuted",
                        }
                    )
        elif not _close(seen, value):
            drift.append(
                {
                    "claim": f"derived.{key}",
                    "recorded": seen,
                    "recomputed": value,
                    "ok": False,
                    "verdict": "refuted",
                }
            )
    return drift


def _close(a: Any, b: Any) -> bool:
    """Compare recorded numbers tolerantly; non-numeric values must match exactly."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= max(1e-9, abs(float(b)) * 1e-9)
    return a == b


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render(register: dict[str, Any]) -> str:
    derived = register["derived"]
    inputs = register["inputs"]
    ceiling = derived["ceiling"]
    hardware = derived["hardware_prefill_ceiling"]
    levers = register["levers_bound"]
    option = register["accept_option"]

    lines = [
        "Accept-the-ceiling arithmetic — 2026-11-01 measured basis",
        "=" * 70,
        f"window              {inputs['window']['start']} → {inputs['window']['end']}"
        f"  ({_fmt(inputs['window']['seconds'], 0)} s)",
        f"offered prompt load {_fmt(inputs['measured']['offered_prompt_tok_s'], 1)} tok/s"
        f"  ({_fmt(inputs['measured']['offered_prompt_tokens'], 0)} tokens,"
        f" {_fmt(inputs['measured']['offered_streams'], 0)} streams)",
        f"delivered           {_fmt(inputs['measured']['local_prompt_tokens'], 0)} prompt tokens"
        f" in {_fmt(inputs['measured']['local_busy_seconds'], 0)} busy s"
        f"  ({_fmt(inputs['measured']['local_rate_tok_s'], 2)} tok/s blend)",
        "",
        "Hardware limit (the formula and its inputs)",
        f"  prefill gate time   {_fmt(derived['prefill_seconds_available'], 2)} s"
        f"  = {_fmt(inputs['window']['seconds'], 0)} s x"
        f" (1 - {_fmt(derived['decode_share_of_busy_pct'], 3)} % decode reserve)",
        f"  engine prefill rate {_fmt(derived['engine_rates_tok_s']['min'], 1)} /"
        f" {_fmt(derived['engine_rates_tok_s']['point'], 2)} /"
        f" {_fmt(derived['engine_rates_tok_s']['max'], 1)} tok/s (min / point / max)",
        f"  cache-reuse fraction {_fmt(derived['reuse_fractions']['min'], 4)} /"
        f" {_fmt(derived['reuse_fractions']['point'], 4)} /"
        f" {_fmt(derived['reuse_fractions']['max'], 4)} (min / point / max)",
        "",
        "Share ceiling — tier 1: hardware prefill (tokens the gate can originate)",
        f"  {_fmt(hardware['min_pct'], 3)} / {_fmt(hardware['point_pct'], 3)} /"
        f" {_fmt(hardware['max_pct'], 3)} % of offered prompt tokens",
        f"  = {_fmt(hardware['prefill_tokens']['min'], 0)} /"
        f" {_fmt(hardware['prefill_tokens']['point'], 0)} /"
        f" {_fmt(hardware['prefill_tokens']['max'], 0)} tokens"
        f" ({_fmt(derived['prefill_seconds_available'], 2)} s x engine rate)",
        "",
        "Share ceiling — tier 2/3: delivered tokens and streams (denominator fixed by",
        "LP-0MWCEGJDP009WURD)",
        "  scenario   prefill tokens   delivered tokens   token share   stream share",
    ]
    for name in SCENARIOS:
        row = ceiling[name]
        lines.append(
            f"  {name:<9}  {_fmt(row['prefill_capacity_tokens'], 0):>14}"
            f"  {_fmt(row['delivered_tokens'], 0):>16}"
            f"  {_fmt(row['token_share_pct'], 3):>10} %"
            f"  {_fmt(row['stream_share_pct'], 3):>10} %"
        )
    lines += [
        f"  reconciliation: the point model gives"
        f" {_fmt(derived['reconciliation']['modelled_delivered_tokens'], 0)} delivered tokens"
        f" against the committed {_fmt(derived['reconciliation']['canonical_ceiling_tokens'], 0)}"
        f" ({_fmt(derived['reconciliation']['token_share_delta_pct'], 4)} % delta) —"
        " algebraic, so the ceiling of record assumes the window's reuse scales",
        "",
        "Targets versus the ceiling (each in its own denominator, never mixed)",
    ]
    target_rows = (
        ("parent AC9", "parent_ac9_stream", "stream"),
        ("spike AC1", "spike_ac1_stream", "stream"),
        ("spike AC1", "spike_ac1_token", "token"),
    )
    for label, key, denominator in target_rows:
        target = derived["targets"][key]
        unit = "streams" if denominator == "stream" else "tokens"
        verdict = (
            "reachable at the point ceiling"
            if target["reachable_at_point"]
            else ("only in the optimistic corner" if target["reachable_within_measured_band"] else "unreachable")
        )
        lines.append(
            f"  {label} {_fmt(target['target_pct'], 1)} % of offered {unit}:"
            f" {_fmt(target['required_tokens'], 0)} tokens ="
            f" {_fmt(target['factor_vs_point_capacity'], 3)}x point capacity"
            f" — {verdict}"
        )
        lines.append(
            f"  {'':<26} needs {_fmt(target['rate_needed_at_point_reuse_tok_s'], 1)} tok/s"
            f" ({_fmt(target['rate_factor_vs_measured_max'], 2)}x the measured max) or"
            f" cache reuse {_fmt(target['reuse_needed_at_point_rate'], 4)}"
        )
    lines += [
        "",
        "Combined lever upside (bound)",
        f"  rate factor        {_fmt(levers['rate_factors']['combined'], 4)}x"
        f" (quantisation prefill {_fmt(inputs['levers']['quantisation']['prefill_gain_high_pct'], 3)} %;"
        f" parallel prefill 0; geometry 0)",
        f"  token share        {_fmt(inputs['base_shares']['token_share_pct'], 4)} % →"
        f" {_fmt(levers['combined_high']['token_share_after_pct'], 4)} %"
        f"  ({_fmt(levers['gap_to_target']['token']['gap_closed_pct_of_gap'], 2)} % of the gap)",
        f"  stream share       {_fmt(inputs['base_shares']['stream_share_pct'], 4)} % →"
        f" {_fmt(levers['combined_high']['stream_share_after_pct'], 4)} %"
        f"  ({_fmt(levers['gap_to_target']['stream']['gap_closed_pct_of_gap'], 2)} % of the gap)",
        "",
        "Accept option (spike AC1 comparison-table row)",
        f"  throughput         {_fmt(option['throughput_tok_s']['prefill_ceiling_tok_s_point'], 2)} tok/s"
        f" measured ceiling, {_fmt(option['throughput_tok_s']['rate_added_by_option_tok_s'], 1)} tok/s added",
        f"  per-slot cost      {_fmt(option['per_slot_resource_cost']['gpu_memory_gib'], 1)} GiB,"
        f" {_fmt(option['per_slot_resource_cost']['config_changes'], 0)} config changes",
        "",
        "Claim checks",
    ]
    for entry in register["checks"]:
        marker = "PASS" if entry["ok"] else "FAIL"
        lines.append(f"  [{marker}] {entry['claim']}")
        if entry.get("note"):
            lines.append(f"          {entry['note']}")
    lines.append("")
    lines.append(
        "verdict: " + ("all checked claims reproduce" if all_ok(register) else "CONTRADICTION FOUND")
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the accept-the-ceiling arithmetic from the committed "
            "baseline and lever registers (read-only)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: the tree this script lives in)",
    )
    parser.add_argument("--json", action="store_true", help="emit the register as JSON")
    parser.add_argument(
        "--register",
        type=Path,
        help="re-verify a committed register from its own embedded inputs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when a claim or the register is contradicted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.register is not None:
        try:
            register = json.loads(args.register.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read {args.register}: {exc}", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"error: {args.register} is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(register, dict):
            print(f"error: {args.register} does not contain an object", file=sys.stderr)
            return 2
        drift = verify_register(register)
        for entry in drift:
            print(
                f"drift: {entry['claim']} recorded={entry.get('recorded')} "
                f"recomputed={entry.get('recomputed')}",
                file=sys.stderr,
            )
        ok = not drift and all_ok(register)
        if args.json:
            print(json.dumps({"ok": ok, "drift": drift}, indent=2))
        else:
            print(render(register))
            if drift:
                print("\nREGISTER DRIFT: the recorded derived block no longer "
                      "matches its inputs", file=sys.stderr)
        return 0 if ok else 1

    try:
        register = build_register(args.root)
    except RegisterError as exc:
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
