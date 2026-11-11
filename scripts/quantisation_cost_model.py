#!/usr/bin/env python3
"""Cost the model-weight quantisation lever for the local prefill ceiling.

Read-only analysis for **LP-0MWCEHYOO002TRGD** (spike parent
LP-0MVXY8ZBZ0003042, question AC1c). It answers, with arithmetic another
engineer can re-run:

1. **What does each candidate weigh?** — the exact weight bytes, parameters and
   bits-per-weight of every GGUF that is on this host, read from the file's own
   tensor table (no estimation). Plus the llama.cpp *plain-preset* prediction
   for the presets that are **not** on this host (Q4_K_M), computed by applying
   llama.cpp's own per-tensor type rules to the shipped tensor table.
2. **What does that buy for VRAM?** — the weight delta against the shipped
   `UD-Q5_K_M`, i.e. the shared-weights term of the per-slot memory cost.
3. **What does that buy for throughput?** — bounded, not guessed:

   * **prefill** — an *absolute ceiling* (every weight byte on the critical
     path: throughput scales as the byte ratio) and a *measured-share ceiling*
     (only the share of a prefill pass that is weight movement scales), with the
     measured prefill rate per prompt-size bucket as the anchor;
   * **decode** — the byte-ratio ceiling, anchored on the measured decode rate
     and the model's own per-token active-weight bytes.
4. **What does it buy for the spike's target metric?** — the local-share
   arithmetic: faster decode frees slot-seconds, which raises the capacity
   ceiling the local share is capped by.

Why this shape
--------------
Weight quantisation changes exactly one thing: **how many bytes one weight
occupies**. It does not change FLOPs, tokens, routing or the KV cache. So the
whole question is "which term of the engine's time is weight movement?", and
that question is answered from measurement, not from a rule of thumb:

* the shipped baseline is measured from the GGUF tensor table (24.629 GiB at
  6.104 bpw — the shipped `UD-Q5_K_M` is *not* a plain k-quant, it keeps 255
  tensors at Q8_0);
* the Q4_K_S candidate is **measured** the same way (19.448 GiB of base weights
  in the host's `UD-Q4_K_S` MTP GGUF — the same 34.661 B parameters);
* the prefill anchors are measured production `prompt eval time` lines, bucketed
  by prompt size (``scripts/prefill_throughput_report.py``);
* the decode anchor is the measured production decode rate, and the per-token
  active-weight bytes are computed from the tensor table (8 of 256 experts plus
  the shared expert, attention and output tensors).

The refutation that makes this cheap to settle: a prefill pass streams
``weight_bytes`` per 1024-token micro-batch, which at the measured 242 tok/s is
**~5 GB/s** — two orders of magnitude below what the host's memory can sustain.
Weight bytes therefore cannot be the prefill bottleneck, so no weight quant can
lift prefill by the byte ratio; only the small measured share of a pass that
*is* weight movement can move. Decode is the opposite regime (batch 1, no
weight reuse), which is why the lever is a decode/VRAM lever and not a prefill
lever.

Usage::

    # regenerate the committed register (needs the two GGUFs on this host)
    python3 scripts/quantisation_cost_model.py \\
        --gguf shipped=$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/*/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf \\
        --gguf q4_k_s=$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/*/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \\
        --prefill-json /tmp/prefill.json \\
        --json > docs/dev/quantisation-cost-2026-11-11.json

    # re-verify the committed register (self-contained: it embeds its inputs)
    python3 scripts/quantisation_cost_model.py \\
        --register docs/dev/quantisation-cost-2026-11-11.json --check

    # inspect one GGUF's own tensor table
    python3 scripts/quantisation_cost_model.py --gguf x=/path/to/model.gguf --layout
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# ggml type tables: (block size in weights, bytes per block)
# Source: ggml/src/ggml.c `type_traits[]` (ggml_blck_size / ggml_type_size).
# Types the shipped model or its k-quant candidates can contain are listed; an
# unknown type is a loud failure rather than a silently wrong byte count.
# ---------------------------------------------------------------------------
TYPE_BYTES: dict[str, tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q8_1": (32, 36),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
    "MXFP4": (32, 17),
}

#: ggml type enum -> name (ggml/include/ggml.h `enum ggml_type`).
TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    20: "IQ4_NL",
    23: "IQ4_XS",
    30: "BF16",
    39: "MXFP4",
}

#: GGUF metadata value types -> byte width (for skipping).
_KV_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

#: Micro-batch passed to the engine by `models.ini` (`ubatch-size = 1024`).
UBATCH_TOKENS = 1024

#: Weights-preset names this model implements.
PRESETS = ("q5_k_m", "q4_k_m", "q4_k_s")

#: Plain-preset default type (`GGML_TYPE_Q5_K`/`Q4_K` for the k-quant mixtures).
PRESET_DEFAULT = {"q5_k_m": "Q5_K", "q4_k_m": "Q4_K", "q4_k_s": "Q4_K"}


class GgufError(RuntimeError):
    """A GGUF file could not be read/understood."""


# ---------------------------------------------------------------------------
# GGUF tensor-table reader
# ---------------------------------------------------------------------------
def read_gguf_tensor_table(path: str | Path) -> list[dict]:
    """Return ``[{name, type, params, dims}]`` from a GGUF file's header.

    Only the metadata and tensor-info sections are read, so a 25 GiB model
    costs a few milliseconds and never touches the tensor data.
    """
    path = Path(path)
    with path.open("rb") as fh:
        if fh.read(4) != b"GGUF":
            raise GgufError(f"{path}: not a GGUF file")
        version = struct.unpack("<I", fh.read(4))[0]
        if version not in (2, 3):
            raise GgufError(f"{path}: unsupported GGUF version {version}")
        n_tensors = struct.unpack("<Q", fh.read(8))[0]
        n_kv = struct.unpack("<Q", fh.read(8))[0]

        def read_str() -> str:
            n = struct.unpack("<Q", fh.read(8))[0]
            return fh.read(n).decode("utf-8", errors="replace")

        for _ in range(n_kv):
            read_str()
            vtype = struct.unpack("<I", fh.read(4))[0]
            if vtype == 8:
                read_str()
            elif vtype == 9:
                etype = struct.unpack("<I", fh.read(4))[0]
                count = struct.unpack("<Q", fh.read(8))[0]
                for _ in range(count):
                    if etype == 8:
                        read_str()
                    else:
                        fh.read(_KV_BYTES[etype])
            else:
                fh.read(_KV_BYTES[vtype])

        tensors: list[dict] = []
        for _ in range(n_tensors):
            name = read_str()
            n_dims = struct.unpack("<I", fh.read(4))[0]
            dims = [struct.unpack("<Q", fh.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", fh.read(4))[0]
            struct.unpack("<Q", fh.read(8))  # data offset, unused
            params = 1
            for d in dims:
                params *= d
            if ggml_type not in TYPE_NAMES:
                raise GgufError(f"{path}: unknown ggml type {ggml_type} on {name}")
            tensors.append(
                {
                    "name": name,
                    "type": TYPE_NAMES[ggml_type],
                    "params": params,
                    "dims": dims,
                }
            )
    return tensors


def tensor_bytes(ggml_type: str, params: int) -> int:
    """Byte size of ``params`` weights stored as ``ggml_type``."""
    block, size = TYPE_BYTES[ggml_type]
    if params % block:
        raise GgufError(f"{params} weights not divisible by {ggml_type} block {block}")
    return (params // block) * size


# ---------------------------------------------------------------------------
# llama.cpp's per-tensor quantisation rules
#
# Both functions below are ports of llama.cpp's own logic, so the prediction for
# a preset that is *not* on this host uses the same rules `llama-quantize` would
# apply rather than a bits-per-weight rule of thumb:
#   * `allows_quantization`  <- `tensor_allows_quantization`  (src/llama-quant.cpp)
#   * `tensor_category`      <- `tensor_get_category`         (src/llama-quant.cpp)
#   * `use_more_bits` / `preset_type` <- `llama_tensor_get_type_impl`
# ---------------------------------------------------------------------------
_NO_QUANT_SUBSTRINGS = (
    "_norm.weight",
    "ffn_gate_inp.weight",  # expert router — never quantised
    "ssm_conv1d",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_dt.bias",
    "ssm_a",
    "altup",
    "laurel",
    "per_layer_model_proj",
    "attn_rel_b.weight",
    ".position_embd",
)

_LAYER_RE = re.compile(r"blk\.(\d+)\.")


def allows_quantization(name: str, n_dims: int) -> bool:
    """``tensor_allows_quantization``: 2D/3D ``weight`` tensors, minus the
    norms, the expert router and the small Mamba/SSM state tensors."""
    if n_dims < 2 or not name.endswith("weight"):
        return False
    return not any(s in name for s in _NO_QUANT_SUBSTRINGS)


def tensor_category(name: str) -> str:
    """``tensor_get_category``: the broad category the preset rules key on."""
    if name == "output.weight":
        return "OUTPUT"
    if name == "token_embd.weight":
        return "TOKEN_EMBD"
    for substr, category in (
        ("attn_qkv.weight", "ATTN_QKV"),
        ("attn_kv_b.weight", "ATTN_KV_B"),
        ("attn_v.weight", "ATTN_V"),
        ("attn_k.weight", "ATTN_K"),
        ("attn_q.weight", "ATTN_Q"),
        ("attn_output.weight", "ATTN_OUTPUT"),
        ("ffn_up", "FFN_UP"),
        ("ffn_gate", "FFN_GATE"),
        ("ffn_down", "FFN_DOWN"),
    ):
        if substr in name:
            return category
    return "OTHER"


def use_more_bits(i_layer: int, n_layers: int) -> bool:
    """llama.cpp's ``use_more_bits`` — the layers a `_M` mixture upgrades."""
    return (
        i_layer < n_layers // 8
        or i_layer >= 7 * n_layers // 8
        or (i_layer - n_layers // 8) % 3 == 2
    )


def _is_attn_v(category: str) -> bool:
    return category in ("ATTN_V", "ATTN_QKV", "ATTN_KV_B")


def preset_type(name: str, dims: list[int], preset: str) -> str:
    """Type ``llama-quantize --type <preset>`` would pick for one tensor.

    Implemented subset: non-Falcon architectures (this host's `qwen35moe`
    hybrid) with more than 8 experts. That is the branch set the shipped model
    and its Q4 candidates fall in; the 8-expert and 70B special cases are
    deliberately absent (they never apply here) and a wrong assumption shows up
    as a size mismatch against the measured layouts, which the checks catch.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
    if not allows_quantization(name, len(dims)):
        return None  # callers keep the tensor's original type
    category = tensor_category(name)
    match = _LAYER_RE.match(name)
    layer = int(match.group(1)) if match else 0

    # This model: 40 blocks, 30 of them attn_qkv (hybrid SSM layers) and 10
    # attn_v, so the attn-v-like counter and the attention count are 40;
    # each block contributes ffn_down_exps + ffn_down_shexp, so n_ffn_down = 80.
    n_attention_v = 40
    n_ffn_down = 80

    # OUTPUT/TOKEN_EMBD: the OUTPUT branch bumps to Q6_K whenever the default
    # type is not already Q8_0 (line ~455); token_embd keeps the default type.
    if category == "OUTPUT":
        return "Q6_K"
    if category == "TOKEN_EMBD":
        return PRESET_DEFAULT[preset]

    if _is_attn_v(category):
        if preset == "q4_k_s":
            return "Q5_K" if layer < 4 else "Q4_K"
        return "Q6_K" if use_more_bits(layer, n_attention_v) else PRESET_DEFAULT[preset]

    if category == "FFN_DOWN":
        if preset == "q4_k_s":
            return "Q5_K" if layer < n_ffn_down // 8 else "Q4_K"
        return "Q6_K" if use_more_bits(layer, n_ffn_down) else PRESET_DEFAULT[preset]

    return PRESET_DEFAULT[preset]


# ---------------------------------------------------------------------------
# Layout summaries
# ---------------------------------------------------------------------------
def _role(name: str) -> str:
    """Coarse role used for the register's per-role byte breakdown."""
    if name.endswith("_norm.weight") or name.startswith("output_norm"):
        return "norm"
    if "ffn_gate_exps" in name:
        return "expert_gate"
    if "ffn_up_exps" in name:
        return "expert_up"
    if "ffn_down_exps" in name:
        return "expert_down"
    if "_shexp" in name:
        return "shared_expert"
    if name.startswith("token_embd"):
        return "token_embd"
    if name.startswith("output"):
        return "output"
    if "attn_" in name:
        return "attention"
    if "ffn_" in name:
        return "ffn_other"
    if "ssm_" in name:
        return "ssm"
    return "norm"


#: Tensors that exist only in the MTP (next-token-prediction) preset.
def is_mtp_tensor(name: str) -> bool:
    match = _LAYER_RE.match(name)
    return bool(match and int(match.group(1)) >= 40)


def layout_summary(tensors: list[dict], *, exclude_mtp: bool = False) -> dict:
    """Parameters, weight bytes, bits-per-weight and the per-role breakdown."""
    rows = [t for t in tensors if not (exclude_mtp and is_mtp_tensor(t["name"]))]
    params = 0
    total = 0
    roles: dict[str, dict[str, float]] = {}
    types: dict[str, float] = {}
    for t in rows:
        b = tensor_bytes(t["type"], t["params"])
        params += t["params"]
        total += b
        role = roles.setdefault(_role(t["name"]), {"params": 0, "bytes": 0})
        role["params"] += t["params"]
        role["bytes"] += b
        types[t["type"]] = types.get(t["type"], 0) + b
    if not params:
        raise GgufError("empty tensor table")
    return {
        "tensors": len(rows),
        "params": params,
        "weight_bytes": total,
        "gib": round(total / 2**30, 4),
        "bits_per_weight": round(total * 8 / params, 4),
        "type_bytes": {k: int(v) for k, v in sorted(types.items())},
        "role_bytes": {
            k: {
                "params": v["params"],
                "bytes": v["bytes"],
                "gib": round(v["bytes"] / 2**30, 4),
                "share_pct": round(100.0 * v["bytes"] / total, 3),
            }
            for k, v in sorted(roles.items(), key=lambda kv: -kv[1]["bytes"])
        },
    }


def predict_preset(tensors: list[dict], preset: str) -> dict:
    """Byte size ``llama-quantize --type <preset>`` would produce.

    Tensors the preset does not touch keep their measured type, exactly as
    ``quantize`` copies them through.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
    params = 0
    total = 0
    types: dict[str, float] = {}
    for t in tensors:
        chosen = preset_type(t["name"], t["dims"], preset)
        ggml_type = chosen or t["type"]
        b = tensor_bytes(ggml_type, t["params"])
        params += t["params"]
        total += b
        types[ggml_type] = types.get(ggml_type, 0) + b
    return {
        "preset": preset,
        "params": params,
        "weight_bytes": total,
        "gib": round(total / 2**30, 4),
        "bits_per_weight": round(total * 8 / params, 4),
        "type_bytes": {k: int(v) for k, v in sorted(types.items())},
    }


# ---------------------------------------------------------------------------
# Throughput bounds
# ---------------------------------------------------------------------------
def active_bytes_per_decode_token(tensors: list[dict], layout: dict) -> dict:
    """Weight bytes one decode step streams, from the model's own topology.

    A batch-1 decode touches one token per expert it routes to (8 of 256 by
    `expert_used_count`), the shared expert, and every non-expert weight
    (attention/SSM, the output matrix, norms).
    """
    expert_used, expert_count = 8, 256
    expert_bytes = 0
    shared_bytes = 0
    other_bytes = 0
    for t in tensors:
        b = tensor_bytes(t["type"], t["params"])
        if "_exps" in t["name"]:
            expert_bytes += b
        elif "_shexp" in t["name"]:
            shared_bytes += b
        elif t["name"] == "token_embd.weight":
            continue  # one row, negligible
        else:
            other_bytes += b
    per_expert = expert_bytes / expert_count
    active = per_expert * expert_used + shared_bytes + other_bytes
    return {
        "routed_expert_bytes": int(per_expert * expert_used),
        "shared_expert_bytes": int(shared_bytes),
        "other_bytes": int(other_bytes),
        "active_bytes": int(active),
        "active_gib": round(active / 2**30, 4),
        "expert_used": expert_used,
        "expert_count": expert_count,
        "weight_bytes": layout["weight_bytes"],
    }


def prefill_weight_share(
    pair: tuple[float, float], weight_bytes: int, bandwidth_bytes_s: float
) -> dict:
    """Share of a prefill pass that is weight movement.

    ``pair`` is ``(mean prompt tokens, aggregate tokens per second)`` measured
    for one prompt-size bucket. A pass streams every weight (a 1024-token
    micro-batch touches all 256 experts), so the weight time per pass is
    ``weight_bytes / bandwidth`` and the pass takes ``UBATCH_TOKENS / rate``.
    """
    mean_tokens, rate = pair
    if rate <= 0:
        raise ValueError("measured rate must be positive")
    passes = math.ceil(mean_tokens / UBATCH_TOKENS)
    pass_seconds = UBATCH_TOKENS / rate
    request_seconds = 1000.0 * mean_tokens / rate / 1000.0
    weight_ms_per_pass = 1000.0 * weight_bytes / bandwidth_bytes_s
    weight_ms = weight_ms_per_pass * passes
    total_ms = request_seconds * 1000.0
    return {
        "mean_prompt_tokens": mean_tokens,
        "measured_tokens_per_second": rate,
        "micro_batches": passes,
        "pass_seconds": round(pass_seconds, 4),
        "request_prefill_seconds": round(total_ms / 1000.0, 3),
        "weight_ms_per_pass": round(weight_ms_per_pass, 3),
        "weight_share_pct": round(100.0 * weight_ms / total_ms, 3),
        "weight_stream_bytes_per_second": round(weight_bytes / pass_seconds, 1),
    }


def candidates(
    shipped_bytes: int,
    candidate_bytes: dict[str, int],
    prefill_buckets: list[dict],
    bandwidth_bytes_s: float,
    decode: dict,
    capacity: dict,
) -> dict:
    """Per-candidate cost table: size, VRAM delta and bounded throughput."""
    out: dict[str, dict] = {}
    for name, weight_bytes in candidate_bytes.items():
        ratio = weight_bytes / shipped_bytes
        gains = {}
        for bucket in prefill_buckets:
            share = bucket["weight_share_pct"] / 100.0
            ceiling_gain = 100.0 * (1.0 / ratio - 1.0)
            share_gain = 100.0 * share * (1.0 / ratio - 1.0)
            gains[bucket["bucket"]] = {
                "weight_share_pct": bucket["weight_share_pct"],
                "prefill_gain_ceiling_pct": round(ceiling_gain, 3),
                "prefill_gain_measured_share_pct": round(share_gain, 3),
                "prefill_tokens_per_second_measured": bucket["measured_tokens_per_second"],
                "prefill_tokens_per_second_ceiling": round(
                    bucket["measured_tokens_per_second"] / ratio, 2
                ),
                "prefill_tokens_per_second_measured_share": round(
                    bucket["measured_tokens_per_second"] * (1.0 + share_gain / 100.0), 2
                ),
            }
        decode_gain = 100.0 * (1.0 / ratio - 1.0)
        decode_seconds_after = decode["slot_seconds"] / (1.0 + decode_gain / 100.0)
        freed = decode["slot_seconds"] - decode_seconds_after
        capacity_gain = 100.0 * freed / capacity["local_busy_seconds"]
        out[name] = {
            "weight_bytes": weight_bytes,
            "gib": round(weight_bytes / 2**30, 4),
            "delta_vs_shipped_gib": round((weight_bytes - shipped_bytes) / 2**30, 4),
            "byte_ratio": round(ratio, 4),
            "prefill": gains,
            "decode": {
                "byte_ratio_bound_pct": round(decode_gain, 3),
                "measured_tokens_per_second": decode["tokens_per_second"],
                "tokens_per_second_ceiling": round(
                    decode["tokens_per_second"] * (1.0 + decode_gain / 100.0), 2
                ),
                "slot_seconds_freed": round(freed, 1),
                "slot_seconds_share_of_busy_pct": round(capacity["decode_slot_share_pct"], 3),
            },
            "capacity": {
                "free_slot_seconds_gain_pct": round(capacity_gain, 3),
                "local_token_share_before_pct": capacity["local_token_share_pct"],
                "local_token_share_after_pct": round(
                    capacity["local_token_share_pct"] * (1.0 + capacity_gain / 100.0), 4
                ),
                "local_stream_share_before_pct": capacity["local_stream_share_pct"],
                "local_stream_share_after_pct": round(
                    capacity["local_stream_share_pct"] * (1.0 + capacity_gain / 100.0), 4
                ),
            },
        }
    return out


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
def _bucket_rows(prefill_json: dict) -> list[dict]:
    """Reduce a `prefill_throughput_report.py --json` payload to five rows."""
    rows = []
    for name, block in prefill_json["buckets"].items():
        if not block["requests"]:
            continue
        rows.append(
            {
                "bucket": name,
                "requests": block["requests"],
                "prompt_tokens": block["prompt_tokens"],
                "prompt_ms": block["prompt_ms"],
                "mean_prompt_tokens": round(
                    block["prompt_tokens"] / block["requests"], 3
                ),
                "aggregate_tokens_per_second": block["aggregate_tokens_per_second"],
            }
        )
    return rows


def build_register(
    *,
    shipped_tensors: list[dict],
    shipped_source: str,
    q4_k_s_tensors: list[dict],
    q4_k_s_source: str,
    prefill_json: dict,
    decode_tokens_per_second: float,
    capacity: dict,
    decode_provenance: str,
    capacity_provenance: str,
    generated_at: str,
) -> dict:
    """Assemble the committed register from measured inputs."""
    shipped = layout_summary(shipped_tensors)
    # The host's Q4_K_S file is the MTP preset, whose base weights are the same
    # 34.661 B parameters as the shipped model; the extra nextn block is
    # excluded so the two layouts are comparable.
    q4s_measured = layout_summary(q4_k_s_tensors, exclude_mtp=True)
    q4s_full = layout_summary(q4_k_s_tensors)
    plain = {p: predict_preset(shipped_tensors, p) for p in PRESETS}

    # UD-Q4_K_M estimate: the measured UD-Q4_K_S weights scaled by the plain
    # preset's own Q4_K_M/Q4_K_S ratio (unsloth's dynamic-quant delta between
    # the two Q4 presets is assumed to track llama.cpp's plain delta).
    ratio_m_over_s = plain["q4_k_m"]["weight_bytes"] / plain["q4_k_s"]["weight_bytes"]
    ud_q4_k_m_bytes = int(round(q4s_measured["weight_bytes"] * ratio_m_over_s))

    shipped_bytes = shipped["weight_bytes"]
    decode_topology = active_bytes_per_decode_token(shipped_tensors, shipped)
    bandwidth = decode_topology["active_bytes"] * decode_tokens_per_second
    buckets = []
    for row in _bucket_rows(prefill_json):
        info = prefill_weight_share(
            (row["mean_prompt_tokens"], row["aggregate_tokens_per_second"]),
            shipped_bytes,
            bandwidth,
        )
        buckets.append({**row, **info})

    decode_slot_seconds = capacity["local_completion_tokens"] / decode_tokens_per_second
    capacity = {
        **capacity,
        "decode_slot_seconds": round(decode_slot_seconds, 1),
        "decode_slot_share_pct": round(
            100.0 * decode_slot_seconds / capacity["local_busy_seconds"], 3
        ),
    }
    decode = {
        "tokens_per_second": decode_tokens_per_second,
        "slot_seconds": decode_slot_seconds,
        "provenance": decode_provenance,
    }

    cand_bytes = {
        "ud_q4_k_m_estimated": ud_q4_k_m_bytes,
        "ud_q4_k_s_measured": q4s_measured["weight_bytes"],
    }
    cand = candidates(
        shipped_bytes, cand_bytes, buckets, bandwidth, decode, capacity
    )
    cand["ud_q4_k_m_estimated"]["provenance"] = "derived-estimate"
    cand["ud_q4_k_m_estimated"]["derivation"] = (
        "measured UD-Q4_K_S base weights x (plain Q4_K_M / plain Q4_K_S)"
    )
    cand["ud_q4_k_s_measured"]["provenance"] = "measured"

    register = {
        "schema_version": SCHEMA_VERSION,
        "work_item": "LP-0MWCEHYOO002TRGD",
        "parent_work_item": "LP-0MVXY8ZBZ0003042",
        "purpose": (
            "Cost the model-weight quantisation lever (Q5_K_M -> Q4_K_M / "
            "Q4_K_S) for the spike's AC1 comparison table: weight/VRAM effect, "
            "bounded prefill and decode throughput effect, and the resulting "
            "local-share arithmetic."
        ),
        "read_only": True,
        "generated_at": generated_at,
        "provenance_classes": {
            "measured": (
                "Read from a GGUF tensor table or a measured production "
                "llama-server/proxy log window. Not re-derived here."
            ),
            "derived-estimate": (
                "Arithmetic over measured figures registered in this file. "
                "Labelled an estimate everywhere it is quoted."
            ),
            "config-derived": (
                "Taken from the shipped configuration (models.ini, "
                "proxy/config.yaml) with its defining key recorded."
            ),
        },
        "engine": {
            "arch": "qwen35moe",
            "block_count": 40,
            "expert_count": 256,
            "expert_used_count": 8,
            "ubatch_tokens": UBATCH_TOKENS,
            "ubatch_source": "models.ini:ubatch-size = 1024",
            "shipped_preset": "UD-Q5_K_M",
            "source": "models.ini [Qwen3] hf-repo suffix and the GGUF tensor table",
        },
        "layouts": {
            "shipped_ud_q5_k_m": {
                "label": "shipped UD-Q5_K_M (unsloth dynamic quant)",
                "provenance": "measured",
                "source": shipped_source,
                **shipped,
            },
            "measured_ud_q4_k_s_base": {
                "label": "UD-Q4_K_S, base-model weights (MTP nextn block excluded)",
                "provenance": "measured",
                "source": q4_k_s_source,
                **q4s_measured,
            },
            "measured_ud_q4_k_s_file": {
                "label": "UD-Q4_K_S as shipped in the MTP preset GGUF (whole file)",
                "provenance": "measured",
                "source": q4_k_s_source,
                **q4s_full,
            },
            "predicted_plain": {
                "label": "llama-quantize plain presets applied to the shipped tensor table",
                "provenance": "derived-estimate",
                "rules": (
                    "llama.cpp src/llama-quant.cpp: tensor_allows_quantization, "
                    "tensor_get_category, use_more_bits, llama_tensor_get_type_impl"
                ),
                **plain,
            },
        },
        "inputs": {
            "prefill": {
                "provenance": "measured",
                "source": "scripts/prefill_throughput_report.py",
                "window": prefill_json.get("window", {}),
                "totals": {
                    "requests": prefill_json.get("requests"),
                    "prompt_tokens": prefill_json.get("prompt_tokens"),
                    "aggregate_tokens_per_second": prefill_json.get(
                        "aggregate_tokens_per_second"
                    ),
                    "per_request": prefill_json.get("per_request"),
                },
                "buckets": buckets,
            },
            "decode": {
                "provenance": "measured",
                "tokens_per_second": decode_tokens_per_second,
                "provenance_note": decode_provenance,
                "active_bytes_per_token": decode_topology,
                "implied_bytes_per_second": round(bandwidth, 1),
            },
            "capacity": {
                "provenance": "measured",
                "provenance_note": capacity_provenance,
                "local_busy_seconds": capacity["local_busy_seconds"],
                "local_completion_tokens": capacity["local_completion_tokens"],
                "local_prompt_tokens": capacity["local_prompt_tokens"],
                "offered_prompt_tokens": capacity["offered_prompt_tokens"],
                "local_token_share_pct": capacity["local_token_share_pct"],
                "local_stream_share_pct": capacity["local_stream_share_pct"],
                "decode_slot_seconds": capacity["decode_slot_seconds"],
                "decode_slot_share_pct": capacity["decode_slot_share_pct"],
            },
            "per_slot_resources": {
                "provenance": "config-derived",
                "session_slot_pool_size": 2,
                "local_model_ctx_size": 262144,
                "per_slot_ctx": 131072,
                "cache_ram_mib": 40960,
                "ctx_checkpoints": 32,
                "checkpoint_mib": 62.8,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "note": (
                    "Only the shared weights term moves with the weight quant; "
                    "cache-ram, the per-prompt checkpoint cost (~2.0 GiB) and "
                    "the q8_0 KV cache are weight-independent."
                ),
                "sources": ["models.ini", "proxy/config.yaml"],
            },
        },
        "candidates": cand,
    }
    register["checks"] = checks(register)
    return register


def checks(register: dict) -> list[dict]:
    """Self-verification of every derived figure in a register."""
    results: list[dict] = []

    def add(claim: str, published, reproduced, ok: bool) -> None:
        results.append(
            {
                "claim": claim,
                "published": published,
                "reproduced": reproduced,
                "ok": bool(ok),
                "verdict": "reproduces" if ok else "drift",
            }
        )

    layouts = register["layouts"]
    shipped = layouts["shipped_ud_q5_k_m"]
    shipped_bytes = shipped["weight_bytes"]

    # 1. The two measured GGUF layouts must carry the same parameter count.
    add(
        "measured Q4_K_S base weights have the shipped parameter count",
        shipped["params"],
        layouts["measured_ud_q4_k_s_base"]["params"],
        shipped["params"] == layouts["measured_ud_q4_k_s_base"]["params"],
    )

    # 2. bits-per-weight reproduces from bytes and params.
    bpw = round(shipped_bytes * 8 / shipped["params"], 4)
    add("shipped bits-per-weight", shipped["bits_per_weight"], bpw,
        abs(bpw - shipped["bits_per_weight"]) < 1e-4)

    # 3. Candidate bytes reproduce from the layouts they are derived from.
    plain = layouts["predicted_plain"]
    ratio = plain["q4_k_m"]["weight_bytes"] / plain["q4_k_s"]["weight_bytes"]
    ud_q4_k_m = int(
        round(layouts["measured_ud_q4_k_s_base"]["weight_bytes"] * ratio)
    )
    add(
        "UD-Q4_K_M estimate from UD-Q4_K_S x plain Q4_K_M/Q4_K_S",
        register["candidates"]["ud_q4_k_m_estimated"]["weight_bytes"],
        ud_q4_k_m,
        ud_q4_k_m == register["candidates"]["ud_q4_k_m_estimated"]["weight_bytes"],
    )
    add(
        "UD-Q4_K_S candidate is the measured base layout",
        register["candidates"]["ud_q4_k_s_measured"]["weight_bytes"],
        layouts["measured_ud_q4_k_s_base"]["weight_bytes"],
        register["candidates"]["ud_q4_k_s_measured"]["weight_bytes"]
        == layouts["measured_ud_q4_k_s_base"]["weight_bytes"],
    )

    # 4. Byte ratios and VRAM deltas reproduce.
    for name, cand in register["candidates"].items():
        ratio = cand["weight_bytes"] / shipped_bytes
        add(
            f"{name}: byte ratio",
            cand["byte_ratio"],
            round(ratio, 4),
            abs(ratio - cand["byte_ratio"]) < 5e-5,
        )
        delta = (cand["weight_bytes"] - shipped_bytes) / 2**30
        add(
            f"{name}: VRAM delta (GiB)",
            cand["delta_vs_shipped_gib"],
            round(delta, 4),
            abs(delta - cand["delta_vs_shipped_gib"]) < 5e-4,
        )

    # 5. The decode anchor's implied bandwidth reproduces from the topology.
    decode = register["inputs"]["decode"]
    topology = decode["active_bytes_per_token"]
    implied = topology["active_bytes"] * decode["tokens_per_second"]
    add(
        "decode anchor implies the registered bytes/second",
        decode["implied_bytes_per_second"],
        round(implied, 1),
        abs(implied - decode["implied_bytes_per_second"]) < 1.0,
    )

    # 6. Every prefill bucket reproduces its weight share and the 5 GB/s-scale
    #    claim that refutes a weight-bandwidth bottleneck.
    bandwidth = decode["implied_bytes_per_second"]
    for bucket in register["inputs"]["prefill"]["buckets"]:
        info = prefill_weight_share(
            (bucket["mean_prompt_tokens"], bucket["aggregate_tokens_per_second"]),
            shipped_bytes,
            bandwidth,
        )
        add(
            f"prefill bucket {bucket['bucket']}: weight share (%)",
            bucket["weight_share_pct"],
            info["weight_share_pct"],
            abs(info["weight_share_pct"] - bucket["weight_share_pct"]) < 1e-3,
        )
        add(
            f"prefill bucket {bucket['bucket']}: weight-stream bytes/second",
            bucket["weight_stream_bytes_per_second"],
            info["weight_stream_bytes_per_second"],
            abs(
                info["weight_stream_bytes_per_second"]
                - bucket["weight_stream_bytes_per_second"]
            )
            < 1.0,
        )

    # 7. Candidate gains reproduce from the buckets and ratios.
    for name, cand in register["candidates"].items():
        ratio = cand["weight_bytes"] / shipped_bytes
        for bucket_name, gain in cand["prefill"].items():
            ceiling = 100.0 * (1.0 / ratio - 1.0)
            add(
                f"{name}: prefill ceiling gain (%) at {bucket_name}",
                gain["prefill_gain_ceiling_pct"],
                round(ceiling, 3),
                abs(ceiling - gain["prefill_gain_ceiling_pct"]) < 1e-3,
            )
            share_gain = ceiling * gain["weight_share_pct"] / 100.0
            add(
                f"{name}: prefill measured-share gain (%) at {bucket_name}",
                gain["prefill_gain_measured_share_pct"],
                round(share_gain, 3),
                abs(share_gain - gain["prefill_gain_measured_share_pct"]) < 1e-3,
            )
        decode_gain = 100.0 * (1.0 / ratio - 1.0)
        add(
            f"{name}: decode byte-ratio gain (%)",
            cand["decode"]["byte_ratio_bound_pct"],
            round(decode_gain, 3),
            abs(decode_gain - cand["decode"]["byte_ratio_bound_pct"]) < 1e-3,
        )

    # 8. Capacity arithmetic reproduces.
    capacity = register["inputs"]["capacity"]
    seconds = capacity["local_completion_tokens"] / register["inputs"]["decode"][
        "tokens_per_second"
    ]
    add(
        "decode slot-seconds",
        capacity["decode_slot_seconds"],
        round(seconds, 1),
        abs(seconds - capacity["decode_slot_seconds"]) < 0.5,
    )
    share = 100.0 * seconds / capacity["local_busy_seconds"]
    add(
        "decode slot-share of local busy (%)",
        capacity["decode_slot_share_pct"],
        round(share, 3),
        abs(share - capacity["decode_slot_share_pct"]) < 1e-3,
    )
    return results


def all_ok(results: list[dict]) -> bool:
    return all(r["ok"] for r in results)


def recheck_register(register: dict) -> list[dict]:
    """Re-verify a committed register from its own embedded inputs."""
    return checks(register)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_gguf_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, _, path = text.partition("=")
    return label, Path(path).expanduser()


def _repo_constants() -> dict:
    """Constants of record for the shipped geometry (config-derived)."""
    return {
        "decode_tokens_per_second": 30.7,
        "decode_provenance": (
            "measured production decode rate, p50, docs/dev/single-slot-impact.md "
            "W1 (2026-10-23/24); 37.7 tok/s in W2 is quoted in the report as the "
            "sensitivity"
        ),
        "capacity": {
            "local_busy_seconds": 1223.0,
            "local_completion_tokens": 5027,
            "local_prompt_tokens": 877908,
            "offered_prompt_tokens": 125577780,
            "local_token_share_pct": 0.6991,
            "local_stream_share_pct": 2.5727,
        },
        "capacity_provenance": (
            "measured, docs/dev/local-share-baseline-2026-11-01.json "
            "(2026-11-01 13:12:48 -> 14:00:03)"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gguf",
        action="append",
        default=[],
        type=_parse_gguf_arg,
        metavar="LABEL=PATH",
        help="GGUF file to read (repeatable); `shipped` is the shipped preset",
    )
    parser.add_argument("--prefill-json", type=Path)
    parser.add_argument("--register", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="re-verify a committed register"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--layout", action="store_true", help="print each GGUF's tensor-table summary"
    )
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)

    if args.check:
        if not args.register:
            parser.error("--check needs --register")
        register = json.loads(args.register.read_text())
        results = recheck_register(register)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for row in results:
                mark = "ok  " if row["ok"] else "DRIFT"
                print(f"[{mark}] {row['claim']}: {row['published']} vs {row['reproduced']}")
        if not all_ok(results):
            print("register check FAILED", file=sys.stderr)
            return 1
        print(f"register check: {len(results)} claims reproduce", file=sys.stderr)
        return 0

    if args.layout:
        if not args.gguf:
            parser.error("--layout needs at least one --gguf")
        out = {}
        for label, path in args.gguf:
            tensors = read_gguf_tensor_table(path)
            out[label] = {
                "path": str(path),
                "whole_file": layout_summary(tensors),
                "base_only": layout_summary(tensors, exclude_mtp=True),
            }
        print(json.dumps(out, indent=2))
        return 0

    labels = dict(args.gguf)
    if "shipped" not in labels:
        parser.error("--gguf shipped=PATH is required to build the register")
    if not args.prefill_json:
        parser.error("--prefill-json PATH is required to build the register")

    generated_at = args.generated_at
    if generated_at is None:
        from datetime import UTC, datetime

        generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    many = [label for label in labels if label.startswith("q4_k_s")]
    if not many:
        parser.error("--gguf q4_k_s=PATH (the UD-Q4_K_S file) is required")
    q4_k_s_label = many[0]
    constants = _repo_constants()

    register = build_register(
        shipped_tensors=read_gguf_tensor_table(labels["shipped"]),
        shipped_source=Path(labels["shipped"]).name,
        q4_k_s_tensors=read_gguf_tensor_table(labels[q4_k_s_label]),
        q4_k_s_source=Path(labels[q4_k_s_label]).name,
        prefill_json=json.loads(args.prefill_json.read_text()),
        decode_tokens_per_second=constants["decode_tokens_per_second"],
        decode_provenance=constants["decode_provenance"],
        capacity=constants["capacity"],
        capacity_provenance=constants["capacity_provenance"],
        generated_at=generated_at,
    )
    if args.json:
        print(json.dumps(register, indent=2))
        return 0
    failed = [c for c in register["checks"] if not c["ok"]]
    for c in register["checks"]:
        print(f"[{'ok  ' if c['ok'] else 'DRIFT'}] {c['claim']}: {c['published']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
