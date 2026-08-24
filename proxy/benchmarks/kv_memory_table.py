#!/usr/bin/env python3
"""F4: Per-config per-slot KV-cache headroom table (LP-0MSC95W3T000CCYC).

Produces the audit-required headroom table for every configuration of the
ctx-size evaluation: per-slot KV memory, total KV memory at peak slot
usage, and headroom versus measured available memory.

Methodology (measured, F4 comment LP-C0MSEGJO5A006TV05):
- KV cache memory is TOTAL-CTX bound, not slot-count bound: with q8_0 KV
  quantisation on the 10-layer hybrid/recurrent Qwen3, 131072 total ctx
  cells consume 1362.7 MiB and 262144 total ctx cells consume ~2720-2725
  MiB (measured from llama-server logs during the F2/F3 benchmark runs).
- The per-token KV cost is therefore 1362.7 / 131072 MiB per total ctx
  cell; per-slot KV = per-token cost x per-slot ctx (per_slot = total_ctx
  // slots). 454 MiB/slot at 6x43.7K and 1360 MiB/slot at 2x131K match
  the F4 findings.
- Model: Qwen3 35B Q5_K_M = 24.7 GiB (measured weight size).
- Available memory: ~87 GiB (124 GiB total / ~87 GiB available measured
  from /proc/meminfo across F2/F3 run snapshots; the "~71GB available"
  intake claim is CONFIRMED upward).

Usage:
    python3 proxy/benchmarks/kv_memory_table.py [--json]
"""
import argparse
import json
import sys

KV_MIB_PER_TOKEN = 1362.7 / 131072  # MiB of q8_0 KV per total-ctx cell (measured)
MODEL_GIB = 24.7                     # Qwen3 35B Q5_K_M weight size (GiB)
AVAILABLE_GIB = 87.0                 # measured available memory (GiB)
# (label, total_ctx, slots) — covers both day/night schedules; per-slot ctx
# depends only on the slot count at a given total ctx.
CONFIGS = [
    ("8x32.8K", 262144, 8),
    ("6x43.7K", 262144, 6),
    ("4x65.5K", 262144, 4),
    ("3x87.4K", 262144, 3),
    ("2x131K", 262144, 2),
    ("3x43.7K live baseline", 131072, 3),
]

MIB_PER_GIB = 1024.0


def build_table():
    """Return per-config rows: per-slot KV, total KV, model+KV, headroom."""
    rows = []
    for label, total_ctx, slots in CONFIGS:
        per_slot_ctx = total_ctx // slots
        per_slot_kv_mib = KV_MIB_PER_TOKEN * per_slot_ctx
        total_kv_mib = KV_MIB_PER_TOKEN * total_ctx
        model_plus_kv_gib = MODEL_GIB + total_kv_mib / MIB_PER_GIB
        headroom_gib = AVAILABLE_GIB - model_plus_kv_gib
        rows.append(
            {
                "config": label,
                "total_ctx": total_ctx,
                "slots": slots,
                "per_slot_ctx": per_slot_ctx,
                "per_slot_kv_mib": round(per_slot_kv_mib, 1),
                "total_kv_mib": round(total_kv_mib, 1),
                "model_plus_kv_gib": round(model_plus_kv_gib, 2),
                "headroom_gib": round(headroom_gib, 2),
            }
        )
    return rows


def _render_markdown(rows):
    header = (
        "| Config | Slots | per-slot ctx | per-slot KV (MiB) | total KV (MiB) "
        "| Model+KV (GiB) | headroom (GiB) |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['slots']} | {r['per_slot_ctx']} "
            f"| {r['per_slot_kv_mib']} | {r['total_kv_mib']} "
            f"| {r['model_plus_kv_gib']} | {r['headroom_gib']} |"
        )
    return "\n".join(lines)


def _render_json(rows):
    return json.dumps({"configs": rows}, indent=2)


def main():
    parser = argparse.ArgumentParser(description="F4 KV headroom table per config")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args()
    rows = build_table()
    if args.json:
        print(_render_json(rows))
    else:
        print(_render_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
