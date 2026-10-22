#!/usr/bin/env python3
"""A/B prompt-eval throughput benchmark for the llama-server ubatch change.

Measures llama-server's OWN reported prompt-eval throughput
(``timings.prompt_per_second`` / ``timings.prompt_ms``) for fixed-size prompts.
This is the quantity the ubatch-size change targets: prefill (prompt eval) is
the dominant term in local first-byte latency, and llama.cpp scales prompt
processing with the physical micro-batch (``--ubatch-size``) up to the GPU's
compute/memory limit.

Usage:
    ubatch_ab_benchmark.py --base-url http://127.0.0.1:8080 --model Qwen3 \
        --label baseline-1024 --out /tmp/ubatch-baseline.json

Records one row per repetition with prompt tokens, prompt_ms and
prompt_per_second, then reports p50/p90 per prompt size and an overall p90
over the 20k+ token batches (matching the epic's baseline definition).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Deterministic filler so every repetition sends the same token count without a
# tokenizer dependency. One line is ~12 tokens; the prompt is repeated prose.
_LINE = (
    "The proxy routes a chat completion to a local llama-server instance when "
    "the prompt fits the local slot and the projected prefill fits the wait "
    "budget, otherwise it falls through to the next remote provider. "
)


def build_prompt(approx_tokens: int, salt: int = 0) -> str:
    """Build a prompt of roughly *approx_tokens* tokens (~4 chars/token).

    *salt* makes each repetition's text unique so llama.cpp's slot prompt
    cache cannot satisfy it (a cached prefix would report a near-zero prompt
    eval time and make the A/B meaningless). ``cache_prompt: false`` is also
    sent, but unique text is the load-bearing guard.
    """
    target_chars = max(200, approx_tokens * 4)
    repeats = max(1, target_chars // len(_LINE))
    body = "".join(f"[{salt}:{index}] {_LINE}" for index in range(repeats))
    return "Summarise the following text in one sentence.\n\n" + body


def measure(base_url: str, model: str, prompt: str, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
        # Defeat the slot prompt cache so prompt_ms measures a real prefill.
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    wall = time.monotonic() - started
    timings = body.get("timings") or {}
    prompt_tokens = int(timings.get("prompt_n") or 0)
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    per_second = timings.get("prompt_per_second")
    if per_second is None and prompt_ms > 0:
        per_second = prompt_tokens / (prompt_ms / 1000.0)
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_ms": round(prompt_ms, 1),
        "prompt_per_second": round(float(per_second or 0.0), 1),
        "wall_seconds": round(wall, 2),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sizes", default="8192,16384,24576",
        help="Comma-separated prompt sizes in tokens",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    rows: list[dict] = []
    salt = int(time.time())
    for size in sizes:
        for rep in range(args.reps):
            prompt = build_prompt(size, salt=salt * 100 + rep)
            try:
                row = measure(args.base_url, args.model, prompt, args.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"ERROR size={size} rep={rep}: {exc}", file=sys.stderr)
                row = {
                    "prompt_tokens": 0, "prompt_ms": 0.0,
                    "prompt_per_second": 0.0, "wall_seconds": 0.0,
                    "error": str(exc),
                }
            row.update({"requested_tokens": size, "rep": rep})
            rows.append(row)
            print(
                f"  size={size:>6} rep={rep} tokens={row['prompt_tokens']:>6} "
                f"prompt_ms={row['prompt_ms']:>9} "
                f"t/s={row['prompt_per_second']:>7}",
                flush=True,
            )

    ok = [r for r in rows if r["prompt_per_second"] > 0]
    per_size = {}
    for size in sizes:
        values = [
            r["prompt_per_second"] for r in ok if r["requested_tokens"] == size
        ]
        per_size[str(size)] = {
            "n": len(values),
            "p50": round(percentile(values, 50), 1),
            "p90": round(percentile(values, 90), 1),
            "max": round(max(values), 1) if values else 0.0,
        }
    # The epic's baseline is stated over 20k+ token batches.
    big = [r["prompt_per_second"] for r in ok if r["prompt_tokens"] >= 20_000]
    summary = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sizes": sizes,
        "reps": args.reps,
        "rows": rows,
        "per_size": per_size,
        "overall_20k_plus": {
            "n": len(big),
            "p50": round(percentile(big, 50), 1),
            "p90": round(percentile(big, 90), 1),
            "min": round(min(big), 1) if big else 0.0,
        },
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in ("per_size", "overall_20k_plus")}, indent=2))
    return 0 if big else 1


if __name__ == "__main__":
    raise SystemExit(main())
