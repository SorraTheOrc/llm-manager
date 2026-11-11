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

``--concurrency N`` (LP-0MWCEHDRV000B7HD) sends N unique prompts at once and
records the round's **aggregate** throughput (tokens processed over the round's
wall window) beside the per-request rate. Concurrency 1 is the shipped
serialised gate (``local_max_concurrent_prefills: 1``); concurrency 2 matches
the two-slot pool, so it is the highest useful parallel-prefill depth on this
host.

Records one row per repetition with prompt tokens, prompt_ms and
prompt_per_second, then reports p50/p90 per prompt size and an overall p90
over the 20k+ token batches (matching the epic's baseline definition).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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


def aggregate_round(rows: list[dict]) -> dict:
    """Aggregate prefill throughput of one round (LP-0MWCEHDRV000B7HD).

    Aggregate throughput is the tokens the round actually prefilled divided by
    the wall window in which it prefilled them — **not** the sum of the
    per-request rates, which double-counts overlapped time. Two requests each
    sustaining 200 tok/s inside the same 100 s window process 40,000 tokens, so
    the aggregate is 400 tok/s; if the second request merely waits its turn the
    window is 200 s and the aggregate stays at 200 tok/s. That difference is the
    whole parallel-prefill question.

    Failed requests contribute no tokens but still bound the window: the time
    they occupied is time the GPU did not spend on useful prefill.
    """
    if not rows:
        return {
            "prompt_tokens": 0,
            "window_seconds": 0.0,
            "tokens_per_second": 0.0,
        }
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    started = min(float(r["started_offset_s"]) for r in rows)
    ended = max(float(r["ended_offset_s"]) for r in rows)
    window = max(ended - started, 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "window_seconds": round(window, 3),
        "tokens_per_second": (round(prompt_tokens / window, 1) if window > 0 else 0.0),
    }


def run_round(
    prompts: list[str],
    measure_fn,
    concurrency: int,
) -> list[dict]:
    """Run one round of *prompts*, *concurrency* of them in flight at once.

    ``measure_fn(prompt)`` performs one measurement and returns its row. Every
    row gains ``request_index`` (its position in *prompts*, so callers can map
    results back without logging 40k-token prompt bodies) and the window offsets
    ``started_offset_s`` / ``ended_offset_s`` measured from the round's start,
    which :func:`aggregate_round` needs.

    With ``concurrency > 1`` the requests are released together through a
    barrier, so the measured window really is overlapped GPU time rather than
    the serialised sum it would be if the threads started lazily.
    """
    concurrency = max(1, int(concurrency))
    round_start = time.monotonic()
    barrier = threading.Barrier(len(prompts)) if concurrency > 1 else None

    def run_one(index: int) -> dict:
        prompt = prompts[index]
        if barrier is not None:
            barrier.wait()
        started = time.monotonic() - round_start
        try:
            row = measure_fn(prompt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            row = {
                "prompt_tokens": 0,
                "prompt_ms": 0.0,
                "prompt_per_second": 0.0,
                "wall_seconds": 0.0,
                "error": str(exc),
            }
        row["request_index"] = index
        row["started_offset_s"] = round(started, 3)
        row["ended_offset_s"] = round(time.monotonic() - round_start, 3)
        return row

    if concurrency == 1:
        return [run_one(index) for index in range(len(prompts))]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(run_one, range(len(prompts))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sizes",
        default="8192,16384,24576",
        help="Comma-separated prompt sizes in tokens",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Requests in flight at once per round (default 1 = serialised, "
            "the shipped local_max_concurrent_prefills: 1 shape). >1 tests "
            "whether overlapping prefills raise AGGREGATE throughput "
            "(LP-0MWCEHDRV000B7HD); it bypasses the proxy's prefill gate and "
            "talks to the engine directly, as the ubatch A/B always has."
        ),
    )
    args = parser.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    concurrency = max(1, args.concurrency)
    rows: list[dict] = []
    rounds: list[dict] = []
    salt = int(time.time())
    for size in sizes:
        for rep in range(args.reps):
            prompts = [build_prompt(size, salt=salt * 100 + rep * 1000 + index) for index in range(concurrency)]
            round_rows = run_round(
                prompts,
                lambda prompt: measure(args.base_url, args.model, prompt, args.timeout),
                concurrency,
            )
            for row in round_rows:
                row.update({"requested_tokens": size, "rep": rep})
                rows.append(row)
            aggregate = aggregate_round(round_rows)
            rounds.append(
                {
                    "requested_tokens": size,
                    "rep": rep,
                    "concurrency": concurrency,
                    **aggregate,
                }
            )
            per_request = " ".join(f"t/s={row['prompt_per_second']:>7}" for row in round_rows)
            tokens = "+".join(str(row["prompt_tokens"]) for row in round_rows)
            print(
                f"  size={size:>6} rep={rep} c={concurrency} tokens={tokens:>13} "
                f"window_s={aggregate['window_seconds']:>8} "
                f"aggregate_t/s={aggregate['tokens_per_second']:>7} "
                f"[{per_request}]",
                flush=True,
            )

    ok = [r for r in rows if r["prompt_per_second"] > 0]
    per_size = {}
    for size in sizes:
        values = [r["prompt_per_second"] for r in ok if r["requested_tokens"] == size]
        aggregate_values = [
            r["tokens_per_second"] for r in rounds if r["requested_tokens"] == size and r["tokens_per_second"] > 0
        ]
        per_size[str(size)] = {
            "n": len(values),
            "p50": round(percentile(values, 50), 1),
            "p90": round(percentile(values, 90), 1),
            "max": round(max(values), 1) if values else 0.0,
            "aggregate": {
                "n_rounds": len(aggregate_values),
                "p50": round(percentile(aggregate_values, 50), 1),
                "p90": round(percentile(aggregate_values, 90), 1),
                "max": round(max(aggregate_values), 1) if aggregate_values else 0.0,
            },
        }
    # The epic's baseline is stated over 20k+ token batches.
    big = [r["prompt_per_second"] for r in ok if r["prompt_tokens"] >= 20_000]
    big_aggregate = [
        r["tokens_per_second"] for r in rounds if r["requested_tokens"] >= 20_000 and r["tokens_per_second"] > 0
    ]
    summary = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": concurrency,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sizes": sizes,
        "reps": args.reps,
        "rows": rows,
        "rounds": rounds,
        "per_size": per_size,
        "overall_20k_plus": {
            "n": len(big),
            "p50": round(percentile(big, 50), 1),
            "p90": round(percentile(big, 90), 1),
            "min": round(min(big), 1) if big else 0.0,
        },
        "overall_20k_plus_aggregate": {
            "n_rounds": len(big_aggregate),
            "p50": round(percentile(big_aggregate, 50), 1),
            "p90": round(percentile(big_aggregate, 90), 1),
            "min": round(min(big_aggregate), 1) if big_aggregate else 0.0,
        },
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "concurrency",
                    "per_size",
                    "overall_20k_plus",
                    "overall_20k_plus_aggregate",
                )
            },
            indent=2,
        )
    )
    return 0 if big_aggregate else 1


if __name__ == "__main__":
    raise SystemExit(main())
