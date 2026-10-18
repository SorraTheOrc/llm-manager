# Slot sweep 1/2/3 — results note (LP-0MVAZPQUM006EAC7)

Run: 2026-10-17 14:21 UTC (slot 1 & 2; slot 3 aborted — see validity note on the work item).

## Environment

- Live proxy at /home/rgardler/projects/llm; harness operated on it via `LLAMA_PROXY_BENCHMARK_ROOT`.
- Mode: **fast** (held with the mode-switch lock, LP-0MVCBMLP7009MN81).
- Base config slot pre-run: **2** (`session_slot_pool_size: 2`, sha256 0c9b565e…).
- Model: plan → local-qwen3/Qwen3, ctx 262144, per-slot physical N=1 258048 / N=2 126976 / N=3 83285.
- Concurrency 3 (≥ observed 2–3 sessions + queue depth); prompt target 2000 tokens; slot cache NOT cleared (`--clean-cache` off).

## Command (reproducible)

```
LLAMA_PROXY_BENCHMARK_ROOT=/home/rgardler/projects/llm \
  python3 -m proxy.benchmarks.slot_benchmark \
  --sweep --concurrency 3 --prompt-tokens 2000 \
  --output-dir proxy/benchmarks/results/slot_sweep_1_2_3_20261017_1421
```

## Results (12/12 completed each; errors 0)

| N | avg dur | avg TTFT* | avg TPS* | queue-timeout | queue rate | local share |
|---|--------:|----------:|--------:|--------------:|-----------:|------------:|
| 1 | 135s    | 0.07s     | 92.3     | 12/17         | 70.6%        | 0.0% |
| 2 | 75s     | 0.05s     | 153.3    | 1/4           | 25.0%        | 0.0% |
| 3 | —       | —         | —        | —             | —           | —   (aborted) |

*fresh-run TTFT/TPS reflect requests resolved REMOTE (local lease starved by live
audit/herdr sessions); they are NOT local decode numbers. See validity note.

## Config restoration evidence

- Pre-run slot: 2 (sha256 0c9b565e82f9778c015504848a1b1b53781a60da9fc2117559d54bcec9e0e44b).
- Post-run restore (finally block): `Restored config: … config.yaml -> session_slot_pool_size: 2`; verified `git status` clean at main checkout after discard of run-2 N=3 abort + explicit `git checkout HEAD -- proxy/config.yaml`.
- Proxy left healthy, fast mode, mode-switch lock engaged (released/restored separately by the operator flow).

## Files

- `slot_1.json`, `slot_2.json` — per-run raw results with reproducibility + viability metadata.
- `slot_benchmark_report.json`, `slot_benchmark_summary.txt` — combined report.
- `INVALID_EARLIER_slot_sweep_1_2_3_20261017_1313/` — run 1 (backend-unreachable + N=3 slot mismatch), preserved but NOT to be cited.
