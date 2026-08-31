# Benchmark Suite: KV Quantization and Configuration Changes

A repeatable benchmark suite for evaluating changes that affect KV cache
storage, quantization, batching, or concurrency. Produces objective metrics
for rollout decisions via A/B comparison.

## Quick Start

```bash
# 1. Record a baseline run
python -m proxy.benchmarks.run_benchmark --baseline

# 2. Apply your config change (e.g., update models.ini quant)

# 3. Record a candidate run
python -m proxy.benchmarks.run_benchmark --candidate --config models.ini

# 4. Compare results
python -m proxy.benchmarks.compare_results baseline_<timestamp>.json candidate_<timestamp>.json
```

## Files

| File | Purpose |
|------|---------|
| `run_benchmark.py` | Main benchmark runner — executes requests against the proxy and records metrics |
| `compare_results.py` | Delta computation and gating policy checker |
| `prometheus_snapshot.sh` | Helper that polls `/admin/metrics` and GPU/system memory endpoints during runs |
| `README.md` | This file — gating policy and usage documentation |

## Requirements

- Python 3.10+ with `httpx`:
  ```bash
  pip install httpx
  ```
- A running llama-proxy instance (default: `http://localhost:8000`)
- llama-server serving at the proxy's backend port (default: `8080`)
- (Optional) `rocm-smi` for GPU VRAM metrics
- (Optional) `curl` for admin metrics polling

## Usage

### run_benchmark.py

```bash
# Record baseline
python -m proxy.benchmarks.run_benchmark --baseline

# Record candidate with custom config
python -m proxy.benchmarks.run_benchmark --candidate --config models.ini

# Custom parameters
python -m proxy.benchmarks.run_benchmark --candidate \
    --base-url http://localhost:8000 \
    --model plan \
    --num-requests 20 \
    --concurrency 4 \
    --max-tokens 256 \
    --output my_candidate.json

# With prometheus snapshot
python -m proxy.benchmarks.run_benchmark --candidate \
    --snapshot-script proxy/benchmarks/prometheus_snapshot.sh
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--baseline` | — | Record baseline metrics |
| `--candidate` | — | Record candidate metrics |
| `--config` | — | Path to `models.ini` (for quantization info) |
| `--output` | `<run_type>_<timestamp>.json` | Output file path |
| `--base-url` | `http://localhost:8000` | Proxy base URL |
| `--model` | `plan` | Model name to benchmark. Use a configured alias (`plan`, `author`, `code`) — `Qwen3` does NOT resolve to a model config, so requests bypass `proxy_with_fallback` smart routing (no routing clamp, no remote fallback) and are dispatched directly to local. |
| `--num-requests` | `5` | Number of requests to send |
| `--concurrency` | `1` | Concurrent request count |
| `--max-tokens` | `128` | Max tokens per response |
| `--timeout` | `60.0` | Request timeout (seconds) |
| `--prompts` | — | JSON file with prompts array |
| `--snapshot-script` | — | Path to `prometheus_snapshot.sh` |

### compare_results.py

```bash
# Generate Markdown report
python -m proxy.benchmarks.compare_results baseline.json candidate.json

# Generate JSON output (for programmatic consumption)
python -m proxy.benchmarks.compare_results baseline.json candidate.json --json

# Custom thresholds
python -m proxy.benchmarks.compare_results baseline.json candidate.json \
    --memory-threshold 20 \
    --latency-threshold 15 \
    --tps-threshold 15
```

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | All gates passed |
| 1 | One or more gates failed |
| 2 | Inconclusive (missing data) |

### prometheus_snapshot.sh

```bash
# Collect 3 samples at 5s intervals
bash proxy/benchmarks/prometheus_snapshot.sh

# Write to file with 10s intervals
bash proxy/benchmarks/prometheus_snapshot.sh \
    --output /tmp/metrics.txt \
    --interval 10 \
    --admin-port 8080
```

## Large-prompt prefill validation (ctx-size evaluation)

The harness supports validating prefill at 30K–120K token prompt sizes before
any configuration evaluation is trusted (F1: LP-0MSC95VTC008GVR7).

### Fixtures

Large-prompt fixtures live in `proxy/benchmarks/large_prompts.json` — a JSON
dict of named prompts (`30k`, `60k`, `90k`, `120k`) generated deterministically
(seeded RNG, English prose ≈3 chars/token) by
`generate_large_prompt_fixture()`. Regenerate with:

```python
from benchmarks.slot_benchmark import generate_large_prompt_fixture, save_large_prompt_fixture
prompts = {f"{n//1000}k": generate_large_prompt_fixture(token_target=n) for n in (30000, 60000, 90000, 120000)}
save_large_prompt_fixture("proxy/benchmarks/large_prompts.json", prompts)
```

### Dry-run sweep (no requests sent)

Validates that all large prompts load, fit within the configured ctx-size, and
that the output JSON schema is consumable by `compare_results.py`:

```bash
python -m proxy.benchmarks.run_benchmark --baseline --dry-run \
    --prompts proxy/benchmarks/large_prompts.json \
    --output /tmp/dry_run_sweep.json --model plan
```

Completes in well under 15 minutes (typically <1s). `ctx_size` is read from
the per-model `ctx-size` in `models.ini`; prompts whose estimated token count
exceeds ctx-size produce a warning.

### Live prefill sweep with memory capture

Reproducible command sequence to drive real prefill at 30K–120K and capture
GPU/KV memory during the run:

```bash
# 1. (Optional) record GPU/system memory during the run in a background shell:
bash proxy/benchmarks/prometheus_snapshot.sh --output /tmp/metrics.txt --interval 10 &

# 2. Send one request per large prompt size (adjust --num-requests as needed):
python -m proxy.benchmarks.run_benchmark --baseline \
    --prompts proxy/benchmarks/large_prompts.json \
    --num-requests 4 --max-tokens 128 \
    --output /tmp/large_prompt_sweep.json --model plan

# 3. Inspect the summary (prefill throughput = t/s, TTFT, P95):
python -m proxy.benchmarks.compare_results /tmp/dry_run_sweep.json /tmp/large_prompt_sweep.json
```

Note: `prometheus_snapshot.sh` captures GPU VRAM (rocm-smi) and system memory
always; `llama_kv_cache_used_bytes`/`llama_kv_cache_capacity_bytes` from
`/metrics` require llama-server to be started with `--metrics`. The `/slots`
endpoint exposes per-slot `n_ctx`/`n_past` regardless.

### slot_benchmark.py cold vs warm phases

The slot-count benchmark (`slot_benchmark.py`) supports production-
representative measurements:

| Flag | Default | Purpose |
|------|---------|---------|
| `--clean-cache` | OFF | Clear slot cache before a run. Default OFF preserves production's warm `slot-save-path` persistence; use `--clean-cache` only for cold-start characterization. |
| `--phase cold\|warm` | `cold` | Records which measurement phase produced the JSON output. Warm = steady-state (≥30 min live traffic / ≥20 completed local turns); cold = first runs post-restart. |
| restart timestamps | — | `proxy_restart_time` and `llama_ready_time` ISO timestamps are recorded in each run's JSON `config` for reproducible measurement windows. |

```bash
# Warm steady-state run, preserving the live slot cache:
python -m proxy.benchmarks.slot_benchmark --slots 6 --phase warm

# Cold-start run with a clean slate:
python -m proxy.benchmarks.slot_benchmark --slots 6 --phase cold --clean-cache
```

### Measured findings (ctx-size eval F2/F3, 2026-08-04)

1. **Routing-estimate tokenizer mismatch**: the large-prompt fixtures (seeded
   prose, `large_prompts.json`) tokenize ~1.69x higher under Qwen3's native
   tokenizer than tiktoken cl100k estimates (90930 chars → est 22732, actual
   38529). Prompts can pass the routing clamp yet exceed the KV slot → llama-server
   HTTP 400 → remote fallback. See `docs/llama-router.md`.
2. **Cold vs warm collapse**: a 30K local prefill takes 222.7s cold (re-prefill
   storm) vs 8.6s warm — measure both phases, report the warm (steady-state)
   number as production-representative.
3. **Large local prefills are slow**: a 60K prompt (~77K actual tokens) takes
   ~600s to prefill locally in cold AND warm; >2-min prefills can trip the proxy's
   dispatch-lease orphan_cleanup (restarting the prefill). Configs admitting
   >40K prompts expose this.
4. **KV memory is total-ctx-bound, not slot-bound** (q8_0, 10 layers): 131072
   total ctx → 1362.7 MiB KV; 262144 → ~2720 MiB, regardless of slot split.

#### Per-config per-slot KV headroom table (F4, LP-0MSC95W3T000CCYC)

Reproducible via `python3 proxy/benchmarks/kv_memory_table.py` (or `--json`).
Method: q8_0 KV per-token cost measured from llama-server logs
(1362.7 MiB / 131072 total ctx cells); model Qwen3 35B Q5_K_M = 24.7 GiB;
~87 GiB available (124 GiB total, measured across F2/F3 run snapshots):

```
| Config | Slots | per-slot ctx | per-slot KV (MiB) | total KV (MiB) | Model+KV (GiB) | headroom (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 8x32.8K | 8 | 32768 | 340.7 | 2725.4 | 27.36 | 59.64 |
| 6x43.7K | 6 | 43690 | 454.2 | 2725.4 | 27.36 | 59.64 |
| 4x65.5K | 4 | 65536 | 681.4 | 2725.4 | 27.36 | 59.64 |
| 3x87.4K | 3 | 87381 | 908.5 | 2725.4 | 27.36 | 59.64 |
| 2x131K | 2 | 131072 | 1362.7 | 2725.4 | 27.36 | 59.64 |
| 3x43.7K live baseline | 3 | 43690 | 454.2 | 1362.7 | 26.03 | 60.97 |
```

Memory is NOT a constraint for any candidate (≥59 GiB headroom at all
configs); the intake's "~71GB available" claim is confirmed upward.

#### Metric semantics (F1/F2/F3 variances, noted in the 2026-08-24 audit)

The ACs reference "prefill throughput (t/s)". In the recorded F2/F3 JSONs:

- `tokens_per_second` is **generation** throughput (`completion_tokens / elapsed`),
  not prefill throughput. Prefill t/s is derivable as `prompt_tokens /
  time_to_first_token_seconds` (e.g. ~125 tok/s documented in F3 comments).
- `time_to_first_token_seconds` is a **heuristic estimate** for non-streaming
  requests (`elapsed / (completion + 1) * 1.5`) rather than a server-reported
  TTFT (`run_benchmark.py` RequestResult).
- P95 fields are **aggregate** per run; each prompt size has 1 sample per run,
  so per-size P95 is not first-class in the JSON (derivable from `requests`).

These semantics are consistent across all F2/F3 JSONs and `compare_results.py`
consumes the same summary keys, so delta comparisons remain valid.

#### Run-window error taxonomy (F3 AC3, LP-0MSC95W0H0003JOL)

Candidate benchmark JSONs record errors whose root cause is external to the
original 503-protocol concern (`backend_unavailable` during llama-server
restarts). Triage of every recorded error in `benchmark-results/F3_candidates/`:

- **2× HTTP 500 `Router.Unavailable`** (`candidate_2x131K_q8_cold.json`,
  90K/120K fixtures): the remote provider (deepseek-v4-flash-free) was down
  during the run window — a remote-outage, not a local restart-window 503.
  The proxy correctly fell through local→remote per the routing clamp.
- **1× 1200s client timeout** (`candidate_3x87.4K_q8_warm.json`, 60K fixture):
  the ~77K-token local prefill exceeded the proxy's dispatch-lease
  `orphan_cleanup` and was restarted, never completing — a known large-prefill
  lease hazard tracked as follow-up LP-0MSEHMMBK0062ZPI.

The 503 `backend_unavailable` path itself is code-verified
(`proxy/proxy/backend_health.py`, `proxy/proxy/router_helpers.py`) and
test-covered (`proxy/tests/test_backend_resilience.py`).

## Gating Policy

The following thresholds define minimum acceptable criteria for candidate
configurations. All thresholds are configurable via CLI flags or by editing
`compare_results.py`.

### Memory Gate

| Threshold | Default | Description |
|-----------|---------|-------------|
| `memory_reduction_pct` | 25% | Minimum reduction in KV cache footprint (candidate vs baseline) |

A candidate must demonstrate at least a 25% reduction in memory (RSS or KV
cache bytes) to be considered for rollout. A reduction below this threshold
may still be acceptable if other gates pass strongly and the change brings
other benefits (e.g., improved quality).

### Latency Gate

| Threshold | Default | Description |
|-----------|---------|-------------|
| `max_latency_regression_pct` | 10% | Maximum allowed increase in average request duration |

A candidate must not regress average request latency by more than 10%.
Minor regressions (5–10%) may be acceptable if the memory savings are
significant (>30%).

### Throughput Gate (TPS)

| Threshold | Default | Description |
|-----------|---------|-------------|
| `max_tps_regression_pct` | 10% | Maximum allowed decrease in tokens-per-second throughput |

A candidate must not regress average tokens-per-second by more than 10%.
This prevents quantization changes that severely degrade generation speed.

### Time-to-First-Token Gate (TTFT)

| Threshold | Default | Description |
|-----------|---------|-------------|
| `max_ttft_regression_pct` | 10% | Maximum allowed increase in time-to-first-token |

A candidate must not regress average time-to-first-token by more than 10%.
This is especially important for interactive use-cases.

### Quality Considerations

While the benchmark suite focuses on performance metrics, quantization changes
can affect output quality. Operators should also evaluate:

- **Token-level divergence**: Compare generated tokens between baseline and
  candidate for identical prompts. High divergence may indicate quality loss.
- **Perplexity delta**: If available, measure perplexity on a held-out
  evaluation set. A small increase (< 5%) is generally acceptable.
- **Human evaluation**: For critical applications, run a blind A/B test with
  human raters before full rollout.

### Override Procedure

If a candidate fails one or more gates but provides other compelling benefits:

1. Document the specific gate failures and their magnitudes.
2. Provide a rationale for why the regression is acceptable in context.
3. Obtain sign-off from the team lead or designated reviewer.
4. Override thresholds via CLI for the specific comparison:

   ```bash
   python -m proxy.benchmarks.compare_results baseline.json candidate.json \
       --memory-threshold 15 \
       --latency-threshold 20
   ```

## Example Workflow

```bash
# Step 1: Ensure proxy is running
# Verify proxy is running (check /health endpoint)
curl -s http://127.0.0.1:8000/health

# Step 2: Record baseline with current config
python -m proxy.benchmarks.run_benchmark --baseline --output baseline.json

# Step 3: Modify config (e.g., change quantization in models.ini)
# Change hf-repo from Q5_K_M to Q4_K_M for Qwen3

# Step 4: Restart proxy to pick up new config
# Restart proxy (kill and re-run start-proxy.sh)
pkill -f 'uvicorn proxy.server' && sleep 2 && bash proxy/scripts/start-proxy.sh

# Step 5: Record candidate
python -m proxy.benchmarks.run_benchmark --candidate --config models.ini --output candidate.json

# Step 6: Compare
python -m proxy.benchmarks.compare_results baseline.json candidate.json > report.md

# Step 7: Review report
cat report.md
```

## Output Format

### Benchmark result JSON

```json
{
  "config": {
    "run_type": "baseline",
    "model": "Qwen3",
    "prompts": ["..."],
    "num_requests": 5,
    "quantization": "Q5_K_M",
    "ctx_size": 65000
  },
  "requests": [
    {
      "request_index": 0,
      "prompt": "Explain quantum computing...",
      "status": "completed",
      "total_duration_seconds": 2.345,
      "prompt_tokens": 15,
      "completion_tokens": 128,
      "tokens_per_second": 54.58,
      "time_to_first_token_seconds": 0.234,
      "error": null
    }
  ],
  "summary": {
    "total_requests": 5,
    "completed": 5,
    "errors": 0,
    "avg_total_duration_seconds": 2.345,
    "avg_tokens_per_second": 54.58,
    "avg_time_to_first_token_seconds": 0.234,
    "total_prompt_tokens": 75,
    "total_completion_tokens": 640,
    "memory_snapshot_bytes": 8000000000
  },
  "timestamp": "2026-07-13T00:00:00Z"
}
```

### Comparison report (Markdown)

The comparison tool generates a report with:

- Configuration comparison table
- Summary comparison with deltas
- Gating policy results (PASS/FAIL/SKIP per gate)
- Overall verdict
