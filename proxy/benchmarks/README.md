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
    --model Qwen3 \
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
| `--model` | `Qwen3` | Model name to benchmark |
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
proxyctl status

# Step 2: Record baseline with current config
python -m proxy.benchmarks.run_benchmark --baseline --output baseline.json

# Step 3: Modify config (e.g., change quantization in models.ini)
# Change hf-repo from Q5_K_M to Q4_K_M for Qwen3

# Step 4: Restart proxy to pick up new config
proxyctl restart

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
