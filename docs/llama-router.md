# Llama Router Mode

This note summarizes how to run llama-server in router mode for multi-model co-residency with the proxy.

## Overview

Router mode allows llama-server to host multiple models concurrently by launching child processes per model.
The proxy can keep embeddings and a primary model available without restarts.

## Configuration

- `models.ini` defines router presets (default location: repo root `models.ini`).
- `proxy/config.yaml` enables router mode and can preload models.

Example `server` config:

```yaml
server:
  llama_router_mode: true
  llama_router_preload:
    - "embeddings"
    - "qwen3"
  llama_models_max: 2
  max_concurrent_queries: 4
  backend_retry_attempts: 3
  backend_retry_base_delay_seconds: 0.25
  backend_retry_max_delay_seconds: 2.0
  backend_retry_jitter_ratio: 0.25
  llama_watchdog_interval_seconds: 5
  llama_backend_probe_timeout_seconds: 2
  llama_self_heal_max_attempts: 3
  llama_self_heal_window_seconds: 300
  llama_self_heal_backoff_base_seconds: 1
  llama_self_heal_retry_after_seconds: 30
```

## Running

Start the proxy normally. The proxy will start llama-server in router mode via `start-llama.sh` and preload models.

To run llama-server directly in router mode:

```bash
./start-llama.sh router
```

The router exposes:

- `GET /models` to list models.
- `POST /models/load` to load a model.

## Notes

- `models.ini` must include both the embeddings model and the primary model preset.
- `llama_models_max` limits concurrent models and controls LRU eviction.
- The proxy health endpoint (`/health`) is readiness-gated (`ready: true/false`) and includes active backend probing (`backend_reachable`). After router/worker crashes it reports `status: degraded` until watchdog recovery completes.
- Router worker zombie/defunct states are treated as backend failures and trigger self-healing.
- During active self-healing, requests return `503` with `Retry-After: 30` and a `backend_recovery_in_progress` error payload.
- Backend crash-path signals are exposed via `/health` and `/admin/metrics` in `backend_signals`, and current recovery progress is reported in `backend_recovery`.
- Repro fault injection script: `proxy/scripts/fault-injection-backend-crash.sh` captures health/metrics snapshots and log signatures during a forced backend crash.

## KV-cache quantization (LP-0MSDCLQ2W001LGWC)

KV-cache data type is a first-class llama-server preset option, so it is configured
per-model in `models.ini` (in the model's own section, like `ctx-size`):

```ini
[Qwen3]
cache-type-k = q8_0
cache-type-v = q8_0
```

Allowed values (lowercase): `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`.
`f16` is the llama-server default; `q8_0` roughly halves KV read cost per decoded
token and is the recommended default for large-context sessions; `q4_0` saves more
VRAM at a small quality cost.

Rationale (LP-0MSDCLQ2W001LGWC): decode is memory-bandwidth-bound on the Strix Halo
iGPU (~256 GB/s shared). Each decoded token reads the model weights plus the session
KV, and the KV term grows linearly with context (at f16, ~20 KB/token → ~1.1 GB at a
57K-token session). Quantizing KV to q8_0 cuts that term roughly in half, improving
large-context decode throughput ~1.2–1.4x without changing ctx-size or slot count.

For single-model (non-router) startup, `start-llama.sh` reads `cache-type-k` /
`cache-type-v` from `models.ini` and passes `--cache-type-k` / `--cache-type-v` to
llama-server (defaulting to f16 when unset).

## Session context-pressure warning (LP-0MSDCLQ2W001LGWC)

Sessions with contexts near the per-slot limit decode far slower (KV reads scale
with context), and compaction is performed by the agents, not the proxy. The proxy
emits a `context_pressure` WARNING at routing time when a session's estimated
context reaches the configured fraction of the effective per-slot context
(`ctx_size / slots - 4096` output headroom).

```yaml
server:
  context_pressure_warn_ratio: 0.8  # 0 disables; default 0.8
```

The warning names the session and the ratio so operators/agents can compact before
decode degrades. See `proxy/tests/test_context_pressure_warning.py`.

## Routing-estimate tokenizer mismatch (LP-0MSAOQTJS000FFVM F2/F3 finding)

The smart-routing clamp (`_effective_large_context_thresholds` in
`proxy/proxy/provider.py`) estimates prompt tokens with tiktoken (cl100k) via
`count_text_tokens`. Benchmark measurements (2026-08-04) found tiktoken
**undercounts Qwen3-native tokens by ~1.69x for dense prose**: a 90930-char
fixture estimates 22732 tokens but Qwen3's tokenizer produces 38529.

Consequences:
- A prompt can pass the clamp check (est < per_slot − 4096) yet exceed the KV
  slot at decode time → llama-server HTTP 400 → remote fallback. Measured on
  4x65.5K (60K fixture: est 45357 < clamp 61440, actual 77060 > 65536 slot) and
  8x32.8K (30K fixture: est 22732 < clamp 28672, actual 38529 > 32768 slot).
- Effective local capacity for dense prose is ~39K tokens regardless of slot
  size until the estimator is corrected.

Mitigations (see follow-up LP-0MSEGPO77005CYCQ): use the Qwen3-native tokenizer
in routing estimates, or set `token_estimate_multiplier` (~1.69) on the
plan/author/code model entries. No multiplier is currently set.
