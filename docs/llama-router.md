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

Mitigations (implemented in follow-up LP-0MSEGPO77005CYCQ F2/F3):
- Server-level `token_estimate_multiplier: 1.69` is set in `proxy/config.yaml`
  and applied consistently to BOTH the routing estimate (provider.py) and the
  slot-persistence estimate (session.py) via `_get_token_estimate_multiplier`
  in `proxy/proxy/provider.py`, so the routing clamp and the persistence cap
  compare Qwen3-native token counts. Per-model `token_estimate_multiplier`
  overrides the server-level value for routing.
- The slot-persistence cap `session_slot_max_prompt_tokens` is derived
  dynamically from the effective per-slot clamp
  (`local_model_ctx_size // active_slots - 4096` output headroom, the same
  source as the routing clamp) when the config key is absent/0, so it
  auto-adapts to slot-count/ctx-size changes.

## Warm-cache context-threshold routing (LP-0MSB2RASV009WFGI)

When routing requests to the local llama-server, the proxy applies a two-tier
context-size check before committing to local, in
`_should_bypass_local_for_large_context` (proxy/proxy/provider.py):

1. **Warm-cache threshold (hard cap):** If the estimated total prompt context
   exceeds `local_large_context_warm_cache_threshold` (default `100000` in
   `proxy/config.yaml`), local is bypassed regardless of cache state. This
   prevents routing excessively large total contexts to the local model slot
   even when the KV cache is warm.

2. **Cold-cache new-token check:** The number of uncached tokens is computed
   as `new_tokens = estimated_tokens × (1 − cached_ratio)`. If
   `new_tokens > cold_cache_threshold`, the prefill is considered too
   expensive and local is bypassed; otherwise the request routes local.

The `cached_ratio` is tracked per-session (see `proxy/session.py` delta
routing classification). A ratio of `0.0` (unknown sessions) is conservative:
local is bypassed whenever new tokens exceed the threshold. A ratio of `1.0`
(full warm cache) yields `new_tokens = 0`, so local is always used unless the
warm-cache hard cap applies. A threshold of `0` disables the bypass entirely.

Config keys (both nested under `server:` and flat forms are supported):

```yaml
server:
  local_large_context_fallback_threshold: 60000        # cold-cache new-token cap
  local_large_context_warm_cache_threshold: 100000     # total-context hard cap
```

## Per-period ctx_size in slot_schedule (LP-0MSLNK96T0018W4D)

`slot_schedule` entries may carry an optional `ctx_size`: the total context
across all slots (llama-server `--ctx-size`) while that entry is active.
When absent, the global `local_model_ctx_size` applies.

```yaml
server:
  slot_schedule:
    enabled: true
    entries:
      - time: "10:00"
        slots: 3
      - time: "23:59"
        slots: 2
        ctx_size: 262144   # overnight: 2 slots @ 256K
```

At a transition the proxy restarts llama-server with the new `--parallel`
AND context size, and the routing clamp (`_effective_large_context_thresholds`)
plus the `session_slot_max_prompt_tokens` dynamic derivation use the ACTIVE
period's `(ctx_size, slots)` — so overnight the per-slot cap becomes
`262144 // 2 - 4096 = 126976` while daytime stays `131072 // 3 - 4096 = 39594`.

**Router-mode mechanism:** a global `--ctx-size` on the router command line
would override per-model INI `ctx-size` for EVERY model (CLI args take highest
precedence in llama.cpp's preset merge), ballooning the embed model's KV cache.
Instead `start-llama.sh` patches ONLY the local model's `ctx-size` into a temp
copy of the preset (`LLAMA_CTX_SIZE`/`LLAMA_CTX_MODEL` exported by the proxy
lifecycle) and points `--models-preset` at it.

**Consistency invariant (F3 lesson):** the routing clamp must never admit
prompts larger than the real per-slot context after llama.cpp rounds
`n_ctx_seq` UP to a multiple of 256 (`262144/2 → 131072/slot`,
`131072/3 → 43776/slot`). The clamp `(ctx_size // slots - 4096)` is always ≤
the rounded per-slot context — enforced by
`proxy/tests/test_ctx_slot_validation.py::TestCtxSlotConsistency`.
