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
