# Integration Tests — Host-First Flow

This document describes how to run the integration tests for the host-first
llama-server startup flow and the proxy lifecycle.

## Prerequisites

- Python 3.10+
- pytest (`pip install pytest requests`)
- llama-server built and installed (see `docs/llama-router.md`)
- GPU with ROCm drivers (for live tests only)

## Test structure

Integration tests are in `proxy/tests/test_host_flow_integration.py` and are
split into two categories:

### 1. Mocked tests (default, no GPU required)

These tests validate the startup logic, fallback behavior, and state transitions
using monkeypatched subprocess calls. They run automatically as part of the
normal pytest suite:

```bash
# From the project root
python3 -m pytest proxy/tests/test_host_flow_integration.py -v
```

These tests cover:
- Host-start startup success and fallback
- Router mode (model=None → "router" argument)
- `llama_allow_host_fallback: false` behavior (no host-start attempt)
- Model loading state consistency after startup
- Progress logging parsing and formatting

### 2. Live tests (GPU required, opt-in)

These tests require a running llama-server and proxy on the development machine.
They are skipped by default and must be explicitly enabled. Live tests are in
a separate file to prevent accidental collection during normal test runs:

```bash
# Ensure llama-server and proxy are running
./scripts/start-llama.sh router
./proxy/scripts/start-proxy.sh

# Run live host-flow tests
RUN_LIVE_HOST_FLOW=1 python3 -m pytest proxy/tests/test_host_flow_live_e2e.py -v
```

Optional environment variables:
| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_LIVE_HOST_FLOW` | — | Set to `1`, `true`, or `yes` to enable live tests |
| `LIVE_PROXY_BASE_URL` | `http://localhost:8000` | Base URL of the running proxy |
| `LIVE_LLAMA_BASE_URL` | `http://localhost:8080` | Base URL of the running llama-server |

These tests cover:
- llama-server health endpoint
- Proxy health endpoint
- Embedding request via proxy
- Chat completion via proxy

### 3. GPU Offload Verification Tests (no GPU required for unit tests)

GPU offload verification tests are in `proxy/tests/test_gpu_offload_verification.py`.
These validate that the infrastructure for ROCm GPU offload (models.ini `[global] ngl`
parsing, environment variable propagation, router-mode command construction) is in place:

```bash
python3 -m pytest proxy/tests/test_gpu_offload_verification.py -v
```

For live GPU offload verification steps, see `docs/gpu-offload-verification.md`.

## Running all tests

```bash
# All proxy tests (includes mocked integration tests)
python3 -m pytest proxy/tests/ -v

# All project tests (build scripts + proxy)
python3 -m pytest tests/ proxy/tests/ -v
```

## On-demand live-server tests (opt-in, disabled by default)

Tests that spawn and kill real OS processes, or exercise the **live** proxy /
llama-server / TTS server, are **opt-in** so a routine `pytest` run can never
accidentally kill a running service, crash it via GPU contention, or close an
SSH session (see LP-0MS6R13CP009VO24). They follow the same convention as the
`e2e_live` / `RUN_LIVE_HOST_FLOW` gates.

- **pytest markers:** `e2e_live` (live proxy), `tts_integration` (live TTS),
  `live_port_kill` (spawns/kills real OS processes) — defined in
  `proxy/pytest.ini`.
- **Environment gates:**

  | Variable | Enables |
  |----------|---------|
  | `RUN_LIVE_PROXY_E2E=1` | Live proxy E2E tests (`test_plan_fallback_live_e2e.py`, `test_embeddings_integration.py`, `test_embeddings_concurrent.py`, `test_model_audit_plan_routing.py`) |
  | `RUN_LIVE_TTS=1` | Live TTS server tests (`test_tts_integration.py`) |
  | `RUN_LIVE_HOST_FLOW=1` | Live host-flow tests (`test_host_flow_live_e2e.py`) |
  | `LIVE_PORT_KILL_TESTS=1` | Shell script that spawns and kills real processes (`tests/test_start_proxy_restart.sh`) |

Examples:

```bash
# start-proxy.sh restart port-cleanup integration test (spawns real listeners
# and kills them with kill/fuser -k on random ports)
LIVE_PORT_KILL_TESTS=1 bash tests/test_start_proxy_restart.sh

# Embeddings/chat integration tests against the live proxy (real inference)
RUN_LIVE_PROXY_E2E=1 python3 -m pytest proxy/tests/test_embeddings_integration.py -v
RUN_LIVE_PROXY_E2E=1 python3 -m pytest proxy/tests/test_embeddings_concurrent.py -v

# TTS integration tests against the live tts-server
RUN_LIVE_TTS=1 python3 -m pytest proxy/tests/test_tts_integration.py -v
```

Without the env var each module prints a SKIP notice. The default `pytest` run
never executes these tests.

The embeddings integration tests (`test_embeddings_integration.py`) poll the
proxy until the embeddings alias answers with a 200 instead of relying on a
single tight-timeout request, and use generous health-check timeouts, so a
healthy-but-loaded proxy (e.g. concurrent chat streams contending for the
single GPU) does not cause spurious failures or skips (see LP-0MS9FM27K007NCNE).

## CI integration

To run the live tests in CI, the CI runner must have:
- A GPU with ROCm drivers
- llama-server built and running on port 8080
- The proxy running on port 8000

For CI environments without a GPU, only the mocked tests run (no
`RUN_LIVE_HOST_FLOW` flag set).
