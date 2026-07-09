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

## CI integration

To run the live tests in CI, the CI runner must have:
- A GPU with ROCm drivers
- llama-server built and running on port 8080
- The proxy running on port 8000

For CI environments without a GPU, only the mocked tests run (no
`RUN_LIVE_HOST_FLOW` flag set).
