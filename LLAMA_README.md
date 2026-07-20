# Configuration

```bash
# Install Podman
sudo apt install podman -y

# Ensure you can run containers
sudo loginctl enable-linger $USER
```

# Build the container image

```bash
podman build -t localhost/llama-rocm:gfx1151 -f Containerfile .
```

# Building llama.cpp with SSL support

The `-hf` flag for downloading models from Hugging Face requires SSL support. 
Build llama.cpp with one of the following SSL options:

```bash
# Navigate to llama.cpp source directory
cd ~/llama.cpp  # or wherever your llama.cpp source is located

# Pull latest changes
git pull

# Build with SSL support (OpenSSL - requires libssl-dev/openssl-devel)
cmake -B build -S . \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="gfx1151" \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DLLAMA_OPENSSL=ON

cmake --build build --config Release -j$(nproc)

# Install the binaries (optional; skip if you use build/bin directly)
# If sudo cannot write install_manifest.txt in the build dir, use --prefix ~/.local
sudo cmake --install build
```

# Benchmarking with llama-bench.sh

The project includes a benchmark wrapper (`scripts/llama-bench.sh`) around
llama.cpp's `llama-bench` tool for systematically measuring model performance
on Strix Halo (gfx1151) hardware.

## Usage

```bash
# Quick benchmark of a specific model with optimal defaults
./scripts/llama-bench.sh --model gemma

# Full parameter sweep on a model (GPU layers, threads, batch, flash-attn, KV cache)
./scripts/llama-bench.sh --sweep --model gemma

# Benchmark all discovered models (one run each)
./scripts/llama-bench.sh --all

# Full sweep across all models
./scripts/llama-bench.sh --all --sweep

# Preview commands without running
./scripts/llama-bench.sh --dry-run --all

# List discovered models
./scripts/llama-bench.sh --list-models

# List models as JSON
./scripts/llama-bench.sh --list-models-json
```

## Output

Results are output as JSON to stdout. In dry-run mode, planned commands are printed
to stderr so stdout remains clean for machine consumption.

```bash
# Save results to a file
./scripts/llama-bench.sh --model gemma > benchmark-results.json

# Pipe through jq for analysis
./scripts/llama-bench.sh --model gemma | jq '.results[].avg_ts'
```

## Benchmark Parameters

The script sweeps the following parameters (configurable in the script defaults):

| Parameter | Flag | Default | Sweep Values |
|-----------|------|---------|-------------|
| GPU layers | `-ngl` | 99 | 99, 80, 60 |
| Threads | `-t` | 16 | 16, 12, 8 |
| Batch size | `-b` | 2048 | 4096, 2048, 1024 |
| Micro batch | `-ub` | 512 | 512, 256 |
| Flash attn | `-fa` | 1 | 1, 0 |
| KV cache K | `-ctk` | f16 | f16, q8_0 |
| KV cache V | `-ctv` | f16 | f16, q8_0 |
| Prompt len | `-p` | 512 | 512, 1024 |
| Gen len | `-n` | 128 | 128, 256 |

## Model Discovery

The script automatically discovers GGUF models from:
1. HuggingFace cache (`~/.cache/huggingface/hub/`)
2. llama.cpp local cache (`~/.cache/llama.cpp/`)

Models can be filtered by name substring using `--model`. Incomplete downloads
are skipped with a warning.

## Library Path

The script automatically sets `LD_LIBRARY_PATH` to include
`~/llama.cpp/build/bin/` so that `llama-bench` can find its shared libraries
(`libllama.so.0`, `libggml*.so.0`).

## Custom Parameters

Override individual parameters via environment variables or by editing the
script defaults at the top of `scripts/llama-bench.sh`.

## Slot save/restore endpoints (KV persistence)

Slot persistence is required for stable KV cache reuse across requests. The
llama-server API exposes slot save/restore endpoints using a query parameter
syntax:

```bash
# Save slot 0 prompt cache
curl -X POST "http://127.0.0.1:55833/slots/0?action=save" \
  -H "Content-Type: application/json" \
  -d '{"filename":"slot_session.bin"}'

# Restore slot 0 prompt cache
curl -X POST "http://127.0.0.1:55833/slots/0?action=restore" \
  -H "Content-Type: application/json" \
  -d '{"filename":"slot_session.bin"}'
```

These files are stored under the directory supplied with `--slot-save-path`.
When running in router mode, the slot endpoints live on the **child model
ports** (not the router port). Use `GET /slots?model=<model-id>` on the router
port to find active slots.

**Note:** slot save/restore is not supported for multimodal models. If your
model downloads a `mmproj` file automatically, disable it with `--no-mmproj`
(or `no-mmproj = true` in `models.ini`) so slot persistence works.

### Rebuild recipe (reproducible)

This repo expects llama.cpp commit:

```bash
git -C ~/llama.cpp rev-parse HEAD
# e97492369888f5311e4d1f3beb325a36bbed70e9
```

To reproduce the build:

```bash
cd ~/llama.cpp

git fetch --all --tags
git checkout e97492369888f5311e4d1f3beb325a36bbed70e9

cmake -B build -S . \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="gfx1151" \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DLLAMA_OPENSSL=ON

cmake --build build --config Release -j$(nproc)

# Install the binaries (optional; skip if you use build/bin directly)
# If sudo cannot write install_manifest.txt in the build dir, use --prefix ~/.local
sudo cmake --install build
```

Ensure the server is started with `--slot-save-path /home/rgardler/projects/llm/slot-cache`
(see `start-llama.sh` + `models.ini`) so the slot endpoints return 200.

If OpenSSL is not available, install it first:
```bash
# Fedora/RHEL
sudo dnf install openssl-devel
```

Alternative SSL options (use one):
- `-DLLAMA_OPENSSL=ON` - Use system OpenSSL (recommended)
- `-DLLAMA_BUILD_BORINGSSL=ON` - Build with BoringSSL
- `-DLLAMA_BUILD_LIBRESSL=ON` - Build with LibreSSL

# Running the Proxy

The recommended way to interact with llama-server is through the LLama Proxy Server, which provides a unified OpenAI-compatible API endpoint, web UI, and automatic model switching.

```bash
cd proxy
sudo ./install.sh
sudo systemctl start llama-proxy
```

Access the web UI at `http://localhost:8000/`. See [proxy/README.md](proxy/README.md) for full documentation.

# Session-Based Prompt Caching

The proxy supports session-based incremental prompt ingestion through `X-Session-Id` headers. When a session ID is provided or auto-generated, the proxy:

1. Tracks per-session message history
2. Sends only new messages (delta) on subsequent requests within the same session
3. Passes `session_id` and `cache_prompt` to llama-server for KV cache reuse

This significantly reduces CPU usage and latency for multi-turn conversations. See [proxy/README.md](proxy/README.md#session-based-incremental-ingestion) for detailed documentation and examples.

# Starting the LLM (Direct)

To run llama-server directly without the proxy:

```bash
~/project/llm/start-llama.sh [model]
```

See Llama on Evo X1 - [pablo-ross/strix-halo-gmktec-evo-x2 ](https://github.com/pablo-ross/strix-halo-gmktec-evo-x2)
