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
