# Configuration

```bash
# Install Podman
sudo apt install podman -y

# Install Distrobox (toolbox is not available in Ubuntu repos)
curl -s https://raw.githubusercontent.com/89luca89/distrobox/main/install | sudo sh

# Ensure you can run containers
sudo loginctl enable-linger $USER
```

# Build the contianer image

```bash
podman build -t localhost/llama-rocm:gfx1151 -f Containerfile .

distrobox create \
  --name llama \
  --image localhost/llama-rocm:gfx1151 \
  --additional-flags "--device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render --security-opt=seccomp=unconfined"
```

# Building llama.cpp with SSL support

The `-hf` flag for downloading models from Hugging Face requires SSL support. 
Build llama.cpp with one of the following SSL options:

```bash
# Enter the container
distrobox enter llama

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

# Install the binaries
sudo cmake --install build
```

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

# Starting the LLM (Direct)

To run llama-server directly without the proxy:

```bash
~/project/llmstart-llama.sh [model]
```


See Llama on Evo X1 - [pablo-ross/strix-halo-gmktec-evo-x2 ](https://github.com/pablo-ross/strix-halo-gmktec-evo-x2)

