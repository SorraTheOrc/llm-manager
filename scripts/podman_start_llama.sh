#!/usr/bin/env bash
set -euo pipefail

# Wrapper to start llama-server inside a podman container named 'llama'.
# Designed to be invoked by the proxy lifecycle as the configured
# llama_start_script so the proxy can spawn/manage the container-backed
# router. The script will block (exec) into podman exec so the parent
# process remains alive while the in-container server runs.

model="${1:-}"
container="${PODMAN_LC_CONTAINER:-llama}"
host_repo="${LLAMA_HOST_REPO:-/home/rgardler/projects/llm}"
image="${LLAMA_CONTAINER_IMAGE:-localhost/llm-llama:local}"

# Required env vars we will forward into the container (if present)
env_vars=(PORT LLAMA_SERVER_BIN LLAMA_MODELS_PRESET LLAMA_MODELS_MAX LLAMA_SLOT_SAVE_PATH LLAMA_SERVER_NO_MMAP HSA_OVERRIDE_GFX_VERSION ROCM_LLVM_PRE_VEGA)

if ! command -v podman >/dev/null 2>&1; then
  echo "podman not found in PATH" >&2
  exit 2
fi

# Create the container if it does not exist (non-interactive)
if ! podman container exists "$container" >/dev/null 2>&1; then
  echo "Creating container '$container' from image $image"
  podman create --name "$container" \
    --device /dev/kfd --device /dev/dri --security-opt label=disable \
    -p 127.0.0.1:8080:8080 \
    -v "$host_repo":/work:rw \
    "$image" sleep infinity
fi

# Ensure the container is running
running=$(podman inspect -f '{{.State.Running}}' "$container" 2>/dev/null || echo "false")
if [ "$running" != "true" ]; then
  echo "Starting container $container"
  podman start "$container"
fi

# Make sure the start script inside the mount is executable
podman exec "$container" chmod +x /work/start-llama.sh || true

# Build env args to pass into podman exec
env_args=()
for v in "${env_vars[@]}"; do
  val="${!v:-}"
  if [ -n "$val" ]; then
    env_args+=(--env "$v=$val")
  fi
done

# Decide command inside container
if [ -z "$model" ] || [ "$model" = "router" ]; then
  echo "Executing router inside container $container"
  exec podman exec "${env_args[@]}" "$container" /work/start-llama.sh router
else
  echo "Executing model '$model' inside container $container"
  exec podman exec "${env_args[@]}" "$container" /work/start-llama.sh "$model"
fi
