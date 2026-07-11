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
env_vars=(PORT LLAMA_PARALLEL LLAMA_MODELS_PRESET LLAMA_MODELS_MAX LLAMA_SLOT_SAVE_PATH LLAMA_SERVER_NO_MMAP HSA_OVERRIDE_GFX_VERSION ROCM_LLVM_PRE_VEGA)

if ! command -v podman >/dev/null 2>&1; then
  echo "podman not found in PATH" >&2
  exit 2
fi

# Create the container if it does not exist (non-interactive)
if ! podman container exists "$container" >/dev/null 2>&1; then
  echo "Creating container '$container' from image $image"

  # Host HuggingFace cache path; can be overridden with LLAMA_HOST_HF_CACHE
  host_hf_cache="${LLAMA_HOST_HF_CACHE:-${HOME}/.cache/huggingface/hub}"
  hf_mount_args=()
  if [ -n "$host_hf_cache" ]; then
    if [ -d "$host_hf_cache" ]; then
      hf_mount_args+=( -v "$host_hf_cache":/root/.cache/huggingface/hub:rw )
    else
      echo "Host HF cache not found at $host_hf_cache; creating"
      mkdir -p "$host_hf_cache" || true
      hf_mount_args+=( -v "$host_hf_cache":/root/.cache/huggingface/hub:rw )
    fi
  fi

  podman create --name "$container" \
    --device /dev/kfd --device /dev/dri --security-opt label=disable \
    --network host \
    -v "$host_repo":/work:rw \
    "${hf_mount_args[@]}" \
    "$image" sleep infinity
fi

# Warn if an existing container lacks the host HuggingFace cache mount
if podman container exists "$container" >/dev/null 2>&1; then
  host_hf_cache="${LLAMA_HOST_HF_CACHE:-${HOME}/.cache/huggingface/hub}"
  hf_mounted=$(podman inspect "$container" --format '{{json .Mounts}}' 2>/dev/null | grep -c "$host_hf_cache" || true)
  if [ "$hf_mounted" -eq 0 ]; then
    echo "WARNING: Existing container '$container' does not mount the host HF cache at $host_hf_cache" >&2
    echo "The container may re-download large model files instead of reusing host cache." >&2
    echo "" >&2
    echo "To recreate the container with the HF cache mount, run:" >&2
    echo "  podman stop $container && podman rm $container && $0 $*" >&2
    echo "" >&2
    echo "To override the HF cache path, set LLAMA_HOST_HF_CACHE:" >&2
    echo "  LLAMA_HOST_HF_CACHE=/custom/cache/path $0 $*" >&2
  fi
fi

# Ensure the container is running
# Get the full container state; valid startable states are: created, exited
# States like "stopping" or "paused" cannot be started and require recreation.
container_status=$(podman inspect -f '{{.State.Status}}' "$container" 2>/dev/null || echo "")

if [ -z "$container_status" ]; then
  # Container doesn't exist or inspect failed; will be handled by creation logic
  true
elif [ "$container_status" = "running" ]; then
  # Already running, nothing to do
  true
elif [ "$container_status" = "created" ] || [ "$container_status" = "exited" ]; then
  # Valid startable state
  echo "Starting container $container"
  podman start "$container"
else
  # Improper state (e.g. stopping, paused) — force-remove and recreate
  echo "WARNING: Container '$container' is in state '$container_status' which is not startable. Force-removing and recreating." >&2
  podman rm -f "$container"

  # Recreate the container (reuses the same logic as initial creation)
  host_hf_cache="${LLAMA_HOST_HF_CACHE:-${HOME}/.cache/huggingface/hub}"
  hf_mount_args=()
  if [ -n "$host_hf_cache" ]; then
    if [ -d "$host_hf_cache" ]; then
      hf_mount_args+=( -v "$host_hf_cache":/root/.cache/huggingface/hub:rw )
    else
      echo "Host HF cache not found at $host_hf_cache; creating"
      mkdir -p "$host_hf_cache" || true
      hf_mount_args+=( -v "$host_hf_cache":/root/.cache/huggingface/hub:rw )
    fi
  fi

  podman create --name "$container" \
    --device /dev/kfd --device /dev/dri --security-opt label=disable \
    --network host \
    -v "$host_repo":/work:rw \
    "${hf_mount_args[@]}" \
    "$image" sleep infinity

  echo "Starting recreated container $container"
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

# If a server is already listening inside the container on :8080,
# avoid starting another and instead attach to the server log so the
# parent process remains alive for the proxy's process supervision.
if podman exec "$container" sh -c "ss -ltnp 2>/dev/null | grep -E ':8080' >/dev/null 2>&1"; then
  echo "Detected backend listening inside container; attaching to logs"
  exec podman exec "${env_args[@]}" "$container" tail -F /work/llama-server.log
fi

# Decide command inside container
if [ -z "$model" ] || [ "$model" = "router" ]; then
  echo "Executing router inside container $container"
  exec podman exec "${env_args[@]}" "$container" /work/start-llama.sh router
else
  echo "Executing model '$model' inside container $container"
  exec podman exec "${env_args[@]}" "$container" /work/start-llama.sh "$model"
fi
