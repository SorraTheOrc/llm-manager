ROCm Upgrade: Preflight and runbook

This document describes the preflight checks, upgrade steps, verification, and rollback procedures for upgrading ROCm on the host and validating container compatibility.

1) Preflight (non-destructive)

- scripts/rocm-preflight.sh --dry-run --json
  - Produces a JSON summary with the following keys:
    - ok: boolean — overall pass/fail for basic checks
    - repo_present: boolean — whether /etc/apt/sources.list.d/rocm.list exists
    - repo_versions: array — candidate ROCm versions parsed from the repo file
    - rocm_smi_present: boolean — whether rocm-smi is available
    - rocm_smi_version: string — captured rocm-smi version/tag (if present)
    - uname: string — kernel uname -a
    - gpu_detected: boolean — whether an AMD GPU was detected via lspci
    - gpu_lines: array — matching lspci lines
    - errors: array — non-fatal issues observed during preflight

Run: scripts/rocm-preflight.sh --dry-run --json

2) Container verification (non-destructive)

- scripts/verify-container-rocm.sh --dry-run --json
  - Validates that the ROCm 7.2.4 official base image can be pulled and that a container runtime is available
  - When run without --dry-run the script attempts to podman/docker pull and run a container to execute rocm-smi and a trivial HIP/py check

Run: scripts/verify-container-rocm.sh --dry-run --json

3) Build verification (llama.cpp)

- scripts/build-llama-test.sh --dry-run --json
  - Plans the steps to clone llama.cpp and build with HIP flags

Run: scripts/build-llama-test.sh --dry-run --json

4) Container rebuild (production)

- scripts/rebuild-container.sh --update-containerfile --image rocm/dev-ubuntu-24.04:7.2.4 --build --tag llm/llama-server:rocm-7.2.4 --json
  - Steps performed:
    - Update Containerfile FROM line (backup made as Containerfile.bak)
    - Build the image with podman/docker and tag it
    - Verify the image via scripts/verify-container-rocm.sh (non-dry-run recommended in staging)

- To push the built image to a registry, add `--push`:

  ```bash
  scripts/rebuild-container.sh --image rocm/dev-ubuntu-24.04:7.2.4 \
    --build --push --tag registry.example.com/llm/llama-server:rocm-7.2.4
  ```

  The script validates registry credentials before pushing. If no credentials are found for the
  registry, the script fails with a clear message:

  ```
  Registry credentials not found for 'registry.example.com'. Run 'podman login registry.example.com' first.
  ```

  To log in to the registry before pushing:

  ```bash
  # Podman
  podman login registry.example.com

  # Docker
  docker login registry.example.com
  ```

  > **Note:** The `--push` flag automatically implies `--build`. You do not need to pass both.

5) Rebuild llama.cpp and deploy

- scripts/rebuild-llama.sh --dry-run --json
  - Clone latest master, build with HIP flags, deploy binary to /home/rgardler/llama.cpp/build/bin/llama-server and attempt to restart using start-llama.sh

6) Host ROCm upgrade (manual step recommended)

- scripts/install-rocm.sh --dry-run --json
  - Produces planned steps to update /etc/apt/sources.list.d/rocm.list to ROCm 7.2.4, import GPG key, and perform apt update
- After review, run as root (careful): scripts/install-rocm.sh --version 7.2.4
- Note: Non-dry-run will write /etc/apt/sources.list.d/rocm.list and attempt to import the GPG key. It is recommended to review the file and key import step before executing.

7) Verification and rollback

- Run scripts/verify-upgrade.sh --dry-run --json to review the planned verification steps
- Run scripts/verify-upgrade.sh --json to perform health and test completion requests against the running llama-server
- Verify llama-server health: `curl http://localhost:8080/health`
- Verify proxy health: `curl http://localhost:8000/health`
- If using systemd services, check status: `systemctl --user status llama-server.service llama-proxy.service` or `sudo systemctl status llama-server.service llama-proxy.service`
- Rollback: scripts/rollback-rocm.sh --dry-run --json
  - If a backup of /etc/apt/sources.list.d/rocm.list exists (.bak) run scripts/rollback-rocm.sh (as root) to restore it
- **Note**: The host-first deployment model (via `start-llama.sh`) is documented in `docs/systemd/` and `proxy/README.md` → "Host-first deployment" section. Ensure systemd units are updated after ROCm upgrades if llama-server binaries change location.

8) Monitoring notes

- Consider adding a Prometheus rule to alert on llama-server health endpoint failures and unexpected rocm-smi outputs.

9) References

- Containerfile — suggested FROM update to rocm/dev-ubuntu-24.04:7.2.4
- start-llama.sh, install_proxy.sh — reboot/restart considerations
- docs/systemd/ — systemd unit files for llama-server and proxy
- proxy/README.md — "Host-first deployment" section with host-fallback configuration
- HOST_INSTALL.md — host installation instructions for llama-server

