# Systemd Service Units

This directory contains systemd unit files for the LLM proxy stack using
the **podman-container** deployment path.

## Services

| Unit | Purpose |
|------|---------|
| `llama-podman-wrapper.service` | Runs llama-server inside a podman container via `scripts/podman_start_llama.sh` |
| `llama-proxy.service` | Runs the proxy (uvicorn) on port 8000; depends on the podman wrapper |

## Installation

### User Service (Recommended)

User services run under your login session and don't require root.

```bash
# Copy unit files
mkdir -p ~/.config/systemd/user
cp llama-podman-wrapper.service llama-proxy.service ~/.config/systemd/user/

# For user services, the unit files reference the repository paths directly.
# Adjust User/Group and paths in the unit files if your setup differs.

# Reload, enable, and start
systemctl --user daemon-reload
systemctl --user enable --now llama-podman-wrapper.service
systemctl --user enable --now llama-proxy.service

# Check status
systemctl --user status llama-podman-wrapper.service
systemctl --user status llama-proxy.service

# View logs
journalctl --user -u llama-podman-wrapper.service -f
journalctl --user -u llama-proxy.service -f
```

### System Service (Alternative)

Use this approach only if the services must survive logout or run
independently of any user session. Requires root or sudo access.

```bash
# Copy unit files
sudo cp llama-podman-wrapper.service llama-proxy.service /etc/systemd/system/

# Reload, enable, and start
sudo systemctl daemon-reload
sudo systemctl enable --now llama-podman-wrapper.service
sudo systemctl enable --now llama-proxy.service

# Check status
sudo systemctl status llama-podman-wrapper.service
sudo systemctl status llama-proxy.service

# View logs
sudo journalctl -u llama-podman-wrapper.service -f
sudo journalctl -u llama-proxy.service -f
```

### Customizing Paths

If your repository is at a different path than `/home/rgardler/projects/llm`,
edit the unit files to update:
- `WorkingDirectory`
- `ExecStart` paths
- `Environment` paths (for PYTHONPATH, config, etc.)
- `User`/`Group` if different

## Verification

After installation and startup:

```bash
# Check proxy health
curl http://localhost:8000/health

# Check llama-server health (container)
curl http://localhost:8080/health

# Verify services are running
systemctl --user is-active llama-podman-wrapper.service
systemctl --user is-active llama-proxy.service
```

## Cross-references

- `docs/systemd/llama-server.service` — Alternative host-direct (non-container) systemd unit
- `docs/systemd/llama-proxy.service` — Proxy systemd unit docs (host-direct variant)
- `docs/INTEGRATION.md` — Integration test instructions
- `proxy/README.md` — Proxy configuration and startup
