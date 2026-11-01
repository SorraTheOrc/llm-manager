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

# These templates are already correct for a user service: they reference the
# repository paths directly and deliberately declare no User=/Group=.

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

> **Never set `User=`/`Group=` in a user service.** The systemd *user* manager
> already runs as your user and cannot call `setgroups`, so a unit that declares
> them aborts at exec time with:
>
> ```
> Failed to determine supplementary groups: Operation not permitted
> Main process exited, code=exited, status=216/GROUP
> ```
>
> Because `llama-proxy.service` declares
> `Requires=llama-podman-wrapper.service`, a broken wrapper takes the proxy down
> with it — both units fail and the stack stays dead after every boot
> (LP-0MVWIA33O004RKP7). `User=`/`Group=` belong **only** in the system-service
> variant below.

### System Service (Alternative)

Use this approach only if the services must survive logout or run
independently of any user session. Requires root or sudo access.

A system service **must** declare `User=`/`Group=`, so it cannot reuse the
user-service templates above. Use the system-service unit and setup commands
documented in `docs/systemd/llama-proxy.service` (which carries both the
user-service and system-service variants), and add the equivalent wrapper unit
under `/etc/systemd/system/` with `User=`/`Group=` set to the account that owns
this checkout.

```bash
# System units live here and REQUIRE User=/Group=
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

Do **not** add `User=`/`Group=` while editing user services — see the warning
above. Those directives apply only to a system-service installation.

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
- `docs/systemd/llama-proxy.service` — Proxy systemd unit docs (user-service **and** system-service variants)
- `docs/INTEGRATION.md` — Integration test instructions
- `proxy/README.md` — Proxy configuration and startup
- `tests/test_systemd_unit_templates.py` — Regression test pinning the
  `User=`/`Group=` rule above (LP-0MVWIA33O004RKP7)
