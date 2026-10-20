---
name: start-proxy
description: "Start or restart the LLM proxy server and all its backend services (llama-server, TTS server) using the start-proxy.sh script with nohup. Trigger on ANY user query requesting a proxy start, restart, or relaunch: 'start proxy', 'restart proxy', 'start the proxy', 'proxy is down', 'restart the proxy', 'restart it', 'restart the server', 'reboot the proxy'."
---

# Start/Restart Proxy Skill

Start or restart the LLM proxy server and all backend services (llama-server,
TTS server) in a single command.

## Prerequisites

The proxy project must be at `/home/rgardler/projects/llm/proxy` with the
`start-proxy.sh` script present at `proxy/scripts/start-proxy.sh`.

## Usage

Run from the proxy directory:

```bash
cd /home/rgardler/projects/llm/proxy
nohup bash scripts/start-proxy.sh --restart &>/tmp/proxy-startup.log &
```

Wait for startup to complete before verifying:

```bash
sleep 30
curl -s http://localhost:8000/health | python3 -m json.tool
```

Confirm all three services are healthy:
- `status`: `"healthy"`
- `llama_server_running`: `true`
- `tts_server_running`: `true`
- `tts_server_healthy`: `true`

If the health check returns `"degraded"`, check the startup log:

```bash
tail -30 /tmp/proxy-startup.log
```

## Behaviour

- `--restart` kills any running proxy, llama-server, and TTS processes before
  starting fresh.
- `nohup` keeps the process running after the terminal session ends.
- The script auto-resolves API keys from `~/.pi/agent/auth.json`.
- A restart **fails closed** when an inherited environment API key disagrees
  with `~/.pi/agent/auth.json`, so a rotated `auth.json` is never silently
  ignored. The startup log names the variable with both 8-character
  fingerprints (`fp=xxxxxxxx`) and prints the remediation. Resolve it by
  unsetting the stale variable in the launching shell (e.g.
  `unset OPENCODE_API_KEY`), or re-run with `--use-env-keys` — or set
  `LLAMA_PROXY_USE_ENV_KEYS=1` for an unattended restart that cannot pass
  flags — when the environment is the intended key source. See
  `proxy/README.md` → "API key rotation and stale env keys".
- Startup takes 30-60 seconds (model loading for llama-server + TTS server).

## Errors

| Symptom | Likely Cause |
|---------|-------------|
| Port conflict (8080/8081/8000) | Stale process from previous run — rerun with `--restart` |
| `ERROR: stale API key environment variable(s) detected — refusing to start` | An inherited environment key differs from `~/.pi/agent/auth.json` after a rotation — `unset` the named variable, re-export the current key, or pass `--use-env-keys` (`LLAMA_PROXY_USE_ENV_KEYS=1` when unattended) |
| TTS server not starting | Check `scripts/start-qwentts.sh` path in config.yaml |
| llama-server not starting | Check the binary at path from config.yaml or in PATH |
| GPU contention | TTS and LLM share the same GPU — requests may 503 during concurrent generation |
