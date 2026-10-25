---
name: start-proxy
description: "Start or restart the LLM proxy server and all its backend services (llama-server, TTS server) using the start-proxy.sh script. Trigger on ANY user query requesting a proxy start, restart, or relaunch: 'start proxy', 'restart proxy', 'start the proxy', 'proxy is down', 'restart the proxy', 'restart it', 'restart the server', 'reboot the proxy'."
---

# Start/Restart Proxy Skill

Start or restart the LLM proxy server and all backend services (llama-server,
TTS server) in a single command.

## Prerequisites

The proxy project must be at `/home/rgardler/projects/llm/proxy` with the
`start-proxy.sh` script present at `proxy/scripts/start-proxy.sh`.

## Usage

Run from the proxy directory (the script detaches by default — it validates
synchronously, launches uvicorn in its own session, prints one handoff line and
exits 0):

```bash
cd /home/rgardler/projects/llm/proxy
bash scripts/start-proxy.sh --restart
```

The launcher prints a single handoff line:

```
Proxy launched detached: child pid <pid>, output in /tmp/proxy-startup.log | health check: curl -sf http://127.0.0.1:8000/health
```

> **NEVER start the proxy in the foreground under an agent tool call.**
> `start-proxy.sh` is detached by default for exactly this reason
> (LP-0MVGT0E6W000Q8CY): a tool-call timeout SIGKILLs the whole process group,
> and a foreground uvicorn dies mid-startup with no `Shutting down` line,
> leaving the service dead. Do not add `--foreground` (the interactive-debug
> opt-in) and do not wrap the command in `exec`/`tail` pipes that hold the
> caller's stdout open. The command above is the exact detached invocation.

Wait for startup to complete before verifying — startup failures after the
handoff (bind errors, model load, import errors) appear only in the log, so
**poll `/health`** instead of trusting the launcher's exit code:

```bash
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8000/health && break; sleep 2; done
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

### Flags and options

- `--restart` — kill any running proxy, llama-server, and TTS processes before
  starting fresh (this is the normal agent-triggered restart).
- `--foreground` — **interactive debugging only**: blocking `exec` of uvicorn in
  the current process group. Never use this under an agent tool call (a tool
  timeout kills the server). Unattended callers may set
  `START_PROXY_FOREGROUND=1` instead of the flag.
- `--log-file <path>` — where the detached server's stdout/stderr are written
  (default `/tmp/proxy-startup.log`); also read the handoff line for the active
  log path.

## Behaviour

- **Detached by default** (LP-0MVGT0E6W000Q8CY): after synchronous validation,
  uvicorn is launched with `setsid` (or `nohup` + `disown` when `setsid` is
  unavailable) with stdout/stderr redirected to the log file, then the script
  prints one handoff line and exits 0. Neither `nohup` nor a background `&` is
  needed around the command — the script already detaches itself.
- `--restart` kills any running proxy, llama-server, and TTS processes before
  starting fresh.
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
| Proxy log ends abruptly with **no `Shutting down` line**, and the triggering tool result was `Command timed out after <N> seconds` on a command that ran `start-proxy.sh` | The launcher ran uvicorn in the foreground and the tool timeout killed the whole process group. Sibling services (llama-server/TTS) are likely still alive. See the recovery box below |

### Recovery: proxy killed by a tool-call timeout

**Signature** — you hit this failure if ALL of these hold:

1. The proxy's log ends abruptly with **no `Shutting down` line** (SIGKILL, not
   a clean shutdown).
2. The triggering tool result read `Command timed out after <N> seconds`.
3. The launcher was run in the foreground (e.g. `start-proxy.sh` without the
   default detach, or under `--foreground`).
4. Sibling services (llama-server on 8080, TTS on 8081) are still alive — the
   proxy is the only casualty.

If the fast check `curl -sf http://127.0.0.1:8000/health` fails while
`curl -sf http://127.0.0.1:8080/health` succeeds, the fix is a plain restart —
**do not re-investigate the launch path**:

```bash
cd /home/rgardler/projects/llm/proxy
bash scripts/start-proxy.sh --restart
```

llama-server is preserved (warm KV / prompt cache) by `--restart`'s default
when its signature matches, so the restart is fast. Then poll `/health` as in
[Usage](#usage).
