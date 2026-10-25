## Global agent guidance

Read the global agent instructions at `~/.pi/agent/AGENTS.md` — they define the core principles, the Worklog (wl) work-item workflow, and the coding disciplines that apply to every project. That file is installed from the SorraAgents project (`~/projects/SorraAgents/AGENTS_GLOBAL.md`) by `~/projects/SorraAgents/scripts/install_pi.sh`, which symlinks it into place.

## Project-specific guidance

Follow the global AGENTS.md in addition to the rules below. The local rules below take priority in the event of a conflict.

### NEVER kill the LLM proxy

The LLM proxy runs as a Python process (`python3 -m uvicorn proxy.server:app`) alongside other Python processes (agent runners, audit runners, pi subprocesses). A blanket `killall`/`pkill` targeting `python3` or other broad process names WILL kill the proxy and take it offline for every consumer (local agents, herdr workers, remote LAN clients).

- **NEVER kill the proxy.** Do not run `killall`, `pkill`, `kill -9`, or any equivalent against `python3`, `uvicorn`, `proxy.server`, or broad process groups.
- To stop hung agent/audit subprocesses, kill the **specific PIDs only** — never a blanket process-name kill that could match the proxy.
- **If the proxy needs a restart, use the start-proxy skill:** `/skill:start-proxy --restart`. This performs the controlled teardown and relaunch of the proxy and its backends; do not kill processes manually and restart ad hoc.

### `forcing full prompt re-processing` is expected on Qwen3.x — NOT an error

`llama-server.log` periodically emits:

```
slot update_slots: id N | task M | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see llama.cpp#13194)
```

**Never report this as a cache failure, a defect, or an error condition, and do not file a work item for it.** Qwen3.6-35B-A3B is a hybrid attention + recurrent/SWA architecture: llama.cpp cannot rewind to the context checkpoints it created earlier in the slot, so it invalidates them (`erased invalidated context checkpoint ... n_swa = 0`) and re-prefills. The prompt cache is unaffected and still recovers prefixes normally.

Measured 2026-10-18 over 792 dispatches — caching is healthy:

- median `cached_ratio` **0.952** (p75 0.995); 568 `cache_state=warm` vs 224 `cold`
- per-task `n_past` shows real prefix reuse before any re-prefill
- prompt cache well under its cap: 7 prompts / ~7.5 GiB of the 40 GiB `cache-ram`

The message describes a **checkpoint rollback**, not a prompt-cache miss, and re-prefill is bounded by the last usable checkpoint. It does cost real latency when a large prompt shares only a short prefix (one observed dispatch took 113 s to first byte after re-prefilling ~19.6K tokens), but that is expected behaviour for this architecture: attribute slow first bytes to prompt divergence, not to a cache defect.

Background: `models.ini` documents the hybrid-arch caveats (`swa-full` is a no-op for this arch; `ctx-checkpoints = 32` at ~62.8 MiB each is the dominant per-prompt memory term).

## Testing

Run the full test suite through the test skill — `/skill:test` (equivalently
the cached `run_tests.py` runner) — which executes the suite in quiet mode
and caches results per git state (2-hour TTL, git-state fingerprint) so a
green run at an unchanged commit is reused instead of re-executed. Direct
`pytest`/`npm test` invocations bypass the cache and are reserved for
single-file or targeted runs.

### Git hooks (dev-push full-suite gate)

This clone's committed hooks must be activated once per clone:
`scripts/install_git_hooks.sh` (idempotent; sets `core.hooksPath=.githooks`
for the main checkout and all worktrees). Once active, `.githooks/pre-push`
runs the **full suite** on every push to `dev`/`main` and refuses a red push;
feature-branch pushes skip the suite. Verify with
`scripts/install_git_hooks.sh --check`. Details, bypass knobs
(`TEST_SCOPE_SKIP=1`) and a stdin based push simulation:
[docs/dev/git-hooks.md](docs/dev/git-hooks.md).
