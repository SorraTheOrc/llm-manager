## Global agent guidance

Read the global agent instructions at `~/.pi/agent/AGENTS.md` — they define the core principles, the Worklog (wl) work-item workflow, and the coding disciplines that apply to every project. That file is installed from the SorraAgents project (`~/projects/SorraAgents/AGENTS_GLOBAL.md`) by `~/projects/SorraAgents/scripts/install_pi.sh`, which symlinks it into place.

## Project-specific guidance

Follow the global AGENTS.md in addition to the rules below. The local rules below take priority in the event of a conflict.

### NEVER kill the LLM proxy

The LLM proxy runs as a Python process (`python3 -m uvicorn proxy.server:app`) alongside other Python processes (agent runners, audit runners, pi subprocesses). A blanket `killall`/`pkill` targeting `python3` or other broad process names WILL kill the proxy and take it offline for every consumer (local agents, herdr workers, remote LAN clients).

- **NEVER kill the proxy.** Do not run `killall`, `pkill`, `kill -9`, or any equivalent against `python3`, `uvicorn`, `proxy.server`, or broad process groups.
- To stop hung agent/audit subprocesses, kill the **specific PIDs only** — never a blanket process-name kill that could match the proxy.
- **If the proxy needs a restart, use the start-proxy skill:** `/skill:start-proxy --restart`. This performs the controlled teardown and relaunch of the proxy and its backends; do not kill processes manually and restart ad hoc.

## Testing

Run the full test suite through the test skill — `/skill:test` (equivalently
the cached `run_tests.py` runner) — which executes the suite in quiet mode
and caches results per git state (2-hour TTL, git-state fingerprint) so a
green run at an unchanged commit is reused instead of re-executed. Direct
`pytest`/`npm test` invocations bypass the cache and are reserved for
single-file or targeted runs.
