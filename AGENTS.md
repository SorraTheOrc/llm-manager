## Global agent guidance

Read the global agent instructions at `~/.pi/agent/AGENTS.md` — they define the core principles, the Worklog (wl) work-item workflow, and the coding disciplines that apply to every project. That file is installed from the SorraAgents project (`~/projects/SorraAgents/AGENTS_GLOBAL.md`) by `~/projects/SorraAgents/scripts/install_pi.sh`, which symlinks it into place.

## Project-specific guidance

Follow the global AGENTS.md in addition to the rules below. The local rules below take priority in the event of a conflict.

## Testing

Run the full test suite through the test skill — `/skill:test` (equivalently
the cached `run_tests.py` runner) — which executes the suite in quiet mode
and caches results per git state (2-hour TTL, git-state fingerprint) so a
green run at an unchanged commit is reused instead of re-executed. Direct
`pytest`/`npm test` invocations bypass the cache and are reserved for
single-file or targeted runs.
