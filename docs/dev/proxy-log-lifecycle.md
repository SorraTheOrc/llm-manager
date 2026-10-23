# Proxy log lifecycle — restart safety, single ownership, coverage

Owner: LP-0MVLHMAWV008EKTI (child of LP-0MVLGNUDJ009NRX2).
Key files: `proxy/proxy/utils.py` (`setup_logging`, `archive_live_log`,
`prune_logs`), `proxy/proxy/single_instance.py`, `proxy/proxy/server.py`
(startup guard), `proxy/scripts/start-proxy.sh`,
`proxy/scripts/reap-stale-proxy.sh`, `scripts/lib/proxy_logs.py`,
`scripts/prefill_recovery_verification.py`.

## What went wrong on 2026-10-23

The post-fix verification window could not be measured because
`/var/log/llama-proxy` had a **3 h 35 m hole**:

| File | First timestamp | Last timestamp |
|---|---|---|
| `proxy.log.2026-10-23_12` | 2026-10-23 12:29:16 | 2026-10-23 18:29:12 |
| *(missing)* | — | — |
| `proxy.log` | 2026-10-23 22:04:16 | live |

`proxy.log`'s birth time was 18:29:18, but its first line was the **22:04:16
startup line** — the interval in between (including the fix go-live at
20:08:53) was gone from every file, plain or `.gz`.

Two contributing defects were observed together:

1. **The live window did not survive a restart.** Append mode is the default
   for `TimedRotatingFileHandler`, but append alone does not guarantee the
   window: a restart (or another process touching the same path) left
   `proxy.log` starting at the new process's first line.
2. **Two `uvicorn proxy.server:app` processes were alive at once**, both with
   `proxy.log` open for write. The older one (started 21:33:34) owned **no
   listener** — it was a wedged twin that no port check could ever catch, yet
   it could still rotate/truncate the log the live process was writing.
   `TimedRotatingFileHandler.doRollover` also `os.remove()`s an existing
   same-named rotated file, so a rotation from the wrong process can delete a
   window outright.

A third defect was in the measurement: `--measure` reported on whatever lines
it happened to parse, so a missing file boundary looked like a normal window.

## 1. A restart archives the live log (never truncates it)

`setup_logging` calls `archive_live_log()` **before** the file handler opens
`proxy.log`:

- a non-empty live file is renamed to `proxy.log.YYYY-MM-DD_HHMMSS`, named for
  its own last write, so the window is explicit and discoverable by the same
  `proxy.log.` prefix rules as every other rotated log;
- the new process always starts a fresh `proxy.log` and logs
  `Rotated previous proxy log aside: <name> (pre-restart window preserved)`,
  so restarts are attributable from the log itself;
- an existing rotated file is **never overwritten** — a name collision adds a
  `.1`, `.2`, … suffix instead of deleting either window;
- an empty (or missing) live file is left alone: restarts without traffic
  create no empty archives;
- a previous writer that is still alive keeps writing to the *archived inode*
  instead of interleaving output with the new process.

Startup archives are pruned and compressed by `prune_logs()` exactly like
in-process rotations (both `%Y-%m-%d_%H` and `%Y-%m-%d_%H%M%S` names are
understood, with or without a collision suffix and `.gz`).

Net effect: after `bash scripts/start-proxy.sh --restart`, the previous process
interval is always present as its own file, and a window spanning the restart
parses lines on both sides with no gap.

## 2. Exactly one proxy owns the log directory

Two independent guards:

- **In-process `flock`** (`proxy/single_instance.py`, acquired in
  `_startup_config_logging`, released at shutdown): the first proxy to start
  holds `<log_dir>/proxy.lock` for its whole lifetime and records its pid. A
  second instance **fails fast** with
  `Another LLama Proxy instance is already running (holder pid N, …)` instead
  of becoming a second writer. The kernel releases the lock when the holder
  exits (including `SIGKILL`), so a crash never leaves a stale lock. Set
  `LLAMA_PROXY_SKIP_INSTANCE_LOCK=1` for isolated/dev runs that intentionally
  share a log directory.
- **Stale-twin reaping** (`proxy/scripts/reap-stale-proxy.sh`, called by
  `start-proxy.sh` on every non-`--restart` launch): a process shaped like
  `uvicorn proxy.server … --port N` that is **alive but holds no listener** is
  a wedged twin. The launch-time port check has already refused when a healthy
  proxy owns the port, so any survivor at that point is stale: it is
  `SIGTERM`ed (grace 5 s) and `SIGKILL`ed if needed. Matching is per port, so a
  dev instance on another port is untouched; `--dry-run` reports without
  killing.

`ps` after any start/restart therefore shows a single `proxy.server:app`
process, and a second instance that cannot bind exits instead of lingering.

## 3. Coverage is part of the measurement verdict

`scripts/lib/proxy_logs.py::measure_log_coverage()` computes, per discovered
file, the first/last timestamp of the lines inside the requested window and
reports an **adjacent-file gap** when the distance between the previous file's
last line and the next file's first line exceeds a threshold.

`scripts/prefill_recovery_verification.py --measure` includes this in its
report and in `all_verdicts_met`:

```bash
python3 scripts/prefill_recovery_verification.py --measure \
  --since "2026-10-23 22:04:14" --until "2026-10-24 22:04:14" --json
```

- `--max-log-gap-seconds` (default `3600`) is the bound: a gap larger than one
  hour inside the window fails the verdict.
- A gap is reported with both file names and the exact gap in seconds, so the
  lost window is named rather than silently averaged away.
- Overlapping files (logrotate `copytruncate` duplicates a tail) are not gaps;
  ordinary idle time *inside* one file is not a gap either — only a file
  boundary that skipped time is.
- `window_start_gap_seconds` / `window_end_gap_seconds` are informational (the
  lead-in before the first retained line, and the distance to a still-live
  window end); they do not fail the verdict.

## Verifying a change to this area

```bash
# targeted
python3 -m pytest proxy/tests/test_restart_safe_logging.py \
                 proxy/tests/test_single_instance.py \
                 tests/test_proxy_log_coverage.py \
                 tests/test_reap_stale_proxy_shell.py -q

# live: one restart, then measure a window spanning it
cd proxy && bash scripts/start-proxy.sh --restart
cd .. && python3 scripts/prefill_recovery_verification.py --measure \
  --since "$(date -d '-2 hours' '+%Y-%m-%d %H:%M:%S')" --json
```

Expected: the report lists the archived pre-restart file and the new
`proxy.log` as two spans, `gaps` is empty, and `coverage.verdict_met` is
`true`.
