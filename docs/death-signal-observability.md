# Death-signal observability: true cause + death-window capture

Work item: **LP-0MW3K7WD6003TJKZ** (parent **LP-0MW2RI9AM007O1FO**).

## Why

The proxy used to report a blanket `llama_server_exit_code_137` for every
backend death — 133 times across the retained log rotations — while
`/var/log/kern.log` recorded `SIGSEGV` / general-protection faults for the same
processes, and the TTS watchdog only logged `code=-9`.

One integer hid two different facts:

- **which signal** ended the process — `137` is `128 + SIGKILL`, but the same
  process can die of `SIGSEGV` (`139`) or `SIGTERM` (`143`) instead;
- **who delivered it** — the proxy's own process-group `SIGKILL` during a
  restart looks identical to an external kill once it is serialised into
  `137`.

Nothing on the host retained the state *before* the death, so the attribution
work had no evidence to work with.

## The death record

Every time the backend or TTS watchdog detects a death it now:

1. decodes the termination cause into a `restart_reason`;
2. takes one final sample of the host state;
3. writes a `death_window` record to the proxy log directory
   (`death_window_<label>_<UTC timestamp>.json`);
4. stores the record in the service's recovery state, exposed on `/health`:
   `backend_recovery_state["last_death"]` for llama-server,
   `tts_recovery_state["last_death"]` for TTS.

### `restart_reason` vocabulary

| Cause | `restart_reason` |
|---|---|
| killed by `SIGSEGV` | `llama_server_signal_SIGSEGV` |
| killed by `SIGKILL` | `llama_server_signal_SIGKILL` |
| killed by `SIGTERM` | `llama_server_signal_SIGTERM` |
| normal exit status 3 | `llama_server_exit_code_3` |
| undecodable status | `llama_server_unknown_exit` |

TTS uses the same vocabulary with the `tts_server` label
(`tts_server_signal_SIGKILL`, …). The reason is logged on every watchdog and
self-heal line, so a restart is never anonymous.

### How the cause is decoded

`proxy/death_capture.py` provides two decoders:

- `decode_wait_status(status)` — a **raw POSIX wait status**
  (`os.WIFSIGNALED`/`os.WTERMSIG`). Use this for anything `wait()`-shaped; a
  synthetic `SIGSEGV` status decodes to `SIGSEGV(11)/139`.
- `decode_returncode(code)` — a `subprocess.Popen` return code: `None` while
  alive, negative for a signal death, and the shell `128 + N` convention for a
  wrapper (`137` → `SIGKILL`, `139` → `SIGSEGV`, `143` → `SIGTERM`).

`probe_death(proc)` prefers the **raw** status read with `waitid(WNOWAIT)`
*before* the `Popen` handle reaps the child. That distinction matters here:
`start-llama.sh` runs `llama-server` as a foreground child, so when the wrapper
itself is killed by the proxy's process-group `SIGKILL` the raw status says
`CLD_KILLED`/`SIGKILL` (self-inflicted), while a wrapper that exits `137`
normally means *its child* was killed by `SIGKILL`. The two are no longer
indistinguishable.

> **Documented ambiguity.** A process that deliberately calls `exit(137)` is
> indistinguishable from one killed by `SIGKILL` at the `Popen` return-code
> layer; `decode_returncode` therefore reports the shell convention (a signal
> death). The raw-status path has no such ambiguity.

### `death_window` record format

```json
{
  "kind": "death_window",
  "label": "llama_server",
  "detected_at": "2026-11-05T13:40:02.114000+00:00",
  "detected_at_unix": 1762340402.114,
  "window_seconds": 120.0,
  "restart_reason": "llama_server_signal_SIGSEGV",
  "death": {
    "kind": "signal",
    "code": 139,
    "signal_name": "SIGSEGV",
    "signal_number": 11,
    "describe": "SIGSEGV(11)/139"
  },
  "summary": {
    "gtt_used_max_bytes": 37480960000,
    "mem_available_min_bytes": 86973087744,
    "process_rss_max_bytes": {"llama_server": 18446744073, "proxy": 3972844748},
    "oom_score_adj": {"llama_server": 800, "proxy": -900},
    "cgroup_memory_events": {"oom": 0, "oom_kill": 0, "max": 12}
  },
  "sample_count": 118,
  "samples": [
    {
      "timestamp": 1762340282.114,
      "mem_total_bytes": 133169152000,
      "mem_available_bytes": 86973087744,
      "gtt_used_bytes": 37480960000,
      "gtt_total_bytes": 147000000000,
      "processes": {
        "llama_server": {"pid": 4242, "rss_bytes": 18446744073, "oom_score_adj": 800}
      },
      "cgroup_memory_events": {"oom": 0, "oom_kill": 0}
    }
  ],
  "dump_path": "/var/log/llama-proxy/death_window_llama_server_20261105T134002Z.json"
}
```

Field notes:

- `summary` holds the extremes across the window (peak GTT, lowest
  `MemAvailable`, peak per-process RSS, the last observed `oom_score_adj` and
  the final cgroup `memory.events` counters) — read these first.
- `samples` is capped (default 60) as well as time-bounded, so a fast sampler
  cannot flood the record.
- `cgroup_memory_events` is the cgroup's own kill accounting: a death with
  `oom_kill` unchanged is **not** an OOM kill.

## Arming the capture helper

Capture is off by default (it costs one `/proc` + sysfs read per second).
Enable it in `proxy/config.yaml` under `server:`:

```yaml
server:
  death_capture_enabled: true          # arm the ring buffer
  death_capture_interval_seconds: 1    # sampling interval
  death_capture_window_seconds: 120    # how far back a death record reaches
  death_capture_capacity: 240          # hard sample cap
```

Restart the proxy (`/skill:start-proxy --restart`) and confirm it armed:

```bash
grep "death capture" /var/log/llama-proxy/proxy.log
# death capture: armed (interval=1.0s, window=120.0s, capacity=240, dir=...)
# death capture: not armed (set server.death_capture_enabled=true ...)
```

After the next death, read the record:

```bash
ls -t /var/log/llama-proxy/death_window_*.json | head -1
jq '{restart_reason, death, summary}' "$(ls -t /var/log/llama-proxy/death_window_*.json | head -1)"
```

The record is also summarised in one log line so it is visible without opening
the file:

```text
death_window label=llama_server cause=SIGSEGV(11)/139 restart_reason=llama_server_signal_SIGSEGV window=120.0s samples=118 gtt_max=34.9GiB mem_available_min=81.0GiB cgroup_oom_kills=0 dump=/var/log/llama-proxy/death_window_llama_server_...json
```

## Interpreting a death

| Observation | Reading |
|---|---|
| `death.kind="signal"`, `SIGKILL`, `cgroup_oom_kills` unchanged | an external or self-inflicted `SIGKILL`, **not** an OOM kill |
| `death.kind="signal"`, `SIGSEGV` | a crash inside the process (see `docs/segfault-libggml-cpu-LP-0MVZVZLCS000LQNB.md`) |
| `death.kind="exit"`, `code=137` | the wrapper exited normally with `137` — its child was `SIGKILL`ed (shell convention), the wrapper itself was not signalled |
| `summary.mem_available_min_bytes` collapsing towards `MemTotal` before the death | host memory pressure (see `docs/` OOM notes) |
| `summary.gtt_used_max_bytes` near physical RAM | GPU-backed (amdgpu GTT) exhaustion |
| `oom_score_adj` lower than intended | OOM protection was not applied to that process |

## Tests

- `proxy/tests/test_death_capture.py` — the decoders (synthetic
  `SIGSEGV`/`SIGKILL`/`SIGTERM`/normal-exit statuses), the raw-status peek, and
  the ring/window/record/dump behaviour with synthetic samples.
- `proxy/tests/test_death_signal_observability.py` — the watchdog wiring: a
  llama `SIGSEGV` is logged/stored as `SIGSEGV(11)/139` (never `exit code=137`),
  and a TTS death carries its `restart_reason` into the log, the recovery state
  and the self-heal path.
