# Class A sender identified: the launcher's own port-kill fallback, run by its tests

Work item: **LP-0MW3PLEH8008O4I7** (parent **LP-0MW2RI9AM007O1FO**).
Builds on the class evidence in
[`death-attribution-LP-0MW3K7WYV001XLL1.md`](death-attribution-LP-0MW3K7WYV001XLL1.md)
(victim set, signal, ruled-out hypotheses, capture harness).

All evidence below was collected **read-only** on the live host on 2026-11-05
between 15:58Z and 16:13Z, except one **deliberate, bounded reproduction** at
16:08:59Z (see §3) that restarted the TTS backend once; it self-healed in 2.5 s.
Every claim carries the command that reproduces it.

---

## 1. Verdict

| Question | Answer |
|---|---|
| Who sends the `SIGKILL`? | **This repository's own `proxy/scripts/start-proxy.sh --restart`**, executed against the live host by **`tests/test_start_proxy_self_kill.py`**. |
| Which of its phases? | Phase 1–2 (`_kill_matching TERM/KILL 'llama-server' \| 'tts-server' \| 'qwentts' \| 'uvicorn proxy\.server'`) and **phase 3, the port fallback `fuser -k 8081/tcp` / `fuser -k 8080/tcp`**. |
| Why is phase 3 the killer? | The test sandboxes `ss`, `netstat`, `pgrep`, `python3` — but **not `fuser`**. Phases 1–2 select their victims through the stubbed `pgrep`, so they only ever signal the test's own `sleep` dummy. Phase 3 selects by **port**, which no `pgrep` stub can sandbox: `fuser -k` sends **`SIGKILL`** to whatever holds 8081 (the live `tts-server`) and, when `KEEP_LLAMA=0`, 8080 (the live `llama-server` router). |
| Why `SIGKILL` and not `SIGTERM`? | `fuser -k` sends `SIGKILL` by default. The death is therefore invisible in `kern.log`, uncatchable, and reported by the proxy as `SIGKILL(9)/137` — class A exactly as characterised. |
| Why does it look load-correlated? | The correlation in the attribution doc §6 is **agent/test activity**, not proxy traffic. Every dev-push runs the full suite; every suite run of this module kills the backends twice. The four quiet hours (06:14 → 10:14) are the hours when no suite ran. |
| Who is the sender, in process terms? | The sender is not one process but a short chain, all `uid=1000` (`rgardler`), all in `cgroup=/user.slice/user-1000.slice/session-7.scope`: the sandboxed `bash …/start-proxy.sh --foreground --restart …` (parent) → `xargs -r kill -s TERM/KILL` and `fuser -k 8081/tcp` (children). Captured in §3. |

---

## 2. Why this was hiding

Three coincidences kept the sender unnamed:

1. **The launcher is a legitimate production tool.** `start-proxy.sh --restart`
   is *supposed* to stop the backends. In its own sandbox the same code runs
   against the *host's* ports because the ports are hard-coded (`LLAMA_PORT=8080`,
   `TTS_PORT=8081`) and the port phase has no injectable override.
2. **The kill arrives from a 5–40 ms child.** A `kill`-by-name sweep can be
   caught by a name/argv decoy; a `fuser -k <port>` child cannot — the decoy
   experiment in the attribution doc §9 (`exec -a <backend path> sleep`) was
   never signalled because the decoy held no port.
3. **It never kills the proxy.** The sandbox stubs `python3` and `pgrep`, so
   `uvicorn proxy\.server` is either stubbed away or unreachable; the proxy's log
   stays unrotated and its own `killpg` escalations never fire. The evidence the
   attribution doc used to rule the launcher *out* (a launcher restart kills the
   proxy and rotates the log — neither happened) is exactly what a **sandboxed**
   launcher restart looks like.

---

## 3. Evidence — one real class-A death, attributed

### 3.1 The probe

`scripts/probe_sigkill_sender.py` (committed by this item) was armed at 200 Hz
with a 120 s lookback while the pre-fix form of the regression test was run once:

```bash
python3 scripts/probe_sigkill_sender.py --rate 200 --window 120 --duration 150 \
    --out-dir /tmp/sigkill-probe
# then, in another shell, the pre-fix test that performs a real --restart
python3 -m pytest tests/tmp_repro_prefix_self_kill.py::test_restart_with_pattern_in_reason_does_not_kill_the_launcher
```

Probe output (verbatim, one line per death):

```
probe: mode=proc rate=200.0Hz window=120.0s victims=['tts-server']
label=tts-server victim_pid=2179500 mode=proc window=120.0s senders=2 \
  comm=xargs pid=2184188 uid=1000 cgroup=/user.slice/user-1000.slice/session-7.scope \
  ppid=2184173 argv='xargs -r kill -s TERM' | \
  comm=fuser pid=2184306 uid=1000 cgroup=/user.slice/user-1000.slice/session-7.scope \
  ppid=2184173 argv='fuser -k 8081/tcp'
```

The same record names the parent of both children — the shell that decided to
kill by port:

```json
"senders": [
  {"argv": ["fuser", "-k", "8081/tcp"], "exe": "/usr/bin/fuser",
   "comm": "fuser", "pid": 2184306, "uid": 1000, "gid": 1000,
   "cgroup": "/user.slice/user-1000.slice/session-7.scope", "ppid": 2184173,
   "parent": {"comm": "bash", "pid": 2184173, "uid": 1000,
              "cgroup": "/user.slice/user-1000.slice/session-7.scope",
              "argv": ["bash",
                       "/tmp/pytest-of-rgardler/pytest-2876/test_restart_with_pattern_in_r0/proxy/scripts/start-proxy.sh",
                       "--foreground", "--restart", "--reason",
                       "operator mitigation forced llama-server restart",
                       "--port", "45541"]}}
]
```

### 3.2 The victim's own record, same episode

`proxy/proxy/death_capture.py` wrote the matching `death_window` record in the
same millisecond (probe `detected_at_unix = 1793894939.819`,
`death_window.detected_at_unix = 1793894939.819`):

```json
{"kind": "death_window", "label": "tts_server",
 "detected_at": "2026-11-05T16:08:59.818831+00:00",
 "detected_at_unix": 1793894939.819,
 "restart_reason": "tts_server_signal_SIGKILL",
 "death": {"code": 137, "describe": "SIGKILL(9)/137", "kind": "signal",
           "signal_name": "SIGKILL", "signal_number": 9},
 "sample_count": 113}
```

### 3.3 The proxy line and the recovery

```
2026-11-05 16:08:59,818 - ERROR - TTS watchdog: TTS server died: SIGKILL(9)/137 (restart_reason=tts_server_signal_SIGKILL)
2026-11-05 16:08:59,819 - ERROR - death_window label=tts_server cause=SIGKILL(9)/137 ...
2026-11-05 16:08:59,820 - WARNING - TTS self-heal attempt 1/3 started (window=120s, restart_reason=tts_server_signal_SIGKILL)
2026-11-05 16:09:00,350 - INFO - Started TTS server (PID: 2184333)
2026-11-05 16:09:02,357 - INFO - TTS self-heal succeeded on attempt 1/3
```

### 3.4 Cross-check, reproducibly

```bash
python3 scripts/probe_sigkill_sender.py --mode correlate \
  --death-window /var/log/llama-proxy/death_window_tts_server_20261105T160859Z.json \
  --events /tmp/sigkill-probe/sender_events.jsonl --correlate-window 10
# candidates=3
```

`kern.log` is **empty for this episode by construction**: a `SIGKILL` produces no
kernel record (attribution doc §3). That absence is the class signature, not a
gap in the evidence — the kernel-side proof requires the root probe in §5.

### 3.5 The same shape, before the capture

The 25 Hz pid witness armed by the attribution child (`/tmp/killer_hunt.jsonl`)
recorded the identical chain before each earlier death:

| death (UTC) | kill-like children observed | parent shell |
|---|---|---|
| 15:55:58 → 15:56:01 TTS | `xargs -r kill -s KILL` 2179228, `fuser -k 8081` 2179241 | `bash …/pytest-2869/test_restart_with_pattern_in_0/proxy/scripts/start-proxy.sh` (2179160) |
| 15:43:24, 15:43:36 TTS | `xargs -r kill -s KILL` 2170973 (+ sibling test) | `bash …/pytest-2862/test_pid_guard_protects_a_ren0/…` (2170878) |
| 15:13:10 → 15:13:14 TTS | `xargs -r kill -s KILL` 2154827, `fuser -k 8081` 2154868 | `bash …/pytest-2849/…` (2154779) |
| 14:44:31 → 14:44:34 TTS | `fuser -k 8081` 2120996, `xargs -r kill -s KILL` 2121173 | sandboxed launcher |

Two deaths ≈ 12 s apart = the module's two tests, one `--restart` each.

---

## 4. The fix

Three changes, all small and test-covered:

1. **`tests/test_start_proxy_self_kill.py`** — stub `fuser` (recording its argv)
   and run the sandbox with `START_PROXY_PORT_KILL_DISABLED=1`. The module's own
   docstring already claimed "the live proxy, llama-server, TTS server and ports
   8000/8080/8081 are never touched"; it now is true.
2. **`proxy/scripts/start-proxy.sh`** — phase 3 (the only phase that selects a
   victim by *port* rather than by a selected identity) is skipped, with a
   notice, when `START_PROXY_PORT_KILL_DISABLED=1`. Default behaviour is
   unchanged, so an operator restart still frees stale ports.
3. **Two behavioural tests** — `test_restart_never_reaches_the_live_port_kill_fallback`
   (guard set ⇒ `fuser` never executed) and
   `test_port_kill_fallback_still_runs_when_the_guard_is_unset` (guard unset ⇒ the
   fallback still targets `8080/tcp` and `8081/tcp`).

The invariant this restores: **a test never signals a process it did not start.**
`pgrep`-selected phases are sandboxable by a stub; a port-selected phase is not,
so it must be disabled — not merely "expected to find nothing".

### 4.1 Post-change window

Probe armed at 200 Hz for 70 s while the fixed module ran both `--restart` tests:

```
probe: mode=proc rate=200.0Hz window=120.0s victims=['tts-server']
# no death record, no sigkill_sender_*.json, exit code 2 (armed, nothing seen)
$ awk '$0 >= "2026-11-05 16:09:30"' /var/log/llama-proxy/proxy.log | grep -cE "died: SIGKILL|death_window label"
0
```

Then a 95 s window over the wider launcher suite
(`tests/test_start_proxy_launch_modes.py`, `tests/test_start_proxy_restart_signature.py`,
16 passed, 58 s):

```
events 31 real fuser 0       # 0 dispatches of /usr/bin/fuser; 4 stub invocations
deaths since 16:10:00: 0
```

For the same modules before the fix, each suite run produced two deaths (§3.5).

---

## 5. Using the probe

```bash
# unprivileged correlation (no root needed) — the mode used above
python3 scripts/probe_sigkill_sender.py --duration 1800 --out-dir /var/log/llama-proxy

# authoritative attribution: the sender of every SIGKILL, from the kernel
sudo python3 scripts/probe_sigkill_sender.py --mode bpftrace --duration 3600 \
    --out-dir /var/log/llama-proxy

# cross-check one captured death against the probe's event log
python3 scripts/probe_sigkill_sender.py --mode correlate \
    --death-window /var/log/llama-proxy/death_window_<label>_<stamp>.json \
    --events /var/log/llama-proxy/sender_events.jsonl
```

Bounded run, JSON records in the proxy's `death_window` envelope, exit `0` when
a death was captured, `2` when the probe ran and observed nothing. **Exit 2 means
"no evidence", never "no sender"**: leave the probe armed and re-run.

### 5.1 Privilege state on this host (2026-11-05)

```bash
cat /proc/sys/kernel/yama/ptrace_scope          # 1
cat /proc/sys/kernel/perf_event_paranoid        # 4
cat /proc/sys/kernel/unprivileged_bpf_disabled  # 2
sudo -n true                                    # sudo: a password is required
command -v bpftrace auditctl                    # /usr/bin/bpftrace ; auditctl missing
```

So: `bpftrace` is installed but needs operator `sudo`; **`auditd` is not
installed on this host**, so the attribution doc's §9 `auditctl` rule is not
available without a package install. Unprivileged `--mode proc` is what named
the sender here; `--mode bpftrace` is the documented, bounded command for the
kernel-side confirmation when an operator is at the keyboard.

---

## 6. Limitations

- `--mode proc` **correlates**: it names processes that were running and look
  like signal senders at the moment a victim disappeared. It cannot prove the
  signal came from them; that is why §3 quotes the victim's own `death_window`
  record, the proxy line and the parent's argv in the same millisecond, and why
  §5 keeps the root probe as the authoritative path.
- The death rate depends on how often the full suite runs. A cold day can
  produce no death at all in a probe window; per AC5 that is recorded, not
  closed on.
- **Live state note (2026-11-05 16:08Z):** `llama-server` was already down —
  the proxy reports `llama_server_running: true`, `backend_reachable: false`,
  `Router list models failed`, slots snapshot ~55 min old. That is the *adoption*
  detection defect of attribution §7 (a separate discovered item), not a killer:
  it is why `--mode proc` listed only `tts-server` as a live victim during the
  capture.

---

## 7. Reproducing everything in this document

```bash
# 1. the regression module alone, with the probe armed
python3 scripts/probe_sigkill_sender.py --rate 200 --window 120 --duration 90 \
    --out-dir /tmp/probe & sleep 2
python3 -m pytest tests/test_start_proxy_self_kill.py -q      # fixed: 0 deaths
# pre-fix (git show <pre-fix-sha>:tests/test_start_proxy_self_kill.py) : 2 deaths

# 2. correlate with the death the proxy recorded
ls -t /var/log/llama-proxy/death_window_tts_server_*.json | head -1
python3 scripts/probe_sigkill_sender.py --mode correlate --death-window <that file> \
    --events /tmp/probe/sender_events.jsonl

# 3. the unit tests for the probe's parsing/formatting
python3 -m pytest tests/test_probe_sigkill_sender.py -q
```
