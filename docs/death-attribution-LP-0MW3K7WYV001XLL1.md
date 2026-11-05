# Attribution: who kills `llama-server` and the TTS server

Work item: **LP-0MW3K7WYV001XLL1** (parent **LP-0MW2RI9AM007O1FO**).
Supersedes the OOM hypothesis in the parent item for every death observed on
2026-11-04/05. Consumes the true-signal reporting and capture harness from
**LP-0MW3K7WD6003TJKZ** (`docs/death-signal-observability.md`).

All evidence below was collected **read-only** on the live host on
2026-11-05 between 13:56Z and 14:40Z. Every claim carries the command that
reproduces it.

---

## 1. Verdict

| Question | Answer |
|---|---|
| Which signal ends `llama-server`? | **`SIGKILL`**, every single death. Not `SIGSEGV`, not `SIGTERM`. |
| Which signal ends `tts-server`? | **`SIGKILL`** (same signal, same episodes). |
| Why does the proxy say `137`? | `start-llama.sh` runs under `set -euo pipefail`; a `SIGKILL`ed child makes the **wrapper** exit `128+9 = 137`, and the pre-observability watchdog printed that number undecoded. `137` is the *wrapper's* status, not the child's cause. |
| Are the kernel `SIGSEGV`/`GPF` lines the cause? | **No — they are a second, distinct death class.** They explain at most 16 of today's 48 death episodes, they name a *thread id* (not the supervised pid), and a `SIGKILL` never produces a kernel record at all. |
| Kernel OOM? | **Ruled out.** Last global OOM was `2026-11-02T12:03:08Z`. |
| memcg / cgroup OOM? | **Ruled out.** `memory.events` `oom_kill` stays at the 11-02 value during deaths. |
| `systemd-oomd`? | **Ruled out.** It has logged no kill since the 2026-10-31 boot. |
| The proxy's own `killpg` escalation? | **Ruled out.** The escalation logs; it never fired in the retained logs. |
| The launcher (`start-proxy.sh`)? | **Ruled out** for these deaths — a launcher restart also kills the *proxy* and leaves a rotated log; neither happened. |
| GPU-driver reset? | **Ruled out as the killer** — a driver/ROCm fault surfaces as `SIGSEGV`/`GPF` (class B below), never as `SIGKILL`. |
| So what kills them? | An **unprivileged userspace process** that `kill(2)`s the two GPU backends and not the proxy. It is **load-correlated** (see §6). The sending pid is not observable without root on this host (§5). |

Two independent death classes exist, and the fix children must address them
separately:

- **Class A — external `SIGKILL`** (majority): no kernel record, no OOM record,
  victim is whichever `llama-server` is currently the router (and, in the same
  episode, `tts-server`).
- **Class B — backend crash** (minority, ~1 in 3 episodes today): a thread hits
  `#GP`/`#PF` in `libggml-cpu.so.0.9.11` (or `libamdhip64.so.7.2.70204`), the
  kernel logs `traps:`/`segfault`, and *in the same episode* the supervised
  router is also `SIGKILL`ed — i.e. class B episodes are class A kills that
  happen to land on a process that just faulted.

---

## 2. Evidence base — how each fact was measured

| Instrument | What it gives | Artefact |
|---|---|---|
| True-signal death reporting (**deployed by this work item**) | `SIGKILL(9)/137` vs `SIGSEGV(11)/139` on every death line | `/var/log/llama-proxy/proxy.log` |
| `death_window` records | pre-death host state per death | `/var/log/llama-proxy/death_window_*.json` |
| Independent 2 Hz process witness | victim pid/pgid/ppid/rss, host state, kernel lines in the same interval | `/tmp/death_witness.jsonl` (script kept with this doc, see §9) |
| Kernel log | `traps:` / `segfault` / OOM lines | `/var/log/kern.log` |
| bash job-status line | which supervises what, and the raw job signal | `/var/log/llama-proxy/llama-server*.log` |
| cgroup accounting | OOM kills inside the service cgroup | `/sys/fs/cgroup/.../memory.events` |

The observability harness shipped **disarmed** (`server.death_capture_enabled:
false`), which is why the parent item found no `death_window` records even
though deaths were happening every few minutes. This change arms it.

---

## 3. Fact 1 — the death signal is `SIGKILL`

Deployed LP-0MW3K7WD6003TJKZ (`d66a76b18`) and captured the next two deaths:

```
2026-11-05 14:04:12,795 - ERROR - TTS watchdog: TTS server died: SIGKILL(9)/137 (restart_reason=tts_server_signal_SIGKILL)
2026-11-05 14:04:12,796 - ERROR - death_window label=tts_server cause=SIGKILL(9)/137 ...
2026-11-05 14:04:14,859 - ERROR - watchdog detected llama-server death: SIGKILL(9)/137 (restart_reason=llama_server_signal_SIGKILL) model=Qwen3
2026-11-05 14:04:14,860 - ERROR - death_window label=llama_server cause=SIGKILL(9)/137 ...
```

```bash
grep -E "signal_SIGKILL" /var/log/llama-proxy/proxy.log
jq '{restart_reason, death, summary}' /var/log/llama-proxy/death_window_llama_server_20261105T140414Z.json
```

Independent confirmation from bash, which is the *direct parent* of the router
(`start-llama.sh` line 273 is `"${LLAMA_CMD[@]}"`); every rotated
`llama-server*.log` ends with the same line, and never with
`Segmentation fault`:

```bash
grep -hE "Killed|Segmentation fault" /var/log/llama-proxy/llama-server*.log | sort | uniq -c
# 15 × "start-llama.sh: line 273: <pid> Killed"   (Segmentation fault: 0)
```

The witness independently recorded eight disappearances in that window with
**zero** kernel-log lines in the same interval — a `SIGKILL` is invisible in
`kern.log`, so "no SIGSEGV at the death time" was never exculpatory evidence.

The victim set is the **process tree**, not a single binary:

```
start-llama.sh (bash, pgid/sid = its own pid)
└── llama-server            <- ROUTER: the process bash supervises, dies by SIGKILL
    ├── llama-server --alias Qwen3 --port <ephemeral>      <- model worker
    └── llama-server --alias mxbai-embed --port <ephemeral> <- model worker
tts-server (exec'd from start-qwentts.sh, pgid/sid = its own pid)
```

Workers die 1–3 s *after* the router: they are `llama.cpp` router children that
exit on stdin `EOF` (`cmd_child_to_router:ready` /
"child server monitoring thread … waiting for EOF on stdin"). So a single
external `SIGKILL` of the router produces a *burst* of deaths, which is why the
event counts in the parent item overstate the number of killers.

---

## 4. Fact 2 — why the proxy reported `137` (AC1)

`start-llama.sh` line 2 is `set -euo pipefail`. The router is started as a
foreground job:

```bash
# start-llama.sh (router branch)
"${LLAMA_CMD[@]}"
exit 0
```

With `set -e`, a job killed by `SIGKILL` aborts the wrapper **with that job's
status**, i.e. the shell convention `128 + 9 = 137`; bash first prints
`...: <pid> Killed  "${LLAMA_CMD[@]}"` to the log it inherits. The `exit 0`
never runs.

The proxy's `Popen` handle is the **wrapper**, so:

- `llama_process.poll()` → `137`
- the pre-observability watchdog printed it raw: `llama_server_exit_code_137`
- `decode_returncode(137)` (LP-0MW3K7WD6003TJKZ) maps it to `SIGKILL` — which is
  what the deployed code now reports.

The masking is therefore *one wrapper exit status*, not a proxy bug: the proxy
faithfully reported its child's status, but `137` describes the wrapper's exit,
while the actual death cause of `llama-server` is only recoverable from the
signal encoded in it. A `SIGSEGV` in the child would have produced
`128 + 11 = 139` (and `Segmentation fault`) — never observed.

Why the kernel `SIGSEGV`/`GPF` lines looked contradictory: they name a **thread
id**, not the process pid of the router. Verified on a live router:

```bash
p=$(pgrep -f "llama-server --models-preset" | head -1)
echo "pid $p threads: $(ls /proc/$p/task | wc -l) max-tid $(ls /proc/$p/task | sort -n | tail -1)"
# pid 2081668 threads: 40  max-tid 2084568   (tid up to pid+2900)
```

So `kern.log`'s `llama-server[2001313]` and the supervised `2001224` are the same
or sibling processes in the router tree — comparing those two numbers directly
(the parent item's method) can never match.

---

## 5. Fact 3 — what is ruled out, with evidence

| Hypothesis | Evidence | Reproduce |
|---|---|---|
| Kernel global OOM | Last `Out of memory:` is `2026-11-02T12:03:08Z` (the LP-0MVZ810NG008A3EW incident). Nothing on 11-04/05. | `grep -iE "out of memory|oom-kill" /var/log/kern.log` |
| memcg/cgroup OOM | The service cgroup's `memory.events` `oom_kill` is frozen at `2` — the 11-02 global event — across every captured death. | `grep -E "^oom_kill " /sys/fs/cgroup/user.slice/user-1000.slice/session-7.scope/memory.events` |
| `systemd-oomd` | Active since 2026-10-31 14:28 with **no kill record** since (its only recorded kill is 2026-10-28 `node_exporter`). | `journalctl -u systemd-oomd --no-pager` |
| Proxy `_kill_process_group` | Every escalation logs `Killing process group for PID …` (`proxy/proxy/lifecycle.py:1296`); **zero** occurrences in the retained logs. A deliberate stop also logs `Stopping llama-server wrapper` — one occurrence, `2026-11-04 22:23:24` (a post-mortem stop). | `grep -c "Killing process group" /var/log/llama-proxy/proxy.log*` |
| `start-proxy.sh --restart` (launcher) | A launcher restart kills the *proxy* too and rotates `proxy.log` (`Rotated previous proxy log aside`). Only two rotations exist for today: `13:23:44` (an operator restart) and `14:03:36` (this work item's deploy). Neither coincides with a death episode. | `ls -t /var/log/llama-proxy/proxy.log*` |
| GPU-driver reset / ROCm fault | Real, but a *distinct* class (B): a driver fault is delivered as `SIGSEGV`(139)/`#GP`, and it is what `kern.log` shows. It cannot produce `SIGKILL`. | `grep -E "traps:|segfault" /var/log/kern.log` |
| Host memory pressure | At the captured kills `MemAvailable` was 94–120 GiB of 124 GiB and GTT 24–32 GiB of 128 GiB. | `jq .summary /var/log/llama-proxy/death_window_*.json` |
| Agent-issued blanket kill | No session under `~/.pi/agent/sessions` contains a `pkill`/`killall`/`fuser -k` **bash tool invocation** in the incident window; the only agent kill today was `kill 1112535` (one specific stale driver pid, 03:05Z). | JSONL scan of every `bash` tool call since 03:00Z (§9) |

### Why the sending pid is not in this record

The host blocks every unprivileged observation of signal delivery:

```bash
cat /proc/sys/kernel/yama/ptrace_scope        # 1  (attach only to descendants)
cat /proc/sys/kernel/perf_event_paranoid      # 4
cat /proc/sys/kernel/unprivileged_bpf_disabled  # 2  (unprivileged BPF fully off)
sudo -n true                                  # password required — no CAP_SYS_ADMIN/PTRACE
```

`SIGKILL` cannot be caught, the kernel logs nothing for it, `journald` on this
host retains only a handful of unrelated lines per hour (`journalctl --since
… | wc -l` = 16 lines across seven minutes), and the victims cannot be traced.
Naming the sender therefore needs one of: a root-run `bpftrace` probe on
`tracepoint:signal:signal_generate`, an `auditd` rule
(`-a always,exit -F arch=b64 -S kill -F a1=9`), or making the victim a
descendant of the tracer. §9 records the decoy experiment that was attempted
instead (argv[0] set to the real backend paths, `strace`-traced) — it survived
the whole observation window, so the kill is **not** a plain
`pkill -f llama-server`-style sweep of the binary name either; it carries the
victim's pid/port or an internal proxy handle.

---

## 6. Fact 4 — the kill is load-correlated, not OOM- or time-correlated

Deaths per hour vs proxy request volume (`Stream started` lines), from the
retained logs:

| hour (UTC) | streams | death lines |
|---|---|---|
| 11-04 23 | 1549 | 0 |
| 11-05 03 | 42 | 2 |
| 11-05 05 | **1690** | **19** |
| 11-05 06 | 84 | 4 |
| 11-05 07–09 | 168 / 103 / 78 | **0 / 0 / 0** |
| 11-05 10 | 2247 | 13 |
| 11-05 11 | **2738** | **26** |
| 11-05 12 | 1843 | 14 |

A four-hour quiet stretch (06:14 → 10:14) with the same binaries, the same
config, the same cgroup and no memory pressure produced **no** deaths. Whatever
sends the `SIGKILL` is triggered by proxy traffic, which points at a
request-path or load-guard action rather than a clock.

---

## 7. Fact 5 — a second, silent failure mode: undetected backend death

Separate from the killer, and equally fatal to a ≥24 h window: when the proxy
**adopts** a running backend at startup, it holds no `Popen` handle, so a later
backend death is invisible to the `poll()`-based watchdog. Observed live:

```
13:23:50 - Adopted running llama-server (model=router parallel=2 router_mode=True); preserving warm prompt cache
13:28:04 - <last line of llama-server.log: "line 273: 2036051 Killed">
   ... 35 minutes of nothing ...
13:59:48 - /health: {"llama_server_running": true, "backend_reachable": false,
                     "slots_surface_status": "stale", "llama_log_age_seconds": 1852}
```

No death line, no self-heal, no restart — the local route stayed dead from
13:28 until the deploy restart at 14:03 while `/health` still claimed the
backend was running and the traffic silently fell back to remote providers.

```bash
grep -cE "died|exit code=|self-heal|TTS watchdog" /var/log/llama-proxy/proxy.log   # 0 before the restart rotation
```

Filed as a discovered work item (see §10) — it is a *detection* defect, not a
killer, and it is on its own sufficient to void a measurement window.

---

## 8. Fix direction for the fix children (AC5, AC4)

1. **OOM hypothesis is refuted for 2026-11-04/05.** LP-0MW3K7XJ30027RV6
   (`oom_score_adj` protection) must be re-scoped: it is still a *latent*
   protection gap — measured on the live router, `oom_score_adj` is **`0`**, not
   the intended `800`, so `protect_backend_process()` is not reaching the
   router — but it is not the cause of these deaths and must not be presented
   as the fix for them.
2. **Class B (crashes) keeps its own children** (LP-0MW3K7Y3C003WEXW
   `libggml-cpu`, LP-0MW3K7YO1005C41X `libamdhip64`). The observed fault sites
   are deterministic: `libggml-cpu.so.0.9.11` offsets `0x51048`/`0x51051`/
   `0x51057` and `libamdhip64.so.7.2.70204` offset `0x2b931f`, 16–22 events/day.
3. **Class A (external `SIGKILL`) needs a new child**: identify and eliminate
   the sender (root-level observation, §5), and make the victim immune: run the
   router under its own service cgroup/user so a stray sweep cannot reach it,
   and/or restart it under `prctl(PR_SET_DUMPABLE)`-independent supervision.
4. **Detection must not depend on a `Popen` handle.** The adopted-backend path
   (§7) needs a liveness probe that is independent of the process handle, or
   adoption must be refused.
5. **The ≥24 h window child (LP-0MW3K7Z7T0094JK6) may only start after**
   class A is stopped; today's kill rate (48 episodes / 15.7 h in the retained
   logs) makes a 24 h window arithmetically impossible.

---

## 9. Reproducing the observation

Scripts used (kept out of the repo by design — they are throwaway probes, not
product code):

- `/tmp/death_witness.py` — 2 Hz sampled process-tree witness: emits
  `spawn`/`death` JSONL with pid, ppid, pgid, sid, RSS, `oom_score_adj`, host
  `MemAvailable`/GTT, cgroup `memory.events`, and the `kern.log` lines written
  in the same interval.
  `nohup python3 -u /tmp/death_witness.py > /tmp/death_witness.jsonl &`
- `/tmp/killer_hunt.py` — 25 Hz `/proc/loadavg`-diffed new-pid recorder (catches
  short-lived `kill`/`pkill`/`fuser` helpers).
- The decoy experiment: three `exec -a <real backend path> sleep 1800`
  processes under `strace -f -e trace=%signal`, to catch a name-pattern sweep
  and read `si_pid`. **Result: no decoy was ever signalled**, so the sender does
  not target the binary name/argv[0].

Root-only follow-ups (need operator `sudo`), in order of value:

```bash
# 1. who sends the signal
sudo bpftrace -e 'tracepoint:signal:signal_generate /args->sig == 9 /
  { printf("%s pid=%d -> %s pid=%d\n", comm, pid, args->comm, args->pid); }'

# 2. or the audit trail
sudo auditctl -a always,exit -F arch=b64 -S kill -F a1=9 -k llm_sigkill
```

Then re-read `/var/log/llama-proxy/death_window_*.json` (now armed) for the
pre-death state of each kill.

---

## 10. Limitations of this record

- **`proxy/config.yaml` line-citation ceiling.** `proxy/tests/test_slot_pool_sizing_doc.py`
  cites `proxy/config.yaml:1781/1441/1447`, and as of this commit the cited
  symbols sit *exactly* at the 60-line `CITATION_DRIFT_TOLERANCE`. Any insertion
  above line 1781 in that file reddens the suite. Arming `death_capture_enabled`
  is therefore a **one-line, zero-delta** edit; add future config commentary
  below line 1781 or refresh the citations in the sizing doc first.
- The **sender pid** for class A is not identified (no privileged observation
  available); the class, the victim set, the signal, the timing and every
  alternative hypothesis are established. This is stated as a limitation rather
  than papered over.- `death_window` records captured on 2026-11-05 14:04 carry only 1–2 samples
  because capture was disarmed at the time; they are armed as of this change,
  so subsequent deaths carry the full 120 s window.
- `journald` retention on this host is too thin to be used as evidence for
  absence of a `systemd-oomd` kill in a specific second; the oomd conclusion
  rests on its own unit log, which *does* record kills (2026-10-28 example).
