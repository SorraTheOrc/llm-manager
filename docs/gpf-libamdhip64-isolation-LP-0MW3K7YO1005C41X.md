# Bound `libamdhip64` GPF: GPU-backend process-group isolation

Work item: **LP-0MW3K7YO1005C41X** (second half of parent AC3,
`LP-0MW2RI9AM007O1FO` — "why does TTS die too"). Supersedes the *fix* branch
of the hypothesis space for the `libamdhip64.so.7.2.70204`
general-protection-fault: the fault sits inside ROCm's own HIP runtime, so a
llama.cpp/qwentts rebuild cannot fix it — it is **bounded** instead by
process-group isolation between the two GPU backends.

Tags: `libamdhip64`, `GPF`, `isolation`, `ROCm`, `process-group`.

---

## 1. Verdict: bound, not fix

| Question | Answer |
|---|---|
| What faults? | A thread in llama-server raises `#GP` inside `libamdhip64.so.7.2.70204` at **one deterministic instruction** (26/26 records, 2026-11-02→11-05, identical offset). |
| Whose code is the fault site? | **ROCm's**, not llama.cpp's: the instruction is in the HIP kernel-launch path of `/opt/rocm-7.2.4/lib/libamdhip64.so`. |
| Can a rebuild fix it? | **No.** libamdhip64 ships from the ROCm release; there is no llama.cpp/qwentts flag that changes it (contrast with the `libggml-cpu` sibling, fixed by `-DGGML_CPU_REPACK=OFF`, `LP-0MW3K7Y3C003WEXW`). |
| Fix vs bound? | **Bound.** A pin/downgrade of ROCm or a kernel/driver change is a live-host change with regression risk and needs its own validation window (§5 follow-up). The bound: the fault kills only the faulting thread group (llama-server's tree); the TTS tree must never be in reach. |
| The named guard | **Process-group isolation** (`start_new_session=True` spawns + live disjoint-PGID assertion + fail-closed kill guard + `/health` verdict). |
| Can the GPF take down TTS? | **No, while isolation holds.** A userspace `#GP` is delivered by the kernel to the faulting thread group only; it does not reset the GPU or signal other processes. The cascade risk the guard closes is the *restart/stop* path of one backend signalling the other. |

## 2. Evidence

### 2.1 The fault is one instruction, every time

`/var/log/kern.log`, 26 records between 2026-11-02T13:01Z and 2026-11-05T14:32Z
(`general protection fault ... error:0` = non-canonical/null dereference, not an
alignment fault):

```text
2026-11-02T13:01:47Z traps: llama-server[3910058] general protection fault ip:78de458b931f ... in libamdhip64.so.7.2.70204[2b931f,...]
2026-11-05T05:24:02Z traps: llama-server[1367802] general protection fault ip:...4b931f     ... [2b931f,...]
2026-11-05T14:32:36Z traps: llama-server[2103242] general protection fault ip:7bc510ab931f ... [2b931f,...]
```

Resolving the file offset onto the mounted artifact (the bracket value is the
faulting file offset; `ip - mmap_base + LOAD-segment-offset` agrees):

```bash
addr2line -e /opt/rocm-7.2.4/lib/libamdhip64.so.7.2.70204 -f -C 0x2b931f
# hipRegisterTracerCallback (contains symbol; stripped build, see objdump)
objdump -d --start-address=0x2b92e0 --stop-address=0x2b9360 \
  /opt/rocm-7.2.4/lib/libamdhip64.so.7.2.70204
```

```asm
2b931a:  call   439630 <hipHccModuleLaunchKernel...>   ; launch helper
2b931f:  mov    0x38(%rax),%r9d                        ; <- fault: deref of %rax
```

The faulting instruction is `mov 0x38(%rax),%r9d` immediately after a call in
the HIP `hipHccModuleLaunchKernel` launch path: `%rax` (the helper's returned
handle) is a wild/truncated pointer. `error:0` and the identical
address-arithmetic every time ⇒ a deterministic ROCm runtime call path, not
memory pressure (attribute record:
[`death-attribution-LP-0MW3K7WYV001XLL1.md`](death-attribution-LP-0MW3K7WYV001XLL1.md)
§1 class B).

### 2.2 ROCm userland/kernel alignment (the "cannot fix here" basis)

| Component | Measured value | Command |
|---|---|---|
| Kernel | `7.0.0-38-generic` | `uname -r` |
| amdgpu kernel module | in-tree for `7.0.0-38` (`/lib/modules/7.0.0-38-generic/.../amdgpu.ko.zst`) | `modinfo amdgpu` |
| amdgpu userspace stack | `6.4.60402` (Ubuntu 24.04 `amdgpu-install` packages) | `dpkg -l \| grep amdgpu` |
| ROCm release | `7.2.4` | `rocm-smi --version` |
| `libamdhip64` artifact | `libamdhip64.so.7.2.70204`, built 2026-05-21 | `ls -la /opt/rocm-7.2.4/lib/libamdhip64.so*` |
| GPU | gfx1151 (Strix Halo APU, `1002:1586`) | `cat /sys/class/drm/card0/device/uevent` |

ROCm 7.2.4 userland mounted on a `7.0.0`-series kernel with a `6.4.x`-era
amdgpu userspace stack is a **userland/kernel version coupling** that a rebuild
of llama.cpp cannot align. Aligning it (pin/downgrade ROCm, or move the kernel
driver line) is a host-level change with regression risk — see §5.

### 2.3 The TTS server is a GPU process too

`qwentts.cpp/tts-server` is built with `cmake .. -DGGML_HIP=ON`
(`proxy/scripts/start-qwentts.sh` build hint), so it also maps
`libamdhip64` and shares the GPU. That is exactly why the "other GPU process"
must be unreachable by a llama-side fault/restart.

---

## 3. The protection (AC1 + AC3): process-group isolation, named and guarded

Both backend spawners already start their tree in a fresh session and process
group (`start_new_session=True` — `proxy/proxy/lifecycle.py`
`spawn_and_capture()` and `start_tts_server()`), so the two trees are disjoint
PGIDs by construction:

```
proxy (uvicorn)                    pgid=<proxy pid>
├── start-llama.sh (SID/PGID)      pgid=<llama wrapper pid>   <- llama tree
│   └── llama-server router + per-model workers  (inherit the PGID)
└── start-qwentts.sh (SID/PGID)    pgid=<tts wrapper pid>     <- TTS tree
    └── tts-server                 (inherits the PGID)
```

This item makes the invariant explicit and guarded:

1. **`proxy/backend_isolation.py` — `check_backend_isolation(srv)`**: resolves
   the live PGIDs/SIDs of both backend trees (tracked handles + port discovery
   for adopted llama) and reports `state: isolated | violation | unknown`. A
   `violation` means the two trees share a pid or a process group — the exact
   condition under which one backend's fault/restart would signal the other.
2. **`assert_backend_isolation(srv, logger)`**: the watchdog guard. Both
   watchdog loops (`_backend_watchdog_loop`, `_tts_watchdog_loop`) re-assert
   it (throttled to 30 s) and log a loud **ERROR** on a violation instead of
   a silent cascade. Fail-open on resolution errors.
3. **`_kill_process_group(..., forbidden_pids=...)`** fails closed: the stop
   paths pass the *other* backend's tree and the kill is **refused** (returns
   False, ERROR logged) if any forbidden pid is a member of the target PGID.
   `stop_llama_server` forbids the TTS tree; `stop_tts_server` forbids the
   llama tree. A cross-backend kill therefore becomes impossible even if the
   PGIDs ever collapse.
4. **`/health` → `backend_isolation`**: the verdict is observable without
   shell access (`curl -s localhost:8000/health | jq .backend_isolation`),
   fail-open.

The regression test is the AC2 proof — a real-process death of one tree leaves
the other untouched:

```bash
python3 -m pytest proxy/tests/test_backend_isolation.py -v
```

### 3.1 Exact verify commands (runbook)

```bash
# 1. Which PGID/SID each GPU backend lives in (must be two distinct PGIDs,
#    each equal to its own wrapper's pid — session leaders):
ps -eo pid,ppid,pgid,sid,comm,args | grep -E 'llama-server|tts-server|start-llama|start-qwentts'
#    or pointed:
ps -o pid,ppid,pgid,sid,cmd -p $(pgrep -f 'llama-server --models-preset' | head -1),$(pgrep -x tts-server | head -1)

# 2. Let the proxy assert it (the same check /health runs):
curl -s http://localhost:8000/health | jq .backend_isolation
#    => {"state":"isolated","violation":false,...}

# 3. If /health ever reports "violation": stop both backends manually via the
#    operator path (or `systemctl --user restart llama-proxy`), investigate why
#    the PGIDs collapsed before doing anything else — the guard refuses to
#    stop one backend while it shares a group with the other, by design.
```

### 3.2 Independent supervision (already present, part of the bound)

llama-server and TTS are supervised independently: `_backend_watchdog_loop` /
`_attempt_router_self_heal` for llama, `_tts_watchdog_loop` /
`_attempt_tts_self_heal` for TTS, each with its own attempt caps and health
probe. A llama GPF therefore triggers a llama-only restart; TTS is never
restarted by it. (Attribution note: the majority SIGKILL class A is a separate
killer, `LP-0MW3PLEH8008O4I7`, and is not addressed by this isolation — see
§5.)

---

## 4. The regression test (AC2)

`proxy/tests/test_backend_isolation.py` (18 tests):

- **End-to-end, real processes** (the spawners' exact shape —
  `start_new_session=True` trees): spawned trees get disjoint PGIDs;
  `check_backend_isolation` reports `isolated`; killing one tree's process
  group **leaves the other tree alive**; a kill with a forbidden pid in the
  target group is **refused without signalling anything**.
- **Kill-guard unit tests**: refusal on a shared PGID, proceed on disjoint
  PGIDs, fail-open on an unresolvable (dead) forbidden pid, unchanged
  behaviour with no forbidden pids.
- **Check unit tests**: `unknown` when a backend is absent, `violation` on a
  shared pid and on a shared PGID, adopted llama resolved by port.
- **Guard unit tests**: `assert_backend_isolation` logs ERROR on violation,
  stays quiet when isolated, never raises.
- **/health**: `backend_isolation` field reports isolated/violation and is
  fail-open (`unknown`) when the check itself fails
  (`proxy/tests/test_tts_health_endpoint.py`).

---

## 5. Recorded gap and follow-up (AC4)

**Gap.** The `libamdhip64` GPF itself is not fixed here: it is a ROCm
userland/kernel coupling (ROCm 7.2.4 userland on kernel `7.0.0-38-generic`
with a `6.4.x` amdgpu userspace stack; §2.2). Within this repository, the
maximal mitigation is the bounded crash: llama-server's tree dies and
self-heals, TTS and the proxy continue, and no kill path can cascade.

**Mitigation (shipped here).** Process-group isolation + the fail-closed kill
guard + independent supervision, above. The GPF becomes a llama-only restart
event, invisible to TTS.

**Follow-up (needs the operator + a validation window).**

1. **Align the ROCm line.** Pin/downgrade ROCm (e.g. to the release whose
   `libamdhip64` matches the installed kernel/amdgpu line) or move the kernel
   driver line to match ROCm 7.2.4; then validate a ≥24 h window with zero
   `kern.log` `libamdhip64` lines.
2. **Report upstream.** The fault is a deterministic ROCm runtime call path
   (`hipHccModuleLaunchKernel` region, `libamdhip64.so.7.2.70204`, gfx1151);
   a ROCm issue including the §2.1 decode would be actionable for AMD.
3. **Class A (external SIGKILL) is separate.** The majority death class
   (`SIGKILL` from `start-proxy.sh --restart` phase 3, LP-0MW3PLEH8008O4I7)
   is not a GPU fault and is not bounded by this isolation; the ≥24 h window
   child (LP-0MW3K7Z7T0094JK6) may only start once class A is stopped (see
   `death-attribution-LP-0MW3K7WYV001XLL1.md` §8).

## 6. Where this leaves the ≥24 h window (parent AC3)

- `libggml-cpu` repack crash → fixed by rebuild flag (`-DGGML_CPU_REPACK=OFF`,
  LP-0MW3K7Y3C003WEXW).
- `libamdhip64` GPF → **bounded** here: deterministic llama-only crash with
  independent supervision and a fail-closed isolation guard; TTS cannot be
  signalled by it.
- Class A SIGKILL → remains (LP-0MW3PLEH8008O4I7, `LP-0MW3PLK5M002YPOA`);
  a 24 h window is still blocked until class A is eliminated.
