# llama-server SIGSEGV in `libggml-cpu.so.0.9.11` — diagnosis (LP-0MVZVZLCS000LQNB)

Status: **diagnosis complete, classification: llama.cpp build defect (CPU backend),
not a proxy defect**. No cores exist (`core_pattern=apport`, `ulimit -c 0`), so the
report separates what is *proven* from what is *consistent-but-unproven*.

The companion defect that turned each crash into a restart burst — the self-heal
loop orphaning its own host-first spawn — is fixed separately
(LP-0MVZT9L3N001XABB, commit `8ed4df3f`); the loop-bounding half of this item is
`restart_reason` attribution plus the no-respawn-while-the-port-is-held guard
(`proxy/proxy/backends/llama.py`).

## 1. Proven: the crashes

Six kernel segfaults on 2026-11-02, **all** `llama-server`, **all** in the same
shared object (`/var/log/kern.log`):

| Time (UTC) | pid | fault address | ip | error |
|---|---|---|---|---|
| 13:42:33 | 3932880 | `0x1e03` | `…b289a048` | 4 |
| 16:07:03 | 4002286 | `0x651b00000018` | `…c9a057` | 4 |
| 16:12:01 | 4009585 | `0x769e00000019` | `…c9a057` | 4 |
| 20:50:03 | 4071323 | `0x712b00000019` | `…e9a057` | 4 |
| 21:11:37 | 4092460 | `0x559200000018` | `…e9a057` | 4 |
| 21:38:42 | 4111384 | `0x6308000132ca` | `…69a048` | 4 |

`error 4` = user-mode **read** of a non-present page. The library is
`libggml-cpu.so.0.9.11` in every line — the ggml **CPU backend**, i.e. quantised
tensor compute, not the HIP/GPU backend and not proxy Python code.

Fault-address shape: five of the six are `<32-bit value>00000018` / `…00000019` —
a 64-bit value whose low half is the offset of a struct field (0x18/0x19). That is
the signature of a *corrupted or truncated pointer* being dereferenced, not of a
plain null dereference and not of an alignment fault (an alignment fault is a
`#GP`, logged as `general protection fault`, never as `segfault`).

Two distinct instruction offsets appear (`…048` and `…057`, 15 bytes apart), i.e.
two neighbouring instructions in the same routine.

## 2. Proven: the deployed CPU backend has no arch-specific quant kernels

The deployed artifacts (`/usr/local/bin/llama-server`, `/usr/local/lib64/libggml-cpu.so.0.9.11`,
both dated 2026-04-13) were inspected:

```
$ nm -C --defined-only /usr/local/lib64/libggml-cpu.so.0.9.11 | grep -c _generic
39
$ nm -C --defined-only /usr/local/lib64/libggml-cpu.so.0.9.11 \
    | grep ggml_vec_dot | grep -cE 'avx|neon|sve|rvv'
0
$ objdump -d --start-address=0x3fb30 --stop-address=0x4039f \
    /usr/local/lib64/libggml-cpu.so.0.9.11 | grep -cE 'zmm|ymm'
217
```

* **39** `_generic` (scalar reference) kernels, **0** arch-specialised
  (`ggml_vec_dot_*_avx2`, `…_avx512`, …) quant dot-product kernels.
* The generic reference kernels nevertheless contain **217** AVX/AVX-512
  instructions (`vpermt2b`, `vmovdqa64`, `vpdpwssd`, `vpmovsxbw`, …) — the
  compiler auto-vectorised the scalar reference loops.

So on this build **every quantised dot product runs compiler-vectorised generic
reference code**, not the hand-written, alignment-audited AVX2/AVX-512 kernels
upstream ships. That is a build/configuration defect class: the kernels that
llama.cpp maintains specifically to be correct on this hardware are absent.

`scripts/rebuild-llama.sh` configures with `-DGGML_HIP=ON` only (plus
`LLAMA_OPENSSL`), which is consistent with the missing CPU arch kernels; the
deployed artifacts predate the current script revision and may have been deployed
via `--deploy-only` from an upstream build. **A rebuild with the CPU arch flags
enabled is the candidate fix, and it is a follow-up item, not a silent scope
expansion of this one.**

## 3. Consistent, not proven: the failing code path

The kernel's printed `ip`, mapping start and file offset are mutually
inconsistent for every crash (e.g. `ip …b289a048` with mapping
`…b285a000+ff000` implies file offset `0x40048`, while the bracket prints
`51048`), so the exact instruction cannot be pinned from the log alone and no
core exists to pin it. Both candidate offsets land inside quantised
dot-product / elementwise kernels:

| candidate file offset | enclosing symbol |
|---|---|
| `0x40048`, `0x40057` | `ggml_vec_dot_q4_K_q8_K_generic` (`0x3fb30`, size `0x86f`) |
| `0x51048`, `0x51057` | `ggml_compute_forward_div+0x130` (`0x51030`) |

The **Q4_K** association is corroborated by the model set in play: the presets
that use Q4_K are `[Qwen3-MTP]` (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_S`) and
`[Qwen2.5-7B]` (`Q4_K_M`), while the production `[Qwen3]` preset is `Q5_K_M` and
the embedder is `Q8_0`. Combined with §2 (only generic kernels exist, and the
q4_K one is auto-vectorised), the failing path is most plausibly the
compiler-vectorised generic Q4_K dot product reading past a quant block.

This is recorded as the leading hypothesis, **not** as a proven instruction-level
root cause.

## 4. Ruled out

| Candidate | Evidence |
|---|---|
| Kernel OOM killer | `grep "Out of memory: Killed" /var/log/syslog` — 15 lines, **all at 12:03:08Z**; nothing after 20:00Z, and no OOM at any crash timestamp. |
| `host_pressure.py` SIGKILL | The module has no kill path (observe/protect only; the only `kill` references are docstrings). Its 20:37:17Z assessment: `ok — MemAvailable 121.4 GiB of 124.4 GiB`. |
| GPU VRAM exhaustion on autoload | No VRAM/OOM error in the `llama-server*.log` tails at the crash timestamps. |
| Proxy Python code | All six faults are inside `libggml-cpu.so.0.9.11`; the proxy is a separate process and none of its frames appear. |
| A docs-only commit (`5b9a32ab`) | Touched two docs files only; the churn predates it (first burst 20:50:02Z, segfaults from 13:42:33Z). |

Side observation (not this item's fix): the proxy cannot lower its own
`oom_score_adj` to `-900` (`PermissionError` — it is not started by the systemd
unit), so only the backend's `+800` half is applied. A future global OOM kills
the backend first and leaves the proxy to self-heal.

## 5. Bounded reproduction attempt

`scripts/diagnose_libggml_segfault.sh` is the harness:

* **evidence mode (default, read-only)** — collects the kernel segfault lines,
  the deployed artifact versions/dates, the symbol census from §2, the Q4_K
  presets in `models.ini`, and the last `self-heal` / `host-first` lines from
  `proxy.log`. It never sends a request and never starts a process.
* **`--replay` mode (destructive, opt-in)** — the documented bounded replay: it
  loads the Q4_K preset through the router and issues N bounded
  `POST /v1/chat/completions` requests with the recorded prompt shape (a
  ~16K-token filler prefix + a short instruction, `cache_prompt=false`,
  `max_tokens=8`), then diffs `kern.log` and the `llama-server` log tail. The
  exact command, prompt shape, preset (`Qwen3-MTP`, Q4_K_S) and build are fixed
  in the script, and it refuses to run without `--replay`.

**Why the replay was not executed here:** the host is the operator's production
proxy and a crash would abort the in-flight 24 h measurement window
(`LP-0MVVXGI490089OX5`, closes 2026-11-03 10:40:27Z) that a *different* open item
is measuring. The evidence above is sufficient to classify the fault (CPU backend
build defect) and to stop the churn; the replay is the confirmation step for the
follow-up rebuild item and must be scheduled by the operator.

## 6. What this item fixes

1. **Crash → burst amplification (fixed here):** `restart_reason` on every
   self-heal attempt, no respawn while the port is served by an unmanaged
   backend (the retry adopts instead), and abandoned prefills recorded as
   `backend_restart` evictions so a restart is visible in the eviction
   accounting.
2. **Orphan/port-conflict cascade (fixed in LP-0MVZT9L3N001XABB):** a spawn that
   outlives its readiness wait is adopted when reachable and terminated when
   not, so it can never hold the port against the next attempt.
3. **The crash itself (follow-up):** rebuild llama.cpp with the CPU arch kernels
   enabled (or pin a build that ships them), then re-run the bounded replay.

## 7. Regression coverage

`proxy/tests/test_restart_churn_bounding.py` — `restart_reason` on the attempt
line and in `backend_recovery_state`, the no-respawn-while-the-port-is-held
guard, the bounded retry after a *settled* failure, and the `backend_restart`
eviction accounting (including the "already served its first byte" guard).
`proxy/tests/test_self_heal_orphan_adoption.py` — the orphan-adoption/settle
behaviour from the prerequisite item.
