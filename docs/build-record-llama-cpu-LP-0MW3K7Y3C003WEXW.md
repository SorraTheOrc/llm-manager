# Build record — llama.cpp CPU backend without the repack crash path

Work item: **LP-0MW3K7Y3C003WEXW** (first half of parent AC3
`LP-0MW2RI9AM007O1FO`). Supersedes the "rebuild with arch-specific kernels"
hypothesis recorded in
[`segfault-libggml-cpu-LP-0MVZVZLCS000LQNB.md`](segfault-libggml-cpu-LP-0MVZVZLCS000LQNB.md) §2–§3.

Status: **build configuration and artifact gate shipped; the deployed artifact
is unchanged until the coordinated restart** (see §7). Nothing in this record
restarts, stops or re-anchors the live proxy.

## 1. Root cause (re-derived from the running artifact)

The kernel's fault lines name the file offset inside the shared object, so the
offset can be resolved against the artifact the process actually maps:

```text
2026-11-05T12:23:06 traps: llama-server[1960312] general protection fault
                     ip:...09a057 ... in libggml-cpu.so.0.9.11[51057,...]
2026-11-05T13:09:22 traps: llama-server[2001313] general protection fault
                     ip:...29a051 ... in libggml-cpu.so.0.9.11[51051,...]
2026-11-05T11:10:43 llama-server[1757796]: segfault at 5855000021b5 ip ... a048
                     ... in libggml-cpu.so.0.9.11[51048,...]
```

`scripts/diagnose_libggml_segfault.sh` now resolves those offsets onto symbols
directly (`--locate-offset`), against the library named in
`/proc/<pid>/maps`:

```text
offset 0x51048 -> ggml_cpu_extra_work_size +0x38
offset 0x51051 -> ggml_cpu_extra_work_size +0x41
offset 0x51057 -> ggml_cpu_extra_work_size +0x47
```

`objdump -d --start-address=0x51010` on that library shows what those three
instructions are:

```asm
0000000000051010 <ggml_cpu_extra_work_size>:
   51040:  mov    (%rbx),%rax          ; registered extra buffer type
   51043:  test   %rax,%rax
   51046:  je     51075
   51048:  mov    0x38(%rax),%rdi      ; <-- fault: extra->context
   5104c:  test   %rdi,%rdi
   5104f:  je     51075
   51051:  mov    (%rdi),%rax          ; <-- fault: tensor_traits vptr
   51054:  mov    %r12,%rsi
   51057:  call   *0x18(%rax)          ; <-- fault: virtual work_size()
```

All three fault sites are the **CPU repack extra-buffer-type dispatch**
(`ggml/src/ggml-cpu/traits.cpp:25`), not a quantisation kernel. What the
observed fault addresses say:

| Site | Fault address observed | Reading |
|---|---|---|
| `+0x38` | `5855000021b5` | the buffer-type pointer read from the extra-type vector is not a live object |
| `+0x41` | GPF (non-canonical) | `op->src[0]->extra` is not a live `tensor_traits` |
| `+0x47` | `18` (`%rax` = 0), `797300000019`, `0x5f7800000018` (`%rax` = `<low32>00000000`) | the vptr read out of that object is null or a truncated pointer |

`0x51057` calls through `*(tensor_traits*)+0x18`; upstream
`ggml::cpu::repack::extra_buffer_type::get_tensor_traits()` returns exactly
`op->src[0]->extra`, and only when the source tensor's buffer type is the
repack buffer type (`repack.cpp:4810`). So the fault requires a tensor living in
a **repack** buffer whose `extra` is not a live `tensor_traits` — a lifetime /
initialisation defect in the repack path, not a numerical one. The census
cannot and does not pin which of the two (stale object / unset `extra`) holds;
both are removed by the same build change (§3).

## 2. The earlier census was wrong about this build

The previous conclusion ("39 `_generic` kernels, 0 arch-specific → rebuild with
arch flags") does not hold for the artifact that crashes:

```text
$ scripts/diagnose_libggml_segfault.sh          # census of the RUNNING artifact
artifact:                /home/rgardler/llama.cpp/build/bin/libggml-cpu.so.0.9.11
generic kernels:         32
arch kernels:            27
arch-specific codegen:   yes
  - ggml_vec_dot_q4_K_q8_K: present=True vector_insns=61 generic_tail_call=False
  - ggml_vec_dot_q6_K_q8_K: present=True vector_insns=90 generic_tail_call=False
  - ggml_vec_dot_q4_0_q8_0: present=True vector_insns=14 generic_tail_call=False
CPU repack registered:   YES (any expected); repack code linked: yes
verdict:                 OK
```

Two independent mistakes produced the false reading:

1. **Wrong artifact.** The census inspected `/usr/local/lib64/libggml-cpu.so.0.9.11`,
   while every running server maps `/home/rgardler/llama.cpp/build/bin/…` (see
   the `artifacts of the RUNNING server` section of the harness). The harness
   now resolves both paths from `/proc/<pid>/maps`.
2. **Wrong metric.** Upstream llama.cpp **always** ships the `_generic`
   fallback symbols and never suffixes the arch kernel with `_avx*` — the
   optimised kernel is plain `ggml_vec_dot_q4_K_q8_K`. Counting `_generic`
   symbols therefore cannot distinguish a good build from a bad one, and
   counting `*avx*` names always returns 0. The metric now counts `ymm`/`zmm`
   instructions inside the hot kernel bodies (`scripts/lib/ggml_cpu_census.py`):
   the live build shows 61/90/14 vector instructions, i.e. `-march=native`
   (GGML_NATIVE=ON) is working; a build whose arch entry point merely tail-calls
   `_generic` fails the new gate.

The deployed build's flags, for the record
(`/home/rgardler/llama.cpp/build/ggml/src/CMakeFiles/ggml-cpu.dir/flags.make`):

```text
C_FLAGS   = -O3 -DNDEBUG … -march=native -fopenmp
C_DEFINES = … -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP …
```

Both facts point the same way: the CPU kernels were built correctly, and the
repack extra-buffer path (`-DGGML_USE_CPU_REPACK`) was compiled in.

## 3. The fix

`GGML_CPU_REPACK` is what **registers** the extra buffer type. With it off, the
`#ifdef GGML_USE_CPU_REPACK` block in `ggml_backend_cpu_get_extra_buffer_types()`
(`ggml/src/ggml-cpu/ggml-cpu.cpp:64`) is compiled out, the vector it returns is
**empty**, and the loop body in `ggml_cpu_extra_work_size` — the three faulting
instructions — is never entered. The build therefore:

* sets **`-DGGML_CPU_REPACK=OFF`** (default in `scripts/rebuild-llama.sh`;
  `--with-cpu-repack` restores upstream's default ON for an experiment), and
* sets **`-DGGML_NATIVE=ON`** explicitly, so the arch-kernel half can never be
  silently lost by an ambient cache or a toolchain change.

**Important subtlety — the repack *code* stays linked.** `repack.cpp` is added to
the CPU backend unconditionally (`ggml/src/ggml-cpu/CMakeLists.txt:32`); the flag
only removes the registration and the `REPACK` backend feature. So a census that
looked for repack *symbols* would reject the fixed artifact. The census therefore
reads the one thing that decides whether the crash can happen — the body of
`ggml_backend_cpu_get_extra_buffer_types()`:

```text
# live (GGML_CPU_REPACK=ON)      # staged (GGML_CPU_REPACK=OFF)
call __cxa_guard_acquire@plt     call __cxa_guard_acquire@plt
call ggml_backend_cpu_repack_buffer_type()@plt   <-- registers the crash path
call ggml_backend_cpu_repack_buffer_type()@plt
call __cxa_atexit@plt            call __cxa_atexit@plt
```

Every artifact the script would deploy is censused first
(`scripts/lib/ggml_cpu_census.py`), and the deploy aborts **before** the live
deploy dir is touched when either fact fails — no arch-specific codegen, or the
repack path still registered. That gate is not decorative: it rejected the first
`GGML_CPU_REPACK=OFF` build in this session because the *symbol-presence* metric
was wrong, which is how the registration-based metric above was found.

**Cost of disabling repack.** Repack converts Q4_0/Q4_K weights to a CPU-friendly
layout to speed up CPU matrix multiplication. This host runs the production
preset with `-ngl 80` (all layers offloaded to the iGPU), so the CPU backend is
not doing weight matmuls for the served models; the CPU repack path buys
throughput only when a model actually runs on the CPU. The trade is therefore
"lose a CPU-only optimisation" against "remove a whole class of backend deaths".

## 4. Exact build command

```bash
# Staged artifact (no deploy, no restart — the coordinated step is §7):
scripts/rebuild-llama.sh \
  --repo /home/rgardler/llama.cpp \
  --dir /tmp/llama_rebuild_cpu \
  --build-only --jobs 16 --json \
  --record-file /tmp/llama_build_record.json
```

`--repo /home/rgardler/llama.cpp` pins the rebuild to the **same llama.cpp
commit as the deployed build**, so the only difference between the staged and
the live artifact is the flag change — the A/B is controlled.

The cmake configuration the script applies:

```text
cmake -S . -B build \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DLLAMA_OPENSSL=ON -DGGML_NATIVE=ON -DGGML_CPU_REPACK=OFF
```

The full command, flags, llama.cpp commit, artifact sizes and **sha256 sums**
are written by the script to the build record
(`/tmp/llama_build_record.json`, `"work_item": "LP-0MW3K7Y3C003WEXW"`), together
with the census reports for the built and (when deploying) deployed copies.

## 5. Artefact census of the staged build

Built 2026-11-05 (≈8 min, `-j16`) from llama.cpp `e97492369` — the same commit
as the deployed build — with the only difference being the two CPU flags.

```text
$ /tmp/llama_rebuild_cpu/build/bin/llama-server --version
version: 8782 (e97492369)

$ scripts/lib/ggml_cpu_census.py /tmp/llama_rebuild_cpu/build/bin/libggml-cpu.so.0.9.11 \
      --expect-repack off
artifact:                /tmp/llama_rebuild_cpu/build/bin/libggml-cpu.so.0.9.11
generic kernels:         32
arch kernels:            27
arch-specific codegen:   yes
  - ggml_vec_dot_q4_K_q8_K: present=True vector_insns=61 generic_tail_call=False
  - ggml_vec_dot_q6_K_q8_K: present=True vector_insns=90 generic_tail_call=False
  - ggml_vec_dot_q4_0_q8_0: present=True vector_insns=14 generic_tail_call=False
CPU repack registered:   no (off expected); repack code linked: yes
verdict:                 OK                       (exit 0)
```

Artifact checksums (`sha256sum`), staged next to the deployed originals:

| Artifact | sha256 | bytes |
|---|---|---|
| `libggml-cpu.so.0.9.11` (staged, repack unregistered) | `93b7ca449a0ee010704733afd98d2f104cf866b8edbeac7acfa6cbb098aece59` | 1 329 072 |
| `libggml-cpu.so.0.9.11` (deployed, repack registered) | `f4ab7518d6e39dd60f6dcd58fc6c1fc79d2b02675e8f8cb5739686a5c12b78bb` | 1 329 152 |
| `llama-server` (staged) | `15c85783a116668040ac16f032e56ccf0263498562fea8bdb3bc4fec65d96a1a` | 13 552 136 |

The 80-byte library delta is the whole change: one registration block and the
`REPACK` feature flag.

**The registration site, both artifacts, same commit** — this is the A/B that
proves the flag reached the code that crashes:

```text
# deployed (GGML_CPU_REPACK=ON)              # staged (GGML_CPU_REPACK=OFF)
call __cxa_guard_acquire@plt                 call __cxa_guard_acquire@plt
call ggml_backend_cpu_repack_buffer_type@plt <-- registers the faulting dispatch
call ggml_backend_cpu_repack_buffer_type@plt call __cxa_atexit@plt
call __cxa_atexit@plt                        call __cxa_guard_release@plt
```

With no registration the vector returned by
`ggml_backend_cpu_get_extra_buffer_types()` is empty, so `ggml_cpu_extra_work_size`
returns `false` from its loop head and the three faulting instructions
(`+0x38`/`+0x41`/`+0x47`) are unreachable.

Provenance record (machine-readable, `--record-file`):

```text
$ python3 -c 'import json;d=json.load(open("/tmp/llama_build_record.json"));print(d["work_item"], d["llama_cpp_commit"], d["cpu_repack"])'
LP-0MW3K7Y3C003WEXW e97492369 OFF
```

It also carries every artifact's size and sha256 (44 entries: binary +
`libggml*`/`libllama*`/`libmtmd*` in the build tree) and both census reports, so
a later audit can re-derive this table without the temp directory.

## 6. Rollback

The deploy is a directory-level artifact swap, so the rollback is a redeploy of
the previous artifact:

1. **Keep the previous build tree.** `/home/rgardler/llama.cpp/build/bin.bak-20260818`
   holds the pre-fix artifacts; `scripts/rebuild-llama.sh --deploy-only --dir <that
   tree> --deploy-path …` re-deploys them (add `--skip-census`, or
   `--expect-cpu-repack on`, because that artifact deliberately carries the
   repack path).
2. **Or rebuild with the old configuration:** `scripts/rebuild-llama.sh
   --with-cpu-repack --expect-cpu-repack on`.
3. Either way the proxy self-heals a restarted backend (one reasoned retry,
   recovery in 2–14 s, `restart_reason=llama_server_exit_code_*`), which is the
   documented path in `docs/llama-router.md`; the crash-blocking value of this
   change is lost, the crash is not made worse.

## 7. Deploy and replay — deliberately not done here (AC2/AC4)

This session produced the build configuration, the artifact gate and the staged
artifact. It did **not** touch the running deployment:

* **AC4 (coordination).** Promoting the artifact restarts every llama-server,
  which re-anchors the ≥24 h restart-free window owned by `LP-0MVZEPI3006Y0WN`
  (and measured by `LP-0MVW8S8OZ00449YR`). That restart needs the window owners'
  sign-off and a chosen slot, not an incidental restart from a build script.
* **AC2 (bounded replay).** `scripts/diagnose_libggml_segfault.sh --replay` is
  destructive by design (it drives ~16 K-token prefills at the Q4_K preset and
  can crash the live backend). Running it against the *rebuilt* artifact
  requires that artifact to be deployed first, i.e. the same coordinated
  restart. Equivalent evidence recorded instead:
  1. the fault location is now resolved to three named instructions, and
  2. the staged artifact's `ggml_backend_cpu_get_extra_buffer_types()` contains
     **no** call to `ggml_backend_cpu_repack_buffer_type()`, so the extra
     buffer-type vector is empty at runtime and that code cannot execute
     (`objdump` evidence in §3/§5).

Promotion, when scheduled:

```bash
# 1. deploy the staged artifact (census-gated, refuses a bad artifact)
scripts/rebuild-llama.sh --deploy-only --dir /tmp/llama_rebuild_cpu \
  --record-file /tmp/llama_build_record.json
# 2. bounded reproduction against the promoted build (destructive, ~16K-token prefills)
scripts/diagnose_libggml_segfault.sh --replay
# 3. confirm the census of what is now running
scripts/diagnose_libggml_segfault.sh
```

## 8. The `libamdhip64` GPF is a separate class (not fixed here)

`/var/log/kern.log` on 2026-11-05 records **12** `libggml-cpu` faults (all three
repack offsets above) and **12** `libamdhip64.so.7.2.70204` GPFs, every one at
`ip:…[2b931f,…]`, at *different* timestamps. A llama.cpp CPU-backend rebuild
cannot fix a ROCm userland fault, so this record claims the `libggml-cpu` half of
parent AC3 only; the `libamdhip64` GPF is tracked separately (discovered-from
LP-0MW3K7Y3C003WEXW) as a ROCm 7.2 userland/kernel mismatch investigation.

## 9. Regression coverage

* `tests/test_ggml_cpu_census.py` — the census verdicts: arch-specific codegen,
  the `_generic`-tail-call defect class, repack *registration* (including
  "repack code linked but unregistered", the shipped configuration), unknown
  registration failing closed, and crash-offset → symbol location.
* `tests/test_rebuild_llama.py` — `-DGGML_CPU_REPACK=OFF` + `-DGGML_NATIVE=ON` in
  the plan, `--with-cpu-repack` opt-in, the deploy **refusal** path (nothing
  copied, live server untouched), `--build-only` staging, and the provenance
  record (commit, flags, sha256).
