---
type: concept
created: 2026-06-20
updated: 2026-07-15
sources:
  - [[sources/SRC-2026-06-20-001]]
  - LP-0MPU61APP003SJ24
---

# Accelerator Performance Tuning

System profile optimization and llama.cpp parameter tuning for the Strix Halo
(AMD Ryzen AI MAX+ 395, gfx1151) system.

## llama-bench Baseline Results

Benchmarks run via `scripts/llama-bench.sh` (a wrapper around llama.cpp's
`llama-bench` tool). All tests use default params: `-ngl 99 -t 16 -b 2048 -ub 512
-fa 1 -ctk f16 -ctv f16 -p 512 -n 128 -r 3` unless otherwise noted.

### System

- **CPU:** AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (16 threads)
- **GPU:** AMD Radeon Graphics, gfx1151 (122880 MiB VRAM)
- **Backend:** ROCm (GPU offload with `-ngl 99`)

### Benchmark Results

#### mxbai-embed (bert 335M Q8_0) — 129 MiB

| Test | Tokens/s |
|------|----------|
| Prompt processing (512 tok) | 2,668.86 ± 13.71 t/s |
| Text generation (128 tok) | 280.33 ± 2.37 t/s |

#### Qwen3-Coder-30B-A3B (BF16) — 48 GiB

| Test | Tokens/s |
|------|----------|
| Prompt processing (512 tok) | 306.99 ± 0.40 t/s |
| Text generation (128 tok) | 17.87 ± 0.01 t/s |

#### Qwen3.6-35B-A3B-UD-Q5_K_M (25 GiB, ~35B params, ~3.5B active)

| Test | Tokens/s |
|------|----------|
| Prompt processing (512 tok) | 183.79 ± 3.38 t/s |
| Text generation (128 tok) | 28.27 ± 0.12 t/s |

#### Qwen3.6-35B-A3B-Q8_0 (34 GiB, ~35B params, ~3.5B active)

| Test | Tokens/s |
|------|----------|
| Prompt processing (512 tok) | 180.97 ± 2.92 t/s |
| Text generation (128 tok) | 26.33 ± 0.14 t/s |

### Parameter Sweep: Qwen3.6-35B-A3B-UD-Q5_K_M

Full parameter sweep (864 combinations, plus targeted t=24/t=32 tests).
Swept parameters: ngl={99,80,60}, t={32,24,16,12,8}, b={4096,2048,1024},
ub={512,256}, ctk={f16,q8_0}, ctv={f16,q8_0}, p={512,1024}, n={128,256}.
Flash attention fixed to fa=1 (compatible, negligible difference).

#### Thread Count Impact

| Threads | Avg Prompt t/s | vs t=16 | Avg Gen t/s |
|---------|---------------|---------|-------------|
| 32 (SMT) | 146.3 t/s | **-25%** | 22.5 t/s |
| 24 (SMT) | 185.0 t/s | -5% | 28.2 t/s |
| **16 (native)** 👑 | **195.5 t/s** | — | **28.6 t/s** |
| 12 | ~149 t/s | -24% | ~28.4 t/s |
| 8 | ~113 t/s | -42% | ~28.0 t/s |

**Finding:** `t=16` is the clear sweet spot. SMT threads (24, 32) degrade
prompt processing due to cache contention. Fewer threads underutilize the
16-core CPU. Text generation is largely unaffected — it's bound by model
size/inference, not thread count.

#### GPU Layer Count

| ngl | t=16 Avg Prompt | Best Prompt |
|-----|----------------|-------------|
| 99 | 179.5 t/s | 194.9 t/s |
| **80** 👑 | **180.4 t/s** | **195.5 t/s** |
| 60 | — | — |

**Finding:** `ngl=80` and `ngl=99` are essentially tied. Dropping 19 GPU
layers (~20%) has no penalty — likely because this MoE model's expert layers
run on CPU regardless. ngl=60 not tested.

#### Batch & Micro-Batch Size

| Batch | Best Prompt | Best Gen |
|-------|-------------|----------|
| 4096 | 195.5 t/s | 28.6 t/s |
| 2048 | 193.2 t/s | 28.6 t/s |
| 1024 | 194.1 t/s | 28.6 t/s |

**Finding:** Batch size has negligible impact on this model. All three values
within ~1%. `ub=256` and `ub=512` also tied.

#### KV Cache Type

`ctk=f16/q8_0` and `ctv=f16/q8_0` — all combinations within noise (~1%).
Recommend keeping defaults (`f16`).

#### Overall Best Config

| Metric | Best | Config |
|--------|------|--------|
| **Prompt processing** | **195.5 t/s** | **ngl=80, t=16, b=4096, ub=256, fa=1, ctk=f16, ctv=f16, p=512** |
| **Text generation** | **28.6 t/s** | **ngl=80, t=16, b=2048, ub=256, fa=1, ctk=f16, ctv=f16, n=128** |

#### Sweep Notes

- `llama-bench` reports "backends: CPU" even when GPU offload is active
  (`-ngl 99`). This is a labeling quirk — GPU layers are offloaded correctly.
- The full sweep of 864 combinations took ~2.5 hours on this 25 GiB model
  (~3 combos/min average after accounting for slower t=12/t=8 configs).
- ngl=60 was not reached before the sweep was redirected to test SMT threads.

## Future Work

- Test ngl=60 and lower to find minimum GPU layers without penalty.
- Compare performance across model quantizations (Q5_K_M vs Q8_0) for same arch.
- Investigate batch size scaling effects on generation quality vs throughput.

## Links

- [[sources/SRC-2026-06-20-001]]
- LP-0MPU61APP003SJ24 — Learn to use LlamaBench
