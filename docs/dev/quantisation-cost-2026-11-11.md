# Cost of the model-weight quantisation lever: Q5_K_M vs Q4_K_M / Q4_K_S

**Work item:** LP-0MWCEHYOO002TRGD (spike parent LP-0MVXY8ZBZ0003042, question
AC1c — "model/quantisation / ctx-slot geometry").
**Date:** 2026-11-11. **Host:** Strix Halo (gfx1151), AMD Ryzen AI MAX+ 395,
124 GiB unified memory, local `llama-server` serving Qwen3.6-35B-A3B.
**Status:** read-only costing. No config, launch flag or model file was changed
by this item.

**Artefact:** [`quantisation-cost-2026-11-11.json`](quantisation-cost-2026-11-11.json)
— the machine-readable register behind every figure here, with its inputs
embedded so it can be re-verified after the host's GGUFs or logs rotate out.
**Tooling:** [`scripts/quantisation_cost_model.py`](../../scripts/quantisation_cost_model.py),
pinned by [`tests/test_quantisation_cost_model.py`](../../tests/test_quantisation_cost_model.py).

```bash
# re-verify the committed register (self-contained)
python3 scripts/quantisation_cost_model.py \
    --register docs/dev/quantisation-cost-2026-11-11.json --check

# inspect any GGUF's own tensor table (weights, params, bits-per-weight, roles)
python3 scripts/quantisation_cost_model.py --gguf x=/path/to/model.gguf --layout
```

---

## 1. Answer in one paragraph

A lower weight quant buys **memory and decode latency, not prefill throughput**
— and prefill throughput is the only thing that raises the local share this
spike is about. The shipped `UD-Q5_K_M` carries **24.629 GiB** of weights;
`UD-Q4_K_M` would carry **20.523 GiB** (−4.107 GiB) and `UD-Q4_K_S` **19.448 GiB**
(−5.181 GiB, measured). Prefill moves by **at most ~1.5 %** (bounded below by 0
and above by 20-27 %, and the upper bound is unreachable — see §3), because a
prefill pass at the measured 242 tok/s streams weights at **~6 GB/s**, two
orders of magnitude below what this host's memory can sustain: weight bytes are
demonstrably *not* the prefill bottleneck. Decode, which is batch-1 and reuses
no weight, does scale with the bytes: up to **+20 % (Q4_K_M) / +27 % (Q4_K_S)**
on the measured 30.7 tok/s. Decode is only **13.4 %** of local busy seconds, so
the freed capacity lifts the local token share from **0.699 % to at most
0.715 %/0.719 %** — inside the ceiling this spike already established. The
quality caveat is the real decision: 89.9 % of the bytes being quantised are
MoE expert weights, and this host has **no measured quality delta** for either
candidate.

---

## 2. What is shipped, and what each candidate weighs

Every figure in this table is read from the GGUF's own tensor table
(`scripts/quantisation_cost_model.py --layout`), not derived from a bits-per-
weight rule of thumb. The shipped preset is an unsloth **dynamic** quant
(`UD-`): 255 of its 733 tensors are held at Q8_0 (attention, embeddings,
output) while the 120 expert tensors take Q5_K/Q6_K.

| Layout | Weights | Δ vs shipped | bits/weight | Provenance |
|---|---:|---:|---:|---|
| shipped `UD-Q5_K_M` | **24.629 GiB** | — | 6.104 | measured (host GGUF) |
| `UD-Q4_K_M` | **20.523 GiB** | **−4.107 GiB (−16.7 %)** | 5.086¹ | derived-estimate |
| `UD-Q4_K_S` | **19.448 GiB** | **−5.181 GiB (−21.0 %)** | 4.820 | measured (host GGUF)² |
| plain `Q5_K_M` (llama-quantize) | 23.033 GiB | −1.596 GiB | 5.708 | derived-estimate |
| plain `Q4_K_M` (llama-quantize) | 19.715 GiB | −4.914 GiB | 4.886 | derived-estimate |
| plain `Q4_K_S` (llama-quantize) | 18.683 GiB | −5.946 GiB | 4.630 | derived-estimate |

¹ `20.523 × 8 / 34.661 B` — the estimate inherits the measured byte count, not
an independent bpw.
² measured from the host's `UD-Q4_K_S` **MTP** GGUF, base-model weights only
(the extra nextn block is excluded; the whole file is 19.909 GiB). It carries
the same 34.661 B parameters as the shipped model, so the two layouts are
byte-for-byte comparable.

The three "plain" rows are what `llama-quantize --type <preset>` would produce
from the shipped tensor table, computed by porting llama.cpp's own per-tensor
rules (`src/llama-quant.cpp`: `tensor_allows_quantization`, `tensor_get_category`,
`use_more_bits`, `llama_tensor_get_type_impl`) into the tool. They matter
because they show *where* the saving comes from: plain `Q4_K_M` is 0.81 GiB
smaller than the UD variant precisely because it also drops the Q8_0
attention/embedding tensors that until now have been protected — which is
where the quality risk concentrates (§3.3).

**Where the bytes are (shipped layout):**

| Role | GiB | Share |
|---|---:|---:|
| expert `ffn_down` | 8.385 | 34.0 % |
| expert `ffn_gate` | 6.875 | 27.9 % |
| expert `ffn_up` | 6.875 | 27.9 % |
| attention | 1.017 | 4.1 % |
| output | 0.503 | 2.0 % |
| token embedding | 0.503 | 2.0 % |
| SSM (conv/state) | 0.267 | 1.1 % |
| shared expert | 0.125 | 0.5 % |
| everything else | 0.079 | 0.3 % |

**89.9 % of the weights are the 256-expert MoE tensors**, so a weight quant is
overwhelmingly a decision about expert precision — not about the attention
stack, which the shipped UD quant keeps at Q8_0 on purpose.

---

## 3. AC1 — prefill throughput and VRAM effect, and the expected tok/s gain

### 3.1 Weight quantisation changes exactly one input: bytes per weight

It does not change FLOPs, token counts, routing, the KV cache (`cache-type-k/v:
q8_0`, weight-independent) or the prompt cache (`cache-ram: 40960` MiB, and the
~2.0 GiB/prompt checkpoint cost is a function of `ctx-checkpoints: 32` ×
62.8 MiB, not of the weights). So the only question that decides the throughput
effect is: **which term of the engine's time is weight movement?** Two measured
regimes give two different answers.

### 3.2 Prefill: weight bytes are not the bottleneck, so the lever is small

Measured prefill aggregates (production `prompt eval time` lines, 2026-11-10 →
2026-11-11 20:55 UTC, 4 669 measured prefills,
`scripts/prefill_throughput_report.py`):

| Prompt-size bucket | Requests | Mean prompt | Aggregate tok/s | Weight-time share of a pass | Weight stream rate |
|---|---:|---:|---:|---:|---:|
| < 8 k | 4 302 | 1 111 | 133.8 | 7.20 % | 3.46 GB/s |
| 8 k–20 k | 333 | 11 610 | 198.4 | 6.13 % | 5.12 GB/s |
| 20 k–40 k | 25 | 22 379 | **242.3** | 7.12 % | **6.26 GB/s** |
| 40 k–80 k | 9 | 44 770 | 142.7 | 4.19 % | 3.69 GB/s |

The last two columns are the whole argument. A 1024-token micro-batch
(`models.ini:ubatch-size = 1024`) touches every one of the 256 experts, so it
streams the **entire** 24.629 GiB of weights; at the measured 242 tok/s a pass
takes 4.23 s, i.e. **6.3 GB/s** of weight traffic. The *same* model's batch-1
decode already sustains **88 GB/s** (§3.3) — a measured floor on what this
host's unified LPDDR5X memory can actually stream, against a ~250 GB/s hardware
peak. Weight streaming during prefill is therefore **7 %** of the bandwidth the
engine itself demonstrates: **weight bytes cannot be what caps prefill
throughput**. That is also
consistent with the two earlier negative results — doubling the micro-batch did
not move 27 k prompts (1.04×, `docs/dev/ubatch-ab-2026-10-22.md`) and opening a
second prefill gate did not move aggregate throughput (1.00×,
`docs/dev/prefill-concurrency-2026-11-11.md`). Statements elsewhere in this
repo that prefill is "memory-bandwidth bound" describe the expert GEMM's
roofline; they do not mean weight streaming.

Two bounds follow, and both are recorded per candidate per bucket in the
register:

| Candidate | Prefill gain ceiling (bytes are the whole critical path) | Prefill gain at the measured weight share | 20 k–40 k tok/s |
|---|---:|---:|---:|
| `UD-Q4_K_M` | +20.0 % | **+1.4 %** | 242.3 → 245.8 |
| `UD-Q4_K_S` | +26.6 % | **+1.9 %** | 242.3 → 246.9 |

The **ceiling** column is what the literal byte ratio would buy; it is
unreachable because it assumes prefill time is entirely weight movement, which
the 6 GB/s measurement refutes. The **measured-share** column is the share of a
pass that weight movement actually occupies (≤ 7.2 %), so the expected gain is
**0 to ~1.5 %**, and 0 is the floor because the quantised GEMM's FLOPs are
unchanged.

**Expected prefill gain: ~0–1.5 % on Q4_K_M, ~0–1.9 % on Q4_K_S.** A 1.5 %
prefill gain is not a lever for local share — it is inside the run-to-run noise
of the 20 k–40 k bucket (25 requests).

### 3.3 Decode: batch-1 weight streaming *is* the bottleneck, so the lever works

A batch-1 decode streams one token per expert it routes to (8 of 256,
`expert_used_count`), the shared expert, and every non-expert weight — 2.683 GiB
per token, computed from the shipped tensor table:

| Component | Bytes per decode token |
|---|---:|
| 8 of 256 routed experts | 0.692 GiB |
| shared expert | 0.125 GiB |
| attention + SSM + output + norms | 1.866 GiB |
| **active total** | **2.683 GiB** |

At the measured 30.7 tok/s (`docs/dev/single-slot-impact.md` W1) that is
**88.4 GB/s** — 35 % of the host's theoretical bandwidth, i.e. a genuine
streaming regime rather than a launch-bound one. So the byte ratio translates
into decode speed, bounded above by it:

| Candidate | Byte ratio | Decode ceiling | 30.7 tok/s → |
|---|---:|---:|---:|
| `UD-Q4_K_M` | 0.833 | **+20.0 %** | ≤ 36.8 tok/s |
| `UD-Q4_K_S` | 0.790 | **+26.6 %** | ≤ 38.9 tok/s |

The same bound applies to a *warm* run: W2 measured 37.7 tok/s, so the ceiling
is ≤ 45.2/47.8 tok/s there. The lower bound is 0 (if fixed per-token overheads
dominate), but at 35 % of peak bandwidth the streaming term is clearly large.

### 3.4 The quality-risk caveat (recorded, not adopted)

- **Expert precision is what changes.** 89.9 % of the bytes quantised are the
  MoE expert tensors; `Q4_K_S` drops `ffn_down` to Q4_K on 30 of 40 layers and
  `Q4_K_M` keeps Q6_K on 20 of them. Mixture-of-experts routers are known to be
  precision-sensitive, and a 4-bit expert shrink is exactly the change that
  degrades long-tail reasoning first.
- **No local quality measurement exists.** Nothing in this repository measures
  answer quality between quant levels; the MTP preset happening to ship Q4_K_S
  (`models.ini:[Qwen3-MTP]`) is a precedent, not evidence.
- **A plain `Q4_K_M` also drops the protected tensors.** The shipped UD quant
  holds attention/embeddings/output at Q8_0 (+1.60 GiB over a plain `Q5_K_M`).
  Adopting a *plain* Q4 would give up that protection; adopting `UD-Q4_K_M`
  keeps it for +0.81 GiB over plain. Prefer the UD variant if this is ever
  tried, and gate it on a quality suite rather than on throughput.
- **The risk is asymmetric.** The upside is ~1 % prefill and ≤ 20-27 % decode
  latency on a route already capped by prefill capacity; the downside is an
  unmeasured quality regression on every locally-served turn.

### 3.5 How to turn this estimate into a measurement

The estimate is cheap to replace with a measurement, in a quiet window (the
production proxy holds the GPU; a benchmark competes with live traffic):

```bash
export LD_LIBRARY_PATH="$HOME/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
~/llama.cpp/build/bin/llama-bench \
    -m <UD-Q4_K_M.gguf> -ngl 99 -t 8 -b 8192 -ub 1024 -fa 1 \
    -ctk q8_0 -ctv q8_0 -p 512,4096,8192 -n 128 -r 3 -o json
```

Two findings from attempting it here:

1. **Use `~/llama.cpp/build/bin/llama-bench`, not `/usr/local/bin/llama-bench`.**
   The PATH binary is an older build (2026-04) that cannot load `qwen35moe`; the
   build directory holds the same binary the proxy serves with. Both need
   `LD_LIBRARY_PATH` set to that directory.
2. **The host cannot supply a Q4 measurement today.** The only local lower-quant
   file is the MTP preset (`UD-Q4_K_S`), and the installed build refuses it:

   ```
   llama_model_load: error loading model: missing tensor 'blk.40.ssm_conv1d.weight'
   ```

   (the MTP GGUF's nextn block is incomplete for non-MTP loading). A measured
   Q4 prefill/decode therefore needs a non-MTP `UD-Q4_K_M` (≈21 GiB) download
   first. Until then the numbers in §3.2–§3.3 are the registered bounds, and
   every one of them is labelled.

---

## 4. AC4 — the comparison-table rows

For each candidate: throughput (measured anchor + bounded effect) and per-slot
resource cost.

| | shipped `UD-Q5_K_M` | `UD-Q4_K_M` | `UD-Q4_K_S` |
|---|---:|---:|---:|
| Weights (shared by all slots) | 24.629 GiB | 20.523 GiB (est.) | 19.448 GiB (measured) |
| Weights per slot @ 2 slots | 12.31 GiB | 10.26 GiB | 9.72 GiB |
| Weights per slot @ 1 slot | 24.63 GiB | 20.52 GiB | 19.45 GiB |
| Prompt cache (`cache-ram`) | 40 960 MiB | unchanged | unchanged |
| Per-prompt checkpoint cost | ≈2.0 GiB (32 × 62.8 MiB) | unchanged | unchanged |
| KV cache | q8_0, 131 072 tok/slot | unchanged | unchanged |
| Prefill, 20 k–40 k prompts | 242.3 tok/s (measured) | 242.3–245.8 tok/s | 242.3–246.9 tok/s |
| Prefill, 40 k–80 k prompts | 142.7 tok/s (measured) | 142.7–143.9 tok/s | 142.7–144.3 tok/s |
| Decode | 30.7 tok/s (measured) | ≤ 36.8 tok/s | ≤ 38.9 tok/s |
| Local token share (2026-11-01 window) | 0.699 % (measured) | ≤ 0.715 % | ≤ 0.719 % |
| Local stream share | 2.573 % (measured) | ≤ 2.630 % | ≤ 2.645 % |
| Quality risk | — | unmeasured | unmeasured |

The share arithmetic: 5 027 local completion tokens at 30.7 tok/s is 163.7 s of
decode inside 1 223 s of local busy time (13.4 %). A 20 %/27 % faster decode
frees 27.3/34.4 s, i.e. +2.2 %/+2.8 % of local capacity, which scales the
ceiling the share is measured against. **+0.02/+0.03 pp of token share** — the
spike's 10 % target needs 6.2× this ceiling, so quantisation does not move the
target it is being asked about.

---

## 5. AC2 — reuse, do not duplicate, the launch-flag A/B work

This item costs **model-weight quantisation** only. The other three axes named
in the spike's AC1c already have owners, and none of them is re-benchmarked
here:

| Knobs | Owner | State at 2026-11-11 | Band / gate it requires |
|---|---|---|---|
| `--ubatch-size`, `--batch-size` | **LP-0MVNBJXR6002OSPD** | `blocked`, stage `intake_complete` | 40 k–80 k tokens, one variable at a time, ≥ 3 reps, pre-registered gate (e.g. ≥ 1.3× p90 prompt-eval at 60 k–80 k with decode regression ≤ 2 %); raw JSON to `benchmark-results/` |
| `--threads`, `--swa-checkpoints` (ctx-checkpoints) | **LP-0MVNBJY2R004EZ51** | `blocked`, stage `intake_complete` | same 40 k–80 k band; the checkpoint axis must be priced with the ~62.8 MiB/checkpoint memory term in mind |
| slot geometry (1 × 262 K vs 2 × 131 K) | **LP-0MVNCMEJ6004A2QY** (Arm B run) / **LP-0MVMHNUXW0061FGN** (A/B experiment) | `blocked` / `open`, stage `plan_complete` | ≥ 60 % local busy for a full night window before any arm is measured (`docs/dev/slot-geometry-ab.md` §0); `compare` reports `inconclusive` otherwise |
| context size | LP-0MSAOQTJS000FFVM | `completed` / `done` | out of scope by the spike's constraints |

Both launch-flag items are blocked on the **same** measurement band
(40 k–80 k tokens) that the previous ubatch A/B
(`docs/dev/ubatch-ab-2026-10-22.md`) declared as its retry condition, and the
geometry item is blocked on a **contended** window (≥ 60 % busy) that today's
demand does not provide (local idle 89.8 % of W1). This item deliberately adds
no third benchmark for those knobs; the only new measurement it required was
the **prompt-size-bucketed prefill curve**, which is reusable by all three
owners (the same `scripts/prefill_throughput_report.py`, zero GPU cost).

---

## 6. AC3 — costing the ctx/slot geometry

Summarised from [`docs/dev/single-slot-impact.md`](single-slot-impact.md) §5
(throughput prediction) and [`docs/dev/slot-geometry-ab.md`](slot-geometry-ab.md)
(measurement protocol); neither is re-derived here.

- **Throughput is GPU-bound, not slot-bound.** One slot would be busy only
  **10.5 %** of the day (89.5 % headroom) against 5.2 % at two slots, so
  shrinking the pool cannot reduce the tokens the machine *can* process — it
  changes the order requests get them in.
- **What one slot costs is concurrency and prefix reuse.** Prefills are already
  serialised (`local_max_concurrent_prefills: 1`), so the second slot only ever
  overlapped decode: 300 slot-seconds/day, ~15 turns/day (6.6 % of 225). The
  real penalty is thrash — every session maps to slot 0, and the recorded steal
  rate at pool size 1 is **95 %** for two concurrent sessions, inflating
  re-prefill by up to 11.5 M tokens/day (26.3 % on top of the 43.8 M/day pool).
- **What one slot buys** is a 258 048-token per-slot clamp, but only if
  `local_large_context_economic_ceiling_ratio` (0.4578 → 0.9844) and
  `local_large_context_warm_cache_threshold` (120 000 → 258 048) move with it;
  applied alone, the extra context routes nothing extra. `local_session_tenants`
  must also drop 2 → 1 (it is explicit, so it does not follow the pool).
- **What remains unmeasured:** the arm A/B itself. No geometry was flipped by
  this item or its siblings — the verdict needs a ≥ 60 % busy window, which no
  recorded window provides (`docs/dev/single-slot-impact.md` §5.6 §5.8).

The quantisation lever is **orthogonal** to this axis and composes with it: the
weight saving is a per-*instance* cost, so it is worth 4.1-5.2 GiB whichever
slot count is chosen (its per-slot share doubles at one slot — see §4).

---

## 7. AC5 — the ctx-size axis is not widened

Context size is out of scope: LP-0MSAOQTJS000FFVM evaluated it and closed
(`completed`/`done`), and the shipped `local_model_ctx_size: 262144` equals the
GGUF's own `qwen35moe.context_length`, so there is no headroom to widen into.
This item changes no ctx value; §4's per-slot context column only follows the
slot **count** (262 144 / slots), exactly as `models.ini` documents.

---

## 8. Method and provenance

1. **Measured, not assumed, layouts.** `scripts/quantisation_cost_model.py`
   parses the GGUF header's tensor-info section and computes each tensor's size
   from ggml's own block/byte table, so `24.629 GiB` and `6.104 bpw` are what
   the files contain. The tool reads only metadata — a 25 GiB model costs a few
   milliseconds and no GPU time.
2. **Preset predictions follow llama.cpp's rules, not a bpw rule of thumb.**
   `preset_type()` ports `tensor_allows_quantization`, `tensor_get_category`,
   `use_more_bits` and the non-Falcon/non-8-expert branch of
   `llama_tensor_get_type_impl`; the register's `checks` assert the derived
   sizes reproduce from the measured table.
3. **Prefill anchors are measured production lines.** The four buckets come from
   `scripts/prefill_throughput_report.py` over the retained `llama-server.log`
   window (2026-11-10 00:00 → 2026-11-11 20:55 UTC, 4 669 prefills); they cost
   no GPU time because they are the engine's own `prompt eval time` records.
4. **Decode and demand anchors are the committed baselines.**
   `docs/dev/single-slot-impact.md` W1 (30.7 tok/s decode) and
   `docs/dev/local-share-baseline-2026-11-01.json` (1 223 busy s, 5 027
   completion tokens, 0.699 % token share).
5. **Every figure is classified** (`measured`, `derived-estimate`,
   `config-derived`) in the register, and `--check` re-derives 37 claims from
   the register's own embedded inputs — a tampered or stale register reports
   `drift` and exits non-zero.

---

## 9. What would change the answer

- **A measured Q4 prefill rate.** If a real `UD-Q4_K_M` benchmark shows prefill
  beyond +2 % (i.e. beyond the measured weight share), then the quantised GEMM's
  dequantisation cost is a larger term than this model assumes, and the lever is
  worth more than §3.2 says. The command in §3.5 settles it in one quiet window.
- **A contended window.** Decode's 13.4 % slot share is measured at 10-18 %
  local busy; under sustained contention decode delays more turns, so both the
  throughput and the share effects grow.
- **A quality measurement.** The recommendation ("not a throughput lever; do not
  adopt without a quality gate") flips only if a quality suite shows no
  regression at Q4_K_M — the memory saving would then be free VRAM headroom.
- **A larger context or more slots.** The weight saving is per instance, so it
  becomes more valuable as per-slot KV/checkpoint costs grow.
