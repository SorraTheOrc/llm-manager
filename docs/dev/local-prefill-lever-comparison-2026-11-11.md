# Local-prefill levers: costed four-family comparison and the share target

**Work item:** LP-0MWCEII0V003V76A (spike parent LP-0MVXY8ZBZ0003042, AC1 + AC2).
**Consumes:** LP-0MWCEGJDP009WURD (measured basis), LP-0MWCEGXEK000U6G6
(compaction), LP-0MWCEHDRV000B7HD (parallel prefill), LP-0MWCEHYOO002TRGD
(quantisation / geometry), LP-0MWCEI8QV007NIC0 (accept-the-ceiling arithmetic).
**Records a recommendation on:** LP-0MVWEZCB0006WPBQ (its AC9 share target).
**Status:** read-only. No routing, config or model change is part of this item;
any implementation item it suggests becomes a *new* child of the spike and is
named here, not created here.

**Artefact:** [`local-prefill-lever-comparison-2026-11-11.json`](local-prefill-lever-comparison-2026-11-11.json)
— the machine-readable register behind every figure quoted below, with each
source register's path and every formula recorded.
**Tooling:** [`scripts/compare_prefill_levers.py`](../../scripts/compare_prefill_levers.py),
pinned by [`tests/test_compare_prefill_levers.py`](../../tests/test_compare_prefill_levers.py)
(family arithmetic, denominator discipline, target arithmetic, drift guard,
doc/register consistency).

```bash
# the register, and the human-readable summary
python3 scripts/compare_prefill_levers.py --json \
    > docs/dev/local-prefill-lever-comparison-2026-11-11.json
python3 scripts/compare_prefill_levers.py

# re-verify the committed register from its own embedded inputs (drift guard)
python3 scripts/compare_prefill_levers.py \
    --register docs/dev/local-prefill-lever-comparison-2026-11-11.json --check
```

---

## 1. Answer in one paragraph

**No lever family buys enough aggregate prefill to matter, and the one family
that buys anything at all buys it by spending prefill gate time that is already
idle.** Parallel prefill is measured at **1.00x** aggregate (2.00x wall time,
0.81x per-request latency); a lower weight quant moves prefill by **at most
1.9 %** because prefill streams weights at 6.3 GB/s against the 88 GB/s that
batch-1 decode already sustains; slot geometry adds **0 tok/s**; and compaction
adds **0 tok/s of rate** while converting idle gate-seconds into **+409 126
delivered tokens** — and only if its target is co-lowered to the 25 000-token
cold-prefill budget, at which point the shipped 30k/38k targets and the intake's
57k convert **exactly nothing**. Together the levers bound at **1.019x** on the
gate rate, i.e. **3.562 % of streams** and **1.051 % of tokens** — 13.3 % of the
way to 10 % of streams and 3.8 % of the way to 10 % of tokens. The decision that
follows is therefore **(d) accept the ceiling**, and the specific number to put
in parent **AC9** is **≥ 3.5 % of locally served streams** (0.951 % of offered
prompt tokens): **1.25x** the 2.8 % baseline, **0.587x** the point capacity
ceiling, and reachable inside the window's *idle* prefill gate even if every
added token is cache-cold — a routing/admission target, not a throughput one.

## 2. The denominators (read this before either table)

LP-0MWCEGJDP009WURD showed the published "~5.8 %" and the same window's "~1.6 %"
are both right, for different denominators. They are **not** interchangeable:

| Denominator | Definition | Measured (2026-11-01 window) | Ceiling min / point / max |
|---|---|---:|---|
| **prompt-token share** | local prompt tokens / offered prompt tokens | **0.6991 %** | 0.4133 / **1.6206** / 4.9626 % |
| **stream share** | locally served streams / offered streams | **2.5727 %** | 1.5209 / **5.9637** / 18.2625 % |
| **bridge** | offered prompt tokens per stream ÷ local tokens per stream | 70 233.7 / 19 085.0 = **3.680x** | — |

One stream-share point is worth ≈3.68 token-share points at the current mix, so
a comparison that costs a **tok/s** lever against a **stream** target is wrong
by that factor. Every family below is reported on **both** denominators, and the
ceiling band's width is **cache reuse**, not engine rate (moving reuse across
its measured range moves the ceiling 7.39x; moving the engine rate across its
whole measured band moves it 1.63x).

## 3. AC1 — the four families, costed

| Family | Throughput added | Per-slot resource cost | Token share Δ | Stream share Δ | Verdict |
|---|---:|---|---:|---:|---|
| (a) session size / compaction | +0.0 tok/s (converts idle gate time) | summariser slot: 4.5 GB weights + 1.75 GiB KV = 6.25 GB (5.2 % VRAM), 24.9–45.6 s GPU per compaction | +0.326 pp | +0.915 pp | conditional |
| (b) prefill serialisation / parallel prefill | +0.0 tok/s (1.00x at 156.2 → 156.1 tok/s) | 2 prefilling slots: 2 × 131 072-token ctx, 4.0 GiB checkpoint state, no extra rate | 0.000 pp | 0.000 pp | reject |
| (c) model / quantisation / ctx-slot geometry | prefill +1.9 % at best (decode +20–27 %) | weights 24.629 → 20.523 / 19.448 GiB; geometry trades concurrency + prefix reuse | +0.020 pp | +0.073 pp | not a share lever |
| (d) accept the ceiling | +0.0 tok/s (0 GiB, 0 config changes) | none | 0.000 pp | 0.000 pp | endorsed baseline |

Δ columns are percentage-point changes against the measured 0.6991 % /
2.5727 %; family (c) quotes its best measured-share candidate (`UD-Q4_K_S`).

### (a) Session size / compaction — LP-0MWCEGXEK000U6G6

*Artefact:* [`compaction-benefit-estimate-2026-11-01.md`](compaction-benefit-estimate-2026-11-01.md).

- **Throughput:** adds **0 tok/s** of prefill rate. What it converts is demand:
  **+409 125.6 delivered tokens** (16.4 streams) out of `1 612 s` of **idle**
  prefill gate (56.9 % of the 2 835 s window) — delivered-rate attribution
  **144.3 tok/s** over the window, against a gate that stays at 253.8 tok/s.
- **Per-slot cost:** one dedicated `Qwen2.5-7B` Q4_K_M summariser slot
  (`parallel = 1`, ctx 32 768) — 4.5 GB weights + 1.75 GiB KV = **6.25 GB
  (5.2 % of VRAM)** held for the server's lifetime, **24.9–45.6 s** of GPU time
  per compaction, competing with the gate it relieves.
- **Share:** token 0.699 → **1.025 %** (+0.326 pp), stream 2.573 → **3.488 %**
  (+0.915 pp).
- **The condition that decides it:** the flat 25 000-token cold-prefill cap is
  the binding test, and the refused population is cache-cold, so `new_tokens`
  equals the prompt size. At 57 000 (the intake's median), 38 000 (shipped fast)
  and 30 000 (shipped cheap) the conversion is **0 tokens, 0 pp** — the
  summariser bill with none of the conversion. The demand at the 25k cap is
  28 140.7 gate-seconds against 1 612 idle ones, so the lever is gate-clipped to
  16.4 streams, and a cold token costs **2.83x** the gate time of a blended
  delivered token — the gain is real only while the gate is idle.
- **Verdict:** *conditional* — the only family with a positive share delta, and
  it requires co-tuning `compaction_trigger_ratio` **and** the target to the
  cold-prefill budget, at the cost of ~25k-token client contexts (a quality
  trade) and a permanent summariser slot.

### (b) Prefill serialisation / parallel prefill — LP-0MWCEHDRV000B7HD

*Artefact:* [`prefill-concurrency-2026-11-11.md`](prefill-concurrency-2026-11-11.md).

- **Throughput:** **156.1 tok/s** at concurrency 2 versus **156.2 tok/s** at
  concurrency 1 on the same 44 770-token prompt — ratio **1.00** (best case
  1.00, worst 0.96) — while the wall window doubles (**2.00x**) and the slowest
  request in a round falls to **0.81x** (126.6 tok/s). The 20k–40k production
  bucket is unchanged at 241.0–243.5 tok/s. Both arms sustain **13.5M
  tokens/day** at 40–80k.
- **Per-slot cost:** two prefilling slots' worth of context-checkpoint state
  (2 × 2.0 GiB at 131 072 tokens/slot) occupied for zero additional throughput,
  and a live request queued behind the probe's prefill waited **183.2 s** to
  first byte (p50 6.3 s before it).
- **Share:** 0 pp on both denominators — no rate change.
- **Verdict:** *reject* — serialisation is **not** the binding constraint; the
  GPU conserves memory bandwidth, so concurrency redistributes latency instead
  of adding capacity. Rollback value is the shipped
  `local_max_concurrent_prefills: 1`.

### (c) Model / quantisation / ctx-slot geometry — LP-0MWCEHYOO002TRGD

*Artefact:* [`quantisation-cost-2026-11-11.md`](quantisation-cost-2026-11-11.md).

| Candidate | Weights | Δ | Prefill 20k–40k | Decode ceiling | Token share | Stream share |
|---|---:|---:|---:|---:|---:|---:|
| shipped `UD-Q5_K_M` | 24.629 GiB | — | 242.3 tok/s | 30.7 tok/s | 0.6991 % | 2.5727 % |
| `UD-Q4_K_M` (est.) | 20.523 GiB | −4.107 GiB | 242.3 → 245.8 (+1.4 %) | ≤ 36.8 (+20.0 %) | 0.7147 % | 2.6301 % |
| `UD-Q4_K_S` (meas.) | 19.448 GiB | −5.181 GiB | 242.3 → 246.9 (+1.9 %) | ≤ 38.9 (+26.6 %) | 0.7188 % | 2.6452 % |

- **Throughput:** prefill moves by **0–1.9 %**. Weight movement occupies ≤ 7.1 %
  of a prefill pass (a 1 024-token micro-batch touches all 256 experts, so a
  pass streams all 24.629 GiB in 4.23 s = **6.3 GB/s**, against the **88 GB/s**
  batch-1 decode already sustains) — so the 20–27 % byte-ratio ceiling is
  unreachable and weight bytes are not the prefill bottleneck.
- **Per-slot cost:** weights are shared across slots, so the saving is a
  per-instance 4.1–5.2 GiB (12.31 → 10.26 / 9.72 GiB per slot at 2 slots). The
  40 960 MiB prompt cache, the q8_0 KV cache and the ~2.0 GiB/prompt checkpoint
  term are weight-independent. **89.9 %** of the quantised bytes are MoE expert
  weights, and there is **no local quality delta** for either candidate; the only
  local Q4 file (the MTP preset) cannot be loaded non-MTP, so a measured Q4 arm
  needs a ~21 GiB download first.
- **Share:** +0.020 pp tokens / +0.073 pp streams — decode capacity freed
  (13.4 % of local busy time, 27.3 / 34.4 slot-seconds), not prefill capacity
  gained.
- **Geometry (same child):** **0 tok/s**. One slot buys a 258 048-token clamp
  only with three co-changed knobs and costs a **95 %** steal rate and up to
  **11.5M tokens/day** of extra re-prefill (+26.3 % on the pool); without the
  co-change the warm ceiling stays at 120 000 tokens. The arm A/B itself is
  **unmeasured** — it needs a ≥ 60 % busy window that no recorded window
  provides.
- **Verdict:** *not a share lever.* Quantisation is a decode-latency change with
  an unmeasured quality risk; geometry adds no aggregate throughput and has no
  measurement yet. The ubatch / thread / checkpoint axes keep their own blocked
  owners (LP-0MVNBJXR6002OSPD, LP-0MVNBJY2R004EZ51).

### (d) Accept the ceiling — LP-0MWCEI8QV007NIC0

*Artefact:* [`accept-ceiling-arithmetic-2026-11-01.md`](accept-ceiling-arithmetic-2026-11-01.md).

- **Throughput:** buys **0 tok/s**; it re-scopes the target to what the engine
  already sustains — a delivered blend of 717.83 tok/s and a prefill ceiling of
  **156.2 / 242.25 / 253.8 tok/s** (min / point / max over the regimes the
  traffic actually occupies).
- **Per-slot cost:** **0 GiB, 0 GPU-seconds, 0 config changes**, no summariser
  slot, no model download, no risk added.
- **Share:** endorses **0.41–1.62 % of tokens** and **1.52–5.96 % of streams**
  (planning range min→point; the max corner is labelled an upper bound).
  Parent AC9's 2.8 % is **0.470x** the point capacity — already reachable — while
  **≥ 10 % of streams is 1.677x** the point capacity and **≥ 10 % of tokens is
  6.171x** it (1 495 tok/s, or 95.3 % cache reuse: a different machine).
- **Verdict:** *endorsed as the comparison baseline* every other family is
  measured against.

### The combined bound

All four levers at their best measured end: rate **1.01898x** (quantisation
only), plus ~24.7k freed decode tokens and the compaction conversion — **1.051 %
of tokens** (+0.352 pp) and **3.562 % of streams** (+0.990 pp). That closes
**3.78 %** of the gap to 10 % of tokens and **13.33 %** of the gap to 10 % of
streams; **2.81x** of stream capacity remains missing. No combination of the
spike's levers reaches ≥ 10 % under either denominator.

## 4. AC2 — the share target for parent AC9

Parent **LP-0MVWEZCB0006WPBQ AC9** asks for "local share materially above the
recorded baseline" (2.8 % of streams) without a number. The recommended number,
with its arithmetic:

> **≥ 3.5 % of locally served streams** — equivalently **≥ 0.951 % of offered
> prompt tokens** — over a ≥ 30 min window with ≥ 3 `prefer-local` sessions,
> with AC9's "0 deficit-attributable `local_provider_cooldown` skips" clause
> unchanged.

| Stream share target | Required streams | Delivered prompt tokens | × point capacity | Gate-seconds (own reuse / all-cold) | Idle gate used (all-cold) | Token equivalent | Note |
|---|---:|---:|---:|---|---:|---:|---|
| 2.5727 % | 46.00 | 877 906 | 0.431 | 0 / 0 | 0.0 % | 0.699 % | measured 2026-11-01 window |
| 2.8 % | 50.06 | 955 469 | 0.470 | 94 / 320 | 19.9 % | 0.761 % | parent AC9 baseline |
| 3.0 % | 53.64 | 1 023 717 | 0.503 | 176 / 602 | 37.3 % | 0.815 % | candidate |
| **3.5 %** | **62.58** | **1 194 337** | **0.587** | **382 / 1 306** | **81.0 %** | **0.951 %** | **recommended** |
| 3.7 % | 66.16 | 1 262 584 | 0.620 | 464 / 1 588 | 98.5 % | 1.005 % | prior producer-facing recommendation (LP-0MVXWQQ20002HAX4) |
| 4.0 % | 71.52 | 1 364 956 | 0.671 | 588 / 2 010 | 124.7 % | 1.087 % | candidate |
| 5.0 % | 89.40 | 1 706 195 | 0.838 | 999 / 3 419 | 212.1 % | 1.359 % | candidate |
| 5.9637 % | 106.63 | 2 035 047 | 1.000 | 1 396 / 4 777 | 296.3 % | 1.621 % | point stream ceiling |
| 10.0 % | 178.80 | 3 412 390 | 1.677 | 3 058 / 10 462 | 649.0 % | 2.717 % | prior ≥ 10 % proposal |

"Own reuse" applies the window's 70.771 % cache reuse to the added volume ("all-
cold" assumes every added token is uncached); "idle gate used" divides the
all-cold figure by the window's 1 612 idle prefill gate-seconds, so > 100 %
means the target no longer fits in the gate time the window had spare.

### The arithmetic for 3.5 %

1. **Streams.** 3.5 % × 1 788 offered streams = **62.58 streams**; at the
   measured 19 085 prompt tokens per local stream that is **1 194 337 delivered
   prompt tokens**, i.e. **+316 429** tokens (16.6 streams) over the measured
   877 908 — a **36.0 %** increase in locally delivered prompt tokens.
2. **Capacity.** 1 194 337 / 2 035 052 = **0.587x** the point ceiling — **1.70x
   inside it**. If the gate ran at 100 % prefill duty, delivering that volume
   would need **50.2 %** of it to be cache hits — inside the measured reuse band
   (0.261–0.900) and below the point model's own 0.7077, i.e. strictly inside the
   point ceiling rather than on it.
3. **Gate time.** At the window's own reuse the added volume is 92 489 cold
   tokens = **382 gate-seconds (23.7 % of the 1 612 idle seconds)**; if every
   added token is cold it is **1 306 s (81.0 %)**. It fits either way — so the
   target asks the proxy to *route* work it can already prefill, not to prefill
   faster.
4. **Headroom against 10 %.** ≥ 10 % needs 3 412 390 tokens — **2.86x** this
   target's volume — i.e. 1.677x the point capacity, 406.2 tok/s, or 82.6 % cache
   reuse; 3.5 % needs 1 194 337 tokens and 0.587x capacity.
5. **Why 3.5 % and not the sibling's 3.7 %.** The earlier recommendation
   (LP-0MVXWQQ20002HAX4) was written before the parallel-prefill and
   quantisation measurements landed; at 3.7 % the all-cold gate cost is 98.5 %
   of the window's idle gate — no margin for a cold expansion. 3.5 % leaves
   **19 %** of the idle gate unused and still reads as "materially above" both
   the 2.8 % baseline (**1.25x**) and the measured 2.573 % (**1.36x**).
6. **Why both numbers.** 3.5 % of streams **is** 0.951 % of offered prompt
   tokens — the same volume, expressed in the other denominator. Quote both or
   a reader will assume the 3.68x mix ratio is a discrepancy.

The target also sits below the combined all-lever bound (3.562 % of streams), so
**meeting AC9 requires no lever to be funded**: it is a routing/admission
outcome, which is exactly what the AC9 measurement window is shaped to observe.

## 5. AC3 — recommended next action

**Recommendation: (ii) accept the ceiling and re-scope AC9 to ≥ 3.5 % of
streams.**

1. **Accept the ceiling (primary).** Record the 3.5 %-of-streams target on
   parent LP-0MVWEZCB0006WPBQ (do **not** re-open its completed ACs; the
   recommendation is recorded as a comment, and the parent's AC9 measurement
   children validate it). Rationale: the four families leave the aggregate
   prefill rate where it is, and the all-lever bound (3.562 % of streams /
   1.051 % of tokens) cannot reach ≥ 10 %.
2. **Do not fund a throughput lever for share.** `(b)` parallel prefill measured
   1.00x, `(c)` quantisation ≤ 1.9 % prefill with an unmeasured quality risk, and
   slot geometry 0 tok/s. A ≥ 10 % stream target would need 406.2 tok/s (1.60x
   the best measured rate) or 82.6 % cache reuse.
3. **Optional, bounded: implement family (a) only if the extra +0.92 pp is
   wanted.** That would create two new children of this spike — they are **not**
   created here:
   - `[impl] Co-tune compaction trigger and target to the 25 000-token
     cold-prefill budget` — the only configuration that converts cold prompts;
   - `[verify] Post-change stream-share measurement` over a ≥ 30 min window,
     checking the converted volume against the 1 612 idle gate-seconds it rides
     on.
   Its cost is ~25k-token contexts plus a permanent summariser slot, and it is
   **not** a path to ≥ 10 %.
4. **A follow-up benchmark is not the gate (iii) — with one exception.** No open
   measurement blocks the share decision. The remaining gaps serve levers this
   comparison does not recommend: the slot-geometry arm A/B needs a ≥ 60 % busy
   window; a Q4 prefill/decode arm needs a ~21 GiB GGUF download; the
   ubatch/thread/checkpoint axes have their own blocked children. The exception
   is **quality**: if quantisation is ever pursued for decode latency, gate it on
   a quality benchmark first (89.9 % of the quantised bytes are MoE experts).

**Priority note (intake risk section).** The risk note said to prioritise `(b)`
prefill serialisation and `(c)` model/geometry first if time ran short. Both were
pursued first (LP-0MWCEHDRV000B7HD, LP-0MWCEHYOO002TRGD) and both returned
negative, which is why `(a)` is the only positive family and `(d)` is the
recommended action.

## 6. AC4 — traceability

Every comparison row names the child that produced it and the artefact a reader
can check; the register records the exact source path for each figure.

| Comparison row | Produced by | Artefact | Register field |
|---|---|---|---|
| Measured basis, both denominators, ceilings | LP-0MWCEGJDP009WURD | [`local-share-baseline-2026-11-01.md`](local-share-baseline-2026-11-01.md) | `denominators.*` |
| (a) session size / compaction | LP-0MWCEGXEK000U6G6 | [`compaction-benefit-estimate-2026-11-01.md`](compaction-benefit-estimate-2026-11-01.md) | `families[a_session_compaction].*` |
| (b) prefill serialisation / parallel prefill | LP-0MWCEHDRV000B7HD | [`prefill-concurrency-2026-11-11.md`](prefill-concurrency-2026-11-11.md) | `families[b_parallel_prefill].*` |
| (c) quantisation | LP-0MWCEHYOO002TRGD | [`quantisation-cost-2026-11-11.md`](quantisation-cost-2026-11-11.md) | `families[c_quantisation_geometry].throughput.candidates` |
| (c) ctx/slot geometry | LP-0MWCEHYOO002TRGD | [`single-slot-throughput-baseline-2026-10-31.json`](single-slot-throughput-baseline-2026-10-31.json), [`slot-geometry-ab.md`](slot-geometry-ab.md) | `families[c_quantisation_geometry].per_slot_resource_cost.geometry_*` |
| (d) accept the ceiling, ceilings, AC9/10 % reachability | LP-0MWCEI8QV007NIC0 | [`accept-ceiling-arithmetic-2026-11-01.md`](accept-ceiling-arithmetic-2026-11-01.md) | `families[d_accept_ceiling].*`, `ceiling.*`, `parent_targets.*` |
| Combined lever bound | LP-0MWCEI8QV007NIC0 | [`accept-ceiling-arithmetic-2026-11-01.md`](accept-ceiling-arithmetic-2026-11-01.md) §AC3 | `combined_bound.*` |
| Share target arithmetic | this item | this document, §4 | `share_target.*` |
| Refusal mix / routing vocabulary | LP-0MVXWQQ20002HAX4 | [`routing.md`](../../proxy/docs/routing.md) | cited, not recomputed |

`families[…].register` in the register carries the full repository-relative path
of each source register, and `sources` names all six.

## 7. Caveats and limits

- **The measured basis is one 47-minute window (2026-11-01 13:12:48 →
  14:00:03)** whose raw logs have rotated out; the committed registers are the
  primary evidence, and every ceiling below is an **upper bound** (177 of 1 781
  `Stream finished` lines carry `tokens=missing`).
- **The ceiling band is 12x wide corner-to-corner, and cache reuse is why.** The
  1.62 % / 5.96 % point figures assume the window's 70.8 % reuse scales with
  volume; the marginal refused stream is cold (reuse ≈ 1.0), which is why the
  min corner is below today's delivered share.
- **Compaction's +409k tokens rides on idle gate-seconds** and the summariser's
  GPU time is an estimate (no benchmark was run). In a saturated window the same
  substitution would *lower* the delivered token share.
- **Quantisation's prefill bound is the measured-share end** and has no measured
  quality delta; it cannot be banked without a quality gate.
- **Geometry's arm A/B is unmeasured**, so its "0 tok/s" is an argument (GPU-bound
  throughput) plus a measurement gap, not a measurement.
- **The target is a routing goal measured on a window**, not a config change:
  this item changes nothing in `proxy/`.

## 8. What this item deliberately does not do

- No proxy, routing, config or model change; no new live measurement (AC1/AC2
  are assembly and arithmetic over committed registers).
- No re-opening of parent LP-0MVWEZCB0006WPBQ's completed ACs — the AC2
  recommendation is recorded as a comment on it.
- No implementation child is created: the optional compaction child is named in
  §5 and would be created by the producer if the accept option is not taken.

## Cross-reference

- [LP-0MWCEGJDP009WURD] — [`local-share-baseline-2026-11-01.md`](local-share-baseline-2026-11-01.md): the measured basis and the two fixed denominators.
- [LP-0MWCEGXEK000U6G6] — [`compaction-benefit-estimate-2026-11-01.md`](compaction-benefit-estimate-2026-11-01.md): the compaction conversion, idle-gate bound and summariser slot.
- [LP-0MWCEHDRV000B7HD] — [`prefill-concurrency-2026-11-11.md`](prefill-concurrency-2026-11-11.md): the engine rates and the parallel-prefill rejection.
- [LP-0MWCEHYOO002TRGD] — [`quantisation-cost-2026-11-11.md`](quantisation-cost-2026-11-11.md): the quantisation bound, decode slot-seconds and geometry verdict.
- [LP-0MWCEI8QV007NIC0] — [`accept-ceiling-arithmetic-2026-11-01.md`](accept-ceiling-arithmetic-2026-11-01.md): the ceiling tiers, target reachability and the combined lever bound.
- [`proxy/docs/routing.md`](../../proxy/docs/routing.md) — the routing/admission reference for the refusal classes the reuse discussion leans on.

[LP-0MWCEGJDP009WURD]: ../../docs/dev/local-share-baseline-2026-11-01.md
[LP-0MWCEGXEK000U6G6]: ../../docs/dev/compaction-benefit-estimate-2026-11-01.md
[LP-0MWCEHDRV000B7HD]: ../../docs/dev/prefill-concurrency-2026-11-11.md
[LP-0MWCEHYOO002TRGD]: ../../docs/dev/quantisation-cost-2026-11-11.md
[LP-0MWCEI8QV007NIC0]: ../../docs/dev/accept-ceiling-arithmetic-2026-11-01.md
