# Accept-the-ceiling arithmetic (LP-0MWCEI8QV007NIC0)

Costing of the spike's **option (d): accept the ~5.8 % ceiling and re-scope the
target**. Spike parent LP-0MVXY8ZBZ0003042, AC1d.

This page is the arithmetic the spike's AC1 comparison table cites, derived
entirely from committed registers by
[`scripts/compute_accept_ceiling.py`](../../scripts/compute_accept_ceiling.py)
(a pure analyser — no proxy behaviour change, no new measurement, no running
proxy required). The machine-readable register is
[`accept-ceiling-arithmetic-2026-11-01.json`](accept-ceiling-arithmetic-2026-11-01.json);
every figure below is a field in it.

## Headline

| Question | Answer |
|---|---|
| What can the hardware prefill? | **0.31 / 0.47 / 0.50 %** of the offered prompt tokens (min / point / max) = 384 k / 595 k / 623 k tokens per 47.25 min window |
| What local prompt-token share does that support? | **0.41 / 1.62 / 4.96 %** once cache hits are counted (min / point / max) |
| And as stream share? | **1.52 / 5.96 / 18.26 %** |
| Is the parent AC9 2.8 % stream target reachable? | **Yes** — 0.470x the point capacity |
| Is the ≥10 % stream target reachable? | **Only in the optimistic corner** — 1.677x the point capacity; needs 406 tok/s (1.60x the measured max) or cache reuse 0.826 |
| Is ≥10 % prompt-token share reachable? | **No** — 6.17x the point capacity; needs 1 495 tok/s (5.89x the measured max) or cache reuse 0.953 |
| Does the lever set close the gap to ≥10 %? | **No** — combined bound closes 3.8 % of the token gap and 13.3 % of the stream gap |

**Verdict:** the "accept" option is defensible on measured evidence. The
ceiling is a hardware property (one prefill gate at 156-254 aggregate tok/s),
the levers do not move it, and the only target it supports comfortably is the
one AC9 already set (2.8 % of streams).

## AC1 — the hardware limit, the formula and its inputs

### The model

```
T_prefill = window_seconds x (1 - decode_share)          # gate time available to prefill
C_cold    = T_prefill x R_prefill                        # prompt tokens actually prefilled
C_del     = C_cold / (1 - r)                             # delivered prompt tokens
share     = 100 x C_del / denominator                    # token or stream denominator
```

The three tiers answer three different questions and are never interchanged:

- **tier 1 — hardware prefill ceiling:** `100 x C_cold / offered_prompt_tokens`,
  the share of the offered prompt tokens the gate can *originate*;
- **tier 2 — delivered-share ceiling:** `100 x C_del / offered_prompt_tokens`,
  the same capacity with cache hits counted, which is what "local share" has
  always meant;
- **tier 3 — stream-denominated equivalent:** the same `C_del` divided by the
  current local mean prompt size per stream, over the offered streams.

The denominator is fixed by the verification child
[LP-0MWCEGJDP009WURD]: **offered prompt tokens** for the token share and
**offered streams** for the stream share. Its reconciliation ruled out the two
rival denominators that would have hidden the difference — an *uncached* token
denominator would have to be 27.8 % of offered (cached_ratio 0.72) against a
project median of 0.952, which would put the ceiling near 34 %, and *peak*
offered load is not used by either published figure.

### Inputs (all measured, none taken here)

| Input | Value | Source |
|---|---|---|
| Window | 2026-11-01 13:12:48 → 14:00:03 = 2 835 s | `local-share-baseline-2026-11-01.json` |
| Offered prompt load | 125 577 780 tokens, 1 788 streams = **44 295.5 tok/s** | same |
| Delivered prompt tokens | 877 908 in 1 223 busy s = **717.83 tok/s** (blend) | same |
| Local mean prompt per stream | 19 084.96 tokens | same |
| Local decode slot-seconds | 5 027 completion tokens / 30.7 tok/s = 163.7 s | `quantisation-cost-2026-11-11.json` |
| Decode reserve | 163.7 / 1 223 = **13.389 %** of gate time | derived |
| **Prefill gate time** | 2 835 x (1 - 0.13389) = **2 455.42 s** | derived |
| Engine prefill rate — **min** | **156.2 tok/s** (sustained serial prefill at 44 770-token prompts, engine probe under live load) | `prefill-concurrency-baseline-2026-11-11.json` (E1) |
| Engine prefill rate — **point** | **242.25 tok/s** (mean of the re-measured 20k-40k aggregate: 241.0 / 243.5) | same (P1, P2) |
| Engine prefill rate — **max** | **253.8 tok/s** (W3 solo-prefill p50, the best aggregate measured for this engine) | `single-slot-throughput-baseline-2026-10-31.json` (W3) |
| Cache reuse — **min** | **0.261** (measured aggregate prefix reuse; what reuse looks like across all traffic, cold large prompts included) | same (W4) |
| Cache reuse — **point** | **0.7077** (implied by the window's own occupancy, below) | derived |
| Cache reuse — **max** | **0.900** (admitted-population warmth: new ≈ 10 % of the prompt; an upper bound, a selection effect of residency admission) | `compaction-benefit-estimate-2026-11-01.json` |

The implied point reuse is the only derived input, and it is the gate's own
occupancy: the window used `1 223 s x (1 - 0.13389) = 1 059.25 s` of prefill
time, which at 242.25 tok/s prefills `256 601` tokens; the route delivered
`877 908`, so `r = 1 - 256 601 / 877 908 = 0.7077`.

### Why the measured duty cycle is not part of the ceiling

The work item asks for the hardware limit from the measured engine evidence,
including the W1/W2 busy percentages. Those percentages enter the arithmetic as
the **headroom the ceiling extrapolates across**, not as a term in it:

| Evidence | Value | Role |
|---|---|---|
| Spike-window duty | 1 223 / 2 835 s = **43.14 %** | headroom factor 2.3181x — what the ceiling extrapolates away |
| W1 duty (24 h) | **10.2 %** busy, 89.8 % idle | the engine is idle most of the time on a normal day; capacity, not demand, is the constraint |
| W2 duty (24 h) | **18.5 %** busy, 81.5 % idle | same |
| W1 average concurrency while busy | **1.04** (peak 2) | confirms one prefill gate is the resource, so a single gate's rate is the right ceiling basis |
| W3 paired aggregate | **254 tok/s** (solo p50 253.8) | the max engine rate used above |

A ceiling is a 100 %-duty statement, so duty is divided out; if duty were left
in, the "ceiling" would be a demand forecast and would fall whenever traffic
fell. The single gate is confirmed as the resource by the 1.04 average
concurrency while busy: the engine is not throughput-limited by concurrency it
cannot get, it is limited by the gate it runs at.

### Result

**Tier 1 — hardware prefill ceiling (share of offered prompt tokens):**

| Scenario | Engine rate | Prefill tokens | Share of offered |
|---|---|---|---|
| min | 156.2 tok/s | 383 537 | **0.305 %** |
| point | 242.25 tok/s | 594 826 | **0.474 %** |
| max | 253.8 tok/s | 623 186 | **0.496 %** |

So the single gate originates at most **~0.5 % of the offered prompt tokens**
per window; `C_cold / window = 135.3 / 209.8 / 219.8 tok/s` against an offered
44 295.5 tok/s.

**Tiers 2 and 3 — delivered tokens and streams:**

| Scenario | Reuse | Delivered tokens | Token share | Servable streams | Stream share |
|---|---|---|---|---|---|
| min | 0.261 | 518 994 | **0.413 %** | 27.2 | **1.521 %** |
| point | 0.7077 | 2 035 052 | **1.621 %** | 106.6 | **5.964 %** |
| max | 0.900 | 6 231 861 | **4.963 %** | 326.5 | **18.262 %** |

### Reconciliation with the ceiling of record

The point row reproduces the committed ceiling **exactly** —
2 035 052 delivered tokens against `ceiling_tokens = 2 035 052.48` in
`local-share-baseline-2026-11-01.json` (0.0000 % delta), 1.621 % token share
and 5.964 % stream share.

That equality is **algebraic, not a validation**: once `r` is implied from the
window's occupancy, `R_prefill` and the decode reserve cancel and the model
collapses to the duty-cycle extrapolation
`window / busy_seconds x delivered_tokens`. The consequence is the finding,
not a coincidence:

> The 1.62 % / 5.96 % ceiling of record **assumes the window's 70.8 % cache
> reuse scales with the volume**. On the marginal stream it does not — the
> refusal decomposition says the refused population is cache-cold (reuse ≈ 1.0:
> `large_context_bypass` 32.8 % and `resident_prefix_insufficient` 24.7 % are
> precisely "there is no resident prefix").

Two further readings of the band:

- **Reuse, not gate duty, is the dominant term.** Holding the point rate and
  moving only reuse across its measured range moves the delivered ceiling from
  0.641 % to 4.737 % (**7.39x**); holding the point reuse and moving only the
  engine rate across its whole measured band moves it 156.2 → 253.8 tok/s
  (**1.62x**). Both are recorded as a check in the register.
- **The min corner (0.413 %) is below today's delivered share (0.699 %)** — and
  that is the point, not an error: at the measured *aggregate* reuse (26.1 %) a
  100 %-duty gate delivers less than today's 43 %-duty gate does at 70.8 %
  reuse. A ceiling statement without its reuse assumption is not a ceiling.

The band is wide (12x corner to corner) because reuse and rate are not
independent in reality: a stream only gets cheap to serve once it is resident.
The **defensible planning range is min-to-point (0.41-1.62 %)**, and the max
corner is labelled an upper bound rather than a target.

## AC2 — what target is reachable, and what ≥10 % would take

Each target is computed in its **own** denominator; the two are never mixed.

| Target | Requirement | vs point capacity | What it would take | Verdict |
|---|---|---|---|---|
| Parent AC9: ≥2.8 % of streams | 955 469 tokens | **0.470x** | already inside the measured ceiling (113.7 tok/s, or reuse 0.378) | **reachable** — today's window delivered 2.573 % |
| LP-0MVXWQQ2 AC1: ≥10 % of streams | 3 412 390 tokens | **1.677x** | 406.2 tok/s (1.60x the measured max) **or** cache reuse 0.826 | only in the optimistic corner (90 % reuse at the best rate) |
| ≥10 % of prompt tokens | 12 557 778 tokens | **6.171x** | 1 494.9 tok/s (5.89x the measured max) **or** cache reuse 0.953 | **unreachable** |

Reading:

- **Parent AC9 (2.8 % of streams) is the target the ceiling supports.** It is
  47 % of the point stream ceiling, and the measured window already delivered
  2.573 % — i.e. AC9 is a "hold the line" target, not a growth target.
- **≥10 % of streams is 1.68x the point capacity.** It is inside the measured
  band only at the corner where every served stream is 90 % cache-warm *and*
  the engine runs at its best measured rate — an assumption admission cannot
  sustain as volume grows, because serving more streams means admitting colder
  ones.
- **≥10 % of prompt tokens is not a target, it is a different machine.**
  12.56 M tokens/window is 6.17x the point capacity: it needs 1 495 tok/s of
  aggregate prefill (5.9x the best ever measured on this engine) or 95.3 %
  cache reuse. No lever family in the spike approaches either.
- **The exact algebra for a re-scope.** Raising the stream ceiling to 10 %
  needs `delivered ≥ 3 412 390`, i.e. `C_cold ≥ 997 408` at reuse 0.7077, i.e.
  `R ≥ 406.2 tok/s` — or, at the point rate, `r ≥ 0.8257`.

## AC3 — the levers' combined upside, and whether it closes the gap

The four lever children are priced together, once, at their most favourable
measured end:

| Lever | Measured effect | Applied to the ceiling | Owner |
|---|---|---|---|
| Session compaction (cap at the 25 000-token cold-prefill budget) | +409 126 delivered tokens (16.4 extra streams) by converting idle gate seconds | additive: x1.0190 with the faster rate below | LP-0MWCEGXEK000U6G6 |
| Parallel prefill | aggregate **1.00x** (wall 2.00x, per-request 0.81x) — the gate is GPU memory-bandwidth bound, not serialisation bound | **0** | LP-0MWCEHDRV000B7HD |
| Quantisation (UD-Q4_K_S, measured-share end) | prefill **+1.898 %** in the 20k-40k bucket (the byte-ratio ceiling of 26.6 % is unreachable: a prefill pass streams weights at 6.3 GB/s against the 88 GB/s decode already sustains) | x1.01898 on the gate rate, plus 24 745 tokens of freed decode slot-seconds | LP-0MWCEHYOO002TRGD |
| Context/slot geometry | no aggregate throughput (one slot trades concurrency and prefix reuse at a 95 % steal rate, not tok/s) | **0** | LP-0MWCEHYOO002TRGD |

**Bound: the combined lever set is `x1.0190` on the gate rate plus at most
~24.7 k freed decode tokens — it does not raise the ceiling, it converts
existing headroom.**

| Denominator | Today | After all levers | Gap to ≥10 % | Gap closed | Remaining factor |
|---|---|---|---|---|---|
| prompt tokens | 0.6991 % | **1.0508 %** | 9.30 pp | 0.352 pp = **3.78 %** | **9.52x** |
| streams | 2.5727 % | **3.5624 %** | 7.43 pp | 0.990 pp = **13.33 %** | **2.81x** |

Two caveats that bound the bound:

- Compaction's +409 k tokens rides on the window's **idle** gate seconds
  (1 612 s, 56.9 % of the window). It does not raise the ceiling, and in a
  saturated window the same substitution would lower the delivered token share
  (a cold token costs 2.83x the gate time of a blended delivered token).
- Neither lower-quant candidate has a measured local quality delta (89.9 % of
  the quantised bytes are MoE expert weights), so the quantisation term is an
  upper bound that cannot be banked without a quality gate.

So: **no combination of the spike's levers closes the gap to ≥10 % under either
denominator.** They get a third of the way on streams and 4 % of the way on
tokens. That is the comparison the accept option needs.

## AC4 — throughput and per-slot resource cost of the "accept" option

| Field | Value |
|---|---|
| Throughput (measured ceiling) | 717.83 tok/s delivered blend; prefill ceiling 156.2 / 242.25 / 253.8 tok/s |
| Throughput added by this option | **0 tok/s** — it buys no capacity, it re-scopes the target to the measured engine |
| Per-slot resource cost | **none** — 0 GiB GPU memory, 0 GPU-seconds, 0 config changes, no summariser slot |
| Local share the option endorses | 1.62 % of prompt tokens / 5.96 % of streams (point), 0.41-1.62 % planning range |
| Risk | none new; the option is the status quo, so the spike's risk sits with the levers instead |

For the spike's AC1 table this row is the comparison baseline every other row
is measured against: the other three options must show what they add **over a
0 tok/s, 0 GiB row** — and the bound above is what they add in total.

## Assumptions and limits

- **The measured basis is the 2026-11-01 window**, as the spike's constraint
  requires; the raw logs have rotated out, so the committed registers are the
  primary evidence (recorded by LP-0MWCEGJDP009WURD).
- **The decode reserve is a share of gate time (13.389 %), not an absolute.**
  That is what makes the 47-minute window scalable to 100 % duty: decode volume
  grows with the served stream count, so its share is the scale-invariant form.
- **The engine rates are aggregate production rates**, not a dedicated
  benchmark: the min corner (156.2 tok/s) and the point corner (242.25 tok/s)
  were measured while live traffic shared the GPU, so they are conservative
  relative to a quiet engine. The max corner (253.8 tok/s) is the W3 solo p50.
- **Reuse is the decisive unknown.** It is measured three ways (aggregate
  26.1 %, implied by the window 70.8 %, admitted-population warmth 90 %) and
  the whole 12x width of the ceiling band comes from it. Any future restatement
  of the ceiling must state its reuse assumption.
- **The 2 820 s published ceiling** and the window's 2 835 s differ by 0.53 %;
  this page uses the window (2 835 s) because the basis register does, and the
  reconciliation against the published 2 820 s figure is carried by
  LP-0MWCEGJDP009WURD.
- **Read-only:** no proxy, routing or config change; four new files (script,
  register, this page, tests).

## Reproducing

```bash
# Recompute from the committed registers (read-only, no proxy needed)
python3 scripts/compute_accept_ceiling.py

# Machine-readable register
python3 scripts/compute_accept_ceiling.py --json

# Self-contained drift check of the committed register
python3 scripts/compute_accept_ceiling.py \
  --register docs/dev/accept-ceiling-arithmetic-2026-11-01.json --check
```

The register embeds every input with its provenance and every derived figure,
so `--register ... --check` re-derives the whole artefact from its own inputs
without touching the source registers.

## Cross-reference

- [LP-0MWCEGJDP009WURD] — `local-share-baseline-2026-11-01.{md,json}`: the
  measured basis and the two fixed denominators this page builds on.
- [LP-0MWCEHDRV000B7HD] — `prefill-concurrency-2026-11-11.md`: the engine rates
  and the parallel-prefill rejection (1.00x).
- [LP-0MWCEGXEK000U6G6] — `compaction-benefit-estimate-2026-11-01.md`: the
  compaction lever's added tokens, idle-gate bound and summariser slot cost.
- [LP-0MVCEHYOO002TRGD] — `quantisation-cost-2026-11-11.md`: the quantisation
  prefill bound, the decode slot-seconds freed and the geometry verdict.
- [`single-slot-throughput-baseline-2026-10-31.json`](single-slot-throughput-baseline-2026-10-31.json)
  — W1/W3/W4 figures (engine rates, prefix reuse, slot geometry).
- [`prefill-admission-conversion.md`](prefill-admission-conversion.md) — the
  admission→first-byte conversion record. It prices *conversion* (how much
  queued demand actually becomes a local dispatch), not capacity: it does not
  enter the ceiling arithmetic, but it is the mechanism by which a cold,
  queued prompt becomes the marginal stream this page prices.
- `proxy/docs/routing.md` — the routing/admission reference for the refusal
  classes the reuse discussion leans on.

[LP-0MWCEGJDP009WURD]: local-share-baseline-2026-11-01.md
[LP-0MWCEHDRV000B7HD]: prefill-concurrency-2026-11-11.md
[LP-0MWCEGXEK000U6G6]: compaction-benefit-estimate-2026-11-01.md
[LP-0MVCEHYOO002TRGD]: quantisation-cost-2026-11-11.md
