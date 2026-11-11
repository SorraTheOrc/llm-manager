# Session-compaction benefit for local prefill share — costed estimate

**Work item:** LP-0MWCEGXEK000U6G6 (spike parent LP-0MVXY8ZBZ0003042, question
AC1a — "Session size/compaction: median request is 57k new tokens, top decile
>100k. Estimate compaction benefit.").
**Status:** read-only analysis. No proxy routing, config or behaviour change is
part of this document or its artefact.

**Artefact:** [`compaction-benefit-estimate-2026-11-01.json`](compaction-benefit-estimate-2026-11-01.json)
— the machine-readable register behind every figure quoted here. It embeds its
own inputs (per-class size grids, cache density, gate mix, guard/routing
constants), so it can be re-verified after the source CSV rotates out.

**Tooling:** [`scripts/estimate_compaction_benefit.py`](../../scripts/estimate_compaction_benefit.py),
pinned by [`tests/test_estimate_compaction_benefit.py`](../../tests/test_estimate_compaction_benefit.py)
(admission boundaries, the cap-conversion claim, both ceilings, register
drift).

**Regenerate** (needs the window CSV, which lives outside the repo):

```bash
python3 scripts/estimate_compaction_benefit.py \
    --sessions-csv ~/proxy-usage-reports/postfix-thrash-gate/standard_sessions.csv \
    --basis docs/dev/local-share-baseline-2026-11-01.json \
    --json > docs/dev/compaction-benefit-estimate-2026-11-01.json
```

**Re-verify the committed register** (self-contained):

```bash
python3 scripts/estimate_compaction_benefit.py \
    --register docs/dev/compaction-benefit-estimate-2026-11-01.json --check
```

---

## 1. Answer in one paragraph

Compaction does not make the prefill gate faster: it rewrites a session's
prompt to a smaller one, so it can only change the guard's **size** tests.
Those tests bind in a specific order, and that order decides the lever's value.
The refused population is essentially **cache-cold**, so `new_tokens` equals the
prompt size and the flat `local_cold_prefill_budget_tokens: 25000` test clears
only for a cap **at or below 25 000 tokens**. At that cap the cold refused
population becomes admissible and the window gains **+409 126 local prompt
tokens (+0.33 pp, 0.699 % → 1.025 % of offered prompt tokens; stream share
2.573 % → 3.488 %)** — bounded not by the 1.62 % capacity ceiling this time,
but by the gate's **1 612 idle seconds** in the window (the demand is 28 141
gate-seconds, so only 16.4 of the 285.7 newly admissible requests can actually
be served). At the intake's 57k cap and at the **shipped compaction targets
(38k fast / 30k cheap) the gain is exactly zero** for the cold population: those
prompts are still over the 25 000-token flat cap (and, above the 38 000-token
always-local band, still behind the residency ladder). Enabling compaction at
today's targets would therefore pay the summariser cost without admitting a
single cold prompt; the lever only pays if the target is co-lowered to the
cold-prefill budget.

## 2. Inputs and provenance

| Input | Value used | Source |
|---|---|---|
| Window | 2026-11-01 13:12:48 → 14:00:03 (**2 835 s**), offered **125 577 780** prompt tokens, 1 788 requests | `docs/dev/local-share-baseline-2026-11-01.json` (report of record `~/proxy-usage-reports/postfix-thrash-gate/report.md`) |
| Measured local basis | 877 908 local prompt tokens in 1 223 busy s (**717.8 tok/s** blended), 46 local streams | same |
| Capacity ceiling | **2 035 057** tokens/window = **1.621 %** token share / 5.964 % stream share | same (baseline child's derivation) |
| Prompt-size distribution | 1 649 sized requests (52 of 58 sessions): mean **77 029**, p50 **60 715**, p90 **121 789**, p95 158 274, p99 219 388 | `standard_sessions.csv`, reconstructed as `messages × avg_context_size` per session; reproduces the report's offered prompt tokens to **+1.15 %** |
| Refused-population cache density | 41 000 of 47 722 skip decisions **cold** (reuse = 1.0); p50 reuse 1.0 in every size band | retained `/var/log/llama-proxy` segments 2026-11-10 20:00 → 2026-11-11 20:00 (`routing_skip_local` lines carry `estimated_tokens` + `cached_ratio` + `reason`); the 2026-11-01 raw logs have rotated out |
| Residency evidence | 413 of 417 residency denials in the window had **no local slot at all** | report § *Residency-aware admission* |
| Gate-state mix | **71.6 %** of guard decisions taken with a free prefill gate | retained-window `local_admission_guard` lines (`gate_occupancy=0`, n = 31 958) |
| Guard constants | flat cap **25 000** new tokens, deadline 180 s, safety fraction 0.95 → **171 s**, queue budget **30 s**, free-gate rung **on**, policy `reject`, 3.9 ms/new token, 0.5 ms/cached token | `proxy/config.yaml` (`local_cold_prefill_budget_tokens`, `local_prefill_deadline_seconds`, `local_admission_guard_*`, `contention_queue_max_wait_seconds`), `proxy/proxy/prefill_eta.py` |
| Routing size tests | always-local ≤ **38 000**; economic warm cap **120 000** (`0.4578 × 262 144`); residency ladder above the band | `proxy/config.yaml` (`local_large_context_cold_cache_threshold`, `local_large_context_warm_cache_threshold`, `local_large_context_economic_ceiling_ratio`), `proxy/proxy/provider.py` |
| Engine rates | 253.8 tok/s solo (20–40k), 156.2 tok/s at 44 770 tokens (40–80k), 160.0 tok/s aggregate | `docs/dev/prefill-concurrency-2026-11-11.md` (sibling LP-0MWCEHDRV000B7HD) |

**Why the cache density comes from a different window.** The refused population
is not observable in dispatch lines (a refused prompt never dispatches), and the
2026-11-01 raw log is gone. The 2026-11-10/11 retained window is the only
evidence that carries `estimated_tokens` **and** `cached_ratio` for *skipped*
requests, and it is decisive: 86 % of those decisions are `cache_state=cold`,
with reuse p50 = 1.0 in every size band. That is not a modelling convenience —
a session with no resident slot is denied by the residency ladder *before* any
residency measurement, and `context_too_large` is decided before the ladder at
all. The window's own residency counters agree (413 of 417 denials had no slot).

## 3. AC1 — the prefill-vs-size curve

The guard's own estimator prices `coded × 0.5 ms + new × 3.9 ms`. With the
refused population cold (reuse 1.0) that is simply 3.9 ms per token:

| Prompt | Cold, guard coefficients | Cold, engine 253.8 tok/s | Cold, engine 156.2 tok/s (40–80k regime) | Cache-warm (reuse 0.10) |
|---|---:|---:|---:|---:|
| **Median** 60 715 | **236.8 s** (= 60 715 × 3.9 ms) | 239.2 s | 388.7 s | 51.0 s (= 60 715 × 0.84 ms) |
| **Top decile** 121 789 | **475.0 s** | 479.9 s | 779.7 s | 102.3 s |

Two readings matter for the spike's comparison:

- The guard's 3.9 ms/token and the engine's measured solo 253.8 tok/s
  (3.94 ms/token) agree to 1 %, so the priced curve is honest at the 20–40k
  regime the compacted prompts would land in.
- The measured **40–80k** rate is 156.2 tok/s — the regime the *uncapped*
  median/p90 prompts sit in. Prefill throughput falls by ~1.6× once a prompt
  passes 40k tokens (memory-bandwidth bound, per the sibling measurement), which
  is exactly why a size cap shows up as extra *capacity*, not extra rate.

Same curve after a cap (cold): 25 000 → **97.5 s**, 38 000 → 148.2 s,
57 000 → 222.3 s. A 25k cap makes every request cost less prefill than today's
median prompt does.

## 4. AC2 — costing the candidate thresholds

The model is three tests, applied per request of size `s` (compacted to
`s' = min(s, T)`):

1. **Routing size tests** — `s' > 120 000` is `context_too_large`; `s' ≤ 38 000`
   is always local; between the two, a cold prompt is denied by the residency
   ladder (`resident_prefix_insufficient`) because it has no slot to price
   against.
2. **Flat cold-prefill cap** — refuse when `s' × reuse > 25 000`
   (calibration-independent, checked first).
3. **Queue/eta bounds** — free gate: own eta ≤ 171 s; busy serialised gate:
   projected wait `2 × own eta ≤ 30 s`.

The flat cap really is the dominant refusal: the conversion harness records
3 557 of 3 954 guard rejects (90 %) as `prefill_budget_exceeded` in the
2026-10-27/28 window ([`prefill-admission-conversion.md`](prefill-admission-conversion.md)),
and the retained 2026-11-10/11 window shows 15 654 of 20 045 (78 %).

Admissibility is averaged over the measured gate mix (71.6 % free / 28.4 %
busy), and the resulting volume is capped twice: by the gate's idle seconds in
the window (1 612 s) and by the measured 2 035 057-token capacity ceiling.

### Cold refused population (primary case)

| Cap | Newly admissible (of 399 sized requests in the three size classes; 477 requests in total) | Served (gate-bound) | Added prompt tokens | Local token share | Local stream share |
|---|---:|---:|---:|---:|---:|
| uncapped (shipped) | 0.0 | 0.0 | 0 | 0.699 % | 2.573 % |
| **57 000** (intake median cap) | **0.0** | 0.0 | 0 | **0.699 %** | **2.573 %** |
| 38 000 (shipped fast target) | 0.0 | 0.0 | 0 | 0.699 % | 2.573 % |
| 30 000 (shipped cheap target) | 0.0 | 0.0 | 0 | 0.699 % | 2.573 % |
| **25 000** (cold-prefill budget) | **285.7** | **16.4** | **+409 126** | **1.025 %** | **3.488 %** |

Arithmetic for the 25 000 row, in the open:

- A cold 25 000-token prompt has `new_tokens = 25 000`, which the flat cap
  admits (it refuses on `>` only); own eta = 25 000 × 3.9 ms = **97.5 s** ≤ 171 s
  → admitted when the gate is free. Behind a busy serialised gate the projected
  wait is 2 × 97.5 = 195 s > 30 s, so it is refused → admitted share =
  0.716 × 399 = 285.7 requests.
- Their demand is 285.7 × 25 000 = **7 142 100 tokens**, needing
  7 142 100 / 253.8 tok/s = **28 141 gate-seconds**. The window has
  2 835 − 1 223 = **1 612 idle gate-seconds**, so only
  1 612 × 253.8 / 25 000 = **16.4** can be served → **+409 126 tokens**.
- Share: (877 908 + 409 126) / 125 577 780 = **1.025 %**; streams
  (46 + 16.4) / 1 788 = **3.488 %**.

The 25 000 boundary is exact by construction: `new_tokens = 25 000` passes,
`25 001` is refused (`prefill_budget_exceeded`). The shipped 30k/38k targets and
the 57k cap all sit above it, so every cold prompt they produce is still
refused — the summariser bill with none of the conversion.

### Sensitivity: if the refused population were cache-warm (reuse 0.10)

Warm prompts are already admissible at sizes far above any candidate cap, so a
cap adds little there; the only converted slice is the `> 120 000`
`context_too_large` band, which any cap ≤ 120k lifts (115.3 requests):

| Cap | Newly admissible | Served | Token share | Stream share |
|---|---:|---:|---:|---:|
| 57 000 | 115.3 | 4.4 | 0.900 % | 2.820 % |
| 38 000 | 115.3 | 10.8 | 1.025 % | 3.175 % |
| 30 000 | 115.3 | 13.6 | 1.025 % | 3.335 % |
| 25 000 | 115.3 | 16.4 | 1.025 % | 3.488 % |

(A warm 57k prompt prices at 57 000 × 0.10 = 5 700 new tokens, so it passes
the flat cap; it is served at the slower 40–80k engine rate of 156.2 tok/s,
which is why the 57k row converts less volume in the same idle seconds.)

### The second ceiling: a busy gate admits ~3 846 cold tokens

With prefills serialised (`local_max_concurrent_prefills: 1`) a queued request
projects `2 × own eta` against the 30 s queue budget, so the admission ceiling
behind a busy gate is 30 s / 2 / 3.9 ms = **3 846 cold tokens** — **6.5×
tighter than the 25 000-token flat cap**. No usable context size reaches it, so
the busy gate cannot be cleared by compaction at all; the lever's gain rides
entirely on the free-gate share (71.6 % measured).

## 5. AC3 — throughput and per-slot cost

**Throughput attributable to the lever: 0 tok/s of new prefill rate.** The gate
stays at its measured 253.8 tok/s (20–40k) / 156.2 tok/s (40–80k). What the
lever delivers is *converted idle gate time*:

| Quantity | Value |
|---|---:|
| Idle prefill-gate seconds in the window | **1 612 s** of 2 835 s (56.9 %) |
| Converted volume at the 25k cap | **409 126 prompt tokens** |
| Delivered-rate attribution (volume / window) | **144.3 tok/s** aggregate |
| Demand the lever would like to serve | 28 141 gate-seconds (17× the idle time) |
| Blended cost of a today-token vs a cold token | 717.8 / 253.8 = **2.83×** |

The 2.83× is the honest caveat: a cold 25k prompt consumes 2.83× the gate time
per delivered token that today's blended mix does (mostly cached re-sends), so
the +0.33 pp is real **only while gate time is idle**. In a saturated window
the same substitution would *lower* the delivered token share; compaction buys
no capacity that the GPU does not already have.

**Per-slot resource cost (summariser slot, `Qwen2.5-7B`):**

| Cost | Value | Derivation |
|---|---:|---|
| Weights (Q4_K_M) | ~4.5 GB | `models.ini` `[Qwen2.5-7B]` rationale |
| KV cache at ctx 32 768 (fp16) | **1.75 GiB** | 2 × 28 layers × 4 KV heads × 128 dim × 2 B = **57 344 B/token**, × 32 768 |
| **Total VRAM** | **≈ 6.25 GB (5.2 % of 120 GB)** | holds for the server's lifetime (`parallel = 1`) |
| GPU time per compaction (estimate) | **24.9–45.6 s** | prompt prefill 8 192 tokens at 250–500 tok/s (16.4–32.8 s) + ≤512 output at 40–60 tok/s (8.5–12.8 s) |
| Documented typical latency | < 2 s | `models.ini` comment (short summaries) |
| Slot occupancy | one dedicated slot, `parallel = 1` | `models.ini`; isolates the summariser from the main slot |

The summariser shares the GPU, so its 24.9–45.6 s **competes with the prefill
gate** it is meant to relieve. At 25k and 18 size-refused sessions the
compaction bill is ~450–820 GPU-seconds, against a 1 612 s idle budget — which is
why the gate-bound 16.4 served streams, not the demand, is the right headline.

## 6. AC4 — the compaction state being costed

| Knob | Shipped | Consequence for this estimate |
|---|---|---|
| `compaction_trigger_ratio` | **0** (disabled) | The assembly-time trigger — the only one that shrinks a **local** prompt — is off. Restoring `0.70` would trigger at `0.70 × 131 072 = 91 750` estimated tokens. |
| `compaction_dry_run` | `false` | Enforcement was already live when the operator disabled it, so re-enabling is **live compaction**, not an advisory. |
| Targets | fast **38 000** / cheap **30 000** tokens | Both sit **above** the 25 000 cold-prefill budget → zero conversion of cold prompts (§4). "Enable compaction" and "raise/lower a threshold" are orthogonal: the trigger decides *when* to summarise, the target decides *how small*, and only a target ≤ 25 000 clears the flat cap. |
| `remote_boundary_compaction` | **false** (min 20 000 tokens) | The second, **independent** trigger. It summarises a session's oldest turns on its way to a **remote** provider, cutting remote input-token spend; it never shrinks a local dispatch, so it does not change local prefill share and is out of scope here. |
| `summarizer_model` | `Qwen2.5-7B` (`parallel = 1`, ctx 32 768, max output 512) | The per-slot cost in §5. Isolated from the main Qwen3.6 slot by design (LP-0MTXCQA8I0038J4X). |

**So the two statements the spike must keep apart:** "enable compaction"
(flip `compaction_trigger_ratio` to 0.70 — pays the summariser cost at a 38k
target, converts **nothing** in this window) versus "lower the target to the
cold-prefill budget" (the only configuration that converts cold prompts, at the
cost of ~25k-token contexts and a summarisation per large session).

## 7. Assumptions, limits, and what would change the answer

- **The 57k median is confirmed, the "new tokens" label is not.** The window's
  median *prompt* is 60 715 tokens and p90 is 121 789 — consistent with the
  intake's 57k/>100k. Those are prompt sizes, not new tokens: for the
  *admitted* population 90 % of prompts are cache-warm (new tokens ≈ 10 % of the
  prompt), while the *refused* population is cold (new ≈ prompt).
- **Cache density is transferred** from the 2026-11-10/11 retained window. If
  the 2026-11-01 refused population were materially warmer, the 57k cap would
  convert the `> 120 000` band (115.3 requests, §4 sensitivity) — still not the
  cold headline case.
- **Gate-state mix is measured, not modelled per request.** 71.6 % free / 28.4 %
  busy is an average over 31 958 retained decisions; the window's own free-gate
  share (the intake's "76 of 314" guard refusals) is not re-verifiable because
  those log lines have rotated out (the baseline child records the same
  limitation).
- **Size reconstruction is session-granular** (`messages × avg_context_size`);
  139 of 1 788 requests (6 sessions) carry no token evidence and are excluded
  rather than counted as zero. The reconstruction overstates the offered load's
  mean by ~10 % (77 029 vs 70 234) because the excluded sessions are small.
- **Only size-caused refusals are costed.** `local_concurrency_limit`,
  `local_lease_active`, `tenant_limit` and the unattributed bucket (1 311 of
  1 788 requests) are not convertible by a prompt cap and keep the measured
  baseline unchanged.
- **The summariser cost is estimated**, not measured: no GPU benchmark was run
  (the spike is read-only and the run would compete with live traffic). The
  250–500 tok/s prefill and 40–60 tok/s decode ranges are plausible 7B Q4 rates
  on this iGPU, not measurements; the documented "< 2 s typical" covers the
  short-summary case.
- **Nothing here changes a threshold.** If the spike recommends the 25k lever,
  the change is a co-tuned `compaction_trigger_ratio` **and** target, with the
  cold-prefill budget as the binding number — a separate implementation item,
  not part of this analysis.

## 8. What this item deliberately does not do

- No proxy, routing or config change; no live measurement (AC: read-only).
- No recommendation on the *other* levers (parallel prefill, quantisation,
  geometry, accepting the ceiling) — those are sibling children of the spike
  and are compared in its AC1 table.
- No attempt to re-scope the published ~5.8 % headline: this document uses the
  baseline child's pinned denominators (token share 1.62 % ceiling / stream
  share 5.96 % ceiling) and reports both.
