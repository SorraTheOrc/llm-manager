# Restore-rate metric, per-mode targets & fix ranking (F4)

**Work item:** LP-0MTCMG18A008ZKVT (parent LP-0MTAQNB7J0094X71)
**Baseline:** 2026-08-26 F1 corpus (docs/dev/slot-persistence-baseline-2026-08-26.json)
**Reproduction:** `python3 scripts/restore_metric_fix_ranking.py --baseline
docs/dev/slot-persistence-baseline-2026-08-26.json --markdown`

## Current restore rate (Wilson 95% CI)

| Mechanism | Restore rate | 95% CI | Interpretation |
|---|---|---|---|
| **Proxy slot persistence** (slot_save → slot_restore) | **89.58%** (831/926) | 87.62–91.53% | Healthy when it runs — the mechanism the fixes can move |
| **llama-server native checkpoints** (created → restored) | **4.88%** (154/3,191) | 4.14–5.63% | The incident's "~95% unrestored" number |
| Incident-day reference (claimed 145/2,954) | 4.97% | 4.19–5.75% | Reproduces within tolerance |

**Prefill pool:** 43.8M prefill tokens/day are re-prefilled at the current
native restore rate (46.1M prompt-processing-done tokens × (1 − 4.88%)) —
this is the savings pool for fix ranking.

## Per-mode targets

| Mode | Priority | Target restore rate | Persistence cap | Slots | Rationale |
|---|---|---|---|---|---|
| **fast** | TTFT/P95 | **95%** | 83,285 | 3 | Fast tolerates fallbacks for speed; restore reuse eliminates minutes-long full re-prefills on large sessions → TTFT/P95 drops |
| **cheap** | Cost / local utilization | **90%** | 126,976 | 2 | Cheap must avoid fallbacks (cost); reliable restore keeps oversized sessions local without re-prefill |

Targets are stated on the **proxy slot-persistence rate** — the actionable
knob. The native checkpoint rate (4.88%) is the incident's signal but is not
a directly tunable target (see F2: it is llama-server's ephemeral recovery
mechanism, distinct from proxy persistence).

## Fix options ranked by expected prefill-token savings

Model: `savings = pool × (1 − current_rate) × rate_gain × recovery`, where
`rate_gain` is the fraction of the restore-rate gap the fix closes and
`recovery` is the share of those turns that actually restore (GPU-wedge
mitigation — saves can timeout under load, LP-0MSI1RWLM007N367).

| Rank | Fix | Expected savings (tokens/day) | GPU-wedge risk | Evidence (F2/F3) |
|---|---|---|---|---|
| 1 | **Raise persistence cap to routing clamp + timeout/cooldown plan** | **22,790,282** | medium | F2: 38/48 sessions gated out (1,902 context_too_large + 457 bypass) with 0 saves — the dominant factor; cap raise must carry LP-0MS91DHPZ001VWQO timeout/cooldown |
| 2 | **Session↔slot affinity / ownership continuity fix** | 3,287,059 | low | F2: 166 orphan + 141 evicted lease releases (307 breaks); LP-0MSB0RV72001KNRV registry-leak class |
| 3 | **Restore-before-save ordering for hot sessions** | 438,274 | none | Ordering fix, zero GPU footprint change; recovers same-slot reuse without new saves |
| 4 | **Relax load-aware skip-when-busy for same-session reuse** | 131,482 | low | Only 2 save failures on incident day → gate is conservative vs observed load (LP-0MSI1RWLM007N367) |

**Ranking conclusion:** the persistence-cap raise dominates — it addresses
the sessions the incident actually hit (giant contexts) and captures ~70% of
the addressable savings pool. The /slots 500s and slots_stale (F3) are not
ranked as separate savings options because their restore-path impact is
indirect (they degrade the slot-state view that affinity decisions use);
they were folded into the affinity fix's `rate_gain` and their direct fix is
tracked in F3's ranked list + LP-0MSVP7XJ6008QPKX.

## Feed to F5

- Fast and cheap targets differ (95% vs 90%) because their priorities
  differ; convergence is not precluded (parent risk) — F5 decides.
- Cap-raise candidate values: fast ≥ 83,285, cheap ≥ 126,976 (up to the
  per-slot routing clamp, 87,381 − headroom / 131,072 − headroom), always
  with the timeout/cooldown plan.
- Ranking sensitivities (rate_gain/recovery) are explicit in
  `scripts/restore_metric_fix_ranking.py` so F5 can re-run with mode-specific
  inputs.