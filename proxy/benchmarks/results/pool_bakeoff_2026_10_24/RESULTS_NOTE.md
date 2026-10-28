# RESULTS NOTE — pool-4 / snapshot-restore bake-off replay, 2026-10-24 window

Work item: **LP-0MVOG8TUM005705R** (Bake-off replay of the 2026-10-24 thrash
window), parent **LP-0MVN3KPEC007WWL5** (Evaluate widening the session-slot
snapshot pool 2 → 4). Protocol: C2
`docs/dev/slot-snapshot-pool-sizing-LP-0MVN3KPEC007WWL5.md` §4. Machine-readable
evidence: `summary.json` (committed next to this note).

## Verdict up front

**Candidate B (a decoupled retained-snapshot pool of 4, LRU, live geometry
unchanged) does NOT raise prefix reuse on the 2026-10-24 thrash window.
G1 and G2 fail; G3–G6 pass. The measured reuse uplift is −0.27 pp aggregate and
−0.10 pp median (>20 k prompts) against the calibrated baseline — i.e. a
slightly *negative* delta, inside modelling noise, and far below the C2 §4.3
2 × / +10 pp go-threshold.** The restore mechanism itself is healthy
(restore-success rate 100 %, zero timeout-exceeded operations), so the
no-uplift is **not** a snapshot-engineering failure — it is that (a) most
window reuse comes from *residency*, not snapshot restoration, and (b) the
busy gate suppresses the largest share of the persistence decisions that *could*
have restored.

The diagnostic arm (candidate B geometry with `skip_when_busy` relaxed) is the
largest same-geometry lever: +5.96 pp aggregate / +8.73 pp median over the
baseline, still short of G1 (24.33 % vs 31.2 % median). The dominant exclusion
is `context_too_large`: 687 of 1,016 unique attempts (67.6 %) never reach the
local slot at all on this window (median prompt 148,753 tokens > the 126,976
pool-2 clamp).

Per C2 §4.3 the documented outcome for a single failed gate is **no-go**;
two gates (G1, G2) fail, so candidate B is recorded as **no-go** on this
window and must not ship from this evidence. Candidate A (literal pool 4) is
recorded but **not** a bake-off arm: its apparent uplift (median 42.33 %) is
bought by clamping 946 prompts off the local path entirely, collapsing the
local serve share 29.82 % → 6.89 % (the AC5 constraint C2 §4.1 excludes).

## Replay protocol used (C2 §4)

| Property | Value |
|---|---|
| Fixture | `proxy/tests/fixtures/prefill_thrash_2026_10_24.jsonl` (1,978 admission attempts, 2026-10-24 18:38 → 23:47 local) |
| Replay | `proxy/tests/pool_snapshot_bakeoff.py` — a **model of the pool/restore mechanism only**; routing/admission decisions come from the fixture, gates are the real production predicates (`_slot_id_for_session`, `_slot_persistence_skip_when_busy`, `_slot_in_failure_cooldown` + the real circuit breaker, `effective_per_slot_threshold` clamp) |
| Unique attempts | 1,016 (each capture log double-logs `queue` + `large_context_bypass`; the model consumes one attempt per `(ts, session, estimated_tokens)` — documented decision #1) |
| Restore/save cost | pinned from C3 `proxy/benchmarks/results/slot_snapshot_bands_20261028/summary.json` (0.02–0.09 s per band) vs adaptive timeout `min(3.0 + 0.0015·tokens, 60)` |
| Percentiles | `scripts/analyse_slot_ratio_restore._percentiles(percentages)` — the usage report's definition, no ad-hoc quantile |
| Calibration | proportional: Arm 0 anchored to the measured 26.1 % / 15.6 % / 28-of-227 baseline (LP-0MVN1IQTF0037WTG); every other arm scaled by the same per-metric factor (documented decision #2) |

## Arms and headline numbers

| Arm | live slots | retained pool | skip_when_busy | comparable | aggregate (cal.) | median >20k (cal.) | cold/>20k (cal.) | restore success | exclusions (busy / cooldown / ctx-too-large / timeout) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| arm0 baseline | 2 | — (per-session files) | on | yes | 26.10 % | 15.60 % | 37 / 297 | 29 / 29 (100 %) | 170 / 0 / 687 / 0 |
| arm1 candidate B | 2 | 4 (LRU) | on | yes | 25.83 % | 15.50 % | 38 / 297 | 27 / 27 (100 %) | 170 / 0 / 687 / 0 |
| arm2 candidate A | 4 | — | on | **no** | 34.99 % | 42.33 % | 9 / 64 | 13 / 13 (100 %) | 17 / 0 / 946 / 0 |
| arm3 diagnostic | 2 | 4 (LRU) | **off** | no | 32.06 % | 24.33 % | 26 / 297 | 53 / 53 (100 %) | 0 / 0 / 687 / 0 |

(Calibrated values anchor arm0 to the measured baseline; raw values in
`summary.json`. `cold/>20k` is the fully-cold count over >20k-token local
serves, scaled by the cold-share calibration factor.)

## Gate verdicts (C2 §4.3, comparable arms)

| Gate | Pass | arm0 | arm1 | Why |
|---|---|---|---|---|
| G1 reuse uplift | **fail** | 15.6 / 26.1 % | 15.5 / 25.83 % | median 15.5 % ≪ 31.2 % required; aggregate 25.83 % ≪ 36.1 %; paired per-session median delta ≈ 0 (not > 0) |
| G2 restore success | **fail** (arm1) | 100 %, excl-share 0.5611 | 100 %, excl-share 0.5611 | restore success is fine; `skip_when_busy`+cooldown exclusions are 170 / 303 = 56.1 % of persistence decisions, far above the 20 % ceiling |
| G3 no geometry drift | pass | 2 / 126,976 | 2 / 126,976 | live slots and clamp asserted from the production helper, not assumed |
| G4 cost bounded | pass | — | — | per-band restore cost 0.04–0.09 s inside the adaptive timeout at every band (10k–60k+) |
| G5 no regression | pass | — | — | disk 2.2–9.9 GiB ≤ 20 GiB cleanup cap; local serve share unchanged (0.2982); deadline/waste are admission-side and unchanged by the pool model |
| G6 reproducibility | pass | — | — | deterministic replay: same fixture/config ⇒ byte-identical JSON; arm0 recalibrated to the measured baseline |

## Attribution of the no-uplift (AC4 — attributable, not merely observed)

The exclusion counts per configuration are in `summary.json` and reproduced
here so "no uplift" has named causes:

| Exclusion channel | arm0 | arm1 | arm3 | Interpretation |
|---|---:|---:|---:|---|
| `context_too_large` | 687 | 687 | 687 | 67.6 % of unique attempts exceed the 126,976 clamp — never candidates for local reuse on this window (median prompt 148,753 tokens) |
| `slot_busy` (skip_when_busy) | 170 | 170 | 0 | the load-aware gate suppresses persistence on 56 % of decisions; relaxing it (arm3) recovers +8.73 pp median — the biggest same-geometry lever |
| circuit-breaker cooldown | 0 | 0 | 0 | no breaker trips in the window's persistence path (all saves/restores succeed) |
| `timeout_exceeded` | 0 | 0 | 0 | C3-measured restore cost (≤ 0.09 s) is far inside the 60 s adaptive cap |
| pool exhausted | 26 | 26 | 26 | 2 live slots saturated; sink of the same size in every arm |

Why the retained pool of 4 does not help even where gates allow:

- **Residency dominates.** 190 of 303 local serves (62.7 %) hit the slot still
  owned by the same session (`resident_hits`) — reuse comes from the KV sitting
  in the slot, not from restoring a file. Snapshot restoration accounts for
  only 29 (arm0) / 27 (arm1) reuse events of 44 / 39 restore opportunities.
- **The cap trades retention for bounded disk.** On this window the baseline's
  per-session files (bounded only by the 20 GiB cleanup cap) retain snapshots
  for nearly every session (9.9 GiB used); candidate B's LRU cap of 4 evicts
  16 snapshots (`retention_evictions`) that the baseline keeps, so restores
  *decrease* (44 → 39 opportunities, 29 → 27 successes). A handful of big
  sessions rotate the pool and push the rest out.
- **The busy gate suppresses the restorable ones.** 170 persistence decisions
  are skipped while another session's modelled serve is in flight; arm3 shows
  those 53 restored opportunities are real (53/53 succeed) and worth
  +8.73 pp median — but they do not reach G1 either.

## Environment / fidelity record

- **No live proxy configuration was changed** (analysis only). The sim drives
  production predicates through the documented stub sources (the in-process
  `_slot_owners` registry and `_slot_busy_state_snapshot` value) with a
  simulated clock — timestamps come from the fixture, never a wall clock.
- **Busy window is a conservative lower bound.** A session holds its dispatch
  lease for the modelled prefill of the unreused prompt only (`new_tokens ·
  (1 − reuse)` / band rate); generation time is not in the fixture, so the busy
  gate's suppression share is *under*-estimated, meaning the real-world
  `slot_busy` share is ≥ 170 / 303.
- **Deadline cancellation totals** (125 / 778,863 tokens in the fixture `_meta`)
  are admission-side and unchanged by the pool model; the deduped 1,016-attempt
  view records 80 `prefill_deadline_exceeded`.
- **No production module imports the harness** (AC5 guard in
  `test_pool_snapshot_bakeoff.py`), matching the sibling
  `prefill_thrash_replay_harness` rule.

## Reproduce

```bash
python3 proxy/tests/pool_snapshot_bakeoff.py --write   # regenerates summary.json
python3 proxy/tests/test_pool_snapshot_bakeoff.py -q    # 20 offline tests (determinism, boundary, gate, import guard)
```

## Artefacts

| File | Content |
|---|---|
| `summary.json` | Full machine-readable bake-off report (arms, raw + calibrated metrics, gates, exclusions, counters, delta table) |
| `RESULTS_NOTE.md` | This note |

## Implication for the parent decision (C5)

- **Pool-4 retention does not pay on this window.** Reuse here is residency
  and per-session files; a bounded retained pool of 4 *loses* snapshots the
  per-session baseline keeps and buys nothing in return.
- **The busy gate, not pool size, is the measured suppressor.** arm3's +8.73 pp
  median is the largest same-geometry lever and still misses G1 — the economics
  of restoring a 0.02–0.09 s operation are throttled by load-awareness by
  design (the gate exists to avoid displacing an in-flight prefill), and the
  window's giant prompts (median 148 k) are excluded before persistence is
  even considered.
- **The thresholds are remote.** G1 demands 31.2 % / 36.1 %; the best
  comparable arm reaches 15.5 % / 25.83 %. No same-geometry configuration on
  this window is close, so C5's go/no-go should record **no-go** for candidate
  B and candidate A (AC5-violating), with arm3's busy-gate relaxation noted as
  the only lever the fixture can actually move reuse with.
