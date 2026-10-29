# Pool-4 go/no-go decision record — widening the session-slot snapshot pool

**Item:** `LP-0MVOG8U4D0026YCL` (C5 — pool-4 go/no-go decision record).
**Parent:** `LP-0MVN3KPEC007WWL5` — *Evaluate widening the session-slot snapshot pool
(`session_slot_pool_size` 2 → 4) before adding slots*.
**Protocol:** C2 `docs/dev/slot-snapshot-pool-sizing-LP-0MVN3KPEC007WWL5.md` §4
(bake-off protocol) and §4.3 (pass/fail thresholds G1–G6).
**Evidence consumed:** C3 `proxy/benchmarks/results/slot_snapshot_bands_20261028/summary.json`
(save/restore cost per band, under load) and C4
`proxy/benchmarks/results/pool_bakeoff_2026_10_24/summary.json` (bake-off replay on the
2026-10-24 thrash window).
**Scope:** documentation only — **no production behaviour change** in this child.

---

## Verdict

# **NO-GO**

Widening the session-slot snapshot pool to 4 **does not pay** on the 2026-10-24 thrash
window: the required reuse uplift is not produced and the exclusion share is far above
the ceiling, so C2 §4.3's documented outcome for the two failed gates is **no-go**.

**G1 (reuse uplift) fails** and **G2 (restore success / suppression) fails** on the
comparable arm (candidate B); G3–G6 pass. Measured numbers, all from the C4 replay:

| Gate | Pass condition (C2 §4.3) | Measured | Verdict |
|---|---|---|---|
| G1 | >20k median >= 31.2 % **and** aggregate >= 36.1 %, paired per-session delta > 0 | arm1 median **15.5 %**, aggregate **25.83 %**, per-session delta 0 | **fail** |
| G2 | restore success >= 90 % **and** `skip_when_busy` + cooldown exclusions <= 20.0 % | restore success 100 %; exclusion share **56.1 %** (170 / 303) (<= 20.0 % ceiling) | **fail** |
| G3 | live slots 2, clamp 126,976 (no geometry drift) | 2 / 126,976 | pass |
| G4 | per-band restore cost within the adaptive timeout | 0.04–0.09 s ≤ 18.0 / 40.5 / 60.0 / 60.0 s | pass |
| G5 | no regression (disk ≤ 20 GiB cap; local serve share not lower; waste not up) | disk 2.17–9.88 GiB; local serve share 0.2982 unchanged | pass |
| G6 | reproducible (deterministic replay, arm0 recalibrated to baseline) | byte-identical JSON; arm0 = 26.1 % / 15.6 % | pass |

The verdict agrees with C4's documented outcome
(`proxy/benchmarks/results/pool_bakeoff_2026_10_24/RESULTS_NOTE.md`): candidate B is
recorded as **no-go** on this window and must not ship from this evidence. This is a
conditional no-go against the present evidence, not a permanent verdict on the
mechanism — the measured levers (busy-gate suppression, giant-prompt admission) are
recorded in §4 so the decision can be revisited where the fixture cannot move.

---

## Break-even condition (stated numerically)

**Pool widening pays only if the >20k-token median restore-driven reuse >= 31.2 %**
(2 × the 15.6 % baseline) **and aggregate reuse >= 36.1 %, while** `skip_when_busy` +
cooldown suppression stays <= 20.0 % of persistence decisions** — because a *cold* prefill
of the thrash window's dominant prompts costs 19.96 s (10k band) → 480.8 s (60k+ band)
to populate, whereas a restore adds only 0.02–0.09 s, i.e. restore is roughly
**1 000 ×–5 300 × cheaper** than the cold prefill it replaces. Every restore the busy
gate suppresses therefore forfeits that near-free resume; with 56.1 % of persistence
decisions suppressed and only ±0 pp of median uplift actually restored, the economics
do not clear the gate.

Equivalently: with the measured restore cost (0.09 s at 60k+) and the cold-prefill
median on this window, a retained pool of 4 would need to convert roughly **every
non-suppressed restore opportunity into G1-grade reuse** (>= 31.2 % median) to pay for
its eviction machinery — it converts −0.1 pp instead.

---

## 1. C3 evidence table — save/restore cost per band (under load)

Source: `proxy/benchmarks/results/slot_snapshot_bands_20261028/summary.json`
(C3 `LP-0MVOG8TK6002EWZU`, harness C1 `LP-0MVOG8SYO006R49J`). Values are the
machine-readable artefact's, rounded to 3 dp. Adaptive timeout
`min(3.0 + 0.0015 × tokens, 60)`.

| Band | Prompt tokens | Timeout (s) | Save (s) unloaded → under load | Restore (s) unloaded → under load | Cached ratio | Restore budget headroom (s) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 10k | 9,942 | 18.0 | 0.034 → 0.049 | 0.021 → 0.04 | 0.9996 | 17.96 | plausible |
| 25k | 24,942 | 40.5 | 0.052 → 0.072 | 0.032 → 0.052 | 0.9998 | 40.448 | plausible |
| 40k | 39,942 | 60.0 | 0.095 → 0.111 | 0.049 → 0.067 | 0.9999 | 59.933 | plausible |
| 60k+ | 59,942 | 60.0 | 0.125 → 0.291 | 0.071 → 0.089 | 0.9999 | 59.911 | plausible |

Totals across both runs (unloaded + under load): **0 timeout-exceeded** operations,
**0 failed** operations, 8/8 saves and 8/8 restores successful, restore effectiveness
≥ 99.4 % of prompt tokens (only 4 tokens reprocessed per band). Summary:
`totals.timeout_exceeded_operations = 0`, `totals.failed_operations = 0`.

**Latency is not a blocker.** Restore cost is **0.02–0.09 s** at every band, including
under load (another slot streaming), far inside the 60 s adaptive cap — the mechanism's
cost model (`0.0015 s/token`) describes the populate *prefill* (20 s → 481 s), not the
save/restore path.

**Disk footprint is not a blocker.** A retained pool of 4 at these sizes projects to
6.46 GiB = 32.3 % of the 20 GiB cleanup cap
(`footprint.projected_gib_retained = 6.46`, `projected_cleanup_cap_fraction = 0.323`),
well inside the existing `cleanup-slot-cache.sh` budget.

---

## 2. C4 evidence table — bake-off arms and gate verdicts

Source: `proxy/benchmarks/results/pool_bakeoff_2026_10_24/summary.json`
(C4 `LP-0MVOG8TUM005705R`), deterministic replay of
`proxy/tests/fixtures/prefill_thrash_2026_10_24.jsonl` (2026-10-24 18:38 → 23:47,
5.15 h, 38 sessions, 1,016 unique attempts). Calibrated values anchor arm0 to the
measured baseline (aggregate 26.1 %, median 15.6 % for >20k prompts, 28 cold of 227,
per `LP-0MVN1IQTF0037WTG`).

| Arm | Live slots | Retained pool | `skip_when_busy` | Comparable | Aggregate reuse (cal.) | Median >20k (cal.) | Restore success | Exclusions busy / cooldown / ctx-too-large / timeout |
|---|---:|---:|---|---:|---:|---:|---:|---|
| arm0 — baseline (`session_slot_pool_size: 2`) | 2 | — (per-session files) | on | yes | **26.1 %** | **15.6 %** | 29 / 29 = 100 % | 170 / 0 / 687 / 0 |
| arm1 — candidate B (retained pool of 4, LRU) | 2 | 4 (LRU) | on | yes | **25.83 %** | **15.5 %** | 27 / 27 = 100 % | 170 / 0 / 687 / 0 |
| arm2 — candidate A (literal `session_slot_pool_size: 4`) | **4** | — | on | **no** (AC5-violating) | 34.99 % | 42.33 % | 13 / 13 = 100 % | 17 / 0 / 946 / 0 |
| arm3 — diagnostic (candidate B geometry, busy gate relaxed) | 2 | 4 (LRU) | **off** | no | 32.06 % | 24.33 % | 53 / 53 = 100 % | 0 / 0 / 687 / 0 |

Gate verdicts (C2 §4.3, comparable arms only — arm0/arm1):

| Gate | Pass? | arm0 | arm1 | Why |
|---|---|---|---|---|
| G1 reuse uplift | **fail** | 15.6 / 26.1 % | 15.5 / 25.83 % | median 15.5 % ≪ 31.2 % required; aggregate 25.83 % ≪ 36.1 %; paired per-session median delta ≈ 0 (not > 0) |
| G2 restore success | **fail** | 100 %, excl-share 0.5611 | 100 %, excl-share 0.5611 | restore success is fine; `skip_when_busy` + cooldown exclusions are 170 / 303 = **56.1 %** of persistence decisions, far above the 20 % ceiling (<= 20.0 % required) |
| G3 no geometry drift | pass | 2 / 126,976 | 2 / 126,976 | live slots and clamp asserted from the production helper |
| G4 cost bounded | pass | — | — | per-band restore 0.04–0.09 s inside the adaptive timeout at every band |
| G5 no regression | pass | — | — | disk 2.17–9.88 GiB ≤ 20 GiB cap; local serve share 0.2982 == arm0; waste is admission-side and unchanged |
| G6 reproducibility | pass | — | — | deterministic replay ⇒ byte-identical JSON; arm0 recalibrated to the measured 26.1 % / 15.6 % baseline |

**Candidate A is not a bake-off arm** (C2 §4.1): its apparent uplift (median 42.33 %) is
bought by clamping 946 prompts off the local path entirely, collapsing local serve share
29.82 % → 6.89 % — see §5 (AC5 supersession).

---

## 3. Why the retained pool of 4 does not help (attribution)

The no-uplift is attributable, not merely observed (exclusion counts per arm are in the
C4 artefact):

- **Residency dominates.** 190 of 303 local serves (62.7 %) hit the slot still owned by
  the same session (`resident_hits`) — reuse comes from the KV sitting in the live slot,
  not from restoring a file. Snapshot restoration accounts for only 29 (arm0) / 27
  (arm1) reuse events.
- **The pool cap trades retention for bounded disk.** On this window the baseline's
  per-session files (bounded only by the 20 GiB cleanup cap) retain snapshots for nearly
  every session; candidate B's LRU cap of 4 **evicts 16 snapshots**
  (`retention_evictions`) that the baseline keeps, so restores *decrease* (44 → 39
  opportunities, 29 → 27 successes). A handful of big sessions rotate the pool and push
  the rest out.
- **`skip_when_busy` suppresses the restorable ones.** 170 persistence decisions are
  skipped while another session's modelled serve is in flight — **56.1 %** of
  persistence decisions. The diagnostic arm (arm3, gate relaxed) shows those 53 restored
  opportunities are real (53/53 succeed) and worth **+8.73 pp** median (24.33 %), the
  largest same-geometry lever measured — still short of G1's 31.2 %.
- **The window's prompts are mostly not candidates.** 687 of 1,016 unique attempts
  (67.6 %) exceed the 126,976 clamp (`context_too_large`) and never reach the local slot
  — median prompt 148,753 tokens on this window.

---

## 4. Measured blockers (parent AC4 — reproducible from the work items alone)

The measured blockers and non-blockers, exactly as the parent's AC4 requires them to be
recorded:

| Blocker dimension | Measured value | Blocker? |
|---|---|---|
| Restore latency vs timeout model | 0.02–0.09 s per band vs adaptive timeout 18.0–60.0 s (`min(3.0 + 0.0015·tokens, 60)`) | **no** — 400 ×–2 000 × headroom; the restore mechanism is not latency-bound |
| `skip_when_busy` + cooldown exclusion rate | **56.1 %** of persistence decisions (170 / 303; cooldown 0) vs the 20 % ceiling | **yes** — G2 blocker; the busy gate suppresses most restorable opportunities |
| Restore failures | **0** failed of 29 + 27 attempts (100 % success); 0 timeout-exceeded operations | **no** — failure is not a blocker, suppression is |
| Per-slot context shrink | candidate A would shrink the clamp 126,976 → 61,440 (live slots 2 → 4); candidate B keeps 2 / 126,976 | **no** in this decision — the shrink is *not chosen* (AC5); it is a supersession, see §5 |
| Reuse uplift | arm1 aggregate 25.83 % (vs 36.1 % required), >20k median 15.5 % (vs 31.2 % required); per-session delta ≈ 0 | **yes** — G1 blocker; the pool does not raise reuse on this window |

Recorded verbatim on this item and on the parent `LP-0MVN3KPEC007WWL5` as comments
(companion PRD text), so the decision is reproducible from the work items alone without
re-reading this document.

---

## 5. AC5 invariant statement (parent AC5)

**This decision keeps the live slot count and per-slot context unchanged.**

- Live slots: **2** (llama-server `--parallel`, dispatch-lease cap, slot-ID space).
- Per-slot context clamp: **126,976** (routing warm ceiling, persistence cap).

The verdict above — **NO-GO** — is a decision *not* to change `session_slot_pool_size`
and therefore trivially satisfies parent AC5: no live-geometry change is proposed.

**AC5 supersession callout (never a silent change).** Candidate A (literal
`session_slot_pool_size: 4`) changes live slots 2 → 4 and the per-slot clamp
126,976 → 61,440, i.e. it **is a supersession of AC5** and would require **producer
sign-off** — it is explicitly *not* the scope of this decision and not a bake-off arm
(C2 §4.1). Only candidate B (a decoupled retained-snapshot pool, live geometry
unchanged) satisfies AC5, and it is the arm this record evaluates.

---

## 6. Alternatives under consideration (decision traceability)

The decision is traceable against the option set that motivated the parent's option 3a:

| Alternative | Owning item | Status here |
|---|---|---|
| Virtual slots (N session slots multiplexed onto 2 physical slots) — candidate C | `LP-0MVGD9ZKI004ETTE` | out of scope for this measurement (new component, C2 §3) |
| Slot-count re-evaluation (keeps `session_slot_pool_size: 2`; N=2 verdict) | `LP-0MVGCQ3QN009HRE8` | intact — this decision preserves the N=2 constraint |
| Tenancy / resident-prefix scheduling (`local_session_tenants`, default = pool size) | `LP-0MVN1IQTF0037WTG` | untouched by this decision |

---

## 7. Consequences for the dependent children

**This decision is NO-GO, so candidate B does not ship from this evidence.** The
dependent work-items remain **blocked**:

- `LP-0MVOG8UE60050SKA` — *Ship pool-4 retention and eviction semantics*: remains
  blocked (its input was the retained-pool spec that a *go* would provide).
- `LP-0MVOG8UNX003Z9WN` — *Report restore-hit rate and snapshot pool occupancy*:
  remains blocked (those observability fields describe the shipped pool; they may be
  reconsidered independently if the parent re-scopes, per C2 §5's note that the fields
  are useful for either answer).

No configuration delta, no code change and no new observability fields are introduced
by this record. If the parent later re-opens the decision, the measured levers in §3/§4
(the busy gate, the giant-prompt admission clamp) are the candidates the fixture can
actually move.

---

## 8. Reproduce

The evidence tables above are the output of the committed, machine-readable artefacts —
regenerate, never hand-edit:

```bash
# C3 — save/restore cost per band (requires the live host API; see its RESULTS_NOTE)
python3 -m proxy.benchmarks.slot_snapshot_band_report \
  --unloaded proxy/benchmarks/results/slot_snapshot_bands_20261028/unloaded.json \
  --under-load proxy/benchmarks/results/slot_snapshot_bands_20261028/under_load.json \
  --output-dir proxy/benchmarks/results/slot_snapshot_bands_20261028/ --markdown README.md

# C4 — deterministic bake-off replay (no host needed; regenerates summary.json)
python3 proxy/tests/pool_snapshot_bakeoff.py --write

# Doc-consistency contract (this record vs the artefacts)
python3 -m pytest proxy/tests/test_slot_snapshot_pool_evaluation_doc.py -q
```

## Files and verification

- This document: `docs/slot-snapshot-pool-evaluation-LP-0MVN3KPEC007WWL5.md`.
- Evidence: `proxy/benchmarks/results/slot_snapshot_bands_20261028/` (C3 —
  `summary.json`, `unloaded.json`, `under_load.json`, `README.md`, `RESULTS_NOTE.md`) and
  `proxy/benchmarks/results/pool_bakeoff_2026_10_24/` (C4 — `summary.json`,
  `RESULTS_NOTE.md`).
- Protocol and candidates: `docs/dev/slot-snapshot-pool-sizing-LP-0MVN3KPEC007WWL5.md`
  (C2, §4 protocol / §4.3 thresholds / §5 open question).
- Contract test: `proxy/tests/test_slot_snapshot_pool_evaluation_doc.py` pins every
  number in §1/§2 to the artefacts — the record cannot silently drift from the evidence
  it claims to rest on.
