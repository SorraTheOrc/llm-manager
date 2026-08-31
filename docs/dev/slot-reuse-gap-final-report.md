# KV Slot Save/Restore Reuse-Gap Evaluation — Final Report

**Evaluation parent:** LP-0MTAQNB7J0094X71
**Follow-up implementation:** LP-0MTE9HAF8008909G (feature, created by F5)
**Window:** 2026-08-26 incident + companion data through 2026-08-28 (logs
available on the box)

---

## 1. Root cause — why ~95% of checkpoints are never restored (F2)

**Reproduction:** `python3 scripts/slot_reuse_gap_analysis.py --log-dir
/var/log/llama-proxy --start 2026-08-26 --end 2026-08-27 --llama-file
'*2026-08-27*'`

Two independent persistence mechanisms were conflated in the incident:

| Mechanism | Restore rate (2026-08-26) | Meaning |
|---|---|---|
| Proxy slot save/restore (`slot_save`/`slot_restore`) | **89.58%** (831/926) | The reuse pipeline — healthy when it runs |
| llama-server native checkpoints (created→restored) | **4.88%** (154/3,191) | Ephemeral in-flight recovery; the "~95% unrestored" |

The **real gap**: context-size gating excluded **1,902 `context_too_large`
requests across 38 of 48 sessions** from local dispatch — with **zero**
persistence events (top sessions: 409/267/207/199/183/161 repeated skips, 0
saves). The sessions that need reuse most (giant contexts) never persist.
Secondary: `slots_stale` 47.4%, GET /slots 500s 9.4% + 400s, 307
orphan/evicted lease releases.

Full evidence: `docs/dev/save-restore-reuse-gap-root-cause.md`.

## 2. /slots HTTP 500-storm triage (F3)

**Reproduction:** `python3 scripts/slots_500_triage.py --log-dir
/var/log/llama-proxy --llama-file '*2026-08-27*'`

**All** GET /slots 500s (6,865) and 400s (527) on the incident day are
answered by the **router** (router-mode proxying); model instances answer
only 200. The router proxies /slots to a busy model mid-giant-prefill, the
model cancels the proxied connection (`Connection handling canceled`, 6,940
events ≈ 500 count 1:1), and the router returns 500. 53.3% of 500 windows
contain concurrent prefill/checkpoint activity. Fix ranking:
direct-instance polling / last-known-state (dominant) → concurrency-aware
timeout → fix cancel path → dedupe aggregation. Related items flagged:
LP-0MSVP7XJ6008QPKX, LP-0MSB0RV72001KNRV, LP-0MSHW2AXJ009DO3S.

Full evidence: `docs/dev/slots-500-triage.md`.

## 3. Restore-rate metric & fix ranking (F4)

**Reproduction:** `python3 scripts/restore_metric_fix_ranking.py --markdown`

Current rates with **Wilson 95% CI**: proxy slot persistence 89.58%
(87.62–91.53); llama native 4.88% (4.14–5.63) — the incident's ~5%
(4.97% CI 4.19–5.75 for the 145/2954 reference). Prefill pool: 43.8M
tokens/day re-prefilled at current rates.

**Per-mode targets** (on the proxy slot-persistence rate):
- fast (TTFT/P95 priority, 3 slots, cap 83,285): **95%**
- cheap (cost/local-utilization priority, 2 slots, cap 126,976): **90%**

**Fix ranking** (savings model `pool × gap × rate_gain × recovery`):
1. Cap raise to routing clamp + timeout/cooldown: **22.79M** tokens/day (medium GPU-wedge risk)
2. Session↔slot affinity / ownership continuity: **3.29M** (low)
3. Restore-before-save ordering: **0.44M** (none)
4. Relax skip-when-busy: **0.13M** (low)

Full evidence: `docs/dev/restore-rate-metric-fix-ranking.md`.

## 4. Mode-specific recommendation (F5)

**Reproduction:** `python3 scripts/mode_recommendation.py --markdown`

- **fast** (3 slots, cap 83,285): TTFT/P95 priority. Selected: cap raise +
  affinity + restore-before-save (26.5M tokens/day savings). Fallback rate
  down; TTFT/P95 down.
- **cheap** (2 slots, cap 126,976): cost/local-utilization priority. Same
  fix set (cap raise is the local-reuse lever). Fallback rate down; TTFT/P95
  down (secondary).
- Mode **convergence**: fix sets converge; cap values do not
  (fast 83,285 ≠ cheap 126,976) — parent risk documented.
- **GPU-wedge plan** unchanged (LP-0MS91DHPZ001VWQO): timeout base 3.0s +
  0.0015s/token, cap 60s; cooldown 300s after 3 consecutive failures;
  skip-when-busy on.
- **No code changed** in proxy/ or ds4/ (evaluation only). Follow-up
  implementation: **LP-0MTE9HAF8008909G**.

Full evidence: `docs/dev/mode-specific-recommendation.md`.

## 5. Validation (F6)

- **Full project test suite:** green via `/skill:test` (pytest, full scope)
  after all F1–F5 scripts landed — no regressions.
- **Reproducibility:** F1 corpus regeneration is deterministic over a frozen
  log snapshot (verified: `--regen /tmp/log-snapshot` → `deterministic:
  true`); the committed baseline
  `docs/dev/slot-persistence-baseline-2026-08-26.json` regenerates to the
  same metrics.
- **Scripts:** `scripts/slot_persistence_harness.py`,
  `scripts/slot_reuse_gap_analysis.py`, `scripts/slots_500_triage.py`,
  `scripts/restore_metric_fix_ranking.py`, `scripts/mode_recommendation.py`,
  `scripts/validation_report.py` — all rerunnable end-to-end from a log
  snapshot (usage in each docstring).
- **No source code changed** in `proxy/` or `ds4/`: evaluation deliverables
  are `scripts/*.py`, `proxy/tests/*.py` (test-only), and `docs/dev/*.md`.

## Deliverable index

| Feature | Script | Test | Doc |
|---|---|---|---|
| F1 corpus | scripts/slot_persistence_harness.py | proxy/tests/test_slot_persistence_harness.py | docs/slot-persistence-analysis-harness.md |
| F2 root cause | scripts/slot_reuse_gap_analysis.py | proxy/tests/test_slot_reuse_gap_analysis.py | docs/dev/save-restore-reuse-gap-root-cause.md |
| F3 /slots triage | scripts/slots_500_triage.py | proxy/tests/test_slots_500_triage.py | docs/dev/slots-500-triage.md |
| F4 metric | scripts/restore_metric_fix_ranking.py | proxy/tests/test_restore_metric_fix_ranking.py | docs/dev/restore-rate-metric-fix-ranking.md |
| F5 recommendation | scripts/mode_recommendation.py | proxy/tests/test_mode_recommendation.py | docs/dev/mode-specific-recommendation.md |
| F6 validation | scripts/validation_report.py | proxy/tests/test_validation_report.py | this report |

## Follow-up

**LP-0MTE9HAF8008909G** — Implement KV slot persistence fixes from the
reuse-gap evaluation (fast cap 83,285 / cheap cap 126,976): cap/restore-path
fix, affinity/ownership fix, restore-before-save ordering, /slots 500 +
slots_stale fix; all within the preserved GPU-wedge plan.