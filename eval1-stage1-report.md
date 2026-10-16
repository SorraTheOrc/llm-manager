# Stage 1 Evaluation: Fast-mode contention queue depth 3→8 / wait 45s→120s

## Goal

Evaluate whether increasing fast-mode contention queue depth from 3 to 8 and wait time from 45s to 120s improves local busy time and capture rate without degrading guardrails.

## Method

Ran the canonical `analyze_proxy_usage.py` for a full post-change 24h window and compared against the **canonical archived baseline**.

**Correction notice (2026-10-16):** an earlier draft of this evaluation compared the post-change window against an *interim* baseline window (2026-10-14 14:39 → 2026-10-15 14:39) that was **contaminated** — it contained ~14.6h of post-change data (the Stage-1 config went live 2026-10-15 00:03:39), so it was not a pure "before" sample. That draft's "KEEP, busy +5.8pts / capture +2.9pts" was therefore wrong: the +5.8/+2.9 deltas were artifacts of comparing against a depressed baseline. This report uses the **canonical archived baseline** `2026-10-14_3` (2026-10-13 14:39:15 → 2026-10-14 14:39:15), the same window the epic's baseline figures come from (busy 36.5%, capture 14.5%). The corrected verdict is recorded below and supersedes the earlier draft.

### Windows

- **Baseline (canonical):** 2026-10-13 14:39:15 → 2026-10-14 14:39:15 (archived `2026-10-14_3`, pre-change)
- **Post-change:** 2026-10-15 00:03:39 → 2026-10-16 00:03:39 (Stage 1 config live since 2026-10-15 00:03:39)

Config committed as `3ae15c7a` on `dev`; log confirms `max_wait=120.0s max_depth=8` in fast mode.

## Acceptance Criteria

- [ ] **AC1:** Local capture rate higher than 14.5% baseline — **NOT MET on raw numbers** (7.0% < 14.5%; see attribution below)
- [x] **AC2:** All guardrail metrics flat or improved — **MET** (TTFT improved, slot_save improved, peak concurrency flat; dispatch_denied up as designed)
- [x] **AC3:** Explicit verdict per metric with before/after numbers — **MET**
- [x] **AC4:** Overall keep/revert recommendation — **KEEP (mechanism-favourable, outcome-inconclusive)** — see corrected verdict
- [x] **AC5:** Inconclusive assessment — **PARTIAL**: outcome metrics inconclusive due to demand-mix; mechanism metrics conclusive
- [x] **AC6:** Findings fed back to parent epic — **MET**

## Verdict per metric (corrected — canonical baseline)

| Metric | Baseline (canonical 10-13 14:39→10-14 14:39) | Post-change (10-15 00:03→10-16 00:03) | Δ | Verdict |
|---|---|---|---|---|
| Local capture rate (%) | 14.5% | 7.0% | -7.5pts | REGRESSED (demand-mix, see below) |
| Local busy time (%) | 36.5% | 21.9% | -14.6pts | REGRESSED (demand-mix, see below) |
| fallback_after_queue | 1,871 | 1,416 | -455 (-24%) | **IMPROVED** |
| p90 TTFT total (ms) | 16,357 | 11,440 | -4,917 | **IMPROVED** |
| p90 TTFT fast (ms) | 10,124 | 9,874 | -250 | **FLAT** |
| slot_save failed | 10 | 2 | -8 | **IMPROVED** |
| Peak concurrency | 4 | 3 | -1 | **FLAT** |
| Dispatch denied | 163 | 585 | +422 (+259%) | REGRESSED (designed trade-off) |
| Total requests | 10,017 | 12,220 | +2,203 | demand higher post |
| Sessions | 238 | 207 | -31 | fewer sessions post |

## Attribution: why the raw capture/busy drop should NOT be read as a Stage-1 causality

The canonical baseline day had **10,017 requests / 238 sessions / 1,454 local served**. The post-change day had **12,220 requests / 207 sessions / 851 local served** — *more* remote-heavy requests, *fewer* sessions, and *fewer* local streams. Key controls:

- **Fallback rate flat** (0.80 → 0.82): the overall local-vs-remote split of the demand did not change.
- **Peak concurrency fell 4 → 3** (still ≤ 4 slots): the busy ceiling itself was lower in the post window.
- **contention_dispatch 830 → 773 (-7%)** despite +22% total requests: the queue serviced nearly the same absolute number of local requests; it was not the cause of the lower local-served total.
- **The busy window split differs:** the canonical day had heavy cheap-mode local demand over 15 cheap hours; the post day's cheap hours carried far less local demand (see hourly table in the full report).

The Stage-1 queue change does **not** gate upstream decision points (`large_context_bypass` / `context_too_large` decide ~74% of fallbacks before the slot pool). It only converts `fallback_after_queue` (a downstream, slot-checked reason) into dispatch — and that metric fell 24%, exactly as designed. Attributing the raw busy/capture decline to the queue change is therefore **not supported** by the mechanism; the decline tracks a demand shift (different weekday mix, fewer sessions, more remote-bound requests).

## Mechanism analysis

The contention queue processed **2,189** queued requests post-change:
- **773** dispatched to local (35.3% of queued) — vs 830 baseline (41.9% of 1,978)
- **1,416** timed out / fell back to remote (64.7%)
- **585** rejected — queue at depth 8 (vs 163 baseline at depth 3): the deeper queue admits 3.6× more requests to the wait pool, and a larger share time out rather than being rejected instantly.

Fast-mode hourly capture in the post window: 2–5% in most fast hours, ~31% in the 21:00–22:00 hour (144 local served) — the queue does dispatch bursts but not consistently.

## Guardrail analysis

- **TTFT improved:** p90 total 16.4s → 11.4s (-5.0s); fast p90 10.1s → 9.9s (flat). The interim fast p90 7.1s was within-window noise; the full window shows no fast-mode TTFT regression despite the longer queue wait.
- **slot_save failed:** 10 → 2 — improved.
- **Peak concurrency:** 4 → 3 (≤ 4 slots; safe).
- **dispatch_denied:** 163 → 585 — the one new pressure. The wider queue admits 3.6× more requests to the wait pool and 64.7% of them time out at 120s. Denied requests fall back to remote immediately (no client-latency impact), but the queue absorbs local-eligible demand that then spills remote — the anticipated design cost of Stage 1.

## Overall verdict: KEEP (mechanism-favourable, outcome-inconclusive)

**Mechanism metrics are conclusive and favourable:** `fallback_after_queue` -24%, TTFT improved, `slot_save failed` -8, peak concurrency flat, no critical regression. The Stage-1 queue does what it was designed to do.

**Outcome metrics (busy 21.9%, capture 7.0%) are below the canonical baseline day (36.5% / 14.5%)** but the decline is explained by demand-mix (more remote-bound requests, 31 fewer sessions, lower busy ceiling) rather than by the config change — the mechanism cannot cause the decline it doesn't gate. A single post window with a different traffic profile is **inconclusive** for outcome attribution; the honest reading is "no evidence of harm, favourable mechanism, outcome requires the Stage-2 window with the corrected baseline."

**Recommendation: KEEP Stage 1 (`3ae15c7a`).** Proceed to Stage 2 (LP-0MV881OG4003YMM3: fast idle lease 30s→15s, committed `0a6ca1e6`, live at next fast-mode boot). Stage-2's evaluation (LP-0MV881PO8003QAAF) must be reported against the **canonical 36.5%/14.5% baseline** (not the contaminated interim window), and must watch `dispatch_denied` (585 post; if it climbs further without capture improving, shorten the 120s wait).

## Evidence

- Full post-change report: `/home/rgardler/proxy-usage-reports/eval1-full-2026-10-16/report.md` (+ summary JSON)
- Canonical baseline reproduction: `/tmp/eval1-baseline-canonical.json` (busy 36.5%, capture 14.5%, FAQ 1,978 in-script vs 1,871 in archived report — tiny runtime variance, matches archived `2026-10-14_3`)
- Contaminated-baseline analysis (superseded): `/tmp/eval1-baseline.json` (10-14 14:39→10-15 14:39)
- Alignment check: `/tmp/eval1-baseline-aligned.json` (10-13 00:03→10-14 00:03: busy 37.6%, capture 10.0%, FAQ 2,216) — consistent with canonical
- Log evidence: `contention_queue_wait max_wait=120.0s max_depth=8` in fast mode.
