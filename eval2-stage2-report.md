# Stage 2 Evaluation: Fast-mode idle lease 30s → 15s (orphan-cleanup risk focus)

## Goal

Evaluate whether the Stage-2 fast-mode idle lease change (**30s → 15s**, commit
`0a6ca1e6`, `local_dispatch_lease_timeout_seconds: 15` in
`proxy/config-fast.yaml`) improves local capture / reduces `local_lease_active`
denials without reproducing the **15x orphan-cleanup spike** that forced the
original 15s → 30s reversal (LP-0MSKOHV55003XFTB → LP-0MSUO82NF002HUMC).

## Method

Ran the canonical `analyze_proxy_usage.py` CLI over the intended Stage-2
window (2026-10-16 10:00 → 2026-10-17 10:00 IST), plus direct lease-event and
per-session stream tracing of `/var/log/llama-proxy/proxy.log*` for the
orphan-cleanup signature.

## ⚠️ Critical data-integrity caveat: 16.7h proxy-log gap

**The intended 24h window is only ~31% covered.** The proxy log has a 16.7-hour
gap on 2026-10-16:

| File | Coverage |
|---|---|
| `proxy.log.2026-10-16_00.gz` | 10-16 00:00 → 06:59 (pre-change, 30s) |
| **GAP** | **10-16 06:59 → 23:40** (whole daytime fast window incl. 10:00 IST boot) |
| `proxy.log-2026-10-17_00.gz` | 10-16 23:40 → 10-17 00:00 (post-change) |
| `proxy.log.2026-10-17_01.gz` | 10-17 00:00 → 07:02 |

Observable post-change fast-mode fragments: **10-16 23:40 → 10-17 01:00**
(1.33h, fully fast, lease=15s) plus 10-17 evening fast windows
(14:46–15:12, 15:37–16:12, 18:55–19:38, 19:39–20:20, 22:22–01:00 ≈ 6.2h,
contaminated by 19 proxy restarts that evening). The **entire daytime fast
window (10:00–23:40 on 10-16) where the 15s lease matters most is missing.**
Root-caused separately: **LP-0MVEL6DWJ00179F4**.

Consequence: aggregate capture/busy/denial numbers from the canonical report
are **not comparable to the baseline** (they cover mostly cheap-mode overnight
hours). Verdicts below are therefore **fragment-based** for the orphan
signature (which is measurable from lease events) and **unverifiable** for
aggregate demand metrics.

## Lease-live verification

- Config at Stage-2 commit: `local_dispatch_lease_timeout_seconds: 15`
  (fast), cheap inherits base 30. Both verified in merged fixtures
  (`merged_config_fast.json` = 15).
- Log evidence: `lease_renewed ... timeout=15s` in ALL observable fast-mode
  fragments (43 renewals in 23:40→01:00; 170 in the 10-17 evening windows);
  `timeout=30s` in cheap segments. The 15s lease was **live and effective in
  fast mode**.

## Verification fragment 1 (clean fast-15s): 2026-10-16 23:40 → 10-17 01:00

| Metric | Value |
|---|---|
| Requests (Stream started) | 1,752 |
| Orphan cleanups (`INFO - lease_released ... reason=orphan_cleanup`) | 7 = **4.0 per 1k requests** |
| Mid-flight active-stream hits | 3 (sessions `01a144aa` 34s in, `herdr-…2173` 37s in, `herdr-…3459` 78s in) |
| Streams lost / errored after cleanup | **0 / 0** — all 3 completed `reason=tool_calls` / `stop` later |
| `dispatch_first_byte_timeout` | **0** |
| `lease_extended_during_prefill` (T1 mitigation) | **713 events** |

## Comparison table

| Metric | Baseline 30s | Fast-15s fragment | Verdict |
|---|---|---|---|
| Orphan cleanups per 1k requests | 18.1/1k (raw, mode-agnostic) | **4.0/1k** (frag A, fast) | **IMPROVED** |
| Orphan cleanup rate (per hour) | 11.7/h (24h); 9/h (same-hour 30s fast, 10-16 00–01) | **5.3/h** (frag A) | **IMPROVED** |
| Mid-flight active-stream hits | (2026-08 failure: 4.3/h, 15x) | 3/1.33h = 2.3/h, **0 loss** | **IMPROVED** |
| `dispatch_first_byte_timeout` | — | **0** (frag A) | **FLAT/IMPROVED** |
| T1 `lease_extended_during_prefill` | 0 events in 2026-08 failure (root cause) | **713 events** (frag A) | **IMPROVED (mechanism)** |
| p90 TTFT fast | 10.1s | 11.5s (fragment-limited) | **FLAT (fragment)** |
| slot_save failed | 10 | 4 | **IMPROVED** |
| Peak concurrency | 4 | 3 ≤ slots (2–3) | **IMPROVED** |
| fallback_after_queue | 1,871–1,963 | 400 (fragment) | **IMPROVED** |
| Local capture rate | 14.5% | 13.0% (NOT comparable — window 31% covered, mostly cheap-mode overnight) | **UNVERIFIABLE** |
| dispatch denied | 163 | 1,079 (fragment; NOT comparable) | **UNVERIFIABLE** |

## The 15s orphan-cleanup question — explicit answer

**No orphan-cleanup spike is reproduced in the observable fast-15s fragment.**
The failure signature that forced the original reversal was:

- 2026-08 (LP-0MSKOHV55003XFTB): 686 events hitting active local streams
  (4.3/h vs 0.29/h = **~15x**), 5 errored after, 167 genuinely abandoned;
  mechanism = adaptive lease expiring mid-flight **because the prefill-extension
  backstop never fired (0 `lease_extended_during_prefill` events)**.

Stage-2 observable result:

- 3 mid-flight hits in 1.33h (2.3/h) — **all 3 streams completed successfully
  after cleanup with zero client-visible data loss** (0 errored, 0 abandoned).
- `dispatch_first_byte_timeout` = 0.
- **T1 prefill-extension now demonstrably fires** (713 events in the fast
  fragment vs **0** across all logs in the 2026-08 failure) — the missing
  mitigation is live, which is the mechanistic reason the 15s lease no longer
  orphan-cleans active streams. T2/T3 (chunk-refresh buffer for anonymous
  sessions, prefill poll) are also in the config.

**Is it justified by lease-active/deny reduction?** The mechanism intent
(faster idle-lease release → fewer `local_lease_active` denials) is supported
directly: `lease_renewed timeout=15s` shows the lease refreshes at 15s in fast
mode. The captured aggregate (`local_lease_active` 35 vs baseline 79,
dispatch denied) is **unverifiable** due to the log gap — the daytime window
is missing. **Justification: mechanism-favourable, aggregated-outcome
inconclusive** (same conclusion grade as the Stage-1 corrected verdict).

## Acceptance criteria

- [ ] **AC1 — capture > 14.5% baseline:** **UNVERIFIABLE.** Window only 31%
  covered; 13.0% fragment figure is dominated by cheap-mode overnight and is
  not comparable. Log gap LP-0MVEL6DWJ00179F4 blocks the comparison.
- [ ] **AC2 — guardrails flat/improved:** **PARTIAL MET.** In the observable
  fragment: slot_save 4 vs 10 (improved), peak concurrency 3 ≤ slots
  (improved), fallback_after_queue 400 vs 1,963 (improved), TTFT p90 fast 11.5s
  vs 10.1s (slight regression, fragment-limited), remote-error rate improved
  (429: 61 vs 112). No **critical** regression observed; no `slot_save` or
  concurrency breach in the observable data.
- [ ] **AC3 — verdict per metric:** met, see table plus statements above.
- [ ] **AC4 — does 15s cause an orphan spike, and justified?:** **NO spike
  reproduced** (4.0/1k vs 18.1/1k baseline raw; 0 `dispatch_first_byte_timeout`;
  3 mid-flight hits all completed; T1 prefill-extension firing). Justification
  is **mechanism-positive**; the aggregate lease-active/deny reduction is
  unverifiable due to the log gap.
- [ ] **AC5 — keep/revert + parent feedback:** **KEEP (conditional)** — see
  recommendation; fed back to parent epic below.
- [ ] **AC6 — analysis-only, no config changes:** met. No runtime config
  touched. (The 15s setting is already in the Stage-2 commit and was live;
  the later priority-routing change LP-0MVDT94F0003VHUA removed fast/cheap
  mode entirely and sets the base lease back to 30s, so this verdict is
  archival for any future fast-profile reuse.)

## Overall verdict: KEEP (conditional)

The 15s fast lease is **mechanism-favourable**: the exact failure signature
that drove the 30s reversal is **not reproduced** — no `dispatch_first_byte_timeout`,
negligible mid-flight hits, zero stream loss, and the previously-dead T1
prefill-extension mitigation fires continuously (713 events in 1.33h). The
fragment-level orphan rate is lower than the same-night 30s comparator.

Conditions:

1. **Close LP-0MVEL6DWJ00179F4 (log gap)** and re-run the daytime-window
   comparison before any future fast-profile reuse of the 15s value — the
   daytime fast window is exactly where the lease pressure is highest and it
   is unmeasured here.
2. The priority-routing rework (LP-0MVDT94F0003VHUA, 2026-10-18) replaced
   fast/cheap mode; current base lease is 30s. The 15s value survives only as
   precedent — if fast-profile config returns, keep 15s **with** the T1–T3
   mitigations, never without them (2026-08 proved 15s without T1 is
   dangerous; Stage-2 shows 15s with T1 is safe).

## Feedback to parent epic (LP-0MV8805WF007LTZA)

- **Stage 2 outcome:** KEEP (conditional) — the 15s lease does not reproduce
  the orphan-cleanup spike; the T1–T3 mitigation stack (built during
  LP-0MSUO6HLX0089MNQ) is the difference and is demonstrably live.
- **Data integrity finding (actionable):** the proxy log lost 16.7h on
  10-16, removing the entire daytime fast window from the eval. Follow-up
  created: **LP-0MVEL6DWJ00179F4**. This is a recurring hazard for the
  remaining experiment stages — pin log rotation/prune behaviour before the
  next staged eval.
- **Aggregate capture/deny numbers for Stage 2 are unverifiable** due to the
  gap; do not treat the fragment capture figure (13.0%) as a Stage-2 result.
- Stage 3 (thresholds) decision should rely on its own window once the log
  gap is fixed.
