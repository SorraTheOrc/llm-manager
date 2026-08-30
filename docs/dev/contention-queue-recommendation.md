# Contention-queue tuning — recommendation and config change proposal

**Work item:** LP-0MTF0G5DJ0029LOR (T4 of LP-0MTED3OFP006I7NO) — delivered on the
parent epic LP-0MTED3OFP006I7NO (AC3/AC4).
**Data sources:** T1 simulation harness (b9daa46), T2 burst profile
(d0de129, `contention-queue-burst-profile.md`), T3 projected-gain analysis
(a26bc66, `contention-queue-queue-caps-vs-slots.md`), prior context-size work
(LP-0MSAOQTJS000FFVM / LP-0MSY0SDAS0031Y7F) and the 60K cold-threshold guardrail
experiment (LP-0MSRM54YO007YG0K AC7).
**Analysis window:** 2026-08-22 22:42:05 → 2026-08-28 00:00:00 (8 cheap-mode
proxy windows, T2/T3 corpus).

## Recommendation (summary)

| Lever | Verdict | Quantified impact / window |
|-------|---------|---------------------------|
| Cheap-mode queue caps | **YES — small, config-only change** (wait 60→120s, depth 4→8) | **+35 local dispatches** (~20% of 172 fallback-after-queue); p95 queue wait 120s (model-projected) |
| Cheap-mode slots 2→3 | **NO — defer** (context-bypass penalty) | +1237 lower bound (~17× the queue-cap lever; converts client-visible 503s) — revisit only after the bypass rate is re-measured |
| Fast-mode queue policy | **No change** (policy=fallback, no queue) | n/a — fast-mode invariant preserved |

### Why queue caps now (cheap mode only)

- T3 sweep over the replay corpus: raising `contention_queue_max_wait_seconds`
  60→120 saves **+35 dispatches/window** (172 → 137 fallback-after-queue) at
  **depth 8** with **zero depth-capped fallbacks** beyond a single marginal event
  (120s/8: 1 capped; 120s/12+: 0). T2 confirms the observed queue never exceeds
  depth 4 at the current 60s wait, so depth 8 provides headroom without inviting
  sustained deep queues.
- 180s/300s give diminishing returns (+54/+74) and push model-projected p95
  queue wait to the new cap (180–300s) — **not recommended**; the extra ~20–39
  dispatches cost minutes of added latency and raise the risk of tripping the
  post-deploy guardrail signature (sustained deep-queue samples, fa Δ ≤ 2×
  baseline 230) documented in LP-0MSRM54YO007YG0K AC7.
- Config-only: zero VRAM, zero context risk, no restart of llama-server required
  to change the caps themselves (slots are the restart-required knob; see
  rollout notes). Fast mode keeps `policy: fallback` — untouched.

### Why not slots (now)

- Slots 2→3 projects **+1237** denied→served-local (a **~17×** larger effect;
  the denied stream is the lease-gate 503s at `max_local`=2, quantified in T3).
  But the slots-vs-context trade-off is a **prior NO-GO for 3×87.4K** in fast
  mode (LP-0MSAOQTJS000FFVM; operator supersede LP-0MSY0SDAS0031Y7F kept fast at
  3 slots with the bypass cost accepted). Cheap mode 2×131072 → 3×87381 would
  shrink per-slot context ~33% and is expected to increase
  `large_context_bypass`/`context_too_large` — the exact rate was not
  re-measured (AC4 reuse), so it is recorded as an **OPEN QUESTION**, and the
  +1237 is stated as a **lower bound** (queue-path arrivals would also mostly
  dispatch direct at 3 slots; not double-counted).

## Proposed config change (config-cheap.yaml)

```yaml
  contention_queue_policy: queue            # unchanged
  contention_queue_max_wait_seconds: 120    # was 60
  contention_queue_max_depth: 8             # was 4
```

- Clamps honored: wait ∈ [1, `session_guardrail_max_runtime_seconds`=1800] ✓
  (120 ≤ 1800); depth ∈ [1, 16] ✓ (8 ≤ 16) — router.py:176-200.
- Adaptive-timeout interplay (Q2=a, F4 AC2): queue wait is subtracted from
  `llama_adaptive_timeout_*` (`_set_queue_wait_on_request`, provider.py:2105), so
  total (queue wait + serve) stays within the adaptive budget; a 120s queue wait
  cannot extend a request past its timeout budget. User-visible stalls grow only
  for the ~35 converted requests and are still bounded by the adaptive timeout
  (base 60s + 0.015/token for local).

## Rollout plan

1. Apply the two-line diff at the cheap-mode transition (01:00) so validation is
   isolated to cheap hours; confirm the caps are picked up (log line prints the
   resolved caps — if the caps are not hot-reloadable, restart at the transition;
   slots otherwise need no restart).
2. Observe 8 cheap windows on the same guardrails as LP-0MSRM54YO007YG0K AC7:
   - `contention_fallback_after_queue` must not exceed 2× baseline (~230) — it
     is expected to DECREASE toward ~137;
   - no sustained deep-queue samples (T2 max observed depth 4; keep depth metric
     away from the 8 cap);
   - `local_dispatch_denied` unchanged (slots untouched — this lever does not
     touch the lease gate).
3. If guardrails hold (2 windows), extend to steady state; if any guardrail
   trips, revert to 60/4 immediately (config-only revert, no restart of server
   state).

## Risks

- **Added queueing latency**: the ~35 converted requests stream from local after
  up to ~120s queue wait instead of falling back within 60s. Bounded by the
  adaptive timeout (above), but worst-case interactive latency in cheap hours
  (01:00–10:00) increases for those requests.
- **Guardrail signature**: sustained deep-queue samples are the documented
  failure signature of the 60K experiment (LP-0MSRM54YO007YG0K AC7). Depth is
  capped at 8 and T2 shows observed depth ≤ 4, but this is the primary thing to
  watch.
- **Depth clamp [1,16] / wait clamp [1,1800]**: recommended values fit; a future
  300s+ wait would sit at the p95 cost described above, not a clamp violation.

## Assumptions and OPEN QUESTION entries (recorded on LP-0MTED3OFP006I7NO)

- OPEN QUESTION 1 — **Live validation**: is a live 7–8 window validation run
  required before this goes to steady state, or is the replay-modeled evidence
  (T1 validation ±10% on counts) sufficient for the config change? (Impl item
  rollout includes the guardrail-observation plan; approval of the run is
  operator-side.)
- OPEN QUESTION 2 — **Cheap 3-slot bypass rate**: the prior NO-GO measured fast
  mode 3×87.4K; cheap 3 slots would run the same per-slot context. Re-measure
  `large_context_bypass`/`context_too_large` at 3 cheap slots before reopening
  the slots lever — it is the only missing number between +1237 (lower bound)
  and the slots verdict.
- Assumption: T1/T2/T3 replay-modeled counts (Δ) are the decision basis;
  model-projected absolute waits skew high vs observed (T3 report caveat) and
  are used only as upper bounds.

## Follow-up

- Implementation item created (see worklog): **"Raise cheap-mode contention
  queue caps (wait 60→120s, depth 4→8)"** — concrete diff above, rollout +
  guardrail plan, references this item (discovered-from LP-0MTF0G5DJ0029LOR,
  related-to LP-0MTED3OFP006I7NO).