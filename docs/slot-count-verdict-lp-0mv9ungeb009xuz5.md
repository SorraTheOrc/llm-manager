# Slot-count verdict: fast/cheap slot-count equalisation (LP-0MV9UNGEB009XUZ5)

**Item:** Evaluate performance impact of equalising fast/cheap slot counts (3/2/1); if not viable implement a drain.
**Verdict:** **SINGLE SLOT COUNT IS VIABLE — N = 2** for both modes.
**Date:** 2026-10-17

---

## Recommendation

Use **`session_slot_pool_size: 2` for BOTH fast and cheap** (equalise at 2), and
remove the restart-on-switch cost by converging the configs so a mode switch is a
**proxy-only restart** via the existing llama-server adopt path
(LP-0MUCEFCL6001ZXYN) — the running llama-server is kept warm across the switch.

Rationale, evidence-backed:

1. **N=2 costs zero local-eligible context vs N=1.** The effective WARM routing
   ceiling is the same `100,000` at N=1 **and** N=2 (the latency-derived economic
   ceiling `round(0.3815 × 262144) = 100,000` binds; physical per-slot clamps of
   258,048 / 126,976 are above it). Only **N=3** drops the ceiling to the physical
   clamp **83,285**. So going from today's fast 1-slot to 2 slots loses **no**
   local-eligible traffic; going to 3 slots pushes the 83–100k band remote
   (~4.9% of fast / 3.4% of cheap routing checks — see LP-0MVAZQS82004YTE9).

2. **N=2 halves the queue-timeout / contention rate vs N=1 under the observed
   concurrency.** Two sweep runs at 3 concurrent sessions (≥ observed 2–3 +
   queue depth) measured:
   - N=1: **70.6–75%** of queue-path arrivals timed out and fell back remote.
   - N=2: **20–25%** (12/12 completed both runs, zero errors).
   This matches the item's own contention evidence (fast 1-slot 10:00–14:02 →
   173 `fallback_after_queue`; cheap 3-slot → 207 over a longer window). The
   1-slot pool under 3 concurrent sessions saturates; the 2-slot pool absorbs it.

3. **N=3 gives little extra and costs context.** 3 slots would absorb the queue
   better (cheap's 207 was over a longer window), but it pays the 83,285
   context clamp and adds per-slot memory headroom pressure on a 262,144 ctx
   model (3× physical KV slots). The 83–100k band (~5%) is pushed remote, and
   the routing clamp change silently reduces what local can capture.

4. **Viability signals were clean at N=1 and N=2**: llama-server started at the
   requested `--parallel` every time (N=3 had one observed-slot mismatch in the
   invalid run; recorded, not hidden). No OOM; system MemAvailable ~84 GB at
   all runs.

5. **The mode-switch lock (LP-0MVCBMLP7009MN81)** now lets a performance
   investigation hold the mode, and the bounded drain (LP-0MT631JKW008WAKE)
   remains the safety net for in-flight streams. Converging slot counts makes
   the switch itself non-destructive, which is the whole point of removing the
   per-flip restart cost.

## Measured trade-offs (AC2)

| N | effective WARM (local-eligible ctx) | queue-timeout rate @3-conc | llama start | system mem | local-eligible lost |
|---|------------------------------------:|---------------------------:|-------------|-----------:|--------------------:|
| 1 | 100,000 | **70.6% / 75%** | clean | ~84 GB | none (today's fast baseline) |
| 2 | 100,000 | **20% / 25%** | clean | ~84 GB | **none vs N=1** |
| 3 | 83,285 | (cheap 207 long-window; run-1 slot mismatch) | 1 start mismatch in invalid run | ~56 GB observed | 83–100k band: ~4.9% fast / 3.4% cheap |

No OOM at any N. GPU memory capture was degraded (~0.2 GB reading) and is
reported as non-available rather than hidden.

## Fidelity caveats (AC1/AC2, explicit)

- **Local-slot decode share could not be isolated in the live sweep**: both runs
  returned `local share 0.0%` because live audit/herdr sessions (including this
  investigation's own herdr session) held the local dispatch lease
  (`local_dispatch_denied owner=…`), so benchmark requests queue-timeout to
  remote. Fresh-run TTFT/TPS therefore reflect remote fallback, NOT local
  decode speed — recorded fully in LP-0MVAZPQUM006EAC7's validity note. The
  queue-timeout rates above ARE valid and reproducible (they measure contention
  under real concurrency), which is what the 1/2/3 slot decision hinges on.
- Per-slot memory (full KV footprint) was not measured directly; recorded where
  available, non-available reported explicitly.
- The 83–100k-band loss at N=3 (traffic-mix) is derived from the routing
  `estimated_tokens` distribution over 6 days, not from a controlled N=3 run.

## Verdict branch: VIABLE

Per the task's conditional: **a single slot count is viable** → create the
follow-up work item to converge `session_slot_pool_size` across
`config.yaml` / `config-fast.yaml` / `config-cheap.yaml` and switch modes
non-destructively via the adopt path. The already-implemented bounded drain
(LP-0MT631JKW008WAKE) remains as the safety net regardless.

## Files / evidence

- Sweep results + validity note: LP-0MVAZPQUM006EAC7 (comments), committed under
  `proxy/benchmarks/results/slot_sweep_1_2_3_20261017_1421/` (commit 28e9e61e),
  harness extension (commit 28e9e61e, child 1).
- Traffic-mix analysis: LP-0MVAZQS82004YTE9 (comment); per-slot context derived
  from `_effective_large_context_thresholds` (economic ratio 0.3815, abs warm
  100,000, ctx 262144, headroom 4096).
- Items: LP-0MVCBMERY004SHCK (slot-2 baseline commit), LP-0MVCBMLP7009MN81
  (mode-switch lock), LP-0MVBJXZY5002UW10 / LP-0MVBJPQE50017BGL (auth-key
  resolver, unblocked the sweep).
