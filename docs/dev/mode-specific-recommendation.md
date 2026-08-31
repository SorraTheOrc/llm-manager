# Mode-specific recommendation & follow-up (F5)

**Work item:** LP-0MTCMGJBM007AQ55 (parent LP-0MTAQNB7J0094X71)
**Reproduction:** `python3 scripts/mode_recommendation.py --markdown`

## Summary

Two modes, two priorities, **one shared fix set** (they do not converge on
cap values — fast 83,285 ≠ cheap 126,976 — but they select the same three
fixes; the parent risk "fast/cheap may converge" does not apply to the
selection, only the cap values differ).

## Fast mode (3 slots, cap 83,285 — TTFT/P95 priority)

- **Selected fix set:**
  1. Raise persistence cap to routing clamp + timeout/cooldown plan
  2. Session↔slot affinity / ownership continuity fix
  3. Restore-before-save ordering for hot sessions
- **Expected savings:** ~26.5M prefill tokens/day (cap raise 22.8M +
  affinity 3.3M + ordering 0.44M)
- **Fallback rate:** down — large sessions stay local (fewer remote
  fallbacks), but fast mode still tolerates fallbacks where needed.
- **TTFT/P95:** down — eliminates minutes-long full re-prefills on large
  sessions (the 0.14–0.2 t/s decode collapse from F2/F3 evidence).

## Cheap mode (2 slots, cap 126,976 — cost/local-utilization priority)

- **Selected fix set:** same three fixes (cap raise is the local-reuse
  lever; affinity + ordering prevent the restore miss that forces remote
  fallback).
- **Expected savings:** ~26.5M prefill tokens/day.
- **Fallback rate:** down — reliable restore is the lever that keeps
  oversized sessions local without full re-prefill (the cheap-mode cost
  driver per the parent AC).
- **TTFT/P95:** down (secondary to cost in this mode).

## GPU-wedge plan (LP-0MS91DHPZ001VWQO — applies to both modes)

Any persistence-cap raise carries the existing floor unchanged:

| Parameter | Value | Rationale |
|---|---|---|
| `session_slot_timeout_seconds` | 3.0s base | unchanged |
| `session_slot_timeout_per_token_seconds` | 0.0015 | unchanged |
| `session_slot_max_timeout_seconds` | 60s | hard cap — saves cannot wedge GPU indefinitely |
| `session_slot_max_consecutive_failures` | 3 | circuit breaker unchanged |
| `session_slot_failure_cooldown_seconds` | 300s | cooldown unchanged |
| `session_slot_skip_when_busy` | true | load-aware gate kept on (only 2 save failures observed; conservatism is acceptable) |

The cap itself moves only within the per-slot routing clamp
(`local_model_ctx_size // slots − 4096` headroom): fast 262144//3 − 4096 =
83,285 (unchanged), cheap 262144//2 − 4096 = 126,976 (unchanged). The
recommendation is that the cap **stays pinned to the clamp** (as of
LP-0MTBTCB8D000OQ0C) and the *restore path* is fixed — the gap is not that
the cap is wrong but that the gated-out sessions (F2) and the router /slots
500s (F3) prevent reuse.

## Expected impact, combined

- Fallback events: down across both modes (restore reuse reduces
  `local_concurrency_limit` and `context_too_large` fallbacks).
- TTFT/P95: down on large sessions (the 42.7M prefill-token/day re-prefill
  burden is the pool).
- Prefill tokens/day: −26.5M (best case per F4 model) if all three fixes
  land; the cap raise alone delivers 22.8M.

## Follow-up implementation work item

Created per parent AC #5 (no code change now):
**"Implement KV slot persistence fixes from the reuse-gap evaluation"** —
carries fast/cheap caps, the GPU-wedge plan, and references this parent.
No source code in `proxy/` or `ds4/` was changed by this evaluation (F1–F5
deliverables are scripts under `scripts/`, tests under `proxy/tests/`, and
docs under `docs/dev/`).

## Validated assumptions

- Fast/cheap converge on the fix **set** but not on cap values (parent risk
  documented: `notes.converged = false`).
- GPU-wedge mitigation: the timeout/cooldown floor is preserved verbatim;
  no cap is raised without it.