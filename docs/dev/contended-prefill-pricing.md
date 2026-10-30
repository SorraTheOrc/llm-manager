# Contended prefill pricing and rollback (LP-0MVS8T6F4006DGG3)

This page documents the contended (co-residency) prefill calibration and the
token-based occupant-remaining re-pricing shipped under the parent work item
**LP-0MVS8T6F4006DGG3** — _Condition prefill-ETA calibration on co-residency;
fix occupant-remaining pricing (measured 2-4x optimistic)_ — and how to roll
it back.

## Why

The prefill-ETA admission guard prices a local prefill from observed
`dispatch_first_byte_ms` samples. Prefill is serialised
(`local_max_concurrent_prefills: 1`) but prefill+decode batching is not: an
occupant that is **generating** halves a newcomer's prefill throughput
(measured 2-4x lower tok/s). Two defects followed from that:

1. **Mixed calibration window.** Solitary-dominant samples priced contended
   admissions optimistically. Measured 2026-10-28: a 6,659-token prompt was
   priced at `eta_prefill_s=13.9` (~479 t/s) against the same day's
   prompt-eval median of ~120 t/s (p25 85 / p75 202 / max 763).
2. **Optimistic occupant decay.** The occupant's remaining service was
   `max(0, eta_prefill_s − elapsed)` against its _optimistic_ admission ETA,
   so it converged to ~0 exactly when the occupant was most expensive. At
   2026-10-28 00:28:55 a cold 65-89k-token audit prompt was priced at
   `eta_occupancy_s=16.0`; the request queued the full 60 s, the gate never
   freed, and it fell back remote — a guaranteed loss admission was supposed
   to prevent.

## What ships

### Contended rate bucket (LP-0MVS9Y76R004B5M8)

- `record_dispatch_sample(..., contended=)` stores a sample in a **separate
  contended window**. A sample is contended when the gate was occupied at
  admission (`busy_slots >= 1`) **or** another session's dispatch record was
  live at first-byte time (the OR is deliberate; both signals are
  deterministic).
- `coefficients()` / `estimate_prefill_seconds(..., contended=True)` price from
  that window when it holds `local_admission_guard_min_samples` observations,
  and otherwise fall back to the solitary window, reporting
  `rate_bucket=solitary_fallback` — never a guessed slowdown multiplier.
- Each `local_admission_guard` decision line carries
  `rate_bucket=contended|solitary|solitary_fallback` and `contended_samples=<n>`.
- Warm prompts (`cached_ratio >= 0.9`) stay on the solitary path
  byte-for-byte.

### Occupant-remaining re-pricing (LP-0MVS9YK9P006M1NQ)

`_live_prefill_remaining_seconds` re-prices each occupant's remaining
**uncached new tokens** at the contended rate instead of decaying its
optimistic admission ETA:

```
progress      = min(1, elapsed / contended_full_service)
remaining_new = new_tokens * (1 - progress)
remaining_s   = estimate_prefill_seconds(remaining_new, cached_ratio=0,
                                         contended=True)
```

`contended_full_service` is the contended-rate cost of the occupant's whole
new-token load, so an occupant past its _optimistic_ ETA still reports the work
it genuinely owes. (The plan's shorthand wrote `elapsed / eta_prefill_s`; that
would cap progress at 1 the moment the occupant passed its optimistic ETA and
re-introduce the very convergence-to-zero defect, so progress is measured
against the contended full service instead.)

Fail-open is preserved: a missing/invalid `new_tokens`, `eta_prefill_s` or
`started_at`, an inactive or already-served record, and an uncalibrated
contended window all contribute nothing, leaving the symmetric
`eta * busy_slots` floor in charge. Measured evidence can only raise that
floor, never lower it.

### Observability surface (LP-0MVS9YKHL007UBBZ)

The `local_admission_guard` `LogEvent` carries `rate_bucket`,
`eta_occupancy_s`, `new_tokens` and `contended_samples`, and the
proxy-usage-analysis report breaks verdicts down by `rate_bucket` and by the
new-token bands used by `scripts/prefill_admission_conversion.py`.

## Guardrails (unchanged)

`contention_queue_max_wait_seconds` stays **60** and
`local_prefill_deadline_seconds` stays **120**; the ordering assertion

```
60 < 120 < 840 < 900
(contention_queue_max_wait_seconds
 < local_prefill_deadline_seconds
 < local_dispatch_first_byte_timeout_seconds
 < local_dispatch_max_prefill_seconds)
```

must keep holding. No new configuration knobs were added: the contended window
reuses `local_admission_guard_min_samples`.

## Rollback

The documented rollback is unchanged — **no new knob**:

- `server.local_admission_guard_policy: observe` — log the decisions (including
  `rate_bucket`) but never refuse. Use to validate the contended calibration
  on live traffic.
- `server.local_admission_guard_policy: off` — do not evaluate the guard at
  all.
- `server.local_prefill_deadline_seconds: 0` — disable only the deadline bound.

Reverting the behavioural change is a code revert of the two feature children
above; the config values in the guardrail block must not be altered by the
rollback.

## Plan: changelog — cross-reference

The parent decomposed the fix into these children (implemented in test-first
dependency order):

| # | Work item | Scope |
|---|-----------|-------|
| 1 | `LP-0MVS9Y7FG004QJ2S` | Pre-change admission→first-byte conversion baseline harness (`scripts/prefill_admission_conversion.py`) |
| 2 | `LP-0MVS9Y76R004B5M8` | Contended prefill rate bucket (co-residency calibration + `rate_bucket` log field) |
| 3 | `LP-0MVS9YK9P006M1NQ` | Occupant-remaining priced from new tokens at the contended rate |
| 4 | `LP-0MVS9YKHL007UBBZ` | Decision-log and usage-report surface for `rate_bucket` |
| 5 | `LP-0MVTKTOVI006ZONN` | Post-change 24 h measurement window vs the 85 % baseline (record: [`prefill-admission-conversion-postchange.md`](prefill-admission-conversion-postchange.md)) |
| 6 | `LP-0MVS9Z034006MFZ6` | Docs and rollback (this page) |

Related origin items: `LP-0MVKUKF65008D5UN` (conservative ETA for large
prompts), `LP-0MVICD2YW004DJPV` (occupancy-aware symmetric floor),
`LP-0MVOFVCA2008DQVE` (measured-occupant term) and `LP-0MVKUKEU80022C3U`
(deadline admission bound).

The 24 h post-change evidence for parent AC4 is recorded by
[`prefill-admission-conversion-postchange.md`](prefill-admission-conversion-postchange.md)
(delivered by `LP-0MVTKTOVI006ZONN`): the abandoned-queue-wait share is **60.7 %**
(314/517) over `2026-10-29 00:10:18` → `2026-10-30 00:10:18` UTC, below the
literal 85 % bar. AC4's second clause ("no new `local_eviction_after_acceptance`
events") is **not** met on the literal reading: the genuine comparator is
152 events (6.3 / h) pre-change and 392 (16.3 / h) post-change (×2.6), with
one genuinely new class, `measured_cache_deficit` (0 → 234). That class is
triaged in `LP-0MVV9JWL9003A6XR` as expected behaviour of the measured-cache
early-abort feature, not a defect. The
stricter working baseline (36.9 %) was not cleared — that regression is tracked
in `LP-0MVV1V213002QAIR`.
