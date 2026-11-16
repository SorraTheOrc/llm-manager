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

Since `LP-0MVV29YDQ00437N6` the same line also carries `occupant_hold_s`, the
**measured** occupant hold described below.

### Blocker-hold pricing (LP-0MVV29YDQ00437N6)

Evidence: [`queue-wait-abandonment-diagnosis.md`](queue-wait-abandonment-diagnosis.md)
(`LP-0MVV29XPT003OXI5`, parent `LP-0MVV1V213002QAIR`). The per-wait diagnosis
of the post-change 24 h window measured the occupancy term at **exactly** the
symmetric `eta_prefill_s × busy_slots` floor in **163 of 169** abandoned waits
that had an admission decision: median projection **28.4 s** against the **60 s**
budget actually spent, median shortfall **31.6 s**. So the measured
occupant-remaining term above was inert for the population that loses, and a
holder that kept the slot for the whole wait was priced as if it owed nothing.
The dominant blocker class was `prefill_in_flight` (**65.3 %**);
`lease_holder_generating` was a **13.4 %** minority whose fairness capping stays
owned by `LP-0MVOJ6KAW004EBV0` (this change does not cap renewals).

The change prices, per live prefilling occupant record, the **remaining
dispatch-lease hold** (`expires_at`, the record's own monotonic bound on how
long the local slot stays leased to that holder):

```
occupancy = max(eta_newcomer x busy_slots,                      # floor
                sum(max(occupant_prefill_remaining, lease_hold)))  # measured
```

Properties (all tested in `proxy/tests/test_occupant_hold_pricing.py`):

- **Measured evidence only raises.** The symmetric floor keeps its exact
  meaning; the measured term can never lower it, and a record whose lease has
  expired (or is missing/invalid) contributes nothing.
- **Deliberately conservative direction.** A live lease is an *upper* bound on
  how long the slot stays away (the holder may release early), so the term can
  refuse a wait the queue might have served. That is the documented trade-off:
  refusing at admission is strictly cheaper than burning the whole budget and
  detouring remote anyway, which is what the diagnosis measured happening in
  60.7 % of enqueued waits (`LP-0MVV29XDW006VTWK` baseline).
- **Fail-open preserved.** No usable occupant evidence ⇒ today's floor,
  byte-for-byte. The guard still never prices from a guessed multiplier.
- **Only the diagnosed class.** Only records that are genuinely *prefilling*
  (`active`, no first byte yet) are priced — exactly the existing
  measurement eligibility, so the generating minority keeps today's treatment.
- **No new knob.** The term is not configurable; there is no operator setting
  to tune, disable or weight it.

Attribution: `occupant_hold_s` on the `local_admission_guard` line reports the
measured hold the decision priced with (`0.0` = no usable evidence, never "the
occupant owed nothing"), so a refusal can be attributed to measured blocker
class rather than inferred from `eta_occupancy_s` alone.

## Guardrails (unchanged)

The guardrail values are the shipped ones and were **not** altered by any of
these changes (the queue budget and deadline have their own swept history —
`contention_queue_max_wait_seconds` 60 → 30 in `LP-0MVT1146V007WXA4`,
`local_prefill_deadline_seconds` 120 → 180 — recorded in `proxy/config.yaml`):

```
contention_queue_max_wait_seconds            = 30
local_prefill_deadline_seconds               = 180

30 < 180 < 840 < 900
(contention_queue_max_wait_seconds
 < local_prefill_deadline_seconds
 < local_dispatch_first_byte_timeout_seconds
 < local_dispatch_max_prefill_seconds)
```

The ordering must keep holding; it is asserted from the *shipped* config by
`proxy/tests/test_prefill_budget_config.py`
(`test_shipped_config_passes_the_ordering_validator`,
`test_ordering_holds_for_the_documented_values`). No new configuration knobs
were added: the contended window reuses
`local_admission_guard_min_samples`, and the blocker-hold term
(`LP-0MVV29YDQ00437N6`) is not configurable at all.

## Rollback

The documented rollback is unchanged — **no new knob**:

- `server.local_admission_guard_policy: observe` — log the decisions (including
  `rate_bucket` and `occupant_hold_s`) but never refuse. Use to validate the
  contended calibration and the blocker-hold term on live traffic.
- `server.local_admission_guard_policy: off` — do not evaluate the guard at
  all.
- `server.local_prefill_deadline_seconds: 0` — disable only the deadline bound.

Reverting a behavioural change is a code revert of the feature children below
(including the blocker-hold term, `LP-0MVV29YDQ00437N6`); the config values in
the guardrail block must not be altered by the rollback.

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
| 6 | `LP-0MVV29XDW006VTWK` | Diurnally matched pre-change baseline restatement — 52.6 % working baseline (record: [`prefill-admission-conversion-diurnal-baseline.md`](prefill-admission-conversion-diurnal-baseline.md)) |
| 7 | `LP-0MVS9Z034006MFZ6` | Docs and rollback (this page) |
| 8 | `LP-0MVV29YDQ00437N6` | Blocker-hold pricing: price the diagnosed `prefill_in_flight` blocker class (dispatch-lease hold) in the occupancy term — evidence: [`queue-wait-abandonment-diagnosis.md`](queue-wait-abandonment-diagnosis.md) |
| 9 | `LP-0MVV29YP6006PLNP` | Post-fix 24 h re-measurement (record: [`prefill-admission-conversion-postfix.md`](prefill-admission-conversion-postfix.md)) |

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
early-abort feature, not a defect.

The **working baseline** for that share is the **diurnally matched** pre-change
window `2026-10-28 00:10:18` → `2026-10-29 00:10:18` (24 h, ending exactly at
the deploy, fully pre-change code): **52.6 %** (71/135), recorded in
[`prefill-admission-conversion-diurnal-baseline.md`](prefill-admission-conversion-diurnal-baseline.md)
(`LP-0MVV29XDW006VTWK`). The like-for-like change is `52.6 % → 60.7 %`
(**+8.1 points**), and it was not cleared — that regression is tracked in
`LP-0MVV1V213002QAIR`. The `36.9 %` figure from the `19:00`→`19:00` window is
retained as the **historical** pre-change reference, not as the working
baseline: 15.7 of the 23.8-point move from it is diurnal mix, since both
pre-change windows are served by identical code. The literal **85 %** bar is
unchanged.

The **post-fix re-measurement** (occupant-hold build `bed25a46c`, re-anchored
window `2026-11-06 14:16:00` → `2026-11-07 14:16:00` because the nominal
fix-anchored window was voided by mid-window redeploys and a backend
crash-loop) is recorded in
[`prefill-admission-conversion-postfix.md`](prefill-admission-conversion-postfix.md)
(`LP-0MVV29YP6006PLNP`): the abandoned-queue-wait share is **82.1 %** (23/28)
— still below the literal 85 % bar, but above both the 52.6 % working
baseline (+29.5 points) and the 60.7 % pre-fix post-change share (+21.4
points). The queue volume collapsed (28 waits vs 136/517) and the per-hour
abandoned rate fell to 0.96 /h (from 13.1 /h pre-fix, 2.96 /h baseline), so
the residual queue is tiny but its conditional abandonment is high. Two
configuration confounds bind the comparison and are stated in the record: the
queue budget was swept 60 s→30 s and depth 8→2 on 2026-10-31
(`LP-0MVT1146V007WXA4`, deployed before this fix), and the measured window is
diurnally 14:16→14:16 vs the 00:10→00:10 baseline.
