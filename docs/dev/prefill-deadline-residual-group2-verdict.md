# Group-2 validation — stall-weighted verdict with the honest split

**Work item:** LP-0MW8EAUQU006DB0U (AC5 of the group-2 validation epic
LP-0MW7V14DN007F127).
**Window (group 2):** `2026-11-16 15:20:47` → `2026-11-17 18:20:00` UTC —
**26.987 h**, restart-free (the single group-2 restart is the window's own boot;
certification on LP-0MW8EAU8P0079280).
**Dataset (group 2):**
[`prefill-deadline-residual-group2-24h.json`](./prefill-deadline-residual-group2-24h.json)
(dataset + measurement narrative in
[`prefill-deadline-residual-group2-24h.md`](./prefill-deadline-residual-group2-24h.md)).
**Group-1 comparator:** `2026-11-10 14:18:26` → `2026-11-11 15:18:26` UTC,
**25.0 h**; committed
[`group1-prefill-deadline-residual.json`](../../proxy/docs/validation/group1-prefill-deadline-residual.json)
(recomputed today for its stall block — see §1).
**Measurement tool:** `proxy/scripts/analyse_prefill_deadline_residual.py`.

This is a **reporting** document. It changes no admission decision, threshold,
config or proxy behaviour; it re-tunes nothing to make the numbers land. Every
figure is reproducible from the committed datasets with the commands in §8.

## 0. Verdict at a glance

Certified criterion (producer decision 2026-11-02, option **b2**): user-class
`local_eviction_after_acceptance reason=prefill_deadline` +
`reason=measured_cache_deficit` ≤ **0.5 events/h** over a ≥ 24 h restart-free
window; **not certified** if occupancy-attributed ≥ 25 % or
capacity-attributed ≥ 50 % of the `prefill_deadline` events. Stall-weighted
target (producer decision 2026-11-08, Appendix Q1): a **≥ 90 % reduction in
user-class Σ `elapsed_s`/h versus the group-1 window**, recomputed with the same
analyser command.

| Test | Group 1 (25.0 h) | Group 2 (26.987 h) | Target | Verdict |
|---|---:|---:|---|---|
| **Combined (b2) event rate** (`prefill_deadline` + `measured_cache_deficit`, user) | 0.72/h (18 events) | **0.148/h** (4 events) | ≤ 0.5/h | **rate met** — but (b2) **NOT CERTIFIED** on the guard below |
| Occupancy guard, `prefill_deadline` | 50 % (2/4) | **100 %** (1/1) | < 25 % | **guard FAILS** (one-event base) |
| Capacity guard, `prefill_deadline` | 50 % (2/4) | 0 % (0/1) | < 50 % | guard met (one-event base) |
| **Combined (b2) stall-weighted, user** | 30.444 s/h | **6.844 s/h** | ≥ 90 % reduction | **NOT MET** — **−77.5 %** |
| **All-reasons user stall-weighted** (honest split) | 60.572 s/h | **230.556 s/h** | (same target read broadly) | **NOT MET** — **+280.6 %** (×3.81) |

**One-line reading.** On the two reasons the (b2) criterion names, the event
rate fell to **0.148/h** (met against ≤ 0.5/h) and their stall cost fell
**77.5 %** — but that is short of the **≥ 90 %** producer target, and the
certification still fails because the single surviving `prefill_deadline` event
is 100 % occupancy-attributed. Worse for the stall-weighted reading: across
**all** user-class post-acceptance reasons the stall cost **rose 3.81×**
(60.572 → 230.556 s/h) because the early-abort class `no_progress_watchdog`
moved the cost out of the counted reasons rather than removing it. The
counted-reason improvement is a **relabelling**, not an elimination — the exact
split AC5 exists to expose (§5).

## 1. Window and method

Group 2 was measured by the sibling item LP-0MW8EAU8P0079280 over the certified
restart-free window `2026-11-16 15:20:47` → `2026-11-17 18:20:00` (26.987 h),
anchored on the single group-2 proxy restart (`docs/dev/group-2-restart.md`,
PID 3720326, HEAD `79f53ac1a`). Window certification (no proxy restart, no
llama-server/TTS death or self-heal, no config reload, `/health` healthy) is
recorded there and on its item; this document takes it as given.

**Stall-weighted method (AC3).** The analyser's `stall_cost` block has landed by
run time (`analyse_prefill_deadline_residual.py`, LP-0MW7V30AZ007JB5Z), so:

- **Group 2** uses the analyser's own `stall_cost` block in the committed
  dataset (the block splits `Σ elapsed_s / hours` by reason and by session
  class, with p50/p90/p95/p99 and the under-1 s / under-5 s / at-near-deadline
  buckets). No recomputation was needed.
- **Group 1** predates that block — the committed
  `group1-prefill-deadline-residual.json` has no `stall_cost` key. It was
  **recomputed today with the same analyser version and the same command shape**
  over the group-1 window (no `--llama-log`; the Nov 10–11 `llama-server` logs
  are rotated away, and the stall block needs only `reason` / `session` /
  `elapsed_s`, none of which come from the backend log).

The group-1 recompute is method-faithful: it reproduces the committed dataset's
post-acceptance counts **exactly** — **23 evictions**, the same per-reason split
(`prefill_deadline` 4, `measured_cache_deficit` 14, `no_progress_watchdog` 4,
`lease_evicted` 1), the same **18** user-class combined events and the same
**0.72/h**. Only the stall block was added. The two windows are therefore
compared on an identical basis.

> **Basis caveat (stated, not hidden).** Group 2 was run with
> `--llama-log "/var/log/llama-proxy/llama-server*.log"` and group 1 without it.
> That flag moves only the abandoned-prefill window basis, backend accounting,
> `seconds_basis` and the release→cancel join — **not** `stall_cost`, which
> reads only `elapsed_s` from the eviction line. The stall-weighted comparison
> below is therefore like-for-like even though the two runs differ on the
> backend-log flag.

## 2. Per-reason and combined event rates (AC2)

Post-acceptance evictions and hourly rates, group 1 versus group 2, user and
`audit-*` split. Group-2 hours **26.987**; group-1 hours **25.0**.

| Reason | G1 total | G1 user/h | G2 total | G2 user | G2 audit | G2 total/h | G2 user/h | G2 audit/h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_progress_watchdog` | 4 | 0.160 | 184 | 173 | 11 | 6.818 | 6.411 | 0.408 |
| `lease_evicted` | 1 | 0.040 | 3 | 1 | 2 | 0.111 | 0.037 | 0.074 |
| `measured_cache_deficit` | 14 | 0.560 | 3 | 3 | 0 | 0.111 | 0.111 | 0.000 |
| `idle_timeout` | 0 | 0.000 | 2 | 2 | 0 | 0.074 | 0.074 | 0.000 |
| `prefill_deadline` | 4 | 0.160 | 1 | 1 | 0 | 0.037 | 0.037 | 0.000 |
| **Combined (b2)** `prefill_deadline` + `measured_cache_deficit` | **18** | **0.720** | **4** | **4** | **0** | — | **0.148** | 0.000 |

- **Combined user-class rate: 4 events / 26.987 h = 0.148/h** — **met** against
  the ≤ 0.5/h gate (group 1 was 0.72/h, itself not met).
- **`no_progress_watchdog`** rose from 4 events (0.160/h) to **184 (6.818/h)**:
  it is the dominant class in this window and sits **outside** the (b2)
  criterion as written. It is a genuine user-visible defect (an accepted
  dispatch released before its first byte) and is analysed in stall terms in
  §3/§5. Its mechanism (umbrella-label relabelling) is defect
  **LP-0MWKQJHSM000CI6W** and is **not** fixed here.

## 3. Stall-weighted cost and the ≥ 90 % target (AC3 / AC5c)

`stall_seconds_per_hour = Σ elapsed_s / window_hours`, separated by reason and
session class. User rows only (the audit class carries 13 events / 466.8 s /
17.297 s/h in group 2 and 0 in group 1; it never touches the (b2) pair).

### 3.1 Per-reason user stall cost

**Group 1 (25.0 h) — user**

| Reason | n | Σ s | s/h | p50 | p95 | p99 | max | <1 s | <5 s | ≥179 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prefill_deadline` | 4 | 744.2 | 29.768 | 181.35 | 198.535 | 200.827 | 201.4 | 0 (0 %) | 0 (0 %) | 4 (100 %) |
| `measured_cache_deficit` | 14 | 16.9 | 0.676 | 0.4 | 5.47 | 6.614 | 6.9 | 11 (78.6 %) | 13 (92.9 %) | 0 (0 %) |
| `no_progress_watchdog` | 4 | 743.1 | 29.724 | 187.3 | 188.295 | 188.379 | 188.4 | 0 (0 %) | 0 (0 %) | 4 (100 %) |
| `lease_evicted` | 1 | 10.1 | 0.404 | 10.1 | 10.1 | 10.1 | 10.1 | 0 (0 %) | 0 (0 %) | 0 (0 %) |

**Group 2 (26.987 h) — user**

| Reason | n | Σ s | s/h | p50 | p95 | p99 | max | <1 s | <5 s | ≥179 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_progress_watchdog` | 173 | 5 939.0 | 220.069 | 20.0 | 121.1 | 170.316 | 186.5 | 0 (0 %) | 0 (0 %) | 1 (0.6 %) |
| `prefill_deadline` | 1 | 180.1 | 6.674 | 180.1 | 180.1 | 180.1 | 180.1 | 0 (0 %) | 0 (0 %) | 1 (100 %) |
| `idle_timeout` | 2 | 87.2 | 3.231 | 43.6 | 48.91 | 49.382 | 49.5 | 0 (0 %) | 0 (0 %) | 0 (0 %) |
| `measured_cache_deficit` | 3 | 4.6 | 0.170 | 0.5 | 3.29 | 3.538 | 3.6 | 2 (66.7 %) | 3 (100 %) | 0 (0 %) |
| `lease_evicted` | 1 | 11.1 | 0.411 | 11.1 | 11.1 | 11.1 | 11.1 | 0 (0 %) | 0 (0 %) | 0 (0 %) |

The behaviour the metric is designed to separate is intact: a
`measured_cache_deficit` detour is **0.5 s at the median** (p99 3.5 s, two of
three under 1 s), whereas a `prefill_deadline` stall is **180.1 s** — a full
deadline wait. The event rate prices these identically (+1 each); the stall
rate does not.

### 3.2 Combined (b2) and all-reasons stall rates, and the target

| Aggregate (user class) | G1 Σ s | G1 s/h | G2 Σ s | G2 s/h | Change | p50 G1→G2 | p95 G1→G2 | p99 G1→G2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Combined (b2)**: `prefill_deadline` + `measured_cache_deficit` | 761.1 | **30.444** | 184.7 | **6.844** | **−77.5 %** | 0.45 → 2.05 | 185.165 → 153.625 | 198.153 → 174.805 |
| **All reasons, user** | 1 514.3 | **60.572** | 6 222.0 | **230.556** | **+280.6 % (×3.81)** | 1.1 → 20.0 | 188.33 → 125.17 | 198.54 → 177.888 |

Bucket shares (user): group-1 combined is 11/18 under 1 s and 4/18 at/near the
deadline; group-2 combined is 2/4 under 1 s and 1/4 at/near the deadline.
All-reasons, group-2 is **2 of 180 under 1 s** and 1 in 180 at/near the
deadline — the distribution moved from a few full-deadline stalls to a long
tail of mid-length aborts (§5).

**Target arithmetic (auditable, both windows).**

- **Combined (b2) user stall-weighted:** group 1 **761.1 s / 25.0 h =
  30.444 s/h**; group 2 **184.7 s / 26.987 h = 6.844 s/h**. Reduction
  `(30.444 − 6.844) / 30.444 =` **77.5 %**, which is **NOT ≥ 90 %** →
  **NOT MET**.
- **All-reasons user stall-weighted:** group 1 **1 514.3 s / 25.0 h =
  60.572 s/h**; group 2 **6 222.0 s / 26.987 h = 230.556 s/h**. That is a
  **280.6 % increase** (×3.81), the opposite of a reduction → **NOT MET** under
  the broad reading of "user-class Σ `elapsed_s`/h".

The producer's 2026-11-08 gate is applied to the **user-class combined**
stall-weighted rate (the counterpart of the ≤ 0.5/h combined event rate, as
recorded on LP-0MVO8H7VQ003B12I): **−77.5 % against a ≥ 90 % requirement →
NOT MET**. The all-reasons figure is reported beside it because it is the
honest total cost, and it makes the not-met verdict unambiguous.

## 4. Attribution, residency coverage and deadline-capacity refusals (AC4)

### 4.1 Attribution split (`prefill_deadline` events)

| Attribution | G1 events | G1 share | G2 events | G2 share |
|---|---:|---:|---:|---:|
| occupancy | 2 | 50 % | **1** | **100 %** |
| capacity | 2 | 50 % | 0 | 0 % |
| predicted-warmth | 0 | 0 % | 0 | 0 % |
| **n** | **4** | | **1** | |

The one group-2 `prefill_deadline` event
(`2026-11-17 09:50:49`, `session=herdr-1794900481-910524-32268`) priced
`estimated_tokens=43319` as only `new_tokens=228` (99.5 % assumed warm,
`priced_cached_ratio=0.995`), forecast `eta_prefill_s=89.4`, and stalled the
full `elapsed_s=180.1` on slot 1. It is occupancy-attributed:
`in_flight_requests_max=2`, `available_slots_min=0`, another session's local
stream completed inside its dispatch window. **On a one-event base the 100 %
occupancy share is an arithmetic fact about the event, not an estimate of a
population** — it nonetheless trips the (b2) `< 25 %` occupancy guard, because
a guard computed over `n = 1` is exactly as binary as this.

### 4.2 Residency-unmeasured share

For the population the analyser attributes (the `prefill_deadline` events), the
residency was **unmeasured at admission**: the sole group-2 event has
`cluster_measured_cached_ratio = null` (and `estimated_tokens = 43319 > 38 000`,
the cold threshold), i.e. **1/1 = 100 %** of the deadline events were admitted
with unmeasured residency, priced as (almost) warm. That is the same derivation
the group-1 verdict used (there, 4/4 = 100 %).

The **window-wide share of _all_ admitted prompts** whose residency was
unmeasured is **not carried by the committed dataset** and is not substituted
here: the analyser publishes `cluster_measured_cached_ratio` only on the
attributed deadline rows. Producing an all-admissions ratio would require the
per-admission routing-level `residency_source` / `measurement` fields
(`routing_check` / `routing_resident_prefix` lines) joined to each admission —
**the missing basis is those fields**, which the dataset does not store. Stated,
not estimated.

### 4.3 Admissions refused by the new deadline-capacity bound (AC5e)

The group-2 deadline token capacity shipped by LP-0MW7V2D13004ZO65 is enforced
at admission and publishes its **own distinct reason** on the
`local_admission_guard` decision line:

| Figure | Value |
|---|---|
| Log reason | **`prefill_eta_exceeds_deadline`** |
| Evidence field | **`deadline_capacity_tokens`** = **46 153** (plus `capacity_priced_tokens`, `deadline_capacity_calibrated`) |
| Refusals in the window | **214** (`local_admission_guard decision=reject`, all 214 carry `deadline_capacity_tokens=46153`) |

The count is **directly published**, not derived from the older flat
`prefill_budget_exceeded` reason: `prefill_eta_exceeds_deadline` is a separate
reason token introduced with the deadline-capacity bound, so a
`prefill_budget_exceeded` count (2 709 in this window — the 25 k cold budget)
**must not** be read as the capacity count. The sibling measurement narrative
([`prefill-deadline-residual-group2-24h.md`](./prefill-deadline-residual-group2-24h.md)
§3.4) records **217**; the exact-window recount here is **214**, the difference
being a handful of lines on the window boundary (three additional
`prefill_eta_exceeds_deadline` rejects fall after `18:20:00`). Either way the
capacity bound is live and non-inert (calibrated capacity 46 153 tokens at the
180 s deadline).

## 5. The honest split — relabelling, not elimination

The two halves of AC5 disagree in sign, and that disagreement is the finding:

- On the **counted reasons** (`prefill_deadline` + `measured_cache_deficit`),
  the event rate fell 0.720 → 0.148/h and the stall cost fell −77.5 %. That
  looks like a large win.
- On **all user-class reasons**, the stall cost **rose 60.572 → 230.556 s/h
  (×3.81)**, driven entirely by `no_progress_watchdog` (4 events / 743.1 s in
  group 1 → 173 events / 5 939.0 s in group 2).

The group-1 window already carried `no_progress_watchdog` at a **full-deadline**
shape (4 events, p50 187.3 s, 100 % at/near the deadline) — roughly equal in
cost to its `prefill_deadline` class. In group 2 that class grew by more than an
order of magnitude and changed shape (p50 20.0 s, long tail to 186.5 s): the
early-abort watchdog moving pre-deadline work out of the counted reasons.

This is a **reason conversion**, not a fix, and the mechanism is already
tracked: `e5af94f75` (LP-0MW8VOKB9002W2C1) computes the distinct
`prefill_eta_exceeded` / `prefill_stalled` / `prefill_max_exceeded` reasons and
emits them on `lease_released`, but the deferred release path still emits
`local_eviction_after_acceptance reason=no_progress_watchdog` — the line the
analyser counts (defect **LP-0MWKQJHSM000CI6W**, open). The counted
`prefill_deadline` rate therefore cannot be read without the stall-weighted
total beside it: an event rate can be improved by moving work into a class the
criterion does not count, while the client-visible stall cost goes up.

Read together:

- **≤ 0.5/h combined event rate:** the **rate is met** (0.148/h), but the
  (b2) certification is **NOT CERTIFIED** — the occupancy guard fails at 100 %
  on a one-event base, and the class that left the counted reasons did not
  leave the system.
- **≥ 90 % stall-weighted reduction:** **NOT MET** — −77.5 % on the combined
  pair, **+280.6 %** on all reasons.

## 6. Verdict

| Question | Answer |
|---|---|
| Per-reason user/audit rates (§2) | reported; `no_progress_watchdog` dominates (184 / 6.818/h, user 173 / 6.411/h) |
| Combined user-class event rate (§2) | **0.148/h** — **met** against ≤ 0.5/h |
| (b2) certification | **NOT CERTIFIED** — occupancy guard 100 % ≥ 25 % (n = 1); rate alone is not the criterion |
| Stall-weighted, combined (b2), user (§3) | **6.844 s/h** (group 1: 30.444 s/h) → **−77.5 %**, **NOT ≥ 90 %** |
| Stall-weighted, all reasons, user (§3) | **230.556 s/h** (group 1: 60.572 s/h) → **+280.6 % (×3.81)** |
| p95 / p99 stall seconds, user (§3) | combined 153.6 / 174.8 s; all-reasons 125.2 / 177.9 s |
| Attribution (§4) | occupancy 1 / capacity 0 / predicted-warmth 0 (n = 1) |
| Residency-unmeasured share, deadline events (§4) | **1/1 = 100 %**; all-admissions share not in the dataset (basis named) |
| Deadline-capacity refusals (§4) | **214**, reason `prefill_eta_exceeds_deadline`, field `deadline_capacity_tokens=46153` |
| **Producer ≤ 0.5/h combined rate** | rate **met** (0.148/h); **certification NOT MET** (occupancy guard) |
| **Producer ≥ 90 % stall-weighted reduction** | **NOT MET** (−77.5 % combined; +280.6 % all reasons) |

## 7. Limitations

- **One event carries the `prefill_deadline` verdict and the occupancy guard.**
  n = 1 makes the occupancy share 100 % and the capacity share 0 % by
  arithmetic. This is a statement about the event, not a population estimate.
- **`no_progress_watchdog` is not priced by (b2).** As written the criterion
  names only `prefill_deadline` and `measured_cache_deficit`; the dominant
  abandoned-work class in this window is outside it. Whether to fold it in (and
  whether to split the umbrella label, LP-0MWKQJHSM000CI6W) is a producer
  decision; this document only fixes the numbers and names the mechanism.
- **Backend-log basis differs between the two runs** (§1). The stall-weighted
  comparison is insensitive to it because `stall_cost` reads only `elapsed_s`;
  the deeper backend-accounting fields are not compared here.
- **Workload is not held constant.** Group 2 is 26.987 h against group 1's
  25.0 h, with a different session mix; the comparison is a like-for-like
  *rate*, not a controlled experiment.
- **All-admissions residency share is not producible from the dataset** (§4.2);
  the missing basis is the per-admission `residency_source` / `measurement`
  fields. Not estimated.

## 8. Reproduction

```bash
# Group 2 (uses the analyser's stall_cost block; identical to the committed dataset)
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-16 15:20:47" --end "2026-11-17 18:20:00" \
  --deadline-seconds 180 \
  --llama-log "/var/log/llama-proxy/llama-server*.log" \
  --json --output /tmp/group2-24h.json

# Group 1 (recompute of the stall block; the committed dataset predates it,
# and the Nov 10–11 llama-server logs are rotated away — no --llama-log)
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-10 14:18:26" --end "2026-11-11 15:18:26" \
  --deadline-seconds 180 \
  --json --output /tmp/group1-recompute-stall.json
```

The group-1 recompute reproduces the committed
`group1-prefill-deadline-residual.json` counts exactly (23 evictions, same
per-reason split, 18 user-class combined events, 0.72/h), which is the check
that the two windows are compared on an identical method.

**Headline for the work-item comment:** combined user-class **0.148/h (4
events, rate met, certification NOT MET on a 100 % one-event occupancy
guard)**; stall-weighted combined **30.444 → 6.844 s/h = −77.5 % (NOT MET vs
≥ 90 %)**; all-reasons user stall-weighted **60.572 → 230.556 s/h (+280.6 %,
×3.81)**, driven by `no_progress_watchdog` (umbrella label,
LP-0MWKQJHSM000CI6W) — a relabelling, not an elimination; **214** admissions
refused by the deadline-capacity bound (`prefill_eta_exceeds_deadline`,
`deadline_capacity_tokens=46153`).
