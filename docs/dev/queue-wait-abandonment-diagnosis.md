# Per-wait diagnosis of abandoned contention-queue waits

Why a queue wait that the admission guard admitted still expires at the
`contention_queue_max_wait_seconds` (60 s) budget — diagnosed **per wait**
rather than per session, with the guard's own projection for the *same request*
set against the wait actually observed, and with the class of blocker that held
the slot (`LP-0MVV29XPT003OXI5`, parent `LP-0MVV1V213002QAIR`).

This is a **diagnosis record only**: no proxy behaviour changed. It is the
evidence slice that `LP-0MVV29YDQ00437N6` (price the diagnosed blocker class in
the guard) and `LP-0MVV29Y1M004GAW0` (lease-active re-queue path) act on.

## Headline

- Of the **314** abandoned waits in the post-change 24 h window, **169 (53.8 %)**
  have an admission decision in their own session; **163 of those are `admit`,
  and every one of the 163 was projected below the 60 s budget** — median
  **28.4 s** against the 60 s observed (median shortfall **31.6 s**).
- The other **145 (46.2 %)** have **no admission line in their session at all**.
  They are the `local_lease_active` re-queue path, which enters the queue
  without a guard decision, so the guard never priced them. The same class was
  **5 of 36 (13.9 %)** in the diurnally matched pre-change window.
- The occupancy term the guard priced with is **exactly the symmetric
  `eta_prefill_s × busy_slots` floor in 163 of the 169** abandoned admits; the
  occupant's measured remaining service raised it in only **6**.
- The blocker was **prefilling**, not decoding, in **205 of 314 (65.3 %)**
  abandonments; `lease_holder_generating` is **42 (13.4 %)**. The parent's
  "decode/lease hold unpriced" shape is real but a **minority**.
- **Verdict: a modelling gap plus a coverage gap, amplified by volume — not
  genuine ETA optimism in the newcomer's own term.** The newcomer's own prefill
  term is calibrated (dispatched waits project 21.95 s against 28.33 s
  observed); what is unpriced is (i) the occupant's remaining service and
  (ii) the whole lease-active re-queue population.

## Windows and provenance

| Window (24 h, UTC) | Code | Queue waits | Abandoned | Share | Coverage |
|---|---|---|---|---|---|
| `2026-10-28 00:10:18` → `2026-10-29 00:10:18` | pre-change, ends at the `52085f09` deploy | 68 rows / 67 outcomes | 36 | **53.7 %** (36/67) | **partial** — 2 archives, 6 h 10 m of the window (see below) |
| `2026-10-29 00:10:18` → `2026-10-30 00:10:18` | `52085f09` | 517 | 314 | **60.7 %** (314/517) | **complete** — 7 archives, no adjacent-file gap |

Artefacts: [`queue-wait-abandonment-diagnosis.json`](queue-wait-abandonment-diagnosis.json)
(post-change) and
[`queue-wait-abandonment-diagnosis-diurnal-baseline.json`](queue-wait-abandonment-diagnosis-diurnal-baseline.json)
(pre-change surviving slice).

**Coverage, stated before any number is quoted.** The post-change window is
**fully covered** (`"covered": true`, no gaps; a 1 s lead-in at the window
start). The pre-change window is **not**: 7-day log retention pruned its first
16 h 5 m 56 s, and the two surviving archives leave a 1 h 43 m 48 s internal gap
(`2026-10-28 22:16:12` → `2026-10-29 00:00:00`). Its numbers are therefore a
**surviving-slice** measurement, never a full-window one — exactly the
provenance split the committed
[diurnal baseline record](prefill-admission-conversion-diurnal-baseline.md)
documents. The slice reproduces that record's abandoned share **exactly**
(36/67 = 53.7 %), which is the consistency check that makes it usable as a
comparator for *shape*, not for volume.

Totals agree with the committed
[post-change record](prefill-admission-conversion-postchange.md) exactly
(517 enqueued, 203 dispatched, 314 abandoned, 60.7 %), so the per-wait rows are
the same population that record counts.

## How each row is built

One row per queue wait (`--json` → `rows[]`). Full contract in the module
docstring of [`scripts/queue_wait_diagnosis.py`](../../scripts/queue_wait_diagnosis.py).

- **Enqueue time** comes from the `contention_queue_wait` line. The 2026-10-29/30
  log shape logs **no session** on that line, so the enqueue is paired to its
  outcome by time: the outcome's own `queued_duration` dates the enqueue and an
  unconsumed enqueue line within 2 s of that instant is its enqueue. Measured
  pairs agree to a few milliseconds (e.g. enqueue `14:12:39,052` → fallback
  `14:13:39,053`, `queued_duration=60.00s`). An enqueue with no outcome is still
  emitted as a row (one such row in the pre-change slice).
- **Trigger and cause** are read from the outcome line
  (`LP-0MVV29X2E004INFL`); the pre-change `reason=local_lease_active` token is
  unambiguous legacy trigger evidence and everything else degrades to
  `unknown`, never inferred from the duration.
- **Guard projection** is the correlated `local_admission_guard` decision's
  `projected_wait_s` / `eta_prefill_s` / `eta_occupancy_s` / `busy_slots` /
  `in_flight_prefills` / `rate_bucket` / `new_tokens` / `budget_s`.
- **Correlation** is resolved in the order `own_request` → `session_preceding`
  → `mismatched` → `none` (see *Correlation confidence* below).
- **Blocker class** is reconstructed from lease evidence in the 120 s before the
  enqueue: `dispatch_first_byte_ms` ⇒ `lease_holder_generating`,
  `lease_extended_during_prefill` ⇒ `prefill_in_flight`, `lease_released` ⇒
  cleared; the newest live session wins. `blocker_is_self` flags a wait blocked
  by its own session.
- **Band** re-derives the `LP-0MVV29WR0009DY1A` own-request band, and
  `legacy_band` reproduces the **pre-fix** rule so the phantom-band question can
  be answered per wait.

### Correlation confidence (AC1 honesty)

Both diagnosed windows predate the own-request pricing deploy
(`LP-0MVV29WR0009DY1A`, `6cf5b88c8`, 2026-10-31), so **neither carries a
same-request `new_tokens` on its queue lines**: `own_request` is **0 rows** in
both artefacts and every join is the documented fallback
`session_preceding`, **flagged `ambiguous: true`** (169 of the 169 joined
abandoned rows). The join is never silent. Within that population, **28 of the
169** had exactly one outstanding admission candidate, so for those the join was
forced by the event order and not by the tie-break.

The `own_request` path is exercised end-to-end on the post-deploy log shape —
`--since "2026-11-03 00:00:00" --until "2026-11-04 00:00:00"` yields **4 queue
waits, 4 correlated `own_request`, 0 ambiguous** — but that window runs a
different queue configuration (`max_wait=30.0s`, `max_depth=2`,
`trigger=generating`), so its shares are **not** comparable and are not quoted
as one.

## (a) How many abandoned waits had a same-request admit below budget

Post-change window, 314 abandoned waits:

| Correlation | Abandoned rows | Share |
|---|---|---|
| `session_preceding` (a preceding admission exists; flagged ambiguous) | **169** | 53.8 % |
| `none` (no admission line in the session at all) | **145** | 46.2 % |
| `mismatched` (only another request's admission exists — never joined) | 0 | — |
| `own_request` | 0 | — (log shape predates the field) |

Of the 169 with a preceding admission: **163 `admit`, 6 `reject`**, and **all
163 admits were projected below the 60 s budget** (`admits_below_budget: 163`).
The distribution of that projection against the 60 s the wait actually spent:

| Projected wait | Rows |
|---|---|
| `<20 s` | **75** |
| `20-30 s` | 13 |
| `30-40 s` | 28 |
| `40-50 s` | 37 |
| `50-60 s` | 10 |
| `>=60 s` | 6 |

`median projected_wait_s` **28.4** · `median observed` **60.0** ·
`median(observed − projected)` **+31.6 s** · `projected_lt_observed` **163/163**.

The contrast that isolates *where* the error is: **dispatched** waits in the
same window (162 with a guard decision) have `median projected` **21.95 s**
against `median observed` **28.33 s** — a median gap of **−1.35 s**, i.e. the
guard is *very slightly conservative* for the waits it can serve. The newcomer's
own prefill term is therefore calibrated; the failure is concentrated in the
population that never gets served inside the budget.

**The 145 unpriced abandonments.** These rows have a session but no
`local_admission_guard` decision anywhere before the wait. 99 of them carry
`trigger=local_lease_active` and 46 carry the pre-change `unknown` trigger; 107
were blocked by a prefilling holder, 23 unknown, 15 by a generating holder. A
worked session (`audit-MS-0MVQO6N34007DHP8-parent-504459a5`, 291 log lines in
the retained archives) shows the loop directly — `local_dispatch_denied` →
`contention_queue_fallback_after_queue ... queued_duration=60.00s
reason=local_lease_active` → `local_dispatch_denied` … with **zero**
`local_admission_guard` lines in the whole session (checked across every
retained archive for that session, not only the window). The
`local_lease_active` branch of the dispatcher calls the queue primitive
(`_maybe_queue_for_local_slot`) **without** the admission guard, so this
population is admitted to the queue by construction, not by a priced decision.

## (b) What held the slot — blocker class per trigger

Abandoned waits, post-change window:

| Blocker class | Abandoned | Share of abandoned | `local_lease_active` | `unknown` trigger |
|---|---|---|---|---|
| `prefill_in_flight` | **205** | 65.3 % | 154 | 51 |
| `lease_holder_generating` | **42** | 13.4 % | 8 | 34 |
| `unknown` | 67 | 21.3 % | 63 | 4 |
| **Total** | **314** | | **225** | **89** |

The guard's own occupancy observation corroborates the class it can corroborate:
of the 280 `prefill_in_flight` rows (all outcomes), **135 carry
`in_flight_prefills=1`** on the admission line; of the 102
`lease_holder_generating` rows, **40 carry `in_flight_prefills=0`** (nothing was
prefilling, so the holder could only be decoding). The two are not the same
quantity — `in_flight_prefills` is process-wide while the blocker class names
the slot holder — so the full contingency table is in the artefact
(`blocker.by_class_and_in_flight_prefills`).

**Lookback sensitivity** (the classification's one tunable): at 60 s the split
is `prefill 156 / generating 40 / unknown 118`; at 120 s (the committed default,
equal to `local_prefill_deadline_seconds`) `205 / 42 / 67`; at 300 s
`229 / 47 / 38`; at 600 s `229 / 50 / 35`. `prefill_in_flight` dominates at every
setting and `lease_holder_generating` is stable at 11–16 % — the **verdict does
not depend on the bound**, only the size of the `unknown` bucket does.

`blocker_is_self` is **49 of 517** rows: the waiting session was itself the most
recent lease holder. Those are not a separate mechanism — they are the
lease-active retry loop waiting on its own previous turn.

## (c) Where the phantom `32k-64k` waits came from

**Attribution artefact, proven per wait.** Reproducing the pre-fix banding rule
(the session's latest admission at or before the wait, else the session's latest
admission *even when it is later*) over this window gives:

| Legacy (pre-fix) band | Queued | Abandoned |
|---|---|---|
| `<8k` | 301 | 149 |
| `8k-20k` | 48 | 31 |
| `20k-32k` | 33 | 26 |
| **`32k-64k`** | **45** | **41** |
| `>=64k` | 1 | 0 |
| `unknown` | 89 | 67 |

`32k-64k`: **45 queued / 41 abandoned** — exactly the phantom row the
[post-change record](prefill-admission-conversion-postchange.md) reports (45
queued waits, 41 abandoned, 0 admits in that band). Under the own-request rule
**98 waits (78 of them abandoned) move band**, and the corrected distribution
is:

| Band (own-request rule) | Queued | Dispatched | Abandoned |
|---|---|---|---|
| `<8k` | 260 | 140 | 120 |
| `8k-20k` | 45 | 15 | 30 |
| `20k-32k` | 26 | 7 | 19 |
| **`32k-64k`** | **0** | 0 | 0 |
| **`>=64k`** | **0** | 0 | 0 |
| `unknown` | 186 | 41 | 145 |

No wait remains in `32k-64k`, so the band's 41 abandoned waits were **all**
borrowed from a later admission, not genuinely 32k–64k prompts.

**Mechanism, from a worked session.** Session
`herdr-1793285755-177798-1619` is a serial retry loop (excerpt **abridged** —
the `provider=`/`preference=` fields and the leading
`2026-10-29 <time> - llama-proxy - INFO - ` prefix are elided for width; the
fields quoted are verbatim):

```
14:56:57,855 - contention_queue_fallback_after_queue ... queued_duration=60.00s reason=local_lease_active
14:57:01,369 - local_admission_guard decision=admit ... projected_wait_s=39.3 ... new_tokens=19657 session=herdr-1793285755-177798-1619
14:58:01,370 - contention_queue_fallback_after_queue ... queued_duration=60.00s reason=local_lease_active
14:58:05,652 - local_admission_guard decision=admit ... projected_wait_s=40.1 ... new_tokens=20062 ...
```

The wait abandoned at `14:56:57` enqueued at `14:55:57`, and the session's first
admission line is **3 s after the abandonment** (`14:57:01`) — so the pre-fix
rule banded that wait by an admission belonging to the *next* request. The
own-request rule bands it from its own enqueue pricing (or `unknown` when the log
shape has none), and 186 waits now report `unknown` rather than borrowing.

Banding does not change the totals, so the 60.7 % share is unaffected — only the
per-band split was wrong, and it is now resolved rather than merely re-caveated.

## (d) The diurnally matched +8.1-point gap, split

The like-for-like move is **52.6 % → 60.7 %** (+8.1 points) against the
[diurnally matched baseline](prefill-admission-conversion-diurnal-baseline.md).

**By trigger** (share = abandoned / outcomes, measurement-time full-window
figures, from the committed records):

| Trigger | Pre-change | Post-change | Δ points |
|---|---|---|---|
| `local_lease_active` | 53.3 % (57/107) | 61.1 % (225/368) | **+7.8** |
| `concurrency_limit` | 50.0 % (14/28) | 59.7 % (89/149) | **+9.7** |

Both entry paths rose by a similar amount, so the gap is **not** carried by one
trigger.

**By blocker class** (share of *abandoned* waits; pre-change from the surviving
slice, so a shape comparison only):

| Blocker class | Pre-change (36 abandoned) | Post-change (314 abandoned) | Δ points |
|---|---|---|---|
| `prefill_in_flight` | 55.6 % (20) | 65.3 % (205) | +9.7 |
| `lease_holder_generating` | 11.1 % (4) | 13.4 % (42) | +2.3 |
| `unknown` | 33.3 % (12) | 21.3 % (67) | −12.0 |

The blocker **mix is broadly stable**: prefill-in-flight dominates in both
windows and the generating share barely moves. What changed materially is
**coverage**, not blocker class:

| Measure | Pre-change (surviving slice) | Post-change |
|---|---|---|
| Abandoned waits with a preceding admission | 86.1 % (31/36) | **53.8 % (169/314)** |
| Abandoned waits with **no** admission in the session | 13.9 % (5/36) | **46.2 % (145/314)** |
| Occupancy term = symmetric floor | 30/31 | 163/169 |
| Median projected (abandoned) | 28.2 s | 28.4 s |
| Median observed (abandoned) | 60.0 s | 60.0 s |
| Median projected (dispatched) | 27.3 s | 21.95 s |
| Median observed (dispatched) | 23.32 s | 28.33 s |

The projection the guard makes for an abandoned wait is **the same before and
after** (28.2 s → 28.4 s) and the occupancy term is the symmetric floor in
~96 % of both populations. So the +8.1-point rise is **not** a new mechanism and
not a change in the guard's pricing: the same unpriced-shape population grew
**3.8×** (136 → 517 queued waits) and the fraction of it that bypasses the guard
entirely grew **13.9 % → 46.2 %**.

## Verdict

**A modelling gap compounded by a coverage gap, amplified by traffic volume —
not genuine ETA optimism.** Stated against the three candidates:

1. **Genuine ETA optimism — rejected for the newcomer's own term.** For waits
   that are served, the projection is accurate to within ~1–5 s and slightly
   conservative (dispatched: 21.95 s projected vs 28.33 s observed post-change,
   a −1.35 s median gap; 27.3 s vs 23.32 s pre-change, −2.82 s). The guard's
   prefill estimator is not the defect.
2. **Modelling gap — confirmed, in the occupancy term.** The occupancy term is
   `max(eta_newcomer × busy_slots, occupant_remaining_service)`, and the second
   term is inert: **163 of 169** abandoned admits price exactly
   `eta_prefill_s × busy_slots` (`occupancy_is_symmetric_floor: 163`,
   `occupancy_raised_by_measurement: 6`). So a blocker that holds the slot for
   60 s is priced as if it were prefilling **the newcomer's own prompt**
   (median `eta_prefill_s` 12.3 s), and the projection lands at 28.4 s against a
   60 s wait. The blocker is mostly **prefilling** (65.3 %), so this is the
   *prefill* variant of the gap, not the decode variant the parent hypothesised;
   `lease_holder_generating` is a real but minority 13.4 %.
3. **Coverage gap — confirmed, and it is what moved.** 46.2 % of post-change
   abandonments (145/314) have **no admission decision in their session**,
   because the `local_lease_active` re-queue path queues without consulting the
   guard. Up from 13.9 % pre-change. Those waits are unpriced by construction.
4. **Traffic mix — the amplifier, not the cause.** Both triggers rose by a
   similar amount (+7.8 / +9.7 points) and the blocker mix barely moved, so
   volume did not introduce a new shape; it multiplied an unchanged one and
   shifted its composition towards the unpriced path.

## Candidate behaviour fixes (at most two)

1. **Price the occupant's remaining service so the occupancy term binds**
   (`LP-0MVV29YDQ00437N6`). Expected effect, from these numbers: the **163
   floor-priced abandoned admits** stop being priced as one newcomer-prefill and
   get the holder's measured remaining service instead; those whose measured
   remaining service exceeds the 60 s budget are refused at admission instead of
   expiring in the queue. Rollback is the existing knobs only — the symmetric
   `eta × busy_slots` floor must stay as the lower bound, fail-open must be
   preserved, and `contention_queue_max_wait_seconds` (60) /
   `local_prefill_deadline_seconds` (120) must not move.
2. **Route the `local_lease_active` re-queue through a priced decision**
   (`LP-0MVV29Y1M004GAW0`, already owned — cross-referenced, **not** duplicated
   here). Expected effect: the **145 unpriced abandonments (46.2 %)** acquire a
   projection and a priority, and the "one queue budget per turn" rule stops a
   retry loop from being handed a fresh 60 s per turn. This record supplies the
   population and the per-wait evidence; the fix decision belongs to that item.

Lease-fairness capping remains owned by `LP-0MVOJ6KAW004EBV0`; the
`lease_holder_generating` minority (13.4 %) is evidence for that item, not a
separate fix proposed here.

## What this record does and does not claim

- **Claims:** the per-wait numbers above for the two windows named, from the
  committed JSON artefacts; that the guard's projection for abandoned waits is
  below budget in 163/163 admit cases with a median 28.4 s against 60 s
  observed; that the occupancy term is the symmetric floor in 163/169; that the
  blocker is prefilling in 65.3 % of abandonments; that the phantom `32k-64k`
  row is an attribution artefact reproduced exactly (45 queued / 41 abandoned)
  and resolved to 0 by the own-request rule; that 46.2 % of abandonments bypass
  the guard entirely, up from 13.9 %.
- **Does not claim:** any `own_request` correlation in these two windows (the
  log shape predates it — every join is the flagged fallback); a full-window
  pre-change measurement (retention pruned 16 h of it); any post-fix effect (the
  fixes are proposed, not applied); any change to the parent's literal 85 % bar
  or to the diurnally matched 52.6 % baseline.
- **Known limitations.** The blocker class is a bounded-lookback reconstruction
  (120 s default), so 21.3 % of abandonments are `unknown` — mostly waits whose
  holder left no lease marker inside the bound; the sensitivity table shows the
  verdict is stable across 60–600 s. `observed_wait_s` is **censored** at the
  60 s budget by definition, so "projected vs observed" measures *how far below
  the budget the projection was*, not the true service time.

## Reproducing

```bash
# Post-change window (fully covered; the primary artefact).
scripts/queue_wait_diagnosis.py \
  --since "2026-10-29 00:10:18" --until "2026-10-30 00:10:18" \
  --log-dir /var/log/llama-proxy --json \
  > docs/dev/queue-wait-abandonment-diagnosis.json

# Diurnally matched pre-change window (surviving slice only).
scripts/queue_wait_diagnosis.py \
  --since "2026-10-28 00:10:18" --until "2026-10-29 00:10:18" \
  --log-dir /var/log/llama-proxy --json \
  > docs/dev/queue-wait-abandonment-diagnosis-diurnal-baseline.json

# Blocker-lookback sensitivity (quoted in "(b)").
scripts/queue_wait_diagnosis.py \
  --since "2026-10-29 00:10:18" --until "2026-10-30 00:10:18" \
  --blocker-lookback 300 --log-dir /var/log/llama-proxy --json
```

The commands regenerate the committed JSON **modulo the rotation-dependent
`log_files_scanned` inventory**, and only **while the archives named in
`coverage.files` survive retention** (7 days). Once they are pruned the committed
files are the evidence, and their `coverage` block records exactly what they were
measured over. `tests/test_queue_wait_diagnosis_artefacts.py` pins the window
boundaries, the totals decomposition and the numbers this page quotes.

## Cross-reference

| # | Work item | Scope |
|---|---|---|
| 1 | `LP-0MVV29XPT003OXI5` | This per-wait diagnosis |
| 2 | `LP-0MVV1V213002QAIR` | Parent: diurnal restatement, mix/code separation, working-baseline regression |
| 3 | `LP-0MVV29XDW006VTWK` | Diurnally matched pre-change baseline (52.6 %) |
| 4 | `LP-0MVTKTOVI006ZONN` | Post-change 24 h record (60.7 %) whose phantom band this resolves |
| 5 | `LP-0MVV29WR0009DY1A` | Own-request queue-wait banding (the fix that resolves the phantom band) |
| 6 | `LP-0MVV29X2E004INFL` | `trigger=` / `cause=` attribution |
| 7 | `LP-0MVV29YDQ00437N6` | Fix owner: price the diagnosed blocker class in the guard (candidate fix 1) |
| 8 | `LP-0MVV29Y1M004GAW0` | Fix owner: `local_lease_active` re-queue path (candidate fix 2) |
| 9 | `LP-0MVOJ6KAW004EBV0` | Lease-fairness capping owner (evidence only, not duplicated) |
| 10 | `LP-0MVS9YK9P006M1NQ` | The occupant-remaining re-pricing whose measured term this shows to be inert |
| 11 | `LP-0MW3GU0JO008XOUF` | Log retention (7 days) shorter than the records that quote measured windows |

Related: [`docs/dev/contended-prefill-pricing.md`](contended-prefill-pricing.md),
[`docs/dev/prefill-admission-conversion-postchange.md`](prefill-admission-conversion-postchange.md),
[`docs/dev/prefill-admission-conversion-diurnal-baseline.md`](prefill-admission-conversion-diurnal-baseline.md).
