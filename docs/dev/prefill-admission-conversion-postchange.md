# Post-change admission→first-byte conversion (24 h) — AC4 record

This page is the **post-change 24 h record for parent AC4** of
**LP-0MVS8T6F4006DGG3** (_Condition prefill-ETA calibration on co-residency;
fix occupant-remaining pricing_), produced by the standalone carrier
**LP-0MVTKTOVI006ZONN** after the AC4 measurement was delegated out of the
epic by operator decision (2026-10-29).

It answers parent AC4 directly: _"Post-change, the share of queued local waits
ending in `contention_queue_fallback_after_queue` drops below the 85 % baseline
measured 2026-10-19/20 over a 24 h window, with no new
`local_eviction_after_acceptance` events."_

- **Deployed code under measurement:** `dev` HEAD **`52085f09`**, serving from
  the proxy restarted **2026-10-29 00:10:18 UTC** (pid 1081749, `/health`
  healthy). A second restart at **2026-10-29 12:58:34 UTC** (pid 2378855) did
  not change the served code, so the window remains a single-code-path window.
- **Measured window (fixed):** `2026-10-29 00:10:18` → `2026-10-30 00:10:18`
  UTC (24 h, inclusive) — exactly 24 h after the fix was deployed.
- **Pre-change reference:** [`prefill-admission-conversion.md`](prefill-admission-conversion.md)
  (window `2026-10-27 19:00:00` → `2026-10-28 19:00:00`, 24 h, pre-fix code).

The numbers are produced entirely by
[`scripts/prefill_admission_conversion.py`](../../scripts/prefill_admission_conversion.py)
(a pure log analyser — no proxy behaviour change, no running proxy required):

```bash
scripts/prefill_admission_conversion.py \
  --since "2026-10-29 00:10:18" \
  --until "2026-10-30 00:10:18" \
  --log-dir /var/log/llama-proxy \
  --json > docs/dev/prefill-admission-conversion-postchange.json
```

Raw output: [`prefill-admission-conversion-postchange.json`](prefill-admission-conversion-postchange.json).
Re-running the command above regenerates the committed JSON byte-for-byte
(modulo the rotation-dependent `log_files_scanned` inventory), so no number on
this page is hand-typed: every table below is read off that JSON. The harness
also emits the `local_eviction_after_acceptance` counts (total, by `reason`,
and per band), so AC4's second clause needs no separate log scan.

## Window and inputs

- **Window:** `2026-10-29 00:10:18` → `2026-10-30 00:10:18` (24 h, inclusive).
- **Log files read (7):** `proxy.log.2026-10-29_00.gz`,
  `proxy.log.2026-10-29_06.gz`, `proxy.log.2026-10-29_125828`,
  `proxy.log.2026-10-29_14`, `proxy.log.2026-10-29_140921`,
  `proxy.log.2026-10-29_20`, `proxy.log-2026-10-30_00`.
- **First byte with no admission line:** 16 sessions (reported separately,
  never silently joined to an admission — harness AC2).
- **`unknown` band:** 89 queued waits whose session had no bandable admission
  evidence at or before the queue event (no `new_tokens` on the admit line).
  It carries no admission verdicts, so it does not distort the conversion or
  verdict counts — but its abandoned share (75.3 %) is reported rather than
  hidden.

Bands are on the **priced new (uncached) tokens** of a local dispatch:
`<8k`, `8k-20k`, `20k-32k`, `32k-64k`, `>=64k` (the `20k`/`32k` boundaries
mirror `local_admission_guard_eta_percentile_min_tokens` and
`CONSERVATIVE_MIN_TOKENS`).

## Sanity gate (window validity) — PASSES

The window is only valid if the running proxy actually emits the co-residency
fields; an all-`unknown` bucket would mean the old code was still serving and
the window would be **void**. Per-band `rate_bucket` distribution over the
window (from the committed JSON):

| Band | `contended` | `solitary` | `solitary_fallback` |
|---|---|---|---|
| `<8k` | 423 | 1 094 | 11 |
| `8k-20k` | 259 | 22 | 37 |
| `20k-32k` | 193 | 390 | 74 |
| `32k-64k` | 0 | 855 | 0 |
| `>=64k` | 0 | 662 | 0 |

`contended` admissions exist in the three low bands (875 in total) and
`solitary_fallback` (122) records contended admissions priced from the
solitary window while the contended sample was still short. The window is
therefore **calibrated and serving the fix**, not void. The absence of
`contended` in `32k-64k` / `>=64k` is expected: every dispatch in those bands
was rejected by the flat cold-prefill budget before any pricing bucket could
matter (`prefill_budget_exceeded` 855 / 662).

## Per-band results, pre vs post

### Pre-change (2026-10-27 19:00 → 2026-10-28 19:00, 24 h, pre-fix code)

| Band | Admits | Rejects | Converted | Unconverted | Conv % | Queued waits | Abandoned | Abandoned % | Queue-wait s |
|---|---|---|---|---|---|---|---|---|---|
| `<8k` | 397 | 36 | 27 | 15 | 64.3 % | 57 | 20 | 35.1 % | 2236.8 |
| `8k-20k` | 24 | 152 | 10 | 5 | 66.7 % | 8 | 4 | 50.0 % | 408.9 |
| `20k-32k` | 2 | 692 | 1 | 1 | 50.0 % | 0 | 0 | 0.0 % | 0.0 |
| `32k-64k` | 0 | 1091 | 0 | 0 | 0.0 % | 0 | 0 | 0.0 % | 0.0 |
| `>=64k` | 0 | 1983 | 0 | 0 | 0.0 % | 0 | 0 | 0.0 % | 0.0 |
| **Total** | **423** | **3954** | **38** | **21** | **64.4 %** | **65** | **24** | **36.9 %** | **2645.7** |

### Post-change (2026-10-29 00:10:18 → 2026-10-30 00:10:18, 24 h, `52085f09`)

| Band | Admits | Rejects | Converted | Unconverted | Conv % | Queued waits | Abandoned | Abandoned % | Queue-wait s |
|---|---|---|---|---|---|---|---|---|---|
| `<8k` | 1132 | 396 | 99 | 62 | 61.5 % | 301 | 149 | 49.5 % | 12833.5 |
| `8k-20k` | 97 | 221 | 26 | 18 | 59.1 % | 48 | 31 | 64.6 % | 2299.3 |
| `20k-32k` | 49 | 608 | 5 | 11 | 31.2 % | 33 | 26 | 78.8 % | 1808.3 |
| `32k-64k` | 0 | 855 | 0 | 0 | 0.0 % | 45 | 41 | 91.1 % | 2527.7 |
| `>=64k` | 0 | 662 | 0 | 0 | 0.0 % | 1 | 0 | 0.0 % | 51.6 |
| `unknown` | 0 | 0 | 0 | 0 | 0.0 % | 89 | 67 | 75.3 % | 4800.3 |
| **Total** | **1278** | **2742** | **130** | **91** | **58.8 %** | **517** | **314** | **60.7 %** | **24320.7** |

### Abandoned-queue-wait share, side by side

| Band | Pre-change abandoned % (24 h) | Post-change abandoned % (24 h) | Pre queued | Post queued |
|---|---|---|---|---|
| `<8k` | 35.1 % | 49.5 % | 57 | 301 |
| `8k-20k` | 50.0 % | 64.6 % | 8 | 48 |
| `20k-32k` | 0.0 % | 78.8 % | 0 | 33 |
| `32k-64k` | 0.0 % | 91.1 % | 0 | 45 |
| `>=64k` | 0.0 % | 0.0 % | 0 | 1 |
| `unknown` | — | 75.3 % | — | 89 |
| **Total** | **36.9 %** (24/65) | **60.7 %** (314/517) | **65** | **517** |

### Per-hour rates (the fairer comparison — the two windows differ in diurnal mix)

| Metric | Pre-change / 24 h | Pre / h | Post-change / 24 h | Post / h | Change / h |
|---|---|---|---|---|---|
| Local admits | 423 | 17.6 | 1278 | 53.3 | **×3.0** |
| Local rejects | 3954 | 164.8 | 2742 | 114.3 | ×0.69 |
| Queued local waits | 65 | 2.7 | 517 | 21.5 | **×8.0** |
| Abandoned queue waits | 24 | 1.0 | 314 | 13.1 | **×13.1** |
| `local_eviction_after_acceptance` | 152 | 6.3 | 392 | 16.3 | **×2.6** |

The pre-change window is 19:00→19:00 and the post-change window is
00:10→00:10, and post-change traffic admitted ~3× as many local dispatches per
hour while rejecting 31 % fewer, so the queued-wait denominator grew ~8×.
Both windows are exactly 24 h, so the per-hour columns are like-for-like in
duration, but the **mix** is not identical and must be kept in mind when
reading the share.

> **Comparator correction (2026-10-30, `LP-0MVV9JWL9003A6XR`).** An earlier
> revision of this page recorded the pre-change eviction comparator as
> **463 / 19.3 per h** and concluded the eviction rate was **×0.85 (15 %
> lower)**. That 463 was a substring artefact: 311 of the 463 hits were
> `[remote] POST … body={…}` request lines whose agent prompt merely
> *mentioned* `local_eviction_after_acceptance` (the pre-change window is
> exactly when `LP-0MVQIT9QY007QU27` was being worked). Re-counted with the
> harness's message-anchored classifier the genuine pre-change total is
> **152 (6.3 / h)**, so the eviction rate rose **×2.6**, it did not fall. The
> diurnal-mix caveat still applies to the *share* comparison; the rate change
> is large enough to survive it.

## AC4 second clause — `local_eviction_after_acceptance`

Counted by the committed harness over each window (`evictions` block of the
JSONs). The classifier only counts a line whose logging *message* starts with
`local_eviction_after_acceptance`, so a proxied request body that merely quotes
the string can never be counted (`LP-0MVV9JWL9003A6XR`). The pre-change column
comes from [`prefill-admission-conversion.json`](prefill-admission-conversion.json),
regenerated with the same classifier over its own 24 h window:

| Reason | Pre-change (24 h) | Pre / h | Post-change (24 h) | Post / h | Class |
|---|---|---|---|---|---|
| `measured_cache_deficit` | **0** | 0.00 | **234** | 9.75 | **NEW** |
| `prefill_deadline` | 145 | 6.04 | 68 | 2.83 | pre-existing |
| `no_progress_watchdog` | 2 | 0.08 | 82 | 3.42 | pre-existing |
| `lease_evicted` | 1 | 0.04 | 7 | 0.29 | pre-existing |
| `idle_timeout` | 3 | 0.13 | 1 | 0.04 | pre-existing |
| `orphan_cleanup` | 1 | 0.04 | 0 | 0.00 | pre-existing, absent post-change |
| **Total** | **152** | **6.33** | **392** | **16.33** | **×2.6 higher** |

### `measured_cache_deficit` is a new class, not a pre-existing one

- **Pre-change window: 0.** Its first appearance anywhere in the retained logs
  is **2026-10-29 00:13:30 UTC** — 3 minutes after the deployment window
  opened.
- **Post-change window: 234** — 60 % of all 392 evictions, the dominant class.
- Emitted by `proxy/proxy/measured_cache_abort.py`
  (`MEASURED_CACHE_DEFICIT_REASON`), landed in `87f238a2` (2026-10-27)
  and `1908f431` (2026-10-28) — both ancestors of the deployed
  `52085f09`, so it is part of the code under measurement.
- The explanation is the feature's: `llama.cpp`'s first `prompt_progress`
  event reports the ground-truth reused prefix, and when the measured deficit
  `(total - cache) / total` clears the ratio floor *and* the measured
  new-token work exceeds the admission-priced `new_tokens` by more than the
  token bound, the dispatch is released early (rather than waiting out the
  deadline) with `reason=measured_cache_deficit`. It is **not** a restatement
  of the 120 s deadline or of a warm-predicate miss at admission time.
- **Triage item: `LP-0MVV9JWL9003A6XR`** (eviction triage for this class and
  for the corrected pre-change comparator).

### Verdict on the new class (AC3): expected behaviour of the feature — not a defect

The class fires exactly where its specification says it should, so no blocking
child item is created:

- **Inside the documented envelope for all 234 events.** The measured deficit
  ratio never fell below the `min_deficit_ratio` floor (min 0.2521 against a
  0.25 floor, median 0.617), and the measured new-token work always exceeded
  the priced value by more than `bound_tokens` (min 2 074 against a 2 000
  bound, median 14 402). No event fired on a genuinely warm measurement.
- **It is hitting the population it was built for.** 55 of 234 (23.5 %) measured
  **zero** cached tokens (`measured_cache=0`) — dispatches whose admission
  price implied near-total reuse (`new_tokens` ≪ `estimated_tokens`) but which
  llama-server re-prefilled from scratch, the documented "fake-warm" case.
  The events span 73 distinct sessions, 107 of them released at the very first
  measurement (`elapsed_s ≤ 1`).
- **They are early releases, not deadline rescues.** All 234 carried a
  predicted prefill ETA below the 120 s `local_prefill_deadline_seconds` bound
  (median 17 s), so each is the intended trade: abandon wasted prefill now
  rather than discover the mis-pricing at the deadline.
- **Nuance recorded rather than hidden:** for 4 of 234 nothing had been priced
  as reused (`estimated_tokens == new_tokens`), so those fired on a plain
  under-estimate of the total work. That is still within documented condition
  4 ("measured work exceeds the priced compute by more than the bound") but it
  is the mode to watch if the bound is ever tightened.
- The high firing rate (9.75 /h, 60 % of evictions) is a **monitoring signal**
  about how much traffic is admitted on a warm prediction the server
  contradicts — not a defect in the abort path.

### AC4 second clause verdict: not met on the literal reading

Parent AC4's clause is *"with no new `local_eviction_after_acceptance`
events"*. That is **not met**: 392 ≠ 0 (up from a genuine pre-change 152) and
one of the five post-change classes, `measured_cache_deficit`, is genuinely
new (0 → 234). The new class is triaged in `LP-0MVV9JWL9003A6XR` with the
verdict above (expected behaviour of the measured-cache early-abort feature),
and the one class that disappeared (`orphan_cleanup`) is recorded here too. The
earlier "all five classes pre-existing, lower rate" conclusion on this page was
wrong on both counts and is superseded by this section.

## Verdict against both bars

| Bar | Threshold | Post-change result | Verdict |
|---|---|---|---|
| Literal parent AC4 (2026-10-19/20 reference) | survived share **< 85 %** | **60.7 %** (314/517) | **met** |
| Stricter working baseline (pre-change code, same harness) | must not regress above **36.9 %** | **60.7 %** (> 36.9 %) | **not met — regression recorded** |
| AC4 second clause | no new `local_eviction_after_acceptance` class | 392 events (16.3/h vs a genuine 6.3/h); one new class `measured_cache_deficit` (0 → 234) | **not met — triaged in `LP-0MVV9JWL9003A6XR`** |

- **Parent AC4's headline share bar is satisfied by evidence**: the literal
  85 % bar it names is cleared by 24 points with a committed, reproducible
  record. (The clause's second bar — no new eviction events — is a separate
  row above and is **not** met; it is triaged rather than waived.)
- **The stricter working bar is *not* cleared.** Against the pre-change code
  measured with the same harness on the same 24 h basis, the abandoned share
  rose from 36.9 % to 60.7 % and the abandoned-wait rate rose from 1.0 /h to
  13.1 /h. That is a real regression against the working baseline, and it is
  **not** explained away by the diurnal-mix caveat alone: admits/hour tripled
  and rejects/hour fell, so more traffic reached the queue, but the abandoned
  *share* of that queue also grew in four of five bands.
- The regression is **tracked separately** as **`LP-0MVV1V213002QAIR`**
  ("Contended-pricing: abandoned-queue-wait share regressed to 60.7% vs the
  36.9% working baseline"), discovered from this record, rather than silently
  accepted.
- **Release-blocker status.** The literal AC4 bar (the abandoned-wait share)
  and the working-baseline regression are unchanged by this correction. What
  *is* new is that AC4's second clause no longer holds: the eviction rate rose
  ×2.6 and `measured_cache_deficit` is a genuinely new class. That class is
  triaged as expected behaviour of a shipped feature (`LP-0MVV9JWL9003A6XR`),
  so whether a literally-failing clause blocks the `dev`→`main` promotion is a
  producer decision — the evidence for it is recorded here and on that item.

## Reproducing

1. Run the command at the top of this page (any window; `--log-dir` defaults
   to `/var/log/llama-proxy`).
2. Confirm the JSON's `since`/`until` match the requested bounds and that
   `log_files` span the window.
3. The harness is unit-tested from synthetic log lines
   ([`tests/test_prefill_admission_conversion.py`](../../tests/test_prefill_admission_conversion.py)),
   including pre-change line shapes without `rate_bucket` /
   `contended_samples`, the admission→first-byte join, the
   `local_eviction_after_acceptance` reason counters, and the
   request-body-echo case that inflated the old comparator.
4. Both committed JSONs were regenerated with the message-anchored eviction
   classifier on 2026-10-30 (`LP-0MVV9JWL9003A6XR`); re-running the command at
   the top of this page and the one in
   [`prefill-admission-conversion.md`](prefill-admission-conversion.md)
   reproduces them (modulo the rotation-dependent `log_files_scanned`
   inventory).

## Cross-reference

| # | Work item | Scope |
|---|-----------|-------|
| 1 | `LP-0MVTKTOVI006ZONN` | This post-change 24 h record (AC4 carrier) |
| 2 | `LP-0MVS8T6F4006DGG3` | Parent epic — AC4 evidenced here |
| 3 | `LP-0MVS9Y7FG004QJ2S` | Pre-change baseline harness |
| 4 | `LP-0MVS9Y76R004B5M8` | Contended prefill rate bucket (`rate_bucket` log field) |
| 5 | `LP-0MVS9YK9P006M1NQ` | Occupant-remaining priced from new tokens at the contended rate |
| 6 | `LP-0MVS9YKHL007UBBZ` | Decision-log / usage-report surface for `rate_bucket` |
| 7 | `LP-0MVS9Z0GY008LCJA`, `LP-0MVSSGE21006BSKS` | Superseded time-blocked carriers of this AC |
| 8 | `LP-0MVV1V213002QAIR` | Working-baseline regression follow-up (discovered from this record) |
| 9 | `LP-0MVV9JWL9003A6XR` | Eviction triage: `measured_cache_deficit` new-class verdict + corrected pre-change comparator (152, not 463) |
| 10 | `LP-0MVR7IN0O000DPXN`, `LP-0MVR7HJD0007RFEM` | Measured-cache early-abort feature that introduced the `measured_cache_deficit` class |
| 11 | `LP-0MVQIT9QY007QU27` | `prefill_deadline` eviction work being done during the pre-change window (source of the 311 request-body false positives) |

Related: [`docs/dev/contended-prefill-pricing.md`](contended-prefill-pricing.md)
(the contended pricing and rollback page) and
[`docs/dev/prefill-admission-conversion.md`](prefill-admission-conversion.md)
(the pre-change baseline).
