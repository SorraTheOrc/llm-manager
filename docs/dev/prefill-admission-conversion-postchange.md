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
| `local_eviction_after_acceptance` | 463 | 19.3 | 392 | 16.3 | ×0.85 |

The pre-change window is 19:00→19:00 and the post-change window is
00:10→00:10, and post-change traffic admitted ~3× as many local dispatches per
hour while rejecting 31 % fewer, so the queued-wait denominator grew ~8×.
Both windows are exactly 24 h, so the per-hour columns are like-for-like in
duration, but the **mix** is not identical and must be kept in mind when
reading the share.

## AC4 second clause — `local_eviction_after_acceptance`

Counted by the committed harness over the same window (`evictions` block of
the JSON):

| Reason | Post-change (24 h) | Post / h | Explanation |
|---|---|---|---|
| `measured_cache_deficit` | 234 | 9.75 | Measured cache came back materially below `estimated_tokens`, so the dispatch was re-priced/evicted rather than dispatched on a bad estimate. Pre-existing class (same reason dominates the pre-change window). |
| `no_progress_watchdog` | 82 | 3.42 | Accepted local dispatch made no progress inside the watchdog window (queue/stall), so the lease was released. Pre-existing class. |
| `prefill_deadline` | 68 | 2.83 | The 120 s `local_prefill_deadline_seconds` bound was hit after acceptance. Pre-existing class. |
| `lease_evicted` | 7 | 0.29 | The slot lease was taken by another dispatch. Pre-existing class. |
| `idle_timeout` | 1 | 0.04 | Lease expired while idle. Pre-existing class. |
| **Total** | **392** | **16.33** | Compare pre-change **463** / 19.3 h → **no new class and a lower rate**. |

Every reason class is non-zero but all five are **pre-existing** classes also
present in the pre-change era, and the absolute rate is 15 % **lower** than
pre-change. No new `local_eviction_after_acceptance` class appeared, so AC4's
second clause holds and no triage work item is triggered by it.

## Verdict against both bars

| Bar | Threshold | Post-change result | Verdict |
|---|---|---|---|
| Literal parent AC4 (2026-10-19/20 reference) | survived share **< 85 %** | **60.7 %** (314/517) | **met** |
| Stricter working baseline (pre-change code, same harness) | must not regress above **36.9 %** | **60.7 %** (> 36.9 %) | **not met — regression recorded** |
| AC4 second clause | no new `local_eviction_after_acceptance` class | 5 pre-existing classes, 392 events (16.3/h vs 19.3/h) | **met** |

- **Parent AC4 (LP-0MVS8T6F4006DGG3) is satisfied by evidence**: the literal
  85 % bar it names is cleared by 24 points with a committed, reproducible
  record.
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
  accepted. It is not a release blocker for the `dev`→`main` promotion (the
  literal AC4 bar and the no-new-eviction clause both hold).

## Reproducing

1. Run the command at the top of this page (any window; `--log-dir` defaults
   to `/var/log/llama-proxy`).
2. Confirm the JSON's `since`/`until` match the requested bounds and that
   `log_files` span the window.
3. The harness is unit-tested from synthetic log lines
   ([`tests/test_prefill_admission_conversion.py`](../../tests/test_prefill_admission_conversion.py)),
   including pre-change line shapes without `rate_bucket` /
   `contended_samples`, the admission→first-byte join, and the
   `local_eviction_after_acceptance` reason counters.

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

Related: [`docs/dev/contended-prefill-pricing.md`](contended-prefill-pricing.md)
(the contended pricing and rollback page) and
[`docs/dev/prefill-admission-conversion.md`](prefill-admission-conversion.md)
(the pre-change baseline).
