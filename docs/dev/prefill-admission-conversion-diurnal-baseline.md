# Diurnally matched pre-change baseline (24 h) — abandoned-queue-wait share

This page restates the abandoned-queue-wait baseline for the contended-pricing
fix (`52085f09`) on a **diurnally matched** 24 h window, so the post-change
share is quantified against traffic with the **same daily shape** instead of
against a 19:00→19:00 window with a different mix
(`LP-0MVV29XDW006VTWK`, parent `LP-0MVV1V213002QAIR`).

- **Deploy boundary:** `dev` HEAD **`52085f09`** started serving at
  **2026-10-29 00:10:18 UTC**. Everything before that instant is pre-change
  code.
- **Diurnally matched pre-change window (this record):**
  `2026-10-28 00:10:18` → `2026-10-29 00:10:18` UTC (24 h) — it ends exactly at
  the deploy, so it is **fully pre-change code**, and it is the same
  clock window as the post-change record `2026-10-29 00:10:18` →
  `2026-10-30 00:10:18`.
- **Harness:** the same committed revision as the post-change record —
  [`scripts/prefill_admission_conversion.py`](../../scripts/prefill_admission_conversion.py)
  (a pure log analyser; no proxy behaviour change, no running proxy required).
  No harness code changed for this slice.

> **Coverage notice — read this before quoting a number.** The record was
> measured on **2026-10-30** with the window fully covered. By the time this
> page was committed (**2026-11-05**) log retention (7 days,
> `logging.retention_days` default in `proxy/proxy/utils.py`) had pruned the
> first ~16 h of `2026-10-28`, so **the window can no longer be re-derived in
> full**: the surviving archives cover **6 h 10 m of it (25.7 % of the
> window)**. The committed JSON is therefore today's harness output over the
> **surviving** archives, and the full-window figures are the measurement-time
> harness output recorded on `LP-0MVV1V213002QAIR`. Both are stated separately
> below and are never mixed in one table. See
> [Provenance and coverage](#provenance-and-coverage).

## Why this window

The previous working baseline used `2026-10-27 19:00` → `2026-10-28 19:00`
(a 24 h window that happened to be convenient, not one chosen for comparability).
The post-change window is anchored on the deploy, so it runs `00:10:18`→
`00:10:18`. Comparing them mixes two effects:

- the **diurnal mix** of the traffic (the 19:00→19:00 window is quieter than
  the 00:10→00:10 window), and
- the **code change**.

Anchoring the pre-change window on the same clock times as the post-change
window, and ending it exactly at the deploy, removes the first effect by
construction: both windows then differ only in which code served them.

## Provenance and coverage

Two measurement classes appear on this page. They are labelled on every table
and are never combined into a single figure.

| Class | When | Coverage | Where the numbers live |
|---|---|---|---|
| **Measurement-time full window** | 2026-10-30 | 24 h (complete) | pasted from the harness run recorded in the planning comment on `LP-0MVV1V213002QAIR` |
| **Coverage re-measurement** | 2026-11-05 | 6 h 10 m (25.7 %) | this page's committed JSON, [`prefill-admission-conversion-diurnal-baseline.json`](prefill-admission-conversion-diurnal-baseline.json) |

Surviving archives inside the window, with the intervals each one actually
contributes (read from the files; the harness reports the same two names in
`log_files`):

| Archive | Interval it covers inside the window | Duration |
|---|---|---|
| `proxy.log.2026-10-28_16.gz` | `2026-10-28 16:16:14` → `2026-10-28 22:16:12` | 5 h 59 m 58 s |
| `proxy.log.2026-10-29_001015.gz` | `2026-10-29 00:00:00` → `2026-10-29 00:10:15` | 10 m 15 s |
| **Total surviving coverage** | | **6 h 10 m 13 s (25.7 % of 24 h)** |

Pruned by retention before this page was written: `2026-10-28 00:10:18` →
`16:16:14`, `22:16:12` → `2026-10-29 00:00:00`. The mechanism is the default
`logging.retention_days: 7` (→ `backupCount = 7 × 24 / 6 = 28` six-hour
segments) in `proxy/proxy/utils.py`; the record's window went out of retention
on 2026-11-04, one day before this slice ran. The retention gap for
measurement records is recorded separately (see
[Cross-reference](#cross-reference)).

## Side-by-side: the three windows

Share is `abandoned / queue outcomes` (the harness denominator: `queued_waits`
= dispatched + abandoned), and every window is exactly 24 h so the per-hour
columns are comparable in duration. **Dispatched** for the diurnal pre-change
window is derived (`135 outcomes − 71 abandoned = 64`), because the
measurement-time record reported the outcome split rather than the dispatched
count directly.

| Window (24 h) | Code | Enqueued | Dispatched | Abandoned | **Abandoned share** | Admits | Rejects | Source |
|---|---|---|---|---|---|---|---|---|
| `10-27 19:00` → `10-28 19:00` | pre | 65 | 41 | 24 | **36.9 %** (24/65) | 423 | 3 954 | committed [`prefill-admission-conversion.json`](prefill-admission-conversion.json) — historical 19:00→19:00 reference |
| `10-28 00:10:18` → `10-29 00:10:18` | pre | 136 | 64 (derived) | 71 | **52.6 %** (71/135) | 576 | 4 078 | measurement-time run (2026-10-30), `LP-0MVV1V213002QAIR` |
| `10-29 00:10:18` → `10-30 00:10:18` | `52085f09` | 517 | 203 | 314 | **60.7 %** (314/517) | 1 278 | 2 742 | committed [`prefill-admission-conversion-postchange.json`](prefill-admission-conversion-postchange.json) |

Note on the second row's denominators: the window enqueued 136 waits and
produced 135 outcome lines (64 dispatched + 71 abandoned); the harness share
is computed from the outcome lines, hence `71/135 = 52.6 %`. The third row's
enqueue and outcome counts agree exactly (517/517).

The same three windows at **per-hour rates** — the column pair that makes the
diurnal pair and the code-change step comparable — are in
[Per-hour rates](#per-hour-rates) directly below, and the mix/code-change
decomposition is in [The arithmetic](#the-arithmetic-with-the-mix-contribution-named).

### Per-hour rates

| Metric | `19:00→19:00` pre / h | `00:10→00:10` pre / h | `00:10→00:10` post / h | pre-diurnal → post |
|---|---|---|---|---|
| Local admits | 17.62 | 24.00 | 53.25 | **×2.2** |
| Local rejects | 164.75 | 169.92 | 114.25 | ×0.67 |
| Queued local waits (enqueued) | 2.71 | 5.67 | 21.54 | **×3.8** |
| Abandoned queue waits | 1.00 | 2.96 | 13.08 | **×4.4** |

Reading the per-hour columns: between the two **pre-change** windows
(`19:00→19:00` → `00:10→00:10`, identical code) the abandoned rate already
triples (1.00 → 2.96 /h) and admits/hour rise ×1.36 — that is diurnal mix, not
code. Between the **diurnally matched** pair the abandoned rate rises another
×4.4 and admits/hour another ×2.2, while rejects/hour fall ×0.67: more traffic
reached the queue, and a larger share of that queue was abandoned.

## The arithmetic, with the mix contribution named

```
36.9 %  ──(+15.7 points: diurnal mix, same pre-change code)──▶  52.6 %
52.6 %  ──(+ 8.1 points: the contended-pricing change)──▶       60.7 %
```

- **`36.9 % → 52.6 %` (+15.7 points) is diurnal mix.** Both windows are served
  by pre-change code; only the clock window changes. This step must not be
  attributed to `52085f09`, and the earlier record's implicit comparison of
  `36.9 %` against the post-change figure overstated the code-change effect by
  roughly a factor of three (`15.7` mix + `8.1` code = the full `23.8` point
  move).
- **`52.6 % → 60.7 %` (+8.1 points) is the like-for-like change.** This is the
  diurnally matched comparison: both windows are `00:10:18`→`00:10:18` UTC,
  24 h, differing only in the code that served them.
- **Mix share of the headline: 15.7 / 23.8 = 66 %.** Code-change share:
  `8.1 / 23.8 = 34 %`.

Both windows are kept side by side deliberately: the `36.9 %` figure remains the
**historical** 19:00→19:00 reference (it is what the earlier post-change record
quoted), and `52.6 %` is the **diurnally matched working baseline** used from
here on. The parent epic's literal **85 %** bar is unchanged and unmoved by this
restatement — `60.7 %` clears it by 24 points either way.

## Coverage re-measurement (committed JSON)

The committed JSON is the harness output for the exact window
(`since`/`until` = the deploy-boundary pair) over the archives that survive
today — 6 h 10 m of the window, which carries **67 of the 136 enqueued waits**
(49 % of the window's queued traffic; the surviving slice is the busy
16:16–22:16 evening block, so it is over-represented in traffic per hour).

**Consistency check, which is the point of this re-measurement:** over the
surviving slice the abandoned share is **53.7 % (36/67)**, against **52.6 %
(71/135)** over the full window at measurement time. The truncated slice
reproduces the full-window share to within 1.1 points, so the measurement-time
figure is corroborated rather than contradicted — but it is *not* a
re-measurement of the window, and this page does not present it as one.

Window totals from the committed JSON:

| Metric | Value |
|---|---|
| Enqueued (`contention_queue_wait`) | 67 (all 67 enqueue lines predate the `new_tokens=` enqueue field, so none is self-bandable) |
| Queue outcomes | 67 (31 dispatched + 36 abandoned) |
| **Abandoned share** | **53.7 %** (36/67) |
| Admission verdicts | 226 admit / 1 153 reject |
| Conversion (admit → first byte) | 28 converted / 15 unconverted = 65.1 %; 43 sessions admitted |
| First byte with no admission line | 2 (reported separately, never joined) |
| Queue-wait seconds | 2 905.8 |
| `local_eviction_after_acceptance` | 48, all `reason=prefill_deadline` |
| Queue-wait banding | `session_chain=60`, `unknown=7` (`own_outcome=0`, `own_enqueue=0` — pre-deployment signature: no queue line carries its own pricing) |

### Per band

| Band | Admits | Rejects | Converted | Unconverted | Conv % | Queued waits | Abandoned | Abandoned % | Queue-wait s | Evictions |
|---|---|---|---|---|---|---|---|---|---|---|
| `<8k` | 200 | 10 | 22 | 14 | 61.1 % | 49 | 26 | 53.1 % | 2 123.9 | 34 |
| `8k-20k` | 26 | 120 | 6 | 1 | 85.7 % | 11 | 5 | 45.5 % | 462.2 | 8 |
| `20k-32k` | 0 | 380 | 0 | 0 | 0.0 % | 0 | 0 | 0.0 % | 0.0 | 6 |
| `32k-64k` | 0 | 219 | 0 | 0 | 0.0 % | 0 | 0 | 0.0 % | 0.0 | 0 |
| `>=64k` | 0 | 424 | 0 | 0 | 0.0 % | 0 | 0 | 0.0 % | 0.0 | 0 |
| `unknown` | 0 | 0 | 0 | 0 | 0.0 % | 7 | 5 | 71.4 % | 319.7 | 0 |
| **Total** | **226** | **1 153** | **28** | **15** | **65.1 %** | **67** | **36** | **53.7 %** | **2 905.8** | **48** |

There is **no phantom band**: every band that carries queued waits also carries
admission verdicts, except `unknown` (waits with no own-request evidence, banded
`unknown` by the own-request rule — `LP-0MVV29WR0009DY1A`). `32k-64k` and
`>=64k` carry queued waits only in the post-change window, not here.

### Entry path and abandon cause

| Field | Coverage re-measurement (2026-11-05) | Measurement-time full window (2026-10-30) |
|---|---|---|
| abandoned by cause | `unknown: 36` | every abandonment a budget expiry (`queued_duration ≥ 55 s` for 71/71) |
| enqueued by trigger | `unknown: 67` | — |
| dispatched by trigger | `local_lease_active: 21`, `unknown: 10` | `local_lease_active: 50`, `concurrency_limit: 14` |
| abandoned by trigger | `local_lease_active: 30`, `unknown: 6` | `local_lease_active: 57`, `concurrency_limit: 14` |
| abandoned share by trigger | `local_lease_active: 58.8 %` (30/51), `unknown: 37.5 %` (6/16) | `local_lease_active: 53.3 %` (57/107), `concurrency_limit: 50.0 %` (14/28) |

The pre-change log shape carries no `trigger=` or `cause=` field, so the harness
degrades both to `unknown` rather than inferring them
(`LP-0MVV29X2E004INFL` AC3/AC4). The `local_lease_active` bucket is the one
positive pre-change token (`reason=local_lease_active` on the outcome line); the
measurement-time record's `concurrency_limit` column is that set's **complement**
(135 outcomes − 107 lease-active), not a positively logged trigger. Post-change
lines carry `trigger=` explicitly, which is why the third window's row in the
post-change record resolves without any inference.

## Sanity position — a pre-change window is not a void window (AC5)

The post-change record warns that an all-`unknown` `rate_bucket` bucket would
mean the fix was not being served, i.e. a **void** window. For a pre-change
window the evidence is the mirror image, and this record states the position
explicitly so the two are never confused:

1. **The absence of the post-change fields is positive evidence of pre-change
   code — not missing data.** Across the surviving coverage, **all 1 379
   admission lines carry `rate_bucket` `unknown`** (per band: `<8k` 210,
   `8k-20k` 146, `20k-32k` 380, `32k-64k` 219, `>=64k` 424) because the field
   does not exist in the pre-change log shape; every abandon cause is `unknown`
   and every enqueue trigger is `unknown`. The post-change record's calibration
   evidence is the *presence* of `contended` / `solitary` /
   `solitary_fallback` (875 / 3 023 / 122) — the exact opposite signature. This
   window's signature is the pre-change one, and it is the signature the
   measurement-time run over the full window would have carried too (a window
   served by post-change code cannot lose the fields again).
2. **The new eviction class is absent, as it must be.** `measured_cache_deficit`
   first appears anywhere in the retained logs at **2026-10-29 00:13:30 UTC** —
   3 minutes *after* this window closes. Over the surviving coverage the window
   shows 48 `local_eviction_after_acceptance` events, **all**
   `reason=prefill_deadline` and **zero** `measured_cache_deficit`.
3. **The boundary is structural, not inferred.** The window ends at the deploy
   instant, so every line inside it was served by pre-deploy code; the surviving
   archives are contiguous within the pre-deploy side of that boundary (the
   deploy segment `proxy.log.2026-10-29_001015.gz` ends at
   `2026-10-29 00:10:15`, 3 s before the deploy).
4. **The window is not quiet/empty.** 576 admits and 4 078 rejects at
   measurement time (226 / 1 153 over the surviving slice) — traffic was served
   throughout, so a low queue count cannot be mistaken for a void window.

## What this record does and does not claim

- **Claims:** the diurnally matched pre-change share is **52.6 % (71/135)**, and
  the like-for-like change is **+8.1 points to 60.7 %**. The `36.9 % → 52.6 %`
  step is diurnal mix, not the code change.
- **Does not claim:** a fresh 2026-11-05 measurement of the full window (it
  cannot be re-derived — see the coverage notice), a per-band comparison of the
  diurnal pair (the pre-change per-band split is available only for the
  surviving slice), or any change to the parent's literal 85 % bar.
- **Not affected by** the harness's own band-attribution history: banding does
  not change the totals, and the totals quoted here are the totals
  (`LP-0MVV29WR0009DY1A` corrected the per-band split only). The diurnal
  window's per-band split has no committed full-coverage artefact: the
  measurement-time record reported outcome totals and the trigger split, not the
  per-band table.

## Reproducing

```bash
scripts/prefill_admission_conversion.py \
  --since "2026-10-28 00:10:18" \
  --until "2026-10-29 00:10:18" \
  --log-dir /var/log/llama-proxy \
  --json > docs/dev/prefill-admission-conversion-diurnal-baseline.json
```

1. The command regenerates the committed JSON byte-for-byte **modulo the
   rotation-dependent `log_files_scanned` inventory**, and only **while the two
   archives named in [Provenance and coverage](#provenance-and-coverage)
   survive retention**. Once they are pruned the JSON cannot be regenerated;
   the committed file is then the evidence, and its `log_files` field names
   exactly which archives produced it.
2. Confirm `since` / `until` in the JSON are the deploy-boundary pair — that is
   the invariant that makes the window diurnally matched and fully pre-change
   (pinned by
   [`tests/test_prefill_admission_conversion_diurnal_baseline.py`](../../tests/test_prefill_admission_conversion_diurnal_baseline.py)).
3. Read `log_files` (what contributed) against `log_files_scanned` (what was
   available) to see the coverage actually behind any number on this page.
4. For the full-window figures, the measurement-time harness output is recorded
   in the planning comment on `LP-0MVV1V213002QAIR`; re-running it today is
   *not* possible and is not silently implied by this page.

## Cross-reference

| # | Work item | Scope |
|---|-----------|-------|
| 1 | `LP-0MVV29XDW006VTWK` | This diurnally matched baseline record |
| 2 | `LP-0MVV1V213002QAIR` | Parent: diurnal restatement, mix/code separation, working-baseline regression |
| 3 | `LP-0MVTKTOVI006ZONN` | Post-change 24 h record whose baseline this restates ([`prefill-admission-conversion-postchange.md`](prefill-admission-conversion-postchange.md)) |
| 4 | `LP-0MVS9Y7FG004QJ2S` | Pre-change baseline harness ([`prefill-admission-conversion.md`](prefill-admission-conversion.md), the 19:00→19:00 window) |
| 5 | `LP-0MVV29WR0009DY1A` | Own-request queue-wait banding (no phantom band) |
| 6 | `LP-0MVV29X2E004INFL` | Entry-path (`trigger=`) and abandon-cause (`cause=`) attribution |
| 7 | `LP-0MVS8T6F4006DGG3` | Epic: contended-pricing fix whose AC4 these records evidence |
| 8 | `LP-0MVV9JWL9003A6XR` | Eviction triage: `measured_cache_deficit` new-class verdict |
| 9 | `LP-0MVT110PJ003XYLP` | Standalone contention-queue residual report (same harness vocabulary) |
| 10 | `LP-0MW3GU0JO008XOUF` | Proxy log retention (7 days) is shorter than the records that quote measured windows — why this window can no longer be re-derived (discovered from this record) |

Related: [`docs/dev/contended-prefill-pricing.md`](contended-prefill-pricing.md)
(the contended pricing and rollback page).
