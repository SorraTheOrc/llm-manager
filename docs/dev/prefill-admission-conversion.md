# Pre-change admission→first-byte conversion baseline (LP-0MVS8T6F4006DGG3)

This page is the **pre-change baseline for parent AC4** of
**LP-0MVS8T6F4006DGG3** (_Condition prefill-ETA calibration on co-residency;
fix occupant-remaining pricing_). Parent AC4 requires the share of queued local
waits ending in `contention_queue_fallback_after_queue` to drop below the 85 %
baseline measured 2026-10-19/20 over a 24 h window, with no new
`local_eviction_after_acceptance` events. This record captures that baseline
**before** the contended-pricing fix is deployed, so the post-change
measurement (now delivered by `LP-0MVTKTOVI006ZONN` as
[`prefill-admission-conversion-postchange.md`](prefill-admission-conversion-postchange.md))
has a like-for-like comparison on the current code.

The numbers are produced entirely by
[`scripts/prefill_admission_conversion.py`](../../scripts/prefill_admission_conversion.py)
(a pure log analyser — no proxy behaviour change, no running proxy required).
That page also carries the **post-change 24 h record** (AC4) delivered by
`LP-0MVTKTOVI006ZONN`, including the `local_eviction_after_acceptance` counts
by reason.

```bash
scripts/prefill_admission_conversion.py \
  --since "2026-10-27 19:00:00" \
  --until "2026-10-28 19:00:00" \
  --json > docs/dev/prefill-admission-conversion.json
```

Raw output: [`prefill-admission-conversion.json`](prefill-admission-conversion.json).
The window is 24 h; the `rate_bucket` of every admission is `unknown` because
the **running proxy predates the co-residency log fields** — confirming this is
genuinely a pre-change capture.

## Window and inputs

- **Window:** `2026-10-27 19:00:00` → `2026-10-28 19:00:00` (24 h, inclusive).
- **Log files read (6):** `proxy.log-2026-10-28_00.gz`,
  `proxy.log.2026-10-27_16.gz`, `proxy.log.2026-10-27_22.gz`,
  `proxy.log.2026-10-28_04.gz`, `proxy.log.2026-10-28_10.gz`,
  `proxy.log.2026-10-28_16.gz`.
- **First byte with no admission line:** 6 sessions (reported separately,
  never silently joined to an admission — harness AC2).
- **Queued waits enqueued:** 65 (`contention_queue_wait`) against 65 outcomes
  (41 dispatched + 24 abandoned). None of the enqueue lines carries its own
  pricing (they predate the field added in `LP-0MVV29WR0009DY1A`), so waits are
  banded by the same-session join when that evidence is unused and otherwise
  reported as `unknown` (3 waits here).

Bands are on the **priced new (uncached) tokens** of a local dispatch:
`<8k`, `8k-20k`, `20k-32k`, `32k-64k`, `>=64k` (the `20k`/`32k` boundaries
mirror `local_admission_guard_eta_percentile_min_tokens` and
`CONSERVATIVE_MIN_TOKENS`).

## Per-band baseline

Re-banded on 2026-10-31 with the own-request queue-wait join rule
(`LP-0MVV29WR0009DY1A`, documented in
[the post-change record](prefill-admission-conversion-postchange.md#queue-wait-attribution-lp-0mvv29wr0009dy1a)):
five waits that the earlier rule had banded from a consumed or later admission
now sit in `unknown`. Verdicts and the 36.9 % headline are unchanged.

| Band | Admits | Rejects | Converted | Unconverted | Conv % | Queued waits | Abandoned | Abandoned % | Queue-wait s |
|---|---|---|---|---|---|---|---|---|---|
| `<8k` | 397 | 36 | 27 | 15 | 64.3 % | 55 | 20 | 36.4 % | 2193.7 |
| `8k-20k` | 24 | 152 | 10 | 5 | 66.7 % | 7 | 4 | 57.1 % | 391.8 |
| `20k-32k` | 2 | 692 | 1 | 1 | 50.0 % | 0 | 0 | 0.0 % | 0.0 |
| `32k-64k` | 0 | 1091 | 0 | 0 | — | 0 | 0 | 0.0 % | 0.0 |
| `>=64k` | 0 | 1983 | 0 | 0 | — | 0 | 0 | 0.0 % | 0.0 |
| `unknown` | 0 | 0 | 0 | 0 | — | 3 | 0 | 0.0 % | 60.2 |
| **Total** | **423** | **3954** | **38** | **21** | **64.4 %** | **65** | **24** | **36.9 %** | **2645.7** |

Reject reasons are dominated by the flat cold-prefill budget
(`prefill_budget_exceeded`: 3 557) and the deadline bound
(`prefill_eta_exceeds_deadline`: 184), with queue-budget refusals split across
`eta_exceeds_budget` / `eta_exceeds_budget_direct`.

### `local_eviction_after_acceptance` in the baseline window

The harness's `evictions` block (`LP-0MVV9JWL9003A6XR`) gives the genuine
pre-change comparator for AC4's second clause:

| Reason | Pre-change (24 h) | Pre / h |
|---|---|---|
| `prefill_deadline` | 145 | 6.04 |
| `idle_timeout` | 3 | 0.13 |
| `no_progress_watchdog` | 2 | 0.08 |
| `lease_evicted` | 1 | 0.04 |
| `orphan_cleanup` | 1 | 0.04 |
| **Total** | **152** | **6.33** |

`measured_cache_deficit` is **absent** from this window (0 occurrences): the
running proxy predates the deploy (`52085f09`, serving from 2026-10-29
00:10:18) of the measured-cache early-abort feature
(`LP-0MVR7IN0O000DPXN` / `LP-0MVR7HJD0007RFEM`) that emits it — its commits
were on `dev` by 2026-10-28 05:49 but were not yet serving. The number that
appears in the earlier revision of the post-change record — **463** — was a
naive substring count that included **311 `[remote] POST … body={…}`
request-body lines** quoting the marker in an agent prompt (this window is
when `LP-0MVQIT9QY007QU27` was being worked).
The harness now anchors each marker to the start of the logging message, so
only the 152 genuine records above are counted; the 311 body lines contribute
zero (`LP-0MVV9JWL9003A6XR`).

## Comparison to the parent AC4 reference

The parent's 85 % reference figure was measured on 2026-10-19/20 **before** the
intervening admission-guard work (deadline bound, cold-prefill budget,
occupancy floor). On the current pre-change code the overall abandoned-wait
share is **36.9 %** (24 of 65 queued waits), materially below that historical
85 %.

That makes the comparison two-sided, and both are recorded here:

- **Literal parent AC4:** the post-change share must be **below 85 %**. A
  36.9 % baseline already satisfies that, so the AC4 measurement must not
  regress above 36.9 % / the 85 % reference.
- **Stricter working baseline:** use the captured **36.9 %** (and 36.4 % in
  `<8k`, 57.1 % in `8k-20k`) as the pre-change bar for child
  `LP-0MVS9Z0GY008LCJA`, so the contended-pricing fix is judged against the
  code it actually changes rather than a stale 85 % figure.

The post-change run must also compare its
`local_eviction_after_acceptance` events against this baseline (parent AC4's
second clause). Since this window records 152 genuine events (6.3 / h) and
**no** `measured_cache_deficit`, the clause's "no new events" reading is
checked against 152, not zero — the later post-change record
([`prefill-admission-conversion-postchange.md`](prefill-admission-conversion-postchange.md))
records 392 events including the genuinely new `measured_cache_deficit` class,
which is triaged in `LP-0MVV9JWL9003A6XR`.

## Reproducing

1. Run the command above (any window; `--log-dir` defaults to
   `/var/log/llama-proxy`).
2. Confirm the JSON's `log_files` span the window and `since`/`until` match
   the requested bounds.
3. The harness is unit-tested from synthetic log lines
   ([`tests/test_prefill_admission_conversion.py`](../../tests/test_prefill_admission_conversion.py)),
   including pre-change line shapes without the newer `rate_bucket` /
   `contended_samples` fields and the own-request queue-wait banding rule
   (`LP-0MVV29WR0009DY1A`).

## Cross-reference

| # | Work item | Scope |
|---|-----------|-------|
| 1 | `LP-0MVS9Y7FG004QJ2S` | This baseline harness (pre-change capture) |
| 2 | `LP-0MVS9Y76R004B5M8` | Contended prefill rate bucket (`rate_bucket` log field) |
| 3 | `LP-0MVS9YK9P006M1NQ` | Occupant-remaining priced from new tokens at the contended rate |
| 4 | `LP-0MVS9YKHL007UBBZ` | Decision-log / usage-report surface for `rate_bucket` |
| 5 | `LP-0MVS9Z0GY008LCJA` | Post-change 24 h measurement vs this baseline (superseded by `LP-0MVTKTOVI006ZONN`, which delivered [`prefill-admission-conversion-postchange.md`](prefill-admission-conversion-postchange.md)) |
| 6 | `LP-0MVS9Z034006MFZ6` | Docs and rollback |
| 7 | `LP-0MVV9JWL9003A6XR` | Eviction triage that corrected this baseline's `local_eviction_after_acceptance` comparator (152 genuine, not the 463 substring count) and made the harness count only message-anchored eviction records |
| 8 | `LP-0MVV29WR0009DY1A` | Own-request queue-wait banding — re-banded this baseline (5 waits moved to `unknown`; verdicts unchanged) |

Related: [`docs/dev/contended-prefill-pricing.md`](contended-prefill-pricing.md)
(the contended pricing and rollback page).
