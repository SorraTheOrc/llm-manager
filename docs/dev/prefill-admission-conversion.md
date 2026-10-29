# Pre-change admission→first-byte conversion baseline (LP-0MVS8T6F4006DGG3)

This page is the **pre-change baseline for parent AC4** of
**LP-0MVS8T6F4006DGG3** (_Condition prefill-ETA calibration on co-residency;
fix occupant-remaining pricing_). Parent AC4 requires the share of queued local
waits ending in `contention_queue_fallback_after_queue` to drop below the 85 %
baseline measured 2026-10-19/20 over a 24 h window, with no new
`local_eviction_after_acceptance` events. This record captures that baseline
**before** the contended-pricing fix is deployed, so the post-change
measurement (child `LP-0MVS9Z0GY008LCJA`) has a like-for-like comparison on
the current code.

The numbers are produced entirely by
[`scripts/prefill_admission_conversion.py`](../../scripts/prefill_admission_conversion.py)
(a pure log analyser — no proxy behaviour change, no running proxy required):

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
- **Log files read (6):** `proxy.log`, `proxy.log-2026-10-28_00`,
  `proxy.log.2026-10-27_16`, `proxy.log.2026-10-27_22`,
  `proxy.log.2026-10-28_04`, `proxy.log.2026-10-28_10`.
- **First byte with no admission line:** 6 sessions (reported separately,
  never silently joined to an admission — harness AC2).

Bands are on the **priced new (uncached) tokens** of a local dispatch:
`<8k`, `8k-20k`, `20k-32k`, `32k-64k`, `>=64k` (the `20k`/`32k` boundaries
mirror `local_admission_guard_eta_percentile_min_tokens` and
`CONSERVATIVE_MIN_TOKENS`).

## Per-band baseline

| Band | Admits | Rejects | Converted | Unconverted | Conv % | Queued waits | Abandoned | Abandoned % | Queue-wait s |
|---|---|---|---|---|---|---|---|---|---|
| `<8k` | 397 | 36 | 27 | 15 | 64.3 % | 57 | 20 | 35.1 % | 2236.8 |
| `8k-20k` | 24 | 152 | 10 | 5 | 66.7 % | 8 | 4 | 50.0 % | 408.9 |
| `20k-32k` | 2 | 692 | 1 | 1 | 50.0 % | 0 | 0 | 0.0 % | 0.0 |
| `32k-64k` | 0 | 1091 | 0 | 0 | — | 0 | 0 | 0.0 % | 0.0 |
| `>=64k` | 0 | 1983 | 0 | 0 | — | 0 | 0 | 0.0 % | 0.0 |
| **Total** | **423** | **3954** | **38** | **21** | **64.4 %** | **65** | **24** | **36.9 %** | **2645.7** |

Reject reasons are dominated by the flat cold-prefill budget
(`prefill_budget_exceeded`: 3 557) and the deadline bound
(`prefill_eta_exceeds_deadline`: 184), with queue-budget refusals split across
`eta_exceeds_budget` / `eta_exceeds_budget_direct`.

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
- **Stricter working baseline:** use the captured **36.9 %** (and 35.1 % in
  `<8k`, 50.0 % in `8k-20k`) as the pre-change bar for child
  `LP-0MVS9Z0GY008LCJA`, so the contended-pricing fix is judged against the
  code it actually changes rather than a stale 85 % figure.

The post-change run must also confirm **zero new
`local_eviction_after_acceptance` events** for within-budget prompts (parent
AC4) — this page records the pre-change state only.

## Reproducing

1. Run the command above (any window; `--log-dir` defaults to
   `/var/log/llama-proxy`).
2. Confirm the JSON's `log_files` span the window and `since`/`until` match
   the requested bounds.
3. The harness is unit-tested from synthetic log lines
   ([`tests/test_prefill_admission_conversion.py`](../../tests/test_prefill_admission_conversion.py)),
   including pre-change line shapes without the newer `rate_bucket` /
   `contended_samples` fields.

## Cross-reference

| # | Work item | Scope |
|---|-----------|-------|
| 1 | `LP-0MVS9Y7FG004QJ2S` | This baseline harness (pre-change capture) |
| 2 | `LP-0MVS9Y76R004B5M8` | Contended prefill rate bucket (`rate_bucket` log field) |
| 3 | `LP-0MVS9YK9P006M1NQ` | Occupant-remaining priced from new tokens at the contended rate |
| 4 | `LP-0MVS9YKHL007UBBZ` | Decision-log / usage-report surface for `rate_bucket` |
| 5 | `LP-0MVS9Z0GY008LCJA` | Post-change 24 h measurement vs this baseline |
| 6 | `LP-0MVS9Z034006MFZ6` | Docs and rollback |

Related: [`docs/dev/contended-prefill-pricing.md`](contended-prefill-pricing.md)
(the contended pricing and rollback page).
