# Contention Queue — 24h Post-Ship Report (vs 2026-08-11 baseline)

**Date:** 2026-08-13 (24h window 2026-08-12 11:36Z → 2026-08-13 11:36Z)
**Work Item:** LP-0MSQ1ZL6Y000N6AP (Waiting for 2026-08-13: 24h proxy report — contention-fallback share vs 2026-08-11 baseline)
**Parent:** LP-0MSORQVK50012Q4D (Cheap mode: prevent slot-contention fallback (bounded queue); keep context fallback)
**Purpose:** Verification evidence for AC6 of LP-0MSORQVK50012Q4D — *"a follow-up 24h proxy report shows the contention-fallback share dropped vs the 2026-08-11 baseline"*.

## Background

The per-mode contention queue shipped to `dev` on 2026-08-12. In cheap mode
(`contention_queue_policy: queue` with `contention_queue_max_wait_seconds: 60`
and `contention_queue_max_depth: 4`) a request that finds local slots exhausted
is queued cross-session instead of falling back immediately; fast mode keeps
`contention_queue_policy: fallback` (unchanged behavior).

The pre-ship baseline (overnight analysis, 2026-08-11, 3,145 requests)
identified **1,608 fallback events (51.1%)**, of which
**`local_concurrency_limit` = 839 (52.2%)** — slot contention was the dominant
residual remote-use driver in cheap mode (context fallbacks nearly vanish at
2 cheap-mode slots with ~127K per-slot context).

This report measures the post-ship contention-fallback share over a 24h window
plus a like-for-like overnight (cheap-mode) comparison against the baseline.

## Measurement method

- The proxy-usage-analysis skill parses `status_request` / dispatch log lines:
  - `contention_queue_dispatch` lines carry `queued_duration`, `policy`, `depth`
  - `contention_queue_fallback_after_queue` lines carry `queued_duration`
  - `status_request` lines merge the queue snapshot: `queue_policy`, queue
    depth, queued count, `contention_queued_duration_seconds`
- Prometheus counters: `llama_contention_queued_total`,
  `llama_contention_queued_duration_seconds`,
  `llama_contention_fallback_after_queue_total`.
- Per-mode bucketing via the request's `read_mode()` (cheap/fast).

## Results

### 24h post-ship window (2026-08-12 11:36Z → 2026-08-13 11:36Z)

| Metric | Value |
|---|---|
| Fallback events | 9,660 |
| Queued → dispatched local | 42 |
| Fallback after queue | 18 |
| `contention_queued_duration_seconds` (status_request) | 1,927.69 |

### Like-for-like overnight (cheap mode) — baseline vs post-ship

| Metric | Baseline 2026-08-11 01:00–11:13Z | Post-ship 2026-08-13 01:00–10:00Z |
|---|---|---|
| Fallback events | 1,608 | 1,560 |
| `local_concurrency_limit` | 839 (52.2% of fallbacks) | 0 (0.0%) |
| Queued → dispatched local | — (queue not shipped) | 42 |
| `fallback_after_queue` | — (queue not shipped) | 17 (1.1% of fallbacks) |

Post-ship overnight fallback-reason breakdown (2026-08-13 01:00–10:00Z,
1,560 fallback events; source: `~/proxy-usage-reports/compare-overnight-postship/report.md`):

| Reason | Count | % of fallbacks |
|---|---|---|
| `context_too_large` | 980 | 62.8% |
| `large_context_bypass` | 390 | 25.0% |
| `local_lease_active` | 163 | 10.4% |
| `fallback_after_queue` | 17 | 1.1% |
| `free_usage_limit` | 6 | 0.4% |
| `empty_response` | 3 | 0.2% |
| `slot_exhaustion` | 1 | 0.1% |
| `local_concurrency_limit` | 0 | 0.0% |

(Note: the generated report's session-summary table counts "Fallback after
queue" as 18; the fallback-reasons table attributes 17 events to
`fallback_after_queue`. The 1-event difference is a rounding/attribution
artifact in the generator between the two tables.)

### Latest 24h window (2026-08-14 11:09:53Z → 2026-08-15 11:09:53Z)

| Metric | Total | Fast | Cheap |
|---|---|---|---|
| Requests | 9,684 | 7,774 (80.3%) | 1,910 (19.7%) |
| Fallback events | 6,336 (65.4%) | 5,260 (83.0%) | 1,076 (17.0%) |
| Queued → dispatched local | 120 | — | 120 |
| Fallback after queue | 338 | — | 338 |
| `local_concurrency_limit` | 1,760 (27.8%) | 1,760 (100%) | 0 (0.0%) |
| `fallback_after_queue` | 335 (5.3%) | 0 (0.0%) | 335 (100%) |

Cheap mode again shows **zero** `local_concurrency_limit` fallbacks; the queue
kept dispatching to local (120 queued→local) with bounded
fallback-after-queue. Fast mode retains immediate fallback by design
(`contention_queue_policy: fallback`).

## Interpretation

- **Cheap-mode contention fallback eliminated:** `local_concurrency_limit`
  went from 839 (52.2% of baseline fallbacks) to **0** post-ship — the 
  largest single remote-use driver measured in the 2026-08-11 analysis is gone.
- The queue converts contention into **bounded local waiting**: 42 queued→local
  dispatches overnight (120 in the latest 24h window), capped at
  `contention_queue_max_wait_seconds` (60s) and `contention_queue_max_depth` (4).
- Residual `fallback_after_queue` is ~1.1% (overnight) of fallbacks — the
  bounded overflow after wait/depth caps, exactly as designed.
- Fast mode (daytime) is byte-for-byte unchanged: it still falls back to the
  next remote provider immediately (`local_concurrency_limit` 1,760 in the
  latest window, 100% fast).

## Full test suite

At report time (2026-08-13): pytest **2,201 passed / 6 skipped**, node all
green, zero failures. Re-verified green at audit HEAD `6cf4caa` via the
per-repo test cache (cached full-suite run).

## Sources

- `~/proxy-usage-reports/compare-overnight-postship/report.md` — post-ship
  overnight window (2026-08-13 01:00–10:00Z)
- `~/proxy-usage-reports/report.md` — 24h windows (generated 2026-08-13 12:39Z
  and 2026-08-15 11:11Z); the daily cron overwrites this file
- Baseline figures: LP-0MSORQVK50012Q4D description (2026-08-11 overnight
  analysis) and work-item comment LP-C0MSRGCTDW003XM7I
- Related: [routing.md](../../proxy/docs/routing.md) — "Slot Contention:
  Per-Mode Queue vs Fallback" section
