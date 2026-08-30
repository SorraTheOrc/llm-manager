# Contention-Queue Burst Profile (cheap-mode fan-out)

**Window:** 2026-08-22 22:42:05 -> 2026-08-28 00:00:00
**Files:** 8
**Work item:** LP-0MTF0G4VH003DXB9 (T2 of LP-0MTED3OFP006I7NO)

## 1. Concurrent local requests during bursts

- Full-occupancy contention events (denied + queue-path arrivals): **1654**
- Inter-arrival gaps: median **5.26s**, p90 **221.22s**, p99 **4254.31s**
- Burst definition: runs of full-occupancy contention events (denied + queue-path arrivals), inter-event gap <= 30s, >= 2 events
- Bursts detected: **233**
- Arrivals per burst: median 3, max 54
- Burst duration: median 20.96s, max 236.72s

`local_active_queries` (from `local_dispatch_denied active=`, sampled inside bursts):

| value | count |
|---|---|
| 0 | 67 |
| 1 | 523 |
| 2 | 567 |

`available_slots` (status_request snapshots):

| value | count |
|---|---|
| 0 | 5219 |
| 1 | 6553 |
| 2 | 2549 |

## 2. Queue depth observed

`contention_queue_depth` (status_request snapshots):

| depth | count |
|---|---|
| 0 | 12015 |
| 1 | 1200 |
| 2 | 604 |
| 3 | 282 |
| 4 | 34 |

`depth` after pop (contention_queue_dispatch lines):

| depth | count |
|---|---|
| 0 | 176 |
| 1 | 59 |
| 2 | 12 |
| 3 | 2 |

## 3. Queue-wait durations

Dispatched (`queued_duration` on contention_queue_dispatch):

- n=249, median 17.62s, p90 47.21s, p95 53.38s, max 59.59s

Fell back (`queued_duration` on contention_queue_fallback_after_queue):

- n=166, values [60.0, 60.03] (== wait cap)

## 4. Fallbacks while the queue was non-empty

- `contention_queue_fallback_after_queue` (166 total): 111 with queue depth > 0, 31 with queue depth 0, 24 without a snapshot within 30s
- `Fallback triggered reason=local_concurrency_limit` (4327 total): 1 with queue depth > 0, 0 with queue depth 0, 4326 without a snapshot within 30s

## Methodology

- Script: `python3 proxy/benchmarks/contention_queue_profile.py [--log-files ...] [--start ...] [--end ...] --report docs/dev/contention-queue-burst-profile.md`
- Parsing: T1 harness `contention_queue_simulation.load_events`; status `contention_queue_depth`/`available_slots` and `Fallback triggered` lines parsed by this script
- `available_slots` is collected only from cheap-tier snapshot lines (those carrying `contention_queue_depth`); status lines from other pools (e.g. the mxbai-embed embedding tier) are excluded
- Metric 4 splits events by the nearest status snapshot's queue depth within +-30s: `gt0` (queue non-empty), `eq0` (verified empty), `no_snapshot` (not derivable — no snapshot within the delta; these fallbacks occur in hours without cheap-tier snapshot traffic)
