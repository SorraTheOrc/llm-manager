# Stall-weighted prefill-deadline rate (LP-0MW7V30AZ007JB5Z)

The daily usage report's wasted-GPU-compute surface (LP-0MW2SQNHJ0044PV6)
counts post-acceptance evictions as *events*. An event count cannot price the
defect: a `measured_cache_deficit` abort that detours in under a second and a
`prefill_deadline` stall that burns the whole 180 s backstop are both +1 to
the (b2) rate. This item adds the missing **stall-weighted** view so the
re-scope on LP-0MVO8H7VQ003B12I can choose between an event-rate target and a
stall-cost target with both numbers on the table.

## What is published

- **Stall-weighted rate** `stall_seconds_per_hour = Σ elapsed_s / window_hours`,
  for the whole window, per reason, and per reason × session class
  (user / `audit-*`), alongside the existing per-event rate
  (AC1).
- **Stall percentiles** p50 / p90 / p95 / p99 of `elapsed_s` per reason
  (user traffic), plus the count and share of events under 1 s, under 5 s,
  and at/within 1 s of the deadline (AC2).
- **Definitions appendix** in `report.md` stating explicitly that
  `measured_cache_deficit` aborts and `prefill_deadline` stalls are *not*
  equivalent, why (an early detour versus a full-deadline wait), and quoting
  the window's measured p50/p90 (AC3).
- **JSON**: the daily `--json` gains a `stall_cost` block inside
  `wasted_compute`; every pre-existing key keeps its name, meaning and type
  (AC4, schema-compatibility test).

## Implementation

- `proxy/scripts/analyse_prefill_deadline_residual.py` — `stall_cost()`
  builds the analyser payload block; `--deadline-seconds` (default 180, the
  shipped `local_prefill_deadline_seconds`) bounds the "at/near deadline"
  bucket, and the payload echoes the threshold it used.
- `.pi/skills/proxy-usage-analysis/scripts/wasted_compute.py` — normalises the
  block into a `StallCost` (with the (b2) composite: `prefill_deadline` +
  `measured_cache_deficit` for user traffic) and carries it in `to_json()`.
- `.pi/skills/proxy-usage-analysis/scripts/reporting.py` — summary-table row,
  detail table and the definitions-appendix paragraph.

The stall cost reads only `reason` / `session` / `elapsed_s`, all already on
the `local_eviction_after_acceptance` line: nothing in the proxy's emission or
decision path changes (AC6).

## Certified-window verification (2026-11-06 14:15:50 → 2026-11-08 00:00)

The certified-window proxy logs have rotated out under the 7-day retention
before this item ran (`/var/log/llama-proxy` holds no `proxy.log*` from before
2026-11-09 20:16). The RCA's recorded analyser payload for exactly that window
survived, and the `prefill_deadline` eviction lines were reconstructed verbatim
into `proxy/tests/fixtures/certified_window_2026_11_06_deadline_events.log`,
which the test suite pins.

| Figure | Certified value | Reproduced |
|---|---|---|
| Combined user-class rate (`prefill_deadline` + `measured_cache_deficit`, user) | 2.223/h (75 user events / 33.736 h) | exact |
| `prefill_deadline` user events | 40 | exact (from recorded payload) |
| `prefill_deadline` user stall-weighted rate | Σ 7227.9 s / 33.736 h = 214.2 s/h | exact |
| `prefill_deadline` user p50 / p90 / p95 / p99 | 180.1 / 180.8 / 180.9 / 191.6 s | exact |
| `prefill_deadline` user under 1 s / at-near-deadline | 0 / 40 | exact |
| `measured_cache_deficit` user p50 / p90 / max | 24.9 / 127 / 175.5 s | quoted from item record (raw per-event rows not retained) |
| `measured_cache_deficit` sub-second | 12 of 47 | quoted from item record |

The `measured_cache_deficit` leg is quoted, not recomputed, because the
analyser's recorded payload stores only the attributed `prefill_deadline`
per-event rows; the deficit leg's per-event `elapsed_s` for the certified
window were not retained. The distinction is stated explicitly so neither the
item nor this report overclaims what the pruned logs can no longer prove.

## Re-running a window

```bash
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-10 14:18:26" --end "2026-11-11 15:18:26" \
  --deadline-seconds 180 --json
```

Retained group-1 window example (2026-11-10 14:18:26 → 2026-11-11 15:18:26,
25 h): `prefill_deadline` user 4 events / 744.2 s / 29.8 s/h (p50 181.4 s,
all at the deadline) versus `measured_cache_deficit` user 14 events / 16.9 s /
0.7 s/h (p50 0.4 s, 11 of 14 under 1 s) — the contrast the event rate hides.
