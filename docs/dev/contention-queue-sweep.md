# Contention-queue sweep: residual-queue report

Runbook for the residual-queue telemetry behind the 60 s → 30 s contention-queue
sweep (**LP-0MVIKPRMD000Y97R**, telemetry child **LP-0MVT110PJ003XYLP**).

The sweep shortens `contention_queue_max_wait_seconds` (60 → 30) and
`contention_queue_max_depth` (8 → 2) so the queue becomes a straggler net rather
than a parking lot. **Shipped caps after the sweep: 30 s / depth 2**
(`proxy/config.yaml`, documented in [`proxy/docs/routing.md`](../../proxy/docs/routing.md));
the baseline tables below record the *pre-flip* pair by design, because that is
the measurement the verdict is compared against. Because the flip criteria are stated against measured
numbers, the pre-flip baseline and the post-flip verdict must be produced by
**one definition**. That definition is
`scripts/contention_queue_sweep_report.py`.

The tool is a pure log analyser: it never changes proxy behaviour, needs no
running proxy, and adds **no new status counters** — the log events are the
telemetry contract (the cumulative counters were deliberately removed in
LP-0MU4NACW9001WKBP).

## Invocation

```bash
# Last 24 h (default), human-readable:
./scripts/contention_queue_sweep_report.py

# A closed, reproducible window — the pre-flip baseline:
./scripts/contention_queue_sweep_report.py \
    --since '2026-10-30 05:00' --until '2026-10-31 05:00' --json

# Markdown section for a work-item comment:
./scripts/contention_queue_sweep_report.py --hours 24 --markdown
```

| Flag | Meaning |
|---|---|
| `--log-dir` | Proxy log directory (default `/var/log/llama-proxy`). Live `proxy.log` plus rotated files under **both** naming schemes, plain or `.gz`. |
| `--since` / `--until` | Explicit window bounds (`YYYY-MM-DD HH:MM[:SS]`, ISO 8601, local time; bounds inclusive). `--until` defaults to now. |
| `--hours` | Window length when `--since` is omitted (default 24). Ignored when `--since` is given. |
| `--sweep-budget-seconds` | The budget the sweep applies, used for the win at-or-below share (default 30). |
| `--min-hours` | Window shorter than this is an incomplete measurement (default 24 — the pre-flip baseline precondition). |
| `--max-gap-seconds` | Adjacent-file log gap that makes the window unmeasurable (default 3600). |
| `--json` / `--markdown` | Machine-readable output / markdown section. |

Reuse: log discovery, timestamps and coverage measurement come from
`scripts/lib/proxy_logs.py`; the contention-queue field extractors are loaded
**read-only** from `.pi/skills/proxy-usage-analysis/scripts/log_parser.py` (the
behavioural reference for the line shape), so the two definitions cannot drift.
A checkout without `.pi/` degrades to built-in patterns and says so
(`meta.parser`).

## The three sections

**(a) Ticket-backed dispatches** — `local_admission_reserve_claimed` lines: a
cooperating client presented a reservation ticket and skipped the queue.
Reported **for context only**; it is not a flip gate.

**(b) Residual queue wins** — `contention_queue_dispatch` lines: a queued
request got a local slot. Reported as a count, the share whose
`queued_duration` is at or below `--sweep-budget-seconds` (the "what would the
shorter budget still have won" number), and the wait distribution
(p50/p75/p90/max).

**(c) Residual queue losses** — `contention_queue_fallback_after_queue` lines:
the request parked and then went remote anyway (dead wait). Reported as a
count, aggregate wait, wait-per-hour, and a split by `reason`:

- `local_lease_active` — only the two lease-retention emit sites in
  `proxy/proxy/provider.py` carry it (a foreign session holds the local lease);
- `none` — the line carries no `reason=` at all;
- any other reason is reported under its own token.

No loss is ever dropped: the split always sums to the loss count.

"Residual" is deliberate — it is what the queue still does once cooperating
clients reserve capacity, and the loss split separates an owner-level problem
(lease retention) from genuine capacity contention.

The report also carries the queue-level denominator (`queue_events`, total and
per hour, with the win/loss share) because the flip criteria are phrased
against "queue events", not against wins alone.

Log lines are counted only when the marker **begins the logging message body**
(leftmost prefix anchor), so a proxied request body that quotes a probe line can
never inflate a count (the LP-0MVV9JWL9003A6XR inflation class).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Measurement complete (window ≥ `--min-hours`, no adjacent-file coverage gap, data present). |
| `2` | Measurement **incomplete** — no logs, an empty window, a thin window, or a log-coverage gap. The numbers are still printed (so a partial view is never lost) but the run never silently passes. |
| `1` | Invalid input — bad path, bad timestamp, non-positive budget/gap, window start ≥ end. |

## Pre-flip baseline (60 s / depth 8)

Measured 2026-10-31 with `contention_queue_max_wait_seconds: 60` and
`contention_queue_max_depth: 8`, over the closed window
**2026-10-30 05:00 → 2026-10-31 05:00** (24.0 h, 9 log files, no coverage gap;
exit `0`). Raw machine-readable output:
[`contention-queue-sweep-baseline-2026-10-31.json`](contention-queue-sweep-baseline-2026-10-31.json).

| Section | Metric | Value |
|---|---|---|
| (a) ticket-backed | `local_admission_reserve_claimed` | **0** (0.0/h) |
| (b) wins | `contention_queue_dispatch` | **214** (8.92/h) |
| | at or below the 30 s sweep budget | **124** (57.9% of wins) |
| | over 30 s | 90 |
| | wait p50 / p75 / p90 / max | 22.19 / 36.54 / 48.45 / 59.49 s |
| (c) losses | `contention_queue_fallback_after_queue` | **137** (5.71/h) |
| | aggregate dead wait | **8 222.21 s** (342.59 s/h, mean 60.02 s) |
| | loss wait p50 / p75 / p90 / max | 60.0 / 60.0 / 60.0 / 60.35 s |
| | `reason=local_lease_active` | **73** losses, 4 381.15 s (53.3%) |
| | `reason` absent (`none`) | **64** losses, 3 841.06 s (46.7%) |
| | other reasons | 0 |
| denominator | queue events | **351** (14.62/h) — wins 61.0% / losses 39.0% |

Reading it against the parent's flip criteria (all three must hold over the
7-day observation window):

1. **Dead wait per hour falls ≥ 50%** — baseline is **342.59 s/h**, so the bar is
   ≤ 171.3 s/h. Note the projection below: the budget change alone looks
   unlikely to reach it.
2. **Residual wins stay ≥ 20% of queue events** — baseline is **61.0%**
   (214/351), so there is headroom; at a 30 s budget **124** of those wins
   (57.9%) would still have been retained.
3. **Remote dispatch volume rises < 20% relative** — *not measured here*: remote
   fallback volume is the proxy-usage-analysis skill's report. Use it alongside
   this one for that criterion.

Losses sit at the full budget (p50 = p75 = p90 = 60.0 s): every loss is a
request that parked for the entire window and then paid for a remote provider
anyway. Just over half of that dead wait (53.3%) is the lease-retention branch,
not raw capacity contention.

**Projection caveat (not a measurement).** Applying a 30 s cap to this window
naively — the 90 wins that waited 30–60 s become losses at 30 s — gives about
(137 + 90) × 30 s ≈ 6 810 s / 24 h ≈ **283.8 s/h**, a fall of only ~17%, i.e.
short of the ≥ 50% bar. The depth cap (8 → 2) should shed some of these requests
before they queue at all, which lowers dead wait further but also lowers the win
count, and that effect is **not** modelled here. Do not treat the projection as
a verdict; produce the post-flip report with the same command and read the
measured numbers.

## Post-flip verdict

After the sweep is applied and the proxy restarted, run the identical command
over the post-flip window (7 days / ≥ 1 000 queue events) and compare the same
fields:

```bash
./scripts/contention_queue_sweep_report.py \
    --since '<flip time>' --until '<flip time + 7d>' \
    --sweep-budget-seconds 30 --json > /tmp/contention-queue-sweep-postflip.json
```

The verdict is a comparison of the three criteria above against the baseline
table; the rollback (revert to 60 / 8 and restart the proxy) is documented on
the parent item.
