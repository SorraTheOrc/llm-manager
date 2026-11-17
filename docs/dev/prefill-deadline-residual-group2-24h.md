# Prefill-deadline residual: group-2 ≥ 24 h restart-free measurement

**Work item:** LP-0MW8EAU8P0079280 (AC3/AC4 of the group-2 validation epic
LP-0MW7V14DN007F127).
**Measurement tool:** `proxy/scripts/analyse_prefill_deadline_residual.py`
(evictions and reasons, user/audit split, stall-weighted cost, cause
attribution, abandoned-prefill overlap).
**Dataset (committed beside this document):**
`docs/dev/prefill-deadline-residual-group2-24h.json`.
**Window:** `2026-11-16 15:20:47` → `2026-11-17 18:20:00` (26.987 h), anchored
on the single group-2 restart recorded in `docs/dev/group-2-restart.md`.

This is an **evidence** document. It reports a measurement taken over a
certified restart-free window; it changes no admission decision, no threshold
and no configuration, and it re-tunes nothing to make the numbers land. The
certification verdict on these numbers is owned by the sibling verdict item
LP-0MW8EAUQU006DB0U (parent AC5), not by this document.

## 0. Result at a glance

Certified criterion (producer decision 2026-11-02, option **b2**): user-class
`local_eviction_after_acceptance reason=prefill_deadline` +
`reason=measured_cache_deficit` ≤ **0.5 events/h** over a ≥ 24 h restart-free
window; **not certified** if occupancy-attributed ≥ 25 % or
capacity-attributed ≥ 50 % of the `prefill_deadline` events.

| Figure (this window, user class unless stated) | Group-2 value | Criterion / prior |
|---|---|---|
| Combined (b2) events — `prefill_deadline` + `measured_cache_deficit` | **4** (1 + 3) | 60 on the post-change 24 h window |
| **Combined event rate** | **0.148/h** | ≤ 0.5/h gate → **met** |
| `prefill_deadline` events | **1** (0.037/h) | 26 on the post-change 24 h window |
| `measured_cache_deficit` events | **3** (0.111/h) | 34 on the post-change 24 h window |
| Occupancy attribution, `prefill_deadline` | **1 of 1 = 100 %** | < 25 % guard → **not met**, on a one-event base |
| Capacity attribution, `prefill_deadline` | **0 of 1 = 0 %** | < 50 % guard → met |
| Abandoned-window overlap, measured basis | **0 of 1 = 0 %** (legacy basis also 0 of 1) | ≤ 20 % target → met |
| **Combined stall-weighted rate** | **6.844 s/h** (184.7 s) | the metric that separates a detour from a deadline stall |
| `prefill_deadline` user p50 / p95 | **180.1 s / 180.1 s** | a full-deadline stall, unchanged in shape |
| `measured_cache_deficit` user p50 / max | **0.5 s / 3.6 s** | early detours, 2 of 3 under 1 s |
| **`no_progress_watchdog` (outside b2)** | **184 events, 6.818/h** (user 173 = 6.411/h) | 3 events on the post-change 24 h window — see §4 |

Read in one line: the two reasons the (b2) criterion names are now **rare**
(4 events in 27 h, 0.148/h — under the 0.5/h gate) and their stall cost is
small (6.8 s/h combined), but the `prefill_deadline` guard cannot be satisfied
on a single event (100 % occupancy-attributed), and a **different**
post-acceptance class — `no_progress_watchdog`, 184 events / 6.818/h — grew by
more than an order of magnitude against the 2026-11-06 post-change windows.
That class is not in (b2) as written; it is reported here because it is the
same user-visible defect (an accepted dispatch released before its first byte)
and it dominates this window's abandoned work (§4).

## 1. Window certification (AC3)

The anchor is the **restart**, not a commit date: the group-2 restart of
`2026-11-16T15:20:47Z` (proxy PID `3720326`, deployed HEAD `79f53ac1a`) is the
window's boot, and nothing re-based the proxy afterwards.

| Check | Evidence | Result |
|---|---|---|
| Window start | group-2 restart, `2026-11-16T15:20:47Z` (`docs/dev/group-2-restart.md`) | — |
| Window end | `2026-11-17T18:20:00Z` (analysis time, above the ≥ 24 h mark of `2026-11-17T15:20:47Z`) | — |
| Window length | 26.987 h | ≥ 24 h → met |
| Proxy process unchanged | PID `3720326`, `ps -o lstart` = `Mon Nov 16 15:20:47 2026`, alive at analysis time | — |
| Proxy restarts in window | 1 `Starting LLama Proxy Server` + 1 `Logging initialised pid=3720326` (`2026-11-16 15:20:49`) across every rotated segment covering the window — the window's own boot only | 0 re-bases |
| Log continuity | segments `proxy.log.2026-11-16_15` (15:20:48 → 21:20:47), `proxy.log-2026-11-17_00` (21:20:48 → 00:00:03), `proxy.log.2026-11-16_21` (00:00:05 → 03:20:46), `proxy.log.2026-11-17_03` (03:20:48 → 09:20:46), `proxy.log.2026-11-17_09` (09:20:52 → 15:20:51), `proxy.log` (15:21:01 → 18:19:50) | no gap |
| llama-server death / self-heal | PIDs `3122336` / `3122413` / `3122417` alive since `2026-11-15 01:06`; 0 `llama-server process died`, 0 `self-heal attempt`, 0 `watchdog router self-heal` lines in the window | none |
| TTS death | PID `3721608` up since `2026-11-16 15:20:50`; 0 TTS death lines | none |
| Config reload | 0 `Configuration reloaded` lines | none |
| Death-window artefacts | no `death_window_*.json` with mtime in the window | none |
| `/health` at analysis time | `status=healthy`, `ready=true`, `llama_server_running=true`, `tts_server_running=true` | — |

The dataset's own view of this agrees: `source.restarts_in_window` carries
exactly one entry, the window's own rotation line
(`Rotated previous proxy log aside: proxy.log.2026-11-16_152045`), and
`source.lines_in_window` is 405,856 across 42 log files.

## 2. Exact command and basis assertion (AC3/AC4)

The dataset was produced by this command, verbatim, from the repository root of
the implementation worktree:

```bash
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-16 15:20:47" --end "2026-11-17 18:20:00" \
  --deadline-seconds 180 \
  --llama-log "/var/log/llama-proxy/llama-server*.log" \
  --json --output docs/dev/prefill-deadline-residual-group2-24h.json
```

The `--llama-log` flag is what moves the abandoned-prefill window from the
**legacy token/rate estimate** to the **measured backend stop**. Asserted in
the committed dataset:

| Field | Value |
|---|---|
| `abandoned_prefill_windows.basis` | **`measured_stop`** (not `legacy_rate_estimate`) |
| `abandoned_prefill_windows.measured_available` | `true` |
| `abandoned_prefill_windows.coverage.available` | `true` |
| `attribution.abandoned_prefill_overlap.basis` | **`measured_stop`** |
| `release_cancel_accounting.measured_available` | `true` |
| `release_cancel_accounting.seconds_basis_tok_s` | `141.48` (derived only from Qwen3-attributed prompt-eval lines: 928 contributing of 7,048 total, 6,120 excluded as other ports) |

Per-row basis still degrades where the join could not bound a release
(177 of 193 `abandoned_prefill_windows.rows` are `legacy_rate_estimate`); each
such row is marked `estimated=true` and always carries `legacy_duration_s`, so
a measured row is never silently mixed with an estimate. Why so many rows fall
back, and why that does not move this window's conclusion, is §5.

## 3. Rates, with the honest split

### 3.1 Per-reason event rates (user vs `audit-*`)

193 post-acceptance evictions in 26.987 h = 7.152/h overall.

| Reason | Total | user | audit | total/h | user/h |
|---|---|---|---|---|---|
| `no_progress_watchdog` | 184 | 173 | 11 | 6.818 | 6.411 |
| `lease_evicted` | 3 | 1 | 2 | 0.111 | 0.037 |
| `measured_cache_deficit` | 3 | 3 | 0 | 0.111 | 0.111 |
| `idle_timeout` | 2 | 2 | 0 | 0.074 | 0.074 |
| `prefill_deadline` | 1 | 1 | 0 | 0.037 | 0.037 |
| **Combined (b2)** | **4** | **4** | **0** | — | **0.148** |

The (b2) composite is 1 + 3 = 4 user events over 26.987 h = **0.148/h**.

### 3.2 Stall-weighted cost (Σ`elapsed_s` / window hours)

A count cannot price the class: a 0.5 s detour and a 180 s stall are both +1 to
the event rate. User-class stall cost:

| Reason (user) | events | Σ `elapsed_s` | stall rate | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| `no_progress_watchdog` | 173 | 5,939.0 s | **220.069 s/h** | 20.0 s | 77.9 s | 170.3 s | 186.5 s |
| `prefill_deadline` | 1 | 180.1 s | 6.674 s/h | 180.1 s | 180.1 s | 180.1 s | 180.1 s |
| `measured_cache_deficit` | 3 | 4.6 s | 0.170 s/h | 0.5 s | 2.98 s | — | 3.6 s |
| `idle_timeout` | 2 | 87.2 s | 3.231 s/h | 43.6 s | 48.3 s | — | 49.5 s |
| `lease_evicted` | 1 | 11.1 s | 0.411 s/h | 11.1 s | 11.1 s | — | 11.1 s |
| **Combined (b2)** | **4** | **184.7 s** | **6.844 s/h** | — | — | — | — |

The b2 reasons' stall buckets split the two behaviours exactly as designed:
`prefill_deadline` is 1 of 1 at/near the deadline (180.1 s, none under 1 s),
`measured_cache_deficit` is 2 of 3 under 1 s. The audit class is negligible on
both reasons (0 b2 events); `no_progress_watchdog` audit is 11 events / 403.2 s
(14.941 s/h).

### 3.3 Attribution and coverage guards

| Figure | Value |
|---|---|
| Attributed `prefill_deadline` totals | occupancy **1**, capacity 0, predicted-warmth 0 |
| Session class | user `{occupancy: 1}` |
| Abandoned-window overlap, measured basis | 0 of 1 (0 %) — legacy basis also 0 of 1 |
| Zero-progress while draining (measured basis) | 2 of 193 evictions |

The single `prefill_deadline` event (2026-11-17 09:50:49,
`session=herdr-1794900481-910524-32268`) priced `estimated_tokens=43319`
against `new_tokens=228` (99.5 % of the prompt assumed cached), forecast
`eta_prefill_s=89.4`, stalled the full `elapsed_s=180.1` and was released on
slot 1. It is occupancy-attributed because another tenant held the pool during
its dispatch window. On a one-event base the occupancy share (100 %) exceeds
the 25 % guard arithmetically; that is a statement about the event, not a
population, and the verdict item should say so explicitly.

### 3.4 Local traffic, refusals and the new deadline-capacity bound

Descriptive context for the verdict/regression items (parent AC5e/AC6); no
verdict is claimed here.

| Figure | Value |
|---|---|
| Local streams consumed by the proxy | 4,112 over 26.987 h = **152.4/h** (post-change 24 h window: 2,946 = 122.8/h) |
| `local_admission_guard` decisions | 4,769 admit (`enforced=false`) / 3,219 refuse (`enforced=true`) |
| Refusals: `prefill_budget_exceeded` (25 k cold budget) | 2,738 |
| Refusals: **`prefill_eta_exceeds_deadline`** (group-2 deadline-capacity bound, `deadline_capacity_tokens=46153`) | **217** |
| Refusals: `eta_exceeds_budget_direct` | 249 |
| Refusals: `eta_exceeds_budget` | 15 |

## 4. Observation outside (b2): the `no_progress_watchdog` class

`no_progress_watchdog` is an `EVICTION_REASONS_AFTER_ACCEPTANCE` reason
(`proxy/proxy/admission_events.py`), so it is the same user-visible defect —
an accepted dispatch released before its first byte — but it is not one of the
two reasons the (b2) criterion names. In this window it is the dominant class:

- **184 events / 6.818/h** (173 user / 11 audit), versus 3 events / 0.125/h on
  the post-change 24 h window and 11 / 0.246/h on the 44.73 h window (both
  2026-11-06), and 60 / 2.5/h on the pre-change 2026-10-29 window.
- **6,342.2 s of stall (235.010 s/h)**; user p50 20.4 s, p90 77.08 s,
  p95 118.3 s, p99 169.2 s, max 186.5 s — a long-tailed queueing cost, not a
  detour.
- The events are spread across the whole window (3–15 per hour, hour one
  included), so this is not a late drift; 96 of the 184 priced the **whole**
  prompt as new compute (`new_tokens == estimated_tokens`), i.e. honest cold
  dispatches, and 20 carry `llama_task_id_reason=task_id_not_started` — the
  backend never began the task at all.
- Local traffic rose only 1.24× against the post-change 24 h window
  (152.4/h vs 122.8/h), which cannot explain a 55× increase in this class.

This is reported, not diagnosed: the mechanism (head-of-line blocking behind
the other slot's prefill vs a cold-ETA admission that cannot complete, versus
the group-2 pricing change admitting work that then queues) is a decision for
the verdict item. Per the parent's scope this document only fixes the numbers.
If the (b2) certification is read as "user-visible abandoned prefill work", this
class must be inside the criterion; as written, it is not.

## 5. Backend-log coverage caveat (why many rows are `legacy_rate_estimate`)

The AC-mandated glob `--llama-log "/var/log/llama-proxy/llama-server*.log"`
resolves to **16 files**: the live `llama-server.log` plus the 15 restart
rotations `llama-server.{1..15}.log` written between `2026-11-05` and
`2026-11-15`. Only `llama-server.log` intersects the window (the daily rotation
`llama-server.log-2026-11-17`, which holds 2026-11-16 and is renamed at
`2026-11-17T00:00`, does not match the `*.log` suffix). Consequences, recorded
in the dataset rather than hidden:

- `abandoned_prefill_windows.coverage.verdict = "mismatch"`: 7,886 parsed
  backend task segments against 193 proxy releases. `llama-server` lines carry
  no wall-clock timestamp, so the joiner pairs them in log order; a stream
  wider than the window cannot align, and rows that cannot be bounded stay
  `legacy_rate_estimate` (177 of 193 here).
- `backend_prefill_accounting` (`backend_completed_prefills=7048`,
  `proxy_consumed_local_streams=4112`, `abandoned_prefills=2936`) is a
  **stream-wide** cross-check, not a window figure, and must not be quoted as
  one. The tool's own note says the same.
- `release_cancel_accounting`: 122 of 193 releases have an observed backend
  cancel; 32 are unjoined (`cancel_without_task_id`); 42 (21.8 %) have no
  explained backend stop. Cancel→stop p50 0.0 s, p90 7.24 s, max 10.93 s
  (n = 82) — consistent with the "cancelled at the backend's next socket
  write" model of record.

Sensitivity check (not committed; run back-to-back with the glob command twice
for this document): pointing `--llama-log` at the two files that *do* cover the
window
(`/var/log/llama-proxy/llama-server.log-2026-11-17,/var/log/llama-proxy/llama-server.log`)
moves `seconds_basis_tok_s` 141.48 → 110.32, the measured_stop row count
16 → 7 and the zero-progress-while-draining count 2 → 1, but changes **no**
count, rate, stall figure, attribution or guard in §0–§3: evictions stay 193
with the same per-reason split, the b2 composite stays 4 events / 0.148/h,
`prefill_deadline` stays 1 event / 180.1 s, `measured_cache_deficit` stays
3 events / 4.6 s and the abandoned-window overlap stays 0 of 1. The
conclusion is insensitive to the basis width even though the basis itself is
weak; only the join-derived fields (seconds basis, per-row basis, the
draining-overlap count) move, and those are the fields §2/§5 already label as
join-dependent.

## 6. Comparison with prior windows

| Window | Hours | user `prefill_deadline` | user `measured_cache_deficit` | combined /h | `no_progress_watchdog` |
|---|---|---|---|---|---|
| Pre-change `2026-10-29` 24 h (`prefill-deadline-residual-2026-10-29.json`) | 24.00 | 47 | 94 | 5.875 | 60 (2.500/h) |
| Pre-change matched `2026-10-30` (`…-prechange-matched-2026-10-30.json`) | 24.00 | 19 | 428 | 18.625 | 22 (0.917/h) |
| Post-change 24 h `2026-11-06` (`…-postchange-24h-2026-11-06.json`) | 24.00 | 26 | 34 | 2.500 | 3 (0.125/h) |
| Post-change matched `2026-11-06` (`…-postchange-matched-2026-11-06.json`) | 24.00 | 28 | 34 | 2.583 | 3 (0.125/h) |
| Post-change 44.73 h `2026-11-06` (`…-postchange-full-2026-11-06.json`) | 44.73 | 47 | 35 | 1.833 | 11 (0.246/h) |
| Group-1 window `2026-11-10 14:18:26 → 2026-11-11 15:18:26` (`docs/dev/stall-weighted-prefill-rate.md`) | 25.00 | 4 (744.2 s, 29.8 s/h) | 14 (16.9 s, 0.7 s/h) | 0.720 | not recorded |
| **Group-2 window (this document)** | **26.99** | **1 (180.1 s, 6.674 s/h)** | **3 (4.6 s, 0.170 s/h)** | **0.148** | **184 (6.818/h)** |

Within `prefill_deadline` + `measured_cache_deficit` the group-2 window is the
lowest measured so far on both the event rate (0.148/h vs 0.720/h on the
group-1 window) and the stall cost (6.844 s/h vs 30.5 s/h implied by the
group-1 window's 744.2 + 16.9 s over 25 h). The `no_progress_watchdog` column
moves the other way and is the reason this window is not a clean improvement
story.

## 7. Limitations

- **One event carries the `prefill_deadline` guard.** 1 event makes the
  occupancy share 100 % and the capacity share 0 %; both are arithmetic facts
  about a sample of one, not estimates of a population.
- **The basis is a log stream wider than the window** (§5). Rows that could not
  be bounded by an observed stop are marked `estimated=true`; the window's
  counts, rates and attribution are unaffected by the sensitivity check.
- **Workload is not held constant.** Traffic here is 152.4 local streams/h
  against 122.8/h on the 2026-11-06 24 h window, and the session mix
  (`herdr-*` workers, `audit-*` runs) differs. The comparison in §6 is a
  like-for-like *rate*, not a controlled experiment.
- **Log retention.** Proxy logs are kept ~7 days, so windows before
  `2026-11-09` exist only as their committed datasets; §6 quotes them from the
  recorded payloads, not from a re-run.
- **`no_progress_watchdog` is not priced by (b2).** §4 reports it because it is
  the dominant abandoned-work class in this window; folding it into the
  criterion (or not) is a producer decision.

## 8. Reproduction

```bash
# the committed dataset (identical to docs/dev/prefill-deadline-residual-group2-24h.json)
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-16 15:20:47" --end "2026-11-17 18:20:00" \
  --deadline-seconds 180 \
  --llama-log "/var/log/llama-proxy/llama-server*.log" \
  --json --output /tmp/group2-24h.json
```

The proxy log segments covering the window are listed in §1; the export
retains them until the 7-day rotation drops them, after which this document and
the committed dataset are the record.
