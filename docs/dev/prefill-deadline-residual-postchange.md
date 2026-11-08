# Prefill-deadline residual: post-change ≥ 24 h measurement (abandoned-drain fix)

**Work item:** LP-0MVW8S8OZ00449YR (AC3/AC4 evidence for the epic
LP-0MVVEQW5Y00962T2).
**Measurement tool:** `proxy/scripts/analyse_prefill_deadline_residual.py`
(evictions, reasons, attribution, abandoned-prefill overlap) and
`scripts/measure_warm_path_window.py` (local completions, large-prompt warm
service latency).
**Datasets (committed, this document is traceable to them):**

| Dataset | Window |
|---|---|
| `prefill-deadline-residual-2026-10-29.json` (**pre-existing**, the documented pre-change baseline) | `2026-10-29 14:08:10` → `2026-10-30 14:08:10` |
| `prefill-deadline-residual-postchange-24h-2026-11-06.json` | `2026-11-06 14:15:56` → `2026-11-07 14:15:56` |
| `prefill-deadline-residual-postchange-full-2026-11-06.json` | `2026-11-06 14:15:56` → `2026-11-08 11:00:00` (44.73 h) |
| `prefill-deadline-residual-prechange-matched-2026-10-30.json` | `2026-10-30 16:35:14` → `2026-10-31 16:35:14` |
| `prefill-deadline-residual-postchange-matched-2026-11-06.json` | `2026-11-06 16:35:14` → `2026-11-07 16:35:14` |
| `prefill-deadline-residual-warm-path-2026-11-06.json` | warm-path metrics for all four of the above |

This is an **evidence** document. It reports a measurement; it changes no
admission decision and no threshold, and it does not re-tune anything to make
the numbers land.

## 0. Result at a glance

Certified criterion (producer decision 2026-11-02, option **b2**): user-class
`local_eviction_after_acceptance reason=prefill_deadline` +
`reason=measured_cache_deficit` ≤ **0.5 events/h** over a ≥ 24 h restart-free
window, not certified if occupancy-attributed ≥ 25 % or capacity-attributed
≥ 50 % of `prefill_deadline` events.

| Test | Pre-change | Post-change (24 h) | Post-change (44.73 h) | Verdict |
|---|---|---|---|---|
| **User-class combined rate** | 5.875/h (141 events) | **2.500/h** (60) | **1.833/h** (82) | **NOT certified** — 5.0× / 3.7× the 0.5/h gate |
| occupancy guard (< 25 %) | 15.0 % (9/60) | 22.2 % (8/36) | 13.4 % (9/67) | guard passes |
| capacity guard (< 50 %) | 13.3 % (8/60) | 16.7 % (6/36) | 34.3 % (23/67) | guard passes |
| abandoned-window share, **measured** basis (≤ 20 % target) | 3.3 % (2/60) | **2.8 % (1/36)** | **1.5 % (1/67)** | target met |
| abandoned-window share, **legacy** basis | 96.7 % (58/60) | 75.0 % (27/36) | 68.7 % (46/67) | target **not** met on this basis |
| zero-progress starvations while draining | not measured (pre-capture) | **1 of 80** | **1 of 122** | target zero → **missed by 1**, listed in §5 |
| warm-path / throughput (AC5) | no regression on p50, throughput and < 10 s share; **p90 rises in absolute terms**, explained in §6 | | | see §6 |

Reading it in one line: the mechanism this epic fixed is now **rare on the
measured basis** (2.8 % → 1.5 % of `prefill_deadline` events start inside an
abandoned-prefill window) and throughput rose sharply, but the **rate
criterion still fails by 3.7–5×** because what remains is predicted-warmth and
capacity pricing, not abandoned-drain starvation.

## 1. Windows, and proof the post-change window ran the fixed build (AC1)

The window anchor is a **restart**, not a commit date, and it must be
restart-free across the proxy **and** the backend.

| Fact | Value | Evidence |
|---|---|---|
| Fix commits | `9f3b9c06` (starvation fix on the accepted-dispatch path, 2026-11-02T09:12:09Z) and `eac04d96` (contract pin) | `git log -1` |
| Deployed tree | `581902994`; `git merge-base --is-ancestor 9f3b9c06 HEAD` → **YES**, `eac04d96` → **YES** | `git` |
| Re-base/rotation event | **`2026-11-06T14:15:52Z`** — `Rotated previous proxy log aside: proxy.log.2026-11-06_141548 (pre-restart window preserved)` | `proxy.log.2026-11-06_14` first line |
| Restart kind | controlled: both router instances `exited with status 0`; `Killed` / `exit_code_137` / `process died` occurrences in the backend log since the rotation: **0** | `llama-server.1.log` tail; `grep -c` over `llama-server.log` |
| Proxy process | pid `2634864`, started `2026-11-06T14:15:50Z`, cwd `/home/rgardler/projects/llm/proxy` | `ps -o lstart`, `/proc/2634864/cwd` |
| Backend router | pid `2634942` at `14:15:53Z`; model child pid `2635023` at `14:15:56Z` | `ps` |
| **Verified window start** | **`2026-11-06T14:15:56Z`** (newest of the three starts) | `ps` |
| Restart-free span at analysis end | `2026-11-06T14:15:56Z` → `2026-11-08T11:00:00Z` = **44.73 h** | `ps -o etimes` = 161 591 s |
| No restart inside the span | `Starting LLama Proxy Server` / `self-heal attempt` / `process died, restarting` / `exit_code_137` = **0** in every segment from `proxy.log.2026-11-06_20` through `proxy.log` | per-file `zgrep -c` |
| Health at analysis time | `status=healthy`, `llama_server_running=true`, `backend_reachable=true`, `self_healing_in_progress=false` | `curl -s localhost:8000/health` |

The measurement keeps **two** post-change windows:

- **24 h (`2026-11-06 14:15:56` → `2026-11-07 14:15:56`)** — the primary
  comparison window. It is diurnally matched to the pre-change baseline
  (`14:08:10` → `14:08:10` seven days earlier: same time of day, same 24 h
  shape) and satisfies the ≥ 24 h requirement exactly.
- **44.73 h (`2026-11-06 14:15:56` → `2026-11-08 11:00:00`)** — the whole
  restart-free span available at analysis time, reported because a single 24 h
  window on this host is a small sample (36 and 67 `prefill_deadline` events).

Config live in the post-change window (`proxy/config.yaml`, `server:`):
`local_prefill_deadline_seconds=180`, `local_cold_prefill_budget_tokens=25000`,
`local_measured_cache_abort_enabled=True`,
`local_measured_cache_abort_bound_tokens=2000`,
`local_measured_cache_abort_min_deficit_ratio=0.25`, slot pool = 2
(`status_request … total_slots=2` on 19 303 polls). **The prefill deadline is
180 s here, not the 120 s of the pre-change baseline** — see §7.

## 2. Reproduction (AC6)

Both analyser runs use the same binary and the same `--llama-log` basis as the
pre-change baseline; `--llama-log` is mandatory, without it the tool silently
degrades to `basis=legacy_rate_estimate` and the measured basis disappears.

```bash
# post-change, 24 h (primary)
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-06 14:15:56" --end "2026-11-07 14:15:56" \
  --llama-log "/var/log/llama-proxy/llama-server.log" --json \
  --output docs/dev/prefill-deadline-residual-postchange-24h-2026-11-06.json

# post-change, whole restart-free span
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-06 14:15:56" --end "2026-11-08 11:00:00" \
  --llama-log "/var/log/llama-proxy/llama-server.log" --json \
  --output docs/dev/prefill-deadline-residual-postchange-full-2026-11-06.json

# pre-change comparison pair (earliest 24 h whose raw segments survive retention)
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-10-30 16:35:14" --end "2026-10-31 16:35:14" \
  --llama-log "/var/log/llama-proxy/llama-server.log-2026-10-31,/var/log/llama-proxy/llama-server.log-2026-11-01" \
  --json --output docs/dev/prefill-deadline-residual-prechange-matched-2026-10-30.json

# post-change, same clock times as the pre-change pair
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-06 16:35:14" --end "2026-11-07 16:35:14" \
  --llama-log "/var/log/llama-proxy/llama-server.log" --json \
  --output docs/dev/prefill-deadline-residual-postchange-matched-2026-11-06.json

# AC5 warm path / throughput, all four windows
python3 scripts/measure_warm_path_window.py \
  --since "2026-10-30 16:35:14" --until "2026-10-31 16:35:14" --label prechange_matched --json
python3 scripts/measure_warm_path_window.py \
  --since "2026-11-06 16:35:14" --until "2026-11-07 16:35:14" --label postchange_matched --json
python3 scripts/measure_warm_path_window.py \
  --since "2026-11-06 14:15:56" --until "2026-11-07 14:15:56" --label postchange_24h --json
python3 scripts/measure_warm_path_window.py \
  --since "2026-11-06 14:15:56" --until "2026-11-08 11:00:00" --label postchange_full --json
```

Log-source accounting (from each dataset's `source` block): 62 proxy log files
read per window, 273 086 lines in the 24 h window (51 duplicate rotation lines
dropped), 415 090 in the 44.73 h window (85 dropped). The pre-change baseline
window read 52 files / 353 286 lines when it was measured on 2026-10-30; its
raw segments have since rotated away (§7).

## 3. Evictions per reason, and the reason conversion (AC4)

Eviction-scoped counts (`local_eviction_after_acceptance` records, not raw
`reason=` occurrences — the same string also appears on the
`measured_cache_abort` line of one event, so a raw grep triple-counts):

| reason | pre-change baseline (24 h) | post 24 h | post 44.73 h |
|---|---|---|---|
| `prefill_deadline` | 60 (2.500/h) | **36 (1.500/h)** | **67 (1.498/h)** |
| ├ user class | 47 (1.958/h) | 26 (1.083/h) | 47 (1.051/h) |
| └ audit class | 13 | 10 | 20 |
| `measured_cache_deficit` | 248 (10.333/h) | **40 (1.667/h)** | **41 (0.917/h)** |
| ├ user class | 94 (3.917/h) | 34 (1.417/h) | 35 (0.782/h) |
| └ audit class | 154 | 6 | 6 |
| `no_progress_watchdog` | 60 (2.500/h) | 3 (0.125/h) | 11 (0.246/h) |
| `lease_evicted` | 7 (0.292/h) | 1 (0.042/h) | 3 (0.067/h) |
| `idle_timeout` | 0 | 0 | 0 |
| **all post-acceptance evictions** | 375 (15.625/h) | 80 (3.333/h) | 122 (2.727/h) |
| **user-class combined (the certified metric)** | **141 → 5.875/h** | **60 → 2.500/h** | **82 → 1.833/h** |

So the certified rate **fell by 57 % (24 h) / 69 % (44.73 h)** and still misses
the ≤ 0.5/h gate by 5.0× / 3.7×. `measured_cache_deficit` releases fell by 84 %
(24 h) and 91 % (44.73 h) in absolute terms; because both reasons are counted
together, that fall cannot masquerade as a residual reduction (the AC4
direction) — the two reasons are reported side by side above and the *sum* is
the number the criterion reads.

Distribution of the 36 post-change `prefill_deadline` events (same columns as
the pre-change table in
[prefill-deadline-residual.md](prefill-deadline-residual.md) §5):

| Column | pre-change baseline (n=60) p10 / p50 / p90 | post 24 h (n=36) | post 44.73 h (n=67) |
|---|---|---|---|
| `estimated_tokens` | 20 111 / 78 213 / 113 555 | 40 728 / 65 811 / 88 924 | 42 657 / 66 388 / 85 333 |
| `new_tokens` (priced compute) | 254 / 2 348 / 16 521 | 308 / 930 / 9 756 | 323 / 2 417 / 20 136 |
| `eta_prefill_s` | 9.5 / 46.4 / 76.1 | 6.4 / 77.9 / 152.0 | 7.5 / 121.7 / 163.3 |
| `elapsed_s` | 120.0 / 120.2 / 124.6 | 180.1 / 180.1 / 180.9 | 180.0 / 180.1 / 180.8 |
| priced `cached_ratio` | 0.0 / 0.946 / 0.996 | 0.825 / 0.990 / 0.996 | 0.690 / 0.964 / 0.995 |

The shape is unchanged and **more extreme on the warmth prediction**: the median
event is a 66 k-token prompt admitted at 930 priced new tokens (99 % declared
cached) which then produced no first byte for the whole 180 s lease.

## 4. Abandoned-prefill overlap share, both bases (AC2)

The intake assumption set the target at ≤ 20 % of `prefill_deadline` events
starting inside an abandoned-prefill window. Both bases are reported, because
they answer different questions and the earlier work established that the
**legacy** basis is an estimate, not an observation (see
[abandoned-prefill-release-cancel.md](abandoned-prefill-release-cancel.md)):
legacy bounds the abandoned window at `tokens ÷ 43.1 tok/s`; the measured basis
bounds it at the **observed** backend stop (`basis=measured_stop`).

| Window | legacy basis | measured basis |
|---|---|---|
| pre-change baseline `10-29 14:08` → `10-30 14:08` | **58/60 = 96.7 %** | **2/60 = 3.3 %** (re-run recorded in [prefill-deadline-residual.md](prefill-deadline-residual.md) §6) |
| post 24 h `11-06 14:15:56` → `11-07 14:15:56` | **27/36 = 75.0 %** | **1/36 = 2.8 %** |
| post 44.73 h `11-06 14:15:56` → `11-08 11:00` | **46/67 = 68.7 %** | **1/67 = 1.5 %** |
| pre-change pair `10-30 16:35` → `10-31 16:35` | 22/24 = 91.7 % | not measurable (no closed stop accounting in that window) |
| post-change pair `11-06 16:35` → `11-07 16:35` | 28/39 = 71.8 % | 1/39 = 2.6 % |

**Both readings are reported, neither replaces the other.** On the measured
basis the share falls **3.3 % → 2.8 % (24 h) / 1.5 % (44.73 h)** — the ≤ 20 %
target is met with a wide margin. On the legacy estimate the share falls
**96.7 % → 75.0 % / 68.7 %** — improved, but still far above 20 %, because the
legacy estimator still assumes the backend keeps grinding on the abandoned
prompt for hundreds of seconds, which the
[release→cancel measurement](abandoned-prefill-release-cancel.md) refutes
(p90 cancel→stop 6.6 s; re-measured in this window as p90 ≈ 9.9 s, mean 4.2 s,
n = 19 joined releases in the 24 h window).

## 5. Accepted dispatches that expired with no measured progress (AC3, second half)

Target: **zero** such dispatches. Occurrences are listed individually so they
can be triaged.

The analyser reports one such event in each post-change window. The two flags
are produced by the per-window log-order join (the backend log carries no
wall-clock timestamps, so releases are joined to backend task segments in log
order, and the abandoned window bounds therefore shift slightly between
windows). The raw records behind both flags are:

| Time (UTC) | Session | Reason | Prompt / priced new tokens | `elapsed_s` | Overlap residual | `slot_id` |
|---|---|---|---|---|---|---|
| `2026-11-06 19:42:25` | `herdr-1793993268-934013-4929` | `prefill_deadline` | 71 071 / 2 538 | 180.7 | 3.7 s from that session's own `measured_cache_deficit` release at `19:39:19` | `unknown` |
| `2026-11-07 11:21:40` | `herdr-1794049759-1173564-12330` | `prefill_deadline` | 79 597 / 489 | 180.8 | 7.0 s from that session's own `prefill_deadline` release at `11:15:19` | `unknown` |

Attribution for both is **predicted-warmth** (`backend_progress_observed=false`,
priced ratio ≤ 10 % of the prompt), not occupancy: the abandoned residual is a
single ubatch (1 024 tokens), so "queued behind its own drain" is nominal here —
the dispatch then produced nothing for the full 180 s on a ~80 k prompt whose
own admission ETA was ≈ 166 s. `slot_id=unknown` is the documented
post-F4 production limitation (no slot id on eviction lines), recorded rather
than guessed.

Reproduce the raw lines:

```bash
zcat -f /var/log/llama-proxy/proxy.log* \
  | grep local_eviction_after_acceptance \
  | grep -E "session=herdr-1793993268-934013-4929|session=herdr-1794049759-1173564-12330" \
  | grep -E "reason=(prefill_deadline|measured_cache_deficit)"
```

Per-event attribution of the 36 `prefill_deadline` events in the 24 h window:

| Window | occupancy | capacity | predicted-warmth |
|---|---|---|---|
| pre-change baseline | 9/60 = **15.0 %** | 8/60 = 13.3 % | 43/60 = 71.7 % |
| post 24 h | 8/36 = **22.2 %** | 6/36 = 16.7 % | 22/36 = 61.1 % |
| post 44.73 h | 9/67 = **13.4 %** | 23/67 = 34.3 % | 35/67 = 52.2 % |

Occupancy stays under the 25 % guard in both windows (22.2 %, 13.4 %); capacity
stays under the 50 % guard (16.7 %, 34.3 %). Capacity roughly doubles on the
longer window — worth watching, but it does not trip the guard.

## 6. No warm-path or throughput regression, with deltas (AC5)

Directions AC4 requires: local completions per hour **up or flat**, large-warm
completion latency **flat or down**, `measured_cache_deficit` releases per hour
**down**.

Both windows of each pair start at the same clock time, one week apart:

- **pre-change pair window:** `2026-10-30 16:35:14` → `2026-10-31 16:35:14`
  (the earliest 24 h pre-change window whose raw log segments survive
  retention; see §7).
- **post-change pair window:** `2026-11-06 16:35:14` → `2026-11-07 16:35:14`
  (inside the restart-free span).

Metric definitions live in `scripts/measure_warm_path_window.py`:
*large* = prompt estimate ≥ 20 000 tokens; *warm* = the proxy's own
`cache_state=warm` label on the dispatch line (`restored` is a distinct cache
path and is reported separately, not folded in); *service latency* =
`dispatch_to_first_byte_ms` (first byte **minus** slot queue wait); *fast* =
first byte under 10 000 ms.

| Metric (24 h) | pre-change pair | post-change pair | Δ | direction AC4 needs |
|---|---|---|---|---|
| local completions | 617 (**25.708/h**) | 2 751 (**114.625/h**) | **+88.917/h (+346 %)** | up → **met** |
| large (≥ 20 k) local completions | 255 (10.625/h) | 1 795 (74.792/h) | +64.167/h (+604 %) | up → met |
| large-warm dispatches | 178 | 1 249 | +1 071 | context only |
| large-warm service **p50** | 4 585.1 ms | **3 967.5 ms** | **−617.6 ms (−13.5 %)** | down → **met** |
| large-warm service **p90** | 27 819.5 ms | **38 863.0 ms** | **+11 043.5 ms (+39.7 %)** | down → **NOT met in absolute terms** — see below |
| large-warm first byte < 10 s | 126/178 = **70.79 %** | 922/1 249 = **73.82 %** | +3.03 pp | up → met |
| `measured_cache_deficit` | 477 (**19.875/h**) | 40 (**1.667/h**) | **−18.208/h (−91.6 %)** | down → **met** |
| control: large-**cold** service p50 | 32 258.5 ms | 55 231.9 ms | +22 973.4 ms (+71.2 %) | — |
| control: large-cold service p90 | 99 414.5 ms | 101 660.5 ms | +2 246.0 ms (+2.3 %) | — |
| warm-to-cold p50 ratio (load-normalised) | 0.142 | **0.072** | −0.070 | — |

**The p90 row is reported as adverse, not explained away.** Three facts bound
what it means:

1. **Load rose 4.5×.** Local completions went from 25.7/h to 114.6/h and local
   dispatches from 623 to 2 754 in the two windows. On a 2-slot pool, a
   4.5× higher arrival rate raises the tail of every latency distribution; the
   **cold** path — same GPU, same instrument, no change in this epic — moved
   **+71 % on p50** over the same pair, while the warm p50 *fell*.
2. **Load-normalised, the warm path improved**: the warm-to-cold p50 ratio fell
   0.142 → 0.072, and the share of large-warm dispatches reaching a first byte
   under 10 s rose 70.8 % → 73.8 %.
3. The three metrics that directly test the fix — throughput, warm p50, and the
   `< 10 s` share — all moved in the required direction, and the 24 h primary
   window agrees (large-warm p50 3 967.5 ms, p90 35 952.0 ms, share 74.5 %,
   122.3 local completions/h).

So AC5's deltas are produced and the required direction is met on throughput,
warm p50, the fast share and the deficit-releases-per-hour; the **absolute warm
p90 delta is adverse** and is reported as such. A like-for-like p90 verdict
would need a load-matched comparison window, which the surviving logs do not
contain (§7) — recorded as an open point for the closure verdict rather than
asserted either way.

## 7. Limitations and confounders

- **The prefill deadline changed from 120 s to 180 s** (`31c842157`,
  2026-11-04, LP-0MW1V0LPM000QLUC) between the pre-change baseline and the
  post-change window. A 50 % longer deadline gives every dispatch 50 % more
  time to complete before it is evicted, so the `prefill_deadline` rate fall is
  **partly a policy effect**, and `elapsed_s` is not comparable across the
  change (120.1 s pre, 180.1 s post). The certified criterion counts events, so
  this does not invalidate the rate, but it must be read with it.
- **The pre-change raw logs have rotated away.** `/var/log/llama-proxy`'s
  oldest retained segment is `proxy.log.2026-10-30_16.gz`; a re-run of the
  analyser over the documented baseline window reads 62 files and finds **0**
  lines in window. The baseline column is therefore the **committed dataset**
  `prefill-deadline-residual-2026-10-29.json` (and the measured-basis overlap
  re-run recorded in [prefill-deadline-residual.md](prefill-deadline-residual.md)),
  not a fresh run. AC5's warm-path comparison uses a *different* pre-change
  window (2026-10-30/31) precisely because that is the earliest 24 h with
  surviving segments.
- **The pre-change pair window is a different regime**: 3 proxy restarts inside
  it (`2026-10-31 00:37:53`, `00:41:57`, `14:36:12`), a 120 s deadline,
  527 evictions of which 477 were `measured_cache_deficit`, and a
  capacity-bound attribution profile (58 % capacity vs 13 % pre-change-baseline,
  23 % post). It is usable for latency direction, not for a controlled
  before/after rate comparison — which is why the certified-criterion rates in
  §0/§3 are taken against the documented baseline instead.
- **Backend prompt-eval accounting is unusable for these windows.** The
  analyser's `backend_prefill_accounting` returns a negative
  `abandoned_prefills` count (`-1534` for the 24 h window) because the backend
  log's task segments do not align with the proxy window (the tool says so
  itself: valid only when the backend log covers exactly the window). No
  figure in this document depends on it; the release→cancel latency
  (p90 ≈ 9.9 s, n = 19 joined releases) is reported instead.
- **Abandoned-window coverage verdict is `mismatch`** for these windows
  (1 446 backend task segments vs 80 proxy releases). That is expected — the
  backend log covers far more tasks than the proxy released — and it is the
  same caveat the earlier windows carried. The share, the rates and the
  attribution do not depend on the join; the overlap *column* is indicative and
  is reported on both bases.
- **`slot_id=unknown`** on every post-F4 eviction line (production limitation),
  so the per-event lists in §5 name the session and the time, not the slot.
- **User/audit mix**: "user class" means every session id that does not start
  `audit-`; on this host that is agent traffic (herdr/pi workers), not
  human-facing traffic. The certified rate is therefore gated on agent-session
  evictions, as in the pre-change baseline.

## 8. What this means for the closure verdict (evidence only, no verdict)

- The **mechanism** the epic fixed is measured as rare: 1.5–2.8 % of residual
  deadline events start inside an abandoned-prefill window on the observed
  basis, against a 3.3 % baseline. On the legacy estimated basis it is 68.7–75.0 %
  against 96.7 % — better, still dominant, and still an estimate.
- The **certified rate** is not met: 2.500/h (24 h) and 1.833/h (44.73 h) user
  class against ≤ 0.5/h, with both guards passing. The residual is
  predicted-warmth-dominated (61.1 % / 52.2 %) with capacity second
  (16.7 % / 34.3 %) and occupancy a minority (22.2 % / 13.4 %).
- **Throughput and the warm p50 improved**; the warm **p90** rose in absolute
  terms under a 4.5× load increase and improved relative to the cold path.
- The producer's decision this evidence feeds — hold the threshold, re-scope it,
  or re-scope to shape-only — is **not** taken here. Nothing was re-tuned.
