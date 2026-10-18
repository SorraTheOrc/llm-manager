# Fast<->cheap proxy mode-flip frequency and how to reduce it

**Work item:** LP-0MV9UNE75004903J
**Author:** Map (pi-agent)
**Analysis window:** 2026-10-09 06:06 -> 2026-10-16 07:00 UTC
(≥7 days, continuous rotated logs; pinned via ``--from/--to`` so the rotating
live ``proxy.log`` tail cannot change the numbers)
**Predecessor:** LP-0MV9UNKVK007TN7E (restart frequency by cause) — this item adds the
per-trigger-source split, the intra-day distribution, flip-flop quantification, and
the reduction options.
**Reproduce:** `python3 scripts/analyze_mode_flips.py --from "2026-10-09 06:06" --to "2026-10-16 07:00"` (see [Method](#method-and-reproducibility)).

## Verdict (TL;DR)

1. **Flip rate: ~23/day, structurally driven, not anomalous.** 163 real
   fast<->cheap flips in ≥7 days; the 7 full days (10-09..10-15) average
   **22.9 flips/day** (std 8.1, range 10-32). The studied 24 h period (31 flips)
   sits 1.0sigma above the mean — **within normal variation**.
2. **Trigger split is ~evenly two-sided and activity-driven:** over the window
   **to-cheap 82 / to-fast 76 / schedule 5**. Almost every cheap switch is followed
   by a fast switch when the operator resumes, so flips come in pairs. The
   ContextHub/herdr `mode-switch-worker` drives the large majority; the proxy's
   own time schedule contributes only the 10:00 fast boundary (5 flips); the
   01:00 cheap boundary **never fires** because the idle-cheap worker already
   reached cheap overnight.
3. **The rate is cooldown-bounded by design.** Inter-flip gap median **32 min**
   (~ the 30-min `MODE_SWITCH_COOLDOWN_SECONDS`), zero genuine breaches. This is
   the flapping fix (LP-0MU5BOK3P0090O0V) working.
4. **The real waste is short cheap excursions:** **21 of 81 (26%) cheap periods
   lasted < 5 min** (minimum 7 s) — two full restarts for seconds of cheap mode.
   Each restart discards the warm KV cache and raises the local->remote fallback
   rate ~1.6x.
5. **Recommendation: act.** Primary fix is the *non-destructive* switch (already
   tracked by **LP-0MV9UNGEB009XUZ5**) which removes the per-flip cost; add
   *hysteresis* — raise the fast->cheap cooldown 30 -> 60 min and the idle
   threshold 15 -> 30 min (**LP-0MVAV5J6Y008K5EM**, created). Target:
   **<= 15 flips/day** and **<= 2 sub-5-min cheap excursions/day**.

## Method and reproducibility

A flip is a real fast<->cheap transition. The proxy skips redundant switches
(`set_mode` returns a no-op when the persisted mode already matches), so **every
`Mode-switch restart spawned` == one real transition**. The active mode at each
startup is read from the `Grandfathering: enabled; other-mode config
config-<other>.yaml (current=<mode>)` line that follows the `Starting LLama
Proxy Server` marker; a flip is a change of startup mode between consecutive
restarts.

Trigger attribution (`scripts/analyze_mode_flips.py`):

- **schedule** — a `Mode scheduler: applied scheduled mode <mode>` marker within
  +/-10 min of the flip (the built-in `01:00 cheap` / `10:00 fast` schedule in
  `proxy/mode_schedule.yaml`).
- **to-fast** — any other flip INTO fast: the ContextHub/herdr worker's
  fast-on-command POST, or a manual `POST /admin/set-mode {"mode":"fast"}`.
- **to-cheap** — any other flip INTO cheap: the worker's leader-only idle POST,
  or a manual switch.

> **Attribution (since LP-0MVAPNC2900145RH).** The proxy now logs the
> `/admin/set-mode` caller: every `Mode-switch restart spawned` line carries
> `source=scheduler|worker|api|manual`, `actor`, requested `mode` and an
> optional `reason`, and `start-proxy.sh --restart` echoes its reason into
> the new process's startup log. Worker-vs-manual is therefore separable
> from the proxy log alone on the new logging path, and the mode-flip
> classifier (`analyze_mode_flips.py`) can drop its heuristic fallback for
> marker-carrying lines.

The classifier uses the shared `scripts/lib/proxy_logs.py` helper, so both
rotation schemes (`proxy.log.*` and `proxy.log-*`, plain and `.gz`) are read.
Its pure parsing/classification functions are unit-tested in
`tests/test_mode_flip_analysis.py` (11 tests).

Reproducible commands (the `zgrep` variants are shown for manual verification;
the script is authoritative):

```bash
cd /var/log/llama-proxy

# 1. Total restarts (process starts) in the retained window
zgrep -h "Starting LLama Proxy Server" proxy.log* | wc -l

# 2. Mode-switch restarts
zgrep -h "Mode-switch restart spawned" proxy.log* | wc -l

# 3. Scheduled transitions actually applied (only 10:00 fast ever fires)
zgrep -h "Mode scheduler: applied scheduled mode" proxy.log* | wc -l  # -> 5

# 4. Active startup mode for every start (direction of each flip)
zgrep -h "other-mode config" proxy.log* | wc -l

# 5. One-pass classifier (per-day, per-trigger, intra-day, flip-flop, dwells).
#    --from/--to pin the window because the live proxy.log file rotates and is
#    pruned while the analysis runs; the rotated .gz files are stable.
python3 scripts/analyze_mode_flips.py --from "2026-10-09 06:06" --to "2026-10-16 07:00"
```

> **Reproducibility caveat.** The in-process log handler rotates and prunes the
> live `proxy.log` while the analysis runs (observed during this investigation:
> ~8 restarts of the 10-16 morning tail disappeared from the file set mid-run).
> The `--from/--to` window confines the counts to the stable rotated files, so
> re-running reproduces the same numbers.

## Per-day flip counts, by trigger source

Whole window (2026-10-09 06:06 -> 2026-10-16 07:00), 7 full days plus a partial day:

| Date (UTC) | Flips | schedule | to-fast (ContextHub/manual) | to-cheap (idle/manual) |
|---|---:|---:|---:|---:|
| 2026-10-09 | 20 | 0 | 10 | 10 |
| 2026-10-10 | 16 | 1 | 7 | 8 |
| 2026-10-11 | 10 | 1 | 4 | 5 |
| 2026-10-12 | 32 | 0 | 16 | 16 |
| 2026-10-13 | 24 | 1 | 11 | 12 |
| 2026-10-14 | 31 | 1 | 14 | 16 |
| 2026-10-15 | 27 | 1 | 13 | 13 |
| 2026-10-16 (partial, to 07:00) | 3 | 0 | 1 | 2 |
| **TOTAL** | **163** | **5** | **76** | **82** |

The two-sided **to-fast / to-cheap pair per cycle** is the activity oscillation:
the leader switches to cheap after 15 min idle; the operator's next command flips
it back to fast.

### Intra-day distribution (all flips, UTC)

```
00:00  18 ##################
01:00   1 #
02:00   1 #
03:00   0
04:00   0
05:00   0
06:00   0
07:00   1 #
08:00   3 ###
09:00   7 #######
10:00  13 #############     <- 10:00 scheduled fast boundary + morning activity
11:00  10 ##########
12:00   5 #####
13:00   8 ########
14:00   7 #######
15:00  11 ###########
16:00  11 ###########
17:00   8 ########
18:00   6 ######
19:00   8 ########
20:00  10 ##########
21:00  12 ############
22:00  10 ##########
23:00  13 #############
```

Flips track operator/agent activity: quiet 03:00-07:00 (0-1/h), busy 08:00-24:00
(3-13/h) with the 10:00 spike where the schedule coincides with the workday.

## Was the studied 24 h period unusual?

The item's premise was "30 restarts, of which 15 mode-switch". Reproducible
counts **contradict** that: over 2026-10-14 14:02 -> 2026-10-15 14:02 UTC there
were **32 process starts and 31 flips** (to-fast 15 / to-cheap 15 / schedule 1)
— essentially every restart was a mode switch, ~2x the original figure.

Against the >=7-day baseline:

| Metric | Studied 24 h | Baseline (7 full days) |
|---|---:|---|
| Flips | 31 | 22.9/day mean, 8.1 std, 10-32 range |
| Flips / day | 31 | 22.9 |
| z-score | +1.0sigma | — |
| Trigger mix | to-fast 15 / to-cheap 15 / schedule 1 | to-fast 76 / to-cheap 82 / schedule 5 |

**Verdict: above-average but not unusual** (+1sigma, inside the observed 10-32
range). There is no runaway or crash loop; the 30-min cooldown bounds the cadence
(inter-flip gap median 32 min, zero breaches).

## Trigger-source detail (Q1) and flip-flop quantification (Q2)

**Schedule.** Only **5** `applied scheduled mode` markers fired in ≥7 days — all
`10:00 -> fast`. The `01:00 -> cheap` boundary produced **no** marker because
`_mode_scheduler_step` no-ops when the persisted mode already matches
(`read_mode() == expected`), and the idle-cheap worker had already switched to
cheap the previous evening. Net schedule contribution: **~5/163 = 3%**.

**ContextHub / manual.** The remaining **158/163 = 97%** are non-scheduled
activity-driven switches (~76 into fast, ~82 into cheap). The
`mode-switch-worker` is the dominant source (rationale in
[Attribution limit](#method-and-reproducibility)).

**Flip-flop.** Flips alternate almost perfectly in pairs: 81 full
fast->cheap->fast cycles and 81 cheap->fast->cheap cycles. The cheap-dwell
distribution (time spent in cheap before flipping back to fast, n=81):

| Cheap dwell after a to-cheap flip | Count |
|---|---:|
| < 5 min | 21 |
| 5-15 min | 8 |
| 15-30 min | 19 |
| 30-60 min | 13 |
| 60-120 min | 8 |
| > 120 min | 12 |

- **26% of cheap excursions are effectively waste:** 21 cheap periods lasted
  < 5 min (minimum **7 s**). Two full restarts (fast->cheap, cheap->fast) for
  seconds-to-minutes of cheap mode. Example (verified in the log):
  `2026-10-14 15:57:43 spawn -> 15:57:49 start (cheap)` then
  `2026-10-14 15:58:04 spawn -> 15:58:10 start (fast)` — **21 s apart, net
  mode change zero**. Another at `2026-10-14 20:42:46 -> 20:42:53` (**7 s**).
- **Fast dwelling is cooldown-bounded:** fast dwell median 34.1 min, **minimum
  23.4 min** — the 30-min fast->cheap cooldown holds (0 genuine breaches; 25
  fast->cheap flips land in the [29, 30.5] min boundary band, consistent with the
  ~10-15 s lag between the `set_mode` call and the restart's start marker).
- **No rapid multi-flip storms:** the historical flapping fix means there are no
  4-restarts-in-5-min episodes; the shortest alternation is a single pair.

## Cost per flip (Q3)

Reused from LP-0MV9UNKVK007TN7E and re-verified on the current logs; the restart
destroys the warm KV cache and re-initialises llama-server.

| Metric | Value |
|---|---|
| Restart -> `llama-server is ready` | median **5 s**, max 5 s |
| Restart -> first local (`routing_check`) request | median 5 s, max 105 s |
| Warm llama-server preserved (adopt path) | **1 / 167** |
| Mode-switch drains | 140 (63 `timed_out=True`, 77 clean) |
| Chat requests deferred during a drain | ~372-387 |
| Local->remote fallback rate, 5 min before -> after | 5.9/min -> 9.2/min (**~1.6x**) |
| Restarts with a >=2x post-restart fallback spike | 77 / 170 (45 %) |

The dominant cost is **not** the 5 s restart but the **cold-cache fallback
spike** afterwards (a thundering herd of re-routes to remote providers) plus the
in-flight work lost by drains that exceed the 30 s window (63 timeouts).

**Cost of the wasted excursions specifically:** the 21 sub-5-min cheap periods
account for ~42 of the 163 flips (26%) — i.e. roughly **a quarter of all restart
cost buys almost no cheap-mode benefit**. Eliminating them is the cheapest
available win.

## What else restarts the proxy, and each cause's share (Q4)

Cross-referencing LP-0MV9UNKVK007TN7E (which classified all restarts by cause):

| Cause | Share | Necessary? |
|---|---:|---|
| Mode switch (fast<->cheap) | **~98 %** | Partly — adaptive switching is desired; the *rate* and the *cost per switch* are not |
| TTS watchdog `exit code=0` | ~1 % | No — downstream of the restart's `pkill -f qwentts` |
| llama-server watchdog `exit code=0` | ~0 % (all downstream of a restart) | No — downstream of the restart's `pkill -f llama-server` |
| Unattributed | ~1 % (3 in the wider window) | Tracked by LP-0MVAPNC2900145RH |

There are **no independent crash/self-heal restarts**; essentially 100% of the
restart rate is the mode-switch rate. Reducing flips reduces restarts 1:1.

## ContextHub `mode-switch-worker.ts` (Q1 code detail, AC2)

Source: `../ContextHub/packages/herdr/src/mode-switch-worker.ts` (+ wiring in
`packages/herdr/src/index.ts`). Two triggers, both POSTing `/admin/set-mode`:

- **Fast on command (all instances).** On an agent-route command the worker
  records the operator timestamp, broadcasts it to the shared activity file, and
  fire-and-forget POSTs `{"mode":"fast"}`. When the cached mode says `fast` it
  re-reads `GET /admin/mode` first, so a schedule-driven flip to cheap never
  suppresses an explicit fast request (WL-0MUWIHUL2000JEFR). **Fail-open:** a
  switch failure never blocks or delays command dispatch.
- **Cheap on idle (leader only).** `tick()` runs only when
  `DowntimeWorker.isLeader()` is true — non-leader panes skip the logic entirely
  (the flip-flop fix, WL-0MU6MXCZZ007ZXT9). It fires cheap only when **all** of:
  1. `modeSwitchEnabled` is true;
  2. no switch is already in flight (per-process single-flight);
  3. `now - lastActivity >= modeSwitchIdleThresholdMs`, where the idle clock is
     `max(local clock, shared mode-activity.json)` so **any** pane's activity
     keeps it fast;
  4. the proxy reports idle (`evaluateIdle`; with per-slot identity only >= 1
     free slot is required, otherwise all slots must be free; a null/unreadable
     status is treated as **busy**);
  5. no **model cooldown** is active (`readModelCooldowns` over the machine-wide
     coordination dir) — a pane in proxy backoff holds no slot, so this prevents
     adding concurrency during provider exhaustion;
  6. the freshly-read persisted mode is not already `cheap`.

  **Fail-closed everywhere:** endpoint failures, timeouts, non-2xx, ambiguous
  payloads, an unreadable cooldown signal, and a failed mode read all yield no
  switch and never throw. On plugin restart the idle clock resets to "active
  now", so a fresh full idle window is always required.

**Thresholds / clamps:** idle threshold default **900,000 ms (15 min)**, floor
60 s; poll interval default 10 s, clamped [5 s, 60 s]; admin API timeout 5 s;
scheduler-run watchdog 30 s; coarse `evaluateIdle` requires
`DOWNTIME_PANE_MIN_FREE_SLOTS` free slots when per-slot identity is present.

**Coexistence with the proxy schedule:** the worker's switches are *manual*
overrides (`set_mode(..., manual=True)`), respected until the next scheduled
boundary (LP-0MSMF25V9002AY1J); the schedule then reclaims control. This is why
the 01:00 cheap boundary is a no-op on most days.

**Proxy side:** `proxy/proxy/mode.py` `set_mode()` persists the mode and spawns
`proxy/scripts/start-proxy.sh --restart`; a manual fast->cheap switch inside
`MODE_SWITCH_COOLDOWN_SECONDS = 30 min` is rejected with 429
(LP-0MU6MQIPP0058198). Fast/cheap are separate profiles (`config-fast.yaml` vs
`config-cheap.yaml`) differing in `session_slot_pool_size` (fast = base **1**,
cheap = **3**), which is why the switch re-initialises llama-server instead of
adopting the running one.

## Reduction options (Q5)

| # | Option | Mechanism | Expected effect | Trade-off |
|---|---|---|---|---|
| A | **Non-destructive switch** | Equalise fast/cheap slot pools so the running llama-server is adopted (or restart only the proxy) | Removes the restart + warm-cache loss + ~1.6x fallback spike; **flip count unchanged** but each flip becomes near-free | Needs equal slot pools (fast 1 vs cheap 3): fast gets more concurrency or cheap loses its extra slots. Tracked: **LP-0MV9UNGEB009XUZ5** |
| B | **Longer fast->cheap cooldown** (30 -> 60 min) | Raise `MODE_SWITCH_COOLDOWN_SECONDS` | Caps cooldown-limited flips; median gap 32 -> ~60 min; est. ~23 -> ~15 flips/day | Proxy stays on fast (cloud-backed) longer after activity ends; delays the cost saving |
| C | **Longer idle threshold** (15 -> 30 min) | Raise `DEFAULT_MODE_SWITCH_IDLE_THRESHOLD_MS` | Removes the 21 sub-5-min wasted cheap excursions (which started after ~15 min idle); est. ~26% fewer cheap flips | More cloud spend during 15-30 min idle windows |
| D | **Working-hours fast** | Suppress idle-cheap 08:00-20:00 | Removes daytime cheap+fast pairs (~60% of flips) | No cost saving while "working"; needs a window config |
| E | **Coalesce/defer switches** | Require the target mode to be stable N minutes | Avoids the 7-21 s flip-flops | Delays the operator's fast transition (bad) unless asymmetric |
| F | **Log the caller/reason** | Record the `/admin/set-mode` actor + reason | Attribution only (no reduction) | None. Tracked: **LP-0MVAPNC2900145RH** |

## Recommendation (Q6)

**Act.**

1. **Primary — make the switch non-destructive (A).** This is the highest-leverage
   fix: it removes the dominant harm (warm-cache loss + fallback spike) for all
   ~23 flips/day without changing the (desirable) adaptive behaviour. Already in
   scope of **LP-0MV9UNGEB009XUZ5** — no duplicate item created.
2. **Complementary — add hysteresis (B + C).** Raise the fast->cheap cooldown to
   60 min and the idle threshold to 30 min. Both are single-value config changes;
   B is safe because it only delays the *cheap* direction and never the
   operator's fast path. Created as **LP-0MVAV5J6Y008K5EM**.
3. **Instrument for attribution (F)** so the next analysis can separate worker
   from manual switches: **LP-0MVAPNC2900145RH** (already created).

**Measurable target (B + C):** over the next >= 7-day window,
**flips/day <= 15** (from 22.9) and **sub-5-min cheap excursions/day <= 2**
(from ~3.3), with the fast-on-command latency unchanged (the existing re-read +
POST path stays sub-second).

**Decision recorded (do not duplicate):** the non-destructive switch is already
tracked by LP-0MV9UNGEB009XUZ5; the reduction-by-hysteresis work is
LP-0MVAV5J6Y008K5EM. No further items created.

## Answers to the work-item questions

1. **Flips/day by trigger source** — mean 22.9/day over 7 full days; whole-window
   split **schedule 5 / to-fast 76 / to-cheap 82**; per-day and intra-day tables
   above. The ContextHub worker dominates; the schedule contributes only the
   10:00 fast boundary.
2. **Fast-on-command flip-flop** — yes, it occurs: 21/81 (26%) cheap excursions
   lasted < 5 min, including a 7 s and a 21 s fast->cheap->fast pair. Fast dwell
   is cooldown-bounded (min 23.4 min).
3. **Cost per flip** — 5 s restart, 1 warm-cache loss, ~1.6x fallback spike, 45%
   of restarts with a >=2x spike, 63/140 drains timed out, ~372-387 requests
   deferred.
4. **Other restart causes** — ~98% of restarts are mode switches; the rest are
   downstream watchdog `exit code=0` artefacts of the restart's `pkill`, plus ~1%
   unattributed. No independent crashes.
5. **Reduction options** — A-F above, with expected effect and trade-offs.
6. **Recommendation** — act; primary = non-destructive switch
   (LP-0MV9UNGEB009XUZ5), complementary hysteresis
   (LP-0MVAV5J6Y008K5EM); target <= 15 flips/day and <= 2 sub-5-min excursions/day.

## Follow-up work items created

- **LP-0MVAV5J6Y008K5EM** — Reduce fast<->cheap mode flipping with longer
  cooldown + idle threshold (hysteresis). Chore, medium.
- Linked (pre-existing, not duplicated): **LP-0MV9UNGEB009XUZ5** (non-destructive
  switch), **LP-0MVAPNC2900145RH** (attribution logging), **LP-0MV9UNIOZ009RIAN**
  (grace-period immunity).
