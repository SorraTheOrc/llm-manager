# Prefill-deadline residual: post-fix window measurement and cause attribution

**Work item:** LP-0MVRMOT0N005136O (child of the AC3 tracking item
LP-0MVO8H7VQ003B12I; consumed by the closure verdict item
LP-0MVRMOV8K0007NFO).
**Measurement tool:** `proxy/scripts/analyse_prefill_deadline_residual.py`.
**Datasets:** `prefill-deadline-residual-2026-10-29.json` (the measured
window), `prefill-deadline-residual-baseline-2026-10-25.json` (the 22-event
baseline), `prefill-deadline-residual-fixwindow-2026-10-26.json` (the 114-event
fix-item window).

This is an **evidence** document. It records a measured window, the provenance
of every number, and the per-event cause attribution the AC3 closure verdict
needs. It changes no admission decision: the analyser only reads logs.

## 1. Window and deploy evidence (AC1)

| Item | Value |
|---|---|
| Measured window | `2026-10-29 14:08:10` → `2026-10-30 14:08:10` UTC (24.0 h) |
| Post-F4 restart (Option D live) | `2026-10-29 00:10:19` (log-rotation restart; first `prefill_progress_measured` line ever recorded is `2026-10-29 00:10:30`) |
| Scheduler epic | `8b7ab7a2` — LP-0MVOFVCA2008DQVE occupancy-aware admission, committed `2026-10-25T23:44:00Z`, in `HEAD` throughout |
| Option D (measured `prompt_progress` capture + early abort) | `1908f431` — LP-0MVR7IN0O000DPXN, committed `2026-10-28T05:49:43Z`, live from the `2026-10-29 00:10:19` restart |
| Retry-amplifier fix | `1911e2ec` — LP-0MVQITN6P008DPLB, committed `2026-10-27T10:09:42Z` |

The window start is deliberately 75 s before the `14:09:25` restart so the
window is **exactly 24 h** and byte-comparable with the ≥24 h post-deploy
measurement already published for the same window family
(LP-0MVS03KCV005BTGE, which measured `2026-10-29T14:08:10Z →
2026-10-30T14:08:10Z`).

### Re-base events inside the window (recorded, not discarded)

| Restart | Dev tip running afterwards |
|---|---|
| `2026-10-29 14:09:25` (75 s in) | `5026ec24` — LP-0MVT2KVVS002YZT8, additive cooldown-skip metrics |
| `2026-10-30 10:35:09` (20.4 h in) | `5d295fd0` — `dev` tip at that instant |

Admission-affecting commits that landed **inside** the window and only took
effect at the `10:35:09` restart: `0e270c1f` (LP-0MVT2PY3J00077W6,
`measured_last_dispatch` admission rung), `d463a35a` (LP-0MVT2QFAP0082BKR,
measurement invalidation on eviction/slot release/model switch), `793eceea`
(LP-0MVT2KWF0000QTSM, narrowed cooldown).

Because every figure below is derived from the **on-disk log stream** — no
in-process counter is consumed — a restart does not invalidate the window; it
segments it. Both segments are reported, so the re-base is visible rather than
averaged away:

| Segment | Hours | `prefill_deadline` | user | user/h | `measured_cache_deficit` | attribution (occ/cap/warmth) |
|---|---|---|---|---|---|---|
| `14:09:25 → 10:35:09` | 20.43 | 49 | 36 | 1.76 | 210 | 9 / 6 / 34 |
| `10:35:09 → 14:08:10` | 3.55 | 11 | 11 | 3.10 | 38 | 0 / 2 / 9 |

Config live in the window (`proxy/config.yaml`, `server:` section):
`local_prefill_deadline_seconds=120`, `local_cold_prefill_budget_tokens=25000`,
`local_measured_cache_abort_enabled=True`,
`local_measured_cache_abort_bound_tokens=2000`,
`local_measured_cache_abort_min_deficit_ratio=0.25`, slot pool = 2. No
slot-count change and no config reload occurred inside the window.

## 2. Reproduction

```bash
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-10-29 14:08:10" --end "2026-10-30 14:08:10" \
  --json --output /tmp/ac3-new.json

# the two comparison windows, same command and same classifier
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-10-25 11:46:50" --end "2026-10-25 19:11:00" \
  --json --output /tmp/ac3-baseline.json
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-10-26 09:06:07" --end "2026-10-27 09:06:07" \
  --json --output /tmp/ac3-fixwindow.json

# optional backend accounting cross-check: the llama-server day file that
# covers exactly 2026-10-29 00:00 → 2026-10-30 00:00
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-10-29 00:00:00" --end "2026-10-30 00:00:00" \
  --llama-log /var/log/llama-proxy/llama-server.log-2026-10-30 --json
```

Source files are read from `/var/log/llama-proxy/proxy.log*` (`.gz` handled
transparently); identical lines are de-duplicated because rotation keeps a
pre-restart copy of the same file. The 24 h window read 52 files / 353 286
in-window lines and dropped 318 duplicate lines.

**Cross-validation of the tool.** Run over the two documented windows it
reproduces their published figures exactly: the 22-event/7.4 h baseline
(12 user / 10 audit, `eta_prefill_s` max 201.5, `elapsed_s` max 133.8) and the
114-event fix window (`estimated_tokens` p10/p50/p90 = 2 478 / 35 008 / 66 958
and `new_tokens` p10/p50/p90 = 150 / 2 384 / 14 302, matching
LP-0MVQIT9QY007QU27's recorded table).

## 3. Eviction counts and the Option D conversion (AC5)

| reason | baseline 7.4 h | fix window 24 h | measured window 24 h |
|---|---|---|---|
| `prefill_deadline` | 22 | 114 | **60** |
| `measured_cache_deficit` | — (not deployed) | — (not deployed) | **248** |
| `no_progress_watchdog` | — | — | 60 |
| `lease_evicted` | 0 | 4 | 7 |
| `idle_timeout` | 0 | 2 | 0 |
| `slot_assignment_freed` | 94 (instrumentation false positive, fixed by `fc3c6e2e`) | 0 | 0 |

Counting is eviction-scoped: the same `reason=` string also appears on the
`measured_cache_abort` line of one event, so a raw `grep -o 'reason=' | wc -l`
over-counts. `count_evictions_by_reason()` in the analyser is the eviction-scoped
form, and a test pins it.

**The conversion is real and must not be read as the residual disappearing.**
248 dispatches in the window were released early by Option D instead of being
dragged to the 120 s deadline. The fall in `prefill_deadline` from 114 to 60 is
therefore *partly* a reason conversion, not a pure reduction.

## 4. User-traffic vs audit split (AC2, AC3)

Classifier: the parent AC4 one — a session id beginning `audit-` is agent audit
traffic, every other session is user traffic
(`reporting.AUDIT_SESSION_PREFIX`, mirrored as
`AUDIT_SESSION_PREFIX` in the analyser and pinned by a test).

| Metric | baseline 7.4 h | fix window 24 h | measured window 24 h |
|---|---|---|---|
| `prefill_deadline` total | 22 | 114 | 60 |
| user / audit | **12 / 10** | 83 / 31 | **47 / 13** |
| user per hour | **1.62** | 3.46 | **1.96** |
| audit per hour | 1.35 | 1.29 | 0.54 |
| distinct sessions | 12 | 60 | 22 |

Reading:

- the 22-event baseline's published split (12 user / 10 audit) reproduces
  exactly, so the two reports are comparable (AC2);
- the user-traffic rate fell from 3.46/h (fix window) to **1.96/h**, i.e. −43 %
  — but it is **not zero**, and it is *above* the 1.62/h pre-Option-D baseline;
- the residual therefore survives the scheduler epic **and** Option D, in the
  same shape: the `progress=0` signature is present in **60 of 60** events
  (717 `lease_extended_during_prefill` lines with `progress=None` and 70
  `prefill_wait_exceeds_threshold … progress=0` lines for the 60 victims,
  i.e. at least one per event), exactly as in the baseline (22 of 22).

### Classifier caveat for the verdict item

Under the parent AC4 classifier, "user traffic" means **every non-`audit-`
session**, which on this host is agent traffic: the 47 non-audit events come
from 20 `herdr-*` worker sessions and 27 `pi` session ids (`01a1…`); there is no
human-facing traffic in the window. AC3(a)'s "zero user-traffic evictions"
therefore reads, in practice, as "zero agent-session evictions". A producer
re-scope (path (b)) should say which of the three session families it intends
to gate on — the 22-event baseline's own prose treated a `drive` session as
audit while the classifier in the code does not.

## 5. Distributions (AC3)

`prefill_deadline` events only; percentiles use linear interpolation over the
in-window samples.

| Column | baseline 7.4 h (n=22) p10 / p50 / p90 | fix window (n=114) | measured window (n=60) |
|---|---|---|---|
| `estimated_tokens` (prompt) | 3 589 / 23 152 / 59 923 | 2 478 / 35 008 / 66 958 | **20 111 / 78 213 / 113 555** |
| `new_tokens` (priced compute) | 3 587 / 5 574 / 14 658 | 150 / 2 384 / 14 302 | **254 / 2 348 / 16 521** |
| `eta_prefill_s` | 7.6 / 51.2 / 190.5 | 4.8 / 34.2 / 91.7 | 9.5 / 46.4 / 76.1 |
| `elapsed_s` | 120.1 / 120.1 / 121.0 | 120.1 / 120.1 / 133.8 | 120.0 / 120.2 / 124.6 |
| priced `cached_ratio` (`1 − new/est`) | 0.0 / 0.166 / 0.928 | 0.0 / 0.892 / 0.998 | **0.0 / 0.946 / 0.996** |

The window is now dominated by **large warm-priced prompts**: the median event
is a 78 k-token prompt admission priced at 2 348 new tokens (96.9 % of the
prompt declared cached) and cancelled at the 120 s deadline having produced no
first byte. That is the same class the fix item identified as "fake-warm", but
larger (p50 prompt +123 % vs the fix window) and with an even more optimistic
price (p50 priced cached ratio 0.946 vs 0.892).

## 6. Cause attribution per event (AC4)

Rules are deterministic, ordered, and recorded in the report payload
(`attribution.rules`); every event carries the evidence its verdict consumed.

1. **occupancy** — another tenant occupied the slot pool inside the dispatch
   window: a `lease_extended_during_prefill` from a *different* session, or a
   different session's `provider=local` stream finishing inside the window, or a
   status poll reporting `local_in_flight_requests >= 2`.
2. **predicted-warmth** — no other tenant, and admission priced ≤ 10 % of the
   prompt as new compute (a warmth *prediction*), while the evidence contradicts
   that price: the backend reported **no** prefill progress at all inside the
   window, or the same prompt cluster measured a re-prefill ≥ 4× the priced
   compute.
3. **capacity** — no other tenant and an honest cold price (> 10 %): the GPU
   could not deliver the priced prefill inside the deadline.

| window | occupancy | capacity | predicted-warmth | capture available |
|---|---|---|---|---|
| baseline 7.4 h | 6 | 13 | 3 | no (pre-F4) |
| fix window 24 h | 17 | 49 | 48 | no (pre-F4) |
| **measured window 24 h** | **9** | **8** | **43** | **yes** |

Measured window by session class: user 7 occupancy / 5 capacity / 35
predicted-warmth; audit 2 / 3 / 8.

**The occupancy share is the direct test of this item's original hypothesis:
9 of 60 (15 %).** Zero events had another session's *lease* extended inside
their window; all 9 qualified through `local_in_flight_requests` reaching 2 and
4 of those also had another session's local stream finishing inside the window.
Second-slot head-of-line blocking is therefore a minority cause, not the
residual's shape — consistent with LP-0MVQIT9QY007QU27's earlier finding for
the 2026-10-26 window.

**Predicted warmth is the dominant class (43 of 60, 72 %)** — and it is
*measured*, not inferred, for the first time: 0 of 60 events produced a single
`prefill_progress_measured` line inside their own dispatch window, so the
backend never reported a first `prompt_progress` event, and the warmth the
admission priced was never realised. Where a retry of the same prompt had
measured, the priced compute was wrong by two orders of magnitude (e.g. priced
173 → measured 35 267).

### Contributing mechanism: the abandoned-prefill backlog

58 of the 60 events' dispatches started **inside an abandoned-prefill window**:
an earlier dispatch from the same burst had been released (49 of the 58 by
`measured_cache_deficit`, 9 by `prefill_deadline`) and the backend kept
evaluating its prompt, so the retry was queued behind it and reported no
progress for its whole lease. The evidence is the release ledger joined on
`measured_new_tokens / 43.1 tok/s` (F5's replay rate); median remaining backlog
at the retry's start is 667 s (min 8.5 s, max 1 932 s).

The backlog is visible directly at the backend. For the exactly-aligned
calendar day `2026-10-29 00:00 → 2026-10-30 00:00`:

| Backend vs proxy | Count |
|---|---|
| `llama-server` completed prompt evaluations (`prompt eval time` lines) | 1 092 |
| local streams the proxy actually consumed (`Stream finished … provider=local`) | 807 |
| post-acceptance releases the proxy made in the same day | 389 |
| **prefills computed and thrown away** | **285 (26 %)** |

So the early abort detours the *client* in under a second, but the abandoned
prefill is still computed. That is filed as
`discovered-from:LP-0MVRMOT0N005136O` (see the work item) because cancelling or
queuing against abandoned backend work is an admission/dispatch change, which
this evidence item explicitly out-scopes.

## 7. What the AC3 verdict can conclude (AC6)

Path **(a)** — "the scheduler epic lands and a post-landing window shows **zero
user-traffic** `prefill_deadline` evictions":

- **not met.** 47 user-class evictions / 24 h (1.96/h) in a window where the
  scheduler epic, the budget/deadline work and Option D are all live.
- The occupancy-only residue that (a) was written for is 9 events (0.38/h of
  all traffic classes); the remaining 51 are predicted-warmth (43) and capacity
  (8), i.e. a different failure mode than the one the epic targeted.

Path **(b)** — producer re-scopes parent AC4 with an agreed threshold. The
evidence supporting a re-scope:

1. The residual is no longer occupancy-shaped: 15 % occupancy, 72 %
   predicted-warmth, 13 % capacity.
2. There is a measurable, restart-stable rate to gate on: 1.96 user-class/h
   over 24 h (1.76/h and 3.10/h in the two restart segments), against 1.62/h
   pre-fix and 3.46/h at the fix item's window.
3. The dominant class is *already* being converted (248 early releases), so a
   bare `prefill_deadline` threshold can be gamed by the conversion — any
   re-scoped criterion must count both reasons together (AC5).
4. The abandoned-prefill backlog (285 thrown-away prefills/day) is the
   mechanism that turns an early release into the *next* event, so a fix owned
   by the dispatch path can plausibly move this residual — it is not a
   GPU-capacity floor.

## 8. Limitations

- `llama-server.log` carries no wall-clock timestamp, so per-event attribution
  is proxy-log-only; the backend accounting cross-check is window-level and
  valid only for the exactly-aligned day file.
- The abandoned-prefill window is an *estimate* (tokens ÷ a conservative rate),
  not an observation of the backend's queue. It is recorded as an evidence
  column, never as a verdict class.
- In the two comparison windows no `prefill_progress_measured` line exists (the
  capture shipped with F4), so their `predicted-warmth` share rests on the
  admission price plus the `progress=0` signature, not on measured deficits.
  The report says so explicitly (`attribution.capture_available` /
  `capture_caveat`) rather than implying a measurement that was never taken.
- `slot_id` is `unknown` on every post-F4 eviction line, so slot-level
  occupancy (which of the two slots was occupied) cannot be attributed; pool
  level (`local_in_flight_requests`, other-session leases) is used instead.
