# Post-fix admission→first-byte conversion (24 h) — re-measured record

This page is the **post-fix 24 h re-measurement** carried by **LP-0MVV29YP6006PLNP**
for the parent **LP-0MVV1V213002QAIR** (AC7): after the occupant-hold blocker
pricing landed, is the abandoned-queue-wait share back — or better — than the
diurnally matched working baseline, and does the literal 85 % bar still hold?

- **Fix under measurement:** occupant-hold blocker pricing
  **`bed25a46c`** (`LP-0MVV29YDQ00437N6`), deployed by the
  proxy restart at **2026-11-05 13:23:48 UTC** (attribution
  `deploy LP-0MVYWLBV70018ZZB fix for AC6 verification`; first
  `local_admission_guard` line carrying `occupant_hold_s=` at
  `2026-11-05 13:23:57`).
- **Measured window (re-anchored):** `2026-11-06 14:16:00` →
  `2026-11-07 14:16:00` UTC (24 h). See
  [Window validity and re-anchor](#window-validity-and-re-anchor) — the
  nominal fix-anchored window is **void** because it was contaminated by
  mid-window redeploys and a backend crash-loop, and the record's own
  scheduling rule mandates re-anchoring rather than reporting a contaminated
  24 h.
- **Result:** abandoned-queue-wait share **82.1 % (23/28)** — met the literal
  85 % bar, but did **not** return to the 52.6 % diurnal working baseline
  (+29.5 points). The abandoned-wait **rate** collapsed to **0.96 /h** (vs
  2.96 /h baseline, 13.1 /h pre-fix post-change) because the queue volume
  collapsed (28 waits vs 136/517) — the guard now rejects at the blocker price
  rather than queueing, so the residual queue is tiny but its conditional
  abandonment is high. **Two material configuration confounds** separate the
  reference windows from this measurement and are stated in
  [Reading the numbers](#reading-the-numbers): the queue budget was swept
  60 s → 30 s and the queue depth 8 → 2 on 2026-10-31
  (`LP-0MVT1146V007WXA4`), i.e. **before** this fix deployed, so the post-fix
  window runs a shorter queue budget than the references it is compared
  against.

The numbers are produced entirely by
[`scripts/prefill_admission_conversion.py`](../../scripts/prefill_admission_conversion.py)
(a pure log analyser — no proxy behaviour change, no running proxy required):

```bash
scripts/prefill_admission_conversion.py \
  --since "2026-11-06 14:16:00" \
  --until "2026-11-07 14:16:00" \
  --log-dir /var/log/llama-proxy \
  --json > docs/dev/prefill-admission-conversion-postfix.json
```

Raw output: [`prefill-admission-conversion-postfix.json`](prefill-admission-conversion-postfix.json).
Re-running the command above regenerates the committed JSON byte-for-byte
(modulo the rotation-dependent `log_files_scanned` inventory), so no number on
this page is hand-typed: every table below is read off that JSON. The harness
also emits the `local_eviction_after_acceptance` counts (total, by `reason`,
and per band), so AC3 needs no separate log scan.

## Window validity and re-anchor

The nominal window — the 24 h ending 24 h after the fix deploy,
`2026-11-05 13:23:48` → `2026-11-06 13:23:48` — fails the record's own
preconditions (the LP-0MVTKTOVI006ZONN scheduling rule carried on this item):
*proxies redeployed mid-window, or a backend outage that suppressed local
admission traffic, must re-anchor the window at the next clean deploy rather
than report a contaminated 24 h.*

**Evidence of contamination (nominal window):**

| Time (UTC) | Event | Impact |
|---|---|---|
| 2026-11-05 13:23:48 | fix deploy restart (`bed25a46c`) | window start (attested) |
| 2026-11-05 14:03:41 | **mid-window redeploy** `d66a76b18` (LP-0MW3K7WD6003TJKZ death reporting) | different code served |
| 2026-11-05 14:04 → 14:44 | **llama-server SIGKILL crash-loop** (7 `death_window` records, pids 2114835→2081364); further kills at 15:13 and 22:11 (llama-server.5/4.log `Killed`) | backend repeatedly down; local admission traffic suppressed |
| 2026-11-05 14:46:00 | **mid-window redeploy** `ea251d80c` (LP-0MW3K7WYV001XLL1 death-capture arming) | different code served |
| 2026-11-05 21:52:10 | manual restart (no reason) | restart inside window |
| 2026-11-06 08:26:13 | manual restart (no reason) | restart inside window |

The crash-loop (class A external SIGKILLs, attribution item
`LP-0MW3PLEH8008O4I7`: `tests/test_start_proxy_self_kill.py` ran against the
live host and killed the supervised backends) suppressed local admission for
hours; the harness measured only **7 queued waits** over the nominal "24 h"
(vs 517 in the post-change window) — a 71.4 % share over n=7 is not
comparable. Measured on the void window for the record: 7 enqueued / 5
abandoned, `queue_banding` all `own_outcome`, evictions 126 (2
`backend_restart`, 30 `idle_timeout` — crash artefacts).

**Re-anchor decision.** Per the record's rule the window is re-anchored at the
next clean deploy. The last restart before a long quiet stretch is
**2026-11-06 14:15:52** (manual) — the zero-kill validation item
`LP-0MW3K7Z7T0094JK6` recorded `2026-11-06 14:16:00` as the validated
restart-free anchor (≥ 24 h zero kills). The re-anchored window is therefore
`2026-11-06 14:16:00` → `2026-11-07 14:16:00` (window start = first whole
minute after the restart, per the measurement-windows convention). Verified:

- **Restart-free:** no `Starting LLama Proxy Server` marker in
  `proxy.log.2026-11-06_14.gz`, `_20.gz`, `proxy.log-2026-11-07_00.gz`,
  `_02.gz`, `_08.gz`, `_14.gz`, `_20.gz`, `proxy.log-2026-11-08_00.gz` (next
  restart: 2026-11-10 14:18:30).
- **Zero backend deaths:** no `death_window_llama_server*` file dated
  2026-11-06/07 (crash-loop confined to 2026-11-05); no watch-dog death lines
  in the window span. (The single `Killed` tail of `llama-server.1.log`
  belongs to the 2026-11-11 23:00 restart, outside the window.)
- **Occupant-hold build end-to-end:** `occupant_hold_s=` present on
  `local_admission_guard` lines at `2026-11-06 14:20:18` (window start) and
  `2026-11-07 13:00:01` (window end) — the fixed field is served at both
  bounds.
- **Fix code in the served build:** `bed25a46c` is an ancestor of the loaded
  build (`git merge-base --is-ancestor bed25a46c d03d7a1dc` = yes; dev HEAD at
  the anchor was `d03d7a1dc`, 2026-11-06 13:19). No queue/admission-guard file
  changed between the fix and the loaded build (`git diff --name-only
  bed25a46c 35736a205 -- '*admission*' '*queue*'` empty; the only in-between
  changes are death-capture and unreachable-backend routing observability in
  `provider.py`/`server.py`, which do not alter the contention-queue or
  admission-pricing path measured here).

## Window and inputs

- **Window:** `2026-11-06 14:16:00` → `2026-11-07 14:16:00` (24 h, inclusive).
- **Log files read (5):** `proxy.log.2026-11-06_14.gz`,
  `proxy.log.2026-11-06_20.gz`, `proxy.log-2026-11-07_00.gz`,
  `proxy.log.2026-11-07_02.gz`, `proxy.log.2026-11-07_08.gz` (plus the
  `proxy.log` tail; 41 files scanned globally, 5 intersect the window).
- **Coverage:** exact — `queue_wait_diagnosis` reports `covered: true`, zero
  gaps at both window edges.
- **First byte with no admission line:** 2 sessions (reported separately,
  never silently joined to an admission — harness AC2).
- **Queued waits enqueued:** 28, against 28 queue outcomes (23 abandoned + 5
  dispatched) — the denominator is stated explicitly, not inferred.
- **Band provenance:** `queue_banding` = `own_outcome: 28` — every queued wait
  in this window is banded from its **own** outcome line's `new_tokens`
  (post-fix log shape carries the field), zero `unknown`, zero `session_chain`
  borrow. No re-banding caveats apply to this window.

Bands are on the **priced new (uncached) tokens** of a local dispatch:
`<8k`, `8k-20k`, `20k-32k`, `32k-64k`, `>=64k`.

## Sanity gate (window validity) — PASSES

The window is only valid if the running proxy actually emits the co-residency
and occupant-hold fields; an all-`unknown` bucket would mean the old code was
still serving and the window would be void. Per-band `rate_bucket`
distribution over the window (from the committed JSON):

| Band | `contended` | `solitary` | `solitary_fallback` |
|---|---|---|---|
| `<8k` | 642 | 2 779 | 1 |
| `8k-20k` | 462 | 115 | 0 |
| `20k-32k` | 120 | 233 | 0 |
| `32k-64k` | 0 | 443 | 0 |
| `>=64k` | 0 | 378 | 0 |

`contended` admissions exist and the served code carries `occupant_hold_s` at
both bounds — the window is **served by the fixed build**, not void. The
absence of `contended` in `32k-64k` / `>=64k` mirrors the prior windows: every
dispatch there was rejected by the flat cold-prefill budget before any pricing
bucket mattered.

## Per-band results, reference vs post-fix

### Working baseline — diurnally matched pre-change (2026-10-28 00:10:18 → 2026-10-29 00:10:18, pre-fix code)

The working baseline is the record quoted in
[`prefill-admission-conversion-diurnal-baseline.md`](prefill-admission-conversion-diurnal-baseline.md):

| Figure | Value |
|---|---|
| Code | fully pre-change (ends exactly at the `52085f09` deploy) |
| Enqueued waits | 136 |
| Abandoned | 71 |
| **Abandoned share** | **52.6 % (71/135)** |
| Admits / Rejects | 576 / 4 078 |

(*The committed baseline JSON is a coverage re-measurement over the 6 h 10 m
that survived rotation (67 waits, 53.7 %) — the record states both
provenance classes separately and never mixes them. Its per-band rows are a
partial slice, so this page does not re-derive them.*)

### Pre-fix post-change (2026-10-29 00:10:18 → 2026-10-30 00:10:18, 24 h, `52085f09`)

From the committed [`prefill-admission-conversion-postchange.json`](prefill-admission-conversion-postchange.json):

| Band | Admits | Rejects | Converted | Unconverted | Conv % | Queued waits | Abandoned | Abandoned % | Queue-wait s |
|---|---|---|---|---|---|---|---|---|---|
| `<8k` | 1 132 | 396 | 99 | 62 | 61.5 % | 236 | 101 | 42.8 % | 9 383.0 |
| `8k-20k` | 97 | 221 | 26 | 18 | 59.1 % | 44 | 29 | 65.9 % | 2 138.6 |
| `20k-32k` | 49 | 608 | 5 | 11 | 31.2 % | 26 | 19 | 73.1 % | 1 386.4 |
| `32k-64k` | 0 | 855 | 0 | 0 | 0.0 % | 0 | 0 | 0.0 % | 0.0 |
| `>=64k` | 0 | 662 | 0 | 0 | 0.0 % | 0 | 0 | 0.0 % | 0.0 |
| `unknown` | 0 | 0 | 0 | 0 | 0.0 % | 211 | 165 | 78.2 % | 11 412.8 |
| **Total** | **1 278** | **2 742** | **130** | **91** | **58.8 %** | **517** | **314** | **60.7 %** | **24 320.7** |

### Post-fix re-anchored (2026-11-06 14:16:00 → 2026-11-07 14:16:00, 24 h, occupant-hold build)

| Band | Admits | Rejects | Converted | Unconverted | Conv % | Queued waits | Abandoned | Abandoned % | Queue-wait s |
|---|---|---|---|---|---|---|---|---|---|
| `<8k` | 2 879 | 543 | 123 | 32 | 79.4 % | 28 | 23 | **82.1 %** | 747.3 |
| `8k-20k` | 264 | 313 | 69 | 2 | 97.2 % | 0 | 0 | — | 0.0 |
| `20k-32k` | 10 | 343 | 6 | 0 | 100 % | 0 | 0 | — | 0.0 |
| `32k-64k` | 0 | 443 | 0 | 0 | — | 0 | 0 | — | 0.0 |
| `>=64k` | 0 | 378 | 0 | 0 | — | 0 | 0 | — | 0.0 |
| **Total** | **3 153** | **2 020** | **198** | **34** | **85.3 %** | **28** | **23** | **82.1 %** | **747.3** |

Every queued wait in the post-fix window sits in `<8k` and is banded from its
own outcome line — the `unknown` row is gone (post-fix log shape carries
`new_tokens` on the enqueue/outcome lines).

### Abandoned-queue-wait share, side by side

| Band | Pre-fix post-change % (24 h) | Post-fix % (24 h) | Pre-fix queued | Post-fix queued |
|---|---|---|---|---|
| `<8k` | 42.8 % | **82.1 %** | 236 | 28 |
| `8k-20k` | 65.9 % | — (0 queued) | 44 | 0 |
| `20k-32k` | 73.1 % | — (0 queued) | 26 | 0 |
| `32k-64k` | 0.0 % | — (0 queued) | 0 | 0 |
| `>=64k` | 0.0 % | — (0 queued) | 0 | 0 |
| `unknown` | 78.2 % | — (banding resolved) | 211 | 0 |
| **Total** | **60.7 %** (314/517) | **82.1 %** (23/28) | **517** | **28** |

(The working baseline's per-band rows are a partial coverage slice and are
not reproduced here; its total, **52.6 % (71/135)**, sits beside the other two
totals above.)

### Per-hour rates (duration-like-for-like; the windows differ in diurnal mix and in queue budget)

| Metric | Baseline / 24 h | Baseline / h | Pre-fix post-change / h | Post-fix / h |
|---|---|---|---|---|
| Local admits | 576 | 24.0 | 1 278 → 53.3 | 3 153 → 131.4 |
| Local rejects | 4 078 | 169.9 | 2 742 → 114.3 | 2 020 → 84.2 |
| Queued local waits | 136 | 5.7 | 517 → 21.5 | 28 → 1.17 |
| Abandoned queue waits | 71 | 2.96 | 314 → 13.1 | 23 → 0.96 |
| `local_eviction_after_acceptance` | —¹ | — | 392 → 16.3 | 80 → 3.33 |

(Baseline rows quote [`prefill-admission-conversion-diurnal-baseline.md`](prefill-admission-conversion-diurnal-baseline.md)'s
measurement-time total figures; post-change rows quote the committed postchange JSON; post-fix rows quote the committed postfix JSON.
¹ The baseline record's 48 evictions (all `prefill_deadline`) were counted on the
partial 6 h 10 m coverage slice, so no baseline eviction per-hour rate is quoted
here — see the AC3 table, which uses the full-window historical pre-change
figure (152) instead.)

The post-fix window **queues 18× fewer waits per hour** than the pre-fix
post-change window (1.17 vs 21.5) and **abandons 13.6× fewer per hour** (0.96
vs 13.1) — the guard's occupant-hold pricing rejects at the blocker price
instead of queueing doomed waits. The **share** is nevertheless higher because
the denominator collapsed harder than the numerator: only the residual
hard-to-serve waits still queue, and 23/28 of them expire on the 30 s budget
(see [Reading the numbers](#reading-the-numbers)).

## Entry path and abandon cause (`LP-0MVV29X2E004INFL`)

| Field | Value |
|---|---|
| enqueued / abandoned | 28 / 23 (**82.1 %**) |
| enqueued by trigger | `concurrency_limit: 10`, `local_lease_active: 18` |
| dispatched by trigger | `concurrency_limit: 1`, `local_lease_active: 4` |
| abandoned by trigger | `concurrency_limit: 9` (90.0 % of its 10), `local_lease_active: 14` (77.8 % of its 18) |
| abandoned by cause | `queue_wait_expired: 23` — **zero depth-cap refusals** |

Both triggers carry explicit post-fix `trigger=` fields; every abandonment is
a budget expiry (`queue_wait_expired`), none a `queue_depth_exceeded`. The
`local_lease_active` retry path (AC5's re-queue) contributes 18 of 28 enqueues
and 14 of 23 abandonments — the priority/one-budget fix shipped, but the
lease-active re-queue remains the larger entry path, matching the prior
windows' shape.

## Queue-wait diagnosis (per-wait, `LP-0MVV29XPT003OXI5` harness)

[`queue-wait-diagnosis-postfix-2026-11-06-24h.json`](queue-wait-diagnosis-postfix-2026-11-06-24h.json)
(reproduces with `scripts/queue_wait_diagnosis.py --since "2026-11-06 14:16:00"
--until "2026-11-07 14:16:00" --log-dir /var/log/llama-proxy --json`):

| Figure | Value |
|---|---|
| queue waits | 28 (23 abandoned / 5 dispatched) |
| correlation | `own_request: 28` (100 %; 0 ambiguous/mismatched) |
| band evidence | `own_outcome: 28` (all waits banded from own outcome) |
| blocker class (abandoned) | `prefill_in_flight: 16`, `lease_holder_generating: 7`, `unknown: 0` |
| abandoned with guard evidence | 23/23 — guard admitted all 23 |
| …admitted below budget | 16 (median projected 28.6 s against the 30 s queue budget) |
| …admitted with projected ≤ observation | 16 (projected_lt_observed) |
| median observed wait (abandoned) | 30.0 s — the 30 s queue budget, all 23 |
| median observed wait (dispatched) | 8.1 s |

The idiosyncratic signature of the pre-fix window (guard projected 35.8 s
median against a 60 s observed wait — the "says it fits, abandons anyway"
gap) is nearly gone: the occupant-hold pricing makes the projection land 1.4 s
under the observed 30 s wait, and the residual abandoned waits are the ones
the guard priced within budget but the 30 s queue budget could not honour.
The ratio of abandoned waits that the guard admitted **above** the queue
budget (`>=60` bucket, 7/23 — the `lease_holder_generating` cluster with two
busy slots) is the remaining misprice to watch; it is the same blocker class
AC6 priced, now at 7/23 rather than the pre-fix majority.

## AC3 — `local_eviction_after_acceptance` by reason

Counted by the committed harness over the re-anchored window (`evictions`
block of the JSON), and compared with the prior windows' committed figures:

| Reason | Pre-change (24 h) | Pre / h | Post-change (24 h) | Post / h | Post-fix (24 h) | Post-fix / h | Class |
|---|---|---|---|---|---|---|---|
| `measured_cache_deficit` | 0 | 0.00 | 234 | 9.75 | **40** | 1.67 | triaged (`LP-0MVV9JWL9003A6XR`) |
| `prefill_deadline` | 145 | 6.04 | 68 | 2.83 | **36** | 1.50 | pre-existing |
| `no_progress_watchdog` | 2 | 0.08 | 82 | 3.42 | **3** | 0.13 | pre-existing |
| `lease_evicted` | 1 | 0.04 | 7 | 0.29 | **1** | 0.04 | pre-existing |
| `idle_timeout` | 3 | 0.13 | 1 | 0.04 | 0 | 0.00 | pre-existing, absent post-fix |
| `orphan_cleanup` | 1 | 0.04 | 0 | 0.00 | 0 | 0.00 | pre-existing, absent post-fix |
| **Total** | **152** | **6.33** | **392** | **16.33** | **80** | **3.33** | — |

**Every class is zero or explained, and no class is genuinely new.** All four
classes present post-fix appeared in the post-change window's classifier
(preexisting or already-triaged): `measured_cache_deficit` is the feature's
early-abort class triaged in `LP-0MVV9JWL9003A6XR` (234 → 40, −83 %), the two
absent classes (`idle_timeout`, `orphan_cleanup`) are zero in a restart-free,
healthy-backend window. The cross-window comparator shares the
message-anchored classifier from `LP-0MVV9JWL9003A6XR` — a proxied request
body quoting the string can never be counted. No new triage item is required.
The same window run through the daily usage report
(`analyze_proxy_usage.py --start "2026-11-06 14:16:00" --end "2026-11-07
14:16:00"`) reports the identical counts (80 total; 40 `measured_cache_deficit`,
36 `prefill_deadline`, 3 `no_progress_watchdog`, 1 `lease_evicted`), so the AC3
figure is corroborated by an independent surface.

## Verdict against all three references

| Bar | Threshold | Post-fix result | Verdict |
|---|---|---|---|
| Literal parent AC4 (2026-10-19/20 reference) | survived share **< 85 %** | **82.1 %** (23/28) | **met** |
| Diurnally matched pre-change working baseline | at/below **52.6 %** (71/135) | **82.1 %** (+29.5 points) | **not met** |
| Pre-fix post-change measured share | at/below **60.7 %** (314/517) | **82.1 %** (+21.4 points) | **not met** |

So the strict answer to the parent's question is **no** on the working
baseline: the post-fix share did **not** return to 52.6 % — it is higher than
both the working baseline and the pre-fix post-change share. But that headline
is not the whole answer: the queue **volume** collapsed (136/517 → 28 waits)
and the per-hour abandoned rate fell 13.6× (13.1 → 0.96 /h) versus the pre-fix
state, because the guard now rejects at the occupant-hold price instead of
queueing. The share rose on a tiny residual queue that still mostly expires —
see the confounds below, which bind every comparison.

### Reading the numbers

Two **configuration confounds** separate the reference windows from this
measurement and are not fix effects:

1. **Queue budget 60 s → 30 s, depth 8 → 2 (LP-0MVT1146V007WXA4, swept
   2026-10-31, deployed before this fix).** All 23 abandoned waits here expire
   at exactly 30.0 s — the 30 s `contention_queue_max_wait_seconds` — while the
   references expired at ≥ 55 s on the 60 s budget. A shorter queue budget
   mechanically raises the conditional abandonment of the residual queue (less
   time to be rescued) and collapses the queue volume; the share comparison
   **must** be read with a shorter-budget bias (+share, −volume) against all
   three references.
2. **Diurnal mix.** The post-fix window is 14:16 → 14:16 (deploy-anchored);
   the working baseline is 00:10 → 00:10. The per-hour rows are the
   duration-like-for-like comparison; the shares carry a mix caveat, exactly as
   in the post-change record.

The re-anchor itself (first clean 24 h after the contaminated nominal window)
is a third, documented difference — the nominal window is void, never reported
as a record.

**AC5 (window-elapsed gate).** The item's stand-down clause applies only to a
session starting before the fixed window has elapsed. This measurement ran on
2026-11-12, long after both the nominal window (ended 2026-11-06 13:23:48)
and the re-anchored window (ended 2026-11-07 14:16:00) — the gate's
"window not yet elapsed" note is therefore **not applicable**, matching the
item's scheduling note that the full window has elapsed.

## Cross-reference

| # | Work item | Scope |
|---|-----------|-------|
| 1 | `LP-0MVV29YP6006PLNP` | This re-measured post-fix record |
| 2 | `LP-0MVV1V213002QAIR` | Parent — AC7 post-fix 24 h measurement; closing verdict recorded here |
| 3 | `LP-0MVV29YDQ00437N6` | The fix under measurement — occupant-hold blocker pricing (`bed25a46c`) |
| 4 | `LP-0MVT1146V007WXA4` | **Confound** — queue caps swept 60 s→30 s / depth 8→2 (2026-10-31, pre-fix) |
| 5 | `LP-0MVV29XDW006VTWK` | Diurnally matched pre-change baseline — **52.6 %** (71/135) |
| 6 | `LP-0MVTKTOVI006ZONN` | Pre-fix post-change record — **60.7 %** (314/517) |
| 7 | `LP-0MVV9JWL9003A6XR` | Eviction triage — `measured_cache_deficit` class (present post-fix at 40, −83 %) |
| 8 | `LP-0MW3K7Z7T0094JK6` | Zero-kill 2026-11-06 14:16:00 restart-free validation window — the re-anchor point |
| 9 | `LP-0MW3PLEH8008O4I7` | Class A SIGKILL attribution — the crash-loop that voided the nominal window |
| 10 | `LP-0MVV29XPT003OXI5` | Per-wait queue diagnosis harness used here |
| 11 | `LP-0MVWC0FAT0018PMW` | Measurement-window coordination record (`docs/dev/measurement-windows.md`) |

Related: [`docs/dev/contended-prefill-pricing.md`](contended-prefill-pricing.md)
(the how-to-and-rollback page for the contended pricing / occupant-hold chain).
