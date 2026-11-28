# `slots_stale` post-fix 24 h window — 2026-11-18T10:25Z → 2026-11-19T10:25Z

**Work item:** LP-0MWLZ68SF0052E7Z (measurement child)
**F6 disposition target:** LP-0MVVXGMRN007NCJP — *Re-measure 24 h window; record AC2 disposition*
**Parent:** LP-0MVVCO1N7004TNGY — `slots_stale` 29.59 % residual
**Epic:** LP-0MWM0B7IQ008W2FP — abort-class remediation (one change per restart)
**Dataset:** [`slots-stale-postfix-window-2026-11-18-to-19.json`](slots-stale-postfix-window-2026-11-18-to-19.json)
**Verdict (F6 AC2): MET** — `slots_stale` **4.88 %** of 12,450 status polls
(608 stale) against the delivered age-derived definition (ceiling 300 s).
Reported alongside it, as the AC requires: the **tightened 2 s reading is
31.60 %** of datable polls, so the pass is a property of the 300 s ceiling, not
of snapshots being fresh (§ 5).

> **Decision note (2026-11-22) — the § 5 / § 8 "open question" is decided.**
> The producer took **option 3 (two-tier)** in LP-0MWRXH13L000UXSA: `slots_stale`
> stays at the delivered 300 s ceiling and is described as an
> **unusably-old-or-undatable bound**, never as a freshness claim; freshness is
> the snapshot-age distribution in § 5, **reported not targeted**; and no hard
> 2 s SLO is introduced today. Revisit trigger: set a freshness SLO only when
> undatable polls fall below 0.5 % of a 24 h window (LP-0MWRXFO0I005UBVM closed)
> **and** the nightly age series (LP-0MWRXGPO3004GXD7) has ≥ 14 days of data; a
> threshold that moves ships the relabelling test (AC3 of that decision). The
> applied text lives in
> [`proxy/docs/per-slot-status-verification.md`](../../proxy/docs/per-slot-status-verification.md#measured-outcome--post-fix-24-h-window-lp-0mwlz68sf0052e7z) (F7, LP-0MVVXGN2C004O28B).

---

## 1. Window and restart integrity (AC1)

| Fact | Value |
|---|---|
| Window | `2026-11-18 10:25:00` → `2026-11-19 10:25:00` (local = GMT/UTC), 24.0 h, half-open |
| Anchor restart | **2026-11-18T10:24:26Z**, pid **1789326** (`uvicorn proxy.server:app`) |
| Anchor marker | `2026-11-18 10:24:29,033 - INFO - Logging initialised at /var/log/llama-proxy/proxy.log pid=1789326` |
| Anchor attribution | `source=manual actor=unknown reason=LP-0MVVCO1N7004TNGY: deploy F3 shared /slots snapshot (156f3cd3d) + F4 age-derived slots_stale schema 8 (1aeca7059) + F5 residual attribution (d8ce36dc9)` |
| Startup markers **inside** the window | **0** |
| Markers searched | `Logging initialised`, `Rotated previous proxy log aside`, `Started server process`, `application startup` |
| Files searched | `proxy.log.2026-11-18_10.gz`, `…-11-18_16.gz`, `proxy.log-2026-11-19_00.gz`, `…-11-18_22.gz`, `…-11-19_04.gz`, `…-11-19_10.gz` |

The window starts at the first whole minute after the anchor, so the anchor's
own `Logging initialised` line (10:24:29) sits **outside** it — the rule in
`docs/dev/measurement-windows.md` § 3.4.

The post-window restart does not affect this measurement:

```
2026-11-19 19:36:07,546 - INFO - Logging initialised at /var/log/llama-proxy/proxy.log pid=2765810
2026-11-19 19:36:07,551 - INFO - Proxy restart attribution: source=manual actor=unknown reason=manual (no reason provided)
```

It is **9 h 11 m after** the window closed, so the window was restart-free for
its whole duration.

**Coverage: complete — 24.0 h of 24.0 h (100 %), no gaps.** Six contiguous log
segments carry the window (69,288 + 77,086 + 27,139 + 30,303 + 36,319 + 39
lines); window-start gap 0.0 s, window-end gap 2.0 s.

| Segment | First line in window | Last line in window | Lines |
|---|---|---|---|
| `proxy.log.2026-11-18_10.gz` | 2026-11-18 10:25:00 | 2026-11-18 16:24:26 | 69,288 |
| `proxy.log.2026-11-18_16.gz` | 2026-11-18 16:24:27 | 2026-11-18 22:24:25 | 77,086 |
| `proxy.log-2026-11-19_00.gz` | 2026-11-18 22:24:28 | 2026-11-19 00:00:27 | 27,139 |
| `proxy.log.2026-11-18_22.gz` | 2026-11-19 00:00:29 | 2026-11-19 04:24:26 | 30,303 |
| `proxy.log.2026-11-19_04.gz` | 2026-11-19 04:24:28 | 2026-11-19 10:24:24 | 36,319 |
| `proxy.log.2026-11-19_10.gz` | 2026-11-19 10:24:29 | 2026-11-19 10:24:58 | 39 |

## 2. Deployed build and reproduction (AC2)

| Fact | Value |
|---|---|
| Loaded build at the anchor | **`d8ce36dc9`** — the `dev` HEAD at 2026-11-18T10:24:26Z, i.e. exactly the F5 commit; contains F3 `156f3cd3d` and F4 `1aeca7059` |
| Checkout `dev` HEAD at run time | `33193b28c` (2026-11-20T13:09:42Z) — the checkout tip moves **without** a restart, so it is not the loaded code |
| `status_schema_version` in every window poll | **8** (4,903/4,903 in the first segment; the F4 contract) |
| Artifact directory | `~/proxy-usage-reports/slots-stale-postfix/` |

```bash
# 1. the measurement (as-configured definition)
python3 scripts/capacity_signal_verification.py \
  --since '2026-11-18 10:25:00' --until '2026-11-19 10:25:00' --window-closed \
  --label slots-stale-postfix --limit 250 --json \
  --write-artifacts ~/proxy-usage-reports/slots-stale-postfix

# 2. the tightened (2 s) reading and the snapshot-age distribution
python3 scripts/slots_snapshot_age_distribution.py \
  --since '2026-11-18 10:25:00' --until '2026-11-19 10:25:00' \
  --threshold-seconds 2 --threshold-seconds 5 --threshold-seconds 300 --json \
  > ~/proxy-usage-reports/slots-stale-postfix/age_distribution.json

# 3. archive the raw segments (done while still retained)
python3 scripts/proxy_usage_window_archive.py archive \
  --log-dir /var/log/llama-proxy \
  --dest ~/proxy-usage-reports/slots-stale-postfix/raw \
  --start '2026-11-18 10:25:00' --end '2026-11-19 10:25:00'
```

The raw segments are archived outside the repository (host state, 8 files,
8.8 MB, `gzip+cmp` verified, checksums in `raw/SHA256SUMS`) following
`docs/dev/proxy-usage-evidence-windows.md`; `proxy_usage_window_archive.py
coverage` independently reports **24.0 h measured (100.0 %) — complete** for
the same bounds.

## 3. The as-configured reading (the AC2 verdict figure)

Definition (F4, `status_schema_version: 8`): `slots_stale` is true **iff**
`slots_snapshot_age_seconds > slots_max_snapshot_age_seconds` (delivered
ceiling **300 s** = `max(120, 10 × 30 s poll cadence)`), **or** the poll cannot
date a snapshot at all.

| Metric | Value |
|---|---|
| Status polls | **12,450** |
| `slots_stale=true` | **608 → 4.88 %** (< 5 % target: **met**) |
| — over the 300 s ceiling | 51 (0.41 % of polls; `detail_cache`) |
| — no datable snapshot | 557 (4.47 % of polls; `counts_cache`, `slots_snapshot_source=none`; the contract counts these stale) |
| Derivation | age-derived 12,450 / published-flag 0 / disagreements 0 |
| Stale by poll latency | `<1 s` 550 · `1–2 s` 58 · `2–5 s` 0 · `≥5 s` 0 |

The two halves reconcile with the tool exactly: **51 over-age + 557
undatable = 608**, and the independent
`scripts/slots_snapshot_age_distribution.py` run reproduces
`count 608 / rate 0.0488`.

**Read the composition, not just the rate.** 91.6 % of the residual is
"cannot date the snapshot" (a counts-cache fallback with no dated snapshot),
not "served an old snapshot". Only 51 polls (0.41 %) were served a snapshot
older than the 300 s ceiling.

### Per-phase read path (F3 evidence)

| Phase | stale p50/p90/p99 (ms) | fresh p50/p90/p99 (ms) |
|---|---|---|
| `status_query_ms` | 5 / 22 / 147 | 5 / 7 / 20 |
| `slots_counts_ms` | 310 / 979 / 1,061 | 0 / 1,050 / 1,066 |
| `slots_detail_ms` | 0 / 0 / 0 | 0 / 0 / 0 |

`slots_upstream_calls = 0` on **11,492 of 12,450 polls (92.3 %)**; the fresh
p90 of 1,050 ms on `slots_counts_ms` is carried by the 950 polls (7.6 %) that
did make one coalesced upstream fetch. The status path no longer pays a
blocking per-poll fetch — compare the pre-fix window (§ 6), where stale polls
paid 1,647 / 4,002 / 5,388 ms in `slots_counts_ms` and 1,015 / 2,029 /
3,043 ms in `slots_detail_ms`.

## 4. Capacity signals

| Signal | Value |
|---|---|
| `capacity_hint` | idle 6,780 (54.46 %) · prefill_bound 3,099 (24.89 %) · saturated 2,033 (16.33 %) · abandoned_busy 131 (1.05 %) · unknown 407 (3.27 %) |
| `saturated, available_slots=0` | 582 (4.67 % of polls) |
| `saturated` without live local work | **11 (0.09 % of polls, 0.54 % of saturated)** — **11/11 attributed** (`session_evicted` 8, `live_work_attribution_lag` 3, lookback 600 s, live-work window 10 s) |
| stale snapshot **and** saturated without live work | **0** (judgeable 12,450; over-budget 51; undatable 557) |
| Abandoned local dispatches | 418 (`session_evicted` 409, `measured_cache_deficit` 5, `client_disconnect` 3, `prefill_deadline` 1) |
| Local dispatches | 2,394 (265 sessions, 143 herdr; warm 1,062 / cold 434 / unknown 898) |
| herdr downtime dispatches | **not verifiable** — the retained dispatch log ends 2026-11-18 13:23:42Z, so a count of 0 would mean "not retained", not "none" |

The tool's own headline verdict is `TARGETS MISSED`, driven only by the
`saturated`-without-live-work check (11 vs 0). F6 AC3 is "stays at 0, **or every
residual poll is attributed**"; all 11 are attributed, so AC3 is met. The
`slots_stale < 5 %` check passes.

## 5. The tightened reading (AC3) — 2 s would fail, and that is the finding

Same polls, same window, ceiling reduced to the plan's Q3 candidate
(`2 s` = two 1 s refresher cycles):

| Snapshot age reading | Count | Rate of datable polls | Rate of all status polls |
|---|---|---|---|
| `> 2 s` (plan Q3 candidate) | **3,758** | **31.60 %** | **30.18 %** |
| `> 5 s` | 3,542 | 29.78 % | 28.45 % |
| `> 300 s` (delivered ceiling) | 51 | 0.43 % | 0.41 % |
| undatable (contract stale) | 557 | — | 4.47 % |

**Age distribution** (datable polls, `n = 11,893`; the other 557 could not be
dated): **p50 0.73 s · p90 35.14 s · p99 187.73 s · max 424.20 s**
(min 5.4 × 10⁻⁵ s).

**Where the over-2 s polls sit:** `prefill_bound` **3,057 (81.3 %)** ·
`saturated` 300 (8.0 %) · `unknown` 401 (10.7 %). The 300 s pass therefore
coexists with a snapshot population whose p90 is 35 s: the ceiling absorbs the
prefill-blocked backend regime rather than measuring freshness inside it.

At the tightened 2 s ceiling the same window would read **31.6 % stale — a
miss of the 5 % target by a factor of 6.3**. The verdict is decided by the
ceiling, not by the derivation. This was raised as a **non-blocking OPEN
QUESTION on the parent** (LP-0MVVCO1N7004TNGY) and was **decided on 2026-11-22**
(LP-0MWRXH13L000UXSA, option 3, two-tier) — see the decision note at the top:
keep the 300 s ceiling as an *unusably-old-or-undatable* bound, report this
distribution as the freshness number to watch, and introduce no 2 s SLO today.
This item does not change the ceiling.

## 6. Pre/post comparison (AC4)

`slots_stale` **changed meaning** in schema 8 (F4), so the older rows are
**pre-definition-change** and are not like-for-like: they measured "a phase
served the last-known cache", not "the served snapshot is older than a
ceiling". The post-F1 row carries the instrumentation but predates F3/F4.

| Window (local) | State | Polls | `slots_stale` | Stale by source | Stale by poll latency | `capacity_hint` idle / prefill / sat / abandoned / unknown | Sat. w/o live work |
|---|---|---|---|---|---|---|---|
| 2026-10-29 01:00 → 10-30 01:00 | pre-definition, pre-F1/F3/F4 (`f58e8964`) | 26,019 | 7,698 (**29.59 %**) | detail 7,224 · counts 474 | ≥5 s 2,063 · 2–5 s 3,888 · 1–2 s 1,637 · <1 s 110 | 16,160 / 6,156 / 3,293 / 410 / 0 | 70 (0.27 %) |
| 2026-11-01 → 11-02 (nightly) | pre-definition | 34,368 | 13,912 (**40.48 %**) | — | — | — | — |
| 2026-11-16T15:20:47 → 11-17T15:20:47 | **pre-definition change** (F1 instrumentation live, F3/F4 not) | 29,053 | 5,239 (**18.03 %**) | detail 4,504 · counts 735 | ≥5 s 1,172 · 2–5 s 2,784 · 1–2 s 1,197 · <1 s 86 | 17,017 / 3,882 / 6,521 / 1,099 / 534 | 19 (0.065 %) |
| **2026-11-18T10:25 → 11-19T10:25** | **F3+F4+F5 live, schema 8 (age-derived)** | **12,450** | **608 (4.88 %)** | **over-age 51 (detail_cache) · undatable 557 (counts_cache)** | <1 s 550 · 1–2 s 58 · 2–5 s 0 · ≥5 s 0 | 6,780 / 3,099 / 2,033 / 131 / 407 | 11 (0.09 %), 11/11 attributed |

Per-phase comparison (stale polls, p50/p90/p99 ms):

| Phase | 2026-11-16 → 17 (pre-definition) | 2026-11-18 → 19 (this window) |
|---|---|---|
| `status_query_ms` | 4 / 6 / 21 | 5 / 22 / 147 |
| `slots_counts_ms` | 1,647 / 4,002 / 5,388 | 310 / 979 / 1,061 |
| `slots_detail_ms` | 1,015 / 2,029 / 3,043 | 0 / 0 / 0 |
| upstream calls on stale polls | 0: 992 · 1: 2,780 · 2: 1,467 | 0: 600 · 1: 8 |

## 7. Why the target is met, and what the metric does not capture (AC5)

**Why it is met.** F3 removed the blocking per-poll `/slots` fetch from the
status path: 92.3 % of polls made **zero** upstream calls, the per-slot detail
phase costs 0 ms on every poll, and the shared fetch that remains costs ~1 s at
p90 only on the 7.6 % of polls that refresh. The residual contract-stale polls
are now dominated by an *undatable* snapshot (557) rather than a slow refresh
(51 over-age).

**What `slots_stale` does not capture — read a pass with these qualifiers:**

1. **Freshness inside the ceiling.** p90 age is 35 s, p99 187.7 s, max 424.2 s.
   A 2 s ceiling would read 31.6 % of datable polls (§ 5). A `< 5 %` pass at a
   300 s ceiling says "no poll was served an *unusably* old snapshot", **not**
   "the snapshot was fresh". Any target phrased as freshness must use the
   tightened reading.
2. **It does not distinguish "old" from "cannot date".** 557 of the 608 stale
   polls are the latter; the field lumps them together by contract.
3. **It says nothing about content correctness.** Whether `capacity_hint` /
   `available_slots` were *right* is the separate invariant in § 4 (11/11
   attributed, zero stale-snapshot-and-saturated violations).
4. **Pre-schema-8 numbers are not comparable.** 29.59 % / 40.48 % / 18.03 %
   counted cache-fallback phase flags; 4.88 % counts age (or undatability).
5. **The prefill-blocked regime dominates the tail** (81.3 % of over-2 s polls
   carry `capacity_hint=prefill_bound`): the snapshot ages because the backend
   is blocked in prefill, which no read-path change can fix.
6. **herdr downtime-dispatch coverage is absent** for this window (retention
   ended 2026-11-18 13:23:42Z), so that companion target cannot be judged here.

## 8. Outcome

| AC | Result |
|---|---|
| AC1 window evidenced restart-free, coverage recorded | **met** — 0 in-window startup markers; 24.0 h / 100 %, no gaps (§ 1) |
| AC2 command, bounds, deployed build, artifact dir recorded on F6 | **met** (§ 2, recorded on LP-0MVVXGMRN007NCJP) |
| AC3 as-configured verdict **and** tightened 2 s reading with age distribution | **met** — 4.88 % vs 31.60 % (§§ 3, 5) |
| AC4 pre/post table with the definition change labelled on the parent | **met** (§ 6, recorded on LP-0MVVCO1N7004TNGY) |
| AC5 `< 5 %` → say why and what the metric still does and does not capture | **met** (§ 7); the 2 s ceiling choice was raised as a non-blocking OPEN QUESTION on the parent and is now **decided** — option 3, two-tier (2026-11-22, LP-0MWRXH13L000UXSA) |
| AC6 `proxy/docs/per-slot-status-verification.md` updated with the outcome | **met** |
