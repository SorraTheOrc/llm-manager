# `slots_stale` post-fix 24 h window — 2026-11-26T10:18Z → 2026-11-27T10:18Z

**Work item:** LP-0MWSD6D24002J0K7 — *Re-measure a 24 h restart-free window after
the fix*
**Parent:** LP-0MWRXFO0I005UBVM (AC4 re-measurement, AC5 record)
**Precedent / template:** [`slots-stale-postfix-window-2026-11-18-to-19.md`](slots-stale-postfix-window-2026-11-18-to-19.md)
**Dataset:** [`slots-stale-postfix-window-2026-11-26-to-27.json`](slots-stale-postfix-window-2026-11-26-to-27.json)
**Verdict (AC4): MET** — the undatable share of `/llama/local/status` polls fell
**4.47 % → 0.0368 %** (557/12,450 → 6/16,296) and the contract `slots_stale`
share fell **4.88 % → 0.0368 %** (608/12,450 → 6/16,296) on a **24.0 h
restart-free** window. The target was `< 0.5 %`; the measurement is **13.6×
inside it** and **121× smaller** than the baseline.

> **One number to carry forward:** there is no longer a *provenance gap*. Every
> one of the 6 residual stale polls had **no snapshot at all**
> (`slots_snapshot_source=none`) because the coalesced `/slots` refresh did not
> land inside the request's ~1 s wait — it is a timeout, not an undatable cache
> entry (§ 6). The 2026-11-18→19 window's 557 `counts_cache` undatable polls are
> gone, and 0 polls were served an over-age snapshot.

---

## 1. Window and restart integrity (AC1)

| Fact | Value |
|---|---|
| Window | `2026-11-26 10:18:00` → `2026-11-27 10:18:00` (local = GMT/UTC), 24.0 h, half-open |
| Anchor restart | **2026-11-26T10:17:48Z**, pid **2364780** (`uvicorn proxy.server:app`) |
| Anchor marker | `2026-11-26 10:17:49,970 - INFO - Logging initialised at /var/log/llama-proxy/proxy.log pid=2364780` |
| Anchor attribution | `source=manual actor=unknown reason=LP-0MWKQJHSM000CI6W label split deploy — V1 window LP-0MWM0CP4B001088M, 24 h restart-free` |
| Attribution basis for **this** item (AC1) | § 2 / § 3.1 of [`measurement-windows.md`](measurement-windows.md) bill LP-0MWSD6D24002J0K7 as a window owner of this anchor ("one window, many measurements", § 4.4). Registered **2026-11-26T17:09:23Z** (commit `91ad69d63`, LP-0MWXSPO0G001MSW4) — **17 h before the window closed** |
| Startup markers **inside** the window | **0** |
| Markers searched | `Logging initialised`, `Proxy restart attribution`, `Rotated previous proxy log aside`, `Started server process`, `application startup` |
| Files searched | `proxy.log.2026-11-26_10.gz`, `…-11-26_16.gz`, `…-11-26_22`, `proxy.log-2026-11-27_00.gz`, `…-11-27_04`, `…-11-27_10` |
| Pre-window restarts (outside the window) | `10:17:17` pid 2364484 (`reason=manual (no reason provided)`, superseded 31 s later) and `10:17:48` pid 2364780 (the anchor) |
| Next restart after the window | `2026-11-27T21:25:31Z` pid 2883701 (`reason=LP-0MWM0CPQN009SHJR retry-outcome deploy — opens V2 window …`) — **11 h 07 m after** the window closed |

The window starts at the first whole minute after the anchor, so both
`2026-11-26 10:17` restart lines sit **outside** it — the rule recorded in
`docs/dev/measurement-windows.md` § 3.4.

**Coverage: complete — 24.0 h of 24.0 h (100 %), no gaps.**
`proxy_usage_window_archive.py coverage` independently reports
`complete: true`, `gaps: []`, lead-in 0.0 h, tail 0.0 h. Six contiguous
segments carry the window (184,765 + 226,769 + 79,030 + 68,091 + 81,223 + 63
lines); window-start gap 0.0 s, window-end gap 0.0 s. First poll
`2026-11-26 10:18:00`, last poll `2026-11-27 10:17:59`.

| Segment | First line in window | Last line in window | Lines |
|---|---|---|---|
| `proxy.log.2026-11-26_10.gz` | 2026-11-26 10:18:00 | 2026-11-26 16:17:48 | 184,765 |
| `proxy.log.2026-11-26_16.gz` | 2026-11-26 16:17:49 | 2026-11-26 22:17:48 | 226,769 |
| `proxy.log-2026-11-27_00.gz` | 2026-11-26 22:17:49 | 2026-11-27 00:00:00 | 79,030 |
| `proxy.log.2026-11-26_22` | 2026-11-27 00:00:00 | 2026-11-27 04:17:48 | 68,091 |
| `proxy.log.2026-11-27_04` | 2026-11-27 04:17:50 | 2026-11-27 10:17:48 | 81,223 |
| `proxy.log.2026-11-27_10` | 2026-11-27 10:17:50 | 2026-11-27 10:18:00 | 63 |

## 2. Deployed build and reproduction

| Fact | Value |
|---|---|
| Loaded build at the anchor | **`989358278`** — `git log -1`: `LP-0MWKQJHSM000CI6W: Implementation complete`, 2026-11-26T04:47:16Z |
| Ancestry verified | `git merge-base --is-ancestor` confirms **`aa6115bcc`** (LP-0MWSD6CEE0068T19 — joiner `/slots` snapshot age, this item's own fix) and **`5ef9b5ebb`** (LP-0MWRWBJPP001Q5JX — measured-basis prefill calibration) are both ancestors of `989358278` |
| Checkout `dev` HEAD at run time | `c9c3a2f49` (2026-11-27T21:06:02Z) — the checkout tip moves **without** a restart, so it is not the loaded code |
| `status_schema_version` in every window poll | **8** (16,296/16,296) |
| Artifact directory | `~/proxy-usage-reports/slots-stale-postfix-24h-2026-11-26-to-27/` |

```bash
# 1. the measurement (as-configured definition)
python3 scripts/capacity_signal_verification.py \
  --since '2026-11-26 10:18:00' --until '2026-11-27 10:18:00' --window-closed \
  --label slots-stale-postfix-24h-rearm --limit 250 --json \
  --write-artifacts ~/proxy-usage-reports/slots-stale-postfix-24h-2026-11-26-to-27

# 2. the tightened (2 s / 5 s) reading and the snapshot-age distribution
python3 scripts/slots_snapshot_age_distribution.py \
  --since '2026-11-26 10:18:00' --until '2026-11-27 10:18:00' \
  --threshold-seconds 2 --threshold-seconds 5 --threshold-seconds 300 --json \
  > ~/proxy-usage-reports/slots-stale-postfix-24h-2026-11-26-to-27/age_distribution.json

# 3. the per-hour breakdown (§ 3), reusing the same parser as the tool so the
#    hourly totals reconcile with it (16,296 polls / 6 stale)
python3 - <<'EOF'
import collections, datetime as dt, json
from pathlib import Path
import sys; sys.path.insert(0, "scripts")
from lib import capacity_signals as cs, proxy_logs as pl
since, until = dt.datetime(2026, 11, 26, 10, 18), dt.datetime(2026, 11, 27, 10, 18)
hours = collections.OrderedDict()
for _path, line in pl.iter_proxy_log_lines(Path("/var/log/llama-proxy")):
    if not cs.is_status_poll(line):
        continue
    ts = pl.parse_line_timestamp(line)
    if not cs.in_window(ts, since, until):
        continue
    flag = cs.slots_stale_from_age(cs.slots_age_fields(line))
    if flag is None:
        flag = cs.stale_flag(cs.status_fields(line))
    e = hours.setdefault(ts.replace(minute=0, second=0, microsecond=0),
                         {"polls": 0, "stale": 0})
    e["polls"] += 1
    e["stale"] += 1 if flag else 0
print(json.dumps([{"hour_utc": b.strftime("%Y-%m-%dT%H:00Z"), **v}
                  for b, v in sorted(hours.items())], indent=2))
EOF

# 4. archive the raw segments (done while still retained)
python3 scripts/proxy_usage_window_archive.py archive \
  --log-dir /var/log/llama-proxy \
  --dest ~/proxy-usage-reports/slots-stale-postfix-24h-2026-11-26-to-27/raw \
  --start '2026-11-26 10:18:00' --end '2026-11-27 10:18:00'
```

The raw segments are archived outside the repository (host state, **8 files,
92,800,421 bytes**, `gzip+cmp` verified, checksums in `raw/SHA256SUMS` and
mirrored into the dataset's `artifacts.raw_segment_sha256`), following
`docs/dev/proxy-usage-evidence-windows.md`.

> `capacity_signal_verification.py` stamps `meta.work_item = LP-0MVOG6YRA000T9PK`
> in its output; that is the tool's own owning work item, not this
> measurement's. The dataset records `work_item: LP-0MWSD6D24002J0K7`
> explicitly.

## 3. The as-configured reading (the AC4 verdict figure)

Definition (F4, `status_schema_version: 8`): `slots_stale` is true **iff**
`slots_snapshot_age_seconds > slots_max_snapshot_age_seconds` (delivered
ceiling **300 s** = `max(120, 10 × 30 s poll cadence)`), **or** the poll cannot
date a snapshot at all.

| Metric | Value |
|---|---|
| Status polls | **16,296** |
| `slots_stale=true` | **6 → 0.0368 %** (target `< 0.5 %`: **met**, 13.6× inside) |
| — over the 300 s ceiling | **0** |
| — no datable snapshot | **6 (0.0368 %)** — all `slots_snapshot_source=none`, i.e. no snapshot existed (§ 6) |
| Derivation | age-derived 16,296 / published-flag 0 / disagreements 0 |
| Stale by poll latency | `<1 s` 0 · `1–2 s` 6 · `2–5 s` 0 · `≥5 s` 0 |
| Unattributed stale polls | **0** |

**Before vs after, same definition, same tools:**

| Window | Polls | `slots_stale` | Over-age | Undatable | Undatable share |
|---|---|---|---|---|---|
| 2026-11-18T10:25 → 11-19T10:25 (pre-fix) | 12,450 | 608 (4.88 %) | 51 | 557 | **4.47 %** |
| **2026-11-26T10:18 → 11-27T10:18 (post-fix)** | **16,296** | **6 (0.0368 %)** | **0** | **6** | **0.0368 %** |

### Per-hour breakdown (AC5)

Every hour of the window is covered; 19 of the 25 buckets are a clean 0. The
first bucket (`10:00`) is the partial 10:18–11:00 opening and the last
(`11-27T10:00`) the partial 10:00–10:18 closing — adjacent-hour partials are
expected in a half-open window.

| Hour (UTC) | Polls | `slots_stale` | Rate |
|---|---|---|---|
| 2026-11-26T10:00Z | 621 | 2 | 0.32 % |
| 2026-11-26T11:00Z | 660 | 0 | 0.00 % |
| 2026-11-26T12:00Z | 577 | 0 | 0.00 % |
| 2026-11-26T13:00Z | 704 | 0 | 0.00 % |
| 2026-11-26T14:00Z | 538 | 1 | 0.19 % |
| 2026-11-26T15:00Z | 321 | 1 | 0.31 % |
| 2026-11-26T16:00Z | 818 | 0 | 0.00 % |
| 2026-11-26T17:00Z | 721 | 0 | 0.00 % |
| 2026-11-26T18:00Z | 1,306 | 0 | 0.00 % |
| 2026-11-26T19:00Z | 1,272 | 0 | 0.00 % |
| 2026-11-26T20:00Z | 1,336 | 0 | 0.00 % |
| 2026-11-26T21:00Z | 928 | 2 | 0.22 % |
| 2026-11-26T22:00Z | 1,009 | 0 | 0.00 % |
| 2026-11-26T23:00Z | 1,275 | 0 | 0.00 % |
| 2026-11-27T00:00Z | 352 | 0 | 0.00 % |
| 2026-11-27T01:00Z | 527 | 0 | 0.00 % |
| 2026-11-27T02:00Z | 254 | 0 | 0.00 % |
| 2026-11-27T03:00Z | 491 | 0 | 0.00 % |
| 2026-11-27T04:00Z | 684 | 0 | 0.00 % |
| 2026-11-27T05:00Z | 511 | 0 | 0.00 % |
| 2026-11-27T06:00Z | 203 | 0 | 0.00 % |
| 2026-11-27T07:00Z | 307 | 0 | 0.00 % |
| 2026-11-27T08:00Z | 501 | 0 | 0.00 % |
| 2026-11-27T09:00Z | 327 | 0 | 0.00 % |
| 2026-11-27T10:00Z | 53 | 0 | 0.00 % |
| **Total** | **16,296** | **6** | **0.0368 %** |

No hour exceeds 0.32 %, so the improvement is not carried by a quiet stretch:
the busiest hours (18:00–20:00, 1,272–1,336 polls each) are **0/1,272 … 0/1,336**.

## 4. Agreement check — `slots_stale == over-age + undatable` (AC4)

| Quantity | Value |
|---|---|
| `capacity_signal_verification.py` `slots_stale` | **6** |
| over-age (300 s ceiling) + undatable | **0 + 6 = 6** |
| Agrees | **yes** |
| `slots_snapshot_age_distribution.py` `contract_stale.count` | **6** |
| `slots_snapshot_age_distribution.py` `polls.undatable` | **6** |
| `slots_snapshot_age_distribution.py` datable / no age field | 16,290 / 0 |

Two independently written readers of the same window agree to the poll, and the
age-derivation disagrees with the published flag on **0** of 16,296 polls.

## 5. The tightened reading — snapshot-age distribution

Same polls, same window, ceilings reduced to the freshness candidates the
plan's Q3 considered:

| Snapshot age reading | Count | Rate of datable polls | Rate of all status polls |
|---|---|---|---|
| `> 2 s` (plan Q3 candidate) | **5,618** | **34.49 %** | 34.47 % |
| `> 5 s` | 5,168 | 31.72 % | 31.71 % |
| `> 300 s` (delivered ceiling) | **0** | **0.00 %** | 0.00 % |
| undatable (contract stale) | 6 | — | 0.0368 % |

**Age distribution** (datable polls, `n = 16,290`; the other 6 could not be
dated): **p50 0.74 s · p90 38.51 s · p99 144.89 s · max 295.44 s**
(min 2.0 × 10⁻⁵ s). The max sits just under the 300 s ceiling — the
`slots_stale` pass is still a property of the ceiling absorbing the
prefill-blocked regime, **not** proof that snapshots are fresh; that is the
decided two-tier reading (LP-0MWRXH13L000UXSA, option 3: the ceiling is an
"unusably-old-or-undatable" bound and this distribution is the freshness number
to watch).

Composition of the over-2 s polls: `prefill_bound` **4,610 (82.1 %)** ·
`saturated` 702 (12.5 %) · `unknown` 306 (5.4 %) — the same prefill-bound tail
as the 2026-11-18→19 window (81.3 %), and unchanged by a read-path fix.

> **Revisit trigger status (for the producer, not decided here).** The two-tier
> decision (LP-0MWRXH13L000UXSA) reopens a freshness SLO only when undatable
> polls fall below 0.5 % of a 24 h window **and** the nightly age series
> (LP-0MWRXGPO3004GXD7) has ≥ 14 days of data. The **first clause is now
> satisfied** (0.0368 %); the second is not assessed by this item. No threshold
> was moved here.

## 6. The 6 residual stale polls — all explained, no provenance gap (AC7)

All six have `slots_snapshot_source=none`, `total_slots=0`,
`slots_upstream_calls=0` and an impossible-to-date snapshot: the poll's own
coalesced refresh had not produced a snapshot when the status request returned.
Five waited ~1.01–1.14 s in `status_query_ms` and never reached the counts
phase (`slots_counts_ms=None`, `not_attempted` in the tool's per-phase table);
the first poll of the window (14 s after the anchoring restart) is the same
state with a cold shared fetch and carries `slots_stale_reason=query_timeout`.

| Timestamp (UTC) | `slots_snapshot_source` | `status_query_ms` | `slots_counts_ms` | `total_slots` | `capacity_hint` | Explanation |
|---|---|---|---|---|---|---|
| 2026-11-26T10:18:02Z | `none` | 4 | 1,050 | 0 | idle | first poll of the window, 14 s after the restart: cold shared fetch; `slots_stale_reason=query_timeout` |
| 2026-11-26T10:46:49Z | `none` | 1,144 | — | 0 | idle | no snapshot materialised within the ~1.14 s refresh wait |
| 2026-11-26T14:05:21Z | `none` | 1,038 | — | 0 | idle | no snapshot materialised within the ~1.04 s refresh wait |
| 2026-11-26T15:29:27Z | `none` | 1,010 | — | 0 | idle | no snapshot materialised within the ~1.01 s refresh wait |
| 2026-11-26T21:18:32Z | `none` | 1,091 | — | 0 | idle | no snapshot materialised within the ~1.09 s refresh wait |
| 2026-11-26T21:23:42Z | `none` | 1,008 | — | 0 | idle | no snapshot materialised within the ~1.01 s refresh wait |

**Why this is not the old provenance gap.** In the 2026-11-18→19 window the
557 undatable polls were undatable *because the read path fell back to a
counts cache* — the poll served a stale cache entry and could not date it
(`counts_cache`, no snapshot, no timeout). Here the read path always either
served a dated snapshot (`snapshot` 5,734, `ttl_cache` 9,435, `upstream` 1,121)
or returned having waited out the refresh budget with **no snapshot at all**
(6). The residual is a *refresh-latency* artefact (the backend was busy: all six
read `capacity_hint=idle` but with `total_slots=0`, i.e. `/slots` did not answer
inside the budget), not a provenance hole: the poll's `slots_snapshot_source` is
explicit, and the counts/detail phases were not attempted rather than silently
serving an undatable cache.

**AC7 verdict: met** — 0 over-age polls; every remaining stale poll is
explained above from the poll's own fields.

## 7. Capacity signals

| Signal | Value |
|---|---|
| `capacity_hint` | idle 6,918 (42.45 %) · prefill_bound 4,706 (28.88 %) · saturated 4,010 (24.61 %) · abandoned_busy 352 (2.16 %) · unknown 310 (1.90 %) |
| `saturated, available_slots=0` | 1,928 (11.83 % of polls) |
| `saturated` without live local work | **4 (0.0247 %, 0.10 % of saturated)** — **4/4 attributed** (`session_evicted` 3, `measured_cache_deficit` 1; lookback 600 s, live-work window 10 s), unexplained **0** |
| stale snapshot **and** saturated without live work | **0** (judgeable 16,296; over-budget 0; undatable 6) |
| Abandoned local dispatches | 421 (`session_evicted` 413, `client_disconnect` 3, `measured_cache_deficit` 3, `prefill_deadline` 2) |
| Local dispatches | 2,492 (257 sessions, 111 herdr; warm 1,584 / cold 580 / unknown 328) |
| prefill-thrash cooldown skips | **0** (0 sessions; skips-with-free-GPU 0; releases 0) |
| herdr downtime dispatches | **not verifiable** — the retained dispatch log ends 2026-11-18 13:23:42Z, so a count of 0 means "not retained", not "none" |

Per-phase read path, stale vs fresh (ms, p50/p90/p99):

| Phase | stale | fresh |
|---|---|---|
| `status_query_ms` | 1,010 / 1,144 / 1,144 (n=6) | 5 / 7 / 51 (n=16,290) |
| `slots_counts_ms` | 1,050 / 1,050 / 1,050 (n=1, 5 not attempted) | 0 / 1,050 / 1,127 (n=16,290) |
| `slots_detail_ms` | 0 / 0 / 0 (n=1, 5 not attempted) | 0 / 0 / 0 (n=16,290) |

`slots_upstream_calls = 0` on **16,290 of 16,296 polls (99.96 %)**; the 1,499
polls (9.2 %) that made one coalesced upstream fetch carry the fresh
`slots_counts_ms` p90 of 1,050 ms. The status path never pays a blocking
per-poll fetch on the stale population.

The tool's headline verdict is `TARGETS MISSED`, driven **only** by the
`saturated`-without-live-work check (4 vs 0). All 4 are attributed, so the
companion invariant ("stays at 0, or every residual poll is attributed") holds
— the same shape as the 2026-11-18→19 window (11/11 attributed) and a further
reduction (11 → 4). It is **not** an AC of this item; it is recorded here so the
number is not mistaken for a regression.

## 8. Pre/post comparison over the definition change

`slots_stale` changed meaning in schema 8 (F4), so rows before 2026-11-18 are
**pre-definition-change** and not like-for-like. The three post-schema-8
readings are:

| Window (local) | Polls | `slots_stale` | Over-age | Undatable | Undatable share | Age p50 / p90 / max (s) |
|---|---|---|---|---|---|---|
| 2026-11-18 10:25 → 11-19 10:25 (F3+F4+F5, build `d8ce36dc9`) | 12,450 | 608 (**4.88 %**) | 51 (detail_cache) | 557 (counts_cache) | **4.47 %** | 0.73 / 35.14 / 424.20 |
| **2026-11-26 10:18 → 11-27 10:18 (joiner-age fix `aa6115bcc`, build `989358278`)** | **16,296** | **6 (0.0368 %)** | **0** | **6 (none/timeout)** | **0.0368 %** | 0.74 / 38.51 / 295.44 |

Per-phase comparison (stale polls, p50/p90/p99 ms):

| Phase | 2026-11-18 → 19 | 2026-11-26 → 27 |
|---|---|---|
| `status_query_ms` | 5 / 22 / 147 | 1,010 / 1,144 / 1,144 |
| `slots_counts_ms` | 310 / 979 / 1,061 | 1,050 / 1,050 / 1,050 |
| `slots_detail_ms` | 0 / 0 / 0 | 0 / 0 / 0 |
| stale polls served from `detail_cache` | 51 | **0** |
| stale polls served from `counts_cache` | 557 | **0** |

**What the fix changed.** The `counts_cache` undatable population — 557 polls
that served a counts-cache fallback with no datable snapshot — is **completely
gone**, and the `detail_cache` over-age population fell 51 → 0. The stale
population is now 6 polls whose stress signal is *latency* (the ~1.0–1.1 s
refresh wait in `status_query_ms`), not *provenance*.

## 9. Outcome

| AC | Result |
|---|---|
| AC1 window registered before it closed, restart-free, anchor facts recorded | **met** — § 2/§ 3.1 of `measurement-windows.md` (commit `91ad69d63`, 2026-11-26T17:09:23Z, 17 h before close) bills this item as a window owner; pid, attribution, first whole minute and loaded build all recorded; **0** in-window startup markers; 24.0 h / 100 % coverage (§ 1) |
| AC2/AC4 undatable share 4.47 % → < 0.5 %, dataset + report committed, both figures recorded | **met** — **4.47 % → 0.0368 %** (§ 3); artefacts committed under `docs/dev/`; both figures commented on LP-0MWRXFO0I005UBVM |
| AC4 agreement `slots_stale == over-age + undatable` | **met** — 6 == 0 + 6, two independent readers agree (§ 4) |
| AC5 dataset + report shape: bounds, poll count, per-hour breakdown, both thresholds | **met** — § 1 bounds, § 3 poll count + 25-bucket per-hour table, §§ 3/5 both threshold figures; JSON dataset alongside |
| AC6 both figures on the parent with artefact links | **met** — comment on LP-0MWRXFO0I005UBVM |
| AC7 every remaining stale poll attributable | **met** — 0 over-age; all 6 explained from the poll's own fields as refresh timeouts, no provenance gap (§ 6) |

**Headline for the parent:** undatable `/llama/local/status` polls are
**0.0368 % of a 24 h restart-free window**, down from **4.47 %** — the
provenance-gap class is eliminated, not merely reduced.
