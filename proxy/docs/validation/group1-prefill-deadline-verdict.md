# Group 1 validation — verdict against the ≤ 0.5/h target

**Work item:** LP-0MW8CS862005HFDX (child of the group-1 validation LP-0MW7V0S6A006DKZ4)
**Window:** `2026-11-10 14:18:26` → `2026-11-11 15:18:26` UTC — **25.0 h**, restart-free
**Dataset:** [`group1-prefill-deadline-residual.json`](./group1-prefill-deadline-residual.json)
**Window record:** [`group1-prefill-deadline-residual.md`](./group1-prefill-deadline-residual.md)

This is an **evidence and verdict** document. It changes no admission decision,
no threshold and no config; every number below is reproducible from the committed
dataset with the command in the window record.

## 0. Verdict at a glance

Certified criterion (producer decision 2026-11-02, option **b2**): user-class
`local_eviction_after_acceptance reason=prefill_deadline` +
`reason=measured_cache_deficit` ≤ **0.5 events/h** over a ≥ 24 h restart-free
window; **not certified** if occupancy-attributed ≥ 25 % or the
capacity-attributed ≥ 50 % of `prefill_deadline` events.

| Test | Baseline (2026-11-06 → 11-08, 33.736 h) | This window (25.0 h) | Verdict |
|------|----------------------------------------:|---------------------:|---------|
| **User-class combined rate** | 2.223/h (75 events) | **0.72/h** (18 events) | **NOT MET** — 1.44× the 0.5/h gate |
| occupancy guard (< 25 %) | 15 % (9/60) | **50 % (2/4)** | **guard FAILS** |
| capacity guard (< 50 %) | 36.7 % (22/60) | **50 % (2/4)** | **guard FAILS** |
| abandoned-window share, measured basis (≤ 20 %) | 3.3 % (2/60) | **0 % (0/4)** | target met |
| zero-progress starvations while draining (target 0) | not measured | **0 of 23** | target met |

One-line reading: the user-class combined rate fell from **2.223/h to 0.72/h**
(≈ 68 % reduction) but **still exceeds the 0.5/h target**, and the small
`prefill_deadline` sample (4 events) now splits **2 occupancy / 2 capacity**,
which trips both attribution guards. The surviving events are the
`deadline-capacity` class that **group 2** is specified to address
(`LP-0MW7V2D13004ZO65`: an explicit deadline token capacity enforced at
admission); group 1 (measured-residency routing only) was never expected to
remove them.

## 1. Evictions per reason (parent AC5, per-reason split)

Total post-acceptance evictions: **23 = 0.92/h**. All are **user class**; **0
audit** (`audit-` sessions).

| Reason | Total | User | Audit | /h (all) | Distinct sessions |
|--------|------:|-----:|------:|---------:|------------------:|
| `measured_cache_deficit` | 14 | 14 | 0 | 0.56 | 11 |
| `prefill_deadline` | 4 | 4 | 0 | 0.16 | 3 |
| `no_progress_watchdog` | 4 | 4 | 0 | 0.16 | 3 |
| `lease_evicted` | 1 | 1 | 0 | 0.04 | 1 |

**User-class combined (the certified criterion):**
`prefill_deadline` (4) + `measured_cache_deficit` (14) = **18 events / 25.0 h =
0.72/h**. Baseline 2.223/h → **0.72/h**, a 67.6 % reduction, **met/not-met =
NOT MET** against ≤ 0.5/h.

## 2. Attribution (n = 4 `prefill_deadline` events)

| Attribution | Events | Share | Baseline share |
|-------------|-------:|------:|---------------:|
| occupancy | 2 | 50 % | 15 % |
| capacity | 2 | 50 % | 36.7 % |
| predicted-warmth | 0 | 0 % | 48.3 % |

The **predicted-warmth** class — 48.3 % of the baseline — is now **zero**. That
is the group-1 (measured-residency) effect: admission no longer prices a cold
prompt as warm. What remains is the **capacity/occupancy** class, i.e. prompts
that were correctly priced as cold but still could not finish inside the 180 s
deadline under the observed load.

### Residency-unmeasured share (AC1)

All four `prefill_deadline` events have `estimated_tokens > 38 000`
(range 39 892 – 77 902) and `cluster_measured_cached_ratio = null`, i.e. their
residency was **unmeasured** at admission:

**4/4 = 100 %** of `estimated_tokens > 38 000` deadline events were admitted
with unmeasured residency, priced as cold.

## 3. Surviving user-class events (parent AC5; enumerated because NOT MET)

### 3.1 `prefill_deadline` (4 events, the named attribution detail)

| # | Time (UTC) | Attribution | est. tokens | elapsed s | priced cached ratio | slots avail min | in-flight max | Why it passed the new gates |
|---|-----------|-------------|------------:|----------:|-------------------:|----------------:|--------------:|------------------------------|
| 1 | 2026-11-10 15:03:26 | occupancy | 54 739 | 180.1 | 0.775 | 0 | 2 | Honest cold price (priced ratio 22.5 % > 10 %) so it was admitted as cold; a **second tenant** was in flight (`in_flight_requests_max=2`, `available_slots_min=0`) — the lease had no free slot before the deadline. Group 1 does not bound prefill tokens at admission; that is group 2's deadline-capacity bound. |
| 2 | 2026-11-10 18:25:15 | occupancy | 39 892 | 182.3 | 0.900 | 0 | 2 | Priced almost fully warm (priced ratio 10.0 %, the gate boundary), so it entered as a cheap prefill; the second tenant occupied the pool for the whole dispatch window (18 own lease extensions, zero backend progress observed). Removed only by a deadline-capacity bound on the cold remainder — group 2. |
| 3 | 2026-11-10 18:45:30 | capacity | 77 902 | 201.4 | 0.745 | 0 | 1 | Honest cold price (priced ratio 25.5 %); **no other tenant** (in-flight max 1) so the GPU itself could not deliver 19 854 new tokens inside 180 s. This is exactly the "too many cold tokens for the deadline" case group 2's token capacity refuses up front. |
| 4 | 2026-11-11 10:26:58 | capacity | 51 754 | 180.4 | 0.661 | 0 | 1 | Honest cold price (priced ratio 33.9 %); no other tenant; 17 570 new tokens could not prefill inside 180 s. Same capacity class as event 3. |

All four are **user-class agent sessions** (`herdr-*`). None is
`predicted-warmth`, so none is a regression of the group-1 pricing change.

### 3.2 `measured_cache_deficit` (14 events, distinct sessions listed)

| # | Time (UTC) | Session |
|---|-----------|---------|
| 1 | 2026-11-10 21:08:24 | herdr-1794343995-937074-24910 |
| 2 | 2026-11-10 21:09:37 | herdr-1794343995-937074-24910 |
| 3 | 2026-11-11 00:25:58 | herdr-1794356153-1526795-31748 |
| 4 | 2026-11-11 00:26:18 | herdr-1794356153-1526795-31748 |
| 5 | 2026-11-11 04:44:37 | herdr-1794372248-1834405-30393 |
| 6 | 2026-11-11 06:51:57 | herdr-1794354420-1479438-681 |
| 7 | 2026-11-11 08:06:39 | herdr-1794354420-1479438-681 |
| 8 | 2026-11-11 09:34:16 | herdr-1794383966-1190982-8187 |
| 9 | 2026-11-11 09:35:39 | herdr-1794389737-2251892-8389 |
| 10 | 2026-11-11 09:53:52 | herdr-1794370617-1804602-1876 |
| 11 | 2026-11-11 10:22:53 | herdr-1794392181-2292297-14777 |
| 12 | 2026-11-11 10:46:45 | herdr-1794389312-2247798-2336 |
| 13 | 2026-11-11 10:56:18 | herdr-1794377316-1066425-14299 |
| 14 | 2026-11-11 14:56:08 | herdr-1794408965-2944116-18277 |

These are the `measured_cache_deficit` abort (missed-warmth) class: a request
priced as warm showed a measured deficit above the abort bound. They are not
occupancy/capacity-attributed by the analyser's deadline rules and are listed
for completeness of the censused user-class events.

## 4. Side evidence

- **Abandoned-window overlap (measured basis):** **0/4 = 0 %** of
  `prefill_deadline` events started inside an abandoned-prefill window
  (target ≤ 20 % → met; legacy basis 1/4 = 25 %).
- **Zero-progress while draining:** **0 of 23** evictions
  (`zero_progress_while_draining = {basis: measured_stop, count: 0, total: 23}`)
  → target zero **met**.
- **Backend accounting:** 3 566 completed / 843 abandoned backend prefill
  segments over the window; `backend_prefill_accounting` and the release→cancel
  join are valid because the llama-server log covers the analysed window.

## 5. Limitations

- **Small `prefill_deadline` sample (n = 4).** The attribution guards are
  computed on 4 events, so a single event moves a guard by 25 percentage
  points; the guard verdicts (50 % / 50 %) are directionally clear but the
  sample is too small for a confidence interval. Stated, not smoothed.
- **"User class" = non-`audit-` sessions**, i.e. agent traffic
  (`herdr-*`/`pi` workers), not human-facing traffic. Same convention as the
  baseline.
- **No code/config change** in this validation; the deadline is
  `local_prefill_deadline_seconds = 180` in this window (a longer deadline is a
  separate producer decision and was not taken).

## 6. Conclusion

- **Met/not-met against ≤ 0.5/h:** **NOT MET** (0.72/h).
- **Direction:** strongly improved from the 2.223/h baseline (≈ 68 % fewer
  events), and the **predicted-warmth** class — the group-1 target — is
  **eliminated** (48.3 % → 0 %).
- **What remains** is the **capacity/occupancy** deadline class (2 + 2), which
  is precisely the residual **group 2** (`LP-0MW7V2D13004ZO65`) is specified to
  remove via a published deadline token capacity enforced at admission. Group 1
  alone was not expected to reach ≤ 0.5/h.
