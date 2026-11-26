# Measurement windows — shared restart-free 24 h calendar

**Work item:** LP-0MVWC0FAT0018PMW — *Coordinate restart-free 24 h measurement
windows across concurrent verification items*
**Kind:** process / record (no code change, no scheduling knob)
**Last refreshed:** 2026-11-26T17:00Z — **§ 2 fully refreshed** to the live
2026-11-26T10:17:48Z anchor (pid 2364780) by LP-0MWXSPO0G001MSW4, with the outgoing
2026-11-25 anchor moved into the restart history and every § 3.1 row re-armed. The
2026-11-25 window was **truncated at 23 h 10 m** by that restart — 50 minutes
short of a PASS — so no item may measure it. Earlier refreshes:
2026-11-25T11:15Z (the 2026-11-25 anchor, LP-0MWW0VZUU002E4MN) and
2026-11-22T12:00Z (restart rows for 2026-11-15/16/18/19 plus the post-fix
`slots_stale` window row from LP-0MWLZ68SF0052E7Z). Rows below § 3.1 that still
quote 2026-11-02/11-06 anchors are historical records, not live windows.

This file is the single coordination record for restart-free measurement
windows. It exists because every one of the open verification items below needs
a **restart-free ≥ 24 h** production window, and **every deploy restarts the
proxy**, which resets the in-process residency/lease state those windows
measure. Left uncoordinated, concurrent verification items invalidate each
other's windows, and each invalidation costs a re-arm plus a wasted dispatched
session.

> **One window serves every measurement whose fix is already loaded.** Deploy
> **before** a window opens, never inside one. A restart inside a window still
> invalidates that window — this calendar reduces how often that happens; it
> does not license measuring across a restart.

Companion records:

- [`prefill-recovery-verification.md`](prefill-recovery-verification.md) — the
  24 h prefill-recovery measurement conventions and the deployed-anchor rule
  (*a pushed commit is not a deployed fix*).
- [`proxy-usage-evidence-windows.md`](proxy-usage-evidence-windows.md) — how to
  archive and regenerate the raw log segments a window depends on before
  rotation ages them out.
- [`../restart-frequency-analysis-LP-0MV9UNKVK007TN7E.md`](../restart-frequency-analysis-LP-0MV9UNKVK007TN7E.md)
  — restart-cause taxonomy and the attribution method reused below.
- `.worklog/downtime-coordination.json` — optional machine-readable mirror
  (currently `entries: []`; mechanism A does not require it).

---

## 1. The one rule

A measurement window is **restart-free** by definition. The window anchor is the
**proxy restart that loaded the fix**, and the window start is the **first whole
minute after that restart**, so the restart line itself sits outside the window.
Any subsequent `Starting LLama Proxy Server` marker inside the window — a deploy,
a mode switch, a `--restart`, a crash recovery — re-anchors the window at the
new restart and the partial window never yields a PASS.

Two consequences drive everything else in this file:

1. **Deploys batch before a window opens.** A deploy that would restart the
   proxy must land in the gap *between* windows, not inside one. The interval
   between app-wide deploys is the shared restart-free window.
2. **A restart is attributable.** Every `start-proxy.sh --restart` must carry
   `--reason "<item id>: <what is being deployed>"` so the restart line names
   the deploy and the affected windows instead of
   `source=manual actor=unknown reason=manual`. (Attribution logging itself is
   LP-0MVAPNC2900145RH.)

> **Textbook mutual invalidation (2026-11-02).** Two sessions deployed 72 s
> apart: pid 3684928 at `10:38:10` (the `9f3b9c06` starvation-fix deploy, no
> `--reason`) and pid 3698721 at `10:39:25` (the unloaded-model `/slots` skip,
> `--reason` given). The first deploy's window died before it reached two
> minutes. One batched restart would have served both.

---

## 2. Current restart-free anchor (live)

| Fact | Value |
|---|---|
| Running proxy | pid **2364780**, `ps -o lstart` **2026-11-26T10:17:48Z**; `Logging initialised … pid=2364780` at `2026-11-26 10:17:49,970` |
| Restart attribution | `source=manual actor=unknown reason=LP-0MWKQJHSM000CI6W label split deploy — V1 window LP-0MWM0CP4B001088M, 24 h restart-free` (`--reason` sets the reason text; `source=` stays `manual` for an operator-run restart) |
| Loaded build | checkout tip at launch = **`989358278`** (`LP-0MWKQJHSM000CI6W: Implementation complete`, 2026-11-26T04:47:16Z). It carries every earlier anchor build — `git merge-base --is-ancestor` confirms `aa6115bcc` (LP-0MWSD6CEE0068T19 / LP-0MWRXFO0I005UBVM — joiner `/slots` snapshot age) and `5ef9b5ebb` (LP-0MWRWBJPP001Q5JX — prefill calibration measured basis) are ancestors |
| Behaviour changes in the delta over `5ef9b5ebb` | **one**: `989358278` (LP-0MWKQJHSM000CI6W). The other commit, `4f74f4621` (LP-0MWW0VZUU002E4MN), is this calendar's own docs refresh |
| llama-server | **preserved** by the restart ("Adopted running llama-server … preserving warm prompt cache") — the warm KV / prompt cache survived, so this anchor is not a cold-cache window |
| Checkout tip at refresh | `989358278` (origin/dev) — identical to the loaded build, so another restart right now would load nothing new |
| Window start | **2026-11-26T10:18:00Z** (first whole minute after the restart; both 10:17 restart lines sit outside the window) |
| Earliest a fresh 24 h window can close | **2026-11-27T10:18:00Z** (window start + 24 h) |
| Restart-free so far | counted from 2026-11-26T10:18:00Z; the previous anchor (pid 545169) reached **23 h 10 m** before this restart truncated it |
| Window owner for this anchor | **LP-0MWKQJHSM000CI6W** (named in the `reason=` line, with the V1 window owner **LP-0MWM0CP4B001088M**), plus the re-armed items **LP-0MWSD6D24002J0K7** (child of LP-0MWRXFO0I005UBVM; `startOn` = run-not-before `2026-11-27T10:18:00Z`), **LP-0MWRWBJPP001Q5JX** and **LP-0MVVLQMG1000843Q** (same window — one window, many measurements) |

> **The 2026-11-25 window is dead — do not measure it.** It was armed at
> `2026-11-25T11:08:00Z` for `2026-11-26T11:08:00Z`, but the proxy was restarted
> at `2026-11-26T10:17:48Z`, **23 h 10 m** in and 50 minutes before it closed.
> A partial window never yields a PASS (§ 1), so the items that had armed it are
> re-armed to this anchor in § 3.1 and the evidence for that day is a truncated
> run, not a measurement.

> **Do not restart the proxy before 2026-11-27T10:18:00Z.** A restart inside
> this window re-anchors it and the partial window never yields a PASS (§ 1).
> If a deploy cannot wait, record the re-anchor explicitly and re-arm the
> affected items to `last restart + 24 h` (§ 4.6) — and update this file in the
> same session, so the calendar never instructs a restart that already happened.

> **Available for reuse — producer decision pending.** The gating fix for
> **LP-0MWBZVU04007OII2** (group-3 validation, `01d89a19e` slot-before-admission)
> is still an ancestor of this loaded build, but the anchor it was offered does
> not exist any more: the 2026-11-25 window was truncated, and this anchor's
> `reason=` line names LP-0MWKQJHSM000CI6W only, not that item. Adopting it still
> requires an explicit producer decision recorded on that item and a row added
> below, or a later attributed restart.

> **Why the loaded build ≠ the checkout tip matters.** The proxy runs from a
> shared checkout whose `HEAD` moves when agents commit; the running process
> keeps the code it imported at start. Always record both (loaded build at the
> anchor and the tip at run time) — see
> [`prefill-recovery-verification.md`](prefill-recovery-verification.md#deployment-anchor--a-pushed-commit-is-not-a-deployed-fix).

### Restart history that has invalidated recent windows

Restarts since 2026-10-30 (from `Logging initialised … pid=` markers). Where the
attribution says `source=manual actor=unknown reason=manual`, the `--restart` was
run **without** `--reason`; the rest name their deploy:

| Restart (UTC) | pid | Note |
|---|---|---|
| 2026-10-30T10:35:09 | 3224511 | re-arm #3 arming restart |
| 2026-10-30T23:10:49 | 3797968 | LP-0MVVJYJQ9001I8QK deploy (post-deadline re-dispatch), `--reason` given |
| 2026-10-31T00:37:53 | 3938219 | **restart #2 — attributed below (AC3)** |
| 2026-10-31T00:41:57 | 3944027 | `/health` probe fix deploy (LP-0MVVCXUDZ007XID0 / ac8dc504) |
| 2026-10-31T14:36:12 | 6173 | manual, no reason |
| 2026-10-31T17:05:40 | 208037 | manual, no reason |
| 2026-10-31T23:10:08 | 752082 | manual, no reason |
| 2026-10-31T23:56:09 | 942578 | manual, no reason |
| 2026-11-01T13:12:53 | 1311467 | manual, no reason |
| 2026-11-01T13:17:50 | 1315877 | release-lag fix deploy (LP-0MVUVHBZH007INL1) |
| 2026-11-01T21:26:41 | 1487007 | manual, no reason |
| 2026-11-02T10:38:10 | 3684928 | LP-0MVZ2EPI3006Y0WN deploy of `9f3b9c06` (starvation fix) — `--reason` omitted, so manual/unknown |
| **2026-11-02T10:39:25** | **3698721** | **anchor of the 2026-11-02 refresh** — LP-0MVWBXDZA00494HK live verification (slices 2-3, unloaded-model `/slots` skip) |
| 2026-11-15T01:06:51 | 3122239 | manual, no reason (recovered from the retained segment) |
| 2026-11-16T15:20:49 | 3720326 | `reason=LP-0MW7V14DN007F127: group-2 validation restart (cold-regime ETA + deadline token capacity)` — the re-arm #6 anchor in § 3.4 |
| **2026-11-18T10:24:29** | **1789326** | **anchor of the post-fix `slots_stale` window** (§ 3.4) — `reason=LP-0MVVCO1N7004TNGY: deploy F3 shared /slots snapshot (156f3cd3d) + F4 age-derived slots_stale schema 8 (1aeca7059) + F5 residual attribution (d8ce36dc9)` |
| 2026-11-19T19:36:07 | 2765810 | manual, no reason — 9 h 11 m **after** that window closed, so it did not void it |
| 2026-11-22T13:47:45 | 3779655 | manual, no reason — loaded build `8d494d878`. Its own 24 h window `2026-11-22T13:47Z → 2026-11-23T13:47Z` **closed before** the next restart and is owned by **LP-0MWRR3ZUV008ADVB**; that item must measure before log retention prunes the segments (≈ 2026-11-29) |
| **2026-11-25T11:07:33** | **545169** | **anchor of the 2026-11-25 refresh** — `reason=LP-0MWRXFO0I005UBVM / LP-0MWSD6CEE0068T19: … batched with LP-0MWRWBJPP001Q5JX …`; llama-server preserved; loaded build `5ef9b5ebb`. **Truncated at 23 h 10 m** by the 2026-11-26T10:17:48Z restart: its `2026-11-25T11:08:00Z → 2026-11-26T11:08:00Z` window never closed restart-free, so it is **not** measurable |
| 2026-11-26T10:17:17 | 2364484 | **unattributed restart** — `reason=manual (no reason provided)`; the process was gone 32 s later (superseded by the launch below), so it served no window of its own. Recorded so the truncation chain is complete; the § 5 triage verdict is "superseded by the attributed 10:17:48Z deploy" |
| **2026-11-26T10:17:48** | **2364780** | **anchor of the 2026-11-26 refresh (§ 2)** — `reason=LP-0MWKQJHSM000CI6W label split deploy — V1 window LP-0MWM0CP4B001088M, 24 h restart-free`; llama-server adopted (warm cache preserved); loaded build `989358278` |

---

## 3. The calendar — open measurement items (live snapshot 2026-11-26T17:00Z)

### 3.1 Items that require a restart-free ≥ 24 h window

`Window it expects` is the window the item currently records; where later
restarts have invalidated it, the re-arm the item now needs is stated. Per the
producer's **future-only** decision, **this item does not re-arm expired
windows** — the owning item's own re-arm cycle does, and points here as the
coordination record.

| Item | What it measures | Required deploy commit(s) | Window it expects | State / next action |
|---|---|---|---|---|
| **LP-0MVN44XB0009M33K** | 24 h AC4 resident-prefix admission (scheduling-attributed `prefill_deadline` drop ≥ 90 %) | loaded build at the 2026-11-16T15:20:47Z anchor = `79f53ac1` (`LP-0MW8EAT60003Z28H`) | **measured** — re-arm #6 `2026-11-16T15:21:00Z → 2026-11-17T15:21:00Z` (see § 3.4) | **Done**: PASS — 0 scheduling-attributed of 1 deadline event (22 → 0); see `slot-thrashing-investigation/findings.md`. Open qualifier: the tail-of-window orphaned-marker wedge (`LP-0MWKWGLRT003AUG4`) |
| **LP-0MVP7HX2S008IU57** | router-responder `GET /slots … 500 == 0` after the child-port discovery fix | `f58e8964` + `189864fc` (LP-0MVP7525P008RXMF, deployed 2026-10-26T21:01:42Z) | `2026-10-27T12:00Z → 2026-10-28T12:00Z` — **expired** | Superseded in practice by the `/health`-fix measurement (LP-0MVVLKPBM007F6U6, RED); needs an explicit disposition, not a fresh 24 h window |
| **LP-0MVVLQMG1000843Q** | writer-side sidecar pairing, ≥ 24 h post-deploy | `5c0f80f9` (LP-0MVVI8TEU006KTOZ, 2026-10-31T06:18:04Z) — in **every** build since 2026-11-02 | from the current anchor: **`2026-11-26T10:18:00Z → 2026-11-27T10:18:00Z`** | Open. Two successive opportunities are spent — the pre-2026-11-25 anchor was ended by the 2026-11-25 restart, and the `2026-11-25T11:08:00Z → 2026-11-26T11:08:00Z` window was truncated at 23 h 10 m — so the earliest valid window is now **`2026-11-27T10:18:00Z`** |
| **LP-0MWSD6D24002J0K7** | re-measure the undatable share of `slots_stale` (target < 0.5 %) after the joiner-age fix | `aa6115bcc` (LP-0MWSD6CEE0068T19) **and** `5ef9b5ebb` (LP-0MWRWBJPP001Q5JX) — both ancestors of the loaded build `989358278` | **`2026-11-26T10:18:00Z → 2026-11-27T10:18:00Z`**; `startOn` = `2026-11-27T10:18:00Z` | **Re-armed** (the `2026-11-25T11:08:00Z → 2026-11-26T11:08:00Z` window it was armed on was truncated at 23 h 10 m). Parent AC4/AC5; measure after the window closes. Before: 4.47 % undatable (557/12,450) |
| **LP-0MWRWBJPP001Q5JX** | post-deploy measurement of the measured-basis prefill calibration | `5ef9b5ebb` — an ancestor of the loaded build `989358278` | shares the `2026-11-26T10:18:00Z → 2026-11-27T10:18:00Z` window | **No separate window** (one window, many measurements). Its `2026-11-25` window was truncated with the others, so this anchor is the one that serves it |
| **LP-0MWRR3ZUV008ADVB** | 24 h before/after + regression band after the measured-basis ETA fix | `af0e39273` — in the **older** anchor build `8d494d878` (pid 3779655) | `2026-11-22T13:47Z → 2026-11-23T13:47Z` — **closed**, restart-free, and bounded by the later 2026-11-25 **and** 2026-11-26 restarts | Open: analyse the closed window before log retention (≈ 2026-11-29); no new window needed |
| **LP-0MWBZVU04007OII2** | group-3 validation: one attributed restart + ≥ 24 h restart-free window | `01d89a19e` — an ancestor of the 2026-11-26 loaded build `989358278` | **no reusable window** — `2026-11-25T11:08:00Z → 2026-11-26T11:08:00Z` was truncated at 23 h 10 m, and the 2026-11-26 anchor's `reason=` names LP-0MWKQJHSM000CI6W only | **Producer decision pending**: adopt the current anchor explicitly here, or take a later attributed restart that names this item |
| **LP-0MVU8003X0018WT8** | router-responder `GET /slots … 400 == 0` after the `?model=` fixes | LP-0MVU7ZZTH005FSFL slices (merged; record loaded build at anchor) | `earliest 2026-11-01T23:56Z` (based on the 2026-10-31T23:56 restart) — **superseded** | **Re-arm** to the current anchor (`→ 2026-11-03T10:39:25Z`) |
| **LP-0MVVCO1N7004TNGY** | `slots_stale` over 24 h (target < 5 %; measured 29.59 %) | F3 `156f3cd3d` + F4 `1aeca7059` + F5 `d8ce36dc9`, deployed by the 2026-11-18T10:24:29Z restart | `2026-11-18T10:25:00Z → 2026-11-19T10:25:00Z` — **measured** (see § 3.4) | **Done**: `slots_stale` **4.88 %** (608/12,450) at the 300 s ceiling; tightened 2 s reading 31.60 %. Outcome recorded by LP-0MWLZ68SF0052E7Z |
| **LP-0MVVXGMRN007NCJP** | re-measure the 24 h `slots_stale` window; record AC2 disposition | fix under LP-0MVVCO1N7004TNGY | `≥ 24 h` after that fix is deployed | **Blocked** on the parent fix |
| **LP-0MVZ2EPI3006Y0WN** | restart the proxy onto the starvation fix, then let ≥ 24 h elapse | `9f3b9c06` (LP-0MVW8S8DH008E4W4, 2026-11-02T09:12:09Z) — **deployed** by the 2026-11-02T10:38:10 restart | window anchored at the current restart: `2026-11-02T10:39:25Z → 2026-11-03T10:39:25Z` | **Deploy done**; verification carried by **LP-0MVZ4R2B20027XK3** (run-not-before 2026-11-03T10:39:24Z) |
| **LP-0MVZ4R2B20027XK3** | verify the ≥ 24 h restart-free window for the starvation fix | the build loaded at the 2026-11-02T10:39:24Z anchor (`9f3b9c06` present) | `2026-11-02T10:39:25Z → 2026-11-03T10:39:25Z` (run instant = window end) | Open; do **not** restart the proxy before 2026-11-03T10:39:24Z |
| **LP-0MVW8S8OZ00449YR** | post-change ≥ 24 h measurement (AC3/AC4) | fix from LP-0MVW8S8DH008E4W4 (`9f3b9c06`) | after the LP-0MVZ2EPI3006Y0WN anchor window closes | **Unblocked** once LP-0MVZ4R2B20027XK3's window closes; shares that window |
| **LP-0MVNBJWRH006V9Y6** | baseline prefill measurement capture (≥ 24 h + a matching complete night window) | record `git log -1` at the anchor (child of LP-0MVMHNUAR006SRZT) | ≥ 24 h plus a matching night | **Blocked** on the parent prefill work |
| **LP-0MVNBJYPM004UV56** | post-change latency/stability verification (≥ 24 h + a matching complete night window) | fix from the parent prefill work; record `git log -1` at the anchor | ≥ 24 h plus a night after the change | **Blocked** on the parent prefill work |
| **LP-0MVV1V213002QAIR** | contended-pricing abandoned-queue-wait share on a 24 h window | record `git log -1` at the anchor | 24 h baseline/after window | Open; arm against the current (or next) anchor |
| **LP-0MVV29XDW006VTWK** | restate the abandoned-wait baseline on a diurnally matched 24 h window | fix under LP-0MVV1V213002QAIR | 24 h | **Blocked** on the diagnosis/fix chain |
| **LP-0MVV29YP6006PLNP** | re-measure the 24 h post-fix abandoned-wait share | fix under LP-0MVV1V213002QAIR | 24 h after fix | **Blocked** on the diagnosis/fix chain |
| **LP-0MVVXGI490089OX5** | profile a 24 h window; name the dominant `/slots` refresh contributor | record `git log -1` at the anchor | 24 h window | Open (child of LP-0MVVCO1N7004TNGY); arm against the current/next anchor |

### 3.4 Measured windows (completed — precedent, not actionable)

| Item | Window (UTC) | Anchor / loaded build | Outcome |
|---|---|---|---|
| **LP-0MVN44XB0009M33K** (re-arm #6, AC4 resident-prefix) | `2026-11-16T15:21:00Z → 2026-11-17T15:21:00Z` — 24.0 h, 100 % covered, five log segments | restart `2026-11-16T15:20:47Z` pid 3720326 (`reason=LP-0MW7V14DN007F127: group-2 validation restart`); loaded build `79f53ac1` | **PASS** (both AC6 gates): scheduling-attributed cancellations 22 → **0**, `residency_measured_decisions` 0 → **252**. Qualifier: 14:24:42Z–15:21Z had the local route closed by an orphaned prefill marker (`LP-0MWKWGLRT003AUG4`). Record: `slot-thrashing-investigation/findings.md` § *2026-11-17 Addendum (re-arm #6)* |
| **LP-0MWLZ68SF0052E7Z** (F6 AC2, `slots_stale`) | `2026-11-18T10:25:00Z → 2026-11-19T10:25:00Z` — 24.0 h, **100 % covered**, no gaps, six log segments | restart `2026-11-18T10:24:26Z` pid 1789326 (`reason=LP-0MVVCO1N7004TNGY: deploy F3+F4+F5`); loaded build `d8ce36dc9` (schema 8); 0 in-window startup markers | **MET** on the delivered definition: `slots_stale` **4.88 %** (608/12,450) < 5 % — 51 over-age (300 s ceiling) + 557 undatable. Tightened reading recorded alongside: age > 2 s on **31.60 %** of datable polls (p90 35.14 s, max 424.20 s; 81.3 % of those prefill-bound). `saturated`-without-live-work 11, all attributed. Record: `docs/dev/slots-stale-postfix-window-2026-11-18-to-19.md` |

**Lesson recorded by that run:** anchor the window at the **first whole minute
after** the restart. Pinning the raw restart instant (`15:20:47`) leaves the
anchor's own `Logging initialised … pid=` line (two seconds later) inside the
window, which fails the restart-free evidence rule in § 1 even though no
subsequent restart occurred.

### 3.2 Items with a shorter post-deploy window (same batching rule)

| Item | What it measures | Required deploy commit(s) | Window it expects | Note |
|---|---|---|---|---|
| **LP-0MVUVHBZH007INL1** | release-lag fix: residual `saturated`-without-live-work polls over a night | deploy recorded in `deployment.txt` at the 2026-11-01T13:17:48Z restart | `2026-11-01 20:00 → 2026-11-02 07:00` | The 2026-11-01T21:26 and 2026-11-02T10:38/10:39 restarts fall **inside** this window, making it mixed-build — **re-measure the next night** (`2026-11-02 20:00 → 2026-11-03 07:00`) against the current anchor |
| **LP-0MVWBXDZA00494HK** | live verification: skip signal replaces unloaded-model fallback noise (slices 2-3) | `579e2cc3` + `19d3c927` — **in the loaded anchor build** | a live post-deploy window against the current anchor | Open; runs against the current anchor without a further restart |
| **LP-0MVXY8RY8009OG7I** | post-deploy local stream share and refusal decomposition (≥ 30 min) | `f602491a` (LP-0MVXWQQ20002HAX4) — **in the loaded anchor build** | ≥ 30 min, ≥ 3 concurrent prefer-local sessions | Available now against the current anchor (shorter than 24 h, so it need not wait for a 24 h window) |
| **LP-0MVZ3V9T800872B8** | re-measure a restart-free night window and record the attribution verdict | fix from LP-0MVYZEYZY003IQAH (child); record `git log -1` at the anchor | one closed night window, proxy started **before** it, provably restart-free | **Blocked** on the sibling fixes |

> **Fix owners are not separate window rows.** LP-0MVVI8TEU006KTOZ and
> LP-0MVJXV0MY000O77E each require a ≥ 24 h post-fix window in their ACs, but the
> measurement is carried by **LP-0MVVLQMG1000843Q** (row above), which anchors on
> the deploy of those fixes. Do not arm a second window for the same deploy.

### 3.3 Recently closed (for context, not actionable)

- **LP-0MVVLKPBM007F6U6** — `/health`-fix router-500 window
  `2026-11-01T00:41:55Z → 2026-11-02T00:41:55Z`; measured **RED** (non-zero),
  completed/in\_review. Its window also contained the 2026-11-01 restarts, so
  its verdict is "not 0", not a restart-free pass.
- **LP-0MVVJYJQ9001I8QK** — post-deadline re-dispatch 24 h window, completed.

---

## 4. Batching convention — deploys land before a window opens

1. **Read this calendar before any deploy that restarts the proxy.** The
   dispatcher / operator checks which windows are open and which fixes are
   already loaded.
2. **Batch.** Collect every pending fix that needs a restart (as of
   2026-11-26T17:00Z: **nothing is merged-but-not-loaded** — the checkout tip
   `989358278` equals the anchor's loaded build, so a restart now would load no
   code; the 2026-11-26 restart deployed the single behaviour change
   `989358278`, LP-0MWKQJHSM000CI6W), deploy them in **one** restart, and then
   open **one** shared window from that anchor. Do not restart once per item.
3. **Never deploy inside an open window.** A deploy that must land mid-window is
   either deferred to the window's end, or it explicitly re-anchors and
   invalidates the named windows below — recorded as such, never silently.
4. **One window, many measurements.** A single restart-free window serves every
   item whose fix was loaded at the anchor. Two items with the same loaded build
   must not each arm a separate window.
5. **Attribute the restart.** Always
   `proxy/scripts/start-proxy.sh --restart --reason "<item id>: <what>"`. A
   restart without `--reason` is recorded as `source=manual actor=unknown` and
   must be triaged (§5).
6. **Re-arm, don't pretend.** If a restart lands inside a window, re-arm the
   affected items to `last restart + 24 h` and record both windows; a partial
   window never yields a PASS.

### Arming checklist

- [ ] Confirm the fix is in the **loaded** build (`git log -1` at the anchor;
      a pushed commit is not a deployed fix).
- [ ] Record `ps -o pid,lstart -p $(pgrep -f 'uvicorn proxy.server:app')` and the
      `Proxy restart attribution:` line.
- [ ] Pin start = first whole minute after the restart; end = start + 24 h.
- [ ] Record the window here (item, anchor, end, loaded build).
- [ ] Re-arm on any restart inside the window, naming the restart.

---

## 5. Manual-restart investigation convention

An unattributed restart — `Proxy restart attribution: source=manual
actor=unknown reason=manual (no reason provided)` — is **investigated, not
ignored**. The outcome is either an attribution or the explicit verdict
*cannot be attributed with available evidence* (both are acceptable; silence is
not).

Procedure, in order, stopping as soon as a confident attribution is reached:

1. **Pin the facts.** Restart timestamp, new pid, `ps -o lstart= -p <pid>` (if
   still alive), and the pre-restart log file (`Rotated previous proxy log
   aside: …`).
2. **Distinguish kill vs graceful stop.** A missing graceful-shutdown line plus
   a `start-proxy.sh --restart` signature (`pkill` of the proxy/TTS) means an
   external restart; a systemd unit restart would leave a journal entry.
3. **Check the journal and schedulers.** `journalctl --since/--until` around the
   timestamp; absence of a unit restart rules out systemd. Check cron/timers for
   anything that restarts the proxy.
4. **Check the agents.** Session logs under
   `~/.pi/agent/sessions/` whose mtime brackets the restart; the worklog
   comments around the timestamp; and the per-child locks under
   `.worklog/locks/child-<id>.lock`, which record the holder's `session_id` and
   `started_at` (a lock written before the restart and released after it
   identifies the active implement session even when the transcript has been
   pruned).
5. **Correlate the code.** Identify what the restart **deployed**: a log field or
   behaviour that appears only after the restart pins the commit that was
   loaded (e.g. new `status_request` fields). This is the strongest
   code-level fingerprint.
6. **Record it.** Write the attribution (or the cannot-be-attributed verdict)
   into the owning work item and into §6 below, with the evidence chain and any
   caveat (e.g. pruned transcript).

---

## 6. Restart #2 attribution (AC3) — 2026-10-31T00:37:53Z, pid 3938219

**Verdict: attributed (reconstructive, strong).** Restart #2 was the **deploy
restart for the LP-0MVSD5RJW008F1IB implementation** (commits `1a7bffdb` and
`fad873c7`), issued without `--reason` by that item's implement session
`herdr-1793401807-3797165-32424` (author `Map`). It is *not* an unexplained
incident.

Evidence chain:

1. **Timeline.** `fad873c7` ("LP-0MVSD5RJW008F1IB: Implementation complete") was
   committed `2026-10-31 00:36:31Z`; the push was commented at
   `2026-10-31T00:37:16Z`; restart #2 landed at `00:37:53Z`, 37 s later.
2. **The item's own note.** Its completion comment states that the shipped
   `status_request` enrichment (`prefill_eta_kind`, `prefill_eta_calibrated`,
   `prefill_eta_samples`, …) *"only appear once the proxy is restarted with this
   code"* — i.e. a restart was the expected next step to observe the new fields.
3. **The code fingerprint.** Those enriched fields are **absent** in
   `proxy.log.2026-10-31_003749` (the pre-restart window, 0 occurrences) and
   **present from the first `status_request` after the restart**
   (`2026-10-31 00:37:56`, 138 occurrences in the next few minutes). The restart
   therefore loaded `fad873c7`'s code — it was that item's deploy.
4. **The active holder.** `.worklog/locks/child-LP-0MVSD5RJW008F1IB.lock` names
   session `herdr-1793401807-3797165-32424` (started `00:10:03Z`), the implement
   session live at the restart.
5. **Restart shape.** The pre-restart proxy has **no** graceful-shutdown line and
   the TTS watchdog logged `TTS server exited with code=0` + self-heal at
   `00:37:49` — the `pkill` signature of `start-proxy.sh --restart`, consistent
   with an external manual restart rather than a crash or a systemd action (no
   journal entry exists).

**Caveat.** The session transcript for `herdr-1793401807-3797165-32424` has since
been pruned (oldest retained session in that directory is 2026-10-31T20:20Z), so
the exact command line is not retained. The attribution rests on the
log/timeline/code-fingerprint/lock correlation above, which is strong but
reconstructive.

**Contrast with the neighbouring restarts.**

- **Restart #1** `2026-10-30T23:10:49Z` (pid 3797968) — LP-0MVVJYJQ9001I8QK
  deploy, `--reason` given.
- **Restart #3** `2026-10-31T00:41:57Z` (pid 3944027) — the `/health` probe fix
  deploy (LP-0MVVCXUDZ007XID0, `ac8dc504`), the deploy anchor recorded by
  LP-0MVVLKPBM007F6U6.

So all three restarts that invalidated re-arm #3 are attributable: two were
deliberate measurement deploys and one (restart #2) was a third deploy (the
LP-0MVSD5RJW enrichment) whose `--reason` was omitted. The lesson is the
convention in §4.5: **always pass `--reason`**, so the next such restart is
attributed automatically rather than reconstructed.

---

## 7. Relationship to the 24 h measurement rules

- This calendar is **read-only** with respect to the proxy's scheduling
  behaviour: no knob re-tuning, no code change.
- It **reduces the frequency** of in-window restarts; it does **not** weaken the
  rule that a restart inside a window invalidates that window.
- Expired / stale windows are re-armed by their **owning items'** re-arm cycles,
  which should cite this file as the coordination record (producer decision:
  future-only scope).
- Machine-readable tooling (a gate in `start-proxy.sh`, a JSON store, linting)
  is deliberately **out of scope** for mechanism A. If the calendar proves
  insufficient, file that as a linked follow-up rather than growing this item.

### 7.1 Measurement safety — the llama log may live in a rotated sibling

A restart that **preserves** llama-server (the `--keep-llama-server` default)
rotates `llama-server.log` while the adopted backend still holds the
**pre-rotation inode**. Its output then continues in the renamed sibling
(`llama-server.1.log`), and the watched `llama-server.log` stays frozen —
`/health` shows `llama_log_source_healthy: false` and `llama_log_age_seconds`
keeps growing. Rotation is now deferred while a live writer holds the file
(LP-0MVZ4UDF10029NDS), so this should not occur after that fix is deployed, but
a window measured on an older build can still have its llama-side data in a
rotated sibling.

Before trusting a window's llama log, resolve the file the running backend is
actually writing to:

```bash
readlink /proc/$(pgrep -f 'llama-server.*--models-preset' | head -1)/fd/1
```

If that path is not `…/llama-server.log`, read the window's llama data from the
reported sibling (and record the substitution with the window's evidence).
Never restart llama-server merely to "fix" the log: preserving the warm prompt
cache is deliberate; restart only via `/skill:start-proxy`.


## 8. Maintenance

- Refresh §2 and the calendar table whenever a restart lands or an item is
  re-armed.
- Keep the anchor honest: record the **loaded** build, not the moving checkout
  tip.
- When a window closes, record the verdict on the owning item and remove it from
  §3 (or move it to §3.3).
