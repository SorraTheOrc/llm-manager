# Prefill-deadline residual: V1 label-split ≥ 24 h restart-free window

**Work item:** LP-0MWM0CP4B001088M (child 2 of the abort-class remediation epic
LP-0MWM0B7IQ008W2FP; AC1–AC6).
**Code under test:** the umbrella-label split LP-0MWKQJHSM000CI6W (commit
`989358278`), and nothing else.
**Measurement tool:** `proxy/scripts/analyse_prefill_deadline_residual.py`
(per-reason split, user/audit split, stall-weighted cost, release-context
classification, never-started share).
**Dataset (committed beside this document):**
`docs/dev/prefill-deadline-residual-v1-label-split-24h.json`, with the
admission denominators in
`docs/dev/prefill-deadline-residual-v1-admission-conversion.json` and
`docs/dev/prefill-deadline-residual-group2-admission-conversion.json`.
**Window:** `2026-11-26 10:18:00` → `2026-11-27 10:18:00` UTC exactly 24.000 h,
anchored on the single restart that deployed the split
(`2026-11-26T10:17:48Z`, pid `2364780`).

This is an **evidence** document. It changes no admission decision, no abort
policy, no threshold and no configuration; the only code change it ships is the
analyser surface that classifies the window (`release_context_split`), which is
offline tooling and cannot move what the proxy emits. The certification verdict
on these numbers is owned by LP-0MWM0CTPX002K0Z3 (closure verdict) against the
redesigned gate (LP-0MWM0CSDL008RB1E).

## 0. Result at a glance

The umbrella label is **gone**: 0 `no_progress_watchdog` evictions in the
window, against 184 in the group-2 baseline. The class it used to hide is now
**83 `prefill_eta_exceeded`, 5 `prefill_stalled`, 0 `prefill_max_exceeded`** —
i.e. the umbrella's 184 events were dominated by the runaway-ETA cut, exactly
the hypothesis LP-0MWNRBUUX0015S4A raised from the `lease_released` lines.

| Figure (this window) | Group-2 baseline | Reading |
|---|---|---|
| Post-acceptance evictions | **98** (4.083/h) | 193 in 26.987 h = 7.152/h |
| `no_progress_watchdog` (umbrella) | **0** | 184 (6.818/h) — label retired |
| `prefill_eta_exceeded` | **83** (3.458/h; user 41 = 1.708/h) | 0 on the eviction line (327 on `lease_released`) |
| `prefill_stalled` | **5** (0.208/h; user 3) | 0 — previously indistinguishable from the umbrella |
| `prefill_max_exceeded` | **0** | 0 |
| Watchdog class total (eta + stalled + max + umbrella) | **88** (3.667/h) | 184 (6.818/h) |
| Class per 1,000 `local_admission_guard` admits | **31.3** (user 15.7) | 38.7 (user 36.4) — inside the band |
| User stalled seconds | **2 125.6 s = 88.567 s/h** | 6 221 s = 230.556 s/h |
| Total stalled seconds (user + audit) | **4 619.8 s = 192.492 s/h** | 6 342.2 s = 235.0 s/h |
| Never-started (`llama_task_id_reason=task_id_not_started`) | **9 of 98 = 9.18 %** (all user) | **20 of 184 = 10.87 %** (recomputed 21/193 = 10.88 %) |
| Whole prompt priced as new compute (`new_tokens >= estimated_tokens`) | **11 of 98 = 11.22 %** | **96 of 184 = 52.17 %** (recomputed 98/193 = 50.78 %) |
| Events whose backend task is **not attributable** (`slot_id=unknown`) | **81 of 98 = 82.65 %** | **133 of 193 = 68.91 %** |
| Audit share of the class | **47 of 98 = 47.96 %** | 13 of 193 = 6.74 % |

Read in one line: the split works and the class is now diagnosable — the
overwhelming majority of it is the **runaway-ETA cut on a dispatch that never
started**, at a median of 15.3 s against a 180 s deadline, not a slow prefill.
The never-started share (9.18 %) is **comparable with the group-2 10.87 %**, and
the "fully cold" share (11.22 %) is **five times lower** than group-2's 52.17 %.
The one figure that did not improve is attributability: 82.65 % of the window's
aborts carry `slot_id=unknown`, so the stronger never-started diagnostic is
available for only 9 events.

### 0.1 The pass/fail line the producer set on 2026-11-25

The 2026-11-25 hand-off on this item replaced the `b2` counted-pair criterion
with the **reason-agnostic user-class stall-weighted rate** over *all*
post-acceptance abort reasons, and made the `b2` pair and the occupancy/capacity
attribution **diagnostics only**.

| Gate | Threshold | This window | Verdict |
|---|---|---|---|
| All-reasons **user-class** Σ`elapsed_s`/h | **≤ 6.06 s/h** (≥ 90 % below the group-1 60.572 s/h) | **88.567 s/h** (2 125.6 s over 24.0 h) | **NOT met** — 14.6× the threshold |
| Event base / noise guard | ≥ 10 user-class aborts across all reasons | **51** | met, no extension needed |
| Escalation trigger | first window above 30.29 s/h (half the group-1 baseline) → report as a finding instead of re-measuring | **88.567 s/h > 30.29 s/h** | **escalated** — no confirming window is armed; the finding is this report |
| `b2` pair (diagnostic) | — | `prefill_deadline` user 0 + `measured_cache_deficit` user 3 = **3 events, 0.125/h** | diagnostic: the counted pair is rare, and that is *not* the pass line |
| `prefill_deadline` attribution (diagnostic) | occupancy < 25 %, capacity < 50 % | 2 events: **occupancy 2, capacity 0, predicted-warmth 0** | diagnostic |

So the honest headline for the downstream verdict (**LP-0MVRMOV8K0007NFO** /
**LP-0MVO8H7VQ003B12I**) is: **the reason-agnostic criterion is not met, and it is
not a near miss** — 88.567 s/h against a 6.06 s/h gate. Because that is above
the 30.29 s/h escalation trigger, the producer's rule says to report it as a
finding rather than re-measure. The finding is: the class is dominated by a
**15.3 s-median early cut** (`prefill_eta_exceeded`, 83 events, 2 956.0 s), not
by the deadline-shaped stall the `b2` pair was built around.

## 1. Window certification (AC1, AC2)

The anchor is the **restart that loaded the fix**, not a commit date. Per the
calendar rule (`docs/dev/measurement-windows.md` § 1) the window starts at the
first whole minute after the restart, so the boot line itself sits outside it.

| Check | Evidence | Result |
|---|---|---|
| Restart (AC1) | `2026-11-26T10:17:48Z`, old pid `545169` → new pid `2364780`, `Logging initialised … pid=2364780` at `10:17:49,970` and `Starting LLama Proxy Server` at `10:17:49,975` | one restart |
| Proxy process (AC1) | `ps -o pid,lstart -p 2364780` → `Thu Nov 26 10:17:48 2026`, alive at analysis time (elapsed 1-07:00) | unchanged |
| Deployed HEAD (AC1) | `989358278` (committed `2026-11-26T04:47:16Z`); the next `dev` commit `91ad69d63` is dated `2026-11-26T17:09:23Z`, 6 h 51 m **after** the restart | `989358278` |
| Label split is an ancestor of the running tree (AC1) | `git merge-base --is-ancestor 989358278 HEAD` → exit 0; the split's three-line emission contract is visible on the window's first eviction (10:38:13, `reason=prefill_eta_exceeded … task_started=no progress_observed=no queue_wait_s=unknown`) | met |
| Window | `2026-11-26 10:18:00` → `2026-11-27 10:18:00`, **24.000 h** | ≥ 24 h → met (AC2) |
| Proxy restarts in window | analyser `source.restarts_in_window` = **0**; 1 `Starting LLama Proxy Server` in the whole log stream for the day, at `10:17:49` — 12 s *before* the window | 0 re-bases |
| Log continuity | `proxy.log.2026-11-26_10` (10:17:49 → 16:17:48), `…_16` (16:17:49 → 22:17:48), `…_22` (00:00:00 → 04:17:48), `proxy.log-2026-11-27_00` (22:17:49 → 00:00:00), `…_04` (04:17:50 → 10:17:45) | 639 936 lines, 44 files, no gap |
| llama-server death / self-heal | pids `3779734` / `3779811` / `3779815` up since `2026-11-22 13:47`; 0 `llama-server process died`, 0 `self-heal attempt`, 0 `watchdog router self-heal` lines in the window | none |
| TTS death | pid `2364824` up since `2026-11-26 10:17:50` (this window's own boot); 0 TTS death lines | none |
| Config reload | 0 `Configuration reloaded` lines | none |
| Death-window artefacts | no `death_window_*.json` with an mtime inside the window (all are from 2026-11-05) | none |
| `/health` at analysis time | `status=healthy`, `ready=true`, `llama_server_running=true`, `slots_surface_status=ok` | — |

Two honest notes:

1. **A failed start attempt precedes the anchor by 31 s.** pid `2364484` logged
   `Logging initialised` at `10:17:17` and stopped after 165 lines (its segment
   `proxy.log.2026-11-26_101745`, which also carries the old process's last
   27 s); the old pid `545169` served traffic until `10:17:45`. The window's
   boot — and the only process that ever served this window — is `2364780` at
   `10:17:48`. Neither segment before the anchor contributes a single eviction
   (the 12 s between `10:17:48` and `10:18:00` carry 125 lines, 0 evictions).
2. **The 24 h window is anchored at the restart, measured from the first whole
   minute.** The epic recorded the window as `10:17:48 → 10:17:48`; that variant
   contains exactly one restart line (`10:17:49`) and yields **identical**
   figures (98 evictions, 88 watchdog-class, 9 never-started). The whole-minute
   form is published because it is the documented rule and contains zero
   restart lines.

## 2. Exact command and basis assertion (AC3/AC5)

The dataset was produced by this command, verbatim, from the repository root of
the implementation worktree:

```bash
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-26 10:18:00" --end "2026-11-27 10:18:00" \
  --deadline-seconds 180 \
  --llama-log "/var/log/llama-proxy/llama-server*.log" \
  --json --output docs/dev/prefill-deadline-residual-v1-label-split-24h.json
```

This is the command form the 2026-11-25 hand-off mandates. The joiner is
enabled and reports `abandoned_prefill_windows.basis = measured_stop`,
`measured_available = true`, `seconds_basis_tok_s = 141.48` — but also
**`abandoned_prefill_windows.coverage.verdict = mismatch`** (98 releases against
7 350 backend task segments), because the `llama-server` backend has been up
since 2026-11-22 13:47 and its log stream is not aligned to a 2026-11-26 10:18
window boundary. Every figure in §0.1 and §3 is derived from **proxy log only**
(the stall cost's `source_fields` are `reason`/`session`/`elapsed_s` and the
classification reads the eviction line), so the coverage mismatch moves nothing
that is used for pass/fail; it is recorded so nobody reads the abandoned-prefill
rows as window-certified.

The admission denominators (AC4) were produced by:

```bash
python3 scripts/prefill_admission_conversion.py \
  --since "2026-11-26 10:18:00" --until "2026-11-27 10:18:00" \
  --json > docs/dev/prefill-deadline-residual-v1-admission-conversion.json

python3 scripts/prefill_admission_conversion.py \
  --since "2026-11-16 15:20:47" --until "2026-11-17 18:20:00" \
  --log-dir ~/proxy-usage-reports/contention-queue-postflip-7d/raw \
  --json > docs/dev/prefill-deadline-residual-group2-admission-conversion.json
```

The group-2 denominators are recomputed from the **retained raw archive**
(`~/proxy-usage-reports/contention-queue-postflip-7d/raw/`, the segments the
group-2 document was written from) because the live rotated logs for
2026-11-16/17 have aged out of `/var/log/llama-proxy`.

## 3. Classification of every post-acceptance abort (AC3)

98 evictions, **all 98 carrying the release context** (0 pre-split lines:
`events_with_release_context = 98`).

### 3.1 By emitted reason

| Reason | Total | user | audit | total/h | user/h | Σ elapsed_s | Σ elapsed_s/h |
|---|---|---|---|---|---|---|---|
| `prefill_eta_exceeded` | 83 | 41 | 42 | 3.458 | 1.708 | 2 956.0 | 123.167 |
| `prefill_stalled` | 5 | 3 | 2 | 0.208 | 0.125 | 912.2 | 38.008 |
| `prefill_deadline` | 2 | 0 | 2 | 0.083 | 0.0 | 383.8 | 15.992 |
| `measured_cache_deficit` | 3 | 3 | 0 | 0.125 | 0.125 | 138.3 | 5.763 |
| `lease_evicted` | 3 | 2 | 1 | 0.125 | 0.083 | 161.4 | 6.725 |
| `idle_timeout` | 2 | 2 | 0 | 0.083 | 0.083 | 68.1 | 2.837 |
| **`no_progress_watchdog`** | **0** | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prefill_max_exceeded` | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Total** | **98** | **51** | **47** | **4.083** | **2.125** | **4 619.8** | **192.492** |

The watchdog class AC3 is about (`prefill_eta_exceeded` + `prefill_stalled` +
`prefill_max_exceeded` + the umbrella remainder) is **88 events = 3.667/h**
(user 44 = 1.833/h), **Σ 3 868.2 s = 161.175 s/h**.

### 3.2 By `task_started`

| `task_started` | Events | Share | user | audit | Σ elapsed_s |
|---|---|---|---|---|---|
| `yes` | 8 | 8.16 % | 6 | 2 | 736.3 |
| `no` | 90 | 91.84 % | 45 | 45 | 3 883.5 |
| `unknown` (pre-split) | 0 | 0.0 % | 0 | 0 | 0.0 |

`task_started=no` is **not** the never-started class, and the window shows why.
Splitting the 90 by whether the slot was resolvable at all:

| `task_started=no`, slot… | Events | Share | user | audit | `llama_task_id_reason` | Σ elapsed_s |
|---|---|---|---|---|---|---|
| **known** (task id genuinely absent) | **9** | 9.18 % | 9 | 0 | `task_id_not_started` × 9 | 222.8 |
| **unknown** (not attributable) | **81** | 82.65 % | 36 | 45 | — | 3 660.7 |

So the honest reading is: 9 events (9.18 %) are **evidenced never-started**, and
81 (82.65 %) are **unattributable** — a weaker state that the group-2 baseline
also carried (133 of its 193 evictions, 68.91 %, had `slot_id=unknown`) but
which the old report never separated out.

### 3.3 By `progress_observed`

| `progress_observed` | Events | Share | user | audit | Σ elapsed_s |
|---|---|---|---|---|---|
| `yes` | 6 | 6.12 % | 4 | 2 | 567.9 |
| `no` | 92 | 93.88 % | 47 | 45 | 4 051.9 |
| `unknown` | 0 | 0.0 % | 0 | 0 | 0.0 |

### 3.4 By queue-wait bucket

| Bucket | Events | Share | user | audit | Σ elapsed_s |
|---|---|---|---|---|---|
| `zero` | 0 | 0.0 % | 0 | 0 | 0.0 |
| `(0, 1]` | 0 | 0.0 % | 0 | 0 | 0.0 |
| `(1, 5]` | 0 | 0.0 % | 0 | 0 | 0.0 |
| `(5, 30]` | 2 | 2.04 % | 2 | 0 | 318.7 |
| `> 30` | 0 | 0.0 % | 0 | 0 | 0.0 |
| `unknown` | 96 | 97.96 % | 49 | 47 | 4 301.1 |

Queue wait is measured on only 2 of 98 events. That is **not** evidence that the
class is uncontended: the field is stamped only on the guarded routing path
(`_contention_queue_wait_seconds`), and this window enqueued just 10 waits
(`queue_enqueued = 10`; 8 abandoned, 2 dispatched). The bucket table should be
read as "contention is rare in this window", with the `unknown` mass making no
claim either way.

### 3.5 Never-started share, comparable with group-2 (explicit AC3 ask)

| Figure | V1 (this window) | Group-2 baseline |
|---|---|---|
| `llama_task_id_reason=task_id_not_started` | **9 of 98 = 9.18 %** | **20 of 184 = 10.87 %** (recomputed from the archive: 21/193 = 10.88 %) |
| …user / audit | 9 / 0 | 20 / 0 |
| …Σ elapsed_s | 222.8 s (9.283 s/h) | not published |
| …by reason | `prefill_eta_exceeded` 7, `measured_cache_deficit` 2 | not published |
| Whole prompt priced as new compute | **11 of 98 = 11.22 %** (user 4, audit 7) | **96 of 184 = 52.17 %** (recomputed: 98/193 = 50.78 %) |
| Not attributable (`slot_id=unknown`) | 81 of 98 = 82.65 % | 133 of 193 = 68.91 % |

The never-started share is **stable** (9.18 % vs 10.87 %, a 1.7-point fall on a
9-event base). The "fully cold" share the never-started fix
(LP-0MWM0CR18001SB1U) targets is **five times lower** than group-2's — but this
window's tree carries 55 implementation commits that group-2's did not, so that
movement cannot be attributed to any one of them; it is recorded, not claimed.

### 3.6 Stall shape

| Figure (user class) | Value | Group-2 |
|---|---|---|
| Σ stalled seconds / h | **88.567** | 230.556 |
| p50 / p90 / p95 / p99 | 18.1 / 133.0 / 174.0 / 184.2 s | not published for the class |
| `prefill_eta_exceeded` p50 / p95 | 15.3 / 87.6 s | — |
| `prefill_stalled` p50 / p95 | 182.0 / 185.96 s | — |
| Aborts under 5 s | 10 of 98 (10.2 %) | — |
| Aborts at/near the 180 s deadline | 7 of 98 (7.1 %) | — |

This is the decisive shape result: `prefill_eta_exceeded` aborts at a **15.3 s
median** — the runaway-ETA cut firing early, exactly as
LP-0MWNRBUUX0015S4A predicted from the 327 `lease_released` lines. The genuine
"no progress at all" wedge is the 5-event `prefill_stalled` class, whose 182 s
median is the deadline-shaped stall the umbrella used to hide it inside.

## 4. Observability-only assertion and workload normalisation (AC4)

### 4.1 No decision, policy or threshold changed

Reviewed the label split's diff (`989358278`, 9 files, +540/−13):

| File | Change | Behavioural? |
|---|---|---|
| `proxy/proxy/admission_events.py` | three new reason constants; `_release_context_fields()` appends `task_started`, `progress_observed`, `queue_wait_s` **after** every pre-existing key; `EVICTION_REASONS*` tuples extended | **no** — emission only; no branch reads the new values |
| `proxy/proxy/router_helpers.py` | the deferred watchdog release passes the reason the sweep already computed (`watchdog_reason`) instead of the literal umbrella; the queue wait is stamped on the record at admission | **label only** — the `lease_released` line already carried `watchdog_reason`; no ETA factor, no timeout, no ceiling, no admission threshold is touched |
| `proxy/scripts/analyse_prefill_deadline_residual.py` | reason constants (+22 lines; this item adds `release_context_split`) | **no** — offline analyser |
| `docs/dev/prefill-watchdog-eviction-reasons.md`, `docs/llama-router.md` | contract documentation | no |
| 3 test files + `test_watchdog_reason_split.py` | assertions on the line contract and the two-line reconciliation | no |

No admission decision, abort policy or threshold changed. The window's behaviour
is the deployment's behaviour; the only new thing the proxy does is *say more*
on a line it already emits.

### 4.2 Class rate per 1,000 dispatched local attempts

Denominator: `local_admission_guard decision=admit` (the routing gate's admitted
local dispatches). The window's non-guard admission paths (`routing_idle_admit`
13, `local_last_resort_admit` 4) are reported but excluded from the denominator,
as are the 19 185 rejects.

| Normalised figure | V1 | Group-2 | Band |
|---|---|---|---|
| Admitted local dispatches | 2 809 (117.0/h) | 4 756 (176.2/h) | — |
| Watchdog class per 1,000 admits | **31.33** | **38.69** | inside |
| …user only | **15.66** | **36.38** | below the band |
| All evictions per 1,000 admits | 34.89 | 40.58 | inside |

The all-traffic class rate sits **inside** the group-2 band (31.3 vs 38.7 per
1 000 admitted local attempts, −19 %). The user-only rate is **well below** it
(15.7 vs 36.4), and the reason is workload mix, not a user-specific
improvement: audit sessions are **47.96 %** of this window's class (47 of 98)
against **6.74 %** in group-2 (13 of 193), while total admission volume fell
34 % (117.0/h vs 176.2/h). Normalising per admitted attempt removes the volume
effect and leaves the mix effect, which is why both rows are published.

**Deviation explained, and its limit stated:** the running tree carries **61
commits** (`79f53ac1a..989358278`, 55 of them implementation commits) between
the group-2 measurement and this window. The class rate moving from 38.7 to 31.3
per 1 000 attempts is therefore the *deployment's* movement, not the split's own
share — the split only changes labels. Nothing in this document claims the split
reduced the class; it claims the class is now diagnosable, and the diagnosis
(§3.6) is that the dominant member is an early runaway-ETA cut.

## 5. What this window changes for the rest of the chain

- **The downstream verdict has its answer and it is negative.** The
  reason-agnostic user stall-weighted rate is **88.567 s/h against a 6.06 s/h
  gate**, which is also above the 30.29 s/h escalation trigger. Per the
  2026-11-25 producer decision, LP-0MVRMOV8K0007NFO and LP-0MVO8H7VQ003B12I must
  record the criterion as **not met** and carry this as a finding, not arm a
  confirming window.
- **The never-started fix (LP-0MWM0CR18001SB1U) now has its target defined from
  this window, not group-2:** 9 evidenced never-started events (all user,
  `prefill_eta_exceeded` 7 / `measured_cache_deficit` 2) plus 81 unattributable
  ones whose `slot_id` is unknown. Improving slot attribution is a prerequisite
  for measuring that fix, and it is not owned by this item.
- **The runaway-ETA bound (LP-0MWNRBUUX0015S4A) is confirmed in the class
  itself,** not just on the `lease_released` line: 83 of 88 watchdog-class
  evictions are `prefill_eta_exceeded`, p50 15.3 s against a 180 s deadline.
- **The gate redesign (LP-0MWM0CSDL008RB1E) must count reasons separately.**
  A single "abort" count over this window prices a 15.3 s early cut exactly like
  a 182 s wedge; the stall-weighted split in §3.1 and §3.6 is the shape that
  separates them.

## 6. Re-anchoring (AC6)

Restart anchor for this window: **`2026-11-26T10:17:48Z`** (pid `2364780`),
window `2026-11-26T10:18:00Z → 2026-11-27T10:18:00Z`. The later windows in the
sequence are re-anchored to the close of this window (`2026-11-27T10:18:00Z`) —
the earliest instant at which a new restart can be scheduled without voiding a
measured window — and the run-not-before titles are updated
old (`2026-11-19T10:25:00Z`) → new (`2026-11-27T10:18:00Z`). The per-item
changes are recorded in the work-item comments.

## 7. Rollback

This item ships no proxy behaviour. Reverting it removes the analyser's
`release_context_split` block and this document; the label split
(`989358278`) stays deployed and its emission contract is unchanged. No
configuration value is involved and no restart is needed to roll back.
