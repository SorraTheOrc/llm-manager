# Slot Thrashing Investigation Findings

**Date:** 2026-06-23
**Work Item:** LP-0MQR75QYM001HAUB (Slot thrashing reproduction & evidence)
**Parent:** LP-0MQR0780Z006TLX6 (Improve slot management)

## Executive Summary

Slot thrashing at the proxy level is **confirmed and reproducible**. With `pool_size=1` (the current production configuration) and ≥2 concurrent sessions, **95–98% of all turn transitions** result in a different session acquiring the slot — meaning a session's KV cache is almost certainly invalidated between every turn.

This is not a GPU-level issue; it's a proxy-level coordination gap. The `SlotLockCoordinator` serializes concurrent requests through the same asyncio.Lock, but between turns the lock is released and any waiting session can immediately acquire it. No ordering or affinity mechanism exists.

## Reproducibility

### Repro Script

A standalone, self-contained repro script is provided at:

```
docs/dev/slot-thrashing-investigation/repro.py
```

Usage:
```bash
python3 docs/dev/slot-thrashing-investigation/repro.py --sessions 4 --turns 10 --json
```

The script:
- Imports no production code; mirrors `SlotLockCoordinator` logic exactly
- Simulates N concurrent sessions making multi-turn requests with realistic timing
- Captures every lock acquire/release event with high-resolution timestamps
- Detects and quantifies inter-session slot stealing
- Reports gap windows, hold times, and estimated save/restore cost

### Requirements

- Python 3.10+
- No live GPU needed
- No test harness dependencies

## Evidence

### Steal Rate vs. Pool Size (4 sessions, 10 turns each)

| Pool Size | Steal Rate | Gap Window (avg) | Notes |
|-----------|-----------|-------------------|-------|
| 1 | 95–98% | ~0.03ms | **Production config** — near-total thrashing |
| 2 | 90% | ~7.8ms | Still severe: 2 slots for 4 sessions causes many collisions |
| 3 | 78% | ~6.3ms | Moderate improvement |
| 4 | 40% | ~5.9ms | Significant improvement: one slot per session |
| 8 | 43% | ~5.3ms | More slots than sessions doesn't help (hash collisions) |

### Steal Rate vs. Session Count (pool_size=1)

| Sessions | Steal Rate | Total Turns | Total Steals |
|----------|-----------|-------------|-------------|
| 1 | 0% | 10 | 0 |
| 2 | 95% | 20 | 19 |
| 4 | 98% | 40 | 39 |
| 8 | 98% | 40 | 39 |

### Lock Hold Times (pool_size=1, 4 sessions)

| Metric | Value |
|--------|-------|
| Min hold | ~10ms |
| Max hold | ~31ms |
| Avg hold | ~23ms |
| Overhead (save+restore) | ~10ms/turn |
| Overhead vs response time | 47–79% |

### Gap Window Analysis (pool_size=1, 4 sessions)

| Metric | Value |
|--------|-------|
| Min gap | ~0.004ms |
| Max gap | ~0.061ms |
| Avg gap | ~0.024ms |
| Median gap | ~0.025ms |

Gap windows are near-zero because asyncio's cooperative multitasking schedules the next waiting task immediately upon lock release. There is no OS-level preemption gap. The lock is released, and the next waiting `lock.acquire()` completes in the same event-loop iteration.

### Control: Single Session (pool_size=1, 1 session, 10 turns)

- **0 steals** — confirming the lock works correctly for single-session use.
- The session holds the same slot for all 10 turns with no interleaving.

## Analysis

### Why Thrashing Occurs

1. **pool_size=1** means all sessions hash to `slot_id=0` (SHA-256 mod 1 = 0)
2. **SlotLockCoordinator** creates one `asyncio.Lock` for slot_id=0
3. The lock serializes concurrent requests through slot 0
4. **But**: when a response is complete, the lock is released
5. **Between turns**: any waiting session can immediately acquire the lock
6. The cooperative asyncio event loop means the next scheduled task gets it — there's no ordering, affinity, or backoff

### Why pool_size>1 Helps (Partially)

With `pool_size=4`:
- Sessions distribute across 4 slots (via SHA-256 hash mod 4)
- Sessions on different slots don't contend
- But 4 sessions across 4 slots still has collisions (40% steal rate vs 98%)

With `pool_size > session_count`:
- No additional benefit — hash collisions still occur
- Extra slots don't help if no session is bound to them

### Cache Invalidation Cost

When a session loses its slot between turns:
1. The proxy must issue `POST /slots/{id}?action=save` to save current KV cache
2. Next time the session acquires the slot, it issues `POST /slots/{id}?action=restore` to reload cache
3. Both operations involve disk I/O and HTTP round-trips to llama-server
4. Configured timeout: `session_slot_timeout_seconds: 3.0`

**With 95–98% steal rate, every session pays save+restore cost on every turn** — even if no other session is actively using the GPU.

## Architectural Context

The proxy has three independent slot-related mechanisms:

| Mechanism | Location | Purpose |
|-----------|----------|---------|
| `SlotLockCoordinator` | `session.py:579` | asyncio lock per slot_id; serializes requests to same slot |
| Slot save/restore | `session.py:476-516` | KV cache persistence to disk via llama-server API |
| `_check_slot_availability` | `router_helpers.py:488` | Pre-route check of llama-server /slots endpoint |

The thrashing problem is in **`SlotLockCoordinator`** — not in slot save/restore (which works correctly) or slot availability check (which works correctly).

## Edge Cases and Boundary Conditions

### Inconclusive Reproduction

All runs were conclusive: thrashing is always reproducible with `pool_size=1` and ≥2 sessions.

### Race Condition Window

The gap between lock release and next acquire is typically **<0.1ms** — far smaller than any cache save/restore latency. This means:
- There is effectively no "safe window" where the original session can reclaim the slot
- Every session must save its cache, and the next session must restore/reload its own

### Worst-Case Analysis

Worst case: all 8 sessions active simultaneously, each making sequential multi-turn requests. Every turn of every session incurs save+restore overhead. With `session_slot_timeout_seconds: 3.0`, a single slow save can hold up all other sessions.

## Key Files (Touch Points)

The following sources were read during the investigation:

- `proxy/proxy/session.py` — `SlotLockCoordinator` (line 579), `_save_slot_snapshot` (line 498), `_restore_slot_snapshot` (line 476), `_slot_id_for_session` (line 526)
- `proxy/proxy/router.py` — slot lock acquire (lines 244, 811), slot save/restore (lines 758, 816, 983)
- `proxy/proxy/router_helpers.py` — `_check_slot_availability` (line 488)
- `proxy/config.yaml` — slot configuration (lines 206-209)
- `proxy/tests/test_slot_polling.py` — existing slot polling tests (not modified)

## Instrumentation Touch Points

The repro script (`repro.py`) is self-contained and mirrors `SlotLockCoordinator` logic without modifying production code. No permanent instrumentation was added to the proxy.

If a future implementation wishes to add runtime logging, the touch points would be:

1. `proxy/proxy/session.py` — `SlotLockCoordinator.acquire()` (line 579): add logging of acquire/release with session_id and timestamp
2. `proxy/proxy/session.py` — `_save_slot_snapshot()` (line 498): add logging of save start/end with slot_id and duration
3. `proxy/proxy/session.py` — `_restore_slot_snapshot()` (line 476): add logging of restore start/end with slot_id and duration

## Related Open Items

- **LP-0MQMC4MKY006J08E** (Prompt-cache / session reuse tests & small fixes) — directly related; F1 evidence confirms the cache-invalidation scenario this item aims to address
- **LP-0MQMC4MNU002QJK4** (Cleanup: slot-cache retention & cleanup script) — slot persistence is orthogonal but affects save/restore reliability
- **LP-0MQ0PYH8P008DLPJ** (Web based logging per slot) — instrumentation touch points identified above would support this

## Conclusion

Slot thrashing at the proxy level is **confirmed** for all tested configurations with `pool_size=1` and ≥2 concurrent sessions. The current `SlotLockCoordinator` provides mutual exclusion but does NOT provide session-level slot affinity, leading to near-100% cache invalidation between turns.

Reference data files:
- `repro.py` — Self-contained reproduction script
- This document — Analysis and findings

---

# 2026-08-07 Addendum: Slot save/restore ReadTimeouts persist under concurrent load

**Work item:** LP-0MSI1RWLM007N367 (Slot save/restore ReadTimeouts persist under concurrent load, ~1.8% of saves)
**Feature:** F1 — Confirm mechanism & document root cause (LP-0MSI5B1SW00222XV)
**Analysis basis:** live proxy logs `/var/log/llama-proxy/proxy.log*` and `llama-server.log*`, 2026-08-06

## Hypothesis under test

The parent plan's preliminary RCA: **slot save/restore requests to llama-server are
starved/queued behind busy slot work under concurrent load**, so the proxy's httpx
ReadTimeout fires at exactly the computed adaptive window. Free-slot saves complete in
~100ms; starved saves exceed the timeout; the next request re-attempts the doomed save
~30s later.

## Evidence (proxy logs, 2026-08-06 full day)

`scripts/slot-persistence-correlate.py --start "2026-08-06 00:00:00" --end "2026-08-07 00:00:00"`:

| Metric | Value |
|--------|-------|
| slot_save failed / success | 38 / 2178 (1.71%) |
| slot_restore failed / success | 19 / 1920 (0.98%) |
| Failures with ≥1 concurrent local stream | **52 of 57 (91%)** |
| Failures with no concurrent local stream | 5 |
| Max concurrent local streams at a failure | 3 |
| llama-server `prompt_save` lines | 21 (no timestamps — see below) |

### Finding 1 — Failures cluster under concurrent local load (confirmed)

52 of 57 failures occurred while ≥1 other local stream was active. Example windows:

- **01:57-01:58 (slot 2):** 3 consecutive failures at 01:57:19 / 01:57:50 / 01:58:15 with
  **4 concurrent local streams** (slots 0+2 busy). Meanwhile slot 0's session
  (`019fd490`) saves+restores completed in ~100ms each (e.g. 01:57:17,589 save →
  01:57:17,694 restore).
- **13:31-13:33 (slot 2):** 3 consecutive failures at 13:31:36 / 13:32:20 / 13:33:26 during
the heavy opencode-go fallback burst (4000+ `local_concurrency_limit` fallbacks counted
across the day's rotated proxy logs).
- **22:44-22:45 (slot 1, after the 22:23:59 restart):** restore failed 22:44:28, save failed
  22:45:07 while 3-5 local streams were active. The in-memory circuit breaker was cleared
  by the restart, so the first failures after restart were not suppressed.

### Finding 2 — Failures time out at EXACTLY the adaptive window (confirmed)

Consecutive failures within a burst repeat at the adaptive-timeout cadence: gaps of
25-70s (01:57:19→01:57:50=31s, 13:31:36→13:32:20=44s, 23:27:52→23:29:02=70s, 02:07:15→
02:07:48=33s), matching `3.0s + 0.001s/token × est_tokens` for est 22K-67K tokens. The
proxy **waits the full computed window before giving up** — the copy did not "fail fast",
it was starved for the entire window.

### Finding 3 — Circuit breaker works but is in-memory only (confirmed)

Bursts of exactly 3 failures are followed by a ~540s gap (01:58:15 → 02:07:15,
02:08:20 → 02:21:59), i.e. the 300s cooldown plus the ~240s of the three timeout waits.
After cooldown expiry the proxy retries — and fails again while load persists (02:07:15
→ 02:07:48 → 02:08:20 → cooldown → ...). The restart at 22:23:59 cleared the breaker,
which is why 22:44/22:45 failures reappeared.

### Finding 4 — llama-server logs lack timestamps (instrumentation gap)

llama-server logs 21 `prompt_save` lines (240-340 MiB KV serializations) but **no
timestamps**, so direct timing correlation is impossible from those logs alone. F1 adds
proxy-side elapsed-time instrumentation (`elapsed=`, `timeout=`, `busy=` fields on every
failed save/restore in `_call_slot_endpoint`) so subsequent windows can be quantified
precisely.

## Conclusion: save-starvation hypothesis CONFIRMED

The residual ReadTimeouts are **proxy-side timeout/coordination bound**: under concurrent
local load, llama-server serializes each KV-cache copy behind its slot work, the save is
starved for the entire adaptive window, and the proxy ReadTimeout fires. Free-slot saves
are near-instant; starved saves exceed the timeout. This matches all four parent RCA
findings and adds the load-correlation numbers above.

## Decision record (for F3 implementation)

1. **Busy-signal definition:** skip save/restore when the proxy-side local backend is
   under load. The signal is `local_active_queries > 0` (equivalently
   `active_sessions > 0` from `_slot_busy_state_snapshot`), i.e. **any other local stream
   active at the moment the persistence call would be issued**. Rationale: 91% of failures
   occur with ≥1 concurrent stream; the 5 no-load failures are the tail of bursts where the
   load had just drained.
2. **Timeout rebalance alone would NOT suffice.** The proxy already waits the full adaptive
   window before failing; raising the coefficient/cap would only extend the same doomed
   waits and re-introduce the GPU-wedge exposure LP-0MS91DHPZ001VWQO fixed. The load gate is
   the primary fix; the timeout rebalance is belt-and-braces only.
3. **Config shape (F3):** `session_slot_skip_when_busy: true` in `proxy/config.yaml`
   (enabled by the deployed config; the code gates only when the key is set true,
   so `_build_slot_context` is not coupled to ambient global state). Busy threshold
   derives from the proxy-side snapshot (F1 instrumentation, `active_sessions > 0`),
   applied in `_build_slot_context` so both save and restore paths are gated
   identically. Context-size gate and circuit breaker stay unchanged.

## Mitigation implemented (F3, 2026-08-07)

The approved recommendation (d) was landed in commit `9acfebc`:

- **Load-aware gate (primary):** `_build_slot_context` now returns `(None, None, timeout)`
  (skips save/restore, logs `reason=slot_busy`) when another local session is actively
  streaming — busy signal = `active_sessions > 0` from dispatch-lease state, with the
  requesting session's own lease excluded. Applied uniformly to save and restore.
- **Conservative timeout rebalance (belt-and-braces):** per-token coefficient 0.001 →
  0.0015; cap unchanged at 60s so the circuit breaker still trips within a bounded
  window (no GPU-wedge regression).
- **Unchanged:** context-size gate, adaptive timeout scaling, circuit breaker
  (3 failures → session-slot 300s cooldown), per-slot `SlotLockCoordinator` serialization.

Post-deploy confirmation: track the F4 7-day observation window via
`scripts/slot-persistence-failures.sh` / `slot-persistence-correlate.py`.

## F1 verification-deliverables

- **Instrumentation** in `_call_slot_endpoint`: failed save/restore logs `elapsed=`, `timeout=`,
  and `busy={active_queries, local_active_queries, active_sessions, slot_busy}`; success
  logs `elapsed=` at DEBUG.
- **Correlation script** `scripts/slot-persistence-correlate.py` (or use the
  `--json` output) reproduces the cadence/load analysis above from `/var/log/llama-proxy`
  for any future window.

---

# 2026-08-15 Addendum: 7-day post-fix observation window complete

**Work item:** LP-0MSI1RWLM007N367 (F4 observation window, parent AC2/AC3)

The 7-day post-fix window (2026-08-08 → 2026-08-14, fix released v0.1.11 on
2026-08-08) has elapsed. Full rows + analysis in
[`observation-log.md`](observation-log.md):

- **save failure rate 0.59%** (67/11267) vs 1.71% baseline → **−65.5%**
- **restore failure rate 0.06%** (6/10000) vs 0.98% baseline → **−93.9%**
- 79% of residual failures still cluster under ≥1 concurrent local stream
  (58/73) — same confirmed save-starvation mechanism at reduced rate.
- Residual ReadTimeouts are the **mid-request-onset class**: the load gate
  samples busy state at request start, but the save executes at request end,
  so concurrency that onsets during a request escapes the gate. This is the
  documented residual path; failures are bounded (≈⅓ baseline rate) with
  load context and a non-load residual (6 ConnectError during llama-server
  restarts + 1 restore-400 capacity error).
- **Verdict: parent AC2 satisfied via the bounded/residual-failures branch**
  of the exit criteria (explicit rationale + load context, vs the ~1.8%
  baseline). AC3 verification evidence posted as a work-item comment.

---

# 2026-10-24 window replay: residency-aware admission (executable evidence)

**Work item:** LP-0MVN1IQTF0037WTG (child LP-0MVN4LB5W005WJ71, AC7 integration
half). Driver: `proxy/scripts/replay_prefill_occupancy.py` (new
`--residency-replay` / `--window-fixture` modes); tests:
`proxy/tests/test_prefill_thrash_replay_2026_10_24.py`.

The replay feeds every captured attempt through the **production** ladder
(`provider._should_skip_local` + `proxy.residency.ResidencyView`), not a copy of
its logic, so these numbers cannot drift from the shipped decision.

## 1. The divergence case, measured both ways

`proxy/tests/fixtures/slot_thrash_2026-10-24/tasks_92383_92422.log` (sanitised
extract of `/var/log/llama-proxy/llama-server.log-2026-10-25`, pid 41395):

| task | slot | prompt tokens | `n_past` (measured reuse) | reuse ratio | ladder verdict |
|---|---|---|---|---|---|
| 92383 | 0 | 49,103 | 738 | 1.5 % | **`resident_prefix_insufficient`** |
| 92422 | 1 | 49,103 | 49,103 | 100 % | **admit** |

The same prompt, seconds apart, measured cold then fully resident: slot churn,
not GPU incapacity. Predicted-ratio pricing (sidecar 0.99) admitted the first
attempt and burned its whole prefill deadline; the residency gate refuses it and
keeps the retry.

## 2. The pinned window, replayed end to end

`proxy/tests/fixtures/prefill_thrash_2026_10_24.jsonl` (2026-10-24
18:38–23:47 UTC, 1,978 local admission attempts):

```
attempts replayed:                1978
  over the cold budget:           1960
  admitted:                         51
  idle-exempt (reported apart):     33
  resident_prefix_insufficient:   1927
over-budget cold admissions:         0 (must be 0)
```

Zero over-budget cold admissions without a resident prefix. The 33 idle-exempt
admissions (`route=idle_bypass`) are reported separately: an idle GPU spends
capacity that would otherwise be wasted, so they are not violations — but they
must stay visible, hence the separate bucket. The 18 remaining admits are
prompts below the 25,000-token budget, which route local as before.

Modelling note: the capture has no per-slot prefill progress
(`prefill_tokens_computed: null` — the `/slots` unobservability documented in
`LP-0MUXTW4MD003TVIC`), so each attempt is replayed as what it measured: a
session owning no slot, i.e. a measured zero prefix. That is the conservative
assumption for an AC whose claim is "cold prompts above the budget from
non-resident sessions are never admitted".

# 2026-10-28 Addendum: 24 h AC4 verification (re-arm #2) — FAIL (attribution blind)

**Work item:** LP-0MVN44XB0009M33K (follow-up of LP-0MVN1IQTF0037WTG AC4/AC5).
**Verdict: FAIL / not pass.** The window-scoped measurement succeeded, but the
AC7 observability check failed as a *finding* (all residency decisions
unmeasured), which AC7 defines as "never a pass". This addendum supersedes the
2026-10-24 numbers as the current measurement; the 2026-10-24 figures below
remain **historical reference only**, never a baseline.

## Window (restart-free, attested)

The first pinned window (`2026-10-26T21:02:00Z → 2026-10-27T21:02:00Z`) was
invalidated by two proxy restarts inside it (`2026-10-27T09:33:12Z` pid
3775724 and `2026-10-27T10:16:12Z` pid 3791954, both `source=manual
actor=unknown`). Re-armed per producer decision A to the latest restart + 24 h:

- **Measured window:** `2026-10-27T10:17:00Z → 2026-10-28T10:17:00Z` (start =
  first whole minute after the 10:16:12 restart).
- **Restart-freedom:** PID 3791954, `ps -o lstart` = `Tue Oct 27 10:16:11
  2026`, continuous uptime `1-00:23:07`; the only restart marker in the six
  overlapping files is the 10:16:12 anchor, **before** the window start.
- **Code identity:** shared checkout tip `1908f431` at run; ladder
  (`_should_skip_local` + `ResidencyView` + `routing_resident_prefix`) present;
  config `local_resident_prefix_min_ratio: 0.5`,
  `local_cold_prefill_budget_tokens: 25000`,
  `local_prefill_deadline_seconds: 120`.

## Corrected capture invocation (the documented one omitted `--log`)

`capture_prefill_thrash_window.py` requires `--log`; the earlier snippet in the
work item omitted it. The working, window-scoped invocation (every disjoint
file, including the daily rotation) is:

```bash
python3 proxy/scripts/capture_prefill_thrash_window.py \
  --start "2026-10-27 10:17:00" --end "2026-10-28 10:17:00" \
  --log /var/log/llama-proxy/proxy.log.2026-10-27_10 \
  --log /var/log/llama-proxy/proxy.log.2026-10-27_16 \
  --log /var/log/llama-proxy/proxy.log-2026-10-28_00 \
  --log /var/log/llama-proxy/proxy.log.2026-10-27_22 \
  --log /var/log/llama-proxy/proxy.log.2026-10-28_04 \
  --log /var/log/llama-proxy/proxy.log \
  --out /tmp/ac4_window_rearm2.jsonl
python3 proxy/scripts/replay_prefill_occupancy.py x \
  --window-fixture /tmp/ac4_window_rearm2.jsonl
```

## Measured result

| Metric | Value |
|---|---|
| `prefill_deadline_exceeded` | **99** (user 85 / audit 14) |
| Wasted prefill tokens | **750,042** (overstated per LP-0MVODJ3P6000AIUM: fields recomputed post-hoc) |
| `prefill_budget_exceeded` refusals | 2,546 |
| Provider exhaustion / upstream 503 | 483 / 100 |
| `residency_decisions` | 3,607 (**measured 0 / unmeasured 3,607**) |
| `residency_resident_admits` | 0 |
| `residency_evicted_denials` | 2,113 |
| `residency_tenant_limit_denials` | 15 |
| slot-owner churn | 31 of 100 sessions |

**AC4 split** (join: same session's most recent `routing_resident_prefix` at/
before each deadline event + the `local_eviction` owner at/around it; ambiguous
events are counted separately and NOT attributed — conservative):

| Class | Count |
|---|---|
| Scheduling-attributed (rule b only; rule a unevaluable) | 22 |
| Ambiguous (no join / owner `none`) | 57 |
| Genuine GPU-speed miss | 20 |

**Window-scoped baseline** (capture → replay through the real ladder):
12,105 attempts, 136 deadline-cancelled, ~631,946 wasted tokens modelled;
replay → 11,238 over budget, 11,224 `resident_prefix_insufficient`, 881
admitted, 14 idle-exempt, **0 over-budget cold admissions (must be 0)**.

## Why FAIL — AC7 finding is decisive

The usage report's `## Residency-aware admission` section and churn table
render, but **every one of the 3,607 decisions is unmeasured**: even
`decision=admit` / `residency_source=session_slot` (234 resident decisions)
logs `n_past_actual=none / reuse_ratio=none`. AC7 states that "decisions logged
but all unmeasured is a finding (ladder not emitting), never a pass". As a
consequence, AC4 rule (a) (`reuse_ratio < 0.5`) could not fire for any of the
99 events; 57 events are ambiguous and the ≥ 90 % reduction claim cannot be
computed rigorously against a window-scoped scheduling-attributed baseline.

AC6 gate (ii) — no `local_lease_active` refusal window longer than the 180 s
watchdog — is **met on a per-session basis** (52 refusals; max
`queued_duration` 60.09 s; max per-session refusal span 79.8 s), with one
aggregate multi-session contention episode of 181 s flagged as marginal.

Follow-up: **LP-0MVRZXPH70025ZNR** (`discovered-from:LP-0MVN44XB0009M33K`) —
populate the measured prefix at admission so AC4 rule (a) and AC7 become
evaluable; the 24 h verification re-runs on a fresh restart-free window after
it lands.

## Re-arm #3 — the attribution gap is closed (LP-0MVRZXPH70025ZNR)

The follow-up landed on `dev` on 2026-10-30 (`0e270c1f`, `0ff1d7fe`,
`d463a35a`, `2d91ab27`, `07d0ef40`, with the regression suites `5b2d5fb5`,
`77291663`). The change is measurement-only — no residency threshold, tenancy
bound or pricing rule moved:

- a new rung between the live `/slots` `context_tokens` measurement and the
  ownership-only `session_slot` sentinel reports the last measured dispatch's
  `total` for the `(model, session)` pair as
  `residency_source=measured_last_dispatch` with a **numeric**
  `n_past_actual` / `reuse_ratio`;
- every decision now carries an additive `measurement=` field: `measured`, or a
  closed unmeasured reason (`unavailable:no_slot`,
  `unavailable:no_prior_measurement`, `unavailable:stale_view`,
  `unavailable:capture_disabled`), so `reuse_ratio=none` is no longer the only
  signal;
- captured measurements are invalidated on eviction, slot release and model
  switch, and the rung requires live slot ownership, so a stale ratio cannot be
  fabricated;
- the usage report's `## Residency-aware admission` section reports
  `residency_measured_decisions` plus a per-reason `residency_unmeasured_reasons`
  breakdown, so the "decisions logged but all unmeasured" condition that made
  this window a FAIL is now a visible finding rather than a silent zero.

**Re-arming rule (unchanged in spirit, restated):** the 24 h window starts at
the first whole minute after a proxy restart that loads the fix (a running
uvicorn does not reload code), runs exactly 24 h, and the run instant is the
window end. Arming is performed through the start-proxy skill, never by killing
the proxy. The full procedure, the arming evidence to record and the expected
measurement shape for re-arm #3 are in
[`docs/dev/resident-prefix-rollout.md`](../../resident-prefix-rollout.md) §
*Re-arming the 24 h AC4 window (re-arm #3)*.

**Status:** re-arm #2 stays a FAIL and its numbers above remain the last
complete measurement. The re-arm #3 window is armed against the first fix-loaded
restart after 2026-10-30T07:00:00Z (deliberately after the concurrent
`LP-0MVRMI5HK00558TB` overnight window closes, so that measurement is not
split); no PASS may be claimed until that window closes and is measured.
