# Per-slot details in `GET /llama/local/status`

LP-0MSORPUMX002LLIA — proxy-side enabler for herdr same-slot idle tracking
(ContextHub WL-0MSG7P9N8009PCKG).

## What changed

`GET /llama/local/status` now includes a `slots` array in addition to the
`available_slots` / `total_slots` counts:

```json
{
  "available_slots": 1,
  "total_slots": 2,
  "slots": [
    {"slot_id": 0, "is_processing": true,  "n_decoded": 872},
    {"slot_id": 1, "is_processing": false, "n_decoded": 860}
  ]
}
```

- Each slot dict is compact: `slot_id` (int), `is_processing` (bool),
  `n_decoded` (int|null) — no streaming state or session identifiers.
- Fetched from llama-server `/slots?model=<current>` via the existing
  `_query_slots_detail()` helper (`proxy/proxy/observability.py`), wired in
  `get_llama_local_status` (`proxy/proxy/handlers.py`).
- Bounded by the `STATUS_QUERY_TIMEOUT` window (default 1.0s) via the
  helper's own httpx timeout, so a slow `/slots` response cannot blow the
  endpoint's response budget.
- Fail-open: `slots` falls back to the last-known per-slot detail cache
  (updated by the periodic broadcast/SSE loops) when the fresh query fails
  or returns nothing, so slot identity survives a transient `/slots`
  failure (LP-0MUFSVXID0039ZAQ). It is an empty array only when
  llama-server is not running, no model is loaded yet, or no cached detail
  exists — never a malformed payload. `total_slots` / `available_slots`
  behavior is unchanged.

## Graceful degradation on `/slots` failure (LP-0MSVP7XJ6008QPKX)

After a cheap-mode restart, llama-server's `/slots` endpoint can return
HTTP 500 while the model reloads. Before this fix the status endpoint
surfaced that as `total_slots=0`, fail-closing the herdr downtime worker
(`isIdleStatus` false → zero dispatches overnight, silently).

Now:

- `_query_slots()` records the last successful `(available_slots,
  total_slots)` counts (module-level cache in `proxy/observability.py`).
- On a fresh `/slots` failure (HTTP error, timeout, connection error, or
  unexpected payload), the status endpoint serves the last known counts
  instead of 0/0, bounded by `SLOT_COUNTS_STALE_AFTER_SECONDS` (default
  3600s; env-configurable) so a genuinely-unavailable llama-server
  eventually fail-closes rather than serving stale capacity forever.
- The payload and `status_request` log gain a `slots_stale` boolean that is
  `true` whenever the served slot data (counts or per-slot detail) came from
  the cache (vs a fresh query), so a future silent failure is observable.
- Per-slot detail degrades the same way (LP-0MUFSVXID0039ZAQ): when the
  fresh `_query_slots_detail()` call fails or returns an empty list, the
  handler serves `_last_slot_details_cache` instead of `slots: []`, preserving
  herdr per-slot identity; an empty cache still yields a well-formed empty
  array. A cached-detail response sets `slots_stale: true`.
- Every failed query increments `llama_slots_query_failures_total{reason=...}`
  (Prometheus counter), alerting via `monitoring/slots_query_alerts.yaml`
  when the failure rate is sustained.

## Stale log-derived progress (LP-0MVECWYWK000U3E1)

The `n_decoded` value served for a slot can be supplemented from
`llama-server.log` progress lines (`slot update_slots: ... n_tokens = N
progress = P`) when the `/slots` endpoint is unresponsive during busy
generation. That supplemental path used to pin the UI at stale token counts
forever when the log stopped being written (e.g. an adopted llama-server
whose stdout pipe was never re-attached):

1. `_update_slot_progress_from_log()` re-parsed the same frozen tail every
   cycle and stamped each entry with `time.time()` (the *read* time), so the
   values never aged past the 60 s prune window.
2. The periodic loop aliased `_last_slot_details_cache` to the list it then
   enriched in place, permanently baking the stale counts into the fallback
   cache.
3. Enrichment replaced `n_decoded` whenever the log value was larger — always
   true for stale data.

Now:

- Log-derived entries are stamped with the log file's **mtime** (clamped to
  `now` if the mtime is in the future), not the read time. A frozen log
  self-expires after `SLOT_PROGRESS_MAX_AGE_SECONDS` (60 s) and leaves no
  progress data behind.
- `_slot_progress_is_fresh()` is the single freshness gate used by the prune,
  enrichment, and processing-slot assignment. A stale entry is rejected even
  if injected directly (AC3).
- `_resolve_slot_details_for_enrichment()` stores a pristine copy of the raw
  `/slots` values in `_last_slot_details_cache` and returns a **detached
  copy** for enrichment, so log-derived fields can never poison the fallback
  cache. Both the periodic broadcast loop and the SSE `status_events()` path
  use it.
- A status response served from the cache (`slots_stale: true`) therefore
  reports only genuine `/slots` values — never a log-derived `n_decoded`
  greater than the live value (AC5).
- With a fresh log the override still works exactly as before (AC4), so the
  original unresponsive-`/slots` use case is preserved.

## Why

herdr's downtime worker (ContextHub `packages/herdr/src/downtime-worker.ts`)
needs to dispatch into PARTIAL-idle states (N of M slots free) while
guaranteeing the SAME slots stay free for the full idle threshold. The
counts-only endpoint could not provide per-slot identity, so `evaluateIdle`
deliberately degraded any `0 < N < total` to "ALL slots free". The `slots`
array delivers the missing per-slot identity.

## Live verification (2026-08-14)

The new code was run live (proxy on :8000 from the implementation worktree,
llama-server :8080 router-mode with Qwen3 loaded, 2 active slots) and
herdr's real consumer code
(`parseLlamaStatus` / `isIdleStatus` / `evaluateIdle` from ContextHub
`packages/herdr/src/downtime-worker.ts`, run via tsx) was exercised against
the live endpoint. Results:

| Check | Result |
|-------|--------|
| `parseLlamaStatus` accepted the new payload on all 4 polls | PASS |
| `isIdleStatus(s, 0)` / `evaluateIdle(s, 1)` ran on parsed live status | PASS (idle=false while a slot processed; idle=true when both free) |
| Free slot_ids across 4 polls (15 s window) | `[[1],[1],[0,1],[0,1]]` |
| Same-slot intersection (SAME slots free every poll) | `[1]` — slot 1 stayed free the whole window while slot 0 processed |
| Every slot dict compact `{slot_id, is_processing, n_decoded}` | PASS |

The intersection result is the key AC6 demonstration: during polls 1–2 the
counts-only payload reported `available_slots=1/2` but could not say WHICH
slot was free; the `slots` array shows it was slot 1, and that the same
slot remained free for the full window — exactly what herdr's same-slot
idle tracking (WL-0MSG7P9N8009PCKG) needs.

Repeatable harness: `RUN_LIVE_SLOTS_VERIFY=1 pytest tests/test_slots_live_verify.py -v`
(guarded live test, skipped by default).

Note: herdr's shipped `evaluateIdle` still applies its documented Q7
degradation (`N < total` ⇒ require ALL slots free) until the ContextHub
consumer item WL-0MSG7P9N8009PCKG lands; this item delivers the
proxy-side enabler the consumer will build on.

---

# Truthful local capacity: slot phase + prefill ETA (LP-0MVGDUVGE0024YAI)

`GET /llama/local/status` now publishes enough per-slot detail for a
dispatcher to tell a slot that is **actually working** from one that is merely
**reserved but idle**, and to price the wait before it launches a session. The
change is additive and versioned (`status_schema_version: 3`); a client that
ignores the new fields behaves exactly as before (contract test:
`test_new_capacity_fields_are_additive_and_versioned`). v3
(LP-0MVKSXR4P001V0CM) adds stale-detail reconciliation of `capacity_hint` and
truthful in-flight accounting. v4 (LP-0MVRMI5HK00558TB) adds the abandoned-work
annotation, the `abandoned_busy` hint value and the top-level
`local_abandoned_slots` count.

## Payload (new fields)

```json
{
  "status_schema_version": 4,
  "total_slots": 2,
  "available_slots": 0,
  "contention_queue_depth": 3,          // existing, elsewhere in the payload
  "capacity_hint": "prefill_bound",
  "local_abandoned_slots": 0,
  "prefill_budget_seconds": 120.0,
  "queue_eta_seconds": 240.0,
  "slots_stale": false,
  "slots_detail_stale": false,
  "slot_counts_stale": false,
  "local_in_flight_requests": 1,
  "slots": [
    {"slot_id": 0, "is_processing": true, "n_decoded": 0,
     "context_tokens": 102345, "owner_session_id": "…",
     "lease_state": "generating", "phase": "prefill",
     "prefill_eta_seconds": 187.0, "abandoned": false,
     "abandonment_reason": null, "abandoned_session_id": null,
     "abandoned_at_seconds_ago": null},
    {"slot_id": 1, "is_processing": true, "n_decoded": 41,
     "context_tokens": 4096, "owner_session_id": null,
     "lease_state": "generating", "phase": "decode",
     "prefill_eta_seconds": null, "abandoned": true,
     "abandonment_reason": "prefill_deadline",
     "abandoned_session_id": "herdr-…", "abandoned_at_seconds_ago": 12.4}
  ]
}
```

Per-slot fields (all additive):

| Field | Type | Meaning |
|---|---|---|
| `owner_session_id` | `str \| null` | Session holding a live dispatch lease on the slot (active, or post-stream cooldown). `null` when unowned. |
| `lease_state` | `"generating" \| "idle_leased" \| "free"` | `generating` whenever `is_processing` is true (never `free` while a prefill or decode is in flight); `idle_leased` when not processing but a live lease still reserves it; `free` otherwise. |
| `phase` | `"prefill" \| "decode" \| null` | Inferred proxy-side (see below); `null` when the slot is not processing. |
| `context_tokens` | `int \| null` | Prompt progress (`max(n_past, n_prompt_tokens_processed)`) when the llama.cpp build reports it, else `null`. |
| `prefill_eta_seconds` | `float \| null` | Shared calibrated prefill cost of the slot's loaded context, only for a slot in `prefill`; `null` otherwise or when not computable. |
| `abandoned` | `bool` | `true` when the proxy gave up on the dispatch occupying this slot (see [Abandoned work](#abandoned-work-lp-0mvr5hk00558tb-ac1)) and llama-server has not released it yet. Always present. |
| `abandonment_reason` | `str \| null` | Which abandonment gave the slot up: `prefill_deadline`, `session_evicted`, `no_progress_watchdog`, `client_disconnect` (also `orphan_cleanup` / `measured_cache_deficit` from the sweep and the early abort). `null` when `abandoned` is `false`. |
| `abandoned_session_id` | `str \| null` | The session that was abandoned (the slot's own identity is already `slot_id`). `null` when `abandoned` is `false`. |
| `abandoned_at_seconds_ago` | `float \| null` | Age of the abandonment, so a consumer can reason about how long a ghost may persist (the registry TTL is 900 s). |

Top-level fields: `capacity_hint`, `prefill_budget_seconds`,
`local_abandoned_slots` (count of `abandoned: true` slots),
`status_schema_version` (plus the existing `prefill_eta_seconds` /
`queue_eta_seconds`).

## `lease_state` (AC2)

- `generating` — `is_processing` is true. Takes precedence, so a working slot is
  **never** reported `free`.
- `idle_leased` — not processing, but the slot's session still holds a live
  dispatch lease (active, or inactive inside its post-stream cooldown window,
  `expires_at` in the future). This is the reserved-but-idle case behind 2,633
  of 2,641 `local_dispatch_denied` events while local decoded only 29.3 % of
  wall clock.
- `free` — not processing and no live lease. An **expired** inactive lease
  does not keep a slot `idle_leased`.

Source: `_build_status_slot_owners()` joins the slot registry
(`proxy.session._slot_owners`, `(endpoint, slot_id) → session`) to the
per-session dispatch records (`srv.local_dispatch_records`), normalising keys
through `_dispatch_key_session_id()`.

## `phase` inference (AC5)

llama-server exposes no phase field, so it is derived proxy-side from
`/slots` signals (`infer_slot_phase()`):

- not processing → `null`;
- processing and `next_token.n_decoded` is `None` or `0` → `prefill` (prompt
  progress, when present, corroborates — including the transition where
  `n_past` advances while `n_decoded` stays at zero);
- processing and `n_decoded > 0` → `decode`.

The rule is deliberately conservative for very short prefills: a processing
slot with no decode signal is reported `prefill`, never guessed as `decode`.

## ETA fields (AC3) — one source of truth

Every ETA comes from the shared calibration / admission guard in
`proxy/proxy/prefill_eta.py`; there is no second estimator.

- per-slot `prefill_eta_seconds` — `prefill_eta.per_slot_prefill_eta_seconds()`,
  a thin wrapper over `estimate_prefill_seconds()` (cache-cold, the conservative
  case used by the published reference estimate). `null` when the slot is not
  in `prefill`, or when `context_tokens` is unknown (modern llama.cpp builds
  dropped `n_past`/`n_prompt_tokens_processed`) — never a misleading `0`.
- `queue_eta_seconds` / `prefill_eta_seconds` (top level) — unchanged, from
  `prefill_eta.status_estimate()`.
- `prefill_budget_seconds` — the admission guard's own wait budget
  (`prefill_eta.wait_budget_seconds()`, i.e. the contention-queue max wait);
  `null` when no budget is resolvable.

## `capacity_hint` decision table (AC4)

Evaluated **in order** (`compute_capacity_hint()`):

0. `abandoned_busy` — no slot is free and **nothing** generating is live
   (proxy-owned) work, i.e. every working slot belongs to a dispatch the proxy
   abandoned (LP-0MVRMI5HK00558TB AC1). Busy, but not ours.
1. `idle` — at least one slot is `free` **or** every occupied slot is
   `idle_leased` (nothing is actually generating).
2. `prefill_bound` — no slot is free and at least one occupied slot is in
   `prefill`.
3. `saturated` — no slot is free and every occupied slot is in `decode`.

Two documented extensions: an empty `slots` array (server down / no model / no
cached detail) is `idle` (fail-open — never claim saturation without evidence),
and a pool with no free slot, no prefill and mixed `decode` / `idle_leased`
occupancy is `saturated` (genuine decode work, nothing available to a newcomer).

Rule 0 is conservative in one direction only: it fires **only** when no
generating slot is live, so live work alongside an abandoned slot keeps the
previous verdict. An abandonment never discounts real work, and slots without
the `abandoned` annotation behave exactly as before.

## Abandoned work (LP-0MVRMI5HK00558TB AC1)

An *abandoned* dispatch is one the proxy stopped waiting for while
llama-server still had it — the prefill deadline expired, the session was
evicted, the no-progress watchdog fired, or the client disappeared. Before this
fix the proxy only closed the *downstream* connection, which llama-server acts
on at its next write, so the abandoned prefill ran to completion and a later
fresh `/slots` poll reported the slot as `is_processing` with
`local_in_flight_requests=0` — counted as `saturated`, i.e. live work nobody
was waiting for. Measured 2026-10-24 20:00 → 2026-10-25 07:00: 332
abandonments left **247** such polls (2.12 %).

Two mechanisms now apply, per reason (details and the log lines are in
[routing.md](routing.md#abandoned-local-dispatches-lp-0mvr5hk00558tb-ac1)):

1. **Cancel the upstream task** wherever a handle exists (the deadline and
   session-eviction paths now close the captured upstream stream, as the
   watchdog and client-disconnect paths already did). llama-server aborts at
   its next socket write, and `return_progress` guarantees writes during
   prefill, so the ghost window is short rather than the whole prefill.
2. **Annotate what remains.** The abandoned slot identity (endpoint, slot,
   session, reason, timestamp) is recorded in-process with a 900 s TTL, and the
   status payload publishes it. A slot the backend has not released yet is then
   *busy, not ours*: `capacity_hint` is `abandoned_busy` — **never**
   `saturated` (that must keep meaning live, proxy-owned decode work) and
   **never** `idle` (a busy-but-abandoned pool folded into `idle` would admit a
   newcomer behind the ghost and make the tail worse).

The annotation is explicit and attributable (slot identity + reason + session),
never a blanket discount of llama-server processing: a fresh snapshot whose
slots are genuinely live still reports `saturated`.

The table is authoritative only while the served `/slots` detail is **fresh**.
When `slots_detail_stale` is true, the cached `is_processing`/`phase` describe
*past* work, so the hint is reconciled with live proxy state
(`downgrade_stale_capacity_hint`, LP-0MVKSXR4P001V0CM AC3):

1. **No live in-flight local request** (`local_in_flight_requests == 0`,
   derived from active dispatch records — a retained lease or a leaked
   counter is *not* work) → `idle`, and `available_slots` is reconciled
   upward to the pool size (fail-open, matching the no-model-loaded branch).
2. **In-flight work with fresh decode evidence** (proxy forwarded a chunk
   within `STALE_DECODE_EVIDENCE_MAX_IDLE_SECONDS`) → stays `saturated`.
3. **Any other in-flight work** → `prefill_bound` — never `saturated` on
   stale data.

This is the gate that starved herdr's downtime worker on 2026-10-22: a cached
busy snapshot served for hours while the GPU was idle, and the leaked
`local_active_queries` counter kept `local_active_query` true. Both are now
truthful.

## Freshness honesty (AC7)

`slots_stale` semantics are unchanged (counts **or** per-slot detail came from
the last-known cache); the per-source flags `slot_counts_stale` and
`slots_detail_stale` tell the two apart. When the served slot detail is stale,
**both per-slot and top-level ETA fields are `null`** — a dispatcher never acts
on a stale wait estimate. `prefill_budget_seconds` is a configuration constant
(not poll-derived) and stays truthful.

## Truthful in-flight accounting (LP-0MVKSXR4P001V0CM AC1/AC2)

- `local_active_query` is driven by **active dispatch records**, not the
  `local_active_queries` counter: a retained post-stream lease (record
  `active: false`) or a leaked counter never reports local work in flight.
  Legacy proxies without the record registry keep the counter fallback.
- `local_in_flight_requests` publishes the active-record count itself.
- `local_prefills_in_flight` publishes only prefills backed by a live dispatch
  record (LP-0MVKUKFSO008GI6O AC3) — an orphaned marker is not published.
- The 10 s `_dispatch_cleanup_loop` now **clamps** `local_active_queries` to
  the active-record count instead of only resetting when no records are
  active, so a leak that coexists with real work cannot persist (AC2).

## Consumer contract — herdr `downtime-worker.ts` (AC6)

The consumer change lives in ContextHub
`packages/herdr/src/downtime-worker.ts` and is a separate ContextHub work item
(referenced from this item's description); this document defines the contract it
gates on. Its `parseLlamaStatus` / `LlamaSlot` parser notes already accept an
optional per-slot `owner_session_id`; extend them to also read:

- `slots[].lease_state` — prefer this over "non-empty `owner_session_id` ⇒
  busy". A slot is genuinely busy only when `lease_state === "generating"`;
  `idle_leased` is reserved-but-idle (the false-busy case); `free` is available.
- `slots[].phase` and `slots[].prefill_eta_seconds` — price the wait for a
  prefilling slot instead of treating all busy time as equal (prefill dominates
  hold time: p50 61.8 s, p90 269.8 s).
- `capacity_hint` — a one-field summary: `idle` (safe to dispatch),
  `prefill_bound` (a prefill is in flight), `saturated` (all occupied slots are
  decoding), `abandoned_busy` (the pool is occupied only by work the proxy has
  abandoned — busy, not ours).
- `slots[].abandoned` / `abandonment_reason` / `abandoned_session_id` and the
  top-level `local_abandoned_slots` (LP-0MVRMI5HK00558TB) — for reporting and
  for distinguishing a ghost from live work without re-deriving it.
- `prefill_budget_seconds` — the admission guard's budget, so the dispatcher
  paces to the same number the proxy enforces.

**Unknown hint values must be treated as busy.** The hint set is additive
(`abandoned_busy` was added in `status_schema_version` 4 and more may follow),
so a consumer must never assume the three values it knows are exhaustive: an
unrecognised value means *do not dispatch* — the conservative direction. Because
the herdr `downtime-worker.ts` contract admits **only** on `idle`, a new
non-`idle` value is already conservative for it, so **no ContextHub change is
required** for `abandoned_busy` (only the optional reporting reads above, if
wanted). The alternative — folding abandoned work into `idle` — was rejected:
it would admit a dispatch behind a ghost and lengthen the tail.

All of these are optional: the existing `is_processing` / `n_decoded` /
`available_slots` reads keep working unchanged (AC1).

## Post-fix measurement (2026-10-24 20:00 → 2026-10-25 07:00, Europe/Dublin)

What the reconciliation above actually did on the first post-fix night
(LP-0MVLNBCTN0084MT0, reported by LP-0MVOG7IX6006MZ23). The measured window is
~12 real hours, not 11: Irish summer time ended at 02:00 on 2026-10-25, so the
local label `01:00–01:59` occurs twice inside it. The deployed code was
`56a3a28a`/`2d2b6b51` (live from 2026-10-24 10:35); the child-port resolver
(`f58e8964`, 2026-10-26 12:51 UTC) post-dates the window, so **every** `/slots`
poll measured here traversed the llama-server router.

| Metric | Measured | Target |
|---|---|---|
| Status polls | 11,654 | — |
| `slots_stale=true` | 9,278 (79.61 %) | < 5 % → **missed**, see (3) |
| `capacity_hint` `idle` / `prefill_bound` / `saturated` | 4,577 (39.27 %) / 6,024 (51.69 %) / 1,053 (9.04 %) | — |
| `saturated` with no live local work | 247 (2.12 %) | 0 → **missed**, all attributed — see (2) |
| `saturated` with `available_slots=0` | 890 (7.64 %) | 82.3 % pre-fix control |
| `slots_detail_stale=true` | 9,012 of the 9,278 stale polls | — |
| `proxy_restart` events in window | 0 | — |

What the numbers say about this contract:

1. **Stale detail never produced a false `saturated`.** Of the 9,012
   detail-stale polls, the 2,927 with no live local work were all reconciled to
   `idle` (rule 2 of `downgrade_stale_capacity_hint`) and the rest read
   `prefill_bound` (6,024) or `saturated` with live work (61). Zero polls
   claimed saturation from a cached busy snapshot — the false-signal class that
   starved herdr's downtime worker on 2026-10-22 is measured at zero.
2. **The 247 `saturated`-without-live-work polls are fresh detail, not stale.**
   llama-server kept finishing dispatches the proxy had already abandoned: 332
   abandonments in the window (`prefill_deadline` 260, `session_evicted` 60,
   `no_progress_watchdog` 11, `client_disconnect` 1). Every violating poll is
   attributed to an abandonment within the preceding 600 s
   (`prefill_deadline` 204, `session_evicted` 43, unattributed 0). That is a
   dispatch-lifecycle issue (deadline vs. upstream cancellation), not a
   capacity-signal one — the reconciliation rules do not apply to a genuine
   fresh snapshot.
3. **`slots_stale` is a polling-path metric, not a capacity-truthfulness
   metric.** Its semantics are unchanged by the fix, and the rate tracks the
   `/slots` round-trip: the handler budgets are 2.0 s for the counts query and
   `STATUS_QUERY_TIMEOUT` (1.0 s) for the detail query, while router-serialised
   polls took p50 3,029 ms (detail-cache-served) against 3 ms for a fresh poll.
   The per-hour rate follows GPU load (44.3 % at 03:00, 96.0 % at 02:00), which
   a truthfulness defect would not. The < 5 % target can only be judged on a
   build that contains `f58e8964`; measured here it stays at the pre-fix level
   (79.5 %) by construction.

Re-derivation (LP-0MVOG6YRA000T9PK tool): the artifacts are in
`~/proxy-usage-reports/2026-10-24_post-fix-night-measurement/` — `REPORT.md`,
`measurement.json` / `measurement.txt`, one row per violating poll in
`saturated_without_live_work_attribution.csv`, plus the hourly, crosstab and
dispatch tables — and are regenerated with:

```bash
python3 scripts/capacity_signal_verification.py \
  --since '2026-10-24 20:00' --until '2026-10-25 07:00' \
  --window-closed --label post-fix-night --limit 250 \
  --write-artifacts ~/proxy-usage-reports/2026-10-24_post-fix-night-measurement
python3 ~/proxy-usage-reports/2026-10-24_post-fix-night-measurement/supplementary_derivations.py
```

**Cross-check against the nightly usage report.** The nightly report covering
this window is `~/proxy-usage-reports/2026-10-25_2/report.md` (generated
2026-10-25 05:00, window 2026-10-24 05:00 → 2026-10-25 05:00; the top-level
`report.md` is overwritten by each 05:00 run, so the per-run directory is the
durable surface). The two surfaces agree where they overlap:

- **Local attempts.** The tool's `local_dispatches` (lines carrying
  `dispatch_first_byte_ms=`, i.e. dispatches that reached a first byte) totals
  **206** over the nightly window and **145** over the 20:00–05:00 overlap —
  exactly the nightly report's `Local attempts` (local `Stream started`)
  totals, 206 and 145.
- **Restarts.** The nightly restart timeline ends at 2026-10-24 10:35:06
  (before the window starts); the tool reports no `proxy_restart` events
  inside it.
- **`Local dispatch` is a different metric, not a disagreement.** The nightly
  column counts every `[local] POST` attempt (1,427 over its window, 692 over
  the overlap), including attempts that never delivered a first byte, so it is
  ~4.8× the first-byte count on the overlap (692/145).
- The nightly analyzer publishes neither `slots_stale` nor `capacity_hint`, so
  those figures have no nightly counterpart to compare against. Adding them is
  a change under `.pi/skills/proxy-usage-analysis/` (agent infrastructure, out
  of scope here) and is recorded as a follow-up rather than made silently.

## Post-fix re-measurements (12 h and 24 h, LP-0MVLNBCTN0084MT0)

Two further windows were measured with the same tool to judge the `< 5 %`
`slots_stale` target on a build that contains the child-port resolver
(`f58e8964`, 2026-10-26 12:51 UTC) instead of the router-serialised polling path
measured above.

| Window (local) | Build | Polls | `slots_stale` | `saturated` w/o live work | herdr downtime dispatches |
|---|---|---|---|---|---|
| 2026-10-24 20:00 → 10-25 07:00 (night) | `56a3a28a`/`2d2b6b51` (router path) | 11,654 | 9,278 (79.61 %) | 247 (2.12 %) | 5 |
| 2026-10-28 09:30 → 21:30 (12 h) | + `f58e8964` | 10,817 | 5,832 (53.92 %) | 116 (1.07 %) | 21 (rolling log; durable 21) |
| 2026-10-29 01:00 → 10-30 01:00 (24 h) | + `f58e8964`, HEAD `b12a0299` | 26,019 | **7,698 (29.59 %)** | **70 (0.27 %)** | **81 (durable trail)** |

What the 24 h window says:

1. **`slots_stale` still misses the `< 5 %` target (29.59 %), but it is now a
   latency measurement rather than a router artefact.** 77 % of the residual
   stale polls were served from cache after a refresh lasting longer than the
   budget (`STATUS_QUERY_TIMEOUT` 1.0 s detail / 2.0 s counts): ≥ 5 s 2,063,
   2–5 s 3,888, 1–2 s 1,637, < 1 s 110. The source split is unchanged in
   character — detail cache 7,224, counts cache 474, unknown 0. Removing the
   router roughly halved the rate (79.61 % → 53.92 %) and the resolver build
   roughly halved it again (→ 29.59 %); the remainder belongs to the `/slots`
   read path under load and is carried forward as LP-0MVVCO1N7004TNGY.
2. **False saturation is now rare and still fully attributable.** 70 polls
   (0.27 % of polls, 2.13 % of `saturated` polls) reported `saturated` with no
   live local work, down from 247; 65 are attributed to an abandonment or
   restart within the preceding 600 s (`prefill_deadline` 27,
   `no_progress_watchdog` 20, `session_evicted` 18) and 5 remain unattributable
   from the proxy log. The cache-time reconciliation contract above is intact:
   no stale poll produced `saturated`.
3. **Dispatch recovery is met.** 81 durable herdr downtime dispatches in the
   24 h window (target ≥ 10; kinds `plan` 31, `audit` 18, `intake` 18,
   `implement` 13, `effort-and-risk` 1; 53 distinct items). The rolling
   `.worklog/downtime-dispatches.log` retains only the last 100 rows and does
   **not** cover a 24 h window — audit the durable comment trail instead.
4. **Two proxy restarts occurred inside the window** (2026-10-29 12:58:34 and
   14:09:25); coverage was still `covered` with no internal gaps. Neither
   restart changed the qualitative result.

Re-derivation: `~/proxy-usage-reports/2026-10-29_24h-remeasure/` (`SUMMARY.md`,
`measurement.json` / `measurement.txt`, `saturated_without_live_work_attribution.csv`,
`downtime-dispatches-durable.csv`, `provenance.json`, `run_remeasure.py`):

```bash
python3 scripts/capacity_signal_verification.py \
  --since '2026-10-29 01:00' --until '2026-10-30 01:00' --window-closed \
  --label 24h-remeasure --limit 250 --json \
  --write-artifacts ~/proxy-usage-reports/2026-10-29_24h-remeasure
# or: python3 ~/proxy-usage-reports/2026-10-29_24h-remeasure/run_remeasure.py
```

