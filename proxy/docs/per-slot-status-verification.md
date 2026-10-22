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
change is additive and versioned (`status_schema_version: 2`); a client that
ignores the new fields behaves exactly as before (contract test:
`test_new_capacity_fields_are_additive_and_versioned`).

## Payload (new fields)

```json
{
  "status_schema_version": 2,
  "total_slots": 2,
  "available_slots": 0,
  "contention_queue_depth": 3,          // existing, elsewhere in the payload
  "capacity_hint": "prefill_bound",
  "prefill_budget_seconds": 120.0,
  "queue_eta_seconds": 240.0,
  "slots": [
    {"slot_id": 0, "is_processing": true, "n_decoded": 0,
     "context_tokens": 102345, "owner_session_id": "…",
     "lease_state": "generating", "phase": "prefill",
     "prefill_eta_seconds": 187.0},
    {"slot_id": 1, "is_processing": false, "n_decoded": null,
     "context_tokens": null, "owner_session_id": "…",
     "lease_state": "idle_leased", "phase": null,
     "prefill_eta_seconds": null}
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

Top-level fields: `capacity_hint`, `prefill_budget_seconds`,
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

1. `idle` — at least one slot is `free` **or** every occupied slot is
   `idle_leased` (nothing is actually generating).
2. `prefill_bound` — no slot is free and at least one occupied slot is in
   `prefill`.
3. `saturated` — no slot is free and every occupied slot is in `decode`.

Two documented extensions: an empty `slots` array (server down / no model / no
cached detail) is `idle` (fail-open — never claim saturation without evidence),
and a pool with no free slot, no prefill and mixed `decode` / `idle_leased`
occupancy is `saturated` (genuine decode work, nothing available to a newcomer).

## Freshness honesty (AC7)

`slots_stale` semantics are unchanged. When the served slot data came from the
last-known cache (a failed fresh `/slots` query), **both per-slot and top-level
ETA fields are `null`** — a dispatcher never acts on a stale wait estimate.
`prefill_budget_seconds` is a configuration constant (not poll-derived) and
stays truthful.

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
  decoding).
- `prefill_budget_seconds` — the admission guard's budget, so the dispatcher
  paces to the same number the proxy enforces.

All of these are optional: the existing `is_processing` / `n_decoded` /
`available_slots` reads keep working unchanged (AC1).
