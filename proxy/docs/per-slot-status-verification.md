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
- Fail-open: `slots` is an empty array when llama-server is not running, no
  model is loaded yet, or the slots query fails/times out — never a
  malformed payload. `total_slots` / `available_slots` behavior is
  unchanged.

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
