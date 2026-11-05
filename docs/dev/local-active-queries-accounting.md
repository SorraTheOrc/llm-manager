# `local_active_queries` accounting — the exactly-once credit model

Parent work item: **LP-0MVWF78QK0060TZH** (capacity signal under-reports free
slots). Implementation: **LP-0MVYWLBV70018ZZB** (AC3 fix, AC8 fold-or-prove).
Release-path audit table and per-path verdicts: LP-0MVYWK964006YDQ8, comment
`LP-C0MW31NHW8004PADI` on the parent.

## Why this exists

`local_active_queries` is the proxy's in-flight local-request count. It is read
by `/llama/local/status` (`available_slots` / `capacity_hint`), by admission
pricing and by the prefill saturation gate, so an over-count reports a busy GPU
that is idle and starves the herdr downtime dispatcher (LP-0MVKSXR4P001V0CM).

The self-healing clamp (`_recover_stuck_local_active_queries`) repairs the
counter only when it *exceeds* the number of active dispatch records, and it
logged **95 recoveries in ~12.2 h** post-v0.1.27 (62× `1→0`, 29× `2→1`,
4× `3→2`) — i.e. one leaked increment per ~13 local streams, surviving three
prior leak-fix items. The clamp is a detector and last line of defence; it is
deliberately untouched by this fix (parent AC6).

## The model

The counter has two increment sites and one decrement owner:

- **increment (explicit sessions):** `_try_acquire_local_dispatch`
  (`proxy/proxy/router_helpers.py`)
- **increment (anonymous/non-explicit sessions):**
  `_increment_local_active_queries`
- **decrement:** `_decrement_local_active_queries`, called only from
  `router._cleanup_after_request`

`local_active_queries` therefore counts **credits**, and the rule is:

> One credit and one token per **dispatch** (acquisition). The record keeps its
> outstanding tokens in `dispatch_credits` (oldest first) with
> `dispatch_token` naming the oldest live dispatch — the attribution owner —
> because a record is keyed `(endpoint, session)` and a session's overlapping
> dispatches (a single-flight duplicate, a queued sibling, a back-to-back next
> turn) share it. A credit is released exactly once, by
> `_claim_local_dispatch_credit`, which mutates the counter and the record set
> in **one** critical section (`local_active_queries_lock` → then
> `local_dispatch_records_lock`, the order `_try_acquire_local_dispatch` uses).
> A re-acquisition by the *same* dispatch (the same request re-entering the
> choke point, e.g. a provider retry loop) takes no second credit: it has one
> end-of-request cleanup, so a second increment could never be released.

Consequences, each of which was a measured leak shape:

1. **An in-request re-acquisition takes no second credit.** A request that
   re-enters the choke point while its own token is still outstanding releases
   nothing extra on completion: one dispatch, one credit, one cleanup. The
   record is refreshed (`expires_at`) rather than overwritten, so the live
   request keeps its identity (`started_at`, `stream_cm` pin, evidence,
   progress baseline).
2. **A refusal releases only its own credit.** `_release_single_flight_detour`
   (streaming and buffered) releases the refused request's own credit through
   its token and **does not delete the dispatch record** — the record belongs to
   the in-flight request the coordinator just detected. Pre-fix the streaming
   detour suppressed the decrement *and* deleted that record, which stranded the
   owner's own decrement — the dominant `2→1` / `1→0` shape.
3. **A release core may pop the record and leave the credit outstanding.** The
   unserved-dispatch core (`_release_unserved_dispatch`: prefill deadline,
   measured-cache-deficit abort) and `_release_client_disconnect_dispatch` pop
   the record so the slot frees immediately and leave the decrement to the
   owning request's single end-of-request cleanup. `_claim_local_dispatch_credit`
   recognises that shape as **counter > credits outstanding across all records**
   and releases the outstanding credit there.
4. **Paths that own a release own its credit.** Session eviction
   (`server._release_lease_on_session_eviction`, which releases *every* credit a
   record holds) and the cross-endpoint completion release
   (`_release_completed_dispatch`) now claim the credit with the record instead
   of dropping the record and leaving the increment behind. Because the claim
   and the transition happen in one critical section, two mechanisms racing for
   the same lease (eviction + the in-flight request's cleanup, disconnect
   release + cleanup, sweep + completion cleanup) can never double-decrement:
   the loser finds no credit and is a no-op. That is AC2's "never drops the
   counter below the active-record count".

## Nullable cases, documented

- **Legacy state without a dispatch-record registry** keeps the pre-fix
  unconditional decrement (there is no record to pair a credit with).
- **A failed record creation on the anonymous path** (swallowed by design)
  leaves a credit with no record; the `counter > credits outstanding` branch
  still releases it, and an overwritten anonymous record's leftover credit is
  released the same way.
- **A record without a `dispatch_credits` list** (legacy state, or a hand-built
  record in a test) counts as one credit while it is `active`, so every
  pre-existing shape behaves exactly as before.
- **`counter < credits outstanding`** (the historical under-count the clamp
  never detects) makes the guard skip a release rather than compound the error;
  the next acquisition/cleanup cycle converges and the clamp is not consulted.
- **The AC4 invariant test asserts `counter == active dispatch records`.** With
  multi-dispatch records the counter can legitimately exceed that count (one
  record, two credits); the invariant test's workload keeps one dispatch per
  session, and the credit total is the general ground truth.

## Tests

- `proxy/tests/test_local_active_queries_release_matrix.py` — one driven release
  per AC2-audit row (18 rows + coverage guard) asserting the counter returns to
  its pre-request value *within* the request lifecycle, plus the AC2
  idempotence rows (`test_double_release_of_one_lease_keeps_the_counter_at_zero`,
  `test_eviction_does_not_release_another_sessions_credit`).
- `proxy/tests/test_local_active_queries_invariant.py` — mixed-outcome
  concurrency (success + measured-deficit abort + client cancel + duplicate
  fan-out) asserting `local_active_queries == active dispatch records` at every
  quiescent step and that the clamp's WARNING never appears.
