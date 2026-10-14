# In-flight lease hold: accepted-but-never-progressed local requests

Characterisation and bounding of the residual risk introduced by the
in-flight prefill-liveness signal (`9b7c505c`, LP-0MUXTW4MD003TVIC).
Tracked by **LP-0MUXYXZQB002S391**.

## Summary

`_extend_lease_during_prefill` treats "the upstream request is still open
with no transport error" (`prefill_in_flight=True`) as a prefill-liveness
signal. That removed the dependency on `/slots` (which llama-server blocks
for the whole prompt-processing batch, so it is unobservable under load) and
let long local prefills complete instead of being killed at the 180s
no-progress timeout.

The residual risk is that "connection open, no transport error" is not the
same as "the backend is actively prefilling". A request llama-server accepts
but never progresses is kept alive by the in-flight signal for the whole
no-progress window. This document quantifies how long such a hold lasts, how
much of the local pool it can starve, and which bound applies.

**Conclusion:** the impact is material in the worst case (a single held lease
stalls 100% of a single-slot pool) but is *bounded and low-frequency*. The
effective no-first-byte bound is the **first-byte ceiling
(`local_dispatch_first_byte_timeout_seconds`, default 840s)** — already
*tighter* than the 900s lifetime ceiling — which cancels the pending upstream
open task and releases the lease. No production code change is required; the
deliverable is this characterisation plus the hermetic tests in
`proxy/tests/test_prefill_lease_extension.py`.

## The shape

| Signal | Value in this shape | Effect |
|---|---|---|
| Upstream request | open, no transport error | `prefill_in_flight=True` |
| First byte | never arrives | no chunk-refresh path |
| `/slots` | times out / blocked | `progress=None, alive=False` |
| Numeric progress | none | no progress-advance extension |

Only the in-flight signal keeps the lease alive, so the 180s no-progress
watchdog cannot release it. The bounds that remain are the two hard ceilings.

## Which bound applies

| Bound | Default | Applies to | Effect |
|---|---|---|---|
| `local_dispatch_no_progress_timeout_seconds` | 180s | no observed progress **and** no liveness | **defeated** by the in-flight refresh |
| `local_dispatch_first_byte_timeout_seconds` | 840s | initial first-byte wait in `_await_first_byte_with_prefill_monitor` | `asyncio.TimeoutError`; cancels the pending upstream open task; router calls `_cleanup_after_request` → lease released |
| `local_dispatch_max_prefill_seconds` | 900s | total active lifetime of a dispatch record in `_watchdog_no_progress` | watchdog releases the record and closes a pinned `stream_cm` (upstream cancelled) |

For a request that never delivers a first byte the hold is therefore bounded
by **min(first-byte ceiling, max-prefill ceiling) = 840s** by default. The
first-byte ceiling fires first (840 < 900) and is the authoritative bound;
the 900s lifetime ceiling is the backstop for a stream that *did* deliver a
first byte and then wedged during generation.

> The first-byte ceiling was itself tightened from the old 900s (which raced
> the client's 900s queue timeout) to 840s by LP-0MV6BR3HI00957PF
> (2026-10-13). The holds quoted below were measured while the ceiling was
> still 900s.

## Starvation analysis and quantified impact

While the lease is held the dispatch record is `active=True`, so it occupies
a pool slot under both occupancy gates enforced by `_try_acquire_local_dispatch`:

- the generating-only pool (`local_generating_queries`), and
- the prefill-in-flight guard (`_get_prefill_in_flight_count`).

On the default profile (`session_slot_pool_size: 1`, `proxy/config.yaml`), a
single held lease denies **100% of local capacity** for the duration of the
hold (≤ 840s). On the cheap profile (`session_slot_pool_size: 3`,
`proxy/config-cheap.yaml`), three concurrent holds exhaust the pool. A held
lease therefore denies a fraction `held / pool_size` of local capacity for
up to the first-byte ceiling.

Hermetic evidence for the denial is
`test_in_flight_hold_starves_single_slot_local_pool`: with one held lease on
a single-slot pool, a second session is denied (`acquired is False`,
`owner == "holder"`); after the max-prefill ceiling releases the holder the
waiter is admitted.

### Frequency

The post-fix verification window (LP-0MUXQNPKE008K9MP,
`prefill-lease-extension-verification-2026-10-12.json`, 16.1h on 2026-10-12)
recorded:

| Metric | Value |
|---|---|
| `lease_extended_during_prefill` (in-flight) | 6096 |
| `dispatch_first_byte_timeout` (900s ceiling at the time) | 10 |
| `lease_released reason=no_progress_watchdog` | 14 |
| long (>180s) local prefill completions | 28 |

So a 900s-first-byte hold occurred roughly **0.6 times per hour** in a heavy
window — a small fraction of the 6096 in-flight extensions, most of which
completed normally. The item description records **two** such holds during
the cheap-mode verification specifically.

### Impact

- A held lease denies local dispatch to other sessions for ≤ 840s. Denied
  requests wait in the bounded cross-session contention queue
  (`session_single_flight_queue_timeout_seconds`, 120s cheap / 45s fast) and,
  if the queue is full or the wait expires, fall through to a **remote**
  provider — extra latency and, for paid fallbacks, extra cost.
- It is **not** an unbounded wedge: both ceilings (840s / 900s) release the
  hold, the upstream is cancelled (first-byte timeout cancels the open task;
  the watchdog closes a pinned `stream_cm`), and counters/slot assignments
  are freed.
- A genuinely hung upstream with *both* ceilings disabled (`0`) is the only
  path that could hold indefinitely; that is an explicit operator choice and
  out of scope here.

## Proposal (AC3)

The hold is **already bounded tighter than 900s** by the 840s first-byte
ceiling; the recommendation is to treat that ceiling — not the 900s lifetime
ceiling — as the authoritative no-first-byte bound, and to keep the invariant
`local_dispatch_first_byte_timeout_seconds <= local_dispatch_max_prefill_seconds`
so the first-byte ceiling always fires first and cancels the upstream before
the lifetime watchdog has to.

A tighter bound than 840s was considered and **not recommended**:

- **Trade-off against genuine long prefills.** llama-server returns the SSE
  response headers with the *first token*, so a legitimate long prefill emits
  no bytes for its entire duration. Observed legitimate completions in the
  verification window exceeded 180s; the adaptive lease cap is 1500s.
  Lowering the no-first-byte budget below the longest legitimate prefill
  would reintroduce the original kill/retry loop (LP-0MUXQ5QL40027VZ4). 840s
  sits comfortably above observed prefills while staying below the client's
  900s queue deadline.
- **Alternative discriminating signal — `/slots` `is_processing`.** Rejected:
  `/slots` is unobservable exactly when the signal is needed (llama-server
  blocks it for the whole prompt-processing batch) and a *queued* request's
  slot reports `is_processing=False`, so it cannot distinguish "queued" from
  "wedged".
- **Alternative discriminating signal — queue position.** llama-server does
  not expose a stable queue-position field on the deployed b8782 build, and
  the prefill-aware guard already caps concurrent in-flight prefills at the
  slot count, so a same-endpoint queue is the exception rather than the rule.

If operators want a tighter bound on constrained hardware they can lower
`local_dispatch_first_byte_timeout_seconds` (and keep it ≤ the lifetime
ceiling); the per-token adaptive lease still protects in-progress streams
that have delivered a first byte.

## Acceptance-criteria mapping

1. **met** — `test_accepted_but_never_progressed_held_to_ceiling_then_released`
   reproduces the shape (open in-flight request, no first byte, no `/slots`
   state), shows the in-flight refresh defeats the 180s no-progress watchdog,
   and shows release at `local_dispatch_max_prefill_seconds` with the
   upstream `stream_cm` cancelled and counters/slot freed.
2. **met** — the starvation analysis above (occupancy model, `held / pool_size`
   quantification, observed frequency) with hermetic evidence in
   `test_in_flight_hold_starves_single_slot_local_pool`.
3. **met** — the 840s first-byte ceiling is identified as the tighter
   effective bound and the trade-off against killing genuine long prefills is
   stated; further tightening is not recommended.
4. **met** — no production code change; the added tests live in the existing
   hermetic suite.
