# Lease-owner retention priority in the local contention queue

**Work item:** LP-0MVI2VRT2000V1AP

## The problem

When every local slot is busy, a request enters the bounded contention queue
(`contention_queue_policy: queue`, wait cap 120 s, depth 8). The queue was
plain FIFO, so a session that had just released its local lease and returned
immediately re-queued **behind newcomers**. It then waited the full budget
and fell through to a paid remote provider — and on the next turn came home
again and re-prefilled.

Observed on `audit-WL-0MVFRRKKN00544PQ-phase2_child_5-6e22ef57`
(2026-10-21 12:23:35–12:36:55): **seven local↔remote transitions in twelve
minutes**.

| time | provider | trigger |
|---|---|---|
| 12:24:29 | local | queue dispatch (48.4 s wait) |
| 12:25:27 | local | queue dispatch (49.0 s wait) |
| 12:27:30 | MunitoAI Free | `contention_queue_fallback_after_queue queued_duration=120.00s` |
| 12:28:06 | local | reclaim, `cached_ratio=0.513`, 35.8 s prefill |
| 12:30:30 | OpenCode Go 3 | `contention_queue_fallback_after_queue queued_duration=120.00s` |
| 12:34:43 | OpenCode Go 3 | `local_dispatch_denied ... owner=<other session> active=1` |
| 12:34:48 | local | reclaim, `cached_ratio=0.396`, 2.2 s prefill |

None of the hops was an ETA-guard or context-size decision:
`local_admission_guard decision=admit reason=within_budget` on every attempt
(3.9 s–72.9 s against a 120 s budget) and prompts stayed between 2,067 and
20,468 tokens. The binding constraint was dispatch contention.

## The fix

`proxy/contention_queue.py` now distinguishes **retained** waiters:

```
retained  ⇔  a local lease was granted within contention_queue_lease_grace_seconds
             AND the session still has local cache history
```

- **Grant registry** — `note_local_grant(session_id)` is called wherever a
  dispatch-lease record is created (`router_helpers`, lease acquisition). The
  timestamp survives the lease release, which is what makes a session
  *returning*.
- **Cache-history requirement** — `is_retained_local_owner()` additionally
  requires a surviving cache-ratio record (`_get_cached_ratio_source(...) is
  not None`). A session whose prefix was genuinely evicted (record cleared,
  no snapshot credit) is no cheaper than a newcomer and gets no priority.
- **Queue ordering** — retained waiters are inserted after any existing
  retained waiters but ahead of non-retained ones, so ordering stays FIFO
  within each class. Each reorder logs:

  ```
  contention_queue_retention session=<returner> ahead_of=<newcomer> \
    reason=lease_owner_retention depth=<n>
  ```

- **Invalidation** — session eviction clears the grant
  (`contention_queue.forget_local_grant`) alongside the cached-ratio records,
  so a discarded session cannot keep its priority.

## Routing-hop counter (thrash observability)

`provider.note_routing_hop(session_id, provider, model_name=…, **context)`
records the provider that served a stream and returns the session's
transition count. It emits a `routing_hop` line **only** when the provider
changes:

```
routing_hop session=<id> from=local to=opencode-go-3 hops=1 entry=opencode-go-3
```

Both `Stream started` sites (local in `router.py`, remote in
`proxy_remote.py`) call it, so a thrash pattern is countable per session
directly from the log — the seven-transition case above would show six
`routing_hop` lines. State is cleared on session invalidation.

## Configuration

```yaml
server:
  contention_queue_lease_grace_seconds: 300   # bounds [0, contiguous wait cap]
```

- `300` (default) — a released lease keeps retention for five minutes, longer
  than any single queue budget, so a returning session always re-queues as
  retained.
- `0` — **rollback**: retention disabled, queue order returns to plain FIFO.

## Tests

`proxy/tests/test_contention_queue_retention.py` (15 cases):

| AC | Coverage |
|---|---|
| AC1 | retained waiter woken ahead of a queued newcomer; `contention_queue_retention` log names both sessions and the reason |
| AC2 | grant → release → newcomer queues → returner dispatches first (full eligibility path) |
| AC3 | evicted prefix (`clear_cached_ratio_for_session`) gets no priority and queues behind the newcomer |
| AC4 | prefill-ETA refusal still falls through immediately (`test_local_admission_guard`, unchanged) |
| AC5 | `note_routing_hop` counts transitions per session, only on change |
| AC6 | 7-round replay: 7 local reclaims (0 hops) with retention, 0 reclaims with `grace=0` |

FIFO within each class is pinned by
`test_fifo_preserved_between_retained_waiters`; a lone newcomer still
dispatches normally (`test_newcomer_still_dispatches_when_it_is_alone`).
