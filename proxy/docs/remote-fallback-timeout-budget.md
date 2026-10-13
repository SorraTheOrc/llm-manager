# Bounded remote-fallback timeouts (LP-0MU1RXEPL005CK97)

## Problem

Long agent sessions occasionally sat in the remote fallback tier for 15+
minutes and died with the upstream 900-second queue-timeout
("We were unable to start processing your request within the 900-second
timeout limit"), six times in ~90 minutes on one session.

Evidence (2026-09-14, `/var/log/llama-proxy`):

- 140 `Upstream ReadTimeout`, 114 `Upstream stall after content delivered`,
  33 `Upstream stall: retrying`, 62 `Mid-stream re-route`.
- `Empty response detected on stream attempt 1/4, retrying in 3.00s` — a
  4-attempt empty-body retry loop.
- 2341 fallbacks: local-qwen3 → deepseek-flash 1088, local-qwen3 →
  opencode-go 1025, opencode-go → deepseek-flash 110.

### Root cause: failover was not time-bounded

`upstream_idle_timeout_seconds: 240` is intentional (tolerates the long,
legitimate reasoning pauses of deepseek-v4-flash / opencode-go; see
`proxy/docs/stall-gap-analysis-2026-08-05/summary.md`, which recommends
keeping 240s). But the idle-timeout wait **repeats once per Tier-1 retry**:

- Pre-content stalls retry up to `upstream_retry_max_attempts` (default 3);
  each attempt can consume the full 240 s idle timeout before the next
  `asyncio.TimeoutError`.
- On top of that, empty upstream responses retry up to
  `upstream_empty_retry_max_attempts` (default 3) more times.

Worst case before this fix, one provider alone:

```
(1 initial + 3 stall retries) x 240 s + 3 empty retries x (request + 3 s)
≈ 960 s + overhead  →  exceeds the client's 900 s queue-timeout
```

Failover (routing to the next provider in the fallback chain) only happened
after all of that.

## Fix: `upstream_retry_failover_budget_seconds` (default 300)

A hard wall-clock cap (seconds) on the total time a **single remote
provider** may spend retrying stalls/empty responses — including the
backoff and empty-retry sleeps — before the proxy yields a terminal
`finish_reason: error` (error.type `retry_budget_exhausted`) so the
fallback chain hops to the next provider.

Implementation (`proxy/proxy/proxy_remote.py`, `_handle_remote_streaming`):

1. A `_failover_deadline = monotonic() + budget` is set when the stream
   starts.
2. **Pre-content only:** each chunk read is capped by the remaining budget
   (`min(idle_timeout, watchdog_remaining, budget_remaining)`), and retry /
   empty-retry backoff sleeps are capped the same way, so the budget is a
   hard bound — it cannot be exceeded by sleeps or repeated idle waits.
3. A top-of-loop gate checks the deadline before the stall/empty retry
   handlers: once it is spent (with zero content delivered) the proxy logs
   the exhaustion (provider/model/budget/elapsed/stall & empty retry
   counts), records the provider in the Tier-3 stall circuit breaker, and
   yields the `retry_budget_exhausted` terminal event.
4. Once **any content-bearing chunk** (content, tool_calls, or
   reasoning_content) has been delivered, the budget no longer applies:
   the stream is governed by the idle/activity/max-duration budgets, so a
   productive stream is never cut off mid-generation. (After-content
   stalls already terminate without retry, LP-0MS9FR9LG002AJ4C.)

With defaults (budget 300 s vs idle 240 s) a fully silent gateway now fails
over after **at most ~300–360 s total per provider** instead of ~960 s,
leaving headroom under the 900 s client limit for the rest of the chain.

Config keys (all in `proxy/config*.yaml` under `server:`):

| Key | Default | Meaning |
|---|---|---|
| `upstream_retry_failover_budget_seconds` | 300 | Total retry wall-clock budget per provider (new, LP-0MU1RXEPL005CK97) |
| `upstream_idle_timeout_seconds` | 240 | Per-chunk silence detection (unchanged) |
| `upstream_retry_max_attempts` | 3 | Max stall retries (unchanged) |
| `upstream_empty_retry_max_attempts` | 3 | Max empty-body retries (unchanged) |

The budget is logged on exhaustion:

```
Upstream retry failover budget exhausted: yielding terminal error session=... provider=... model=... budget=300.00s elapsed=... stall_retries=... empty_retries=...
```

## Accepted risk: deepseek/opencode-go time-window gap (AC3)

The deepseek direct entry and all opencode-go entries use

```yaml
available_times: ["00:00-01:00", "04:00-06:00", "10:00-00:00"]   # UTC
```

so **remote fallback is intentionally unavailable** in the plan/author/code
chains during UTC 01:00–04:00 and 06:00–10:00; only `local-qwen3` is
available in those windows (any request that cannot run locally fails
instead of falling back). This windowing is an off-peak quota/cost control
for the free/opencode-go and deepseek gateways; closing it requires an
operator decision on spend, so it is **explicitly documented as accepted
risk** rather than changed silently here.

If the 01:00–04:00 / 06:00–10:00 UTC gap becomes unacceptable, options:
extend the windows to near-24 h for one gateway only (e.g. keep deepseek
windowing but open opencode-go), or add a third always-on remote provider
to the chains. Tracked as accepted risk in LP-0MU1RXEPL005CK97 (AC3);
revisit via a config change + this doc.

## Strictly below the client's 900 s deadline (LP-0MV6BR3HI00957PF)

The client (pi/herdr agent sessions) gives the proxy at most **900 seconds**
before it reports "We were unable to start processing your request within the
900-second timeout limit". Every proxy-side wait that this document (and the
AC1 chain-hold machinery) governs therefore sits **strictly below** 900 s so
the proxy always answers first:

| Bound | Default | Why |
|---|---|---|
| `upstream_retry_failover_budget_seconds` | 300 | Per-provider retry cap (existing) |
| `server.chain_hold_total_seconds` | **720** | Cumulative wait for ALL chain-hold cycles (streaming + non-streaming). Without it `chain_hold_seconds(300) x chain_hold_max_cycles(3) = 900 s` raced the client exactly. 0 disables the cap; values >= 900 are rejected at load. |
| `local_dispatch_first_byte_timeout_seconds` | **840** | Initial first-byte wait for a local dispatch. The legacy default reused the 900 s lifetime wedge knob and logged `dispatch_first_byte_timeout ... ceiling=900s`. 0 disables. |
| `local_dispatch_max_prefill_seconds` | 900 | Lifetime wedge for a dispatch record (unchanged; post-first-byte streams keep their full window). |

`chain_hold_total_seconds` is cumulative across cycles and enforced in both
code paths (`_run_chain_cycles` for non-streaming and the streaming
hold-feedback generator): once the budget is consumed the terminal
exhaustion response is returned instead of holding again. Regression tests:
`proxy/tests/test_chain_hold_total_budget.py`.

The AC1 24-hour post-change measurement for LP-0MU1S617L002Q9PR is tracked
in LP-0MV6BR3PT001WBEI.

## Re-scoped AC2 measurement surface (LP-0MV6CFMU9005WU2E)

AC2 of LP-0MU1S617L002Q9PR is re-scoped (producer decision, 2026-10-13) to:
**warm-cache local dispatches first-byte p99 < 60 s plus a queue-wait cap** —
cold full-prefill and single-slot contention are excluded from the warm
target. Each local ``dispatch_first_byte_ms`` log line now carries:

| Field | Meaning |
|---|---|
| `dispatch_first_byte_ms` | Total dispatch start → first data chunk (unchanged; includes queue wait) |
| `dispatch_to_first_byte_ms` | **Service time** once the slot is held (first-byte minus `queue_wait_ms`; previously identical to the total, now the re-scoped warm metric) |
| `cache_state` | `warm` when the routing `cached_ratio` >= 0.5, else `cold`; fail-open (missing/unknown → `cold`) |
| `cached_ratio` | Cached-token ratio for the (model, session) pair |
| `queue_wait_ms` | Dispatch start → slot acquired (the single-slot contention/queue segment) |

Queue wait is separately capped by the existing contention-queue policy
(`server.contention_queue_max_wait_seconds`, default 60 — pinned by
test_contention_queue.py), so the re-scoped target reads: for warm-cache
local dispatches, `dispatch_to_first_byte_ms` p99 < 60 s, with the queue
wait bounded by the cap. Warm/cold classification and lookup helpers live in
`proxy/test_dispatch_first_byte_metric.py`-covered helpers
(`_classify_cache_state`, `_lookup_dispatch_cache_state`).

## Acceptance criteria coverage

- **AC1 — bounded failover:** `upstream_retry_failover_budget_seconds`
  hard-bounds time-to-failover per provider; the 240 s idle timeout is no
  longer repeated on the critical path (documented here + config comments).
- **AC2 — bounded, configurable, logged empty retries:** empty-response
  retries keep their attempt budget and now additionally respect the
  configured failover budget; exhaustion is logged with the budget.
- **AC3 — deepseek window gap:** documented above as accepted risk.
- **AC4 — regression test:** `proxy/tests/test_remote_failover_budget.py`
  asserts bounded failover for a stalling-then-empty upstream (attempt
  count, elapsed time, terminal `retry_budget_exhausted` event).
- **AC5 — full suite green.**

## Related items

- LP-0MSF1PUM90099ZSW / LP-0MSF5I7XN009ENWQ — 240 s idle-timeout raise.
- LP-0MSF5IAXE005BG33 — stall-gap distribution analysis (keeps 240 s).
- LP-0MRFEXXVC001RYKB — Tier-3 cross-request stall circuit breaker.
- LP-0MRF77A0E0026B9T — empty-response retry origin.