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
(`server.contention_queue_max_wait_seconds`, shipped 30 s since the
LP-0MVT1146V007WXA4 sweep — pinned by `test_contention_queue_docs.py`), so the
re-scoped target reads: for warm-cache local dispatches,
`dispatch_to_first_byte_ms` p99 < 60 s, with the queue wait bounded by the cap. Warm/cold classification and lookup helpers live in
`proxy/test_dispatch_first_byte_metric.py`-covered helpers
(`_classify_cache_state`, `_lookup_dispatch_cache_state`).

## Budget-window exhaustion advertises the window rollover (LP-0MVF0N9SA0071LWM)

A free-tier entry gated by `rate_limit_rpm` (`munitoai-free`) is skipped by
`proxy.provider_budget` **before dispatch** when its rolling 60 s window is
full. That skip applies no provider cooldown and no usage-limit quarantine, so
the exhaustion payload carried `unavailable={}` and `_compute_retry_after`
found no candidate — it advertised `retry_after: 0`. The chain-hold read 0 as
"instant recovery" and held 0 s, replaying all `chain_hold_max_cycles` inside
the same still-full window and surfacing a spurious "All providers exhausted"
(the measured `13/11 requests in 60s window` case).

`_budget_window_retry_after` now derives the wait from the limiter's own
rollover (`seconds_until_capacity`, evaluated with the per-request
`effective_limit`) and takes the **earliest** freeing entry, since the next
cycle can succeed as soon as ANY provider regains capacity. The derived value
is used only when the cooldown/reset computation yields 0, so:

- budget-skip-only exhaustion → positive `retry_after` (≤ the 60 s window),
  bounded by `chain_hold_seconds` / `chain_hold_total_seconds`, so one hold is
  normally enough for the window to roll over;
- exhaustion with a real cooldown / usage-limit reset → unchanged (the
  cooldown-derived wait wins);
- genuine instant recovery (`retry_after == 0`, no budget skip) → still 0 s,
  no added latency (LP-0MU56ZKQD005SX08 behaviour preserved).

The 503 body's JSON `retry_after` and the matching `Retry-After` header carry
the derived value, so clients back off correctly. Regression tests:
`proxy/tests/test_chain_hold_budget_window.py`.

## Provably dead chains fail fast (LP-0MW0KZR7A003UFQK)

A hold only helps when a wait inside the hold budget can change the outcome.
When **every** provider of the residual chain is either

- **deterministically skipped** — `usage_limit_reset_pending`,
  `quota_exhausted` / `quota_hard_stop`, `outside_available_window`: states no
  wait inside the budget can change; or
- **terminally refused** for this request — a non-retryable 4xx (anything in
  400–499 except 408/425/429), which this prompt will always reproduce;

the chain is **not held**: the error is returned immediately and no
`chain_hold_cycle` is consumed.

> **Balance-opted-in entries are not deterministically skipped.** With
> `use_balance: true` (LP-0MW46PBD200249WK), a plan-exhausted entry whose
> rolling/weekly windows still hold headroom keeps dispatching onto prepaid
> balance — it is capacity, not a `quota_exhausted` state, so it cannot make a
> chain provably dead.

Evidence (2026-11-03 08:40–09:05): every remote provider was skipped
deterministically (`munitoai-free` reset ~15.4 h away, `opencode-go-3`
`quota_exhausted` ~14 days, four entries `outside_available_window`), yet the
proxy logged `Chain exhausted ... holding 300s before restarting cycle` three
times — burning an audit subprocess's 1080 s per-call screen budget on holds
for a call that could never succeed. The trigger was the exhaustion tail
substituting the *first provider's raw error* (a bare `400` with no
`Retry-After`) for the generic exhaustion payload: `_bounded_hold_seconds`
reads an absent `retry_after` as "legacy full hold".

### How the verdict is reached

`_resolve_provider_with_exclusions` now records every chain-resolution skip on
the request (`skip_sink` → `_note_chain_resolution_skip`), not just the local
entry's. At the exhaustion tail `_residual_chain_verdict` decides:

| Condition | Verdict |
|---|---|
| every provider is deterministically skipped or terminally refused, and at least one deterministic skip exists | **fail fast** — the verdict is stamped on the request |
| any provider is skipped for a reason a wait can heal (cooldown, stall breaker) or has a transient attempt (5xx/429/408/timeout/connection error/budget skip) | today's bounded hold, byte-identically |
| any reason outside the closed deterministic set (a future reason) | transient by contract — today's bounded hold |

Derived skips (`failure_domain`, `usage_limit_account`) are **neutral**: they
neither prove nor defeat the verdict, because the root provider's own attempt
already decides it.

`_bounded_hold_seconds(..., request)` returns `None` when the request carries
the verdict, so the streaming hold generator and the non-streaming hold loop
both refuse the hold through one code path.

### Honest retry timing

The fail-fast response keeps the first provider's status and body (the client
sees the real `400`, not a generic `all_exhausted`) and gains a `Retry-After`
header plus the matching `retry_after` body field, derived from the
**earliest** known reset — a usage-limit reset countdown, the next
`available_times` window start, or the soonest quota-window reset. The
maximum candidate is deliberately **not** used: on 2026-11-03 it would have
advertised ~14 days because of one dead entry, parking the client long past
the moment capacity returns. `server.max_retry_after_seconds` still caps the
advertisement, and the true reset stays visible in `diagnostics`.

### Log vocabulary (AC3)

One line distinguishes the two outcomes, so a fail-fast refusal is never
counted as hold latency:

```
chain_planned_unavailable model=<path> reason=deterministic_residual_chain hold=skipped retry_after=<s> providers=<n> deterministic_skips=<name:reason,...> terminal=<name,...>
```

`Chain exhausted ... holding <N>s ...` remains the hold. Both are classified
distinctly by `proxy-usage-analysis`.

### Local admission: never bounce onto an empty remote (AC2)

The mirror-image defect on the same incident: at 08:49:40 the local
prefill-ETA guard refused a `remote-ok` request on an enforced ETA bound and
routed it to that dead remote chain (outcome `exhausted`); at 08:50:41 the
identical prompt with the identical ETA was admitted locally and produced a
first byte in 6 s.

The direct (free-slot) guard now takes the ETA refusal **only when at least
one remote provider has capacity** — `_remote_chain_has_capacity` applies the
resolution predicate's own vocabulary read-only: not deterministically
skipped for this request, not in cooldown, no pending usage-limit reset, not
quota-hard-stopped, inside its `available_times` window. When the remote chain
cannot serve, the request is **admitted locally** and waits for its own
prefill (`local_admission_guard_local_wait`), instead of being handed to a
chain that will end `exhausted`.

Scope of the override, deliberately narrow:

- only `reason=eta_exceeds_budget_direct` (a capacity judgement about the
  local engine) is overridable; the request-property refusals
  (`prefill_budget_exceeded`, `context_too_large`, the deadline and
  safety-budget reasons) keep refusing — admitting a prompt that cannot fit
  locally only burns a dispatch;
- only for a chain that has at least one remote provider: a local-only model
  has nothing to bounce to, so its ETA bound stays the honest answer to a
  caller whose budget cannot cover the prefill;
- pin-local / `local-only` intent and the bounded deferral escalation are
  untouched (the pin branch runs first).

Regression tests: `proxy/tests/test_chain_fail_fast.py` (verdict, hold
refusal, retry timing, log line, capacity probe) and the AC2/AC4 rows in
`proxy/tests/test_local_admission_guard.py`.

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
  Attempt-count assertions use jitter-tolerant bounds (`<= 2`) rather than
  exact counts (LP-0MVK8BMGW0059W5S): at these tiny test budgets the exact
  number of attempts depends on wall-clock jitter around the deadline, so a
  single stalled first attempt can exhaust the budget under load. The
  deterministic proofs are the terminal `retry_budget_exhausted` event and
  the budget value in the failover warning log.
- **AC5 — full suite green.**

## Deterministic clock for watchdog/timing assertions (LP-0MVX0NEM7003HRW3)

Every deadline computed inside the stream generator — the watchdog start,
the last-content timestamp that resets the activity budget, and
`_failover_deadline` — is derived from a **single monotonic clock** passed as
`_handle_remote_streaming(clock=...)`. It defaults to `time.monotonic`, so
production behaviour is unchanged (one clock, one arithmetic path).

The seam exists because a watchdog assertion must never be decided by
wall-clock elapsed time: a host under load can starve the event loop between
two awaits, so `asyncio.wait_for` fires its timeout handler before the read
task's step even though the upstream chunk was already available. Before the
seam, the post-timeout re-check read `time.monotonic()` again and a small test
budget (0.2 s) could therefore look expired, producing a spurious
`stream_activity_timeout` error on a stream that had been delivering content
the whole time. Production budgets (1800 s activity / 14400 s duration) are
orders of magnitude beyond any realistic scheduling stall, so the seam is a
test-determinism fix, not a production behaviour change.

`proxy/tests/test_remote_stream_duration_watchdog.py` injects `FakeClock` and
covers both directions:

- content progress spaced below the budget but spanning 3× it (timer reset),
- a starved read that raises `TimeoutError` with zero clock progress (retried
  as an ordinary stall, never classified as an activity expiry).

## Related items

- LP-0MW0KZR7A003UFQK — provably dead chains fail fast; a local ETA refusal
  never bounces a request onto a remote chain with no capacity.

- LP-0MSF1PUM90099ZSW / LP-0MSF5I7XN009ENWQ — 240 s idle-timeout raise.
- LP-0MSF5IAXE005BG33 — stall-gap distribution analysis (keeps 240 s).
- LP-0MRFEXXVC001RYKB — Tier-3 cross-request stall circuit breaker.
- LP-0MRF77A0E0026B9T — empty-response retry origin.
- LP-0MVF0N9SA0071LWM — budget-window exhaustion derives `retry_after` from
  the free-tier window rollover (see above).