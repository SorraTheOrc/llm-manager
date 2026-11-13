# Proxy Request Routing

This document describes how the proxy decides how to route incoming requests.
The core routing functions `proxy_to_local` and `proxy_to_remote` (along with request/response
logging helpers `log_request`, `log_response`, and `log_response_chunk`) live in
`proxy/proxy/router.py`.

For backward compatibility, these five functions are re-exported from `proxy/proxy/server.py`
so that `from proxy.server import proxy_to_local` continues to work.

## Request Routing Flow

The main entry point is `proxy_openai_api` at `/v1/{path:path}`. Here's how routing works:

### 1. Model Identification

- The proxy parses the request body to extract the `model` field
- If no model is specified, it falls back to `current_model` (the locally loaded model)

### 2. Model Configuration Lookup

The `get_model_config()` function matches the model name against the `models` section in `config.yaml` using:

1. **Direct name match** - exact match against model keys (e.g., `anthropic`, `openai`, `qwen3`)
2. **Exact alias match** - case-insensitive match against the `aliases` list
3. **Wildcard pattern match** - fnmatch patterns (e.g., `gpt-*` matches `gpt-4`, `gpt-4-turbo`)

### 3. Route Based on Model Type

| Condition | Route To | Behavior |
|-----------|----------|----------|
| `model_cfg.type == "local"` | `proxy_to_local()` | Routes to local llama-server on `localhost:8080` |
| `model_cfg.type == "remote"` | `proxy_to_remote()` | Routes to external API (OpenAI, Anthropic, GitHub) |
| No model config + `default_remote.enabled` | `proxy_to_remote()` | Falls back to default remote endpoint |
| No model config + `current_model` exists | `proxy_to_local()` | Uses currently loaded local model |
| No model config + nothing else | Returns `400` error | "Unknown model" |

### 4. Local Model Handling

For local models, the proxy checks:

- If the requested model is already loaded (`current_model == llama_model_str` and process running) → route immediately
- If in **router mode** (`llama_router_mode: true`), it queries the router to see if the model is already loaded
- Otherwise, it schedules a **background load** and returns `503 Model Loading` to the client

#### Generating-only dispatch pool

The local model uses a **generating-only dispatch pool** to limit concurrent
sessions that are actively generating tokens. The key design choices are:

- **Prefill does NOT count against the pool** — sessions in the prefill
  phase (waiting for the LLM to start producing) are allowed concurrently,
  up to `session_slot_pool_size` parallel prefills. This prevents the
  llama-server internal queue from saturating (LP-0MTJET4I5009EHNX).
- **Only generating sessions count** — the counter (`local_generating_queries`)
  increments when the first token arrives and decrements on stream end.
- **Self-healing recovery** — if the decrement in the streaming generator's
  `finally` block aborts (e.g., slot-save exception), a session key can leak
  into the generating pool, wedging it at capacity. The dispatch cleanup loop
  (`_dispatch_cleanup_loop`, runs every 10 s) periodically reclaims stale
  generating entries by reconciling them against active dispatch records
  (LP-0MTYAWDCQ006RGYU). This self-healing mechanism recovers the pool within
  ~20 seconds without a proxy restart.

#### Dispatch-lease orphan cleanup: fail-closed liveness (LP-0MVIQ0FJC000YLX0)

Every local dispatch holds an **active dispatch record** (lease + slot
registry entry) until it finishes. A periodic sweep
(`_cleanup_stale_local_dispatch`) releases records whose `expires_at` has
passed. Because a stream can finish late or its release-on-finish can be
skipped, the sweep verifies liveness before freeing a lease:

- **Primary evidence is progress, not `/slots`.** llama-server blocks
  `/slots` for the whole prompt-processing batch, so under a heavy prefill the
  probe times out and used to be read as "dead" (fail-open) — freeing the
  lease and slot of a stream that was still working (measured 298
  `orphan_cleanup` vs 6 `lease_verified_active` in 24 h, with 81 of 118 long
  streams orphan-cleaned mid-flight yet finishing later).
- **Fail-closed rule.** When the `/slots` probe is inconclusive (timeout,
  error, no slot assignment) but the record's `last_progress_ts` is inside
  `server.local_dispatch_no_progress_timeout_seconds` (default 180 s), the
  lease is **extended** (logged `lease_verified_active
  reason=recent_progress stream_abandoned=False`) instead of orphaned. A busy
  backend is not a dead backend.
- **Bounded, not unbounded.** The no-progress watchdog
  (`reason=no_progress_watchdog`) and the max-prefill ceiling
  (`server.local_dispatch_max_prefill_seconds`, default 900 s) still release
  a genuinely wedged record — fail-closed only protects a *fresh* progress
  clock.
- **Genuine orphans cancel upstream.** When a record really is orphaned, the
  sweep cancels the captured upstream stream context (`stream_cm.__aexit__`)
  before freeing the lease, so the llama-server slot is actually released —
  matching what the no-progress-watchdog path already did. Without this the
  proxy freed its bookkeeping while the backend kept prefilling, admitting new
  long prefills onto a busy GPU.
- **Counters stay consistent.** `local_active_queries`, the slot registry and
  `local_prefill_in_flight` are updated on the same paths as the lease, so
  admission and serialisation never see a slot the backend is still using.

#### Client-dead mid-prefill release (LP-0MVLFVY6K003VYEL)

A client that vanishes *during prefill* is released immediately instead of
waiting for the `idle_timeout` sweep. Pre-fix the dispatch was retained as an
inactive record with a fresh `expires_at` plus its `local_prefill_in_flight`
marker, so every other session was refused with
`local_dispatch_denied … reason=local_lease_active` for ~5.5-6 minutes
(observed 2026-10-23 20:58-21:04). The prefill phase emits no chunks, so the
in-loop `is_disconnected()` check and the generator's `GeneratorExit` /
`CancelledError` handlers cannot see it. Two detections now feed one release:

- `_await_first_byte_with_prefill_monitor` polls
  `request.is_disconnected()` on the existing prefill poll cadence
  (`local_dispatch_lease_prefill_poll_seconds`, default 10 s; no new task) and,
  on a *confirmed* disconnect, releases and raises
  `ClientDisconnectedDuringPrefillError` → the router accounts the counters and
  answers 499 (no backend cooldown, no 5xx metric).
- `DisconnectReaper` (5 s) records its confirmed verdict on the ASGI scope
  (`llama_proxy.client_disconnected`) before cancelling the request task, so
  the router's `CancelledError` path can release synchronously without
  re-probing a connection it can no longer read.

`_release_client_disconnect_dispatch` deletes the record, pops the marker,
frees the slot assignment, cancels the captured upstream stream, wakes the
contention queue and logs
`lease_released session=… reason=client_disconnect endpoint=…` (distinct from
`disconnect`, `prefill_deadline`, `idle_timeout`, `no_progress_watchdog`). It
contains **no `await`** so a second cancellation cannot skip it; the two
awaiting steps run as detached tasks. Only an explicit `is_disconnected()`
`True` (or the reaper's recorded verdict) releases — never elapsed time — so
a live but slow prefill is untouched, and the 90 s deadline / 180 s watchdog /
900 s ceiling all still apply.

#### Completion releases a lease that outlived its request (LP-0MVWFKI8O001X694)

A local dispatch that ended **without a first byte** while the request was
re-routed to a remote provider used to leave its dispatch record `active`:
the end-of-request cleanup only transitioned the record of the endpoint the
request finally completed on, so the lease was never marked terminal. The
`no_progress_watchdog` then reclaimed it 180 s later and emitted
`local_eviction reason=tenancy_lease_expired` /
`local_eviction_after_acceptance reason=no_progress_watchdog` for a request
that had **succeeded** — an eviction that feeds the local entry-cooldown
machinery (parent LP-0MVWEZCB0006WPBQ AC8).

`_cleanup_after_request` is the request's single end-of-request owner, so it
now calls `_release_completed_dispatch`, which releases every record still
`active` for the session (i.e. one whose own endpoint never received its
end-of-request transition), pops the session's `local_prefill_in_flight`
marker, frees the slot assignment and wakes the contention queue, logging
`lease_released session=… reason=request_completed endpoint=…` once per
released record. It is **not** an eviction path: no `local_eviction` event, no
`eviction_after_acceptance` counter, no KV-snapshot plan, no entry cooldown
and no abandoned-slot annotation, and it does not cancel the upstream stream
(a lease-lifecycle fix, not a prefill-cancel fix). Genuine stalls are
unaffected: the watchdog, the prefill deadline and the disconnect path
release their own records exactly as before.

Belt-and-braces: a record that somehow survives to the sweep carrying the
`terminal` / `completed_at` stamp is reaped there **without** an eviction and
logged once as `local_dispatch_completed_record_reaped
reason=completed_record_in_sweep`, so a missed completion release is
diagnosable from the log alone.

#### Counter/credit contention invariant (LP-0MVYWKX7200880QX, LP-0MW0TLPOR002TAFO)

`local_active_queries` must equal the number of **outstanding dispatch
credits** at every step of a contended local workload — admission, first byte,
per chunk, a measured cache-deficit eviction, a client cancel and completion.
That equality is the definition of done for exactly-once release accounting,
and it is the condition under which the self-healing clamp
(`_recover_stuck_local_active_queries`) has nothing to repair: an over-count
is what makes `/llama/local/status` report a busy GPU that is idle
(LP-0MVKSXR4P001V0CM).

A dispatch record is keyed `(endpoint, session)` and holds one credit per
in-flight dispatch in `dispatch_credits` (LP-0MVWF78QK0060TZH), so the
counter counts **credits**, not records: two overlapping dispatches of one
session are legitimately `counter == 2` with **one** active record. Every
consumer must therefore use `_outstanding_dispatch_credits`, never
`_count_active_local_dispatch_records` (they are equal only for one-credit
records).

LP-0MW0TLPOR002TAFO closed the last record-count assumptions:

- `_increment_local_active_queries` (the anonymous/non-explicit increment
  path) **appends** a credit to a live record instead of overwriting it — the
  explicit path (`_try_acquire_local_dispatch`) already did. Overwriting
  dropped a sibling's credit from the record set, so `counter (2)` no longer
  matched the record's credits `(1)`.
- `_increment_local_active_queries` also **stamps its request** with the
  credit token (`request._local_dispatch_token`), exactly as the explicit path
  does, so an anonymous dispatch's teardown or cancellation release is
  token-scoped instead of consuming a sibling's credit via the untracked
  (`dispatch_token=None`) path.
- `_cleanup_after_request` drops a non-explicit (or disconnected) session's
  record only when it holds **no** remaining credits; a record shared with a
  still-in-flight sibling is kept, because it is the live-work evidence the
  recovery loops read.
- `_release_client_disconnect_dispatch` (the client-cancel, prefill-deadline
  and reaper-cancellation release) keeps a record that another in-flight
  dispatch still holds a credit on, instead of popping the whole shared
  record; the paired cleanup releases this request's own token-scoped credit
  and leaves the sibling intact.
- `_recover_stuck_local_active_queries` clamps to the outstanding-credit count,
  not the active-record count.

Without these, the recovery loop *destroyed live state* during a healthy
burst — the measured `local_active_queries 2→1` / `1→0`, `active_queries 1→0`
and `local_generating_queries reclaimed 1 stale session` recoveries, while the
dispatch was still running.

`proxy/tests/test_local_active_queries_invariant.py` drives three concurrent
local streams with distinct outcomes through the real `proxy_to_local`
(paced SSE success, measured-deficit abort, pre-first-byte client cancel) and
samples `local_active_queries == _outstanding_dispatch_credits(srv)`
at each quiescent checkpoint under the production
`local_active_queries_lock` → `local_dispatch_records_lock` order, then runs
the real clamp and asserts zero `counter recovered` warnings.
`proxy/tests/test_local_dispatch_counter_credit_accounting.py` pins the
credit-aware behaviour for each exit path — success, client disconnect,
**deadline cancel** and error all restore `local_active_queries`,
`active_queries`, `local_generating_queries` and `local_generating_sessions` —
plus a concurrently-acquired, mixed-exit burst that must settle to zero with no
`counter recovered` line.

#### The recovery WARNING names the request it could not account for

(LP-0MVYWLHNI000AQ7X, parent AC5.) The self-healing clamp is the detector for
this leak class, so its WARNING is kept — but it used to carry only the counts,
which left the recoveries with no antecedent log signal (12 % in the parent's
evidence) unattributable without a fresh investigation. The line now appends
the leaking request's identifying fields when they are determinable:

```
local_active_queries counter recovered: reset from N to M (M active dispatch
records) session=… turn=… record=… release_reason=… release_age_s=…
```

- `session` / `turn` / `record` name the **active** dispatch records (the live
  work the clamp keeps, comma-separated, capped at
  `LOCAL_DISPATCH_RELEASE_HISTORY_LIMIT`); when none is active they name the
  most recently released record, which is the only identity a leak that has
  just been released leaves behind. `record` is the dispatch-record id
  `<endpoint>|<session>`.
- `release_reason` (`measured_cache_deficit`, `prefill_deadline`,
  `client_disconnect`, `explicit_release`, `request_completed`,
  `no_progress_watchdog`, `idle_timeout`, `orphan_cleanup`, `non_explicit`,
  `disconnect`, `session_evicted`) and `release_age_s` come from the most
  recent `lease_released` inside `LOCAL_DISPATCH_RELEASE_MAX_AGE_SECONDS`; an
  older release is unrelated churn and is not named.
- Any field that cannot be determined is `unknown`, so the detector always
  fires and the reset values (`prev` → `active_count`) are unchanged.

The identity is captured where it is already known — on the dispatch record at
admission (`session`, `turn`) and in a bounded in-memory release history next
to the existing `lease_released` line — so the clamp tick reads a snapshot
only: no new lock, no new hot-path work.

### 5. Remote Model Handling

For remote models (`proxy_to_remote()`):

- Constructs the target URL from `endpoint` + path
- Adds API key from environment variable (e.g., `OPENAI_API_KEY`)
- Adds custom headers from config
- Forwards the request via `httpx` (streaming for SSE responses)

### Free-tier request budgets (`rate_limit_rpm`, LP-0MVDQ1GMO00801PL)

Remote entries may declare `rate_limit_rpm`, a per-entry request budget for a
rolling 60-second window. It is enforced by
`proxy.provider_budget.acquire_provider_budget()` *before* dispatch in both
fallback entry points (`_proxy_with_remote_fallback_cycle` and
`_proxy_with_fallback`), on top of the cooldown / `available_times` /
failure-domain gates (which only react *after* a failure).

| Situation | Behaviour |
|-----------|-----------|
| Window has capacity below the new-session cap | Entry is dispatched normally (slot reserved up front). |
| Unreserved part spent — **new** session (no recorded affinity) | Skip immediately, log `Provider budget exhausted: … limit=<rpm-reserve> reserve=<n> reason=provider_budget_exhausted`, record attempt `status=provider_budget_exhausted` with `effective_limit`/`reserve_slots`, continue to the next provider. **No waiting.** |
| Unreserved part spent, reserved slots free — **in-progress** session (affinity recorded on a 2xx dispatch) | Dispatch on the reserved slot: the reserve exists so established sessions keep their provider and prompt cache instead of queueing behind new traffic. |
| Window completely full — **in-progress** session | Wait up to `server.provider_rate_limit_wait_max_seconds` (default **15**) for the window to drain, then dispatch on this entry (`status=provider_budget_exhausted_waited`); if it never drains, pass through like a new session (`reason=provider_budget_exhausted_after_wait`). |
| Waiters already at `server.provider_rate_limit_wait_max_waiters` (default 8) | Pass through (`reason=provider_budget_exhausted_waiters`) so waiters cannot pile up. |
| Client disconnects during the wait | Wait aborts promptly (`reason=provider_budget_exhausted_disconnect`). |
| `rate_limit_rpm` unset or `0` | Ungated: behaviour exactly as before; no affinity recorded. |

**Per-model / per-entry knobs (`budget`, LP-0MVDVK5TM004TQQV).** Every
behaviour knob in the table (`reserve_ratio`, `wait_max_seconds`,
`max_waiters`, `affinity_ttl_seconds`) resolves through
**provider entry `budget` block > `models.<name>.budget` block > `server.*` >
built-in default**, per knob — so `plan`, `author` and `code` can trade
latency against free-tier utilisation independently (e.g. an interactive chain
may wait longer, a background chain may pass through immediately) without
changing any other chain. All layers are optional; absent keys fall through, so
a config without `budget` blocks behaves byte-for-byte as before (identical
`BudgetDecision` values and log lines). Example:

```yaml
models:
  plan:
    budget:
      reserve_ratio: 0.4        # overrides the default 0.3 for this chain
      wait_max_seconds: 10
      max_waiters: 4
      affinity_ttl_seconds: 600
    providers:
      - name: munitoai-free
        rate_limit_rpm: 15
        budget:                 # overrides the model block for this entry only
          reserve_ratio: 0.2
```

- **Validation** matches the server knobs exactly: the ratio is clamped to
  `[0, 1]` and wait/TTL/waiter counts to `>= 0`; a malformed value is logged
  and falls back to the next layer. Never raises, never blocks a request
  (fail open). A non-mapping `budget:` value is ignored with a warning.
- **Provenance.** The effective knob's source layer is logged as
  `budget_source=<knob>@<layer>` (e.g. `budget_source=reserve_ratio@model`)
  on the skip/wait lines and recorded in the diagnostics attempt entry
  `budget_source` — present only when a model/entry block supplied the value,
  so server-defaulted runs keep today's exact log lines.
- **Boundary: the bucket is per provider entry, not per model.** Two models
  tuning the same entry share **one** `rate_limit_rpm` window (one account
  quota); their `budget` blocks change behaviour against that window, never
  its size. Per-model *quota* is out of scope by design.
- **`models.compact` caveat.** `models.compact` is consumed only by the
  compaction summarizer (`proxy/proxy/compaction_summarizer.py`), which calls
  its providers directly over httpx and never reaches
  `acquire_provider_budget()` — a `models.compact.budget` block is accepted and
  resolvable but has **no runtime effect** until that dispatch is gated
  (tracked as LP-0MVFYYYXS0011T58).

**Why the reserve exists (LP-0MVDUN5CY009VD20).** Affinity is recorded on the
first successful dispatch, so *every* later turn of that session counts as
"in-progress". Without a reserve, established sessions queued behind new-session
traffic and waited 15–55 s per turn, which dominated first-token latency
(p90 6.5 s → 12.8 s, p99 21 s → 53 s while generation speed was unchanged).
Reserving `floor(rate_limit_rpm × provider_rate_limit_reserve_ratio)` slots
(4 of 15 at the former budget) keeps a slot available for them, so waits become
rare; the default 15 s cap bounds whatever remains. The reserve rounds down and
is clamped to `rate_limit_rpm - 1`, so at the current budget of 3/min the
reserve is **0**: there are no reserved slots and both new and in-progress
sessions share the 3-slot window.

Affinity is `session → entry`, TTL-scoped by
`server.provider_affinity_ttl_seconds` (default 900), and is **never** shared
across sessions or between sibling entries (e.g. free `munitoai-free` vs paid
`munitoai`). The point of the split behaviour is prompt-cache continuity: a
session already on the free tier keeps its provider, while a batch of new
agent sessions passes straight through instead of queueing behind an exhausted
allowance.

`munitoai-free` ships with `rate_limit_rpm: 3` in `config.yaml` (the former
fast/cheap overlay files were removed with priority-based routing,
LP-0MVDT94F0003VHUA); tune the value from the `remaining=` field in the skip
logs.

**Value history (LP-0MVDVJYN5001BMG4):** raised from 10 to 15 on 2026-10-18
following the measured budget evaluation. Pre-gate proxy-log evidence showed
the upstream 429 onset at ≈20 dispatches/rolling-60 s (429-minute rate: 4–6 %
≤ 19/min, 20 % at 20–29, 47 % at 30–39, 100 % ≥ 40), so 15 sits below the
knee while recovering roughly a quarter of the paid spill observed at 10
(45 % → 27 %). Observation window: begins at the next proxy restart after the
change; a follow-up check-in (24 h later) monitors the upstream 429 rate at
peak and reverts to 10 if it exceeds ~10 % of minutes.

**Value history (LP-0MVF19DKR008VCHO):** lowered from 15 to **2** on 2026-10-19
at the operator's request — the free tier is treated as a bonus tier, so bursts
spill to the paid tiers rather than consuming the free allowance. At the 2/min
budget the reserve ratio (`provider_rate_limit_reserve_ratio`, default 0.3)
rounded down to 0 slots, so new and in-progress sessions both saw the full
2-per-window cap; in-progress sessions could still wait for the window to drain
(bounded by `provider_rate_limit_wait_max_seconds`). Took effect at the next
proxy restart.

**Value history (LP-0MVGGI4Y0003K4AZ):** raised from 2 to **3** on 2026-10-20
at the operator's request — a modest bump that stays far below the measured
upstream 429 onset (≈20 dispatches/rolling-60 s), so the free tier's real
allowance is used a little closer to capacity and requests spill to the paid
tiers less often. The reserve ratio still rounds down
(`floor(3 × 0.3) = 0`), so new and in-progress sessions both see the full
3-per-window cap; in-progress sessions can still wait for the window to drain
(bounded by `provider_rate_limit_wait_max_seconds`). Takes effect at the next
proxy restart.

### Off-peak paid tiers in the `low` chain (LP-0MVF17VP70053JXX)

`low` priority prefers local and queues over paid remote fallback, then uses the
free tier (`munitoai-free`, `rate_limit_rpm: 3`). Because that free tier is the
last *free* entry of the chain, exhausted local slots plus an exhausted free
budget used to leave the request nowhere to go but an upstream 429. `plan`,
`author` and `code` therefore label the three `opencode-go` entries for `low`
as well as `medium`/`high`:

```text
priorities: [low, medium, high]   # opencode-go-3, opencode-go-2, opencode-go
priorities: [medium, high]        # munitoai, deepseek-flash
```

The resulting `low` chain is the model's own provider order filtered by the
label (LP-0MVGPZWY0001JD1D), so it reads directly off the `providers` list:

```text
chain(low, plan)             = [local-qwen3, opencode-go-3, opencode-go-2,
                                opencode-go, munitoai-free, munitoai,
                                deepseek-flash]
chain(low, author | code)    = [local-qwen3, munitoai-free, opencode-go-3,
                                opencode-go-2, opencode-go, munitoai,
                                deepseek-flash]
```

Those entries keep the off-peak `available_times` windows added by
LP-0MSXPHTD9004DZ4P — UTC `00:00-01:00`, `04:00-06:00`, `10:00-00:00` — so the
extra tiers are only eligible outside peak hours; the windows are **not**
widened by this change. The cheap priority window (`server.priority_window`,
default local `Europe/Dublin` 01:00–10:00 = UTC 00:00–09:00) overlaps the
opencode windows in ~3 of its 9 hours (UTC 00:00–01:00 and 04:00–06:00); the UTC
01:00–04:00 and 06:00–10:00 gaps are deliberate paid-spend avoidance.

#### Free tier after the timed accounts in `plan` (LP-0MVWG4U7F0009LTH)

`munitoai-free` draws on a **daily-reset allowance** (LP-0MVT8SAZE00570J5), so
the token it spends is gone for the rest of the UTC day. Placing it directly
after local meant every request that overflowed the local slots burned that
allowance *before* the timed `opencode-go` accounts, even inside the 14 hours
per day those accounts are eligible — wasting free capacity that would
otherwise have covered the gap periods (UTC 01:00-04:00 and 06:00-10:00,
LP-0MU1RXEPL005CK97), when the opencode accounts are unavailable anyway.

The `plan` chain therefore lists `munitoai-free` **after** all three
`opencode-go` entries and **before** the paid `munitoai` tier (`author` and
`code` keep the original order). The entry's own knobs — `rate_limit_rpm: 3`,
`respect_ratelimit_headers`, `max_context_tokens`, `display_name` — and its
absence of `available_times` are unchanged, so the tier stays eligible around
the clock and is still the last **free** resort inside an opencode window once
all three accounts are exhausted.

Capacity vs resilience — the three entries are **not** interchangeable:

| Failure | Quarantine key | Effect on the sibling entries |
|---------|----------------|-------------------------------|
| HTTP 429 / usage limit | `_usage_limit_account_key` = `api_key_env@endpoint:model` | Independent per API key: a 429 on `opencode-go-3` leaves `opencode-go-2` and `opencode-go` eligible → 3× rate-limit capacity (LP-0MSMBWB23009XYPW, LP-0MVCX0E2V001VB5N). The key includes the **model**, so the same key on a different model is a separate account/limit (LP-0MVT8SAZE00570J5 corrected this row). |
| 5xx / stall / terminal error | `_failure_domain_key` = `endpoint:model` | Shared domain: one failure skips all three for the rest of the request → **no** 3× outage resilience (LP-0MSG45I8Q0020N1F). |

The fully paid `munitoai` and `deepseek-flash` entries declare exactly
`priorities: [medium, high]`, so they never join a `low` chain. `munitoai`
carries a 10:00-23:00 UTC `available_times` window (paid dispatches confined
to business hours); `deepseek-flash` keeps its off-peak windows and carries
no spend discipline of its own.

### Prepaid-credit balance opt-in (`use_balance`, LP-0MW46PAU4006SK7Z / LP-0MW46PBD200249WK)

An OpenCode Go account may be configured in the console to fall over onto
prepaid **balance** once its monthly plan window is spent. Each remote
provider entry may declare `use_balance: true|false` so the quota gate knows
which accounts may keep serving on that credit:

| Situation | Behaviour |
|-----------|-----------|
| `use_balance: true` on a quota-managed (`opencode-*`) entry, monthly plan window the **only** blocker | The entry is dispatched and the over-plan spend is billed to prepaid credit — the `dispatch_on_balance` decision (below) replaces the `quota_exhausted` skip. |
| `use_balance: true` with a spent rolling/weekly window, or with `use_balance: false` (or omitted) — the default | A plan-exhausted account waits for its window reset; routing behaves exactly as before the flag existed (the `quota_exhausted` hard stop stands). |
| Malformed value (`"yes"`, `2`, a mapping) | Resolves to `false`; one WARN per entry is logged at load (never a startup failure). |
| Flag on a non-quota-managed entry | Ignored with a debug note — the flag is meaningful only for `opencode-*` accounts. |

Only a real YAML boolean is honoured (fail safe): an omitted key keeps the
existing behaviour, and a quoted string such as `"true"` is *not* accepted.
The shipped `config.yaml` declares the field on the three `opencode-go-*`
accounts in the `plan`/`author`/`code` chains; only `opencode-go` opts in
(`use_balance: true`) — `opencode-go-2` and `opencode-go-3` stay at the
`false` default.

#### The plan-vs-balance dispatch rule

The quota hard stop (`server.quota_exhaustion_percent`, default 99.5) skips a
spent account before any upstream request is attempted. `use_balance: true`
softens exactly one case: the **monthly plan window is the only blocking
signal** — the monthly window is spent (`status` not `ok`, or its percent at
or above the threshold) while rolling and weekly still hold real headroom
(not blocking, and their remaining headroom above `100 - threshold`). Then
the entry keeps dispatching and `_dispatch_on_balance` logs, once per account
per process:

```
dispatch_on_balance (account=OPENCODE_KEY_A, plan_monthly=100.0%, use_balance=true)
```

Anything else keeps the unchanged hard stop (`reason=quota_exhausted`):

- **rolling or weekly spent** — including a window whose headroom is at or
  below `100 - threshold` even with an `ok` status (genuine exhaustion is
  never overridden);
- **no opt-in** — `false`/absent keeps the skip byte-identical;
- **quota behaviour disabled or stale-suspended** — the softening is inert, so
  a stale "plan spent" reading can never bill balance after the plan reset
  (the money guard's stale-state fail-safe survives).

The softening changes **only** the hard-stop predicate. The surge kill
switch, dry-run, stale-state suspension, quota ordering, `available_times`
and the cooldown / usage-limit quarantine gates keep their exact precedence
— a balance-eligible entry is still constrained by its own `available_times`
window and the rest of the resolution gates. Balance absence cannot be seen
from `/v1/usage` (it carries no balance field), so it is handled reactively
from the upstream error itself — see
[Transient overload vs balance exhaustion](#transient-overload-vs-balance-exhaustion-lp-0mw46pbwl008t317):
a `[overloaded]` 429 is a short retryable cooldown, a genuine
balance-exhaustion response (HTTP 402 / a usage-limit body) quarantines the
account with the reset-aware usage-limit quarantine.

### Priority-chain membership (`priorities`, LP-0MVGPZWY0001JD1D)

Priority routing is declared **once**, on the provider definition the operator
is already editing. Each provider entry may carry a `priorities` list of
`low|medium|high`, and the chain for a request is the model's own ordered
`providers` list filtered by the effective priority:

```text
chain(P) = [p for p in providers if P in p["priorities"]]   # order = providers order
```

Rules (all enforced at load by `validate_priority_config` → `ValueError`):

- **No tier implication.** A provider labelled `medium` is *not* thereby
  eligible for `low`; membership is exactly the list written.
- **Model-level opt-in, fail-closed.** If any provider of a model declares
  `priorities`, every provider of that model must declare it, and together
  they must cover `low`, `medium` and `high`. A partially labelled model is a
  load-time error, so adding a provider can never silently join or leave a
  chain.
- **Malformed labels rejected.** A non-list value, an empty list, an unknown
  priority name and a duplicate entry each fail the load.
- **No silent widening.** For a labelled model the filtered chain is used
  as-is; an empty result never falls back to the full `providers` list (which
  would leak the paid tiers during the nine cheap-window hours).
- **Unlabelled models** (no `priorities` anywhere — `embed`, `private`)
  keep the legacy behaviour: every provider is eligible for every
  priority.
- **Per-priority ordering is not supported.** Chain order is always the
  `providers` order, so an entry that must come first for one priority and
  later for another needs a separate entry (or a separate model). This is a
  deliberate limitation of replacing the name-indexed route map.

The former per-model `priority_routes` map (priority → ordered provider
**names**) is **removed** — there is no shim or deprecation window. A config
that still declares it fails at load with a migration message naming the
replacement key; see the README section
[Migrating from `priority_routes`](README.md#migrating-from-priority_routes).

### Per-entry context limits (`max_context_tokens`, LP-0MVF9N31A0006VH5)

Remote tiers have no inherent context ceiling: without a per-entry limit every
request, however large, is dispatched to the first eligible provider. For a
tier whose allowance is expressed per minute, one oversized prompt can spend
the whole window (or trigger an upstream 429), pushing legitimate traffic onto
a paid tier. Each provider entry may therefore declare `max_context_tokens`
(an `int`), the maximum **estimated prompt context** it may serve.

The gate runs in both fallback dispatch loops
(`_proxy_with_remote_fallback_cycle`, `_proxy_with_fallback_cycle`) *before*
`acquire_provider_budget`/dispatch, using the authoritative routing estimate
(`_get_tokenizer_for_model` + `_estimate_effective_prompt_tokens_for_routing` +
`token_estimate_multiplier`) computed **once per request**.

| Situation | Behaviour |
|-----------|-----------|
| Estimated context ≤ entry limit | Entry is dispatched normally (all other gates unchanged). |
| Estimated context > entry limit | Skip the entry: log `routing_skip_provider_context provider=<p> type=<t> status=provider_context_exceeded estimated_tokens=<n> limit=<limit> reason=provider_context_exceeded session=<s> preference=<parsed>`, record attempt `status=provider_context_exceeded` with the estimate and the limit, continue to the next eligible provider. **No cooldown, failure-domain exclusion, usage-limit reset, affinity record or contention-queue entry** — the entry stays eligible for later, smaller requests. |
| `max_context_tokens` absent / `0` / non-positive | Ungated: behaviour byte-for-byte as before. |
| Limit set on a **local** entry | Composes as an additional hard cap: `min(existing local thresholds, max_context_tokens)`. Never loosens `_should_skip_local`. |
| No tier can serve the request (all capped below it) | Only possible if every tier is capped below the estimate; the shipped config prevents it by capping the final `deepseek-flash` tier at the largest upstream window (1,048,576) so an oversized prompt always has a destination. |

Every context skip increments the cumulative
`backend_signals.provider_context_exceeded` counter (mirroring the local
`context_too_large` signal), so the gate is visible on `/health` and
`/admin/metrics` as well as in the `routing_skip_provider_context` log line.

Shipped values (each sourced in a comment next to the entry in
`proxy/config.yaml`, probed 2026-10-19):

| Entry | Limit | Source |
|-------|-------|--------|
| `munitoai-free` | 150000 | free-tier `x-ratelimit-limit-input-tokens` (rolling 60 s input allowance) |
| `munitoai` | 400000 | paid-tier `x-ratelimit-limit-input-tokens` |
| `opencode-go`, `opencode-go-2`, `opencode-go-3` | 1000000 | model-registry context window for `opencode-go/deepseek-v4.1-flash` (Go publishes plan quota as percentages, not per-minute tokens) |
| `deepseek-flash` | 1048576 | `api.deepseek.com` `/models` `context_window` — largest window in the chain |
| `compact` chain | 1048576 | model-registry / `api.deepseek.com` windows |

Some gateways answer HTTP 429 with custom `x-ratelimit-*` headers instead of
the standard `Retry-After` (MunitoAI sends none). The proxy can honour them so
a rate-limited provider is skipped for the upstream's own reset window rather
than being re-probed every ~45 s by the exponential-backoff guess.

Enable it per provider entry, or globally under `server`:

```yaml
- name: munitoai-free
  type: remote
  rate_limit_rpm: 3
  respect_ratelimit_headers: true   # entry flag wins over server.*
```

```yaml
server:
  respect_ratelimit_headers: true   # global default (false)
  ratelimit_cooldown_margin_seconds: 2
```

Behaviour when enabled, on an HTTP 429 only:

- `x-ratelimit-reset-requests`, `-tokens`, `-input-tokens` and
  `-output-tokens` are parsed; durations accept bare seconds or unit suffixes
  (`52s`, `1m`, `1h30m`, `250ms`).
- When one or more dimensions report `x-ratelimit-remaining-*` at or below
  zero, the longest reset among those exhausted dimensions is used; otherwise
  the longest parseable reset is used.
- The cooldown becomes `max(exponential backoff, reset + margin)` — the
  upstream reset is a floor, and the per-provider failure counter still
  extends it under repeated failures. `Retry-After`, when present, still wins
  if longer.
- `x-ratelimit-over-limit` is parsed and recorded for diagnostics; a 429 that
  carries the flag but no parseable reset falls back to plain backoff.
- The resolved cooldown is recorded in the attempt (`cooldown_seconds`,
  `ratelimit_reset_seconds`, `ratelimit_over_limit`) and persisted in
  `provider-state.json`, so the `Skipping provider=… in cooldown (Ns
  remaining)` log line shows the header-derived wait.
- On a **quota** 429 (per-account rate-limit error or
  `x-ratelimit-over-limit: yes`) the parsed reset is *also* applied as an
  account-scoped quarantine (reset + the 2-minute usage-limit margin), keyed
  on `api_key_env` so same-gateway paid siblings stay eligible — see
  [Account-scoped quarantine from reset headers](#account-scoped-quarantine-from-reset-headers-lp-0mvfa1vrt007t9ht).

Default is **off**: every existing provider keeps its pre-change behaviour.
The `munitoai-free` and `munitoai` entries opt in via `config.yaml`.

#### Account-scoped quarantine from reset headers (LP-0MVFA1VRT007T9HT)

A reset header on a **quota** 429 is applied to the failing ACCOUNT, not just
the provider entry. This is the account-scoped half of the header support
above; the entry-level cooldown described there is unchanged.

- **Trigger:** an HTTP 429 whose body is a per-account quota error
  (`type: rate_limit_error`, `code: rate_limit_exceeded`,
  `type: insufficient_quota` or `code: free_daily_limit_exceeded`, top-level or
  nested under `error` — the observed MunitoAI bodies) **or** which carries
  `x-ratelimit-over-limit: yes`, **and** which also yields a parseable
  `x-ratelimit-reset-*` duration. No reset signal (or a zero/absent value)
  means no account quarantine: the pre-existing short provider cooldown and
  per-request account-key exclusion apply exactly as before. A daily free-limit
  429 is handled earlier by the usage-limit reset path instead, so its
  advertised daily window wins over the rolling header window
  (LP-0MVT8SAZE00570J5).
- **Duration:** `parsed reset + _USAGE_LIMIT_RESET_MARGIN_SECONDS` (120 s).
  Two distinct margins apply on such a 429 and they are **not**
  interchangeable: the **2-second** `server.ratelimit_cooldown_margin_seconds`
  pads the provider-ENTRY cooldown, while the **2-minute** usage-limit margin
  pads the persisted ACCOUNT quarantine.
- **Account key:** `_usage_limit_account_key` = `api_key_env` @ the
  entry's failure domain (`endpoint:model`). Entries without an
  `api_key_env` fall back to the failure-domain key. Because the key is the
  API-key account, a paid sibling on the same gateway (e.g. `munitoai` on
  `MUNITOAI_API_KEY` after `munitoai-free` on `MUNITOAI_FREE_API_KEY` hit its
  quota) has an independent key, stays unquarantined, and remains eligible
  for the rest of the request (LP-0MVCX0E2V001VB5N).
- **Consequences:** the quarantine is recorded in `_usage_reset_at` as an
  **explicit** reset (so it is exempt from the guessed-quarantine 24 h cap and
  the self-healing probe), is persisted to `provider-state.json` and therefore
  survives a proxy restart, is recorded in the attempt
  (`account_quarantine_seconds`) and in the exhaustion diagnostics
  (`usage_limit_reset` with `reset_in`/`reset_at`), and feeds
  `_compute_retry_after()` — so an exhausted chain-hold reports the reset as
  its `retry_after` instead of a blind backoff. Chain-hold cycles themselves
  (`chain_hold_seconds` 300 s, `chain_hold_max_cycles` 3, `chain_hold_total_seconds`
  720 s) are unchanged: the account is
  excluded, the request advances down the chain, and siblings keep their
  normal eligibility.
- **No window metadata:** a header-derived quarantine is not tied to a quota
  window, so it is revalidated as a *legacy* entry — the first fresh
  `/v1/usage` poll that shows no window exhausted clears it (see the
  usage-limit section below).
- **Flag gate:** the header is only consulted when
  `respect_ratelimit_headers` is enabled for the entry (or globally); with it
  off the reset is ignored and no account quarantine is created.

### Responses-API providers (`api: openai-responses`)

Some upstreams expose **only** the OpenAI Responses API (`/v1/responses`) and
return HTTP 500 on `/v1/chat/completions` (e.g. the muse model family on
`https://opencode.ai/zen`). For those providers, set `api: openai-responses`
on the remote config entry (LP-0MTGK5DQO001Y8H0):

```yaml
- name: opencode-zen-muse-free
  type: remote
  endpoint: https://opencode.ai/zen
  api_key_env: OPENCODE_API_KEY
  model: muse-spark-1.2-contributor-free
  api: openai-responses
  forward_session_headers: false
```

Behavior when enabled:

- The request path is rewritten from `v1/chat/completions` to `v1/responses`.
- The request body is translated: `messages` → `input` (tool-role messages
  become `function_call_output` items, assistant `tool_calls` become
  `function_call` items), `max_tokens`/`max_completion_tokens` →
  `max_output_tokens`, `reasoning_effort` → `reasoning.effort`.
- Streaming responses are translated Responses SSE → chat/completions SSE
  (`response.output_text.delta` → content deltas,
  `response.function_call_arguments.delta` → tool_calls argument deltas,
  `response.completed` → finish chunk + `[DONE]`), so fallback, watchdog,
  retry, and empty-detection machinery keeps working unchanged.
- Non-streaming responses are translated from Responses JSON (`output[]`,
  `usage.{input,output,total}_tokens`) to the chat/completions JSON shape.

The client always sees chat/completions wire format; `api: openai-responses`
is purely an upstream-adaptation flag. Omit it (default:
`openai-completions`) for normal providers.

## Example Config Structure

```yaml
models:
  openai:                              # Direct name match
    aliases: [gpt-*, o1-*]            # Wildcard aliases
    type: remote
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    
  qwen3:                              # Local model
    aliases: [qwen3*]
    type: local
    llama_model: Qwen3                # Actual model name for llama-server
```

## Routing Examples

- `{"model": "gpt-4"}` → matches `openai` config via `gpt-*` wildcard alias → routes to OpenAI API
- `{"model": "Qwen3"}` → matches `qwen3` config directly → routes to local llama-server
- `{"model": "claude-3-opus"}` → matches `anthropic` config via exact alias → routes to Anthropic API
- No model specified + `current_model` is set → uses the currently loaded local model

## Prefill progress signal provenance (LP-0MW7V2DK400365ER)

During a large local prefill no stream chunk is emitted, so the dispatch lease
and the occupancy surface need a *measured* prefill progress signal to tell
"still working" apart from "stalled". This section records where that signal
comes from, in the order the proxy consults it.

### Sources, in preference order

The probe is `proxy.router_helpers._query_prefill_progress`, polled every
`local_dispatch_lease_prefill_poll_seconds` while the dispatch awaits its first
byte. Each source is bounded by `STATUS_QUERY_TIMEOUT` (default 1 s) and never
blocks the stream loop.

1. **Per-slot `/slots`** (`proxy.observability._query_slots_progress`), on the
discovered model-child port (never the router). A slot's progress is the
maximum of `n_past` and `n_prompt_tokens_processed`; liveness is
`is_processing`. Older llama.cpp builds publish these per-slot fields firmly.
2. **Per-slot log tail** (`proxy.observability.get_slot_prefill_progress`).
   llama-server writes, for every prompt-processing batch:

   ```
   slot update_slots: id  0 | task 2833255 | prompt processing progress, n_tokens = 17690, batch.n_tokens = 1024, progress = 0.999774
   ```

   stderr, which the proxy redirects into `llama-server.log` and parses into
   `_slot_progress_cache` (see `_update_slot_progress_from_log`). The measured
   token count is `n_tokens`; an entry is trusted while the log's mtime is
   within `SLOT_PROGRESS_MAX_AGE_SECONDS` (60 s). This source is what makes
   progress observable on builds that dropped the `/slots` fields (below).
3. **Aggregate `/status`** (`proxy.observability.query_llama_status`), reading
   `kv_cache_tokens` (or `n_past` when present) from `/model`, `/status`,
   `/models`, `/v1/models`. It is a whole-server aggregate, not per-slot, so it
   is the last resort.

### Build-specific caveats

- The deployed llama.cpp **b8782** build removed `n_past` /
  `n_prompt_tokens_processed` / `context_tokens` from the `/slots` response.
  Observed 2026-11-08 on the live child: `/slots` returns `id`, `n_ctx`,
  `is_processing`, `id_task`, `params` and `next_token.{n_decoded,n_remain}` —
  **no per-slot prompt-progress field**.
- That build still emits the `slot update_slots … n_tokens = N` stderr line,
  so source 2 is the supported measured signal there. Nothing in the proxy
  fabricates `n_past` when a build omits it (a fabricated context size would
  mis-price the ETA).
- `query_llama_status`'s `kv_cache_tokens` is build-dependent; `/status`
  itself is *not* a llama.cpp endpoint (it 404s), so the aggregate reads the
  model-list endpoints and returns `null` when the field is absent.

### Unobservable case (fail-safe)

When all three sources yield nothing **and** the slot is not observed
processing, progress is *unobservable*. The proxy then:

- leaves the lease at its adaptive acquisition estimate (no unbounded
  extension — a genuinely progress-less dispatch is still released by
  `local_dispatch_no_progress_timeout_seconds` and the
  `local_dispatch_max_prefill_seconds` ceiling);
- keeps the liveness-only extension path (`is_processing`) as the fail-safe;
- emits a throttled `prefill_progress_unobservable` WARNING (at most once per
  `_PREFILL_WARN_INTERVAL`, 60 s); and
- increments the `proxy_prefill_progress_unobservable_total{reason="no_source"}`
  counter on **every** blind poll, so a persistent rate is visible in
  `/metrics` and not only in the log
  (LP-0MW8VPH2Q007XPHQ, parent AC6/AC11).

### How the signal is consumed

- **Lease extension** (`_extend_lease_during_prefill`): a numeric advance
  refreshes `last_progress_ts` and logs
  `lease_extended_on_measured_progress` / `reason=measured_progress`; liveness
  or an in-flight own-session request extends without refreshing the clock.
- **Early abort** (`_watchdog_release_reason`): `prefill_eta_exceeded` when
  elapsed > `eta_prefill_s * local_prefill_eta_factor` (default 2×),
  `prefill_stalled` when no progress for
  `local_dispatch_no_progress_timeout_seconds`, and `prefill_max_exceeded` at
  the hard ceiling. The `lease_released` WARNING records the reason plus
  `elapsed_s` / `stalled_for_s` (parent AC4/AC14).
- **Occupancy** (`measured_occupied_slot_ids`): the union of `/slots`
  `is_processing` and fresh log-derived progress; published additively as
  `measured_in_flight_requests` so the residual analyser no longer under-counts
  a busy second slot (LP-0MW8VP0XR007UGQF).

## Cold (measured) prefill calibration (LP-0MW957Q41002EXE2)

Q410 makes the large-prompt rate honest: the mixed calibration window is
polluted by instant cache hits (measured: 85 751 tokens in 2.6 s), which puts
the fitted rate 1.4–1.7× above the genuinely-cold regime (measured
3.81–4.26 ms/token ≈ 235–295 tok/s). The fix is a third per-`(model, bucket)`
window that only ever holds **measured** cold prefill samples.

### Sample predicate

A sample enters the cold window only when `measured_cached_ratio < 0.10`
**and** `processed_tokens >= 32 000` (first byte `>= 1 s`), so instant
cache-hit completions can never price the cold rate. The samples come from
the measured-prefill capture (`MeasuredPrefill.measured_cached_ratio` /
`processed`, the same `prefill_progress_measured` flow documented above) at
the point the measurement is finalised — the existing priced window
(`record_dispatch_sample`, `dispatch_first_byte_ms`) is untouched.

### Selection

- Prompts **below** `local_admission_guard_eta_percentile_min_tokens` keep the
existing mixed/contended calibration byte-for-byte (no small-prompt
regression).
- Prompts **at or above** the percentile floor are priced from the cold window
only. When the cold window is not yet calibrated (`cold_samples`
`< local_admission_guard_min_samples`), large-prompt pricing is **inert** —
the guard admits exactly like an uncalibrated bucket, it never falls back to
the mixed window's optimistic rate. The conservative percentile path
(`local_admission_guard_eta_percentile`, default p75, never below the cold
window's p50) still applies, now to the cold distribution; a percentile of
`0` restores the point estimate byte-for-byte.

### Observability

`rate_bucket=cold cold_samples=<n>` on the admission decision/log line and
`prefill_eta_cold_samples` on `/llama/local/status` make the honest split
legible: a `cold_samples` count below the minimum explains an inert
large-prompt admission, and a calibrated `rate_bucket=cold` decision states
exactly which window priced it.

## Large-Context Routing: Mode-Dependent Economic Bypass (LP-0MU5A4QBR003YJM0)

When a request's estimated token count exceeds the cold-cache threshold
(`local_large_context_cold_cache_threshold`), the proxy applies a two-tier
check to decide whether to route local or skip to a remote provider:

1. **Physical capacity check** (`context_too_large`): if the estimated tokens
   exceed the warm-cache threshold (per-slot capacity clamp), the request
   cannot fit the KV slot and always bypasses local.
2. **Economic cold-cache check** (`large_context_bypass`): if the estimated
   new tokens (estimated × (1 - cached_ratio)) exceed the cold threshold,
   the prefill cost is too high for local.

**Mode-dependent behaviour for the economic check** (selected by the active
mode profile's ``local_large_context_economic_bypass_serves_local`` flag —
not a hardcoded mode string):

| Mode | `local_large_context_economic_bypass_serves_local` | Economic bypass behaviour |
|------|-----------------------------------------------------|--------------------------|
| **cheap** | `true` | Request proceeds to local; if all slots are busy it enters the contention queue (bounded by depth and wait caps). The request is **not** recorded as `large_context_bypass` — it is served locally (or queued). |
| **fast** | absent / `false` | Request skips local and falls back to the next remote provider. The skip reason is `large_context_bypass`. |

The physical capacity check applies in both modes — requests that cannot fit
the KV slot are never served locally regardless of mode.

**Observability:** when cheap mode serves an economic-bypass request locally,
the proxy logs a `routing_economic_bypass_local` line with the token counts
and cached ratio. The `routing_skip_local` line (with `reason=large_context_bypass`)
remains parseable by reporting tools and only appears for fast mode.

## Restored-prefix warmness: seeding `cached_ratio` from the slot snapshot (LP-0MVGD9YXH008O304)

The economic check above is only as good as `cached_ratio`. Historically that
ratio came *only* from `provider._last_cached_ratio`, an in-memory dict written
by a previous local response — so a session whose KV prefix was already saved to
disk was still scored cold (82.8 % of routing decisions reported
`cached_ratio=0.00`), every session above 38k tokens was permanently bypassed
(`large_context_bypass`), and each proxy restart reset the whole fleet to cold.

`proxy/proxy/slot_cache_ratio.py` closes that gap with a sidecar written next to
each snapshot (`slot_<session>.bin` → `slot_<session>.bin.ratio.json`):

- **Writer** — after a *successful* `slot_save`, the proxy records the
  prompt-token count that is now resident in the saved KV cache plus a SHA-256
  fingerprint of every message in the request. Both save paths write it: the
  request-completion path and the session **eviction** path, and both couple
  the sidecar's lifetime to the snapshot's (see
  [Lifecycle coupling](#snapshot--sidecar-lifecycle-coupling-lp-0mvk0e80c009huqc)
  below).
- **Reader** — before the cold-cache gate reads the ratio, the snapshot's
  sidecar is validated and used to seed it:

  ```
  cached_ratio = min(1.0, cached_tokens / estimated_tokens) * safety_margin
  ```

  with `session_slot_ratio_safety_margin` (default `0.9`) and
  `session_slot_ratio_ttl_seconds` (default `86400`) configurable on the
  `server` block. The measured fraction is a hard ceiling, so restoration can
  never over-credit (AC3), and the margin absorbs llama.cpp dropping its own
  checkpoints (`forcing full prompt re-processing`).
- **Correctness guard (AC2)** — the recorded message fingerprints must be a
  prefix of the incoming message list. A changed system prompt or a history
  truncated from the front fails the check and earns `cached_ratio = 0`; a
  weak partial credit is still subject to the normal cold-cache gate, so it
  cannot mask an expensive prefill.
- **Fail-open** — missing/corrupt/expired sidecar, other model, absent
  `session_slot_save_path`, or any error degrades to the previous behaviour
  (`cached_ratio = 0`, conservative). Session invalidation retires the snapshot
  and its sidecar together and clears the in-memory ratio.
- **Restart-safe (AC6)** — the sidecar lives on disk, so the first request of a
  session after a proxy restart is scored warm when the snapshot is present
  and valid.

**Observability (AC4):** `routing_check`, `routing_skip_local`,
`routing_economic_bypass_local` and `dispatch_first_byte_ms` now carry
`cache_state=cold|warm|restored`. `restored` means the ratio was seeded from a
snapshot *and* cleared the warm threshold, so restored-prefix wins are counted
separately from genuine prompt-cache hits. `slot_ratio_restored` is logged once
per seed with `cached_tokens`, `estimated_tokens` and the credited
`cached_ratio`.

### Seed-outcome instrumentation (`slot_ratio_seed`, LP-0MVJYZWWW005XGW7)

A ≥24 h measurement (LP-0MVGGPSVM003W35K) found the seeding path made **zero
contact** in 54.4 h while 33 sidecar-bearing sessions hit the local routing gate
2,918 times — yet the only diagnostic was a DEBUG line, invisible at the INFO
level the proxy runs at. Every seed decision for a session that *could* have a
snapshot therefore emits one INFO line (behaviour-neutral — no credit decision
changed):

```
slot_ratio_seed outcome=<outcome> session=<8> model=<model> stored_messages=<n> incoming_messages=<n> age_s=<n> cached_ratio=<x.xx>
```

| `outcome` | Meaning |
|-----------|---------|
| `found` | Sidecar validated and a credit was applied (also emits `slot_ratio_restored`). |
| `missing_sidecar` | `slot_<session>.bin` exists but no readable `.ratio.json` sidecar was written for it (absent, corrupt, or unknown schema version; `stored_messages=0`, `age_s=0`). |
| `missing_snapshot` | A sidecar exists but its snapshot is gone — an orphaned sidecar (counts/`age_s` read from the sidecar). |
| `expired` | `age_s` exceeds `session_slot_ratio_ttl_seconds` (default 86 400). |
| `model_mismatch` | The sidecar was written for a different `llama_model`. |
| `prefix_mismatch` | The stored message fingerprints are not a prefix of the incoming messages (rewritten system prompt or client-compacted history). |
| `empty_ratio` | The sidecar produced no credit (`cached_tokens` or `estimated_tokens` is 0). |

Sessions with **no snapshot and no sidecar at all** stay silent — that is the
common case, so log volume is bounded by snapshot-bearing sessions only. The
line supersedes the old DEBUG `slot_ratio_sidecar skipped reason=prefix_mismatch`
record: there is no duplicated DEBUG/INFO pair for the same event.

**Aggregating the outcomes** (per reporting window, e.g. 24 h of
`llama-proxy.log`):

```bash
for o in found missing_sidecar missing_snapshot expired model_mismatch prefix_mismatch empty_ratio; do
  printf '%-18s %s\n' "$o" "$(grep -c "slot_ratio_seed outcome=$o" llama-proxy.log)"
done
# Age distribution of expired sidecars (is the 24 h TTL binding?):
grep -o 'slot_ratio_seed outcome=expired .*age_s=[0-9]*' llama-proxy.log | grep -o 'age_s=[0-9]*'
# Message-growth pattern for prefix mismatches (stored vs incoming):
grep 'slot_ratio_seed outcome=prefix_mismatch' llama-proxy.log | grep -o 'stored_messages=[0-9]* incoming_messages=[0-9]*' | sort | uniq -c | sort -rn | head
```

A high `prefix_mismatch` share points at the guard (rewritten fronts); a high
`expired` share points at the TTL; `missing_sidecar`/`missing_snapshot` point at
the writer or at slot-file churn; a high `found` with no `slot_ratio_restored`
lines would indicate an error inside `set_cached_ratio`.

**Prefer the tallying tool to hand-greps** (rotated and `.gz` logs, no double
counting, windows handled once):

```bash
python3 scripts/analyse_slot_ratio_restore.py \
  --log-dir /var/log/llama-proxy \
  --slot-cache-dir ./slot-cache \
  --window "last24h=2026-10-28T10:50:26,2026-10-29T10:50:26"
```

It splits the outcomes into the two **lookup** outcomes (`missing_sidecar`,
`missing_snapshot`) and the **guard** outcomes (`expired`, `model_mismatch`,
`prefix_mismatch`, `empty_ratio`, `found`) under `seed_outcomes`, because that
split is what identifies the failure mode: guard outcomes are only reachable
after a snapshot **and** a readable sidecar were found together, so a zero guard
total across a window means the seed path never got past the lookup and the
guard (prefix / TTL) cannot be the cause. The first such verdict, and the two
writer-side gaps that produce it, are recorded in
[`docs/dev/slot-ratio-seed-verdict-LP-0MVK0E8C40035FG3.md`](../../docs/dev/slot-ratio-seed-verdict-LP-0MVK0E8C40035FG3.md).

### Snapshot ↔ sidecar lifecycle coupling (LP-0MVK0E80C009HUQC)

The verdict above showed the seed path never reached its guard at all: over
4,488 decisions in 155 h, `missing_sidecar` (3,402; 75.8 %) and
`missing_snapshot` (1,086; 24.2 %) accounted for **100 %** of outcomes, and
`found` / `expired` / `model_mismatch` / `prefix_mismatch` / `empty_ratio` were
**all exactly 0**. A guard branch is only reachable once a snapshot *and* a
readable sidecar co-exist, so the defect was never the guard, the TTL or the
credit arithmetic — it was **artefact pairing on the writer side**. Three gaps
are now closed, and `prefix_matches()`, `ttl_seconds()` and
`min(1.0, cached/estimated) × margin` are byte-for-byte unchanged.

**What was fixed**

| Failure mode | Before | Now |
|---|---|---|
| Eviction saves were unseedable | `session.save_evicted_session_snapshot()` wrote only `slot_<session>.bin` (98 saves across 32 sessions, overlapping 17 of the 31 `missing_sidecar` sessions) | the save writes the sidecar too, from the session's recorded prompt prefix |
| Request saves without a measured usage block got no sidecar | the write was gated on `_measured_prompt_tokens > 0` | the write is unconditional; the token count falls back to the recorded prefix, then to the router's prompt estimate |
| Sidecar lifetime was not coupled to the snapshot's | the sidecar was deleted only when a caller passed a filename; nothing else pruned it (139 orphans against 0 snapshots on disk) | every snapshot unlink goes through one retirement helper (root layout *and* `{host}-{port}` subdirectories), and `scripts/cleanup-slot-cache.sh` recurses, deletes the pair together and prunes orphan sidecars |

**Which token count gets recorded.** The sidecar must not invent a
measurement, so `provider.sidecar_cached_tokens()` resolves it in decreasing
order of trustworthiness:

1. the upstream `usage` block's `prompt_tokens` for this request (a direct
   measurement — unchanged behaviour);
2. else the prompt prefix recorded by `record_prompt_prefix()` on an earlier
   *measured* local turn, but only when its fingerprints prefix-match the
   saved messages (then it is a genuine lower bound for the saved KV);
3. else `_estimate_prompt_tokens_for_routing()` — the same tokenizer the
   routing decision uses, counting message text only (it ignores tool
   definitions and per-message overhead, so it under-counts rather than
   inflating credit);
4. else **no sidecar at all** — a snapshot with no trustworthy count stays
   unpaired rather than carrying an invented number (which would be reported
   as `empty_ratio` or, worse, over-credit).

The eviction path has no request body, so it only ever uses rule 2: a session
that has never produced a measured local turn gets no sidecar from an eviction
save.

**What still earns nothing** (unchanged, deliberately):

- a **changed prompt** — a rewritten system prompt or a front-truncated history
  fails `prefix_matches()` and credits `0` (AC2 of LP-0MVGD9YXH008O304);
- an **expired sidecar** — older than `session_slot_ratio_ttl_seconds`
  (default 86 400 s);
- a **mismatched model** — the sidecar was written for a different
  `llama_model`;
- a **weak partial credit** — still subject to the normal cold-cache gate, so
  it cannot mask an expensive prefill;
- the **in-memory expected-reuse credit** — `provider.expected_cached_ratio()`
  keeps the same `prefix_matches()` rule for both call sites.

**Expected post-fix signature** (the ≥24 h re-measurement,
LP-0MVK0E8NH007XCV3):

- `slot_ratio_seed outcome=missing_sidecar` ≈ 0 for sessions whose snapshot was
  taken by the eviction path (they now carry a sidecar);
- `slot_ratio_seed outcome=missing_snapshot` ≈ 0 (orphan sidecars are pruned
  with their snapshots and by retention);
- **`found` > 0** and `cache_state=restored` dispatches > 0 — the first real
  evidence that the feature engages;
- `prefix_mismatch` and `expired` become *possible* for the first time. A
  non-zero `prefix_mismatch` count is a **success signal** (the guard is
  finally running on real traffic), not a regression — do not "fix" the guard
  in response to it. It is the next measurement's question, and only then is
  the guard itself the binding constraint.

Aggregate it with the same recipe as the other outcomes, adding the two
derived counters:

```bash
# Orphan sidecars still on disk (should trend to 0 as retention prunes them):
find "$SLOT_CACHE_DIR" -name '*.bin.ratio.json' | while read -r s; do
  [ -f "${s%.ratio.json}" ] || echo "orphan: $s"
done | wc -l
# Pairs restored warm (should be > 0 after the fix):
grep -c 'slot_ratio_restored ' llama-proxy.log
```

#### Retention and pairing in `scripts/cleanup-slot-cache.sh`

The cleanup script now walks the legacy root **and** one level of
`{host}-{port}` endpoint subdirectories, deletes a snapshot and its sidecar as
one unit under both the age and size rules, and prunes `.ratio.json` files
whose snapshot is gone (root and subdirectories alike, `--dry-run` honoured).
Before this, it globbed `"$SLOT_CACHE_DIR"/slot_*.bin` only, so with every live
snapshot in `localhost-8080/` it printed `No slot files found` and never touched
an orphan.

The orphan sweep is **age-bounded like the other two rules**
(LP-0MVUJBFRL005VDNA). A sidecar that lost its snapshot moments ago may belong
to a save still in flight, or to a save that failed between writing the sidecar
and the snapshot, so recent orphans (newer than `--min-age-days`, default 1 day)
are kept and reported as kept in the summary, while aged orphans are pruned.
`--min-age-days 0` disables the protection, exactly as it does for the size
rule — so all three deletion rules share one mental model: age-bounded, never
deleting anything inside the protective window.

## Capacity-based local admission (LP-0MVGDUVS6003T2IX)

The economic gate (`local_large_context_cold_cache_threshold`, 38 000 new
tokens) and the physical cap (per-slot clamp, ~126 976 at N=2 / ctx 262 144)
are **unchanged**. Raising them was rejected twice (LP-0MVGCOZMZ003I6PF,
LP-0MVGD9YLM0055SFF): the 38k–124k band needs ~100 slot-hours per 24 h when
cache-cold against ~26 h of measured idle headroom, so it only fits with warm
caches plus admission control.

Instead, the economic band may be admitted locally while the **projected
slot-seconds** of admissions fit inside a share of **measured** idle slot time:

```
projected_s = prefill ETA (calibrated from dispatch_first_byte_ms) + decode allowance
busy_s      = union of local dispatch intervals inside the horizon
idle_s      = slots * horizon - busy_s
budget_s    = local_capacity_admission_budget_ratio * idle_s

admit iff admitted_s + projected_s <= budget_s
```

```yaml
server:
  local_capacity_admission_enabled: false   # ships off; wiring is now correct
  local_capacity_admission_budget_ratio: 0.8
  local_capacity_admission_horizon_seconds: 3600
  local_capacity_admission_decode_seconds: 5
```

- **Not a threshold widening.** Both existing thresholds keep their values;
  only the *effective* treatment of the economic band changes, and only while
  measured idle capacity supports it.
- **Rolling, data-driven inputs.** `busy_s` is recorded from real dispatch
  intervals (`router.py::_cleanup_after_request`), and the prefill ETA comes
  from `proxy/prefill_eta.py`'s calibration over observed
  `dispatch_first_byte_ms` samples — no fixed capacity constants.
- **Burst-safe.** Admissions are recorded in a rolling ledger inside the
  horizon, so a burst of band requests cannot collectively over-commit the
  budget; once the budget is exhausted the bypass stands (today's behaviour).
- **Never overrides the physical cap.** `context_too_large` still routes
  remote; a prompt that cannot fit a KV slot is never admitted by capacity.
- **Never overrides the prefill deadline** (LP-0MVXWQQ20002HAX4). The gate
  applies the admission guard's *own* deadline bounds to the projected
  **prefill** term (decode allowance excluded): `prefill_s <
  local_prefill_deadline_seconds` and, when configured, `prefill_s <=
  local_admission_guard_budget_safety_fraction × local_prefill_deadline_seconds`.
  Before this bound existed the gate could admit a prompt the deadline would
  then cancel, so enabling it only ever bought a wasted dispatch.
- **The flat cold-prefill cap yields to a capacity admission**
  (LP-0MVXWQQ20002HAX4). A request admitted here is stamped on the request
  (`provider.LOCAL_CAPACITY_ADMISSION_ATTR`) and the dispatch guard no longer
  refuses it on the flat `local_cold_prefill_budget_tokens` count; the
  guard's measured ETA/deadline/safety bounds still apply. Without that
  deference the two gates contradicted each other — routing admitted the band
  and the guard then always refused it, so enabling this key was a no-op.
  Every other path (the idle economic bypass included) keeps the flat cap, and
  disabling the gate is a complete rollback.
- **What bounds the live effect.** The deadline bound prices the band with the
  same conservative estimator the guard uses, so the admission rate is
  whatever the *measured* prefill rate supports. The band starts at
  `local_large_context_cold_cache_threshold` (38 000 new tokens); at the
  guard's conservative ~3.9 ms/token that is ≥ 148 s, beyond the safety
  budget (derived from `local_admission_guard_budget_safety_fraction`, see
  [Measured cache-deficit early abort](#measured-cache-deficit-early-abort-lp-0mvr7in0o000dpxn-benefit-gate-lp-0mvwbd91z004u135)), so with today's pricing the gate admits **nothing** and the key
  ships **off**. It becomes productive once the per-bucket calibration (a
  measured, faster rate) makes a band prompt fit — that is the documented
  enablement precondition, and `routing_capacity_admit` firing on a real band
  prompt is the signal to flip it.
- **Observability.** Each admit logs `routing_capacity_admit provider=… model=…
  reason=capacity_admission projected_s=… prefill_s=… admitted_s=… budget_s=…
  idle_s=… budget_ratio=… horizon_s=… deadline_s=… deadline_budget_s=…
  estimated_tokens=… cached_ratio=… cache_state=… warm_threshold=… session=…`
  and records a `status=capacity_admit` attempt.
- **Rollout / rollback.** Enabled 2026-11-01 once cache warmth
  (LP-0MVGD9YXH008O304), the prefill-ETA guard (LP-0MVGDUV5E005UWBQ) and
  prefill serialisation (LP-0MVGDUUV3001XFRJ) were live; revert by setting
  `local_capacity_admission_enabled: false` (read per request, no restart).
  The decode allowance was re-measured for the same change (109
  completion tokens per local stream ≈ 4.4 s, not the 30 s placeholder).
  The three prerequisites are live, but the 2026-11-01 measurement shows the
  gate admits nothing at the current conservative pricing, so it **ships
  off** (LP-0MVXWQQ20002HAX4); the deadline bound and the flat-cap deference
  above are what make enabling it safe and effective once calibration
  supports it.

## Routing occupancy counts other sessions, not your own lease (LP-0MW0VH2LB008JSKQ)

Two gates decide local admission, and they must agree about the **requesting
session's own** dispatch record:

| Gate | Where | Own record |
| --- | --- | --- |
| routing concurrency (`cur_local >= max_local` → `local_concurrency_limit`) | `_proxy_with_fallback_cycle` → `_local_concurrency_info` | **idle lease excluded** |
| dispatch lease (`_try_acquire_local_dispatch`) | lease choke point | same-session re-acquisition permitted (`own_has_lease`) |

Before this, the routing gate counted *every* active-or-unexpired record —
including the requester's own — so it fired first and the lease gate's
own-lease exemption was unreachable. The 30 s no-preemption retention
(`local_dispatch_lease_timeout_seconds`) exists so a returning session keeps its
slot; the routing gate then denied it that slot, and because `remote-ok` (the
default preference) never queues, the request stepped straight to a remote
provider.

Measured 2026-11-03 (14:25–15:10 UTC, `session_slot_pool_size: 2`): of 82
`local_concurrency_limit` fall-throughs, **15 (18 %)** had the session's own
local turn finish within the 30 s lease timeout — `cur_local = 2` with
`generating = 0`, i.e. both "occupied" slots were finished turns inside their
idle lease, one of them the returning session's own. Three sessions oscillated
local→remote for no capacity reason, one of them pushed remote on four
consecutive turns 4.5–26.2 s after its own local turn.

**The rule.** `_get_local_concurrency_info(..., session_id=…)` excludes the
requesting session's own record **only while it is an idle lease**
(`active: false`). A record that is still `active` is genuine in-flight work, so
a second concurrent request from the same session is gated exactly as before.
`session_id=None` (anonymous callers, and the pool-description readers behind
`/llama/local/status` and `/llama/local/reserve`) excludes nothing, so the
engine-capacity view and the pure-legacy global-counter path are unchanged.

## Dispatch-lease occupancy and the denial line (LP-0MW0KZTWC004HXQ5)

`_try_acquire_local_dispatch` (`proxy/router_helpers.py`) is the single
admission choke point for local work. Its occupancy view and the capacity
view published to clients are **different questions with different sources**,
and conflating them is what made the 2026-11-03 incident undiagnosable.

| Question | Source | Where |
| --- | --- | --- |
| May this session dispatch **now**? | in-process dispatch-lease records + the generating-only and prefill-in-flight counters | `_try_acquire_local_dispatch` |
| Does the **engine** have a free slot? | llama-server's `/slots` (fresh, else the bounded last-known counts) | `_query_slots` → `/llama/local/status` |

`available_slots` / `total_slots` are **never lease-derived**. They come from
llama-server's `/slots`, with the documented fallbacks in
`proxy/handlers.py` (last-known counts when the query fails, the configured
`session_slot_pool_size` when the model is definitively unloaded) and the
documented upward reconciliation with live in-flight work when the served
*detail* is stale. A second derivation from lease records would let the two
views drift apart silently again; there is exactly one.

### Stale records never block (AC1)

Before counting, the admission path runs `_prune_stale_dispatch_records`: a
record is stale only when it is **inactive and past `expires_at`** (the request
finished and its post-request cooldown elapsed). A record that is still
`active` past its expiry is live work whose lease lapsed — the 180 s
no-progress watchdog's concern, never the admission gate's — so it is left for
the watchdog. Pruning also releases the freed session-slot assignment, so the
slot registry cannot describe a slot the gate has already forgotten.

### The denial line carries its evidence (AC2)

Every `local_dispatch_denied` line is followed on the SAME line by an
occupancy snapshot, so a refusal is diagnosable from the log alone:

```
local_dispatch_denied session=… owner=… active=… cold_start=… <turn>
  records_total=… records_active=… records_unexpired=… records_stale=…
  active_counter=… max_local=… generating=… prefill_in_flight=… prefill_cap=…
  llama_available_slots=… llama_total_slots=…
  records=<key>:<active|idle>:<expires_in=Ns|expired>,…
```

`records=` is `none` for an empty registry; the aggregate fields always appear
(`None` when unavailable) so "no data" is never confused with "zero".
`prefill_cap` is the *effective* cap (`local_max_concurrent_prefills`, clamped
to the slot pool), which is what the prefill branch actually compares against.
The field vocabulary is published as `OCCUPANCY_EVIDENCE_FIELDS` so the log
contract cannot drift from the code.

### Divergence is logged, not shipped silently (AC3)

When a refusal contradicts the occupancy evidence, a
`local_occupancy_divergence reason=… session=… owner=… <evidence>` warning is
emitted beside the denial (reading the last known llama counts costs no extra
poll). Two shapes are recognised:

| `reason=` | Meaning |
| --- | --- |
| `llama_reports_free_slots` | llama-server's last known counts report a free slot while the lease gate refused — the 2026-11-03 08:49:31 shape (`available_slots=2` on a two-slot pool) |
| `occupancy_unexplained` | the refusal cannot be reproduced from any gate input in the snapshot (generating, prefill and unexpired-record counts are all below their caps) |

Nothing is inferred from missing data: an absent llama count is not a
disagreement, and a genuinely full pool that llama agrees with logs no
warning at all. The divergence is a *diagnostic*: it never changes the routing
decision, so a divergence is observable without being a behaviour change.

Log level follows blame: `occupancy_unexplained` (our numbers cannot reproduce
the refusal) is a `WARNING`, while `llama_reports_free_slots` is an `INFO` line
beside the denial — an idle-leased slot counts as free to llama while the
prefill cap legitimately refuses, so treating it as a warning would flood the
WARNING stream (2,633 denial lines were observed in one window).


Prefill (dispatch → first byte) does not scale with concurrency on this
backend. Measured 2026-10-19 → 10-20 over 460 paired local dispatches:

| Overlapping prefills | n | first-byte p50 | new tok/s p50 |
| --- | ---: | ---: | ---: |
| 0 | 190 | **23.4 s** | **253.8** |
| 1 | 220 | 85.0 s | 156.6 |
| 2 | 48 | 119.7 s | 128.8 |

47.8 % of prefills shared the GPU with at least one other and aggregate
throughput stayed flat (~254 tok/s) — i.e. concurrency added latency, not
capacity, and inflated total prefill wall time 2.3×.

**Re-measured 2026-11-11 (LP-0MWCEHDRV000B7HD).** With the gate closed, the
20k–40k production bucket sustained 241.0–243.5 tok/s aggregate (per-request p50
258.0–273.2 tok/s, 12 and 13 requests over two windows) — the recorded ceiling,
reproduced. Forcing two concurrent prefills at the engine (gate bypassed, unique
44,770-token prompts, three interleaved rounds) gave aggregate **156.1 tok/s**
vs **156.2 tok/s** serial: ratio 1.00, wall window 2.00×, slowest request 0.81×.
So the flat aggregate is a property of the GPU, not of serialisation —
concurrency remains a latency expense and not a throughput lever. Record:
`docs/dev/prefill-concurrency-2026-11-11.md`,
`docs/dev/prefill-concurrency-baseline-2026-11-11.json`.

The four prefill-share lever families (session compaction, prefill concurrency,
model/quantisation/slot geometry, accept-the-ceiling) are costed on one unit set
— prefill tok/s added, per-slot resource cost, and *both* local-share
denominators — in
`docs/dev/local-prefill-lever-comparison-2026-11-11.md`
(LP-0MWCEII0V003V76A). Two results matter when reading any share number below:
no family adds aggregate prefill rate (the all-lever bound is 1.019× on the gate
rate), and a share figure is meaningless without its denominator — offered
**prompt tokens** (0.41–1.62 % ceiling) and offered **streams** (1.52–5.96 %
ceiling) differ by the 3.68× prompt-size mix. The local-share target it
recommends for the recovery criterion is **≥ 3.5 % of locally served streams**
(= 0.951 % of offered prompt tokens), 0.587× the point ceiling.

At most `server.local_max_concurrent_prefills` prefills now run at once:

```yaml
server:
  local_max_concurrent_prefills: 1   # 1 = serialise (default); 0 = disabled
```

- **Decode is not affected** — only prefill-vs-prefill is serialised; a slot
  that is already generating does not block another slot's prefill. The
  admission guard honours this literally (LP-0MVXWQQ20002HAX4): its occupancy
  term (`gate_occupancy` on the `local_admission_guard` line) is the in-flight
  **prefill** count, capped by the prefill cap — never the generating
  concurrency. A decoder-only neighbour contributed no gate wait, and its
  throughput effect is already priced by the *contended* rate bucket, so
  charging it again as an occupancy multiplier double-counted the slowdown
  and refused work the GPU could start immediately (measured 2026-11-01, one
  hour: 40 refusals with `in_flight_prefills=0 busy_slots=1`). The symmetric
  floor and the measured-occupant term still apply whenever a prefill IS in
  flight, so the deadline bound is unchanged in that case.
- **Waiting is bounded by the existing machinery** — a second prefill waits
  in the contention queue (policy/depth/wait caps) and falls back to remote
  exactly as before when those caps are hit; the prefill-ETA admission guard
  refuses waits that cannot succeed rather than parking the request.
- **No added delay for a lone request** — with nothing else prefilling the
  gate is open (the in-flight count is 0).
- **Observability** — each local dispatch logs
  `prefill_gate_wait_ms=… saturation=… session=… model=… provider=…` (the time
  the request waited specifically for the prefill gate; `0.0` when it was not
  prefill-gated), and `/llama/local/status` publishes
  `local_prefills_in_flight`, `local_max_concurrent_prefills` and
  `prefill_gate_waiters` (the contention queue depth).

## Residency-aware admission ladder (LP-0MVN1IQTF0037WTG)

Local admission used to be priced **per request** from a *prediction*
(`expected_cached_ratio` / the `slot_cache_ratio` sidecar) with no notion of
which session owned which slot. Under contention that prediction is
structurally wrong: 27 distinct sessions cycled through 2 slots in 5.4 h on
2026-10-24, aggregate prefix reuse was **26.1 %** (median **15.6 %** for prompts
>20k tokens, 28 of 227 fully cold), and the *same* 49,103-token prompt measured
`n_past=738` (1.5 % reusable) on one attempt and 100 % on the immediate retry.

A large prompt (estimated tokens above `local_large_context_cold_cache_threshold`)
now walks a **five-rung ladder**, in this order. The slot candidate is assigned
**before** the ladder runs (*Candidate-first ordering*, below), so the
residency rungs measure a real slot rather than predicting from a
not-yet-existing assignment:

1. **Hard context cap** — `context_too_large` when the prompt exceeds the warm
   threshold. Unchanged, never overridable (the prompt cannot fit the KV slot).
2. **Resident-prefix gate** — admit only when the session's *measured* reusable
   prefix is at least `local_resident_prefix_min_ratio` of the prompt;
   otherwise skip local with `resident_prefix_insufficient` and fall through the
   chain as `large_context_bypass` does today. A prompt above the cold threshold
   with **no** measurement at all (`residency_source=unknown`) is treated as
   *cold* and skipped with `unmeasured_residency_cold` (see *Unmeasured is cold*
   below). `0` disables this rung.
3. **Session tenancy** — at most `local_session_tenants` distinct sessions hold a
   local reservation at once (`tenant_limit` otherwise); same-session re-entry is
   always permitted and a tenancy idle-lapses after
   `local_session_tenancy_idle_seconds` of no activity.
4. **Live-priced cold check** — `new_tokens` is priced from the same live
   measurement (`estimated_tokens - min(context_tokens, estimated_tokens)`), so
   the cold budget is evaluated against real prefill work; a cold miss is
   `large_context_bypass`.
5. **Idle-exempt override** — after the economic rungs, so an idle GPU still
   serves cold work (`local_idle_admit_*`, below). It may override rungs 2 and
   4, never rung 1.
6. **Last-resort override** — the final rung (`local_last_resort_*`, below),
   reached only when no remote provider can serve the request at all.

```yaml
server:
  local_resident_prefix_min_ratio: 0.5   # 0 disables the residency gate
  local_unmeasured_residency_cold_enabled: true  # false restores the predicted-ratio fallback
  local_session_tenants: 2               # default: the slot count; 0 disables
  local_session_tenancy_idle_seconds: 30 # default: the adaptive dispatch lease
  local_large_append_bytes_threshold: 32768  # 0 disables the large-append rule
```

**Large-append pricing (`LP-0MW81DLWY003XOZ2`).** Where the ladder does *not*
measure (rung 5, the predicted-ratio price), the ratio still comes from the
previous local response — it cannot see a single message that appends a large
block of new work. On 2026-11-03 08:41:54 a **51,945-byte** tool-role message
(a whole `SKILL.md` pasted into history) was appended, the price came out as
`cached_ratio=0.99 new_tokens=173`, and llama-server then prefilled 13,002
tokens: 53.2 s to first byte, `measured_cache_abort deficit_ratio=0.659`. When
the content appended after the session's recorded prompt prefix exceeds
`local_large_append_bytes_threshold` **bytes**, the pre-dispatch price forces a
full prefill (`cached_ratio = 0`), so the cold threshold is evaluated against
the real prefill work instead of a ratio that was never about this content.
Only the append counts (a large resident prefix plus a small turn keeps the
measured ratio), and with no recorded prefix — a ratio restored from the
slot-snapshot sidecar — the boundary is unknown and the measured ratio is used
unchanged. `0` disables the rule and restores the previous price exactly; a
missing or malformed value keeps the 32 KiB default. Every forced price logs
`routing_large_append_full_prefill model=… session=… appended_bytes=…
threshold=… estimated_tokens=…`, so a session that is suddenly priced cold is
explainable from the log instead of looking like a measurement change.

**Divergence feedback (`LP-0MW81DS4W008DKDV`).** A prediction is sometimes
*disproved* by the dispatch it admitted: llama-server's measured
`prompt_progress` reports `total` / `cache`, and
`proxy.measured_cache_abort` records `deficit_ratio = (total - cache) / total`.
When that deficit is at or above `local_divergence_feedback_min_deficit_ratio`
(default `0.25`, the same floor the early abort uses), the measured truth is
fed back into the session's pricing model rather than left for the next turn
to re-inherit:

- the session's measured ratio becomes the measured `cache / total`;
- the next turn's prefix credit is capped at the `cache` tokens the dispatch
  proved resident, so `expected_cached_ratio` prices the measured work — a
  55,492-token prompt with 200 resident tokens is priced at ≈55k new tokens,
  not the `new_tokens=172` the model had predicted, and so falls under the
  38,000 cold threshold (and under the unmeasured-cold rule);
- the correction is **bounded**: at most
  `local_divergence_feedback_max_turns` recorded dispatches (default `1`) and
  at most `local_divergence_feedback_max_age_seconds` (default `900`), so a
  session can never be poisoned permanently — the following dispatch rewrites
  the basis and a genuinely warm turn restores the credit;
- it is **cleared** with the state it derives from: session invalidation, slot
  reassignment (`session._free_slot_assignment`) and model switch
  (`lifecycle._forget_switched_model_measurements`).

Two log lines make it attributable: `routing_divergence_feedback model=…
session=… deficit_ratio=… measured_reusable_tokens=… measured_new_tokens=…
priced_new_tokens=… turns=…` when the correction is recorded, and
`routing_divergence_feedback_applied model=… session=… recorded_tokens=…
corrected_tokens=… deficit_ratio=…` when a corrected dispatch records its own
prefix. `local_divergence_feedback_enabled: false` restores the previous
pricing byte-for-byte; a missing or malformed numeric knob keeps its shipped
default.

### What to read, what never to call

Admission reads the **cached** slot snapshot only
(`observability._last_slot_details_cache` + its `_last_slot_details_at` stamp,
refreshed about once a second by `_periodic_broadcast_loop()`) joined with the
in-process ownership registry (`session._slot_owners`) through
`proxy.residency.get_residency_view()`.

**Never issue a synchronous `/slots` call on the admission hot path.** Under a
prompt-processing batch llama-server holds `/slots` for more than 15 s (measured
8.7 s, then >15 s, against a 1 s status timeout — `LP-0MUXTW4MD003TVIC`), so a
synchronous call would stall every routing decision. `proxy.residency` is
dependency-free for exactly this reason: no poller, no new task, no upstream
call.

### Candidate-first ordering (`LP-0MW959P9S005DJY7`, `LP-0MW959PWL001WYSM`)

The ladder used to be probed **before** any slot assignment, so on a pool whose
slots are handed back at the end of every dispatch the session owned nothing at
decision time: `measurement=unavailable:no_slot` was emitted on 3 363 of
~4 500 local checks (~75 %), and the decision fell back to the predicted sidecar
ratio — which priced a 55 492-token prompt at `new_tokens=172` and admitted it
warm. The ordering is now:

1. **Assign** a slot candidate (`session.acquire_slot_candidate`) — before the
   verdict, so even the outermost `context_too_large` rung has one. Acquisition
   registers ownership in `session._slot_owners`, the registry every residency
   rung already reads;
2. **Measure** the candidate (`resolve_resident_prefix`), so the measurement is
   of *this* request's slot rather than of a prior turn's;
3. **Price** `new_tokens` from that measurement
   (`estimated_tokens - min(n_past_actual, estimated_tokens)`, the live-prefix
   price) and take the verdict;
4. **Release** the candidate on every refusal rung and on any exception
   (`session.release_slot_candidate` → the same registry release the normal
   end-of-turn path uses), so a refused admission leaves no fabricated
   occupancy — no lease, no dispatch record, no session-slot record. The
   release emits `candidate_released reason=<skip_reason> session=<id>
   slot=<n> endpoint=<url>`, and is idempotent. An **admission** keeps the
   candidate: ownership passes unchanged to the dispatch bookkeeping
   (`_build_slot_context` re-enters the same assignment), so there is no second
   assignment.

**`measurement=unavailable:no_slot` now means the pool is exhausted.** It is no
longer "no slot has been assigned yet" — a candidate is assigned before the
verdict, so the only way a session has no slot at decision time is that none
could be given (the pool is full), or that no pool/session id is configured at
all. A candidate with no prior dispatched measurement reports
`unavailable:no_prior_measurement` on the `session_slot` rung instead; one with
a measurement reports `measured` on `measured_last_dispatch`.

Concurrency, cleanup and the latency this added are pinned in the design doc
(`docs/dev/slot-management-design.md` §7–§10); the field each decision reports
for that latency is `admission_overhead_ms=` (below).

**Unmeasured is cold (`LP-0MW81DFML006O5KL`).** A missing or stale snapshot
(> 5 s), a session that owns no slot, or a build that no longer reports
`context_tokens` (llama.cpp b8782 dropped `n_past`) degrades one rung at a
time: measured ratio → session owns the slot (restore path) → **unmeasured**. An
unmeasured basis is treated as **cold**: the prompt is held to
`local_large_context_cold_cache_threshold` (38 000) and, above it, skipped with
`unmeasured_residency_cold` into the normal remote chain. It no longer
fail-opens to the predicted sidecar ratio — that relaxation is exactly what
admitted the 2026-11-07 20:48:51 55 492-token prompt on
`cached_ratio=1.00 new_tokens=172` and evicted it at 180.1 s. Only a
*measured-resident* basis (`measured` / `measured_last_dispatch` /
`session_slot`) keeps the 120 000 warm threshold. The denial is deliberately
**not** idle-overridable (`IDLE_OVERRIDABLE_SKIP_REASONS`): it reports a missing
measurement, not a price an idle GPU can beat — `local_last_resort_*` stays the
final local override when no remote can serve the request. Absent data never
becomes a hard local outage for prompts *below* the cold threshold: those route
local exactly as before.

**Rollback (one value, byte-for-byte).**
`server.local_unmeasured_residency_cold_enabled: false` — `false` / `0` / `off`
are all accepted, and a missing or malformed value keeps the `true` default, so
a typo can never silently disable the enforcement — restores the pre-change
fail-open behaviour exactly: an unmeasured basis degrades to the predicted
sidecar ratio and logs `decision=admit_predicted` / `decision=skip_predicted`
with `n_past_actual=none`. `local_resident_prefix_min_ratio: 0` remains the
broader switch that disables the whole residency ladder (and therefore this rule
with it).

### Tenancy is a lease, not a lock

Reservations follow the existing dispatch-record lease: held only while real
activity is observed, reclaimed by the existing ~10 s `_dispatch_cleanup_loop`
(observed client disconnect → released immediately) or by the 180 s no-progress
watchdog for a silent half-open drop, never only by the 900 s lifetime ceiling.
On reclaim the evicted session's KV snapshot is saved (`slot_<session>.bin`,
subject to the existing pool-size / busy / circuit-breaker gates) so a returning
client restores warm instead of prefilling cold, and the shared
`local_eviction` ERROR event records the reason (`lease_evicted`,
`slot_assignment_freed`, `resident_prefix_evicted_by_tenant`,
`tenancy_lease_expired`; sink: `proxy/admission_events.py`).

### Observability

Every decision logs
`routing_resident_prefix provider=… model=… decision=<admit|resident_prefix_insufficient|tenant_limit|large_context_bypass|unmeasured_residency_cold> n_past_actual=… estimated_tokens=… reuse_ratio=… slot_owner_session=… residency_source=<measured|measured_last_dispatch|session_slot|non_resident|unknown> measurement=<measured|unavailable:no_slot|unavailable:no_prior_measurement|unavailable:stale_view|unavailable:capture_disabled> min_ratio=… tenants=… occupants=… session=… threshold_applied=<cold|warm|none> threshold_value=<n|none> admission_overhead_ms=<ms|none> slot_candidate=<n|none> new_tokens=<n|none>`.

`slot_candidate=` (LP-0MW959RRO008AAH9 AC10a) is the slot
candidate-first admission assigned before this verdict, i.e. the slot the
decision was measured against. It is the association key the dispatch record
and the eviction event use to attribute the request to a llama slot/task (see
[request ↔ slot/task attribution](#request--llama-slottask-attribution-lp-0mw959rro008aah9))
and is `none` when no candidate was assigned (pool unconfigured or exhausted),
never a fabricated slot. Every pre-existing key keeps its position, so a
positional consumer of the line is unaffected.

**The priced `new_tokens=` (`LP-0MW959SE30012HJJ`, parent AC6).** The
`slot_candidate=` / `n_past_actual=` / `residency_source=` / `measurement=`
fields say what the verdict was measured against; `new_tokens=` is the number it
was **held to** — the priced uncached-token count the cold cap compared. It is
appended **last** (after `slot_candidate=`), so every pre-existing key of the
line keeps its position and a positional consumer of a pre-change line is
unaffected.

| Rung that priced the prompt | `new_tokens=` |
|---|---|
| measured-resident (live `context_tokens`, or the session's last measured dispatch) | `estimated_tokens - min(n_past_actual, estimated_tokens)` — the live-prefix price |
| predicted-ratio rollout (`admit_predicted` / `skip_predicted`, `local_unmeasured_residency_cold_enabled: false`) | `int(estimated_tokens × (1 − cached_ratio))` — the sidecar price |
| hard cap (`context_too_large`), unmeasured-cold (`unmeasured_residency_cold`), tenancy (`tenant_limit`) | `none` — no rung priced the prompt; never a fabricated `0` |

A **refusal** line carries the price wherever the refusing rung had one: a
`resident_prefix_insufficient` denial reports the measured price the prompt was
refused *instead of*, so the refusal is attributable to the basis (the ratio)
rather than looking like a missing price. Both verdicts are emitted by one
logger with one grammar, so a refusal can never quietly drop a field.

A wrong price is therefore traceable from one line, not by archaeology across
the dispatch, eviction and `prompt_progress` lines: `slot_candidate=0
n_past_actual=30000 estimated_tokens=49103 new_tokens=19103
residency_source=measured_last_dispatch measurement=measured decision=admit`
reconstructs the arithmetic that admitted the request.

`measurement=` is the additive provenance field (immediately after
`residency_source=`, before `min_ratio=`) that distinguishes a real measurement
from the ownership sentinel. It carries **exactly one** value per line, drawn
from a closed vocabulary (`provider.MEASUREMENT_VOCABULARY`), so an unmeasured
decision is self-describing instead of a bare `reuse_ratio=none` — "the ladder
cannot measure" is distinguishable from "there is no slot to measure"
(LP-0MVT2Q1SD009JLKD):

| `measurement=` | Emitted when |
|---|---|
| `measured` | the ratio came from a real measurement (`residency_source=measured` live `context_tokens`, or `measured_last_dispatch`) |
| `unavailable:capture_disabled` | the session owns a slot but `local_measured_cache_abort_enabled` is off (the documented rollback), or the switch could not be read (fail-open ⇒ capture not enabled) |
| `unavailable:no_prior_measurement` | the session owns its slot and capture is enabled, but no local dispatch has been measured yet for `(model, session)` |
| `unavailable:no_slot` | no slot candidate could be assigned — **the pool is exhausted** (or no pool/session id is configured), so the decision was taken with no slot to measure. Since candidate-first ordering assigns before the verdict, this no longer means "not yet assigned" |
| `unavailable:stale_view` | the residency view is absent/stale (`residency_source=unknown`), or the decision was logged without resolving residency facts (gate disabled: `local_resident_prefix_min_ratio: 0`) |

Precedence is deterministic and first-match: `measured` →
`capture_disabled` → `no_prior_measurement` → `no_slot` → `stale_view`
(`provider.measurement_reason`). A measured source is never re-labelled by the
rollback switch, and `no_slot` requires a fresh view — an absent/stale view is
reported as `stale_view`, never as a measured zero.

**Backward compatible.** `reuse_ratio=none` remains the numeric-absent form for
every non-measurement (the internal `0.0`/`1.0` gate sentinels are never
surfaced — `LP-0MVNPZY22008B536`), `residency_source` values are unchanged, and
a pre-change line (no `measurement=` field) parses unchanged.

**Which threshold decided it (`LP-0MW81DHPH00650NY`).** The basis says
*whether* residency was measured; `threshold_applied=` says *which token
threshold* the verdict was held to, and `threshold_value=` is that threshold's
configured value (`local_large_context_cold_cache_threshold` cells default
38 000, `local_large_context_warm_cache_threshold` default 120 000). Both are
**appended after `session=`**, so every pre-existing key keeps its relative
order and the `proxy-usage-analysis` parser reads a new line unchanged (the
values are a closed vocabulary, `provider.THRESHOLD_APPLIED_VOCABULARY`):

| `threshold_applied=` | `threshold_value=` | Emitted when |
|---|---|---|
| `cold` | the cold cap (38 000) | no measured residency backs the prompt — the unmeasured-cold denial (`unmeasured_residency_cold`, `residency_source=unknown`) and the measured-zero denial (`resident_prefix_insufficient` with `residency_source=non_resident`) both fall here, because the warm cap has no measurement to stand on |
| `warm` | the warm cap (120 000) | the decision is measured-resident (`measured` / `measured_last_dispatch` / `session_slot`), so the warm cap governs and the prompt may be served locally above the cold threshold |
| `none` | `none` | the decision has no residency basis at all (the ladder is disabled: `local_resident_prefix_min_ratio: 0`), so no token threshold is attributed |

So a local admission of a >38 000-token prompt is attributable from the line
alone: `threshold_applied=cold` with a small `n_past_actual` is a cold-priced
admission, while `threshold_applied=warm` is a measured-resident prompt that
would otherwise have been refused by the cold cap.

**How much latency the admission added (`LP-0MW959R5S009WDM8`).**
Candidate-first admission put two steps on the request hot path — the slot
candidate acquisition and the residency resolution priced against it — so the
line carries their measured cost as `admission_overhead_ms=`, **appended last**
(after `threshold_value=`), in milliseconds with three decimals. It covers only
the added work: the clock starts at the candidate acquisition (in the router's
scope) and stops the moment the residency resolution returns, so it never
includes the rest of the ladder or the upstream request. `none` means *not
measured* (the ladder is disabled: `local_resident_prefix_min_ratio: 0`) and is
never reported as `0.000`. The budget is **p95 ≤ 50 ms**, enforced by replaying
the checked-in 2026-10-24 thrash fixture through the real admission path
(`proxy/tests/prefill_thrash_replay_harness.py`,
`measure_admission_overhead`), which aggregates this same field — the replay
and the operator therefore read one measurement, not two implementations.

**Measurement sources** (rung order): `measured` is the live `/slots`
`context_tokens`; when the deployed build does not report it (llama.cpp b8782
dropped `n_past`) `measured_last_dispatch` uses the session's most recent local
dispatch from the in-process `proxy.measured_prefill` store — llama-server's own
`prompt_progress` `total` for the `(model, session)` pair — before falling back
to the ownership-only `session_slot` sentinel. That `total` is the previous
prompt's size *including* the tokens that dispatch decoded, so the ratio derived
from it is an upper bound on the prefix this request will reuse: an honest
approximation of warmth, never a promise that the whole prefix is still
resident. The rung is guarded by live
ownership and the capture rollback switch, and prices admission from
`min(n_past_actual, estimated_tokens)` reusable tokens, so a *grown* prompt can
be denied with `resident_prefix_insufficient` (the intended AC4 rule (a)
signal, not a regression). The stored measurement is invalidated where the
session loses its slot — eviction, slot release/reassignment and a model
switch (`LP-0MVT2QFAP0082BKR`) — so the next decision for that pair reports
`measurement=unavailable:no_prior_measurement` rather than a stale ratio;
live ownership remains the backstop.
`proxy-usage-analysis` renders the resident-vs-evicted reuse split and
per-session slot-owner churn in its `## Residency-aware admission` section.

Replay evidence: `proxy/scripts/replay_prefill_occupancy.py --residency-replay`
(and `--window-fixture`) plus
`proxy/tests/test_prefill_thrash_replay_2026_10_24.py` replay the pinned
2026-10-24 window through this ladder: zero over-budget cold admissions without
a resident prefix, and tasks 92383/92422 classified cold-then-resident.

## Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J)

Routing is size/economics-driven and used to ignore whether the backend was
doing anything, so a large or cache-cold prompt was bypassed to a paid remote
provider while both slots sat unused (measured: **187**
`local_dispatch_denied` in an hour where local busy was **2.5 %**, and
`large_context_bypass` 1,284/24 h). The economic cold-cache bypass is now
overridable when the GPU has demonstrably been idle:

```yaml
server:
  local_idle_admit_enabled: false   # default off
  local_idle_admit_seconds: 60      # idle age required before overriding
```

**Scope:** only the economic band (`large_context_bypass`) is overridable. The
physical `context_too_large` cap is a capacity limit — a prompt that cannot fit
the KV slot never goes local because of idleness.

**Idle signal** (`proxy/proxy/local_activity.py`, exposed on
`/llama/local/status` as `local_idle_seconds`, `local_last_activity_at`,
`local_last_activity_source`): a monotonic timestamp updated when a local
dispatch starts, on every forwarded SSE chunk, and when the request finishes.
The baseline is process start, so a freshly restarted idle proxy reports its
true idle age rather than "unknown".

**Fail-safe blockers** — any of these keeps today's bypass (fail-closed):

| Condition | Blocker |
| --- | --- |
| Idle age below `local_idle_admit_seconds` (or the knob is missing/malformed/non-positive) | no override |
| Any waiter in the contention queue | `queue_not_empty` |
| A local lease, generation, or prefill is in flight | `local_active` / `generating` / `prefill_in_flight` |
| Backend self-healing | `self_healing` |
| Model switch or background load in progress | `model_switch` / `model_loading` |
| Default local backend not ready / no process (explicit `endpoint` entries are externally managed and skip this gate) | `backend_not_ready` / `backend_not_running` |

**Observability:** each override logs
`routing_idle_admit provider=… model=… reason=economic_bypass_idle
idle_seconds=… threshold_seconds=… estimated_tokens=… new_tokens=…
cached_ratio=… cache_state=… session=…` and records a `local_admit`-class
dispatch attempt (`status=idle_admit`). `proxy-usage-analysis` parses the line
into an `idle_admit` event and reports it as the
`Local admitted while GPU idle (cold-cache override)` row and the
`local_idle_admit` JSON counter.

Idle-admitted requests are expected to be **slow** (prefill p90 269.8 s, max
817 s in the baseline window) — that is the deliberate trade of idle GPU time
for remote spend, and it is why the override never fires while anything is
queued or in flight.

## Last-resort local admission (LP-0MVWAMY8M001KE4W)

The idle-admit rung above still prices the dispatch with the flat
`local_cold_prefill_budget_tokens` cap (25,000), so it declines most of the
work an idle GPU could take. Measured over the 24 h window
2026-10-29 23:43:30 → 2026-10-30 23:43:30:

- the idle ladder fired **1,337** times and refused **1,291 (96.6 %)** on
  `routing_idle_admit_refused reason=prefill_budget_exceeded` — `new_tokens`
  p50 **79,136** (all inside the 120 k warm cap), `idle_seconds` p50 197 s;
- **236** of the window's 686 `All providers exhausted` warnings landed inside
  idle gaps that carried **98 inbound requests and zero streams**;
- in those gaps the remote chain was empty every time
  (`usage_limit_reset_pending` ~23 h, `quota_exhausted` ~18 days,
  `outside available_times`).

The final rung of the admission ladder closes that gap: when there is
**provably nowhere else to send the request**, an idle GPU served slowly beats
a retryable 503.

```yaml
server:
  local_last_resort_idle_seconds: 300     # 0 disables the rung entirely
  local_last_resort_deadline_seconds: 600 # size-appropriate prefill deadline
```

The rung sits **after every existing gate** (an ordinary request never reaches
it) and admits only when all of the following hold:

1. the rung is enabled (`local_last_resort_idle_seconds` > 0);
2. the estimated prompt fits the model's own routing warm cap
   (`local_large_context_warm_cache_threshold` / the per-slot clamp, tightened
   by a per-entry `max_context_tokens`) — a prompt that cannot fit a slot is
   never admitted by this rung;
3. the remote chain is **measured empty at decision time** via
   `_remote_chain_has_capacity` — the resolver's own eligibility vocabulary
   (recorded chain-resolution skips, cooldown, pending usage-limit reset, quota
   hard stop, `available_times`), not "we have not tried yet";
4. nothing local is queued, generating, prefilling, loading or switching
   (`_local_idle_blocker`), which also bounds last-resort concurrency at
   **one** dispatch — a second request while a last-resort dispatch runs is
   refused exactly as today;
5. the local model has been idle for at least the configured threshold.

**Deadline.** A prompt admitted on an idle GPU needs more time than the normal
`local_prefill_deadline_seconds` (a 79 k cold prompt needs ~410 s at the
measured ~193 tok/s), so a last-resort dispatch uses
`local_last_resort_deadline_seconds` instead — clamped to 900 s (the
`local_dispatch_max_prefill_seconds` lifetime wedge bound), falling back to
600 s when malformed, and still bounded by the tighter of itself and
`local_dispatch_first_byte_timeout_seconds`. Every other dispatch keeps the
normal deadline; without this the rung would only convert the 503 into a
`prefill_deadline` eviction.

**Fail-safe.** Any error resolving remote availability or reading the idle
clock denies (today's behaviour) and never fails a request. Over-admitting
locally on an unreadable chain state is the worse error, so the check defaults
to "a remote may exist".

**Kill switch.** `local_last_resort_idle_seconds: 0` (or a
malformed/non-positive value) restores today's decision matrix
byte-for-byte. Default is **ON**.

**Observability.** Each admission logs

```
local_last_resort_admit provider=… model=… reason=remote_exhausted_idle_gpu
  estimated_tokens=… new_tokens=… idle_seconds=… threshold_seconds=…
  deadline_s=… warm_cap=… remote_exhausted=true session=… turn=…
```

and records a `status=last_resort_admit` attempt so the fallback diagnostics
carry it. The post-deploy (≥ 24 h) attribution reads `All providers exhausted`
inside idle gaps together with `local_eviction_after_acceptance`
(`reason=prefill_deadline`) and discarded prefill tokens: the rung is only a
success if the 503s fall and neither eviction metric rises.

## Slot Contention: reservations first, the queue as a straggler net

### The primary cooperative path is a reservation

A dispatcher that wants local capacity should **claim it before sending**
rather than wait for it afterwards. `POST /llama/local/reserve` returns the
same occupancy-aware verdict the queue's admission guard enforces and, on
`admit`, a single-use ticket the request presents as
`X-Local-Reservation: <id>` — a ticketed request dispatches **without
re-checking and without queueing**. `X-Local-Preference`
(`local-preferred` / `remote-ok`, the default) states what the client wants
when local capacity is unavailable; `local-preferred` defers with a retryable
`429` instead of buying a remote tier, while the default `remote-ok` steps
straight to the next provider. Reservations are therefore the
**primary cooperative path**; the protocol is specified in
[Local Admission Reservations and Fallback Intent](#local-admission-reservations-and-fallback-intent-lp-0mvikpcsg002b0nl).

### What the queue is for: stragglers, not parking

When every local slot is busy (``local_active_queries >=
session_slot_pool_size``), a request that did **not** reserve can still
**queue** for a slot instead of falling back remote immediately. Since
LP-0MVT1146V007WXA4 that queue is a bounded **straggler safety net, not a
parking lot**: it covers non-cooperating clients and the seconds-scale gap
before a slot frees, priced at a shipped **30 s** wait budget and a **2**-deep
cap (`proxy/config.yaml`, `server` section). A reservation ticket skips it
entirely, so the queue only ever holds stragglers.

Queue entry is gated by the request's effective priority (LP-0MVDT94F0003VHUA
AC3): ``low`` always queues (prefer a local slot over a paid remote call),
``high`` never queues (immediate remote fallback), ``medium`` follows the
configured policy. The caps bound the queue for every priority:

| Config key | Meaning | Clamped to |
|---|---|---|
| `contention_queue_policy` | ``queue`` or ``fallback`` (applies to ``medium``) | — |
| `contention_queue_max_depth` | Max queued requests | `[1, 16]` |
| `contention_queue_max_wait_seconds` | Max seconds to wait for a local slot | `[1, session_guardrail_max_runtime_seconds]` |

**Shipped values (unified config, slot pool fixed at 2):**

| Priority | Policy | Depth | Max wait | Rationale |
|---|---|---|---|---|
| **low** | ``queue`` (always) | `2` | `30s` | Straggler net — recover the seconds-scale gaps, never park a request |
| **medium** | ``queue`` (config-driven) | `2` | `30s` | Shipped default; a wait is only worth taking when it can win inside 30 s |
| **high** | never queues | — | — | Immediate remote fallback under local contention |

History: the per-mode overlays declared their own queue caps (a deeper depth
and a longer wait, unified in LP-0MV881O5C007PSL4 and LP-0MVIQ0EXV00673HO);
the mode axis is retired, so the policy and both caps are now single
server-level knobs. The depth and wait values were swept to the shipped pair in
LP-0MVT1146V007WXA4 — see *Evidence, flip criteria and rollback* below
and the measurement runbook [contention-queue-sweep.md](../../docs/dev/contention-queue-sweep.md).

Key semantics (see `proxy/proxy/provider.py` `_maybe_queue_for_local_slot` and
`proxy/proxy/contention_queue.py`):

- **Reservations never queue**: a request carrying a claimed
  `X-Local-Reservation` ticket dispatches without re-checking and without
  queueing, which is exactly why the queue holds only stragglers.
- **Cross-session**: the queue is process-global, so a request from session B
  can wait (bounded) behind a long audit stream from session A. This is what
  converts overnight contention fallbacks into queued-local dispatches.
- **Context bypasses**: requests that cannot fit the KV slot
  (`context_too_large`, LP-0MSF8XDG7000PERM) always fall back — this is a
  physical capacity limit, not contention.  The economic cold-cache bypass
  (`large_context_bypass`, LP-0MRE4NBQ5009V5BX) is mode-dependent (LP-0MU5A4QBR003YJM0):
  in **cheap mode** it is recovered (the request proceeds to local and may
  queue); in **fast mode** it bypasses local and falls back to remote providers
  for tail latency.
- **ETA-aware admission ordering** (LP-0MVIQ0G2M0006QF6): with
  `local_max_concurrent_prefills: 1` one long prefill is head-of-line blocking
  for every queued local request (1 827 prefill-saturated queue samples vs 468
  generating in the 24 h baseline). Within each retention class the queue now
  inserts a waiter by **ascending projected prefill ETA**, so a queue of 20 s
  turns is not parked behind one 7-minute prefill. Retention priority is still
  the primary tie-break (ordering only applies *within* a class), a missing
  projection orders last, and a waiter older than
  `contention_queue_eta_promotion_seconds` (default 30) is promoted to the
  front of its class so a long prefill cannot starve. Each reorder logs
  `contention_queue_eta_order ... reason=shortest_eta_first`;
  `contention_queue_eta_ordering: false` restores plain FIFO.
- **Wake signals**: the queue wakes on BOTH `local_active_queries` decrement
  (a local stream ended) AND slot-persistence / lease release (slot
  save/restore frees the backend during model switches).
- **Timeout accounting (Q2=a)**: the queued wait subtracts from the
  client-visible adaptive timeout budget — total (wait + serve) stays within
  `llama_adaptive_timeout_*` (base 60s + 0.015/token, capped at
  `max_runtime_seconds`), so interactive clients never see queue wait + serve
  exceed the adaptive envelope.
- **One queue budget per turn** (LP-0MVV29Y1M004GAW0): a request can reach
  the queue twice in one turn — the `local_concurrency_limit` gate and then
  the `local_lease_active` retry after the local dispatch was denied by the
  holder's lease. The retry passes the request's **priority** (so `high`
  never queues there, matching the documented contract) and the queue wait the
  turn already spent, so the second wait is bounded by the **remainder** of
  `contention_queue_max_wait_seconds` rather than a fresh window. Cumulative
  queue wait for a turn therefore never exceeds the cap, and
  `queued_duration` on the lease path reports the whole turn's wait. The
  enqueue line carries `spent_wait=` and `budget=` alongside the configured
  `max_wait=`.
- **Metrics**: contention-queue observability lives in the per-request log
  lines only — there are no cumulative counters and no dedicated Prometheus
  counters (LP-0MU4NACW9001WKBP: the cumulative queued-count / queued-duration
  / fallback-after-queue counters were removed as unused and confusing). The
  `contention_queue_dispatch` line carries `queued_duration`, `policy`, and
  `depth`; the `contention_queue_fallback_after_queue` line carries
  `queued_duration` (the elapsed wait, F4 AC2). The 24h report pipeline
  (`.pi/skills/proxy-usage-analysis`) parses and aggregates these lines into
  `contention_dispatch` / `contention_fallback_after_queue` counts plus
  total queued duration (`contention_queued_duration_seconds`).

Config reference:

```yaml
server:
  contention_queue_policy: queue
  contention_queue_max_wait_seconds: 30    # clamped to [1, max_runtime_seconds]
  contention_queue_max_depth: 2            # clamped to [1, 16]
```

The queue engages only when ``contention_queue_policy`` is ``queue``. Absent
keys default to ``fallback`` for backward compatibility, so nothing queues.

### Evidence, flip criteria and rollback for the 30 s / depth 2 sweep (LP-0MVIKPRMD000Y97R)

The shipped caps were swept in **LP-0MVT1146V007WXA4** and are pinned by this
document, `proxy/config.yaml` and
`proxy/tests/test_contention_queue_docs.py`. Two measured windows justify them:

| Window | Scope | Measured |
|---|---|---|
| **2026-10-22 19:00 → 22:18** | 3.3 h intake window | 180 queue events: 38 wins (21 %) / 142 losses (79 %), **every loss at the full wait budget** — 8 705 s of dead wait. Wins retained by budget: 10 s → 16 %, 20 s → 24 %, 30 s → 34 %, 45 s → 60 % |
| **2026-10-30 05:00 → 2026-10-31 05:00** | 24 h pre-flip baseline (AC1 trigger precondition) | 351 queue events: 214 wins (61 %) / 137 losses (39 %), loss wait 8 222 s (342.6 s/h); 124 wins (57.9 %) landed at or below 30 s; 73 of the 137 losses were `reason=local_lease_active` |

The baseline is produced by `scripts/contention_queue_sweep_report.py` and
archived in
[contention-queue-sweep-baseline-2026-10-31.json](../../docs/dev/contention-queue-sweep-baseline-2026-10-31.json);
the **same** definition must produce the post-flip verdict (runbook:
[contention-queue-sweep.md](../../docs/dev/contention-queue-sweep.md)), so the
two numbers are never computed two ways.

**Flip criteria** — the sweep is kept only if ALL three hold over the 7-day /
≥ 1 000-queue-event observation window, against that 24 h baseline:

1. aggregate dead wait per hour falls by **>= 50 %** (baseline 342.6 s/h);
2. residual queue wins stay **>= 20 %** of queue events (baseline 61 %);
3. remote dispatch volume rises by **< 20 %** relative (no paid flood).

Anything else is the rollback — **two lines and a proxy restart** (the queue
caps are read at startup, so a config edit alone changes nothing):

```yaml
server:
  contention_queue_max_wait_seconds: 60
  contention_queue_max_depth: 8
```

`contention_queue_policy` and `contention_queue_lease_grace_seconds` (300) are
unchanged by the sweep, and the lease grace deliberately stays above the wait
budget so a returning session still re-queues as retained.

## Prefill-ETA admission guard (LP-0MVGDUV5E005UWBQ, LP-0MVIQ0F8T002FS42)

Queuing only helps when the local prefill can actually finish inside the wait
budget. Evidence for 2026-10-19 07:23 → 2026-10-20 07:23: of 5,805 queued waits
only 855 were dispatched local — **4,947 (85 %) were abandoned after up to
the full budget and then went remote anyway**, while local prefill times during the same
window were p50 61.8 s / p90 269.8 s / max 817 s. Waiting the full budget to maybe avoid
a remote call whose median TTFT is 27 ms is a losing trade unless the prompt
is genuinely cache-warm.

Before a request enters the contention queue **and on the direct (free-slot)
dispatch path immediately before the local request is sent upstream**, the
proxy costs the projected prefill and compares the **occupancy-aware projected
wait** with the caller's wait budget:

```
eta_prefill = coded_tokens * coded_ms_per_token + new_tokens * prefill_ms_per_token
  coded_tokens = est_tokens * cached_ratio          # served from the prompt cache
  new_tokens   = est_tokens * (1 - cached_ratio)    # must actually be prefilled

projected_wait = own_prefill_seconds + occupied_seconds
  occupied_seconds = max(eta_prefill * gate_occupancy, sum(remaining occupant hold))
    # remaining occupant hold is per record
    #   max(remaining occupant prefill service, remaining dispatch-lease hold)
    # — the service is re-priced from the occupant's remaining uncached new
    # tokens at the contended co-residency rate (see below)
  busy_slots       = min(slot_pool, max(generating_concurrency, prefills_in_flight))
  gate_occupancy   = min(busy_slots, prefills_in_flight)   # in-flight PREFILLS only

bound  = wait_budget                                  # 0 when the caller never queues
       if free_gate_deadline_budget and gate_occupancy == 0 and safety_budget > 0:
           max(wait_budget, safety_budget)   # free gate ⇒ no queue, bound = deadline
within = True if wait_budget == 0 else projected_wait <= bound   # no queue ⇒ no queue comparison
admit  if within
       and projected_wait <  local_prefill_deadline_seconds
       and own_prefill   <= safety_budget            # both bound EVERY priority, `high` included
else   refuse local admission immediately -> next provider in the chain
```

**The occupancy term is the PREFILL gate, not the generating concurrency**
(LP-0MVXWQQ20002HAX4). Only prefill-vs-prefill is serialised
(`local_max_concurrent_prefills`); decoding runs in parallel, so a
*decoder-only* neighbour leaves the gate free for this request's prefill. Its
throughput effect is already priced by the contended rate bucket, and charging
it again as an occupancy multiplier double-counted the slowdown. The
`local_admission_guard` line reports both numbers (`busy_slots` = observed
concurrency, `gate_occupancy` = what the wait was costed against).

**A free gate may be bounded by the deadline instead of the queue budget —
opt-in** (LP-0MVXWQQ20002HAX4). `contention_queue_max_wait_seconds` is
calibrated for *queue* losses; a request that starts its prefill immediately
never queues, so its only wait is its own prefill. Measured 2026-11-01
13:12-14:00 (1 788 requests, 58 sessions, local share 2.6 %): 20 of the 76
refusals with a free prefill gate had a measured own ETA of 36-102 s, inside
the safety budget then shipped (114 s), and were refused **only** against the
30 s queue budget — a paid remote call for a prefill the GPU could have
started at once.
Counter-evidence: replaying the 2026-10-23 thrash window with the rung enabled
admits them, but 6 of the 20 are then cancelled at the deadline
(`admitted_and_doomed` 0 → 6) — the very invariant the flat cold-prefill
budget exists to protect. The rung therefore ships **off**
(`local_admission_guard_free_gate_deadline_budget: false`) unless the operator
accepts the trade. **Status 2026-11-01:** enabled by operator decision
(LP-0MVXZWF7A002TUO1 — the post-fix 13:17:50-15:34:42 UTC window measured
1.56 % local share with free-gate refusals as a residual non-quarantine
cause; the trade was accepted for a live re-measurement with the rung on,
recorded in a comment on that item). **Status 2026-11-02 (kept enabled, see
"Measured attribution" below):** after the re-measurement the operator
decided to keep the rung enabled; the measured attribution of its first 24 h
window is LP-0MVZ5WV6W008UO5F.  Flip the config value back to `false` to
restore the queue-bound behaviour. Contended admissions keep the queue bound in
either case, and the deadline + safety bounds still refuse anything that could
not finish. The log line carries `budget_s` (the resolved queue budget) and
`bound_s` (the bound actually enforced), so the two can never be confused.

### The ETA refusal is only taken when a remote can serve (LP-0MW0KZR7A003UFQK)

The guard's refusal is a **capacity** judgement about the local engine, so it
only justifies moving the request remote when a remote provider can actually
take it. On 2026-11-03 08:49:40 the guard refused a `remote-ok` request on an
enforced ETA bound (`eta_prefill_s=45.2 enforced=true`) and routed it to a
chain whose every entry was deterministically skipped; the request ended
`outcome=exhausted`. At 08:50:41 the identical prompt with the identical ETA
was admitted (`enforced=false`, `eta_occupancy_s=0.0`) and produced a first
byte in 6 s.

`_remote_chain_has_capacity` therefore gates the direct (free-slot) rejection
with the resolution predicate's own vocabulary, read-only and in the same
order: not deterministically skipped for this request, not in cooldown, no
pending usage-limit reset, not quota-hard-stopped, inside its
`available_times` window. When no remote can serve, the request is **admitted
locally** and waits for its prefill — the same outcome the client got once the
occupancy cleared — logged as:

```
local_admission_guard_local_wait provider=… model=… session=… preference=…
  reason=eta_exceeds_budget_direct free_slot=true → remote chain has no
  capacity; admitting locally and waiting for the prefill
```

and recorded as a `status=local_admission_override` attempt (so the
diagnostics show why local was used despite the ETA bound). The
`local_admission_guard_direct_fallback` line keeps its meaning — a real
fall-through — and is never emitted on the admit-locally path.

Deliberately narrow, because "admit locally" must not become a way to blow
through the request's own bounds:

| Case | Behaviour |
|---|---|
| `reason=eta_exceeds_budget_direct`, hybrid chain, no remote capacity | **admit locally** and wait for the prefill |
| `reason=eta_exceeds_budget_direct`, hybrid chain, a remote has capacity | unchanged — refuse and route remote |
| request-property reasons (`prefill_budget_exceeded`, `context_too_large`, deadline / safety budget) | unchanged refusal — the prompt cannot be served locally at all |
| local-only chain (no remote provider) | unchanged refusal — there is nothing to bounce to, and the ETA bound is the honest answer to a caller whose budget cannot cover the prefill |
| pin-local / `local-only` intent | unchanged (the pin branch runs first, including bounded deferral escalation) |

### Local-route ConnectError fail-fast + pre-dispatch skip (LP-0MW42T6IJ006EQA3)

The 2026-11-05 21:36 incident: a dead llama-server (killed ≈15:13, port 8080
not listening) was still advertised available, so every request dispatched
local and burned **~28 s** — `backend_retry_attempts=8` backoff (0.25 → 2 s)
x chain-hold cycles of deterministic `ConnectError` (`connect_failures`)
before falling to a remote provider, then poisoned the session with a
cooldown. A refused connect is deterministic: nothing inside the backoff
horizon can make a dead port answer, so retrying it only wastes time.

The fail-fast shipped in LP-0MW0KZR7A003UFQK covers *provably dead remote
chains* (deterministic skips / terminal refusals), not a connection-refused
local backend, and no pre-dispatch check consulted the reachability evidence
`/health` already had. The fix has three parts:

1. **Fail fast on a refused connect (AC1).** The local dispatch's backend
   retry loop (`_call_with_backend_retries`) detects a deterministic
   connection-refused `httpx.ConnectError` (cause chain carries
   `ECONNREFUSED` / `ConnectionRefusedError`) and raises after a **single
   attempt** — no backoff, no retry — logging:

   ```
   backend_fail_fast path=… stream=… attempt=1/8 signal=connect_failures
     error=ConnectError reason=connection_refused
   ```

   The caller's fallback chain serves the next provider immediately. The
   same line (and the skip arm) fires at the cycle level when the
   dispatch surface raised directly, so every refused local entry is
   attributable.

2. **Pre-dispatch skip when the backend is known unreachable (AC2).** The
   routing cycle skips the local provider **before any connection attempt**
   when either signal says unreachable:

   - the armed skip deadline (an in-process TTL deadline underlying
     ``routing_skip_local reason=backend_unreachable``), written by a refused
     local dispatch (fail-fast) and by `/health`'s own probe
     (`backend_reachable=false` — the probe is already paid for, so the
     routing gate gets its freshness without per-request probing);
   - a slots snapshot older than `server.local_slots_max_snapshot_age_seconds`
     (the 6.5 h-stale snapshot of the incident). A missing snapshot is *not*
     evidence of staleness (mirrors `slots_snapshot_too_old`).

   The skip is logged with the existing `routing_skip_local` vocabulary and
   recorded as `status=backend_unreachable_skip`:

   ```
   routing_skip_local provider=local-qwen3 model=… session=… preference=…
     skip_remaining=60s reason=backend_unreachable → skipping local,
     routing to next remote provider
   ```

   Because the skip is checked at the top of the local entry, a chain-hold
   cycle that restarts from the first provider never re-dispatches the dead
   backend — the retry storm (8 attempts x N cycles) cannot recur.

3. **Bounded, self-healing state (AC3).** The skip is TTL-bounded
   (`server.local_backend_skip_seconds`, default 60): a successful local
   dispatch or a healthy `/health` probe clears it immediately, and expiry
   lets the next request re-try local. A recovered backend is therefore used
   again at the next request/probe — never a permanent quarantine, and never
   a per-turn retry storm while the backend is down (the skip persists across
   turns until cleared or expired).

Knobs (all in `proxy/config.yaml`, defaults shown):

| Key | Default | Meaning |
|---|---|---|
| `server.local_backend_skip_seconds` | `60` | TTL of the routing skip armed by a refused connect / failed probe |
| `server.local_slots_max_snapshot_age_seconds` | cadence-derived (~300) | staleness ceiling for the secondary slot-snapshot signal |
| `server.backend_retry_attempts` | `8` | unchanged — still applies to *transient* (retryable) connect errors |

Regression rows live in `tests/test_local_route_connecterror_fail_fast.py`:
(i) dead backend → single-attempt fallthrough to remote; (ii) recovered
backend → local route restored; (iii) routing skips local pre-dispatch when
`/health` shows unreachable.

### Measured attribution of the first rung-enabled 24 h window (LP-0MVZ5WV6W008UO5F)

Window 2026-11-01 11:09:07 → 2026-11-02 11:09:07 (the rung was enabled at
2026-11-01 21:27:04Z): **126** `local_eviction_after_acceptance
reason=prefill_deadline` events, all inside the 25,000-token cold-prefill
budget (max `new_tokens` 23,027), 796,993 priced new tokens discarded —
matching the usage report's `wasted prefill tokens` figure exactly. Joining
each eviction to its admission evidence at the measured dispatch start
(`eviction_ts - elapsed_s`) splits them by admission path
(`attribute_prefill_deadline.py`, see the proxy-usage-analysis
[SKILL.md triage rule](../../.pi/skills/proxy-usage-analysis/SKILL.md)):

| Admission path | Evictions | Priced new tokens discarded |
|---|---|---|
| `rung_accepted` — free-gate rung priced the wait inside the comparator then shipped (114 s; measured band 43.0–112.9 s) but above the 30 s queue budget | 26 | 79,740 |
| `high_priority_inert` — `priority=high` never queues, so the guard was inert (`no_budget`); **no** deadline/safety bound ran (fixed by LP-0MVZF3JZ9009LTN2 — this class must now read 0) | 43 | 257,778 |
| `inert_unlogged` — guard inert (uncalibrated bucket) and silent at the dispatch instant | 47 | 439,307 |
| `priced_within_queue_budget` — admitted inside the 30 s queue budget (5.3–28.0 s) yet stalled the full deadline behind another session's prefill (occupancy priced 0) | 8 | 12,954 |
| `idle_override` — admitted by the idle-capacity override, guard not consulted | 2 | 7,214 |

**Conclusions.** (1) The rung's own accepted trade is **26 / 24 h** — not
126 / 24 h: the other 100 events came from admission paths the rung does not
govern and are tracked separately (LP-0MVZF3JZ9009LTN2 high priority,
LP-0MVZF4SU10032OZH inert-unlogged, LP-0MVZF50MJ00654O1 occupancy-priced-0).
(2) None of the 7 events with a stamped (admission-time) ETA above the then-
shipped 120 s deadline is an ETA-per-token **under-prediction** — AC1's
hypothesis. In all 7
the pricing, where it ran, already priced the prompt *over* the deadline (6
high priority: 278.8–325.9 s; 1 inert medium priority: 170.3 s); each was
admitted because the admission path never enforced the bound. (3) The
attribution is reproducible with:

```bash
python3 .pi/skills/proxy-usage-analysis/scripts/attribute_prefill_deadline.py \
  --start '2026-11-01 11:09:07' --until '2026-11-02 11:09:07'
```

A post-rung window returning toward ~26 `rung_accepted` / 24 h is the
documented accepted behaviour.

The occupancy term matters because prefills are serialised
(`local_max_concurrent_prefills: 1`): the in-flight prefill holds the gate
first, so a newcomer's real wait is its own prefill **plus** the service the
gate already owes. Comparing the own-prefill term alone admitted waits the
proxy's own `/llama/local/status` payload had already flagged as unservable —
the 2026-10-21 14:56→16:49 window produced 302 queued waits, 261 of which fell
back after queueing, 215 of them at the full budget (25,804 s of
aggregate wasted wait), with 247 of the 325 full-budget losses entering the
queue `saturation=prefill`. The occupancy floor is also what the guard uses
when per-slot prefill progress is unobservable
(`prefill_progress_unobservable`, `alive=False`, `in_flight=True`): an
in-flight prefill is treated as occupying the gate for at least the calibrated
estimate, never as "free soon".

**Occupant-evidence pricing** (LP-0MVOFVCA2008DQVE). Pricing the occupant at
the *newcomer's* own `eta_prefill` (the symmetric floor) is structurally wrong
for heterogeneous traffic: a warm 7 s prompt behind a huge cold prefill was
costed at ~14 s, admitted, then cancelled by
`local_prefill_deadline_seconds` with **zero prefill progress** (measured
2026-10-25: 22 `local_eviction_after_acceptance reason=prefill_deadline`
events in 7.4 h, 12 of them user traffic; signature `eta_prefill_s=7.1
elapsed_s=120.1 progress=None alive=False in_flight=True`). Each live
dispatch record already carries the occupant's `new_tokens`, `cached_ratio`,
`model_name`, its optimistic admission `eta_prefill_s` and its `started_at`.
Since LP-0MVS9YK9P006M1NQ the guard no longer decays that optimistic ETA
(`max(0, eta_prefill_s − (now − started_at))`, which under-priced the occupant
by the measured 2-4x and converged to ~0 exactly when it passed its optimistic
ETA). It instead re-prices the occupant's **remaining uncached new tokens** at
the contended co-residency rate:

```
progress      = min(1, elapsed / contended_full_service)
remaining_new = new_tokens * (1 - progress)
remaining_s   = estimate_prefill_seconds(remaining_new, cached_ratio=0,
                                         contended=True)
```

`contended_full_service` is the contended-rate cost of the occupant's whole
new-token load, so an occupant past its *optimistic* ETA still reports the work
it genuinely owes. Over the active, still-prefilling records of *other*
sessions this remaining service is summed (a same-session re-entry never waits
behind itself). Measured evidence can only raise the symmetric floor, never
lower it; a missing/invalid `new_tokens`/`eta_prefill_s`/`started_at`, an
inactive or already-served record, or an uncalibrated contended window
contributes nothing — fail-open by construction, never a guessed multiplier.
The deadline bound is evaluated against this same projected time to first byte,
so admission and the deadline can no longer disagree about who the occupant is.

**Blocker-hold pricing** (LP-0MVV29YDQ00437N6). The per-wait diagnosis
([`docs/dev/queue-wait-abandonment-diagnosis.md`](../../docs/dev/queue-wait-abandonment-diagnosis.md),
`LP-0MVV29XPT003OXI5`) measured the occupancy term at *exactly* the symmetric
floor in 163 of 169 abandoned waits that had an admission decision (median
projection 28.4 s against the 60 s budget actually spent). The dominant blocker
class was `prefill_in_flight` (65.3 %), so each live prefilling occupant now
also prices its remaining **dispatch-lease hold** (`expires_at` on the record —
the monotonic bound on how long the local slot stays leased to that holder) and
the per-record term is `max(prefill service, lease hold)`. Still measured
evidence only: it raises the symmetric floor and never lowers it, it is not
configurable, and a record with neither signal contributes nothing. The
`occupant_hold_s` field on the decision line reports the measured hold
(`0.0` = no usable evidence). The `lease_holder_generating` minority (13.4 %)
is unchanged — renewals are capped by `LP-0MVOJ6KAW004EBV0`, not priced here.

**Contended (co-residency) rate selection** (LP-0MVS9Y76R004B5M8). Prefill is
serialised but prefill+decode batching is not, so an occupant that is
*generating* halves a newcomer's prefill throughput (measured 2-4x lower
tok/s). A calibration sample is tagged **contended** when the gate was occupied
at admission (`busy_slots >= 1`) **or** another session's dispatch record was
live at first-byte time, and stored in its own per-(model, context bucket)
window. A contended admission is priced from that window; when it is not yet
calibrated the guard falls back to the solitary window and reports
`rate_bucket=solitary_fallback` instead of guessing. Warm prompts
(`cached_ratio >= 0.9`) stay on the solitary path byte-for-byte. Each decision
line carries `rate_bucket=contended|solitary|solitary_fallback` and
`contended_samples=<n>`.

Ordering to keep (the guard does not change these values):
`contention_queue_max_wait_seconds (30) < local_prefill_deadline_seconds (180)
< local_dispatch_first_byte_timeout_seconds (840) <
local_dispatch_max_prefill_seconds (900)` — an admitted request must never be
cut off by the deadline, and the deadline must fire before the first-byte
ceiling and the wedge bound. Rollback: `local_admission_guard_policy: observe`
or `off` (or `local_prefill_deadline_seconds: 0` for the deadline bound only).
Background, evidence and the per-child cross-reference: [Contended prefill
pricing and rollback](../../docs/dev/contended-prefill-pricing.md).

`projected_wait` is computed by a single function
(`prefill_eta.projected_wait_seconds`) that also produces the published
`queue_eta_seconds`, so the number a dispatcher paces against and the verdict
the guard enforces can never drift apart (LP-0MVICD2YW004DJPV).

- **Data-driven coefficients** (`proxy/proxy/prefill_eta.py`): every local
  dispatch records `(coded_tokens, new_tokens, dispatch_to_first_byte_ms)` —
  the service time from the existing `dispatch_first_byte_ms` line, queue wait
  excluded — in a rolling window (default 200 samples) keyed by
  `(model, context bucket)` (buckets: <16k, 16k–64k, 64k–128k, ≥128k). A
  two-coefficient least-squares fit produces the per-bucket coefficients,
  which pass through a conservatism factor (1.2 — over-estimating prefers the
  fast remote call) and are clamped to plausible bounds. Defaults (0.5 ms per
  cached token, 3.9 ms per fresh token ≈ 254 tok/s solitary) apply until a
  bucket is calibrated.
- **Inert until calibrated**: with fewer than
  `local_admission_guard_min_samples` (5) observations a bucket, the guard is
  inert — today's queueing behaviour is byte-for-byte unchanged. It can never
  refuse on a guess.
- **Cache-warm stays local**: a 90 %-cached 100k prompt costs
  `0.9*100k*0.5 ms + 0.1*100k*3.9 ms ≈ 84 s` of its own prefill and is
  admitted behind a *free* gate, while the same prompt cache-cold costs
  ≈ 390 s and is refused. Behind an occupied gate the occupancy term applies
  to both, so a wait only stays admissible when the whole projected wait fits
  the budget. Locality is kept where it actually pays.
- **Priority semantics**: `high` never queues, so it has no *queue-wait*
  budget, and `low`/`medium` use `contention_queue_max_wait_seconds`. Every
  priority — `high` included — is bounded by the prefill deadline and its
  deadline-relative safety budget, because those are what actually cancel an
  accepted dispatch; only the queue-wait comparison is skipped for `high`
  (LP-0MVZF3JZ9009LTN2). Before that fix the guard returned inert
  `no_budget` for `high` and never priced it at all, which is why 43 of the
  126 `prefill_deadline` evictions in the 2026-11-01→02 window were
  high-priority dispatches (6 of them already priced above the deadline when
  they were admitted). A `high` prompt that fits both time bounds admits
  exactly as before.
- **Refusal is immediate and visible**: the request falls through to the next
  provider without waiting. If the whole chain is exhausted, the client
  receives a `429` with `X-Local-Admission: reject` /
  `X-Local-Admission-Eta` / `X-Local-Admission-Budget` headers and the
  machine-readable `code: local_admission_reject`. The retry story is
  reason-aware: ETA/occupancy reasons carry `Retry-After` + `retry_at`
  (how much longer the projected wait is than the budget), while
  `reason=prefill_budget_exceeded` (a prompt permanently over the flat
  cold-prefill token cap) carries **no** `Retry-After`, says
  `"retryable": false` and points at compaction — see the
  [per-reason retry contract](#per-reason-retry-contract-lp-0mvntt4x8009io2x).
- **Observability**: each decision logs
  `local_admission_guard decision=admit|reject reason=… enforced=true|false
  eta_prefill_s=… eta_occupancy_s=… projected_wait_s=… budget_s=…
  busy_slots=… in_flight_prefills=… free_slot=…`, i.e. the own prefill, the
  occupancy contribution, the projected wait actually compared against the
  budget and the occupancy evidence behind it — a refusal (or a surprising
  admit) is diagnosable from the log line alone. The 24h report counts them as
  `local_admission_reject` / `local_admission_reject_enforced` /
  `local_admission_admit`. `/llama/local/status` publishes
  `prefill_eta_seconds` (cost of the reference prompt, cache-cold) and
  `queue_eta_seconds` (that same cost repeated for every occupied slot, plus
  the newcomer's own — the guard's `projected_wait`), with
  `queue_eta_occupancy_seconds` / `queue_eta_busy_slots` breaking out the
  occupancy term, so a dispatcher can pace *before* launching a session.
- **Direct (free-slot) dispatch guard** (LP-0MVIQ0F8T002FS42): the same
  decision is applied one step earlier, before a request that found a *free*
  slot is dispatched. A free slot is not permission to start a prefill the
  caller will abandon — baseline 48 of 606 local dispatches had a first byte
  after >120 s, most on a free slot. The direct refusal logs the **governing**
  reason with `free_slot=true`: `reason=eta_exceeds_budget_direct` **only**
  when the free-slot ETA/budget bound is itself the cause (queue-entry
  refusals keep `reason=eta_exceeds_budget`), while a deadline,
  deadline-relative safety-budget or flat cold-prefill refusal keeps its own
  reason (`prefill_eta_exceeds_deadline` / `prefill_eta_exceeds_safety_budget`
  / `prefill_budget_exceeded`) on this path exactly as on the queue path — the
  dispatch path never changes the reason. It records a
  `local_admission_reject` attempt so the chosen fallback provider appears in
  the diagnostics, and falls through to the next provider. It reuses the same
  estimator and the same single budget (`contention_queue_max_wait_seconds`) —
  there is no second budget knob, and a cold/uncalibrated bucket still admits.
  **Deadline refusal** (LP-0MVKUKEU80022C3U): when the projected **time to
  first byte** — own ETA plus the occupied gate, i.e. the same
  `projected_wait` above — reaches ``local_prefill_deadline_seconds``
  (default `DEFAULT_PREFILL_DEADLINE_SECONDS`) the guard also refuses with
  ``reason=prefill_eta_exceeds_deadline`` (both paths). The deadline reason
  dominates the budget reason whenever it fires — it names the hard physical
  constraint that would cancel the dispatch; when the projected first byte is
  below the deadline the budget reason is reported as before. Including the
  occupancy term is the residual fix (LP-0MVOFVCA2008DQVE): a warm prompt
  whose own prefill fits easily can still be cancelled at the deadline while
  another session's prefill occupies the GPU, so it is refused (and falls
  through) instead of being dispatched into a guaranteed cancellation.

Rollout control (`server.local_admission_guard_policy`):

| Value | Behaviour |
|-------|-----------|
| `reject` (default) | Refuse admission when the projected wait exceeds the budget |
| `observe` | Log the verdict (`enforced=false`) but keep queuing — use to validate calibration on live traffic |
| `off` | Do not evaluate the guard at all |

Verification target (48h after rollout): `contention_queue_fallback_after_queue`
drops well below the 4,947 baseline, client-visible latency for refused
requests approaches the remote TTFT baseline instead of ~120 s, local serve
share stays above 2.6 %, and mid-stream error counts do not increase.

## Local-route starvation detector (LP-0MVWFKIVO007EGDU, parent AC11)

`/llama/local/status` separates the **engine** from the **route**:
`available_slots` / `capacity_hint` / `llama_server_running` describe the
former, while `local_route_available` / `local_route_reason` /
`local_route_cooldown_remaining_s` describe the latter. Both are needed because
they disagree exactly when it hurts. On 2026-10-31 the `local-qwen3` entry was
armed **217 times in 5.5 h** (median gap 67 s ≈ the 60 s cooldown TTL) while the
payload kept reporting `available_slots: 2`, `capacity_hint: idle` and
`llama_server_running: true`; local served 165 streams against 5 707 remote ones
— **2.8 % local share**.

### The published route fields (`status_schema_version` 5)

Additive since LP-0MVWFKIKJ005XHAY: every existing key keeps its name and
meaning, and the three fields below describe the *route* rather than the
*engine*. They are built read-only from the in-process armed state at request
time; the routing decision never reads them, so there is no behaviour change
and no new poll.

| Field | Type | Value when nothing is armed | Meaning |
|-------|------|--------------------------------|---------|
| `local_route_available` | bool | `true` | Would chain resolution accept the local route right now? `false` means the route is armed and local dispatches will be refused before the engine is even asked. |
| `local_route_reason` | str or `null` | `null` | Why the route is unavailable, as a canonical routing token: `local_provider_cooldown` (an entry cooldown is armed — the actionable, waitable reason) or `outside_available_window` (every local entry is outside its `available_times` window, which is not a cooldown and therefore carries no countdown). |
| `local_route_cooldown_remaining_s` | int or `null` | `null` | Seconds until the route becomes usable again — the **shortest** active arm across the eligible local entries. `null` when the route is available, and for a non-cooldown reason (never a fabricated `0`). |

The local entries judged are the ones serving the reported local model
(`current_model`), so the chat route and the embed route never decide each
other's availability; a multi-backend model with an armed sibling entry is
still reported available when any eligible entry is usable. With no local
entry configured, or unreadable engine state, the payload stays well-formed and
absent-safe. Note the deliberate split from the engine fields above: a payload
can legitimately read `available_slots: 2`, `capacity_hint: idle` and
`local_route_available: false` at the same time — that is exactly the
2026-10-31 shape, and a dispatcher must gate on the route fields, not on the
slot counts alone.

Nothing in that window said "the route is armed while the engine is idle". Each
arm was its own log line, so the episode was diagnosable only by joining
hundreds of lines after the fact. The detector below emits **one** line for the
whole episode instead.

### The line

```
WARNING local_route_starved armed_s=300 window_s=600 armed_fraction=0.50 arms=5 available_slots=2 llama_server_running=true reasons=local_provider_cooldown:3,measured_cache_deficit:2
```

| Field | Meaning |
|-------|---------|
| `armed_s` | Seconds of the rolling window in which the local route was armed (entry-wide cooldown or per-session/per-prompt thrash window). Overlapping arms count once. |
| `window_s` | Rolling window length (`server.local_route_starvation_window_seconds`, default 600). |
| `armed_fraction` | `armed_s / window_s`; the line only fires at or above `server.local_route_starvation_min_armed_fraction` (default 0.5). |
| `arms` | Arm events in the window; at least `server.local_route_starvation_min_arms` (default 2) are required, because one genuine fault is normal traffic. |
| `available_slots` | Free local slots from the existing coalesced `/slots` sample at emission time — always `> 0`, since a busy engine is not starvation. |
| `llama_server_running` | Always `true`; the warning never fires with the engine down. |
| `reasons` | Per-reason arm counts as `token:count`, most frequent first, so the cause is attributable from the one line. |

The `reasons` tokens are the canonical routing vocabulary, never a second
spelling:

| Token | Armed by | Scope |
|-------|----------|-------|
| `local_provider_cooldown` | A `type: local` entry marked unavailable by the fallback error handlers — a local 5xx (other than a deferred measured-cache-deficit 503) or a connection error, outside the cold-start grace window | whole entry |
| `measured_cache_deficit` | The per-session/per-prompt thrash window armed when the measured cache-deficit guard releases a dispatch early (see the exit table above) | one session + one prompt |
| `prefill_thrash_cooldown` | The same thrash window, armed by `prefill_deadline_exceeded` | one session + one prompt |

### When it fires

- **State-only, no new polling.** The detector is fed by the existing arm sites
  (`provider._handle_http_error_with_cooldown`,
  `provider._handle_connection_error_in_fallback`,
  `router_helpers._record_prefill_thrash_cooldown`) and is evaluated *on those
  events* — no timer, no task, no HTTP call, no `/slots` poll of its own.
- **Only while the engine is healthy.** It reads the same state the status
  surface publishes: `llama_server_running=true`, a **known**
  `available_slots > 0`, and not inside the cold-start grace window
  (`cold_start.is_cold`) — warm-up deferral is expected, never starvation. An
  unknown slot count (no fresh `/slots` sample) keeps it silent rather than
  guessing.
- **Bounded output.** At most one line per 10 minutes
  (`server.local_route_starvation_throttle_seconds`). A route that stays armed
  re-warns after the throttle, so a long episode reads as one line per 10
  minutes instead of a burst.
- **Never on a single fault.** One arm is not starvation, however long its own
  cooldown is: `min_arms` is checked before the armed fraction.

### What it never does

- **No routing input.** The tracker is observability only: no routing decision
  reads it, and it can neither arm, clear nor shorten a cooldown.
  `local_route_starvation.starvation_snapshot()` exposes the state read-only for
  tests and tooling; no `/llama/local/status` field is added for it (the
  `local_route_*` fields above are the published surface).
- **No capacity arms.** Slot-exhaustion / lease-active refusals are a verdict on
  capacity, not on the route, so they are not counted — and the healthy-engine
  condition (`available_slots > 0`) would suppress them in any case.

### Knobs and rollback

Every key is optional and lives in the `server` config section; malformed values
fall back to the documented default, so a typo can never switch the detector off
by accident.

```yaml
local_route_starvation_enabled: true            # 0 disables the detector
local_route_starvation_window_seconds: 600      # rolling window
local_route_starvation_min_armed_fraction: 0.5  # sustained share required
local_route_starvation_min_arms: 2              # one genuine fault stays silent
local_route_starvation_throttle_seconds: 600    # at most one line per 10 min
```

## Local Admission Reservations and Fallback Intent (LP-0MVIKPCSG002B0NL)

Polling `/llama/local/status` and then sending is racy: two clients can observe
the same idle window, and the admission decision is taken *after* the work has
been sent (queue depth reached 7 while `available_slots` still reported 2 with
the prefill gate saturated). The reservation handshake lets a dispatcher
**claim** local capacity atomically, before sending, and express what it wants
to happen if local capacity is not available.

### `POST /llama/local/reserve`

```
POST /llama/local/reserve
{"estimated_tokens": 48000, "session_id": "herdr-...",
 "priority": "medium", "cached_ratio": 0.35,
 "ttl_seconds": 10, "dry_run": false}

200 {"admit": true, "reservation_id": "<opaque>",
     "projected_wait_s": 63.1, "ttl_s": 10}
429 {"admit": false, "reason": "eta_exceeds_budget",
     "queue_eta_s": 187.2, "budget_s": 60.0, "retry_after_s": 28}
# or (LP-0MVKUKEU80022C3U)
429 {"admit": false, "reason": "prefill_eta_exceeds_deadline",
     "queue_eta_s": 187.2, "budget_s": 60.0, "retry_after_s": 28}
```

- The verdict is the **same** occupancy-aware decision the admission guard
  enforces (`prefill_eta.evaluate_admission`), with the same occupancy probe,
  so the two can never disagree for identical inputs.
- `200` claims a ticket: a follow-up request carrying
  `X-Local-Reservation: <id>` dispatches **without re-checking and without
  queueing**. Tickets are single-use, TTL-bound (default 10 s, capped at
  60 s) and scoped to the local model they were issued for.
- `429` refuses immediately: the client defers instead of queueing behind a
  doomed wait. `queue_eta_s` is the projected wait (`prefill_eta_seconds ×
  (busy_slots + 1)`), `budget_s` is `contention_queue_max_wait_seconds`, and
  `retry_after_s` is the overshoot.
- `DELETE /llama/local/reserve/<id>` cancels an outstanding ticket and
  releases its gate claim, so the next ticket can be issued.
- `dry_run: true` returns the verdict without claiming (the admission
  preamble).
- A stale, expired, cancelled or mismatched ticket is ignored: the request
  takes today's ordinary path (never a hard failure).
- Fail-open: an uncalibrated estimator admits (the endpoint never rejects on a
  guess), and a missing `estimated_tokens` is a `400`.

Interaction with the queue: the reservation replaces the *admission* decision,
not the capacity model. `contention_queue_max_wait_seconds` remains the wait
budget both the guard and `/reserve` compare against, and
`contention_queue_max_depth` still bounds the requests that do queue; a
reserved dispatch does not consume queue depth.

### `X-Local-Preference` request header

Exactly two values are live (`VALID_PREFERENCES == (local-preferred, remote-ok)`);
`prefer-local` and `local-only` are **retired** and answered with a `400` before
any routing (see the migration notes below).

| Value | Behaviour |
|-------|-----------|
| `remote-ok` (default) | **Local allowed, remote acceptable.** The local route stays in the chain and is attempted; when local cannot serve **right now** the request steps straight to the next provider — no queue, no deferral, no local-capacity `429`. This is what an absent header (and any unknown value) resolves to |
| `local-preferred` | **Local while the request fits local capacity, remote beyond it.** Pin local **while the request can still fit**. A *deferrable* request — the block is temporal, so waiting can help — waits for local capacity and, when the wait budget runs out, receives the retryable `429` deferral (`X-Local-Route: refused`, `X-Local-Fallback-Reason: <exit token>`, `Retry-After` / `retry_at`) instead of being moved to a paid provider. A *too-large* request — one that outgrows the local serving envelope, so waiting cannot help — is **never deferred and never refused for local capacity**: the local entry leaves the chain and a remote tier serves it (`200`, `X-Local-Route: remote`, no `Retry-After`). See [`local-preferred`: pinned while deferrable, promoted when too large](#local-preferred-pinned-while-deferrable-promoted-when-too-large) and the [per-reason retry contract](#per-reason-retry-contract-lp-0mvntt4x8009io2x) |
The `remote-ok` row above is the *historic* name for "local allowed, remote
acceptable": it once meant "remove the local route from the chain", which is the
opposite of the behaviour it selects today and must not be restored. When local
cannot serve a `remote-ok` request **right now** — the slot is saturated, the
prefill ETA is over the wait budget, the local entry is in cooldown, the
residency ladder skipped it, the lease is held, or the per-entry context cap
excludes it — the request **steps straight to the next provider**: it does
**not** enter the contention queue, does not defer, and never receives a
local-capacity `429`, so its queue wait is always `0`.

Wire compatibility: **`prefer-local` is retired** (LP-0MVUH9PUU005HVS4). It
was the default until the two-value vocabulary; it meant "wait for local while
the request fits the wait budget", which is now expressed by the explicit
`local-preferred`. An explicit `prefer-local` is **not** silently reinterpreted
as the new default — that would deliver a materially different policy with no
signal — it is refused before any routing, queueing, lease, dispatch or
cooldown state is written, with `400` and this body:

```json
{
  "error": "X-Local-Preference value is retired",
  "code": "local_preference_retired",
  "detail": "X-Local-Preference 'prefer-local' is retired and is no longer accepted; send 'local-preferred' or 'remote-ok' instead."
}
```

**Migration:** a client that needs the wait-for-local behaviour sends
`local-preferred`; a client that is happy to step remote sends `remote-ok` (or
omits the header, which is the same thing).

Wire compatibility: **`local-only` is retired** (LP-0MVUH9PAZ002UAPA). It was
the deprecated alias of `local-preferred` (LP-0MVSHWYVE005MS16); it is no
longer accepted. A request that sends it is refused **before** any routing,
queueing, lease, dispatch or cooldown state is written, with `400` and this
body:

```json
{
  "error": "X-Local-Preference value is retired",
  "code": "local_preference_retired",
  "detail": "X-Local-Preference 'local-only' is retired and is no longer accepted; send 'local-preferred' instead. A client that must never spend remote budget should pin a chain with no remote entries (e.g. the private model) rather than a preference value."
}
```

Exactly one `WARNING` names the retired value, its live successor and the
session id (`WARNING … X-Local-Preference='local-only' is retired; rejected
with 400 (code=local_preference_retired) — send 'local-preferred'
(session=…)`), so a deployed client still on the old spelling is identifiable
from `proxy.log` alone.

`code` is the machine-readable branch point: treat it as a permanent client
configuration error, not a retryable one. **Migration:** send
`local-preferred` to keep pin-local (deferral) semantics — a pure spelling
change, no behaviour change. A client that must never spend remote budget does
not need a preference value at all: pin a model whose chain has **no remote
entries** (the `private` model), which no preference — and no promotion — can
ever move to a paid provider.

Unknown values resolve to the default `remote-ok` and log a `WARNING` naming
the value received and the default it resolved to
(`WARNING … unknown X-Local-Preference='local-pref' ; using 'remote-ok'`), so a
client typo is identifiable from `proxy.log` alone; an absent header is the
same resolution without the `WARNING`. The header stays purely additive — never
a `400`, never a silent demotion of a *retired* value (those are rejected).
`high` priority keeps its "never queue" semantics under every preference.

#### Behaviour shift under the `remote-ok` default

The default is `remote-ok`, so headerless traffic now behaves as "take local
only when it is free *right now*". Two observable shifts follow, and **neither
is a local-serving defect** — they are the intended contract:

- **No queue wait by default.** `contention_queue_dispatch` and
  `contention_queue_fallback_after_queue` lines disappear for headerless
  traffic, so the measured `queued_duration` median drops to ~0 and the remote
  share rises whenever the GPU is busy at request time. The contention queue,
  the deferral/retry surface and `POST /llama/local/reserve` are reachable only
  for `local-preferred` clients.
- **Local serving is opportunistic.** A headerless request is served locally
  only when a slot is free at that instant, so local utilisation falls under
  contention and `X-Local-Route: remote` with a fall-through reason becomes the
  common outcome. The prefill-ETA admission guard still decides whether local
  can serve now, but its *refusal* outcome is unreachable for this preference:
  the request steps to the next provider instead of receiving a `429`.

Expect the daily proxy-usage report's local/remote split to move accordingly;
read the shift as the new default, not as a regression.

### `local-preferred`: pinned while deferrable, promoted when too large

`local-preferred` is one intent with two outcomes, split by a single question:
**can waiting make this request fit local capacity?**

The answer is the request's *capacity verdict*, computed exactly once per
request by `admission_reserve.local_capacity_verdict()` from evidence the
routing path already gathered — the admission guard's own decision, the
recorded local-route exit and its token evidence — and never re-priced by the
dispatcher. Two verdicts exist, and each decides what a pinned request does:

| Verdict | Meaning | What `local-preferred` does |
|---|---|---|
| `deferrable` | The block is temporal: a slot, a lease or a queue is busy and will free. Waiting can help. | **Pinned.** The request is never dispatched to a non-local provider. It waits for local capacity and, when the wait cannot help it, is answered with the retryable `429` deferral. |
| `too_large` | The request itself outgrows the local serving envelope, so no amount of waiting helps. | **Promoted.** The local entry leaves the effective chain *before* the dispatch loop (the capacity filter — the only filter that ever removes a local entry) and a remote tier serves the request: no local attempt, no queue entry, no lease, no local-capacity `429`. |

#### Pinned: a deferrable request never goes remote

Enforcement is **structural**, ahead of every non-local dispatch — not by
patching each skip branch at the admission guard. Both dispatch cycles
(`_proxy_with_fallback_cycle` and `_proxy_with_remote_fallback_cycle`) call
`_build_local_only_refusal_for_remote()` at the head of the provider loop,
immediately after the next candidate is resolved and **before** any attempt:
whenever the candidate is not a local entry and the request is pinned — the
pin-local preference (canonical `local-preferred`, including the alias
normalised to it) **and** a non-`too_large` verdict — the request is answered
with the `429` (`X-Local-Route: refused`) instead of ever reaching that
candidate. It is the mirror of the capacity promotion: where a `too_large`
verdict removes the local route from the chain *before* dispatch, a pinned
request refuses *before* any remote entry is dispatched
(LP-0MVRMC1L3006F0F3). The admission guard is still
there; it is simply no longer the only line of defence.

The pin is **not** one universal outcome per exit — it is a split decided by
the request's capacity verdict (`admission_reserve.local_capacity_verdict()`, see
the [verdict table](#local-preferred-pinned-while-deferrable-promoted-when-too-large)):

- **Deferrable** (the block is temporal, so waiting helps): the request is
  never dispatched to a non-local provider. Every exit from the local route
  ends in the retryable `429` deferral — `X-Local-Route: refused`,
  `X-Local-Fallback-Reason: <exit token>`, `Retry-After` / `retry_at` — or the
  request is completed locally. `refusing` rows in the table below.
- **Too large** (the request outgrows the local serving envelope, so waiting
  cannot help): the exit is **not** a refusal. The local entry leaves the
  chain before dispatch and a remote tier serves the request — `200`,
  `X-Local-Route: remote`, the same token as `X-Local-Fallback-Reason`, no
  `Retry-After` and no local-capacity `429`. `promoting` rows below; the
  mechanism is [Promoted: a too-large request leaves the local chain](#promoted-a-too-large-request-leaves-the-local-chain).

So a `local-preferred` request is completed locally, told when to retry
(deferrable), or promoted to a remote tier with the capacity reason stated
(too large) — and a client can tell the last two apart from the status code and
`X-Local-Route` alone, without parsing prose.

A deferral's time is spent in the contention queue. In `queue` mode a pinned
`deferrable` request waits for a local slot up to
`contention_queue_max_wait_seconds`, bounded by `contention_queue_max_depth`.
Both queue exits — the wait budget expiring, or the depth cap refusing to
enqueue it at all — and the `local_lease_active` exit end the deferral in the
retryable `429`, returned before the next candidate is resolved so nothing is
enqueued twice and no lease or attempt state is written for a refusal. A
claimed reservation (`X-Local-Reservation`) is unaffected: a pre-granted ticket
dispatches without re-checking or queueing.

The refusal's `X-Local-Fallback-Reason` names the exit. Each local-route exit
records its reason on the request (`_note_local_fallback_reason`) as it falls
through; the loop-head choke point turns that recorded reason into the refusal
token, and the same token is what the routing log line carries as `reason=` —
the header, the refusal body `reason` and `proxy.log` cannot drift. The
vocabulary is canonicalised in `admission_reserve.LOCAL_FALLBACK_REASON_TOKENS`
(LP-0MVRMC1W6003MV3Y); an exit that recorded nothing is attributed with a
structural token, never a remote fall-through: a chain with **no local entry
at all** refuses with `local_only_no_local_route` (a client that pinned local
expected no remote spend for any reason), and any other unrecorded exit with
`local_route_unavailable`.

For a `too_large` verdict the row's token is not a refusal: the entry leaves
the chain and the same token is stamped as `X-Local-Fallback-Reason` on the
`remote` response that served the request (the pin declines at this choke point
for an in-loop `context_too_large` skip, exactly as the pre-loop promotion
does for the cold-prefill and context-cap classification).

Each row's **Outcome** column says what a pinned `local-preferred` request does
with that exit: `refusing` rows end in the retryable `429` deferral, while
`promoting` rows leave the local chain and are served remotely. The promoting
set is exactly `admission_reserve.CAPACITY_TOO_LARGE_REASONS` —
`prefill_budget_exceeded` and `context_too_large` (the latter also carrying the
per-entry `max_context_tokens` ceiling, which composes into the same skip);
every other row is a temporal (deferrable) exit and therefore refuses.

| Exit from the local route | `X-Local-Fallback-Reason` | Outcome | Log line (`reason=` field) |
|---|---|---|---|
| Admission guard, queue path — projected prefill cannot finish inside the wait budget (`contention_queue_max_wait_seconds`) | `eta_exceeds_budget` | refusing | `local_admission_guard ... reason=eta_exceeds_budget ... enforced=true` |
| Admission guard, queue path — projected first byte reaches `local_prefill_deadline_seconds` | `prefill_eta_exceeds_deadline` | refusing | `local_admission_guard ... reason=prefill_eta_exceeds_deadline ...` |
| Admission guard, queue path — safety-budget bound of the deadline | `prefill_eta_exceeds_safety_budget` | refusing | `local_admission_guard ... reason=prefill_eta_exceeds_safety_budget ...` |
| Admission guard, queue path — flat cold-prefill token cap (`local_cold_prefill_budget_tokens`): the prompt outgrows the local serving envelope | `prefill_budget_exceeded` | promoting | `local_admission_guard ... reason=prefill_budget_exceeded ...` |
| Admission guard, direct free-slot dispatch path | `eta_exceeds_budget_direct` — **only** when the free-slot ETA/budget bound is itself the cause; a deadline, deadline-relative safety-budget or flat cold-prefill refusal keeps its own governing token (`prefill_eta_exceeds_deadline` / `prefill_eta_exceeds_safety_budget` / `prefill_budget_exceeded`), so the direct path never relabels a non-ETA bound (`prefill_budget_exceeded` is a `too_large` verdict there: the pin declines and a remote tier serves the request — see [Promoted](#promoted-a-too-large-request-leaves-the-local-chain)) | refusing | `local_admission_guard ... free_slot=true ...` plus, for a pin-local client, `local_only_refused reason=<token> ... route=refused`. The `local_admission_guard_direct_fallback ... → routing to next provider` line is emitted **only** when the request really falls through to the next provider (`remote-ok` / absent header — the immediate-step default) |
| Local entry in cooldown (chain resolution) | `local_provider_cooldown` | refusing | `Skipping provider=<local>: ... in cooldown (...s remaining) reason=local_provider_cooldown preference=...` |
| Router skip: prompt at/over the warm (KV-slot) cap; the per-entry `max_context_tokens` ceiling composes as an additional cap — the prompt can never fit local, however long it waits | `context_too_large` | promoting | `routing_skip_local ... reason=context_too_large → skipping local, routing to next remote provider` |
| Router skip: economic cold-prefill bypass (an idle-admission denial keeps this token) | `large_context_bypass` | refusing | `routing_skip_local ... reason=large_context_bypass ...` |
| Router skip: residency ladder found too little reusable prefix (an idle-admission denial keeps this token) | `resident_prefix_insufficient` | refusing | `routing_skip_local ... reason=resident_prefix_insufficient ...` |
| Router skip: prompt above the cold threshold with **no** measured residency (`residency_source=unknown`, i.e. no fresh `/slots` view and no slot ownership) — unmeasured is cold (`LP-0MW81DFML006O5KL`), and unlike `resident_prefix_insufficient` this denial is **not** idle-overridable, because it reports a missing measurement rather than a price | `unmeasured_residency_cold` | refusing | `routing_skip_local ... reason=unmeasured_residency_cold ...` |
| Router skip: per-tenant residency cap (`session_tenants`) | `tenant_limit` | refusing | `routing_skip_local ... reason=tenant_limit ...` |
| Router skip: (session, prompt) in post-deadline prefill-thrash cooldown — enforced at provider.py's chain entry **and** at the local dispatch choke points (`proxy_to_local` and `_try_acquire_local_dispatch`). The choke point's refusal is a **deferral**, not a fault: its 503 carries the `prefill_thrash_cooldown` dispatch-reason marker, so the entry is **not** quarantined and other sessions keep their local route (LP-0MVXSSYTN001TBW5) | `prefill_thrash_cooldown` | refusing | `routing_skip_local ... reason=prefill_thrash_cooldown ...` (`entry=router_dispatch` / `entry=lease_acquire` at the choke points) and `Fallback triggered ... reason=prefill_thrash_cooldown` (attempt record `status=local_dispatch_deferral`, no `local_provider_cooldown` arm) |
| Router skip: this request already had a local dispatch cancelled post-deadline (the request-scoped terminal marker), so local is terminal for it. Enforced at the same choke points as the cooldown row. Request-scoped by construction, so a concurrent request from the same session is unaffected. The refusal is a **deferral** carrying the `post_deadline_terminal` dispatch-reason marker, so the entry is **not** quarantined (LP-0MVXSSYTN001TBW5) | `post_deadline_terminal` | refusing | `routing_skip_local ... reason=post_deadline_terminal ...` (`entry=router_dispatch` / `entry=lease_acquire`) and `Fallback triggered ... reason=post_deadline_terminal` (attempt record `status=local_dispatch_deferral`, no `local_provider_cooldown` arm) |
| Local dispatch released by the measured cache-deficit guard (`measured_cache_abort`): the measured prefix deficit proved this prompt would re-prefill far more than admission priced, so the accepted dispatch is released in seconds and the request detours to the next provider. It fires only when all three conditions hold — the deficit ratio floor, the token bound **and** the benefit gate (the measured cold ETA exceeds the request's prefill-deadline budget) — so a **pinned** dispatch whose measured ETA fits that budget is never aborted and never reaches this row (see [Measured cache-deficit early abort](#measured-cache-deficit-early-abort-lp-0mvr7in0o000dpxn-benefit-gate-lp-0mvwbd91z004u135)). The release is a verdict on the prompt, not on local's health: the entry is **not** quarantined, and other sessions keep their local route (LP-0MVWFKESK0005G61) | `measured_cache_deficit` | refusing | `Fallback triggered ... from=<local> to=<next> reason=measured_cache_deficit` (attempt record `status=local_dispatch_deferral`, no `local_provider_cooldown` arm) |
| Contention-queue depth cap reached (never queued), or the wait budget expired | `fallback_after_queue` | refusing | `contention_queue_fallback_after_queue ... preference=...` |
| Local slots busy and the request did not queue (queue disabled, `high` priority, or a `remote-ok` request, which never waits) | `local_concurrency_limit` | refusing | attempt record `status=local_concurrency_limit`; the refusal line `local_only_refused reason=local_concurrency_limit ...` |
| Prefill-gate / lease denial (a lease-active 503 after the retry window) | `local_lease_active` | refusing | `contention_queue_fallback_after_queue ... reason=local_lease_active` |
| Quota hard stop (chain resolution) | `quota_hard_stop` | refusing | `Skipping provider=<local>: ... reason=quota_hard_stop preference=...` |
| Usage-limit reset pending (chain resolution) | `usage_limit_reset_pending` | refusing | `Skipping provider=<local>: usage_limit_reset_pending ... reason=usage_limit_reset_pending ...` |
| Outside the entry's `available_times` window (chain resolution) | `outside_available_window` | refusing | `Skipping provider=<local>: outside its available_times window (UTC) reason=outside_available_window ...` |
| The chain has no local entry at all | `local_only_no_local_route` | refusing | `local_only_refused reason=local_only_no_local_route preference=local-preferred verdict=deferrable` |
| Any exit that left no reason on the request (e.g. a plain post-dispatch slot-exhaustion 503) | `local_route_unavailable` | refusing | `local_only_refused reason=local_route_unavailable preference=local-preferred verdict=deferrable` |
| Bounded deferral — the session already spent `local_deferral_max_refusals` consecutive retryable local-capacity refusals, so the pin is declined instead of refused again | `deferral_refusals_exhausted` | promoting | `local_deferral_escalated session=… refusals=… threshold=… preference=local-preferred reason=deferral_refusals_exhausted` |

The table is pinned by `proxy/tests/test_local_only_refusal_matrix.py` (one row
per exit: a pinned `local-preferred` request ⇒ `429` + token + **no upstream
call**; `remote-ok` and an absent header ⇒ the default's immediate remote
fall-through, with no queue and no deferral) and by `proxy/tests/test_local_fallback_reason_tokens.py`
(per-branch tokens equal the log `reason=` verbatim). Every "not local" line
also carries `preference=<parsed>` and `priority_routing ... session=<id>`
(LP-0MVRMC275000R2W6), and each refusal emits exactly one
`local_only_refused reason=<token> preference=<parsed> verdict=<verdict>` line
— so a pin violation, the exit that caused it, and whether waiting can help are
attributable from `proxy.log` alone. `preference` is the parsed value (a
legacy alias header is normalised to `local-preferred`), and the `verdict`
(`deferrable` / `too_large`) is the same classification the 429 body carries,
computed once by `admission_reserve.local_capacity_verdict`
(LP-0MVSMINPR006SWT6).

**Counting refusals.** `local_only_refused` is the countable token for this
failure class: the guard's own `local_admission_guard decision=reject …` line is
shared with a `remote-ok` request, which never enters the queue at all, so
`decision=reject` alone cannot distinguish the two. The refusal
line always names its dispatch context and the admission evidence it was built
from (LP-0MVNTJQVB007QWEZ):

```
local_only_refused reason=<token> preference=<parsed> verdict=<verdict> \
  session=<id> provider=<local entry> priority=<priority> \
  new_tokens=<n> budget_tokens=<n> eta_prefill_seconds=<s> budget_seconds=<s> \
  route=refused
```

Every optional key is appended only when the refusal actually carries that
value (e.g. a chain-resolution exit has no admission decision, so the token
fields are absent); `route=refused` is constant and mirrors the
`X-Local-Route: refused` response header. `session` is resolved from the
request's session header (`unknown` when it carried none) and `provider` is the
local entry the refusal was taken on — the same values the guard decision line
reports. So `grep 'local_only_refused ' proxy.log | grep 'session=<id>'` answers
"was this client's 429 a local-only refusal, and why" without correlating
headers, and `grep -c 'local_only_refused route=refused'` counts the class.

Conversely, no log line may claim routing that did not happen: for a pin-local
refusal the direct-guard dispatch point emits `local_only_refused` and **not**
`local_admission_guard_direct_fallback … → routing to next provider`, which is
reserved for a real fall-through (the `remote-ok` / absent-header clients that
are served by the next provider).

**Worked example.** The 2026-10-25 refusal of session
`herdr-1792897705-4039879-16152` — a cache-cold 100k-token prompt priced far
outside the wait budget and refused by the direct free-slot guard while a
local slot was free — reproduces in
`proxy/tests/test_local_admission_guard.py::test_direct_guard_pin_refusal_is_attributable_and_claims_no_fallback`
and leaves this output (fields elided with `…` are unchanged by this contract;
the reproducer asserts the refusal line below is the exact line the code
emits, and that no fall-through line accompanies it). `budget_seconds=60.0` is
the budget that was in force on 2026-10-25, recorded verbatim — the shipped
budget has since been swept (see the straggler-net section above):

```text
local_admission_guard decision=reject reason=eta_exceeds_budget_direct enforced=true eta_prefill_s=468.0 … new_tokens=100000 budget_tokens=0 … free_slot=true priority=low policy=reject model=Qwen3 … provider=local-llama session=herdr-1792897705-4039879-16152
local_only_refused reason=eta_exceeds_budget_direct preference=local-preferred verdict=deferrable session=herdr-1792897705-4039879-16152 provider=local-llama priority=low new_tokens=100000 budget_tokens=0 eta_prefill_seconds=468.0 budget_seconds=60.0 route=refused
```

The example contains no `local_admission_guard_direct_fallback` line at all:
this request was refused, never routed onwards.

#### Promoted: a too-large request leaves the local chain

A `too_large` verdict is never deferred and never refused for local capacity.
It is decided **before** the dispatch loop, once the authoritative routing
estimate exists, by `_promote_too_large_local_request()`: the local entry is
filtered out with the same `apply_local_preference()` capacity filter used for
every other promotion, so the remote tiers serve the request with no local
attempt, no queue entry and no lease. The per-request
routing line reports `outcome=promoted`, which is how an operator tells a
capacity promotion from a client that never had a local route.

Two thresholds define the local serving envelope:

| Threshold | Config key (shipped value) | What it bounds |
|---|---|---|
| Cold-prefill token cap | `server.local_cold_prefill_budget_tokens` (`25000`) | The **new (uncached)** tokens local is willing to prefill. The expected cached ratio is applied first, so a long prompt whose reusable prefix keeps its *new*-token count under the cap stays eligible; the comparison is `new_tokens > budget_tokens`. |
| Local context limit | `server.local_model_ctx_size` (`262144`) per local slot — `local_model_ctx_size // slots` minus the output headroom — composed with a local entry's optional `max_context_tokens` ceiling | The physical KV-slot ceiling (the warm routing cap). A prompt at or over it can never be served locally, however long a client is willing to wait. |

The same classification reaches the pin from inside the loop too: a router skip
that reports `context_too_large` (or the guard's `prefill_budget_exceeded`)
yields a `too_large` verdict, so the pin declines and the request falls through
to the remote tiers instead of being refused. A `too_large` request therefore
never receives a local-capacity `429` with retry timing: those two reasons are
compaction signals, and a `local-preferred` client meets them as a refusal only
when its chain has no remote tier to promote to (the remote-less carve-out
below), where the refusal carries no `Retry-After` and `retryable: false`.

#### Worked example: the 2026-10-28 incident

A `local-preferred` request for `new_tokens=6627` — comfortably inside the
`25000`-token cold-prefill budget — arrived while the single local slot was
mid-prefill. The admission guard measured a projected wait more than twice the
wait budget and logged:

```
local_admission_guard decision=reject reason=eta_exceeds_budget_direct
  eta_prefill_s=31.7 eta_occupancy_s=31.7 projected_wait_s=63.4
  budget_s=60.0 busy_slots=1
```

Under the old pin-local reading this was the end of the request: the only
outcomes were a silent dispatch to a paid provider or a terminal "All providers
exhausted" body, neither of which the client could act on. The capacity verdict
is `deferrable` — the prompt fits locally, only the *moment* does not — so the
request stays pinned and is answered with the retryable `429`:
`X-Local-Route: refused`, `X-Local-Fallback-Reason: eta_exceeds_budget_direct`,
body `code: local_admission_reject`, `retryable: true`, `Retry-After: 4`
(`ceil(63.4 − 60.0)`, the overshoot over the wait budget, and `retry_at` naming
the same instant in RFC 3339 UTC), with `new_tokens: 6627` /
`X-Local-Admission-New-Tokens: 6627` against `budget_tokens: 25000` /
`X-Local-Admission-Budget-Tokens: 25000`. A client that honours the retryable
`429` retries once the advertised slot frees and is served locally; nothing is
silently dispatched to a paid provider, and a momentarily busy local slot never
surfaces as a terminal exhaustion. (This direct free-slot refusal is the one
shape that does not carry the body `verdict` field yet —
LP-0MVSS435600370D6.)

#### Remote-less (`private`) chains still defer

Stripping never *manufactures* an empty chain. When every entry is
`type: local` — the `private` model — the capacity filter returns the chain
intact, so its existing local attempt and deferral behaviour applies unchanged:
a busy local slot still queues and still ends in the retryable `429` rather
than a paid fallback. `local-preferred` promotion needs a remote tier to promote
to, so a remote-less chain cannot promote at all; its too-large prompts are
bounded by the local capacity limits instead.

#### Bounded deferral: after three refusals the request goes remote

A pinned request is never *silently* promoted while it still fits local — but
it is also never deferred for ever. The proxy keeps a per-session count of
**consecutive retryable local-capacity refusals** (keyed by the resolved
session id, TTL one hour, store bounded) and escalates at the threshold
`local_deferral_max_refusals` (default `3`):

| Stage | What the client sees |
|---|---|
| Refusals 1–3 | Unchanged retryable `429`: `X-Local-Route: refused`, `X-Local-Fallback-Reason: <exit token>`, `retryable: true`, `Retry-After` + `retry_at`. The body additionally reports `deferral_refusals` (count so far) and `deferral_refusals_remaining` (`0` on the third), so the client can see the escalation coming — compact or switch preference before it happens. |
| The next request from that session | **Not refused.** The local entry leaves the chain before dispatch and a remote tier serves the request: the upstream status, `X-Local-Route: remote`, `X-Local-Fallback-Reason: deferral_refusals_exhausted`, **no** `Retry-After` (the terminal path advertises no wait it will not honour), and one `local_deferral_escalated session=… refusals=… threshold=…` log line. |

`deferral_refusals_exhausted` is a *different* token from the `too_large`
promotion, so a client (and an operator reading `proxy.log`) can always tell
"we stopped waiting for you" from "your prompt cannot fit local".

The counter resets when the session is served locally, when a non-retryable
refusal (a prompt that must be compacted) or a `too_large` promotion happens,
when the entry's TTL expires and when the threshold is reconfigured. Three
cases never escalate:

* **No resolvable session id** — there is no stable key to count against, so
  today's unbounded refusal behaviour is preserved.
* **A chain with no remote entry** (e.g. `private`) — nothing to promote to, so
  the pin keeps refusing; that is the documented zero-remote-spend escape hatch.
* **`local_deferral_max_refusals: 0`** — the escalation is off and behaviour is
  exactly as before the bound existed (the rollback).

#### Client migration: what a client author must do

**One change is required: replace `local-only` with `local-preferred`.** The
old spelling is retired and now answers `400` with
`code: local_preference_retired` (see [the retirement
note](#x-local-preference-request-header)) instead of being normalised, so a
client that still sends it gets no answer at all. The rename is the whole
migration — the request intent and the deferral behaviour described here are
unchanged.

What *is* needed for end-to-end deferral is client-side and tracked outside
this repository, in the ContextHub store:

- **ContextHub epic WL-0MVSI1JMN002CGC0** — the client-side `local-only` →
  `local-preferred` rename: herdr **WL-0MVSI1RX2005O9NE** and the pi extension
  **WL-0MVSI1S5P003UAMB**.
- **WL-0MVSBF5XZ001H7V7** — the client honouring the retryable `429`. This is
  the half that makes deferral real end to end: a client that treats the `429`
  as terminal (or silently retries a different model) gets none of the benefit,
  however correct the proxy side is. Read `retryable`; when it is `true`, wait
  for `Retry-After` (or until `retry_at`) and resend the **same** request
  pinned to local; when it is `false`, compact the prompt to
  `budget_tokens` and resend instead of retrying.

All of the above is client-side and out of scope for this repository: the
contract in this document is what those clients implement against.

### Response headers

Every response carries `X-Local-Route`:

- `local` — served by a local provider.
- `remote` — served by a remote provider; `X-Local-Fallback-Reason` names why
  local was not used — one of the exit tokens in the
  [pin-local section](#local-preferred-pinned-while-deferrable-promoted-when-too-large),
  or `no_local_provider` when the local route was never attempted (a chain
  that never had a local entry, or a labelled chain with no entry for the
  request's priority). A `local-preferred` request promoted because it is
  `too_large` also lands here, with the capacity reason as its token: no
  `Retry-After`, no local-capacity `429`. Promotion never manufactures an empty
  chain — a chain with **no non-local entry** (the `private` model) is never
  stripped, so it keeps deferring and refuses with `Retry-After` instead of
  falling through to a provider that does not exist.
- `refused` — a refusal response: the pin-local deferral or refusal, or the
  admission-guard refusal after the rest of the chain is exhausted.
  `X-Local-Fallback-Reason` carries the exit's reason token, byte-identical to
  the body `reason`. Refusals also carry `Retry-After` (only when waiting can
  help), `retry_at` (the same instant, RFC 3339 UTC) and the
  `X-Local-Admission*` estimate and token headers — see the
  [per-reason retry contract](#per-reason-retry-contract-lp-0mvntt4x8009io2x).

`remote-ok` (the default, and what an absent header resolves to) keeps the
local route and is enforced at the **admission** decision rather than at the
chain: on every one of the exits above it falls through to the next provider,
immediately, with no queue wait and no local-capacity `429`. The local entry is
never removed from the chain *before* dispatch, so a `remote-ok` request
reaches a local-route exit exactly like an absent header does — it simply never
waits at one. The pin adds the deferral for `local-preferred` while its verdict
is `deferrable`, and drops the local entry entirely when the verdict is
`too_large`. The only chain-level strip that exists is the capacity filter, and
no preference value can trigger it.

### Per-reason retry contract (LP-0MVNTT4X8009IO2X)

Every refusal is a `429` stamped `X-Local-Route: refused` that answers two
questions a scheduler needs: **whether** waiting can help (`retryable`) and,
when it can, **when** (`Retry-After` and an absolute `retry_at`). Every field
and header a client must honour:

- status `429` + `X-Local-Route: refused` — the request was refused, not served
  and not moved remote. The same headers appear on the deferral (a pinned
  request whose wait ran out) and on the admission refusal.
- `X-Local-Fallback-Reason` (header) and `reason` (body) — the exit token,
  byte-identical on both surfaces, naming *why* local could not serve the
  request. The token vocabulary is the exit table in the
  [pin-local section](#local-preferred-pinned-while-deferrable-promoted-when-too-large).
- `Retry-After` — relative seconds to wait before retrying. Present **only**
  when waiting can help; absent when it cannot.
- `retry_at` — the same instant as an absolute RFC 3339 UTC timestamp, so a
  client need not trust its own clock. Present exactly when `Retry-After` is.
- `retryable` — `false` when no amount of waiting can make this request fit
  local capacity; the remedy is compaction, not a timer.
- `suggested_action` — human-readable remediation: "compact and resend" when
  `retryable: false`, "wait / reduce the size / route remote" when `true`. On
  `retryable: false` the shipped string still carries a pre-deferral clause
  suggesting a switch to `remote-ok` to reach a remote fallback. A `too_large`
  `local-preferred` request is already promoted automatically, so treat the
  compaction half (resend at `budget_tokens` or less) as the actionable remedy.
- `verdict` — `deferrable` when the block is temporal (waiting can help) or
  `too_large` when the request itself exceeds the local serving envelope
  (`prefill_budget_exceeded` / `context_too_large`). Additive: every other
  field/header keeps its value. It is computed exactly once by
  `admission_reserve.local_capacity_verdict` in the pin-local refusal builder,
  which is the only verdict computation in `provider.py`
  (LP-0MVSMINPR006SWT6). A deferral therefore always reads
  `verdict: deferrable` — that is what kept the request pinned. Every
  **pin-local** refusal carries it: the queue/lease deferral and the direct
  free-slot admission refusal share the one builder, so both report the verdict
  and the same body/header shape (LP-0MVNTJJGX00375VQ AC1/AC2). The
  *chain-exhaustion* admission-refusal variant (the `local_admission_reject`
  response a non-pinned request reaches when the rest of the chain is
  genuinely exhausted) is the one surface that does not (LP-0MVSS435600370D6).
- Token evidence — `X-Local-Admission-New-Tokens` / body `new_tokens` and
  `X-Local-Admission-Budget-Tokens` / body `budget_tokens` on **every**
  refusal, so a client can size the compaction: how far over the flat
  `server.local_cold_prefill_budget_tokens` cap the request was.
- Admission estimate — `X-Local-Admission: reject` plus
  `X-Local-Admission-Eta` / `X-Local-Admission-Budget` (the projected prefill
  ETA and the wait budget in seconds), and the same numbers in the body as
  `eta_prefill_seconds`, `budget_seconds` and, when the guard measured the
  occupancy ahead of the request, `projected_wait_seconds`.
- `code` — the stable discriminator `local_admission_reject`, so a client can
  classify the refusal without relying on the prose in `detail`. It is never
  one of the exhaustion codes (`all_exhausted`, `outside_time_window`,
  `all_slots_exhausted`): those classify a chain that really did exhaust, while
  `local_admission_reject` classifies a request the **local** capacity guard
  refused (LP-0MU56ZM0K005F69G, LP-0MVNTJJGX00375VQ).
- `error` / `detail` — `error: "Local admission refused"`, with a `detail`
  naming the pin intent (`X-Local-Preference: local-only` / `local-preferred`)
  and the guard's refusal. A pin-local refusal is a capacity/intent outcome,
  not chain exhaustion: the rest of the chain was never consulted, so these
  responses never say `"All providers exhausted"` — that prose is reserved for
  a genuinely exhausted chain, where it stays byte-compatible for older
  clients (LP-0MVNTJJGX00375VQ AC1/AC3).

**Client action on a `local_admission_reject` (LP-0MVNTJJGX00375VQ AC5).** Read
`retryable` (and `code`), never the prose:

- `retryable: true` — honour `retry_at` (equivalently `Retry-After`) and resend
  the **same** request pinned to local; a slot, lease or queue is what blocked
  it and waiting can help.
- `retryable: false` — compact the prompt to `budget_tokens` (or below the local
  context ceiling) and resend; no amount of waiting makes this request fit
  local capacity.
- To let a remote provider serve such a request instead, send
  `X-Local-Preference: remote-ok` (or omit the header, which is the default):
  `remote-ok` never pins, so the remote fallback is reachable again. A client that cannot change its routing
  must read the refusal as "local cannot serve this request now", never as "all
  providers are down" — the two are different `code` values and different
  prose.

| Reason | `Retry-After` | `retryable` | `retry_at` | Client action |
|---|---|---|---|---|
| `prefill_budget_exceeded` | absent | `false` | absent | compact the prompt |
| `context_too_large` | absent | `false` | absent | compact the prompt |
| `eta_exceeds_budget` | wait − budget, ≥1 s, capped | `true` | RFC 3339 UTC | wait, then retry |
| `eta_exceeds_budget_direct` | wait − budget, ≥1 s, capped | `true` | RFC 3339 UTC | wait, then retry |
| `prefill_eta_exceeds_deadline` | wait − budget, ≥1 s, capped | `true` | RFC 3339 UTC | wait, then retry |
| `prefill_eta_exceeds_safety_budget` | wait − budget, ≥1 s, capped | `true` | RFC 3339 UTC | wait, then retry |
| `cooldown` | real cooldown remainder | `true` | RFC 3339 UTC | wait, then retry |
| `all_slots_exhausted` | measured occupant remainder, else `slot_unavailable_retry_after`; ≥1 s, capped | `true` | RFC 3339 UTC | wait, then retry |

**The dispatch path never changes the reason.** `eta_exceeds_budget_direct` is
the free-slot *label* of the ETA/budget bound **only**: a deadline,
deadline-relative safety-budget or flat cold-prefill refusal keeps its
governing reason (`prefill_eta_exceeds_deadline` /
`prefill_eta_exceeds_safety_budget` / `prefill_budget_exceeded`) on **both**
the queue and the free-slot path, so the row above applies to it unchanged
wherever it fires. Relabelling a non-ETA bound as the ETA/budget reason would
advertise a timer for a prompt that waiting cannot heal — the 2026-10-25
retry-storm shape. The split is pinned end-to-end on the free-slot path by
`proxy/tests/test_local_admission_guard.py::test_direct_path_names_the_governing_reason_end_to_end`
and `::test_direct_free_slot_budget_refusal_keeps_its_reason`, the guard's
label rule by `proxy/tests/test_prefill_eta_deadline.py`'s direct-path tests,
and this wording by `proxy/tests/test_direct_path_refusal_docs.py`.

**Which reasons have no timer — and why.** `prefill_budget_exceeded` is a
property of the *request*: the prompt is permanently over the flat
`local_cold_prefill_budget_tokens` cap, so any advertised delay would invite
exactly the futile retry storm observed on 2026-10-25 (the client backed off
seven times on `Retry-After: 1` before reporting the 429 terminal).
`context_too_large` is the same shape at the physical KV-slot cap: only
compaction (shorten the session history) changes the verdict.

**When waiting helps, the timer is real.** For the ETA/occupancy reasons it is
`ceil(projected_wait_seconds − budget_seconds)`, at least 1 s and capped by
`server.max_retry_after_seconds`; `retry_at` is that same delay from now, RFC
3339 UTC. For `cooldown`/lease-held/contention-queue-expiry skips the timer is
derived from the *real* expiry state (the cooldown remainder, the lease
remainder or the queue exit) — never from the ETA formula — so the hint tracks
when local capacity actually returns.

**Slot exhaustion carries the same vocabulary (LP-0MVUXY1CL005KWCC).** The
chain-level slot-exhaustion `429` (`code: all_slots_exhausted`, produced when
every provider in the chain was local and every local slot was busy) is not a
pin-local refusal — it is the terminal exhaustion surface, so it carries no
`X-Local-Route` header — but it answers the same two questions in the same
fields: `Retry-After` equal to the body `retry_after`, an absolute `retry_at`
(RFC 3339 UTC) and `retryable: true`. The delay is the **measured** remaining
service owed by the live occupant(s) when the proxy can observe it, otherwise
`server.slot_unavailable_retry_after` (default `5` s — the knob the
router-level slot `503` advertises, so the two slot surfaces agree on the order
of magnitude); it is floored at 1 s and capped by
`server.max_retry_after_seconds`. The advertised timer also bounds a chain hold
(`min(chain_hold_seconds, retry_after)`) instead of the legacy full
`chain_hold_seconds` hold. Client guidance: honour `retry_at` when present and
retry once when it elapses; the value is a measured remainder or the configured
short slot default, never an optimistic guess about a decoding occupant.

The real-expiry mapping is the contract of LP-0MVQ23B80009OZ9V. Every
non-admission exit advertises the expiry of the state that actually blocked
it, never the ETA formula:

- a local entry **cooling down** → the cooldown remainder
  (`_provider_unavailable_until`, the state behind `unavailable_providers`);
- a **quarantined local account** → the usage-limit reset remainder
  (`_usage_reset_at`);
- a local entry **outside its `available_times` window** → the seconds until
  the next window opens;
- a **held local dispatch lease** → the lease remainder (the live dispatch
  record's `expires_at`, else the configured
  `local_dispatch_lease_timeout_seconds`);
- a **contention-queue exit** → the queue's wait window
  (`contention_queue_max_wait_seconds`);
- a **post-deadline prefill-thrash cooldown** → the session's cooldown
  remainder.

A chain-resolution exit (cooldown / reset / window) records no expiry on the
request, so it is recomputed from live local availability state at refusal
time; an exit that measures its own expiry (lease, queue, thrash cooldown)
records it as it falls through. Only the pin-local refusal advertises these
values — only a pinned `local-preferred` request advertises them, while the
default `remote-ok` steps straight to the next provider, and the ETA/occupancy
reasons keep their `ceil(wait − budget)` timer.

**Local-preferred means pinned while deferrable, promoted when too large.** A
pin-local request whose capacity verdict is `deferrable` is never moved to a
paid provider: it is answered with this `429` (`X-Local-Route: refused`, body
`verdict: deferrable`) even while remote providers are available, and the
advertised timer says when a retry can succeed — the walk-through and the full
exit→token table are in
[`local-preferred`: pinned while deferrable, promoted when too large](#local-preferred-pinned-while-deferrable-promoted-when-too-large).
A request that is `too_large` is **never refused for local capacity**: waiting
cannot help, so the local entry leaves the chain and a remote tier serves it
(`outcome=promoted`). Under `remote-ok` (the default) an exit instead falls
through to the next provider immediately, and waiting is the client's own
choice — a client that wants the proxy to wait opts in with `local-preferred`.

`/llama/local/status` publishes the reservation counters
(`reserved_outstanding`, `reserved_issued`, `reserved_used`,
`reserved_cancelled`, `reserved_expired`) — `issued − used − cancelled −
expired` is the reserved-but-unused signal for ticket hoarding.

## Cross-host audit admission budget (LP-0MW95FYMH006XYNK)

Audits run against the **one local `llama-server` GPU that every host and
project shares**, and per-host guards cannot serialise them: with two agent
hosts and four projects, the per-host one-audit-per-host guards permitted
five audits plus a plan run to overlap on the single GPU (2026-11-03
08:30–09:10: 592,452 prompt tokens / 4,345 decoded tokens across 30 streams
in 40 minutes). The proxy is the one component every host talks to, so it
owns the audit cap and the live lease registry (`proxy/proxy/audit_budget.py`):
the ContextHub per-host `budget.json` (`maxAudit`) is **no longer
authoritative** for audit admission.

### The gate: a long-TTL audit-tier lease

The proxy issues **audit-tier leases** — long-TTL (default 1 h), renewable,
releasable — keyed by audit session (`X-Session-Id` prefix `audit-`). At most
`audit_admission.max_audits` (default 1) leases are live at once, so a second
**different** audit session is refused with a stable token and waits instead
of competing:

```
GET /llama/local/audit/availability
  -> {cap: 1, live: 0, available: 1, leases: []}
  -> {cap: 1, live: 1, available: 0,
     leases: [{session_id, client_ip, acquired_at, expires_at, expires_in_s}]}

POST /llama/local/audit/lease  {"session_id": "audit-...", "cap": 1}
  -> 200 {admit: true, lease_id, cap, live, available, ttl_s, expires_at, ...}
  -> 429 {admit: false, reason: "audit-budget-saturated", cap, live,
         available: 0, retry_after_s: <bounded>}

POST   /llama/local/audit/lease/renew   {"session_id": "audit-..."}
  -> 200 {renewed: true, lease_id, live, cap, expires_at, ...}
       # no live lease -> 200 {renewed: false} (next acquire starts fresh)

DELETE /llama/local/audit/lease/<session_id>
  -> 200 {released: true, live}   |   200 {released: false} (no-op)
```

- **Single mutation path.** Acquire/renew/release are the only ways to change
the live lease set; a refused acquire performs no partial state change.
- **Idempotent re-acquire.** The same session re-acquiring (or renewing) its
own lease is a renew — never a self-refusal. A lease older than its TTL is
reclaimed (`audit_lease_reclaimed reason=expired`), so a crashed holder can
never wedge the budget.
- **Proxy clock only.** Expiry comparisons use the proxy's monotonic clock, so
a skewed client host cannot hold a stale lease.
- **Client cap clamped.** A client may send a requested `cap`; it is clamped
to the proxy's configured `audit_admission.max_audits` (never honoured above
it, never a capacity grant the proxy does not own).
- **Fail-safe.** An internal registry failure returns the stable token
`audit-budget-unavailable` with a bounded `retry_after_s`; the availability
read reports `error: registry_unavailable` instead of asserting capacity. The
audit skill's documented fallback is a bounded wait then legacy per-host
behaviour — a proxy error never blocks every audit indefinitely.
- **Kill switch.** `audit_admission.max_audits: 0` disables audit admission
entirely (mirrors the ContextHub `maxAudit: 0` kill switch).

### Dispatcher integration (parent AC3)

The herdr dispatcher's capacity decision consumes this **lease-derived
availability signal** (`GET /llama/local/audit/availability`) instead of raw
version counts or a per-host `budget.json` assertion, so dispatcher admission
and proxy refusal can never disagree — even when ContextHub runs on several
servers. `budget.json` is read for a startup migration warning only
(`conflicting client-side cap`), never for admission.

`/llama/local/status` publishes the audit-budget counters
(`audit_leases_live`, `audit_leases_issued`, `audit_leases_reclaimed`,
`audit_leases_released`, `audit_refused_saturated`,
`audit_refused_unavailable`).

## Stream Error Handling (recovery-first + informative-error fallback)

### Prefill deadline (LP-0MVIQ0G2M0006QF6)

Before the first byte, a local dispatch is bounded by
`server.local_prefill_deadline_seconds` (0 disables). The
effective first-byte ceiling is the tighter of this and
`local_dispatch_first_byte_timeout_seconds` (840 s), so by default the deadline
is the bound that fires. On expiry the pending upstream request is cancelled
(the monitor cancels its `open_task`), the local dispatch aborts with the
retryable 503 the first-byte ceiling already produces — so the chain detours
to the next provider — and the dispatch lease/slot is released. The event logs
`prefill_deadline_exceeded session=… deadline=…s` so the detour is
attributable; `dispatch_first_byte_timeout` is logged when the (looser) wedge
ceiling is what fired instead.

A dispatch cancelled by the deadline never produced a first byte, so nothing
was prefilled and nothing is retained: the router calls
`_release_prefill_deadline_dispatch` in the same branch, which deletes the
dispatch record and pops the session's `local_prefill_in_flight` marker
(logged `lease_released session=… reason=prefill_deadline endpoint=…`). Only
that session's lease/marker are touched. Before this release the cleanup path
merely marked the record inactive with a *future* `expires_at`
(`lease_renewed`) and left the marker behind — the mechanism by which one
wedged session held the local pool for ~4 h on 2026-10-23
(LP-0MVKUKFSO008GI6O AC1).

The deadline sits **above** the admission budget
(`contention_queue_max_wait_seconds`, 30 s) so a request legitimately admitted
inside its budget is never cut off, and **below**
`local_dispatch_first_byte_timeout_seconds` (840 s), which is itself below
`local_dispatch_max_prefill_seconds` (900 s) and below the client's own 900 s
queue deadline. The ordering is asserted at startup by
`_startup_validate_prefill_timeout_config` (via
`prefill_eta.validate_prefill_timeout_ordering`): a violating configuration is
**fatal**, because an out-of-order timeout set either cuts off requests the
admission guard legitimately admitted or races the client's own deadline. It
is the hard backstop that does not depend on ETA-estimator quality.

#### Measured cache-deficit early abort (LP-0MVR7IN0O000DPXN, benefit gate LP-0MVWBD91Z004U135)

Admission-time warmth is only a *prediction*: `routing_resident_prefix`
decisions carry `n_past_actual=none`, so a slot owner is admitted at
`reuse_ratio=1.0` even when llama-server is about to re-prefill the prompt from
scratch (measured 2026-10-26: 29 of 116 deadline evictions were ~56k prompts
priced at 17-182 new tokens). With `server.local_measured_cache_abort_enabled`
on, the proxy asks llama-server for `prompt_progress` inside the SSE stream
(`return_progress: true` on a **copy** of the client body — the key never
reaches recorded payloads) and, at the first progress event, compares the
**measured** compute (`total - cache`) with the admission-priced `new_tokens`.
When the measurement proves the dispatch is re-prefilling far more than priced,
the accepted dispatch is released in **seconds** (not at
`local_prefill_deadline_seconds`), the shared
`local_eviction_after_acceptance reason=measured_cache_deficit` event records
the evidence (`total`, `measured_cache`, `priced_new_tokens`, `elapsed_s`), and
the request receives the same retryable 503 the deadline returns, so it detours
to the next remote provider.

The abort needs **all three** conditions, in this enforced order:

1. **deficit ratio floor** — `(total - cache) / total >=
   local_measured_cache_abort_min_deficit_ratio` (shipped **0.25**). A genuine
   warm dispatch shows `cache >= 0.99 × total` (ratio <= 0.01), so no prompt
   size and no token bound can ever trip the abort on it.
2. **token bound** — the measured new tokens exceed the priced `new_tokens` by
   more than `local_measured_cache_abort_bound_tokens` (shipped **2 000**).
3. **benefit gate** — the measured deficit priced with the admission guard's
   own calibrated model (`proxy.prefill_eta.estimate_prefill_seconds` at
   `cached_ratio=0.0`) **exceeds the request's own prefill-deadline budget**:

   ```text
   safety budget (the comparator) = local_admission_guard_budget_safety_fraction
                                    × local_prefill_deadline_seconds
   ```

   This is the **one canonical derivation** for the comparator in this
   document (the matching canonical block in `proxy/config.yaml` sits at
   `local_admission_guard_budget_safety_fraction`); every other mention here
   says "the safety budget" or "the comparator" instead of repeating a number.
   With the shipped keys it is **171 s**. `proxy/config.yaml` carries the same
   two keys, so a retune of either moves the figure — and the
   documentation-drift test in
   `proxy/tests/test_measured_cache_abort_benefit_docs.py` fails naming the
   file, the expected figure and the exact string to substitute.

   Falling back to the bare `local_prefill_deadline_seconds` when the safety
   fraction is `0` (the "no safety margin" rollback). The
   comparator comes from `proxy.prefill_eta.prefill_deadline_budget_seconds` —
   the same helper the admission guard enforces — so the advertised and the
   enforced threshold cannot drift.

The rule in one line: **abort only when the measured cold ETA exceeds the
prefill-deadline budget.** An ETA equal to the budget is *within* budget and
continues. The gate prices the **measured** `total - cache`, never the
admission-priced `new_tokens`: with the incident's `total=14380 cache=0`,
`priced_new_tokens=48` the deficit is 14 380 tokens, which measures 56.1 s at
the shipped rate fit (38.8 s at the host median 371 t/s, 84.6 s at p25) — the
8.2 s a 49-token price produces is the fake-warm optimism the abort exists to
correct, so it must never price the gate.

The admission **wait** budget is explicitly **NOT** the comparator. It bounds
queueing (`budget_seconds`, derived from `contention_queue_max_wait_seconds`,
shipped **30 s**; the recorded incident carried 60 s) and is already spent by
the time a dispatch is accepted and prefilling, so
`local_prefill_deadline_seconds` is what actually cancels it. The choice is
load-bearing: against a 60 s
comparator the incident prompt (84.6 s at the host p25 rate) still aborts, i.e.
the defect survives; against the 114 s comparator then shipped it continued at
every measured rate with at least 29 s of margin.

**A pinned dispatch whose measured ETA fits the budget is never aborted.**
This is the point of the gate. A `local-preferred` (pin-local) request may not
detour: the contract converts the abort's retryable 503 into a retryable
`429 Local admission refused` and the abort also arms
`local_prefill_thrash_cooldown_seconds` (**120 s**), so an unjustified abort is
a long-lived hard refusal for exactly the client that cannot escape it. When
the gate declines, nothing else happens: no release, no 503, no detour and no
cooldown (the arming site sits inside the abort branch and is never reached),
and the dispatch is served locally.

**Not a provider fault — the detour does not quarantine the entry.** The
release is a verdict on the *prompt*, not on local's health: the 503 carries
the `measured_cache_deficit` dispatch-reason marker (on the response header and
on the in-process attribute, so a lost header cannot silently re-enable the
quarantine), and the fallback handler **defers** it instead of arming
`provider_cooldown_seconds` for the `local-qwen3` entry (LP-0MVWFKESK0005G61).
The attempt record reads `status=local_dispatch_deferral`, no
`local_provider_cooldown` arm is produced, and every other session keeps its
local route while the GPU is free. This mirrors the cold-start grace immunity
([Local cold-start immunity](#local-cold-start-immunity-lp-0mv9unioz009rian)):
same deferral shape, different reason. Only the aborting session is kept off
local, by the per-session/per-prompt `local_prefill_thrash_cooldown_seconds`
window armed before the 503 is returned; an unmarked local 503 outside the
cold window keeps the entry quarantine unchanged, so `local-qwen3 in cooldown`
stays trustworthy evidence of a real fault. The slot marker the abort leaves
behind is not freed by the abort: it is annotated `abandoned_busy` while
llama-server drains the cancelled prefill and released on the first fresh
`/slots` snapshot that reports the slot free (see
[Abandoned local dispatches](#abandoned-local-dispatches-lp-0mvrmi5hk00558tb-ac1))
— there is no per-reason TTL for this reason.

**Vocabulary.** The gate adds exactly one `detail` token, `within_eta_budget`,
and no new reason token, no new response header and no refusal body change (the
refusal surface belongs to `LP-0MVSS435600370D6`). Every value on the decision:

| `detail=` | Verdict | Emitted when |
|---|---|---|
| `disabled` | continue | `local_measured_cache_abort_enabled` is `0`/`false`/`off` (the documented feature rollback) |
| `no_measurement` | continue | no `prompt_progress` measurement, or a field missing |
| `invalid_measurement` | continue | `total <= 0`, `cache < 0` or `cache > total` |
| `invalid_priced_new_tokens` | continue | the admission-priced `new_tokens` is absent or malformed |
| `below_deficit_ratio_floor` | continue | condition 1 failed |
| `within_bound` | continue | condition 2 failed |
| `within_eta_budget` | continue | conditions 1-2 passed and the measured ETA fits `budget_s` (the gate's one new token) |
| `deficit_over_bound` | **abort** | all three conditions passed — `reason=measured_cache_deficit` |
| `decision_error` | continue | fail-open: an unpriceable deficit, an unusable budget (`budget_s=0`), or any internal error |

**Observability.** Both verdicts are reported on the same
`measured_cache_abort` line, so a tolerated deficit is explainable from
`proxy.log` alone:

```text
# continue (benefit gate declined the abort)
measured_cache_abort session=<id> model=<model> detail=within_eta_budget total=<n> cache=<n> \
  measured_new_tokens=<n> priced_new_tokens=<n> deficit_ratio=<r> bound_tokens=<n> \
  measured_eta_s=<s> budget_s=<s> → measured deficit fits the prefill budget, continuing local dispatch

# abort
measured_cache_abort session=<id> model=<model> reason=measured_cache_deficit detail=deficit_over_bound \
  total=<n> cache=<n> measured_new_tokens=<n> priced_new_tokens=<n> deficit_ratio=<r> \
  bound_tokens=<n> measured_eta_s=<s> budget_s=<s> elapsed_s=<s> → cancelling local dispatch early and detouring to the next provider
```

`budget_s` is the comparator **actually enforced** (the safety budget, derived
the same way the guard derives it), never the admission wait budget;
`measured_eta_s` / `budget_s` render
`unknown` when the gate did not price the deficit.

**Rollback.** Two independent, value-only rollbacks: set
`local_measured_cache_abort_enabled: 0` to restore the exact pre-change
behaviour end to end (no `return_progress` injection, no capture, no early
release), or retune the comparator's input — `local_prefill_deadline_seconds`
(and `local_admission_guard_budget_safety_fraction`, which the admission guard
already enforces; `0` compares against the bare deadline). No code change and
no restart-ordering constraint beyond the usual process restart. The keys
themselves are documented in `proxy/config.yaml`.

#### Cold-prefill admission budget (LP-0MVN3KMGI009BZ0D)

A parallel, *measured* cap on what the GPU must actually compute:
`server.local_cold_prefill_budget_tokens` (default in the shipped config
**25 000**; **0 disables** it and is the documented rollback) bounds the **new
(uncached, to-be-prefilled)** tokens admitted for a local dispatch — not the
prompt size, so a cache-warm request stays admissible however large its prompt.
It applies on **every** admission path, including the idle/economic bypass
that admitted 33 oversized cold prompts in the measured 2026-10-24 18:38–23:47
window. Over-budget requests are refused at admission and detour to the next
provider; they are never dispatched and then cancelled at the deadline
(baseline: 125 deadline cancellations and ~779k prefill tokens computed then
discarded in that 5.2 h window).

The cap alone is a bare token count, so it is paired with two
deadline-relative knobs:

- `local_admission_guard_eta_percentile_min_tokens` (shipped **20 000**,
  default **32 768** = the pre-existing hard-coded floor) — the prompt size
  from which conservative p75 percentile pricing engages. Lowering it to
  20 000 makes the p75 pricing cover the whole 25k budget band. If it exceeds
  the budget, percentile pricing would never engage for a budgeted prompt and
  a startup WARNING says the band guard is silently dead.
- `local_admission_guard_budget_safety_fraction` (shipped **0.95**, default
  **0.0** = disabled) — a budgeted prompt is admitted only when its
  conservative p75 ETA fits inside the safety budget
  (`fraction × local_prefill_deadline_seconds`; the canonical derivation is in
  [Measured cache-deficit early abort](#measured-cache-deficit-early-abort-lp-0mvr7in0o000dpxn-benefit-gate-lp-0mvwbd91z004u135) → *benefit gate*).
  A 25k cold prompt prices ~93 s at the band's p75 rate
  (~270 t/s) and is admitted; a band-slow 25k (~132 s at ~190 t/s) is refused.

`local_admission_guard_reference_tokens` (shipped **25 000**, previously
32 000) aligns the published `/llama/local/status` reference prompt with the
budget so there is one number instead of two. Since LP-0MVSD5RJW008F1IB the
reference used once dispatches have been observed is the **median** of the last
`local_admission_guard_reference_window_samples` prompt sizes (shipped **20**;
`0` restores the single-most-recent-dispatch behaviour byte-for-byte) capped by
`local_admission_guard_reference_max_tokens` (`0`/absent = uncapped), so one
huge dispatch cannot pin the published cache-cold bound for a whole window.
`cold_threshold` (derived,
38 000) remains a large-context routing bypass and becomes a non-binding upper
bound once this measured budget exists.

**Rollback** (restores the pre-LP-0MVN3KMGI009BZ0D behaviour with values only —
no code change): `local_prefill_deadline_seconds: 90`,
`local_cold_prefill_budget_tokens: 0` (cap disabled),
`local_admission_guard_eta_percentile_min_tokens: 32768` (the old floor),
`local_admission_guard_budget_safety_fraction: 0.0` (bound disabled) and
`local_admission_guard_reference_tokens: 32000`.

Note: the existing `stream_heartbeat_interval_seconds` (default 10 s) emits
`data: {"type":"heartbeat",…}` frames during prefill **after** the upstream
SSE headers arrive. Frames *before* the headers (the strict prefill window)
need the local dispatch to return a streaming response before the upstream
first byte; that gap is tracked separately.

#### Post-deadline prefill-thrash cooldown (LP-0MVKUKFHH005UDWL, rescoped by LP-0MVT2KSXY002O75C)

A deadline cancellation is *evidence that the prefill cannot complete* — the
root cause of the thrash: the client re-dispatches the identical prompt and the
whole prefill burns again (the incident's 41 local POSTs, 35 inside
02:00-03:00, re-prefilling near-duplicates of the same 28k-124k-token prompts).

**Duration: 120 s** (`server.local_prefill_thrash_cooldown_seconds`; was 600 s
until 2026-10-31, `LP-0MVWC8JCG007GAPW`). The window is armed by both arming
sites below, including the measured-cache early abort, which fires ~0.1 s into
a dispatch — so at 600 s a 0.1 s GPU saving bought a ten-minute *client*
outage, a 6000x asymmetry. It is worst for a **pin-local** client
(`X-Local-Preference: local-preferred`), which by contract receives a retryable
`429 Local admission refused` instead of falling through to a paid provider, so
every retry inside the window is a hard refusal. Measured 2026-10-31: 330
`measured_cache_abort` events across 28 sessions (249 at `deficit_ratio=1.0`),
and 113 refusals for the single session `herdr-1793416360-4155786-6690`; the
post-cooldown retries measured `cache=0` again, so the longer window was not
buying cache warmth — only delay. Two minutes still spans two client retry
intervals, so the retry-storm defence is intact.

120 s is deliberately **interim**: the benefit gate in `LP-0MVWBD91Z004U135`
stops the abort firing for prompts whose measured cold prefill fits the
request's budget, after which this value should be re-tuned upward rather than
left at 120 by default.

**What it refuses.** The window is keyed by **(session, prompt fingerprint)**,
not by session alone. The fingerprint is a 32-hex-char SHA-256 prefix over the
**current turn** of the request's `messages` — everything after the last
assistant message (sorted keys, compact separators); message order is
significant, and a body with no usable `messages` yields no fingerprint.
Turn scoping is what lets the two fingerprinting sites agree on one request
(LP-0MVSAMN14001DMTY): `router.py` arms from the *delta* body
`_handle_session` rewrote, while `provider.py` fingerprints the *original*
client body, and the delta is always a trailing suffix of it. So:

- a re-dispatch of the **failing prompt** is still refused — on every local
  dispatch entry, because the gate sits in `provider.py` *before* the
  concurrency / queue / smart-routing gates, so a refused request never queues
  for a slot it would be forbidden to use;
- a **different prompt** from the same session — usually far smaller — is
  admitted. That is the measured over-reach this fixes: 2026-10-24 20:00 →
  2026-10-25 07:00, 916 `routing_skip_local … reason=prefill_thrash_cooldown`
  skips across 46 sessions, **241 of them for 21 `herdr-*` sessions** (the
  downtime dispatcher starving its own workload), 337 of them where the
  proxy's own nearest status poll within 120 s reported
  `local_in_flight_requests=0`.
- a **fail-open fallback** keeps the gate engaged when a fingerprint cannot be
  derived: an arming site without a fingerprint arms the session-wide
  (empty-fingerprint) window, and a gate without one consults **every** window
  of the session. The narrowing is lost, never the gate.

**When it is released.** Besides expiry and a comparable-size successful first
byte (`local_prefill_thrash_recovery_fraction`, LP-0MVQITN6P008DPLB), the
window is released when the proxy's own evidence proves the GPU free
(`_release_prefill_thrash_cooldown_on_free_gpu`).

The first-byte clear stays wired and stays scoped (same fingerprint, ≥ the
recovery fraction of the armed size), but since the gate now refuses a
cooling-down session *before* any local dispatch, a request can only be the
release trigger for a window it does not match. In practice a live window is
released by expiry or by the free-GPU proof; the table below lists all of
them:

| Release trigger | Condition (all must hold) | Log line |
|---|---|---|
| `free_gpu` | no `active` dispatch record; a **fresh** `/slots` snapshot (`proxy.residency` view `known`); **no** processing slot (an abandoned ghost still processing blocks it) | `prefill_thrash_cooldown_released session=… reason=free_gpu cooldown_remaining=Ns endpoint=…` |
| comparable first byte | a successful local dispatch prefilled ≥ `local_prefill_thrash_recovery_fraction` of the armed size, for the same fingerprint | (cleared silently on the success path) |
| expiry | lazy prune at the next request | — |
| knob 0 | `server.local_prefill_thrash_cooldown_seconds: 0` — disables arming, skipping and release | — |

The `free_gpu` release is what removes the measured free-GPU skips; it is not a
blanket disable (any of a stale snapshot, a live dispatch or a processing slot
keeps the window armed), and it was sequenced after the abandoned-work fix
below, because a ghost still computing legitimately blocks the release.

**Wiring.** Arming: `router.py` on `prefill_deadline_exceeded` and on the
measured-cache-deficit early abort (both log
`prefill_thrash_cooldown_started session=… cooldown_s=…`), with the request's
**original** prompt fingerprint (captured before `_handle_session` rewrites the
body to the delta) and the delta's prompt size.

**Terminal for the request.** The same two sites also mark the *request*
(`_mark_local_dispatch_terminal` — an attribute on the `Request`, so it dies
with it): after a cancellation, local is terminal for that request, which
closes the header-less shape where the window is armed under a generated
session id no gate site can re-derive from the headers. A concurrent request
from the same session carries its own `Request` object and is covered by the
session-wide window instead.

**Gating — at the acquisition choke points, not only the chain.**
`provider.py`'s chain entry logs `routing_skip_local … cooldown_remaining=Ns
reason=prefill_thrash_cooldown → skipping local`, but the in-request local
retry loops re-enter `proxy_to_local` *after* that decision, so the same gate
also sits at every point that can acquire a local dispatch lease
(`_local_dispatch_gate`, consulted by `proxy_to_local` before any local
resource is acquired and by `_try_acquire_local_dispatch` before the record is
created; refusals log `entry=router_dispatch` / `entry=lease_acquire`, with
`reason=prefill_thrash_cooldown` or `reason=post_deadline_terminal`). A refusal
returns the retryable 503 `backend_error` shape — never
`local_lease_active`/`no_slots_available`, which would make provider.py queue
for a slot the gate is refusing to grant.

**Hold bound.** The AC1/AC2 acceptance bound is a **<=300 s (5 min) hold** on
any local dispatch that never delivers a first byte — the epic's per-hold
bound (LP-0MVKN9IRP003XJH6 AC1). The terminal marker and the choke-point gate
hold that bound by construction: a post-deadline cancellation releases its
lease in the same second (`lease_released … reason=prefill_deadline`, 0.0 s),
and the same request can no longer re-acquire a fresh lease, so the
~308 s unproductive hold of the 2026-10-26 incident (311-322 s denials) cannot
recur. Remaining unproductive holds (new cold prefills, deadline evictions of
*other* requests, no-progress watchdog) are each bounded: the watchdog fires
after `local_dispatch_no_progress_timeout_seconds` (180 s) plus the sweep,
well inside the 300 s bound. The measurement gate that verifies this at
>=24 h is `scripts/prefill_recovery_verification.py --measure`
(`--wedge-bound-seconds`, default 300 = the bound), documented in
[docs/dev/prefill-recovery-verification.md](../../docs/dev/prefill-recovery-verification.md).

Clearing: the router's first-byte success path. The store is in-process (`_prefill_thrash_cooldowns`,
keyed `(session, fingerprint)`; a proxy restart resets it with the routing
state it guards) in `proxy/proxy/router_helpers.py`. Unit and routing-level
behaviour are pinned by `proxy/tests/test_prefill_thrash_cooldown.py`; the skip
counts are measured by `scripts/capacity_signal_verification.py`
(`prefill_thrash_cooldown` block — both the strict "fresh poll" attribution and
the relaxed "no live local work" population the 336-skip baseline was derived
with).

#### Per-dispatch live-work attribution (LP-0MVYZEYZY003IQAH)

`local_in_flight_requests`, `local_owner_session_id` and `local_active_query`
describe the **in-flight dispatch** from admission through first byte — the
whole prefill — not just generation. A dispatch is live work while its record
is `active`; a `lease_renewed` retained lease (`active: false`, future
`expires_at`) reserves the slot for the owning session but is **not** live
work. The status handler samples this attribution **after** the `/slots` query
in the same poll, so fresh busy detail cannot be paired with an attribution
snapshot taken before a dispatch admitted during a multi-second poll
(LP-0MVZ3V8UR003KCSZ).

A dispatch record is keyed `(endpoint, session_id)`, so back-to-back turns of
one session share a key. Each acquisition stamps a per-dispatch token on the
record and the owning request; a teardown (`_decrement_local_active_queries`)
or release (`_release_local_dispatch`, `_release_unserved_dispatch`) carrying a
different token is a no-op (LP-0MVZ3V8J1006YGFN). Without that scoping, a late
turn-N teardown cleared turn-N+1's live-work attribution for its whole prefill
— the 2026-11-01/02 residual class (72 polls, 65 of them mis-attributed to
`session_evicted` by the nearest-event lookback).

The cache-served `available_slots` paired with a fresh `saturated` hint is a
**separate** family tracked by LP-0MVXSX0TZ00798KA, not fixed here.

#### Request ↔ llama slot/task attribution (LP-0MW959RRO008AAH9)

A dispatched local request is attributable to the llama-server slot **and task**
it used (parent AC10, merged from LP-0MW0KZUUS002LICZ). Before this, the dispatch
record hard-coded `slot_id: None, llama_task_id: None`, so the eviction evidence
from the 2026-11-03 incident read `slot_id=unknown llama_task_id=unknown` and a
later event (progress, cancellation, eviction) could not be mapped onto a
backend task at all.

**Source of the id — cached state only.** `_query_slots_detail()` now carries the
`id_task` the deployed build publishes for each slot (additively, last in the
entry). Like every other residency input, it is read from
`observability._last_slot_details_cache`: llama-server holds `/slots` for >15 s
while processing a prompt batch, so the association **never** issues an upstream
query (`observability.slot_task_id_from_cache`).

**Where the ids are persisted.** The accepted-dispatch evidence stores the
candidate `slot_id` stamped on the admission snapshot
(`provider._stamp_local_admission_evidence`) and the `llama_task_id` resolved for
it; `_pin_dispatch_slot` resolves both from the ownership registry as a fallback
when no snapshot was threaded. The task id is resolved **again at release**,
because the slot is usually still idle when the record is created (the request
has not reached llama-server yet) while at release it is running this request.
A record created outside the guarded routing path keeps the explicit `None`
markers — only the guarded path claims an association.

**The eviction line.** The `local_eviction_after_acceptance` line is
append-only: `slot_id=` and `llama_task_id=` keep their positions and, when a
**known** slot cannot be attributed, a `llama_task_id_reason=` field is appended
after `endpoint=` so an unavailable id is self-describing instead of a bare
`unknown`. A record whose slot is unknown (a legacy path) keeps the historical
`unknown/unknown` rendering. An *idle* slot keeps the `id_task` of the last task
it ran (measured 2026-10-31, see `proxy/metrics.py`), so its id is never
attributed to a new dispatch — it reports `task_id_not_started` instead,
precisely so a request is never mapped onto the wrong backend task.

| `llama_task_id_reason=` | Emitted when |
|---|---|
| `no_slot` | the record has no slot id to associate against (legacy path; not emitted) |
| `snapshot_unavailable` | no successful `/slots` sample has been taken yet (cache empty) |
| `slot_not_sampled` | the snapshot has no entry for this slot |
| `task_id_unpublished` | the build publishes no `id_task` at all |
| `task_id_not_started` | the slot is not running a task (idle, or the `-1` sentinel) |

The vocabulary is closed (`observability.SLOT_TASK_UNAVAILABLE_REASONS`), and
the association never changes a routing or eviction decision — this is
attribution only. Tests: `proxy/tests/test_dispatch_slot_task_association.py`
(cache → resolver → dispatch record → admission line → eviction line, plus the
refused-admission case, which records no association and no lease).

#### Abandoned local dispatches (LP-0MVRMI5HK00558TB AC1)

An *abandoned* dispatch is one the proxy stops waiting for while llama-server
still has it: it produced no first byte and the proxy gave up on the
deadline, the session, the client or the watchdog. Abandoned work is not live
work, and it must not read as live work either.

Measured before this change (LP-0MVOG7IP80003MLV, 2026-10-24 20:00 →
2026-10-25 07:00): 332 abandonments (`prefill_deadline` 260,
`session_evicted` 60, `no_progress_watchdog` 11, `client_disconnect` 1) left
247 fresh `/slots` polls reporting `capacity_hint=saturated` with
`local_in_flight_requests=0` — ghosts, counted as live.

Two mechanisms, applied per reason:

| Abandonment reason | Mechanism | Log line |
|---|---|---|
| `prefill_deadline` | **Cancel**: the captured upstream stream (`cm`) is closed by `_release_unserved_dispatch` (`LP-0MVT2KSN3007QYFK`); the slot is recorded as abandoned | `lease_released … reason=prefill_deadline`, `abandoned_slot_recorded …` |
| `session_evicted` | **Cancel** the pinned `stream_cm` of each released record, then **annotate** what could not be cancelled | `lease_released … reason=session_evicted`, `abandoned_slot_recorded …` |
| `no_progress_watchdog` | **Cancel** the record's `stream_cm` (existing, `LP-0MVIQ0FJC000YLX0`) + **annotate** | `lease_released … reason=no_progress_watchdog`, `abandoned_slot_recorded …` |
| `client_disconnect` | **Cancel** the record's `stream_cm` (existing, `LP-0MVLFVY6K003VYEL`) + **annotate** | `lease_released … reason=client_disconnect`, `abandoned_slot_recorded …` |
| `orphan_cleanup` / `measured_cache_deficit` | **Cancel** + **annotate** (same shared core); both now carry the dispatch record's pinned slot identity and the upstream stream, so the earliest abort path is attributable too (`LP-0MVUVHAQP001IRKK`) | `lease_released … reason=<that reason>`, `abandoned_slot_recorded … slot_source=…` |

A `measured_cache_deficit` abandonment only exists because the benefit gate
**allowed** the abort: all three conditions held, so the measured cold ETA
exceeds the request's safety budget (the comparator, derived from
`local_admission_guard_budget_safety_fraction`; falling back to the bare
deadline when the safety fraction is `0`) and waiting could not have succeeded.
A deficit the gate declines continues the dispatch and abandons nothing (see
[Measured cache-deficit early abort](#measured-cache-deficit-early-abort-lp-0mvr7in0o000dpxn-benefit-gate-lp-0mvwbd91z004u135)).

The cancel is real on **every** release path, not decoration. llama-server
aborts at its next write to the socket, and `return_progress` guarantees such
writes throughout prefill. The sequence is always
`Connection handling canceled` → `stop: cancel task, id_task = N` →
`slot release: … stop processing`.

Per-path status — what each abandonment does to the backend:

| Release path | Cancels the upstream stream? | Records slot identity? |
|---|---|---|
| `prefill_deadline` | **yes** — the handler passes `upstream_stream_cm=cm` and `abandoned_slot_id` | yes |
| `session_evicted` | **yes** — the pinned `stream_cm` of each released record | yes |
| `no_progress_watchdog` | **yes** — the record's `stream_cm` (`LP-0MVIQ0FJC000YLX0`) | yes |
| `client_disconnect` | **yes** — the record's `stream_cm` (`LP-0MVLFVY6K003VYEL`) | yes |
| `orphan_cleanup` | **yes** — same shared cancel/annotate core | yes |
| `measured_cache_deficit` (Option D) | **yes since `LP-0MVUVHAQP001IRKK`** — the `MeasuredCacheAbortDetourError` arm in `proxy/proxy/router.py` now passes both `upstream_stream_cm=cm` and `abandoned_slot_id=slot_id`; before that change it closed `cm` itself and recorded no slot identity | yes |

The per-path behaviour above is measured, not inferred. The evidence lives in
[abandoned-prefill-release-cancel.md](../../docs/dev/abandoned-prefill-release-cancel.md)
(child `LP-0MVW8S7QY002S7DV`, parent `LP-0MVVEQW5Y00962T2`) and is taken on
the paths that actually produce the releases, not on the deadline path alone:

- **Per-release cancel rate** (canonical 24 h window, 2026-10-29 14:08 →
  2026-10-30 14:08 UTC): **300 of 375 post-acceptance releases (80.0 %)** had
  an observed backend cancel. **66 % of that window's releases are Option D**
  (`measured_cache_deficit`, 248 of 375), so the rate is dominated by the path
  that produces most releases rather than by the deadline.
- **H1 refuted — an abandoned prefill does not run to completion.** Over a
  day-aligned 48 h window **748 of 2 581** backend launches stopped before
  emitting `prompt eval time`, and post-cancel progress is at most 4 batches
  and ≤ 1 746 tokens (one to two `ubatch`es).
- **The stop is bounded.** `cancel → stop processing` p90 **6.585 s**, exactly
  one `--ubatch-size 1024` at the window's measured **155.51 tok/s**. The
  recorded bound is **6.6 s**.

The 2026-10-28 sampler (**42 of 82** `Connection handling canceled` events,
never reaching `prompt processing done`) is retained here as history: it was
taken on the **deadline** path and must not be quoted as a generic
per-release rate. What the logs cannot always supply is *attribution*, not
cancellation: 227 of the 375 releases pair with a bare
`operator(): http client error: Connection handling canceled` line that
carries no task id (`unjoined_reason=cancel_without_task_id`), so the cancel is
observed but the task cannot be named.

Because the backend can lag the close, the abandonment is also **recorded**
(`_record_abandoned_slot`, `proxy/proxy/router_helpers.py`) with a bounded TTL
(`DEFAULT_ABANDONED_SLOT_TTL_SECONDS`, 900 s — longer than the prefill-thrash
cooldown so the annotation outlives the window in which the ghost can still be seen
processing; the cooldown was 600 s when this bound was chosen and is 120 s
since 2026-10-31, `LP-0MVWC8JCG007GAPW`, so the margin only grew), keyed by slot identity so the registry cannot grow past the pool,
and pruned lazily on read. It is published on `/llama/local/status`:
`status_schema_version` 4 introduced the abandonment fields — each slot carrying
`abandoned`,
`abandonment_reason`, `abandoned_session_id`, `abandoned_at_seconds_ago` and
`abandoned_scope`, the top level carrying `local_abandoned_slots`, and
`capacity_hint` gaining the additive value `abandoned_busy` for a pool
occupied *only* by abandoned work. (The payload version is `5` since the
read-only `local_route_*` fields of LP-0MVWFKIKJ005XHAY, see
[the published route fields](#the-published-route-fields-status_schema_version-5);
the abandonment fields are unchanged by that bump.)
`saturated` therefore keeps meaning live, proxy-owned decode work, and a
busy-but-abandoned pool is never reported `idle`. See
[per-slot-status-verification.md](per-slot-status-verification.md) for the
status semantics and the consumer rule (any unrecognised hint value → treat as
busy).

**No abandonment is ever dropped.** The measured residual after the first fix
(`LP-0MVT2KW5A008X037`: 12 polls / 0.11 % of a comparable night still reported
`saturated` with no live work) was an *annotation identity* gap, not a
liveness gap: the proxy knew it had given up, but the recorded identity did not
match the slot llama-server was still draining — and a release that carried no
identity at all (`measured_cache_deficit`) could not name a slot even in
principle. Both shapes are now covered:

- **Identity is resolved before the assignment is freed**, in precedence order:
  the caller's explicit slot id → the dispatch record's `slot_id` pinned at
  admission (`_pin_dispatch_slot`, so it survives the later assignment free) →
  the live session assignment. The provenance is published on the log as
  `slot_source=explicit|assignment|endpoint_wide` on every
  `abandoned_slot_recorded` line.
- **The resolved id is re-verified** against the slot llama-server reports for
  that session; a mismatch logs
  `abandoned_slot_mismatch session=… recorded_slot=… observed_slot=…` and
  records the endpoint-wide entry **in addition to** the exact one.
- **Unknown slot → endpoint-wide entry.** When no slot identity can be
  resolved, the entry is stored under
  `ABANDONED_SLOT_ENDPOINT_WIDE` (`slot_id=None`, `scope="endpoint"`) instead
  of being discarded. The sentinel is a negative `int` because llama-server slot
  ids are non-negative, so it can never collide with a real slot; there is at
  most one such entry per endpoint, so growth stays bounded.

**Why a draining-only pool never reads `saturated`.** The status builder
(`annotate_slots_for_status`) matches the slot's own entry first
(`abandoned_scope="slot"`), then falls back to the endpoint-wide entry
(`abandoned_scope="endpoint"`) — but **only** for a slot that is
`is_processing` with `owner_session_id: null`, i.e. neither a dispatch lease
nor a live dispatch record owns it. That guard is what keeps the truthful-status
contract intact: an endpoint-wide entry can only ever describe a *draining
ghost*, so a genuinely live, proxy-owned slot still reports `saturated` and
`capacity_hint` rule 0 (`abandoned_busy` only when **no** generating slot is
live) is unchanged in intent. This is why the pool's hint is `abandoned_busy`
rather than `saturated` while llama-server finishes a task the proxy walked
away from.

The entry is dropped as soon as the backend **disproves** it: a **fresh**
`/slots` snapshot reporting the slot not processing clears that slot's entries,
and a fresh snapshot with nothing processing clears the endpoint-wide entry. A
stale snapshot never clears anything (cached `is_processing: false` is a fossil,
not proof of release) and a snapshot with no slots at all changes nothing.
Clearing and annotating are read-only status decoration: no lease is released,
no counter is decremented, no capacity is acquired.

**Abandoned work is still not dispatched behind.** The annotation changes what
the status *says*, never what the gate *admits*: a slot llama-server is still
draining has no free capacity, so `available_slots` and the admission gate are
untouched, and `abandoned_busy` is a **non-idle** hint — a consumer that admits
only on `idle` (herdr's downtime worker, for instance) does not dispatch behind
a ghost. Folding abandoned work into `idle` was rejected for exactly that
reason: it would start a second prefill behind a draining one and lengthen the
tail.

**Measurement.** `scripts/capacity_signal_verification.py` attributes a
violating poll to a preceding `lease_released` event, and its
`_ABANDON_REASONS` set now includes `measured_cache_deficit` (imported from
`proxy.measured_cache_abort`): that release gives up on a dispatch while
llama-server keeps working on it, so leaving it out made 5 of the 12 residual
AC4 polls unattributable. Widening the set changes the historical baseline —
the published 247 polls / 2.12 % pre-fix and 12 polls / 0.11 % post-fix counts
were computed with the narrower four-reason set — and every tool report states
that discontinuity rather than silently re-basing the comparison
(`LP-0MVUVHBD40067FE3`).

Interaction with the cooldown above: the annotation is what makes the *skip*
decision honest (a ghost no longer reads as live work), while the ghost itself
keeps the *free-GPU release* from firing — the GPU really is busy until
llama-server finishes the cancelled task.

#### Release, not a short TTL, for the deficit reason (LP-0MVWFKF4M009AEJ9)

A deficit-aborted slot is released by the **observed-free** path, not by a
reason-scoped shortened TTL. The planned 60 s TTL for `measured_cache_deficit`
was dropped once the release-on-observed-free mechanism landed for every reason
(LP-0MVUVHB1V006HA5I): an exact release cannot re-open the draining-ghost gap
(LP-0MVUTYHG7001S6BM), whereas any fixed short window re-opens it whenever a
drain outlasts the window. What a consumer therefore sees for this reason is:

- immediately after the abort, the slot stays **annotated** —
  `abandoned_busy`, `abandonment_reason=measured_cache_deficit`,
  `abandoned_scope="slot"` (or `"endpoint"` when no identity could be
  resolved) — because llama-server is still computing the cancelled prefill;
- on the first **fresh** (`stale=false`) `/slots` snapshot that reports the
  slot not processing, the entry is dropped and `local_abandoned_slots` /
  `capacity_hint` reflect real capacity on that same poll;
- a **stale** snapshot never releases anything (a cached
  `is_processing=false` is a fossil, not proof), so a long drain keeps the
  annotation until the backend actually disproves it.

`DEFAULT_ABANDONED_SLOT_TTL_SECONDS` (900 s) is only the registry bound that
keeps a forgotten entry from growing the registry past the pool; it is not the
release trigger, and no per-reason TTL exists. Pinned end to end by
`TestDeficitAbortMarkerRelease` in
`proxy/tests/test_abandoned_local_dispatch.py`.

#### Regression coverage for the prefill-thrash wedge

`proxy/tests/test_prefill_thrash_wedge.py` reproduces the 2026-10-23 02:00-
03:00 incident (LP-0MVKN9IRP003XJH6): a session whose prefill never produces a
first byte hits `prefill_deadline_exceeded`, detours, and — pre-fix — the
detour only marks the dispatch record inactive with a future `expires_at`
(`lease_renewed timeout=…s`) while the `local_prefill_in_flight` marker
survives. The same session re-acquires immediately (`_extend_lease_during_prefill`
refreshes `last_progress_ts` from the in-flight marker, so the 180 s
no-progress watchdog never fires), and for anonymous sessions the leaked
marker saturates the prefill cap so every other session is denied
(`local_dispatch_denied` / `contention_queue_fallback_after_queue
reason=local_lease_active`).

The module drives the real routing helpers (`_try_acquire_local_dispatch`,
`_await_first_byte_with_prefill_monitor`, `_extend_lease_during_prefill`,
`_release_prefill_deadline_dispatch`, `_cleanup_after_request`,
`_cleanup_stale_local_dispatch`) with a stubbed stale-`/slots` llama-server
and an injected monotonic clock, so the whole 5-minute simulated window runs
in well under a second. The harness mirrors the post-fix router sequence
(deadline release → cooldown → cleanup) and its contract assertions — lease
and in-flight marker released on the detour, at most one local dispatch
inside the window — are green: LP-0MVKUKFHH005UDWL added the post-deadline
thrash cooldown and LP-0MVKUKFSO008GI6O added the truthful lease/marker
accounting.

The 2026-10-26 10:39-10:47 shape (LP-0MVSAM2AV0077UAE deterministic tests,
LP-0MVSAMN14001DMTY fix) is a different failure mode on the same gate: the
lease really is released, yet the *same request* re-dispatches local ~150 ms
later and arms a fresh lease, and the armed window never yields a skip line
because the retry loops bypass the chain gate. The same module pins the fix:
the choke-point gate refuses the re-acquisition, the request-scoped marker
makes the deadline detour terminal for local inside that request, the
turn-scoped fingerprint makes the arm and check keys agree, and the release
clears the record and marker before awaiting the upstream stream close, so
`_extend_lease_during_prefill … in_flight=True` can never resume for a
dispatch that never produced a byte.

#### Lease and in-flight truthfulness (LP-0MVKUKFSO008GI6O)

Three accounting rules keep the local gate honest, so one wedged session
cannot hold the pool:

1. **An open request is not progress.** `_extend_lease_during_prefill` pushes
   `expires_at` out on the in-flight signal but does **not** refresh
   `last_progress_ts`. Only observed numeric progress or observed slot
   liveness (`is_processing`, LP-0MUXQ5QL40027VZ4) advances the no-progress
   clock, so a genuinely progress-less in-flight record is still released by
   `local_dispatch_no_progress_timeout_seconds` (180 s) instead of surviving
   to the 900 s ceiling. The incident's
   `lease_extended_during_prefill progress=None alive=False in_flight=True`
   loop refreshed the clock every 10 s and defeated the watchdog.
2. **The in-flight signal is own-session only.** The trigger is accepted only
   when `local_prefill_in_flight` holds an entry for the extending record's
   own session, and the record is addressed by
   `_dispatch_lease_key(endpoint, session_id)` — another session's open
   request can never refresh this record's `expires_at`/`last_progress_ts`.
3. **Occupancy counts only live leases.** The admission gate's
   `in_flight_prefills` (and therefore `busy_slots`) uses
   `_get_live_prefill_in_flight_count`, which counts an in-flight marker only
   when a live (active/unexpired) dispatch record backs it. An orphan marker
   left by a released/deleted record no longer reports
   `busy_slots=1 … available_slots=0` for every other session (the incident's
   52 admission-guard rejections in the hour).

Coverage: `proxy/tests/test_lease_truthfulness.py` (release on deadline,
two-session extension isolation, orphan-marker occupancy, watchdog firing for
progress-less in-flight records).

When a routed stream fails, the proxy follows a **recovery-first** strategy
with an **informative-error fallback** (from the Aug 3 error analysis,
LP-0MSDFKCK4007CPMY; see `proxy/docs/error-analysis-2026-08-03.md`):

1. **Recover first** — pre-content mid-stream failures (empty response,
   connect failure, pre-content stall) re-route to the next healthy provider
   in the configured chain, bounded by the remaining providers and the
   content-delivered boundary (no re-route once any content chunk has been
   forwarded). Failed providers enter the existing cooldown / stall circuit
   breaker. The pre-flight re-route is implemented in
   `proxy/proxy/provider.py` (`_preflight_streaming_response`,
   `StreamingPreContentError`); it is gated on a remaining provider being
   available so the last provider in a chain still surfaces its terminal
   error to the client. Tracked in **LP-0MSETOTWY000SU0Z** (implements
   LP-0MSDRRDWK009QT4E).
2. **Informative error fallback** — when recovery is impossible (all
   providers failed/in cooldown, or content already delivered), the
   synthetic `finish_reason: error` SSE event carries a structured
   `error` payload (`type`, `message`, `provider`, `model`, `entry`,
   `suggested_action`) so the client/operator can act. Backward compatible:
   `finish_reason: error` semantics unchanged, no client-side change.
   All eight emission sites use the shared helper
   `_build_stream_error_event` / `_stream_error_event_bytes` in
   `proxy/proxy/proxy_remote.py`. Tracked in **LP-0MSETOTWY000SU0Z** (implements
   LP-0MSDRRJPF0052STT).

Emission sites and the single-helper enrichment point are documented in
`proxy/docs/sse-error-emission-audit.md`.

## Local cold-start immunity (LP-0MV9UNIOZ009RIAN)

After a proxy restart, requests that race llama-server warm-up receive
HTTP 503 (or connection errors) and previously entered the standard
per-failure 60s provider cooldown — the dominant source of post-restart
local quarantines (~12 of 45 cooldown episodes in one analysed 24h window).

**Exemption, strictly scoped:** while the cold-start grace window is active
(`cold_start.is_cold()` is True — the single immunity-boundary signal), a
**local** HTTP 503 or connection error is **deferred** with the
`local_cold_start_retry_after_seconds` Retry-After hint instead of being
quarantined: no `provider_cooldown_seconds` cooldown is applied, the
exponential-backoff failure counter is not bumped, and the next fallback/
chain cycle can try local again immediately. The deferral is recorded as a
`cold_start_deferral` attempt entry for observability.

**Genuine faults are never masked:** any other 5xx during the window (e.g.
500/502), and any HTTP 503 or connection error after the window expires,
still quenches normally. The window is re-armed per restart by
`cold_start.note_model_loaded()`; router-level concurrent-dispatch gating
(`local_cold_start_max_concurrent`) is unchanged.

**Reason-scoped sibling: the measured-cache-deficit detour
(LP-0MVWFKESK0005G61).** The measured cache-deficit early abort (see
[Measured cache-deficit early abort](#measured-cache-deficit-early-abort-lp-0mvr7in0o000dpxn-benefit-gate-lp-0mvwbd91z004u135))
uses the same deferral shape for a different reason. Its 503 means the router
released *that dispatch* because the measured prefix deficit proved the prompt
could not be served inside its own prefill budget — a verdict on the prompt,
which says nothing about local's health. The 503 therefore carries the
`measured_cache_deficit` marker and is deferred the same way: recorded as
`local_dispatch_deferral`, no `provider_cooldown_seconds` entry cooldown, no
failure-counter bump, and the response's own `Retry-After` returned as the
hint. The cold-start branch keeps precedence; an unmarked local 503 outside the
window is still quarantined. The aborting prompt remains refused by the
per-session/per-prompt `local_prefill_thrash_cooldown_seconds` window (other
sessions and other prompts are unaffected), so the quarantine removed here is
only the entry-wide one that starved unrelated sessions. The rollback is the
detour's own feature flag (`local_measured_cache_abort_enabled: 0`), not a
change to any knob in this section.

**Reason-scoped sibling: the post-deadline thrash-gate refusals
(LP-0MVXSSYTN001TBW5).** The router's post-deadline local gate (see the
`routing_skip_local` exit rows above) returns the same retryable `503` for two
refusals, and both take this deferral shape for the same reason the deficit
detour does — they are verdicts on *one request/prompt*, never on local's
health:

| Gate refusal | Marker / `X-Local-Fallback-Reason` | Meaning |
|---|---|---|
| `post_deadline_terminal` | `post_deadline_terminal` | This very request already had a local dispatch cancelled post-deadline; local is terminal for it. Request-scoped, so a concurrent request from the same session is unaffected |
| `prefill_thrash_cooldown` | `prefill_thrash_cooldown` | The session's post-deadline window is armed for this prompt (`local_prefill_thrash_cooldown_seconds`) |

The refusal `503` carries the reason twice — as `X-Local-Dispatch-Reason` on the
wire and as an in-process attribute on the response — so the fallback loop
defers that one request instead of calling
`mark_provider_unavailable("<local>", provider_cooldown_seconds)`. The attempt
record reads `status=local_dispatch_deferral` with the gate reason, the
`Fallback triggered` line names `reason=post_deadline_terminal` /
`reason=prefill_thrash_cooldown` (never the generic `reason=HTTP 503`), no
failure counter is bumped, and the refusal's own `Retry-After` is returned as
the hint. `local_route_starved` therefore sees no arm from these refusals.

Measured why this matters (window 2026-10-31 23:56 → 2026-11-01 12:12, build
`5d8318e3`): `1 193 of 1 369 (87.1 %)` `local-qwen3 in cooldown` skips belonged
to bursts whose nearest antecedent was this unmarked gate `503`, and 23 of 41
cooldown bursts started 0.1–8 s after a `post_deadline_terminal` refusal — one
neighbour's aborted prompt became a 60 s cross-session outage of the local
route. The refusing request still detours to the next provider inside the same
request (unchanged), and genuine local faults (an unmarked local `5xx`, a
post-grace connection error, slot exhaustion) keep their documented cooldowns:
the exemption is reason-scoped, never "any `503` from local".

## Usage-limit reset quarantine (GoUsageLimitError, insufficient_quota)

HTTP 429 responses whose error type is `GoUsageLimitError` (a usage-limit
variant distinct from `FreeUsageLimitError`), or which carry the
`error.type == "insufficient_quota"` / `error.code ==
"free_daily_limit_exceeded"` markers MunitoAI sends for its daily free limit,
are treated as usage-limit **reset** events rather than generic rate-limit
events. The markers are read **case-insensitively from the top level and from
a nested `error` dict** (gateways disagree about the nesting, and the code
alone is enough for payloads that omit `type`). The quarantine is
**account-scoped** (`_usage_limit_account_key` = `api_key_env@endpoint:model`):
distinct API keys on the same gateway keep independent limits
(LP-0MSMBWB23009XYPW), and the 2-minute `_USAGE_LIMIT_RESET_MARGIN_SECONDS`
safety margin is always added.

The reset duration is resolved in this order, **per quota window**
(LP-0MVPAE77Z002VGIA; supersedes the fixed monthly reading of
LP-0MVF7MPZN009GOJB):

1. **Authoritative quota snapshot, window-aware** — for an `opencode-*` account
   with a **fresh** quota snapshot and quota behaviour strictly enabled, the
   window the 429 names in `metadata.limitName`
   (`rolling`/`weekly`/`monthly`, normalised case/whitespace) determines the
   quarantine:
   - a known window with a positive reset countdown quarantines for **that
     window's** `seconds_to_reset` + the margin (`source=snapshot`,
     `window=<name>`); a weekly-limited account therefore returns within the
     week, never waiting for the monthly reset;
   - a known window whose snapshot reset is non-positive (contradictory, stale
     evidence) is bounded by the named period capped at
     `_usage_limit_guess_cap_seconds` and marked *guessed*
     (`source=snapshot-bounded`), so the self-healing probe revalidates it;
   - an absent/empty/unknown `limitName` (e.g. `daily`) is bounded by the
     **soonest usable window** reset and marked *guessed* — never the monthly
     value while a shorter window is still open;
   - when no window yields a usable reset, the account-level
     `seconds_to_reset` (monthly) path is retained unchanged.
   The chosen window is persisted with the quarantine
   (`usage_reset_window` in `provider-state.json`) and revalidated on the poll
   cadence: as soon as a fresh `/v1/usage` poll shows that window reset
   (effectively empty, or its countdown extended past the quarantine), the
   quarantine is cleared and the account returns to rotation without a proxy
   restart. Legacy entries with no window metadata (schema version 1) clear on
   the first healthy poll that shows no window exhausted.
2. **Message duration** — for a non-`opencode-*` entry (no quota source), or
   when the snapshot is stale/absent or the quota source is disabled
   (`quota_surge_enabled: false`) / dry-run, an explicit `Resets in 22hr
   43min` duration in the message is honoured in full (`source=message`).
3. **Absolute reset instant** — an absolute instant such as `It resets at
   2026-10-30 00:00 UTC` (the observed MunitoAI free-tier daily-limit 429) is
   parsed into seconds-until-reset and honoured as explicit evidence
   (`source=timestamp`). Supported shapes are `YYYY-MM-DD HH:MM UTC`,
   `YYYY-MM-DD HH:MM:SS UTC`, ISO-8601 with `Z`, and ISO-8601 with an explicit
   `±HH:MM`/`±HHMM` offset; a bare timestamp with **no** timezone evidence is
   treated as unparseable rather than guessed in local time. The resulting
   quarantine is **bounded to the period the payload names**
   (`free_daily_limit_exceeded` / "daily … limit" → 24 h) plus the margin, so
   a malformed, far-future or clock-skewed timestamp cannot lock the account
   out indefinitely (LP-0MVT8SAZE00570J5 AC4).
4. **Guessed period** — when there is no explicit duration or usable
   timestamp, `metadata.limitName` / the payload's period wording
   (daily/weekly/monthly) supplies a guessed period capped at
   `_usage_limit_guess_cap_seconds` (24 h by default), logged as
   `source=guess`. This guessed branch is **fallback-only**: it is unreachable
   for a fresh-snapshot `opencode-*` account, and the full 30-day period is
   never used without explicit evidence (LP-0MUQTCMW2001VXH8). A past or
   unparseable absolute timestamp falls back here; an `insufficient_quota`
   body with no reset information at all is quarantined *guessed* for the cap
   (self-healing probe, never persisted) instead of being re-probed every
   ~60 s.

Because this check runs **before** `_handle_http_error_with_cooldown()` /
`_apply_ratelimit_account_quarantine()`, an advertised daily reset always wins
over a rolling `x-ratelimit-*` header window (LP-0MVT8SAZE00570J5 AC3).

Routing decisions during the block log `usage_limit_reset_pending` with the
reset time and the window that holds the account out (e.g.
`window=weekly`); once the reset time arrives (or a fresh poll proves the
window reset) the account becomes eligible again without operator
intervention.
`FreeUsageLimitError` responses without a reset time get the default 3-hour
cooldown (10800s; per-provider overrides were retired with
`opencode-big-pickle` in LP-0MT652JRM004ZLSI). Tracked in
**LP-0MSLJPOCC0001ROJ**; the snapshot-based retirement of the guess is
**LP-0MVF7MPZN009GOJB**.

## Transient overload vs balance exhaustion (LP-0MW46PBWL008T317)

Once a `use_balance: true` account may dispatch on credit, two reactive paths
— both invisible to the proactive `/v1/usage` gate, which has no balance
field — must be told apart when the upstream answers:

- **Transient overload** — an HTTP 429 whose body carries the `[overloaded]`
  marker (observed `429 ... [overloaded] the server is overloaded; retry
  shortly`) with `Retry-After: 2` / `retry-after-ms: 2000` (live sampling
  alternated 3 × 429 then 2 × 200). Classified by `_is_overloaded_error()`
  **distinctly from every quota / usage-limit classifier**, before
  `_handle_http_error_with_cooldown()`.

  Behaviour (AC1/AC2/AC5/AC10):

  1. The entry gets a **short retryable cooldown** honouring the upstream's
     own wait — `Retry-After` (or the `retry-after-ms` millisecond variant),
     floored at 1 s and capped by `server.overloaded_cooldown_max_seconds`
     (default 30 s) — so a wild header can never park a transiently
     overloaded entry for a long cooldown. Logged at INFO with
     `reason=overloaded`, attempt `status=overloaded`.
  2. **No failure-domain exclusion, no account quarantine, no
     sibling-fallback streak**: the entry's failure domain
     (`https://opencode.ai/zen/go:deepseek-v4-flash`) is never poisoned, so
     sibling accounts on the same gateway stay eligible (AC1/AC10). The
     cooldown is deliberately not exponential — repeated overloads stay
     short.
  3. The **next chain-hold cycle re-tries the same entry** once the bounded
     cooldown elapses — a 429-then-200 sequence succeeds with a ~2 s hold
     (`min(chain_hold_seconds, retry_after)`). A chain whose first entry
     keeps overloading falls through to the next provider after the bounded
     holds instead of exhausting or hanging, and the exhaustion response
     still carries a truthful `retry_after` (the remaining cooldown).

  The standard `Retry-After` / `retry-after-ms` headers are honoured by this
  path directly; `respect_ratelimit_headers` (the opt-in for the custom
  `x-ratelimit-*` headers) is **not** required for them.

- **Genuine balance exhaustion** — the prepaid credit / usage is spent. The
  reactive proof is the upstream error itself: an HTTP 402 "Payment
  Required", or a usage-limit body (`insufficient_quota` /
  `free_daily_limit_exceeded` / `GoUsageLimitError` /
  `FreeUsageLimitError` / `usage_limit_reached`) on a **non-429** status
  (429s keep routing through the quota / usage-limit reset path above).
  Classified by `_is_balance_exhaustion_error()` and quarantined via
  `_balance_exhaustion_reset_info()` (AC3/AC6):

  1. The failing **API-key account** is quarantined with the **reset-aware
     usage-limit quarantine** — a body `Resets in ...` duration or absolute
     reset instant, else the account's real `/v1/usage` snapshot reset, else
     the bounded guess — so the entry is **not retried in a loop**.
  2. The quarantine is **cumulative**: `_set_usage_limit_quarantine()` only
     ever extends — a shorter balance signal can never clear or shorten a
     longer usage-limit quarantine already recorded (AC4).
  3. Keyed on `api_key_env@endpoint:model` (account-scoped), so paid
     siblings on the same gateway stay eligible. Logged at INFO with
     `reason=balance_exhaustion`, attempt `status=balance_exhaustion` (a 402
     that also carries a resolvable usage-limit body is logged by the
     usage-limit reset path instead — same quarantine, same real reset
     timing).

## Sibling-fallback circuit breaker (LP-0MTPMF03P0046MFG)

> **Scope:** applies to **remote providers only** (LP-0MUNTOGLB005LUKY).
> Local providers are explicitly exempt: a local empty response never
> increments the streak and never triggers the 600s quarantine. The extended
> cooldown exists to stop retrying remote sibling API keys against a dead
> gateway; a local entry has no sibling keys, and quarantining it removes the
> only healthy local capacity from the chain. Local providers keep their
> existing normal handling (the `empty_response_max_cooldown_seconds` cap of
> 10s, `local_slot_exhaustion_retry_attempts`, the no-progress watchdog and
> the dispatch lease).

A remote provider that repeatedly returns an **empty response** or **stalls**
(zero usable content) across retry cycles is quarantined for an **extended
cooldown** so the fallback chain permanently skips it and routes to the next
healthy sibling — instead of re-attempting the same dead provider dozens of
times per request/client retry.

**Motivating incident (2026-09-06 RCA):** session
`herdr-1788651245-1996040-22961` retried `opencode-go/muse-spark-1.2-contributor`
~80 times in 32 minutes, each failing with `empty_response` (upstream closed
without delivering content). The short 60s Tier-2 provider cooldown expired
between client retries, so every new request POSTed the dead gateway again.
`empty_response` events were never counted by the Tier-3 stall circuit breaker
(which only tracks idle-timeout stalls), so nothing ever escalated to a long
quarantine.

**Behaviour:**

1. Every `empty_response` / `stall_*` / zero-content stream error on a
   **remote** provider entry increments a per-entry consecutive-failure
   streak (`_sibling_failure_count` in `proxy/proxy/provider.py`), shared
   across requests (module-level state, like the other cooldown mechanisms).
   `_record_sibling_failure()` takes an optional `provider_type` argument
   (`None` defaults to remote, preserving existing callers); `"local"`
   returns early without marking the provider unavailable or incrementing
   the streak.
2. Once the streak reaches `sibling_fallback_threshold` (default 2) within
   `sibling_fallback_window_seconds` (default 600s), the provider is marked
   unavailable for `sibling_fallback_cooldown_seconds` (default 600s) — via
   the existing `mark_provider_unavailable()` — and a
   `Sibling-fallback circuit breaker triggered: ...` warning is logged.
   Both the provider ENTRY and its shared BRAND are quarantined, so
   same-gateway API-key siblings (`opencode-go-2` / `opencode-go-3`) are also
   skipped (mirrors the Tier-3 stall breaker's brand-level quarantine and the
   `_entry_cooldown_key` brand check).
3. On success the streak resets (`_reset_sibling_failure_count`); stale
   streaks age out of the sliding window.
4. 429/402 rate-limit responses are **not** counted (they use the existing
   per-model / usage-limit cooldowns).

**Recording points** (`proxy/proxy/provider.py`): the three non-streaming
empty-body sites and the four pre-content streaming re-route catches
(`StreamingPreContentError` / `StreamingRecoverableAfterReasoningError` in
both `_proxy_with_remote_fallback_cycle` and `_proxy_with_fallback_cycle`) —
the same places the short Tier-2 cooldown is applied. Streaming failures are
counted when a sibling is available to re-route to (the pre-flight consumed
the stream and raised); once a sibling becomes eligible again the streak
accumulates and the dead provider is skipped.

Config reference (server section):

```yaml
server:
  sibling_fallback_threshold: 2        # consecutive failures before quarantine
  sibling_fallback_cooldown_seconds: 600  # extended cooldown applied on trip
  sibling_fallback_window_seconds: 600    # sliding window for the streak
```

See `proxy/tests/test_provider_fallback.py` (`TestSiblingFallbackCore` and the
`sibling_*` async tests) for coverage of thresholding, extended cooldown,
window expiry, brand quarantine, 429 non-counting, and direct sibling routing
after the threshold trips.

## Planned-unavailable chain: scheduled gaps are deferred and reported (LP-0MVWC9CZH004OZYN)

Some gaps in a model's provider chain are **scheduled**, not failures: an
`available_times` window is closed until a known UTC edge, a usage-limit
quarantine has a known reset instant, or a quota window rolls over at a known
time. Before this change the client received the opaque
`503 All providers exhausted` and a cap-derived `Retry-After`, so an
unattended worker retried every minute for hours even though the honest answer
was "the chain reopens at 10:00 UTC".

When the *whole* residual chain is blocked by those known-until gates, the
proxy now returns a distinguishable **planned-unavailable** response instead.

### The response

```json
{
  "error": "All providers exhausted",
  "code": "no_provider_available_in_window",
  "retryable": true,
  "retry_after": 3600,
  "retry_at": "2026-10-31T10:00:00+00:00",
  "retry_after_truncated": true,
  "detail": "no provider is available now: the earliest reopening provider is opencode-go (blocked by outside_available_window) at 2026-10-31T10:00:00+00:00"
}
```

- `code` is the stable discriminator (`EXHAUSTION_CODE_TIME_WINDOW_BLOCKED`).
- `error` keeps the historical `All providers exhausted` prose, so older
  clients that string-match it are unaffected (additive fields only).
- `Retry-After` (header) always equals the body `retry_after`.
- `detail` names the earliest reopening provider and the gate that closed it.

### The known-until vs transient split

Classification reads the **chain-resolution skip record** (`{provider:
skip_reason}` on the request), *not* the per-attempt status vocabulary — the
two differ (`quota_hard_stop` vs `quota_exhausted`,
`usage_limit_reset_pending` vs `usage_limit_reset`), and classifying on the
wrong one silently disables the verdict.

| Class | Skip reasons |
|---|---|
| **Known-until** (a wait will reopen it) | `outside_available_window`, `usage_limit_reset_pending`, `quota_exhausted`, `quota_hard_stop` |
| **Derived** (neutral: does not veto the verdict) | `failure_domain`, `usage_limit_account` |
| **Transient** (a single one keeps today's behaviour) | `provider_cooldown`, budget skips, and anything unrecognised |

The closed sets live in `DETERMINISTIC_CHAIN_SKIP_REASONS` /
`DERIVED_CHAIN_SKIP_REASONS` in `proxy/proxy/provider.py`. A **new** skip
reason defaults to transient — it must be added to the set deliberately.

> **Balance-opted-in entries never appear here.** With `use_balance: true` a
> plan-exhausted entry whose rolling/weekly windows still hold headroom is
> **not** skipped at all — it dispatches onto prepaid balance (see
> [Prepaid-credit balance opt-in](#prepaid-credit-balance-opt-in-use_balance-lp-0mw46pau4006sk7z))
> — so it produces no skip record and can never make a chain known-until.

A terminal request-scoped refusal (a non-retryable HTTP 4xx, i.e. anything in
400–499 except 408/425/429) does not defeat the verdict: no wait can change a
prompt the upstream will always reject (`_attempt_is_terminal_request_scoped`).
A single transient attempt (connection error, 5xx, 429, timeout, cooldown,
budget skip) keeps today's generic exhaustion behaviour byte-identically.

### The earliest-instant rule (vs the chain-hold's maximum)

`retry_at` is the **earliest** instant *any* chain entry becomes eligible —
the minimum over window edges and known resets — because that is what a client
needs. It is deliberately *not* the maximum: the maximum is dominated by the
slowest blocker (measured 2026-11-03: one entry ~15 h out, another ~14 days),
and advertising it would park the client far longer than capacity needs.

The chain-hold wrapper keeps its own conservative semantics: it bounds a hold
by the real wait and refuses to hold a provably dead chain at all, so the two
numbers are intentionally kept separate (`_earliest_deterministic_reset_seconds`
for the client; `_bounded_hold_seconds` for the wrapper).

### The truncation flag (no silent cap)

`server.max_retry_after_seconds` still caps the advertised delay so no client
is told to sleep for days. When the true wait exceeds the cap the body carries
the **untruncated** `retry_at` **and** `retry_after_truncated: true`, so a
client can choose to sleep through the whole window instead of polling every
hour. The cap is never silent.

### Pinned `local-preferred` clients

A pinned request never consults the remote tiers for a *service* decision, but
its refusal must still be honest about *when* the chain can serve. When the
local route is refused and the remote slice of the chain is deterministically
known-until blocked, the pinned client receives the same
`no_provider_available_in_window` response (with the earliest remote reopening
instant) rather than a bare `429 Local admission refused` carrying an unrelated
local-capacity timer.

### Fail-open

Any unparseable `available_times` window, missing reset instant, unknown gate
status, or internal exception falls back to today's generic exhausted
response. A provider with no computable reopening instant disqualifies the
classification; the proxy never fabricates a `retry_at`.

### Observability

One `chain_planned_unavailable` line per request (never per entry) names the
aggregate instant and every blocked entry with its gate and reopening instant:

```
chain_planned_unavailable model=plan retry_at=2026-10-31T10:00:00Z retry_after=6240 providers=4 deterministic_skips=opencode-go:outside_available_window,... blocked=[opencode-go:outside_available_window@2026-10-31T10:00:00Z, ...] terminal=none
```

The `proxy-usage-analysis` report classifies this line distinctly from
`Chain exhausted ... holding` ("Chain-end outcomes"), and never counts a
fail-fast as hold latency.

### Operator decision: defer and report, never paper over the gap

Adding an always-on remote provider to cover the scheduled gaps was
**explicitly rejected** (operator decision 2026-10-31). During the scheduled
windows some requests *must* be deferred; this response makes that deferral
honest and actionable rather than appearing as a failure. Do not re-time a
provider's `available_times` to hide a gap either — the gap is the design.

## Per-prompt-fingerprint backoff for deterministic local 400s (LP-0MW95BCVV004LO0K)

A local 400 is a request-shape incompatibility, so it falls back to the next
provider *without* cooling the local entry down. That is right for a one-off,
but wrong for a history llama-server rejects every time (the compacted-pointer
shape from LP-0MTXEBQ4E001BMI2): the client re-sends a byte-identical
conversation, gets the same 400, and the proxy routes remote again — forever,
with no state that says "this exact history is provably rejected".

`proxy/fingerprint_backoff.py` supplies the missing identity and throttle:

- **Fingerprint.** `compute_prompt_fingerprint(messages)` is a SHA-256 hash
  chain over the canonical message array with the per-send fields
  (`timestamp`, `request_id`, `id`) stripped at every depth, so a resend of
  the same history fingerprints identically across processes and sessions.
- **Backoff keyed by the fingerprint, not the session.** Two sessions looping
  on the same bad conversation share one verdict
  (`{fingerprint, session_id, retry_after, resend_count}`); the ladder is
  1 min → 5 min → 15 min, capped at 15 min.
- **Reset rules.** A successful **local** stream clears the verdict for the
  history it served; a session that sends a *different* history drops its
  previous verdict and starts at the bottom of the ladder. A remote success
  deliberately does **not** reset: the wedged loop is local 400 → remote
  success → local 400, so counting the remote success as recovery would stop
  the ladder advancing.
- **Hot path.** The fingerprint is cached per session as a hash chain: a
  byte-identical resend hashes nothing, an appended turn hashes only the new
  messages, and a rewritten history (compaction) is re-hashed from scratch.
- **In-memory and bounded.** Both the verdict table and the session cache are
  capped and pruned, like the sibling cooldown and streak state. A proxy
  restart loses them — acceptable, because a restart also loses the wedged
  client's in-flight request.

Wiring in `provider.py`:

- `_observe_local_http_error_400` records the verdict on every local 400 and
  **extends** the pre-existing `_local_http_400_streaks` entry with
  `fingerprint`, `backoff_resend_count`, `backoff_retry_after` and
  `backoff_until` (the per-(session, provider) streak semantics are unchanged,
  LP-0MTXEBQ4E001BMI2). The deterministic-repeat WARNING carries the
  fingerprint prefix, the resend count and the current `retry_after`.
- `_reset_local_http_400_streak` clears the verdict on the same two local
  success paths that reset the streak.
- `fingerprint_backoff_state(session_id, body_json)` is the read side the
  typed-error layer (LP-0MW95BDJQ002DSVI) and the terminal breaker
  (LP-0MW95BE7X00574OL) consume: it reports `{fingerprint, entry, expired,
  remaining_seconds}` (or `None` when the request carries no history), and it
  reports an expired entry too, because the resend that follows continues the
  ladder rather than restarting it.

Bodies without a `messages` array record no verdict: there is no conversation
to fingerprint.

## Typed deterministic client error before fallback (LP-0MW95BDJQ002DSVI)

Backoff alone still leaves the wedged session waiting on a remote fallback for
a request the local model rejects *every* time. The typed error closes that
loop: when the same history is provably rejected (see the section above), the
local-400 path answers the client directly instead of routing on.

**When it fires.** Both halves of the evidence are required, and the pair is
what keeps the error honest:

- the per-`(session, provider)` streak has reached
  `_DETERMINISTIC_LOCAL_400_THRESHOLD` — these are not one-off 400s, and
- the per-prompt-fingerprint entry's `resend_count` has reached the same
  threshold — *this history* was already rejected. The streak alone is not
  enough: a session can repeat 400s with a changed history (a compaction
  rewrite, a new user turn), and calling that `identical_history` would be a
  lie, so the fresh fingerprint's count of 1 filters it out.

A body without a `messages` array has no fingerprint and therefore never
produces the typed error.

**What the client gets.** HTTP `400` with

```json
{
  "error": "deterministic_client_error",
  "reason": "identical_history",
  "retry_after": 300,
  "body_snippet": "<first 512 chars of the llama-server rejection>"
}
```

plus a `Retry-After` header carrying the same seconds, `X-Provider` naming the
local provider that rejected it, and
`X-Deterministic-Client-Error: identical_history` so a client can branch
without parsing the body. `retry_after` is the fingerprint ladder's remaining
window (or its next rung when the window has just expired), so the header, the
body and the enforced backoff always agree. The error also increments
`proxy_http_errors_total{status="400", reason="deterministic_client_error"}`
as a distinct reason from `local_http_400` and `local_http_400_deterministic`.

**Why before fallback matters.** The response is returned from the local-4xx
branch *before* the `continue` that would move to the next provider, so:

- no remote dispatch happens (a remote cannot fix a request-shape rejection
  that is deterministic for this history; routing remote only spends quota),
- no `chain_hold_cycle` is consumed — the request never reaches the
  `ChainExhaustedError` path the hold wrapper reacts to, so a wedged session
  cannot hold the chain and starve audit or other sessions (the 2026-11-03
  incident shape), and
- the local provider is still not cooled down: the next request — with a
  compacted or otherwise changed history — dispatches local normally.

**Single 400s are unchanged.** A first local 400 still logs the body snippet,
increments `local_http_400` and falls back to the next provider exactly as
before. Only a repeat of a history already known to be rejected is answered
with the typed error.

## Terminal breaker: `session_wedged` (LP-0MW95BE7X00574OL)

The typed error tells a client its history is provably rejected; a client that
ignores it and keeps resending the *same* bytes has no path left, so the proxy
stops answering with a retryable verdict and says the session is wedged
(parent AC3).

**When it fires.** The terminal error is checked on the same evidence as the
typed error (it is reached from `_build_typed_400_response`, before the typed
response is built) and needs one thing more: the history's shared
`resend_count` must reach
`deterministic_client_error_escalation_threshold` — default `5`, read at the
top level of the proxy config or inside its `server` section, exactly like the
other proxy knobs. A non-positive, non-integer or boolean value falls back to
the default, so a mistyped knob can neither disable the breaker nor convict a
fresh session. Because the typed evidence is a prerequisite, even a threshold
of `1` never convicts a lone 400.

**What the client gets.** HTTP `400` with

```json
{
  "error": "session_wedged",
  "cause": "repeated_deterministic_400",
  "hint": "compaction may help",
  "retry_after": 900,
  "resend_count": 5
}
```

plus a `Retry-After` header carrying the same seconds, `X-Provider` naming the
local provider, and — so a client can tell the two errors apart **without
parsing the body** (parent AC5) —
`X-Session-Wedged: repeated_deterministic_400` alongside
`X-Deterministic-Client-Error: session_wedged` (the typed error's value there
is `identical_history`). It also increments
`proxy_http_errors_total{status="400", reason="session_wedged"}`, distinct
from `local_http_400`, `local_http_400_deterministic` and
`deterministic_client_error`.

**How long it asks the client to wait.** `retry_after` is the ladder's
remaining window, floored at `SESSION_WEDGED_MIN_RETRY_AFTER_SECONDS` (60 s)
and capped by the ladder's 15-minute ceiling (900 s): a wedged session must
not be told to retry immediately, and the number always agrees between the
header and the body.

**What resets it.** Exactly the events that end the loop: a successful **local**
stream (which also clears the backoff) and a **different** history (compaction,
truncation, a new conversation). The counter is shared by fingerprint across
sessions — two workers resending the same bad history trip one breaker — and
is in-memory only, so a proxy restart clears it along with the backoff state.
Until it fires, resends keep receiving the typed `deterministic_client_error`,
and the request still never reaches the fallback chain or a chain-hold cycle.

**Wiring.** `provider._build_session_wedged_response` builds the terminal
response from `fingerprint_backoff.session_wedged_error`; the threshold and
key live in `proxy/fingerprint_backoff.py`
(`DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD`,
`ESCALATION_THRESHOLD_CONFIG_KEY`).
