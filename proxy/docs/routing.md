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
priorities: [low, medium, high]   # local-qwen3, munitoai-free,
                                 # opencode-go-3, opencode-go-2, opencode-go
priorities: [medium, high]        # munitoai, deepseek-flash
```

The resulting `low` chain is the model's own provider order filtered by the
label (LP-0MVGPZWY0001JD1D), so it reads directly off the `providers` list:

```text
chain(low) = [local-qwen3, munitoai-free, opencode-go-3, opencode-go-2, opencode-go]
```

Those entries keep the off-peak `available_times` windows added by
LP-0MSXPHTD9004DZ4P — UTC `00:00-01:00`, `04:00-06:00`, `10:00-00:00` — so the
extra tiers are only eligible outside peak hours; the windows are **not**
widened by this change. The cheap priority window (`server.priority_window`,
default local `Europe/Dublin` 01:00–10:00 = UTC 00:00–09:00) overlaps the
opencode windows in ~3 of its 9 hours (UTC 00:00–01:00 and 04:00–06:00); the UTC
01:00–04:00 and 06:00–10:00 gaps are deliberate paid-spend avoidance.

Capacity vs resilience — the three entries are **not** interchangeable:

| Failure | Quarantine key | Effect on the sibling entries |
|---------|----------------|-------------------------------|
| HTTP 429 / usage limit | `_usage_limit_account_key` = `api_key_env@endpoint:model` | Independent per API key: a 429 on `opencode-go-3` leaves `opencode-go-2` and `opencode-go` eligible → 3× rate-limit capacity (LP-0MSMBWB23009XYPW, LP-0MVCX0E2V001VB5N). The key includes the **model**, so the same key on a different model is a separate account/limit (LP-0MVT8SAZE00570J5 corrected this row). |
| 5xx / stall / terminal error | `_failure_domain_key` = `endpoint:model` | Shared domain: one failure skips all three for the rest of the request → **no** 3× outage resilience (LP-0MSG45I8Q0020N1F). |

The fully paid `munitoai` and `deepseek-flash` entries declare exactly
`priorities: [medium, high]`, so they never join a `low` chain: they carry no
window discipline that would confine spend to off-peak hours.

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
  local_capacity_admission_enabled: false   # rollback switch, default off
  local_capacity_admission_budget_ratio: 0.8
  local_capacity_admission_horizon_seconds: 3600
  local_capacity_admission_decode_seconds: 30
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
- **Observability.** Each admit logs `routing_capacity_admit provider=… model=…
  reason=capacity_admission projected_s=… admitted_s=… budget_s=… idle_s=…
  budget_ratio=… horizon_s=… estimated_tokens=… cached_ratio=… cache_state=…
  warm_threshold=… session=…` and records a `status=capacity_admit` attempt.
- **Rollout / rollback.** Ships **off**. Enable only once cache warmth
  (LP-0MVGD9YXH008O304), the prefill-ETA guard (LP-0MVGDUV5E005UWBQ) and
  prefill serialisation (LP-0MVGDUUV3001XFRJ) are live in the running proxy and
  the 24 h measurement (LP-0MVGHXD59008UOUD) is green; revert by setting
  `local_capacity_admission_enabled: false` (read per request, no restart).

## Prefill serialisation (LP-0MVGDUUV3001XFRJ)

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

At most `server.local_max_concurrent_prefills` prefills now run at once:

```yaml
server:
  local_max_concurrent_prefills: 1   # 1 = serialise (default); 0 = disabled
```

- **Decode is not affected** — only prefill-vs-prefill is serialised; a slot
  that is already generating does not block another slot's prefill.
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
now walks a **five-rung ladder**, in this order:

1. **Hard context cap** — `context_too_large` when the prompt exceeds the warm
   threshold. Unchanged, never overridable (the prompt cannot fit the KV slot).
2. **Resident-prefix gate** — admit only when the session's *measured* reusable
   prefix is at least `local_resident_prefix_min_ratio` of the prompt;
   otherwise skip local with `resident_prefix_insufficient` and fall through the
   chain as `large_context_bypass` does today. `0` disables this rung.
3. **Session tenancy** — at most `local_session_tenants` distinct sessions hold a
   local reservation at once (`tenant_limit` otherwise); same-session re-entry is
   always permitted and a tenancy idle-lapses after
   `local_session_tenancy_idle_seconds` of no activity.
4. **Live-priced cold check** — `new_tokens` is priced from the same live
   measurement (`estimated_tokens - min(context_tokens, estimated_tokens)`), so
   the cold budget is evaluated against real prefill work; a cold miss is
   `large_context_bypass`.
5. **Idle-exempt override** — last, so an idle GPU still serves cold work
   (`local_idle_admit_*`, below). It may override rungs 2 and 4, never rung 1.

```yaml
server:
  local_resident_prefix_min_ratio: 0.5   # 0 disables the residency gate
  local_session_tenants: 2               # default: the slot count; 0 disables
  local_session_tenancy_idle_seconds: 30 # default: the adaptive dispatch lease
```

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

**Fail-open on unknown state.** A missing or stale snapshot (> 5 s), a session
that owns no slot, or a build that no longer reports `context_tokens` (llama.cpp
b8782 dropped `n_past`) degrades one rung at a time: measured ratio → session
owns the slot (restore path) → today's predicted-ratio decision. Admission never
turns absent data into a hard local outage.

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
`routing_resident_prefix provider=… model=… decision=<admit|resident_prefix_insufficient|tenant_limit|large_context_bypass> n_past_actual=… estimated_tokens=… reuse_ratio=… slot_owner_session=… residency_source=<measured|measured_last_dispatch|session_slot|non_resident|unknown> measurement=<measured|unavailable:no_slot|unavailable:no_prior_measurement|unavailable:stale_view|unavailable:capture_disabled> min_ratio=… tenants=… occupants=… session=…`.

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
| `unavailable:no_slot` | a **fresh** view shows the session owns no slot, so its reusable prefix is a measured zero |
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

## Slot Contention: Per-Mode Queue vs Fallback (LP-0MSORQVK50012Q4D)

When every local slot is busy (``local_active_queries >= session_slot_pool_size``),
the proxy's behavior depends on the per-mode contention policy declared in the
active mode config:

| Mode (config file) | `contention_queue_policy` | Slot-contention behavior |
|--------------------|---------------------------|--------------------------|
| cheap (config-cheap.yaml) | `queue` | Queued cross-session, bounded by `contention_queue_max_wait_seconds` (60) and `contention_queue_max_depth` (8); dispatched local when a slot frees in time, otherwise falls back to the next remote provider |
| fast (config-fast.yaml) | `queue` | Historical: bounded by `contention_queue_max_wait_seconds` (45) and `contention_queue_max_depth` (3); unified to cheap caps (depth 8, wait 60s) in LP-0MV881O5C007PSL4 / LP-0MVIQ0EXV00673HO |

Fast and cheap now share the same queue caps (depth 8, wait 60s) since
LP-0MV881O5C007PSL4 unified them; the fast mode note above is historical.
See LP-0MTQYIK4Z008XF2V / LP-0MVIQ0EXV00673HO for tuning rationale.

Key semantics (see `proxy/proxy/provider.py` `_maybe_queue_for_local_slot` and
`proxy/proxy/contention_queue.py`):

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
  # cheap profile (config-cheap.yaml)
  contention_queue_policy: queue
  contention_queue_max_wait_seconds: 60    # clamped to [1, max_runtime_seconds]
  contention_queue_max_depth: 8            # clamped to [1, 16]
```

The queue engages only when ``contention_queue_policy`` is ``queue`` AND the
proxy is in the corresponding operating mode. Absent keys default to
``fallback`` for backward compatibility.

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
  occupied_seconds = max(eta_prefill * busy_slots, sum(remaining occupant service))
    # remaining occupant service is re-priced from the occupant's remaining
    # uncached new tokens at the contended co-residency rate (see below)
  busy_slots       = min(slot_pool, max(generating_concurrency, prefills_in_flight))

admit  if projected_wait <= wait_budget
       and projected_wait <  local_prefill_deadline_seconds
else   refuse local admission immediately -> next provider in the chain
```

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
`contention_queue_max_wait_seconds (60) < local_prefill_deadline_seconds (120)
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
- **Priority semantics are unchanged**: `high` never queues, so it has no
  wait budget and the guard never applies to it; `low`/`medium` use
  `contention_queue_max_wait_seconds`.
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
  after >120 s, most on a free slot. The direct refusal logs
  `reason=eta_exceeds_budget_direct free_slot=true` (queue-entry refusals keep
  `reason=eta_exceeds_budget`), records a `local_admission_reject` attempt so
  the chosen fallback provider appears in the diagnostics, and falls through
  to the next provider. It reuses the same estimator and the same single
  budget (`contention_queue_max_wait_seconds`) — there is no second budget
  knob, and a cold/uncalibrated bucket still admits.
  **Deadline refusal** (LP-0MVKUKEU80022C3U): when the projected **time to
  first byte** — own ETA plus the occupied gate, i.e. the same
  `projected_wait` above — reaches ``local_prefill_deadline_seconds``
  (default 120 s) the guard also refuses with
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

| Value | Behaviour |
|-------|-----------|
| `prefer-local` (default) | Today's behaviour: use local when it can serve within the budget, else fall through to the next provider |
| `local-preferred` | **Local while the request fits local capacity, remote beyond it.** Pin local **while the request can still fit**. A *deferrable* request — the block is temporal, so waiting can help — waits for local capacity and, when the wait budget runs out, receives the retryable `429` deferral (`X-Local-Route: refused`, `X-Local-Fallback-Reason: <exit token>`, `Retry-After` / `retry_at`) instead of being moved to a paid provider. A *too-large* request — one that outgrows the local serving envelope, so waiting cannot help — is **never deferred and never refused for local capacity**: the local entry leaves the chain and a remote tier serves it (`200`, `X-Local-Route: remote`, no `Retry-After`). See [`local-preferred`: pinned while deferrable, promoted when too large](#local-preferred-pinned-while-deferrable-promoted-when-too-large) and the [per-reason retry contract](#per-reason-retry-contract-lp-0mvntt4x8009io2x) |
| `remote-ok` | Skip the local route entirely (background / low-priority work). The local provider is removed from the chain |

Wire compatibility (LP-0MVSHWYVE005MS16): `local-only` is the **legacy,
deprecated** alias of `local-preferred`. It is still accepted and normalised to
`local-preferred` at parse time — a client that keeps sending it retains
pin-local semantics and is never demoted to `prefer-local` — and it logs one
`WARNING` deprecation line per request. No removal date is scheduled, so
migrating a client is optional: the alias is the same request intent, and an
un-migrated client needs no code change to get the deferral semantics.

Unknown values (and an absent header) behave as `prefer-local`, so the header
is purely additive. `high` priority keeps its "never queue" semantics under
every preference.

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
| `too_large` | The request itself outgrows the local serving envelope, so no amount of waiting helps. | **Promoted.** The local entry leaves the effective chain *before* the dispatch loop (the same filter `remote-ok` uses) and a remote tier serves the request: no local attempt, no queue entry, no lease, no local-capacity `429`. |

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
candidate. It is the mirror of `remote-ok`: where `remote-ok` removes the local
route from the chain *before* dispatch, a pinned request refuses *before* any
remote entry is dispatched (LP-0MVRMC1L3006F0F3). The admission guard is still
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
| Admission guard, direct free-slot dispatch path | `eta_exceeds_budget_direct` (the deadline/safety/budget reasons keep their own token) | refusing | `local_admission_guard ... free_slot=true ...` plus, for a pin-local client, `local_only_refused reason=<token> ... route=refused`. The `local_admission_guard_direct_fallback ... → routing to next provider` line is emitted **only** when the request really falls through to the next provider (`prefer-local` / `remote-ok` / absent header) |
| Local entry in cooldown (chain resolution) | `local_provider_cooldown` | refusing | `Skipping provider=<local>: ... in cooldown (...s remaining) reason=local_provider_cooldown preference=...` |
| Router skip: prompt at/over the warm (KV-slot) cap; the per-entry `max_context_tokens` ceiling composes as an additional cap — the prompt can never fit local, however long it waits | `context_too_large` | promoting | `routing_skip_local ... reason=context_too_large → skipping local, routing to next remote provider` |
| Router skip: economic cold-prefill bypass (an idle-admission denial keeps this token) | `large_context_bypass` | refusing | `routing_skip_local ... reason=large_context_bypass ...` |
| Router skip: residency ladder found too little reusable prefix (an idle-admission denial keeps this token) | `resident_prefix_insufficient` | refusing | `routing_skip_local ... reason=resident_prefix_insufficient ...` |
| Router skip: per-tenant residency cap (`session_tenants`) | `tenant_limit` | refusing | `routing_skip_local ... reason=tenant_limit ...` |
| Router skip: (session, prompt) in post-deadline prefill-thrash cooldown | `prefill_thrash_cooldown` | refusing | `routing_skip_local ... reason=prefill_thrash_cooldown ...` |
| Contention-queue depth cap reached (never queued), or the wait budget expired | `fallback_after_queue` | refusing | `contention_queue_fallback_after_queue ... preference=...` |
| Local slots busy with the queue disabled (fast-mode fallback policy) | `local_concurrency_limit` | refusing | attempt record `status=local_concurrency_limit`; the refusal line `local_only_refused reason=local_concurrency_limit ...` |
| Prefill-gate / lease denial (a lease-active 503 after the retry window) | `local_lease_active` | refusing | `contention_queue_fallback_after_queue ... reason=local_lease_active` |
| Quota hard stop (chain resolution) | `quota_hard_stop` | refusing | `Skipping provider=<local>: ... reason=quota_hard_stop preference=...` |
| Usage-limit reset pending (chain resolution) | `usage_limit_reset_pending` | refusing | `Skipping provider=<local>: usage_limit_reset_pending ... reason=usage_limit_reset_pending ...` |
| Outside the entry's `available_times` window (chain resolution) | `outside_available_window` | refusing | `Skipping provider=<local>: outside its available_times window (UTC) reason=outside_available_window ...` |
| The chain has no local entry at all | `local_only_no_local_route` | refusing | `local_only_refused reason=local_only_no_local_route preference=local-preferred verdict=deferrable` |
| Any exit that left no reason on the request (e.g. a plain post-dispatch slot-exhaustion 503) | `local_route_unavailable` | refusing | `local_only_refused reason=local_route_unavailable preference=local-preferred verdict=deferrable` |

The table is pinned by `proxy/tests/test_local_only_refusal_matrix.py` (one row
per exit: a pinned `local-preferred` request ⇒ `429` + token + **no upstream
call**; `prefer-local` and an absent header ⇒ today's remote fall-through,
byte-compatible) and by `proxy/tests/test_local_fallback_reason_tokens.py`
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
shared with a `prefer-local` request that is refused locally and *then served
remotely*, so `decision=reject` alone cannot distinguish the two. The refusal
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
reserved for a real fall-through (the `prefer-local` / `remote-ok` / absent
header clients that are served by the next provider).

#### Promoted: a too-large request leaves the local chain

A `too_large` verdict is never deferred and never refused for local capacity.
It is decided **before** the dispatch loop, once the authoritative routing
estimate exists, by `_promote_too_large_local_request()`: the local entry is
filtered out with the same `apply_local_preference()` filter `remote-ok` uses,
so the remote tiers serve the request exactly as if the client had sent
`remote-ok` — no local attempt, no queue entry, no lease. The per-request
routing line reports `outcome=promoted`, which is how an operator tells a
capacity promotion from a client asking for remote.

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

#### Client migration: what a client author must do

**No change is required to keep pin-local semantics.** A client that already
sends `local-only` may keep sending it: the alias is accepted and normalised to
`local-preferred`, so the request keeps the deferral behaviour described here
(one deprecation `WARNING` is logged per request). Renaming the value is a
clean-up, not a migration gate.

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
  or `no_local_provider` when the local route was never attempted (a
  `remote-ok` chain strip that removed every local entry, or a chain that
  never had one). A `local-preferred` request promoted because it is
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

`prefer-local` (the default) keeps today's behaviour unchanged: on every one
of the exits above it falls through to the next provider exactly as before.
The pin adds the deferral for `local-preferred` while its verdict is
`deferrable`, and drops the local entry entirely when the verdict is
`too_large`. `remote-ok`, by contrast, is enforced structurally: the local
route is removed from the chain *before* dispatch (`apply_local_preference`),
so a `remote-ok` request never reaches a local-route exit at all.

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
  suggesting a switch to `prefer-local` to reach a remote fallback; a
  `too_large` `local-preferred` request is already promoted automatically, so
  treat the compaction half (resend at `budget_tokens` or less) as the
  actionable remedy. The string refresh is tracked in LP-0MVTGI2RP009SRKN.
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
  response a `prefer-local` request reaches when the rest of the chain is
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
  `X-Local-Preference: prefer-local` (or `remote-ok`): both stop pinning, so the
  remote fallback is reachable again. A client that cannot change its routing
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
values — `prefer-local` keeps today's remote fall-through, and the ETA/occupancy
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
(`outcome=promoted`). Under `prefer-local` (the default) an exit instead falls
through to the next provider, and waiting is the client's own choice.

`/llama/local/status` publishes the reservation counters
(`reserved_outstanding`, `reserved_issued`, `reserved_used`,
`reserved_cancelled`, `reserved_expired`) — `issued − used − cancelled −
expired` is the reserved-but-unused signal for ticket hoarding.

## Stream Error Handling (recovery-first + informative-error fallback)

### Prefill deadline (LP-0MVIQ0G2M0006QF6)

Before the first byte, a local dispatch is bounded by
`server.local_prefill_deadline_seconds` (default **120 s**, 0 disables). The
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
(`contention_queue_max_wait_seconds`, 60 s) so a request legitimately admitted
inside its budget is never cut off, and **below**
`local_dispatch_first_byte_timeout_seconds` (840 s), which is itself below
`local_dispatch_max_prefill_seconds` (900 s) and below the client's own 900 s
queue deadline. The ordering is asserted at startup by
`_startup_validate_prefill_timeout_config` (via
`prefill_eta.validate_prefill_timeout_ordering`): a violating configuration is
**fatal**, because an out-of-order timeout set either cuts off requests the
admission guard legitimately admitted or races the client's own deadline. It
is the hard backstop that does not depend on ETA-estimator quality.

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
  conservative p75 ETA fits inside `0.95 × local_prefill_deadline_seconds`
  (= 114 s at 120 s). A 25k cold prompt prices ~93 s at the band's p75 rate
  (~270 t/s) and is admitted; a band-slow 25k (~132 s at ~190 t/s) is refused.

`local_admission_guard_reference_tokens` (shipped **25 000**, previously
32 000) aligns the published `/llama/local/status` reference prompt with the
budget so there is one number instead of two. `cold_threshold` (derived,
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

**What it refuses.** The window is keyed by **(session, prompt fingerprint)**,
not by session alone. The fingerprint is a 32-hex-char SHA-256 prefix over the
request's `messages` (sorted keys, compact separators); message order is
significant, and a body with no usable `messages` yields no fingerprint. So:

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
(`_release_prefill_thrash_cooldown_on_free_gpu`):

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
fingerprint and its prompt size. Gating: `provider.py`, which logs
`routing_skip_local … session=… cooldown_remaining=Ns
reason=prefill_thrash_cooldown → skipping local`. Clearing: the router's
first-byte success path. The store is in-process (`_prefill_thrash_cooldowns`,
keyed `(session, fingerprint)`; a proxy restart resets it with the routing
state it guards) in `proxy/proxy/router_helpers.py`. Unit and routing-level
behaviour are pinned by `proxy/tests/test_prefill_thrash_cooldown.py`; the skip
counts are measured by `scripts/capacity_signal_verification.py`
(`prefill_thrash_cooldown` block — both the strict "fresh poll" attribution and
the relaxed "no live local work" population the 336-skip baseline was derived
with).

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
| `orphan_cleanup` / measured cache deficit | **Cancel** + **annotate** (same shared core) | `lease_released … reason=<that reason>` |

The cancel is real on this build, not decoration. llama-server aborts at its
next write to the socket, and `return_progress` guarantees such writes
throughout prefill. Evidence (2026-10-28 `llama-server.log`, 82
`Connection handling canceled` events): **42** of them cancelled a task that
never reached `prompt processing done`, i.e. mid-prefill cancellation is
achievable and observable here. The sequence is always
`Connection handling canceled` → `stop: cancel task, id_task = N` →
`slot release: … stop processing`.

Because the backend can lag the close, the abandonment is also **recorded**
(`_record_abandoned_slot`, `proxy/proxy/router_helpers.py`) with a bounded TTL
(`DEFAULT_ABANDONED_SLOT_TTL_SECONDS`, 900 s — longer than the 600 s cooldown
so the annotation outlives the window in which the ghost can still be seen
processing), keyed by slot identity so the registry cannot grow past the pool,
and pruned lazily on read. It is published on `/llama/local/status`:
`status_schema_version` 4, each slot carrying `abandoned`,
`abandonment_reason`, `abandoned_session_id` and `abandoned_at_seconds_ago`,
the top level carrying `local_abandoned_slots`, and `capacity_hint` gaining the
additive value `abandoned_busy` for a pool occupied *only* by abandoned work.
`saturated` therefore keeps meaning live, proxy-owned decode work, and a
busy-but-abandoned pool is never reported `idle`. See
[per-slot-status-verification.md](per-slot-status-verification.md) for the
status semantics and the consumer rule (any unrecognised hint value → treat as
busy).

Interaction with the cooldown above: the annotation is what makes the *skip*
decision honest (a ghost no longer reads as live work), while the ghost itself
keeps the *free-GPU release* from firing — the GPU really is busy until
llama-server finishes the cancelled task.

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
