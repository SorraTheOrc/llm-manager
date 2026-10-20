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
| Unreserved part spent — **new** session (no recorded affinity) | Skip immediately, log `Provider budget exhausted: … limit=<rpm-reserve> reserve=<n> reason=rate_limited`, record attempt `status=rate_limited` with `effective_limit`/`reserve_slots`, continue to the next provider. **No waiting.** |
| Unreserved part spent, reserved slots free — **in-progress** session (affinity recorded on a 2xx dispatch) | Dispatch on the reserved slot: the reserve exists so established sessions keep their provider and prompt cache instead of queueing behind new traffic. |
| Window completely full — **in-progress** session | Wait up to `server.provider_rate_limit_wait_max_seconds` (default **15**) for the window to drain, then dispatch on this entry (`status=rate_limited_waited`); if it never drains, pass through like a new session (`reason=rate_limited_after_wait`). |
| Waiters already at `server.provider_rate_limit_wait_max_waiters` (default 8) | Pass through (`reason=rate_limited_waiters`) so waiters cannot pile up. |
| Client disconnects during the wait | Wait aborts promptly (`reason=rate_limited_disconnect`). |
| `rate_limit_rpm` unset or `0` | Ungated: behaviour exactly as before; no affinity recorded. |

**Why the reserve exists (LP-0MVDUN5CY009VD20).** Affinity is recorded on the
first successful dispatch, so *every* later turn of that session counts as
"in-progress". Without a reserve, established sessions queued behind new-session
traffic and waited 15–55 s per turn, which dominated first-token latency
(p90 6.5 s → 12.8 s, p99 21 s → 53 s while generation speed was unchanged).
Reserving `floor(rate_limit_rpm × provider_rate_limit_reserve_ratio)` slots
(4 of 15 at the former budget) keeps a slot available for them, so waits become
rare; the default 15 s cap bounds whatever remains. The reserve rounds down and
is clamped to `rate_limit_rpm - 1`, so at the current budget of 2/min the
reserve is **0**: there are no reserved slots and both new and in-progress
sessions share the 2-slot window.

Affinity is `session → entry`, TTL-scoped by
`server.provider_affinity_ttl_seconds` (default 900), and is **never** shared
across sessions or between sibling entries (e.g. free `munitoai-free` vs paid
`munitoai`). The point of the split behaviour is prompt-cache continuity: a
session already on the free tier keeps its provider, while a batch of new
agent sessions passes straight through instead of queueing behind an exhausted
allowance.

`munitoai-free` ships with `rate_limit_rpm: 2` in `config.yaml` (the former
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
spill to the paid tiers rather than consuming the free allowance. At 2/min the
reserve ratio (`provider_rate_limit_reserve_ratio`, default 0.3) rounds down to
0 slots, so new and in-progress sessions both see the full 2-per-window cap;
in-progress sessions can still wait for the window to drain (bounded by
`provider_rate_limit_wait_max_seconds`). Takes effect at the next proxy
restart.

### Off-peak paid tiers in the `low` chain (LP-0MVF17VP70053JXX)

`low` priority prefers local and queues over paid remote fallback, then uses the
free tier (`munitoai-free`, `rate_limit_rpm: 2`). Because that free tier is the
last entry of the chain, exhausted local slots plus an exhausted free budget
used to leave the request nowhere to go but an upstream 429. `plan`, `author`
and `code` therefore extend their `low` chain with the three `opencode-go`
entries:

```text
low: [local-qwen3, munitoai-free, opencode-go-3, opencode-go-2, opencode-go]
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
| HTTP 429 / usage limit | `_usage_limit_account_key` = `api_key_env@endpoint` | Independent per API key: a 429 on `opencode-go-3` leaves `opencode-go-2` and `opencode-go` eligible → 3× rate-limit capacity (LP-0MSMBWB23009XYPW, LP-0MVCX0E2V001VB5N). |
| 5xx / stall / terminal error | `_failure_domain_key` = `endpoint:model` | Shared domain: one failure skips all three for the rest of the request → **no** 3× outage resilience (LP-0MSG45I8Q0020N1F). |

The fully paid `munitoai` and `deepseek-flash` entries never join a `low` chain:
they carry no window discipline that would confine spend to off-peak hours.

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
| Estimated context > entry limit | Skip the entry: log `routing_skip_provider_context provider=<p> type=<t> status=provider_context_exceeded estimated_tokens=<n> limit=<limit> reason=provider_context_exceeded session=<s>`, record attempt `status=provider_context_exceeded` with the estimate and the limit, continue to the next eligible provider. **No cooldown, failure-domain exclusion, usage-limit reset, affinity record or contention-queue entry** — the entry stays eligible for later, smaller requests. |
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

`github-primary` is deliberately left ungated: it declares no `model`, so a
per-entry ceiling cannot be sourced.


Some gateways answer HTTP 429 with custom `x-ratelimit-*` headers instead of
the standard `Retry-After` (MunitoAI sends none). The proxy can honour them so
a rate-limited provider is skipped for the upstream's own reset window rather
than being re-probed every ~45 s by the exponential-backoff guess.

Enable it per provider entry, or globally under `server`:

```yaml
- name: munitoai-free
  type: remote
  rate_limit_rpm: 2
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

- **Trigger:** an HTTP 429 whose body is a per-account rate-limit error
  (`type: rate_limit_error` or `code: rate_limit_exceeded`, top-level or
  nested under `error` — the observed MunitoAI body) **or** which carries
  `x-ratelimit-over-limit: yes`, **and** which also yields a parseable
  `x-ratelimit-reset-*` duration. No reset signal (or a zero/absent value)
  means no account quarantine: the pre-existing short provider cooldown and
  per-request account-key exclusion apply exactly as before.
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

- **Writer** — after a *successful* `slot_save`, the proxy records the measured
  `prompt_tokens` from the upstream `usage` block (the whole served prompt is
  now resident in the saved KV cache) plus a SHA-256 fingerprint of every
  message in the request. No usage block → no sidecar (today's behaviour).
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
  (`cached_ratio = 0`, conservative). Session invalidation deletes both the
  snapshot and its sidecar and clears the in-memory ratio.
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
| cheap (config-cheap.yaml) | `queue` | Queued cross-session, bounded by `contention_queue_max_wait_seconds` (120) and `contention_queue_max_depth` (8); dispatched local when a slot frees in time, otherwise falls back to the next remote provider |
| fast (config-fast.yaml) | `queue` | Short-queue: bounded by `contention_queue_max_wait_seconds` (45) and `contention_queue_max_depth` (3); falls back to remotes sooner than cheap to minimise latency |

Fast mode defaults (depth 3 / wait 45s) are deliberately smaller than cheap
(depth 8 / wait 120s) so burst traffic during peak hours spills to remotes
more quickly — see LP-0MTQYIK4Z008XF2V for the tuning rationale.

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
  contention_queue_max_wait_seconds: 120    # clamped to [1, max_runtime_seconds]
  contention_queue_max_depth: 8             # clamped to [1, 16]

  # fast profile (config-fast.yaml)
  contention_queue_policy: queue
  contention_queue_max_wait_seconds: 45     # clamped to [1, max_runtime_seconds]
  contention_queue_max_depth: 3             # clamped to [1, 16]
```

The queue engages only when ``contention_queue_policy`` is ``queue`` AND the
proxy is in the corresponding operating mode. Absent keys default to
``fallback`` for backward compatibility.

## Prefill-ETA admission guard (LP-0MVGDUV5E005UWBQ)

Queuing only helps when the local prefill can actually finish inside the wait
budget. Evidence for 2026-10-19 07:23 → 2026-10-20 07:23: of 5,805 queued waits
only 855 were dispatched local — **4,947 (85 %) were abandoned after up to
120 s and then went remote anyway**, while local prefill times during the same
window were p50 61.8 s / p90 269.8 s / max 817 s. Waiting 120 s to maybe avoid
a remote call whose median TTFT is 27 ms is a losing trade unless the prompt
is genuinely cache-warm.

Before a request enters the contention queue (and before a dispatch that
would start another concurrent prefill), the proxy therefore costs the
projected prefill and compares it with the caller's wait budget:

```
eta_prefill = coded_tokens * coded_ms_per_token + new_tokens * prefill_ms_per_token
  coded_tokens = est_tokens * cached_ratio          # served from the prompt cache
  new_tokens   = est_tokens * (1 - cached_ratio)    # must actually be prefilled

admit  if eta_prefill <= wait_budget
else   refuse local admission immediately -> next provider in the chain
```

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
  `0.9*100k*0.5 ms + 0.1*100k*3.9 ms ≈ 84 s` and is admitted, while the same
  prompt cache-cold costs ≈ 390 s and is refused. Locality is kept where it
  actually pays.
- **Priority semantics are unchanged**: `high` never queues, so it has no
  wait budget and the guard never applies to it; `low`/`medium` use
  `contention_queue_max_wait_seconds`.
- **Refusal is immediate and visible**: the request falls through to the next
  provider without waiting. If the whole chain is exhausted, the client
  receives a retryable `429` with `Retry-After` (how much longer the prefill
  would need than the budget) and `X-Local-Admission: reject` /
  `X-Local-Admission-Eta` / `X-Local-Admission-Budget` headers, plus the
  machine-readable `code: local_admission_reject`.
- **Observability**: each decision logs
  `local_admission_guard decision=admit|reject reason=… enforced=true|false
  eta_prefill_s=… budget_s=…`. The 24h report counts them as
  `local_admission_reject` / `local_admission_reject_enforced` /
  `local_admission_admit`. `/llama/local/status` publishes
  `prefill_eta_seconds` (cost of the reference prompt, cache-cold) and
  `queue_eta_seconds` (that cost repeated for every occupied slot, plus the
  newcomer's own) so a dispatcher can pace *before* launching a session.

Rollout control (`server.local_admission_guard_policy`):

| Value | Behaviour |
|-------|-----------|
| `reject` (default) | Refuse admission when the ETA exceeds the budget |
| `observe` | Log the verdict (`enforced=false`) but keep queuing — use to validate calibration on live traffic |
| `off` | Do not evaluate the guard at all |

Verification target (48h after rollout): `contention_queue_fallback_after_queue`
drops well below the 4,947 baseline, client-visible latency for refused
requests approaches the remote TTFT baseline instead of ~120 s, local serve
share stays above 2.6 %, and mid-stream error counts do not increase.

## Stream Error Handling (recovery-first + informative-error fallback)

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

## Usage-limit reset quarantine (GoUsageLimitError)

HTTP 429 responses whose error type is `GoUsageLimitError` (a usage-limit
variant distinct from `FreeUsageLimitError`) are treated as usage-limit
**reset** events rather than generic rate-limit events. The proxy parses the
reset duration from the provider message (`Resets in 22hr 43min`) — falling
back to `metadata.limitName` (daily/weekly/monthly) when the message carries
no explicit duration — adds a 2-minute safety margin, and quarantines the
**whole failure domain** (all entries sharing the endpoint, e.g. both
`opencode-go` and `opencode-go-2` on `https://opencode.ai/zen/go`) until the
computed reset time passes. Routing decisions during the block log
`usage_limit_reset_pending` with the reset time and do not contact the
upstream; once the reset time arrives the domain becomes eligible again
without operator intervention. `FreeUsageLimitError` responses without a
reset time get the default 3-hour cooldown (10800s; per-provider overrides
were retired with `opencode-big-pickle` in LP-0MT652JRM004ZLSI). Tracked in
**LP-0MSLJPOCC0001ROJ**.

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
