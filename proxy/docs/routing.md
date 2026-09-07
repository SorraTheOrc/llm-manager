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

### 5. Remote Model Handling

For remote models (`proxy_to_remote()`):

- Constructs the target URL from `endpoint` + path
- Adds API key from environment variable (e.g., `OPENAI_API_KEY`)
- Adds custom headers from config
- Forwards the request via `httpx` (streaming for SSE responses)

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

## Slot Contention: Per-Mode Queue vs Fallback (LP-0MSORQVK50012Q4D)

When every local slot is busy (``local_active_queries >= session_slot_pool_size``),
the proxy's behavior depends on the per-mode contention policy declared in the
active mode config:

| Mode (config file) | `contention_queue_policy` | Slot-contention behavior |
|--------------------|---------------------------|--------------------------|
| cheap (config-cheap.yaml) | `queue` | Queued cross-session, bounded by `contention_queue_max_wait_seconds` (60) and `contention_queue_max_depth` (4); dispatched local when a slot frees in time, otherwise falls back to the next remote provider |
| fast (config-fast.yaml) | `fallback` | Today's behavior — skip to the next remote provider immediately |

Key semantics (see `proxy/proxy/provider.py` `_maybe_queue_for_local_slot` and
`proxy/proxy/contention_queue.py`):

- **Cross-session**: the queue is process-global, so a request from session B
  can wait (bounded) behind a long audit stream from session A. This is what
  converts overnight contention fallbacks into queued-local dispatches.
- **Context bypasses never queue**: requests that cannot fit the KV slot
  (`context_too_large` / `large_context_bypass`, LP-0MSF8XDG7000PERM /
  LP-0MRE4NBQ5009V5BX) fall back exactly as before — they are physical
  capacity limits, not contention.
- **Wake signals**: the queue wakes on BOTH `local_active_queries` decrement
  (a local stream ended) AND slot-persistence / lease release (slot
  save/restore frees the backend during model switches).
- **Timeout accounting (Q2=a)**: the queued wait subtracts from the
  client-visible adaptive timeout budget — total (wait + serve) stays within
  `llama_adaptive_timeout_*` (base 60s + 0.015/token, capped at
  `max_runtime_seconds`), so interactive clients never see queue wait + serve
  exceed the adaptive envelope.
- **Metrics**: queued count, queued duration, and fallback-after-queue count
  are exposed via `proxy/proxy/contention_queue.py::metrics()` and the
  `status_request` / `contention_queue_dispatch` /
  `contention_queue_fallback_after_queue` log lines (Prometheus counters in
  `proxy/proxy/metrics.py`) so the 24h proxy report can quantify the gain. The
  `contention_queue_dispatch` line carries `queued_duration`, `policy`, and
  `depth`; the `contention_queue_fallback_after_queue` line carries
  `queued_duration` (the elapsed wait, F4 AC2). The status endpoint
  (`/llama/local/status`) includes a live contention-queue snapshot via
  `observability.contention_queue_snapshot`. The 24h report pipeline
  (`.pi/skills/proxy-usage-analysis`) parses and aggregates these lines into
  `contention_dispatch` / `contention_fallback_after_queue` counts plus
  total queued duration (`contention_queued_duration_seconds`).

Config reference:

```yaml
server:
  # cheap profile
  contention_queue_policy: queue              # or "fallback" (fast)
  contention_queue_max_wait_seconds: 60      # clamped to [1, max_runtime_seconds]
  contention_queue_max_depth: 4              # clamped to [1, 16]
```

The queue engages only in cheap operating mode (`proxy.mode.read_mode() ==
"cheap"`); absent keys default to fallback for backward compatibility.

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
   provider entry increments a per-entry consecutive-failure streak
   (`_sibling_failure_count` in `proxy/proxy/provider.py`), shared across
   requests (module-level state, like the other cooldown mechanisms).
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
