# Is the local LLM involved in Responses API calls?

**Work Item:** LP-0MTY7EKZC006T893  
**Type:** task (read-only investigation)  
**Date:** 2026-09-26  
**Active mode at time of analysis:** `cheap` (`/admin/mode` → `{"mode":"cheap"}`,
`LLAMA_PROXY_CONFIG=proxy/config-cheap.yaml`)  
**Evidence:** `proxy/config.yaml`, `proxy/config-fast.yaml`, `proxy/config-cheap.yaml`,
`proxy/proxy/proxy_remote.py`, `proxy/proxy/provider.py`, `proxy/proxy/router.py`,
`proxy/proxy/router_helpers.py`, `proxy/proxy/compaction*.py`,
`proxy/proxy/tokenizers.py`, `proxy/proxy/utils.py`, `proxy/proxy/lifecycle.py`,
live `/var/log/llama-proxy/proxy.log` (2026-09-26).

---

## Executive summary

**The local llama-server (Qwen3/Qwen2.5 GPU inference) is NOT on the critical
path of a Responses API call to a remote model such as Muse
(`muse-spark-1.3-contributor` via `opencode-go`).** The chat↔responses
translation is pure Python; the only local resources touched by the request
path are the CPU tokenizer (file I/O) and ordinary Python/session bookkeeping.

When a request *is* served by the local LLM, it is served **instead of** the
remote responses call — not as a step towards it. The two are alternatives in
the routing fallback chain, never sequential.

Three specific confirmations:

1. **Translation is pure Python.** `_translate_chat_to_responses`,
   `_translate_responses_to_chat`, and `_translate_responses_stream` are
   dict/SSE transforms with no model call (`proxy_remote.py:379,493,571`).
2. **No local tier in the compaction summariser.** The configured
   `models.compact` chain is remote-only (`opencode-go-compact` →
   `deepseek-flash-compact`); the local Qwen2.5-7B summariser is only built as
   a fallback when `models.compact` is absent, which it is not
   (`compaction_summarizer.py:730` vs `:353`).
3. **Compaction is disabled anyway.** `compaction_trigger_ratio: 0` in all
   three profiles, so `plan_session_compaction` returns
   `reason="compaction_disabled"` before any summariser is called
   (`compaction.py:685`, `compaction.py:193`). Live log confirms:
   `compaction_bypass_eval ... reason=compaction_disabled`.

**Where latency investigation should focus:** the upstream gateway
(`https://opencode.ai/zen/go`), the proxy's remote streaming/retry machinery
(`proxy_remote.py`), network, and provider-side rate limiting — **not** local
GPU inference. See AC5.

---

## AC1 — Is the local LLM invoked on the critical path of a responses call?

**Answer: No — not for a request that is actually answered by the remote
Responses API provider.**

### Why

A remote responses call is only dispatched after every local provider in the
chain has been *bypassed or denied*. The relevant logic is in
`_proxy_with_fallback_cycle` (`provider.py:4726`), which iterates the model's
ordered `providers` list:

- If the provider is `type: local`, the router runs the smart-routing check and
  may call `_dispatch_local` → `proxy_to_local` → local llama-server inference
  (`provider.py:5699`).
- If the provider is `type: remote`, the router calls `ptr_remote` →
  `proxy_to_remote` directly (`provider.py:5741`). This path contains **no**
  local llama-server call.

For a remote call to happen, the local candidate must have been skipped
(`_should_skip_local`, `provider.py:5529`) or denied (`local_dispatch_denied`,
`router.py:1087`). Either way the local model never runs inference for that
request.

### The specific code paths that *would* invoke the local LLM (and why they do not here)

| Path | When it runs | Local inference? |
|---|---|---|
| `_dispatch_local` → `proxy_to_local` (`provider.py:2933`, `router.py:774`) | Local provider selected and lease acquired | **Yes** — but this is a local answer, not a responses call |
| `_evaluate_compaction_for_bypass` → `evaluate_and_apply_compaction` (`provider.py:5153`, `router_helpers.py:2504`) | Only inside the local-provider branch when local is skipped | **No** — summariser is remote; and compaction is disabled |
| `build_local_summarizer` (`compaction_summarizer.py:353`) | Only when `models.compact` is absent | **No** — `models.compact` is present |
| Background backend health / model lifecycle (`backend_health.py`, `lifecycle.py:678-770`) | Periodic task, model load/unload | Not on the request path (may add GPU contention) |

### Empirical confirmation (live log, 2026-09-26)

```
13:43:06 routing_check provider=local-qwen3 ... estimated_tokens=77674 ... new_tokens=77674
13:43:06 routing_economic_bypass_local provider=local-qwen3 ...   # cheap-mode rescue → try local
13:43:06 [local] POST http://192.168.0.199:8000/v1/chat/completions   # request logged by proxy_to_local
13:43:06 local_dispatch_denied session=... owner=... active=3 cold_start=False
13:43:06 [remote] POST ... -> https://api.deepseek.com            # fell through to remote
13:43:08 Stream started: provider=deepseek model=deepseek-flash
```

Two points worth highlighting:

- The `[local] POST` line is **inbound request logging** (`log_request`,
  `router_helpers.py:208`), emitted before the lease check. It is not evidence
  of local inference.
- `local_dispatch_denied` shows the local slot was owned by another session, so
  the local model produced no tokens; the request fell through to the remote
  provider.

The `compaction_bypass_eval ... reason=compaction_disabled` lines confirm the
summariser was never invoked.

---

## AC2 — Full information flow for a Muse Responses API call

### Flow (client → proxy → provider → client)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLIENT (Pi / herdr)                                                            │
│   POST /v1/chat/completions  {model: "plan"|"code", messages: [...]}           │
└───────────────────────────────┬────────────────────────────────────────────────┘
                                │  (all local components below are CPU / Python)
┌───────────────────────────────▼────────────────────────────────────────────────┐
│ PROXY (FastAPI, ui.py:1189)                                                     │
│  1. Body parse + system-prompt compose / message delta      [CPU, no model]     │
│  2. get_model_type() → first provider type (provider.py:1701)                   │
│  3. proxy_with_fallback → _proxy_with_fallback_cycle (provider.py:4726)         │
│     per provider, in order:                                                     │
│       ├─ LOCAL provider:                                                        │
│       │    a. Token estimate (native tokenizer file I/O or tiktoken) [CPU]      │
│       │    b. _should_skip_local / lease check / contention queue   [CPU]       │
│       │    c. compaction eval (disabled) → remote summariser if on  [REMOTE]    │
│       │    d. _dispatch_local → proxy_to_local → llama-server       [GPU]       │
│       │       (only if selected AND lease acquired — no remote call follows)    │
│       └─ REMOTE provider (Muse, api: openai-responses):                         │
│            a. rate-limit check                                      [CPU]       │
│            b. proxy_to_remote (proxy_remote.py:739):                            │
│               • responses_mode → rewrite path to /v1/responses (751-753)        │
│               • _translate_chat_to_responses(body)                  [CPU]       │
│               • build headers, API key from env/auth.json           [CPU/IO]    │
│            c. httpx POST → https://opencode.ai/zen/go/v1/responses  [NETWORK]   │
│            d. streaming: _translate_responses_stream → chat SSE     [CPU]       │
│            e. return to client                                                  │
│  4. post-request: traffic recording (fire-and-forget file I/O)      [CPU/IO]    │
└───────────────────────────────┬────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────────────────┐
│ UPSTREAM: opencode.ai/zen/go (Muse) — NOT local                                 │
└────────────────────────────────────────────────────────────────────────────────-┘
```

### Local components touched, classified

| # | Component | Function / location | Class | On critical path? |
|---|---|---|---|---|
| 1 | Body parse + system-prompt compose | `ui.py:1148-1176`, `prompt_resolver.py` | non-GPU (CPU) | Yes (negligible) |
| 2 | Native tokenizer estimate | `provider.py:331`, `:1444`, `tokenizers.py` | non-GPU (CPU + file I/O) | Yes (negligible) |
| 3 | tiktoken / char estimate | `utils.py:61`, `lifecycle.py:420` | non-GPU (CPU) | Yes (negligible) |
| 4 | Session manager (delta/history) | `router_helpers.py:_handle_session` | non-GPU (CPU + file I/O) | Yes |
| 5 | Smart-routing gate | `provider.py:_should_skip_local` (`:1504`) | non-GPU (CPU) | Yes |
| 6 | Local lease / contention queue | `router.py:1000-1130`, `contention_queue.py` | non-GPU (in-memory) | Yes |
| 7 | Compaction planning | `compaction.py:592,685` | non-GPU (CPU) | Yes; returns disabled |
| 8 | Remote summariser (if enabled) | `compaction_summarizer.py:730,646` | **REMOTE** (Muse/DeepSeek) | No local GPU |
| 9 | Local summariser (fallback only) | `compaction_summarizer.py:353` | GPU/LLM | **Not invoked** (`models.compact` present) |
| 10 | Local completion | `router.py:774`, `provider.py:2933` | GPU/LLM | **Not invoked** (local skipped/denied) |
| 11 | Responses translation | `proxy_remote.py:379,493,571` | non-GPU (CPU) | Yes (pure Python) |
| 12 | Traffic recording | `router_helpers.py:3560`, `session_recorder.py` | non-GPU (file I/O, async) | Fire-and-forget |
| 13 | Metrics / logging | `metrics.py`, `observability.py` | non-GPU (CPU) | Yes (negligible) |
| 14 | Backend health / model lifecycle | `backend_health.py`, `lifecycle.py:678-770` | GPU/LLM-capable | **No** — background loop |
| 15 | Embeddings (`mxbai-embed`) | web UI chat console only (`handlers.py:501-511`) | GPU/LLM | No — not on the API path |
| 16 | Tokenizer data files | `tokenizer_data/qwen3/tokenizer.json` | non-GPU (disk) | Yes (lazy load, cached) |

**Net:** the only local resources on the remote-call path are CPU (tokenizer,
Python transforms, logging). No GPU inference.

### Note on the "conversion"

The brief refers to "responses API → chat API conversion". In code the primary
direction for Muse is **chat → responses** on the outbound hop and
**responses → chat** on the inbound stream:

- `_translate_chat_to_responses` (`proxy_remote.py:379`) — `messages` → `input`,
  `max_tokens` → `max_output_tokens`, `reasoning_effort` → `reasoning.effort`,
  tool-call reshaping.
- `_translate_responses_to_chat` (`proxy_remote.py:493`) — non-streaming reply.
- `_translate_responses_stream` (`proxy_remote.py:571`) — SSE event mapping
  (`response.output_text.delta`, `response.output_item.added`,
  `response.function_call_arguments.delta`, `response.completed`,
  `response.failed`).

All three are deterministic Python; the local model is not used to "convert"
anything.

---

## AC3 — Compaction summariser chain

**Chain (current, all profiles):** remote-only.

```yaml
# proxy/config-cheap.yaml:428-445 (identical shape in config.yaml / config-fast.yaml)
models:
  compact:
    providers:
      - name: opencode-go-compact        # tier 1: Muse via opencode-go
        type: remote
        endpoint: https://opencode.ai/zen/go
        model: muse-spark-1.3-contributor
        api: openai-responses            # ← the ONLY active responses provider
      - name: deepseek-flash-compact     # tier 2 fallback: DeepSeek direct
        type: remote
        endpoint: https://api.deepseek.com
        model: deepseek-flash
```

- Selected programmatically by `build_compact_summarizer`
  (`compaction_summarizer.py:730`), which resolves the chain with
  `resolve_provider` (cooldowns / `available_times` / failure domains honoured)
  and posts each tier via `_post_compact_tier` (`:646`) with its own `httpx`
  client. It does **not** go through the router's `_proxy_with_fallback_cycle`.
- **Local summariser involvement:** none, as long as `models.compact` declares
  at least one provider. `_compact_model_config` (`:567`) returns `None` only
  when `models.compact` is absent, in which case `build_local_summarizer`
  (`:353`, Qwen2.5-7B at `http://localhost:8080/v1/chat/completions`) is used.
  That is a backwards-compatibility path, not the current configuration.
- **Invocation point:** prompt-assembly time — `_handle_session` for local
  dispatch, and `_evaluate_compaction_for_bypass` (`provider.py:5153`) in the
  local-provider branch when local is skipped. For a pure remote-only model
  (e.g. `compact`), the fallback cycle does not evaluate compaction.
- **Enablement state:** `compaction_trigger_ratio: 0` in `config.yaml`,
  `config-fast.yaml`, and `config-cheap.yaml`. `compaction_trigger_tokens`
  returns `0`, so `plan_session_compaction` short-circuits with
  `reason="compaction_disabled"` and **never calls the summariser**. This
  matches the live log (`compaction_bypass_eval ... reason=compaction_disabled`).
  `compaction_dry_run: false` is irrelevant while the trigger is 0.

> **Doc drift note:** `docs/llama-router.md` ("Session compaction config") still
> describes the local Qwen3 summariser and a default ratio of 0.70. The code and
> config now use the remote `models.compact` chain and ratio 0. This investigation
> is the current source of truth; the llama-router section should be refreshed
> when convenient.

---

## AC4 — Token estimation

**Token estimation never uses llama-server inference (GPU). It is CPU-only.**

Two estimators are used, both CPU:

1. **Routing / persistence estimator** — `_estimate_prompt_tokens_for_routing`
   (`provider.py:331`) and `_estimate_effective_prompt_tokens_for_routing`
   (`:1444`). For a model declaring `tokenizer: qwen3`, the native tokenizer
   loads a vendored `tokenizer_data/qwen3/tokenizer.json` via the lightweight
   `tokenizers` library (`tokenizers.py`), counts locally, and forces the
   multiplier to 1.0. Otherwise it falls back to tiktoken
   (`utils.count_text_tokens`, `utils.py:61`), then to a byte heuristic.
2. **Adaptive-timeout estimator** — `lifecycle._estimate_prompt_tokens`
   (`lifecycle.py:420`) is a ~4-bytes-per-token character heuristic, used only
   by `_compute_request_timeout` (`router_helpers.py:3488`).

**Latency characterisation:** negligible relative to a network LLM call. The
tokenizer is loaded once and cached (`functools.cache`); per-request counting is
a single in-process pass over the prompt text. It cannot be a source of
second- or minute-scale latency on a remote responses call.

**Live-log corroboration:** the `routing_check` lines emit
`estimated_tokens=...` for large 60-90K-token prompts within milliseconds of
each other, alongside `[remote]` dispatch — consistent with CPU estimation, and
there is no `llama-server` request between them for requests that route remote.

---

## AC5 — Recommendations

Based on the evidence, the local LLM is **not** the cause of slow Muse
Responses API calls. Redirect the latency investigation to:

1. **Upstream gateway** (`https://opencode.ai/zen/go`). The prior RCA
   (LP-0MTV77DAT0018ZUS) already identified upstream `ReadTimeout`,
   `stall_after_content`, and no auto-retry after content as the dominant
   failure modes. That is the most likely explanation for intermittent slowness.
2. **Proxy remote streaming/retry machinery** (`proxy_remote.py`
   `_handle_remote_streaming`, preflight re-route, Tier-1/Tier-2/Tier-3
   cooldown and failure-domain logic). These add bounded waits and re-routes
   around a slow upstream; measure hops and retry cycles per request.
3. **Network path** between the proxy host and `opencode.ai` (TLS, DNS,
   egress) — `upstream_request_timeout_seconds` (default 120 s) and
   `upstream_idle_timeout_seconds` (240 s) bound stalls, so a slow-but-alive
   upstream shows up as latency, not error.
4. **Provider rate limiting / account tiers** (opencode-go 429s, `available_times`
   windows). The `opencode-go*` siblings share a gateway; quota exhaustion
   forces retries/skips that look like slowness.

**Findings that redirect away from the local LLM:**

- The responses translation is deterministic Python (AC1/AC2).
- The compaction summariser has no local tier and is disabled (AC3).
- Token estimation is CPU-only and negligible (AC4).
- For remote-routed requests the live log shows `local_dispatch_denied` /
  `context_too_large` *before* the remote dispatch — the local model produced
  no tokens.

**Residual caveats / where local can still matter:**

- **Background GPU contention.** The model-health loop
  (`llama_model_health_interval_seconds: 10`) and model load/unload
  (`lifecycle.py`) run independently of the request. On this Strix Halo APU the
  GPU-wedge detector is disabled (pinned busy counter,
  `llama_gpu_wedge_detection_enabled: false`), but a load/unload cycle triggered
  by another session could transiently occupy the GPU. This affects *local*
  requests, not the remote responses call, but a co-located heavy local workload
  could saturate shared resources (CPU, disk, RAM) and indirectly slow the
  proxy process. If slow Muse calls correlate with active local generation,
  check per-session local activity.
- **Config-dependence.** If Muse is re-enabled in `plan`/`author`/`code`
  (currently commented out) the model becomes local-first; requests small/warm
  enough to pass the routing gate will be answered by Qwen3, and only
  skipped/denied ones reach Muse. That still does not put local inference on
  the Muse path. If `models.compact` were removed, the local summariser would
  come back into play when compaction is enabled — but only at prompt-assembly
  time, and only when the trigger ratio is non-zero.

---

## References

- Related: LP-0MTV77DAT0018ZUS (Muse mid-stream stall RCA —
  `docs/muse-stall-rca-LP-0MTV77DAT0018ZUS.md`), LP-0MTT0O74N009E7N2
  (remote-only compaction chain), LP-0MTGK5DQO001Y8H0 (Responses API
  translation), LP-0MSEQ71IF0003FRT (native tokenizer).
- Config: `proxy/config.yaml`, `proxy/config-fast.yaml`, `proxy/config-cheap.yaml`.
- Code: `proxy/proxy/proxy_remote.py`, `provider.py`, `router.py`,
  `router_helpers.py`, `compaction.py`, `compaction_summarizer.py`,
  `tokenizers.py`, `utils.py`, `lifecycle.py`.
- Existing behavioural tests backing these claims:
  `proxy/tests/test_compaction_compact_summarizer.py::test_never_calls_local_llama_server`,
  `::test_compact_model_declares_remote_chain`,
  `::test_local_fallback_when_no_compact_model`;
  `proxy/tests/test_responses_api_translation.py`;
  `proxy/tests/test_qwen3_tokenizer.py`;
  `proxy/tests/test_compaction_bypass.py`.
