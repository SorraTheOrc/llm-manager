# Error analysis: llama-proxy, 2026-08-03

**Work item:** Aug 3 error taxonomy & root-cause report (LP-0MSDP2P6S007ZDMY)
**Parent:** Analyze opencode-go errors and find a better handling strategy (LP-0MSDFKCK4007CPMY)
**Date:** 2026-08-03
**Data:** `/var/log/llama-proxy/proxy.log*`, window `2026-08-03 00:00:00 → 2026-08-04 00:00:00` (355 error events; see `proxy/docs/error-analysis-2026-08-03/` for the reproducible extraction, F1)

## Headline numbers

| Error class | Count | Client-visible? |
|---|---|---|
| `Stream finished: reason=error` (synthetic SSE) | **127** | **Yes** — the "unspecified error" |
| `[remote] upstream error` (HTTP) | 112 | No — triggers fallback chain |
| `backend_retry` timeouts | 93 | No (retry in progress) |
| `slot_save failed` (local) | 17 | Indirect (degraded local slots) |
| `Stream error:` (proxy-side exception) | 6 | Yes (same events as 6 of the 127) |
| **Total** | **355** | |

> The 6 `Stream error:` events (local/Qwen3, NameError/RemoteProtocolError) are
> the same incidents as 6 of the 127 `Stream finished: reason=error` events:
> the proxy logs the exception warning and then emits the synthetic finish
> event. So the **client-visible stream-error count is 127**, not 133.

## Error taxonomy

### 1. Mid-stream stalls after content delivered — 120 of 127 stream errors (root cause #1)

**Count:** 120 `Upstream stall after content delivered: terminating stream without retry` warning lines; split 92 `opencode-go/deepseek-v4-flash` + 28 `opencode/deepseek-v4-flash-free`.

**Evidence (representative lines):**

```
2026-08-03 10:13:14,158 - WARNING - Upstream stall after content delivered: terminating stream without retry session=019fc52e-... provider=opencode model=deepseek-v4-flash-free
2026-08-03 13:58:xx - WARNING - Upstream stall after content delivered: ... provider=opencode-go model=deepseek-v4-flash
```

**Root cause (confidence: HIGH):** the content-aware per-stream retry policy
(LP-0MS9FR9LG002AJ4C, `_terminate_after_content` in
`proxy/proxy/proxy_remote.py` L1216) detects an upstream idle timeout
(`server.upstream_idle_timeout_seconds`, default 120s) after at least one
content-bearing chunk (`content`, `tool_calls`, or `reasoning_content`) was
already forwarded to the client. It deliberately **terminates immediately
with a synthetic `finish_reason: error` instead of re-sending the whole
request** (a re-send would duplicate output and re-bill the multi-hundred-KB
prompt). The upstream (opencode.ai zen) pauses >120s mid-stream on large
contexts / long reasoning chains; the proxy's stall detector fires while the
upstream is still alive.

**Impact:** this is the dominant client-visible failure (94% of stream
errors). Each one stops the agent session with an unspecified error.

### 2. Pre-content stall, retries exhausted — ~3 of 127 (minor)

**Evidence (evidence.txt `stream_finish_error` section, session/provider/model):**

```
2026-08-03 … - WARNING - Upstream stall: max retries exhausted … session=019fc… provider=opencode-go model=deepseek-v4-flash
```

3 `Upstream stall: max retries exhausted` warning lines, all
`opencode-go/deepseek-v4-flash` (session+entry evidence in
`proxy/docs/error-analysis-2026-08-03/errors.csv` rows with `error_detail`
`stall_exhausted`).

**Root cause (confidence: MEDIUM):** Tier-1 stall retries (`_retry_count >=
max_retries`, default 3; `proxy_remote.py` L863) exhausted before any content
was delivered. The upstream repeatedly stalled on connect/read.

**Related work items:** the content-aware retry boundary
(LP-0MS9FR9LG002AJ4C) and upstream cooldown handling
(LP-0MRGU0I91006ODFD) directly target this window; recovery-first
recommendation LP-0MSDP2PDB004GV86 quantifies the avoidance (see below).

### 3. Local stream exceptions — 6 (NameError ×3, RemoteProtocolError ×3)

**Evidence (evidence.txt `stream_error` section — full 6 lines):**

```
2026-08-03 00:00:56,367 - WARNING - Stream error: session=019fc48a-… provider=local model=Qwen3 error=NameError
2026-08-03 12:47:13,378 - WARNING - Stream error: session=019fc754-… provider=local model=Qwen3 error=NameError
2026-08-03 12:54:59,822 - WARNING - Stream error: session=f04af558-… provider=local model=Qwen3 error=NameError
2026-08-03 17:01:35,360 - WARNING - Stream error: session=019fc83a-… provider=local model=Qwen3 error=RemoteProtocolError
2026-08-03 17:01:35,363 - WARNING - Stream error: session=019fc831-… provider=local model=Qwen3 error=RemoteProtocolError
2026-08-03 17:01:35,368 - WARNING - Stream error: session=019fc312-… provider=local model=Qwen3 error=RemoteProtocolError
```

**Root cause (confidence: MEDIUM):** local llama-server stream exceptions
(`proxy_remote.py` L1157 generic-exception site / `router.py` L1315 local
path). `RemoteProtocolError` at 17:01:35 ×3 in three sessions suggests a
llama-server connection drop (possibly restart or slot churn); `NameError` at
00:00/12:47/12:54 is a proxy-side code bug worth investigating
(see follow-up recommendation R4 / LP-0MSDRRPV0001TCLX).

**Related work items:** the local ctx-size increase
(LP-0MSAOQTJS000FFVM) addresses the slot pressure that degrades local
availability; follow-up tuning work item LP-0MSDRRPV0001TCLX tracks the
`NameError` investigation.

### 4. Upstream HTTP errors — 112 (handled by fallback, not client-visible)

| Status | Count | Type | Affected URLs |
|---|---|---|---|
| 400 | 89 | `invalid_request_error` — *"reasoning_content in thinking mode must be passed back"* | opencode.ai/zen/go ×68, api.deepseek.com ×20, opencode.ai/zen ×1 |
| 402 | 19 | `server_error` — *"Upstream request failed"* | opencode.ai/zen/go ×12, opencode.ai/zen ×7 |
| 429 | 4 | `FreeUsageLimitError` — *"Rate limit exceeded"* | opencode.ai/zen ×4 |

**Root cause (confidence: HIGH):**
- **400:** request-shape incompatibility — `reasoning_content` preservation
  (LP-0MSCGTYWA006NAZC) is not accepted by some upstream endpoints in
  thinking mode. These are request-level errors that the fallback chain
  (`provider.py` `_is_http_error_status`) absorbs by routing to the next
  provider; they do **not** surface to the client as stream errors, but they
  burn provider chain slots and add latency.
- **402:** upstream billing/quota rejection (opencode zen go tier).
- **429:** `FreeUsageLimitError` — the 3-hour per-model cooldown
  (LP-0MRGU0I91006ODFD) suppresses repeat fallbacks to the exhausted free
  tier (4 occurrences vs 112 upstream errors; cooldown working).

#### 4a. `reasoning_content` round-trip repair (LP-0MSGU3JNU0092AFQ)

Remote thinking-mode providers (Console `opencode.ai/zen`, Console Go
`opencode.ai/zen/go`, `api.deepseek.com`) require the `reasoning_content`
field to be present on assistant messages in multi-turn requests. The client
(opencode) drops the **empty** `reasoning_content: ""` that the upstream
emitted on tool-call-only turns, so the field is entirely absent on those
messages when the history is re-sent. The upstream rejects the whole request
with HTTP 400: *"The `reasoning_content` in the thinking mode must be passed
back to the API."*

**Fix (deployed 2026-08-06):**

1. `proxy_remote.py::_sanitize_remote_messages` now injects
   `reasoning_content: ""` (matching upstream emission) on every assistant
   message where the field is missing or `null` — additive-only; existing
   values are never touched. This makes the forwarded payload exactly match
   what the upstream itself produced, eliminating the validation trigger.
2. `provider.py` intercepts the specific 400 in both `proxy_with_fallback`
   and `proxy_with_remote_fallback`: when all providers are exhausted and
   the first error is this reasoning_content round-trip 400, the proxy
   returns a synthetic JSON error (`code: reasoning_content_roundtrip`,
   `suggested_action` remediation) instead of the raw upstream body — the
   error never reaches the client as an opaque upstream failure.

**Probe verification (2026-08-06):** `reasoning_content: ""` is accepted by
all three endpoints (opencode.ai/zen, opencode.ai/zen/go,
api.deepseek.com), including the exact recorded 69-message production payload
on api.deepseek.com. The missing-field shape was also accepted at probe time,
indicating the upstream rejection was a strict/transient validation state;
the injection removes the risk entirely by matching upstream emission.

### 5. `backend_retry` timeouts — 93 (transient, absorbed)

| Error | Count | Signal |
|---|---|---|
| `ReadTimeout` | 75 | `timeout_failures` |
| `ConnectTimeout` | 13 | `connect_failures` |
| `ReadError` | 5 | `read_failures` |

**Evidence (evidence.txt `backend_retry` section — sample lines):**

```
2026-08-03 00:01:24,113 - WARNING - backend_retry path=v1/chat/completions stream=True attempt=4/8 delay=1.818s signal=timeout_failures error=ReadTimeout
2026-08-03 00:15:59,668 - WARNING - backend_retry path=v1/chat/completions stream=True attempt=1/8 delay=0.208s signal=timeout_failures error=ReadTimeout
2026-08-03 00:25:05,545 - WARNING - backend_retry path=v1/chat/completions stream=True attempt=1/8 delay=0.251s signal=timeout_failures error=ReadTimeout
```

Full 93 lines in `proxy/docs/error-analysis-2026-08-03/evidence.txt`.

**Root cause (confidence: MEDIUM):** upstream connect/read timeouts during
the Tier-2 retry backoff (`backend_retry ... attempt=N/8`). Transient unless
clustered; correlated with the opencode-go stalls above (upstream unresponsive
for extended periods).

**Related work items:** the upstream cooldown / retry-accounting work item
LP-0MRGU0I91006ODFD governs the Tier-2 backoff; the content-aware retry
boundary LP-0MS9FR9LG002AJ4C stops retrying once content is delivered (the
dominant stall case).

### 6. `slot_save failed` — 17, all `ReadTimeout/ReadTimeout`

**Evidence:** 17 WARNING lines between 13:39–18:00, e.g.
`slot_save failed slot=2 error=ReadTimeout/ReadTimeout`.

**Root cause (confidence: MEDIUM-HIGH):** local llama-server slot
persistence (KV-cache save) times out under context pressure. Related work
item **LP-0MSAOQTJS000FFVM** (increase local ctx-size) is the tracked fix.
Degrades local availability and pushes sessions to remote providers
(compounding the fallback load that drives #1/#4).

## Quantified impact of the proposed strategies

Assessed against the 127 client-visible stream errors (the operator-facing
failure).

### Strategy A — Recovery-first silent continue (LP-0MSDP2PDB004GV86)

Re-route to the next healthy provider on failure, respecting the
content-delivered boundary.

| Window | Aug 3 count | Safe to re-route? | Avoided |
|---|---|---|---|
| Pre-content stall, retries exhausted (#2) | ~3 | **Yes** — no content delivered, re-send is cheap | 3 |
| Empty-response / non-SSE retry paths (sites 1,2,4,5) | ~1-4 (gap window) | **Yes** | ~1-4 |
| **After-content stall (#1)** | **120** | **No** — content already sent; re-routing duplicates output | 0 |
| Local generic exceptions (#3) | 6 | **No** (mid-stream) | 0 |
| **Total avoided** | | | **~4–7 of 127 (3–6%)** |

> **Key finding:** recovery-first alone solves at most ~3–6% of the Aug 3
> stream errors, because the dominant failure (120/127) happens *after*
> content is delivered — the boundary where re-routing is unsafe by design
> (LP-0MS9FR9LG002AJ4C). Recovery-first is still worth implementing for the
> pre-content window (empty responses, connect failures, pre-content stalls),
> but it is **not** the main fix for the observed failure mode.

### Strategy B — Informative-error SSE fallback (LP-0MSDP2PH20079WQ7)

Enrich the synthetic `finish_reason: error` event with a structured
`error` payload (`type`, `message`, provider, model, suggested action).

| Outcome | Aug 3 count | Coverage |
|---|---|---|
| Client-visible stream errors that would carry the enriched payload | **127 / 127** | **100%** |

> Every one of the 127 synthetic error events is emitted by one of the eight
> audited sites (LP-0MSDP2P9X002A12I) that can attach provider/model/error
> type with a shared helper. The client continues to see `finish_reason:
> error` (backward compatible) but now with actionable detail — e.g. "stall
> after content delivered (ReadTimeout), provider opencode-go/
> deepseek-v4-flash, retry with full context".

### Combined recommendation

- **Recovery-first for the pre-content window** (bounded re-route to the
  next provider; ~3–6% of Aug 3 errors) — cheap, correct, prevents session
  stops on empty responses/connect failures.
- **Informative-error fallback for everything else** (100% coverage) — the
  essential fix for the after-content stalls that dominate Aug 3, plus a
  guidance channel for every other failure class.
- **Address the underlying stalls** (config/tuning): raise
  `upstream_idle_timeout_seconds` above 120s or add upstream heartbeats for
  long-reasoning models; investigate the local `NameError` (R4) and ctx-size
  pressure (LP-0MSAOQTJS000FFVM).

## Linkage to related work items

- LP-0MRGU0I91006ODFD — FreeUsageLimitError 3-hour cooldown (working; 429s absorbed)
- LP-0MS9FR9LG002AJ4C — content-aware per-stream retry (root cause #1 policy)
- LP-0MSAOQTJS000FFVM — local ctx-size evaluation (root cause #6)
- LP-0MSCGTYWA006NAZC — reasoning_content preservation (root cause #4, 400s)
- LP-0MSDP2P3E0053WOD — reproducible extraction harness (this report's data)
- LP-0MSDP2P9X002A12I — SSE emission-site audit (this report's site map)

## Artifacts

- `proxy/docs/error-analysis-2026-08-03/` — harness outputs: `errors.csv`,
  `counts.csv`/`counts.json`, `evidence.txt`, `summary.md` (F1)
- `proxy/docs/sse-error-emission-audit.md` — emission-site audit (F3)

---

# Recommendation 1 — Recovery-first silent continue (LP-0MSDP2PDB004GV86)

## Goal

When an upstream provider fails mid-stream, the proxy first tries to
**recover automatically** — re-route to the next healthy provider — so the
agent session does not stop and the operator does not have to type
"continue". Recovery is bounded and content-aware; when recovery is
impossible the proxy escalates to the informative error (Recommendation 2).

## Safe vs unsafe recovery windows (content-delivered boundary)

Reuse the existing content-delivered boundary (LP-0MS9FR9LG002AJ4C,
`_has_content` / `_terminate_after_content` in `proxy_remote.py`).

| Window | When | Safe to re-route? | Rationale |
|---|---|---|---|
| Pre-content failure | No `content`/`tool_calls`/`reasoning_content` chunk forwarded yet (audit sites 1, 2, 4, 5 and pre-content stalls at site 3) | **Yes** | Re-sending the request is cheap (no duplicate output); the client has not received any assistant content, so the re-routed response is indistinguishable from a first attempt |
| After-content failure | At least one content-bearing chunk already forwarded (audit site 6/7 — the Aug 3 dominant case, 120/127) | **No** | Re-sending would duplicate output already visible to the client and re-bill the full prompt; must terminate (current behaviour) and escalate to the informative error |
| Non-timeout stream exception | `RemoteProtocolError`, `NameError` etc. (audit sites 6/8) | **Only if zero content** | Same boundary applies; non-timeout errors after content must not re-route |

## Bounded retry / re-route policy

- **Re-route attempts:** cap re-routes per request at the number of
  *remaining* providers in the configured chain (`proxy/config.yaml`
  `providers` list) — never loop back to an already-failed provider in the
  same request. Aug 3 chains are 4-5 providers long (local-qwen3 →
  opencode-deepseek-free → opencode-go-2-deepseek → opencode-go-deepseek →
  deepseek-v4-flash); a re-route consumes at most 4 extra attempts.
- **Per-provider cooldown:** reuse the existing Tier-2/Tier-3 mechanisms
  (`mark_provider_unavailable`, `_check_stall_circuit_breaker`,
  LP-0MRFEXXVC001RYKB, LP-0MRGU0I91006ODFD) — a provider that failed
  mid-stream enters cooldown so a later request does not immediately try it
  again. Aug 3 evidence: 429 cooldown already suppressed repeat fallbacks
  (4 events); the same pattern extends to stall/ReadTimeout failures.
- **No infinite loops:** if every provider in the chain fails or is in
  cooldown, stop re-routing and escalate (Recommendation 2).

## Escalation

When recovery is exhausted (all providers failed/in cooldown) **or** the
failure is after-content, emit the informative-error SSE event
(Recommendation 2) — never a silent stream abort.

## Architecture fit (no client-side changes)

- The re-route decision must be made **inside** `_handle_remote_streaming()`
  (proxy_remote.py) or signalled back to the fallback loop
  (`proxy_with_remote_fallback()`, provider.py L1893). Today the streaming
  generator runs detached from the fallback chain once a 2xx
  `StreamingResponse` is returned (`_handle_streaming_success`,
  provider.py L1587); a recovery signal (e.g. a special internal chunk or an
  exception carrying the provider name) would let the fallback loop pick the
  next provider and re-issue the request with the same session.
- Pre-content re-route is a request-level retry on a *different* provider —
  the client sees a normal stream either way. No client cooperation needed.

## Quantified impact (Aug 3)

- **~4–7 of 127 client-visible stream errors avoided (3–6%)** — the
  pre-content window (empty responses, connect failures, pre-content stall
  exhaustion).
- **120/127 NOT avoidable** — after-content stalls; recovery is unsafe there
  by design.
- Cheap to build; prevents session stops in the pre-content window; the
  main value is architectural (it is the prerequisite for the informative
  error escalation path).

## Follow-up implementation work item (spec)

**Title:** Recovery-first silent continue for pre-content mid-stream failures

**Scope:** `proxy/proxy/proxy_remote.py` (signal recoverable pre-content
failures), `proxy/proxy/provider.py` (`proxy_with_remote_fallback` re-route
on signal, bounded by remaining providers + per-provider cooldown),
`proxy/config.yaml` docs.

**ACs (draft):**
1. Pre-content stall/empty-response/connect failure on a remote provider
   re-routes to the next provider in the configured chain without a
   client-visible break, bounded by remaining providers.
2. After-content failures never re-route (content-delivered boundary
   preserved; existing LP-0MS9FR9LG002AJ4C tests still pass).
3. Providers that fail mid-stream enter the existing cooldown/circuit-breaker
   mechanisms; no infinite recovery loops.
4. When all providers fail/cooldown, the request escalates to the
   informative-error event (Recommendation 2) instead of aborting silently.
5. Full test suite passes; new hermetic tests for the re-route decision
   (pre-content vs after-content, bounded attempts, cooldown).

---

# Follow-up — Chain-hold retry for exhausted chains (LP-0MSH94Z7K007VKC9)

## Problem (2026-08-06 09:05–09:08 cluster)

When every provider in a model's fallback chain is unavailable (final model
unreachable), the proxy returned an error to the client immediately. In the
09:05 cluster, Console Go's stall circuit breaker (180s), the free tier's
3-hour cooldown, and the deepseek direct time-window gap left the `plan`
chain with zero redundancy for ~3 minutes, producing 31 "All providers
exhausted" errors. Most chain exhaustion is transient (60s cooldowns, 180s
stall-circuit-breaker cooldowns, 5–10 min time-window edges), so erroring
immediately discarded requests that would have succeeded moments later.

## Mechanism

Both fallback entry points (`proxy_with_fallback` /
`proxy_with_remote_fallback`) now run their provider chain as a CYCLE under a
cycle-hold wrapper. When a cycle exhausts every provider (a distinguishable
`ChainExhaustedError` raised from the exhaustion tail), the wrapper holds the
request for `server.chain_hold_seconds` (default 300) then starts a NEW cycle
from the FIRST provider with fresh per-request state — giving short cooldowns
time to expire. The number of hold-retry cycles is bounded by
`server.chain_hold_max_cycles` (default 3; 0 = infinite); after the bound the
existing exhaustion/error response is returned unchanged.

- Streaming requests receive periodic SSE comment lines
  (`: chain exhausted (<diagnostics>); retrying from <first> in <Ns>`) during
the hold (client surfacing tracked in SA-0MSHAKSEA001LQ6T); non-streaming
requests are held silently.
- A client disconnect aborts the hold promptly (no wasted waiting).
- The hold only defers the exhaustion verdict — successful responses,
provider ordering, and existing cooldown/circuit-breaker behavior are
unchanged. This complements Recommendation 1's recovery-first strategy: it
covers the case where the ENTIRE chain is exhausted (not just one provider)
and Recommendation 2's escalation remains the terminal response after the
bound.

---

# Recommendation 2 — Informative-error SSE fallback (LP-0MSDP2PH20079WQ7)

## Goal

When recovery is genuinely impossible (Recommendation 1 exhausted, or
after-content failure where re-routing is unsafe), the synthetic SSE error
carries a **structured, human/agent-readable payload** instead of an
unspecified `finish_reason: error`. The client keeps working today
(backward compatible) and a future client can render the detail.

## Structured payload schema

Enrich the synthetic event (all 8 audit sites) with an `error` object
inside `choices[0]`:

```json
{
  "choices": [{
    "delta": {},
    "finish_reason": "error",
    "index": 0,
    "error": {
      "type": "stall_after_content",
      "message": "Upstream idle timeout after content delivered (120s no data)",
      "provider": "opencode-go",
      "model": "deepseek-v4-flash",
      "entry": "opencode-go-2-deepseek",
      "suggested_action": "Retry the request with full context, or route to a healthier provider"
    }
  }]
}
```

Field semantics:

| Field | Source | Example |
|---|---|---|
| `type` | Failure class (see below) | `stall_after_content`, `stall_exhausted`, `empty_response`, `stream_exception`, `upstream_http` |
| `message` | Human-readable one-liner with the underlying exception/timeout | "Upstream idle timeout after content delivered (120s no data)" |
| `provider` | The provider that failed (already in scope at every site) | `opencode-go` |
| `model` | The model name (already in scope) | `deepseek-v4-flash` |
| `entry` | Config entry name (already logged; LP-0MSC7F7BG0043TE1) | `opencode-go-2-deepseek` |
| `suggested_action` | Static map from `type` + retry context | see below |

`suggested_action` mapping (static, per failure type):

- `stall_after_content` → "Upstream paused >120s after output started; retry the
  request with full context"
- `stall_exhausted` → "Upstream stalled repeatedly; provider placed in cooldown; the
  next provider in the chain will be used"
- `empty_response` → "Upstream returned no content; retried N times; check upstream
  status or route manually"
- `stream_exception` → "Proxy stream error (NameError/RemoteProtocolError); check
  proxy/llama-server logs"
- `upstream_http` → "Upstream HTTP <status> (<type>); see proxy logs"

## Trigger conditions (used only when recovery is impossible)

1. **Recovery exhausted:** every provider in the chain failed or is in
   cooldown (Recommendation 1 escalation).
2. **Content already delivered:** stall/exception after the content
   boundary (audit sites 6/7, 8-after-content) — re-routing is unsafe, so
   emit the enriched error immediately.
3. **Non-timeout exceptions** where retry is not attempted (audit site 6)
   — enrich instead of a bare finish.

## Backward compatibility

- The client (pi/opencode-go) today reads `choices[0].finish_reason` and
  treats `error` as terminal; adding the `error` object key changes nothing
  for it — no client-side change required (constraint satisfied).
- The event shape `{"choices":[{"delta":{},"finish_reason":"error",
  "index":0, ...}]}` remains valid SSE for any OpenAI-style decoder.
- A future client build can read `choices[0].error` to render the detail.

## Minimal enrichment point

A single shared helper (e.g. `_build_stream_error_event(provider, model,
entry, error_type, message, suggested_action)` in `proxy_remote.py`)
replacing the 8 duplicated dict literals (audit: LP-0MSDP2P9X002A12I). All
sites already have `provider`, `model_name`, `entry`, the exception type, and
retry counters in scope. `router.py` imports the same helper for the local
path. Lowest risk: no per-site logic, no behavioural change to the
`finish_reason` field.

## Quantified impact (Aug 3)

- **127/127 client-visible stream errors (100%)** would carry the enriched
  payload — the complement of the recovery-first avoidance number.
- The dominant class (after-content stalls, 120/127) is exactly the case
  where this fallback is required (recovery is unsafe), so the two
  recommendations compose: recovery-first for the small pre-content window,
  informative error for everything else.

## Follow-up implementation work item (spec)

**Title:** Informative-error SSE payload for synthetic finish_reason:error

**Scope:** `proxy/proxy/proxy_remote.py` (shared helper + 7 sites),
`proxy/proxy/router.py` (1 site), docs.

**ACs (draft):**
1. All 8 synthetic `finish_reason: error` emission sites emit the enriched
   event via one shared helper; event includes `error.type`, `error.message`,
   `error.provider`, `error.model`, `error.entry`, `error.suggested_action`.
2. Existing clients are unaffected: `finish_reason` semantics unchanged;
   no client-side change required.
3. `suggested_action` maps from the failure type (stall/empty/exception/http)
   per the schema above.
4. Full test suite passes; new hermetic tests assert the enriched event
   shape at each site class (unit-level generator tests).
