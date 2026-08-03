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

**Evidence:** 3 `Upstream stall: max retries exhausted` warning lines, all
`opencode-go/deepseek-v4-flash`.

**Root cause (confidence: MEDIUM):** Tier-1 stall retries (`_retry_count >=
max_retries`, default 3; `proxy_remote.py` L863) exhausted before any content
was delivered. The upstream repeatedly stalled on connect/read.

### 3. Local stream exceptions — 6 (NameError ×3, RemoteProtocolError ×3)

**Evidence:**

```
2026-08-03 00:00:56,367 - WARNING - Stream error: session=019fc48a-... provider=local model=Qwen3 error=NameError
2026-08-03 17:01:35,360 - WARNING - Stream error: ... provider=local model=Qwen3 error=RemoteProtocolError
```

**Root cause (confidence: MEDIUM):** local llama-server stream exceptions
(`proxy_remote.py` L1157 generic-exception site / `router.py` L1315 local
path). `RemoteProtocolError` at 17:01:35 ×3 in three sessions suggests a
llama-server connection drop (possibly restart or slot churn); `NameError` at
00:00/12:47/12:54 is a proxy-side code bug worth investigating
(see follow-up recommendation R4).

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

### 5. `backend_retry` timeouts — 93 (transient, absorbed)

| Error | Count | Signal |
|---|---|---|
| `ReadTimeout` | 75 | `timeout_failures` |
| `ConnectTimeout` | 13 | `connect_failures` |
| `ReadError` | 5 | `read_failures` |

**Root cause (confidence: MEDIUM):** upstream connect/read timeouts during
the Tier-2 retry backoff (`backend_retry ... attempt=N/8`). Transient unless
clustered; correlated with the opencode-go stalls above (upstream unresponsive
for extended periods).

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
