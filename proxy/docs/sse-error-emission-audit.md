# Audit: synthetic SSE `finish_reason: error` emission sites

**Work item:** Audit synthetic SSE finish_reason:error sites (LP-0MSDP2P9X002A12I)
**Parent:** Analyze opencode-go errors and find a better handling strategy (LP-0MSDFKCK4007CPMY)
**Date:** 2026-08-03

## Summary

Every client-visible "unspecified error" on the proxy today is the synthetic
SSE event

```json
{"choices":[{"delta":{},"finish_reason":"error","index":0}]}
```

emitted with **no `error` field**, followed by `data: [DONE]`. The client
(pi/opencode-go) treats `finish_reason: error` as a terminal failure with no
payload, so the operator sees an unspecified error and must type "continue".

This document exhaustively maps every code path that emits this event, the
trigger condition, the retry context, whether content had already been
delivered, and the minimal enrichment points for a future informative error
(work item LP-0MSDP2PH20079WQ7).

## Emission sites

All sites emit the same shape:

```python
_final_error_obj = {"choices": [{"delta": {}, "finish_reason": "error", "index": 0}]}
```

### A. `proxy/proxy/proxy_remote.py` — remote streaming path

All seven sites live inside `_handle_remote_streaming()`
(defined at line 510), which runs **after** the fallback chain
(`proxy_with_remote_fallback()` in `provider.py`) has already returned a
`StreamingResponse` to the client. Mid-stream failures therefore **cannot**
trigger the provider fallback chain — the generator synthesises the error
event and stops.

| # | Line | Trigger condition | Retry context | Content delivered? | Notes |
|---|------|-------------------|---------------|--------------------|-------|
| 1 | 774 | Empty-retry (LP-0MRF77A0E0026B9T) returned a **non-streaming / HTTP ≥400** response | Empty-response retry loop (`_empty_retry_count` vs `empty_max_attempts`, default 1) | No (empty response) | "Retry returned a non-streaming response — don't retry further"; yields error + break |
| 2 | 821 | **Empty-response retry exhaustion** (`_empty_retry_count > empty_max_attempts`) | After all empty retries returned empty or failed to connect | No | Logs `Empty upstream response: max retries exhausted` |
| 3 | 863 | **Stall max-retries exhausted** (`_retry_count >= max_retries`, default 3) | Tier-1 stall retry loop (bounded exp. backoff 1s/2s/4s, LP-0MRE52D3C001KP1H) | Possibly (stall detected on current attempt) | Also records stall in Tier-3 circuit breaker (LP-0MRFEXXVC001RYKB) |
| 4 | 1050 | Stream completed normally (`saw_done`/`saw_finish`) but **no content-bearing chunk** and empty retries exhausted | Empty-response retry loop | No | "If no content and retries exhausted, yield synthetic error so the fallback chain can activate" — but the fallback chain is already past |
| 5 | 1092 | `StopAsyncIteration` (upstream closed without `[DONE]`) and **no content** and empty retries exhausted | Empty-response retry loop | No | Same fallback-chain comment |
| 6 | 1157 | **Generic stream exception** (`except Exception as exc`, e.g. `RemoteProtocolError`, `ReadTimeout`) | No retry on non-timeout errors | Possibly | Logs `Stream error: ... error=<ExcName>`; yields synthetic error + break |
| 7 | 1216 | **Stall after content delivered** (`_terminate_after_content`, LP-0MS9FR9LG002AJ4C) | No retry — content-aware policy terminates immediately | **Yes** | "Upstream stall after content delivered: terminating stream without retry"; records Tier-3 stall |

### B. `proxy/proxy/router.py` — local streaming path

| # | Line | Trigger condition | Retry context | Content delivered? | Notes |
|---|------|-------------------|---------------|--------------------|-------|
| 8 | 1315 | **Generic stream exception** in local llama-server streaming (`except Exception as exc`) | No retry; finally-block cleans up and saves slot | Possibly | Logs `Stream error: ... provider=local model=Qwen3 error=<ExcName>`; emits error event + `data: [DONE]` (LP-0MS14PM7I0077MXD) |

### C. `proxy/proxy/provider.py` — fallback chain exhaustion (HTTP, not SSE)

The fallback chain (`proxy_with_remote_fallback()`, line 1893) handles
**pre-stream** failures (HTTP ≥400, connection errors, empty responses). When
every provider is exhausted it returns a buffered HTTP error response
(`_build_exhausted_response()`) — **not** the synthetic SSE event. The chain
does not observe mid-stream `finish_reason: error` events because they are
emitted after the `StreamingResponse` has already been returned to the
client.

## Why the client surfaces an unspecified error

1. The proxy emits `{"delta":{},"finish_reason":"error","index":0}` — the
   delta is empty, `finish_reason` is `error`, and there is **no `error`
   field**, no provider/model, and no message.
2. The proxy then emits `data: [DONE]` (sites 8 in router.py and the local
   path always; the remote path sites terminate the generator after the
   error event, and the caller's stream loop terminates — see
   `proxy/server.py` streaming dispatch).
3. The client's SSE decoder sees a terminal `finish_reason: error` chunk
   with no error payload. It cannot distinguish this from an upstream
   failure, a fallback exhaustion, or a mid-stream stall; it aborts the
   request and reports an unspecified error, requiring the operator to type
   "continue" (per intake interview, rgardler).

## Why mid-stream errors bypass the fallback chain

- `proxy_with_remote_fallback()` (provider.py:1893) iterates providers and
  returns the first **successful** response. A successful streaming response
  is detected by `_handle_streaming_success()` (provider.py:1587) when the
  response is a 2xx `StreamingResponse` — which is true as soon as
  `proxy_to_remote()` returns a generator-backed response.
- From that point, `_handle_remote_streaming()`'s generator runs detached
  from the fallback loop: stalls, empty responses, and stream exceptions
  produce the synthetic SSE error instead of a re-route.
- The only recovery that exists is **within** the generator: Tier-1 stall
  retries (site 3) and empty-response retries (sites 1-2, 4-5), both bounded
  and content-aware (LP-0MS9FR9LG002AJ4C). Once those are exhausted, or
  content was already delivered (site 6), or the exception is non-timeout
  (site 7), the error is terminal for this provider.

## Minimal enrichment points

The AC for the informative-error work item (LP-0MSDP2PH20079WQ7) requires a
single best enrichment point. Findings:

- **A single wrapper is feasible and lowest-risk.** All eight sites build the
  same dict literal. A module-level helper in `proxy_remote.py` (e.g.
  `_build_stream_error_event(provider, model, entry, error_type, message)`)
  that returns `{"choices":[{"delta":{},"finish_reason":"error","index":0,
  "error": {...}}], ...}` would let all sites attach a structured payload
  without per-site logic. `router.py` would import the same helper.
- **Payload candidates already exist in scope at each site:**
  - `provider` / `model_name` / `entry` (function parameters)
  - error type from `type(exc).__name__` (sites 3, 7), `error=ReadTimeout/...`
    (sites 1-2, 4-5), `_terminate_after_content` stall state (site 6)
  - retry counts (`_retry_count`, `_empty_retry_count`)
- **Backward compatibility:** the client today treats `finish_reason: error`
  as unspecified; adding an `error` object key inside the same SSE event
  changes nothing for the current decoder (it reads `choices[0].finish_reason`),
  while a future client can render the structured error. This satisfies the
  "no client-side change required" constraint.
- **Recovery routing** (LP-0MSDP2PDB004GV86) would additionally need the
  generator to signal "recoverable, zero content" to the fallback loop —
  the sites marked *Content delivered? = No* (1, 2, 4, 5) and pre-content
  stalls (3) are the candidates; sites 6 and 7 (post-content / generic
  exceptions after content) are unsafe.

## Evidence from Aug 3 logs

See `proxy/docs/error-analysis-2026-08-03/` (harness output, F1):
127 `Stream finished: reason=error` events (93 opencode-go/deepseek-v4-flash,
28 opencode/deepseek-v4-flash-free, 6 local/Qwen3) are the client-visible
synthetic errors; the proxy-side `Stream error:` lines (6, local Qwen3
NameError/RemoteProtocolError) correspond to site 6 (generic exception) and
site 8 (router.py local path).
