# Stall-gap distribution analysis (LP-0MSF5IAXE005BG33)

Refines the F4 initial raise of `upstream_idle_timeout_seconds` from 120s to
240s (LP-0MSF5I7XN009ENWQ) for parent LP-0MSF1PUM90099ZSW.

## Method

- **Input**: `/var/log/llama-proxy/proxy.log` (post-restart window,
  2026-08-05 02:32 → 03:33) and `proxy/session-recordings/` (raw SSE chunk
  streams per session).
- **Script**: `extract_stall_gaps.py` (this directory), reproducible with:
  `python3 proxy/docs/stall-gap-analysis-2026-08-05/extract_stall_gaps.py`
- **Artifacts**: `stall-gaps.json` (machine-readable summary).

### What is measurable vs by-construction

The proxy's idle detection fires exactly `upstream_idle_timeout_seconds`
after the last upstream chunk (the chunk read is wrapped in
`asyncio.wait_for(timeout=...)`), so the **observed gap is equal to the
configured timeout by construction** (120s for every stall in this window).
What actually varies, and is measured here:

1. **Stream duration** (Stream started → stall detected) — how long a
   request had been running when the stall hit.
2. **Retry outcome** — whether the session continued after the stall
   (upstream slow-but-alive) vs ended in a client-visible error.
3. **Stream composition** from session recordings (reasoning-only vs
   tool-calls-only vs content-committed) — the re-route eligibility signal
   for F2/F3 (zero final content + zero tool_calls ⇒ re-route).

## Results (post-restart window)

| Metric | Value |
|---|---|
| Stall detections | 38 |
| After-content terminations | 16 |
| With Tier-1 retry | 24 |
| Session continued after stall (upstream slow-but-alive) | 33 / 38 |
| Client-visible `Stream finished: reason=error` | 16 |

### Stream duration (Stream started → stall detected), seconds

| | n | min | p50 | p90 | p95 | max |
|---|---|---|---|---|---|---|
| duration | 36 | 120.0 | 138.7 | 417.3 | 429.5 | 547.4 |

The p50 of ~139s reflects the 120s idle timeout plus ~19s of upstream
streaming; the p90/p95 of ~420-430s reflect requests that stalled repeatedly
across retries.

### Split by provider

| Provider | Stalls | After-content | Session continued |
|---|---|---|---|
| opencode-go | 22 | 6 | 22 |
| opencode | 16 | 11 | 11 |

### Stream composition (from session recordings; 22 of 38 sessions have recordings)

| Composition | Count | Re-route eligible? |
|---|---|---|
| tool_calls_only | 16 | No (terminate per Q1) |
| reasoning_only | 6 | **Yes** |
| content_committed | 0 | No (committed) |
| no_recording | 16 | n/a |

**6 reasoning-only streams** are re-route eligible (zero final content, zero
tool_calls). **4 of the 16 client-visible after-content errors** were
reasoning-only stalls that would now be re-routed by F2/F3 instead of
surfacing an error.

## Impact quantification (AC4)

- **(a) 240s raise alone**: direct evidence of "upstream resumed between
  120s and 240s" is unavailable (the proxy terminates at the timeout). Proxy
  evidence: 33/38 stalls occurred in sessions that continued afterward —
  the upstream was often slow-but-alive, supporting a longer timeout.
- **(b) re-route behavior (F2/F3)**: 4 of the 16 client-visible
  `stall_after_content` errors in this window were reasoning-only stalls and
  would have been re-routed to the next provider instead of erroring.

## Recommendation (AC3)

**Keep `upstream_idle_timeout_seconds: 240`.** Rationale:

- Stream-duration p95 (429.5s) exceeds the 120s timeout, and 33/38 stalls
  were in sessions that later continued — many stalls are slow-but-alive
  upstreams rather than dead connections.
- Raising further (e.g. 480s) would slow true-failure detection (up to 8
  minutes of silence) with diminishing returns, since the re-route behavior
  (F2/F3) already rescues reasoning-only stalls without waiting.
- The primary mitigation going forward is the **mid-stream re-route**
  (LP-0MSF1PUM90099ZSW F2/F3), not the timeout value.

## Follow-up

- If post-deployment monitoring shows the 240s value still producing
  client-visible stalls on reasoning-only streams at scale, re-run this
  analysis over a longer window and consider a provider-specific timeout.
  Recorded in parent LP-0MSF1PUM90099ZSW (AC5).
