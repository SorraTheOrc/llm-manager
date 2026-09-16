# RCA: Muse (opencode-go) Mid-Stream Stalls — LP-0MTV77DAT0018ZUS

**Date:** 2026-09-11  
**Work Item:** LP-0MTV77DAT0018ZUS  
**Status:** RCA complete — fix proposal attached  
**Evidence:** Live proxy.log (2026-09-11), proxy/config.yaml, proxy/proxy/proxy_remote.py, proxy/proxy/provider.py, proxy/proxy/stall_circuit_breaker.py

---

## AC1 — Error Taxonomy & Evidence

### Error classes observed in current proxy.log (2026-09-11)

| # | Error class | Count | Exemplar log line | Owner |
|---|---|---|---|---|
| 1 | **Upstream ReadTimeout** | 33 | `WARNING - Upstream ReadTimeout session=herdr-1789093098-304094-13583 provider=opencode-go model=muse-spark-1.3-contributor` | Proxy Tier-1 retry |
| 2 | **stall_after_content** | 8 | `Stream finished: reason=error error_type=stall_after_content error_message=Upstream idle timeout after content delivered (240s no data)` | **Proxy — terminal, no auto-retry** |
| 3 | **stream_exception (RemoteProtocolError)** | 1 | `Stream finished: reason=error error_type=stream_exception error_message=Local stream error (RemoteProtocolError)` | Local llama-server |
| 4 | **context_too_large** | 2 | `Fallback triggered for model=v1/chat/completions, from=local-qwen3, to=opencode-go-2, reason=context_too_large` | Routing fallback |
| 5 | **max retries exhausted** | 2 | `Upstream stall: max retries exhausted session=herdr-1789093098-304094-13583 provider=opencode-go model=muse-spark-1.3-contributor retries=4` | Proxy Tier-1 exhaust → Tier-3 circuit breaker |
| 6 | **failure.domain skips** | 30 | `Skipping provider=opencode-go-3: same failure domain as an already-failed entry (domain=https://opencode.ai/zen/go:muse-spark-1.3-contributor)` | Tier-2 failure-domain grouping |

### No `stall_exhausted` in current log

The `stall_exhausted` path (line 1500 of `proxy_remote.py`) fires only after **3 idle retries** within the streaming loop. In practice, the pre-content `ReadTimeout` path (line 1864) is hit first (httpx ReadTimeout fires at `upstream_idle_timeout_seconds: 240`), which sets `_terminate_after_content = True` when `_has_content` is already true. So stalls in production always hit the `stall_after_content` path, not `stall_exhausted`.

### Error flow diagram

```
Request → local-qwen3 (cold/warm threshold) → opencode-go-3 → opencode-go-2 → opencode-go → deepseek-v4-flash (timed)

Per stream:
  240s idle → httpx.ReadTimeout or idle timeout
    → _has_content == True → _terminate_after_content = True
    → _build_stream_error_event(stall_after_content)
    → YIELD terminal error chunk → CLIENT SEES ERROR

  _has_content == False → _should_retry = True
    → Bounded exponential backoff (2/4/8/16s, max 3 attempts)
    → After 3 retries → _build_stream_error_event(stall_exhausted or stall_after_content)
    → StreamingPreContentError → fallback chain

After content delivered: NO retry. Terminal error.
```

---

## AC2 — RCA: Why Manual `continue` is Required

### Root Cause #1: Deliberate no-retry on post-content stalls

**Code location:** `proxy/proxy/proxy_remote.py:_handle_remote_streaming`, ~line 1927

```python
# After-content stall/ReadTimeout: terminate the stream immediately
# with a synthetic finish_reason: error instead of restarting the
# whole request (LP-0MS9FR9LG002AJ4C).
if _terminate_after_content:
    ...
    _final_error_obj = _build_stream_error_event(
        error_type="stall_after_content",
        suggested_action="Retry the request with full context, or route to a healthier provider",
    )
    yield _final_error_bytes
    break   # ← exits the streaming generator; no retry loop, no fallback
```

This is **intentional design** (LP-0MS9FR9LG002AJ4C). The rationale was: after content is delivered, re-dispatching risks duplicate tool calls and billing. However, this creates a **dead end** — the error chunk reaches the client (Pi/herdr) as a terminal event, and neither the proxy nor the client auto-retries.

### Root Cause #2: Failure domain eliminates all Muse siblings

**Code location:** `proxy/proxy/provider.py:_should_fallback`, ~line 1544

When a single opencode-go entry fails, the failure domain mechanism (LP-0MSG45I8Q0020N1F) groups all three opencode-go entries by their canonical key:

```
domain = "https://opencode.ai/zen/go:muse-spark-1.3-contributor"
```

All 30 failure-domain skip logs confirm: **one entry failure kills all three** because they share the same endpoint + model. This is correct behavior (same upstream gateway), but it means there is **no healthy Muse sibling** to fall back to.

### Root Cause #3: DeepSeek is gated by `available_times`

**Config:** `proxy/config.yaml` line 268

```yaml
- name: deepseek-v4-flash
  provider: deepseek
  endpoint: https://api.deepseek.com
  model: deepseek-v4-flash
  available_times: ["00:00-01:00", "04:00-06:00", "10:00-00:00"]
```

At the time of the observed stalls (08:00–09:50 UTC), DeepSeek was **outside its available window** (`00:00-01:00`, `04:00-06:00`, `10:00-00:00` means the gap `06:00-10:00` is excluded). With all Muse entries in the same failure domain and DeepSeek unavailable, the fallback chain is **effectively exhausted**.

### Root Cause #4: Context magnifies the failure

**Config:** `proxy/config.yaml` line 614 — `compaction_dry_run: true`

Compaction is in dry-run mode: advisory logging only, zero dispatch change. Sessions grow unchecked (62k → 125k tokens observed in related items).

**Effect:** Large prompts (`estimated_tokens > cold_threshold 38000`) are forced to remote providers via the warm/cold threshold routing in `provider.py:2853`. The local Qwen3 cannot handle large-context requests, so every large prompt goes directly to Muse with no local fallback. The compaction dry-run means the prompt never gets summarized, so context keeps growing, and every new request hits Muse again.

**Evidence:** `routing_check` logs show `estimated_tokens` climbing and `cached_ratio: 1.00` — prompts are large enough that the cached context dominates, but compaction never fires.

---

## AC3 — Fix Proposal

### Fix A: Auto-recovery for post-content stalls (Proxy-side)

**Owner:** Proxy  
**Complexity:** Medium  
**Risk:** Medium (billing/duplicate content concerns)

**Proposal:** When `stall_after_content` is emitted, instead of yielding a terminal error chunk, **buffer the partial content and re-dispatch** to the next healthy provider.

**Implementation approach:**

1. When `_terminate_after_content` is set, instead of breaking immediately:
   - Store the buffered content chunks (`collected_chunks`)
   - Mark the stream as "recoverable" (new error type: `stall_after_content_recoverable`)
   - Raise `StreamingRecoverableAfterContentError` (new exception, sibling to `StreamingPreContentError`)
   - The fallback chain catches it, discards the partial content prefix (or replays it), and re-dispatches to the next provider

2. **Idempotency guard:** The re-dispatch uses the **full request context** (not just the partial response), so if the upstream somehow processes both requests, the client receives a complete answer and can discard partials. The partial content is never yielded to the client until the final answer is ready.

3. **Alternative (simpler):** Pi-side auto-`continue` on `error_type: stall_after_content`. The proxy emits the error as today; Pi/herdr detects the error type and automatically issues a retry with the same context. This requires no proxy changes.

**Config touch points:** None required (default behavior change). Could add `server.auto_recovery_stall_after_content: true` for opt-in.

**Trade-offs:**
- (+) Fully automatic — no operator input needed
- (+) Uses existing fallback chain mechanics
- (+) Logs `suggested_action` and increments stall metrics (observability maintained)
- (−) Partial content already sent to client must be handled (discard or replay)
- (−) Risk of duplicate billing if upstream processes both requests (mitigated by full-context re-dispatch)

### Fix B: Context mitigation — enable live compaction

**Owner:** Proxy + operator  
**Complexity:** Low  
**Risk:** Low

**Proposal:** Flip `compaction_dry_run: false` in `proxy/config.yaml`.

**Rationale:** Dry-run compaction generates advisory warnings but never actually compacts the session. The session context grows unchecked, forcing more requests to large-context remote providers. With live compaction enabled, sessions will summarize when `estimated_tokens > 0.70 × per_slot_threshold`, keeping context within the local model's capacity.

**Config change:**
```yaml
# proxy/config.yaml line 614
compaction_dry_run: false   # was: true
```

**Trade-offs:**
- (+) Directly addresses the context magnification root cause
- (+) Low risk — compaction already tested; dry-run mode is the experiment phase
- (−) Summarization loses some detail (trade-off is inherent to compaction)
- (−) May affect prompt-sensitive tasks that rely on full context

### Fix C: Expand DeepSeek available_times (optional, operator decision)

**Owner:** Operator  
**Complexity:** None  
**Risk:** Low (billing)

If the operator wants to widen DeepSeek's availability window, the `available_times` for the `deepseek-v4-flash` entries (lines 223, 239, 255, 268, 291, 307, 324, 340, 358, 402) can be expanded. This would provide a healthy fallback when Muse is in cooldown, reducing the "chain exhausted" problem.

**Example:** `"00:00-23:59"` (always available) or `"06:00-23:59"` (remove the 06:00–10:00 gap observed during the stall window).

---

## AC4 — Verification Plan

1. **Shadow replay:** Replay the reference session `herdr-1789093098-304094-13583` through the modified proxy with a test config that adds a 4th opencode-go entry or widens DeepSeek availability. Verify that `stall_after_content` triggers auto-fallback instead of terminal error.

2. **Synthetic stall test:** Inject a mock upstream that sends content then sleeps for >240s. Verify:
   - With Fix A (proxy): the stream continues on the next provider; no terminal `stall_after_content` error
   - With Fix A (Pi-side): the `error_type: stall_after_content` triggers auto-continue
   - With Fix B (compaction): `estimated_tokens` stays below `cold_threshold`, requests route to local

3. **Context growth test:** Enable `compaction_dry_run: false`, run a long session, verify `compaction_compacted_total{reason="trigger_ratio"}` increases and `estimated_tokens` stays bounded.

---

## AC5 — Documentation

This document serves as the primary RCA. Changes from this fix (compaction flip, auto-recovery code) will be tracked via follow-up work items.

---

## Summary of Findings

| # | Finding | Root cause | Fix priority |
|---|---|---|---|
| 1 | Proxy emits terminal error on post-content stalls, no auto-retry | `_terminate_after_content` → terminal yield, not re-dispatch | **High — Fix A** |
| 2 | All 3 opencode-go entries share failure domain → all skipped | Correct behavior, but eliminates healthy siblings | Addressed by Fix A (fallback to DeepSeek/local) |
| 3 | DeepSeek unavailable during observed stall window | `available_times` excludes 06:00–10:00 UTC | **Medium — Fix C** |
| 4 | Compaction dry-run → context grows unchecked → more remote calls | `compaction_dry_run: true` | **High — Fix B** |
| 5 | Pi/herdr requires manual `continue` to retry | No auto-continue logic for `stall_after_content` | Addressed by Fix A (proxy or Pi-side) |

## Recommended Action Order

1. **Flip `compaction_dry_run: false`** (Fix B) — low risk, addresses context root cause
2. **Implement auto-recovery for `stall_after_content`** (Fix A) — proxy-side re-dispatch or Pi-side auto-continue
3. **Review `available_times` for DeepSeek** (Fix C) — operator decision

## Follow-up Work Items

- **Fix B (compaction):** Can be implemented as a one-line config change; create a chore task.
- **Fix A (auto-recovery):** Requires code changes. Create a feature task for proxy-side re-dispatch or a task for Pi-side auto-continue.
- **Fix C (DeepSeek window):** Operator decision; no code change required.

---

## Related Work

| Item | Title | Status | Relevance |
|------|-------|--------|-----------|
| LP-0MSF1PUM90099ZSW | Fall back on mid-stream stalls | completed/done | Pre-content mid-stream re-routing (completed) |
| LP-0MSDRRDWK009QT4E | Silent continue for pre-content failures | completed/done | Pre-content recovery (completed) |
| LP-0MTPMF03P0046MFG | Switch sibling after repeated failures | completed/done | Sibling fallback (completed) |
| LP-0MTV5GFL8007WSK2 | Fix compaction dry-run logging | completed/in_review | Compaction system |
| LP-0MS9FR9LG002AJ4C | _terminate_after_content design | — | The design that causes this problem |
| LP-0MSG45I8Q0020N1F | Failure domain grouping | — | Same-gateway sibling exclusion |
| LP-0MRFEXXVC001RYKB | Stall circuit breaker (Tier-3) | — | Cross-request stall tracking |
