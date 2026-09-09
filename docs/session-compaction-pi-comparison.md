# Compaction Comparison: Pi vs Proxy

**Work Item:** LP-0MTT890RO000AEQM — Compare Pi and proxy compaction approaches  
**Status:** Planning / Analysis (no implementation)

---

## Executive Summary

Both Pi (the coding agent, `@earendil-works/pi-coding-agent`) and the proxy (`proxy/compaction.py`) implement session compaction to manage context window overflow, but their architectures are fundamentally different:

| Dimension | Pi (agent-side) | Proxy |
|---|---|---|
| **Trigger** | Threshold-based: `tokens > contextWindow - reserveTokens` (default reserve 16384) | Ratio-based: `tokens > trigger_ratio × effective_per_slot_threshold` (default 0.70) |
| **Strategy** | Single: summarize oldest + keep recent N tokens | Dual: fast=truncate, cheap=summarize+truncate |
| **Summarizer** | Built-in structured prompt via LLM call | Optional local llama-server (Qwen3) or none |
| **Retention invariant** | First K entries preserved (by UUID pointer to firstKeptEntryId) | System prompt + first user prompt preserved verbatim |
| **Backstop** | None (compaction throws on failure) | `truncate_backstop()` — drops oldest turns if summary alone over budget |
| **File operations** | Tracked and embedded in summary (`<read-files>`, `<modified-files>`) | Not tracked |
| **Retry** | `retryAssistantCall` with configurable policy | Fail-open (return `""` on any error) |
| **Dry-run** | Not implemented | Yes (`compaction_dry_run` config flag) |
| **Churn tracking** | Not implemented | Yes (`CompactionChurnCollector`, < 1/session/hour target) |
| **Structured logging** | Via `session_compact` / `compaction_start/end` events | Single `compaction_event` log line with full field set |
| **Branch summary** | Separate but analogous feature | Not implemented |
| **Split-turn handling** | Yes — summarizes turn prefix separately | No |
| **Custom instructions** | Yes — `/compact <instructions>` or extension override | No |
| **Previous summary merging** | Yes — incremental updates when a previous summary exists | No (always fresh summary) |
| **Turn boundaries** | Never cuts at tool results; respects entry-level turns | Groups messages into turns via `pair_turns()`, drops whole turns only |
| **Token estimation** | Per-message `estimateTokens()` with role-specific heuristics | Char-based: `len(content) // 4` per message (lightweight) |
| **Abort / cancellation** | `AbortController` with hooks | No abort mechanism |

**Key finding:** The proxy's approach is **simpler by design** (it operates at the HTTP proxy layer, not the agent layer), but **lacks several capabilities that Pi has**, including file operation tracking, retry, custom instructions, and branch summarization. Where the proxy's strategy diverges intentionally (fast mode = truncate, cheap mode = summarize+truncate), those are sound operational choices.

The proxy **does not need to replicate all of Pi's features** — it serves a different role. But the proxy should adopt several of Pi's safeguards (retry, churn tracking, branch summaries) and consider adding file operation tracking.

---

## 1. Pi Compaction Architecture

### 1.1 Trigger Mechanism

**Code:** `shouldCompact(contextTokens, contextWindow, settings)`

Pi's trigger is simple:
```
shouldCompact = contextTokens > (contextWindow - reserveTokens)
```

- **Default reserve:** 16,384 tokens (configurable in `DEFAULT_COMPACTION_SETTINGS`)
- **Context window:** The model's declared `contextWindow`
- **Fires:** Automatically before every assistant response (`_compactBeforeNextAssistantResponse`)
- **Manual entry:** `/compact` command or RPC call

**Key design decision:** Pi checks compaction at two points:
1. Before sending each assistant response (auto)
2. When the agent completes (`agent_end`) — catches responses that were aborted mid-way

This means Pi is **proactive**: it compacts before the context overflows, not after.

### 1.2 Preparation Phase

**Code:** `prepareCompaction(pathEntries, settings)`

The preparation phase is the most complex part of Pi's compaction:

1. **Idempotency check:** If the last entry is already a compaction, return `undefined` (nothing to do)
2. **Find previous compaction:** Search backwards for the last `compaction` entry; its `summary` becomes `previousSummary`
3. **Calculate tokens:** `tokensBefore = estimateContextTokens(buildSessionContext(pathEntries).messages)`
4. **Find cut point:** `findCutPoint(pathEntries, startIndex, endIndex, keepRecentTokens)`
   - Walks backwards from newest entries
   - Accumulates estimated message sizes
   - Stops when accumulated tokens ≥ `keepRecentTokens` (default 20,000)
   - Finds the closest valid cut point (never at tool results)
   - Returns: `firstKeptEntryIndex`, `turnStartIndex`, `isSplitTurn`
5. **Extract messages to summarize:** All entries between `boundaryStart` and `historyEnd`
6. **Extract file operations:** From tool calls in messages + from previous compaction's details
7. **Handle split turns:** If cutting mid-turn, separate the turn prefix for its own summary

### 1.3 Cut Point Detection

**Code:** `findCutPoint()`, `findValidCutPoints()`, `isCutPointMessage()`

Pi's cut point algorithm is sophisticated:

- **Valid cut points:** User messages, assistant messages, bashExecution, custom messages, branchSummary, compactionSummary
- **Invalid cut points:** toolResult (must follow its tool call)
- **Turn boundaries:** If cutting at an assistant message with tool calls, the tool results that follow it are kept
- **Split-turn handling:** If cutting mid-turn, the turn prefix is summarized separately with a specialized prompt (`TURN_PREFIX_SUMMARIZATION_PROMPT`)

This ensures Pi never cuts in the middle of a logical interaction (tool call → result).

### 1.4 Summarization

**Code:** `generateSummary()`, `generateSummaryWithUsage()`, `compact()`

The summarization process:

1. **Prompt construction:**
   - Convert messages to LLM format via `convertToLlm()` (handles custom types)
   - Serialize to text via `serializeConversation()` (prevents the model from continuing the conversation)
   - Wrap in `<conversation>` tags
   - If `previousSummary` exists, use `UPDATE_SUMMARIZATION_PROMPT` (incremental merge)
   - Otherwise use `SUMMARIZATION_PROMPT` (structured format with Goal, Constraints, Progress, Key Decisions, Next Steps, Critical Context)

2. **Max tokens budget:** `min(0.8 × reserveTokens, model.maxTokens)` — caps summary size

3. **Retry mechanism:** `completeSummarization()` wraps the LLM call in `retryAssistantCall()` for transient failures

4. **Validation:** Checks response for tool calls (error if present), checks for truncation (error if `stopReason === "length"`)

5. **File operations:** After summary generation, appends `<read-files>` and `<modified-files>` XML sections

6. **Split-turn summaries:** If `isSplitTurn`, generates a separate turn-prefix summary and merges both into one final summary

### 1.5 Summarization System Prompt

**Code:** `SUMMARIZATION_SYSTEM_PROMPT`

```
You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.
```

### 1.6 Structured Summary Format

**Initial summary:** `SUMMARIZATION_PROMPT` produces:
```markdown
## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Constraints, preferences, or requirements]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, examples, or references needed to continue]

## File Operations
<read-files>
...
</read-files>
<modified-files>
...
</modified-files>
```

**Incremental update:** `UPDATE_SUMMARIZATION_PROMPT` merges new information into the existing structure, preserving all sections and updating progress.

### 1.7 Session Entry Format

Pi stores compaction results as typed session entries:
```typescript
{
  type: "compaction",
  summary: string,
  tokensBefore: number,
  firstKeptEntryId: string,
  usage: { input, output, cacheRead, cacheWrite, totalTokens, cost },
  details: { readFiles: string[], modifiedFiles: string[] },
  isSplitTurn: boolean,
  // ... other session metadata
}
```

### 1.8 Auto-Compaction Flow

**Code:** `agent-session.js:_compactBeforeNextAssistantResponse()`

1. After assistant response completes, check if compaction is needed
2. If in-progress, skip (abort controller)
3. Call `prepareCompaction()` → if no work needed, return early
4. Call `compact()` → generates summary
5. Create compaction entry and save session
6. Emit `session_compact` event to extensions

---

## 2. Proxy Compaction Architecture

### 2.1 Trigger Mechanism

**Code:** `should_compact_session(estimated_tokens, mode, config)`, `compaction_trigger_tokens()`

```python
trigger = compaction_trigger_ratio × effective_per_slot_threshold
should_compact = estimated_tokens > trigger
```

- **Fast mode:** `0.70 × 83,285 = 58,300` tokens (3 slots × 262,144 / 4 headroom)
- **Cheap mode:** `0.70 × 61,440 = 43,000` tokens (2 slots × 262,144 / 4 headroom)
- **Per-slot threshold:** `ctx_size // slots - 4096` (routing clamp)
- **Trigger ratio:** Configurable `server.compaction.trigger_ratio` (default 0.70)
- **Disabled:** When ratio ≤ 0

The trigger is computed at **prompt assembly time** in the routing layer, not at the session layer.

### 2.2 Strategy Per Mode

| Mode | Strategy | Rationale |
|---|---|---|
| **fast** | Truncate (drop oldest whole turns) | Speed priority; remote fallback acceptable |
| **cheap** | Summarize + truncate backstop | Cost avoidance; no remote |

### 2.3 Planning Phase

**Code:** `plan_session_compaction(messages, config, mode, summarizer, estimate_tokens, backstop)`

1. **Check trigger:** If below trigger or disabled → `action="noop"`
2. **Check summarizer:** If `summarizer is None` → `action="remote_with_guidance"` (non-compactable)
3. **Retention set:** Extract system prompts + first user prompt verbatim (AC1 invariant)
4. **Pair turns:** `pair_turns()` groups messages into logical turns (new user message = new turn)
5. **Select recent window:** Walk turns from newest, accumulate tokens until target reached
6. **Summarize middle:** Call `summarizer(middle_messages)` for the folded portion
7. **Build compacted list:** `[retained] + [summary marker] + [recent turns]`
8. **Backstop:** If over target and `backstop=True`, call `truncate_backstop()`

### 2.4 Summarizer

**Code:** `build_local_summarizer(config, llama_port, timeout_seconds)`

The proxy's summarizer is a callable backed by a local llama-server (Qwen3):

```python
body = {
    "model": model_name,          # From config (default "Qwen3")
    "messages": [
        {"role": "system", "content": _SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ],
    "max_tokens": max_tokens,     # From config (default 512)
    "stream": False,
    "temperature": 0.2,
}
```

**Key design decision:** The proxy's summarizer is **optional and fail-open**:
- If unavailable → `action="remote_with_guidance"` (never fails the dispatch)
- If HTTP error → returns `""` (warning logged, dispatch continues)
- No retry mechanism (unlike Pi)
- Short timeout (default 30 seconds)

### 2.5 Summarization System Prompt

**Code:** `compaction_summarizer.py:_SUMMARIZER_SYSTEM_PROMPT` (via `provider.py`)

The system prompt is minimal (compared to Pi's structured format):
```
Summarise the conversation context. Preserve essential instructions, decisions,
and key facts. Keep it concise.
```

This is a **significant difference**: Pi uses a highly structured format (Goal, Constraints, Progress, etc.) while the proxy uses an open-ended prompt. This affects summary quality and downstream usability.

### 2.6 Transcript Formatting

**Code:** `_transcript_for_summarizer(middle_messages)`

The proxy serializes messages into a simple transcript format:
```
user: <content>

assistant: <content>

user: <content>
```

For multi-part content (text + images), it concatenates text parts with newlines.

### 2.7 Backstop (truncate_backstop)

**Code:** `truncate_backstop(compacted_messages, target_tokens, estimate_tokens)`

If the summary path still leaves the session over budget:
1. Find the summary marker to identify the protected region
2. Everything before the marker is never touched (AC2)
3. Drop oldest whole turns from the recent region
4. Return status: `noop` / `dropped` / `exhausted`

### 2.8 Decision Flow

**Code:** `decide_session_compaction(messages, config, mode, ...)`

```
if dry_run mode:
    → plan what WOULD happen, log advisory, NEVER apply
else:
    → plan compaction
    → log event
    → apply if compactable (return compacted messages)
    → remote_with_guidance if non-compactable
    → noop if below trigger
```

### 2.9 Structured Logging

**Code:** `log_compaction_event(plan_result, session_id, ...)`

One log line per compaction event:
```
compaction_event session=abc12345 mode=fast action=compact reason=compacted_within_target
    pre_tokens=62400 post_tokens=35200 turns_summarized=12 turns_dropped=0
    summary_tokens=480 dry_run=false
```

Log level: INFO for tidy compactions, WARNING for lossy events (backstop drop, exhaustion, remote fallback).

---

## 3. Side-by-Side Comparison

### 3.1 Functional Parity Matrix

| Feature | Pi | Proxy | Gap |
|---|---|---|---|
| **Trigger mechanism** | ✓ (threshold-based) | ✓ (ratio-based, per-mode) | Both trigger proactively; proxy has mode-specific thresholds |
| **Retention invariant** | ✓ (first K entries via UUID) | ✓ (system prompt + first user prompt) | **Equivalent** — both preserve essential context |
| **Turn-boundary respect** | ✓ (never cut at tool results) | ✓ (pair_turns, drop whole turns) | **Equivalent** — both respect logical boundaries |
| **Summarization via LLM** | ✓ (structured prompt) | ✓ (local llama-server, open prompt) | **Gap: summary quality** (Pi's structured format) |
| **Incremental summarization** | ✓ (merge with previousSummary) | ✗ | **Gap: Pi updates, proxy regenerates** |
| **Split-turn handling** | ✓ (separate turn-prefix summary) | ✗ | **Gap: Pi handles mid-turn cuts** |
| **Backstop truncation** | ✗ | ✓ (truncate_backstop) | **Advantage: proxy** — Pi has no safety net |
| **Dry-run mode** | ✗ | ✓ (`compaction_dry_run` flag) | **Advantage: proxy** — Pi has no preview |
| **File operation tracking** | ✓ (read/modified in summary) | ✗ | **Gap: Pi tracks file ops** |
| **Retry on summarization failure** | ✓ (retryAssistantCall) | ✗ (fail-open) | **Gap: Pi retries, proxy fails** |
| **Churn tracking** | ✗ | ✓ (`CompactionChurnCollector`) | **Advantage: proxy** — Pi has no rate limiting |
| **Structured logging** | ✓ (events via hooks) | ✓ (single log line) | **Both have logging** — different mechanisms |
| **Branch summarization** | ✓ (separate feature) | ✗ | **Gap: Pi has branch summaries** |
| **Custom instructions** | ✓ (`/compact <instructions>`) | ✗ | **Gap: Pi supports focus areas** |
| **Abort/cancellation** | ✓ (`AbortController`) | ✗ | **Gap: Pi can abort in-progress compaction** |
| **Token estimation accuracy** | ✓ (per-message, role-specific) | ✓ (char/4 heuristic) | **Comparable** — both approximate |
| **Usage tracking** | ✓ (input/output/cost) | ✗ (no usage tracking) | **Gap: Pi tracks token usage** |
| **Extension hooks** | ✓ (`session_before_compact`) | ✗ | **Gap: Pi has extension API** |
| **Summarizer fail-open** | ✗ (throws on failure) | ✓ (returns "", logs warning) | **Advantage: proxy** — never blocks dispatch |

### 3.2 Architecture Comparison

| Aspect | Pi | Proxy |
|---|---|---|
| **Layer** | Agent/session layer (inside the coding agent) | HTTP proxy layer (before routing to llama-server) |
| **Data model** | Typed session entries with UUIDs | OpenAI-style message lists |
| **Persistence** | Compaction entry saved to session file | No persistence (stateless, applied per-request) |
| **Statefulness** | Session-aware (knows previous summary, entry history) | Stateless (works with current message list) |
| **Extensibility** | Extension hooks, custom instructions | Config-driven (ratio, thresholds) |
| **Deployment** | Part of the agent binary | Part of the proxy server |

---

## 4. Gap Analysis

### 4.1 Critical Gaps (proxy should adopt from Pi)

#### G1: Summarization Prompt Quality
**Severity: HIGH**

Pi uses a structured format that produces consistent, well-organized summaries:
```
## Goal
## Constraints & Preferences  
## Progress (Done/In Progress/Blocked)
## Key Decisions
## Next Steps
## Critical Context
```

The proxy uses an open-ended prompt. While simpler, it produces less structured summaries that may miss important context (file paths, decisions, next steps).

**Impact:** Lower-quality summaries → less useful context for resumed sessions → potential quality degradation in downstream use.

**Recommendation:** Adopt a structured prompt format for the proxy's summarizer. This is a prompt engineering change, not a code change.

#### G2: Incremental Summarization
**Severity: MEDIUM**

Pi merges new information into the existing summary (`UPDATE_SUMMARIZATION_PROMPT`). The proxy always generates a fresh summary from scratch.

**Impact:** Fresh summaries lose continuity across compaction events. If a session compacts multiple times, each summary starts from scratch rather than building on the previous one.

**Recommendation:** Add `previousSummary` tracking to the proxy's compaction flow. This requires storing the last summary in session state.

#### G3: Retry on Summarization Failure
**Severity: LOW**

Pi retries summarization calls on transient failures (socket drops, etc.). The proxy fails immediately on any error.

**Impact:** Rare — summarization calls are fast and local. But in degraded conditions (llama-server under load), transient failures may be non-trivial.

**Recommendation:** Add retry logic to `build_local_summarizer()` with a configurable retry count (1-3 retries, exponential backoff).

### 4.2 Moderate Gaps (proxy should consider adopting from Pi)

#### G4: File Operation Tracking
**Severity: MEDIUM**

Pi tracks file reads and writes, embedding them in the summary as XML sections. The proxy does not.

**Impact:** Summary misses file context — which files were read, which were modified. This is critical for the proxy's primary users (agent framework consumers who need to continue work).

**Recommendation:** Add file operation extraction in the proxy's summarization path. This requires access to the message history's tool calls (which the proxy already has access to, as messages are OpenAI-style).

#### G5: Split-Turn Handling
**Severity: LOW**

Pi can handle mid-turn compaction by summarizing the turn prefix separately. The proxy groups turns and doesn't split them.

**Impact:** The proxy's approach is more conservative (never splits a turn) but may discard more recent work in edge cases where a single turn exceeds the budget.

**Recommendation:** Low priority. The proxy's "never split a turn" policy is safer. Only consider if evidence shows mid-turn splits are necessary in practice.

### 4.3 Features Where Proxy Already Exceeds Pi

| Feature | Why Proxy's Approach is Better |
|---|---|
| **Backstop truncation** | Pi has no safety net — if summarization fails to fit the budget, there's nothing. The proxy's `truncate_backstop` provides a deterministic fallback. |
| **Dry-run mode** | Pi has no preview. The proxy's dry-run mode enables safe rollout (warn-only) and operational monitoring. |
| **Churn tracking** | Pi has no rate limiting. The proxy's churn collector enables < 1 compaction/session/hour enforcement. |
| **Mode-specific strategies** | Pi uses a single strategy. The proxy's fast=truncate, cheap=summarize+truncate is a more nuanced approach matched to use cases. |
| **Fail-open summarizer** | Pi throws on summarization failure (blocks the session). The proxy returns `""` and continues (never blocks dispatch). |
| **Per-mode thresholds** | Pi has a single context window. The proxy has fast/cheap modes with different thresholds, matching different operational needs. |

---

## 5. Recommended Actions

### 5.1 Immediate (High Impact, Low Effort)

#### R1: Improve the Summarization System Prompt (HIGH, LOW)
**Effort:** ~1 hour (prompt engineering only)  
**Work Item:** Create as a child task

Change the proxy's summarizer prompt from the current open-ended format to a structured format matching Pi's:

```python
SUMMARIZER_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.

Use this EXACT format:
## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]

## File Operations (if any)
<read-files>
...
</read-files>
<modified-files>
...
</modified-files>
"""
```

This is the single highest-impact change — it addresses the quality gap without any code changes.

#### R2: Add File Operation Tracking (MEDIUM, LOW)
**Effort:** ~2 hours  
**Work Item:** Create as a child task

Add file operation extraction to the proxy's summarizer:
1. Scan `middle_messages` for tool calls (read, write, edit operations)
2. Build `readFiles` and `modifiedFiles` sets (deduplicated, sorted)
3. Append `<read-files>` and `<modified-files>` sections to the summary
4. Return file operations as part of the compaction result

This matches Pi's behavior exactly and provides critical file context in summaries.

### 5.2 Short-Term (Medium Impact, Low-Medium Effort)

#### R3: Add Retry Logic to Summarizer (LOW, LOW)
**Effort:** ~1 hour  
**Work Item:** Create as a child task

Add retry to `build_local_summarizer()`:
```python
MAX_RETRIES = 2
RETRY_DELAY = 0.5  # seconds

def _summarizer(middle_messages):
    ...
    for attempt in range(MAX_RETRIES + 1):
        try:
            # existing HTTP call
            ...
        except Exception as exc:
            if attempt < MAX_RETRIES:
                logger.warning("summarizer attempt %d failed: %s, retrying...", attempt + 1, exc)
                time.sleep(RETRY_DELAY)
                continue
            logger.warning("summarizer failed after %d attempts: %s", MAX_RETRIES + 1, exc)
            return ""
    return ""
```

#### R4: Add Dry-Run Mode to Pi (ADVANTAGE PROXY → PI)
**Effort:** ~4 hours  
**Work Item:** Not in scope for this proxy project (Pi is in a separate repo)

Document the gap. Pi should consider adding a dry-run/preview mode like the proxy's. This is a Pi-side change.

#### R5: Add Incremental Summarization Support (MEDIUM, MEDIUM)
**Effort:** ~4-8 hours  
**Work Item:** Create as a child task

Track the previous compaction summary in session state and pass it to the summarizer. Use an "update" prompt variant (similar to Pi's `UPDATE_SUMMARIZATION_PROMPT`) that merges new information into the existing summary.

This requires:
1. Storing the last summary in session state (e.g., in `slot-cache` or session metadata)
2. Detecting that a previous summary exists
3. Using the incremental prompt format
4. Merging the previous summary with the new information

### 5.3 Long-Term (Strategic)

#### R6: Churn Rate Enforcement in Pi (ADVANTAGE PROXY → PI)
**Effort:** ~4 hours  
**Work Item:** Not in scope for this proxy project

Document the gap. Pi should consider adding a churn rate limiter to prevent excessive compaction events (< 1/session/hour).

#### R7: Branch Summarization in Proxy (ADVANTAGE PROXY → PI)
**Effort:** ~8-16 hours  
**Work Item:** Not in scope for this proxy project (separate feature)

Pi's branch summarization is analogous to compaction — it summarizes a diverged conversation branch. The proxy doesn't need this unless it develops branching session support.

---

## 6. Quality Assessment

### 6.1 Overall Assessment

| Dimension | Pi Score | Proxy Score | Notes |
|---|---|---|---|
| **Trigger sophistication** | 7/10 | 9/10 | Proxy's per-mode, ratio-based approach is more nuanced |
| **Summarization quality** | 9/10 | 5/10 | Pi's structured format wins; proxy prompt needs improvement |
| **Safety/reliability** | 6/10 | 9/10 | Proxy's fail-open, backstop, dry-run are superior |
| **Feature completeness** | 9/10 | 6/10 | Pi has more features (incremental, split-turn, file ops, etc.) |
| **Operational visibility** | 6/10 | 9/10 | Proxy's structured logging, churn tracking, dry-run are superior |
| **Extensibility** | 8/10 | 5/10 | Pi's extension hooks vs proxy's config-only |

**Overall: Pi 7.7/10 vs Proxy 7.2/10**

The proxy is competitive but has a clear gap in summarization quality (prompt format) and some feature gaps (incremental, file ops). These are addressable.

### 6.2 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Summarization quality degradation from prompt change | Low | Medium | A/B test with existing sessions; monitor quality |
| Retry adds latency to summarization calls | Medium | Low | Short delays (0.5s), max 2 retries; total overhead bounded |
| File operation tracking misses some operations | Low | Low | Conservative: only track standard tool names |
| Incremental summary drift over many compactions | Medium | Medium | Periodic "reset" summaries; quality monitoring |

---

## 7. Implementation Roadmap

```
Phase 1 (Immediate - Week 1)
├── R1: Improve summarizer system prompt (prompt engineering)
├── R2: Add file operation tracking (code change)
└── R3: Add retry logic (code change)

Phase 2 (Short-term - Week 2-3)
├── R5: Add incremental summarization support (code change)
├── R4: Document dry-run gap for Pi (analysis only)
└── R6: Document churn tracking gap for Pi (analysis only)

Phase 3 (Long-term - As needed)
├── R7: Branch summarization (if proxy develops branching)
└── Quality experiments (A/B testing of new summary format)
```

---

## 8. Conclusion

The proxy's compaction system is **architecturally sound and operationally mature** — it handles the proxy's specific needs (mode-specific strategies, dry-run, backstop, fail-open) well. However, it has **clear quality gaps** compared to Pi in summarization quality (prompt format, incremental updates, file operations).

The recommended actions (R1, R2, R3) are **low-effort, high-impact** improvements that bring the proxy's summarization quality to parity with Pi. The proxy does not need to replicate all of Pi's features (it serves a different layer of the stack), but it should adopt Pi's best practices for summarization quality.

**The proxy's approach is already "at least as good" in terms of reliability and operational features. After implementing R1-R3, it will be "at least as good" in summarization quality as well.**
