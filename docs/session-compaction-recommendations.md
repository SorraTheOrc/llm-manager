# Mode-Specific Session Compaction Recommendations

F4 of `LP-0MTAQNAQT002L746` (proactive session compaction for oversized
contexts, fast & cheap modes). Synthesizes F1 (context-size distributions,
`LP-0MTC87GBV0031F4B`), F2 (correlation, `LP-0MTC8A2UB0040NKQ`), F3 (design
options, `LP-0MTC8B0BN009BMTJ`), and the hard-routing-cap companion
evaluation (`LP-0MTAQNAIH001RN1S`, `docs/dev/hard-local-routing-cap-evaluation.md`).

## 0. Decision summary

| | **fast mode** | **cheap mode** |
|---|---|---|
| Strategy | **B: hard cap + auto-truncate** | **C: hybrid (summarize + cap + remote for extreme)** |
| Compaction trigger | `estimated_tokens > 58,300` (0.70 × per-slot clamp 83,285) | `estimated_tokens > 43,000` (0.70 × static clamp 61,440) |
| Compaction target | ≤ `cold_cache_threshold` 38,000 | ≤ 30,000 (recent-N + system prompt) |
| Automation | **automatic** on breach (deterministic truncate; low risk) | **automatic** on breach (only acceptable path; remote only when non-compactable) |
| Admission cap (companion) | `local_large_context_hard_cap: 70000` | `local_large_context_hard_cap: 61440` |
| Expected fallback effect | −50–60% of oversized-session `context_too_large` skips | near-zero net cost increase; remote spend confined to extreme tail |
| Go/no-go | Gated on LP-0MSG9PUHU0059TTZ quality bar; A/B rollout | Same gate; implement as compact-or-route (no silent queue) |

This item is evaluation + recommendation only (**no behavior change**); the
implementation becomes a follow-up work item after approval.

## 1. Evidence recap (F1 + F2)

Per-session **peak** estimated context (routing-time `estimated_tokens`,
proxy `routing_check` lines), 2026-08-24..26:

| Day | fast median | fast p90 | fast p95 | fast max | cheap median | cheap p90 | cheap max |
|-----|------------|----------|----------|----------|-------------|-----------|-----------|
| 08-24 | 32.8K | 89.9K | 125.8K | 248.9K | 19.3K | 57.9K | 137.0K |
| 08-25 | 37.3K | 115.4K | 140.1K | 424.1K | 32.4K | 64.6K | 109.1K |
| 08-26 | 27.4K | 92.1K | 167.4K | 651.4K | 30.0K | 64.6K | 162.4K |

- **Breach rates** vs per-mode caps (F1): fast (≥83,285): 12.8% / 15.1% /
  10.4% of sessions; cheap (≥61,440): 9.1% / 15.3% / 11.1%.
- **Wasted prefill** (F2, Aug 26): **281.2M of 367.7M estimated prefill
  tokens (76.5%)** were spent on routing checks where the session context
  ratio exceeded 1.0 (can never be resident in one slot → full re-prefill
  every turn, no reusable KV).
- **Concentration:** the top-15 sessions by wasted work account for **99.1%**
  of all wasted prefill tokens — a small tail of long-lived audit/implement
  sessions causes nearly all the waste.
- **Decode collapse:** 33 llama-server decode observations < 1 t/s on Aug 26
  (min 0.18 t/s) against a ~22.9 t/s median — the collateral effect of
  near-cap re-prefills on concurrent decodes.

## 2. Fast mode recommendation

### Chosen strategy: B — hard cap + auto-truncate

Rationale: fast mode tolerates fallback and prioritizes speed. Truncation is
deterministic (no summarizer call, no quality-injection surface), fast,
highly testable, and — because the truncated prefix is stable — the best
interaction with slot save/restore. F3's matrix ranks B highest on
determinism/testability and slot-restore interaction, with quality risk
bounded to "drop whole turns, never split, never drop the system prompt".

### Trigger thresholds

- **Compaction trigger:** `estimated_tokens > 0.70 × per_slot_clamp`
  = **58,300 tokens** (fast clamp 83,285). Rationale: 0.70 sits below the
  existing `context_pressure_warn_ratio` point (0.8 × 83,285 = 66.6K) so
  compaction is *proactive* rather than reactive to warnings, and leaves
  ≥ 25K of growth headroom below the 83,285 clamp before the next breach —
  sessions compact infrequently.
- **Compaction target:** `≤ cold_cache_threshold (38,000)` — i.e. right at
  the existing economic new-token threshold (`local_large_context_cold_cache_threshold`,
  fast profile) where a resumption re-prefill is cheap. Concretely: keep the
  system prompt + the most recent turns up to a ~38K native-token budget;
  drop oldest whole turns first.
- **Admission cap (interlock):** `local_large_context_hard_cap: 70000`
  (companion LP-0MTAQNAIH001RN1S §5.1) remains the *routing* cap. The
  compaction trigger (58.3K) fires below it, so sessions hit compaction
  before they ever approach the hard cap; the hard cap stays as a final
  bypass guard (route remote, never futile-prefill).
- **Automation: automatic on breach.** Truncation is deterministic and
  cheap; risk of silent content loss is managed by always logging every
  compaction event (session, before/after counts, turns dropped). Advisory
  only is rejected because F2 shows the tail needs *enforcement*, not
  another log line.

### Expected effect (extrapolated from F1/F2, ordered by confidence)

1. **Prefill waste:** ~76.5% of estimated prefill work (Aug 26) was on
   ratio>1.0 sessions — with a 58.3K trigger + 38K target these checks
   become ratio<1.0 local serves. Wasted-prefill tokens → near zero for the
   top-15 sessions (99.1% of the waste).
2. **Local utilization:** faster (smaller) prefills reuse slot KV below the
   persistence cap; expected recovery of the decode-collapse pattern (~74%
   avg t/s recovery per the companion's 3.3 table). Session save/restore
   engages again for compacted sessions.
3. **Fallback rate:** the oversized-session `context_too_large` /
   `large_context_bypass` skips drop by roughly the share of checks that
   become compacted-local. Mean +6% remote fallbacks for the hard cap alone
   (companion); **net fallback rate expected to fall** once compaction keeps
   sessions under the cap, because busy-slot contention (`local_concurrency_limit`,
   `local_lease_active`) — the largest fallback buckets — also drops.
4. **Remote cost:** near-zero change in fast mode; only the extreme tail
   (>70K and non-compactable) routes remote, and those volume small vs total
   traffic.

Qualitative confidence: high on waste reduction (measured), medium-high on
contention reduction (mechanistic), medium on end-user P95 (needs the
LP-0MSG9PUHU0059TTZ experiment to quantify the quality side).

## 3. Cheap mode recommendation

### Chosen strategy: C — hybrid (summarize + cap + remote for extreme)

Rationale: cheap mode must avoid remote spend, so **proactive compaction is
the ONLY acceptable path for oversized cheap sessions** (F3; companion §5.2).
Cost reduction is the goal; a cheap session > 61,440 that stays oversized
either wastes a near-full 131K re-prefill or queues forever — both wrong.
C = A + B backstop + explicit remote for the residual extreme.

### The "compact and serve local" vs "reject/queue" line

- **Compact and serve local** when, after summarization + truncation, the
  session fits under 0.90 × static clamp (61,440) = ~55K — almost always the
  case; summarization targets ≤ 30,000.
- **Remote (with guidance), never silent reject/queue** when the session is
  non-compactable (summarizer unavailable or exceeds budget even after
  compaction) — mirror the companion §5.2's compaction gate: return a
  `context_too_large`-style informative 4xx with `Retry-After` guidance, or
  route remote per the existing fallback accounting. Cheap sessions must
  never sit stuck; `queue` is forbidden for oversized contexts (contention
  policy is `queue`, and context bypasses never queue, LP-0MSORQVK50012Q4D AC4).

### Trigger thresholds

- **Compaction trigger:** `estimated_tokens > 0.70 × static_clamp`
  = **43,000 tokens** (cheap static clamp 61,440). Rationale: below the
  warn point (0.8 × 61,440 = 49,152), proactive, ≥ 18K headroom before next
  breach.
- **Compaction target:** `≤ 30,000` (system prompt + recent N turns + oldest
  summarized). Summarize (A) first, then hard-cap (B) as backstop if the
  summary pass cannot reach budget.
- **Automation: automatic on breach** — cheap mode has no acceptable
  alternative path; automatic compaction plus remote only for the
  non-compactable extreme.

### Expected effect

- Remote cost: near-zero for the compactable majority; spend confined to the
  non-compactable extreme tail (session > 61,440 post-compaction), which is
  rare (0.4% of events per companion §5.2) and already going remote today.
- Fallback avoidance: oversized cheap sessions stop monopolizing local slots
  (cheap contention), removing their contribution to cheap fallbacks from
  busy-slot pressure while actually serving locally.

## 4. Threshold specification (both modes)

| Parameter | fast | cheap | Rationale |
|---|---|---|---|
| Compaction trigger (ratio) | 0.70 × 83,285 = **58,300** | 0.70 × 61,440 = **43,000** | below warn ratio (0.8), proactive, ≥17–25K bump headroom |
| Compaction target (post) | ≤ **38,000** (= cold threshold) | ≤ **30,000** | cheap re-prefill; matches economic cold line (fast) |
| Warn ratio (existing) | 0.8 | 0.8 | unchanged; compaction now acts on it |
| Routing hard cap (companion) | 70,000 | 61,440 | unchanged; final bypass guard |
| Drop policy | drop oldest whole turns | summarize oldest, then drop whole turns | never split a turn, never drop system prompt |
| Event logging | required (session, before/after, turns dropped) | required (same + summary length) | traceability for quality regression |

## 5. Interaction analysis

- **Slot save/restore:** compaction keeps sessions below the persistence cap
  (`session_slot_max_prompt_tokens`, derived from the routing clamp per
  companion), so compacted sessions restore efficiently and the dropped
  prefix is never re-prefilled. Truncation (B) is strictly more restore-
  friendly than summarization (A) because the remainder is byte-stable;
  hence B for fast, and B-backstop in cheap.
- **Hard-routing-cap companion (LP-0MTAQNAIH001RN1S):** complementary, not
  competing. The hard cap stops *futile* prefills at the routing layer
  (70K fast / 61.4K cheap, bypass local instantly); compaction stops
  sessions from *reaching* the cap on every turn. Together: compact under
  trigger → serve local; trigger→cap window → compacted local; ≥ cap →
  remote/gate. `local_large_context_warm_cache_threshold` should clamp to
  the hard cap (companion §5.1/5.2), keeping `context_too_large` accounting
  consistent.
- **KV-cache quantization (LP-0MSDCLQ2W001LGWC, q8_0):** orthogonal and
  additive — halves KV read band per token for whatever context remains.
  Compaction shrinks *context length*; quantization shrinks *bytes/token*;
  both raise decode headroom on the shared memory-bandwidth bottleneck.
  No conflict; recommended to keep both.
- **Existing context_pressure warning path (`proxy/provider.py`):** the
  `context_pressure_ratio` / `should_warn_context_pressure` computation
  (0.8 × `effective_per_slot_threshold`, configurable
  `context_pressure_warn_ratio`) is exactly the trigger hook — the change is
  to *act* on the signal at prompt assembly, not just log it.

## 6. Go/no-go for implementation

- **No-go blockers (any → do not implement):**
  1. The LP-0MSG9PUHU0059TTZ quality experiment rejects compacted-local vs
     uncompacted-remote on the rubric (task-completion bar not met).
  2. Operator rejects the drop/summarize policy (content-loss risk not
     accepted) without a documented retention mirror (client-side full
     history).
  3. A follow-up measurement shows compaction triggering more than ~1× per
     session per hour (churn/instability) — thresholds must be re-tuned,
     not shipped.
- **Go conditions:** quality bar met; compaction event logging in place;
  thresholds confirmed against a 48h dry-run (warn-only mode) using the
  F1/F2 distribution data; implementation work item created with the
  trigger/target constants above.

## 7. Implementation notes (for the follow-up item)

- Compaction is a **prompt-assembly** concern (session history as a message
  list), not a KV concern — it composes with slot save/restore naturally.
- Dry-run first: emit a `compaction_advisory` log line (session, est tokens,
  would-drop turns, target size) for 48h, measure against §6.3, then enable
  enforcement.
- Cheap-mode remote/gate decision must be wired through the existing
  fallback-reason accounting so the remote-cost impact is measurable
  (compare `WR_k4`-style reason buckets pre/post rollout).