# Session Compaction Design Options — Evaluation

F3 of `LP-0MTAQNAQT002L746` (proactive session compaction for oversized
contexts, fast & cheap modes). Companion evaluation docs:

- `docs/dev/hard-local-routing-cap-evaluation.md` — LP-0MTAQNAIH001RN1S (hard
  routing cap; raised session-slot persistence cap to the routing clamp)
- `proxy/docs/context-compaction-eval/distribution.md` + `correlation.md` —
  F1 (`LP-0MTC87GBV0031F4B`) context-size distributions and F2
  (`LP-0MTC8A2UB0040NKQ`) oversized-session correlation
- Blocked experiment LP-0MSG9PUHU0059TTZ — quality-validation framework to
  revive for the chosen strategy
- LP-0MSDCLQ2W001LGWC — existing compaction tooling (context_pressure
  warning, KV cache quantization)

## 1. Why compaction is needed (evidence recap)

From F1/F2 on the 2026-08-24..26 window (proxy `routing_check`
`estimated_tokens` as ground truth; report CSVs carry empty context columns):

| Day | Sessions w/ routing checks | Fast breach rate (>= 83,285) | Cheap breach rate (>= 61,440) | `context_pressure` warnings |
|-----|---------------------------|------------------------------|-------------------------------|----------------------------|
| 08-24 | 253 | 12.8% | 9.1% | 1,234 |
| 08-25 | 191 | 15.1% | 15.3% | 1,362 |
| 08-26 | 180 | 10.4% | 11.1% | 1,731 |

- Peak per-session estimated context reached 651,408 tokens (Aug 26); fast
  p95 was 125K–167K across the three days.
- On Aug 26, **281.2M of 367.7M estimated prefill tokens (76.5%)** were
  wasted on routing checks where the session context ratio exceeded 1.0 — a
  session that can never be resident in one slot re-prefills its entire
  context every turn with no reusable KV.
- The decode collapse corroborates: 33 llama-server decode observations
  below 1 t/s on Aug 26 (min 0.18 t/s) against a ~22.9 t/s median.

These are the root cause of the fallback storm: slots routinely busy with
huge re-prefills -> `local_concurrency_limit` / `local_lease_active` /
`context_too_large` skips dominated the fallback reason mix.

## 2. Operating constraints

- **fast mode** (3 × 262,144; per-slot clamp 83,285): fallbacks to remote are
  acceptable; speed matters most. TTFT is dominated by prefill, so any
  strategy that keeps estimated context well under the per-slot clamp
  directly improves P95 via slot save/restore reuse.
- **cheap mode** (2 × 262,144; routing clamp 100,000; static clamp 61,440):
  remote fallbacks are to be avoided (cost). For oversized sessions, local
  compaction is the only acceptable path; the line between "compact and serve
  local" and "reject/queue" must be drawn.
- **No behavior change without approval**: this is a design option doc; the
  chosen strategy becomes a follow-up implementation work item.
- **Quality-loss risk must be flagged, not assumed away** — the operator
  depends on session content (audits, implement runs); silent truncation is
  unacceptable.

## 3. Candidate strategies

### Strategy A — Summarize-oldest-turns (TCX-style)

**Mechanism**: when a session crosses the trigger (`context_pressure` ratio
>= warn ratio, default 0.80 of the effective per-slot clamp), compact the
history: keep the system prompt + the most recent N turns verbatim, and fold
the oldest turns into a generated summary injected at the top of the prompt.
Serving compaction via

- the proxy's own summarization pass before dispatch (needs a cheap
  summarizing model call, or a cached local summary), or
- a client-frame instruction (the agent applying the summarize step itself),
  per the blocked LP-0MSG9PUHU0059TTZ design which planned exactly this
  "trim older turns, keep system + recent N" shape with a rubric-based
  quality bar.

**Effect on metrics**:
- Fallback rate: high reduction — sessions stay under the per-slot clamp, so
  `context_too_large` / `large_context_bypass` disappear for them; busy-slot
  contention falls with prefill time.
- Local utilization: up (slot save/restore becomes effective again below the
  persistence cap).
- Remote cost: near-zero cost increase (summaries are short); one
  summarization call per compaction event.

**Quality risk**: HIGH for detail-recall (older instructions, error
messages, file contents collapsed into a summary lose exactness), LOW for
recency (recent N turns are verbatim). The LP-0MSG9PUHU0059TTZ experiment
framework (task suite, three-arm A/B/C, rubric within-X%-of-baseline bar) is
the right validation vehicle; the summary must be measured against that bar
before rollout.

### Strategy B — Hard cap + auto-truncate

**Mechanism**: enforce an absolute prompt budget per session (e.g. 80% of the
per-slot clamp minus output headroom). Whenever the estimated context exceeds
the budget, drop the oldest turns until under the threshold. Deterministic,
no model call, trivially fast, and cheap to implement.

**Effect on metrics**:
- Fallback rate: high reduction (same mechanism as A but with hard drops).
- Local utilization: up, but the same concern as A: dropped content is gone
  unless the client retains it.
- Remote cost: neutral (no extra calls).

**Quality risk**: HIGH for abrupt loss (a dropped turn may contain a
critical instruction or in-flight tool result the operator still needs; the
work item's risk note is well founded). MEDIUM for well-structured
conversations where older turns are mostly static preamble — which is why
this is *less lossy than summarization for well-structured sessions* (no
invention, no paraphrase drift): content is dropped, never distorted. The
hardest part is deciding turn-boundary attribution: drop whole turns, never
split mid-turn, and never drop the system prompt.

**Interaction with slot save/restore**: best of the three — a truncated
session under the persistence cap restores efficiently, and the dropped
prefix is never re-prefilled again.

### Strategy C — Hybrid (summarize + cap + remote fallback for extreme)

**Mechanism**: (1) summarize oldest turns (A), (2) apply a hard cap (B) as a
backstop when summarization cannot reach the budget (e.g. summarizer
unavailable or output still over cap), and (3) for sessions that still
exceed the per-slot budget after compaction, route to remote with the
`context_too_large` path instead of looping local prefill attempts.

**Effect on metrics**:
- Fallback rate: low residual (only the truly extreme, compaction-resistant
  sessions fall back — exactly the ones the F2 data shows should never be
  locally prefilled: ratio > 1.0 after compaction means the turn is
  futile-local).
- Local utilization: up for the compactable majority; the residual extreme
  stops monopolizing slots.
- Remote cost: bounded and justified — fallback only where compaction cannot
  possibly fit the slot.

**Quality risk**: MODERATE (inherits A's summarization loss for the
compacted majority; B's hard cap as backstop may still drop content in
worst case). Operational benefit: prevents the futile-prefill loop that F2
quantified as 76.5% of prefill work.

## 4. Quality-risk matrix

| Criterion | A: Summarize-oldest | B: Hard cap + truncate | C: Hybrid |
|---|---|---|---|
| Quality preservation | Medium (detail loss in summary; recency verbatim) | Medium-high (verbatim remainder, abrupt loss risk) | Medium (A then B backstop) |
| Implementation complexity | High (summarization call, summary cache/injection, client co-op or proxy pass) | Low (threshold check + turn-dropping in prompt assembly) | High (A + B + routing decision) |
| Effectiveness at reducing prefill waste | High | High | Highest (residual extreme routed out) |
| Interaction with slot save/restore | Good (session under cap restores; summary is stable) | Best (stable truncated prefix restores cleanly) | Good (as A; extreme never local) |
| Interaction with hard-routing-cap companion (LP-0MTAQNAIH001RN1S) | Compatible — clamp raised persistence cap; compaction keeps sessions under the clamp so save/restore engages earlier | Compatible — same effect | Compatible — extreme routed out as the companion's cap intends |
| Determinism / testability | Low (summarizer nondeterminism) | High | Medium |
| Risk if summarizer/LLM unavailable | Blocked (must fall back) | None | Uses B as backstop |

## 5. Recommendation sketch (mode-specific detail in F4, LP-0MTC8BWV50012WSW)

- **fast mode**: Strategy C is overkill on the quality axis; **B (hard cap +
  auto-truncate)** with a generous budget (e.g. 80% of 83,285 ≈ 66K) is
  simplest, deterministic, and maximally restorable; fallback to remote is
  already acceptable in fast mode so the residual extreme needs no special
  handling beyond the existing `context_too_large` routing.
- **cheap mode**: remote is to be avoided; **C (hybrid)** is the only path
  that both protects cost and bounds prefill waste for the extreme tail —
  summarize+cap first (A+B), with the residual extreme explicitly routed
  remote rather than queued, per the cheap-mode constraint.

Both must be validated against the LP-0MSG9PUHU0059TTZ quality bar before
any rollout; triggers and aggressiveness are quantified in F4.

## 6. Validation approach (revives LP-0MSG9PUHU0059TTZ)

- Reuse the three-arm experiment design: A = uncompacted remote
  (baseline), B = compacted local, C = uncompacted local (ceiling).
- Quality bar: compacted-local within X% of uncompacted-remote on a
  task-completion rubric, with no increase in failure/retry rate.
- Metrics: TTFT (P95), total latency, local busy %, remote token reduction,
  fallback rate, and — from F2's methodology — wasted-prefill tokens
  (expected near zero once the strategy holds sessions under the clamp).

## 7. Implementation notes (for the follow-up item)

- Trigger point is already computed in `proxy/provider.py`
  (`should_warn_context_pressure` / `context_pressure_ratio`, default warn
  ratio 0.80 against `effective_per_slot_threshold`); the new step is
  *acting on* the warning/advisory rather than only logging it.
- Turn-boundary attribution lives in prompt assembly (session history as
  message list), so compaction is a prompt-assembly concern, not a KV
  concern; slot saving continues to work on the compacted prompt.
- Do NOT silently drop content: log every compaction event (session,
  before/after token counts, what was dropped or summarized) so quality
  regressions are traceable.