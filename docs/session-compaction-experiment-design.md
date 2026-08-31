# Session-Compaction Quality Experiment — Design

LP-0MSG9PUHU0059TTZ (Phase 2 of the compaction evaluation; epic
LP-0MTCW79RR000LFMJ). Validates that **compacting session history to fit
under the routing clamp and serving locally** yields the gains estimated in
the F1/F2 log study *without degrading output quality*. Outcome **gates
Tier 1 enforcement** (LP-0MTCWE8NG003P0SD). Tier 2 (LP-0MTBOX45O005LD1S,
hard-routing cap + cheap compaction gate) is a routing-level safety net and
does not depend on this experiment.

## 1. Objective and hypothesis

**Hypothesis (H1).** For sessions whose estimated context exceeds the
per-mode compaction trigger, serving a *compacted* prompt locally on Qwen3
produces output quality within a pre-registered tolerance of the current
baseline (uncompacted → remote deepseek-v4-flash), while eliminating the
wasted-prefill behavior quantified in F2.

**Why the risky population matters.** F1/F2 (2026-08-24..26, `proxy/docs/
context-compaction-eval/`) show the waste concentrates in the oversized
tail: **76.5% of estimated prefill work on 08-26 (281.2M of 367.7M tokens)
was spent on routing checks where the session context ratio exceeded 1.0**
(can never be resident in one slot → full re-prefill every turn), and the
**top-15 sessions account for 99.1% of all wasted prefill**. Breach rates
vs the per-mode caps were 9.1–15.3% of sessions/day. The same tail caused a
decode collapse on 08-26 (33 llama-server observations < 1 t/s vs a
~22.9 t/s median) — the collateral effect of near-cap re-prefills on
concurrent decodes. Compaction targets exactly these sessions; quality risk
is highest precisely there (long, detail-heavy histories), which is why a
controlled experiment must measure it before any enforcement rolls out.

## 2. Task suite

Drawn from logged real sessions (or a synthetic suite mirroring them),
covering the traffic that breaches the triggers:

| Category | Share of suite | Rationale (from logs) |
|---|---|---|
| Code editing / multi-file refactors | 30% | Long-lived implement/audit sessions dominate the F2 top-waste list |
| Q&A over long documents | 25% | Detail-recall sensitive; older content collapses into summary |
| Agent tool-call loops | 25% | In-flight tool results and error text must survive compaction |
| Long-context reasoning (chain-of-thought, analysis) | 20% | Full-context population — the "risky" cohort |

**Selection rules** (pre-registered):
- Include only sessions with ≥ 1 routing check `estimated_tokens` above the
  per-mode **compaction trigger** (fast > 58,300 / cheap > 43,000 — F4
  thresholds) — the population the strategy will actually act on.
- Stratify by mode (fast/cheap) and by context band: trigger–cap
  (compactable), cap–2×cap, > 2×cap (extreme; non-compactable after one
  summarization pass).
- Each task is replayed from the session transcript up to the point where
  the trigger was first breached, with the operator's final turn as the
  task's target request.
- Minimum 30 tasks per arm per mode in the trigger–cap band; ≥ 10 per arm
  in the extreme band (smaller because the population is small: 0.4% of
  events exceed the cap post-compaction per F4 §3).

Synthetic mirroring is permitted only where transcripts are unavailable
(e.g. sanitized); every synth task must reproduce the transcript's
length/detail structure so the quality measurement transfers.

## 3. Conditions (three arms)

| Arm | Prompt | Served by | Role |
|---|---|---|---|
| **A — baseline** | uncompacted (full history) | remote `deepseek-v4-flash` (current behaviour) | status quo; quality + cost anchor |
| **B — compacted-local** | compacted per §4 | local Qwen3 | proposed behaviour |
| **C — ceiling** | uncompacted (where the clamp allows) | local Qwen3 | documents local capability at full context; sanity bound for B |

Arm C runs only where the uncompacted context fits under the routing clamp
(fast ≤ 83,285 / cheap ≤ 100,000 warm clamp at schedule; hard cap
70,000 / 61,440 per Tier 2 as the ceiling). Where it cannot run, B vs A
is the primary comparison and C's role is explanatory only.

Same tasks across arms; within-session blocking (each task is its own
block), random arm order, blinded rubric scoring (see §5).

## 4. Compaction strategy (concrete, from F3/F4)

- **fast profile — Strategy B (hard cap + auto-truncate)**: when
  `estimated_tokens > 58,300` (0.70 × per-slot clamp 83,285), drop oldest
  whole turns (never split a turn, never drop the system prompt) until
  ≤ **38,000** (= `local_large_context_cold_cache_threshold`, the economic
  re-prefill line). Deterministic; no summarizer call.
- **cheap profile — Strategy C (hybrid)**: when `estimated_tokens > 43,000`
  (0.70 × static clamp 61,440), summarize oldest turns (A) then apply the
  hard cap (B) as backstop, targeting ≤ **30,000**; sessions still
  non-compactable after compaction are *excluded from arm B* and flagged as
  the extreme tail (they follow the Tier 2 gate: 429 compaction-gate
  response, never silent remote/queue).

Arm B uses the Tier 1 machinery (LP-0MTCWE8NG003P0SD, warn-only dry-run
first) once implemented; until then a **manual equivalent** — the harness's
prompt-assembly pass implementing exactly the drop/summarize rules above on
the extracted transcript — is used, with before/after counts logged per the
F3 §7 traceability requirement (session, before/after tokens, turns dropped,
summary length).

## 5. Quality measurement and bar

### Metrics (primary — quality)

1. **Task-completion rate** — binary per task, rubric-defined success
   envelope (e.g. target output satisfies stated requirement).
2. **Rubric output quality** — 1–5 per dimension (correctness, completeness,
   detail recall, instruction adherence, formatting), scored blind by a
   rubric judge (LLM-as-judge with a fixed, pre-committed rubric + human
   spot-check on ≥ 10% of responses; judge is arm-blinded).
3. **Semantic-equivalence score (B vs A)** — paired, where completion is
   ill-defined (open-ended tasks).
4. **Failure / truncation rate** — response errors, fallback
   `context_too_large` / `large_context_bypass` events, cap gates during the
   task.
5. **Retry rate** — client-initiated retries observed in the replay.

### Pre-registered quality bar (primary gate)

- Mean rubric score: **B ≥ 0.95 × A** (compacted-local within 5% of
  uncompacted-remote), with a non-inferiority one-sided test at α = 0.05.
- Task-completion rate: **B ≥ A − 3 percentage points** (lower bound of the
  paired difference).
- No increase in failure|gate events (B vs A) beyond sampling noise
  (chi-square / exact test at α = 0.05); cap gates in B only for
  non-compactable extremes.
- Arm C is reported but not gated: it bounds what local can do, and a B ≪ C
  gap signals the compaction strategy (not the local model) is the
  bottleneck — triaged, not rejected, on that signal alone.

### Metrics (secondary — operational/efficiency)

- Latency: TTFT (P50/P95) and total-task latency (P50/P95), per arm.
- Local busy % during the run; remote request count and remote token
  reduction (B vs A).
- Wasted-prefill tokens (F2 methodology) per task and aggregated; decode
  t/s sampled where llama-server logs permit (window-attributed, per
  caveats in `proxy/docs/context-compaction-eval/README.md`).
- Cost delta (B vs A): remote-token spend + summarizer calls (cheap arm).

### Pre-registered efficiency gate (secondary)

- **Reject efficiency claim if** wasted-prefill tokens are not reduced by
  ≥ 25% on the risky population, or TTFT P95 in B is worse than A by
  > 20% where the same task is servable locally (i.e. compaction buys
  nothing operationally).

## 6. Sample size and run design

- Pilot: 30–50 tasks/arm/mode from the trigger–cap band + ≥ 10/arm/mode
  extreme; revisit power after scoring variance is observed.
- Planned power note: for a 3pp completion delta at α = 0.05 / β = 0.20,
  ~310 blocks/arm — the pilot will confirm whether that scale is needed or
  whether the observed variance supports a smaller matched design (the
  within-task blocking is the main variance reducer).
- Runs are replay-only (no live traffic injection); each arm's replay runs
  through the proxy's normal routing path with the session `model`
  overridden per arm (remote / local) and mode pinned, reusing existing
  routing checks, `context_pressure` estimates and the Tier 1 dry-run log
  surface. **No speculative new infra** — see §7.

## 7. Harness reuse and new work items

**Existing machinery reused (AC2 — no speculative infrastructure):**
- `proxy/scripts/analyze_context_distribution.py` + `correlate_oversized_sessions.py`
  (F1/F2) — task-suite selection from `routing_check` / `context_pressure`
  log lines.
- `proxy/provider.py` `context_pressure_ratio` /
  `should_warn_context_pressure` (0.8 × effective per-slot threshold) —
  trigger hook; compaction is an *action on* this advisory, not a new signal.
- Tier 1 (LP-0MTCWE8NG003P0SD) warn-only dry-run — once implemented, arm B
  runs through it; until then the harness's manual prompt-assembly pass.
- Tier 2 (LP-0MTBOX45O005LD1S) gate/headers — defines the non-compactable
  extreme handling and gives the cap ceiling for arm C.

**New work items (created on approval of this design):**
1. Execution item (parent: this item) — run the experiment, score, report
   go/no-go against §5/§6. Reference: `docs/session-compaction-experiment-design.md`.
2. Harness script item (child of the execution item, small) — replay +
   scoring harness (`proxy/scripts/run_compaction_experiment.py`, eval-only,
   mirrors the F1/F2 eval-script pattern: no behavior change, fixture-based
   tests). Scoped lean: transcript extraction → prompt assembly (manual
   compaction pass) → three-arm replay → metric CSV → rubric scoring input.

## 8. Go / no-go rules (pre-registered)

**Go** (all): quality bar §5 met (B non-inferior to A on rubric + completion
+ no failure increase); efficiency gate §6 met or explicitly waived as
secondary; compaction event logging present (F3 §7 traceability).

**No-go** (any one):
- B rubric < 0.95 × A on the primary test (or completion B < A − 3pp).
- Failure|gate rate in B exceeds A at α = 0.05.
- Operator rejects the drop/summarize policy (content-loss risk) without a
  documented retention mirror (client-side full history; ContextHub
  WL-0MTBOXEGR009KDP6 counterpart).
- Churn: compaction would trigger > 1× per session per hour (tuned against
  the F1 distributions, per F4 §6.3) — threshold re-tune required, not ship.

**Recommendation (this document):** **proceed to execution.** The F1/F2
evidence (76.5% prefill waste, 99.1% concentration in 15 sessions, decode
collapse) makes the efficiency gain near-certain for the oversized tail; the
open question is exclusively the quality side, which this experiment is
designed to answer with pre-registered, arm-blinded measurement. Execution
is gated on this item's review; Tier 1 (LP-0MTCWE8NG003P0SD) remains
blocked until the bar is met.

## 9. Risks and caveats

- **Model confound**: arms differ in both compaction AND serving model
  (remote deepseek-v4-flash vs local Qwen3). Arm C bounds the model
  contribution at full context; B vs A remains the decision-relevant pair
  (the proposed end state is compacted-local).
- **Summarizer nondeterminism** (cheap arm): arm B cheap uses the manual
  summarize pass; record summary inputs/outputs for reproducibility.
- **Judge bias**: blind scoring, fixed pre-committed rubric, human
  spot-check ≥ 10%.
- **Log caveats**: llama-server logs are window-attributed, not hour-exact
  (README.md); treat decode t/s as directional only.
- **Extreme band rarity**: 0.4% of events exceed the cap post-compaction
  (F4 §3); the extreme-band arm is sized to detect gross quality cliffs,
  not fine deltas.

## Cross-references

- Evidence F1: LP-0MTC87GBV0031F4B (`distribution.md`), F2:
  LP-0MTC8A2UB0040NKQ (`correlation.md`).
- Strategy F3: LP-0MTC8B0BN009BMTJ (`docs/session-compaction-design.md`),
  F4: LP-0MTC8BWV50012WSW (`docs/session-compaction-recommendations.md`).
- Tier 2 cap evaluation: LP-0MTAQNAIH001RN1S
  (`docs/dev/hard-local-routing-cap-evaluation.md`), implementation
  LP-0MTBOX45O005LD1S.
- Client-side compaction UX: contextual WL-0MTBOXEGR009KDP6 (ContextHub).