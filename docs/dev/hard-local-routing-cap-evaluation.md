# Hard local-routing cap evaluation — stop futile over-slot prefills

- Work item: LP-0MTAQNAIH001RN1S
- Date: 2026-08-27
- Evaluation only — no config change. Implementation/rollout tracked as a
  follow-up work item after operator approval (AC4).

## 1. Context and mechanism

The local Qwen3 slot's KV capacity is defined by `ctx_size / slots`.
Current live configs (supersede LP-0MSY0SDAS0031Y7F):

| Mode | Slots | ctx | Per-slot (llama `n_ctx_slot`) | Routing clamp (`per_slot - 4096`) |
|---|---|---|---|---|
| fast | 3 | 262144 | 87381 (padded 87552) | **83285** |
| cheap | 2 | 262144 | 131072 | min(100000, 126976) = **100000** |

Currently the routing decision has **no hard cap below the clamped warm
threshold**: for estimated context in the `(cold, warm]` band

- fast: `(38000, 83285]`
- cheap: `(42000, 100000]`

the request is dispatched local whenever the session's cached-token ratio is
high enough that `new_tokens = estimated × (1 − ratio) <= cold`. A session at
60–83K estimated tokens with a partly-warm cache is therefore routed local,
llama-server re-prefills the **entire** prompt (cache restore is skipped for
oversized contexts — the derived `session_slot_max_prompt_tokens` cap —
or fails with ReadTimeout under load), and the prefill saturates the Strix
Halo iGPU's shared memory bandwidth for minutes while other slots' decodes
starve.

Observed 2026-08-26/27 evidence (see §4): prefills up to **85,724 tokens**
(~98% of the fast per-slot capacity), repeated per turn; decodes collapsing
to **0.11–0.2 t/s** while another slot runs a ≥50K prefill.

## 2. Methodology

- Parsed `/var/log/llama-proxy/llama-server.log` + `llama-server.log-2026-08-27`
  (`proxy/benchmarks/evaluate_hard_routing_cap.py`, committed) — 2,270
  prefill-complete events, 4,386 decode samples.
- Checked routing decisions in `/var/log/llama-proxy/proxy.log*` (2026-08-27
  00:00 → 14:58 UTC, 1,934 `routing_check` events).
- Corroborated with the 24h proxy-usage report for 2026-08-25 14:37 →
  2026-08-26 14:37 (`~/proxy-usage-reports/2026-08-26_3/report.md`).
- llama-server logs carry no timestamps; event proximity is measured in
  sequential-log lines (window 800, tuned to the measured decode-collapse
  signal). Per-slot attribution via `slot update_slots: id X` lines;
  "cross-slot" prefills are those of another slot (the bandwidth consumers).

## 3. Findings

### 3.1 Prefill-size distribution (fast-mode profile applied to both log days)

| Bucket | Events | % | Tokens | % tokens |
|---|---|---|---|---|
| < 38K (under cold, always local) | 1,644 | 72.4% | 30,883,328 | 48.0% |
| 38–83.3K (in-band, cached-ratio check) | 622 | 27.4% | 33,078,533 | 51.4% |
| > 83.3K (context_too_large, skipped today) | 4 | 0.2% | 339,850 | 0.5% |
| **Total** | **2,270** | | **64,301,711** | |

Max observed prefill: **85,724 tokens**. 51 events > 70K (3.9M tokens),
135 events > 61.4K (9.6M tokens).

### 3.2 The gap: near-cap requests are still dispatched local

Of 1,934 routing decisions (15 h): **182 had estimated context 60–83.3K
(in-band for fast)** and **89 (49%) of those were dispatched local**
(cached ratio high enough that `new_tokens <= 38000`). Per day this is
roughly **90–140 futile near-cap local dispatches** (extrapolated from
15 h of logs: 89 × 24/15 ≈ 142).

These are the futile prefills: 60–83K actual tokens prefilled at
~50–105 t/s prompt-eval (median 105 t/s, p10 47.6 t/s) = **10–26 minutes
per prefill**, leaving only 4–27K of KV headroom for output, and
re-prefilled on every turn when persistence is skipped.

### 3.3 Decode starvation (cross-slot prefill contention)

| Cross-slot prefill size (within 800 lines) | Evals | Avg decode t/s | <5 t/s | <2 t/s |
|---|---|---|---|---|
| < 10K | 346 | 80.9 | 1 (0.3%) | 1 |
| 10–30K | 611 | 106.0 | 16 (2.6%) | 6 |
| 30–50K | 1,567 | 84.1 | 73 (4.7%) | 31 |
| ≥ 50K | 1,862 | 78.5 | 133 (7.1%) | 62 |

- 223 of 4,386 decode samples (5.1%) collapsed below 5 t/s; **133 (60%)**
  had a ≥50K cross-slot prefill nearby. Worst observed: **0.11 t/s**
  (matches the 0.14–0.2 t/s incident reports).
- Same-slot context growth also slows decode (KV read scales with context),
  but the cross-slot signal above isolates the *bandwidth contention*
  mechanism: avg decode drops ~74% when another slot prefills ≥50K.

### 3.4 Fallback-rate context (24h report, 2026-08-25 14:37 → 08-26 14:37)

- Fallback events: 3,094 (56.6% of 5,466 requests).
  - `context_too_large`: **966 (31.2%)** — already routed remote today.
  - `large_context_bypass`: 458 (14.8%).
  - `local_concurrency_limit`: 1,309 (42.3%) — slot contention.
- A hard cap shifts in-band traffic from "dispatched local / slow prefill"
  into the already-exercised `context_too_large`-style remote path — no new
  fallback mechanism is introduced.

### 3.5 Remote-path readiness for oversized requests

- Remote tiers already absorb 3,283 requests/day (60.1%); the oversized
  requests land on the same opencode-go→deepseek chain.
- Observed remote pressure on 2026-08-26: 184 upstream 429s in the 24h
  window (5.6% of remote), 44 backend_retry, 75 stream_finish errors.
- The incremental remote load from the hard cap is small (see §5): +57–61
  events/day at a 70K cap — well within the observed remote envelope,
  provided the request bypasses local **instantly** (no dispatch attempt).

## 4. Reproduction script

`proxy/benchmarks/evaluate_hard_routing_cap.py` — replay
`llama-server.log*` with a candidate `--hard-cap` and `--mode`, reports the
gated prefill volume and the decode-collapse correlation:

```bash
python3 proxy/benchmarks/evaluate_hard_routing_cap.py \
    --logs /var/log/llama-proxy/llama-server.log \
    --logs /var/log/llama-proxy/llama-server.log-2026-08-27 \
    --mode fast --hard-cap 70000
```

Sample run (fast mode, logs above):

```
--- Prefill distribution (current policy) ---
     under_cold: 1644 events (72.4%),   30883328 tokens (48.0%)
        in_band:  622 events (27.4%),   33078533 tokens (51.4%)
context_too_large:    4 events ( 0.2%),    339850 tokens ( 0.5%)
Total: 2270 events, 64301711 tokens (mean 28326, max 85724)

--- With hard cap 70000 ---
      above_cap:   61 events ( 2.7%),    4622059 tokens ( 7.2%)
→ additionally gated beyond current clamp: 57 events / 4282209 tokens
--- Decode collapse vs CROSS-slot prefill (line window 800) ---
Collapsed <5 t/s: 223 (5.1%), of which with a >=50K cross-slot prefill nearby: 133
```

## 5. Mode-specific recommendation

### 5.1 Fast mode — hard cap 70,000, bypass local instantly

**Threshold:** `local_large_context_hard_cap: 70000` (new key).

Rationale:

- Gates the prefill tail that produces the starvation: everything above 70K
  actual (51 events / 3.9M tokens / day) plus the in-band dispatched-local
  requests above 70K estimated (41 of 89 per 15 h).
- 70K ≈ 80% of the fast per-slot clamp (83,285) and coincides with the
  `context_pressure_warn_ratio` point (0.8 × 83285 = 66.6K). Sessions above
  this decode so slowly (KV bandwidth) that serving them locally is not
  worth the prefill cost or the collateral starvation of other slots.
- Leaves 17.4K of KV headroom below the physical 87.4K slot, absorbing the
  estimate-to-actual gap (chat-template/tool-def overhead, ~2–5K) *and*
  reserving a usable output budget (~17K tokens) — fixing the observed
  truncation-with-zero-headroom failure mode (LP-0MSAZXXDY005AWA1).
- Extrapolated effect on fallback rate: +57–61 remote events/day (mean
  ~60/966 existing `context_too_large` fallbacks = **+6%** on top of an
  already-routine remote path). Local utilization impact: −2.7% of prefill
  events, −7.2% of prefill tokens (both small).
- P95/TTFT win: the eliminated near-cap prefills are 10–26 min each;
  removing them from local dispatch removes their collateral effect on every
  concurrent decode (avg t/s +74% recovery per the 3.3 table). End-user P95
  improves by more than the +6% extra remote fallbacks cost, because the
  remote path for oversized requests has bounded latency (~120 s upstream
  timeout, 240 s idle) — far below a 10-min futile prefill.

**Interaction with existing keys (fast):**

- `local_large_context_cold_cache_threshold` (38000): **unchanged** — the
  economic new-token threshold; the hard cap only removes the upper part of
  the cached-ratio band's dispatch window, the (cold, hard_cap] band remains
  non-empty for the ratio check.
- `local_large_context_warm_cache_threshold` (100000, clamped to 83285):
  **clamp to the hard cap instead** — i.e. warm becomes
  `min(warm_config, hard_cap, per_slot-4096)` so `context_too_large` fires
  at the hard cap, keeping the existing skip-reason accounting.
- `session_slot_max_prompt_tokens` (0 = derived): derive from the hard cap —
  persistence is only useful below the routing cap; above it the request is
  never local, so persistence estimates above the cap are moot. Deriving
  both from one source keeps them consistent (AC3 of LP-0MSEGPO77005CYCQ).

### 5.2 Cheap mode — hard cap 61,440 (boot clamp), compact-or-route

**Threshold:** `local_large_context_hard_cap: 61440` (cheap profile).

Rationale:

- Cheap mode *must avoid remote fallback* (cost). Its slots are 131K —
  prefills above the routing clamp (100K) are rare (0.4% events).
- The cheap static profile already uses `local_model_ctx_size: 131072`
  (`config-cheap.yaml`), and the boot/scheduled clamp math (2×262144)
  yields **61,440** at boot (`131072//2 - 4096`). A hard cap at 61,440
  aligns the routing decision with the *conservative* boot clamp: nothing is
  dispatched local that would not fit the conservative per-slot view.
- Gating today: +159 events / 11.0M tokens (7.0% of events, 17.2% of tokens)
  — larger than fast mode's, but these are cheap-mode high-context sessions.
  They must NOT silently fall back (cost + the cheap contention policy is
  `queue`, and context bypasses never queue today, LP-0MSORQVK50012Q4D AC4).
- **Defined path for oversized cheap contexts (AC3):** never leave them
  stuck. Preferred: **session-compaction gate** — return a
  `context_too_large`-style informative 4xx with a `Retry-After`-style
  guidance header telling the agent to compact (the proxy already emits
  `context_pressure` warnings at 0.8×clamp suggesting compaction, and the
  companion session-compaction evaluation is in flight). Fallback to remote
  only when the session is NOT compactable within the contention queue
  budget (existing `contention_queue_*` caps apply).

**Interaction with existing keys (cheap):**

- `local_large_context_cold_cache_threshold` (42000): unchanged (the
  `(42000, 61440]` band stays non-empty for the ratio check).
- `local_large_context_warm_cache_threshold` (100000): clamp to 61440 so
  oversized cheap contexts route to the compaction gate, not into a
  near-full-slot prefill.
- `session_slot_max_prompt_tokens` (derived): derive from the hard cap as in
  fast mode.

## 6. Edge-case analysis (AC3)

1. **Tokenizer-estimate caveats (F2/F3 of LP-0MSAOQTJS000FFVM):** the
   routing estimate uses the Qwen3-native tokenizer on message content.
   llama-server's actual `n_tokens` additionally includes the chat template,
   tool definitions and BOS/EOS overhead (~2–5K for tool-heavy agents).
   Observed consequence: max actual prefill 85,724 > clamp 83,285. The hard
   cap at 70K (fast) / 61.4K (cheap) absorbs this gap with 17.4K / 69.6K
   margin — no prompt can pass the cap and still overrun its KV slot.
2. **Output-token headroom:** the routing clamp already reserves 4096; the
   hard cap reserves 17.4K (fast) / 69.6K (cheap) output headroom against
   the physical slot, so `max_output_tokens` truncation misreads
   ("maximum output token limit", LP-0MSAZXXDY005AWA1) disappear for gated
   sessions.
3. **Session compaction interaction:** oversized sessions must have a
   defined path. Fast: instant remote (fallback is acceptable and is
   already the norm, 56.6% fallback rate). Cheap: compaction-gate/queue —
   never a silent 10-min prefill; guidance surfaces the existing
   `context_pressure` compaction signal. Companion item (session-compaction
   evaluation) remains the owner of the compaction UX.
4. **`slot_schedule` / `--parallel` alignment (constraint):** the hard cap
   is derived from the *active* schedule entry's `(ctx_size, slots)` exactly
   like the current clamp (`_effective_large_context_thresholds` /
   `effective_per_slot_threshold`), so fast 23:59→3 and cheap 10:00→2
   entries each resolve their own cap. No change to llama-server `--parallel`
   or `slot_schedule`.
5. **Warm-cache efficiency:** sessions below the hard cap that use the ratio
   check keep their warm-cache wins unchanged; the hard cap only removes the
   zone where the warm-cache benefit is outweighed by prefill cost and
   cross-slot starvation.
6. **Out of scope (unchanged):** the 150K–480K HTTP-400 cluster
   (LP-0MSC1BNP90017L9K) — those requests already beat the clamp and are
   served by a separate fix.

## 7. Expected net effect summary

| Metric (per day) | Today (measured) | With hard cap (fast 70K / cheap 61.4K) |
|---|---|---|
| Prefill events gated (never local) | 4 `context_too_large` + 966 fallbacks | +57–61 fast +159 cheap additional |
| Prefill tokens kept off the iGPU | 0.34M | +4.3M fast / +11.0M cheap |
| Futile near-cap local dispatches (est 60–83K) | ~90–140 | → 0 (all above cap route immediately) |
| Decode collapse <5 t/s w/ ≥50K cross prefill | 133/15h (60% of collapses) | → ~0 for gated prefills |
| Decode avg under 50K+ cross prefill | 78.5 t/s | recovers toward ~90–106 t/s |
| Remote load increase | — | +6% of today's remote requests |
| Local utilization | — | −2.7% events / −7.2% tokens (fast) |

Net: a hard routing cap at **fast 70,000 / cheap 61,440** eliminates the
10–26-minute futile prefills and the collateral decode starvation they cause
(ranked the top injury in the 2026-08-26 incident), at a small, bounded
(~6%) incremental remote cost in fast mode and a compaction-gated path in
cheap mode — with zero config change in this item.

## 8. Open questions for the operator (approval gate)

1. **Fast-mode hard cap value**: 70,000 (recommended, = 0.8 × clamp) vs
   75,000 (only 26 additional events/day gated — fewer remote fallbacks but
   keep 12.5K headroom) vs 83,285 (current clamp; gates almost nothing new).
2. **Cheap-mode oversized path**: reject-with-compaction-guidance
   (recommended) vs queue-then-fallback vs immediate fallback (cost risk).
3. **Cheap hard cap value**: 61,440 (boot-clamp alignment, recommended) vs
   70,000 (fewer gated events) vs 100,000 (current clamp; no change).
4. **Config-surface shape**: new explicit `local_large_context_hard_cap`
   key per mode vs a ratio of the per-slot clamp (e.g. `0.8`) computed at
   runtime — the ratio form auto-adapts to future slot-schedule changes.
5. Whether the hard cap should also gate **non-session (anonymous)**
   requests the same way (recommended: yes — same KV capacity applies).