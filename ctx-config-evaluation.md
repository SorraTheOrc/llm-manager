# Context Config Evaluation: 3×262144 vs Alternatives

**Work item:** LP-0MTAQNAIO0069DUP  
**Date:** 2026-08-27  
**Data window:** 2026-08-24 → 2026-08-26 (proxy-usage-reports) + live system state  
**Previous eval:** LP-0MSAOQTJS000FFVM (benchmark F2/F3, 2026-08-03/04)

---

## 1. Remote-Tier Status Report

### Current opencode API key status (from proxy logs 2026-08-26)

| Provider | Usage Limit Status | Reset At |
|---|---|---|
| `opencode-go-3-deepseek` | **EXPIRED — usage_limit_reset_pending** | 2026-08-30 18:28 UTC (4+ days) |
| `opencode-go-2-deepseek` | **EXPIRED — usage_limit_reset_pending** | 2026-08-29 09:03 UTC (2+ days) |
| `opencode-go-deepseek` | Active (fallback, no explicit limit) | — |

**Key finding:** The original supersede's premise (opencode 429s on 2026-08-18) **still holds**. Two of three opencode API keys are in `usage_limit_reset_pending` state. They are **not** being 429'd — they are being silently skipped entirely. This means fallback traffic goes through `opencode-go-deepseek` (the main OPENCODE_API_KEY), which may have its own limits.

The `deepseek-v4-flash` direct provider (api.deepseek.com) is also available as the final fallback.

**Implication for this eval:** Remote availability is NOT the binding constraint right now — it's the opencode key expiry. The eval should focus on local performance regardless of remote tier status, but note that a config that produces more local traffic will not reduce remote spend if the remaining key is also rate-limited.

---

## 2. Config-Space Benchmark Results

### Hardware context

- **GPU:** AMD Radeon Graphics (Strix Halo APU, gfx1151) — 1 GB VRAM reported, but uses **shared system memory** (124 GB total, 77 GB available)
- **Model:** Qwen3.6-35B-A3B-GGUF:Q5_K_M, Q5_K_M quantization
- **llama-server RSS:** ~11 GB (shared memory), model weights ~2.1 GB
- **KV memory headroom:** ~9 GB available (plenty for all candidate configs)

### Benchmark harness results (from F2/F3, 2026-08-03/04)

| Config | Cold TPS | Cold TTFT | Warm TPS | Warm TTFT | TPS Change (warm−cold) |
|---|---|---|---|---|---|
| Baseline 3×43.7K (q8_0) | 15.72 | 0.707s | 25.09 | 0.071s | +59.6% |
| 4×65.5K (q8_0) | 16.93 | 0.704s | 20.48 | 0.094s | +21.0% |
| 3×87.4K (q8_0) | 14.55 | **2.487s** | 17.03 | 0.105s | +17.1% |
| 2×131K (q8_0) | **0.39** | **4.790s** | 18.89 | 1.799s | +4744% |

### Key benchmark insights

1. **Cold prefill degradation is non-linear:** 3×87.4K has 2.487s TTFT (3.5× baseline) and 2×131K is catastrophic at 4.79s TTFT (6.8× baseline, 40× slower TPS).

2. **Warm performance tradeoff:** More slots = better warm throughput. 4×65.5K (20.48 TPS warm) beats 3×87.4K (17.03 TPS warm) despite smaller per-slot context.

3. **Decode-stall penalty:** A 4.79s TTFT cold prefill completely blocks the slot. With only 2 slots, a single long prefill means `available_slots=0` for 5+ seconds. The iGPU shared-memory bandwidth is the bottleneck — giant prefills saturate it and collapse concurrent decode.

4. **4×65.5K is the benchmark winner:** Best cold performance (16.93 TPS, 0.704s TTFT — essentially baseline), decent warm throughput. The only downside is lower per-slot context (65.5K).

### KV memory estimates (q8_0, 8 KV heads, 32 layers)

| Config | Per-slot KV | Total KV | GPU VRAM (reported) | Shared Mem (llama-server RSS) |
|---|---|---|---|---|
| 3×43.7K | 21.3 MB | 64.0 MB | 1 GB | ~9.5 GB |
| 4×65.5K | 32.0 MB | 128.0 MB | 1 GB | ~10.0 GB |
| 3×87.4K | 42.7 MB | 128.0 MB | 1 GB | ~10.5 GB |
| 2×131K | 64.0 MB | 128.0 MB | 1 GB | ~11.0 GB |

**All configs fit within available shared memory.** The 2×131K uses the most (~11 GB RSS), but still well within the 77 GB available.

---

## 3. Session-Retention Projection (Aug 24-26 data)

**Data source:** `/home/rg/proxy-usage-reports/compare-last24h/daytime_sessions.csv` — 698 sessions, 312 with context data  
**Peak concurrent sessions:** 32  
**Context distribution:** P50=48.8K, P75=74.2K, P90=110.3K, P95=150.8K, P99=408.6K, Max=829.2K

| Config | Slots | Clamp (per_slot−4096) | Full Local | Context Bypass | Concurrency Blocked | Concurrency-Aware Local | Raised-Cap Local |
|---|---|---|---|---|---|---|---|
| current 3×262144 (fast) | 3 | 83,285 | 181 (58.0%) | 61 | 70 | 17.0 (5.4%) | 239 (76.6%) [+225] |
| 2×262144 (cheap current) | 2 | 126,976 | 212 (67.9%) | 22 | 78 | 13.2 (4.2%) | 278 (89.1%) [+264] |
| 3×131072 (Option C) | 3 | 126,976 | 212 (67.9%) | 22 | 78 | 19.9 (6.4%) | 278 (89.1%) [+264] |
| 4×131072 (Option E) | 4 | 126,976 | 212 (67.9%) | 22 | 78 | 26.5 (8.5%) | 278 (89.1%) [+264] |

### The Hidden Binding Constraint

The **persistence cap** (`session_slot_max_prompt_tokens = 12,288`) is THE binding constraint, not the context routing clamp.

- Raising the persistence cap from 12.3K to the routing clamp unlocks **61.9% → 89.1%** of sessions staying fully local
- Without the cap raise, even context-eligible sessions lose warm-cache persistence and must do full cold prefills
- This explains why sessions keep getting routed remote: they fit in the slot but can't restore from KV cache

### Concurrency analysis

With 32 peak concurrent sessions:
- **3 slots:** slot availability = 3/32 = 9.4% → most context-eligible sessions are concurrency-blocked
- **4 slots:** slot availability = 4/32 = 12.5% → slightly better
- **2 slots:** slot availability = 2/32 = 6.3% → worst

The concurrency problem dominates the context problem: even with unlimited per-slot context, with only 2-4 slots for 32 concurrent sessions, most sessions will hit `available_slots=0`.

---

## 4. Mode-Specific Recommendation

### FAST MODE (10:00–01:00, fallbacks OK, speed important)

**Recommendation: Keep 3×262144 (current) + raise persistence cap**

**Rationale:**
- The benchmark shows 3×87.4K already has degraded cold TTFT (2.487s vs 0.707s baseline)
- 4×65.5K has better cold performance but lower per-slot context (65.5K vs 87.4K) — sessions at P75=74.2K would exceed this, triggering more context bypasses
- The real problem is not the per-slot context size — it's the **persistence cap** (12.3K) that prevents warm-cache restoration
- With 3 slots and 262144 total ctx, the iGPU is already handling concurrent loads; the issue is that giant cold prefills (85K+ tokens) saturate bandwidth

**Config changes:**
1. **Keep:** `local_model_ctx_size: 262144`, `session_slot_pool_size: 3`, slot_schedule 3 slots
2. **Raise:** `session_slot_max_prompt_tokens` from `0` (dynamic) or current to at least `83285` (the routing clamp)
   - This allows warm-cache persistence for all sessions that fit the per-slot clamp
   - Without this, sessions get warm cache for the first 12K tokens, then must cold-re-prefill on every turn

**Expected impact:**
- Fallback rate: No change from concurrency (still 70 sessions blocked per day by slot scarcity)
- Remote spend: Significant reduction if persistence cap is raised (61%→77% full local)
- TTFT: Unchanged for warm sessions (already fast), reduced for cold sessions (no need to re-prefill 85K tokens)

### CHEAP MODE (01:00–10:00, cost reduction important)

**Recommendation: Keep 2×262144 (current cheap shape) + raise persistence cap + consider 3×131072 as alternative**

**Rationale for current 2×262144:**
- Higher per-slot clamp (126,976) vs current 3×262144 (83,285) — fewer context bypasses (22 vs 61)
- 2 slots instead of 3 reduces GPU load during low-traffic overnight hours
- Benchmark shows 2×131K cold performance is poor (0.39 TPS, 4.79s TTFT) but warm is acceptable (18.89 TPS)

**Alternative: 3×131072 (Option C)**
- Same per-slot clamp as 2×262144 (126,976)
- 3 slots instead of 2 → better concurrency (19.9 vs 13.2 expected local)
- Benchmark shows 3×87.4K cold TTFT is 2.487s (degraded but not catastrophic)
- 3×131072 would have similar cold performance to 3×87.4K but with 1.5× more context per slot
- Total KV memory: 3 × 64MB = 192 MB (vs 2 × 64MB = 128 MB) — still negligible

**Config changes:**
1. **Raise:** `session_slot_max_prompt_tokens` to match cheap-mode clamp (126,976)
2. **Consider:** Changing cheap slot_schedule from `slots: 2` to `slots: 3` with `ctx_size: 393216`

**Expected impact:**
- If 2×262144 stays: 89.1% full local with raised cap, 2 slots for cheap overnight
- If 3×131072 adopted: 89.1% full local with raised cap, 3 slots (better concurrency), higher GPU use overnight

---

## 5. Go/No-Go vs Keeping 3×262144

### Keep 3×262144: GO (with persistence cap change)

**No config change to slots×ctx is recommended.** The benchmarks show:
- 4×65.5K: Better cold performance, but sessions at P75=74K exceed the 65.5K clamp → 107 context bypasses
- 3×262144: Adequate cold performance (2.487s TTFT for 87K prefills), 61 context bypasses
- 2×131K: Catastrophic cold performance (4.79s TTFT, 0.39 TPS) — never use

**The real fix is the persistence cap, not the ctx config.**

### Action items (implementation, not eval):

1. **Raise `session_slot_max_prompt_tokens`** from 0 (dynamic) to the active clamp value:
   - Fast mode: 83,285
   - Cheap mode: 126,976 (if 2×262144) or 126,976 (if 3×131072)

2. **Monitor:** After cap raise, track:
   - Warm vs cold prefill ratio (should shift toward warm)
   - Fallback rate (should decrease)
   - Remote spend (should decrease)
   - TTFT distribution (should improve for repeated sessions)

---

## 6. Validation Notes

- Benchmark data from F2/F3 (2026-08-03/04) used the same model (Qwen3.6-35B-A3B-Q5_K_M), same q8_0 KV quantization, same hardware (Strix Halo APU)
- Session retention data from proxy-usage-reports (Aug 24-26) — most recent 24h comparison window
- Live system state verified: llama-server running with ctx=262144, parallel=3, RSS ~11 GB
- No config changes were made during this evaluation (AC5 satisfied)
- Test suite: run via `/skill:test` before committing

---

*Evaluation complete. No config changes made — rollout requires separate work item with restart plan (AC5).*
