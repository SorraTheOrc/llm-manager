# Busy-engine local-status read measurement & occupancy-authority verdict

**Work item:** LP-0MW3ED5IC000L82Z (child of LP-0MW10EZXX004M7BI)
**Measurement window:** 2026-11-09, live engine (no synthetic numbers)
**Base commit:** `dev` @ `7059cc651`
**Session:** herdr-1794195907-3592844-23050
**Engine:** llama.cpp build `b8782-e97492369`; router `:8080`; Qwen3 model child `:43745`; proxy `:8000`; presets `slots=2,ctx=131072` per slot.

This report answers one question with evidence: **which read path is authoritative for
local-engine occupancy while a generation runs, and what poll budget does that justify?**
It replaces the intake's OPEN QUESTION ("must be settled before code") with a measured
verdict.

---

## 1. Method

- Latency sampled with `httpx` at the client, per-request timeout stated. Child and router
  sampled **interleaved** over the same window so both see the same engine load.
- Stub latency for the ingest scripts is not used; every number below is a real read
  against the running engine (measurement scripts: `/tmp/measure_busy.py`,
  `/tmp/measure_contended.py`, `/tmp/measure_prefill.py`).
- Load conditions measured separately:
  1. **Decode-only** — one or both slots decoding ordinary prompts.
  2. **Contended prefill** — a controlled ~20k-token prompt prefill driven directly at the
     child, which is the condition that reproduces the intake's multi-second hangs.
- A scratch `--metrics` instance (mxbai-embed, CPU-only, `-ngl 0`, port `:18099`) was used
  to validate the `/metrics` endpoint and its cost **without touching production**. It was
  stopped afterwards; production `/health` verified `200` on `:8000` and `:8080`.

---

## 2. Results

### 2.1 `/slots` under decode-only load

| Endpoint | Samples | p50 | p95 | max | Returned-with-processing |
|---|---|---|---|---|---|
| child `:43745/slots?model=Qwen3` (one slot decoding) | 20 | **31 ms** | **124 ms** | 1.875 s | 20/20 |
| router `:8080/slots?model=Qwen3` (same window) | 20 | **31 ms** | **32 ms** | 32 ms | 20/20 |
| child `:43745/slots?model=Qwen3` (both slots decoding) | 20 | **37 ms** | **37 ms** | 38 ms | 20/20 (2 slots) |

**Decode alone does not block `/slots` on this build.** p50 is ~31–37 ms and p95 stays
inside both poll budgets (2.0 s counts, 1.0 s detail) with two slots decoding.

### 2.2 `/slots` under a long prefill (the real defect)

A controlled ~20k-token prompt prefill on the child, sampled while it ran:

| Metric | Value |
|---|---|
| Samples | 11 (8 s timeout each) |
| Samples > 1 s | **10 / 11** |
| Samples that timed out at 8 s | 10 |
| p50 latency | **8.009 s** (timeout-bounded) |

**The blocking is prefill-bound, not decode-bound.** During prompt processing `/slots`
does not answer inside either budget; this is the intake's multi-second hang (`1.06 s`,
`6.93 s`, `>12 s` observed earlier under live audit prefills). This matches the sibling
long-prefill item (LP-0MW7V2DK400365ER): the `/slots` read serialises behind prefill.

### 2.3 Cheap endpoints (both child and router)

| Endpoint | p50 | p95 | max |
|---|---|---|---|
| child `/health` | 0.28 ms | 0.50 ms | 0.61 ms |
| child `/props` | 0.38 ms | 0.60 ms | 0.65 ms |
| child `/v1/models` | 0.24 ms | 0.53 ms | 0.69 ms |
| router `/health` | 0.43 ms | 0.92 ms | 1.14 ms |
| router `/props` | 0.23 ms | 0.30 ms | 0.31 ms |
| router `/v1/models` | 0.43 ms | 0.55 ms | 0.59 ms |

**Confirmed: sub-millisecond under load, on both child and router.** `/props` advertises
`total_slots=2`, `is_sleeping=false`, `endpoint_slots=true`, **`endpoint_metrics=false`**.

### 2.4 `/metrics`

- **Production Qwen3 child:** `GET /metrics` → **HTTP 501** (0.28 ms). `--metrics` is not in
  the preset; `/props` reports `endpoint_metrics=false`. `llamacpp:requests_processing`
  therefore does **not** exist on the deployed engine.
- **Scratch `--metrics` instance (mxbai-embed, CPU):** `/props` reports
  `endpoint_metrics=true`; `GET /metrics` → **HTTP 200 in 0.41–4.43 ms** (well under the
  50 ms threshold) and exports `llamacpp:requests_processing` (present, value 0 at idle).
  The 3 concurrent CPU embeddings completed between samples, so a `requests_processing`
  vs `/slots` paired-agreement figure was **not** conclusively captured on the scratch
  instance (the decode-instance agreement check is part of the production rollout).

### 2.5 Nightly throughput baseline (for the ±2 % threshold)

From `~/proxy-usage-reports/performance_metrics_state.yaml` (same fingerprint
`slots=2,ctx=131072`):

| night | decode tok/s | prompt-eval tok/s |
|---|---|---|
| 2026-11-03 | 34.2 | 128.6 |
| 2026-11-04 | 33.1 | 111.0 |
| 2026-11-05 | 33.2 | 111.7 |
| 2026-11-06 | 34.4 | 125.5 |
| 2026-11-07 | 34.3 | 116.2 |
| 2026-11-08 | 33.9 | 110.1 |

Decode speed: mean ≈ **33.85 tok/s**, σ ≈ **0.57 tok/s (~1.7 %)**, max–min spread
**~3.8 %**. The nightly series is an observational baseline: its night-to-night noise is
*at or above* the ±2 % threshold, so a single before/after night cannot resolve a 2 %
change. A multi-night mean (≈5 nights per side ⇒ standard error ≈0.76 %) would.

---

## 3. Thresholds, applied mechanically

The success thresholds were fixed before measuring. Each is applied to the measured data:

| Threshold | Measured | Verdict |
|---|---|---|
| `/metrics` p95 ≤ 50 ms **under decode** | Not measurable on the deployed engine (501, `endpoint_metrics=false`); 0.41–4.43 ms on a CPU scratch instance, not under decode | **FAIL / not satisfied** |
| `requests_processing` agrees with `/slots` occupancy ≥ 95 % | Not measurable on production (metric absent); scratch instance did not hold requests between samples (inconclusive) | **FAIL / not satisfied** |
| decode throughput within ±2 % | Baseline captured (33.1–34.4 tok/s, σ≈1.7 %, spread 3.8 %); "after" requires the flag enabled (no production change here) | **not measured** (baseline only) |

**Result: the `/metrics` path is rejected for adoption now.** All three thresholds are
unsatisfied, and — independently of measurement — `/metrics` is *aggregate* (no per-slot
id/phase/`context_tokens`), so it could not replace `/slots` for the per-slot fields herdr
and the Web UI consume. It remains a viable *supplement* only after the flag is enabled and
the three thresholds are measured under decode.

---

## 4. Verdict

### 4.1 Authoritative source per payload field

| Payload field(s) | Authoritative source | Why |
|---|---|---|
| `total_slots`, geometry, `is_sleeping` | **`/props`** | Sub-ms under load on both ports; carries `total_slots` and capability flags. |
| liveness (`llama_server_running`) | **`/health`** | Sub-ms; never blocks behind prefill. |
| cheap model state | **`/v1/models`** | Sub-ms. |
| `available_slots`, per-slot `slots[]` (`is_processing`, `phase`, `context_tokens`, lease/abandonment), per-slot ETA | **`/slots`** (bounded, with last-known fallback) | The **only** per-slot source; responsive under decode (~31–37 ms) but blocks for the duration of a long prefill (10/11 reads >1 s; 8 s timeouts). Must be bounded and explicitly stale, never blocking. |
| coarse occupancy via `requests_processing` | **`/metrics`** — **not adopted now** | Unavailable on the deployed engine (501); aggregate only; adoption deferred to LP-0MW3ED6K5000RT5R pending the measured thresholds. |

**Fallback (adopted):** `/props` for geometry, `/health` for liveness, `/v1/models` for cheap
model state, and a **bounded** `/slots` read for the rich per-slot fields, with last-known
caches and explicit staleness when a read times out.

### 4.2 Justified poll budget

- **Detail (`STATUS_QUERY_TIMEOUT`) stays bounded at ~1 s — do not raise it to cover
  prefill.** Under decode `/slots` is ~31 ms (p95 124 ms), so 1 s is ample. During a long
  prefill it will time out; raising the budget to the prefill duration (multi-second to
  >12 s) would serialise every 1 s status poller behind the prefill and blow the status
  response budget. A timeout must degrade explicitly (serve last-known + `slots_stale` /
  `slots_detail_stale`), never block.
- **Counts stays bounded (~2 s or lower).** `total_slots` can be sourced from the sub-ms
  `/props` read instead of a second `/slots` counts read.
- **Cheap fields (`/props`, `/health`, `/v1/models`) use a short budget (<1 s)** and are
  always answered.

### 4.3 What "active generation" means for the staleness target

Decode-only windows are **not** stale on this build (p95 124 ms). The "≈100 % stale"
baseline is specific to **long-prefill** windows. The policy must therefore treat prefill
periods as the expected staleness window and make that staleness honest (last-known counts
+ explicit flags + no fossil `generating`), rather than pretending a prefill-blocked
`/slots` is fresh.

---

## 5. Not measured here (and who owns it)

- `requests_processing` vs `/slots` agreement under **decode**: needs a `--metrics`
  generation instance. Owned by LP-0MW3ED6K5000RT5R (production `--metrics` rollout + its
  throughput check).
- decode throughput **after** enabling `--metrics`: same owner; the baseline above is the
  "before" series it should compare against across several nights.
- A conclusive paired agreement sample on the scratch instance (CPU embeddings completed
  between samples).

No synthetic throughput or occupancy value was substituted. These gaps do not change the
verdict: the `/metrics` path is **not adopted now**, so the fallback in §4.1 stands.

---

## 6. Reproducing

The controlled measurements are two files kept outside the tree during the run
(`/tmp/measure_busy.py`, `/tmp/measure_prefill.py`); the reusable *test* reproduction of the
degraded status payload is the committed harness from LP-0MW3ED3XN003TPXI
(`proxy/tests/test_slots_busy_engine_policy.py`, commit `7059cc651`). This report records the
live-engine measurement; the harness pins the payload contract in CI.
