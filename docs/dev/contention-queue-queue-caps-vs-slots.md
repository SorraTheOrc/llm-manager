# Queue Caps vs Slots — Projected Gain Analysis

**Window:** 2026-08-22 22:42:05 -> 2026-08-28 00:00:00
**Files:** 8
**Work item:** LP-0MTF0G54K003F9SI (T3 of LP-0MTED3OFP006I7NO)

## Semantics and scope

The two levers affect **different event streams** and the difference matters for
interpreting the numbers (verified against router/provider source, see T3 notes):

- **denied** (`local_dispatch_denied`) events are **lease-gate 503 rejections** for
  explicit-session requests: the `max_local`-sized dispatch-lease pool (2 slots) is
  fully reserved by other sessions (active + inactive unexpired leases), so the new
  request is rejected with a client-visible 503 — it never reaches the contention
  queue.  These are the bursty stream quantified in T2 (median inter-arrival ~4–5s).
- **queue-path arrivals** (dispatch + fallback_after_queue events) occur when both
  slots are saturated for the queue decision path; they wait up to the cap, then
  either dispatch local or fall back to the next remote provider (client still
  receives a response, just not from the cheap tier).

Consequences:
- **Raising slots** converts 503 failures into served local requests (availability
  + cost win) and, secondarily, removes almost all queue pressure (fewer requests
  ever see both slots busy).  The denied-saved count below is a **lower bound**
  (queue-path arrivals would also mostly dispatch directly under 3 slots).
- **Raising queue caps** converts remote-fallback responses into local ones (cost
  / latency-selectivity win) but does nothing for the denied/503 stream.

## Baseline (current config: 2 slots, wait 60s, depth 4)

- Dispatched (model): **243**
- Fallback-after-queue (model): **172**
- Depth-capped fallbacks: **0** (depth 4 is never hit at 60s
  wait — T2 confirms max queue depth 4 with zero depth-capped falls)
- Queue-wait (model): median=41.1s p90=60.0s p95=60.0s max=60.0s

> Caveat: the T1 model was validated on **counts** (±10%; dispatch −2.4%, fallback
> +3.6% vs observed).  Model-projected waits skew higher than observed (observed
> dispatch-wait median 17.6s, p90 47.2s per T2) because requests are served at the
> next slot-free event.  Use the **Δ counts** for decisions; treat wait columns as
> model-projected upper-ish bounds.

## Queue-cap tuning scenarios

| wait | depth | Δ dispatched | Δ fallbacks | depth-capped | max queue | wait p95 |
|------|-------|:-----------:|:-----------:|:----------:|:---------:|:--------:|
| 120s | 8 | +35 | -35 | 1 | 8 | 120s |
| 120s | 12 | +35 | -35 | 0 | 9 | 120s |
| 120s | 16 | +35 | -35 | 0 | 9 | 120s |
| 180s | 8 | +54 | -54 | 15 | 8 | 180s |
| 180s | 12 | +54 | -54 | 2 | 12 | 180s |
| 180s | 16 | +54 | -54 | 0 | 13 | 180s |
| 300s | 8 | +69 | -69 | 37 | 8 | 300s |
| 300s | 12 | +72 | -72 | 18 | 12 | 300s |
| 300s | 16 | +74 | -74 | 4 | 16 | 300s |

Reading: each row shows the projected extra local dispatches (Δ dispatched) and the corresponding reduction in fallback-after-queue if the policy caps were raised to (wait, depth).  Δ fallbacks == −Δ dispatched because every saved request was previously a fallback-after-queue timeout.
### Queue-wait impact (model-projected dispatched waits)

| wait | depth | median | p90 | p95 | max |
|------|-------|:------:|:---:|:---:|:---:|
| 120s | 8 | 66.7s | 120.0s | 120.0s | 120.0s |
| 120s | 12 | 67.1s | 120.0s | 120.0s | 120.0s |
| 120s | 16 | 67.1s | 120.0s | 120.0s | 120.0s |
| 180s | 8 | 70.7s | 180.0s | 180.0s | 180.0s |
| 180s | 12 | 88.1s | 180.0s | 180.0s | 180.0s |
| 180s | 16 | 89.0s | 180.0s | 180.0s | 180.0s |
| 300s | 8 | 66.6s | 300.0s | 300.0s | 300.0s |
| 300s | 12 | 89.7s | 300.0s | 300.0s | 300.0s |
| 300s | 16 | 111.8s | 300.0s | 300.0s | 300.0s |

The price of each saved dispatch is added queue wait for the requests that previously timed out at 60s; at a 300s wait cap the p95 dispatched wait reaches the cap (some requests that would have been served far sooner by remote fallback now sit in the queue for minutes).

## Slots increase scenarios

| slots | Δ dispatched (denied saved) | denied remaining |
|-------|:----------------------------:|:----------------:| 
| 3 | +1237 | 2 |
| 4 | +1237 | 2 |

The 2 remaining denied events report active=3+ and are outliers from agent/audit session traffic (e.g. `audit-`/`herdr-` sessions) rather than cheap-mode chat fan-out; both 3-slot and 4-slot scenarios save the same 1237 because no cheap-mode denied request reports active=3.

### Slots 3 — context-bypass penalty

*Prior work (LP-0MSAOQTJS000FFVM / LP-0MSY0SDAS0031Y7F) found that increasing local model slots shrinks per-slot context, increasing ``large_context_bypass`` and ``context_too_large`` fallbacks.  Adding a 3rd slot shrinks the per-slot context by ~33 % (262144/3≈87.4K tokens/slot vs 131072 at 2 slots).  The exact bypass penalty was measured in the context-size evaluation work items (LP-0MSAOQTJS000FFVM); see those items for the measured ``large_context_bypass`` rate increase when slots were increased.  This estimate does NOT include the secondary queue improvement from fewer queue-path arrivals (denied events that bypass the queue entirely reduce queue pressure).*

### Slots 4 — context-bypass penalty

*Prior work (LP-0MSAOQTJS000FFVM / LP-0MSY0SDAS0031Y7F) found that increasing local model slots shrinks per-slot context, increasing ``large_context_bypass`` and ``context_too_large`` fallbacks.  Adding a 4th slot shrinks the per-slot context by ~50 % (262144/4=65536 tokens/slot vs 131072 at 2 slots).  The exact bypass penalty was measured in the context-size evaluation work items (LP-0MSAOQTJS000FFVM); see those items for the measured ``large_context_bypass`` rate increase when slots were increased.  This estimate does NOT include the secondary queue improvement from fewer queue-path arrivals (denied events that bypass the queue entirely reduce queue pressure).*

## Headline comparison

| Lever | Best Δ served-local | Added queue wait | Notes |
|-------|:-------------------:|:----------------:|-------|
| Queue cap (wait=300s, depth=16) | +74 | p95 300s (vs 60s) | +4 depth-capped fallbacks; no context risk |
| Slots → 3 | +1237 * (lower bound) | 0s (no queue) | converts 503s to served-local; context-bypass penalty (see below) |

**Headline:** *Slots* is the dominant lever: adding a 3rd cheap-mode slot projects **~1237** requests served-local (a **~17×** larger effect than the best queue-cap tuning at **~74**) for the same 8 replayed windows — and it also eliminates almost all queue pressure.  The cost is the context-bypass penalty (per-slot context shrinks ~33% at 3 slots; more ``large_context_bypass``/``context_too_large`` fallbacks per LP-0MSAOQTJS000FFVM) plus the VRAM cost of a 3rd KV slot.  Queue-cap tuning is config-only (zero VRAM, zero context risk) and worth doing if slots stay at 2: raising wait to 120s + depth to at least 8 recovers ~35 dispatches (~20% of fallback-after-queue) with bounded added wait; larger waits (180s–300s) add diminishing returns and push p95 wait toward the cap.
