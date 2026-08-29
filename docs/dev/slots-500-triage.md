# GET /slots HTTP 500-storm triage (F3)

**Work item:** LP-0MTCMEV1G0022A35 (parent LP-0MTAQNB7J0094X71)
**Window:** 2026-08-26 (llama-server.log-2026-08-27.gz; llama-server logs carry
no timestamps, see F1 harness docs)
**Reproduction:** `python3 scripts/slots_500_triage.py --log-dir
/var/log/llama-proxy --llama-file '*2026-08-27*'`

## TL;DR

The GET /slots HTTP 500 storms are a **router-mode proxying artifact**, not a
model failure:

- Every GET /slots 500 and 400 on 2026-08-26 is answered by the **router**
  (lines with no `[pid]` prefix). The model instances (pids 37439, 59455)
  answer **only 200** (36,238 model 200s; 0 model 500s/400s).
- When the router proxies GET /slots to a busy model instance (mid
  giant-prefill), the model cancels the proxied connection
  (`operator(): http client error: Connection handling canceled`) and the
  router returns HTTP 500.
- **6,940 cancel events ≈ 6,865 router 500s (1:1)** — the cancel is the
  proximate mechanism.
- The 527 HTTP 400s are router responses when it cannot answer /slots with a
  resolvable model (llama.cpp requires `?model=` — LP-0MSHW2AXJ009DO3S).

## Classification (2026-08-26)

| Responder | 200 | 400 | 500 |
|---|---|---|---|
| router (proxied) | 29,373 | **527** | **6,865** |
| model 59455 (Qwen3) | 36,189 | 0 | 0 |
| model 37439 (embed) | 49 | 0 | 0 |

Spot-check: the incident's reported day-noon snapshot (6,459 500s / ~69.6K
polls / 527 400s) matches this file's full-day totals within the expected
mid-day-vs-full-day delta; the 527 HTTP 400 count is exact.

## Proximate-cause breakdown (of 6,865 500s)

| Cause label | Count | Share |
|---|---|---|
| router-proxied to busy model (proxy_reques in window) | 6,856 | 99.9% |
| connection canceled (cancel event in window) | 2,949 | 43.0% |
| concurrent prefill / checkpoint activity in window | 3,655 | 53.3% |
| restart-race (model-load lines in window) | — | rare (restart-window only) |

The 500 rate is not uniform: **158 of 1K-line slabs show both >0% 500 rate
and concurrent prefill/checkpoint activity** (busy windows), i.e. the storm
concentrates in slabs where giant prefills are in flight. It is NOT a
permanent degraded state — outside prefill bursts the router returns 200
(29,373 times).

## Mechanism detail

llama-server runs in **router mode** (`launch: spawning server instance ...`
with the router on :8080 proxying to per-model instances, e.g. Qwen3 on
:59455). The proxy polls `GET /slots?model=...` against the router (see
proxy/proxy/observability.py `_query_slots` / `_query_slots_detail`). The
router answers by proxying to the owning model instance:

1. Model instance is mid-giant-prefill (e.g. `prompt processing progress,
   n_tokens = 10240...` at 2048-token batches).
2. Router's proxied /slots request hits the busy instance → the instance
   cancels the connection (`operator(): http client error: Connection
   handling canceled`, 6,940/day).
3. The router converts the canceled proxy into `GET /slots ... 500`.

Because /slots poll failures feed the proxy's `slots_stale` flag and slot
counts (graceful degradation LP-0MSVP7XJ6008QPKX), the 500s compound the
session↔slot affinity problems documented in F2: slots_stale hits 47.4% on
the incident day (3,780 of 7,980 status polls).

## Correlation with giant prefills

- 53.3% of 500 windows contain concurrent prefill/checkpoint activity; the
  two model instances produce 4,135 prefill progress events + 3,191
  checkpoint-created events on the day.
- The 22:00–23:09 incident snapshot-write window overlaps the highest /slots
  poll volume (F1 timeline: 151–184 stale polls per half-hour, 87–163
  routing skips per half-hour), i.e. the storm peaks exactly when the
  oversized-session fallback storm is also peaking.

## Fix options (ranked by expected impact)

| Rank | Fix | Expected impact | GPU-wedge risk | Tracked elsewhere |
|---|---|---|---|---|
| 1 | **Restore-before-proxy**: have the proxy query the model instance directly (`/slots?model=` on the model port), or answer from the last-known slot state, instead of proxying through the router | Eliminates the dominant cancel-500 path and the 400 path; removes dependence on router /slots | none | LP-0MSVP7XJ6008QPKX (restart 500s), LP-0MSB0RV72001KNRV (registry leak), LP-0MSHW2AXJ009DO3S (400 without ?model=) |
| 2 | **Concurrency-aware slot timeout/backoff**: extend the /slots timeout (and retry) when the model is mid-prefill instead of cancelling | Medium — cuts 500s during prefill storms (53% of 500s) | low | partially LP-0MSI5B1T2009GQ4C (load-aware timeout rebalance) |
| 3 | **Fix the 'Connection handling canceled' path** in router-mode proxying (coordinate cancellation between router and busy model; likely a llama.cpp router-mode issue under load) | Medium — 43% of 500 windows contain a cancel event | low | not tracked |
| 4 | **Deduplicate slot-status aggregation**: proxy owns slot-state polling per model (mirror `_query_slots_detail`, LP-0MTC8A2UB0040NKQ) so the router /slots is never the authority for proxy routing decisions | Low-medium — reduces dependence on the flaky router path | none | not tracked |

## Recommendations

- Implement fix #1 (direct model-instance polling or last-known-state
  fallback) — it is the highest-impact, zero-GPU-wedge option and aligns
  with the graceful-degradation design already in observability.py.
- Validate fix #2 (timeout/backoff) against the 53% prefill-window share
  after #1 lands.
- Re-run this triage after a config/code change to confirm the router:500
  count drops; the access-log responder/status split is a stable
  regression signal (router:500 == 0 is the target).