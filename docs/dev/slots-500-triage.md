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

## Residual-path analysis (LP-0MVP7525P008RXMF, 2026-10-26)

The rank-1 fix landed (direct child-port polling in the status handler and
SSE broadcast loop) but was **inert in production** because child-port
discovery silently failed, plus two call sites never consulted it:

| Responder (2026-10-26 window) | 200 | 400 | 500 |
|---|---|---|---|
| router (no `[pid]` prefix) | 36,062 | 86 | **1,754** |
| model child Qwen3 | 37,851 | 0 | 0 |
| model child mxbai-embed | 4 | 0 | 0 |

- 1,754 router 500s ≈ 1,819 `Connection handling canceled` events (1:1, the
  same mechanism as 2026-08-26) and 1,327 of them fall in the last 10% of
  the log, interleaved with a 49,152-token prefill (`n_tokens = 49152,
  progress = 0.406`).
- Root cause A (discovery): `_discover_local_child_port()` only understood
  the startup spawn line. After logrotate the live `llama-server.log` had
  **zero** spawn lines (they are in rotated files) yet was 9.6 MB of real
  content, so `_is_sparse_log()` returned False and the helper cached
  `None` for the whole life of the llama-server pid.
- Root cause B (call sites): `proxy/proxy/ui.py` SSE initial payload built
  the /slots URL from `server.llama_server_port` with no discovery, and
  `_check_slot_availability()` skipped discovery whenever the local provider
  endpoint (the router) was supplied.

### Fix (AC1-AC6)

1. **Rotation-proof discovery.** `_discover_local_child_port()` now scans a
   second, live source — the router's per-request line `proxying request to
   model <model> on port <port>` — which keeps being written for the whole
   life of the process and survives logrotate. The tail of the current log
   is scanned (tail-first, then the head spawn line, then rotated logs for
   the sparse-fallback case). Positives are cached per `(pid, model)` and
   re-verified every `CHILD_PORT_CACHE_TTL` (default 300 s); negatives are
   cached with a short TTL (`CHILD_PORT_MISS_TTL`, default 30 s) so a signal
   that appears later (e.g. the first proxied request after a restart) is
   picked up instead of being pinned to `None`.
2. **One shared resolution path.** `_resolve_local_slots_target()`
   (router_helpers.py) now resolves the target for *every* proxy-initiated
   `/slots` query: status handler, SSE broadcast + initial payload, slot
   availability check, prefill-progress/lease-liveness probes, and the
   per-model slot poller. An external endpoint (non-loopback host, or a
   loopback port that is not the router port) is left untouched; the local
   router endpoint is rewritten to the discovered child port.
3. **Loud degradation.** On discovery failure the resolver logs a throttled
   warning (`slots_child_port_unavailable: ...` — one per 300 s) and
   increments `llama_slots_child_port_fallbacks_total{reason=...}` (router
   mode vs single-server mode), so the silent `except Exception: pass`
   regression cannot recur undetected.

### Observability signal

- `proxy.log`: `slots_child_port_unavailable: no model child port
  discovered for model=<m>; falling back to router port 8080 — ...`
- `/metrics`: `llama_slots_child_port_fallbacks_total` (by reason).
- `llama-server.log`: post-restart `/slots` polls should be answered by the
  model child (`[pid]`-prefixed lines) rather than the router; router :500
  counts toward 0. Router-responder `/slots` traffic is proxy-originated only
  from slot polling and (before LP-0MVVCXUDZ007XID0) the `/health`
  reachability probe; the probe no longer queries `/slots` (see below).

### Live verification (pre-deploy, 2026-10-26 production log)

```text
spawn-only (old path): None          # inert as documented
live proxy line      : 41395
child discovery      : 41395         # routes /slots to the child
resolve target       : (None, 41395) # endpoint=http://localhost:8080 rewritten
```
## Post-fix 24 h window verification (LP-0MVP7HX2S008IU57, 2026-10-27 → 2026-10-28)

**Work item:** LP-0MVP7HX2S008IU57 (verifies the LP-0MVP7525P008RXMF fix)
**Window:** 2026-10-27T12:00:00Z → 2026-10-28T12:00:00Z (measured
2026-10-30T19:09Z; raw counts in
[`router-slots-500-window-2026-10-27-to-28.json`](router-slots-500-window-2026-10-27-to-28.json))

### TL;DR — verdict RED

Direct child-port polling now carries essentially all slot traffic (135,028
child 200s vs 505 router 200s in the 48 h superset), and the router 500 rate
fell ~95 % (2,753 on 2026-10-26 → 128 on 2026-10-27). It is **not zero**:
**1,171 router-responder `GET /slots ... 500` lines** remain in the window
superset, and they have a *different* root cause from the pre-fix storm.

### Method and window file sets

`llama-server.log` access lines carry **no timestamps**, so the window is
scoped by file. `logrotate` writes a `copytruncate` copy at 00:00 UTC whose
name is the rotation date and whose content is the previous 24 h, so the
window straddles exactly two files (a superset with 12 h of extra traffic on
each side):

| File | Content range (UTC) | In-window coverage |
|---|---|---|
| `llama-server.log-2026-10-28.gz` | 2026-10-27T00:00Z → 2026-10-28T00:00Z | second half of the window (12:00 → 24:00) |
| `llama-server.log-2026-10-29.gz` | 2026-10-28T00:00Z → 2026-10-29T00:00Z | first half of the window (00:00 → 12:00) |

`proxy.log` **is** timestamped and self-rotates every 6 h, so the window was
filtered by each file's own observed first/last timestamp. Six files tile the
window with no overlap:
`proxy.log.2026-10-27_10.gz`, `proxy.log.2026-10-27_16.gz`,
`proxy.log-2026-10-28_00.gz`, `proxy.log.2026-10-27_22.gz`,
`proxy.log.2026-10-28_04.gz`, `proxy.log.2026-10-28_10.gz`
(15,953 `status_request` polls, 211,192 log lines in total).

```bash
python3 scripts/slots_500_triage.py --log-dir /var/log/llama-proxy \
  --llama-file 'llama-server.log-2026-10-28.gz' --summary
python3 scripts/slots_500_triage.py --log-dir /var/log/llama-proxy \
  --llama-file 'llama-server.log-2026-10-29.gz' --summary
```

### Router vs child responder split (per file, AC1/AC2)

| File | router 200 | router 400 | router 500 | child 200 | child 400 | child 500 |
|---|---|---|---|---|---|---|
| `llama-server.log-2026-10-28.gz` (2026-10-27) | 79 | 243 | **128** | 72,959 (41395: 72,955, 59947: 4) | 0 | 0 |
| `llama-server.log-2026-10-29.gz` (2026-10-28) | 426 | 2,164 | **1,043** | 62,069 (41395: 31,919, 42051: 30,150) | 0 | 0 |
| **superset total** | **505** | **2,407** | **1,171** | **135,028** | **0** | **0** |

Every router-responder request came from `127.0.0.1` (the proxy process —
`ss` sampling over 60 s showed no other local client of `:8080`). Child
instances answered **only** 200/404: the 500s are purely a router-mode
proxying artefact, as in the original triage.

Daily router-500 trend (same measurement, one file per UTC day):

| Day | 2026-10-25 | 2026-10-26 | 2026-10-27 | 2026-10-28 | 2026-10-29 | 2026-10-30 (partial) |
|---|---|---|---|---|---|---|
| router 500s | 20,134 | 2,753 | 128 | 1,043 | 604 | 140 |

### `slots_stale` share (AC3)

| Figure | Window value | Pre-fix baseline |
|---|---|---|
| **Primary:** all `status_request` polls | **5,005 / 15,953 = 31.4 %** | 579 / 899 = 64.4 % (2026-10-26) |
| Secondary: active-prefill subset (`active_query=true` or `local_active_query=true`) | 3,739 / 5,968 = 62.7 % | — |

### Fallback signals (AC4)

- **Counter delta: unavailable.** The process that served the window
  (pid 3791954, started 2026-10-27T10:16:11Z) has since been replaced (pid
  3224511, started 2026-10-30T10:35:07Z), so the monotonic
  `llama_slots_child_port_fallbacks_total` cannot be read for the window.
  No proxy restart occurred **inside** the window (zero `Starting LLama Proxy
  Server` lines across all six window files).
- **In-window warning lines: 1** (throttled to one per 300 s, so this
  undercounts) — `model=gpt120` at 2026-10-28T10:45:00.404Z. `gpt120` is a
  local **chat** model that was not loaded, i.e. a chat-path fallback →
  `LP-0MVVCY3HQ0009F0K` (filed `discovered-from:LP-0MVP7525P008RXMF`).
- **`mxbai-embed` fallbacks in-window: 0.**

### Residual router 500s: the `/health` probe, not slot polling

The remaining 1,171 router 500s are explained by the proxy's **`/health`
reachability probe**, which used to target the router by design:

- `proxy/proxy/handlers.py:1170` — `health_check()` calls
  `srv._probe_backend_reachable(llama_port)` on **every** `/health` request.
- `proxy/proxy/lifecycle.py:725` — `_probe_backend_reachable()` probed
  `"/slots?model=<current_model>"` first, then `/models`, `/health`, all on
  `http://localhost:<router_port>`.

**Fixed (LP-0MVVCXUDZ007XID0):** the router-mode probe now issues `/models`
and `/health` only. Both answer reachability without touching the busy child,
and the removed `/slots` path could not change the verdict anyway — a 5xx fell
through to the next path while any 2xx-4xx already returned `True`. The probe
stays cheap: no child-port log discovery, no router-proxied slot query.

Because that probe never calls `_resolve_local_slots_target()`, it increments
neither the fallback counter nor the `slots_child_port_unavailable` warning —
the exact observed shape (1,171 router 500s, ~0 fallback signals).

Live reproduction (2026-10-30T19:18Z): 10 sequential
`curl -s -o /dev/null localhost:8000/health` calls produced exactly one
router-responder `GET /slots` line each — **6 × 500**, 3 × 200, 1
indeterminate. Filed as `LP-0MVVCXUDZ007XID0`
(`discovered-from:LP-0MVP7525P008RXMF`, priority high).

### Limitations

- `llama-server` lines are timestamp-free, so the two kept files are a
  **superset** (2026-10-27T00:00Z → 2026-10-29T00:00Z). Zero router 500s
  across the superset would have proven zero inside the window; a non-zero
  count does **not** localise the 500s to the window, so the in-window count
  is bounded (`≤ 1,171`) rather than exact.
- The pre-window baseline snapshot (2026-10-27T12:00Z) was never taken; the
  counter-based reconstruction is moot because the serving proxy has since
  restarted.
- Fallback WARNING lines are throttled (one per 300 s) and therefore
  undercount; the counter is authoritative but was unreadable for this window.

### Target signal going forward

`router:500 == 0` is the regression target. The residual non-zero floor in
this window came from the `/health` reachability probe, which polled
`GET /slots?model=<current>` on the router once per `/health` request; that
probe path is removed (LP-0MVVCXUDZ007XID0 — router mode now probes
`/models` + `/health` only, so `/health` no longer touches the router's
`/slots` and no longer queues behind a busy child).

Consequences when reading these figures:

- Counts measured **before** the LP-0MVVCXUDZ007XID0 fix (i.e. this window)
  carry the probe's 500s; they are expected noise, not a polling regression.
- Counts measured **after** the fix ships should reach 0; if they do not, the
  500s are a genuine polling-path regression and
  `llama_slots_child_port_fallbacks_total` (plus the child-vs-router 200
  split) tells you which path regressed.
- The post-fix 24 h confirmation window is AC3 of LP-0MVVCXUDZ007XID0, using
the recipe in this document.

## The responder split counts *every* status code (LP-0MVVWTWGY004BO5H)

The tool used to pre-seed a fixed set of status counters
(`slots_200`/`slots_400`/`slots_500`) and then index it with whatever status
the access log reported, so the **first non-200/400/500 `/slots` line aborted
the entire run**:

```text
error: analysis failed: 'slots_503'
KeyError: 'slots_503'
```

Since `llama-server` child instances do answer `/slots` with other codes
(`503` while a child is shutting down or saturated), the documented
`--log-dir` invocation over the live corpus failed outright.

Fixed behaviour — read the split as an **open** key space:

- the per-status counters are defaulting (`Counter`/`defaultdict`), so any
  status the corpus reports appears as `slots_<status>` in
  `classification.stats` (`slots_503`, `slots_502`, `slots_429`, …) instead of
  raising `KeyError`;
- `classification.responder_status` carries one key per
  `<responder>:<status>` pair actually seen, so an unexpected status is
  attributed to the responder that produced it (`router:503`, `<pid>:503`)
  rather than dropped;
- a `/slots` line whose status cannot be parsed (a shape `_ACCESS_RE` does not
  match) is reported as `unknown:?` and counted in
  `classification.stats.slots_unparsed`. It is **not** added to
  `slots_total`, which remains the sum of statuses the tool could parse (the
  denominator of the 500 rate);
- `per_file_slots` keeps its per-file status tally, `?` for unparsed lines.

The `router` vs child classification (`router` = no `[pid]` prefix, otherwise
the child pid) is unchanged, so the F3 numbers above stay reproducible.

### Live corpus run (2026-10-31, `--summary`)

```text
python3 scripts/slots_500_triage.py --log-dir /var/log/llama-proxy --summary
```

exits 0 and reports the child 503s that previously crashed the run:

```text
slots_total 871,519   slots_200 756,429   slots_400 13,563
slots_500   101,510   slots_503      17   slots_unparsed 10
router:200  126,405   router:400  13,563   router:500  101,510
child 503 keys: 36521, 38029, 41395, 43181, 43957, 44791, 50193,
                52221, 54265 (×2), 54457 (×2), 56105, 56943, 59057,
                60939, 60965 (×2)  → 17 attributed child 503s
unknown:?   10       (9 child 503 lines concatenated after `llama-server`
                      progress-dot output, plus one `GET /slots/1 … 404`)
```

Reading caveat: those dotted 503 lines are real child responses whose line
prefix defeats `_ACCESS_RE`, so they land in `unknown:?` rather than being
attributed to their pid — the responder split therefore *under*-counts child
503s while still reporting them. Tracked separately
(`discovered-from:LP-0MVVWTWGY004BO5H`).
