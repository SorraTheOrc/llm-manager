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
- `/metrics`: `llama_slots_child_port_skipped_total{reason="model_not_loaded"}`
  (LP-0MVWBXEAI008YZ7B, slice 3 of LP-0MVVCY3HQ0009F0K) — incremented once per
  **deliberately skipped** poll, i.e. a `/slots` query for a model the router's
  TTL-cached `/models` loaded set definitively reports as *not loaded*. Such a
  poll issues **no** request and does **not** touch
  `llama_slots_child_port_fallbacks_total`, so the two counters partition the
  traffic: **fallback** = a *loaded* model whose child port could not be
  discovered (the LP-0MVP7525P008RXMF AC3 regression signal), **skipped** = an
  unloaded model (expected — there is no slot state to learn). `proxy.log`
  carries the matching throttled debug line (one per 300 s):
  `slots_child_port_skipped: model not loaded — skipping /slots poll for
  model=<m> (source=<s>)`.
- Loadedness is read from the **same** cached `/models` snapshot as the model
  resolution rule (`is_model_known_unloaded()`, no second cache). Only a
  model the router lists **and** reports as not loaded is skipped; an
  errored/stale probe or a name the router does not list is *indeterminate*
  and fails open to polling as before (the guard never skips on
  `srv.current_model` disagreement alone).
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
- **Unloaded-model polls are no longer issued at all** (LP-0MVWBXEAI008YZ7B):
  a `/slots` query for a model the router's cached `/models` snapshot reports
  as not loaded is skipped before any request, so it contributes neither a
  router request nor a router-responder 400/500. The skip is counted in
  `llama_slots_child_port_skipped_total{reason="model_not_loaded"}`: a
  non-zero value there is a *deliberate* poll suppression, not a missing
  poll, and the unloaded-model class (`gpt120` in the 2026-10-28 window) no
  longer reaches the router at all.
- **Attribution rule when a router-responder `/slots` 400/500 does appear.**
  Read the two counters, not the access line (the query string is absent): a
  **skipped** delta means the proxy suppressed the poll, so a post-guard
  unloaded-model event cannot be proxy-originated and an unexplained 400/500
  is either the unknown/missing class (a proxy defect) or an external probe; a
  **fallback** delta together with the `slots_child_port_unavailable` warning
  means a *loaded* model's child port was not discoverable and the poll did
  traverse the router — that is the genuine discovery regression to chase.
- An **unloaded**-class 400 is expected truth and is not a regression: the
  definitive model-absent answer keeps callers failing open
  (`slots_failure_is_definitive_model_absent()`, see *The three router `/slots`
  400 classes* below), and skipping the poll is admission-neutral.
- The post-fix 24 h confirmation window is AC3 of LP-0MVVCXUDZ007XID0, using
  the recipe in this document. The skip rule itself is verified by
  `proxy/tests/test_slots_unloaded_model_skip.py`
  (`TestUnloadedModelSkipSignal` = AC1/AC6, `TestLoadedModelFallbackSignal`
  = AC2, `TestFailOpenCharacterisation`, `TestAdmissionUnchanged` = AC7).

## The three router `/slots` 400 classes

A router-responder `GET /slots ... 400` line **cannot be attributed from
`llama-server.log`**: llama.cpp's router access line omits the query string, so
`srv log_server_r: done request: GET /slots 127.0.0.1 400` is identical with
and without `?model=` (measured 2026-10-29). Per-query attribution (a caller
tag plus a `http_400` failure metric on the detail/progress paths) is deferred
to LP-0MVTDMXXH005D5D5; until then the proxy distinguishes the classes by
construction:

| Class | Request | Interpretation | Proxy response |
|---|---|---|---|
| **Missing** `?model=` | `GET /slots` with no model | Proxy defect: a `/slots` construction site dropped the parameter | Not reachable: the query helpers append `?model=` unconditionally, and the static fence `proxy/tests/test_slots_model_param_guard.py` fails the suite on any model-less construction |
| **Unknown / unresolvable** | `GET /slots?model=<id>` where `<id>` is not a `/models` id/preset | Proxy defect: a hard-coded or guessed default instead of a resolved id | Not reachable through the rule: `resolve_local_model_id()` (`proxy/proxy/router_helpers.py`) resolves explicit → `srv.current_model` → the sole non-embed `/models` preset → the canonicalised live child-port log signal → the log id verbatim → a *logged* `"Qwen3"` last resort. Guarded by `proxy/tests/test_slots_model_id_resolution.py` |
| **Unloaded** | `GET /slots?model=<valid id>` with no live instance | **Expected truth**: idle-model unload, or the window before a preload finishes | The definitive "model absent" signal: `slots_failure_is_definitive_model_absent()` (`proxy/proxy/observability.py`) returns True for `http_400`/`http_404`, and callers fail open instead of substituting cached slot counts — a fossil snapshot of the last request (LP-0MSVP7XJ6008QPKX, LP-0MVKSXR4P001V0CM AC3) |

Reading a 400 in a triage window:

- `router:400` counts in the responder split are **bounded, not attributed**:
a non-zero count is acceptable only when each event is explained by the
unloaded-model class or by a call site removed in LP-0MVP7HV7Q001AJF8. An
unexplained event reopens the item.
- Attribute the **unloaded** class with the proxy counters, not the access
  line: `llama_slots_child_port_skipped_total{reason="model_not_loaded"}`
  counts polls the proxy deliberately suppressed (no request issued at all,
  since LP-0MVWBXEAI008YZ7B), so a **skipped** delta cannot produce a 400;
  `llama_slots_child_port_fallbacks_total{reason="no_child_port_router_mode"}`
  plus the `slots_child_port_unavailable` warning counts polls that *did*
  traverse the router because a loaded model's child port was not
  discoverable (a regression, never the unloaded class). A 400 explained by
  neither counter is the unknown/missing class and reopens the item.
- The proxy-side metric `llama_slots_query_failures_total{reason="http_400"}`
records the class for the status path (`_query_slots`); the detail and progress
paths record no failure metric yet — that gap is LP-0MVTDMXXH005D5D5.
- A **missing** or **unknown** 400 is a defect even when the request would
succeed on retry: the `?model=` is what the router resolves, so a wrong id
degrades silently to "0/0 slots" through fail-open.
- The README states the same rule for operators
(`proxy/README.md` → *Child-port discovery* → *Model resolution rule*).

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
## Post-fix 24 h window verification (LP-0MVVLKPBM007F6U6, 2026-11-01 → 2026-11-02) — RED

**Work item:** LP-0MVVLKPBM007F6U6 (verifies the LP-0MVVCXUDZ007XID0 `/health` probe fix)
**Window:** 2026-11-01T00:41:55Z → 2026-11-02T00:41:55Z (fix-era restart 2026-10-31T00:41:55Z + 24 h;
measured 2026-11-02T08:00Z; raw evidence in
[`router-slots-500-window-2026-11-01-to-02.json`](router-slots-500-window-2026-11-01-to-02.json))

### TL;DR — verdict RED, the probe fix holds, a residual floor persists

Router-responder `GET /slots ... 500` is **not zero** in the window:
**≈ 385 in-window (superset 752 across three files)**, i.e. AC1 of the
verification item is unmet. The `/health` probe **contributes 0** (verified
live: 10 sequential `/health` curls produced 0 new `/slots` lines, and the
deployed code probes `/models` + `/health` only). The known `gpt120`
chat-fallback confound (LP-0MVVCY3HQ0009F0K) is **absent**. The residual is
proxy-initiated Qwen3 (chat model) slot polling that still traverses the
router port :8080 and 500s on busy-child moments at a persistent ~16/day
floor (rate-matched to the parent window's non-probe share; the parent's
full attribution of 1,171 to the probe overstates its share). Filed as
**LP-0MVYZ8Z16004QIRG**.

### Method and window file sets

Access lines carry no timestamps; the window is scoped by file. Rotation
boundaries: 11-01 00:00 copytruncate → `llama-server.log-2026-11-01.gz`;
11-01 21:26:39 in-process restart rotation → `llama-server.1.log`;
11-02 00:00 copytruncate → `llama-server.log-2026-11-02`; live
`llama-server.log` continues to measurement time. Superset
[11-01 00:00 → 11-02 07:57] ⊇ window.

| File | Content range (UTC) | router 200 | router 400 | router 500 | child 200 | child 400/500 |
|---|---|---|---|---|---|---|
| `llama-server.1.log` | 11-01 00:00 → 21:26:39 | 401 | 1,864 | **331** | 60,301 | 0 |
| `llama-server.log-2026-11-02` | 11-01 21:26:39 → 11-02 00:00 | 17 | 92 | **31** | 5,614 (+1×503) | 0 |
| `llama-server.log` (live) | 11-02 00:00 → 07:57 | 396 | 1,941 | **390** | 19,412 | 0 |
| **superset total** | | **814** | **3,897** | **752** | **85,328** | **0** |

In-window estimate ≈ 385 (time-proportional; bounded 385 ≤ in-window ≤ 752).

### Restart integrity (AC3)

Three `Starting LLama Proxy Server` events inside the window (11-01
13:12:53, 13:17:50, 21:26:41 UTC); current pid 1487007 (lstart
11-01T21:26:39Z) replaced the fix-era pid 3944027 (lstart 10-31T00:41:55Z).
At least one in-window pid change ⇒ **`llama_slots_child_port_fallbacks_total`
delta unavailable**; per-file log counts only. The llama-server router was
pid 32809 until 21:26:39, then 54819 (no router respawn at 13:12/13:17).

### `slots_stale` share and fallback signals (AC2)

- 20,926 `status_request` polls; **slots_stale=True 8,929 / 20,926 = 42.7 %**
  (above the parent post-fix window's 31.4 %, below the pre-fix 64.4 %
  baseline); active-prefill subset 14,482 / 20,926 = 69.2 %.
- 3 throttled `slots_child_port_unavailable` WARNINGs (one per restart,
  `source=log_signal`, model=Qwen3); current-process counter
  `llama_slots_child_port_fallbacks_total{reason="no_child_port_router_mode"} 1`
  and `llama_slots_query_failures_total{reason="timeout"} 9,001` (delta N/A).

### Residual 500s: the probe is not the source

The probe fix is effective on both axes: the live repro adds 0 `/slots`
lines per `/health` call (pre-fix: one per call, 6×500 per 10 calls), and
`proxy/proxy/lifecycle.py:751-754` probes `/models` + `/health` only in
router mode. The residual router 500/400 lines are therefore a separate
Qwen3 chat-model polling path that still hits the router port (live `ss`
shows persistent proxy→:8080 connections), matching the same busy-child /
unloaded-model mechanism documented above. Target signal remains
`router:500 == 0`.

## Post-fix 24 h window verification (LP-0MVP7HX2S008IU57, 2026-11-16 → 2026-11-17) — router 500s **zero**, fallback axis RED

**Work item:** LP-0MVP7HX2S008IU57 (verifies the LP-0MVP7525P008RXMF child-port
discovery fix)
**Window:** 2026-11-16T15:20:47Z → 2026-11-17T15:20:47Z (anchored to the
group-2 proxy restart `pid 3720326`, `lstart` 2026-11-16T15:20:47Z; measured
2026-11-17T18:20:11Z, 2 h 59 m after the window closed; raw counts in
[`router-slots-500-window-2026-11-16-to-17.json`](router-slots-500-window-2026-11-16-to-17.json))

### TL;DR — AC1 met (router 500s **0**), AC4 unmet (one boot-time Qwen3 fallback)

- **Router-responder `GET /slots ... 500` = 0** across a 42 h 20 m superset
  (109,330 `/slots` events, 15 h 20 m before the window and 2 h 56 m after it)
  that contains every in-window access. The residual floor measured in the
  2026-11-01 → 11-02 window (752 router 500s in a 31.9 h superset, ≈ 16/day
  in-window) **has not recurred**: 0 router 500s on every measured content day
  from 2026-11-07 except a single line on 2026-11-13.
- **One chat-model fallback occurred**, at window start **+4.4 s**
  (`model=Qwen3`, `source=log_signal`) — the anchor restart's own boot
  transient, before child-port discovery had resolved. Counter delta **1**,
  warning lines **1**, and **0** for the remaining 23 h 59 m 56 s. AC4's
  chat-model zero-tolerance rule therefore fails ⇒ **verdict RED** by the
  documented rule, attributed to boot rather than to the busy-child / unloaded
  model mechanisms. Filed as **LP-0MWL0R0UT004G9XE**.
- `slots_stale` share **18.03 %** of all `status_request` polls — the lowest
  measured post-fix figure (31.4 % in 2026-10-27 → 28, 42.7 % in 2026-11-01 →
  02, pre-fix baseline 64.4 %).

### Method and window file sets

Access lines in `llama-server*.log` carry no timestamps, so the window is
scoped by file: the daily `copytruncate` copy whose name is the rotation date
and whose content is the **previous** UTC day, plus the live file. The union
is a 42 h 20 m **superset** of the 24 h window (15 h 20 m of pre-window and
2 h 56 m of post-window traffic), so zero router 500s across it proves zero
inside the window.

Every access line in both files belongs to the same model child,
**container pid 44409** — the persistent host-first `llama-server` (host pid
3122336, started 2026-11-15T01:06:52Z, *before* the window). One generation
served the whole superset; because it predates the window, the pid does not
bound the file set in time (unlike the 2026-10-27 window, where the child
restarted with the window anchor).

| File | Content range (UTC) | router 200 | router 400 | router 500 | child 200 | child 400/500 |
|---|---|---|---|---|---|---|
| `llama-server.log-2026-11-17` | 11-16 00:00 → 11-17 00:00 | 1 | 1,070 | **0** | 60,893 | 0 |
| `llama-server.log` (live) | 11-17 00:00 → 18:20 | 0 | 310 | **0** | 47,056 | 0 |
| **superset total** | 11-16 00:00 → 11-17 18:20 | **1** | **1,380** | **0** | **107,949** | **0** |

Router-responder traffic is **1.26 %** of all `/slots` events (1,381 of
109,330) — every slot query except the boot poll and the 400s is served
directly by the child port.

`proxy.log` is timestamped and self-rotates every 6 h, so each candidate's own
first/last line timestamps were derived and the files that tile the window
kept (`proxy.log.2026-11-16_152045` ends at 15:20:45, before the window; the
live `proxy.log` starts at 15:21:01, after the window end):

| File | First line (UTC) | Last line (UTC) | In-window lines | `status_request` polls | `slots_stale=True` | fallback WARNs |
|---|---|---|---|---|---|---|
| `proxy.log.2026-11-16_15` | 11-16 15:20:48.866 | 11-16 21:20:47.400 | 155,471 | 13,695 | 2,532 | 1 |
| `proxy.log-2026-11-17_00` | 11-16 21:20:48.857 | 11-17 00:00:03.557 | 62,202 | 7,507 | 941 | 0 |
| `proxy.log.2026-11-16_21` | 11-17 00:00:05.674 | 11-17 03:20:46.648 | 28,776 | 1,771 | 489 | 0 |
| `proxy.log.2026-11-17_03` | 11-17 03:20:48.056 | 11-17 09:20:46.252 | 45,875 | 2,529 | 506 | 0 |
| `proxy.log.2026-11-17_09` | 11-17 09:20:52.805 | 11-17 15:20:51.185 (clipped at 15:20:47) | 71,009 | 3,551 | 771 | 0 |
| **totals** | 11-16 15:20:48 → 11-17 15:20:47 | | **363,333** | **29,053** | **5,239** | **1** |

### Router vs child responder split — `scripts/slots_500_triage.py` output verbatim (AC2)

```text
$ python3 scripts/slots_500_triage.py --log-dir /var/log/llama-proxy --llama-file llama-server.log-2026-11-17 --json
  "classification": {
    "responder_status": {
      "44409:200": 60893,
      "router:200": 1,
      "router:400": 1070
    },
    "proximate_causes": {},
    "stats": {
      "slots_total": 61964,
      "slots_200": 60894,
      "slots_400": 1070,
      "slots_500": 0,
      "cancel_events": 400,
      "model_400_events": 8,
      "proxying_events": 5391,
      "prefill_events": 4975,
      "checkpoint_events": 5918,
      "near_proxy_500s": 0,
      "near_cancel_500s": 0,
      "near_prefill_500s": 0,
      "near_checkpoint_500s": 0
    }
  }

$ python3 scripts/slots_500_triage.py --log-dir /var/log/llama-proxy --llama-file llama-server.log --json
  "classification": {
    "responder_status": {
      "44409:200": 47056,
      "router:400": 310
    },
    "proximate_causes": {},
    "stats": {
      "slots_total": 47366,
      "slots_200": 47056,
      "slots_400": 310,
      "slots_500": 0,
      "cancel_events": 271,
      "model_400_events": 4,
      "proxying_events": 3975,
      "prefill_events": 3684,
      "checkpoint_events": 4444,
      "near_proxy_500s": 0,
      "near_cancel_500s": 0,
      "near_prefill_500s": 0,
      "near_checkpoint_500s": 0
    }
  }
```

(`--json` output truncated to the `classification` block; the full documents
are `per_file_slots` / `correlation` / `fix_options` as for every run of this
CLI. The live file grew between the two capture instants — 18:20:11Z is the
single measurement instant for both blocks above.)

Daily router-500 trend, same measurement over the daily `copytruncate` copies
(labelled by **content** day — the copy named by its rotation date holds the
previous UTC day; 2026-11-11 is a partial copy, 2,434 child 200s):

| Content day | 11-07 | 11-08 | 11-09 | 11-10 | 11-11* | 11-12 | 11-13 | 11-14 | 11-15 | 11-16 | 11-17 (to 18:20) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| router 500s | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | **0** |
| router 400s | 193 | 496 | 343 | 611 | 31 | 439 | 515 | 93 | 264 | 1,070 | 310 |

Reading: the residual floor from the 2026-11-01 → 11-02 window
(`~16/day in-window`, 752 in the superset) is **absent**: every measured day
from 2026-11-07 shows 0 router 500s except a single line on 2026-11-13,
consistent with the remediation closed as LP-0MVYZ8Z16004QIRG on 2026-11-04.
The target signal `router:500 == 0` now holds over a full 24 h window.

### Restart integrity (AC6)

- Window proxy `pid 3720326`, `lstart` **2026-11-16T15:20:47Z**, still running
  at measurement (`etime` 1-02:58:55) ⇒ **no pid change, no restart inside the
  window**.
- Exactly **one** `Starting LLama Proxy Server` line in the window
  (15:20:49.786) — the anchor process's own boot line, 2.8 s after its
  `lstart`; 0 in the other four `proxy.log` files.
- A single model-child generation (`pid 44409`) serves every `/slots` line in
  both `llama-server` files.
- Because the serving process **started at the window start**, its monotonic
  counters *are* the window counters: no post-window restart destroyed the
  counter delta this time (unlike the 2026-10-27 → 28 and 2026-11-01 → 02
  windows).

### Fallback signals (AC4)

```text
# HELP llama_slots_child_port_fallbacks_total Proxy /slots queries falling back to the router port because no model child port could be discovered, by reason
# TYPE llama_slots_child_port_fallbacks_total counter
llama_slots_child_port_fallbacks_total{reason="no_child_port_router_mode"} 1.0
llama_slots_query_failures_total{reason="timeout"} 12210.0
```

| Signal | Window value | Verdict |
|---|---|---|
| `llama_slots_child_port_fallbacks_total{reason="no_child_port_router_mode"}` delta | **1** | chat-model zero-tolerance **fails** |
| `slots_child_port_unavailable` WARNING lines | **1** (`model=Qwen3`, 15:20:51.411Z) | — |
| chat-model (`Qwen3`) fallbacks, steady state (15:20:52 → 15:20:47 next day) | **0** | clean |
| embed-model (`mxbai-embed`) fallbacks | **0** | non-blocking, none |

The single fallback is the **first poll of the boot that anchors the window**:

```text
2026-11-16 15:20:51,406 - INFO - Started TTS server (PID: 3721608)
2026-11-16 15:20:51,408 - INFO - Slot occupancy sampler started (interval 10.0s)
2026-11-16 15:20:51,411 - WARNING - slots_child_port_unavailable: no model child port discovered for model=Qwen3 (source=log_signal); falling back to router port 8080 — every proxy-initiated /slots poll will traverse t...
2026-11-16 15:20:51,435 - WARNING - llama-server --version returned 127
2026-11-16 15:20:51,560 - INFO - Adopted running llama-server (model=router parallel=1 router_mode=True); preserving warm prompt cache
```

The same restart-correlated transient appeared in the 2026-11-01 → 02 window
(3 × `model=Qwen3`, each within 3 s of a restart). It is a boot-ordering
issue, not the busy-child / unloaded-model mechanism the fix targets, and it
makes a zero-tolerance fallback check unsatisfiable on any restart-anchored
window. Filed as **LP-0MWL0R0UT004G9XE**; the `gpt120` unloaded-model confound
(LP-0MVVCY3HQ0009F0K) did **not** fire this window (4 `gpt120` requests, 0
fallback warnings).

### `slots_stale` share (AC3)

| Figure | Window value | Comparison |
|---|---|---|
| **Primary:** all `status_request` polls | **5,239 / 29,053 = 18.03 %** | 31.4 % (10-27 → 28), 42.7 % (11-01 → 02), **64.4 %** pre-fix baseline (2026-10-26) |
| Secondary: active-prefill subset (`active_query=true` or `local_active_query=true`) | **5,018 / 16,197 = 30.98 %** (the subset is 55.75 % of all polls) | — |

### Baseline and reproducibility (AC5)

- **Baseline available this window**: the serving process started exactly at
  the window start, so its counters were 0 at *t0* by construction — the
  counter delta needs no reconstruction (the previous window's missed-snapshot
  limitation does not apply).
- No `llama-server.log` line-count baseline was captured; the file-set
  argument is the over-count direction instead — a 42 h 20 m superset (15 h
  20 m before the window, 2 h 56 m after it) with 0 router 500s.
- Commands used: the two `slots_500_triage.py` invocations above; the
  `proxy.log` derivation loop (`awk` first/last timestamp per candidate) plus
  a window-clipping tally over the five kept files; `ps -o pid,lstart,etime`
  and a single `/metrics` scrape per site.

### Residual router 400s — observation only

1,380 router-responder `/slots ... 400` in the superset (1.26 % of slot
events) — the `?model=` / model-resolution class, not this item's AC. The
dedicated 24 h 400 == 0 verification is **LP-0MVU8003X0018WT8** (same window);
these counts are recorded here for it.

### Limitations

- `llama-server` access lines are timestamp-free ⇒ the window is scoped by
  file and proven by the over-count direction: the 42 h 20 m superset (15 h
  20 m pre-window, 2 h 56 m post-window) reports 0 router 500s, and both
  overhangs carry none.
- The persistent host-first `llama-server` (container pid 44409 / host pid
  3122336) predates the window, so its lines cannot be time-bounded by pid as
  in the 2026-10-27 window.
- The fallback counter delta **includes the window's first 4.4 s** (the anchor
  restart's boot transient) because process lifetime == window.
- `proxy.log` window end is clipped at 15:20:47 inside
  `proxy.log.2026-11-17_09`, whose last line is 15:20:51.
- `llama_slots_query_failures_total{reason="timeout"} 12,210` is a
  process-lifetime value (window + overhang) and is reported as an observation
  only.
