# Local slot observability

How `GET /llama/local/status` (and, from LP-0MW1VVMI1005XU0J, `/health`) reports
the local llama-server slot surface, what each freshness field means, and how
to recover when the surface degrades.

Background: on 2026-11-04 the payload published
`slots_snapshot_age_seconds=29825.67` (8 h 17 m) **and kept claiming
availability** from that frozen list (`available_slots=2 capacity_hint=idle`),
so the downtime dispatcher — which fails closed on `slots_stale` — scheduled no
work for over eight hours. This document describes the bounds and signals that
make that impossible.

## Snapshot-age ceiling: `local_slots_max_snapshot_age_seconds`

A per-slot snapshot may be served from the last-known cache while a fresh
`/slots` read is failing, but only while it is young enough to be useful.

| Source | Setting | Notes |
|---|---|---|
| config | `server.local_slots_max_snapshot_age_seconds` | takes precedence |
| environment | `LOCAL_SLOTS_MAX_SNAPSHOT_AGE_SECONDS` | fallback when config is absent |
| default | **derived**: `max(120, 10 × STATUS_POLL_CADENCE_SECONDS)` | 300 s with the 30 s default cadence |

The default is deliberately *derived in code* from the status poll cadence
(`observability.STATUS_POLL_CADENCE_SECONDS`), not hard-coded to an hour: the
guarantee is that no consumer can act on a snapshot older than roughly the next
few polls. The absolute ceiling is **not** the same knob as
`SLOT_COUNTS_STALE_AFTER_SECONDS` (1 h), which bounds only the `available`/
`total` counts cache.

An unparseable, non-positive or absent value falls through to the next source,
so a bad setting can never disable the bound or raise inside the status
endpoint.

### What happens past the ceiling

When the served snapshot's age exceeds the ceiling:

- `slots` is empty — the fossil list is no longer presented as live;
- `available_slots` and `total_slots` are `0` — availability is **not** claimed
  from the stale list (the counts fail-open on the configured pool does not
  apply either);
- `capacity_hint` is `unknown` (never `idle`);
- `slots_detail_stale` is `true`;
- `slots_stale` is `true`;
- `slots_stale_reason` is `snapshot_too_old`.

Below the ceiling nothing changes: the snapshot is still served with its true
age in `slots_snapshot_age_seconds`, `slots_snapshot_source="snapshot"`, and
the transport-level reason (`query_timeout` / `query_error`).

An **unset** fetch stamp is treated as an unknown age, not an infinitely old
one, so an un-initialised cache never trips the ceiling.

## Visibility on `/health`

A stale slot surface is a serving deficiency, so it is visible on `/health`
without reading `/metrics`:

| Field | Meaning |
|---|---|
| `slots_surface_status` | `ok` (fresh), `stale` (degraded — see the reason), `unknown` (backend unreachable, so nothing can be asserted) |
| `slots_stale` | mirrors the status payload's flag |
| `slots_stale_reason` | the same enumerated reason the status payload publishes |
| `slots_snapshot_age_seconds` | age of the served snapshot, or `null` when unknown |
| `slots_max_snapshot_age_seconds` | the applied ceiling |

The values come from one shared evaluation (`observability.evaluate_slots_surface`),
and `/health` mirrors the **last published** `/llama/local/status` reason — so a
consumer that only polls `/health` still sees the cause.

**`ready` and the top-level `status` are deliberately unaffected.** `ready`
gates request admission; an unobservable slot *surface* does not stop the proxy
serving a request, and flipping `ready` would fail-closed every consumer that
reads it for liveness (the 2026-11-02 orphan-backend regression,
LP-0MVZT9L3N001XABB AC2, is the cautionary precedent). Use
`slots_surface_status != "ok"` as the non-OK signal.

An ordinary idle window (llama-server down, or no model loaded yet) reports
`slots_surface_status: "unknown"`, never `"stale"`, so a normal quiet period is
not reported as a defect.

### Recovery

No restart is required — and none should be performed to clear this condition.
The next successful `GET /slots?model=<id>` poll refreshes the detail cache,
which clears the staleness reason and restores the live list automatically.

If `snapshot_too_old` persists, the cause is a *failing* `/slots` read, not a
stale ceiling. Follow `slots_stale_reason`:

| `slots_stale_reason` | Meaning | Operator action |
|---|---|---|
| `router_model_missing` | the proxy could not resolve a model id to query (its resolution fell through to the last resort) | confirm the router reports a loaded model in `GET /models`; check `current_model` |
| `child_port_undiscovered` | router mode, but no model child port could be discovered from `llama-server`'s log | check the log path is current and the log contains `proxying request to model … on port …` |
| `query_timeout` | the `/slots` fetch timed out | check llama-server load; the router serialises `/slots?model=…` behind a busy child |
| `query_error` | any other fetch failure (non-200, connection error, malformed body) | inspect `llama-server`'s log for the `/slots` status code |
| `snapshot_too_old` | the served snapshot exceeded `local_slots_max_snapshot_age_seconds` | fix the underlying fetch failure; no restart needed |

Every value is a member of `observability.SLOTS_STALE_REASONS`. Treat an
unrecognised value as **stale** (fail closed), never as fresh.

## Staleness never starves the local route

A stale surface is an observation about the *poll*, not proof the backend is
down, so the pre-dispatch routing gate
(`observability._local_backend_unreachable_remaining_seconds`) only treats it as
unreachability when a **reachability probe measured unreachable within the
probe window**.

Background (LP-0MWCOMI160011F6C, 2026-11-11): a llama-server respawn moved the
child port at 20:55 and the `/slots` probe then failed for 93 minutes. The
counts cache froze, and once it passed `local_slots_max_snapshot_age_seconds`
the gate skipped local on every request (60 s at a time) — with no probe on the
skip path, so the skip was self-sustaining and kept a healthy, idle backend
starved while `/health` reported `backend_reachable: true` the whole time.

| Setting | Meaning |
|---|---|
| `server.local_reachability_probe_max_age_seconds` | freshness window of a measured reachability verdict (default **60 s**) |
| `server.local_backend_skip_seconds` | TTL of a skip decided from that verdict (default **60 s**) |

The gate's decision order:

1. **Armed skip deadline** (`local_backend_unreachable_until`) — set by a
   deterministic connection-refused dispatch or a failed reachability probe;
   unchanged behaviour, so a refused connect still fail-fasts immediately.
2. **Stale snapshot + recent `unreachable` probe verdict** — skip, bounded by
   `local_backend_skip_seconds`.
3. **Anything else** — dispatch local (fail open). A stale snapshot alone, an
   expired (or absent) probe verdict, or a passing probe never skips, so a
   recovered backend is re-admitted on the next probe cycle instead of waiting
   for a lucky `/slots` success.

Reachability evidence is published by `lifecycle._probe_backend_reachable`
(the `/health` and `/models` probe that already runs on every `/health` poll)
and by the fail-fast arm path, and is bounded by
`local_reachability_probe_max_age_seconds` so a verdict can never decide a skip
forever.

### `/slots` failure classification

`llama_slots_query_failures_total{reason=...}` labels every failed `/slots`
read. `observability.classify_slots_transport_failure` maps an exception onto
that vocabulary once for all three `/slots` callers (counts, detail, occupancy
sampler):

| Exception | Label |
|---|---|
| `httpx.TimeoutException` (`ConnectTimeout`, `ReadTimeout`, `WriteTimeout`, `PoolTimeout`), builtin `TimeoutError` | `timeout` |
| `httpx.NetworkError` (`ConnectError`, `ReadError`, `WriteError`, `CloseError`), builtin `ConnectionError`/`OSError` | `connection_error` |
| anything else | `unknown` |

Before this mapping every httpx transport error landed in `unknown`, which is
why the 93-minute outage could not be diagnosed from the metric.

## Verification script

`scripts/verify_slots_observability.py` produces the reproducible PASS/FAIL
evidence block for this contract. It is read-only: plain HTTP GETs and log
reads, no state mutation.

```bash
# against a running proxy (recommended: pass the router so the model-parameter
# contract is probed directly)
python3 scripts/verify_slots_observability.py --router-url http://localhost:8080

# machine-readable
python3 scripts/verify_slots_observability.py --router-url http://localhost:8080 --json
```

Exit `0` means every check passed; `1` means at least one failed and the failing
checks are named.

Checks performed:

| Check | What it proves |
|---|---|
| `slots_stale=false` | the surface is not degraded (reason printed alongside) |
| `snapshot age below ceiling` | a served snapshot is within `slots_max_snapshot_age_seconds`; "no snapshot served" passes, because a live read has nothing stale |
| `health surface ok` | `slots_surface_status == "ok"` |
| `health/status reason agreement` | `/health` and `/llama/local/status` report the same reason |
| `router requires ?model= and answers it` | **a model-less `/slots` returns 400 and `/slots?model=<id>` returns 200** — the exact contract the 2026-11-04 freeze tripped over |
| `/slots 200 in llama-server log` | the backend actually served a slot read after the restart |

> **Why the model-parameter check is a probe, not a log grep.** llama.cpp's
> `log_server_r` line does **not** include the query string on this build — it
> logs `GET /slots 127.0.0.1 200` — so a `GET /slots?model=…` string cannot be
> read out of the log. Probing the router directly is the same evidence and
> strictly stronger: it proves the 400/200 asymmetry rather than inferring it.
> When a build does log the query string, the script's log matcher accepts a
> model-bearing `200` as well.

### Recorded live verification (2026-11-04)

After the proxy was restarted onto the fixed build
(`bash scripts/start-proxy.sh --restart`, healthy immediately), the verifier
reported, verbatim:

```text
available_slots: 2
health_slots_stale_reason: None
slots_detail_stale: False
slots_max_snapshot_age_seconds: 300.0
slots_snapshot_age_seconds: None
slots_snapshot_source: None
slots_stale: False
slots_stale_reason: None
slots_surface_status: ok
total_slots: 2

[PASS] slots_stale=false: slots_stale=False reason=None
[PASS] snapshot age below ceiling: no snapshot served (live read, nothing stale)
[PASS] health surface ok: slots_surface_status='ok'
[PASS] health/status reason agreement: health=None status=None
[PASS] router requires ?model= and answers it: model-less=400 model-bearing=200 model='Qwen3'
[PASS] /slots 200 in llama-server log: found a successful /slots read after the restart

RESULT: PASS
```

Recovery required **no llama-server restart** — the surface cleared as soon as
`/slots` answered.
