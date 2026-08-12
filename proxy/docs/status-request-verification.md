# status_request log payload — live verification (LP-0MSK9XXCN0077CMA, LP-0MSKV3IEQ004ZV88)

This document records the operational verification of AC4 for
LP-0MSK9XXCN0077CMA ("Proxy: log status_request response payload + client IP")
and the additive client-identity extension for poller attribution from
LP-0MSKV3IEQ004ZV88 (source port + stable header-based `client_id`).

## What was verified

`GET /llama/local/status` logs a self-describing `status_request` line that
includes the client IP (with source), the client source port (direct
connections), a stable `client_id` from the session headers when present, the
response payload (idle/busy, slot counts, model) and request latency — visible
in the plain-text proxy.log.

Field semantics (additive — existing fields unchanged):

- `client_ip` / `client_ip_source` — resolved via X-Forwarded-For / X-Real-IP
  when present (`source=header`), else from the direct connection
  (`source=direct`); `client_ip=unknown` only when no client address exists.
- `client_port` — `request.client.port` for direct connections; `unknown` for
  reverse-proxy header paths (headers carry no source port) and when no client
  address/port is present.
- `client_id` — first present of `x-session-id` / `session_id` /
  `x-client-request-id` (the session-header convention in
  `proxy/proxy/session.py`); **omitted entirely** (not `None`/`unknown`) when
  none are sent.

## Method

1. A real HTTP poll was issued against the running proxy
   (`curl http://127.0.0.1:8000/llama/local/status`), once directly and once
   with an `X-Forwarded-For` header to exercise the reverse-proxy header path.
2. The live log (`/var/log/llama-proxy/proxy.log`) was grepped for
   `status_request` lines.

## Observed log lines

Real poller attribution (herdr downtime worker at `192.168.0.199`, direct
connection — `client_ip_source=direct`):

```
2026-08-08 22:45:12,876 - INFO - status_request active_query=true available_slots=3 client_ip=192.168.0.199 client_ip_source=direct current_model=Qwen3 latency_ms=8 llama_server_running=true local_owner_lease_remaining_seconds=None local_owner_session_id=None model_switch_in_progress=false total_slots=3
```

Direct curl poll from localhost (`client_ip_source=direct`):

```
2026-08-08 22:45:13,103 - INFO - status_request active_query=true available_slots=3 client_ip=127.0.0.1 client_ip_source=direct current_model=Qwen3 latency_ms=5 llama_server_running=true local_owner_lease_remaining_seconds=None local_owner_session_id=None model_switch_in_progress=false total_slots=3
```

curl poll with `X-Forwarded-For: 203.0.113.42` — reverse-proxy header path
(`client_ip_source=header`, resolved to the header value):

```
2026-08-08 22:45:13,112 - INFO - status_request active_query=true available_slots=3 client_ip=203.0.113.42 client_ip_source=header current_model=Qwen3 latency_ms=5 llama_server_running=true local_owner_lease_remaining_seconds=None local_owner_session_id=None model_switch_in_progress=false total_slots=3
```

New-style lines with source port + client_id (local verification run of the
LP-0MSKV3IEQ004ZV88 code, direct connections):

```
2026-08-09 14:46:22,596 - INFO - status_request active_query=false available_slots=0 client_id=sess-verify-1 client_ip=127.0.0.1 client_ip_source=direct client_port=57472 current_model=None latency_ms=0 llama_server_running=false local_active_query=false local_owner_lease_remaining_seconds=None local_owner_session_id=None model_switch_in_progress=false total_slots=0
```

Each of three consecutive requests has a distinct `client_port` (57472, 57486,
57496) — ephemeral ports are unique per TCP connection, so pollers on one host
are distinguishable by IP+port even without `client_id`.

## Source-port & client_id attribution — ≥5-min segment

A 5.5-minute verification window (2026-08-09 13:46:39Z–13:52:09Z) exercised the
new code with six simulated pollers at staggered cadences (30s/45s/60s/90s,
offset starts) — mirroring the ~8-pollers-at-~30s pattern observed in the RCA
(WL-0MSK9TUCA00206M7). Each poller sent its own `X-Session-Id` header and a
fresh TCP connection, so the segment shows per-client attribution by
`client_id` **and** distinct `client_port` values:

```
# 47 status_request entries in the 5.5-min window; per-client breakdown:
     11 client_id=herdr-downtime-b
     11 client_id=herdr-downtime-a
      8 client_id=opencode-statusline-a
      7 client_id=opencode-statusline-b
      6 client_id=grafana-monitor
      4 client_id=misc-agent
# every request has a distinct client_port (47/47 unique) — IP+port attribution
# works even without client_id
```

Example lines from the window:

```
2026-08-09 14:46:39,688 - INFO - status_request active_query=false available_slots=0 client_id=herdr-downtime-a client_ip=127.0.0.1 client_ip_source=direct client_port=40380 current_model=None latency_ms=0 llama_server_running=false local_active_query=false local_owner_lease_remaining_seconds=None local_owner_session_id=None model_switch_in_progress=false total_slots=0
2026-08-09 14:46:42,689 - INFO - status_request active_query=false available_slots=0 client_id=grafana-monitor client_ip=127.0.0.1 client_ip_source=direct client_port=56258 current_model=None latency_ms=0 llama_server_running=false local_active_query=false local_owner_lease_remaining_seconds=None local_owner_session_id=None model_switch_in_progress=false total_slots=0
2026-08-09 14:46:46,690 - INFO - status_request active_query=false available_slots=0 client_id=opencode-statusline-a client_ip=127.0.0.1 client_ip_source=direct client_port=56268 current_model=None latency_ms=0 llama_server_running=false local_active_query=false local_owner_lease_remaining_seconds=None local_owner_session_id=None model_switch_in_progress=false total_slots=0
```

Full segment: `~/.local/state/llama-proxy-dev/logs/proxy.log` on the verifying
host (dev-mode log dir); the live equivalent after deployment is
`/var/log/llama-proxy/proxy.log` on the proxy host.

## Findings

- Every `status_request` line is self-describing: `client_ip` + `client_ip_source`
  are present for both direct and reverse-proxy (header) clients, so pollers are
  attributable.
- Direct connections additionally log `client_port` (the requester's ephemeral
  source port), giving IP+port attribution — the strongest signal for distinct
  pollers sharing one host. Header paths report `client_port=unknown`.
- `client_id` (from `x-session-id` / `session_id` / `x-client-request-id`) is
  logged when a poller identifies itself and omitted otherwise — additive only.
- All response fields are present as `key=value` pairs: `active_query`,
  `available_slots`, `current_model`, `latency_ms`, `llama_server_running`,
  `local_owner_lease_remaining_seconds`, `local_owner_session_id`,
  `model_switch_in_progress`, `total_slots`.
- Idle/busy state is visible (`active_query=true`, `available_slots`/`total_slots`).
- The endpoint JSON response contract is unchanged (logging only).

## Deployment & rollout

Logging changes are code-only and require a proxy restart on the host; there is
no config or schema migration.

1. Merge `dev` → `main` (release process) and pull on the proxy host
   (`/home/rgardler/projects/llm`).
2. Restart the proxy service:

   ```bash
   sudo systemctl restart llama-proxy.service   # ExecStart: uvicorn proxy.server:app --port 8000
   sudo systemctl status llama-proxy.service
   ```

3. Smoke-check the new fields are live:

   ```bash
   curl -s http://127.0.0.1:8000/llama/local/status > /dev/null
   grep status_request /var/log/llama-proxy/proxy.log | tail -1   # expect client_port=<NNNN>
   ```

4. **Notify the herdr team** once live (RCA WL-0MSK9TUCA00206M7 owners), so
   idle-window RCA investigators know pollers are attributable by IP+port
   without `ss`/tcpdump snapshots.
5. Post-rollout, capture a real ≥5-min segment from `/var/log/llama-proxy/proxy.log`
   and confirm ≥5 distinct `client_ip`/`client_port` pairs over the window
   (see the segment section above for the expected shape).

## Acceptance-criteria mapping (LP-0MSKV3IEQ004ZV88)

| Parent AC | Where covered |
|-----------|---------------|
| AC1 — status_request logs client IP + source port (unknown for header paths) | Features 1+2 (unit/endpoint/formatter tests; `_resolve_client_port`) |
| AC2 — ≥5-min segment shows per-client attribution (distinct IPs/ports) | Feature 7 — this doc, segment section |
| AC3 — existing log consumers/tooling unaffected (additive) | Features 1–6 (existing `test_status_logging.py` / lease tests still green) |
| AC4 — unit tests for port resolution + fallback; existing tests pass | Features 1+2 (port tests), features 3+5 (identity tests) |
| AC5 — deployment documented (restart + herdr notification) | Feature 7 — Deployment & rollout section |

## Reproduce

```bash
curl -s http://127.0.0.1:8000/llama/local/status > /dev/null
curl -s -H "X-Forwarded-For: 203.0.113.42" http://127.0.0.1:8000/llama/local/status > /dev/null
curl -s -H "X-Session-Id: my-client" http://127.0.0.1:8000/llama/local/status > /dev/null
grep status_request /var/log/llama-proxy/proxy.log | tail -3
```

Requires a running proxy (see `scripts/start-proxy.sh`).
