# status_request log payload — live verification (LP-0MSK9XXCN0077CMA)

This document records the operational verification of AC4 for
LP-0MSK9XXCN0077CMA ("Proxy: log status_request response payload + client IP").

## What was verified

`GET /llama/local/status` logs a self-describing `status_request` line that
includes the client IP (with source), the response payload (idle/busy, slot
counts, model) and request latency — visible in the plain-text proxy.log.

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

## Findings

- Every `status_request` line is self-describing: `client_ip` + `client_ip_source`
  are present for both direct and reverse-proxy (header) clients, so pollers are
  attributable.
- All response fields are present as `key=value` pairs: `active_query`,
  `available_slots`, `current_model`, `latency_ms`, `llama_server_running`,
  `local_owner_lease_remaining_seconds`, `local_owner_session_id`,
  `model_switch_in_progress`, `total_slots`.
- Idle/busy state is visible (`active_query=true`, `available_slots`/`total_slots`).
- The endpoint JSON response contract is unchanged (logging only).

## Reproduce

```bash
curl -s http://127.0.0.1:8000/llama/local/status > /dev/null
curl -s -H "X-Forwarded-For: 203.0.113.42" http://127.0.0.1:8000/llama/local/status > /dev/null
grep status_request /var/log/llama-proxy/proxy.log | tail -3
```

Requires a running proxy (see `scripts/start-proxy.sh`).
