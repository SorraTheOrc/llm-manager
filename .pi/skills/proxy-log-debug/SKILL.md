---
name: proxy-log-debug
description: "EXECUTE immediately via /skill:proxy-log-debug; do NOT ask permission or confirm. Diagnose a proxy log error or failure-domain/cooldown skip. Trigger on user queries such as: 'proxy log error', 'why was provider X skipped', 'same failure domain', 'what failed on this endpoint', 'decode this proxy log line', 'why did the request fall through to deepseek', 'opencode-go in cooldown'."
---

# Proxy Log Debug

Diagnose a single proxy-log error, provider skip, or fallback-chain event
**without rediscovery**: where the logs live, how to query them, what each
failure line means, and how the failure-domain vs cooldown exclusion works.

For aggregate 24h usage/fallback/error reports, use
`/skill:proxy-usage-analysis` instead — that skill owns the windowed
report pipeline. This skill is for **point diagnosis** of a specific event
("what failed at 13:49?", "why was opencode-go skipped?").

## Log locations (read-only)

| Source | Path | Notes |
|---|---|---|
| Live proxy log | `/var/log/llama-proxy/proxy.log` | 6-hourly rotation, 90-day retention |
| Rotated proxy logs | `/var/log/llama-proxy/proxy.log*` | Siblings (`proxy.log.YYYY-MM-DD_HH`, `.gz`); a rotated file routinely holds data past its encoded rotation time — **filter per-line by timestamp, never by filename** |
| Proxy-usage reports | `~/proxy-usage-reports/report.md`, `errors.json`, `errors.csv` | Daily 05:00 cron (`cron.log`); `errors.json` has provider/model × status breakdowns |

**Do NOT use** these for log reads:

- `journalctl -u llm-proxy` — empty; the proxy does not log to systemd.
- `proxy/proxy/logs/` and `proxy/logs/` in the repo — stale/benchmark logs, not the live stream.
- `GET /logs/tail?source=proxy` — **hangs** (observed 2026-09-10); always `grep` the files directly.
- The uvicorn stdout terminal — unreachable from other sessions.

Live log lines have **no session UUID on skip lines**; correlate by
timestamp (`±1s`) and `provider=` / `session=` fields on neighbouring lines.

## Query recipes

All recipes assume `LOG=/var/log/llama-proxy/proxy.log` (or the relevant
rotated sibling; `zgrep` for `.gz`).

**1. Failure chain around a timestamp** — the skip lines never carry the
cause; the cause is on the lines immediately above (substitute the HH:MM
of interest for `13:49` in every recipe):

```bash
grep "13:49" $LOG | grep -E "ReadTimeout|stall|upstream error|Skipping provider|exhausted|provider=opencode" | tail -30
```

**2. What failed on a failure domain (endpoint)** — search by endpoint
fragment or provider name in the same minute window:

```bash
grep "13:49" $LOG | grep -E "opencode.ai/zen/go|provider=opencode-go"
```

**3. Full session trace** — when a `session=<uuid>` is known:

```bash
grep "01a08a8f-a1a4-700b-8479-1053187f4888" $LOG
```

**4. Provider exhaustion summary** — shows per-provider cooldown remaining
at exhaustion:

```bash
grep "All providers exhausted" $LOG | tail -10
# ... unavailable={'opencode-go-2': 58, 'deepseek-v4-flash': 44}
# (values = cooldown seconds remaining per entry)
```

**5. Cooldown vs failure-domain check** — distinguish the two skip reasons:

```bash
grep -E "in cooldown|same failure domain|same usage-limit account|usage_limit_reset_pending|outside its available_times" $LOG | tail -20
```

## Failure-domain vs cooldown (the key distinction)

Two independent exclusion mechanisms; confusing them misdiagnoses skips.

| | Failure-domain skip | Cooldown |
|---|---|---|
| Scope | **Per-request only** (`attempted_domains` set, reset each request) | **Cross-request**, time-based (`mark_provider_unavailable`, seconds–minutes) |
| Key | Normalized **endpoint URL** (`_failure_domain_key`, `proxy/proxy/provider.py:3027`) — scheme+host lowercased, default ports dropped, trailing slash/fragment stripped | **Provider entry name** (`_entry_cooldown_key`) |
| Log line | `Skipping provider=X: same failure domain as an already-failed entry (domain=<url>)` (`provider.py:3125`) | `Skipping provider=X: <key> in cooldown (<N>s remaining)` |
| Meaning | "Don't retry the same broken gateway via a different API key this request" | "This entry failed recently; quarantined across requests" |
| Example | `opencode-go`, `opencode-go-2`, `opencode-go-3` all map to `https://opencode.ai/zen/go` → one fails, all three skipped for the rest of that request | `unavailable={'opencode-go-2': 58}` at exhaustion = 58s remaining on that entry |

**Usage-limit quarantine is a third mechanism**, keyed on the API-key
**account** (`_usage_limit_account_key`, `provider.py:3079`), not the
endpoint: distinct `api_key_env` entries on the same gateway have
independent limits. Log lines: `same usage-limit account as an
already-exhausted entry` (per-request) and `usage_limit_reset_pending
(account=…, reset_in=…s)` (cross-request). A 402/GoUsageLimitError on one
key does NOT exclude sibling keys; a stall/5xx on the shared endpoint DOES.

## Error taxonomy (what adds a domain to `attempted_domains`)

Every path below ends with `attempted_domains.add(_failure_domain_key(...))`
in the fallback loop (`_proxy_with_remote_fallback_cycle` /
`_proxy_with_fallback_cycle`, `provider.py` fallback section), which is what
produces the "same failure domain" skips downstream.

| Log signature | Cause | Code ref |
|---|---|---|
| `Upstream ReadTimeout … provider=X` → `Upstream stall: retrying … attempt=N backoff=Ns` → `Upstream stall: max retries exhausted … retries=3` | Upstream stopped responding mid-stream (idle timeout); Tier-1 retries (3× backoff) exhausted. `ReadTimeout` is a connection error → `connection_error` attempt, entry cooldown, domain added (fallback-loop `except` tail). The intermediate `retrying … attempt=N` lines are normal transient recovery, not failures — only `max retries exhausted` poisons the domain | `_is_connection_error`, `_handle_connection_error_in_fallback` (`provider.py`) |
| `[remote] upstream error … status=4xx/5xx` → `HTTP <status>` | Upstream returned HTTP ≥ 400. Entry cooldown + domain added (`provider.py:4545`). 429s keep `all_slot_exhaustion` semantics; 402 with reset-time → usage-limit account quarantine instead of/in addition | `_handle_http_error_with_cooldown` (`provider.py:3762`), `_usage_limit_reset_seconds` (`provider.py:3279`) |
| `free_usage_limit` | Free-tier usage-limit error body → fixed cooldown (`_FREE_USAGE_LIMIT_COOLDOWN_SECONDS`) + domain added (`provider.py:4522`) | `_is_free_usage_limit_error` (`provider.py:3210`) |
| `empty_response` | 2xx with zero content (after internal generator retries) → cooldown + sibling-failure streak + domain added (`provider.py:4584`) | `_is_empty_response` (`proxy/proxy/utils.py:186`), `_handle_empty_response_with_cooldown` (`provider.py:3846`) |
| `stream_error` (`finish_reason:error`, zero content) | Pre-content stream failure → `StreamingPreContentError` → entry cooldown + sibling streak + domain added (`provider.py:4405`) | `_preflight_streaming_response` (`provider.py:3488`), `StreamingPreContentError` (`provider.py:3365`) |
| `stream_reroute` (`stall_after_reasoning`) | Reasoning delivered but no final content/tool_calls → `StreamingRecoverableAfterReasoningError` → same request re-routed, domain added (`provider.py:4436`) | `StreamingRecoverableAfterReasoningError` (`provider.py:3381`) |
| `All providers exhausted … unavailable={…}` | Chain end. `unavailable` maps entry → cooldown seconds remaining. `diagnostics=attempts` (when logged) lists per-provider `status` (`connection_error`, `HTTP 503`, `empty_response`, `stream_error`, `usage_limit_reset`, …). Adjacent lines to expect: `Returning first provider error response instead of generic exhausted message … (status=N)` (the chain surfaces the first error, e.g. a 402, rather than the generic 503) and `Chain exhausted …; holding Ns before restarting cycle from the first provider (cycle=C, max_cycles=M)` (chain-hold retry wrapper, LP-0MSH94Z7K007VKC9 — the request is HELD, not failed, and a new cycle restarts) | Exhaustion tail + `_run_chain_cycles` (`provider.py`) |

Related cross-request mechanisms (NOT per-request domain skips):

- **Tier-3 stall circuit breaker** — N stalls in a sliding window → brand-level
  cooldown (`proxy/proxy/stall_circuit_breaker.py`; defaults in
  `provider.py:157-164`).
- **Sibling-fallback circuit breaker** (`_record_sibling_failure`,
  `provider.py:2120`) — consecutive empty/stall failures → extended cooldown
  so the retry cycle skips to a sibling.

## Config reference (`proxy/config.yaml`)

Provider entries that share an `endpoint` share a failure domain, even with
different `api_key_env` values. Current remote topology (verify against
`proxy/config.yaml` — model names rotate; endpoints are the stable key):

| Endpoint (failure domain) | Entries | Distinct `api_key_env`? |
|---|---|---|
| `https://opencode.ai/zen/go` | `opencode-go`, `opencode-go-2`, `opencode-go-3` (+ `-deepseek` variants) | Yes (`OPENCODE_API_KEY`, `OPENCODE_2_API_KEY`, `OPENCODE_3_API_KEY`) — independent usage limits, shared stall fate |
| `https://api.deepseek.com` | `deepseek-v4-flash` (+ variants) | `DEEPSEEK_API_KEY` |

Fallback order within a model block (e.g. `plan:`) is top-down:
`local-qwen3` → `opencode-go-3` → `opencode-go-2` → `opencode-go` →
`deepseek-v4-flash`. Note `deepseek-v4-flash` has an `available_times`
window — outside it the chain can exhaust with only opencode entries tried.

To map an endpoint URL to a provider brand from a bare log line, the
`proxy-usage-analysis` skill's endpoint inference covers
`opencode.ai/zen/go` → `opencode-go`, `opencode.ai/zen` → `opencode`,
`api.deepseek.com` → `deepseek`, `models.inference.ai.azure.com` → `github`.

## Worked example (2026-09-10 13:49)

Observed lines:

```text
2026-09-10 13:49:13,010 - WARNING - Upstream ReadTimeout session=… provider=opencode-go model=muse-spark-1.3-contributor
2026-09-10 13:49:13,011 - WARNING - Upstream stall: max retries exhausted session=… provider=opencode-go model=muse-spark-1.3-contributor retries=3
2026-09-10 13:49:13,012 - INFO - Skipping provider=opencode-go-3: same failure domain as an already-failed entry (domain=https://opencode.ai/zen/go)
2026-09-10 13:49:13,012 - INFO - Skipping provider=opencode-go: same failure domain as an already-failed entry (domain=https://opencode.ai/zen/go)
2026-09-10 13:49:14,053 - WARNING - All providers exhausted for model=v1/chat/completions; unavailable={'opencode-go-2': 58, 'deepseek-v4-flash': 44}
```

Diagnosis: `opencode-go` stalled (ReadTimeout, 3 retries exhausted) →
`connection_error` → domain `https://opencode.ai/zen/go` added to
`attempted_domains` → `opencode-go-2`/`opencode-go-3` skipped **per-request**
(not cooldown) → chain fell through to `deepseek-v4-flash` (different
domain), which was itself in cooldown (44s) → exhausted. Correct behaviour:
retrying sibling API keys against a non-responsive gateway is futile.

## Safe live checks (no log reads)

```bash
curl -s http://localhost:8000/health | python3 -m json.tool   # backend_recovery, backend_signals, self_healing
curl -s http://localhost:8000/metrics | grep -E "proxy_error|fallback|cooldown|remote_stream"
```

Both return in <2s. Do NOT `curl /logs/tail` (hangs). Do NOT `pkill`/`killall`
anything to "restart logging" — the proxy shares the box with other Python
processes (see `start-proxy` skill; use `/skill:start-proxy --restart` if a
restart is genuinely needed).
