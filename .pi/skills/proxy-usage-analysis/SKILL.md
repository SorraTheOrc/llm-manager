---
name: proxy-usage-analysis
description: "Analyze the last 24h of llama-proxy logs (/var/log/llama-proxy/proxy.log*) into per-session daytime/nighttime CSVs and a Markdown report with data-backed configuration recommendations. Trigger on user queries such as: 'analyze proxy usage', 'proxy usage report', 'why is the proxy falling back so much', 'proxy fallback analysis', 'local model utilization', 'generate the daily proxy report'."
---

# Proxy Usage Analysis

Turn the last 24 hours of llama-proxy session and fallback activity into a
digestible per-session CSV record (split into daytime / nighttime buckets per
the slot schedule) and an operator-facing Markdown report with highlighted,
data-backed recommendations.

## When to use

- An operator wants a quick daily read on whether the local models
  (llama-server via the proxy) are used well, how much traffic fell back to
  remote providers, why, and what to change.
- Investigating slot counts (6 daytime / 8 nighttime), context limits, or
  routing thresholds: the report's fallback-reason breakdown and context
  pressure stats show whether configuration changes are warranted.

## Inputs

- Log source (read-only): `/var/log/llama-proxy/proxy.log` plus rotated
  siblings (`proxy.log.YYYY-MM-DD_HH`, 6-hourly rotation, 90-day retention).
- Config (reference only, for slot schedule + thresholds):
  `proxy/config.yaml` in the llm project (auto-discovered by walking up from
  the current directory, or pass `--config`).

## Usage

```bash
python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
    --log-dir /var/log/llama-proxy \
    --hours 24
```

Outputs go to `~/proxy-usage-reports/` by default (override with
`--output-dir`).

Options:

| Flag | Default | Purpose |
|---|---|---|
| `--log-dir` | `/var/log/llama-proxy` | Directory containing `proxy.log*` |
| `--hours` | `24` | Analysis window length |
| `--start` / `--end` | — | Explicit ISO window (`YYYY-MM-DD HH:MM:SS`); overrides `--hours` |
| `--output-dir` | `~/proxy-usage-reports` | Where the CSVs and report are written |
| `--config` | auto-discovered | Path to `proxy/config.yaml` |
| `--json` | off | Print a machine-readable JSON summary instead of the text summary |
| `--quiet` | off | Suppress the stdout summary |

## Outputs

Written to `--output-dir` (default `~/proxy-usage-reports`):

- `daytime_sessions.csv` — one row per **daytime** session (10:00–23:59,
  6 slots per the configured schedule). One row per session, covering ALL
  sessions in the window (local-only and fallback).
- `nighttime_sessions.csv` — one row per **nighttime** session (00:00–09:59,
  8 slots).
- `report.md` — the aggregate report: a single **Session summary** table
  (sessions, requests, local/remote split, classifications, fallback events,
  dispatch denials, context sizes — each with **Total / Day / Night**
  columns), fallback-reason and routing-skip breakdowns, per-model
  breakdown, and highlighted recommendations. Every day/night count carries
  its share of the metric's total (e.g. `285 (74.4%)`), and each
  recommendation's evidence cites the total plus the day/night split.

CSV columns: session id, start/end time, duration, number of messages,
start/avg/max context size, avg/max response size, initial model assignment
(provider + model), time of move to a remote model (empty if never fell
back), fallback reason (empty if never fell back), bucket, slots,
local/remote request counts, dispatch denials.

## How it works

1. **File discovery** — the live `proxy.log` plus every rotated file whose
   name-encoded rotation time is at/after the window start.
2. **Streaming parse** — files are read line by line (never loaded into
   memory; the live log can exceed 700 MB). Only structured prefixes are
   parsed: `Stream started`, `Stream finished`, `Fallback triggered`,
   `routing_skip_local`, `local_dispatch_denied`. Unparseable lines are
   counted and skipped, never fatal.
3. **Session grouping** — a session is identified by its UUID
   (`session=<uuid>`). Per-session context/response sizes use the
   authoritative `tokens=prompt/completion/total` from `Stream finished`
   lines (payloads in logs are truncated and never used for sizes).
4. **Day/night bucketing** — derived from the `slot_schedule` in
   `proxy/config.yaml` (10:00 → 6 slots day, 23:59 → 8 slots night), keyed by
   session start time; nothing is hardcoded.
5. **Recommendations** — rule-based heuristics, each citing the data that
   supports it (see below).

## Interpreting the report

- **Session classification**: local-only vs fell back (local → remote) vs
  remote-only (never used local). A high remote-only share with
  `warm_cache_bypass` reasons usually means routing thresholds are too low or
  caches are cold.
- **Fallback reasons**:
  - `local_concurrency_limit` / `local_lease_active` / `slot_exhaustion` →
    slot pool contention → raise `session_slot_pool_size` or the
    `slot_schedule` slot counts (keep llama-server `--parallel` aligned).
  - `large_context_bypass` → prompts exceed the large-context routing
    thresholds / per-slot context → raise local ctx-size
    (`models.ini`), `local_large_context_*_threshold`, or
    `session_slot_max_prompt_tokens`. Related work item:
    **LP-0MSAOQTJS000FFVM** (evaluate increasing the local ctx-size).
  - `warm_cache_bypass` → cache not warm at routing time → consider raising
    `local_large_context_warm_cache_threshold` or improving slot-cache
    warm-up / session affinity.
  - `HTTP 4xx/5xx`, `empty_response`, timeouts → remote provider issues
    (credentials, rate limits) — not slot-related.
- **Context pressure**: sessions whose max context approaches
  `local_model_ctx_size / slots` can force `large_context_bypass`.
- **Day vs night**: a large fallback-rate gap between buckets suggests the
  slot schedule under-serves one period.

## Testing

```bash
cd .pi/skills/proxy-usage-analysis
python3 -m pytest tests -q
```

The suite covers log-line parsing, session aggregation, fallback attribution,
day/night bucketing, recommendation rules, and an end-to-end run, using
fixtures copied from real `/var/log/llama-proxy/proxy.log` lines.

## Limitations

- `Fallback triggered` lines carry no session UUID; per-session attribution
  prefers the session's own `routing_skip_local` line and otherwise the
  nearest fallback event within 60s of the first remote stream.
- Sessions spanning a slot-schedule transition may observe 503s during the
  drain window; those are expected, not errors.
- A session is included when it has at least one `Stream started` inside the
  window; the day/night bucket is keyed by its first in-window stream.
- Log-format drift is tolerated (missing fields default to empty), but a
  major format change may require updating the regexes in `scripts/log_parser.py`.
