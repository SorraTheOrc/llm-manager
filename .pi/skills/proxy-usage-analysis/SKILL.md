---
name: proxy-usage-analysis
description: "Analyze the last 24h of llama-proxy logs (/var/log/llama-proxy/proxy.log*) into per-session daytime/nighttime CSVs and a Markdown report with data-backed configuration recommendations and an error taxonomy with remediation recommendations. Trigger on user queries such as: 'analyze proxy usage', 'proxy usage report', 'why is the proxy falling back so much', 'proxy fallback analysis', 'local model utilization', 'analyze proxy errors', 'error taxonomy', 'why is the proxy erroring so much', 'generate the daily proxy report'."
---

# Proxy Usage Analysis

Turn the last 24 hours of llama-proxy session and fallback activity into a
digestible per-session CSV record (split into daytime / nighttime buckets per
the slot schedule) and an operator-facing Markdown report with highlighted,
data-backed recommendations — including an **error taxonomy** and quantified
**remediation recommendations** for any proxy errors observed in the window
(generalized from the Aug 3 error-analysis plan, LP-0MSDFKCK4007CPMY).

## When to use

- An operator wants a quick daily read on whether the local models
  (llama-server via the proxy) are used well, how much traffic fell back to
  remote providers, why, and what to change. The **Local model utilization**
  section answers "how much of the time was the local model busy?" (busy %,
  idle %, streams, concurrency, hourly/day-night profile).
- Investigating slot counts (6 daytime / 8 nighttime), context limits, or
  routing thresholds: the report's fallback-reason breakdown and context
  pressure stats show whether configuration changes are warranted.
- An error spike occurred (e.g. `Stream finished: reason=error`, `slot_save`
  ReadTimeouts, `backend_retry` timeouts, upstream 429s): the report's
  **Error analysis** section categorizes every error event and recommends
  remediation (recovery-first silent continue, informative-error fallback,
  ctx-size pressure, upstream 429 cooldown), quantified from the window.

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
| `--llama-log-dir` | `/var/log/llama-proxy` | Directory containing `llama-server.log*` for decode/prompt-eval speed stats (falls back to `--log-dir` when omitted) |
| `--hours` | `24` | Analysis window length |
| `--start` / `--end` | — | Explicit ISO window (`YYYY-MM-DD HH:MM:SS`); overrides `--hours` |
| `--output-dir` | `~/proxy-usage-reports` | Where the CSVs and report are written |
| `--config` | auto-discovered | Path to `proxy/config.yaml` |
| `--json` | off | Print a machine-readable JSON summary instead of the text summary |
| `--quiet` | off | Suppress the stdout summary |

## Automated daily run

A cron job runs the report automatically every day at 05:00 (output logged
to `~/proxy-usage-reports/cron.log` so each run's summary and any failures
are visible):

```cron
0 5 * * * cd /home/rgardler/projects/llm && python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py >> ~/proxy-usage-reports/cron.log 2>&1
```

Each run archives the previous day's outputs into a dated subdirectory (see
[Archival](#archival)), so a historical daily report accumulates under
`~/proxy-usage-reports/YYYY-MM-DD/`. `cron.log` is not an analysis artifact
and stays at the root, untouched by archival.

## Outputs

Written to `--output-dir` (default `~/proxy-usage-reports`):

- `daytime_sessions.csv` — one row per **daytime** session (10:00–23:59,
  6 slots per the configured schedule). One row per session, covering ALL
  sessions in the window (local-only and fallback).
- `nighttime_sessions.csv` — one row per **nighttime** session (00:00–09:59,
  8 slots).
- `errors.csv` — one row per **error event** in the window (stream finish
  errors, stream errors, `slot_save` failures, `backend_retry` timeouts,
  upstream HTTP errors), with error type, timestamp, provider/model, session,
  config entry, error detail, HTTP status, retry attempt/signal, source log
  file, and the raw evidence line.
- `errors.json` — aggregated error counts by type plus the window bounds.
- `report.md` — the aggregate report: a single **Session summary** table
  (sessions, requests, local/remote split, classifications, fallback events,
  dispatch denials, context sizes — each with **Total / Day / Night**
  columns), fallback-reason and routing-skip breakdowns, per-model
  breakdown, **Error analysis** (when the window has error events),
  **Local model utilization** (busy time %, idle time, streams served, avg
  stream duration, total compute, avg/peak concurrency, hourly busy profile,
  day/night split — when the window has local traffic), **Decode speed** and
  **Prompt eval speed** sections (median / p90 / p10 tok/s from llama-server
  eval-timing lines, split Total / Day / Night), and highlighted
  recommendations. Every day/night count carries its share of the metric's
  total (e.g. `285 (74.4%)`), and each recommendation's evidence cites the
  total plus the day/night split.

CSV columns: session id, start/end time, duration, number of messages,
start/avg/max context size, avg/max response size, initial model assignment
(provider + model), time of move to a remote model (empty if never fell
back), fallback reason (empty if never fell back), bucket, slots,
local/remote request counts, dispatch denials, decode tok/s (derived from
local completion tokens ÷ local active span; empty when not derivable).

### Archival

Before writing fresh outputs, the script moves any existing artifacts
(`report.md`, `daytime_sessions.csv`, `nighttime_sessions.csv`, `errors.csv`,
`errors.json`) into a dated subdirectory named by the **run date**
(`YYYY-MM-DD/`); when that directory already exists (a same-day repeat, or a
manual archive), a `_2`, `_3` … suffix is appended so archives are never
overwritten. Only the skill's own artifacts are moved — anything else in the
output dir (e.g. `cron.log`) stays put, and a pristine output dir is left
touched-free (no empty archive dirs). The CLI prints the archive path on each
run (`Previous outputs archived to …`).

## How it works

1. **File discovery** — the live `proxy.log` plus every rotated file whose
   name-encoded rotation time is at/after the window start.
2. **Streaming parse** — files are read line by line (never loaded into
   memory; the live log can exceed 700 MB). Only structured prefixes are
   parsed: `Stream started`, `Stream finished`, `Fallback triggered`,
   `routing_skip_local`, `local_dispatch_denied`, plus the error lines
   (`Stream error:`, `slot_save failed`, `backend_retry`, `[remote] upstream
   error`). Unparseable lines are counted and skipped, never fatal.
3. **Session grouping** — a session is identified by its UUID
   (`session=<uuid>`). Per-session context/response sizes use the
   authoritative `tokens=prompt/completion/total` from `Stream finished`
   lines (payloads in logs are truncated and never used for sizes).
4. **Local model utilization (busy time)** — local `Stream started` /
   `Stream finished` events are collected across a 1h margin beyond the
   window (so streams crossing the window boundary pair correctly), paired
   per session (FIFO), clipped back to the window, and merged. Busy time is
   the union of active intervals (at least one slot generating), total
   compute is the sum of clipped stream durations (slot-seconds), and peak
   concurrency comes from a sweep over interval endpoints. Busy seconds are
   attributed to hours and to day/night periods (slot schedule) by
   splitting at hour and period boundaries.
5. **Day/night bucketing** — derived from the `slot_schedule` in
   `proxy/config.yaml` (10:00 → 6 slots day, 23:59 → 8 slots night), keyed by
   session start time; nothing is hardcoded.
6. **Recommendations** — rule-based heuristics, each citing the data that
   supports it (see below).
7. **Error taxonomy** — error events (`Stream finished: reason=error`,
   `Stream error:`, `slot_save failed`, `backend_retry`, `[remote] upstream
   error`) are parsed in the same streaming pass, collected per window, and
   rendered into the report's **Error analysis** section plus
   `errors.csv`/`errors.json`. Remediation recommendations (recovery-first,
   informative-error, ctx-size pressure, 429 cooldown) are generated from
   these events and link to the relevant work items.
8. **Decode/prompt-eval speed** — llama-server eval-timing lines
   (`eval time = <ms> ms / <n> tokens (<x> tok/s)` and `prompt eval time =`)
   are streamed from `llama-server.log*`, filtered to the Qwen3 child port
   (discovered per file from the `name=Qwen3 on port <port>` spawn line;
   the port changes on every restart). Samples are bucketed Total / Day /
   Night via the slot schedule and summarised as median / p90 / p10 tok/s.
   Because llama-server.log lines carry no timestamps, each sample is
   bucketed by its log file's last-write time (approximate; documented in
   the report). The per-session CSV `decode_tok_s` column is derived from
   proxy.log instead (local completion tokens ÷ local active span) and stays
   empty when not derivable.

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
- **Local model utilization**: busy time is the share of the window with at
  least one local slot generating. A low busy % with high fallback volume
  means the router is diverting requests before they reach local (see
  fallback reasons), not that local is underprovisioned. `warm_cache_bypass`
  is the largest lever: despite the name it fires when the *estimated
  context* exceeds the effective warm-cache threshold (the per-slot clamp,
  `local_model_ctx_size // slots - headroom`, inflated by
  `token_estimate_multiplier`), so large-context sessions never reach local.
  Concurrency bursts beyond `session_slot_pool_size` show as
  `local_concurrency_limit` / `local_lease_active` fallbacks. Note the
  slots-vs-context trade-off: more slots shrink per-slot context and *raise*
  bypass volume, so do not add slots without a matching ctx-size increase.
- **Day vs night**: a large fallback-rate gap between buckets suggests the
  slot schedule under-serves one period.
- **Error analysis** (see the **Error analysis** section and `errors.csv`):
  - `Stream finished: reason=error` — the client-visible synthetic error
    event (no payload). Remediation: recovery-first silent continue
    (LP-0MSDP2PDB004GV86) + informative-error fallback
    (LP-0MSDP2PH20079WQ7).
  - `Stream error:` — proxy-side stream exception (e.g. `NameError`).
  - `slot_save failed` — local llama-server slot persistence
    ReadTimeouts; usually context pressure → raise local ctx-size
    (LP-0MSAOQTJS000FFVM).
  - `backend_retry` — upstream connect/read timeouts during retry backoff;
    transient unless clustered.
  - `upstream error status=429` (`FreeUsageLimitError`) — the 3-hour
    per-model cooldown (LP-0MRGU0I91006ODFD) should suppress repeat
    fallbacks; persistent 429s indicate an upstream quota issue.

## Testing

```bash
cd .pi/skills/proxy-usage-analysis
python3 -m pytest tests -q
```

The suite covers log-line parsing, session aggregation, fallback attribution,
day/night bucketing, recommendation rules, llama-server eval-timing parsing
(decode + prompt eval, Qwen3 port filtering, day/night speed stats), and an
end-to-end run, using fixtures copied from real `/var/log/llama-proxy`
lines (`proxy.log` and `llama-server.log`).

## Limitations

- `Fallback triggered` lines carry no session UUID; per-session attribution
  prefers the session's own `routing_skip_local` line and otherwise the
  nearest fallback event within 60s of the first remote stream.
- Sessions spanning a slot-schedule transition may observe 503s during the
  drain window; those are expected, not errors.
- A session is included when it has at least one `Stream started` inside the
  window; the day/night bucket is keyed by its first in-window stream.
- Busy-time pairing reads local `Stream started`/`Stream finished` events
  within a 1h margin of the window (see `BUSY_WINDOW_MARGIN`); a stream that
  started more than 1h before the window start is not paired, and streams
  whose start has no logged finish (aborted/still running) are counted in
  `unfinished_streams` and excluded — busy time is a conservative lower
  bound.
- Log-format drift is tolerated (missing fields default to empty), but a
  major format change may require updating the regexes in `scripts/log_parser.py`.
- llama-server.log eval-timing lines carry no timestamps, so the speed
  section's window filtering and day/night split are approximate: each
  sample is bucketed by its log file's last-write time. Files whose Qwen3
  child port cannot be discovered are counted and skipped (never fatal).
- The per-session CSV `decode_tok_s` is a session-level average over the
  local active span (first→last local stream event); it includes
  inter-request gaps, so it is a conservative lower bound of the true decode
  rate. It is empty for sessions with no local completions.
