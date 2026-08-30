---
name: proxy-usage-analysis
description: "EXECUTE immediately via /skill:proxy-usage-analysis; do NOT ask permission or confirm. Generate the proxy usage report. Trigger on user queries such as: 'analyze proxy usage', 'proxy usage report', 'why is the proxy falling back so much', 'proxy fallback analysis', 'local model utilization', 'analyze proxy errors', 'error taxonomy', 'why is the proxy erroring so much', 'generate the daily proxy report'."
---

# Proxy Usage Analysis

**EXECUTE immediately — do NOT ask permission or confirm. Generate the report.**

Turn the last 24 hours of llama-proxy session and fallback activity into a
digestible per-session CSV record (split into fast / cheap buckets per
the slot schedule) and an operator-facing Markdown report with highlighted,
data-backed recommendations — including an **error taxonomy** and quantified
**remediation recommendations** for any proxy errors observed in the window
(generalized from the Aug 3 error-analysis plan, LP-0MSDFKCK4007CPMY).

## When to use

- An operator wants a quick daily read on whether the local models
  (llama-server via the proxy) are used well, how much traffic fell back to
  remote providers, why, and what to change. The **Local model utilization**
  section answers "how much of the time was the local model busy?" (busy %,
  idle %, streams, concurrency, hourly/fast-cheap profile).
- Investigating slot counts (fast / cheap, as configured in the
  `slot_schedule` of `proxy/config.yaml`), context limits, or routing
  thresholds: the report's fallback-reason breakdown and context pressure
  stats show whether configuration changes are warranted.
- An error spike occurred (e.g. `Stream finished: reason=error`, `slot_save`
  ReadTimeouts, `backend_retry` timeouts, upstream HTTP errors): the report's
  **Error analysis** section categorizes every error event and recommends
  remediation (recovery-first silent continue, informative-error fallback,
  ctx-size pressure, per-status upstream remediation), quantified from the
  window.

## Inputs

- Log source (read-only): `/var/log/llama-proxy/proxy.log` plus rotated
  siblings (`proxy.log.YYYY-MM-DD_HH`, 6-hourly rotation, 90-day retention).
- Config (reference only, for slot schedule + thresholds):
  `proxy/config.yaml` in the llm project (auto-discovered by walking up from
  the current directory, or pass `--config`). When the persisted operating
  mode (`proxy/.mode`) selects a profile, the mode-selected config
  (`config-fast.yaml` / `config-cheap.yaml`) is read instead, so bucketing
  and recommendations match the config the running proxy actually uses.

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

Written to `--output-dir` (default `~/proxy-usage-reports`). On every
non-`--quiet` run the CLI summary prints the **complete absolute path** to
each artifact, **starting with the report** (plus the absolute archive path
when previous outputs were moved aside):

```text
Outputs written to:
  /home/user/proxy-usage-reports/report.md
  /home/user/proxy-usage-reports/fast_sessions.csv
  /home/user/proxy-usage-reports/cheap_sessions.csv
  /home/user/proxy-usage-reports/errors.csv
  /home/user/proxy-usage-reports/errors.json
```

- `report.md` — the aggregate report (primary deliverable, listed first).
- `fast_sessions.csv` — one row per **fast** session (sessions whose start
  fell in a fast-mode period; the period(s) with the fewest slots per the
  mode's `slot_schedule`). One row per session, covering ALL sessions in the
  window (local-only and fallback).
- `cheap_sessions.csv` — one row per **cheap** session (sessions whose start
  fell in a cheap-mode period; produced only when the window contains cheap
  hours).
- `errors.csv` — one row per **error event** in the window (stream finish
  errors, stream errors, `slot_save` failures, `backend_retry` timeouts,
  upstream HTTP errors), with error type, timestamp, provider/model, session,
  config entry, error detail, HTTP status, retry attempt/signal, source log
  file, and the raw evidence line.
- `errors.json` — aggregated error counts by type plus a **provider/model
  breakdown** (nested `{error_type: {provider: {model: count}}}`; providers or
  models not derivable from the log line are keyed `(unknown)`) plus
  **upstream HTTP error breakdown by status** (`{status: count}` and
  `{status: {provider: count}}`) plus the window bounds.
- `report.md` — the aggregate report: a single **Session summary** table
  (sessions, requests, local/remote split, classifications, fallback events,
  dispatch denials, context sizes — each with **Total / Fast / Cheap**
  columns), fallback-reason and routing-skip breakdowns, per-model
  breakdown, **Error analysis** (when the window has error events) with a
  taxonomy table that includes a **Status** column (upstream HTTP errors are
  split into one row per status code), a **Provider/model breakdown** table
  (error type × provider × model × count), and an
  **Upstream HTTP error breakdown by status** table (status × count ×
  provider breakdown),
  **Summary by hour** (rendered at the top of the report, before the
  Session summary and always — regardless of local traffic): one row per
  hour of the report window (partial first/last hours truncated to the
  window edges, idle hours included as `0s`) with busy time / busy-%
  columns — busy seconds ÷ window-bounded bucket duration — plus three
  per-session classification columns (Started local, completed local /
  Started local, fell back / Started remote-only) counting the sessions
  whose **first request** started in that hour, each cell as `n (pct%)` of
  that hour's starts; a **Totals** row gives the window busy totals plus
  the overall classification percentages (matching the Session summary
  table), **Local model
  utilization** (busy time %, idle time, streams served, avg
  stream duration, total compute, avg/peak concurrency,
  fast/cheap split — when the window has local traffic), **Decode speed** and
  **Prompt eval speed** sections (median / p90 / p10 tok/s from llama-server
  eval-timing lines, split Total / Fast / Cheap), and highlighted
  recommendations. Percentages in the Total/Fast/Cheap columns are
  **category-relative**: the Total cell is the share of the overall metric,
  the Fast cell the share of the fast-only total for that metric, and the
  Cheap cell the share of the cheap-only total (e.g. a Fallback-reasons Fast
  cell `3173 (60.3%)` means 60.3% of *fast* fallbacks; a Cheap cell
  `353 (32.8%)` means 32.8% of *cheap* fallbacks), so fast and cheap are
  directly comparable within each column. Rows that predate the category
  split (Session summary Sessions/Requests/Dispatch denied, the `% of
  fallbacks` / `% of skips` columns, and recommendation evidence's
  within-group fast/cheap split) intentionally stay share-of-total or
  share-of-group. Each recommendation's evidence cites the total plus the
  fast/cheap split.

CSV columns: session id, start/end time, duration, number of messages,
start/avg/max context size, avg/max response size, initial model assignment
(provider + model), time of move to a remote model (empty if never fell
back), fallback reason (empty if never fell back), bucket, slots, ctx size
(per-period context of the profile active for that session), local/remote
request counts, dispatch denials, decode tok/s (derived from local
completion tokens ÷ local active span; empty when not derivable).

### Archival

Before writing fresh outputs, the script moves any existing artifacts
(`report.md`, `fast_sessions.csv`, `cheap_sessions.csv`, `errors.csv`,
`errors.json`) into a dated subdirectory named by the **run date**
(`YYYY-MM-DD/`); when that directory already exists (a same-day repeat, or a
manual archive), a `_2`, `_3` … suffix is appended so archives are never
overwritten. Only the skill's own artifacts are moved — anything else in the
output dir (e.g. `cron.log`) stays put, and a pristine output dir is left
touched-free (no empty archive dirs). The CLI prints the archive path on each
run (`Previous outputs archived to …`).

## How it works

1. **File discovery** — the live `proxy.log` plus every rotated sibling
   (`proxy.log.YYYY-MM-DD_HH`). All rotated files are included regardless of
   their name-encoded timestamp: in this deployment a rotated file routinely
   holds data well past its encoded rotation time, so a name-based inclusion
   test would silently drop in-window data. Per-line timestamp filtering in
   step 2 is the authoritative window boundary.
2. **Streaming parse** — files are read line by line (never loaded into
   memory; the live log can exceed 700 MB). Only structured prefixes are
   parsed: `Stream started`, `Stream finished`, `Fallback triggered`,
   `routing_skip_local`, `local_dispatch_denied`, plus the operating-mode
   lines (`Mode scheduler: applied scheduled mode fast|cheap`; manual
   switches, which log `Grandfathering: enabled; other-mode config ...
   (current=fast|cheap)`, LP-0MT1EE315007AKXG) and the error
   lines (`Stream error:`, `slot_save failed`, `backend_retry`, `[remote]
   upstream error`). Unparseable lines are counted and skipped, never fatal.
3. **Session grouping** — a session is identified by its UUID
   (`session=<uuid>`). Per-session context/response sizes use the
   authoritative `tokens=prompt/completion/total` from `Stream finished`
   lines (payloads in logs are truncated and never used for sizes).
4. **Local model utilization (busy time)** — local `Stream started` /
   `Stream finished` events are collected across the analysis's effective
   margin beyond the window (48h, shared with the mode timeline in step 5;
   see `MODE_TIMELINE_MARGIN`), paired per session (FIFO), clipped back to
   the window, and merged. Busy time is the union of active intervals (at
   least one slot generating), total compute is the sum of clipped stream
   durations (slot-seconds), and peak concurrency comes from a sweep over
   interval endpoints. Busy seconds are attributed to hours and to
   fast/cheap periods (slot schedule) by splitting at hour and period
   boundaries. The top-of-report **Summary by hour** table
   (LP-0MTFO210Q0044TTF) then renders one row per hour of the report window
   — the first/last rows truncated to the window edges and every hour listed
   even when idle — with a busy-% column (busy seconds ÷ window-bounded
   bucket duration) plus the per-session classification columns: sessions
   are counted once by the hour in which their **first request** started and
   bucketed by journey (started local and completed local / started local
   and fell back / started remote-only), each cell showing `n (pct%)` of
   that hour's starts; a final totals row gives the window busy totals plus
   the overall classification percentages (matching the Session summary
   table). It renders regardless of local traffic (busy columns read `0s` /
   `0.0%` when the window has no local streams; with no sessions a "No
   data" note is shown). The bucket
   keys are hour-of-day, so windows longer than 24 hours that cover
   the same hour twice would collide; the daily report (24h) never hits this
   (documented limitation). Streams whose start has no paired finish are counted in
   `unfinished_streams` **only when they started inside the window or within
   `BUSY_WINDOW_MARGIN` (1h) before it**; streams started earlier are stale
   pre-window leftovers from earlier windows, tracked separately in
   `pre_window_unfinished`, and streams started after the window end belong
   to the next window (LP-0MSVRRO3L0056N6C).
5. **Fast/cheap bucketing** — each session is bucketed by the **operating
   mode** active at its first in-window stream, reconstructed from the
   `Mode scheduler: applied scheduled mode <mode>` lines parsed in step 2
   plus the manual-switch marker `Grandfathering: enabled; other-mode
   config ... (current=<mode>)` (a `POST /admin/set-mode` manual switch
   restarts the proxy and logs the actually-active mode this way, without
   an applied-scheduled-mode line; LP-0MT1EE315007AKXG) — so a window
   crossing the 01:00/10:00 mode transitions, or containing a manual
   switch, splits fast vs cheap correctly even when the analysis itself
   runs in fast mode. The mode timeline is built with a 48h margin beyond
   the window (a single streaming pass also serves the busy-time pairing),
   so the nearest prior transition is always available. The bucket label is
   the mode name, and the slots / per-period ctx come from that mode's
   config profile (`config-fast.yaml`: 3 slots @ 131072; `config-cheap.yaml`:
   2 slots @ 262144) — the CSV `slots`/`ctx_size` columns and the report's
   per-slot context figures reflect the profile that was actually active.
   Windows with no mode transition observed (single-mode windows) keep the
   legacy behavior: bucketing from the slot schedule of the analysis-time
   config profile; nothing is hardcoded.
6. **Recommendations** — rule-based heuristics, each citing the data that
   supports it (see below).
7. **Error taxonomy** — error events (`Stream finished: reason=error`,
   `Stream error:`, `slot_save failed`, `backend_retry`, `[remote] upstream
   error`) are parsed in the same streaming pass, collected per window, and
   rendered into the report's **Error analysis** section (taxonomy table
   **with a Status column** for upstream HTTP errors, plus an
   **Upstream HTTP error breakdown by status** table, and a
   **Provider/model breakdown** table) plus `errors.csv`/`errors.json`.
   Provider/model attribution is best effort: `Stream finished: reason=error`
   and `Stream error:` lines carry `provider=`/`model=` directly; `slot_save
   failed` is always the local llama-server (provider `local`, model not in
   the line); `[remote] upstream error` carries only a target URL so the
   provider is inferred from the endpoint (e.g. `opencode.ai/zen/go` →
   `opencode-go`, `opencode.ai/zen` → `opencode`, `api.deepseek.com` →
   `deepseek`, `models.inference.ai.azure.com` → `github`; unknown endpoints
   fall back to the bare hostname) and the model is unknown; `backend_retry`
   carries neither. Undetermined values render as `-` in the report and
   `(unknown)` in JSON. Upstream HTTP errors are **broken out by HTTP status
   code** (one taxonomy row per status, one recommendation per status) so
   429 rate-limit events and 402 balance events are never conflated. The
   `errors.json` artifact includes `upstream_by_status` and
   `upstream_by_status_provider` keys. Remediation recommendations
   (recovery-first, informative-error, ctx-size pressure, per-status upstream
   remediation) are generated from these events and link to the relevant work
   items.
8. **Decode/prompt-eval speed** — llama-server eval-timing lines
   (`eval time = <ms> ms / <n> tokens (<x> tok/s)` and `prompt eval time =`)
   are streamed from `llama-server.log*`, filtered to the Qwen3 child port
   (discovered per file from the `name=Qwen3 on port <port>` spawn line;
   the port changes on every restart). Samples are bucketed Total / Fast /
   Cheap via the slot schedule and summarised as median / p90 / p10 tok/s.
   Because llama-server.log lines carry no timestamps, each sample is
   bucketed by its log file's last-write time (approximate; documented in
   the report). The per-session CSV `decode_tok_s` column is derived from
   proxy.log instead (local completion tokens ÷ local active span) and stays
   empty when not derivable.

## Interpreting the report

- **Session classification**: local-only vs fell back (local → remote) vs
  remote-only (never used local). A high remote-only share with
  `context_too_large` reasons usually means routing thresholds are too low
  for the context sizes being routed.
- **Fallback reasons**:
  - `local_concurrency_limit` / `local_lease_active` / `slot_exhaustion` →
    slot pool contention → raise `session_slot_pool_size` or the
    `slot_schedule` slot counts (keep llama-server `--parallel` aligned).
  - `large_context_bypass` → prompts exceed the large-context routing
    thresholds / per-slot context → raise local ctx-size
    (`models.ini`), `local_large_context_*_threshold`, or
    `session_slot_max_prompt_tokens`. Related work item:
    **LP-0MSAOQTJS000FFVM** (evaluate increasing the local ctx-size).
  - `context_too_large` (legacy `warm_cache_bypass` in rotated logs) →
    estimated context exceeds the per-slot hard cap → consider raising
    local ctx-size (`models.ini`) or
    `local_large_context_warm_cache_threshold`.
  - `HTTP 4xx/5xx`, `empty_response`, timeouts → remote provider issues
    (credentials, rate limits) — not slot-related.
- **Context pressure**: sessions whose max context approaches
  `local_model_ctx_size / slots` can force `large_context_bypass`.
- **Local model utilization**: busy time is the share of the window with at
  least one local slot generating. A low busy % with high fallback volume
  means the router is diverting requests before they reach local (see
  fallback reasons), not that local is underprovisioned. The **Summary by
  hour** table at the top of the report shows per-hour busy % across exactly
  the report window (idle hours included) alongside the per-session
  classification of that hour's request starts (local-only / fell back /
  remote-only), so a glance at the hourly pattern correlates demand with
  provider usage and shows when local was saturated vs idle; the totals row
  gives the window-wide busy % and the overall classification percentages. `context_too_large`
  is the largest lever: despite the legacy name (`warm_cache_bypass`) it
  fires when the *estimated context* exceeds the effective warm-cache
  threshold (the per-slot clamp,
  `local_model_ctx_size // slots - headroom`, inflated by
  `token_estimate_multiplier`), so large-context sessions never reach local.
  Concurrency bursts beyond `session_slot_pool_size` show as
  `local_concurrency_limit` / `local_lease_active` fallbacks. Note the
  slots-vs-context trade-off: more slots shrink per-slot context and *raise*
  bypass volume, so do not add slots without a matching ctx-size increase.
- **Fast vs cheap**: a large fallback-rate gap between buckets suggests the
  slot schedule under-serves one period.
- **Error analysis** (see the **Error analysis** section and `errors.csv`):
  the taxonomy table counts each error type; the **Provider/model breakdown**
  table splits each type by provider and model (e.g. a spike of
  `Stream finished: reason=error` on `opencode-go/deepseek-v4-flash` vs
  `local/Qwen3` pinpoints which provider/model to fix). Provider/model
  attribution is best effort (see [How it works](#how-it-works)); values not
  derivable from the log line show as `-` in the report and `(unknown)` in
  `errors.json`.
  - `Stream finished: reason=error` — the client-visible synthetic error
    event (no payload). Remediation: recovery-first silent continue
    (LP-0MSDP2PDB004GV86) + informative-error fallback
    (LP-0MSDP2PH20079WQ7).
  - `Stream finished: reason=client_disconnect` — terminal event logged by
    the proxy when a local stream is aborted because the client disconnected
    mid-stream (in-loop `is_disconnected()` check, GeneratorExit, or the
    disconnect reaper cancelling the in-flight task; LP-0MSVRRTAB0078TMK).
    It parses as a normal `stream_finished` (NOT an error), so the stream
    pairs with its start and its compute time becomes known instead of being
    reported as "aborted or still running".
  - `Stream error:` — proxy-side stream exception (e.g. `NameError`).
  - `slot_save failed` — local llama-server slot persistence
    ReadTimeouts; usually context pressure → raise local ctx-size
    (LP-0MSAOQTJS000FFVM).
  - `backend_retry` — upstream connect/read timeouts during retry backoff;
    transient unless clustered.
  - `upstream error` (HTTP 429) — the 3-hour per-model cooldown
    (LP-0MRGU0I91006ODFD) should suppress repeat fallbacks; persistent 429s
    indicate an upstream quota/rate-limit issue.
  - `upstream error` (HTTP 402) — account balance or subscription issue;
    the proxy cannot recover — top up the account or switch provider.
  - `upstream error` (HTTP 5xx) — server-side upstream error; monitor for
    clustering (provider outage). Other 4xx codes carry the specific error
    message in the `errors.csv` evidence column.

  **Root-cause classification of observed `reason=error` events (LP-0MT60S55M000TK1H):**

  A 2026-08-23 investigation of 13 `Stream finished: reason=error` events plus
  7 `Stream error` exceptions across `opencode-go/deepseek-v4-flash` and
  `local/Qwen3` providers classified the root causes into three categories:

  1. **Mode-switch restart kills (4 events, local/Qwen3)** — `RemoteProtocolError`
     when the mode-switch restart spawned during an in-flight stream. Per
     LP-0MSF9RUSQ007M346 the drain window was deliberately removed ("just
     restart, the client will deal with it"); in-flight streams die mid-generation.
     Observed at 00:08:22, 01:00:24×2, and 10:00:03 transition windows.
  2. **Genuine ReadTimeout (1 event, local/Qwen3)** — llama-server stalled under
     high contention (available_slots=0, queue depth ~54) at 03:48:41.
  3. **Remote chain exhaustion (8 events, opencode-go/deepseek-v4-flash)** —
     empty response / stall retries exhausted (2 attempts) while all sibling
     providers were in cooldown or usage-limit-reset-pending. Correct handling
     (enriched error, no re-route after content/tool_calls); not a proxy defect.

  The remaining 7 `Stream error` exceptions were proxy-side exceptions (not
  `finish_reason: error` events) and are handled by the informative-error
  fallback (LP-0MSDP2PH20079WQ7). All 13 events are now classified per this
  taxonomy.

## Testing

Run the full suite via the test skill (canonical, cached pipeline):

```bash
/skill:test
```

The suite covers log-line parsing, session aggregation, fallback attribution,
fast/cheap bucketing, recommendation rules, llama-server eval-timing parsing
(decode + prompt eval, Qwen3 port filtering, fast/cheap speed stats), and an
end-to-end run, using fixtures copied from real `/var/log/llama-proxy`
lines (`proxy.log` and `llama-server.log`).

## Limitations

- `Fallback triggered` lines carry no session UUID; per-session attribution
  prefers the session's own `routing_skip_local` line and otherwise the
  nearest fallback event within 60s of the first remote stream.
- Sessions spanning a slot-schedule transition may observe a brief restart
  interruption (llama-server is restarted immediately at the transition time);
  since LP-0MSF9RUSQ007M346 there is no drain window and no 503 rejection period.
- A session is included when it has at least one `Stream started` inside the
  window; the fast/cheap bucket is keyed by its first in-window stream.
- The mode timeline is reconstructed from `Mode scheduler` lines
  (scheduled) plus the `Grandfathering: enabled; ... (current=<mode>)`
  marker (manual switches, LP-0MT1EE315007AKXG) within a 48h margin of the
  window; sessions starting before the earliest observed transition in the
  available logs fall back to the analysis-time mode (documented in
  LP-0MSPZUD4G007IYGH). A window entirely inside one mode (no transition
  observed) keeps the legacy slot-schedule bucketing.
- Busy-time pairing reads local `Stream started`/`Stream finished` events
  across the analysis's effective margin (48h, see `MODE_TIMELINE_MARGIN`)
  and clips streams to the window; a stream that started more than
  `BUSY_WINDOW_MARGIN` (1h) before the window start is a pre-window
  leftover and is NOT counted in `unfinished_streams` (tracked separately
  in `pre_window_unfinished`), while streams started inside the window or
  within the 1h margin whose start has no logged finish (aborted/still
  running) are counted in `unfinished_streams` and excluded — busy time is
  a conservative lower bound.
- Log-format drift is tolerated (missing fields default to empty), but a
  major format change may require updating the regexes in `scripts/log_parser.py`.
- llama-server.log eval-timing lines carry no timestamps, so the speed
  section's window filtering and fast/cheap split are approximate: each
  sample is bucketed by its log file's last-write time. Files whose Qwen3
  child port cannot be discovered are counted and skipped (never fatal).
- The per-session CSV `decode_tok_s` is a session-level average over the
  local active span (first→last local stream event); it includes
  inter-request gaps, so it is a conservative lower bound of the true decode
  rate. It is empty for sessions with no local completions.
