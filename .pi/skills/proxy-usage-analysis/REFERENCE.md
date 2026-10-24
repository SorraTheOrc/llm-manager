# Proxy Usage Analysis — reference

Substantive reference material for the `proxy-usage-analysis` skill: log
inputs, report outputs, the report interpretation guide, and known
limitations. The skill itself is a thin, CLI-focused wrapper — see
[SKILL.md](SKILL.md) for invocation, options, and the daily cron.

## Inputs

- Log source (read-only): `/var/log/llama-proxy/proxy.log` plus rotated
  siblings. Two rotation mechanisms produce two naming patterns —
  in-process rotation (`proxy.log.YYYY-MM-DD_HH`, dot) and logrotate safety
  net (`proxy.log-YYYY-MM-DD_HH`, dash) — and either may be gzipped (`.gz`).
  The script discovers all of them and decompresses `.gz` transparently; the
  log directory should contain both types.
- llama-server log source (read-only): `/var/log/llama-proxy/llama-server.log`
  plus its rotations — numbered in-process rotations
  (`llama-server.N.log`) and logrotate daily archives
  (`llama-server.log-YYYY-MM-DD[.gz]`) — used for the Decode speed /
  Prompt eval speed sections (see [How it works](#how-it-works) step 9).
- Config (reference only, for slot schedule + thresholds):
  `proxy/config.yaml` in the llm project (auto-discovered by walking up from
  the current directory, or pass `--config`). The persisted operating-mode
  axis was removed (LP-0MVDT94F0003VHUA AC4), so the single unified
  `proxy/config.yaml` is read, and its `server.priority_window`
  (`enabled` / `start` / `end`, local time) is the fast/cheap schedule when a
  window's logs carry no mode-schedule evidence, so the standard/low session
  bucketing and recommendations match the config the running proxy actually
  uses (LP-0MVKSXXIR006OBI2).

## Outputs

Written to `--output-dir` (default `~/proxy-usage-reports`). On every
non-`--quiet` run the CLI summary prints the **complete absolute path** to
each artifact, **starting with the report** (plus the absolute archive path
when previous outputs were moved aside):

```text
Outputs written to:
  /home/user/proxy-usage-reports/report.md
  /home/user/proxy-usage-reports/standard_sessions.csv
  /home/user/proxy-usage-reports/low_priority_sessions.csv
  /home/user/proxy-usage-reports/errors.csv
  /home/user/proxy-usage-reports/errors.json
```

- `report.md` — the aggregate report (primary deliverable, listed first).
- `standard_sessions.csv` — one row per **standard** session (sessions
  whose start fell outside the cheap priority window). One row per session,
  covering ALL sessions in the
  window (local-only and fallback).
- `low_priority_sessions.csv` — one row per **low** session (sessions whose
  start fell inside the cheap priority window; produced only when the window
  contains low-priority periods,
  hours).
- `errors.csv` — one row per **error event** in the window (stream finish
  errors, stream errors, `slot_save` failures, `backend_retry` timeouts,
  upstream HTTP errors), with error type, root cause, timestamp,
  provider/model, session, config entry, error detail, HTTP status, retry
  attempt/signal, source log file, and the raw evidence line.
- `errors.json` — aggregated error counts by type plus a **provider/model
  breakdown** (nested `{error_type: {provider: {model: count}}}`; providers or
  models not derivable from the log line are keyed `(unknown)`) plus a
  **root-cause breakdown** for stream errors (`by_root_cause`, nested
  `{error_type: {root_cause: count}}`; see the root-cause taxonomy below)
  plus **upstream HTTP error breakdown by status** (`{status: count}` and
  `{status: {provider: count}}`) plus the window bounds.
- `--json` summary field `compaction` — machine-readable compaction summary
  (same as in the report): `"compaction": {"events": N, "compact_events":
  N, "backstop_events": N, "churn_events": N, "sessions_with_compaction":
  N, "by_action": {action: count}, "by_reason": {...}, "bucket_counts":
  {mode: count}, "bucket_dry_live": {mode: {label: count}}, "dry_run":
  {dry_run/live}, "avg_pre_tokens": i, "avg_post_tokens": i,
  "avg_tokens_saved": n, "stayed_local": N, "fell_back": N,
  "would_have_avoided": N, "would_still_fallback": N, "avoided": N,
  "backstop_sessions": N} — so `python3 scripts/analyze_proxy_usage.py … --json | jq .compaction` is usable without parsing Markdown.
- `--json` summary field `hold_cycle_routing_skip_events` — number of
  hold-cycle routing-skip re-emissions (`chain_hold_cycle>0`) in the
  window, excluded from every demand/local-skip counter
  (LP-0MUIEO6IP0030CT3): `jq .hold_cycle_routing_skip_events`.
- `--json` summary field `provider_context_skips` — provider context-ceiling
  skips (LP-0MVMBM1Y3001TPRT): `{"total": N, "share_of_skips": f,
  "standard": N, "low": N, "by_provider": {provider: count},
  "by_limit": {limit: count}}`. `share_of_skips` uses **all** skip events
  (`routing_skip_local` + provider-context) as its denominator; an event whose
  line carried no `limit` is keyed `(unknown)` rather than dropped.
- `report.md` — the aggregate report: a single **Session summary** table
  (sessions, requests, local/remote split, classifications, fallback events,
  dispatch denials, context sizes — each with **Total / Fast / Cheap**
  columns), a **Total tokens** section (Input / Output / Total counts and
  % share, with fast/cheap and by-model breakdowns; see [Total tokens](#total-tokens)
  below), fallback-reason and routing-skip breakdowns, a **Provider context
  skips** section (`routing_skip_provider_context` events rendered only when
  the window has them: count, % of all skip events, Standard/Low split,
  provider/type breakdown and the `limit` ceiling each provider hit, plus the
  median/p90/max estimated prompt that was blocked), per-model
  breakdown, **Error analysis** (when the window has error events) with a
  taxonomy table that includes a **Status** column (upstream HTTP errors are
  split into one row per status code), a **Provider/model breakdown** table
  (error type × provider × model × count), and an
  **Upstream HTTP error breakdown by status** table (status × count ×
  provider breakdown),
  **Mode timeline & restarts** (when the window contains mode transitions
  or restart markers): one row per mode-switch / restart event with the
  attribution (source / actor / reason) when present in the logs, plus the
  applied/requested operating mode (mode-switch rows show the applied mode;
  restart rows show the restart kind and, when logged, the requested mode).
  Events without markers fall back to `unknown` display values so historical
  logs degrade gracefully; source values render as the canonical
  `scheduler` / `worker` / `api` / `manual` labels (LP-0MVC144KS001PDR4).
  **Summary by hour** (rendered at the top of the report, before the
  Session summary and always — regardless of local traffic): one row per
  **absolute wall-clock hour** of the report window (partial first/last hours
  truncated to the window edges, idle hours included as `0s`) with busy time
  / busy-% columns — busy seconds ÷ window-bounded bucket duration,
  defensively clamped to 100% — plus five per-hour **activity** columns
  counted by each event's own timestamp within the bucket: `Requests started`
  (all `Stream started` events), `Local attempts` (`Stream started` with
  `provider=local`), `Local served` (`Stream finished` with
  `provider=local`), `Fallbacks` (`Fallback triggered` events), and
  `Local-skip` (`routing_skip_local` events); a **Totals** row gives the
  window busy totals plus the sum of each activity column over the window.
  **Chain-hold de-duplication (LP-0MUIEO6IP0030CT3):** the `Local-skip`
  columns count only genuine demand — a held request re-emitted across
  chain-hold cycles (`chain_hold_cycle>0`, LP-0MU5AIAAY003KVM0) is counted
  once (its cycle-0 occurrence), and the re-emissions are surfaced separately
  in a **Hold-cycle re-emissions** section (per-session breakdown) plus a
  header line (`hold-cycle re-emissions (excluded from demand)`). The
  `routing_skip_local reasons` table and its `% of skips` percentages use the
  same de-duplicated counts, and the `--json` summary exposes the re-emission
  volume as `hold_cycle_routing_skip_events`. Historically, local-skip
  totals for windows containing held requests were inflated (one held
  request counted N+1 times); the intended count drop is therefore an
  accuracy fix, not lost data.
  Absolute-hour buckets keep the two edge partials of a rolling window
  distinct, so no busy duration is double-counted and no row exceeds 100%
  (the previous hour-of-day keying could collide them and emit impossible
  >100% rows, LP-0MTYAZFGN003BYQS); the `--json` summary exposes the same
  per-hour activity as `hourly_activity`, plus `hourly_busy` and
  `hourly_session_classification`, all keyed by absolute hour. Each
  `hourly_activity` entry (LP-0MVKUKG3F0035I85) carries `state` (the
  `+`-joined availability label), `states` (the ordered tuple), and the raw
  evidence counters `local_dispatch_attempts`, `local_denials`,
  `lease_active_fallbacks` alongside `local_served`, so a machine consumer
  can audit the availability verdict instead of trusting it, **Local model
  utilization** (busy time %, idle time, streams served, avg
  stream duration, total compute, avg/peak concurrency,
  fast/cheap split — when the window has local traffic) followed by
  **Prefill vs decode** (phase slot-time split: prefill dispatch → first byte
  vs decode first byte → finish, each as slot-seconds and % of combined phase
  slot time summing to 100%, with p10/p50/p90/max per phase and unknown-stream
  counts; the `--json` summary exposes it as `local_busy.prefill_vs_decode`),
  **Time to first
  token** (p10 / median / p90 of dispatch → first-token latency, split into
  **Total / Fast / Cheap** rows using the same three-tier fast/cheap bucketing
  (mode timeline → `server.priority_window` → slot schedule) as
  the other breakdowns, from local `dispatch_first_byte_ms=` and remote
  `ttft_seconds=` log lines; the `--json` summary exposes the same split under
  `ttft` with `total` / `fast` / `cheap` sub-objects), **Decode speed** and
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

## Verifying the daily path is agent-free

The daily report is produced solely by the 05:00 cron — the deterministic
CLI is the only trigger and no scheduler dispatch entry (herdr
scheduled prompt) remains. Confirm from the repository root:

```bash
# 1. No scheduler dispatch entry for the skill remains
test ! -e .worklog/scheduled-prompts.json \
  || ! grep -q 'proxy-usage-analysis' .worklog/scheduled-prompts.json

# 2. The cron entry is present and calls only the CLI
crontab -l | grep proxy-usage-analysis

# 3. No agent-orchestration narration remains in the thin skill wrapper
! grep -q 'do NOT ask permission' .pi/skills/proxy-usage-analysis/SKILL.md
```

All three checks must pass. `.worklog/scheduled-prompts.json` is a
gitignored, per-project runtime config (absent ⇒ empty set); removing the
entry is a local runtime edit, not a committed artifact.

## How it works

1. **File discovery** — the live `proxy.log` plus every rotated sibling
   (`proxy.log.YYYY-MM-DD_HH` and `proxy.log-YYYY-MM-DD_HH`, both dot and
   dash naming, plain or `.gz`). All rotated files are included regardless of
   their name-encoded timestamp: in this deployment a rotated file routinely
   holds data well past its encoded rotation time, so a name-based inclusion
   test would silently drop in-window data. Per-line timestamp filtering in
   step 2 is the authoritative window boundary. **Gzip-compressed files are
   decompressed transparently** — reading a `.gz` as plain text yields zero
   parseable lines while still counting the file as scanned, which silently
   discards its entire contents.
2. **Streaming parse** — files are read line by line (never loaded into
   memory; the live log can exceed 700 MB). Only structured prefixes are
   parsed: `Stream started`, `Stream finished`, `Fallback triggered`,
   `routing_skip_local`, `routing_skip_provider_context` (a per-provider
   context-ceiling skip rendered as its own **Provider context skips**
   breakdown in the report; parsed into `provider_context_skip`
   events carrying `provider`, `type`, `reason`, `estimated_tokens`,
   `limit`, `session`, and the source file/line as evidence,
   LP-0MVMBM1Y3001TPRT), `local_dispatch_denied`, `[local] POST` (every
   dispatch attempt at the local llama-server — including attempts that
   never yield a first byte, LP-0MVKUKG3F0035I85), the server-side
   compaction events (`compaction_event`, `compaction_backstop`,
   `compaction_churn` — parsed tamerly with missing fields defaulting to
   None, LP-0MTHCTLAF00147IT), plus the operating-mode lines (`Mode
   scheduler: applied scheduled mode fast|cheap`; manual switches, which
   log `Grandfathering: enabled; other-mode config ...
   (current=fast|cheap)`, LP-0MT1EE315007AKXG), and the error lines (`Stream
   error:`, `slot_save failed`, `backend_retry`, `[remote] upstream error`).
   Restart markers (`Mode-switch restart spawned:`, `restart_services:
   router-mode restart complete`) and the startup attribution line
   (`Proxy restart attribution:`) are parsed into `restart` events
   (LP-0MUQO82LJ008F1FS, LP-0MVC0S6U5003SEDJ). Attribution markers
   (`source=`, `actor=`, `reason=`) appended by LP-0MVAPNC2900145RH to
   restart/mode-switch/startup lines are extracted into structured
   fields; absent markers degrade gracefully to `None`. A `mode=`
   marker on restart lines (the set_mode() spawn carries the mode being
   switched to) is extracted into the event's `mode` field
   (LP-0MVC144KS001PDR4) so the report can surface the requested mode. A
   `chain_hold_cycle=<N>` token on a `routing_skip_local` line is parsed
   into the event's `chain_hold_cycle` field (LP-0MU5AIAAY003KVM0); lines
   without the token (pre-chain-hold rotated logs) parse with
   `chain_hold_cycle=None`, which aggregates as cycle 0 (genuine demand).
   Local `dispatch_first_byte_ms=` lines (LP-0MTSSM5SO003PKU0) parse into
   `ttft` events carrying the prefill breakdown: `ttft_ms` (dispatch → first
   byte total), `dispatch_to_first_byte_ms` (total minus queued wait),
   `queue_wait_ms`, `cache_state` and `cached_ratio`, plus `session` and
   `model` (LP-0MVJN025B001I8AL). `log_parser.pair_ttft_to_streams()` pairs
   them FIFO per session (Nth TTFT ↔ Nth `stream_started`, the same
   convention the busy-time calculation uses for start/finish pairs); streams
   without a TTFT line pair with `ttft=None` — unknown prefill, reported
   rather than dropped.
   Unparseable lines are counted and skipped, never fatal.

   **`Fallback triggered` log contract (LP-0MVMBMSWR0036RES).** The proxy
   emits one line per fallback, with the request's resolved session and a
   per-session fallback ordinal trailing the reason:

   ```text
   Fallback triggered for model=<path>, from=<provider>, to=<provider>, reason=<reason>[ limit=<n> remaining=<n> waited=<seconds>] session=<uuid|unknown> attempt=<n>
   ```

   - `session=` is the request's resolved session UUID, or `unknown` when the
     request carried none (the parser degrades `unknown` to "no session").
   - `attempt=` is the session's 1-based fallback ordinal. It keeps increasing
     across requests, so two fallbacks from the same session in the same
     second are distinguishable.
   - `limit=` / `remaining=` / `waited=` are the budget snapshot
     (`provider_budget.BudgetDecision`: `limit_rpm`, `remaining`,
     `waited_seconds`) and are emitted **only** when the reason is
     budget-driven (`rate_limited*`). The decision itself is unchanged.
   - All evidence fields trail `reason=`, so pre-change rotated logs (no
     evidence fields) still parse: every evidence field degrades to `None`.
     Consumers that anchored `reason=` to end-of-line were updated in the same
     change.
3. **Session grouping** — a session is identified by its UUID
   (`session=<uuid>`). Per-session context/response sizes use the
   authoritative `tokens=prompt/completion/total` from `Stream finished`
   lines (payloads in logs are truncated and never used for sizes).
4. **Local model utilization (busy time)** — local `Stream started` /
   `Stream finished` events are collected across the analysis's effective
   margin beyond the window (48h, shared with the mode timeline in step 5;
   see `MODE_TIMELINE_MARGIN`), paired per session (FIFO), clipped back to
   the window, and merged. **Busy is the union of dispatch → finish
   intervals** (LP-0MVJN9H2I00116FP): each paired stream's interval starts at
   its back-calculated dispatch time (`ttft.ts - ttft_ms/1000`, i.e. the
   prefill line's first-byte timestamp minus its measured duration) and ends
   at its `Stream finished` line, so the prefill (prompt-processing) phase is
   counted alongside decode. A stream with no first-byte line falls back to
   its decode interval (`Stream started` → `Stream finished`) and is
   attributed to the lower bound — busy is never overstated by a fabricated
   prefill. The decode-only union is retained as
   `BusyStats.decode_busy_seconds`, the labelled **lower bound** of the busy
   figure; `busy_seconds` is the labelled **upper bound**
   (`BUSY_DEFINITION = "dispatch_to_finish"`,
   `BUSY_UPPER_BOUND_LABEL`/`BUSY_LOWER_BOUND_LABEL`). All derived figures work
   on the new intervals: total compute is the sum of clipped stream durations
   (slot-seconds), peak concurrency comes from a sweep over interval
   endpoints, and the fast/cheap and hourly splits are attributed from the
   dispatch → finish union (`fast_busy_seconds`, `cheap_busy_seconds`,
   `hourly_busy`); `prefill_busy_seconds` is the wall-clock prefill span
   (upper − lower). The `--json` summary exposes the same fields under
   `local_busy` — the definition marker `"definition": "dispatch_to_finish"`,
   `decode_busy_seconds` (retained decode-only union, the lower bound),
   `prefill_busy_seconds` (prefill span, derivable from the two unions),
   `prefill_extended_streams` (how many streams carried a measured prefill)
   and the human labels `upper_bound_label` / `lower_bound_label` — so a
   machine consumer never has to infer the busy semantics
   (LP-0MVJNW6SN008AKH3). Every pre-change key is retained unchanged, and the
   phase split stays at `local_busy.prefill_vs_decode`. Busy seconds are attributed to **absolute** hours and
   to fast/cheap periods (slot schedule) by splitting at hour and period
   boundaries. The top-of-report **Summary by hour** table
   (LP-0MTFO210Q0044TTF, reworked by LP-0MTYAZFGN003BYQS) then renders one
   row per **absolute wall-clock hour** of the report window — the first/last
   rows truncated to the window edges and every hour listed even when idle —
   with a busy-% column (busy seconds ÷ window-bounded bucket duration,
   clamped to 100%), a **State** column carrying the hour's local-availability
   classification (LP-0MVKUKG3F0035I85, see below), plus seven per-hour
   activity columns counted by each event's own timestamp in the bucket:
   `Requests started` (all `Stream started`), `Local attempts` (local `Stream
   started`), `Local served` (local `Stream finished`), `Local dispatch`
   (every `[local] POST` dispatch attempt, including attempts that never
   stream), `Lease denials` (`local_dispatch_denied` refusals), `Fallbacks`
   (`Fallback triggered`), and `Local-skip` (`routing_skip_local`); a final
   totals row gives the window-level state plus the window busy totals and
   the sum of each activity column. Counting by event
   timestamp means long-running sessions contribute to every hour they touch,
   not just their first-request hour. It renders regardless of local traffic
   (busy columns read `0s` / `0.0%` when the window has no local streams;
   with no events a "No data" note is shown). Absolute-hour bucket keys keep
   the two edge partials of a window straddling an hour distinct, so no busy
   duration is double-counted and no row exceeds 100% (the previous
   hour-of-day keying collided them). Streams whose start has no paired finish are counted in
   `unfinished_streams` **only when they started inside the window or within
   `BUSY_WINDOW_MARGIN` (1h) before it**; streams started earlier are stale
   pre-window leftovers from earlier windows, tracked separately in
   `pre_window_unfinished`, and streams started after the window end belong
   to the next window (LP-0MSVRRO3L0056N6C).
   **Prefill statistics** (LP-0MVJN45LF009CO2V) are computed in the same pass
   from the local `dispatch_first_byte_ms=` events: `aggregate()` hands the
   margin-widened local TTFT events to `compute_busy_stats(ttft_events=...)`,
   which pairs them FIFO per session to the local streams and reports
   `BusyStats.prefill_stats` (`PrefillStats`). Each stream's prefill interval
   is `[ttft.ts − ttft_ms/1000, ttft.ts]`; the duration comes from
   `dispatch_first_byte_ms`, falling back to the logged
   `dispatch_to_first_byte_ms + queue_wait_ms` split and then to pairing the
   dispatch and first-byte timestamps (a non-positive pairing is not
   evidence). `PrefillStats` exposes one `PrefillSample` per stream (session,
   first-byte timestamp, seconds) plus `p10`/`p50`/`p90`/`max_seconds` and
   `total_seconds`. **Every prefill figure is slot-seconds, not wall-clock**
   (`unit`): `total_seconds` sums the per-stream durations (total prefill
   work), so it is deliberately *not* comparable with the wall-clock union in
   `BusyStats.busy_seconds`. Streams with no first-byte line are unknown
   prefill — excluded from the percentiles but retained in
   `stream_count`, so an aborted request is visible instead of silently
   dropped. A stream is attributed to the window containing its first-byte
   timestamp, and its interval is not clipped: a dispatch before
   `window_start` keeps its full duration and a first byte after
   `window_end` belongs to the next window (no double counting).
   `prefill_stats` is `None` only when the caller supplied no TTFT events.
   The `--json` summary exposes the same statistics as a top-level
   `local_prefill` object — `unit` (`slot-seconds`), `slot_seconds` (window
   total), `count` (measurable samples), `streams` (all local streams
   considered), `unknown_prefill` (streams with no observable prefill) and
   `p10` / `p50` / `p90` / `max_seconds` — with `0`/`null` values (never a
   missing key) when there is local traffic but no measurable prefill, and
   `null` when the window has no local traffic at all (matching `local_busy`)
   (LP-0MVJNW6SN008AKH3).
   **Decode statistics** (LP-0MVJNDOUU0059MMO) are computed in the same pass
   on the same paired local streams, without needing TTFT events:
   `BusyStats.decode_stats` (`DecodeStats`) holds one `DecodeSample` per
   stream whose `Stream started` (first byte) falls in the window, with the
   duration `Stream finished` − `Stream started`. Like prefill it is
   **slot-seconds, not wall-clock** (`unit`), attributed by first-byte
   timestamp and not clipped; a stream with no logged finish is unknown
   decode (excluded from the percentiles, retained in `stream_count`).
   `PrefillStats` and `DecodeStats` share the `SlotSecondsStats` base
   (samples, percentiles, `total_seconds`, `unknown_count`), so the two
   phases are directly additive: the `## Prefill vs decode` section
   (LP-0MVJNDOUU0059MMO) splits local phase slot time into prefill and decode
   as absolute slot-seconds and as a percentage of combined phase slot-time
   that always sums to 100%, with p10/p50/p90/max per phase and an explicit
   slot-seconds-not-wall-clock statement. A window with no prefill samples
   prints a named reason and points at the `## Decode speed` /
   `## Prompt eval speed` eval-timing diagnostic rather than leaving an
   unattributed blank (a window with no local traffic prints a named "no
   local traffic" reason). The same split is in the JSON summary at
   `local_busy.prefill_vs_decode`.
5. **Fast/cheap bucketing** — a session is bucketed by the **profile active
   at its first in-window stream**, from three sources in precedence order
   (LP-0MVKSXXIR006OBI2):
   1. **Mode timeline** (historical windows) — reconstructed from the
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
      This source **wins** whenever the logs carry the evidence, so archived
      windows keep their historical attribution.
   2. **Priority window** (the current deployment) — the fast/cheap mode
      schedule was retired (LP-0MVDT94F0003VHUA); cheap semantics now come
      from `server.priority_window`, which caps every request to `low`
      priority inside its local-time window. When no mode transitions exist,
      that window is the fast/cheap schedule: traffic inside it is **cheap**,
      traffic outside **fast**. `start`/`end` are inclusive/exclusive local
      times and `start > end` wraps past midnight (e.g. `22:00`–`06:00`,
      split into two cheap segments); `enabled: false` attributes no cheap
      time at all. The profile's slots/ctx still come from the unified
      config, so only the time axis is re-labelled.
   3. **Slot schedule** (legacy fallback) — bucketing from the analysis-time
      config's `slot_schedule` entries by slot count; nothing is hardcoded.
   The report's `## Priority window timeline & restarts` section names which
   source applied, so the attribution is auditable (a retired mode schedule
   with a `server.priority_window` source shows the window bounds; a window
   with mode evidence states that the priority window was not applied).
6. **Total tokens** — a `## Total tokens` section rendered **immediately before**
   `## Fallback reasons` in every run, including zero-traffic windows (shows
   `0` counts with a `_No tokens in window._` note for the by-model table).
   Totals are `Input (prompt) + Output (completion)` summed from the
   authoritative `tokens=prompt/completion/total` on `Stream finished` lines
   (never payload truncation), with a top row for Input / Output / Total
   (absolute + % of grand total), a fast vs cheap sub-table (absolute +
   % of total per bucket; bucketing follows the same three-tier fast/cheap
   logic (mode timeline → `server.priority_window` → slot schedule),
   unattributed streams excluded or documented), and a by-model
   sub-table (one row per `provider/model`, absolute + %, sorted by Total desc;
   unknown provider/model renders as the `(unknown)` placeholder).
7. **Recommendations** — rule-based heuristics, each citing the data that
   supports it (see below).
8. **Error taxonomy** — error events (`Stream finished: reason=error`,
   `Stream error:`, `slot_save failed`, `backend_retry`, `[remote] upstream
   error`) are parsed in the same streaming pass, collected per window, and
   rendered into the report's **Error analysis** section (taxonomy table
   **with a Status column** for upstream HTTP errors, plus a
   **Root-cause breakdown** table for stream errors, an
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
   items. The **Server-side compaction** section (LP-0MTHCTLAF00147IT) is
   rendered immediately after the Session summary whenever the window has
   compaction telemetry (and otherwise as a "No compactions observed in
   window" note). It covers Total / Fast / Cheap split (using the
   compaction event's own `mode` field, aligned with session bucketing),
   compact vs remote_with_guidance vs backstop action split, dry-run
   (`dry_run=true`) vs live (`dry_run=false`) labelling, token/turn
   economics (avg pre_tokens → post_tokens → tokens saved, avg
   turns_summarized), and fallback-avoidance impact (compacted sessions that
   stayed local vs still fell back, plus an estimate of avoided
   `large_context_bypass`/`context_too_large` fallbacks via the effective
   large-context warm threshold; dry-run estimates are labelled
   "would-have-avoided" to avoid overstating). The `--json` summary
   includes a `compaction` object (see [Outputs](#outputs)).
9. **Decode/prompt-eval speed** — llama-server eval-timing lines
   (`eval time = <ms> ms / <n> tokens (<x> tok/s)` and `prompt eval time =`)
   are streamed from `llama-server.log*`, filtered to the Qwen3 child port
   (discovered per file from the `name=Qwen3 on port <port>` spawn line;
   the port changes on every restart). Rotation is recognised in both
   schemes: numbered in-process rotations (`llama-server.N.log`) and the
   logrotate daily archives (`llama-server.log-YYYY-MM-DD`), decompressing
   `.gz` transparently. When rotation splits the Qwen3 spawn line away from a
   child log, the port is **inferred** from the single, otherwise-unannounced
   port carried by that file's eval lines (ports announced for other models,
   e.g. `name=mxbai-embed on port N`, are excluded); a file whose port stays
   ambiguous is skipped with a named reason. Samples are bucketed Total /
   Fast / Cheap with the same three-tier fast/cheap attribution as the
   session tables (mode timeline → `server.priority_window` → slot schedule)
   and summarised as median / p90 / p10 tok/s. Because llama-server.log lines
   carry no timestamps, each sample is bucketed by its log file's last-write
   time (approximate; documented in the report). An empty window is
   **self-diagnosing**: the section prints
   the log directory scanned, the files scanned, the eval lines matched and
   a named cause — `verbosity/build` (no `eval time` lines in the files at
   all), `rotation` (files outside the window), or `attribution failure`
   (Qwen3 port undeterminable) — instead of an unattributed blank
   (LP-0MVJNIZRO008S031). The same context is in the `--json` summary under
   `decode_speed` / `prompt_eval_speed` (`log_dir`, `files_discovered`,
   `files_outside_window`, `files_unrecognised`, `qwen3_ports`,
   `eval_lines_total`, `eval_lines_matched`, `file_diagnostics`, and
   `empty_window_reasons` when the window is empty). The per-session CSV
   `decode_tok_s` column is derived from proxy.log instead (local completion
   tokens ÷ local active span) and stays empty when not derivable.

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
- **Provider context skips** (`routing_skip_provider_context`): a remote
  entry's `max_context_tokens` ceiling was exceeded by the estimated prompt,
  so that provider was skipped **for this request only** (no cooldown, no
  failure-domain exclusion, no usage-limit reset). Read the dedicated section
  as a chain-cost signal, not an outage: the request was still served, but each
  skip walked the chain further. The `limit` column names the ceiling that was
  hit, so a provider whose limit explains most skips is the one to raise (or
  reorder behind a larger-context entry); a ceiling that is low relative to the
  sessions' peak context overlaps the `context_too_large` /
  `large_context_bypass` advice above (same size problem, remote side). A
  session whose own provider-context skip precedes its first remote stream is
  attributed that skip (`fallback_reason=provider_context_exceeded`) unless it
  also logged a `routing_skip_local` line, which takes precedence.
- **Context pressure**: sessions whose max context approaches
  `local_model_ctx_size / slots` can force `large_context_bypass`.
- **Decode speed / Prompt eval speed**: when these sections are empty, read
  the **Why no eval-timing samples** block rather than assuming a data
  problem: `verbosity/build` means the files were read but carry no
  `eval time` lines (llama-server log verbosity or build flags), `rotation`
  means every `llama-server*` file falls outside the window (retention/-
  rotation too aggressive for the window), and `attribution failure` means
  the eval lines exist but no Qwen3 child port could be determined (an
  ambiguous or non-Qwen3 port). The block always names the log directory,
  the files scanned and the eval lines matched.
- **Local model utilization**: busy time is the share of the window with at
  least one local slot occupied **from dispatch to finish** (prefill +
  decode, LP-0MVJN9H2I00116FP); the retained decode-only figure is its
  labelled lower bound, and a stream with no first-byte line never inflates
  it. The **Prefill vs decode** section then splits the *phase* slot time
  into prefill (dispatch → first byte) and decode (first byte → finish) as
  slot-seconds and as a share of combined phase slot time (always 100%
  together, LP-0MVJNDOUU0059MMO) — read it for where slot time goes (prefill
  rarely dominates at 46% local occupancy, but it does on large cold
  prompts), not for window occupancy: phase slot-seconds sum per-stream
  durations and are not comparable with the wall-clock busy %. A low busy % with high fallback volume
  means the router is diverting requests before they reach local (see
  fallback reasons), not that local is underprovisioned. The **Summary by
  hour** table at the top of the report shows per-hour busy % and per-hour
  activity across exactly the report window (idle hours included), so a
  glance at the hourly pattern correlates demand with provider usage and
  shows when local was saturated vs idle; the totals row gives the
  window-wide busy % and the summed activity counts. `context_too_large`
  is the largest lever: despite the legacy name (`warm_cache_bypass`), it
  fires when the *estimated context* exceeds the effective warm-cache
  threshold (the per-slot clamp,
  `local_model_ctx_size // slots - headroom`, inflated by
  `token_estimate_multiplier`), so large-context sessions never reach local.
  Concurrency bursts beyond `session_slot_pool_size` show as
  `local_concurrency_limit` / `local_lease_active` fallbacks. Note the
  slots-vs-context trade-off: more slots shrink per-slot context and *raise*
  bypass volume, so do not add slots without a matching ctx-size increase.
- **Hour state (local availability)**: the **State** column of **Summary by
  hour** answers "was the local model *idle*, or *unable to serve*?" — a
  question busy-% cannot answer, because busy is computed only from paired
  `Stream started`/`Stream finished` events, so an hour where work never
  yields a first byte renders as `0s / 0.0%`. The four states are derived
  only from the logs, and a short footnote after the table repeats their
  definitions in every report:

  | State | Meaning | Evidence |
  |---|---|---|
  | `serving` | decode happened | ≥1 local `Stream finished` |
  | `prefill-only-no-decode` | prompts were prefilled, nothing was decoded | ≥1 `[local] POST` and **no** local `Stream finished` |
  | `lease-blocked` | local demand was refused because another session held the dispatch lease | ≥1 `local_dispatch_denied` or `reason=local_lease_active` fallback |
  | `idle` | no local demand at all | no dispatch attempt, no refusal, no local completion |

  The states are **not exclusive** and are `+`-joined in the cell, in the
  fixed order `serving`, `prefill-only-no-decode`, `lease-blocked`; `idle`
  only ever appears alone. The 2026-10-23 02:00-03:00 incident is the canonical
  example: a wedged prefill lease held the gate while the GPU prefilled ~2M
  prompt tokens and decoded nothing, so the hour renders
  `prefill-only-no-decode+lease-blocked | 0s | 0.0%` with `Local dispatch`
  173, `Local served` 0 and `Lease denials` 141 — an outage, not idle. The
  totals row carries the classification of the summed window evidence (the
  window-level verdict, so a single wedged hour is still visible in a 24h
  report). A high `lease-blocked` hour with a low busy % means the gate was
  held (see the `local_lease_active` fallback guidance above);
  `prefill-only-no-decode` without denials means dispatch happened but never
  completed, i.e. cancelled/aborted prefill (check
  `local_prefill_deadline_seconds` versus projected prefill ETA).
- **Total tokens**: Input + Output + Total (prompt + completion summed from
  `tokens=prompt/completion/total` on `Stream finished` lines; streams without
  `tokens=` contribute `0`). Check the per-bucket and per-model tables in the
  **Total tokens** section — all shares are of the same grand total, so
  fast% + cheap% = 100% and model %s sum to 100%. A large Input / Output
  disparity can reveal long-prompt sessions driving capacity pressure.
- **Fast vs cheap**: a large fallback-rate gap between buckets suggests the
  profile under-serves one period. Read the `## Priority window timeline &
  restarts` note first: it states whether the split came from the mode
  timeline (historical logs), the configured `server.priority_window` (the
  current cheap window), or the legacy slot schedule.
- **Error analysis** (see the **Error analysis** section and `errors.csv`):
  the taxonomy table counts each error type; the **Provider/model breakdown**
  table splits each type by provider and model (e.g. a spike of
  `Stream finished: reason=error` on `opencode-go/deepseek-v4-flash` vs
  `local/Qwen3` pinpoints which provider/model to fix). Provider/model
  attribution is best effort (see [How it works](#how-it-works)); values not
  derivable from the log line show as `-` in the report and `(unknown)` in
  `errors.json`.
  - `Stream finished: reason=error` — the client-visible synthetic error
    event. It carries the enriched `error_type`/`error_message`/
    `suggested_action` payload, and each event is classified with a
    **root cause** (see [Root-cause taxonomy](#root-cause-taxonomy)). The
    recovery-first re-route and informative-error fallback are **already
    shipped and live** (LP-0MSETOTWY000SU0Z, LP-0MT60S55M000TK1H), so the
    report's recommendation states the strategy is live and names the
    dominant residual cause instead of prescribing adoption.
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
  - `upstream error` (HTTP 429) — the mechanism depends on the error type,
    mirroring the proxy's precedence. A `GoUsageLimitError` (or any
    usage-limit error with a computable reset) takes the account-level
    usage-limit reset quarantine (LP-0MSLJPOCC0001ROJ); its duration is
    derived from the message's reset time or `metadata.limitName`
    (daily/weekly/monthly). A `FreeUsageLimitError` with no computable reset
    falls back to the 3-hour per-model cooldown (LP-0MRGU0I91006ODFD).
    Persistent 429s indicate an upstream quota/rate-limit issue; a bucket
    mixing both error types is reported with both mechanisms. The log parser
    extracts only `error.type`, so the report names the mechanism rather than
    echoing the computed period.
  - `upstream error` (HTTP 402) — account balance or subscription issue;
    the proxy cannot recover — top up the account or switch provider.
  - `upstream error` (HTTP 5xx) — server-side upstream error; monitor for
    clustering (provider outage). Other 4xx codes carry the specific error
    message in the `errors.csv` evidence column.

  <a id="root-cause-taxonomy"></a>
  **Root-cause taxonomy for stream errors (LP-0MUQO82LJ008F1FS / LP-0MUQO862C004UT7T):**

  Every `stream_finish_error` / `stream_error` event is classified with a
  deterministic `root_cause` from log fields plus restart correlation (a
  documented ±120s window around any restart marker: `Mode-switch restart
  spawned:`, `Proxy restart attribution:` (startup attribution, LP-0MVC0S6U5003SEDJ),
  `restart_services: router-mode restart complete`, or a mode-switch line):

  1. **`upstream_stall_after_content`** — the enriched `error_type` is
     `stall_after_content`: the upstream stalled *after* content was delivered,
     so re-route is impossible and the informative error is the correct
     terminal outcome (by design; no proxy-side action).
  2. **`mode_switch_restart`** — a `local` stream error within the correlation
     window of a restart / mode-switch marker: the restart killed the
     in-flight llama-server stream. The bounded drain bounds the window
     (LP-0MUQO862J001VBWU).
  3. **`remote_chain_exhaustion`** — any other remote-provider stream error
     (empty response / stall retries exhausted while siblings were in
     cooldown). Correct handling (enriched error, no re-route after
     content/tool_calls); not a proxy defect. The client-side retry closes
     the loop (LP-0MUQO862N002559W).
  4. **`other`** — no restart correlation and no known enriched type; never
     fabricated. Inspect the `errors.csv` evidence.

  The classification is surfaced as the `root_cause` column in `errors.csv`,
  the `by_root_cause` breakdown in `errors.json`, and the **Root-cause
  breakdown** table in the report. The 2026-09-25 → 2026-09-26 window split
  was **24 / 6 / 1** (stall / restart / remote). The prior LP-0MT60S55M000TK1H
  investigation of 13 `reason=error` events used the same three real classes
  (restart kills, remote chain exhaustion, one genuine ReadTimeout); this
  taxonomy generalises it and adds the deterministic restart correlation.

  **Client visibility:** the enriched `error` object is emitted in the SSE
  event, but `@earendil-works/pi-ai`'s `mapStopReason()` collapses
  `finish_reason: "error"` to the generic `Provider finish_reason: error`
  and drops `type`/`message`/`suggested_action` before the client sees it.
  Tracked upstream as
  [earendil-works/pi#10752](https://github.com/earendil-works/pi/issues/10752);
  until it lands, the ContextHub recovery extension retries the generic
  string (WL-0MUQO8AEE003G1Z2, verified by LP-0MUQO862N002559W).

## Limitations

- **Hold-cycle re-emissions are excluded from local-skip demand counters
  (LP-0MUIEO6IP0030CT3).** A held request re-emitted across chain-hold
  cycles (`chain_hold_cycle>0`) counts once (its cycle-0 occurrence) in the
  `Local-skip` columns, the `routing_skip_local reasons` table, and the
  `--json` counters; the re-emissions appear only in the **Hold-cycle
  re-emissions** section and the `hold_cycle_routing_skip_events` JSON
  field. Local-skip totals for windows containing held requests therefore
  fall versus pre-change reports — this is the intended accuracy fix, not
  lost data.
- **Per-session fallback attribution (LP-0MVMBMSWR0036RES).** Since
  `Fallback triggered` carries `session=`, a session's fallback reason is taken
  from its **own** evidence and never inherited from a neighbouring line. The
  precedence is (1) the session's own `routing_skip_local` line at/before its
  first remote stream, (2) the session's own `routing_skip_provider_context`
  line at/before its first remote stream (LP-0MVMBM1Y3001TPRT — it names the
  provider and the `limit` that blocked that chain step, so it outranks the
  generic fallback line), (3) the session's own `Fallback triggered` line
  at/before its first remote stream, (4) the nearest **session-less** fallback
  line within 60s before the first remote stream — reachable only for rotated
  logs predating the session stamp — and (5) no own evidence at all. The
  provenance is exposed as the `fallback_attribution` column (`own_skip`,
  `own_provider_skip`, `own_fallback`, `legacy_nearest`, `lease_denied`,
  `unattributed`) plus the `no_remote_fallback` state for sessions that never
  used a remote provider; the report's `Session fallback attribution` line and
  the JSON summary's `fallback_attribution` map count every state. A session
  with no fallback line but `local_dispatch_denied` evidence is labelled
  `lease_denied` (`reason=local_lease_active`) — a lease/concurrency refusal,
  not an inherited spill. Before this change a session with no fallback line of
  its own inherited an unrelated neighbour's `rate_limited` within the 60s
  window (11 of 26 clean local-using sessions in the 2026-10-23/24 window were
  mislabelled that way).
- Provider context-ceiling skips (`routing_skip_provider_context`) are counted
  per event, not per provider chain: one request that exceeds several entries'
  ceilings logs one line per entry, so the count is *entry* attempts skipped.
  The line records no cooldown and the entry stays eligible for smaller
  requests. Older rotated lines may omit `limit=`; those events still parse and
  their ceiling is reported as `not logged` rather than zero.
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
  LP-0MSPZUD4G007IYGH). A window with no mode evidence buckets from the
  configured `server.priority_window` (local time, wrap-aware, `enabled:
  false` disables it; LP-0MVKSXXIR006OBI2) — the retired mode schedule's
  replacement — and only a window with neither signal keeps the legacy
  slot-schedule bucketing. The report's `## Priority window timeline &
  restarts` section states which of the three sources was used.
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
- Rotated logs may be gzipped. `iter_events` decompresses `.gz` files
  transparently (via `open_log_text`), so discovery never has to distinguish
  plain from compressed siblings. A truncated/corrupt gzip member would raise,
  but the surrounding deployment rewrites complete members only.
- Compaction events are window-filtered by per-line timestamp (`ts`) and
  by the same streaming, per-line margin as session / fallback / error
  events; the bucket (fast/cheap) of each `compaction_event` comes from its
  own `mode` field (the mode active at dispatch), not the analysis-time
  config. Backstop drops (`compaction_backstop` / `backstop_dropped` /
  `backstop_exhausted`) log as `compaction_backstop` lines at WARNING level
  (no `mode` field there) — they surface as the distinct "backstop
  sessions" row and count toward the action breakdown.
- llama-server.log eval-timing lines carry no timestamps, so the speed
  section's window filtering and fast/cheap split are approximate: each
  sample is bucketed by its log file's last-write time. Files whose Qwen3
  child port cannot be discovered are counted and skipped (never fatal).
- **The hour availability state uses proxy-log evidence only**
  (LP-0MVKUKG3F0035I85). llama-server writes its prefill/decode evidence
  (`prompt processing progress`, `slot update_slots`, `eval time`) with **no
  per-line timestamp**, so those lines cannot be attributed to an hour
  deterministically — for the 6-hourly rotation a wedge window's GPU evidence
  can only be placed in wall-clock time by an analyst reconstructing a time
  base externally. The proxy's own `[local] POST` dispatch-attempt marker is
  therefore the per-hour "prefill started" signal: it is emitted for every
  attempt (including ones that never produce a first byte) with an exact
  timestamp, and its lack of a following local `Stream finished` is exactly
  the "no decode" half of `prefill-only-no-decode`. Prompt-token volume and
  `eval time` counts remain available at window level in the **Prefill vs
  decode** and **Decode speed** sections; they do not feed the per-hour
  state.
- The per-session CSV `decode_tok_s` is a session-level average over the
  local active span (first→last local stream event); it includes
  inter-request gaps, so it is a conservative lower bound of the true decode
  rate. It is empty for sessions with no local completions.
