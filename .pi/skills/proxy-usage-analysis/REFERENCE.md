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
- `--json` summary field `local_turn_coverage` — the deterministic local-turn
  coverage behind the report's `## Local model usage by session` section
  (LP-0MVM93DE6009JGBU): `{"sessions_in_window": N,
  "sessions_with_local": M, "percentiles": [{"label": "P10",
  "local_turns": f, "remote_turns": f, "local_turn_pct": f,
  "context_at_fallback": i|null}, …], "total": {"local_turns": N,
  "remote_turns": N, "local_turn_pct": f, "context_at_fallback": i|null,
  "answered_turns": N, "unanswered_turns": N, "context_samples": N},
  "reasons": [{"reason": r, "sessions": N,
  "pct_of_sessions_with_local": f}, …], "sessions": [{…per-session row
  with `reason`, `context_at_fallback` and `context_at_fallback_source`…}],
  "notes": {"sessions_with_fallback": N,
  "declined_before_local_turn": N, "context_fallback_sessions": N,
  "context_fallback_before_local_turn": N, "provider_budget_exhausted_sessions": N}}`.
  A `% local turns` of 100 means the session never used remote; a null
  `context_at_fallback` means the logs carried no token evidence (the report
  renders `n/a`, never an estimate).
- `--json` summary field `engine_cross_check` — the engine cross-check
  (LP-0MVNWULLI006ZYXX): proxy-inferred phase slot-time vs llama-server's own
  summed per-task eval durations. `{"unit": "slot-seconds",
  "engine_metric": "sum of per-task eval durations (slot-seconds, not
  wall-clock)", "metric_caveat": str, "interpretation": str,
  "window_approximation": str, "threshold_pct": f,
  "threshold_enabled": bool, "status": "ok"|"partial"|"unavailable",
  "prefill": {"phase", "proxy_slot_seconds", "proxy_samples",
  "proxy_streams", "engine_seconds", "engine_samples", "delta_seconds",
  "delta_pct", "available", "reason", "divergent"}, "decode": {…same…},
  "available_phases": [phase], "divergent_phases": [phase],
  "engine_unavailable_reasons": [str]}`. `delta` is proxy − engine and
  `delta_pct` is of the engine figure; an unavailable phase carries `null`
  deltas plus a named `reason` — never `0`/`100`. Additive: every pre-existing
  key is unchanged.
- `--json` summary field `provider_context_skips` — provider context-ceiling
  skips (LP-0MVMBM1Y3001TPRT): `{"total": N, "share_of_skips": f,
  "standard": N, "low": N, "by_provider": {provider: count},
  "by_limit": {limit: count}}`. `share_of_skips` uses **all** skip events
  (`routing_skip_local` + provider-context) as its denominator; an event whose
  line carried no `limit` is keyed `(unknown)` rather than dropped.
- **Prefill-regression keys** (LP-0MVOAQDB4002D56G, certifying the cold-prefill
  admission budget of LP-0MVN3KMGI009BZ0D). Five additive keys, so the 24 h
  AC4 certification no longer needs a hand-written log scan; the report renders
  them in a **Prefill admission regression** section whenever any is non-zero.
  - `prefill_deadline_events` — `prefill_deadline_exceeded` WARNING lines
    (an admitted local prefill cancelled at the deadline).
  - `prefill_deadline_events_by_class` — `{"user": N, "audit": M}`. A
    session id beginning `audit-` is agent **audit** traffic (gate sweeps,
    herdr child sessions); every other session — including an unattributed
    one — is **user** traffic. Audit runs are real inference but are not the
    workload the budget is priced for, so the deadline-rate criterion is gated
    on `user`.
  - `prefill_budget_refusals` — `local_admission_guard` events with
    `decision=reject reason=prefill_budget_exceeded` (over-budget prompts
    refused before any dispatch; other reject reasons are not counted here and
    remain visible in `local_admission_reject`).
  - `wasted_prefill_tokens` — sum of `new_tokens` over
    `local_eviction_after_acceptance` events whose `reason` is
    `prefill_deadline`: tokens the GPU computed for a dispatch then abandoned
    before its first byte. Other eviction reasons are excluded. The summed
    `new_tokens` are the **admission-time** values the routing decision priced
    and the dispatch record stored verbatim (LP-0MVODJ3P6000AIUM AC5): the
    budget's own numbers, never a recompute at eviction time. Records without
    an admission snapshot contribute 0 (explicit `unknown` marker), so the
    figure cannot over-account a re-priced prompt. In particular
    `measured_cache_deficit` (below) is **excluded**: that dispatch was
    released early, so its tokens are not a deadline cancellation and would
    over-state this figure.

- **Measured-cache evictions (Option D, LP-0MVR7IN0O000DPXN / guardian
  LP-0MVR7K74S001PBNI).** A dispatch whose *measured* `prompt_progress`
  (`total`/`cache`) shows a re-prefill far above the admission-priced
  `new_tokens` is released **before** the 120 s deadline and logged
  `local_eviction_after_acceptance ... reason=measured_cache_deficit` with the
  measured/priced token fields. It is attributable as its own reason, distinct
  from `prefill_deadline`:
  - parsed verbatim (never folded into the tenancy `local_eviction` kind nor
    into the deadline class);
  - counted in `local_evictions_after_acceptance_by_reason` under the
    `measured_cache_deficit` key, so a post-deploy window can compare it with
    the residual `prefill_deadline` count;
  - listed as its own row in the report's **Post-acceptance evictions**
    breakdown; the event's admission-priced `new_tokens` stays readable on the
    event (and in the raw log line) for the wasted-token comparison, while
    `wasted_prefill_tokens` continues to mean deadline cancellations only.
  - `provider_exhausted_errors` — `All providers exhausted` lines
    (client-visible chain failures); `upstream_status_503_errors` —
    `upstream_http_error` events with `status == 503` (provider-side 5xx).
    The two are separate on purpose: a provider can return 503 while another
    still serves the request.
- **Contention-prediction keys (LP-0MVGR2OXT0096OGK).** Three additive keys
  measure whether the payload's `lease_state` / `prefill_eta_seconds` signals
  predict real local contention; the report renders the same figures in a
  **Local contention prediction vs reality** section (M1/M2/M3).
  - `denial_attribution` — M1: each `local_dispatch_denied` (which carries no
    `lease_state`) is joined to its decision-block `local_admission_guard`
    `reject` record (nearest within `join_window_seconds`, ±60 s, inclusive) and
    the pool state at denial time is derived from the guard's measured
    `busy_slots` / `in_flight_prefills` fields: **`idle_leased`** (no slot
    decoding, no prefill in flight — the idle-lease false-busy case),
    **`prefill_bound`** (≥1 prefill running; serialised prefills make it the
    gating resource), **`saturated`** (slots decoding, no prefill), or
    **`unknown`** (legacy guard line without the fields). Entries: `denials`,
    `attributed`, `unattributed` (no guard in the 60 s block — never guessed),
    `by_verdict`, `by_guard_reason`, `idle_leased_share_pct`, `per_hour`
    (per absolute hour: `denials`, the four verdict columns,
    `unattributed`, `busy_seconds`, `busy_pct`), `capacity_hint` (the payload's
    own claim over `status_request` polls), and `verdict`
    (`matches_reality` / `does_not_match_reality` / `inconclusive`: the signal
    does not match when ≥50% of attributed denials fired with the pool measured
    idle).
  - `prefill_eta_error` — M3: two comparisons, because a cache-cold bound and
    a per-request estimate answer different questions
    (LP-0MVSD5RJW008F1IB). The pre-existing keys keep their names and shapes
    for trend continuity; `comparison` (`bound_vs_served_remainder`) states
    which comparison `verdict` describes.
    - **bound vs served remainder** — predicted `prefill_eta_seconds` (from
      `status_request` polls) vs observed llama-server `prompt eval time`
      durations, quantile-matched: `predicted` / `observed` (`samples`, `p50`,
      `p90`), `error_p50` / `error_p90` (predicted − observed, seconds),
      `relative_error_p50_pct` / `relative_error_p90_pct`, `tolerance_pct`
      (25, reported for continuity but **not** applied to this comparison),
      `bound` (`verdict`, `holds`, `predicted_p50`, `observed_p90`,
      `observed_samples`, `min_samples` 30, `margin_seconds`) and `verdict` =
      `bound_holds` when the served p90 ≤ the published p50, `bound_violated`
      otherwise, `inconclusive` below 30 observed dispatches. The old
      two-sided `over_predicts` / `under_predicts` wording is **gone** for this
      comparison: a cross-population comparison cannot score calibration.
    - **matched per-dispatch accuracy** — `matched`: predicted = the ETA the
      admission guard priced for that dispatch (`eta_prefill_s`), observed =
      the same dispatch's `dispatch_to_first_byte_ms`, joined on `session=`.
      Entries: `pairs`, `dispatches`, `unmatched_dispatches`,
      `admitted_pairs` / `observe_pairs` (mutually exclusive: an ordinary
      admitted dispatch, or an observe-policy would-be refusal — note the
      log's `enforced` token means "the refusal was applied", so `policy=`
      decides the split), `enforced_refusals` (guard records dropped because
      the refusal really fired: that request never dispatched, so it cannot be
      the owner of a first byte), `predicted_p50` / `predicted_p90`,
      `observed_p50` / `observed_p90`, `relative_error_p50_pct` /
      `relative_error_p90_pct` (percentiles of the per-pair relative error),
      `tolerance_pct` (25) and `verdict` (`matches_reality` within tolerance,
      `over_predicts` / `under_predicts`, `inconclusive` with no pairs). Only
      matched pairs are scored; a dispatch whose guard logged no priced ETA
      (inert/uncalibrated, or an inactive observe-mode decision) is counted as
      unmatched and never folded in.
  - `status_capacity_hint` — the same `capacity_hint` distribution as a
    top-level key (`{idle: n, prefill_bound: n, saturated: n}`), so
    `jq .status_capacity_hint` answers "what did the payload claim while the
    denials happened?" without parsing Markdown.
- **Local capacity signals key (LP-0MVU8O4EL005IYP9).** One additive key
  mirrors the **Local capacity signals** section, so automation reads exactly
  what the Markdown shows:
  - `capacity_signals.slots_stale` — `{status_polls, count, denominator, rate,
    target, passed}`: the status-poll count, the polls with
    `slots_stale=true`, the denominator (polls that carry the `slots_stale`
    field — schema-1 polls are excluded, never counted as fresh), the rate,
    the configured target (`--stale-rate-target`, default `0.05`) and the
    strict `rate < target` verdict. `passed` is `null` when the window has no
    status polls: the Markdown section is omitted then, so a consumer must not
    read a silent `true` as an all-clear. The derivation and the target check
    are shared with `scripts/capacity_signal_verification.py`
    (`scripts/lib/capacity_signals.py`), and status polls use the reference
    tool's half-open `[window_start, window_end)` window while every other
    section keeps its historical closed interval.
  - `capacity_signals.capacity_hint` (LP-0MVU8O9OC005YQU1) — `{status_polls,
    counts, shares}`: the `capacity_hint` histogram over status polls and each
    token's share of them. Every token keeps its own row — a new claim name, a
    case variant or `unknown` for an unreadable value is never dropped or
    re-bucketed — so a vocabulary change is visible in the JSON the night it
    lands.
  - `capacity_signals.saturated_without_live_work` (LP-0MVU8O9OC005YQU1) —
    `{saturated, count, rate_of_polls, rate_of_saturated}`: the
    `capacity_hint=saturated` polls that report no live local work
    (`local_in_flight_requests > 0` authoritative at schema 3, else the
    `local_active_query` / `active_query` fallback), with its share of all
    polls and of `saturated` polls. The rule and the tallies are the shared
    `scripts/lib/capacity_signals.py` implementation, so this key and
    `scripts/capacity_signal_verification.py` publish identical figures.
  - `capacity_signals.targets` / `capacity_signals.all_targets_met` — one
    entry per capacity target (`name`, `target`, `observed`, `passed`), the
    list the section's one-line verdict names, and the aggregate outcome
    (`null` when the window has no status polls, matching
    `slots_stale.passed`).
  - `capacity_signals.coverage` (LP-0MVU8O9OC005YQU1) — `{covered,
    window_start_gap_seconds, window_end_gap_seconds, max_gap_seconds}`: the
    span of the observed polls against the report window. When the retained
    logs do not reach both edges (1 h tolerance) the section labels the
    figures a **partial** measurement instead of letting a truncated
    retention window read as a complete night.
- **Contended-pricing split keys (LP-0MVS9YKHL007UBBZ, parent
  LP-0MVS8T6F4006DGG3).** Two additive keys split the same
  `local_admission_reject` / `local_admission_admit` verdicts, so a regression
  to the optimistic rate is visible in the standard report instead of
  requiring ad-hoc log archaeology. Both are **omitted-when-empty** maps (a
  bucket/band with no verdict is absent, never zero-filled):
  - `local_admission_by_rate_bucket` — `{rate_bucket: {"admit": n, "reject":
    n}}` over the `rate_bucket=` the prefill ETA was calibrated from:
    `contended` (priced against the contended window), `solitary` (the history
    this parser always saw), `solitary_fallback` (the contended window was
    uncalibrated, so the solitary window was used — the deliberately
    residual-optimistic case) and `unknown` (a pre-change line with no
    `rate_bucket` field at all). A high `solitary_fallback` share is the
    signal that the contended window still needs samples.
  - `local_admission_by_token_band` — `{band: {"admit": n, "reject": n,
    "converted": n, "unconverted": n, "conversion_share": f|null}}` over the
    **admission-conversion harness's** bands
    (`scripts/prefill_admission_conversion.py`: `<8k`, `8k-20k`, `20k-32k`,
    `32k-64k`, `>=64k`, plus `unknown` for a line with no priced
    `new_tokens`), so the report's conversion numbers are directly comparable
    with the harness's baseline/post-change measurement. `converted` /
    `unconverted` are **session-level**: a session with `decision=admit` is
    converted when a **local** first byte (`dispatch_first_byte_ms`) follows
    it; a remote first byte never converts a local admission, and a band with
    no admitted session reports `conversion_share: null` rather than `0.0`.
  The parsed `local_admission_guard` fields behind these keys —
  `rate_bucket`, `contended_samples`, `eta_occupancy_s` and the priced
  `new_tokens` — are likewise additive: a pre-change line leaves every one of
  them `None` and parses exactly as before.
- `report.md` — the aggregate report: a single **Session summary** table
  (sessions, requests, local/remote split, classifications, fallback events,
  dispatch denials, context sizes — each with **Total / Fast / Cheap**
  columns), a **Total tokens** section (Input / Output / Total counts and
  % share, with fast/cheap and by-model breakdowns; see [Total
  tokens](#interpreting-the-report) below), fallback-reason and routing-skip breakdowns, a **Provider context
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
  fast/cheap split, plus the hardware-side **GPU busy — power-based** row when
  Prometheus is reachable — see [GPU busy time from power](#gpu-busy-time-from-power-prometheus)
  — when the window has local traffic) followed by
  **Local contention prediction vs reality** (LP-0MVGR2OXT0096OGK, when the
  window has a denial or status poll): M1 per-hour denial attribution by
  derived pool state plus the guard-reason and `capacity_hint` splits; M2 the
  window's denials-per-hour and busy fraction compared with the pre-change
  baseline (2,641 denials / 24 h at 29.3% busy); M3 the quantile-matched
  *bound vs served remainder* comparison at p50/p90 with the one-sided
  `bound_holds` / `bound_violated` verdict, and M3b the *matched per-dispatch*
  accuracy (guard ETA → that dispatch's first byte); and one explicit
  matches/does-not-match-reality verdict for the denial signal and one for the
  cache-cold bound), then
  **Local capacity signals** (LP-0MVU8O4EL005IYP9, omitted when the window has
  no status polls): the `slots_stale` stale-poll rate for the window — status
  polls, polls with `slots_stale=true`, the schema-aware denominator, the rate,
  the configurable target (`--stale-rate-target`, default < 5 %) and an
  explicit met/unmet outcome, with the excluded schema-1 poll count stated —
  derived by the same implementation as
  `scripts/capacity_signal_verification.py` and exposed as
  `--json`'s `capacity_signals.slots_stale`, then
  **Admission verdicts by rate bucket and prompt band** (LP-0MVS9YKHL007UBBZ,
  when the window has any `local_admission_guard` decision): the admit/reject
  verdicts split by the pricing `rate_bucket` (`contended` / `solitary` /
  `solitary_fallback` / `unknown`) and by the conversion harness's new-token
  bands, with per-band admission → local first-byte conversion counts
  (buckets/bands with no verdict are omitted), then
  **Prefill vs decode** (phase slot-time split: prefill dispatch → first byte
  vs decode first byte → finish, each as slot-seconds and % of combined phase
  slot time summing to 100%, with p10/p50/p90/max per phase and unknown-stream
  counts; the `--json` summary exposes it as `local_busy.prefill_vs_decode`),
  **Engine cross-check** (LP-0MVNWULLI006ZYXX: the proxy-inferred prefill /
  decode slot-seconds compared against llama-server's own summed per-task eval
  durations — `prompt eval time` and `eval time` — with the absolute delta and
  the delta as a percentage of the engine figure, the explicit slot-seconds
  (not wall-clock, not comparable with `busy_seconds`) caveat, the window
  boundedness caveat, and an `Unavailable — <phase>: <reason>` note when the
  engine or proxy samples are missing; a phase above
  `local_engine_cross_check_threshold_pct` raises a warning recommendation and
  the `--json` summary exposes it as the additive top-level
  `engine_cross_check` key),
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
request counts, local turns answered/unanswered, per-session `% local turns`,
context size at fallback with its provenance
(`routing_skip` / `remote_stream` / `session_context`, empty when no token
evidence exists), dispatch denials, decode tok/s (derived from local
completion tokens ÷ local active span; empty when not derivable).

### Local-turn coverage (`## Local model usage by session`)

Since LP-0MVM93DE6009JGBU the report carries a per-session local-usage
section — which sessions actually used the local model, how many turns it
answered, what share of each session's turns were local, and why each session
moved to remote. It is rendered from data the pipeline already collects (no
new log source):

```text
## Local model usage by session

| Percentile | # turns local | # turns remote | % local turns | Context size at fallback |
| P10 | … |
| P50 | … |
| P90 | … |
| Total | … |

### Fallback reason distribution
| Reason | Sessions | % of sessions-with-local |

### Per-session coverage
| Session ID | # turns local | # turns remote | % local turns |
  Reason for remote fallback | Context size at fallback |
```

Operational definitions:

- **Covered session** — at least one local `Stream started` inside the
  window. A local turn counts as *answered* when a local, non-error
  `Stream finished` pairs with it (FIFO, the busy/TTFT pairing convention); a
  routed-but-unanswered turn (error, abort, log gap) is counted and flagged
  in the local column — never dropped.
- **`% local turns`** = `local / (local + remote) × 100` per session, so a
  session that never used remote reads 100 %. The summary rows are
  P10/P50/P90 (linear interpolation) over the covered sessions; the total row
  sums the turn columns, recomputes the share from those sums and shows the
  mean context-at-fallback of the sessions with token data.
- **Fallback reason** — the session's own attributed reason
  (`fallback_reason`), else its attribution state: `no_remote_fallback` (never
  used remote) or `unattributed` (no own evidence). A reason is never
  inherited from a neighbouring session (LP-0MVMBMSWR0036RES).
- **Context size at fallback** — the session's `routing_skip_local
  estimated_tokens` at/before its recorded fallback, else the first remote
  `Stream finished` prompt tokens, else its own `start_context_size` /
  `max_context_size`. Sessions with no token evidence render `n/a` plus a
  named limitation; the report never fabricates an estimate.

Sorting: the summary percentile rows ascend (P10 → P90); the per-session table
is sorted by `# turns local` **descending** (more local turns is the goal),
then by session id. When no session used local, the section renders the named
line `No session in the window used the local model.` instead of empty tables.
The same derivation is exposed as the `local_turn_coverage` `--json` key, so
Markdown and machine-readable views cannot disagree.

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
   **Per-turn correlation id (`turn=`, LP-0MVMBM467005BT4A).** Every
   structured line one client request emits carries a short correlation id:

   ```text
   Stream started: provider=<p> model=<m> session=<uuid> request=<preview> turn=a1b2c3d4
   Stream finished: reason=<r> tokens=<p>/<c>/<t> session=<uuid> provider=<p> model=<m> turn=a1b2c3d4
   routing_skip_local ... session=<uuid> chain_hold_cycle=<n> turn=a1b2c3d4
   routing_skip_provider_context ... session=<uuid> preference=<pref> turn=a1b2c3d4
   Fallback triggered for model=<path>, from=<a>, to=<b>, reason=<r> session=<uuid> attempt=<n> turn=a1b2c3d4
   routing_hop session=<uuid> from=<a> to=<b> hops=<n> turn=a1b2c3d4
   local_dispatch_denied session=<uuid> owner=<owner> active=<n> cold_start=<bool> turn=a1b2c3d4
   [local] POST <url> body=<preview> session_id=<uuid> session={...} slot=<id> turn=a1b2c3d4
   [remote] POST <url> -> <endpoint> body=<preview> session_id=<uuid> session={...} slot=<id> turn=a1b2c3d4
   dispatch_first_byte_ms=<ms> ... session=<uuid> model=<m> priority=<p> estimated_tokens=<n> turn=a1b2c3d4
   ttft_seconds=<s> session=<uuid> provider=<p> model=<m> turn=a1b2c3d4
   Stream error: session=<uuid> provider=<p> model=<m> error=<type> turn=a1b2c3d4
   ```

   - **Format.** Eight lowercase hex characters (`uuid4().hex[:8]`), emitted as
     the last `key=value` on the line by
     `proxy/turn_id.py::turn_field()`. Short on purpose: it keeps lines
     readable while making a per-session collision negligible.
   - **Stability.** One id per client request, generated at the request's entry
     point (`_run_chain_cycles`, `proxy_to_local`, `proxy_to_remote`, and
     idempotently at `log_request`) and **never** re-generated: it is stable
     across local↔remote fallback hops, chain-hold retry cycles, provider
     reroutes and stream start/finish. Two turns of the same session always get
     different ids.
   - **Parsing.** `log_parser` extracts it into `LogEvent.turn` on
     `stream_started`, `stream_finished`, `stream_finish_error`, `fallback`,
     `routing_skip`, `provider_context_skip`, `dispatch_denied`,
     `local_dispatch` and `ttft` events. Absent on any other kind (and on every
     pre-change rotated line), where it stays `None`.
   - **Exact attribution.** With ids present the joins are exact and no longer
     depend on ordering: `pair_ttft_to_streams()` pairs a
     `dispatch_first_byte_ms=`/`ttft_seconds=` line with the `Stream started`
     line of the same turn, `pair_stream_events()` pairs `Stream started` with
     its own `Stream finished` (busy time, decode/prefill slot-seconds), and
     `_attribute_fallback()` prefers the fallback/skip/denial evidence carrying
     the first remote stream's turn. Lines **without** ids (rotated logs
     predating the field) keep the historical per-session FIFO and
     nearest-line-within-60s heuristics, so historical windows still analyse
     unchanged.
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
     budget-driven (`provider_budget_exhausted*`). The decision itself is unchanged.
   - All evidence fields trail `reason=`, so pre-change rotated logs (no
     evidence fields) still parse: every evidence field degrades to `None`.
     Consumers that anchored `reason=` to end-of-line were updated in the same
     change.
3. **Session grouping** — a session is identified by its UUID
   (`session=<uuid>`). Per-session context/response sizes use the
   authoritative `tokens=prompt/completion/total` from `Stream finished`
   lines (payloads in logs are truncated and never used for sizes). Local
   streams report real measured usage since LP-0MVMHNVKT003PRIK (see
   [Local token reporting and the cold-cache cap](#local-token-reporting-and-the-cold-cache-cap)),
   and per-model token totals are attributed to the provider/model stamped on
   each `Stream finished` line — the one that actually served that stream —
   rather than to the session's initial provider.
4. **Local model utilization (busy time)** — local `Stream started` /
   `Stream finished` events are collected across the analysis's effective
   margin beyond the window (48h, shared with the mode timeline in step 5;
   see `MODE_TIMELINE_MARGIN`), paired **exactly by `turn=`** where the ids
   are present and per session (FIFO) for pre-change lines, clipped back to
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
   `prefill_extended_streams` (how many of the in-window paired streams
   carried a measured prefill — always ≤ `streams`, because the
   margin-widened collection also carries pairs from earlier windows and
   only pairs whose clipped busy interval is counted are included,
   LP-0MVKN85WO007O00H)
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
   **Engine cross-check** (LP-0MVNWULLI006ZYXX) validates that split against the
   engine's own accounting: `SpeedStats.engine_seconds` sums the
   Qwen3-attributed, in-window `prompt eval time` / `eval time` durations
   (produced inside the same attribution loop as the speed percentiles, so the
   port rules are inherited verbatim) and the `## Engine cross-check` section
   compares them with `PrefillStats.total_seconds` / `DecodeStats.total_seconds`
   — absolute delta and percentage of the engine figure. Both sides are
   **sums of per-task durations (slot-seconds)**, never wall-clock, so neither
   is comparable with `BusyStats.busy_seconds` / `busy_pct` (a merged union)
   and no arithmetic mixes them. A phase with no engine samples (or no proxy
   samples) renders an explicit `Unavailable — <phase>: <reason>` note reusing
   the parser's named diagnostics, with `null` deltas in JSON — never a
   fabricated `0` or `100 %`. A phase whose absolute percentage delta exceeds
   `local_engine_cross_check_threshold_pct` (default 25 %, `0` disables) raises
   a warning recommendation (`medium`, `high` at twice the threshold). The
   JSON summary carries the whole comparison under the additive top-level
   `engine_cross_check` key.
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
   unknown provider/model renders as the `(unknown)` placeholder). The section
   also states the **measured-usage coverage** — how many `Stream finished`
   lines carried `tokens=` and how many did not, split by provider — so
   "size unknown" is visible instead of being silently counted as zero
   (`streams_with_usage` / `streams_without_usage` /
   `streams_without_usage_by_provider` in the JSON summary). Terminal lines
   with no usage carry `tokens=missing` (LP-0MVMBM6N400541EH), and the
   per-session counts also appear as the `Streams without usage` column of the
   **Per-session coverage** table in *Local model usage by session* (next to
   the `Context size at fallback` figure whose `n/a` it explains) and as
   `streams_with_usage` / `streams_without_usage` in the per-session CSVs.
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
   tokens ÷ local active span) and stays empty when not derivable. The same
   attribution loop also sums the per-kind **engine seconds**
   (`SpeedStats.engine_seconds`, from `prompt_eval_seconds` /
   `decode_seconds`) for the `## Engine cross-check` section
   (LP-0MVNWULLI006ZYXX) — the engine's own per-task duration totals, summed
   (never wall-clock) and therefore comparable with the proxy's phase
   slot-seconds but not with `busy_seconds`.

### Local token reporting and the cold-cache cap

Local streams report **measured** usage since LP-0MVMHNVKT003PRIK. Before that
change, 0 of ~225 local `Stream finished` lines in a 24h window carried
`tokens=`, so local prompt sizes were unknown, the per-model token table
attributed local tokens only via sessions whose *initial* provider was local,
and the usage-driven cache-credit path (`update_cached_ratio` /
`record_prompt_prefix` in `proxy/provider.py`) could never fire for local.

The proxy now adds `stream_options: {"include_usage": true}` to local
streaming requests, so llama-server returns its OpenAI-compatible `usage`
block (`prompt_tokens`, `completion_tokens`, `total_tokens`,
`prompt_tokens_details.cached_tokens`). Two log-format details matter when
reading the logs:

- llama-server emits the usage block in a **separate chunk after** the
  `finish_reason` chunk, so the proxy defers the `Stream finished` line by one
  chunk and emits it as
  `Stream finished: reason=<r> tokens=<p>/<c>/<t> session=<uuid> provider=local model=<m>`.
  A stream that never delivers usage (client disconnect, aborted prefill)
  still logs exactly one terminal line, marked `tokens=missing`.
- **`tokens=missing` (LP-0MVMBM6N400541EH).** When a terminal stream reports no
  usage at all, the `Stream finished` line carries `tokens=missing` in the
  position where `tokens=<p>/<c>/<t>` sits — e.g. `Stream finished:
  reason=stop tokens=missing session=<uuid> provider=local model=<m>`. The
  marker is emitted from the same code path as the counted field
  (`proxy/proxy/router_helpers.py`: the deferred-finish flush at `[DONE]` /
  end-of-stream, the immediate finish-chunk emit for providers that never
  report usage, and the buffered `stream=false` reply), so "no data" is
  distinguishable from "zero tokens" instead of the field silently vanishing.
  It never overrides real usage: a present `usage` block still logs
  `tokens=<p>/<c>/<t>` unchanged. Consumers matching
  `tokens=(\d+)/(\d+)/(\d+)` simply do not match the marker (verified for
  `scripts/idle_admit_impact.py`, `scripts/slot-persistence-correlate.py`,
  `proxy/scripts/analyze_context_distribution.py` and
  `proxy/scripts/run_compaction_experiment.py`), and the analyser records it as
  `usage_missing=True` on the stream-finished event. Those streams are counted
  (`streams_without_usage`, window level and per session) and excluded from
  every token/context figure rather than treated as zero — which is why a
  session's `Context size at fallback` can render `n/a`, and the *Per-session
  coverage* table carries its `Streams without usage` count so that `n/a` is
  explained by the logs rather than being an unexplained gap.
- When the client did not ask for usage itself, the usage-only chunk is not
  forwarded, so the client-visible SSE stream is unchanged. Buffered
  (non-streaming) local replies log the same terminal line with an extra
  `stream=false` marker; it carries measured tokens but is **not** paired with
  a `Stream started` (there is none), so it never perturbs busy/decode/first-byte
  statistics.

**Cold-cache cap semantics.** Two independent limits guard local routing
(`_should_skip_local` in `proxy/provider.py`):

- the **cold cap** (`local_large_context_cold_cache_threshold`, 38K by default)
  bounds the **new (uncached) tokens** — `new_tokens = estimated_tokens ×
  (1 − cached_ratio)`. With a measured previous turn, `cached_ratio` reflects
  the reusable prefix, so a 40K-total turn whose only ~10K tokens are new is
  still admitted;
- the **warm cap** (`local_large_context_warm_cache_threshold`) bounds **total
  context** regardless of cache warmth, and the per-entry
  `max_context_tokens` can only tighten it.

Measured usage makes the first limit honest: a turn whose measurement is
missing is priced as fully cold (`cached_ratio = 0.0`), which is what pushed
large-context sessions permanently onto remote providers.

### GPU busy time from power (Prometheus)

The **Local model utilization** table's primary busy figures are derived from
proxy logs (the dispatch → finish slot union). Alongside them, when Prometheus
is reachable, the report adds a **GPU busy — power-based (hardware)** row and
exposes the same data under `--json`'s `gpu_busy` key. This is the
**authoritative hardware-side** busy fraction and does **not** depend on the
proxy's stream-lifecycle logging — but it is **not** the local model
utilisation figure: decisions use the log-derived `local_busy` (see
[Which busy figure is authoritative for decisions](#which-busy-figure-is-authoritative-for-decisions)).

Why power rather than `gpu_busy_percent`: on this Strix Halo APU that counter
**saturates at ~20 W** — it responds to load within seconds but reads ~100 % for
any draw above ~20 W, so it reports idle-with-context stretches as busy
(LP-0MS9GA6QL007XTBP; measured 2026-10-28 over 7 days: 94.5 % mean busy against
28.1 % of samples at compute power, 45.2 % false-busy; LP-0MVSQQO910077LYL). It
is not a utilisation signal. AMD's own tooling is not an alternative either:
`amd-smi` reports `N/A` for usage/power/clocks/energy on this APU, and
`amdtop`/`rocm-smi` display the very same saturating counter (see
[monitoring/README.md](../../../monitoring/README.md), *Why not `amd-smi`,
`rocm-smi` or `amdtop`?*). Power draw discriminates
cleanly (idle ~10–15 W, computing ≥ 90 W), and the `gpu:computing:bool`
recording rule (`gpu:power_watts:amdgpu >= bool 90`) is evaluated by Prometheus
every **15 s** from the node_exporter hwmon series. The report queries
`avg_over_time(gpu:computing:bool[Nh])` for the window, so the busy fraction is
computed over the raw 15 s samples (see
[monitoring/README.md](../../../monitoring/README.md) for the full pipeline).

- The 15 s cadence is comfortably finer than the workload timescales (median
decode 2.4 s, worst prefill 108 s), so sub-15-second sampling would add no
accuracy.
- The row is **best-effort**: if Prometheus is unreachable the query returns
  `None`, the row is omitted from the report, and `--json`'s `gpu_busy` is
  `null`. A report run never fails because monitoring is down.
- The `gpu_busy` block carries `computing_fraction` (0–1), `busy_pct`,
  `busy_seconds`, `idle_seconds`, `total_seconds`, and `smoothed_power` — the
  `gpu:power_watts:avg5m` trace at 1-minute resolution, derived from the 15 s
  raw series by a 5-minute rolling average, for warm-up/cooldown context.

#### Which busy figure is authoritative for decisions

**Decisions use the log-derived `local_busy` figure** — the
`Busy time — upper bound (dispatch → finish)` row and `--json`'s `local_busy`
key (the dispatch → finish slot union defined under
[How it works](#how-it-works) step 4). Gates, alerts and threshold
comparisons quote *that* number: it is the only figure measuring
**proxy-served** local work (slot occupancy, queued time included), so it is
what "how busy was the local model?" means when deciding whether local
capacity is contended (e.g. the `> 60 %` local-busy gate).

**Prefer `gpu_busy` for hardware questions only** — was the GPU computing at
all, is another workload using it, is it power/thermally limited. The row is
**hardware activity, not local model utilisation**: it counts any GPU draw
≥ 90 W (other workloads, requests the proxy never served) and misses
proxy-served work during log gaps, so it must never be presented as the local
utilisation figure, and no gate or alert may quote it as the busy %.

The two figures legitimately diverge by design — see [Why the two figures
diverge](#why-the-two-figures-diverge) for the causes (single source; not
repeated here).

#### Why the two figures diverge

The log union measures *slot occupancy* (including queued time and excluding
any work the proxy did not log); the power figure measures when the
*hardware* actually computed. A persistent gap between them usually means log
gaps, unfinished streams, or GPU work outside the proxy's slot accounting, so
the same window can legitimately report `local_busy` 12.6 % next to a
power-based 39.3 % — and only the former answers whether local capacity was
contended.

## Legacy reason normalisation

The parser maps legacy reason spellings to their current names so historical and
current windows report one label (``_normalize_reason`` in
``log_parser.py``):

| Legacy (rotated logs) | Current |
|---|---|
| `warm_cache_bypass` | `context_too_large` |
| `rate_limited` (prefix) | `provider_budget_exhausted` (variant suffixes preserved: `_after_wait`, `_waiters`, `_disconnect`, `_waited`) |

`rate_limited*` was renamed to `provider_budget_exhausted*`
(LP-0MVMBM8YS003Y2JG) because operators and tooling misread it as an upstream
rate-limit rejection; it is the proxy's own pre-flight request-budget skip on a
healthy provider (see [Interpreting the report](#interpreting-the-report)).

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
  - `provider_budget_exhausted` (legacy `rate_limited*` in rotated logs
    predating LP-0MVMBM8YS003Y2JG) → the proxy's own per-provider RPM
    budget (`rate_limit_rpm`) was exhausted before dispatch, so the request
    passed to the next provider. It is *pacing, not an upstream 429* → if
    the volume is unwanted, raise that entry's `rate_limit_rpm` or reorder
    the chain (raising `models.ini` ctx-size does NOT help).
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
  it. This is the **decision figure** — gates and alerts quote it, not the
  power-based `gpu_busy` row (see [Which busy figure is authoritative for
  decisions](#which-busy-figure-is-authoritative-for-decisions)). The **Prefill vs decode** section then splits the *phase* slot time
  into prefill (dispatch → first byte) and decode (first byte → finish) as
  slot-seconds and as a share of combined phase slot time (always 100%
  together, LP-0MVJNDOUU0059MMO) — read it for where slot time goes (prefill
  rarely dominates at 46% local occupancy, but it does on large cold
  prompts), not for window occupancy: phase slot-seconds sum per-stream
  durations and are not comparable with the wall-clock busy %. **Engine
  cross-check** then validates that phase split against llama-server's own
  per-task eval durations (LP-0MVNWULLI006ZYXX). Read a divergence as a
  *direction*, not a defect: **proxy above engine** means the proxy accounted
  for phase time the engine never (fully) performed (queueing, retries,
  cancelled work), while **engine above proxy** means unattributed local
  traffic (unlogged or partially-logged streams, or eval lines whose proxy
  events fell outside the window). Two caveats always apply: (a) both figures
  are sums of durations (slot-seconds), never wall-clock, so neither is
  comparable with `busy_seconds`/`busy_pct`; and (b) llama-server lines carry
  no timestamps, so a window much shorter than the live `llama-server.log`
  compares an in-window proxy figure against the file's whole span and shows
  engine ≫ proxy as a windowing artefact. Absolute percentage deltas above
  `local_engine_cross_check_threshold_pct` (default 25 %, `0` disables) raise a
  warning recommendation. A low busy % with high fallback volume
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
  `tokens=` contribute `0` and are counted in the section's
  **Streams without usage** line, split by provider — a non-zero local entry
  means that stream predates measured local usage or was aborted). Check the
  per-bucket and per-model tables in the **Total tokens** section — all shares
  are of the same grand total, so fast% + cheap% = 100% and model %s sum to
  100%. A large Input / Output disparity can reveal long-prompt sessions
  driving capacity pressure.
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

- **Denial attribution is a timestamp-block join, not a session join
  (LP-0MVGR2OXT0096OGK M1).** `local_dispatch_denied` carries no `lease_state`
  and the guard record frequently logs `session=unknown`, so a denial is
  attributed to the nearest `local_admission_guard` `reject` within ±60 s
  (inclusive). Measured on the 2026-10-27 16:16 → 2026-10-28 15:56 window
  (111 denials, 3,558 guard rejects): the nearest guard is 0–20 ms away for the
  majority, ±30 s covers 68.5%, ±60 s 73.9%, ±120 s 82.9%. 60 s is the widest
  bound that stays inside one contention burst; the remainder is reported as
  `unattributed` rather than borrowed from a stale earlier burst.
- **Prefill ETA is a bound, compared to the served remainder at matched
  quantiles (LP-0MVGR2OXT0096OGK M3, rescoped by LP-0MVSD5RJW008F1IB).**
  llama-server eval-timing lines carry no per-line timestamp (the parser
  assigns one file-mtime timestamp per log file), so a `prefill_eta_seconds`
  poll cannot be paired with the specific `prompt eval` it predicted.
  `prefill_eta_error` therefore compares the p50/p90 of the prediction
  distribution against the p50/p90 of the observed durations. The prediction is
  the cache-cold *reference-prompt* cost while the observation is the *uncached
  remainder* of mostly-warm prompts, so a wide gap is expected by construction
  and is **not evidence of a calibration defect**. The verdict is therefore
  one-sided (`bound_holds` / `bound_violated` / `inconclusive`: does the served
  p90 stay within the published p50 bound?), and the calibration question is
  answered separately by `prefill_eta_error.matched`, which joins the guard's
  priced ETA to the same dispatch's `dispatch_to_first_byte_ms` via `session=`
  and applies the 25 % tolerance. A window with no guard records (inert guard,
  or pre-change logs) reports the matched figure as `inconclusive` with
  `unmatched_dispatches` counted — never as a fabricated match.

- **Engine eval timings are summed, never placed in time
  (LP-0MVNWULLI006ZYXX).** llama-server writes no per-line timestamps
  (`grep -cE "[0-9]{2}:[0-9]{2}:[0-9]{2}" llama-server.log` → 0), so the
  `## Engine cross-check` figures are **sums of per-task durations** and cannot
  be turned into a time series, an hour, or a wall-clock union. Two
  consequences follow: the totals are comparable with the proxy's phase
  **slot-seconds** but never with `busy_seconds`/`busy_pct`; and window
  attribution is per file — a whole log file counts if its last write falls in
  the window, so a window shorter than the live `llama-server.log` yields an
  engine total that covers more than the window (engine ≫ proxy, a windowing
  artefact rather than unattributed traffic). Deriving wall-clock GPU busy time
  from these logs is impossible by construction and is out of scope (the
  power-based `gpu_busy` telemetry covers hardware utilisation).
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
  its own inherited an unrelated neighbour's `provider_budget_exhausted` within the 60s
  window (11 of 26 clean local-using sessions in the 2026-10-23/24 window were
  mislabelled that way).
- **Per-turn attribution is exact where `turn=` ids exist
  (LP-0MVMBM467005BT4A).** Since the field was introduced, every structured
  line a request emits (routing skips, context skips, fallbacks, routing hops,
  dispatch denials, `[local]`/`[remote]` POST, `Stream started`,
  `Stream finished`, TTFT, error lines) carries the request's 8-hex id, so
  TTFT→stream, start→finish, and skip/fallback→stream joins are made by id and
  cannot mix up two interleaved turns of one session. Log lines written before
  the field existed carry no id, so windows spanning the change fall back to the
  FIFO/nearest-line heuristics for those lines — the report states the pairing
  method in its busy-time note. A turn that logged no id (e.g. a task detached
  from the request context) is likewise paired by the legacy rules.
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
- **Local-turn coverage (`## Local model usage by session`,
  LP-0MVM93DE6009JGBU).** `% local turns` uses the *routed* turn counts
  (local + remote `Stream started` events inside the window), so a session
  whose local turn started before the window and finished inside it is not
  counted as a local turn. An unanswered local turn (an errored
  `Stream finished`, an aborted request with no terminal line) is counted and
  flagged rather than dropped, so a session can read, e.g.,
  `2 (1 unanswered)`. Context size at fallback is the session's own evidence
  only — its `routing_skip_local estimated_tokens`, else the first remote
  stream's prompt tokens, else its `start_context_size` / `max_context_size`;
  a session whose logs carry no token evidence renders `n/a` with a named
  limitation (3 of 26 local-using sessions in the 2026-10-23/24 reference
  window: `empty_response` / `fallback_after_queue` sessions). The percentile
  summary's context column is computed over the sessions that have a value,
  so its sample count can be smaller than the covered-session count — the
  section states both.
- **`provider_budget_exhausted` is budget pacing, not an upstream 429.** The
  reason comes from `provider_budget.py`'s own per-provider RPM accounting:
  the configured `rate_limit_rpm` window for that provider entry is exhausted,
  so the request passes to the next provider immediately. Upstream HTTP 429
  responses are a different signal and surface as `upstream_http_error
  status=429` events, not as `provider_budget_exhausted` fallbacks. The name
  was renamed from `rate_limited` (LP-0MVMBM8YS003Y2JG) precisely so the two
  can never be conflated; rotated logs carrying the pre-rename spelling are
  normalized to the new one by the parser (see the
  [Legacy reason normalisation](#legacy-reason-normalisation) section), so a window spanning
  the rename still reports one row.
