---
name: proxy-usage-analysis
description: "Generate the proxy usage report from the last 24 hours of llama-proxy logs via the deterministic CLI at .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py. The daily report is produced unattended by the 05:00 cron; this skill documents the same CLI for on-demand runs. Trigger on user queries such as: 'analyze proxy usage', 'proxy usage report', 'why is the proxy falling back so much', 'proxy fallback analysis', 'local model utilization', 'analyze proxy errors', 'error taxonomy', 'why is the proxy erroring so much', 'generate the daily proxy report'."
---

# Proxy Usage Analysis

Turn the last 24 hours of llama-proxy session and fallback activity into a
per-session CSV record (split into fast / cheap buckets — from the mode
timeline when the logs carry it, otherwise from the configured
`server.priority_window`, otherwise the slot schedule) and an operator-facing
Markdown report with highlighted, data-backed recommendations — including an
**error taxonomy**, a deterministic **root-cause breakdown** for any proxy
errors observed in the window, and a **provider context-skips** breakdown for
the per-provider context-ceiling skips (`routing_skip_provider_context`).

The report also carries a **wasted GPU compute** surface — a summary table
near the start, a detail section and an appendix — so the volume of GPU
compute burned on work no client received, and its cause, are tracked
rather than hand-scanned. See [Wasted GPU
compute](#wasted-gpu-compute-lp-0mw2sqnhj0044pv6).

The report also opens with a **performance metrics scorecard**: the ten
metrics the proxy is held to, each with its best recorded baseline, an
evidence-based target, this window's actual value and a direction-aware
result, followed by a per-metric detail block (importance, calculation,
levers, trade-offs). See [Performance metrics
scorecard](#performance-metrics-scorecard-lp-0mw0fu5z6006a4p3).

The pipeline is fully deterministic: every report section is produced by
rule-based Python, not an LLM. The skill is a thin, CLI-focused wrapper
around the pipeline. The daily report is produced unattended by the 05:00
cron, and the same CLI is invoked for on-demand runs.

**No behavioural change from the per-turn correlation id.** Since
LP-0MVMBM467005BT4A the proxy appends `turn=<8 hex>` to every structured line
a request emits, and the parser/aggregation use it to join a turn's lines
exactly. The report keeps its sections, columns and JSON keys; only the
pairing *method* improves (exact where ids exist, unchanged FIFO/nearest-line
heuristics for pre-change log lines). Nothing in the CLI, the flags or the
daily cron path changes — see
[REFERENCE.md](REFERENCE.md#how-it-works) for the field contract.

## Invocation

Run the CLI non-interactively (the canonical entry point):

```bash
python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
    --log-dir /var/log/llama-proxy \
    --hours 24
```

Outputs go to `~/proxy-usage-reports/` by default (override with
`--output-dir`). On demand, `/skill:proxy-usage-analysis` invokes this same
CLI.

Options:

| Flag | Default | Purpose |
|---|---|---|
| `--log-dir` | `/var/log/llama-proxy` | Directory containing `proxy.log*` |
| `--llama-log-dir` | `/var/log/llama-proxy` | Directory containing `llama-server.log*` for decode/prompt-eval speed stats (falls back to `--log-dir` when omitted) |
| `--hours` | `24` | Analysis window length |
| `--start` / `--end` | — | Explicit ISO window (`YYYY-MM-DD HH:MM:SS`); overrides `--hours` |
| `--output-dir` | `~/proxy-usage-reports` | Where the CSVs and report are written |
| `--config` | auto-discovered | Path to `proxy/config.yaml` |
| `--stale-rate-target` | `0.05` | Target share of status polls allowed to report `slots_stale=true`; drives the **Local capacity signals** section's met/unmet outcome and `--json`'s `capacity_signals.slots_stale.passed`. Mirrors `scripts/capacity_signal_verification.py --stale-rate-target` |
| `--json` | off | Print a machine-readable JSON summary instead of the text summary (includes `local_busy` — the **decision figure**, with `definition`, `decode_busy_seconds`, `prefill_busy_seconds` and the labelled bounds — `gpu_busy` — hardware-only power data (`computing_fraction`, `busy_pct`, `idle_pct`, `unknown_pct`, the three `shares_pct` values, a `coverage` sample block and `smoothed_power`), **not** local model utilisation — `local_prefill`, `local_busy.prefill_vs_decode`, and `engine_cross_check` — proxy slot-time vs llama-server eval timings — plus `performance_metrics` — the scorecard block keyed by metric id and its `performance_metrics_counts` totals, and `wasted_compute` — the wasted-GPU-compute block, `null` when the analysis was not produced) |
| `--quiet` | off | Suppress the stdout summary (including the corrected `busy (dispatch → finish)` line); use `--json` for the same figures) |
| `--metrics-config` | `metrics/performance_metrics.yaml` | Path to the committed performance-metric defaults (metric detail copy and any seeded history); read-only at runtime, never written |
| `--metrics-state` | `<output-dir>/performance_metrics_state.yaml` | Path to the live performance-metric history file this run records its window in |
| `--no-metrics-update` | off | Render the performance metrics section but do not record this window (no state write) |
| `--seed-metrics-history [DIR]` | `~/proxy-usage-reports` | One-off backfill: seed the best-3 history from archived `report.md` files and exit without analysing logs |
| `--no-discord` | off | Do not post the report (or a failure notice) to the Discord webhook; the report notes `disabled (--no-discord)` |
| `--discord-webhook-url` | — | Discord webhook URL, overriding `discord.webhook_url` in the worklog config. A credential: never written to the report, `--json` or the logs |
| `--discord-config` | auto-discovered `.worklog/config.yaml` | Path to the worklog config holding `discord.webhook_url` |

## Daily cadence (05:00 cron)

A cron job runs the report automatically every day at 05:00 (output logged
to `~/proxy-usage-reports/cron.log` so each run's summary and any failures
are visible). This is the sole daily trigger — no agent or LLM is in the
loop:

```cron
0 5 * * * cd /home/rgardler/projects/llm && python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py >> ~/proxy-usage-reports/cron.log 2>&1
```

That line is unchanged by delivery (LP-0MW2BD6G8001E960): the report is
posted to Discord by default, so no new flag is needed and a failure is
announced on the same channel.

Each run archives the previous day's outputs into a dated subdirectory (see
[Archival](REFERENCE.md#archival)), so a historical daily report accumulates
under `~/proxy-usage-reports/YYYY-MM-DD/`. `cron.log` is not an analysis
artifact and stays at the root, untouched by archival.

## Discord delivery (LP-0MW2BD6G8001E960)

The report is pushed to the operator instead of being parked on disk. Three
outcomes, all recorded in the report's own `- Discord delivery:` header
bullet so the artefact says how it travelled:

| Situation | What happens | The report says |
|---|---|---|
| Webhook configured | `report.md` is posted as a multipart **file attachment** (`proxy-usage-report-YYYY-MM-DD.md`) with a short body — window, headline figures and the top three recommendations by severity | `report sent as an attachment (…)` |
| No webhook configured | Nothing is sent | `not configured — no \`discord.webhook_url\` in <path searched>; report not sent to Discord` |
| Send failed | Nothing further is attempted (delivery is fail-soft; the analysis result is never lost) | `failed — <reason>` |
| `--no-discord` | Nothing is sent | `disabled (--no-discord)` |

**Where the credential comes from.** `--discord-webhook-url`, else
`discord.webhook_url` in `.worklog/config.yaml` — auto-discovered by walking
up from the working directory and then from the repository root derived from
the script path, so a run from any directory finds the project config. The
URL is a credential and is never written to the report, the `--json` summary,
`cron.log` or an error message; only its *source* (the config path) is named.

**A failed run announces itself.** Any failure in `main()` — an unexpected
exception, an invalid window, an unreadable log directory — posts
`⚠️ proxy-usage-analysis failed on <host>: <reason>` (with the window and the
cron log path) before the original error text and exit code are preserved.
The notifier is fail-soft: a dead webhook never masks the real failure. So
silence on the channel means "ran and was fine", never "crashed at 05:00".

**Tests never post.** The skill's `tests/conftest.py` replaces the network
seam for the whole suite, so a test can only reach Discord by injecting its
own `poster`; delivery behaviour is covered in
`tests/test_discord_notify.py`.

## Outputs

Written to `--output-dir` (default `~/proxy-usage-reports`): `report.md`
(primary deliverable), `fast_sessions.csv`, `cheap_sessions.csv`,
`errors.csv`, `errors.json`, and the live performance-metric history
`performance_metrics_state.yaml`. On every non-`--quiet` run the CLI summary
prints the complete absolute path to each artifact, starting with the
report. `report.md` also carries the **Provider context skips** section
(count, share of all skip events, standard/low split, provider breakdown and
the `limit` ceiling each provider hit) whenever the window contains
`routing_skip_provider_context` lines; the `--json` summary exposes the same
figures as `provider_context_skips`. The report's header carries one
`- Discord delivery:` bullet describing how it was delivered, and `--json`
exposes the same outcome as `discord_delivery` (`configured`, `sent`, `note`,
`error`, `report_path`, `attachment_name`, `webhook_source`).

See [REFERENCE.md](REFERENCE.md#outputs) for the full schema of each
artifact and the archival behaviour.

## Performance metrics scorecard (LP-0MW0FU5Z6006A4P3)

The report opens with `## Performance metrics summary`: the ten metrics the
proxy is held to, one row each, with the columns **Metric | Baseline |
Target | Tolerance | Actual | Result**. It is the report's first section after
the header bullets — before `## Performance metric details`, `## Summary by
hour` and every data section — so a stakeholder who reads only this table
still learns whether the proxy improved in this window.

| Column | Meaning |
|---|---|
| Metric | The registry label; row order is the registry order, which never changes between runs |
| Baseline | The single best recorded value and the date it was measured |
| Target | The mean of the three best recorded measurements (direction-aware) |
| Tolerance | The relative band around the target the verdict was judged against — `±10 %` by default, overridable per metric |
| Actual | This window's measurement, `n/a` when it cannot be measured |
| Result | `Met` (within the tolerance band of the target), `Unmet`, or `Insufficient history` below three measurements |

The rules behind those numbers:

- **Direction.** Each metric is registered as higher-is-better or
  lower-is-better, and every comparison respects it — a decoding speed of
  33.1 tok/s against a 32.4 tok/s target is `Met`, a 9.7 denials/h against
  9.1 denials/h is `Unmet`.
- **Retention.** The three best *windows* are retained behind each target,
  one measurement per window, so a re-run of the same day replaces that day's
  value instead of appending a duplicate.
- **Target.** The mean of those three retained measurements; below three the
  target is `n/a`.
- **Verdict.** The actual is compared with the target inside the metric's
  **tolerance band**: a metric is `Met` when the actual is within the
  tolerance band of the target, and any actual better than the target is also
  `Met`. The band is direction-aware and relative to the target —
  higher-is-better needs `actual >= target × (1 − tolerance)`,
  lower-is-better needs `actual <= target × (1 + tolerance)` — and both edges
  count as met, so ordinary run-to-run noise (a decode median 0.3 tok/s under
  target) is not reported as a regression. Fewer than three measurements
  reports **Insufficient history** — never a false Met/Unmet — and a window
  with no measurable actual reports `n/a` with `Insufficient history`, never
  `0`.
- **Tolerance.** The band defaults to **±10 %** of the target for every
  metric and is rendered in the `Tolerance` column, so a reader can see what
  produced each verdict. A metric whose own noise needs a wider or tighter
  band sets `tolerance_pct` in the committed defaults file (beside a one-line
  rationale for any value other than 10); an absent, non-numeric or negative
  value falls back to the 10 % default and never fails the run. A **zero
  target** has a zero-width band — 10 % of zero is zero — so only an actual of
  zero is `Met` there; this is stated rather than replaced by an invented
  absolute epsilon.
- **Configuration era.** Every measurement records the slot geometry it was
  taken under (`slots=<n>,ctx=<per-slot context>`). When a metric's retained
  best three span more than one era the section prints a **cross-era note**
  naming the metric and the eras, because such a target averages unlike
  configurations.

`## Performance metric details` follows the table immediately: one `###`
subsection per metric, headed by the metric id and label (for example
`### decode_speed — Decode speed`), stating why the metric matters, how it is
calculated and from which data, the levers available to change it and the
trade-offs of trying to improve it. The prose comes from the committed
defaults file rather than the renderer, so a wording change is a defaults-file
change; each subsection links to the data section reporting its figures
instead of repeating those tables.

Where the numbers live:

- `.pi/skills/proxy-usage-analysis/metrics/performance_metrics.yaml` — the
  **committed defaults**: empty history plus each metric's detail copy and
  optional `tolerance_pct` band. **Read-only at runtime**; no run ever writes
  it.
- `<output-dir>/performance_metrics_state.yaml` — the **live state**: every
  recorded measurement, written once per run while the report is built.

Four flags relocate those files, suppress the write, or seed the history from
the archive:

| Flag | Default | Purpose |
|---|---|---|
| `--metrics-config PATH` | `metrics/performance_metrics.yaml` | Committed defaults to read (detail copy and any seeded history); never written |
| `--metrics-state PATH` | `<output-dir>/performance_metrics_state.yaml` | Live history file this run records its window in |
| `--no-metrics-update` | off | Render the scorecard without recording this window (no state write) |
| `--seed-metrics-history [DIR]` | `~/proxy-usage-reports` | One-off backfill: seed the best-3 history from the archived `report.md` files and exit without analysing logs |

**The existing 05:00 cron invocation is unchanged.** No metrics flag is
needed: a default run reads the committed defaults, writes
`<output-dir>/performance_metrics_state.yaml` and renders the scorecard, so
the cron line under [Daily cadence](#daily-cadence-0500-cron) stays exactly
as it is. `--json` carries the same figures as a `performance_metrics` block
keyed by metric id (plus `performance_metrics_counts`), so an audit reads
`baseline` / `target` / `tolerance_pct` / `actual` / `verdict` without parsing
Markdown.

Metric definitions, the two file schemas and a worked backfill run:
[REFERENCE.md § Performance metrics scorecard](REFERENCE.md#performance-metrics-scorecard-lp-0mw0fu5z6006a4p3).

## GPU busy time from power (Prometheus)

When Prometheus is reachable, the **Local model utilization** section adds a
**GPU busy — power-based (hardware)** row that measures when the GPU actually
drew compute-level power (≥ 90 W), instead of relying on the proxy's
stream-lifecycle logs. On this Strix Halo APU `gpu_busy_percent` is a two-state
detector whose step sits at **~30 W** — below the idle-with-display band, so it
reads 99.2 % while idling with a display against 98.7 % while computing and
cannot separate the two — and `amd-smi` reports `N/A`,
so power draw is the discriminating signal; `amdtop`/`rocm-smi` show the same
counter (see [monitoring/README.md](../../../monitoring/README.md),
*Why not `amd-smi`, `rocm-smi` or `amdtop`?*, for the measurement and its
PromQL recipe; re-measured 2026-11-06 in LP-0MW5K7WKZ005VJXA). The figure is
computed over the raw **15-second** hwmon series (`monitoring/gpu_compute_rules.yaml`)
by classifying **every expected 15 s instant** across the window as
`computing`, `idle` or `unknown`: an instant is observed only when a sentinel
proves the sampling pipeline was alive there (`up{job="llama-proxy"}`,
`up{job="node-exporter"}` or a fresh `llama_slots_sample_timestamp_seconds`
heartbeat) and the rule has a sample. Absence is never counted as `idle` and
never added to the numerator, so a monitoring gap cannot inflate the figure;
the `computing` + `idle` + `unknown` shares cover the window exactly and sum
to 100 %. A window with no observed instants at all keeps the best-effort
"no row" behaviour rather than reporting a fabricated 0 %. The report also
adds the 5-minute-smoothed power trace for context. Best-effort: if Prometheus
is down the row is omitted and `--json`'s `gpu_busy` is `null` — the report
never fails. Full pipeline and rationale:
[monitoring/README.md](../../../monitoring/README.md) and
[REFERENCE.md](REFERENCE.md#gpu-busy-time-from-power-prometheus).

**This row is hardware activity, not local model utilisation.** It counts any
GPU consumer (`gpu:computing:bool`) — including work the proxy never served —
and misses proxy-served work during log gaps, so it must never be read (or
quoted) as the local model utilisation figure. **Decisions use the log-derived
`local_busy`** (the `dispatch → finish` slot union: the report's `Busy time —
upper bound (dispatch → finish)` row and `--json`'s `local_busy` key); quote
`gpu_busy` only for hardware questions (was the GPU computing at all,
thermals, power draw). The decision rule, the preference rule and the causes
of the gap:
[REFERENCE.md § Which busy figure is authoritative for decisions](REFERENCE.md#which-busy-figure-is-authoritative-for-decisions).

**The one reported GPU-activity measure.**
[monitoring/README.md](../../../monitoring/README.md#one-reported-gpu-activity-measure)
defines exactly one reported GPU-activity figure: the attributable
`gpu:computing_workload:bool` share produced **only** by
`scripts/gpu_utilisation.py`. `gpu_busy` here is a **scoped derivation** of
that doctrine — the power-only `gpu:computing:bool` share (any GPU consumer) —
and `llama_slots_busy` is the rule's input; `local_busy` is local
capacity/contention (a proxy-accounting question, the figure gates quote). The
`--json` key scope, one sentence each:

- `local_busy` — the log-derived `dispatch → finish` slot-union share measuring local capacity/contention (the figure gates and alerts quote), not a GPU measurement.
- `gpu_busy` — the power-derived `gpu:computing:bool` share measuring hardware activity for any GPU consumer, not local model utilisation.

## Engine cross-check (slot-time vs llama-server eval timings)

The **Engine cross-check** section (LP-0MVNWULLI006ZYXX) independently
validates the report's most consequential local figures: the proxy-inferred
prefill/decode slot-seconds from **Prefill vs decode** are compared against
llama-server's own per-task eval timings — summed `prompt eval time` seconds
vs proxy prefill slot-seconds, and summed `eval time` seconds vs proxy decode
slot-seconds — each with the absolute delta and the delta as a percentage of
the engine figure. Engine seconds come from the same Qwen3-port attributed,
in-window eval lines that feed **Decode speed** / **Prompt eval speed**, so the
port-attribution rules (spawn-line port, `mxbai-embed` excluded, ambiguous
multi-port files skipped) are reused, never reimplemented.

**Metric types (read this before comparing anything).** Both figures are
**slot-seconds** — sums of durations, i.e. total phase work, not wall-clock.
They are therefore *not* comparable with the wall-clock `busy_seconds` /
`busy_pct` union in **Local model utilization**, and no arithmetic in the
report mixes the two. Engine lines carry no timestamps, so the engine total
can only be summed; the section also states that a window much shorter than
the live `llama-server.log` compares an in-window proxy figure against the
file's whole span (engine ≫ proxy is then a windowing artefact).

**Interpreting a divergence.** A non-zero delta is expected — the proxy counts
per-stream phases plus its own queueing, retries and cancellations, the engine
counts per-task eval calls. Direction is the signal:

- **proxy above engine** — queueing, retries or cancelled work: phase time the
  proxy accounted for that the engine never (fully) performed;
- **engine above proxy** — unattributed local traffic (unlogged or
  partially-logged streams, or eval lines whose proxy events fell outside the
  window).

**Threshold.** A phase whose **absolute** percentage delta exceeds
`local_engine_cross_check_threshold_pct` (in `proxy/config.yaml`, default
**25.0 %**, `0` disables) raises a warning recommendation — `medium`, escalated
to `high` at twice the threshold. The default exists because the structural
per-stream vs per-task skew is normal; only a large divergence signals an
accounting problem. Fractional values are supported (the loader coerces the
scalar as a float).

**Honest degradation.** When the llama logs are absent, unreadable, or carry no
Qwen3-attributed eval lines for the window, the section renders an explicit
`Unavailable — <phase>: <reason>` note (reusing the parser's named
`verbosity/build` / `rotation` / `attribution failure` diagnostics) with `-`
deltas — never a misleading `0` or `100 %`. `--json` exposes the same
comparison additively under `engine_cross_check`.

## Post-acceptance evictions (alert threshold + triage rule)

`local_eviction_after_acceptance` (parent AC7, LP-0MVN4LCQJ0034S5S) is an
**ERROR** the proxy emits when an *accepted* local dispatch (admitted, dispatch
record created) is released **before producing a first byte** — cancelled by
the prefill deadline, the no-progress watchdog, an idle timeout, the orphan
sweep, a lease reclaim or a slot-assignment free.

The report prints the count with a per-reason breakdown, and the `--json`
summary exposes it as `local_evictions_after_acceptance` (total) plus
`local_evictions_after_acceptance_by_reason` (all six reasons are
addressable). Both keys are additive; no existing key changed.

**Alert threshold: `> 0`.** For a prompt inside the configured cold-prefill
budget, *any* such event is an alert — not a trend to watch. It means the
budget/pricing model admitted a dispatch the GPU could not serve, which is
exactly the failure the budget exists to prevent (baseline: 125 deadline
cancellations and ~779k discarded prefill tokens in the 2026-10-24 18:38–23:47
window). The rung trade-off that deliberately accepts some of these is
quantified below — the threshold itself does not change.

**Triage rule.** Any `local_eviction_after_acceptance` for a within-budget
prompt is treated as a **defect** and filed as a **critical `test-failure`-class
work item** through the triage path:

```bash
python3 $(skill_path triage)/scripts/check_or_create.py \
  '{"test_name": "local_eviction_after_acceptance:<reason>",
    "stdout_excerpt": "<the ERROR line>",
    "stack_trace": "", "parent_work_item_id": "<current work item>"}'
```

Do not dismiss the event as churn: a lease expiring *after* a served response
is normal and is not reported (classification is by the dispatch record's
`first_byte_seen` flag), so everything that does appear here is unserved work.
If the events cluster in one prompt band, the band's calibrated rate is wrong —
re-measure before changing the budget.

**Attributing a `prefill_deadline` spike (LP-0MVZ5WV6W008UO5F).** The count
alone cannot answer triage's question — *which admission path let this
dispatch start?* — because the free-gate rung is only one of several answers.
Run the attribution CLI over the alert window before filing, and record its
class split on the work item:

```bash
python3 .pi/skills/proxy-usage-analysis/scripts/attribute_prefill_deadline.py \
  --start '<window start YYYY-MM-DD HH:MM:SS>' \
  --until '<window end YYYY-MM-DD HH:MM:SS>' [--json]
```

Each eviction is joined to its admission evidence at the measured dispatch
start (`eviction_ts - elapsed_s`) and classified (`--json` carries the same
summary plus per-event rows):

| Class | Meaning | Action |
|---|---|---|
| `rung_accepted` | `local_admission_guard decision=admit` priced inside the deadline-relative bound but above the queue budget — the free-gate rung's documented admitted-and-doomed trade | compare the rate against the documented accepted level (26 / 24 h, ~80k priced tokens, measured 2026-11-01→02); no fix unless the rate grows |
| `priced_within_queue_budget` | admitted inside the 30 s queue budget yet no first byte — the slot was held by another session's prefill (occupancy priced ~0) | occupancy-observability defect — file a `discovered-from` bug (measured 8 / 24 h) |
| `high_priority_inert` | `priority=high` was *genuinely* inert: a logged `no_budget` decision with every time bound disabled, so nothing was priced | **fixed** (LP-0MVZF3JZ9009LTN2; attribution corrected under LP-0MVZQUT3A0039MLY): `high` is now bounded by the deadline and safety budget like every other priority. An **unlogged** high-priority eviction is `inert_unlogged`; only a genuine logged `no_budget` line keeps this class, which must otherwise read 0 |
| `inert_unlogged` | no guard line at the dispatch instant and no override — an inert decision that logged nothing (uncalibrated bucket, or `high` priority with no logged `no_budget` line) | observability defect — file a `discovered-from` bug (measured 47 / 24 h) |
| `idle_override` / `economic_override` / `capacity_override` | a routing override admitted the dispatch without the guard | assess the override's own trade |
| `priced_above_bound` | admitted although the priced wait exceeded the bound it was compared against | impossible by construction — treat as a guard bug |

**Measured accepted level (rung enabled).** For the window
2026-11-01 11:09:07 → 2026-11-02 11:09:07 (the rung
`local_admission_guard_free_gate_deadline_budget: true` was enabled at
2026-11-01 21:27:04Z, LP-0MVXZWF7A002TUO1) the split of 126
`prefill_deadline` evictions was: `rung_accepted` 26 (79,740 priced new
tokens), `high_priority_inert` 43, `inert_unlogged` 47,
`priced_within_queue_budget` 8, `idle_override` 2 — all inside the 25k
cold-prefill budget, 796,993 priced new tokens discarded in total, 7 with a
stamped ETA above the 120 s deadline (6 high priority + 1 inert; none is an
ETA-per-token under-prediction — in each case the pricing, where it ran,
already said "over the deadline"). **The rung's own trade is therefore 26 /
24 h, not 126 / 24 h**: the other 100 belong to admission paths the rung does
not govern, tracked in LP-0MVZF3JZ9009LTN2 (high priority),
LP-0MVZF4SU10032OZH (inert unlogged) and LP-0MVZF50MJ00654O1 (occupancy
priced at 0). A post-rung window returning toward ~26/24 h with the rung on
is the documented accepted behaviour.

**Fix status of those classes.** LP-0MVZF3JZ9009LTN2 landed first (producer
decision: "high-priority are bound in the same way as anything else"), so a
window measured after it should show `high_priority_inert` at **0** — a
non-zero count is a regression, not an expected class. Its attribution was
corrected under LP-0MVZQUT3A0039MLY: an unlogged `priority=high` eviction is
now `inert_unlogged` (it was previously mislabelled `high_priority_inert`),
and `high_priority_inert` survives only for a genuine logged `no_budget`
decision, so the "must read 0" rule is no longer diluted by uncalibrated high
dispatches. LP-0MVZF4SU10032OZH (inert unlogged) and LP-0MVZF50MJ00654O1
(occupancy priced at 0) remain open: those classes still appear until they
land.

**Reading the numbers.** `estimated_tokens` / `new_tokens` / `eta_prefill_s` are
the **admission-time** values the routing decision priced and the dispatch
record stored verbatim (LP-0MVODJ3P6000AIUM): `new_tokens > local_cold_prefill_budget_tokens`
is therefore genuine evidence of an over-budget admission, never a re-pricing
artefact. A record created without an admission snapshot reports `0`/`unknown`
instead of a recomputed value.

## Wasted GPU compute (LP-0MW2SQNHJ0044PV6)

The report answers the operating question behind the eviction count: **how
much GPU compute did this window burn on work nobody received, and why?** The
goal of the surface is to make *minimising wasted GPU compute* a tracked
number rather than a hand-scan of `proxy.log`.

It appears in three places in `report.md`:

| Where | What it holds |
|---|---|
| `## Wasted GPU compute (summary)` — after the scorecard, before the first data section | one row per headline figure with its target and verdict: evictions/h, the user-class composite rate vs its threshold, occupancy and capacity shares, wasted GPU tokens, restart-free status, overall verdict |
| `## Wasted GPU compute` — in the body, after **Prefill admission regression** | per-reason table (total / user / audit / per hour), attribution table with shares, waste table, release→cancel table |
| `## Appendix: wasted GPU compute — definitions and basis` — last section | term definitions, the measured vs legacy abandoned-prefill basis, thresholds and their origin, and the per-window caveats |

The figures, and what each is for:

| Figure | Meaning | Why it matters |
|---|---|---|
| Post-acceptance evictions (total, /h) | every `local_eviction_after_acceptance`, all reasons | the gross volume of accepted-then-unserved work |
| **User-class composite rate** | `prefill_deadline` + `measured_cache_deficit`, non-`audit` sessions, per hour | the conversion-proof gate: `measured_cache_deficit` is an *early abort* of the same defect, so counting it stops a reason conversion looking like an improvement |
| Occupancy share | share of deadline events where another tenant held the slot pool | fairness/head-of-line class; the only one an admission change can fix |
| Capacity share | share attributed to an honest cold price the GPU could not deliver | distinguishes "our pricing is wrong" from "this box is too slow" |
| Wasted GPU tokens | priced prefill discarded at the deadline + tokens evaluated after the backend cancel | the number to drive down; the tail is measured (≤ 1 ubatch by design) |
| Release→cancel (p50/p90) | latency from release to the backend stopping that task | sizes the real abandoned window instead of estimating it from tokens ÷ rate |
| Restart-free | no proxy restart and no backend restart that discarded an in-flight dispatch | a window with a restart cannot certify anything: the restart re-prices capacity and drops prefills |

**Thresholds** are the producer-agreed criterion recorded on
LP-0MVN3KMGI009BZ0D (option b2): ≤ **0.5** user-class events/h, occupancy ≤
**25 %**, capacity < **50 %**, measured over a **restart-free ≥ 24 h** window.
The appendix repeats them next to the window's own caveats, so a figure is
never read as certified when it is not.

**One source of truth.** `scripts/wasted_compute.py` does not re-implement
the attribution rules: it drives the committed analyser
`proxy/scripts/analyse_prefill_deadline_residual.py` (imported by path) and
normalises its payload. If the analyser is absent (a non-source install) or the
analysis fails, the section renders a named reason instead of a number, and
the run still succeeds.

**Honest degradation.** Three inputs can be missing or unreliable, and each
turns into a stated caveat rather than a wrong figure: no `llama-server` log
(the per-release and backend numbers are unavailable), a log whose coverage
verdict is not `exact` (the per-release join is *indicative* — this is the
confusion that produced a wrong "restart-free" claim on 2026-11-04), and a
window containing a restart (`verdict: NOT CERTIFIED`).

**Runtime cost.** The analyser re-reads the same `proxy.log*` files the report
already parsed. Measured on this host: a 6.99 h window (108 k in-window lines,
56 files) completes in well under a minute; a 24 h cron run therefore pays a
few extra minutes on top of its existing tens of seconds. The report stays
deterministic and the extra work never fails the run.

## Contention-prediction measurement (LP-0MVGR2OXT0096OGK)

Since the payload change (per-slot `lease_state` / `phase` /
`prefill_eta_seconds`, top-level `capacity_hint`; LP-0MVGDUVGE0024YAI), the CLI
measures whether those signals predict **real** local contention. The report
renders a **Local contention prediction vs reality** section and `--json`
exposes the same numbers under `denial_attribution`, `prefill_eta_error` and
`status_capacity_hint` whenever the window has a denial or a status poll:

- **M1 (denial attribution).** `local_dispatch_denied` lines carry no
  `lease_state`, so each denial is joined to its decision-block
  `local_admission_guard` `reject` record (nearest within ±60 s, inclusive)
  and the pool state at denial time is derived from the guard's measured
  `busy_slots` / `in_flight_prefills` fields: `idle_leased` (no slot decoding,
  no prefill in flight — the false-busy case), `prefill_bound` (≥1 prefill
  running), `saturated` (slots decoding), `unknown` (legacy lines). Denials
  with no guard in the block are reported `unattributed`, never guessed.
- **M2 (local busy fraction vs denials).** The window's busy fraction and
  per-hour busy% (from `BusyStats`) sit beside the denials, and the
  window's denials-per-hour / busy fraction are compared with the pre-change
  baseline (2,641 denials / 24 h at 29.3% busy).
- **M3 (prefill ETA accuracy).** `prefill_eta_error` answers two questions
  (LP-0MVSD5RJW008F1IB): (a) *did the published cache-cold reference bound
  hold?* — the quantile-matched bound vs served-remainder comparison, with a
  one-sided `bound_holds` / `bound_violated` / `inconclusive` verdict; and
  (b) *was the per-request pricing right?* — the `matched` figure, which joins
  the admission guard's priced ETA to the **same** dispatch's
  `dispatch_to_first_byte_ms` via `session=` and applies the 25 % tolerance
  (`matches_reality` / `over_predicts` / `under_predicts` / `inconclusive`).
  A wide predicted−observed gap in (a) is expected by construction and is not
  evidence of a calibration defect (llama-server lines carry no per-line
  timestamps, so the quantile comparison cannot pair 1:1 — see REFERENCE.md
  Limitations).

## Local capacity signals (LP-0MVU8O4EL005IYP9)

The report renders a **Local capacity signals** section immediately after
**Local contention prediction vs reality**, publishing the `slots_stale`
(stale-poll) rate for the report's own window so polling-path health is
visible nightly without running `scripts/capacity_signal_verification.py` by
hand and cross-checking two surfaces:

- **`slots_stale` — stale-poll rate vs target.** The status-poll count
  (`status_request` lines carrying `capacity_hint=`), the polls with
  `slots_stale=true`, the rate, the configured target (default < 5 %) and an
  explicit **met** / **unmet** outcome. The denominator is the polls that
  carry the `slots_stale` field at all: a schema-1 poll has no freshness field
  and is **excluded from the rate**, never counted as fresh (the excluded
  count is stated in the section). `slots_stale` means the served slot counts
  *or* the per-slot detail came from the last-known cache because the fresh
  `/slots` query failed — a **polling-path** signal, not a capacity verdict,
  and only comparable between nights measured on the same build.
- **Configurable target.** `--stale-rate-target` (default `0.05`) overrides
  it for both the rendered outcome and the `--json` verdict, so the cron line
  can carry a new target without a code change.
- **`capacity_hint` distribution and `saturated`-without-live-work.** Every
  observed hint token (and an unexpected one — a new claim name, or `unknown`
  for an unreadable value) is listed with its count **and** share of status
  polls; nothing is dropped or re-bucketed, so a vocabulary change is visible
  the night it lands. The violations table reports `capacity_hint=saturated`
  polls with **no** live local work, with both shares (of all polls and of
  `saturated` polls). Live work is the shared rule: `local_in_flight_requests
  > 0` (schema 3, authoritative) or the `local_active_query` /
  `active_query` flags (schema-2 fallback). A one-line verdict names each
  missed target explicitly.
- **Coverage caveat.** The section states the span of the polls it measured;
  when the retained logs do not reach both edges of the window it labels the
  figures a **partial** measurement (`--json`:
  `capacity_signals.coverage`) instead of letting a truncated retention
  window read as a complete night.
- **Shared derivation.** The status-poll detection, field extraction, the
  denominator, the rate and the target check live once, in
  `scripts/lib/capacity_signals.py`, and are imported by both this skill
  (through `scripts/capacity_signals.py`, a documented path-based shim) and
  `scripts/capacity_signal_verification.py` — the two surfaces cannot
  disagree. Status polls are aggregated with the reference tool's half-open
  `[window_start, window_end)` window, so a poll exactly at the window end
  belongs to the next night.
- **Same-process import safety.** The derivation tool imports the live
  `MEASURED_CACHE_DEFICIT_REASON` constant from the proxy package
  (`proxy.measured_cache_abort`) so its wording cannot drift from the proxy's
  own reason string. The repo's `proxy/` directory is a PEP 420 namespace dir
  without an `__init__.py`, so `aggregation.compaction_impact`'s best-effort
  `from proxy.provider import ...` can leave a half-imported `proxy` entry in
  `sys.modules` when it (expectedly) fails; a later same-process import of the
  real package would then resolve against that stale cache. The aggregation
  drops the broken entry on failure and the tool refreshes the cache before
  importing, so the two never collide under the full test suite's single
  process (LP-0MVYNLWWL009R49W).
- **Standalone equivalent.** For an ad-hoc window — or to cross-check a night
  — run the same derivation without the rest of the report:
  `python3 scripts/capacity_signal_verification.py --since '<start>' --until
  '<end>' --window-closed --json`. Its docstring ("Derivation of every emitted
  figure") and the module docstring of `scripts/lib/capacity_signals.py` are
  the **source of truth** for the wording used above; the two must never
  describe the same rule differently.
- **Unattended safety.** The section is **omitted** (no heading, no zero-fill)
  when the window has no status polls, so an empty window — or a log directory
  with no logs at all — renders a clean report and the 05:00 cron exits 0. In
  that case `--json` reports `capacity_signals.slots_stale.passed: null`
  rather than a misleading `true`. The capacity-signal work added **no
  network call, no LLM call and no interactive prompt** to the pipeline: the
  figures are derived from the same log lines the report already parses, and
  the optional Prometheus GPU row keeps its pre-existing best-effort
  behaviour (fail-soft, never a cron failure).

### Overriding the stale-rate target on the cron line

`--stale-rate-target` is the only knob: it changes the rendered met/unmet
outcome **and** `--json`'s `capacity_signals.slots_stale.passed` /
`capacity_signals.targets`, so a new target needs no code change. Put it on the
cron line (the daily job otherwise uses the `0.05` default):

```cron
0 5 * * * cd /home/rgardler/projects/llm && python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py --stale-rate-target 0.10 >> ~/proxy-usage-reports/cron.log 2>&1
```

Equivalently, for a one-off re-run over an arbitrary window, pass the same flag
together with `--start` / `--end`:

```bash
cd /home/rgardler/projects/llm
python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
  --start '2026-10-24 20:00:00' --end '2026-10-25 07:00:00' \
  --output-dir ~/proxy-usage-reports/recheck --stale-rate-target 0.10 --json
```

`--json` exposes the same figures under `capacity_signals.slots_stale`
(`status_polls`, `count`, `denominator`, `rate`, `target`, `passed`), plus
`capacity_signals.capacity_hint` (`status_polls`, `counts`, `shares`),
`capacity_signals.saturated_without_live_work` (`saturated`, `count`,
`rate_of_polls`, `rate_of_saturated`), `capacity_signals.targets` (each
check's `name`, `target`, `observed`, `passed`), `capacity_signals.coverage`
(`covered`, the two edge gaps, `max_gap_seconds`) and
`capacity_signals.all_targets_met` (`null` on a window with no polls).

## Local-turn coverage (LP-0MVM93DE6009JGBU)

The report answers "which sessions actually used the local model, how many
turns did it answer, what share of each session's turns were local, and why
did each session move to remote" in a **Local model usage by session** section
rendered from data the pipeline already collects:

- a summary percentile table (P10/P50/P90 + total) over the sessions with ≥1
  local turn: `# turns local`, `# turns remote`, `% local turns` and context
  size at fallback (linear interpolation, the skill's usual convention);
- a fallback-reason counts-and-percent table (a reason is categorical, so it
  has no meaningful percentile);
- a per-session table (session id, turn counts, `% local turns`, fallback
  reason, context size at fallback) sorted by local turns descending — more
  local turns is the goal, so the best sessions read first.

Derivation rules that matter when reading it:

- an unanswered local turn (error / abort / log gap) is counted and flagged
  rather than dropped, never silently discarded;
- `% local turns` is `local / (local + remote)` per session, so a local-only
  session reads 100 %;
- the fallback reason is the session's **own** attribution
  (`no_remote_fallback` when it never used remote, `unattributed` when it has
  no own evidence). `unattributed` means exactly "no own evidence" — it is
  **not** a statement that the session's line arrived late: the own-evidence
  join is turn-first, so an own `Fallback triggered` or routing-skip line
  carrying the first remote stream's `turn=` is attributed however much later
  the proxy logged it (the pre-flight-buffering order — the proxy emits the
  fallback line after `Stream finished` on the streaming-success path), while
  turn-less rotated lines keep the timestamp rule. A reason is never inherited
  from a neighbouring session (LP-0MVMBMSWR0036RES);
- context size at fallback is the session's own
  `routing_skip_local estimated_tokens`, else the first remote stream's prompt
  tokens, else its session context — sessions with no token evidence render
  `n/a` with a named limitation, never a fabricated estimate.

The same figures are in the `--json` summary under `local_turn_coverage` and
in the per-session CSVs.

Interpretation notes the section prints itself when the data applies:
context-size fallbacks are `estimated_tokens` threshold decisions (they fire
only after a prompt grows past the cold/warm thresholds, not on a session's
first message) and `provider_budget_exhausted` is the proxy's own per-provider
RPM budget
pacing (`provider_budget.py`), **not** an upstream HTTP 429 — those surface as
`upstream_http_error status=429` events.

## Context size at fallback by percentile band (LP-0MVZ6EAZR001DX5E)

The report answers "does the dominant reason for falling back change as context
grows, and does the uncached work per turn grow with it?" in a **Context size
at fallback by percentile band** section rendered immediately after *Local
model usage by session* and before *Fallback reasons*:

- the cohort is the neighbouring section's cohort (sessions with ≥1 local turn)
  restricted to those with token evidence at their fallback; the section states
  how many sessions are included and how many are excluded for missing token
  evidence (excluded sessions are counted, never rendered as 0);
- ten **disjoint** decile bands `P0–P10` … `P90–P100` (the report's usual
  linear-interpolation percentile convention; a session exactly on a cut sits in
  the higher band, the lowest value in `P0–P10` and the highest in
  `P90–P100`), each with its session count, average total context size and
  average final-turn new tokens. An empty band renders `-`, never a fabricated
  mean;
- **final-turn new tokens** are the new (uncached) prompt tokens of the
  session's last turn, taken from that turn's admission/routing line: its
  measured `new_tokens=`, else `estimated_tokens= × (1 − cached_ratio=)`. A
  session whose final turn carries no such evidence is excluded from that
  column's mean and tallied as `n/a` — never imputed as 0;
- a **top-two fallback reasons per band** table: each band's two most common
  reasons as a count and a share of that band's sessions (ties break
  alphabetically), plus an explicit `other` remainder so every row reads as
  shares of 100 %.

No LLM call, no network call and no new log field: everything is derived from
lines the pipeline already parses. On a window with no local turns the section
is omitted entirely (the 05:00 cron still exits 0) while `--json` reports
`context_at_fallback_by_percentile` with an empty band list and zero cohort
counts.

## Contended-pricing surface (LP-0MVS9YKHL007UBBZ)

Since the contended-pricing fix (parent LP-0MVS8T6F4006DGG3: contended
admissions priced against a contended rate bucket, uncalibrated contended
window → `rate_bucket=solitary_fallback`), the report surfaces the decision
fields instead of hiding them in raw log archaeology:

- the parser carries `rate_bucket`, `contended_samples`, `eta_occupancy_s` and
  the priced `new_tokens` on `local_admission_guard` events (all additive — a
  pre-change line leaves each `None`);
- the report renders an **Admission verdicts by rate bucket and prompt band**
  section and `--json` exposes the same split as the additive keys
  `local_admission_by_rate_bucket` and `local_admission_by_token_band` (with
  per-band admission → local first-byte conversion counts; buckets/bands with
  no verdict are omitted, and a high `solitary_fallback` share is the signal
  that the contended window still needs samples) — see
  [REFERENCE.md](REFERENCE.md#outputs).

The banded conversion numbers are directly comparable with the
admission-conversion harness (`scripts/prefill_admission_conversion.py`),
which defines the bands both views share; a test pins the two band tables
against drift.

**Quick verification over a synthetic log slice** (a small window containing
all three buckets and one banded admit → local first byte):

```bash
mkdir -p /tmp/rate-bucket-check && cat > /tmp/rate-bucket-check/proxy.log <<'EOF'
2026-10-28 10:00:00,123 - INFO - local_admission_guard decision=reject reason=eta_exceeds_budget enforced=true eta_prefill_s=41.2 eta_occupancy_s=16.0 projected_wait_s=57.2 budget_s=60.0 deadline_s=120.0 deadline_budget_s=114.0 new_tokens=6659 budget_tokens=40000 rate_bucket=contended contended_samples=6 busy_slots=1 in_flight_prefills=1 free_slot=false priority=medium policy=reject model=Qwen3 samples=12 provider=local-qwen3 session=herdr-abc
2026-10-28 10:00:30,123 - INFO - local_admission_guard decision=admit reason=within_budget enforced=true eta_prefill_s=91.8 eta_occupancy_s=0.0 projected_wait_s=91.8 budget_s=60.0 deadline_s=120.0 deadline_budget_s=114.0 new_tokens=25000 budget_tokens=40000 rate_bucket=solitary contended_samples=0 busy_slots=0 in_flight_prefills=0 free_slot=true priority=medium policy=reject model=Qwen3 samples=12 provider=local-qwen3 session=herdr-abc
2026-10-28 10:01:00,123 - INFO - local_admission_guard decision=admit reason=within_budget enforced=true eta_prefill_s=3.1 eta_occupancy_s=2.0 projected_wait_s=5.1 budget_s=60.0 deadline_s=120.0 deadline_budget_s=114.0 new_tokens=12000 budget_tokens=40000 rate_bucket=solitary_fallback contended_samples=0 busy_slots=1 in_flight_prefills=0 free_slot=false priority=medium policy=reject model=Qwen3 samples=12 provider=local-qwen3 session=herdr-def
2026-10-28 10:01:30,123 - INFO - dispatch_first_byte_ms=8123.4 dispatch_to_first_byte_ms=8120.1 cache_state=cold cached_ratio=0.000 queue_wait_ms=3.3 session=herdr-abc model=Qwen3 priority=high estimated_tokens=25000
EOF
python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
    --log-dir /tmp/rate-bucket-check \
    --start "2026-10-28 09:00:00" --end "2026-10-28 12:00:00" \
    --output-dir /tmp/rate-bucket-out --quiet \
&& grep -A12 '## Admission verdicts by rate bucket and prompt band' /tmp/rate-bucket-out/report.md
```

The section should show `contended 0/1`, `solitary 1/0` (converted in `20k-32k`)
and `solitary_fallback 1/0` rows, and the `--json` run returns the same split
under `local_admission_by_rate_bucket` / `local_admission_by_token_band`.

## Further reference

Report interpretation, root-cause taxonomy, implementation notes, and
limitations: [REFERENCE.md](REFERENCE.md). The repeatable check that the
daily path is agent-free is documented in
[REFERENCE.md#verifying-the-daily-path-is-agent-free](REFERENCE.md#verifying-the-daily-path-is-agent-free).

## Testing

Run the full suite via the test skill (canonical, cached pipeline):

```bash
/skill:test
```

The suite covers log-line parsing, session aggregation, fallback attribution,
fast/cheap bucketing (mode timeline → `server.priority_window` → slot
schedule, including window parsing, midnight wrapping and disabling),
recommendation rules, llama-server eval-timing parsing
(decode + prompt eval, Qwen3 port filtering, fast/cheap speed stats), and an
end-to-end run, using fixtures copied from real `/var/log/llama-proxy`
lines (`proxy.log` and `llama-server.log`).

Two invariants have their own dedicated suites:

- `tests/test_no_ctx_size_advice.py` — no generated recommendation and no
document advises raising the local context size (the ceiling is fixed by the
model and the hardware), and the completed ctx-size work item is never cited
as an open action.
- `tests/test_discord_notify.py` — webhook discovery (flag → worklog config →
`--discord-config`), the multipart attachment payload, message truncation, the
report's `- Discord delivery:` bullet in all four states, and the CLI wiring
for both a delivered report and an announced failure. No test performs network
I/O: `tests/conftest.py` replaces `discord_notify._post_multipart` for the
whole suite, and delivery tests inject their own `poster`.
