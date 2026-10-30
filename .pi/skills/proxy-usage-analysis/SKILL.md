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
| `--json` | off | Print a machine-readable JSON summary instead of the text summary (includes `local_busy` — the **decision figure**, with `definition`, `decode_busy_seconds`, `prefill_busy_seconds` and the labelled bounds — `gpu_busy` — hardware-only power data (`computing_fraction`, `busy_pct`, `smoothed_power`), **not** local model utilisation — `local_prefill`, and `local_busy.prefill_vs_decode`) |
| `--quiet` | off | Suppress the stdout summary (including the corrected `busy (dispatch → finish)` line); use `--json` for the same figures) |

## Daily cadence (05:00 cron)

A cron job runs the report automatically every day at 05:00 (output logged
to `~/proxy-usage-reports/cron.log` so each run's summary and any failures
are visible). This is the sole daily trigger — no agent or LLM is in the
loop:

```cron
0 5 * * * cd /home/rgardler/projects/llm && python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py >> ~/proxy-usage-reports/cron.log 2>&1
```

Each run archives the previous day's outputs into a dated subdirectory (see
[Archival](REFERENCE.md#archival)), so a historical daily report accumulates
under `~/proxy-usage-reports/YYYY-MM-DD/`. `cron.log` is not an analysis
artifact and stays at the root, untouched by archival.

## Outputs

Written to `--output-dir` (default `~/proxy-usage-reports`): `report.md`
(primary deliverable), `fast_sessions.csv`, `cheap_sessions.csv`,
`errors.csv`, and `errors.json`. On every non-`--quiet` run the CLI summary
prints the complete absolute path to each artifact, starting with the
report. `report.md` also carries the **Provider context skips** section
(count, share of all skip events, standard/low split, provider breakdown and
the `limit` ceiling each provider hit) whenever the window contains
`routing_skip_provider_context` lines; the `--json` summary exposes the same
figures as `provider_context_skips`.

See [REFERENCE.md](REFERENCE.md#outputs) for the full schema of each
artifact and the archival behaviour.

## GPU busy time from power (Prometheus)

When Prometheus is reachable, the **Local model utilization** section adds a
**GPU busy — power-based (hardware)** row that measures when the GPU actually
drew compute-level power (≥ 90 W), instead of relying on the proxy's
stream-lifecycle logs. On this Strix Halo APU `gpu_busy_percent` saturates at
~20 W (it reads ~100 % for any draw above that) and `amd-smi` reports `N/A`,
so power draw is the discriminating signal; `amdtop`/`rocm-smi` show the same
saturating counter (see [monitoring/README.md](../../../monitoring/README.md),
*Why not `amd-smi`, `rocm-smi` or `amdtop`?*). The report
queries `avg_over_time(gpu:computing:bool[Nh])` — evaluated by Prometheus over
the raw **15-second** hwmon series (`monitoring/gpu_compute_rules.yaml`) —
plus the 5-minute-smoothed power trace for context. Best-effort: if Prometheus
is down the row is omitted and `--json`'s `gpu_busy` is `null` — the report
never fails. Full pipeline and rationale:
[monitoring/README.md](../../../monitoring/README.md) and
[REFERENCE.md](REFERENCE.md#gpu-busy-time-from-power-prometheus).

**This row is hardware activity, not local model utilisation.** It counts any
GPU draw ≥ 90 W — including work the proxy never served — and misses
proxy-served work during log gaps, so it must never be read (or quoted) as the
local model utilisation figure. **Decisions use the log-derived `local_busy`**
(the `dispatch → finish` slot union: the report's `Busy time — upper bound
(dispatch → finish)` row and `--json`'s `local_busy` key); quote `gpu_busy`
only for hardware questions (was the GPU computing at all, thermals, power
draw). The decision rule, the preference rule and the causes of the gap:
[REFERENCE.md § Which busy figure is authoritative for decisions](REFERENCE.md#which-busy-figure-is-authoritative-for-decisions).

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
window).

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

**Reading the numbers.** `estimated_tokens` / `new_tokens` / `eta_prefill_s` are
the **admission-time** values the routing decision priced and the dispatch
record stored verbatim (LP-0MVODJ3P6000AIUM): `new_tokens > local_cold_prefill_budget_tokens`
is therefore genuine evidence of an over-budget admission, never a re-pricing
artefact. A record created without an admission snapshot reports `0`/`unknown`
instead of a recomputed value.

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

Derivation rules that matter when reading it: an unanswered local turn (error
/ abort / log gap) is counted and flagged rather than dropped; `% local turns`
is `local / (local + remote)` per session, so a local-only session reads
100 %; the fallback reason is the session's **own** attribution
(`no_remote_fallback` when it never used remote, `unattributed` when it has no
evidence); and context size at fallback is the session's own
`routing_skip_local estimated_tokens`, else the first remote stream's prompt
tokens, else its session context — sessions with no token evidence render
`n/a` with a named limitation, never a fabricated estimate. The same
figures are in the `--json` summary under `local_turn_coverage` and in the
per-session CSVs.

Interpretation notes the section prints itself when the data applies:
context-size fallbacks are `estimated_tokens` threshold decisions (they fire
only after a prompt grows past the cold/warm thresholds, not on a session's
first message) and `provider_budget_exhausted` is the proxy's own per-provider
RPM budget
pacing (`provider_budget.py`), **not** an upstream HTTP 429 — those surface as
`upstream_http_error status=429` events.

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
