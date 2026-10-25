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
| `--json` | off | Print a machine-readable JSON summary instead of the text summary (includes `local_busy` — with `definition`, `decode_busy_seconds`, `prefill_busy_seconds` and the labelled bounds — `local_prefill`, and `local_busy.prefill_vs_decode`) |
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
