---
name: proxy-usage-analysis
description: "Generate the proxy usage report from the last 24 hours of llama-proxy logs via the deterministic CLI at .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py. The daily report is produced unattended by the 05:00 cron; this skill documents the same CLI for on-demand runs. Trigger on user queries such as: 'analyze proxy usage', 'proxy usage report', 'why is the proxy falling back so much', 'proxy fallback analysis', 'local model utilization', 'analyze proxy errors', 'error taxonomy', 'why is the proxy erroring so much', 'generate the daily proxy report'."
---

# Proxy Usage Analysis

Turn the last 24 hours of llama-proxy session and fallback activity into a
per-session CSV record (split into fast / cheap buckets per the slot
schedule) and an operator-facing Markdown report with highlighted,
data-backed recommendations — including an **error taxonomy** and a
deterministic **root-cause breakdown** for any proxy errors observed in the
window.

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
| `--json` | off | Print a machine-readable JSON summary instead of the text summary |
| `--quiet` | off | Suppress the stdout summary |

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
report.

See [REFERENCE.md](REFERENCE.md#outputs) for the full schema of each
artifact and the archival behaviour.

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
fast/cheap bucketing, recommendation rules, llama-server eval-timing parsing
(decode + prompt eval, Qwen3 port filtering, fast/cheap speed stats), and an
end-to-end run, using fixtures copied from real `/var/log/llama-proxy`
lines (`proxy.log` and `llama-server.log`).
