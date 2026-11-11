# Proxy Usage Analysis Skill

The proxy-usage-analysis skill generates usage reports from llama-proxy logs,
converting 24 hours of session and fallback activity into per-session CSV
records and an operator-facing Markdown report.

## Pipeline overview

One deterministic pipeline turns the logs into the daily artifacts — no LLM is
in the loop:

```text
proxy.log + llama-server.log
        │  streaming parse (log_parser.py / llama_log_parser.py, window-filtered)
        ▼
   aggregation.py      sessions, fallbacks, busy time, TTFT, cache, denials, ETA
        │              accuracy — the ten scorecard inputs
        ▼
  performance_metrics.py   metric registry, best-3 history, targets, verdicts
        │
        ▼
   reporting.py ──► report.md        scorecard first, then the per-metric
        │                            details, then the data sections
        ├─────────► standard_sessions.csv, low_priority_sessions.csv,
        │           errors.csv, errors.json and the --json summary
        ▼
  <output-dir>/performance_metrics_state.yaml    live best-3 history,
                                                 rewritten once per run

  report.md ──► discord_notify.py ──► Discord webhook (report as an
                                      attachment; a failed run posts a
                                      failure notice instead)
```

- **Scorecard.** `report.md` opens with `## Performance metrics summary`: the
  ten metrics the proxy is held to, each with its best recorded baseline, an
  evidence-based target (the mean of the three best measurements) and this
  window's `Met` / `Unmet` / `Insufficient history` result — followed by a
  `## Performance metric details` block with one subsection per metric
  (importance, calculation, levers, trade-offs).
- **State file.** The committed `metrics/performance_metrics.yaml` holds the
  metric definitions and detail copy and is **never** written at runtime; the
  mutable history lives in `<output-dir>/performance_metrics_state.yaml`, so
  the nightly run never dirties the repository.
- **Backfill.** A one-off `--seed-metrics-history [DIR]` seeds the best-3
  history from the archived `report.md` files (idempotent per window), so the
  scorecard is useful on its first run instead of reporting `Insufficient
  history` for three days.
- **Delivery.** A successful run posts `report.md` to the Discord webhook in
  `.worklog/config.yaml` as an attachment and records the outcome in the
  report's own `- Discord delivery:` bullet; a failed run posts a failure
  notice to the same webhook. `--no-discord` keeps a run local. The webhook
  URL is a credential and is never written to any artefact.
- **No ctx-size advice.** The local context is fixed by the model and the
  hardware, so no recommendation advises raising it — the large-context rules
  name prompt compaction, slot rebalancing and the by-design remote bypass
  instead.

Metric definitions, both file schemas and the backfill command:
[REFERENCE.md](REFERENCE.md#performance-metrics-scorecard-lp-0mw0fu5z6006a4p3).

## Attribution Markers

Since **LP-0MVAPNC2900145RH** the proxy logs restart and mode-switch lines
with attribution markers — ``source=<value>``, ``actor=<value>``, and
``reason=<value>`` key=value pairs — that identify the origin of each
restart/mode-switch event.

These marker rules cover restart and mode-switch events only. Per-session
**fallback attribution** — which session's `Fallback triggered` or routing-skip
line explains a move to remote, joined turn-first by `turn=` — is documented in
[REFERENCE.md](REFERENCE.md#limitations) under *Per-session fallback
attribution*.

### How attribution is consumed

The analysis pipeline (``analyze_proxy_usage.py``) consumes attribution data
through the following path:

1. **Log parsing** (`log_parser.py`): The streaming log parser extracts
   ``source``, ``actor``, and ``reason`` from mode-switch, restart-spawned,
   restart-complete, and startup-attribution log lines, storing them on the
   resulting :class:`LogEvent` objects.

2. **Restart classification**: Attribution markers on `restart` events are
   used to classify mode switches by trigger origin:

   | `source` value | Classification |
   |----------------|----------------|
   | `scheduler`    | Scheduled mode transition |
   | `worker`       | ContextHub/herdr idle-cheap or fast-on-command |
   | `api`          | External API call (`/admin/set-mode`) |
   | `manual`       | Direct operator action |

3. **Report generation**: The generated report surfaces attribution breakdowns
   for restart frequency and mode-switch triggers, enabling operators to
   distinguish automated (scheduler/worker) from manual mode changes.

4. **Mode-flip analysis** (`analyze_mode_flips.py`): The mode-flip classifier
   uses `source`-based classification (``source=scheduler`` → ``schedule``,
   ``source=worker`` → ``to-fast``/``to-cheap``) with heuristic fallback for
   historical unattributed logs.

### Backward compatibility

Attribution markers are **appended** to existing log lines — they do not
replace or modify the original format. Lines from before LP-0MVAPNC2900145RH
do not carry attribution, and the parser stores ``None`` for ``source``,
``actor``, and ``reason`` on those events. Downstream analysis falls back to
heuristic classification when attribution is absent.

## See Also

- [SKILL.md](SKILL.md) — skill invocation and CLI usage
- [REFERENCE.md](REFERENCE.md) — detailed reference, output schemas, and
  implementation notes
- Log parser attribution logic:
  [log_parser.py](scripts/log_parser.py)
- Mode-flip classification: [analyze_mode_flips.py](../../../scripts/analyze_mode_flips.py)
