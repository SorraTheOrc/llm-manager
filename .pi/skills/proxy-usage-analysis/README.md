# Proxy Usage Analysis Skill

The proxy-usage-analysis skill generates usage reports from llama-proxy logs,
converting 24 hours of session and fallback activity into per-session CSV
records and an operator-facing Markdown report.

## Attribution Markers

Since **LP-0MVAPNC2900145RH** the proxy logs restart and mode-switch lines
with attribution markers — ``source=<value>``, ``actor=<value>``, and
``reason=<value>`` key=value pairs — that identify the origin of each
restart/mode-switch event.

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
