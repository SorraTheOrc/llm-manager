# Session Compaction Evaluation — Analysis Scripts

Evaluation artifacts for LP-0MTAQNAQT002L746 ("Proactive session compaction
for oversized contexts"). **Evaluation only — no behavior change**; the
proxy code is untouched by these scripts.

## Scripts

| Script | Outputs | Purpose |
|---|---|---|
| `proxy/scripts/analyze_context_distribution.py` | `proxy/docs/context-compaction-eval/distribution.{json,md}` | Per-mode per-day session estimated-context distributions (median/mean/p90/p95/max), breach counts at the per-mode caps, `context_pressure` / `routing_skip_local` tallies (F1, LP-0MTC87GBV0031F4B). |
| `proxy/scripts/correlate_oversized_sessions.py` | `proxy/docs/context-compaction-eval/correlation.{json,md}` | Hourly timeline (pressure / skips / denials / 5xx / checks), per-session correlation (peak context, prefill work, wasted work at ratio>1.0, skips, pressure), llama-server decode/prefill evidence (F2, LP-0MTC8A2UB0040NKQ). |

Design/recommendation documents: `docs/session-compaction-design.md` (F3,
LP-0MTC8B0BN009BMTJ) and `docs/session-compaction-recommendations.md` (F4,
LP-0MTC8BWV50012WSW).

## Data sources

- `/var/log/llama-proxy/proxy.log*` (plain + rotated `.gz`): timestamped
  `routing_check` (per-request `estimated_tokens`, `warm_threshold` = mode
  clamp 83285 fast / 100000 cheap), `context_pressure` warnings,
  `routing_skip_local` skips, `local_dispatch_denied`, upstream 5xx.
- `/var/log/llama-proxy/llama-server*.log` (no timestamps; decode/prefill
  evidence attributed by rotation-file close time — see the correlation
  report's caveats).
- The report CSVs (`~/proxy-usage-reports/2026-08-2X/`) are **not** used for
  context sizes: their context columns are empty; the proxy logs are the
  ground truth for `estimated_tokens`.

## Reproduce

```bash
# F1 — distributions and breach counts (defaults: 2026-08-24..26)
python3 proxy/scripts/analyze_context_distribution.py \
    --log-dir /var/log/llama-proxy \
    --output-dir proxy/docs/context-compaction-eval

# F2 — correlation for the incident day
python3 proxy/scripts/correlate_oversized_sessions.py \
    --log-dir /var/log/llama-proxy \
    --day 2026-08-26 \
    --output-dir proxy/docs/context-compaction-eval
```

Both scripts accept `--json` to print the machine-readable report instead of
Markdown. Re-running them regenerates the committed artifacts deterministically.

## Tests

```bash
python3 -m pytest proxy/tests/test_analyze_context_distribution.py proxy/tests/test_correlate_oversized_sessions.py -q
```

Tests use fixture log lines in a temp dir; they never touch live logs.

## Known caveats (from the reports)

- llama-server logs carry no timestamps — decode/prefill evidence is
  window-attributed, not hour-exact.
- Proxy rotated logs have a ~2h gap on 2026-08-26 22:00–24:00; earlier
  hour-22 figures (4,541 fallbacks, 280 5xx, 42.7M prefill) cover other
  windows and differ from the calendar-day recomputation.
- Sessions with `session=unknown` routing checks (unattributed, ~30/day) are
  excluded from session aggregation.