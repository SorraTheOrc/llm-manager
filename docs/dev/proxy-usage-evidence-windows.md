# Proxy-usage evidence windows: archiving and regeneration

Rotated `proxy.log` segments are the only record of what the proxy served, and
they are **not** retained contractually in this deployment — in practice they
last roughly a week (verified 2026-10-28: the oldest retained line was
`2026-10-20 13:10:30`, i.e. ~8 days). Any investigation that quotes a 24 h
window must therefore (a) copy the raw segments covering that window somewhere
safe *while the window is still retained*, and (b) record the exact command
that produced every number, so the figures stay reproducible afterwards.

The +96 % logged-token investigation (LP-0MVIAEARO0001O7R) exists because an
earlier pass measured four windows into `/tmp` and lost both the outputs and,
by the time the archiving work item ran, the 2026-10-18/19 source segments.
`scripts/proxy_usage_window_archive.py` is the deterministic procedure that
prevents a repeat.

## Tool

```bash
# 1. What does the corpus still cover for this window, and where are the holes?
python3 scripts/proxy_usage_window_archive.py coverage \
  --log-dir /var/log/llama-proxy \
  --start "2026-10-20 14:44:27" --end "2026-10-21 14:44:27"

# 2. Copy + verify the raw segments before retention rotates them away.
python3 scripts/proxy_usage_window_archive.py archive \
  --log-dir /var/log/llama-proxy --dest ~/proxy-usage-reports/cost/raw \
  --start "2026-10-18 00:00:00" --end "2026-10-22 23:59:59"

# 3. Regenerate the window and append a reproducible manifest block.
python3 scripts/proxy_usage_window_archive.py run-window --name post24 \
  --start "2026-10-20 14:44:27" --end "2026-10-21 14:44:27" \
  --output-dir ~/proxy-usage-reports/cost/post24 \
  --log-dir /var/log/llama-proxy \
  --manifest ~/proxy-usage-reports/cost/raw/manifest.txt
```

Contract of the three subcommands:

- **`coverage`** — thin CLI over the shared `scripts/lib/proxy_logs.py`
  discovery/coverage implementation (both rotation naming schemes, plain or
  gzip), reporting per-file spans, adjacent-file gaps, measured hours and a
  `complete` verdict. A window whose segments are gone reports `0.0 h measured`
  instead of a plausible-looking partial read.
- **`archive`** — copies each in-range segment, verifies the copy with
  `gzip -t` **and** a byte comparison against the source (`gzip+cmp`), records
  SHA-256 sums, and never writes to the source directory. Re-runs are
  idempotent; an archived file whose content diverges from the source is a hard
  error rather than a silent overwrite.
- **`run-window`** — runs `analyze_proxy_usage.py` as-is (never modified here),
  captures its stdout, parses the headline numbers out of the generated
  `report.md`, and appends a manifest block containing the boundaries, the exact
  command, the headline numbers and the coverage verdict. A failed run writes no
  manifest block, so the manifest only ever describes an artifact that exists.
  Tokens-per-request is derived from the report's own totals, not re-derived
  from logs.

`llama-server.log*` segments carry **no** per-line timestamps in this
deployment, so their inclusion is decided by the rotation date encoded in the
filename (`llama-server.log-YYYY-MM-DD.gz`); the live `llama-server.log` is
never archived because it cannot be ranged.

## Result for the 2026-10-18 → 2026-10-22 investigation

Full record: `~/proxy-usage-reports/cost/README.md` and
`~/proxy-usage-reports/cost/raw/manifest.txt` (path outside the repo on
purpose — the raw segments are 7.8 MB of log data and the artifacts are host
state, not source).

| window | boundary | status |
|---|---|---|
| `base24` | 2026-10-18 14:37:48 → 2026-10-19 14:37:48 | **preserved original run** — logs had rotated away; the original 2026-10-19 14:37:56 report is mirrored at `~/proxy-usage-reports/2026-10-20/` (421,848,059 tokens, 12,810 requests — reproduces the recorded 421.7 M / 12,810 within 0.04 %) |
| `ref24` | 2026-10-19 14:44:27 → 2026-10-20 14:44:27 | **unavailable** — only the last 1.57 h retained (6.5 %); no recorded baseline exists. Use `now24` as the reference window instead |
| `post24` | 2026-10-20 14:44:27 → 2026-10-21 14:44:27 | regenerated — 759,248,067 tokens / 8,359 requests over **21.83 h of 24 h** (2.17 h hole: the `proxy.log-2026-10-21_00.gz` daily rotation was itself rotated away) |
| `now24` | 2026-10-21 14:44:27 → 2026-10-22 14:44:27 | regenerated — 490,817,220 tokens / 7,867 requests over **24.0 h of 24 h**, complete |

Two lessons worth carrying into the next investigation:

1. **A gap-adjusted total is not a total.** `post24`'s absolute token volume is
   8.4 % below the previously recorded 828.5 M, and the missing 2.17 h is 9.0 %
   of the window: the gap-adjusted expectation is 753.6 M, so the observed
   759.2 M is within +0.75 %. The quantity the investigation actually quotes —
   **tokens per request** (90,830 vs ≈90,675, +0.17 %) — is unaffected, because
   the missing segment removes requests and their tokens in proportion. Quote
   the ratio, or gap-adjust the absolute.
2. **Archive at measurement time, not later.** The natural reading of "record
   the figures so they are reproducible" is to keep the numbers; retention
   makes keeping the *segments* the thing that actually matters. Run
   `archive` in the same session as the measurement.

## Related

- `.pi/skills/proxy-usage-analysis/SKILL.md` — the analysis CLI used unchanged
  for every window here.
- `scripts/lib/proxy_logs.py` — shared rotation-aware discovery/coverage.
- `tests/test_proxy_usage_window_archive.py` — pins the coverage verdicts,
  copy faithfulness and manifest record this document relies on.
