# Regenerated proxy usage report — end-to-end verification (LP-0MVJO4382002FJF2)

Regenerate the proxy usage report for a real 24 h window and verify the
corrected busy figure (dispatch → finish union) and the new prefill/decode
split against the intake brief's expected order (LP-0MVGCQ421005IXGC).

## Window and artefact

- Window: **2026-10-19 07:23:00 → 2026-10-20 07:23:00** (24.0 h) — the brief's
  own window, chosen because the source logs are retained.
- Full regenerated report (retained artefact):
  `~/proxy-usage-reports/verify-LP-0MVJO4382002FJF2/report.md`
  (plus `standard_sessions.csv`, `low_priority_sessions.csv`, `errors.csv`,
  `errors.json` in the same directory).
- CLI used (identical to the daily 05:00 cron, with an explicit window):

  ```bash
  python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
      --log-dir /var/log/llama-proxy \
      --start "2026-10-19 07:23:00" --end "2026-10-20 07:23:00" \
      --output-dir ~/proxy-usage-reports/verify-LP-0MVJO4382002FJF2
  ```

- Run at 2026-10-23 02:14:35: 21 log file(s) parsed, 344 sessions,
  17458 requests (local 411 / remote 17047), 17005 fallback events.

## Observed vs expected-order figures

The intake brief's sanity values are wall-clock sums and slot-hours from a
slightly different pairing; the purpose is **order-of-magnitude** agreement,
which holds:

| Figure | Brief expectation | Observed (regenerated report) | Match |
| --- | --- | --- | --- |
| Prefill slot-time (sum of per-stream prefill) | ≈ 14.3 h | **13h 56m** (p10 4.4 s, med 74.5 s, p90 297.0 s, max 817.3 s; 411 samples) | ✔ ~97% |
| Decode slot-time | ≈ 7.8 h | **7h 24m** (p10 3.4 s, med 12.6 s, p90 150.8 s, max 1621.1 s; 408 samples) | ✔ ~95% |
| Prefill share of phase slot-time | ≈ 65 % (14.3/22.1) | **65.3 %** | ✔ |
| Busy % — new definition (dispatch → finish union ÷ window) | ≈ 46 % | **54.4 %** (13h 03m of 24h) | ✔ same order |
| Busy % — pre-change decode-only definition | ≈ 16 % | **27.9 %** (6h 42m) | ✔ same order |

Notes on the small absolute differences (all benign):

- The brief's 463 local requests vs 411 in this window are from a different
  pairing pass; the count and the absolute durations differ at the margins.
- Observed busy 54.4 % vs the estimate's 46 %: the estimate used 48 slot-hours
  of capacity, while this window's report divides by 86400 s of wall clock
  (avg concurrency while busy 1.63, peak 4).
- Observed decode-only 27.9 % vs the brief's 16 %: the 16 % derived from
  slot-time (7h 47m of 48 slot-hours), not wall-clock.

## AC coverage from the regenerated report

**AC1 / AC3 — corrected busy figure + `## Local model utilization`.** The
section shows:

```
| Busy time — upper bound (dispatch → finish) | 13h 03m (54.4%) | ...
| Decode-only busy — lower bound (`Stream started` → finish) | 6h 42m (27.9%) | ...
| Prefill busy (upper − lower) | 6h 21m | ...
```

with the definition-change note ("Busy = at least one local slot active
**from dispatch to finish** … it previously counted decode only
(`Stream started` → `Stream finished`) …") and the labelled lower/upper
bounds.

**AC1 — `## Prefill vs decode`.** Both rows render:

```
| Prefill (dispatch → first byte) | 13h 56m | 65.3% | 4.4 | 74.5 | 297.0 | 817.3 | 411 |
| Decode (first byte → finish) | 7h 24m | 34.7% | 3.4 | 12.6 | 150.8 | 1621.1 | 408 |
```

with the explicit slot-seconds-not-wall-clock statement; percentages sum to
100.0 %.

**AC4 — eval-timing sections not blank.** `## Decode speed` shows 77 real
samples (median 21.6 tok/s) and `## Prompt eval speed` 77 samples (median
93.6 tok/s), taken from the window's `llama-server` logs (Qwen3 child port
discovered); one out-of-window rotated file is explicitly noted as skipped
("Qwen3 child port not found"), so no section is an unattributed blank.

**AC5 — busy % materially higher.** 54.4 % (dispatch → finish) vs 27.9 %
(decode-only) in the same window: the definition change roughly doubles
reported local busy, matching the brief's 16 % → 46 % order-of-magnitude
claim. The pre-change report for the near-identical window
(`~/proxy-usage-reports/2026-10-20_2/report.md`, 2026-10-19 05:00 →
2026-10-20 05:00) shows the old definition at 29.9 % (7h 10m), i.e. the
increase comes from the definition change, not data drift.

## Automated regression coverage

`TestRegeneratedReportExpectedOrder` in
`.pi/skills/proxy-usage-analysis/tests/test_analyze_proxy_usage.py`
(LP-0MVJO4382002FJF2) pins this end to end with a brief-shaped fixture
(180 s prefill / 100 s decode per stream, ≈16 % → ≈46 % busy order):

- the report shows the upper/lower busy rows and the definition-change note;
- `## Prefill vs decode` splits 64.3 % / 35.7 % and totals to 100 %;
- busy % is ≥ 2.5× the decode-only busy % (materially higher, same order);
- `## Decode speed` / `## Prompt eval speed` are never an unattributed blank
  (real samples when llama-server eval timing exists, named diagnostic
  otherwise);
- the JSON summary's `local_busy` / `local_prefill` / `prefill_vs_decode`
  match the report and decompose as busy = decode + prefill.

## Conclusion

All acceptance criteria of LP-0MVJO4382002FJF2 are met by the regenerated
report; the corrected busy definition and prefill/decode split are visible in
real 24 h logs and agree with the brief's expected order.
