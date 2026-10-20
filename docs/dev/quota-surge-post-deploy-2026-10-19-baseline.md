# Quota surge post-deploy allowance utilisation — 2026-10-20

Generated 2026-10-19T22:57:35.155546Z by `scripts/quota_allowance_report.py` (F10, LP-0MVF7KTMI005B667).

**Target:** ≥ 90% monthly consumption at reset per account (residual ≤ 10% — the margin the rolling 20% / weekly 50% caps need). **Hard stop:** 99.5% (`quota_exhaustion_percent`).

## Invocation (reproducibility, AC13d)

```bash
scripts/quota_allowance_report.py --log-dir /var/log/llama-proxy --start 2026-10-19T18:00:00 --end 2026-10-20T00:00:00 --output-md docs/dev/quota-surge-post-deploy-2026-10-19-baseline.md --output-json docs/dev/quota-surge-post-deploy-2026-10-19-baseline.json --quiet
```

- log dir: `/var/log/llama-proxy`
- window: `2026-10-19T18:00:00Z` → `2026-10-20T00:00:00Z`
- filters: threshold `99.5%`, surge window `7` days, max poll gap `6.0` h
- machine-readable data: `docs/dev/quota-surge-post-deploy-2026-10-19-baseline.json`

## Per-account results

No monthly period has completed inside the analysed window yet.

### Periods in the window

**OPENCODE_2_API_KEY**

- reset `2026-10-30T01:03:42Z` (in progress): latest reading 95% (241.28 h before the scheduled reset; not yet a moment-of-reset measurement), surge 0 h (final window 0 h), hard stop 0 h, skips 0 (dry-run 0), poll failures 0, stale suspensions 0 → **in_progress**

**OPENCODE_3_API_KEY**

- reset `2026-11-17T11:05:08Z` (in progress): latest reading 13% (683.31 h before the scheduled reset; not yet a moment-of-reset measurement), surge 0 h (final window 0 h), hard stop 0 h, skips 0 (dry-run 0), poll failures 0, stale suspensions 0 → **in_progress**

**OPENCODE_API_KEY**

- reset `2026-10-23T09:43:54Z` (in progress): latest reading 93% (81.95 h before the scheduled reset; not yet a moment-of-reset measurement), surge 0 h (final window 0 h), hard stop 0 h, skips 0 (dry-run 0), poll failures 0, stale suspensions 0 → **in_progress**

## Outcome notes

- Nothing to classify yet: no account has crossed a monthly reset inside this window. Re-run after each account's next reset (schedule in `docs/dev/quota-surge.md` §9.1).

## Summary

- accounts measured: 3
- completed periods: 0
- target met (≥ 90%): 0
- below target: 0
- overspend (≥ 99.5%): 0
- insufficient data: 0
