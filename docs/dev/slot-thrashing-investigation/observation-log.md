# Slot-persistence failure observation log (LP-0MSI1RWLM007N367 F4)

7-day post-fix observation window for the load-aware save gate + timeout
rebalance (commit `9acfebc`). Regenerate daily with:

```bash
./scripts/slot-persistence-failures.sh               # today
./scripts/slot-persistence-failures.sh --date YYYY-MM-DD
```

Each run appends a row to the table below and to the parent work item
(LP-0MSI1RWLM007N367). The **window starts only after the fix is deployed**
(pushed to `dev` and promoted); the pre-fix baseline row is the reference the
7-day window is measured against.

Exit criteria (parent AC2): **0 failures during the window** OR bounded/residual
failures with explicit rationale and load context, vs the ~1.8% baseline.

> The window survives the day 3-slot → night 8-slot config cycle because each
> daily row reads the full calendar day's proxy logs.

## Post-fix observation rows (appended per day)

| Date | slot_save failed/total (%) | slot_restore failed/total (%) | max concurrent streams at failure | concurrency fallbacks near failures |
|------|----------------------------|-------------------------------|-----------------------------------|-------------------------------------|

## Pre-fix baseline (reference)

| Date | slot_save failed/total (%) | slot_restore failed/total (%) | max concurrent streams at failure | concurrency fallbacks near failures |
|------|----------------------------|-------------------------------|-----------------------------------|-------------------------------------|
| 2026-08-06 | 38/2216 (1.71%) | 19/1939 (0.98%) | 3 | 2073 |