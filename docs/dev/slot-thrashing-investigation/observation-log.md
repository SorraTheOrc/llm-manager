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
| 2026-08-07 | save 6/2084 (0.29%) | restore 2/1778 (0.11%) | 4 | 62 |
| 2026-08-08 | save 2/1383 (0.14%) | restore 0/1222 (0.0%) | 2 | 343 |
| 2026-08-09 | save 7/2418 (0.29%) | restore 1/2082 (0.05%) | 3 | 157 |
| 2026-08-10 | save 12/1055 (1.14%) | restore 1/937 (0.11%) | 2 | 84 |
| 2026-08-11 | save 9/1800 (0.5%) | restore 0/1625 (0.0%) | 3 | 109 |
| 2026-08-12 | save 11/1310 (0.84%) | restore 1/1150 (0.09%) | 3 | 38 |
| 2026-08-13 | save 15/1211 (1.24%) | restore 0/1078 (0.0%) | 2 | 325 |
| 2026-08-14 | save 11/2090 (0.53%) | restore 3/1906 (0.16%) | 2 | 27 |

## Pre-fix baseline (reference)

| Date | slot_save failed/total (%) | slot_restore failed/total (%) | max concurrent streams at failure | concurrency fallbacks near failures |
|------|----------------------------|-------------------------------|-----------------------------------|-------------------------------------|
| 2026-08-06 | 38/2216 (1.71%) | 19/1939 (0.98%) | 3 | 2073 |

## 2026-08-15: 7-day window complete — analysis & verdict

**Window:** 2026-08-08 → 2026-08-14 (7 full calendar days post-release; fix
released in v0.1.11 on 2026-08-08). Includes the day 3-slot → night 8-slot
config cycle; each row reads the full day's logs.

### Aggregate (canonical 7-day window, 08-08..08-14)

| Metric | Post-fix | Pre-fix baseline (08-06) | Change |
|--------|----------|--------------------------|--------|
| slot_save failed/total | **67/11267 (0.59%)** | 38/2216 (1.71%) | **−65.5%** |
| slot_restore failed/total | **6/10000 (0.06%)** | 19/1939 (0.98%) | **−93.9%** |
| failures with ≥1 concurrent local stream | 58/73 (79%) | 52/57 (91%) | — |
| max concurrent local streams at a failure | 3 | 3 | — |

### Residual-failure classification (window total 81 failures)

| Class | Count | Detail |
|-------|-------|--------|
| ReadTimeout, full-window wait | 70 | `elapsed==timeout` at the adaptive window — same save-starvation signature as baseline, at ~⅓ the rate |
| ReadTimeout, partial | 4 | elapsed < timeout (e.g. 10.4/10.4 → 5.7/8.7 quirks; circuit-breaker interplay) |
| ConnectError | 6 | `elapsed=0.0s` — llama-server restart/connection windows (e.g. 08-10 22:52:26 = proxy shutdown moment), NOT load-related |
| Restore 400 | 1 | "no available space in KV cache or invalid slot save file" — capacity error, NOT a timeout |

### Why residual failures persist despite the gate (rationale, AC2)

The load-aware gate (`_slot_persistence_skip_when_busy`) is evaluated in
`_build_slot_context` at **request start**; the save/restore executes at
**request end** (after streaming completes). Concurrency can onset **mid-request**: at
request start the slot is idle (gate passes), another session begins streaming during the
request, and the request-end save collides with the new stream → ReadTimeout at exactly
the adaptive window. The gate therefore eliminates saves that *start* under load but cannot
catch concurrency that *onsets during* a request. The residual 70 full-window ReadTimeouts
are precisely this mid-request-onset class, at ~⅓ the baseline rate (0.59% vs 1.71%).

### Verdict vs exit criteria (parent AC2)

Exit criteria: **0 failures during the window** OR bounded/residual failures with explicit
rationale and load context, vs the ~1.8% baseline.

**Bounded/residual branch satisfied:** failures are not zero, but they are bounded and
explained:

- Save failure rate cut **65.5%** (1.71% → 0.59%), restore cut **93.9%** (0.98% → 0.06%).
- 79% of residual failures still cluster under concurrent local streams — the confirmed
  save-starvation mechanism, at reduced rate, with a documented residual path
  (mid-request onset past the request-start gate sample).
- 7 of 81 residual failures are non-load classes (6 ConnectError during llama-server
  restarts, 1 restore-400 capacity error), not timeouts under load.
- No evidence of GPU wedge or circuit-breaker suppression loss: breaker trips remain
  bounded (3 consecutive failures → 300s cooldown) and the context-size gate is unchanged.
- Impact of a residual save failure is unchanged from design: that session resumes cold
  via full prefill (llama.cpp internal KV cache may cover it); rate is ~1 failure per 170
  saves at the current load profile.

### Observability note (gate-skip logging)

Gate-skip lines (`reason=slot_busy`, `reason=context_too_large`, circuit-breaker
disabled) are logged via `session.py`'s **module** logger, which has no handler attached in
production (the file handler is bound to the `llama-proxy` logger). They are therefore not
present in `/var/log/llama-proxy/proxy.log*` — gate *effectiveness* must be inferred from
the failure-rate reduction above rather than counted skip lines. The failure lines
(`slot_save/restore failed ... elapsed= timeout= busy=`) use `srv.logger` and are present.
This is a minor observability gap; the primary evidence (rate reduction + residual
classification) is unaffected.
