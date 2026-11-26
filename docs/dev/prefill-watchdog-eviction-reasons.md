# Prefill-watchdog eviction reasons — the umbrella split (LP-0MWKQJHSM000CI6W)

The no-progress watchdog releases an accepted local dispatch with a **distinct
reason** it computed, and the post-acceptance eviction line
(`local_eviction_after_acceptance`) reports that same reason. Before this
change the release path stamped the umbrella label `no_progress_watchdog` on the
eviction line even when the `lease_released` line carried
`prefill_eta_exceeded`, so the residual analyser counted a *label* rather than a
*behaviour* and a relabelling could be mistaken for a fix.

Observability only: no admission decision, abort policy or threshold changed.

## Reason vocabulary

`:mod:`proxy.admission_events` is the single source of truth; these are the
watchdog's four values.

| `reason=` | Meaning | Set by |
|---|---|---|
| `prefill_eta_exceeded` | Still inside the hard ceiling, but elapsed exceeds `eta_prefill_s * local_prefill_eta_factor` (default factor 2.0). A **runaway prefill** — progressing but far slower than priced. | `_watchdog_release_reason` |
| `prefill_stalled` | No measured progress for `local_dispatch_no_progress_timeout_seconds` (default 180 s). A **genuine wedge**. | `_watchdog_release_reason` |
| `prefill_max_exceeded` | Total active time exceeds `local_dispatch_max_prefill_seconds` (default 900 s), regardless of progress. The hard backstop. | `_watchdog_release_reason` |
| `no_progress_watchdog` | **Fallback only** — a watchdog release that no distinct reason attributes. It no longer stands in for a behavioural class. | release path fallback |

Priority order is earliest-abort first: `prefill_eta_exceeded` →
`prefill_stalled` → `prefill_max_exceeded`. A record without a
`last_progress_ts` baseline (legacy records) returns no distinct reason and is
left to the pre-existing expiry/orphan path.

## Eviction line contract

The line below is emitted at **ERROR**:

```
local_eviction_after_acceptance session=<id> reason=<reason> \
  estimated_tokens=<n> new_tokens=<n> eta_prefill_s=<s|unknown> \
  elapsed_s=<s> slot_id=<id|unknown> llama_task_id=<id|unknown> \
  endpoint=<name>[ llama_task_id_reason=<why>] \
  task_started=<yes|no> progress_observed=<yes|no> queue_wait_s=<s|unknown> \
  [<caller extra_fields>]
```

- **Every pre-existing key keeps its position.** The only change to an existing
  key is the `reason=` *value* (the distinct reason instead of the umbrella).
- `llama_task_id_reason=` (LP-0MW959RRO008AAH9) appears only when a known slot
  could not be attributed.
- The **release-context evidence is always appended**:
  - `task_started=yes|no` — the llama-server task id resolved from the cached
    `/slots` snapshot (the same evidence the `task_id_not_started` reason names;
    `no` is the never-started class).
  - `progress_observed=yes|no` — a **measured** numeric token progress was
    recorded (`last_progress > 0`). A liveness-only lease extension does not
    count as progress.
  - `queue_wait_s=<s|unknown>` — the measured contention-queue wait stamped on
    the dispatch record at admission; `unknown` when the request never queued.
  - Caller-specific `extra_fields` (e.g. the measured/priced numbers behind a
    `measured_cache_deficit` release) still come last.
- `parse_fields` consumers are unaffected: the line remains flat `key=value`
  and the new keys are appended, never interleaved.

This is the field set the V1 window (LP-0MWM0CP4B001088M AC3) classifies every
post-acceptance abort by: emitted reason, `task_started`, `progress_observed`
and queue-wait bucket.

## Reconciliation between the two lines

For every watchdog release the sweep emits exactly one `lease_released` line and
exactly one `local_eviction_after_acceptance` line, and **the two per-reason
breakdowns must match**. In the group-2 window they did not: 327
`lease_released reason=prefill_eta_exceeded` against 0 evictions carrying that
reason. A mixed-window replay test
(`proxy/tests/test_watchdog_reason_split.py::test_mixed_window_reconciles_the_two_breakdowns`)
now fails if the two paths drift again.

The residual analyser (`proxy/scripts/analyse_prefill_deadline_residual.py`)
groups by whatever `reason=` the line carries, so the distinct reasons appear in
its per-reason split and attribution without a schema change. The named
constants are `REASON_PREFILL_ETA_EXCEEDED`, `REASON_PREFILL_STALLED`,
`REASON_PREFILL_MAX_EXCEEDED` and `REASON_NO_PROGRESS_WATCHDOG`.

## Rollback

The change is label-only. Rolling back is reverting the commit; there is **no
config value** to change and no restart-side effect. The distinct reasons are
also emitted on the pre-existing `lease_released` line, which is unchanged by
this work.
