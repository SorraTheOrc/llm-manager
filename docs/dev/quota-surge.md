# Quota-aware OpenCode Go surge

**Work item:** LP-0MVF7KRH30000KOA (Configuration keys and quota-surge design doc)
**Parent epic:** LP-0MVF4FIP1001RD8P (Quota-aware OpenCode Go surge: spend the
full monthly allowance before it resets)

This document describes the shipped quota-surge behaviour: the problem it
solves, the `/v1/usage` contract it depends on, the derivation and pacing
models, the peak/off-peak trade-off, the safety guards, and the operational
runbook. Every behavioural claim here is covered by an automated test; the
claim → test map lives in the docstring of
[`proxy/tests/test_quota_config_docs.py`](../../proxy/tests/test_quota_config_docs.py)
and was recorded as a review note on the work item (epic AC12d).

- [1. The problem](#1-the-problem)
- [2. The `/v1/usage` contract](#2-the-v1usage-contract)
- [3. Derivation model](#3-derivation-model)
- [4. Pacing and scoring model](#4-pacing-and-scoring-model)
- [5. Peak/off-peak trade-off](#5-peakoff-peak-trade-off)
- [6. Safety](#6-safety)
- [7. Configuration reference](#7-configuration-reference)
- [8. Operational runbook](#8-operational-runbook)

---

## 1. The problem

OpenCode Go / Go Plus allowances are **use-it-or-lose-it**: the plan is a
monthly dollar amount per account, and any allowance still unspent at
`resetsAt` is simply lost — there is no carry-over. Two further properties make
the naive "just route in config order" behaviour wasteful:

- **Reset dates are per account and are not calendar-aligned.** A live reading
  on 2026-10-19 showed three accounts resetting roughly seven days apart
  (`2026-10-23`, `2026-10-30`, `2026-11-17`), so any surge window must be
  computed per account from its own `resetsAt` rather than from a fixed
  calendar rule.
- **The spend spread was extreme.** The same reading showed two of three
  accounts close to exhaustion while the third was nearly untouched:

  | entry | `api_key_env` | rolling | weekly | monthly | monthly `resetsAt` |
  |---|---|---|---|---|---|
  | `opencode-go`   | `OPENCODE_API_KEY`   | 0% | 0% | **88%** | 2026-10-23T08:43:55Z |
  | `opencode-go-2` | `OPENCODE_2_API_KEY` | 0% | 0% | **95%** | 2026-10-30T00:03:43Z |
  | `opencode-go-3` | `OPENCODE_3_API_KEY` | 3% | 1% | **5%**  | 2026-11-17T10:05:09Z |

Before this feature the proxy was blind to the remaining allowance: routing
inside the `opencode-go*` chain was static, and exhaustion was learned
*reactively* from a `GoUsageLimitError` / 429. The surge feature adds a
proactive, authoritative reading and steers traffic toward the account that can
absorb it before its own reset.

## 2. The `/v1/usage` contract

The authoritative quota source is:

```
GET {endpoint}/v1/usage        Authorization: Bearer <api key>
                               Cache-Control: no-store
                               Accept: application/json
```

A successful body has this shape (values illustrative):

```json
{"usage": {
   "rolling": {"status": "ok", "percent": 0, "resetsAt": "2026-10-19T15:22:54.000Z"},
   "weekly":  {"status": "ok", "percent": 0, "resetsAt": "2026-10-26T00:00:00.000Z"},
   "monthly": {"status": "ok", "percent": 88, "resetsAt": "2026-10-23T08:43:55.000Z"}
}}
```

Properties the implementation relies on:

- **Only the `usage` object carries quota information.** Response headers carry
  no `x-ratelimit-*`, `retry-after` or reset hints on success *or* error, and
  the response `usage` token counts carry no cost/remaining figure.
- **`cache-control: no-store`** — the response is never cached; the proxy both
  honours this and requests it.
- **No model selector.** `?model=…` and `?detailed=true` are ignored, so quota
  is **per API key (per account), not per model or endpoint**.
- **No credits/billing endpoint exists** (`/v1/credits`, `/v1/billing`,
  `/v1/quota`, `/v1/me`, `/v1/account` all return 404), so the console-only
  "Use balance" pay-as-you-go fallback is not API-visible. This is why
  [`quota_exhaustion_percent`](#6-safety) is a conservative hard stop.

**Graceful degradation is mandatory.** The endpoint is undocumented, so a poll
failure (non-2xx, timeout, malformed body, missing key) must never fail or
delay a client request: the poller keeps the last known-good values, records
the error as `last_status = "poll_error:<ExceptionName>"`, and logs at WARN
(rate-limited to one WARN per account per 300 s). Client-facing reads are a
pure in-memory cache read and never wait for an in-flight poll
(`test_poll_failure_keeps_previous_values_and_warns`,
`test_cache_read_does_not_wait_for_inflight_poll`).

## 3. Derivation model

`proxy/quota_state.py` derives one immutable `QuotaSnapshot` per account
(keyed by `api_key_env`, so two entries that share a key share one snapshot
and one poll — `test_shared_api_key_env_shares_one_snapshot_and_poll`).

- **Per-window headroom is `100 − percent`**, clamped to `[0, 100]`.
- **`effective_headroom` is the minimum of the three window headrooms**
  (`rolling`, `weekly`, `monthly`). The monthly figure alone is **not**
  decisive: a spent rolling (5-hour) or weekly window blocks more traffic even
  when most of the monthly allowance is intact. Feeding
  `rolling=95, weekly=10, monthly=40` yields `effective_headroom = 5`
  (`test_derivation_reports_windows_and_minimum_headroom`).
- **A window whose `status` is anything other than `"ok"`, or is entirely
  absent, is treated conservatively as blocking** and contributes `0`
  headroom — unknown status is never optimistically trusted
  (`test_non_ok_window_status_is_blocking`, `test_missing_window_is_treated_as_blocking`).
- **A missing or unparseable `resetsAt` yields `seconds_to_reset <= 0`**, never
  a guessed reset (`test_missing_or_invalid_resets_at_yields_zero`). This
  retires the class of failure behind LP-0MUQTCMW2001VXH8, where a guessed
  reset locked an account out for up to 30 days.
- **The account-level `seconds_to_reset` reports the monthly window** — the
  allowance period that resets per account. The rolling and weekly windows
  reset on much shorter cadences and are exposed per window for callers that
  need them.
- **An account is `exhausted`** when any window blocks or no effective
  headroom remains.

The poller is **interval-gated** (at most one poll per distinct `api_key_env`
per `quota_poll_interval_seconds`) and **single-flighted** (concurrent pollers
coalesce into one in-flight request per account), so a burst of client requests
can never multiply upstream `/v1/usage` calls
(`test_interval_gating_allows_one_poll_per_interval`,
`test_concurrent_refreshes_coalesce_into_one_request`). Only `opencode-*`
entries (or an explicit `quota_surge_accounts` allow-list) are polled at all
(`test_non_opencode_entries_are_not_polled`).

## 4. Pacing and scoring model

The producer decision (epic Q2 → option 2c) is a per-account **urgency
score**:

```
score = effective_headroom × quota_pacing_target / seconds_to_reset
```

i.e. "percent of the monthly allowance at risk per second until reset",
scaled by the pacing target so the default `0.95` aims to finish at ~95%
consumed rather than exactly 100%. The score is:

- **monotonically non-decreasing in headroom** at a fixed time-to-reset
  (`test_urgency_score_is_non_decreasing_in_headroom`), and
- **non-decreasing as the reset approaches** at fixed headroom
  (`test_urgency_score_is_non_decreasing_as_reset_approaches`).

Within the `opencode-go*` chain, eligible entries are sorted by **descending**
score and spliced back into the positions the quota entries occupied, so every
non-quota provider keeps its configured position. The sort is stable, so equal
scores (and entries with no snapshot) tie-break on config order deterministically
(`test_prefers_account_with_more_headroom`,
`test_preference_reverses_with_percentages`,
`test_equal_headroom_preserves_config_order`).

### 4.1 Achievability — why the ramp starts before the final week

The documented Go caps bound how fast a residual can physically be discharged:

- a **rolling 5-hour window at 20%** of the monthly cap, and
- a **weekly window at 50%** of the monthly cap.

Over `T` seconds the caps permit `min(0.20·T/5h, 0.50·T/7d)` of the monthly
allowance; the weekly cap is always the binding term at long horizons
(`achievable_fraction`, `cap_rate_per_second` —
`test_achievable_fraction_follows_the_documented_cap_relationship`,
`test_cap_rate_is_the_binding_sustainable_rate`). A large residual therefore
**cannot** be discharged in a last-minute burst: with 90% left and 2 days to
reset the weekly cap allows at most ~14% more, so the ramp must already have
started — even though the reset is *before* the final week. Conversely, 10%
left with 6 days to reset is comfortably spendable, so there is no ramp.
`ramp_started` returns `True` exactly when the residual exceeds the achievable
fraction (`test_ramp_started_achievability_table`,
`test_ramp_started_is_reported_outside_the_surge_window`).

The urgency score is capped at the caps' binding sustainable rate, so no
returned score ever implies spending more than a rolling (20%) or weekly (50%)
cap in one window (`test_no_score_exceeds_a_window_cap`,
`test_score_sweep_never_breaches_a_window_cap`). Outside the surge window and
outside the ramp condition the model returns the baseline preference `0.0`, so
a caller that sorts by score behaves exactly as it did before the feature
(`test_baseline_outside_surge_window`).

The **final-week surge window** is `0 < seconds_to_reset ≤ quota_surge_days ×
86400` (default 7 days), measured per account from its own monthly `resetsAt`
(`surge_window_active`, `test_surge_window_boundaries`).

## 5. Peak/off-peak trade-off

DeepSeek V4.1 Flash is priced at **$0.15/$0.60 per 1M tokens off-peak** and
**$0.30/$1.20 peak** (peak hours `01:00-04:00` and `06:00-10:00` UTC, Mon–Fri,
which the existing `available_times` windows deliberately avoid for
pricing reasons). Because the monthly allowance is a **dollar** amount, tokens
spent in peak hours deplete it **twice as fast** — `PEAK_BURN_MULTIPLIER =
2.0` (`test_burn_multiplier_represents_the_2x_peak_rate`).

The model therefore treats peak spend as a last resort: `peak_escalation_allowed`
is `False` whenever the window is already `available_times`-eligible (off-peak),
and is only allowed when `ramp_started` says waiting off-peak would strand the
residual (`test_peak_escalation_only_when_otherwise_unspendable`). Surge never
relaxes any other gate for this purpose — peak escalation is a pricing/urgency
decision only.

In routing terms, once an account is inside its final week the
`available_times` restriction is lifted for **that account** (section 6), so
peak hours become usable; the 2× burn rate is what makes that escalation
effective at discharging an otherwise-unspendable residual. The per-assessment
`burn_multiplier`, `peak_escalation` and `off_peak_eligible` fields are exposed
by `quota_pacing.assess_pacing` for observability
(`test_assessment_exposes_peak_weighting`).

## 6. Safety

All quota-derived decisions are switched through the single predicate
`quota_state.quota_behaviour()`, which returns one of four modes —
`enabled`, `dry_run`, `suspended`, `disabled`. Ordering, the
`available_times` bypass and the quota-driven skip all branch on that one
object, so "all quota behaviour off" is provable by construction
(`test_behaviour_modes_and_precedence`).

- **Kill switch — `quota_surge_enabled: false`.** Turns every quota-derived
  decision off: no ordering, no bypass, no quota-driven skip. Routing is
  byte-identical to the configuration without quota state
  (`test_kill_switch_is_byte_identical_to_today`). The **separate cost guard**
  (`quota_exhaustion_percent`) is a money guard and stays active. The poller
  keeps running so state and observability stay current.
- **Dry run — `quota_surge_dry_run: true`.** Computes and logs the decisions it
  *would* take (`reason=quota_surge_dry_run`; DEBUG `quota_ordering` lines with
  `applied=False`) but leaves the selected provider, eligibility, `retry_after`
  and next-window values unchanged. The flag is re-read per request
  (`test_dry_run_leaves_routing_unchanged`,
  `test_dry_run_logs_the_bypass_it_would_apply`,
  `test_dry_run_keeps_eligibility_and_retry_after_unchanged`).
- **Stale-state fail-safe — `quota_state_max_age_seconds` (default 3600 s).**
  When the last successful poll for an account is older than this, surge is
  **suspended** for that account: no bypass, no quota-driven re-ordering or
  skip, so routing reverts to the configured `available_times` and config
  order. One WARN per account per suspension with
  `reason=quota_surge_suspended`, and one INFO on recovery
  (`test_stale_state_suspends_the_bypass_and_warns_once`,
  `test_stale_state_reverts_ordering_to_config_order`). Missing / never-polled
  state never suspends (nothing is stale) and never enables surge
  (`test_missing_state_never_suspends`).
- **Hard stop / cost guard — `quota_exhaustion_percent` (default 99.5).** An
  `opencode-*` account whose monthly consumption is at or above this percentage
  — or whose `effective_headroom` is at or below `100 − threshold` (a spent
  rolling or weekly window) — is **skipped before any upstream request is
  attempted**, logged at INFO with `reason=quota_exhausted`
  (`test_quota_skip_table`). The skip applies whether or not surge is active
  (`test_skip_applies_when_surge_disabled_or_dry_run`) and is cumulative with
  the reactive usage-limit quarantine — it never clears, shortens or overrides
  it (`test_gousagelimiterror_quarantine_and_hard_stop_are_cumulative`,
  `test_quota_skip_never_overrides_an_active_quarantine`). A mid-surge crossing
  stops surge for that account on the very next dispatch decision, with zero
  further requests to that entry (`test_mid_surge_crossing_hard_stops_next_dispatch`,
  `test_single_dispatch_followed_by_crossing_is_the_only_request`). This is the
  guard that prevents spending real pay-as-you-go money on an account with the
  console "Use balance" option enabled. Values outside `[1, 100]` are clamped
  and non-numeric values fall back to the default, with one WARN per process
  (`test_invalid_values_are_clamped`, `test_invalid_threshold_warns_exactly_once`).
- **The final-week surge bypasses `available_times` only.** Nothing else is
  relaxed: entry cooldown, `rate_limit_rpm` request budget, the sibling-failure
  circuit breaker, failure-domain grouping, usage-limit quarantine and
  stall/quarantine gates all keep running unchanged
  (`test_rate_limited_entry_still_skipped_under_surge`,
  `test_circuit_broken_entry_still_skipped_under_surge`,
  `test_usage_limit_quarantined_entry_still_skipped_under_surge`,
  `test_failure_domain_sibling_still_excluded_under_surge`). An exhausted entry
  is never bypassed (`test_exhausted_entry_is_not_bypassed`), the bypass clears
  once the reset passes (`test_bypass_clears_once_the_reset_passes`), and it is
  per account, not global (`test_bypass_window_is_per_account`).

## 7. Configuration reference

All keys live under `server` in `proxy/config.yaml` (the readers also accept
the flat spelling). Each is documented inline next to its default in
`proxy/config.yaml`; the drift test
[`proxy/tests/test_quota_config_docs.py`](../../proxy/tests/test_quota_config_docs.py)
asserts that the values below match both the on-disk config and the code's
absent-key fallback.

| Key | Default | Purpose |
|---|---|---|
| `quota_surge_enabled` | `true` | Master kill switch for all quota-derived decisions |
| `quota_surge_dry_run` | `false` | Compute/log decisions without applying them |
| `quota_poll_interval_seconds` | `900` | Minimum gap between `/v1/usage` polls per account |
| `quota_surge_days` | `7` | Length of the final surge window, in days (per account) |
| `quota_exhaustion_percent` | `99.5` | Hard stop: monthly percent at/above which an account is skipped |
| `quota_state_max_age_seconds` | `3600` | Age beyond which stale state suspends surge (fail-safe) |
| `quota_pacing_target` | `0.95` | Fraction of the residual the ramp aims to spend by `resetsAt` |
| `quota_surge_accounts` | *(unset = all `opencode-*`)* | Optional allow-list of entry names or `api_key_env` values |

## 8. Operational runbook

### 8.1 Disable quickly

1. **Stop all quota-derived behaviour without a restart** — set
   `server.quota_surge_enabled: false` and reload the config (the flag is read
   per request through the central predicate). Routing becomes byte-identical
   to pre-feature behaviour; only the money-safe hard stop stays active.
2. **Stop it from *changing* routing but keep alerting** — set
   `server.quota_surge_dry_run: true`; the decisions are logged with
   `reason=quota_surge_dry_run` but never applied.
3. **Stop the poller entirely** — restart the proxy; the poller is started and
   stopped with the server lifecycle (`start_quota_poller` /
   `stop_quota_poller`). Do not stop the proxy by killing processes; use the
   `start-proxy` skill (`/skill:start-proxy --restart`).
4. **Restrict the blast radius** — set `server.quota_surge_accounts` to a
   single entry/`api_key_env` and only that account is managed.

### 8.2 Read the new metrics

Prometheus (`GET /metrics`) exposes, per `{account, entry}`:

| Metric | Meaning |
|---|---|
| `quota_rolling_percent` / `quota_weekly_percent` / `quota_monthly_percent` | Percent consumed in each window |
| `quota_effective_headroom` | Remaining percent across the binding (minimum) window |
| `quota_seconds_to_reset` | Seconds until the monthly reset |
| `quota_resets_at_timestamp_seconds` | Absolute monthly reset time (Unix seconds) |
| `quota_surge_active` | `1` while surge is active for the account |
| `quota_exhausted` | `1` when the account is exhausted |
| `quota_last_poll_timestamp_seconds` | Last successful poll time |
| `quota_poll_failures_total{account,reason}` | Failed `/v1/usage` polls by reason |

The same per-account state is carried in the `quota` section of the `/events`
SSE status payload and rendered as a cell on the home page
(`quota <N>% eff · monthly <M>% · resets <countdown> · surge`), degrading
cleanly (`quota: null`, no cell) when no state has been polled yet. No API key
**value** is ever exposed — only the account key derived from `api_key_env`
(`test_metrics_gauges_carry_the_exposed_values`,
`test_status_events_payload_carries_the_quota_section`,
`test_home_page_rows_show_quota_fields`,
`test_no_api_key_value_is_exposed_on_any_surface`).

Log lines an operator can grep:

| Line | Level | Meaning |
|---|---|---|
| `reason=quota_surge` | INFO | Final-week bypass entered/left for an account (once per transition) |
| `reason=quota_exhausted` | INFO | Account skipped by the hard stop |
| `reason=quota_surge_dry_run` | INFO | Would-be decision (dry run) |
| `reason=quota_surge_suspended` | WARN | Stale state suspended surge for an account |
| `reason=quota_surge_resumed` | INFO | Surge recovered after a fresh poll |
| `quota_ordering` (`applied=…`) | DEBUG | Per-request ranking decision |
| `quota poll failed account=… error=…` | WARN | Failed poll; last-known values kept |

### 8.3 Confirm surge is working after deploy

1. **Poller is alive** — `quota_poll_failures_total` is not climbing and
   `quota_monthly_percent` is non-zero for the accounts in use; the home page
   shows a `quota …` cell for each `opencode-*` entry.
2. **Derivation is sane** — `quota_effective_headroom` equals the minimum of
   the three percent gauges (`100 − percent`); `quota_seconds_to_reset` counts
   down toward `quota_resets_at_timestamp_seconds`.
3. **Surge is active where expected** — for an account inside its final week,
   `quota_surge_active == 1`, the home page shows `surge`, and a
   `reason=quota_surge` INFO line marks the entry transition.
4. **Routing follows the score** — with two eligible accounts, DEBUG
   `quota_ordering` lines show the higher-headroom/nearer-reset account ranked
   first; reversing the percentages reverses the order.
5. **Guards hold** — `quota_exhausted == 1` accounts produce
   `reason=quota_exhausted` skips with no upstream request; a stale account
   (`quota_last_poll_timestamp_seconds` older than
   `quota_state_max_age_seconds`) logs `reason=quota_surge_suspended` and falls
   back to configured `available_times`.

The end-to-end flow (poll → derive → order → surge → hard stop → stale
suspend) is exercised as a single integration test against a fake upstream
(`test_full_flow_poll_order_surge_hard_stop_stale`), and the pre-existing
routing gates are asserted unaffected under surge
(`test_pre_existing_routing_gates_unaffected_under_surge`).

## 9. Post-deploy measurement (F10, LP-0MVF7KTMI005B667)

Closes the loop on the epic's outcome: per account, how much of the monthly
allowance was spent **at the moment of reset**, and whether the target
(≥ 90% consumption at reset, ≤ 10% residual) was met. The measurement is
log-derived, deterministic and reproducible — the tool is
`scripts/quota_allowance_report.py` (tested in `tests/test_quota_allowance_report.py`),
which reuses the shared rotation-aware log discovery (`scripts/lib/proxy_logs.py`)
and mirrors the rule-based (no-LLM) pipeline of the
`.pi/skills/proxy-usage-analysis` skill.

### 9.1 When to measure (per-account schedule)

Resets are per account and **not calendar-aligned**. The first *full* period
after deployment ends at each account's first reset **after** 2026-10-19:

| Account | First reset after deploy | First full post-deploy period ends |
|---|---|---|
| `OPENCODE_API_KEY` | 2026-10-23 | **2026-11-23** |
| `OPENCODE_2_API_KEY` | 2026-10-30 | **2026-11-30** |
| `OPENCODE_3_API_KEY` | 2026-11-17 | **2026-12-17** |

Run the report once after **each** account's reset (or once after 2026-12-17
for the full set) — the tool reports every completed period in the window,
so a later run covers all of them retroactively.

### 9.2 Exact invocation (recorded for reproducibility, AC13d)

```bash
python3 scripts/quota_allowance_report.py \
    --log-dir /var/log/llama-proxy \
    --start 2026-10-19T18:00:00 \
    --end 2026-12-18T00:00:00 \
    --output-md docs/dev/quota-surge-post-deploy-2026-12-18.md \
    --output-json docs/dev/quota-surge-post-deploy-2026-12-18.json
```

The window **must start at deployment** (2026-10-19 ~18:46 UTC, when quota
polling began) so the first period's consumption is attributed to the post-epic
deployment. The invocation (argv, window, filters) is embedded in the JSON
artifact; re-running the same command over the same logs reproduces the same
figures byte-for-byte.

### 9.3 What the report contains

Per account, per completed period: the monthly percentage at reset (the last
poll at/before `resetsAt`), the residual (`100 − percent`), the coverage gap
(how far that reading was from the reset), surge-active hours (total and
inside the final 7 days), hard-stop dwell, quota-driven skips (applied vs
dry-run), poll failures and stale suspensions. Outcomes:

- `target_met` — ≥ 90% at reset.
- `below_target` — < 90%: **follow-up work item** with a concrete tuning
  proposal (`quota_pacing_target`, `quota_surge_days`, ramp start) and the
  observed data (AC13c).
- `overspend` — at/above `quota_exhaustion_percent` at reset: the guard did
  not hold; **critical** follow-up immediately (AC13c).
- `insufficient_data` — no usable reading near the reset.

Surge hours come from the `quota_surge_bypass entered|left` transitions,
which the proxy logs only when an entry is evaluated **outside** its
configured `available_times` window (the off-peak gaps `01:00–04:00` and
`06:00–10:00` UTC; inside `10:00–00:00`/`00:00–01:00`/`04:00–06:00` the
window already allows dispatch, so no transition is emitted). Zero surge
hours therefore means "no out-of-window dispatch was attempted", not "the
surge feature was off" — a conservative lower bound on bypass use.

Dwell at the hard stop is derived from consecutive polls at/above
`quota_exhaustion_percent`, so it is reported only while polling covers the
episode (bounded by the 15-minute interval).

### 9.4 Baseline (2026-10-19)

`docs/dev/quota-surge-post-deploy-2026-10-19-baseline.{md,json}` record the
start-of-measurement state so a later period report can be diffed against it.
No completed period existed at that point (nothing to classify).

Leading indicators from the baseline (as of deployment evening, 2026-10-19):

- `OPENCODE_API_KEY`: 93% monthly, reset 2026-10-23 — inside its final
  week. No `reason=quota_surge` transition had been logged by the baseline
  because the evening window (`10:00–00:00`) already permits dispatch and
  transitions only fire for out-of-window evaluations (`01:00–04:00`,
  `06:00–10:00` UTC).
- `OPENCODE_2_API_KEY`: pinned at 95% monthly for the whole observed window
  (rolling/weekly 0% → effective headroom 5, ranked below the other accounts)
  — on track to finish ≥ 90% at its 2026-10-30 reset.
- `OPENCODE_3_API_KEY`: 13% monthly, reset 2026-11-17, with a flat spend
  curve — it is the at-risk account. It is ranked first by urgency score but
  receives almost no dispatch, so whether it closes the gap depends on how
  much traffic the chain gets at all (routing/steering, outside quota surge's
  control). The 2026-12-17 measurement will confirm the outcome; if it
  finishes below 90% the follow-up proposal will additionally flag the
  `quota_surge_accounts` / dispatch-volume question.

## 10. References

- `proxy/quota_state.py` — poller, derivation, cache, central predicate.
- `proxy/quota_pacing.py` — pure pacing/urgency/peak model.
- `proxy/provider.py` — headroom ordering, final-week bypass, hard stop.
- `proxy/metrics.py`, `proxy/ui.py` — observability surfaces.
- Epic discovery findings (2026-10-19) — LP-0MVF4FIP1001RD8P.
- Related: LP-0MS4ETBNO0022QAC (`available_times`), LP-0MSLJPOCC0001ROJ
  (`GoUsageLimitError` quarantine), LP-0MUQTCMW2001VXH8 (retired guessed
  reset), LP-0MVDQ1GMO00801PL (per-provider request budget).
