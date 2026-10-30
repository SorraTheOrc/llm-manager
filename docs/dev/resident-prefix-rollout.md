# Resident-prefix admission: staged rollout, observation and rollback

**Work item:** LP-0MVN1IQTF0037WTG (child LP-0MVN4LC7N005XH7P). Design and
rung-by-rung behaviour: `proxy/docs/routing.md` →
*Residency-aware admission ladder*.

## What ships, and where it is gated

| Knob (`server:`) | Shipped | `0` means | Notes |
|---|---|---|---|
| `local_resident_prefix_min_ratio` | `0.5` | residency gate disabled (predicted-ratio decision as before) | the rollback switch |
| `local_session_tenants` | `2` (= `session_slot_pool_size`) | tenancy bound disabled | same-session re-entry always allowed |
| `local_session_tenancy_idle_seconds` | `30` (adaptive dispatch-lease base) | — (non-positive falls back to the default) | never ≥ the 180 s no-progress watchdog |

The ladder runs **only** for prompts above
`local_large_context_cold_cache_threshold`; everything below it routes exactly
as before. `local_idle_admit_enabled` / `local_idle_admit_seconds` still apply
and remain the **last** rung, so an idle GPU still serves cold work.
`contention_queue_*` semantics are untouched: a request denied by the residency
gate or the tenancy bound never enters the queue (same as today's
`large_context_bypass`).

## Staged rollout (three steps, each observable)

1. **Land on `dev`** (this is what the item's commits do). Full suite green at
   the pushed tip — the dev-push pre-push gate owns execution; the cached
   evidence is read with
   `python3 .pi/skills/test/scripts/run_tests.py --summary --suite all --json`
   (never re-run ad hoc during implementation).
2. **Deploy via the release** (`/skill:ship release`, `dev` → `main`). The
   deploy instant is what arms the 24 h AC4 verification item
   (`LP-0MVN44XB0009M33K`) — landing on `dev` does **not** start that clock.
3. **First report cycle after deploy.** Run the usage analysis for the first
   full window and confirm the rollout is *actually on*:

   ```bash
   python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
     --hours 24 --output-dir ~/proxy-usage-reports/rollout
   ```

   Expect in `report.md` / the JSON summary:

   - the `## Residency-aware admission` section exists;
   - `residency_decisions` > 0 with a split across `residency_resident_admits`,
     `residency_evicted_denials` and (usually small) `residency_tenant_limit_denials`;
   - the per-session slot-owner churn table is populated for churned sessions;
   - `local_evictions` counts appear with reasons from the closed vocabulary
     (`lease_evicted`, `slot_assignment_freed`,
     `resident_prefix_evicted_by_tenant`, `tenancy_lease_expired`).

   **Absence of the section means the ladder is not emitting**
   (`routing_resident_prefix` lines) — that is a finding, not a pass. Check the
   provider log level and that the deployed config carries the three knobs.

## What to watch in the first cycle

| Signal | Where | Healthy | Investigate when |
|---|---|---|---|
| Resident admits vs evicted denials | `## Residency-aware admission` | resident admits dominate for multi-turn sessions | nearly all denials ⇒ either genuine churn or a residency-view bug (check `residency_source`) |
| `residency_source=unknown` share | same section, per-decision lines | small | large ⇒ stale snapshot (the ~1 s poller is not refreshing) — the gate fails open, so local keeps working, but measurement is lost |
| `residency_source=measured_last_dispatch` share | same section, per-decision lines | most resident decisions (the deployed b8782 build reports no per-slot `context_tokens`, so `measured` is rare) | `session_slot` dominating ⇒ no prior dispatch was captured (non-streaming clients, or `local_measured_cache_abort_enabled: 0`) |
| `measurement=measured` share | same section | tracks the two measured sources | absent everywhere ⇒ neither measurement source is live |
| `measurement=unavailable:*` breakdown | same section, per-decision lines | dominated by `no_prior_measurement` while the capture feature ramps up | `stale_view` large ⇒ the ~1 s `/slots` poller is not refreshing; `capture_disabled` large ⇒ the rollback switch is off (or the config cannot be read); `no_slot` large ⇒ genuine churn |
| Tenant-limit denials | `residency_tenant_limit_denials` | present but modest | dominating ⇒ the bound is too tight for the workload; raise `local_session_tenants` or lower the idle-lapse |
| Evictions | `local_evictions` by reason | roughly tracks disconnects/cancels | `tenancy_lease_expired` dominant ⇒ sessions are being abandoned silently (client-side problem, proxy is doing the right thing) |
| Remote spend | report summary, local vs remote | no step change | a large local→remote shift ⇒ the 0.5 default is too aggressive for this workload (see Risks in the parent item) |
| `prefill_deadline_exceeded` | error section | falling | rising ⇒ the 24 h AC4 verification (`LP-0MVN44XB0009M33K`) is the place to measure it properly |

## Measurement sources at admission (LP-0MVT2PY3J00077W6)

The deployed llama.cpp b8782 build no longer reports per-slot
`context_tokens`, so the live `/slots` measurement is unavailable in
production. Each `routing_resident_prefix` decision therefore reports the best
*available* measurement, labelled by `residency_source` and the additive
`measurement=` field:

| `residency_source` | Source | `n_past_actual` |
|---|---|---|
| `measured` | live `/slots` `context_tokens` (cached snapshot, ≤ 5 s old) | the reported context size |
| `measured_last_dispatch` | llama-server's own `prompt_progress` `total` for the session's most recent local dispatch (`proxy.measured_prefill` store, keyed `(model, session)`) | that dispatch's total |
| `session_slot` | ownership only — the sentinel rung | `none` (no numeric ratio) |
| `non_resident` / `unknown` | no slot of this session's / no fresh view | `none` |

Rung order: `measured` → `measured_last_dispatch` → `session_slot` →
`non_resident`/`unknown`. The middle rung fires only while the session still
owns a live slot and while capture is enabled, and it prices admission from
`min(n_past_actual, estimated_tokens)` reusable tokens rather than the old
implicit "everything is reused" assumption — so a **grown** prompt whose
measured reuse fell below `local_resident_prefix_min_ratio` is now denied with
`resident_prefix_insufficient`. That is the intended AC4 rule (a) signal (a
visible cold-prefill admission), not a regression.

**Known limits.** A buffered (non-streaming) local dispatch carries no
`prompt_progress` events, so it records no measurement; and the stored
measurement is only valid while its session still owns the slot it was
captured against.

**Invalidation (LP-0MVT2QFAP0082BKR).** The `(model, session)` entry is
cleared wherever that ownership ends, so a stale `total` cannot be re-reported
as a numeric reuse ratio:

| Event | Where | Scope cleared |
|---|---|---|
| eviction (every reclaim reason) | `router_helpers._record_local_eviction` | all models for the session |
| slot release / reassignment | `session._free_slot_assignment` | all models for the session |
| model switch (and a crashed-backend restart) | `lifecycle.ensure_model_loaded` | everything except the target model's names; **all** entries on a restart |

Clearing never raises and never changes a non-residency decision. Two
backstops remain: admission still requires live slot ownership, and the store
is keyed by `(model, session)` so a decision only ever reads the request's own
model. After an invalidation the next decision for that session reports
`measurement=unavailable:no_prior_measurement` instead of a ratio — the
honest non-measurement, not a fabricated one.

**Rollback.** `local_resident_prefix_min_ratio: 0` restores the predicted-ratio
decision (the gate off, this rung never consulted);
`local_measured_cache_abort_enabled: 0` restores the pre-change
`session_slot` behaviour while leaving the residency gate itself on.

## Unmeasured-reason vocabulary (LP-0MVT2Q1SD009JLKD)

A decision that cannot be measured says **why**: every
`routing_resident_prefix` line carries exactly one `measurement=` value from a
closed vocabulary, so "the ladder cannot measure" is distinguishable from
"there is no slot to measure" and the unmeasured share of AC4 is attributable.

| `measurement=` | Trigger |
|---|---|
| `measured` | a real measurement (`measured` live `context_tokens`, or `measured_last_dispatch`) |
| `unavailable:capture_disabled` | slot-owning session, but `local_measured_cache_abort_enabled` is off — or the switch cannot be read (fail-open ⇒ capture not enabled) |
| `unavailable:no_prior_measurement` | slot-owning session, capture enabled, but nothing measured yet for `(model, session)` |
| `unavailable:no_slot` | a fresh view shows no slot of this session's (a measured zero) |
| `unavailable:stale_view` | the residency view is absent/stale (`residency_source=unknown`), or the decision was logged without residency facts (gate disabled) |

Precedence is deterministic (first match wins): `measured` →
`capture_disabled` → `no_prior_measurement` → `no_slot` → `stale_view`
(`provider.measurement_reason`). For a slot-owning session the capture
rollback wins over a missing prior measurement, so an operator can tell "the
feature is off" from "nothing was measured". `reuse_ratio=none` remains the
numeric-absent form for every non-measurement — the internal `0.0`/`1.0` gate
sentinels are never surfaced.

**Reading the split.** At steady state expect `measured` for resident
decisions. A rise in `no_prior_measurement` means streaming local dispatches
are not being captured (or are all non-streaming); `capture_disabled` means the
rollback switch is off; `stale_view` means the `/slots` snapshot is not being
refreshed; `no_slot` is genuine non-residency.

## Rollback

Single knob, no restart needed for the *decision* (config reload picks it up):

```yaml
server:
  local_resident_prefix_min_ratio: 0   # predicted-ratio decision everywhere
```

Then, if tenancy is the suspect rather than the residency gate:

```yaml
server:
  local_session_tenants: 0             # tenancy bound disabled
```

Both are the documented `0`-disables paths and are covered by
`proxy/tests/test_resident_prefix_config_docs.py` (the readers really do
behave that way, not just the comments). Eviction logging, snapshot-on-eviction
and the usage-report section stay active in both cases — they are observability,
not admission.

## What this rollout deliberately does not change

- `local_prefill_deadline_seconds` — owned by `LP-0MVN3KMGI009BZ0D` (90 → 120);
  the residency work neither reads nor rewrites it.
- `local_max_concurrent_prefills` — unchanged (prefill serialisation is a
  separate rolling decision, `LP-0MVGDUUV3001XFRJ`).
- `session_slot_pool_size` (the snapshot pool size) — unchanged; widening it to
  4 is evaluated separately in `LP-0MVN3KPEC007WWL5` (at 3 slots the per-slot
  context drops from 262,144 to 87,381 tokens, which excludes the long-context
  sessions this work is about).
