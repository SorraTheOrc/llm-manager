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
| Tenant-limit denials | `residency_tenant_limit_denials` | present but modest | dominating ⇒ the bound is too tight for the workload; raise `local_session_tenants` or lower the idle-lapse |
| Evictions | `local_evictions` by reason | roughly tracks disconnects/cancels | `tenancy_lease_expired` dominant ⇒ sessions are being abandoned silently (client-side problem, proxy is doing the right thing) |
| Remote spend | report summary, local vs remote | no step change | a large local→remote shift ⇒ the 0.5 default is too aggressive for this workload (see Risks in the parent item) |
| `prefill_deadline_exceeded` | error section | falling | rising ⇒ the 24 h AC4 verification (`LP-0MVN44XB0009M33K`) is the place to measure it properly |

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
