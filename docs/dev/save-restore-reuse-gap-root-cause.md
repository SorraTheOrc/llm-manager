# Save/restore reuse-gap root cause (F2)

**Work item:** LP-0MTCMEOHB002X1JN (parent LP-0MTAQNB7J0094X71)
**Window:** 2026-08-26 (proxy events) + llama-server.log-2026-08-27.gz (native
checkpoint events; llama-server logs carry no timestamps, so day attribution
is per-file, see the F1 harness docs)
**Reproduction:** `python3 scripts/slot_reuse_gap_analysis.py --log-dir
/var/log/llama-proxy --start 2026-08-26 --end 2026-08-27 --llama-file
'*2026-08-27*'` — every number below is regenerable from the log snapshot.

## TL;DR

There are **two independent persistence mechanisms**, and the incident's
"~95% checkpoints unrestored" conflates them:

1. **Proxy slot save/restore** (the mechanism designed for session↔slot
   reuse): `slot_save` / `slot_restore` events in proxy.log. On the incident
   day it restored at **89.7%** (926 saves → 831 restores) — it works…
   *when it runs*.
2. **llama-server native context checkpoints** (`created context checkpoint`
   in llama-server.log): an internal, ephemeral mechanism used to snapshot
   in-flight KV state mid-processing. Rather than a "session reuse" API, it
   is a recovery aid and its restore rate (154/3191 = **4.83%**) is not a
   session-reuse metric.

The **real reuse gap** is not that restores fail — it is that persistence is
**skipped entirely for the sessions that need it most**. Context-size gating
(`routing_skip_local reason=context_too_large`) excluded **1,902 requests
across 38 sessions** from local dispatch on 2026-08-26, and those sessions
get **zero** slot_save/slot_restore events (0 saves, 0 restores). Sessions
with the most repeated context — 409, 267, 207, 199, 183, 161 routing skips
— never persisted even once.

## Factor breakdown (2026-08-26)

| Factor | Count | Dominance |
|---|---|---|
| **Context-size gating** (`context_too_large` skips) | 1,902 (of 2,359 skips; 457 `large_context_bypass`) | **DOMINANT** — 38/48 sessions gated out entirely |
| Proxy slot persistence success | 926 saves / 831 restores = **89.7%** | Healthy when triggered |
| llama-server native checkpoints | 3,191 created / 154 restored = **4.83%** | Ephemeral native mechanism; not the session-reuse path |
| Leases (churn / affinity) | 3,186 events; 166 orphan releases + 141 evicted (307 affinity breaks) | Secondary — contributes to session↔slot instability |
| `slots_stale` | 3,780 of 7,980 polls (47.4%) | Secondary — degrades the proxy's slot-state view |
| GET /slots HTTP 500 | 6,865 of 73,003 polls (9.4%) + 527 HTTP 400 | Secondary — poll failures bias the slot-state view |
| Load-aware gating | 2 save failures, both `busy_info.slot_busy=true` | Negligible on this day (LP-0MSI1RWLM007N367 handled) |
| Circuit-breaker cooldown | 0 `persistence disabled` events | None observed |

## Why checkpoints are saved but not restored

### Mechanism confusion (the headline)

The incident's "2,954 checkpoints saved vs 145 restored" is a count of
llama-server's **native** `created context checkpoint` events, not proxy
slot persistence. Evidence:

- Proxy slot persistence restored at **89.7%** on the same day (926 saves →
  831 restores) — if the proxy's mechanism were broken at 95%, restores
  would be ~46, not 831.
- Native checkpoint restores are almost entirely **same-slot, same-task**
  (slot 0: 74 restores, 9 foreign; slot 1: 40/6; slot 2: 40/11) — i.e. a
  task resuming in the slot it was interrupted in. This is in-flight
  recovery, not cross-turn session reuse.
- Native checkpoints are **strictly smaller** than the proxy's persistence
  target: checkpoints >50K tokens restored at 37/429 = 8.6%, and the 
  proxy never calls llama-server `prompt_load` (0 load lines) — the proxy
  does not read the native checkpoint store for session reuse.

So "95% of checkpoints never restored" describes the native checkpoint
mechanism's recovery semantics, not a failing reuse pipeline. The reuse
pipeline (proxy slot persistence) is separate and healthy when invoked.

### The actual reuse gap: persistence is skipped for oversized sessions

`_build_slot_context` (proxy/proxy/session.py) skips persistence when the
request context exceeds `session_slot_max_prompt_tokens` (config: 83,285
class cap). On 2026-08-26:

- 1,902 requests were skipped local with `reason=context_too_large` (plus
  457 `large_context_bypass`).
- 38 of 48 sessions (79%) had routing skips with **zero** persistence events.
- The heaviest-reuse sessions never persisted: 409/267/207/199/183/161
  repeated oversized checks, 0 saves each.

These are exactly the sessions that would benefit most from KV reuse
(multi-hundred-KB contexts re-prefilled every turn). The size gate that
protects against GPU wedging (LP-0MS91DHPZ001VWQO) also disables reuse for
the sessions where reuse saves the most.

### Secondary contributors

1. **slots_stale (47.4% of status polls)** — the proxy's slot-state view is
   stale almost half the time, weakening session↔slot affinity decisions
   (which session owns which slot), so even when a slot could be restored,
   the proxy may not trust/act on its slot registry.
2. **GET /slots 500s (9.4%)** — llama-server access-log errors on /slots
   polls feed the staleness (LP-0MSVP7XJ6008QPKX / LP-0MSB0RV72001KNRV
   fixed other /slots issues; the residual 9.4% remains).
3. **Lease churn (166 orphan + 141 evicted releases)** — orphan-cleanup
   releases break session→slot continuity; a session whose lease was
   reclaimed between turns cannot restore to "its" slot.

None of these alone explain the gap; they compound the primary context-gate
exclusion.

## Timeline correlation: snapshot writes 22:02–23:09

The incident's 22 `slot_*.bin` snapshots (100–560 MB) align with the
22:00–23:09 window, consistent with a burst of **saves that DO happen** for
the 3 persisted sessions in that window (42 saves / 32 restores 22:00–23:09
in proxy.log-2026-08-27_00.gz) — while simultaneously the oversized-session
gate fired 87 skips @ 22:00, 54 @ 22:30, 163 @ 23:00 and `slots_stale`
spiked (151/111/118 polls). So even during the save burst, the sessions
being saved are a small minority (3 sessions) while the context-gated
majority keeps re-prefilling remote.

## Root cause, ranked

1. **Size-gating excludes the reuse-neediest sessions** (dominant): the
   `session_slot_max_prompt_tokens` cap means oversized sessions never get
   persistence → 42.7M prefill tokens/day with no restore path.
2. **Mechanism conflation inflated the incident's "95%" claim**: the native
   checkpoint restore rate is a recovery metric, not the reuse rate; proxy
   slot persistence restored at 89.7%.
3. **slots_stale + /slots 500s + lease churn** (secondary): degrade the
   proxy's slot-state view and session↔slot affinity, compounding (1) when
   the proxy does attempt restore.

## Fix direction (feed F4/F5)

- **Cap derived and validated** (LP-0MTIFR5W3006UAX8 / LP-0MTE9HAF8008909G): the
  persistence cap is now derived from the per-slot routing clamp via
  `effective_per_slot_threshold` (83285 fast, 126976 cheap). Integration tests
  (`test_persistence_cap_modes.py`) verify save→restore cycles at both caps,
  restore-rate baselines (>80% proxy slot restores for >50K contexts), and GPU-
  wedge safeguards (adaptive timeout, circuit breaker, skip-when-busy). The
  `session_slot_max_prompt_tokens: 0` config triggers the dynamic derivation.
- Fix the residual /slots 500s + slots_stale so the proxy's slot-state view
  is trustworthy for affinity decisions.
- Consider lease continuity (fewer orphan releases) so sessions keep their
  slot mapping between turns.

## Validated assumptions

- llama-server native checkpoint format is compatible across restarts:
  **NOT verified on this day's data** — 0 proxy `prompt_load` calls, so the
  proxy never exercises cross-restart native restore; the assumption is
  untested and noted (parent A/R).