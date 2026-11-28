## Wasted GPU compute (summary)

GPU compute burned on work no client received (dispatch accepted, then released before its first byte). Detail and definitions follow in **Wasted GPU compute** and the appendix. Thresholds are the producer-agreed criterion on LP-0MVN3KMGI009BZ0D.

| Figure | Value | Target | Verdict |
|---|---|---|---|
| Post-acceptance evictions | 5 (0.50/h) | minimise | — |
| User-class prefill failures (`prefill_deadline` + `measured_cache_deficit`) | 3 (0.30/h) | ≤ 0.5/h | PASS |
| Occupancy-attributed share | 0% of 3 deadline events | ≤ 25% | PASS |
| Capacity-attributed share | 0% | < 50% | PASS |
| Wasted GPU tokens (priced prefill discarded at the deadline + evaluated after cancel) | 9000 | minimise | — |
| Restart-free window | no — 0 proxy / 1 backend restart(s) | required | FAIL |
| User-class prefill **stall** (Σ elapsed s/h, `prefill_deadline` + `measured_cache_deficit`) | 36.1 s/h (361 s over 3 events) | minimise | — |
| **Overall** | | | **NOT CERTIFIED** |

Caveats: no measured prefill-progress capture in this window: the attribution rests on the admission price alone; no llama-server log supplied: abandoned-prefill overlap, release->cancel latency and backend waste are unavailable; window contains 1 backend restart(s) that discarded in-flight dispatches: not certifiable (a restart with no in-flight dispatch leaves no eviction event, so this is a lower bound); restart-free window required for a certifiable rate; use a window that starts after the last proxy/backend restart.

## Wasted GPU compute

Cause and cost of the dispatches released after acceptance in this window. Rates use the analysed window length; `user` excludes `audit-*` sessions, which are agent gate runs rather than served traffic. Attribution covers `prefill_deadline` events only: `predicted-warmth` = we priced a warm prefix that did not exist, `capacity` = an honest price the GPU still could not deliver, `occupancy` = another tenant held the slot pool.

| Reason | Total | User | Audit | Per hour | User per hour |
|---|---|---|---|---|---|
| `prefill_deadline` | 3 | 2 | 1 | 0.30 | 0.20 |
| `backend_restart` | 1 | 1 | 0 | 0.10 | 0.10 |
| `measured_cache_deficit` | 1 | 1 | 0 | 0.10 | 0.10 |

| Attribution | Events | Share |
|---|---|---|
| predicted-warmth | 3 | 100% |
| capacity | 0 | 0% |
| occupancy | 0 | 0% |

| Waste | Value |
|---|---|
| Priced prefill tokens discarded at the deadline | 9000 |
| Tokens evaluated after the backend cancel | 0 |
| Total wasted GPU tokens | 9000 |

| Release → cancel | Value |
|---|---|
| Releases with an observed backend cancel | 0 of 0 |
| Cancel → backend stop (p50 / p90) | unavailable |
| Abandoned-prefill overlap (basis `legacy_rate_estimate`) | 0 of 3 deadline events |

#### Stall cost (what the event rate cannot see)

The rows above count events; these price them in **seconds a user waited**. A `measured_cache_deficit` detour that aborts inside a second and a `prefill_deadline` stall that burns the whole 180 s backstop are +1 each to the rate and orders of magnitude apart here — which is why making aborts faster cannot move the rate but does move this table.

| Reason | Events (user / audit) | User stall | User s/h | p50 | p90 | p95 | p99 | Under 1 s | Under 5 s | At/near deadline |
|---|---|---|---|---|---|---|---|---|---|---|
| `prefill_deadline` | 3 (2 / 1) | 360.2 s | 36.0 s/h | 180.1 s | 180.1 s | 180.1 s | 180.1 s | 0 (0%) | 0 (0%) | 2 (100%) |
| `backend_restart` | 1 (1 / 0) | 3.1 s | 0.3 s/h | 3.1 s | 3.1 s | 3.1 s | 3.1 s | 0 (0%) | 1 (100%) | 0 (0%) |
| `measured_cache_deficit` | 1 (1 / 0) | 0.4 s | 0.0 s/h | 0.4 s | 0.4 s | 0.4 s | 0.4 s | 1 (100%) | 1 (100%) | 0 (0%) |

**User-class composite** (the (b2) population, stall-weighted): 3 event(s), 360.6 s of user waiting, **36.1 s/h** — p50 180.1 s, p90 180.1 s, p95 180.1 s, p99 180.1 s; under 1 s 1 (33%), under 5 s 1 (33%), at/near the deadline 2 (67%).

**Whole window** (every reason, user + audit): 5 event(s), 543.8 s of waiting, 54.4 s/h.

Caveats: no measured prefill-progress capture in this window: the attribution rests on the admission price alone; no llama-server log supplied: abandoned-prefill overlap, release->cancel latency and backend waste are unavailable; window contains 1 backend restart(s) that discarded in-flight dispatches: not certifiable (a restart with no in-flight dispatch leaves no eviction event, so this is a lower bound); restart-free window required for a certifiable rate; use a window that starts after the last proxy/backend restart.

## Appendix: wasted GPU compute — definitions and basis

| Term | Definition |
|---|---|
| Post-acceptance eviction | `local_eviction_after_acceptance`: the dispatch was routed to the local model and released before the client received a first byte |
| `prefill_deadline` | the hard backstop expired with no first byte: the client waited the whole deadline, then was detoured |
| `measured_cache_deficit` | the measured prefill contradicted the warm-cache price, so the dispatch was released early (a cheaper outcome for the same defect) |
| User-class composite rate | `prefill_deadline` + `measured_cache_deficit` for non-`audit` sessions, per hour. The two reasons are counted together so a reason conversion cannot pass as the defect disappearing |
| Occupancy / capacity share | share of `prefill_deadline` events attributed to another tenant / to an honest price the GPU could not deliver; the guards are occupancy ≤ 25 % and capacity < 50 % |
| Priced prefill tokens discarded | the `new_tokens` each deadline event had priced as new compute, summed: GPU work promised and lost |
| Tokens evaluated after cancel | tokens the backend computed after the release (≤ one `ubatch` by design): the measured tail of waste |
| Restart-free window | no proxy restart (analyser re-base events) and no backend restart that discarded an in-flight dispatch. A restart both re-prices capacity and drops prefills, so a window with one cannot certify anything |
| Stall-weighted rate | `Σ elapsed_s / window_hours` over the same events the per-event rate counts, split by reason and by user / `audit` session. It prices **how long users waited**, where the per-event rate prices **how many dispatches failed**. Both are reported because neither alone answers the residual question |
| Stall percentiles | p50 / p90 / p95 / p99 of `elapsed_s` per reason for user traffic: the shape of the wait, not just its total |
| Under 1 s / under 5 s / at-near deadline | the share of a reason's events that ended within 1 s, within 5 s, or at (within 1 s of) the prefill deadline. A reason whose events are mostly under 1 s is the early detector *working*, not a stall |

**`measured_cache_deficit` and `prefill_deadline` are not equivalent.**

The user-class composite rate counts the two reasons together so that a *conversion* between them cannot be mistaken for the defect disappearing. That protects the count, but it deliberately treats every event as equally costly, and they are not:

- A **`measured_cache_deficit`** release is the early detector **working**: the measured prefill contradicted the warm-cache price, so the dispatch was abandoned and detoured after a short wait. Making it faster is an improvement, and it leaves the event rate unchanged.
- A **`prefill_deadline`** release is the client **waiting the whole backstop** (180 s by default) and then being detoured anyway. It is the worst outcome for the user and the one the deadline exists to bound.

Measured in this window (user traffic, `elapsed_s` to the release): `measured_cache_deficit` p50 0.4 s / p90 0.4 s over 1 event(s), 1 (100%) of them under 1 s, **0.0 s/h** in total; `prefill_deadline` p50 180.1 s / p90 180.1 s over 2 event(s), **36.0 s/h** in total against a 180 s backstop. The event counts are comparable; the waiting is not.

Read the per-event rate for **whether the defect is occurring** and the stall-weighted rate for **what it costs**. A re-scope choosing between an event-rate target and a stall-cost target needs both.

**Measurement basis.** `legacy_rate_estimate` (`measured_stop` joins each release to the backend's observed cancel→stop latency; `legacy_rate_estimate` bounds it by `tokens ÷ 43.1 tok/s` and overstates the backlog by two to three orders of magnitude). Backend log coverage: `none supplied`.

**Thresholds.** ≤ 0.5 user-class events/h; occupancy ≤ 25 %; capacity < 50 %; measured over a restart-free ≥ 24 h window (producer decision, option b2, recorded on LP-0MVN3KMGI009BZ0D AC4).

**Caveats.**

- no measured prefill-progress capture in this window: the attribution rests on the admission price alone
- no llama-server log supplied: abandoned-prefill overlap, release->cancel latency and backend waste are unavailable
- window contains 1 backend restart(s) that discarded in-flight dispatches: not certifiable (a restart with no in-flight dispatch leaves no eviction event, so this is a lower bound)
- restart-free window required for a certifiable rate; use a window that starts after the last proxy/backend restart
- A backend restart with no dispatch in flight leaves no eviction event, so the restart-free flag is a lower bound on restarts.
- The attribution rules, abandoned-prefill model and backend accounting are computed by `proxy/scripts/analyse_prefill_deadline_residual.py`; this report consumes its payload rather than recomputing them.
