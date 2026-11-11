# Group 1 validation — hand-off and unblock record

**Work item:** LP-0MW8CS8NY0087XBV (child of the group-1 validation LP-0MW7V0S6A006DKZ4)
**Parent ACs:** AC7 (hand-off), AC8 (unblock dependents)

This record is the traceable hand-off for the group-1 validation. The measured
numbers are posted on the tracking item **LP-0MVO8H7VQ003B12I**; this file and
the two datasets below are the committed evidence the closure child
(LP-0MWBLEQ1S003MUTF) cites.

## Measured numbers (window 25.0 h, restart-free, 2026-11-10T14:18:26Z → 2026-11-11T15:18:26Z)

| Metric | Baseline | Group 1 window | Verdict |
|--------|---------:|---------------:|---------|
| User-class combined (`prefill_deadline` + `measured_cache_deficit`) | 2.223/h | **0.72/h** | NOT MET (≤ 0.5/h) |
| `prefill_deadline` user | 1.186/h | 0.16/h (4) |  |
| `measured_cache_deficit` user | 1.037/h | 0.56/h (14) |  |
| Attribution (n=4): occupancy / capacity / predicted-warmth | 15 % / 36.7 % / 48.3 % | **50 % / 50 % / 0 %** | predicted-warmth eliminated |
| Abandoned-window share (measured) | 3.3 % | 0 % (0/4) | target met |
| Zero-progress starvations while draining | — | 0 of 23 | target met |

Interpretation (full detail in the verdict):

- **Predicted-warmth is eliminated** (48.3 % → 0 %) — the group-1 measured-residency
  effect.
- The rate fell **≈ 68 %** but is **NOT MET** against ≤ 0.5/h; the residual is the
  **capacity/occupancy** deadline class that **group 2** (`LP-0MW7V2D13004ZO65`,
  published deadline token capacity enforced at admission) is specified to remove.

## Committed evidence

| Artifact | Commit | Path |
|----------|--------|------|
| Window + dataset | `abdf2d397` | `proxy/docs/validation/group1-prefill-deadline-residual.{json,md}` |
| Verdict | `4dad2ee4c` | `proxy/docs/validation/group1-prefill-deadline-verdict.md` |
| Restart evidence (PIDs / HEAD / `/health`) | — | comments on LP-0MW8COT80004L1SR; pre-restart log preserved as `proxy.log.2026-11-10_141824` |

## Unblock status (parent AC8)

- The group-1 validation parent **LP-0MW7V0S6A006DKZ4** dependency edges are
  satisfied: `LP-0MW7V292T0020G02` (measured-residency, in group-1 scope) is
  `completed`. `LP-0MW7V29MA009MCPI` is **out of group-1 scope** (producer Q1,
  2026-11-11) and carries its own deployed validation
  `LP-0MWBZVU04007OII2`; the parent was already reset `blocked` → `open` (Q3) and
  stays `open` until this epic advances.
- **Group-2 validation LP-0MW7V14DN007F127**: its group-1 dependency is now
  satisfied, but it **remains legitimately `blocked`** on
  `LP-0MW7V2D13004ZO65` (group-2 implementation epic) and on
  `LP-0MW7V2DK400365ER` (now `completed`). Its own AC9 clears the `blocked`
  status once all blockers are terminal — it must **not** be cleared here.

## Notes

- No code, config or threshold change: this is a measurement + documentation
  validation.
- "User class" = non-`audit-` sessions (agent traffic), same convention as the
  baseline.
