# Group-2 serialisation proof (LP-0MW8EAT60003Z28H)

Evidence for parent LP-0MW7V14DN007F127 AC2 (serialisation: one group under
test per restart) and AC9 (stale `blocked` cleared).

Recorded 2026-11-16 (group-2 validation dispatch). Read-only checks only —
no proxy restart performed by this child (the single group-2 restart is
performed by child LP-0MW8EATPF00924OF).

## AC1 — Serialisation proof (parent AC2)

The restart performed by the sibling child (LP-0MW8EATPF00924OF, group-2 AC1)
happens **after** the group-1 validation is terminal with a certified ≥ 24 h
window.

| Fact | Value |
|---|---|
| Group-1 validation item | **LP-0MW7V0S6A006DKZ4** — `completed` / `in_review` |
| Group-1 restart | 2026-11-10T14:18:26Z (PID 316897, old PID 2634864) |
| Group-1 restart-free window | 2026-11-10T14:18:26Z → 2026-11-11T15:18:26Z (**25.0 h**) |
| Group-1 dataset | `proxy/docs/validation/group1-prefill-deadline-residual.json` (commit `abdf2d397`) |
| Group-1 verdict | `proxy/docs/validation/group1-prefill-deadline-verdict.md` (commit `4dad2ee4c`) |
| Group-1 measurement child | LP-0MW8CS7O7000YPDB — `completed` |

All three outbound `depends-on` edges of the parent are satisfied:

| Blocker | Status |
|---|---|
| LP-0MW7V2D13004ZO65 — group-2 impl epic (cold-regime ETA + deadline capacity) | `completed` / `in_review` |
| LP-0MW7V2DK400365ER — group-2 impl (feed measured prefill progress) | `completed` |
| LP-0MW7V0S6A006DKZ4 — group-1 validation | `completed` / `in_review` |

Verified: `wl dep list LP-0MW7V14DN007F127 --json` → all outbound edges
terminal.

## AC2 — One group per restart (parent AC2)

Both group-2 implementation items are terminal (see table above). No group-2
validation window has been measured before this restart chain; the group-1
window closed 2026-11-11T15:18:26Z and no certified group-2 measurement exists
yet. The restart in child LP-0MW8EATPF00924OF is therefore the **single**
group-2 validation restart, satisfying the "one group per restart" rule.

**Observed intermediate restart (unattributed).** The running proxy was last
restarted at **2026-11-15T01:06:50Z** (`Proxy restart attribution:
source=manual actor=unknown reason=manual (no reason provided)`, PID 3122239).
That restart happened during the group-2 implementation period — the
cold-window calibration fix (`4a72facb0`) merged at 2026-11-15T01:10:04Z and
the replay proof (`18dc37d43`) at 2026-11-15T05:09:55Z, i.e. **after** it. The
running tree therefore lacks the calibration fix that makes the deadline-
capacity bound non-inert (parent AC9 note), so it is **not** a valid group-2
validation tree. The group-2 restart (child AC1) loads the full group-2 tree
(commits `7f6033901`…`4a72facb0`, all ancestors of `origin/dev`), stays the
single certified restart, and its ≥ 24 h window has never been measured before.

## AC3 — Stale `blocked` cleared (parent AC9)

`wl dep add` had set the parent to `blocked`. With all three blockers terminal
(2026-11-15), the block is stale; the parent was cleared to `open` with the
transition recorded in the parent comment (AC9, 2026-11-15T05:21Z). No
further lifecycle change is needed; this item is `open` and startable.

## Interlock with the ≥ 24 h window child

The measurement child (LP-0MW8EAU8P0079280) depends on the restart child; per
the measurement-windows convention (`docs/dev/measurement-windows.md`) its
window is anchored at the restart and must elapse ≥ 24 h before measurement.
The group-1 precedent held that child `blocked` + `[run-not-before …]` until
the window closed; the same gating is applied to the group-2 window child.
