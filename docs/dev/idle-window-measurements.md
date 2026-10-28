# Idle-aware local admission: before/after measurement

Reproducible measurement for LP-0MVGHXD59008UOUD — *Measure idle-aware local
admission impact over ≥24 h*. The behaviour under test is shipped by
LP-0MVGD9Z8Z0055Y5J: when the local backend has been idle for
`server.local_idle_admit_seconds` (default 60) and nothing is queued, in
flight or loading, an **economic** `large_context_bypass` is served locally
instead of falling back to a paid remote provider. The override only fires
while the GPU is demonstrably idle; the physical capacity cap
(`context_too_large`) is never overridden.

The override is **off in the shipped default** (`server.local_idle_admit_enabled:
false`). The ≥ 24 h window of record (2026-10-22 21:48:05 → 2026-10-23 21:48:05)
ran with it enabled (LP-0MVJXGSEG0045IAF) and measured **31 admissions with 0
conversions** — every admitted prompt was refused by the ETA admission guard or
cancelled at the prefill deadline before a first byte. The numbers, the
downstream lease-gate reading and the recorded decision are in
[Lease-gate decision](#lease-gate-decision-lp-0mvk35ux20081bib-the-dispatch-lease-gate-stays-as-is)
below; the machine artifact is
`docs/dev/idle-window-measurement-2026-10-22-to-2026-10-23.json`.

## Tool

`scripts/idle_admit_impact.py` scans the proxy log stream
(`/var/log/llama-proxy`, live + rotated under either naming scheme, plain or
`.gz`, via `scripts/lib/proxy_logs.py`) for a **baseline window** and a
**post-deploy window** and reports the acceptance-criteria metrics for both.
It never changes proxy behaviour and does not need the proxy to be running.

```bash
# Last 24 h vs the 24 h before it (default; useful for a first look)
./scripts/idle_admit_impact.py

# Explicit windows, machine-readable artifact
./scripts/idle_admit_impact.py \
    --before-start '2026-10-19 07:23' --before-end '2026-10-20 07:23' \
    --after-start '2026-10-21 09:00' --after-end '2026-10-22 09:00' \
    --deployed-at '2026-10-21 08:55' --window-closed --json > idle_admit_24h.json

# Markdown section for a work-item comment or docs artifact
./scripts/idle_admit_impact.py --markdown
```

Exit codes: `0` = PASS (measurement complete, every AC check met), `2` = WARN
(measurement incomplete or a check missed — the tool never silently passes),
`1` = invalid input / no logs.

## What is measured

The tool also carries an instrumentation addition: the local
`dispatch_first_byte_ms` log line now ends with `priority=<tier>
estimated_tokens=<n>`. Both values were already computed inside
`proxy_to_local`; emitting them lets a dispatch be attributed to a priority
tier (AC4) and a context bucket (AC2) without heuristics. Every pre-existing
field keeps its position, so existing consumers
(`scripts/dispatch_latency_verification.py`, `proxy-usage-analysis`) parse the
line unchanged.

**AC1 — window aggregates.** `local_idle_admit` count, local serve share
(`Stream finished provider=local` ÷ all finished streams),
`large_context_bypass` count (demand only: `chain_hold_cycle > 0`
re-emissions are excluded, mirroring the usage report), remote fallback count
(with the per-reason split), `local_dispatch_denied` count and local prefill
slot-seconds (`dispatch_to_first_byte_ms` sum, the slot-held service time —
the same quantity the intake brief reports as 14.3 h of prefill in the
baseline window).

**AC2 — idle-admitted vs non-idle local dispatches, same context bucket.**
Each `routing_idle_admit` event is paired with the next local dispatch in the
**same session** inside the correlation window (default 60 s, claimed at most
once). Paired requests are bucketed by the prompt size on the `idle_admit`
line (authoritative) and compared bucket-by-bucket with non-idle dispatches,
reporting `dispatch_first_byte_ms` percentiles, `dispatch_to_first_byte_ms`
percentiles and slot-seconds **with sample counts**. Idle-admitted prompts are
expected to be *slow* — that is the trade, not a defect — so there is no
threshold on this comparison, only a stated cost. For logs predating the
dispatch-line instrumentation, the bucket falls back to the session's most
recent `routing_check` line.

**AC1 (parent LP-0MVK16GME007EYUW) — lease-gate shape attribution.** The idle
override can decide to admit a request and then be refused one gate later by
the **local dispatch lease gate** (`local_lease_active`), because another
session merely *holds* a local lease. The tool reports how much of the idle
capacity that gate costs, split by blocking shape, so the parent decision is
made from counts rather than inference (LP-0MVK35HBI003SB3Z).

| Log line | Field | Meaning |
|---|---|---|
| `local_dispatch_denied` | `cold_start=False` | the denial *is* a `local_lease_active` refusal (a `cold_start=True` denial is a warmup deferral and is counted separately) |
| `local_dispatch_denied` | `owner=` | the foreign session holding the lease |
| `local_dispatch_denied` | `active=` | occupancy count at refusal: `0` → shape A, `>= 1` → shape B |
| `lease_released` | `reason=` | why the lease ended: `no_progress_watchdog`, `idle_timeout`/`prefill_deadline` (adaptive expiry), `explicit_release`, `client_disconnect`, `orphan_cleanup`, `session_evicted` |
| `status_request` | `local_active_query=` / `local_owner_session_id=` / `local_owner_lease_remaining_seconds=` | whether the owning session actually had GPU work, and how long its lease had left |
| `contention_queue_dispatch` / `contention_queue_fallback_after_queue` | `reason=local_lease_active` | whether the bounded (60 s) contention queue converted the refusal or fell through to remote |

**Attribution rule** (stated in the report as `lease_gate.attribution_rule`):

- **Shape A — inactive cooldown lease.** `active=0`: no *active* record exists,
  so the pool is held only by inactive leases inside their base 30 s cooldown
  (the owner finished; its record stays reserved for its session by the
  no-preemption policy). Bounded and small — it can only convert the "frees in
  a few seconds" cases.
- **Shape B — `active` record with no GPU work behind it.** `active>=1`: an
  active record exists, which is either a wedged/abandoned stream or a long
  silent prefill. The sub-split is corroborated, never assumed — the owner's
  earliest `lease_released` within 900 s of the refusal, plus any
  `status_request` naming that owner, decides between `no work behind it`
  (`no_progress_watchdog`, `orphan_cleanup`, `idle_timeout`, or
  `local_active_query=false`), `silent prefill` (`prefill_deadline`),
  `owner working` (a completion/disconnect reason — i.e. genuine contention,
  not an idle lease) and `unconfirmed` (no evidence).

Counts for both shapes, the shape-B evidence split, the `lease_released`
reason groups, the contention-queue outcomes and the `status_request`
aggregate are emitted per window in JSON (`lease_gate`, `lease_released`,
`lease_status`) and as rows in `--markdown`.

**Reading the answer for the parent AC1.** `shape_a` + `shape_b` = the
attributable `local_lease_active` refusals in the window;
`shape_b.no_work_behind_it` is the part that a verified-idle preemption
predicate could take immediately, `shape_b.silent_prefill` is the part the
180 s no-progress watchdog / adaptive expiry already reclaims, and
`shape_b.owner_working` is the refusal the gate is *right* to enforce. Quote
`refusals`/`samples` alongside any percentage: percentages are withheld
(`null`) when the window holds fewer than `--min-lease-samples` (default 5)
attributable refusals, and the run then ends as a WARN (exit 2) with an
explicit "too few lease-gate samples" note so a thin window can never read as
a confident split. Denials predating the `active=` field are counted in
`unattributed_refusals` and are never silently folded into either shape.

**AC3 — denials and mid-stream timeouts.** `local_dispatch_denied` is compared
as a **rate per hour** (windows may differ in length); an increase beyond 5 %
fails the check. Mid-stream timeouts (`ReadTimeout`, `empty_response`,
`ConnectTimeout` fallback reasons, `slot_save failed ... error=ReadTimeout`,
`dispatch_first_byte_timeout`) are split three ways:

- **attributed** — the timeout names a session and falls inside the
  correlation window of an idle-admitted request in that session. Any
  attributed timeout fails AC3;
- **coincident** — the line carries no session id (most `Fallback triggered`
  and `slot_save failed` lines do not) but falls inside the correlation window
  of some idle-admitted request. Reported, never *blamed*;
- **unrelated** — everything else.

**AC4 — interactive traffic.** For the interactive tier
(`--interactive-priority`, default `high`) the p50/p90 first-byte latency of
**non-idle** dispatches is compared before vs after with a ±10 % noise band.
Idle-admitted dispatches are the *treatment*, so they are excluded from the
control distribution and counted separately
(`ac4.idle_admitted_by_priority`); when the interactive tier regresses, the
tool names the idle-admitted prompts at that tier as the causal evidence.

**Counterfactual (approximation).** For a window in which the override was
never enabled the tool estimates how often it *would* have fired: an economic
bypass is *eligible* when the gap since the last system-wide local activity
(a local dispatch or a local stream finishing) is at least the idle
threshold. The real guard also refuses while anything is leased, queued or
prefilling, so the true count is `<= eligible_bypasses`. Always read it as an
estimate, never as a measurement.

## Verdict and the AC5 recommendation

| Check | Pass condition |
|---|---|
| AC1 window | post-deploy window ≥ 24 h (and starts at/after `--deployed-at` when given) |
| AC1 signal | at least one `routing_idle_admit` event in the post-deploy window |
| AC2 samples | at least 5 idle-admitted dispatches paired with their dispatch event |
| AC3 denials | per-hour `local_dispatch_denied` does not rise by more than 5 % |
| AC3 timeouts | zero timeouts attributable to idle-admitted requests |
| AC4 interactive | non-idle `high`-priority p50/p90 first byte within 10 % |
| AC1 (LP-0MVK16GME007EYUW) lease gate | at least 5 attributable `local_lease_active` refusals in the post-deploy window (below it the split percentages are withheld and the run is a WARN) |

The AC5 recommendation is deterministic:

- `measure` — no (or too few) idle-admitted samples: not measurable yet. The
  follow-up names the enablement step.
- `enable_by_default` — the remote fallback rate fell by at least 2 % **and**
  denials did not rise **and** no timeout is attributable **and** interactive
  latency is within noise.
- `tune` — anything else, with the tuning decision spelled out (priority
  gating, prompt-size cap, threshold increase), which is exactly the
  "concrete follow-up records the tuning and why" that AC5 accepts.

Exit code is `0` only when every check passes **and** the recommendation is
`enable_by_default`; a `tune` outcome is a completed measurement but exits
`2` so the operator sees the decision is not "enable as-is".

## Runbook: measuring the override

1. **Enable and start the window.** Set `server.local_idle_admit_enabled: true`
   (keep `server.local_idle_admit_seconds: 60` for the first window), commit and
   push it to `dev`, then make the running proxy pick it up:

   ```bash
   curl -s -X POST http://localhost:8000/admin/reload-config
   ```

   `POST /admin/reload-config` re-reads `proxy/config.yaml` into `srv.config`
   without restarting anything, so in-flight streams (and any agent talking to
   the proxy) are not interrupted. Record the reload time — that is
   `--deployed-at` and the earliest valid post-deploy window start. A full
   restart (`/skill:start-proxy --restart`) also works but is heavier: it drops
   in-flight requests and resets the process-local idle clock.
2. **Verify it is live.** The next request that would have been an economic
   bypass while the GPU is idle logs
   `routing_idle_admit ... reason=economic_bypass_idle`; on-demand check:
   `grep -c routing_idle_admit /var/log/llama-proxy/proxy.log`.
3. **Wait ≥ 24 h.** The window cannot be shortened; the tool refuses to call a
   shorter window a post-deploy window under `--window-closed`.
4. **Run the measurement** with explicit windows, `--deployed-at` and
   `--window-closed`, and attach the JSON artifact to the work item.
5. **Act on the recommendation.** `enable_by_default` → keep the flag on and
   record the numbers next to the knob in `proxy/config.yaml`. `tune` → open
   the follow-up the tool proposes (priority gating / prompt-size cap /
   threshold) and keep the flag on or off according to that decision.

Rollback is the flag back to `false` plus the same `POST /admin/reload-config`
— there is no other state to unwind.

Notes when reading a window:

- The idle clock is process-local monotonic time and resets to process start,
  so the first `local_idle_admit_seconds` after a proxy restart cannot fire.
  Windows that contain restarts still measure correctly, they just under-count
  the first minute after each restart.
- An idle-admitted 120 K prompt can hold a slot for minutes. A same-window
  overlap with interactive traffic is a real finding, not a config error —
  that is what AC4 exists to catch.
- `proxy-usage-analysis` reports the same override from the other side (the
  `Local admitted while GPU idle (cold-cache override)` row and the
  `local_idle_admit` JSON counter). Use it for the AC1 headline numbers when
  the daily report is the artifact of record; this tool adds the per-request
  comparison, the attribution and the verdict.

## Evidence recorded so far

Basis for the intake brief (window 2026-10-19 07:23 → 2026-10-20 07:23, with
the tool):

| Metric | Baseline |
| --- | ---: |
| `local_idle_admit` | 0 (override disabled) |
| Local serve share | 2.34 % |
| `large_context_bypass` | 1,284 |
| Remote fallback events | 17,005 |
| `local_dispatch_denied` | 2,635 |
| Local prefill slot-seconds | 50,187 s (13.9 h) |

Counterfactual estimate over 2026-10-21 19:00 → 2026-10-22 19:00 (override
still disabled): 1,235 economic bypasses, **937 (75.9 %)** with ≥60 s of
system-wide local idle before them (idle-gap p50 129 s, p90 1,617 s). The
override would therefore have converted a large share of remote fallbacks —
which is why the 24 h measurement is worth running before enabling it by
default. This is an estimate, not a measurement: leases, queued requests and
in-flight prefills are not reconstructable from logs.

The AC4 rows are unmeasurable in any window logged before the
`priority=`/`estimated_tokens=` instrumentation is deployed; the tool says so
explicitly instead of reporting a vacuous pass.

**Lease-gate reading, 2026-10-25 → 2026-10-27 (illustration, not the decision).**
A live run over the two most recent 24 h windows read: 338 and 291
`local_lease_active` refusals respectively, **all shape B** (`active>=1`;
shape A never appeared in either window). In the first window the shape-B
split was 223 (66 %) with no work behind it (`no_progress_watchdog` /
`idle_timeout` / `local_active_query=false`), 112 (33 %) silent prefills
(`prefill_deadline`) and 3 (1 %) genuinely working owners; the second window
was 134 (46 %), 84 (29 %), 1 (0.3 %) with 72 (25 %) unconfirmed. The bounded
contention queue converted 73 of 177 and 26 of 60 of them to a local dispatch.
That is the shape of the answer the parent AC1 asks for; the recorded decision
and its window live with LP-0MVK35UX20081BIB — see the next section.

## Lease-gate decision (LP-0MVK35UX20081BIB): the dispatch lease gate stays as is

**Decision window of record:** 2026-10-22 21:48:05 → 2026-10-23 21:48:05
(24.0 h, `--window-closed`, `--deployed-at '2026-10-22 21:48'`) — the window
opened by LP-0MVJXGSEG0045IAF. Machine artifact:
`docs/dev/idle-window-measurement-2026-10-22-to-2026-10-23.json`. Reproduce
with:

```bash
./scripts/idle_admit_impact.py \
    --before-start '2026-10-21 21:48' --before-end '2026-10-22 21:48' \
    --after-start '2026-10-22 21:48' --after-end '2026-10-23 21:48' \
    --deployed-at '2026-10-22 21:48' --window-closed --json
```

### AC1 — the measured conversion rate, and what the lease gate costs it

**31 `routing_idle_admit` events converted 0 times (0/31 local first bytes).**
The tool's pairing (`ac2.idle_admitted_sample_counts` = 0 paired, 31 unpaired)
and a per-event trace of the same window agree, so the zero is not a
correlation-window artefact:

| Outcome of the 31 admits | Count | Evidence |
| --- | ---: | --- |
| Converted (local first byte) | **0** | no `dispatch_first_byte_ms` line on any admitted session |
| Dispatched, then cancelled at the prefill deadline | 22 | `prefill_deadline_exceeded ... deadline=90s → cancelling local dispatch and detouring to the next provider`, preceded by `lease_extended_during_prefill ... progress=None alive=False in_flight=True` (no observable prefill progress) |
| Refused on arrival by the ETA admission guard | 9 | `local_admission_guard decision=reject reason=eta_exceeds_budget_direct enforced=true` (projected prefill 63–362 s vs the 60 s queue budget; e.g. 118,064 tokens → `eta_prefill_s=118.1`) |
| Refused by the dispatch lease gate (`local_lease_active`) | **0** | no `local_dispatch_denied session=<admitted session>` line exists for any of the 31 |

The deadline in force during the window was 90 s (the shipped knob is now 120 s);
the direction is the same for both — the admitted prompts' projected prefills
far exceed either value.

**So the lease gate costs the override zero measured conversions: not one of the
requests it admitted was refused by `local_lease_active`.** The constraint that
actually eats the win is prefill feasibility — the 90 s prefill deadline (22) and
the ETA guard's 60 s queue budget (9). The counterfactual's 75.9 % therefore
over-estimates for a different reason than the lease gate: the admitted prompts
(39 K–119 K tokens, all `cache_state=cold`) cannot reach a first byte inside the
deadline at all.

### AC1 — which blocking shape accounts for the refusals

| Metric | before (10-21 21:48 → 10-22 21:48) | after / window (10-22 21:48 → 10-23 21:48) |
| --- | ---: | ---: |
| `local_lease_active` refusals (`cold_start=False`) | 1,451 | 1,731 |
| **shape A** — inactive cooldown lease (`active=0`) | **0** | **0** |
| **shape B** — `active` record (`active>=1`) | 1,451 (100 %) | 1,731 (100 %) |
| — no work behind it | 1,147 | 1,674 (96.7 %) |
| — silent prefill (`prefill_deadline`) | 0 | 0 |
| — genuinely working owner | 0 | 0 |
| — unconfirmed | 304 | 57 (3.3 %) |
| `lease_released` (all reasons) | 590 | 348 |
| — `idle_timeout` / `session_evicted` / `no_progress_watchdog` / `prefill_deadline` | 250 / 124 / 22 / **0** | 207 / 114 / 27 / **0** |
| Contention queue: dispatched local / fell through | 235 / 487 | 100 / 522 |

Shape A never occurs in either window, and **no** `lease_released` line carries
`reason=prefill_deadline`. The refusals are entirely **shape B**: `active`
records with no GPU work behind them — wedged/abandoned streams released by the
no-progress watchdog (27 in the window) or expiring by adaptive idle timeout
(207), corroborated by `status_request ... local_active_query=false` for 40.
Silent prefills are not among the refusal causes in this window: the dominant
shape-B evidence is `adaptive_expiry` (1,458 of 1,674) — the long adaptive
lease on a record with no progress behind it — followed by
`no_progress_watchdog` (176) and `status_idle_owner` (40).

### AC2 — the decision and its rationale

**Option 1 — leave the lease gate as is.** No gate change is activated, and the
option-4 follow-up LP-0MVK368JZ000TDW7 is closed with this decision recorded on
it. Shapes converted: **none** (no change) — and none of the alternatives has a
measured shape to convert in this window.

- **Option 3 (remaining-time gate)** converts shape A only; shape A is 0 in both
  measured windows, so it would convert nothing.
- **Option 4 (preempt verified-idle foreign lease)** is the only option whose
  predicate targets what the window actually contains (1,674 shape-B
  no-work-behind-it refusals). It is still not activated: the requests it would
  release the gate for are exactly the admitted prompts that cannot reach a
  first byte here, so it would turn refusals into *aborted* prefills — ~90 s of
  slot time each for zero conversions — while charging the displaced owner a
  re-prefill. The 42 refusals owned by already-admitted (then aborted) sessions
  show the cost direction without buying any of the win.
- **Option 2 (blanket preempt)** has the same zero-win economics without the
  verified-idle predicate — strictly worse than option 4.
- **Option 5 (prefill-aware idle clock)** has no support in the numbers: zero
  refusals attribute to a silent prefill (`prefill_deadline` = 0 in both
  windows' `lease_released` reasons and in shape-B corroboration). The prefill
  blindness that matters is the override's own admission — it admits prompts
  whose projected prefill exceeds the deadline/budget — and that is filed as a
  follow-up rather than solved by a clock change.

**Load caveat.** The after window was ~2× busier than the before window
(`large_context_bypass` 1,790 → 3,702; remote fallbacks 6,953 → 12,888;
denials 66.0 → 72.1 per hour), so 0/31 is a heavy-load measurement. The
mechanism (admit → ETA refusal or deadline abort) does not depend on load, and
no gate change is justified either way — the decision is "do not change the
gate", not "the gate costs nothing under all loads".

### AC5 — follow-ups filed rather than added to this scope

The real blocker is filed as **LP-0MVRCLPVI0057116** (*Idle-admit eligibility
must match prefill feasibility (measured 0/31 conversion)*), the AC5 tuning
follow-up of LP-0MVGHXD59008UOUD, linked to LP-0MVNBJYE3003MF6J (ETA-guard /
prefill-deadline interaction). Audit-session lease hygiene — why an `active`
record with no work exists for minutes — stays with the existing wedge/orphan
items (LP-0MVO8H7VQ003B12I, LP-0MUCEFB8E003YVFF); it is not re-filed here.

### Appendix Q3 — the probe's ~705 s lease, settled

The original probe's own refusal is inside this window:

```
2026-10-22 21:57:13 local_dispatch_denied session=idle-admit-verify-1792702147
  owner=audit-AH-0MVIMPC11001F7WU-verdict_reask-11ea1acf active=1 cold_start=False
2026-10-22 21:59:26 lease_released session=audit-AH-0MVIMPC11001F7WU-verdict_reask-11ea1acf
  reason=no_progress_watchdog endpoint=http://localhost:8080
2026-10-22 21:59:26 dispatch_cleanup removed 1 stale lease(s)
```

Shape B, and the owning record was reclaimed by the **no-progress watchdog** —
a wedged `active` record with no work behind it, matching the adaptive-lease
arithmetic (30 s base + 45,000 × 0.015 = 705 s). It was **not** a long silent
prefill: the window's 348 `lease_released` lines carry zero
`reason=prefill_deadline`, and zero shape-B refusals corroborate to
`prefill_deadline`. **Q3 is settled from the window's `lease_released`
reasons.** (The probe's next request, once the lease cleared, was then refused
by the ETA guard — `eta_exceeds_budget_direct`, 53.6 s projected prefill vs the
60 s budget — which is the same prefill-feasibility constraint described above,
not a lease condition.)

## Tests

`tests/test_idle_admit_impact.py` drives the CLI against synthetic logs and
asserts the metric contract above (aggregates, hold-cycle exclusion, session
pairing and the correlation window, bucket comparison, three-way timeout
attribution, the AC4 control distribution, the recommendation rules, the
counterfactual, and the CLI's exit codes / JSON shape / markdown rendering).
The lease-gate suite asserts the refusal count (cold-start denials excluded),
the `active=`-field A/B split, the shape-B corroboration (`lease_released`
reasons, `status_request` `local_active_query=false`, `prefill_deadline` =
silent prefill, completion reasons = genuinely working), the `lease_released`
reason groups, the `status_request` aggregates, the contention-queue
outcomes, the too-few-samples WARN (percentages withheld, exit 2), and that a
log with no lease-gate lines leaves every pre-existing output field
unchanged.
`proxy/tests/test_dispatch_log_measurement_fields.py` drives a real local
stream through `proxy_to_local` and asserts the dispatch line carries
`priority=` and `estimated_tokens=` without disturbing the pre-existing
fields.
