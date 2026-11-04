# Slot-save failure cliff triage — LP-0MVVX6RM3001P4EY

**Verdict:** the 66–67 % "router-path" slot-save failure rate measured on the
parent window was **a measurement artefact plus one real, path-specific
defect**. The router save path is healthy (97.9 % success in-window). The
failures are **eviction-path saves pinned to the static 3.0 s base timeout**:
`plan_eviction_snapshot` has no request body, its token estimate is 0, so the
deployed adaptive timeout (`session_slot_timeout_per_token_seconds: 0.0015`,
cap `session_slot_max_timeout_seconds: 60`) contributes nothing. Hypothesis 1
(timeout too short for large snapshots) is **supported for the eviction path
only**; hypotheses 2 (slot contention) and 3 (llama-server regression) are
**refuted** as the primary cause. A minimal proxy-side fix — feed the
session's recorded prefix into the existing adaptive formula and tag `source=`
on every slot-save failure line — ships with this item. No tuning values
changed (AC5).

Artifacts (committed alongside this report):

- `docs/dev/slot-save-failure-attribution-LP-0MVVX6RM3001P4EY.json` — main
  window 2026-10-29T12:58:32Z → 2026-10-31T04:09:00Z (39.17 h).
- `docs/dev/slot-save-failure-attribution-prefix-LP-0MVVX6RM3001P4EY.json` —
  pre-fix window 2026-10-28T12:58:32Z → 2026-10-29T12:58:32Z (24 h).

## AC1 — measurement (reproducible)

Command line (run from the repo root; output written only via the script,
never hand-edited):

```bash
python3 scripts/attribute_slot_save_failures.py \
  --log-dir /var/log/llama-proxy \
  --window '2026-10-29T12:58:32,2026-10-31T04:09:00' --json \
  > docs/dev/slot-save-failure-attribution-LP-0MVVX6RM3001P4EY.json
```

Headline counts (main window; the pre-fix JSON holds the same fields):

| Metric | Pre-fix 24 h (10-28T12:58:32 → 10-29T12:58:32) | Main window (10-29T12:58:32 → 10-31T04:09:00, 39.17 h) |
|---|---|---|
| Router saves ok / failed | 103 / 0 (0 % fail) | **46 / 1 (2.1 % fail)** |
| Eviction saves ok / failed | 9 / 33 (78.6 % fail) | **60 / 93 (60.8 % fail)** |
| Failure reasons | ReadTimeout 33 | ReadTimeout 94 |
| `timeout=` on failures | 3.0 s × 33 | **3.0 s × 93**, 35.6 s × 1 |
| `elapsed=` on failures | 3.0–3.5 s | 3.0–3.8 s × 92, 9.3 s × 1, **35.6 s × 1** |
| Failures at zero proxy queries | 3 / 33 | **11 / 94** |
| `slot persistence skipped reason=slot_busy` | 155 | **375** |
| Seed outcomes (`missing_sidecar` / `missing_snapshot` / other) | 303 / 392 / 0 | 863 / 32 / 0 |

Path attribution method: `slot_save failed slot=N …` lines carry neither
`session=` nor `source=`, but every eviction attempt logs a paired
`slot_save session=… saved=False source=eviction` line at the same second
and slot, so a failure is attributed to the eviction path when such a pair
exists and to the router path otherwise. The single in-window router failure
(`timeout=35.6s`, unpaired) is the one whose adaptive timeout actually
scaled.

**Save-duration distribution: NOT logged.** The success path logs duration
only at DEBUG (`slot_save ok slot=N elapsed=… timeout=…`) and the deployed
logging level is INFO (0 DEBUG lines across the whole deployed log history,
verified by grep). The report therefore does **not** estimate success
durations; the failure-side `elapsed=` (all ≈ the applied timeout) is the
only duration evidence.

## AC2 — hypothesis verdicts

### H1 — timeout too short for large snapshots: **SUPPORTED** (eviction path only)

Deciding evidence:

1. **93 of 94 failures used `timeout=3.0s`** — the static base
   (`session_slot_timeout_seconds`), not the adaptive value. Only the one
   router failure shows a scaled timeout (35.6 s = 3 + 0.0015 × ~21.7 k
   tokens), and it still waited its full window (`elapsed=35.6s`).
2. Eleven of the 94 failures occurred with **zero active queries and zero
   active sessions** — a 3 s bound that fails on an otherwise idle server is
   a size/timeout problem, not a load problem.
3. llama-server serializes whole KV caches; observed `prompt_save` state
   sizes in the window's logs reach **1,155 MiB (105,109-token prompt)** and
   322–412 MiB routinely — physically incompatible with a 3 s copy.
4. Root cause in code:
   `plan_eviction_snapshot` → `_build_slot_context(server_config, session_id,
   None, …)`; with `body_json=None` the token estimate is 0, so
   `session_slot_timeout_per_token_seconds` contributes `0.0015 × 0 = 0` and
   the timeout stays at the base. The router path (which passes the request
   body) is on the adaptive formula — hence its ~100 % success.

### H2 — slot contention / concurrency: **REFUTED** as the primary cause

- `slot_busy=false` at every failure, and **11/94 failures occur at
  `active_queries=0`, `active_sessions=0`** — the timeout fires with no
  competing work, so contention cannot be the deciding mechanism.
- Concurrency is an *aggravator*: most failures (83/94) do have ≥1 active
  query at the failure second (the eviction itself runs while the local slot
  is servicing work), but the identical 3.0 s bound and elapsed≈3.0 s shape
  holds across the whole load spectrum, including the idle cases.
- `slot persistence skipped reason=slot_busy` (375 in-window) is a *skip*
  (the save is not attempted), not a failure; it reduces supply separately
  and is not part of the 94-failure set.

### H3 — llama-server-side regression: **REFUTED**

- The failure signature is identical **pre-fix**: 33/33 failures at
  `timeout=3.0s`, 3/33 at zero queries, all ReadTimeout.
- The decisive mechanism is proxy-side (the eviction timeout derivation),
  whose behaviour did not change across the window. The router path — same
  llama-server, same GPU — shows no regression (103/103 pre-fix, 46/47
  post-fix).
- A 7-day horizon (2026-10-24 → 10-31, `--window
  '2026-10-24T00:00:00,2026-10-31T23:59:59'`) confirms the same shape:
  router 495/500 ok; eviction 158/257 ok with **257/257 failures at
  `timeout=3.0s`**, 75 failures at zero queries. No follow-up is filed
  against llama-server configuration.

## AC3 — decision

**Cause: the eviction save path bypasses the adaptive timeout.** Implemented
in this item (with tests):

1. `_build_slot_context(…, estimated_tokens=None)` accepts an explicit
   estimate; `plan_eviction_snapshot` feeds it the session's **recorded
   measured prefix** (`provider.get_recorded_prompt_prefix`, the same
   trustworthy count the sidecar writer uses), so the eviction path now runs
   the same adaptive formula as the router path and is bounded by the same
   cap (60 s). A session with no recorded prefix keeps the base timeout
   (3.0 s) — unknown context, no invention.
2. Every `slot_save`/`slot_restore` failure line now ends with
   `source=router|eviction|unknown`, so a failure can be attributed to its
   path without timestamp-pairing inference (the gap that produced the
   parent's 66–67 % "router-path" misreading). `attribute_slot_save_failures.py`
   prefers the field when present.

Only the eviction **planner** changed behaviour; no `config.yaml` value was
touched. Post-deploy verification metric: the eviction ok/fail ratio and the
`timeout=` histograms in these two JSON artifacts (success durations are not
logged at INFO — see AC1).

Follow-ups: none required. (The 39.17 h window's 375 `slot_busy` skips are a
separate supply signal already owned by the load-aware gating work; this item
records them, it does not tune them.)

## AC4 — supply impact (share of seeds attributable to a failed save)

`seeds_by_last_attempt` in the JSON attributes every
`missing_sidecar`/`missing_snapshot` seed to the session's **most recent
save attempt** (ok or failed):

| Seed outcome | After router ok | After eviction ok | After a **failed** save | No save in window |
|---|---|---|---|---|
| `missing_sidecar` (863) | 245 | 263 | **316 (36.6 %)** | 39 |
| `missing_snapshot` (32) | — | 9 | **20 (62.5 %)** | 3 |

- Failed saves attribute **336 / 895 = 37.5 %** of the two seed classes to the
  save-failure cliff (which for `missing_snapshot` — no snapshot at all — is
  directly causal).
- The `missing_sidecar` majority (508/863) follows *successful* saves and is
  the separate pairing gap already owned by the path-alignment fix (sibling
  item LP-0MVVI8TEU006KTOZ); a failed save writes no sidecar, so the two
  causes are additive, not overlapping.
- Interpretation for parent AC3: even with the path-alignment fix, `found`
  cannot rise to its potential while 37.5 % of seeds follow a save that never
  completed — this item's fix removes that floor.

With the fix, a 40 k-token eviction session's save timeout goes from 3.0 s to
`min(3 + 0.0015×40 000, 60) = 60 s`; a 20 k-token session to 33 s; a session
with no recorded prefix is unchanged (3.0 s). No value changed, so AC5's
old/new naming is about the *applied timeout policy*, not a config value.

## AC5 — no masking

- No `config.yaml` tuning value was changed. Old → new *effective behaviour*
  for the eviction path only: save timeout always the static base (3.0 s) →
  `min(3.0 + 0.0015 × recorded-prefix-tokens, 60) s`, justified by AC2's H1
  evidence (93/93 eviction failures at 3.0 s, 11 at zero load, llama-server
  state sizes to 1,155 MiB).
- The failure rate was not tuned away: `session_slot_skip_when_busy` and the
  circuit breaker are untouched.
- Changed-scope suite is green through the test skill (see git history for
  the run); the full suite runs at the dev-push gate.
