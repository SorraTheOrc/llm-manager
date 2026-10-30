# Slot-ratio seed fix (5502ee44) post-deploy verification — 2026-10-30

Work item: LP-0MVTJVNSO008IZEB — validates LP-0MVK0E80C009HUQC (fix `5502ee44`)
and answers AC3 of LP-0MVJXV0MY000O77E.

## Verdict

**NOT FIXED.** Over the ≥24 h post-fix window (2026-10-29T12:58:32Z →
2026-10-30T12:58:32Z) there are **0 `cache_state=restored` dispatches, 0
`slot_ratio_restored` seed events and 0 `slot_ratio_seed outcome=found`
decisions** — identical to the 155 h pre-fix baseline. The restored-prefix
credit still does not fire, and `missing_sidecar` (snapshot present, no
sidecar) now dominates at 98 % of seed decisions. AC1/AC2/AC3/AC6 are unmet;
AC4 is this report; AC5's honest verdict is below with a follow-up work item
filed. **The epic's AC3 stays unmet.**

## Command lines (exact, reproducible)

Primary 24 h pair (pinned timestamps, no `now` shorthand):

```bash
python3 scripts/analyse_slot_ratio_restore.py \
  --log-dir /var/log/llama-proxy \
  --slot-cache-dir /home/rgardler/projects/llm/slot-cache \
  --window "post-fix=2026-10-29T12:58:32,2026-10-30T12:58:32" \
  --window "pre-fix=2026-10-28T12:58:32,2026-10-29T12:58:32" \
  --json > docs/dev/slot-ratio-restore-postfix-2026-10-30.json
```

Supplementary windows (full post-fix span and the post-restart span):

```bash
python3 scripts/analyse_slot_ratio_restore.py \
  --log-dir /var/log/llama-proxy \
  --slot-cache-dir /home/rgardler/projects/llm/slot-cache \
  --window "post-fix-full=2026-10-29T12:58:32,2026-10-30T21:46:00" \
  --window "post-restart=2026-10-30T10:35:09,2026-10-30T21:46:00" \
  --json > docs/dev/slot-ratio-restore-postfix-2026-10-30-full.json
```

Both JSON payloads are committed next to this report so every count quoted
here is reproducible (`windows[].seed_outcomes`, `windows[].slots`,
`windows[].local_prefill_ms.by_context_bucket`, `sidecars`,
`restored_engagement`).

## Deploy / code-state checks (item step 1)

- Fix `5502ee44` is an ancestor of `origin/dev` (`f0b420de` …) and of local
  `dev` (`5b35dd80` at report time) — verified with
  `git merge-base --is-ancestor 5502ee44 dev`.
- Running proxy PID `3224511` (started 2026-10-30T10:35:09Z, cwd
  `/home/rgardler/projects/llm/proxy`) loads the same tree, so the deployed
  process carries the fix: `provider.sidecar_cached_tokens`,
  `provider.get_recorded_prompt_prefix`, the router sidecar write outside the
  measured-usage gate, `session._write_eviction_sidecar` + snapshot/sidecar
  retirement, and the recursive `cleanup-slot-cache.sh`.
- **Restarts inside the window — noted (item step 1: a restart resets the
  in-memory ratio but not the sidecars):**
  - 2026-10-29T14:09:25Z — deployed LP-0MVT2KVVS002YZT8 (additive
    prefill-thrash cooldown-skip metrics).
  - 2026-10-30T10:35:09Z — deployed LP-0MVT2QFAP0082BKR (measured-prefill
    invalidation on eviction/slot-release/model-switch; touches
    `measured_prefill`, not the seed/sidecar path).
  The window still satisfies ≥24 h; both restarts fall inside it and neither
  can manufacture restored dispatches (they clear in-memory ratios).

## Headline counts (24 h windows, primary)

| Metric | Pre-fix (10-28T12:58:32 → 10-29T12:58:32) | Post-fix (10-29T12:58:32 → 10-30T12:58:32) |
|---|---|---|
| `cache_state=restored` dispatch lines | **0** | **0** |
| `slot_ratio_restored` seed events | **0** | **0** |
| `slot_ratio_seed outcome=found` | 0 (0 %) | **0 (0 %)** |
| `missing_sidecar` | 303 (43.6 %) | **605 (98.1 %)** |
| `missing_snapshot` | 392 (56.4 %) | **12 (1.9 %)** |
| `expired` / `model_mismatch` / `prefix_mismatch` / `empty_ratio` | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| guard outcomes (seed past lookup) | 0 | **0** |
| seed decisions (total) | 695 | 617 |
| routing decisions | 14,087 | 10,248 |
| local serve share | 2.69 % | 4.37 % |
| slot saves ok / failed | 103 / 33 (24.3 % fail) | 34 / 65 (**65.7 % fail**) |
| eviction saves ok / failed | 9 / 33 | 42 / 64 |
| `slot persistence skipped reason=slot_busy` | 155 | 260 |
| `slot_restore failed` | 0 | 0 |

Extended windows confirm the same shape: the full 32.79 h post-fix span
(10-29T12:58:32 → 10-30T21:46:00) shows 778 seed decisions, **0 found**, 766
`missing_sidecar` (98.5 %), 0 restored dispatches; the post-restart span
(10-30T10:35:09 → 10-30T21:46:00, 11.18 h) shows 221 seed decisions, **0
found**, 221 `missing_sidecar` (100 %), 0 restored dispatches. A direct grep
for `slot_ratio_restored|cache_state=restored` over the full post-fix log
span returns 0 matches (independent of the tool).

## AC2 — restored vs cold dispatch-to-first-byte, same context bucket

No restored dispatches exist (n=0), so the AC2 restored-vs-cold comparison
cannot be stated. For reference, `dispatch_first_byte_ms` in the same windows
(no restored group to compare):

| Bucket | Pre-fix cold | Pre-fix warm | Post-fix cold | Post-fix warm |
|---|---|---|---|---|
| ≤ 38,000 | n=205, p50 9.45 / p90 51.82 | n=454, p50 2.73 / p90 21.27 | n=269, p50 6.41 / p90 47.07 | n=609, p50 2.39 / p90 11.21 |
| 38,000–120,000 | n=0 | n=86, p50 5.45 / p90 18.73 | n=1, p50 47.0 / p90 47.0 | n=2, p50 70.32 / p90 93.27 |

Sample counts for both groups are stated; the restored group is empty in
every bucket, so AC2 is **unmet** (not applicable to state numbers).

## AC3 — seed-outcome vocabulary shift

Stated above. Expected post-fix signature per LP-0MVK0E8C40035FG3 was
`missing_sidecar`/`missing_snapshot` → ~0 and `found` > 0 with `expired` /
`prefix_mismatch` possible. Observed: `found` still 0, guard still never
reached (guard_outcomes_total = 0), and `missing_sidecar` **increased** from
44 % to 98 % while `missing_snapshot` fell 56 % → 2 % (the on-disk legacy
orphan sidecars are being cleared, so fewer seeds find a sidecar without a
snapshot). The expected shift **did not happen** — AC3 unmet.

## AC6 — on-disk sidecar/snapshot pairing at report time

At report time (2026-10-30T21:40Z):

```
snapshots=0 sidecars=28        # -> 0 paired files
orphan sidecars=28             # sidecars whose snapshot is gone
orphan snapshots=0
```

vs the plan-time baseline (2026-10-29T17:40Z): 6 snapshots, 144 sidecars,
142 orphan sidecars, 4 orphan snapshots. The legacy orphan sidecar pile is
clearing (142 → 28), but **every sidecar on disk was saved after the deploy**
(`saved_at` 2026-10-30T04:02 → 19:01Z) with its snapshot already retired, so
orphan growth for post-deploy saves **persists** — AC6's "paired files
present, no orphan growth" is not met (0 paired files at report time).

## AC5 — honest interpretation and follow-up

`missing_sidecar` (605/617) means that at seed time the session's snapshot
exists but its `.ratio.json` sidecar does not — the exact writer-side gap
`5502ee44` was supposed to close. Corroborating evidence in-window: only
34/99 (34 %) router-path saves and 42/106 (40 %) eviction-path saves
succeeded, so the writer side frequently reaches the "snapshot saved, sidecar
missing" state through one of:
1. the eviction path's `_write_eviction_sidecar` returning early when
   `get_recorded_prompt_prefix` has no recorded prefix for the session, and
2. the router path falling back to `sidecar_cached_tokens` (which returns 0
   → no sidecar written) when no usage block / prefix measurement exists.

Additionally the save-failure cliff itself worsened post-fix
(ReadTimeout-dominated: 52 of 106 `slot_save` failures in-window, 65.7 %
rate vs 24.3 % pre-fix), shrinking sidecar supply. None of this makes the
restored credit dishonest to report: 0 restored dispatches across ≥24 h and
778 seed decisions is the number, and it is unchanged from the pre-fix
baseline — the credit neither fired nor paid for itself.

**Follow-up item filed: LP-0MVVI8TEU006KTOZ** (bug: writer-side sidecar still
absent at seed time post-fix — 98 % `missing_sidecar`, 0 found/restored over
≥24 h; investigate eviction-sidecar prefix availability, the
`sidecar_cached_tokens` 0-fallback skip, save/retire path divergence and the
save-failure cliff).

## Confounds / notes

- Snapshot-bearing traffic is thin and dominated by short-lived
  audit/child-pi sessions; 617 seed decisions is a small sample, but it is
  larger than the 0-restored baseline span and uniformly negative.
- No attribution of local-prefill movement to this path — the policy in the
  item's Risks/notes applies: `found`/`restored` are zero.
- Pre-existing unrelated red: dev tip carries 2 inherited
  `test_slot_pool_sizing_doc` citation-drift failures (LP-0MVTHCT8W003VAGE);
  unrelated to this measurement.

## Traceability

- child of: LP-0MVJXV0MY000O77E (AC3 remains unmet)
- supersedes: LP-0MVK0E8NH007XCV3 (closed)
- validates: LP-0MVK0E80C009HUQC (`5502ee44`), LP-0MVK0E8C40035FG3 (`97b95fb6`)
- JSON: `docs/dev/slot-ratio-restore-postfix-2026-10-30.json` (24 h pair),
  `docs/dev/slot-ratio-restore-postfix-2026-10-30-full.json` (full + post-restart)
- Extends: `docs/dev/slot-prefix-warmness-measurement-LP-0MVGGPSVM003W35K.md`,
  `docs/dev/slot-ratio-seed-verdict-LP-0MVK0E8C40035FG3.md`
