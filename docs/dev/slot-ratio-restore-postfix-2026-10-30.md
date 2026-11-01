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

The verdict was **re-confirmed on 2026-10-31T04:10Z** over a 39.17 h post-fix
window (895 seed decisions) with an independent `zgrep` over the whole log
history — see [Addendum: 2026-10-31 re-measurement](#addendum-2026-10-31-re-measurement-extended-window).
The only movement is on AC6: orphan sidecars fell to 0 and the first cleanly
**paired** snapshot+sidecar save is on disk.

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
  `docs/dev/slot-ratio-restore-postfix-2026-10-30-full.json` (full + post-restart),
  `docs/dev/slot-ratio-restore-postfix-2026-10-31.json` (extended re-measurement)

## Addendum: 2026-10-31 re-measurement (extended window)

Purpose: the 2026-10-30 report answered the 24 h requirement; this addendum
extends the post-fix window to the full span available at
`2026-10-31T04:09Z` (39.17 h, i.e. 15 h beyond AC1's ≥24 h minimum) so the
zero-result verdict rests on a larger sample, and re-runs the AC6 pairing check
against the current disk state. **The verdict is unchanged.**

### Command line (exact, reproducible)

```bash
python3 scripts/analyse_slot_ratio_restore.py \
  --log-dir /var/log/llama-proxy \
  --slot-cache-dir /home/rgardler/projects/llm/slot-cache \
  --window "post-fix=2026-10-29T12:58:32,2026-10-30T12:58:32" \
  --window "pre-fix=2026-10-28T12:58:32,2026-10-29T12:58:32" \
  --window "post-fix-latest-24h=2026-10-30T04:09:00,2026-10-31T04:09:00" \
  --window "post-fix-extended=2026-10-29T12:58:32,2026-10-31T04:09:00" \
  --window "post-restart2=2026-10-31T00:41:55,2026-10-31T04:09:00" \
  --json > docs/dev/slot-ratio-restore-postfix-2026-10-31.json
```

Independent cross-check (gz-aware, whole log history, no tool involved):

```bash
cd /var/log/llama-proxy
zgrep -oh "slot_ratio_restored" proxy.log* | wc -l          # -> 0
zgrep -oh "cache_state=restored" proxy.log* | wc -l        # -> 0
zgrep -oh "slot_ratio_seed outcome=[a-z_]*" proxy.log* | sort | uniq -c
#  3863 slot_ratio_seed outcome=missing_sidecar
#  1130 slot_ratio_seed outcome=missing_snapshot
#     0 found / expired / model_mismatch / prefix_mismatch / empty_ratio
```

### Headline counts (extended windows)

| Metric | post-fix 24 h | latest 24 h (10-30T04:09 → 10-31T04:09) | post-fix extended (39.17 h) | post-restart2 (3.45 h) |
|---|---|---|---|---|
| `cache_state=restored` dispatches | **0** | **0** | **0** | **0** |
| `slot_ratio_restored` seed events | **0** | **0** | **0** | **0** |
| `slot_ratio_seed outcome=found` | **0 (0 %)** | **0 (0 %)** | **0 (0 %)** | **0 (0 %)** |
| `missing_sidecar` | 605 (98.1 %) | 455 (95.8 %) | **863 (96.4 %)** | 45 (100 %) |
| `missing_snapshot` | 12 (1.9 %) | 20 (4.2 %) | 32 (3.6 %) | 0 |
| `expired` / `model_mismatch` / `prefix_mismatch` / `empty_ratio` | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| guard outcomes (seed past lookup) | 0 | 0 | **0** | 0 |
| seed decisions (total) | 617 | 475 | **895** | 45 |
| routing decisions | 10,248 | 11,983 | 19,704 | 562 |
| slot saves ok / failed | 34 / 65 (65.7 %) | 22 / 54 (71.1 %) | 46 / 94 (**67.1 % fail**) | 0 / 4 |
| `slot persistence skipped reason=slot_busy` | 260 | 191 | 375 | 28 |
| `slot_restore failed` | 0 | 0 | 0 | 0 |

Across the whole log history the guard outcomes (`expired`,
`model_mismatch`, `prefix_mismatch`, `empty_ratio`) remain **0**: the seed path
never reaches the guard because it never gets past `read_slot_ratio_sidecar()`.
That is the defining signature of the writer-side gap, not of a TTL or prefix
guard rejecting good sidecars.

### AC2 restated (`dispatch_first_byte_ms`, the `dispatch_to_first_byte_ms` field of the AC)

The field is `dispatch_first_byte_ms` in the log and in
`scripts/analyse_slot_ratio_restore.py`; the AC's `dispatch_to_first_byte_ms`
names the same measurement. Restored group **n=0** in every window and bucket,
so the restored-vs-cold comparison cannot be stated. Cold/warm reference, same
context buckets, extended window:

| Bucket | cold | warm | restored |
|---|---|---|---|
| ≤ 38,000 | n=440, p50 6.17 / p90 49.30 | n=961, p50 2.27 / p90 11.17 | n=0 |
| 38,000–120,000 | n=1, p50 47.00 / p90 47.00 | n=3, p50 84.44 / p90 96.09 | n=0 |

### AC6 pairing check at 2026-10-31T04:12Z

Exact item command, re-run:

```bash
cd /home/rgardler/projects/llm/slot-cache
echo "snapshots=$(find . -name '*.bin' | wc -l) sidecars=$(find . -name '*.bin.ratio.json' | wc -l)"
# snapshots=3 sidecars=1
find . -name '*.bin' -exec basename {} \; | sed 's/\.bin$//' | sort > /tmp/snaps
find . -name '*.bin.ratio.json' -exec basename {} \; | sed 's/\.bin\.ratio\.json$//' | sort > /tmp/sides
comm -13 /tmp/snaps /tmp/sides | wc -l   # orphan sidecars -> 0
comm -23 /tmp/snaps /tmp/sides | wc -l   # orphan snapshots -> 2
comm -12 /tmp/snaps /tmp/sides | wc -l   # paired -> 1
```

| Report time | snapshots | sidecars | paired | orphan sidecars | orphan snapshots |
|---|---|---|---|---|---|
| plan-time baseline (2026-10-29T17:40Z) | 6 | 144 | 2 | 142 | 4 |
| first report (2026-10-30T21:40Z) | 0 | 28 | 0 | 28 | 0 |
| **this addendum (2026-10-31T04:12Z)** | 3 | 1 | **1** | **0** | 2 |

What changed, and what it means for AC6:

- The legacy orphan-sidecar pile (142 → 28 → **0**) is fully cleared by the
  recursive `cleanup-slot-cache.sh` retirement path, so the writer-side fix's
  *retirement* half is demonstrably working.
- For the first time since the deploy there is a **paired** save on disk:
  `slot_herdr-1793415813-4139491-4538.bin` +
  `localhost-8080/slot_herdr-1793415813-4139491-4538.bin.ratio.json`, written
  2026-10-31T04:09:08Z, carrying `cached_tokens=20824`, `prompt_tokens=20824`
  and 4 message fingerprints. That is direct evidence the writer path *can*
  produce a complete, seedable sidecar on the deployed tree.
- The 2 orphan snapshots are both post-deploy and both recent:
  `slot_audit-LP-0MVSD5RJW008F1IB-parent-7caeb9ce.bin` (01:14:50Z) and
  `slot_herdr-1793415438-4119054-16480.bin` (04:09:37Z, a session still live
  at report time, so its sidecar may yet be written).
- AC6 asks for "paired files present, no orphan growth": *paired files
  present* is now satisfied, but orphan growth for post-deploy saves is **not**
  eliminated (2 orphan snapshots; the audit-session one is a genuine orphan).
  AC6 therefore remains **partially met** rather than met.
- The pairing gap is also why `missing_sidecar` still dominates at seed time:
  a snapshot exists for the session, `read_slot_ratio_sidecar()` returns
  `None`, and the seed aborts before the guard. Root-cause work continues in
  LP-0MVVI8TEU006KTOZ, which was updated with these new on-disk observations.
- Extends: `docs/dev/slot-prefix-warmness-measurement-LP-0MVGGPSVM003W35K.md`,
  `docs/dev/slot-ratio-seed-verdict-LP-0MVK0E8C40035FG3.md`
