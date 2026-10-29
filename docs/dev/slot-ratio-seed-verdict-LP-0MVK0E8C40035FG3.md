# Slot-sidecar seed failure mode — verdict from instrumented outcome counts

- **Item:** LP-0MVK0E8C40035FG3 (verification-only child of LP-0MVJXV0MY000O77E)
- **Depends on:** LP-0MVJYZWWW005XGW7 (INFO `slot_ratio_seed outcome=…` logging, shipped v0.1.25)
- **Baseline compared against:**
  [slot-prefix-warmness measurement (LP-0MVGGPSVM003W35K)](slot-prefix-warmness-measurement-LP-0MVGGPSVM003W35K.md)
- **Date of measurement:** 2026-10-29 10:50 UTC
- **Code touched:** none in `proxy/proxy/` (analysis tooling and docs only)

## Verdict

**The failure mode is a snapshot ↔ sidecar lifecycle gap, not a guard rejection.**
Of 4,488 seed decisions over 155 h, **every single one failed at the lookup**
(`missing_sidecar` 75.8 %, `missing_snapshot` 24.2 %), and **no decision ever
reached the guard**: `found`, `expired`, `model_mismatch`, `prefix_mismatch` and
`empty_ratio` are all exactly 0.

Consequently:

- **Candidate mode 1 (prefix mismatch) — ruled out.** 0 of 4,488 decisions
  (0.00 %). The `prefix_matches()` guard is not merely rare, it is *unreachable*
  in the observed traffic: by construction it is only evaluated after a
  `slot_<session>.bin` candidate exists **and** a readable, versioned sidecar is
  found next to it, which never happened.
- **Candidate mode 2 (expired sidecar / 24 h TTL) — ruled out.** 0 of 4,488.
  Additionally, every sidecar that *did* exist was young: the `age_s`
  distribution of the snapshotless sidecars is p50 **1.26 h**, p90 **14.1 h**,
  max **16.05 h**, all far below the 86,400 s `session_slot_ratio_ttl_seconds`
  default. The TTL could not have suppressed a single decision.
- **Candidate mode 3 (lookup miss) — confirmed**, but not for the reason
  recorded at intake ("directory names matched" — they still do). The lookup
  miss is a *pairing* miss: a snapshot without a sidecar, or a sidecar without
  its snapshot. The reader is looking in the right directory; the writer is not
  keeping the two artefacts together.

## The counts

Command (from the repository root; reads every rotated and `.gz` file exactly
once, so no line is double counted and no compressed history is skipped):

```bash
python3 scripts/analyse_slot_ratio_restore.py \
  --log-dir /var/log/llama-proxy \
  --slot-cache-dir /home/rgardler/projects/llm/slot-cache \
  --window "post-instrumentation=2026-10-23T00:00:00,2026-10-29T11:00:00" \
  --window "last24h=2026-10-28T10:50:26,2026-10-29T10:50:26"
```

Add `--json` for the machine-readable form (`windows[].seed_outcomes`). The same
tally by hand, per the recipe in
[`proxy/docs/routing.md`](../../proxy/docs/routing.md):

```bash
cd /var/log/llama-proxy
for o in found missing_sidecar missing_snapshot expired model_mismatch prefix_mismatch empty_ratio; do
  printf '%-18s %s\n' "$o" "$( { zgrep -h "slot_ratio_seed outcome=$o " proxy.log* ; } | wc -l)"
done
```

### Window A — full post-instrumentation span (155.0 h, 2026-10-23 00:00 → 2026-10-29 11:00)

| `outcome` | count | share | sessions | `age_s` p50 / max |
|---|---:|---:|---:|---|
| `found` | **0** | 0.00 % | 0 | — |
| `missing_sidecar` | **3,402** | 75.80 % | 31 | 0 / 0 (not recorded for this outcome) |
| `missing_snapshot` | **1,086** | 24.20 % | 4 | 4,527 (1.26 h) / 57,792 (16.05 h) |
| `expired` | **0** | 0.00 % | 0 | — |
| `model_mismatch` | **0** | 0.00 % | 0 | — |
| `prefix_mismatch` | **0** | 0.00 % | 0 | — |
| `empty_ratio` | **0** | 0.00 % | 0 | — |
| **total** | **4,488** | 100 % | 35 | — |

- lookup outcomes: **4,488 / 4,488 (100 %)**; guard branches reached: **0**.
- `slot_ratio_restored` events: **0**; `cache_state=restored` routing lines: **0**.
- writer side: request-path `slot_save` **550 ok / 124 failed (18.4 %)**;
  eviction-path saves **98 saved / 110 failed across 32 sessions**;
  `slot persistence skipped reason=slot_busy` **1,253**.
- disk: **139** orphan `.ratio.json` sidecars (all `Qwen3`), oldest
  2026-10-20 13:27:34, newest 2026-10-29 10:09:29, and **0** `slot_*.bin`
  snapshots present.

### Window B — last 24 h (2026-10-28 10:50:26 → 2026-10-29 10:50:26)

| `outcome` | count | share | sessions |
|---|---:|---:|---:|
| `missing_sidecar` | 312 | 50.57 % | 6 |
| `missing_snapshot` | 305 | 49.43 % | 3 |
| all other outcomes (incl. `found`, `expired`, `prefix_mismatch`) | **0** | 0.00 % | 0 |
| **total** | **617** | 100 % | 9 |

`missing_snapshot` `age_s` in this window: p50 3,295 s (0.92 h), p90 8,628 s
(2.40 h), max 9,696 s (2.69 h) — again nowhere near the 24 h TTL.

A single 24 h window (617 decisions) is thin but not ambiguous: it contains the
same two lookup outcomes and the same zero guard outcomes as the 155 h window,
so the verdict does not depend on one long window (the risk noted at intake).

## Where the pairing breaks (writer side, code + counts)

1. **Eviction saves never write a sidecar.** `write_slot_ratio_sidecar()` has
   exactly one caller: the request-completion path in `proxy/router.py`.
   `proxy/session.py:session.save_evicted_session_snapshot()` — the
   tenancy/lease reclaim save that exists so an evicted session restores warm —
   writes only the `.bin`. 98 successful eviction saves across 32 sessions in
   window A, overlapping **17 of the 31** sessions that report
   `missing_sidecar`. Those snapshots are structurally unseedable: they can
   never produce a credit, no matter what the guard does.
2. **The request-path sidecar write is conditional.** Even on the request path
   the sidecar is written only when `_measured_prompt_tokens > 0` (no usage
   block extracted ⇒ snapshot saved, sidecar skipped). This is the plausible
   source of `missing_sidecar` for the affected sessions that never had an
   eviction save; the outcome line cannot separate the two, because
   `missing_sidecar` records `stored_messages=0` by design and does not log the
   resolved paths (follow-up instrumentation item filed).
3. **Sidecar lifetime is not coupled to its snapshot's.** The sidecar is written
   *after* the snapshot in a separate best-effort step, and is deleted only from
   `_invalidate_session_and_slot()` when the caller passes a filename; nothing
   else prunes it. The nightly `scripts/cleanup-slot-cache.sh` globs
   `"$SLOT_CACHE_DIR"/slot_*.bin` — **root only, non-recursive**, and it does not
   know about `.ratio.json` at all — so it reports "No slot files found" (all
   live snapshots live in the `localhost-8080/` subdirectory) and never removes
   an orphan sidecar. Result: 139 sidecars on disk with no snapshot, and a
   session whose sidecar outlived its snapshot reports `missing_snapshot` on
   **every** subsequent request, forever.
4. **Concentration, not a long tail.** Two sessions dominate: one long-lived
   agent session (`herdr-17…`, 2,687 decisions: 1,665 `missing_sidecar` +
   1,022 `missing_snapshot`) and one audit session family. Because a broken pair
   persists, a handful of sessions generate thousands of identical failures —
   which is why the *count share* is a property of the defect, not of traffic
   breadth.

## Implied fix

**None of the four candidate guard relaxations is implied by the data** —
"fingerprint the non-system messages only", "stable prompt-prefix anchor",
"credit the last saved overlapping messages" and "re-evaluate the TTL default"
all relax a guard that **never executed**, so each would leave the observed
0 % restored-prefix engagement exactly as it is (and the last would trade a 24 h
TTL for nothing).

The fix is **writer-side lifecycle coupling** in the seed path's producer:

1. write the sidecar on the **eviction** save path too (the session's last
   recorded prompt prefix and prompt-token estimate are already available via
   `record_prompt_prefix`), and
2. write it on the request path even when no usage block was measured, deriving
   `cached_tokens` from the same recorded prefix/estimate instead of skipping
   the write;
3. delete the sidecar wherever the snapshot is deleted (one helper used by every
   unlink site) and teach `scripts/cleanup-slot-cache.sh` to recurse into
   endpoint subdirectories and prune orphaned `.ratio.json` files;
4. leave `prefix_matches()`, the TTL default and the credit arithmetic
   **unchanged**.

**Why this still satisfies AC2 of LP-0MVGD9YXH008O304** ("a snapshot that does
not prefix-match the incoming prompt must not be credited"): the fix repairs
only *artefact pairing*, not the guard. The credit is still applied exclusively
after the stored message fingerprints prefix-match the incoming messages, with
the same measured-tokens × safety-margin cap (AC3). A genuinely changed prompt
still earns nothing; a genuinely reusable prefix finally gets the chance to be
evaluated at all. The guard should be revisited only *after* this fix, and then
from new `slot_ratio_seed` counts — if `prefix_mismatch` appears in volume once
pairs exist, that is the moment the guard is the binding constraint, and the
data will say so directly.

## Validation of the fix (what the sibling item should show)

Expected post-fix signature in a ≥24 h window:

- `slot_ratio_seed outcome=missing_sidecar` ≈ 0 for sessions whose snapshot was
  taken by the eviction path (they now carry a sidecar);
- `slot_ratio_seed outcome=missing_snapshot` ≈ 0 (orphan sidecars pruned with
  their snapshots);
- `found` > 0 and `cache_state=restored` dispatches > 0 — the first real
  evidence that the feature engages;
- guard outcomes (`prefix_mismatch`, `expired`) become *possible* for the first
  time; their counts are the next measurement's question, not this one's.

## Caveats and attribution

- The counts prove the **stage** conclusively (lookup vs guard); the split of
  `missing_sidecar` between "eviction save wrote no sidecar" and "request save
  skipped the sidecar for lack of measured usage" is per-session inferred, not
  logged — hence the follow-up instrumentation item.
- Local-prefill movement in this window must not be attributed to the seed path:
  engagement was 0 throughout (expected-reuse credit, admission ordering and
  other changes dominate). This verdict uses only the seed-outcome and
  slot-save tallies.
- `forcing full prompt re-processing` in `llama-server.log` is expected on this
  hybrid-architecture Qwen3 model and is unrelated (see `AGENTS.md`).
- No credit logic was changed by this item: `git diff` over `proxy/proxy/` is
  empty. The only code change is the seed-outcome tally added to
  `scripts/analyse_slot_ratio_restore.py` (with tests), so this analysis is
  reproducible rather than hand-grepped.
