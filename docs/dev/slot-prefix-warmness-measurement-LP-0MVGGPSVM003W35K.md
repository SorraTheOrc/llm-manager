# Restored slot-prefix warmness — ≥24 h before/after measurement

**Work item:** LP-0MVGGPSVM003W35K (AC5 measurement for LP-0MVGD9YXH008O304)
**Report date:** 2026-10-22 · **Data:** `/var/log/llama-proxy` (proxy logs, `.gz` aware)
**Tool:** `scripts/analyse_slot_ratio_restore.py`; machine-readable figures in
`docs/dev/slot-prefix-warmness-metrics-LP-0MVGGPSVM003W35K.json`

## Verdict

**The restored-prefix warmness change did not engage, so it cannot be
confirmed effective — and the observed local-prefill improvement must NOT be
attributed to it.** Follow-up: **LP-0MVJXV0MY000O77E**. In a **54.4 h** post-deploy window the proxy logged **zero**
`slot_ratio_restored` seed events and **zero** `cache_state=restored` lines,
although the sidecar writer (same code path) produced **37** `slot_*.bin.ratio.json`
sidecars and sessions that own sidecars hit the routing gate **2,918** times
after their sidecar existed (33 sessions). AC3 has no samples to compare; AC5
therefore triggers: a follow-up work item is created to instrument and fix the
seeding path.

The headline metrics moved in the expected direction, but the movement is
explained by *other* changes that landed in the same window (expected-reuse
credit LP-0MVI2XOTY008Q4UV, idle-aware admission LP-0MVGD9Z8Z0055Y5J,
prefill serialisation LP-0MVGDUUV3001XFRJ, capacity gate LP-0MVGDUVS6003T2IX,
prefill-ETA admission LP-0MVIQ0F8T002FS42 / LP-0MVIQ0G2M0006QF6, contention
max-wait 120→60 s LP-0MVIQ0EXV00673HO, and the operator's in-flight **ubatch
1024→2048 A/B** on `models.ini`/`start-llama.sh`).

## Windows

| Window | Local span | Hours |
| --- | --- | ---: |
| Baseline (intake) | 2026-10-19 07:23:12 → 2026-10-20 07:23:12 | 24 |
| Post-deploy (full) | 2026-10-20 13:27:34 → 2026-10-22 19:50:00 | 54.37 |
| Post-deploy (last 24 h) | 2026-10-21 19:50:00 → 2026-10-22 19:50:00 | 24 |

Post-deploy start is the first `slot_<session>.bin.ratio.json` sidecar write
(2026-10-20 13:27:34), i.e. the first traffic served by the LP-0MVGD9YXH008O304
build. All timestamps are log-local (IST, UTC+2).

## Results

| Metric | Baseline (intake) | Post-deploy (full) | Post-deploy (last 24 h) |
| --- | ---: | ---: | ---: |
| Routing decisions (`routing_check`) | 6,734 | 15,021 | 5,021 |
| `cached_ratio=0.00` decisions | 5,616 (83.4 %) | 10,783 (71.8 %) | 3,430 (68.3 %) |
| Mean `cached_ratio` | 0.136 | 0.245 | 0.292 |
| `large_context_bypass` | 1,284 | 4,111 | 1,340 |
| Local skips (total) | 4,566 | 9,762 | 2,933 |
| Local serve share | 411 / 17,459 (2.4 %) | 1,543 / 19,684 (7.8 %) | 862 / 6,659 (12.9 %) |
| Local prefill samples | 411 | 1,543 | 862 |
| Local prefill p10 | 4.43 s | 0.78 s | 0.80 s |
| Local prefill **p50** | 74.45 s | 6.99 s | 6.09 s |
| Local prefill p90 | 297.0 s | 65.1 s | 47.9 s |
| Local prefill max | 817.3 s | 591.6 s | 591.6 s |
| `cache_state=warm` dispatches | n=203, p50 69.6 s | n=1,203, p50 5.95 s | n=727, p50 5.31 s |
| `cache_state=cold` dispatches | n=208, p50 81.1 s | n=340, p50 13.4 s | n=135, p50 13.0 s |
| `cache_state=restored` dispatches | 0 | 0 | 0 |
| `slot_ratio_restored` seeds | 0 | 0 | 0 |
| `slot_save` ok / failed | 56 / 5 (8.2 %) | 211 / 15 (6.6 %) | 110 / 8 (6.8 %) |
| `slot_restore failed` | 0 | 0 | 0 |

Sidecar inventory at report time: **37** `slot_*.bin.ratio.json` (model Qwen3,
oldest 2026-10-20 13:27:34, newest 2026-10-22 19:43:10) — the writer is active.

### Like-for-like: local prefill by context bucket (routing estimate)

| Bucket | State | Baseline | Post-deploy (full) | Post-deploy (last 24 h) |
| --- | --- | ---: | ---: | ---: |
| ≤ 38,000 | warm | n=101, p50 27.5 s, p90 157.4 s | n=609, p50 4.37 s, p90 30.3 s | n=320, p50 4.43 s, p90 28.8 s |
| ≤ 38,000 | cold | n=195, p50 76.7 s, p90 288.3 s | n=318, p50 12.6 s, p90 80.8 s | n=124, p50 10.6 s, p90 78.5 s |
| 38,000–120,000 | warm | n=101, p50 111.3 s, p90 371.5 s | n=594, p50 8.6 s, p90 89.0 s | n=407, p50 6.13 s, p90 57.1 s |
| 38,000–120,000 | cold | n=11, p50 189.2 s, p90 313.4 s | n=22, p50 106.6 s, p90 382.6 s | n=11, p50 72.8 s, p90 326.7 s |
| 38,000–120,000 | **restored** | — | **0 samples** | — |
| > 120,000 | any | — (no local dispatch) | — (no local dispatch) | — |

## Acceptance criteria

1. **≥24 h post-deploy window analysed side-by-side with baselines** — DONE
   (54.4 h window, tables above; tool reproduces every figure).
2. **`cache_state=restored` counted separately from warm/cold** — partially:
   the tool reports restored separately (`0` everywhere) and separates restored
   wins from genuine hits; the report tooling that LP-0MVGCQ421005IXGC owns
   (local prefill/busy metrics) is still open and does **not** yet render
   restored counts — noted on that item.
3. **Seeded dispatches lower first-byte than cold at same context bucket** —
   NOT MEASURABLE: zero seeded dispatches. The like-for-like table above is the
   warm-vs-cold evidence only; there are no restored samples at any bucket.
4. **`slot_save`/`slot_restore` failure rates reported** — DONE: fail rates
   8.2 % → 6.6 % (full) / 6.8 % (last 24 h); `slot_restore failed` 0 in all
   windows. Save/restore unreliability (5 events in the baseline window) is
   **ruled out** as the reason for zero seeds: saves mostly succeed, sidecars
   are written, and the failure rate did not rise.
5. **Change confirmed effective, or follow-up item** — FOLLOW-UP: the change
   made zero observable contact with routing, so it is not confirmed; a
   concrete follow-up (diagnose why seeding never fires; instrument it) is
   filed as **LP-0MVJXV0MY000O77E** (Instrument and fix the slot sidecar seed
   path), with the evidence, hypotheses and re-measurement plan.

## Why zero seeds? (evidence and hypotheses)

The seed path requires (a) the sidecar file be found under
`{session_slot_save_path}/{host}-{port}/slot_<session>.bin.ratio.json`,
(b) model match, (c) TTL ≥ 0, and (d) the stored message fingerprints to be a
prefix of the incoming messages; only then does it log
`slot_ratio_restored ...` at INFO.

Observed:
- the writer runs (37 sidecars) and names files exactly as the reader looks
  them up (`slot-cache/localhost-8080/`, endpoint `localhost:8080`);
- 33 sessions with sidecars reached the local routing gate **2,918** times
  after their sidecar existed (incl. right after the 19:27 and 19:41 restarts,
  when the in-memory ratio was empty and `cache_state=cold` was logged — e.g.
  session `herdr-1792667491-…` with a sidecar saved at 14:13 scored `cold`
  at 19:30 with `cached_ratio=0.00`, messages 696) — yet no seed ever logged.

The remaining suspects, in order of likelihood:
1. **Prefix mismatch** — the guard's `prefix_matches` check fails on real
   resumed conversations. The sidecar stores fingerprints of the *full*
   message list it saved; a resumed request that compacted/rewrote its history
   (pi does client-side compaction; the proxy's own compaction is disabled —
   `compaction_bypass_eval … reason=compaction_disabled`), or whose first
   message (system prompt) is regenerated per request, fails the prefix check
   and earns nothing. The check is exactly AC2's guard, but as designed it can
   never credit the common case.
2. **Snapshot staleness** — `slot_save` for large sessions long before the
   resume (sidecar at 276 messages vs 696 at routing), so compaction of the
   front is more likely and the credit window (24 h TTL) is short relative to
   how long a session stays alive.
3. **Endpoint string drift between save and seed** — checked and inconsistent
   with the data (dir names match), but worth confirming by logging the
   candidate list.

The current code logs seed outcomes only at DEBUG (`slot_ratio_sidecar
skipped reason=prefix_mismatch …`), invisible at the INFO level the proxy runs
at — that is the instrumentation gap the follow-up fixes.

## Confounds (why the headline improvements are not attributable)

- **Code drift:** the baseline→post window includes the idle-admission,
  prefill-serialisation, capacity-gate, expected-reuse credit, prefill-ETA and
  contention changes listed above; each shifts routing, ratio, and prefill
  latency independently.
- **Operator A/B in flight:** `models.ini` / `start-llama.sh` raise
  `ubatch-size` 1024→2048 (LP-0MVIQ0FSX006HQOS); the A/B restarts llama-server
  and empties the prompt cache, biasing first-byte stats.
- **Restart churn:** the proxy restarted repeatedly in-window (e.g. 10-20 13:27,
  10-22 03:55, 07:27, 13:27, 19:27, 19:41); warmness across restarts is
  precisely the scenario the feature targets, so this is a *relevant* window —
  it makes the zero-seed finding stronger, though it also adds noise.
- **Remote spend:** 89–98 % of streams still remote (remote fallback, rate
  limits); per-count metrics (bypasses) are not normalised to demand size.

## How to reproduce

```bash
# Query a single window (JSON):
./scripts/analyse_slot_ratio_restore.py \
  --window 'post=2026-10-20T13:27:34,2026-10-22T19:50:00' --json
# Default comparison (baseline + post full + post last-24h), with sidecar inventory:
./scripts/analyse_slot_ratio_restore.py --report-end '2026-10-22 19:50:00' \
  --slot-cache-dir /home/rgardler/projects/llm/slot-cache
```

The extractor reads every `proxy.log*` exactly once (`.gz`-aware); it does not
concat-compress the files the way the intake's ad-hoc commands did, so figures
are reproducible and complete. The recomputed baseline (83.4 % ratio-0, 1,284
bypasses, 2.4 % serve share, p50 74.5 s / p90 297 s, 251→203 warm / 212→208
cold dispatch states, slot_save 56/5) agrees with the intake's published
baselines to within measurement noise (published: 82.8 %, 1,284, 2.6 %,
61.8 s median, 251/212).
