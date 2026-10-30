# What `session_slot_pool_size` actually controls, and the pool-4 bake-off protocol

**Parent:** `LP-0MVN3KPEC007WWL5` — *Evaluate widening the session-slot snapshot pool
(`session_slot_pool_size` 2 → 4) before adding slots*.
**Delivered by:** `LP-0MVOG8T9F0031JP5` (define pool-4 semantics and bake-off protocol).
**Scope:** documentation only — **no production behaviour change** in this item.
**Verdict of this document:** the parent's "pool 4" is **not a snapshot-retention knob**.
In this codebase `session_slot_pool_size` is simultaneously the llama-server
`--parallel` value, the dispatch-lease concurrency cap, the slot-ID space, and the
persistence on/off gate. A bare `2 → 4` therefore **shrinks per-slot context to 61,440
tokens** and contradicts parent AC5. Making "4 retained snapshots" real without
touching live geometry is **candidate B** (a decoupled snapshot/restore pool), and
that requires the producer's answer to the OPEN QUESTION in §5 before C6/C7 proceed.

Reproduce every number in §2 with the unit test named in §7 — the table is the output
of the production helper, not a hand calculation.

---

## 1. Every behaviour `session_slot_pool_size` controls today

`session_slot_pool_size` is set once in `proxy/config.yaml:1680` (currently `2`). It is
read by **nine** distinct mechanisms. The config comment already warns that it must
match llama-server's `--parallel` (`proxy/config.yaml:1660-1679`).

| # | Behaviour | Mechanism | Citation |
|---|---|---|---|
| 1 | **llama-server `--parallel` / `-np` (live slot count)** — this is what allocates GPU contexts | `session_slot_pool_size` is exported as `LLAMA_PARALLEL`; the start script consumes it for `--parallel` (router mode) and for both `-np` and `--parallel` (single-model mode) | `proxy/proxy/lifecycle.py:1600-1604`; `start-llama.sh:239` (default), `:252` (`--parallel`), `:501` (default), `:511` (`-np`), `:514` (`--parallel`) |
| 2 | **Per-slot physical context** — llama-server divides the total `-c` by the slot count, so per-slot ctx is *derived, never configured* | `local_model_ctx_size // slots` | `proxy/proxy/provider.py:1788` (`effective_per_slot_threshold`), headroom constant `provider.py:1779` (`_LOCAL_ROUTING_OUTPUT_HEADROOM = 4096`) |
| 3 | **Large-context routing clamp (warm ceiling)** — prompts above the clamp bypass local | warm threshold is clamped to `min(configured_warm, effective_per_slot_threshold(ctx, slots))` | `proxy/proxy/provider.py:1900` (`_effective_large_context_thresholds`), used for skip decisions `provider.py:1943`, `provider.py:2107` |
| 4 | **Persistence context cap** — a request larger than the clamp never attempts save/restore, so router and persistence agree | `session_slot_max_prompt_tokens` falls back to `compute_hard_routing_cap` and then to `effective_per_slot_threshold(ctx, slots)` | `proxy/proxy/session.py:1229-1235`, `proxy/proxy/provider.py:2064` (`compute_hard_routing_cap`) |
| 5 | **Dispatch-lease concurrency cap** — how many sessions may hold a local lease at once | `max_local = session_slot_pool_size`, passed into the lease acquire, with the cold-start cap applied on top | `proxy/proxy/router.py:213-232` (`_get_local_max_concurrent_queries`), `router.py:1397` (call site), `proxy/proxy/router_helpers.py:3263` (`_try_acquire_local_dispatch`), `provider.py:6329` (`_get_local_concurrency_info` — `max_local` for the saturation view) |
| 6 | **Slot-ID space** — the numeric range a session may be assigned, and pool exhaustion | `_slot_id_for_session` returns the lowest free id in `[0, pool_size)`, `None` when the pool is full | `proxy/proxy/session.py:1000-1034` |
| 7 | **Persistence on/off gate (global)** — pool size 0 disables all snapshotting | `_slot_persistence_enabled(slot_path, pool) = bool(slot_path and pool > 0)` | `proxy/proxy/session.py:488`, gate applied in `_build_slot_context` at `session.py:1185-1190` and in the eviction-save planner `session.py:848-871` |
| 8 | **Tenancy default ("N tenants, N slots")** — the number of distinct sessions allowed to hold a large-prompt local reservation | `_default_tenants` uses the pool size when `local_session_tenants` is unset | `proxy/proxy/router_helpers.py:2929-2937`; config comment `proxy/config.yaml:1373` (live value `proxy/config.yaml:1381` = 2) |
| 9 | **Capacity-admission slot count** and **status-endpoint fail-open total** | utilisation/admission accounting reads the pool; the `/status` handler reports the pool as `total_slots` when no model is loaded | `proxy/proxy/capacity_admission.py:124-129`; `proxy/proxy/handlers.py:426` (docstring), `:656`, `:702`, `:829` |

Two further couplings that are *not* pool size but are commonly confused with it:

- **Prefill concurrency is independent** — `local_max_concurrent_prefills: 1`
  (`proxy/config.yaml`, prefill-serialisation block) serialises prefills, so extra live
  slots add *decode* concurrency, not prefill throughput. This is the reason the parent
  rejects "more live slots" as the fix.
- **On-disk retention is independent** — snapshot files are already **per session**
  (`slot_<session_id>.bin`, `proxy/proxy/session.py:1126-1128`), so the number of files on
  disk is bounded by the number of *sessions seen*, not by the pool. Retention is governed
  by `scripts/cleanup-slot-cache.sh` (age 7 d, size cap 20 GB, keep-recent 1 d —
  `scripts/cleanup-slot-cache.sh:3-12`, defaults at `:27-29`), never by
  `session_slot_pool_size`. Therefore "more retained snapshots than live slots" is a
  *different change* from "more live slots" — this is the whole point of candidate B.

---

## 2. Per-slot geometry at pool 2 / 3 / 4

Total context is `local_model_ctx_size = 262,144` and the routing output headroom is
`_LOCAL_ROUTING_OUTPUT_HEADROOM = 4,096` (`proxy/proxy/provider.py:1779`). The clamp is
`ctx // slots - headroom` (`provider.py:1788-1805`), the same helper production uses for
routing (`provider.py:1900`) and for the persistence cap (`session.py:1229-1235`).

| Pool N | Raw per-slot ctx (`262144 // N`) | **Routing/warm clamp** (`effective_per_slot_threshold`) | Live GPU contexts |
|---:|---:|---:|---:|
| 2 | 131,072 | **126,976** | 2 |
| 3 | 87,381 | **83,285** | 3 |
| 4 | 65,536 | **61,440** | 4 |

Note the trap that makes the parent's headline number look safer than it is: "65,536 at
N=4" is the *raw* division. The **usable clamp is 61,440** — 4,096 tokens lower — and
that is the number the router, the persistence cap and the warm ceiling all use. The
thrash window's dominant prompt (49,103 tokens, `LP-0MVN1IQTF0037WTG`) is still inside the
N=4 clamp, but the **headroom collapses from 77,873 tokens at N=2 to 12,337 tokens at
N=4**, so any growth in prompt size immediately falls off the local path — and each extra
live slot also multiplies the KV residency the GPU must hold.

Asserted, not hand-computed: `proxy/tests/test_slot_pool_sizing_doc.py` recomputes the
table from `proxy.provider.effective_per_slot_threshold` and asserts the doc's table and
this row-by-row.

---

## 3. The three candidate readings of "4 snapshots" (parent AC3)

Parent AC3 says "retention/eviction semantics for **4 snapshots**". Because pool size is
nine mechanisms at once (§1), that sentence has three mechanically different readings. Each is
recorded with its exact config delta, its live-slot and per-slot-ctx consequence, the
parent AC it violates or satisfies, and the item that owns it.

### Candidate A — `session_slot_pool_size: 4` literally (config-only)

- **Config delta:** `proxy/config.yaml:1680` `session_slot_pool_size: 2 → 4`
  (+ llama-server restart, since it is exported as `--parallel`).
- **Live slots:** 2 → **4** (llama-server `--parallel 4`, four GPU contexts).
- **Per-slot ctx:** 126,976 → **61,440** (clamp).
- **Snapshots retained:** still one file per session on disk; retention unchanged.
- **Consequence:** gains *residency* (4 sessions may hold leases) but pays
  126,976 → 61,440 per-slot context, pushing the 61,440–126,976 prompt band remote. With
  prefill serialised (`local_max_concurrent_prefills: 1`) it adds no prefill throughput.
- **Parent ACs:** **violates AC5** ("no change to live slot count or per-slot ctx"). Also
  cuts against AC2's premise, because the reuse loss from a smaller clamp can swamp the
  reuse gain the item is trying to buy.
- **Owner:** this child / the parent's C6 decision — **not** `LP-0MVGD9ZKI004ETTE`.

### Candidate B — decouple the retained-snapshot pool from the live-slot pool

- **Config delta:** `session_slot_pool_size` stays **2**; add a *separate* snapshot-pool
  size (default 4) governing how many sessions may hold a **retained** snapshot/restore
  reservation, with explicit eviction: least-recently-active slot assignment is freed and
  its `slot_<session>.bin` is saved before reuse
  (`proxy/proxy/session.py:848-871` `plan_eviction_snapshot` /
  `save_evicted_session_snapshot`, already gated by every existing safeguard).
- **Live slots:** unchanged at **2**. **Per-slot ctx:** unchanged at **126,976**.
- **Snapshots retained:** 4 (bounded, with LRU eviction and the existing disk cap).
- **Consequence:** more sessions keep a restorable prefix without shrinking context. The
  cost is restore latency on every swap (measured in `LP-0MVOG8TK6002EWZU`) and the
  exposure to `session_slot_skip_when_busy` / circuit-breaker suppression
  (`proxy/proxy/session.py:667-692`, `:562-580`) when the other live slot is streaming.
- **Parent ACs:** **satisfies AC5** (no change to live slot count or per-slot ctx),
  **satisfies AC3** (explicit retention/eviction semantics for 4 snapshots), and is the
  only candidate that answers AC2's question ("does a wider *snapshot* pool raise reuse?")
  without confounding the measurement with a geometry change.
- **Owner:** the N=2 constraint candidate B preserves is owned by `LP-0MVGCQ3QN009HRE8`
  (slot-count re-evaluation, keeps `session_slot_pool_size: 2`); the decoupling semantics
  are shipped by **C7** `LP-0MVOG8UE60050SKA` (retention/eviction) plus
  `LP-0MVOG8UNX003Z9WN` (observability), both conditional on the §5 answer and on a
  "go" from C6 `LP-0MVOG8U4D0026YCL`.

### Candidate C — virtual slots: N session slots multiplexed onto 2 physical slots

- **Config delta:** `session_slot_pool_size` stays **2** (physical); the proxy gains a
  session→physical-slot multiplexer with save/restore on swap.
- **Live slots:** **2** physical; the proxy-side *session* slot count is a new,
  larger number (the spike proposes e.g. 8).
- **Per-slot ctx:** unchanged at **126,976**.
- **Consequence:** the most capability (N logical sessions per physical slot) and the most
  machinery (a swap scheduler, fairness, cold-start interaction). Note that with a
  session↔physical-slot multiplexer at N_session = 4, candidate C **collapses into
  candidate B**: the two differ only in how much of the multiplexing is made explicit in
  configuration. B is the minimal form of C that answers the parent's question.
- **Parent ACs:** satisfies AC5 and AC3. It is however *out of scope* for the parent's
  AC2 measurement as written, because the reuse uplift would be produced by a new
  component rather than by a config change.
- **Owner:** `LP-0MVGD9ZKI004ETTE` ("Spike: quantify llama-server slot save/restore cost
  as the basis for virtual slots", `open`/`plan_complete`, harness child
  `LP-0MVKU0RG0002WWF2`).

### Comparison

| | A (config pool 4) | B (decoupled snapshot pool) | C (virtual slots) |
|---|---|---|---|
| Config delta | `session_slot_pool_size: 4` | new snapshot-pool key, pool stays 2 | none; new multiplexer |
| Live slots | **4** | 2 | 2 physical, N logical |
| Per-slot ctx (clamp) | **61,440** | 126,976 | 126,976 |
| Retained snapshots | unchanged (per session) | **4**, LRU | N logical |
| Parent AC5 | **violates** | satisfies | satisfies |
| Owner | C6 (this chain) | constraint: `LP-0MVGCQ3QN009HRE8`; ship: C7/C8 (this chain) | `LP-0MVGD9ZKI004ETTE` |

---

## 4. Bake-off protocol (numerically specified)

This protocol is what C4 `LP-0MVOG8TUM005705R` (bake-off replay) and C5
`LP-0MVOG8U4D0026YCL` (go/no-go record) must execute and report. It is specified here so
the decision cannot drift between the two children.

### 4.1 Fixture

| Property | Value |
|---|---|
| Window | **2026-10-24 18:38 → 23:47 local** (5.15 h) — the thrash window |
| Capture | `proxy/tests/fixtures/prefill_thrash_2026_10_24.jsonl` (1,978 attempts, header + one record per local admission attempt) |
| Capture command | `proxy/scripts/capture_prefill_thrash_window.py --start "2026-10-24 18:38" --end "2026-10-24 23:47"` (recorded in the fixture `_meta.capture_command`) |
| Documented totals | 1,978 attempts · **125** `prefill_deadline_exceeded` · 33 idle-bypass over 25k new tokens · modelled waste 778,863 prefill tokens (`_meta.totals`) |
| Session count in fixture | 38 distinct sessions |
| **Baseline to beat** | aggregate prefix reuse **26.1 %**; **median 15.6 % for prompts >20 k tokens** (227 prompts, 28 fully cold), measured on this window by `LP-0MVN1IQTF0037WTG` |
| Arms | **Arm 0** = control, production config (`session_slot_pool_size: 2`, one file per session, no retained-pool bound). **Arm 1** = candidate B (live slots 2, retained snapshot pool 4, LRU eviction) |
| Arm isolation | one arm per run. Candidate B changes no `--parallel` value, so it needs no llama-server restart; if any arm changes geometry (it must not, see G3) record the restart boundary and exclude the arm's own restart from its window |
| Harness | reuse `proxy/benchmarks/slot_snapshot_benchmark.py` (`LP-0MVOG8SYO006R49J`) for save/restore cost, and `scripts/analyse_slot_ratio_restore.py` for reuse/`cache_state` accounting — **do not build a second harness** |

Candidate **A** is *not* an arm of this bake-off. It changes live geometry, so measuring it
here would confound the reuse question with the AC5 violation; A may only be measured under
a separate geometry experiment (see `docs/dev/slot-geometry-ab.md` for that shape).

### 4.2 Metrics (all mandatory, all reported numerically)

| Metric | Definition | Source |
|---|---|---|
| **Reuse uplift (primary)** | aggregate `cached_tokens / estimated_tokens` over the window, and the **median per-prompt reuse for prompts >20 k tokens**, each reported for Arm 0 and Arm 1, with the delta and a paired comparison over the same sessions | proxy log `dispatch_first_byte_ms` / `slot_ratio_restored` / `usage` blocks, via `scripts/analyse_slot_ratio_restore.py` |
| **Restore-success rate (primary)** | `slot_restore` successes / attempts, split by `cache_state` (`cold`/`warm`/`restored`) | `scripts/slot_persistence_harness.py` `slot_restore_events` |
| **`skip_when_busy` exclusions** | count of save/restore decisions suppressed by the load-aware gate (per arm), as a share of attempted persistence decisions | `proxy/proxy/session.py:667-692`; counted as persistence-skip reasons in `scripts/slot_persistence_harness.py` `skip_events` |
| **Circuit-breaker cooldown exclusions** | count of decisions suppressed by the per-slot failure cooldown, per slot id | `proxy/proxy/session.py:562-580`; `slot persistence disabled slot=… cooldown_remaining=…` log line |
| **Save cost / restore cost per band** | wall-clock p50/p90 per band (10 k / 25 k / 40 k / 60 k+), saved bytes, and the adaptive-timeout model `min(3.0 + 0.0015 × tokens, 60)` for comparison | `LP-0MVOG8TK6002EWZU` (C3), `proxy/benchmarks/slot_snapshot_benchmark.py` |
| **Disk footprint** | total `session_slot_save_path` bytes and file count per arm, plus bytes per retained snapshot | `scripts/slot-persistence-harness.py` `slot_cache_inventory`; `scripts/cleanup-slot-cache.sh --dry-run` |
| **Deadline cancellations** | `prefill_deadline_exceeded` count and modelled wasted prefill tokens per arm (baseline 125 / 778,863) | the fixture totals above, re-counted per arm |
| **Local serve share** | local streams / all `Stream started` in the window (guards against an arm that improves reuse by serving less) | `scripts/analyse_slot_ratio_restore.py` |

### 4.3 Pass / fail thresholds C4/C5 apply

The go/no-go is **pass only if all six gates hold**; a single failed gate is a **no-go**
that must be recorded with its number (parent AC4: "record the measured blockers … so the
decision is reproducible").

| Gate | Pass condition | Rationale |
|---|---|---|
| **G1 — reuse uplift** | Arm 1 **median reuse for >20 k-token prompts ≥ 31.2 %** (i.e. ≥ 2 × the 15.6 % baseline) **and** Arm 1 aggregate reuse ≥ 26.1 % + 10 pp (i.e. **≥ 36.1 %**). The paired per-session median delta must also be **> 0**. | 2 × / +10 pp is the smallest uplift that can plausibly change the economics; anything less does not justify new eviction machinery |
| **G2 — restore success** | Arm 1 `slot_restore` success rate **≥ 90 %** of attempts, **and** the sum of `skip_when_busy` + cooldown exclusions **≤ 20 %** of persistence decisions | below 90 % the pool is churn, not reuse; above 20 % suppression the feature is unusable under real load (`session_slot_skip_when_busy: true` is on) |
| **G3 — no geometry drift** | live slots **2** and per-slot clamp **126,976** in both arms (asserted from the running config, not assumed) | enforces parent AC5 mechanically |
| **G4 — cost bounded** | per-band restore wall-clock p90 **within** the adaptive timeout `min(3.0 + 0.0015 × tokens, 60)` for the 10 k and 25 k bands; for 40 k/60 k the median may exceed the 60 s cap only if G1 is met **without** relying on that band | the mechanism currently helps only the <40 k band (parent "Why") |
| **G5 — no regression** | deadline cancellations **≤ 125** and modelled waste **≤ 778,863 tokens**; local serve share **not lower** than Arm 0; disk footprint **≤ the 20 GB cleanup cap** (`scripts/cleanup-slot-cache.sh` default) | an arm that trades wasted prefill for a lost local share is not a win |
| **G6 — reproducibility** | both arms' captures, JSON results and the exact commands committed; rerun of Arm 0 reproduces the 26.1 % / 15.6 % baseline within noise | parent AC2 "reproducible" |

If G1 and G2 pass but G4 fails only for the 60 k+ band, the documented outcome is **go,
band-limited**: ship the pool-4 retention with a `session_slot_max_prompt_tokens`-style
upper bound at 40 k so the expensive band is never swapped. Any other partial pass is a
**no-go**.

### 4.4 Measured outcome (C4's replay, 2026-10-28)

C4 `LP-0MVOG8TUM005705R` executed the §4 protocol offline against the checked-in
fixture (`proxy/tests/pool_snapshot_bakeoff.py`, deterministic replay — no live
config change). Machine-readable evidence: `proxy/benchmarks/results/pool_bakeoff_2026_10_24/summary.json`;
human-readable report: `proxy/benchmarks/results/pool_bakeoff_2026_10_24/RESULTS_NOTE.md`.

**Verdict: candidate B is a no-go on this window under C2's thresholds.** The retained
pool of 4 measures **no reuse uplift** (aggregate 25.83 % vs the calibrated 26.1 %
baseline; >20 k-token median 15.5 % vs the 31.2 % G1 threshold), so **G1 fails**. **G2
fails** on the exclusion share: `skip_when_busy` suppresses 170 of 303 persistence
decisions (56.1 % vs the 20 % ceiling), though every restore that was attempted
succeeded (100 %). G3–G6 pass. Attribution (per-arm exclusion counts in
`summary.json`): 687 of 1,016 unique attempts (67.6 %) exceed the pool-2 clamp and never
reach the local slot at all; the diagnostic arm (busy gate relaxed) is the only
same-geometry configuration that moves the >20 k median (+8.73 pp to 24.33 %, still
below G1).

---

## 5. OPEN QUESTION for the producer (blocks C6/C7)

> **OPEN QUESTION (producer):** Parent AC5 states "**no change to live slot count or
> per-slot ctx** in this item". Candidate A (`session_slot_pool_size: 4`) *necessarily*
> changes both — 2 → 4 live slots and 126,976 → 61,440 per-slot clamp — because in this
> codebase that one key is the llama-server `--parallel` value, the lease cap, the slot-ID
> space and the persistence gate (§1). Candidate B (a decoupled retained-snapshot pool,
> live slots fixed at 2) satisfies AC5 and is the only reading under which "4 snapshots"
> means what the parent's title says.
>
> **Question:** does AC5 forbid candidate A outright — i.e. is **candidate B** the required
> interpretation of AC3's "4 snapshots"? Or is a geometry change acceptable here (which
> would also make the parent's AC2 reuse number non-comparable with the 15.6 % baseline,
> and would need the separate 3/4-slot discussion in `LP-0MVGCQ3QN009HRE8` reopened)?
>
> **Until this is answered, C6 and C7 remain conditional:** C6
> (`LP-0MVOG8U4D0026YCL`, go/no-go) may execute the §4 bake-off and record numbers, but
> may not record a "go"; C7 (`LP-0MVOG8UE60050SKA`, retention/eviction semantics) must not
> be implemented. C8 (`LP-0MVOG8UNX003Z9WN`, observability) is unaffected — the fields it
> adds are useful for either answer.

This question is mirrored in the parent item's Appendix.

---

## 6. What this document does *not* decide

- It does **not** change configuration, code or tests beyond the doc-consistency unit
  test named below (parent AC6: documentation only).
- It does **not** claim reuse *will* improve: the 2026-10-24 restoration economics
  (~75 s for 50 k tokens, capped at 60 s) mean the mechanism is only expected to pay in
  the <40 k band. That is precisely what G1/G4 test.
- It does **not** re-open the N=2 live-slot verdict
  (`docs/slot-count-reevaluation-LP-0MVGCQ3QN009HRE8.md`); candidate B keeps that verdict
  intact by construction.

---

## 7. Files and verification

- This document: `docs/dev/slot-snapshot-pool-sizing-LP-0MVN3KPEC007WWL5.md` (parent AC6
  links it from `LP-0MVN3KPEC007WWL5`).
- Table verification: `proxy/tests/test_slot_pool_sizing_doc.py` — asserts the §2 table
  equals `proxy.provider.effective_per_slot_threshold(262144, N)` for N = 2/3/4, asserts
  the clamp-vs-raw gap is exactly `_LOCAL_ROUTING_OUTPUT_HEADROOM`, and re-checks every
  `file:line` citation in §1 against the current source (a 60-line drift tolerance keeps
  unrelated commits on the shared `dev` branch from reddening the suite, while a moved or
  removed symbol still fails).
- Prior evidence: `docs/slot-count-verdict-lp-0mv9ungeb009xuz5.md` (N=2 verdict),
  `docs/slot-count-reevaluation-LP-0MVGCQ3QN009HRE8.md` (per-slot table at 95 % of
  per-slot, concurrency vs capacity), `docs/dev/slot-geometry-ab.md` (the 1 vs 2 live-slot
  experiment shape, which candidate A would re-run).
- Related items: `LP-0MVN1IQTF0037WTG` (reuse baseline 26.1 % / 15.6 %), `LP-0MVGD9ZKI004ETTE`
  (virtual slots — candidate C), `LP-0MVGCQ3QN009HRE8` (slot-count re-evaluation),
  `LP-0MVOG8SYO006R49J` (save/restore harness), C3 `LP-0MVOG8TK6002EWZU`, C4
  `LP-0MVOG8TUM005705R`, C5 `LP-0MVOG8U4D0026YCL`, C7 `LP-0MVOG8UE60050SKA`, C8
  `LP-0MVOG8UNX003Z9WN`.
