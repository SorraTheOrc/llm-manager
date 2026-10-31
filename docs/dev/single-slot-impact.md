# Single-slot impact: derived vs static inventory

**Work item:** LP-0MVTV6FGT000A19P (Inventory derived vs static slot mechanisms)
**Parent:** LP-0MVTF26NP003KXKJ (Investigate what happens if I switch to a single
slot)
**Sibling slices of the same parent**

| Slice | Work item | What it fills in this report |
|---|---|---|
| Pinned derivations | LP-0MVTV6F65005HKQZ (completed) | The executable assertions every derived number below is taken from |
| Throughput prediction | LP-0MVTV6G0N0053V0G | [§5](#5-throughput-prediction-reserved-for-lp-0mvtv6g0n0053v0g) |
| Metrics gap list | LP-0MVTV6GA8002P7QU | [§6](#6-metrics-gap-list-reserved-for-lp-0mvtv6ga8002p7qu) |
| Doc cross-links and 2-slot claims | LP-0MVTV6GK1001ED02 | Corrections to the other slot documents |

---

## 0. Scope (what this report is and is not)

- **In scope:** an inventory of every routing/balancing mechanism and metric on
  the local dispatch path, each with its value at the shipped two-slot geometry,
  its value at one slot, a derived/static classification and `file:line`
  evidence ([§3](#3-inventory)); the headline answer — how many items adjust
  themselves, which stay static, and the exact config keys a coherent switch
  needs ([§1](#1-verdict)); and the provenance of every number
  ([§4](#4-method-and-provenance)).
- **This is a prediction plus an auto-adjustment audit, not the geometry
  verdict.** Whether one slot at 262,144 tokens is better than two at 131,072
  is decided by the measured A/B in LP-0MVMHNUXW0061FGN (protocol:
  `docs/dev/slot-geometry-ab.md`). Nothing here pre-empts that verdict.
- **No runtime behaviour changed in this work.** The deliverable is this
  document; the only executable artefact is the doc-fidelity test
  `proxy/tests/test_single_slot_inventory_doc.py`, which re-derives the table's
  numbers through the production helpers and pins the `key: old -> new` list
  against the shipped `proxy/config.yaml`.
- **Assumed geometry for "one slot".** `session_slot_pool_size: 1` with
  `local_model_ctx_size: 262144` unchanged, i.e. 1 x 262,144 per-slot context
  and a 258,048 clamp — the A/B's Arm B geometry
  (`docs/dev/slot-geometry-ab.md:9`-`20`). A reduced total context is a
  different question and is not analysed here.
- **Read-only runtime inspection only.** The observability rows below were
  audited from the endpoints' code paths and payload documentation, not by
  polling a live proxy: no GPU load was added and no slot was contended for.

---

## 1. Verdict

### 1.1 Headline counts

The inventory has **41 rows**: **17 auto-adjust** (`derived-auto`), **5 are
static and need a manual change** (`static-needs-manual-change`) and **19 are
not affected by the slot count at all** (`not-applicable` — their value is
identical at both geometries **by design**, which for the cold threshold and
the prefill cap is a requirement, not an accident).

The short answer: **nothing on the routing path silently keeps two-slot
behaviour.** Every routing *threshold* and every *concurrency* bound is either
re-derived from `session_slot_pool_size` at request time or is a static
constant that does not interact with the slot count. What does **not** follow
the slot count is the small set of constants in §1.3 — and one of them
(`local_session_tenants`) is an explicit value whose own documented default
would have followed.

### 1.2 The items that stay static

Five rows are `static-needs-manual-change`; four are config keys and one is a
code literal:

1. `session_slot_pool_size` — the switch itself (`proxy/config.yaml:1704`).
2. `local_large_context_economic_ceiling_ratio` — the economic-ceiling ratio
   (`proxy/config.yaml:1410`).
3. `local_large_context_warm_cache_threshold` — the absolute warm cap
   (`proxy/config.yaml:1332`).
4. `local_session_tenants` — the large-prompt tenancy bound
   (`proxy/config.yaml:1386`). **This is the finding worth acting on:** the
   key is an *explicit* `2`, and an explicit value bypasses the "N tenants, N
   slots" default (`proxy/proxy/router_helpers.py:3112`-`3134`), so it does
   **not** follow a switch to one slot. Left at `2`, two sessions could each
   hold a large-prompt reservation in a one-slot pool and evict one another.
   This key is absent from the A/B arm table
   (`docs/dev/slot-geometry-ab.md:116`-`121`), which is why it is called out
   here.
5. `handlers.py`'s fail-open pool literal of **3 slots**
   (`proxy/proxy/handlers.py:683`, `:729`, `:859`). Three call sites read
   `server_cfg.get("session_slot_pool_size", 3) or 3`, so the stale `3` is used
   only when the key is entirely absent — but then `/llama/local/status`
   advertises three slots at *either* geometry. The configured value wins
   whenever the key is present, so this is a latent defect rather than a live
   one; it is listed because a single-slot switch makes a 3-slot advertisement
   more wrong, not less.

Everything else is either derived (§3, `derived-auto`) or provably
slot-invariant (§3, `not-applicable`).

### 1.3 The exact `key: old -> new` list for a coherent single-slot switch

```text
session_slot_pool_size:                      2      -> 1
local_large_context_economic_ceiling_ratio:  0.4578 -> 0.9844
local_large_context_warm_cache_threshold:    120000 -> 258048
local_session_tenants:                       2      -> 1
```

Confirmed **unchanged** (and they must stay unchanged):

```text
local_large_context_cold_cache_threshold:    38000  -> 38000
local_model_ctx_size:                        262144 -> 262144
```

The first three keys are the A/B arm table's patch list
(`docs/dev/slot-geometry-ab.md:116`-`121`). The fourth is this report's
addition: `local_session_tenants: 2` (`proxy/config.yaml:1386`) is explicit, so
it does not track the pool the way its documented default would. The cold
threshold stays at 38,000 because it is an economic **new-token** threshold whose
only requirement is to stay below the warm ceiling, so the `(cold, warm]` band
in which the cached-ratio check operates never collapses
(`proxy/proxy/provider.py:2036`; LP-0MSI2M5BT004BCDP).

Note that the ratio and the warm cap are a **co-change**: the absolute warm
value is only an upper bound on `round(ratio × local_model_ctx_size)`, so
raising the absolute alone moves nothing
(`proxy/proxy/provider.py:1962`-`1974`; LP-0MVHSET0A008VU95).

### 1.4 Why doubling the per-slot context routes nothing extra on its own

At one slot the physical clamp doubles from **126,976** to **258,048**
(`proxy/proxy/provider.py:1890`), and that is the only routing threshold that
moves. The effective warm ceiling stays at **120,000**, because:

- the economic term is `round(0.4578 × 262144) = 120,010` and scales with the
  **model** context, not the slot count (`proxy/proxy/provider.py:1943`);
- the configured absolute cap `local_large_context_warm_cache_threshold`
  (120,000) bounds that 120,010 **down** to 120,000
  (`proxy/proxy/provider.py:1972`; `proxy/config.yaml:1332`);
- the effective warm ceiling is the **minimum** of the physical clamp and the
  economic ceiling (`proxy/proxy/provider.py:2043`-`2054`), i.e.
  `min(258,048, 120,000) = 120,000`.

So with today's constants the physical term is not the binding one at either
geometry: `min(126,976, 120,000) = 120,000` at two slots and
`min(258,048, 120,000) = 120,000` at one. **Doubling the per-slot context
routes nothing extra on its own** — the extra 138,048 tokens of per-slot
capacity are unreachable for routing until the ratio and the absolute cap are
raised together (rows 4 and 6 of §1.3). That is exactly why the A/B's Arm B
patch moves both.

This is executable, not narrative:
`proxy/tests/test_provider_per_slot_ctx.py::TestSingleSlotThresholdDerivations::test_absolute_warm_cap_dominates_physical_clamp_at_1_slot`
asserts the 120,010 ratio term, the 120,000 capped ceiling, the 258,048 clamp,
`warm_1 == 120000`, `warm_1 < clamp_1` and `warm_1 == warm_2`.

---

## 2. Classification vocabulary

Three canonical values, used for every row and asserted by
`proxy/tests/test_single_slot_inventory_doc.py`:

| Value | Meaning |
|---|---|
| `derived-auto` | Computed at request/response time from `session_slot_pool_size` (and `local_model_ctx_size`). Its value follows a geometry change with no edit — including values that are legitimately invariance, e.g. the economic ceiling, which is derived from the model context and therefore correctly stays put. |
| `static-needs-manual-change` | A constant (config key or code literal) whose value must be edited by hand for a coherent single-slot switch, or which would otherwise contradict the new pool size. |
| `not-applicable` | Not slot-geometry dependent. The value is identical at both geometries **by design** and must stay identical (an economic threshold that would collapse a band, a prefill cap that already serialises, a metric with no slot dimension). |

---

## 3. Inventory

Every row records the four fields required by AC2: **value @ 2 slots**,
**value @ 1 slot**, **classification** and **evidence**. Derived values are
those pinned by LP-0MVTV6F65005HKQZ — the test names are given per group, and
§4 lists the derivation-to-test mapping.

### 3.1 Slot geometry and the routing thresholds

| # | Mechanism / metric | Value @ 2 slots | Value @ 1 slot | Classification | Evidence |
|---|---|---|---|---|---|
| 1 | `session_slot_pool_size` — the geometry switch; exported as `LLAMA_PARALLEL` for llama-server `--parallel` | 2 | 1 | `static-needs-manual-change` | `proxy/config.yaml:1704`; `proxy/proxy/lifecycle.py:1658` |
| 2 | Per-slot context (`local_model_ctx_size // slots`, realised by llama-server's `--ctx-size` / `--parallel`) | 131,072 | 262,144 | `derived-auto` | `proxy/config.yaml:1431`; `docs/dev/slot-geometry-ab.md:14` |
| 3 | Physical per-slot clamp (`ctx // slots − 4,096` output headroom) | 126,976 | 258,048 | `derived-auto` | `proxy/proxy/provider.py:1890` (`effective_per_slot_threshold`) |
| 4 | `local_large_context_economic_ceiling_ratio` | 0.4578 | 0.4578 | `static-needs-manual-change` | `proxy/config.yaml:1410` |
| 5 | Economic ceiling ratio term (`round(ratio × local_model_ctx_size)`) | 120,010 | 120,010 | `derived-auto` | `proxy/proxy/provider.py:1943` (`effective_large_context_economic_ceiling`) |
| 6 | `local_large_context_warm_cache_threshold` — absolute warm cap, an upper bound on row 5 | 120,000 | 120,000 | `static-needs-manual-change` | `proxy/config.yaml:1332`; bound applied at `proxy/proxy/provider.py:1972` |
| 7 | Economic ceiling after the absolute bound (`min(rows 5, 6)`) | 120,000 | 120,000 | `derived-auto` | `proxy/proxy/provider.py:1972` |
| 8 | Effective warm ceiling — `min(physical clamp, economic ceiling)`, then clamped by the hard-routing cap | 120,000 | 120,000 | `derived-auto` | `proxy/proxy/provider.py:2043`-`2068` (`_effective_large_context_thresholds`) |
| 9 | `local_large_context_cold_cache_threshold` — economic new-token threshold, never clamped or scaled | 38,000 | 38,000 | `not-applicable` | `proxy/config.yaml:1297`; unclamped at `proxy/proxy/provider.py:2036` |
| 10 | Hard-routing cap (`round(ratio × min(warm, clamp))`); both ratio keys retired to `0` → disabled | 0 (disabled) | 0 (disabled) | `derived-auto` | `proxy/proxy/provider.py:2166` (`compute_hard_routing_cap`); `proxy/config.yaml:1412` |

Pinned by: `TestSingleSlotThresholdDerivations::test_four_derivations_pinned_per_geometry`,
`::test_absolute_warm_cap_dominates_physical_clamp_at_1_slot`,
`::test_cold_warm_band_non_empty_and_cold_slot_invariant`,
`TestHardRoutingCapGeometryDerivation::test_ratio_half_cap_pinned_per_geometry`,
`::test_shipped_config_cap_disabled_per_geometry`,
`TestShippedGeometryMatchesPinnedFixture::test_shipped_value_matches_pinned_fixture`,
`::test_hard_routing_cap_ratios_retired` (all in `proxy/tests/`).

> Row 10 stays `derived-auto` because the *derivation* rescales with the
> geometry: the token cap is 0 at both geometries only because the ratio is
> retired to 0, and the non-zero case is pinned at both geometries so a future
> re-enable cannot land on an unpinned value.

### 3.2 Admission, leasing and the contention queue

| # | Mechanism / metric | Value @ 2 slots | Value @ 1 slot | Classification | Evidence |
|---|---|---|---|---|---|
| 11 | `local_session_tenants` — large-prompt tenancy bound (explicit value overrides the slot-count default) | 2 | 2 (stays 2 — must be edited) | `static-needs-manual-change` | `proxy/config.yaml:1386`; default path `proxy/proxy/router_helpers.py:3112`-`3134` |
| 12 | Dispatch lease pool `max_local` — concurrent local dispatch leases | 2 | 1 | `derived-auto` | `proxy/proxy/provider.py:6542` (`_get_local_concurrency_info`) |
| 13 | Local concurrency limit read by the router | 2 | 1 | `derived-auto` | `proxy/proxy/router.py:215`-`233` (`_get_local_max_concurrent_queries`) |
| 14 | Capacity-admission slot-pool sizing helper | 2 | 1 | `derived-auto` | `proxy/proxy/capacity_admission.py:124`-`130` (`_slots`) |
| 15 | `local_max_concurrent_prefills` — concurrent local prefills | 1 | 1 | `not-applicable` | `proxy/config.yaml:1725`; `proxy/proxy/router_helpers.py:2710` |
| 16 | `contention_queue_max_wait_seconds` — queue wait budget | 60 s | 60 s | `not-applicable` | `proxy/config.yaml:1776`; clamped at `proxy/proxy/router.py:267`-`291` |
| 17 | `contention_queue_max_depth` — queue depth cap | 8 | 8 | `not-applicable` | `proxy/config.yaml:1777`; clamped at `proxy/proxy/router.py:293`-`306` |
| 18 | `contention_queue_lease_grace_seconds` — lease-owner retention grace | 300 s | 300 s | `not-applicable` | `proxy/config.yaml:1807`; clamped at `proxy/proxy/router.py:311`-`330` |
| 19 | `contention_queue_eta_ordering` (+ `..._eta_promotion_seconds`) — ETA admission ordering | on (30 s) | on (30 s) | `not-applicable` | `proxy/config.yaml:1786`, `:1790` |
| 20 | `local_slot_exhaustion_retry_attempts` / `_retry_delay_seconds` — slot-exhaustion retries | 2 / 0.1 s | 2 / 0.1 s | `not-applicable` | `proxy/config.yaml:1261`-`1262`; `proxy/proxy/provider.py:5805`-`5833` |

Rows 15-20 are `not-applicable` on purpose: they bound *how a request waits*,
not *how many requests run*, so a one-slot pool changes what they observe but
not what they are configured to do. Row 15 is the one that matters for the
prediction: with `local_max_concurrent_prefills: 1`, prefills are already
serialised at two slots, so a second slot only ever overlapped **decode**
(`proxy/config.yaml:1706`-`1725`).

### 3.3 Slot affinity and KV-cache persistence

| # | Mechanism / metric | Value @ 2 slots | Value @ 1 slot | Classification | Evidence |
|---|---|---|---|---|---|
| 21 | Slot affinity — addressable slots for session assignment | 2 (sessions spread over slots 0-1) | 1 (every session maps to slot 0) | `derived-auto` | `proxy/proxy/session.py:1043`-`1073` (`_slot_id_for_session`) |
| 22 | Persistence gate `session_slot_max_prompt_tokens` — derived from the per-slot clamp when unset | 126,976 | 258,048 | `derived-auto` | `proxy/proxy/session.py:1252`-`1285` |
| 23 | Context-pressure ratio denominator (`estimated / per-slot clamp`); advisory threshold currently 0 (disabled) | /126,976 | /258,048 | `derived-auto` | `proxy/proxy/provider.py:1399`-`1436`; threshold `proxy/config.yaml:1445` |

Row 21 is the mechanic behind the counter-evidence this report must not hide:
with one slot every session shares slot 0, so each turn transition evicts the
previous session's KV prefix — measured at a 95 % steal rate for two
concurrent sessions in the slot-thrashing investigation
(`docs/dev/slot-management-design.md:38`-`48`; "cache preservation: poor").
Row 23 is derived but currently inert because the advisory threshold is 0.

### 3.4 Observability endpoints and per-slot detail

| # | Mechanism / metric | Value @ 2 slots | Value @ 1 slot | Classification | Evidence |
|---|---|---|---|---|---|
| 24 | `/llama/local/status` `available_slots` / `total_slots` (live `/slots` query) | 2 / 0-2 | 1 / 0-1 | `derived-auto` | endpoint `proxy/proxy/handlers.py:374`; query `proxy/proxy/handlers.py:658`; `proxy/proxy/observability.py:519`-`573` |
| 25 | `/llama/local/status` `slots[]` per-slot detail (`slot_id`, `is_processing`, `n_decoded`, `context_tokens`, `owner_session_id`, `lease_state`, `phase`, `prefill_eta_seconds`) | 2 entries | 1 entry | `derived-auto` | `proxy/proxy/observability.py:658`-`700`; annotated at `proxy/proxy/handlers.py:459`-`470` |
| 26 | Fail-open pool literal in the status/health slot paths (`server_cfg.get("session_slot_pool_size", 3) or 3`) | 2 from config (literal `3` only when the key is absent) | 1 from config (literal `3` only when the key is absent) | `static-needs-manual-change` | `proxy/proxy/handlers.py:683`, `:729`, `:859` |
| 27 | `/health` slot counts | none — the payload carries `status` / `ready` / `llama_server_running` / `backend_reachable`, not slot counts | none | `not-applicable` | `proxy/proxy/handlers.py:1192`-`1264` |
| 28 | Last-known slot-counts cache (`SLOT_COUNTS_STALE_AFTER_SECONDS`) and `slots_stale` / `slots_detail_stale` flags | follows the live pool (2) | follows the live pool (1) | `derived-auto` | `proxy/proxy/observability.py:164`-`210`; flags at `proxy/proxy/handlers.py:912`-`953` |
| 29 | Abandoned-slot counter `local_abandoned_slots` in the status poll | derived from live slot state | derived from live slot state | `derived-auto` | `proxy/proxy/handlers.py:948`-`953` |

Row 27 is a factual correction to the parent's assumed surface: **`/health`
does not report slot counts**; the counts and the per-slot detail live on
`/llama/local/status` only.

### 3.5 Metrics and the analysis report

| # | Mechanism / metric | Value @ 2 slots | Value @ 1 slot | Classification | Evidence |
|---|---|---|---|---|---|
| 30 | Prometheus `llama_slots_query_failures_total{reason}` | counter, no slot dimension | counter, no slot dimension | `not-applicable` | `proxy/proxy/metrics.py:158`-`162` |
| 31 | Prometheus `llama_slots_child_port_fallbacks_total{reason}` | counter, no slot dimension | counter, no slot dimension | `not-applicable` | `proxy/proxy/metrics.py:163`-`168` |
| 32 | herdr's consumption of `/llama/local/status` (same free slots tracked across polls; cross-repo WL-0MUFP30T2003OX1F) | 2 slots tracked | 1 slot tracked | `derived-auto` | `proxy/proxy/handlers.py:459`-`470`; `proxy/proxy/observability.py:166`-`177` |
| 33 | Report metric: local busy % (dispatch → finish wall-clock union) | geometry-independent | geometry-independent | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:575`-`631` |
| 34 | Report metric: turns served locally (local-turn coverage) | geometry-independent | geometry-independent | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:359`-`376` |
| 35 | Report metric: input/output tokens processed locally (`Stream finished` `tokens=<p>/<c>/<t>`) | geometry-independent | geometry-independent | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:817`-`846` |
| 36 | Report metric: diverts by reason (`routing_skip_local` reason distribution, `Fallback triggered`) | geometry-independent | geometry-independent | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:376`-`412`, `:528` |
| 37 | Report metric: queue waits (`queue_wait_ms` in the prefill breakdown) | geometry-independent | geometry-independent | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:544`-`548` |
| 38 | Report metric: dispatch-to-first-byte (`dispatch_to_first_byte_ms`) | geometry-independent | geometry-independent | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:544`-`548` |
| 39 | Report metric: prefill slot-seconds | slot-seconds, summed per stream | slot-seconds, summed per stream | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:641`-`664` |
| 40 | Report metric: decode slot-seconds | slot-seconds, summed per stream | slot-seconds, summed per stream | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:664`-`680` |
| 41 | Report metric: decode tok/s (Decode speed, from `llama-server.log` eval timing) | geometry-independent | geometry-independent | `not-applicable` | `.pi/skills/proxy-usage-analysis/REFERENCE.md:788`-`801` |

Rows 33-41 are `not-applicable` because none of them divides by the slot count:
the analysis computes wall-clock unions, per-stream slot-second sums and
per-stream rates, all of which stay comparable across geometries. That is also
why they can evidence the A/B — see §6 for the ones that *cannot* see
single-slot starvation.

---

## 4. Method and provenance

1. **Derived numbers are taken from the sibling test item, not re-derived by
   hand.** LP-0MVTV6F65005HKQZ pinned each derivation at `(ctx=262144, slots=1)`
   and `(ctx=262144, slots=2)` against a fixture mirroring the shipped
   `server` block (`proxy/tests/config_test_utils.py`, `geometry_config` /
   `SHIPPED_ROUTING_GEOMETRY`):

   | Derivation | Pinned by (test) | 2 slots | 1 slot |
   |---|---|---|---|
   | Physical per-slot clamp | `test_four_derivations_pinned_per_geometry` | 126,976 | 258,048 |
   | Effective warm ceiling | `test_four_derivations_pinned_per_geometry`, `test_absolute_warm_cap_dominates_physical_clamp_at_1_slot` | 120,000 | 120,000 |
   | Cold threshold | `test_four_derivations_pinned_per_geometry`, `test_cold_warm_band_non_empty_and_cold_slot_invariant` | 38,000 | 38,000 |
   | Hard-routing cap (shipped, ratio 0) | `test_four_derivations_pinned_per_geometry`, `test_shipped_config_cap_disabled_per_geometry` | 0 | 0 |
   | Hard-routing cap (ratio 0.5) | `test_ratio_half_cap_pinned_per_geometry` | 60,000 | 60,000 |
   | Economic ceiling ratio term | `test_absolute_warm_cap_dominates_physical_clamp_at_1_slot` | 120,010 | 120,010 |
   | Shipped constants vs fixture | `TestShippedGeometryMatchesPinnedFixture::test_shipped_value_matches_pinned_fixture` | — | — |
   | Retired cap ratios | `TestShippedGeometryMatchesPinnedFixture::test_hard_routing_cap_ratios_retired` | — | — |

2. **Constants cite their defining line.** Every non-derived value in §3 cites
   the `proxy/config.yaml` or module line that defines it. The two Arm B target
   values (258,048 and 0.9844) are taken from the A/B arm table
   (`docs/dev/slot-geometry-ab.md:116`-`121`); the resulting warm ceiling
   (258,048) is the single-slot clamp itself, i.e. `min(round(0.9844 ×
   262144), 258048) = 258048` evaluated through the same pinned helper
   (`proxy/proxy/provider.py:1943`-`1985`, `:2043`-`2054`).

3. **Doc fidelity is executable.**
   `proxy/tests/test_single_slot_inventory_doc.py` parses the §3 tables and
   asserts (a) every AC1-required mechanism is present, (b) every row carries
   all four fields with a canonical classification, (c) the four derived
   threshold rows equal the production helpers' output at 1 and 2 slots, and
   (d) each `key: old` in §1.3 still matches the shipped `proxy/config.yaml`.
   A future config or code drift therefore fails the suite instead of leaving a
   silently stale report.

4. **No measured baselines are asserted in this slice.** All measured figures
   (busy %, prefill percentiles, re-prefill volume, steal rate) belong to the
   throughput prediction (§5) and are cited there with their window and source;
   the only measured figure quoted here is the 95 % steal rate, with its origin
   (`docs/dev/slot-management-design.md:38`-`48`).

---

## 5. Throughput prediction (reserved for LP-0MVTV6G0N0053V0G)

*This section is filled by the throughput-prediction slice.* It reuses the
classifications from §3 and must cover, at request level and including tokens
processed locally: turns served locally, input/output tokens processed locally,
per-session first-byte and decode latency, head-of-line blocking and queue
waits, slot reuse/thrash and re-prefill volume, and concurrency headroom — each
with the metric and the A/B arm comparison that would confirm or refute it.

Inputs it inherits from this slice:

- `derived-auto` rows 2, 3, 5, 7, 8 and 10 fix the routing envelope: at one
  slot, **only the per-slot clamp doubles**; every routing threshold keeps its
  two-slot value until the §1.3 co-change is applied.
- `derived-auto` row 12 fixes the concurrency envelope: the dispatch lease pool
  becomes 1, so `max_local` no longer admits a second concurrent session.
- `not-applicable` row 15 says prefills were already serialised at two slots,
  so the single-slot cost lands on **decode** overlap, not on prefill overlap.
- `derived-auto` row 21 names the thrash mechanism (all sessions → slot 0) and
  points at the measured steal rate it must reason with.

## 6. Metrics gap list (reserved for LP-0MVTV6GA8002P7QU)

*This section is filled by the metrics-gap slice.* Every conclusion of §5 must
be mapped to an existing metric that can detect it, and every blind spot filed
as a follow-up work item.

Inputs it inherits from this slice:

- Rows 33-41 are the existing measurement surface, and all of them are
  geometry-independent — none of them is divided by, or scoped to, a specific
  slot.
- Rows 24-29 publish **pool-level** counts and **per-slot** detail, but the
  proxy publishes **no per-slot occupancy metric** (the only Prometheus
  `llama_slots_*` series are the two failure counters, rows 30 and 31). Per-slot
  occupancy metrics are the known gap and are already tracked as the blocked
  item **LP-0MVT1O0VG00905EL** ("Publish slot occupancy metrics from the
  proxy") — they are not implemented here.
- Row 32 is the other consumer whose blind spots matter: herdr reads pool counts
  and per-slot detail, so a starved single slot is visible to herdr only through
  `slots[]` poll data, not through a metric.

---

## 7. Citations appendix

Code and config (`proxy/` relative to the repository root):

| Citation | What it establishes |
|---|---|
| `proxy/config.yaml:1297` | `local_large_context_cold_cache_threshold: 38000` |
| `proxy/config.yaml:1332` | `local_large_context_warm_cache_threshold: 120000` |
| `proxy/config.yaml:1386` | `local_session_tenants: 2` (explicit) |
| `proxy/config.yaml:1410` | `local_large_context_economic_ceiling_ratio: 0.4578` |
| `proxy/config.yaml:1412`-`1414` | hard-routing cap ratio keys retired |
| `proxy/config.yaml:1261`-`1262` | slot-exhaustion retries 2 / 0.1 s |
| `proxy/config.yaml:1431` | `local_model_ctx_size: 262144` |
| `proxy/config.yaml:1445` | `compaction_trigger_ratio: 0` (advisory threshold disabled) |
| `proxy/config.yaml:1704` | `session_slot_pool_size: 2` |
| `proxy/config.yaml:1725` | `local_max_concurrent_prefills: 1` |
| `proxy/config.yaml:1776`, `:1777`, `:1807` | queue wait 60 s, depth 8, lease grace 300 s |
| `proxy/config.yaml:1786`, `:1790` | ETA ordering on, 30 s promotion |
| `proxy/proxy/lifecycle.py:1658` | `session_slot_pool_size` exported as `LLAMA_PARALLEL` |
| `proxy/proxy/provider.py:1399`-`1436` | `context_pressure_ratio` denominator |
| `proxy/proxy/provider.py:1890` | `effective_per_slot_threshold` (physical clamp) |
| `proxy/proxy/provider.py:1943`-`1985` | `effective_large_context_economic_ceiling` (ratio term + absolute bound) |
| `proxy/proxy/provider.py:1972` | the absolute warm cap applied as an upper bound |
| `proxy/proxy/provider.py:2036` | cold read unclamped |
| `proxy/proxy/provider.py:2043`-`2068` | `_effective_large_context_thresholds` (min of terms, hard-cap clamp) |
| `proxy/proxy/provider.py:2166` | `compute_hard_routing_cap` |
| `proxy/proxy/provider.py:5805`-`5833` | slot-exhaustion retry knobs |
| `proxy/proxy/provider.py:6542` | `max_local` from `session_slot_pool_size` |
| `proxy/proxy/router.py:215`-`233` | `_get_local_max_concurrent_queries` |
| `proxy/proxy/router.py:267`-`330` | contention-queue clamp bounds |
| `proxy/proxy/router_helpers.py:2710` | `local_max_concurrent_prefills` read |
| `proxy/proxy/router_helpers.py:3112`-`3134` | `resolve_session_tenants` default = slot count |
| `proxy/proxy/capacity_admission.py:124`-`130` | capacity-admission slot pool |
| `proxy/proxy/session.py:1043`-`1073` | `_slot_id_for_session` slot affinity |
| `proxy/proxy/session.py:1252`-`1285` | `session_slot_max_prompt_tokens` derivation |
| `proxy/proxy/observability.py:164`-`210` | last-known slot counts + staleness window |
| `proxy/proxy/observability.py:519`-`573` | `_query_slots` (available/total counts) |
| `proxy/proxy/observability.py:658`-`700` | `_query_slots_detail` (per-slot detail) |
| `proxy/proxy/handlers.py:374` | `/llama/local/status` handler |
| `proxy/proxy/handlers.py:459`-`470` | per-slot detail consumed by herdr's downtime worker |
| `proxy/proxy/handlers.py:683`, `:729`, `:859` | fail-open pool literal `3` |
| `proxy/proxy/handlers.py:912`-`953` | `slots_stale` / `slots_detail_stale` / abandoned slots |
| `proxy/proxy/handlers.py:1192`-`1264` | `/health` (no slot counts) |
| `proxy/proxy/metrics.py:158`-`168` | the only two `llama_slots_*` series |

Documents and analysis catalogue:

| Citation | What it establishes |
|---|---|
| `docs/dev/slot-geometry-ab.md:9`-`20` | Arm A / Arm B geometry definitions |
| `docs/dev/slot-geometry-ab.md:116`-`121` | the arm patch table (keys and values) |
| `docs/dev/slot-management-design.md:38`-`48` | 95-98 % steal rate with `pool_size=1` |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:359`-`376` | local-turn coverage (turns served locally) |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:376`-`412`, `:528` | divert / skip reason distribution |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:544`-`548` | prefill breakdown (`queue_wait_ms`, `dispatch_to_first_byte_ms`) |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:575`-`631` | local busy % definition |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:641`-`664` | prefill slot-seconds |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:664`-`680` | decode slot-seconds |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:788`-`801` | decode tok/s |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:817`-`846` | local token reporting and the cold-cache cap |

Related work items: LP-0MVMHNUXW0061FGN (measured A/B, owns the geometry
verdict), LP-0MVT1O0VG00905EL (slot occupancy metrics, blocked),
LP-0MVTZUZR7K009Q98J (drop `slot_schedule` dependency), LP-0MVHT41X9000AXFB
(stale warm-threshold constants in analysis scripts).
