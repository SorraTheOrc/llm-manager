# Single-slot impact: derived vs static inventory

**Work item:** LP-0MVTV6FGT000A19P (Inventory derived vs static slot mechanisms)
**Parent:** LP-0MVTF26NP003KXKJ (Investigate what happens if I switch to a single
slot)
**Sibling slices of the same parent**

| Slice | Work item | What it fills in this report |
|---|---|---|
| Pinned derivations | LP-0MVTV6F65005HKQZ (completed) | The executable assertions every derived number below is taken from |
| Throughput prediction | LP-0MVTV6G0N0053V0G | [§5](#5-throughput-prediction-lp-0mvtv6g0n0053v0g) |
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

## 5. Throughput prediction (LP-0MVTV6G0N0053V0G)

*A desk prediction, not a measurement.* This section reasons from §3's
classifications and from a set of recorded baselines; it does **not** flip the
geometry, restart the proxy or measure an arm. Every figure is registered in
`docs/dev/single-slot-throughput-baseline-2026-10-31.json` with its window, its
source document and its provenance class (`measured`, `derived-estimate`,
`config-derived`), and `proxy/tests/test_single_slot_throughput_doc.py` fails if
this text and that register disagree or if an estimate is not labelled one.

### 5.1 The short answer

At today's load, one slot costs **latency and prefix reuse, not throughput**:

- **Turns and completion tokens barely move.** Local work is nearly serial
  already: W1 was busy `10.2` % of the day with average concurrency `1.04` and
  peak `2`. The only work a second slot actually performs is the `300`
  slot-seconds/day of overlap (`3.4` % of busy time, D1/D2), i.e. roughly `15`
  local turns/day (`6.6` %, D3/D4) can be delayed. Predicted turn loss: 0 to
  `2` % of the `225` local turns/day, because most of those delays fit inside
  the existing `60` s queue budget.
- **Prompt tokens processed locally are set by routing, not by slot count.**
  Without the §1.3 co-change the effective warm ceiling stays `120,000`
  (D17), so the measured local prompt volume — `5,242,181` tokens/day in W1 and
  `12,123,352` in W2 — is unchanged: the extra per-slot capacity is
  unreachable for routing. With the co-change the ceiling becomes `258,048`
  (D16) and the ceiling-divert pool (`4,443`/day in W1, `4,667`/day in W2)
  becomes admissible, but capacity limits conversion to about `5.6` % of that
  pool's cold volume (D22) — a tail effect on diverts, not a volume effect on
  local tokens.
- **The costs land on the first-byte tail and on prefix reuse.** A converted
  258K-class prompt pays a cold prefill of `1,017`–`1,317` s (D11/D12), and
  minimum-size converted diverts pay about `473` s each (D24). One slot also
  means concurrent sessions share slot 0, so the recorded `95` % steal rate
  (W7) replaces today's "each session owns a slot", inflating re-prefill by up
  to `11.5M` tokens/day (`26.3` % on top of the `43.8M`/day pool, D9/D10).
- **Headroom is not the constraint.** One slot would be busy `10.5` % of the day
  (D5), leaving `89.5` % headroom (D6). Fewer slots therefore do not shrink the
  *tokens the machine can process*; they lengthen the *waits requests pay*.

### 5.2 Baselines of record

| Window | Span | Source | Retention | Used for |
|---|---|---|---|---|
| W1 | 2026-10-23 10:52:37 → 2026-10-24 10:52:37 (24.0 h) | `~/proxy-usage-reports/2026-10-25/report.md` | partial | demand, latency, diverts, tokens |
| W2 | 2026-10-30 05:00:01 → 2026-10-31 05:00:01 (24.0 h) | `~/proxy-usage-reports/report.md` | full | tokens, provenance, denials |
| W3 | 2026-10-19 → 2026-10-20, 460 paired dispatches | `proxy/docs/routing.md:826`-`847` | recorded-only | prefill overlap rate |
| W4 | 2026-10-24, 5.4 h | `proxy/config.yaml:1366`-`1380` | recorded-only | prefix reuse |
| W5 | 2026-08-26 plus companion data to 2026-08-28 | `docs/dev/restore-rate-metric-fix-ranking.md:1`-`20` | recorded-only | re-prefill pool, restore rates |
| W6 | 2026-10-30 05:00 → 2026-10-31 05:00 (24.0 h) | `docs/dev/contention-queue-sweep-baseline-2026-10-31.json` | committed | queue waits |
| W7 | not recorded in the source | `docs/dev/slot-management-design.md:28`-`48` | recorded-only | steal rate by pool size |

Plus the engine-side large-prompt measurement — prompt-eval p90 `189` s, max
`591` s (`docs/dev/ubatch-ab-2026-10-22.md:8`-`11`).

W1's report predates the `tokens=missing` accounting and its pre-2026-10-24 log
segments have rotated out of retention, so it is the window of record for
demand, latency and divert shape only; token provenance is measured on W2.
W3, W4 and W7 are not re-runnable from retained logs; they are quoted as
recorded measurements and §5.8 names the measurement that refreshes each.

### 5.3 Prediction register

Every row names its mechanism with a `file:line` citation, its baseline with the
window and the registered figures, the predicted direction and magnitude at one
slot, the metric that would confirm or refute it, and the Arm A / Arm B
comparison.

| # | Effect | Mechanism (`file:line`) | Baseline (window · figures) | Prediction at 1 slot | Confirming metric | Arm A vs Arm B |
|---|---|---|---|---|---|---|
| E1 | Turns served locally | `max_local` derives from `session_slot_pool_size` (`proxy/proxy/provider.py:6542`); router limit `proxy/proxy/router.py:215`-`233` | W1 · `W1.local_turns`, `W1.local_sessions`, `W1.avg_concurrency_while_busy`; W2 · `W2.dispatch_denied`, `W2.denials_saturated`, `W2.denials_prefill_bound` | direction: flat to slightly down; magnitude: 0 to `2` % of `225` turns/day (at most `15` turns delayed, D3/D4), and no loss while the wait stays inside the `60` s queue budget | `turns_served_locally`, `diverts_by_reason`, `queue_waits`, `fallback_after_queue` | Arm A (2 x 131,072) vs Arm B (1 x 262,144): criterion (a) requires Arm B `turns_served_locally` at least equal to Arm A |
| E2 | Prompt tokens processed locally | 3-term warm ceiling `proxy/proxy/provider.py:2002`-`2040`; physical clamp `proxy/proxy/provider.py:1890`; dispatch lease pool `proxy/proxy/provider.py:6542` | W1 · `W1.local_prompt_tokens`, `W1.ceiling_diverts`, `W1.divert_context_too_large`; W2 · `W2.local_prompt_tokens`, `W2.ceiling_diverts`; W4 · `W4.aggregate_prefix_reuse_pct` | no co-change: unchanged (`5,242,181` and `12,123,352` per day, D17); with co-change: at most `5.6` % of the `394M` cold ceiling-divert volume convert, i.e. about `21.9M` cold prompt tokens/day (D19-D22) | `diverts_by_reason`, `context_ceiling_diverts`, `prefill_slot_seconds` | Arm A vs Arm B: criterion (b) requires Arm B `context_ceiling_diverts` to fall by at least 50 %; `prefill_slot_seconds` shows the prefill cost of that fall |
| E3 | Completion tokens processed locally | decode shares one llama.cpp batch across the pool; one slot per session (`proxy/proxy/session.py:1062`-`1095`) | W1 · `W1.local_completion_tokens`, `W1.decode_tok_s_median`; W2 · `W2.local_completion_tokens`, `W2.decode_tok_s_median` | direction: unchanged in volume — `48,700` and `244,657` per day, `216` completion tokens per local turn (D8); only the *timing* of those tokens changes | `decode_slot_seconds` per completed turn, plus the capture's archived `Decode speed` table | Arm A vs Arm B: compare `decode_slot_seconds` divided by `turns_served_locally`; the archived `Decode speed` medians (`30.7` / `37.7` tok/s) normalise the comparison |
| E4 | Per-session first-byte latency | prefills already serialised (`proxy/config.yaml:1759`, `proxy/proxy/router_helpers.py:2709`); one slot cannot overlap a prefill with another slot's decode | W1 · `W1.prefill_p50_seconds`, `W1.prefill_p90_seconds`; W3 · `W3.solo_first_byte_p50_seconds`, `W3.one_overlap_first_byte_p50_seconds`; engine · D13, D14 | same-class prompts: `4.4` s p50 and `35.3` s p90, flat to slightly worse; co-change admissions: `473` s per minimum-size converted prompt and `1,017`-`1,317` s at the `258,048` ceiling (D11/D12/D24), so the p90 rises with the converted share | `local_dispatch_first_byte` (p50/p90), `ttft_all_streams` | Arm A vs Arm B: criterion (c) allows Arm B `first_byte_p90` to worsen by at most 30 % |
| E5 | Per-session decode latency | two streams share one decode batch at 2 slots; at 1 slot a stream decodes alone (`proxy/proxy/provider.py:6542`) | W1 · `W1.decode_tok_s_p10`, `W1.decode_tok_s_p90`; W2 · `W2.decode_tok_s_p10`, `W2.decode_tok_s_p90` | direction: per-stream decode flat to slightly better, the low tail (`8.4` / `12.8` tok/s p10) improving once nothing shares the batch; the *second* stream's end-to-end turn worsens by its queue wait | `decode_slot_seconds`, and the capture's archived `Decode speed` p10/p50/p90 (no `decode_tok_s` path exists in `arm.json`) | Arm A vs Arm B: compare the two captures' archived `Decode speed` tables against their `decode_slot_seconds` |
| E6 | Head-of-line blocking and queue waits | one dispatch lease (`proxy/proxy/provider.py:6542`); queue caps `proxy/config.yaml:1810`-`1811`, `:1820`, `:1841`; router clamps `proxy/proxy/router.py:267`-`330` | W6 · `W6.queue_events`, `W6.queue_wins`, `W6.queue_losses`, `W6.queue_wait_p50_seconds`, `W6.queue_wait_p90_seconds`, `W6.queue_wait_max_seconds` | queue entries rise by the overlap turns only: `351` to about 366/day; losses `137` to about 150/day, concentrated where a prefill-bound wait exceeds the `60` s budget (mean loss wait `60.02` s today) | `queue_waits`, `fallback_after_queue`, `queue_wait_seconds` | Arm A vs Arm B: `queue_waits` and `fallback_after_queue` compared side by side in the two arm captures |
| E7 | Slot reuse, thrash and re-prefill volume | one slot per session, lowest free slot (`proxy/proxy/session.py:1062`-`1095`); persistence prompt cap `proxy/proxy/session.py:1271`-`1285` | W7 · `W7.steal_pct_pool1_two_sessions`, `W7.steal_pct_pool2_four_sessions`; W4 · `W4.aggregate_prefix_reuse_pct`, `W4.median_reuse_large_prompts_pct`; W5 · `W5.reprefill_pool_tokens_per_day`, `W5.native_checkpoint_restore_pct` | steal rate goes from 0 % between two sessions with a slot each to `95` % on one slot; re-prefill rises by up to `11.5M` tokens/day (D9/D10) when sessions interleave, and by nothing while only one session is active — which was `89.8` % of the W1 day | `prefill_slot_seconds` per served turn, `errors.slot_save_failed`, the engine cross-check | Arm A vs Arm B: a rise in Arm B `prefill_slot_seconds` divided by `turns_served_locally` is thrash cost; a flat ratio is no penalty |
| E8 | Concurrency headroom | pool size sets concurrent local streams (`proxy/config.yaml:1738`); occupancy measured by the analyzer | W1 · `W1.compute_slot_seconds`, `W1.peak_concurrency`, `W1.busy_pct`; W2 · `W2.compute_slot_seconds`, `W2.peak_concurrency` | one slot's daily capacity is `10.5` % used (D5), leaving `89.5` % headroom (D6); the two-stream peak occupied only `300` s (D1), so neither geometry approaches the throughput ceiling | `local_busy_pct`, `avg_concurrency`, `peak_concurrency`, `local_concurrency_limit` | Arm A vs Arm B: `local_busy_pct` (criterion (d): Arm B must not regress at all) with the `avg_concurrency` / `peak_concurrency` pair |

### 5.4 Effect detail and arithmetic

**E1 — turns served locally.** Two mechanisms bound local turns: the dispatch
lease pool, whose size *is* `session_slot_pool_size`
(`proxy/proxy/provider.py:6542`), and the bounded contention queue that decides
what a refused request does next. Halving the pool halves the number of
sessions that can hold a lease at once, but the measured overlap says how much
that matters: W1's total compute was `2h 31m` of slot-seconds inside `2h 26m` of
busy wall-clock, so a second stream was resident for `300` s of the day (D1),
`3.4` % of busy time (D2). At W1's mean stream duration `40.5` s that is about
`15` turns (D3), `6.6` % of `225` (D4). W2's denial attribution sharpens this:
of `332` denials only `8` were pool-*saturation* denials, against `229`
prefill-bound ones — the denial mix is already dominated by prefill
serialisation, not by slot count.

**E2 — prompt tokens processed locally.** Without the §1.3 co-change the
routing envelope is untouched (physical clamp doubles but is not the binding
term), so local prompt tokens stay at the measured `5,242,181` (W1) and
`12,123,352` (W2). With the co-change the ceiling is `258,048`, and the question
becomes capacity, not routing:

- every ceiling divert carries at least the `120,000` warm ceiling in prompt
  tokens — W1 recorded `2,064` `context_too_large` diverts plus `2,379`
  `large_context_bypass` diverts — so its `4,443` ceiling diverts represent at
  least `533M` prompt tokens/day (D19);
- at W4's measured aggregate prefix reuse of `26.1` %, at least `73.9` % of that
  volume is cold (D18), i.e. `394M` cold tokens/day (D20);
- one dedicated slot at W3's measured solo prefill rate of `253.8` tok/s can
  prefill `21.9M` cold tokens/day (D21);
- so at most `5.6` % of the ceiling-divert cold volume fits (D22) — about `183`
  minimum-size prompts/day if a whole slot-day went to them (D23).

The consequence is that Arm B's divert reduction is a *tail* effect: the
prompts it wins are the ones whose cold remainder is small, and each one that is
not small costs `473` s of first byte or more (D24).

**E3 — completion tokens processed locally.** Completion volume is demand, not
geometry: `48,700` (W1) and `244,657` (W2) tokens/day, `216` per local turn
(D8). What one slot changes is batching. Decode is where concurrent streams
share work at two slots — the prefill analogue was measured in W3, where
overlapping prefills halved per-request throughput while aggregate throughput
stayed flat at `254` tok/s. The same shape is expected for decode: aggregate
decode throughput conserved, per-stream latency improved at one slot, and the
aggregate throughput *ceiling* untouched because it is the GPU, not the slot
count, that binds.

**E4 — first-byte latency.** Prefills are already serialised by
`local_max_concurrent_prefills: 1`, so the prefill-vs-prefill queueing term does
not change: same-class prompts keep `4.4` s p50 / `35.3` s p90 (W1), and W3
measured `23.4` s of first byte with no overlapping prefill against `85.0` s
with one. The change
is that at two slots a second slot can *decode* while slot 0 prefills, whereas
at one slot a prefill occupies the only slot. For prompts that stay in the same
routing class the added exposure is bounded by the overlap term (E1); for
prompts admitted only because of the co-change it is the prefill itself —
`1,017` s at W3's solo rate and `1,317` s at W2's median prompt-eval rate for a
`258,048`-token prompt (D11/D12), against an engine-side p90 of `189` s and max
of `591` s already observed on 20k+ prompts (`docs/dev/ubatch-ab-2026-10-22.md:8`-`11`).

**E5 — decode latency.** Measured per-stream decode is `30.7` tok/s median in
W1 and `37.7` in W2 (p90 `39.4` and `43.7`), with a long low tail (`8.4` and
`12.8` tok/s p10). The low
tail is where two streams sharing one decode batch shows up. At one slot that
tail should narrow, and the cost moves to end-to-end time for the waiting
stream. This is the one effect the baselines cannot size: nothing attributes
decode rate to concurrency, so it is stated as a direction with the confirming
metric named, not as a number.

**E6 — head-of-line blocking and queue waits.** The queue is bounded
(`60` s wait, depth 8, lease grace 300 s, ETA ordering on) and already in use:
`351` entries/day in W6, `214` of which won their slot (p50 wait `22.19` s, p90
`48.45` s, max `59.49` s) and `137` of which lost at a mean `60.02` s. Adding
the overlap turns is the increment: about `15` extra entries/day (E1), taking
the entry count to roughly 366/day and the loss count to roughly 150/day, with
the risk concentrated on prefill-bound collisions whose remaining duration
exceeds the budget — decode remaining durations (p90 `22.2` s in W2) fit, large
prefills (p90 `189` s engine-side, D13) do not.

**E7 — slot reuse, thrash and re-prefill.** Slot assignment takes the lowest
free slot for the session (`proxy/proxy/session.py:1062`-`1095`), so with a
pool of one every session maps to slot 0. At two slots two concurrent sessions
each own a slot and steal nothing from each other; the recorded `95` % steal
rate at pool size 1 for two sessions (W7) is what one slot substitutes for it.
W4 shows what thrash already costs at two slots — `27` sessions cycling through
`2` slots in `5.4` h with only `26.1` % aggregate prefix reuse — and W5 prices
the pool: `43.8M` re-prefill tokens/day against a `4.88` % native checkpoint
restore rate. The worst-case increment from one slot is the local prompt volume
times the steal share, `11.5M` tokens/day (D9), i.e. `26.3` % on top of the
existing pool (D10). Mitigating term: steals need *interleaving*. While a single
session is active it holds slot 0 and nothing is stolen, and in W1 local was
idle `89.8` % of the day.

**E8 — concurrency headroom.** W1's measured compute is `2h 31m` of
slot-seconds. At two slots that is `5.2` % of daily capacity; at one slot it is
`10.5` % (D5), i.e. `89.5` % of a single slot would still be idle (D6). W2 tells
the same story at higher load: `4h 37m` of compute, `18.5` % busy, peak
concurrency `2`. Neither window is throughput-bound, so reducing the pool cannot
reduce the tokens the machine is *able* to process; it can only change the order
in which requests get them.

### 5.5 Token provenance (AC3)

**Authoritative source.** Every token figure here comes from the
`Stream finished` line's `tokens=<prompt>/<completion>/<total>` fields, which
carry llama-server's measured usage for local streams (the proxy requests
`stream_options.include_usage`,
`.pi/skills/proxy-usage-analysis/REFERENCE.md:865`-`891`), and the attribution
rule is REFERENCE §3 (`:591`-`598`). Log payload previews are truncated and are
never used for sizes.

**Attribution rule.** A stream's tokens are attributed to the provider and model
stamped on its own `Stream finished` line — the one that actually served the
stream — not to the session's initial provider. A session that starts local and
falls back therefore splits its tokens across providers, and the `local` row of
the report's token table *is* the local share. That is the figure quoted as
"tokens processed locally" above (`5,242,181` prompt / `48,700` completion in
W1; `12,123,352` / `244,657` in W2).

**Local-versus-remote turn split.** A *covered session* is one with at least one
local `Stream started`; a local turn is *answered* when a local, non-error
`Stream finished` pairs with it — exactly by `turn=` where the id is present,
per-session FIFO otherwise. `% local turns` is `local / (local + remote) × 100`,
so a session that never used remote reads 100 %. W1: `225` local turns over `26`
covered sessions out of `175` sessions; W2: `790` local turns over `175`
covered sessions out of `620`.

**Lines without token evidence.** In W2, `22,568` of `25,446` `Stream finished`
lines carry `tokens=` (`88.7` % coverage, `W2.usage_coverage_pct`); the
remaining `2,878` carry `tokens=missing` (MunitoAI 1,230, OpenCode 1,643,
local `5`) and are excluded from the token totals rather than counted as zero.
The local exception rate is therefore `5` in `790` finished local streams
(`99.4` % coverage, `W2.local_usage_coverage_pct`), so the local token figures
used above are measured for all but `5` turns. **W1 records no such count** —
the report version predates the `tokens=missing` accounting and the first 13 h of
its logs have rotated away — so the W1 token split is quoted as that report
produced it, with the missing-usage count for W1 explicitly *not* recovered
(§5.8 keeps it as an open re-measurement).

**Derived figures are labelled.** The per-turn means `23,300` prompt tokens and
`216` completion tokens (D7/D8) are arithmetic over measured totals, not
measurements, and are labelled estimates. The capacity bounds (D18-D22) are
estimates with their formulas and assumptions recorded in the register; the
warm ceilings `120,000` / `258,048` are `config-derived` through the production
helpers, not hand arithmetic.

### 5.6 The uncontended-evidence caveat (AC4)

Both baseline windows are essentially **uncontended**, and the prediction must
be read with that in mind:

- W1: local idle `89.8` % of the window, busy `10.2` %, average concurrency
  while busy `1.04`, peak `2`.
- W2: local idle `81.5` %, busy `18.5` %, average concurrency `1.04`, peak `2`.

The serialisation cost of one slot is therefore **under-observed today**: a
second slot is in use for `300` slot-seconds per day (D1), so the data cannot
show what happens when two sessions genuinely contend for long stretches, and
every magnitude above is a prediction bounded by today's demand rather than a
measured result. The measured A/B exists for exactly this reason — the reporter
returns `inconclusive` unless both arms sustain at least 60 % busy
(`docs/dev/slot-geometry-ab.md:41`-`48`) — and **no conclusion in this section
may be presented as a measured single-slot result.** Nothing here was measured
by flipping the geometry; the geometry was not flipped in this work.

### 5.7 Conditional on the static knobs (AC5)

Any effect stated above as depending on a static knob is conditional on the
manual change in §1.3:

| Prediction | Conditional on |
|---|---|
| the effective warm ceiling rises to `258,048`, so ceiling diverts become admissible (E2, E4) | `local_large_context_economic_ceiling_ratio: 0.4578 -> 0.9844` **and** `local_large_context_warm_cache_threshold: 120000 -> 258048`, applied together |
| the extra per-slot context routes nothing extra on its own (E2 no-co-change case) | `session_slot_pool_size: 2 -> 1` applied **alone**, leaving the ceiling at `120,000` |
| two sessions can hold large-prompt reservations in one pool (E6, E7 pressure) | `local_session_tenants: 2 -> 1` |
| the `(cold, warm]` cached-ratio band stays non-empty | `local_large_context_cold_cache_threshold` **stays** `38000` (unchanged) |
| the per-slot clamp doubles to `258,048` and the ceiling can reach it | `local_model_ctx_size` **stays** `262144` — the switch changes the split, not the total context |

The stale `3`-slot literal in `proxy/proxy/handlers.py` is not an input to any
prediction: it is a latent observability defect that would advertise three slots
at either geometry when the key is absent (§1.2 item 5, and the metrics-gap
slice). Everything else in the prediction rides on `derived-auto` or
`not-applicable` rows, which need no manual change (§3).

### 5.8 What must be re-measured (AC6)

| Claim that depends on it | Why the baseline must be refreshed | Confirming measurement |
|---|---|---|
| prefill rate `253.8` tok/s, hence `473`-`1,317` s of converted first byte (D11/D12/D24) | W3 (2026-10-19 → 10-20) cannot be re-run from retained logs, and batch/ubatch experiments since then moved prompt-eval rate | re-run the prefill-overlap sweep with the W3 method (`proxy/docs/routing.md:826`-`847`), or read Arm B's `prefill_slot_seconds` against the prompt tokens it served |
| steal rate `95` % at one slot (E7) | W7 records no measurement window, and its underlying findings document is not in this tree | Arm B `prefill_slot_seconds` per served turn against Arm A, plus the engine cross-check (`## Engine cross-check`, `local_engine_cross_check_threshold_pct: 25.0`) |
| cold share `73.9` % and the `394M` cold ceiling volume (D18-D20) | W4 measured 5.4 h at two slots; the diverts are larger prompts, whose own reuse was `15.6` %, so the cold share is optimistic | the arm captures' `diverts_by_reason` plus their `prefill_slot_seconds` |
| queue scaling `351` → about 366 entries/day (E6) | W6 counts queue behaviour at two slots; the increment assumes added entries equal the overlap turns | `queue_waits` and `fallback_after_queue` in both arm captures |
| W1's missing-usage count (AC3) | not emitted by that report version and the logs have rotated | re-run the analyzer on a retained 24 h window with the W2 pattern and record `streams_without_usage_by_provider` |
| every magnitude in E1 and E8 | both are functions of measured concurrency, which is `1.04` while busy — a contended window has a different shape | a contended (at least 60 % busy) window in which `avg_concurrency` exceeds 1.04, as §5.6 requires before any arm is measured |

### 5.9 What would refute this prediction

- Arm B `turns_served_locally` materially below Arm A (criterion (a)) — turns do
  not survive the single slot.
- Arm B queue entries far above the ~366/day bound, or losses far above
  ~150/day — head-of-line blocking is worse than the overlap term predicts.
- Arm B `prefill_slot_seconds` per served turn materially above Arm A — thrash
  costs more than the `11.5M` tokens/day bound.
- Arm B `first_byte_p90` worse than 30 % (criterion (c)) — the first-byte tail
  cost is real rather than tail-only.
- a contended window where `avg_concurrency` climbs well above 1.04 — the
  overlap term grows and E1/E6/E8 must be recomputed before the geometry is
  judged.

### 5.10 Inputs inherited from §3

- `derived-auto` rows 2, 3, 5, 7, 8 and 10 fix the routing envelope: at one
  slot, **only the per-slot clamp doubles**; every routing threshold keeps its
  two-slot value until the §1.3 co-change is applied.
- `derived-auto` row 12 fixes the concurrency envelope: the dispatch lease pool
  becomes 1, so `max_local` no longer admits a second concurrent session.
- `not-applicable` row 15 says prefills were already serialised at two slots, so
  the single-slot cost lands on **decode** overlap, not on prefill overlap.
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
| `docs/dev/single-slot-throughput-baseline-2026-10-31.json` | every figure quoted in §5, with its measurement window, source document and provenance class |
| `proxy/docs/routing.md:826`-`847` | prefill-overlap measurement (W3): first-byte p50 23.4 s solo vs 85.0 s with one overlap, aggregate throughput flat at ~254 tok/s |
| `docs/dev/ubatch-ab-2026-10-22.md:8`-`11` | engine-side large-prompt prompt-eval tail (p50 16 s, p90 189 s, max 591 s) |
| `docs/dev/restore-rate-metric-fix-ranking.md:16`-`18` | 43.8M re-prefill tokens/day and the 4.88 % native restore rate (W5) |
| `docs/dev/contention-queue-sweep-baseline-2026-10-31.json` | queue entries, waits and losses for the 24 h contention window (W6) |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:359`-`376` | local-turn coverage (turns served locally) |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:376`-`412`, `:528` | divert / skip reason distribution |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:544`-`548` | prefill breakdown (`queue_wait_ms`, `dispatch_to_first_byte_ms`) |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:575`-`631` | local busy % definition |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:641`-`664` | prefill slot-seconds |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:664`-`680` | decode slot-seconds |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:788`-`801` | decode tok/s |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:865`-`891` | local token reporting (`tokens=` / `tokens=missing`) and the cold-cache cap |
| `.pi/skills/proxy-usage-analysis/REFERENCE.md:591`-`598` | token attribution rule: authoritative `tokens=` on `Stream finished`, provider/model stamped on the line |

Related work items: LP-0MVMHNUXW0061FGN (measured A/B, owns the geometry
verdict), LP-0MVT1O0VG00905EL (slot occupancy metrics, blocked),
LP-0MVTZUZR7K009Q98J (drop `slot_schedule` dependency), LP-0MVHT41X9000AXFB
(stale warm-threshold constants in analysis scripts).
