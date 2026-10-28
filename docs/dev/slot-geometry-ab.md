# Slot-geometry A/B protocol: 1 x 262,144 vs 2 x 131,072 (LP-0MVMHNUXW0061FGN)

Repeatable measurement protocol for the parent experiment *"1 slot at 262K ctx
vs 2 slots at 131K ctx"*, delivered by child LP-0MVNCMDW50043RSP.

The question is a geometry trade-off, not a feature:

- **Arm A (current)** — `--parallel 2` with `--ctx-size 262144` ⇒ **2 slots x
  131,072 tokens**, warm routing ceiling ~120 K.
- **Arm B (candidate)** — `--parallel 1` with `--ctx-size 262144` ⇒ **1 slot x
  262,144 tokens**, warm routing ceiling ~258 K, at the cost of serialising
  concurrent sessions behind one slot.

Per-slot context is *derived*, never configured directly: llama-server divides
`--ctx-size` by `--parallel`, and the proxy lifecycle exports
`session_slot_pool_size` as `LLAMA_PARALLEL`
(`proxy/proxy/lifecycle.py`). The routing clamp is
`ctx_size // slots - 4096` (`proxy/proxy/provider.py:effective_per_slot_threshold`,
LP-0MSAZXXDY005AWA1) and the warm ceiling is
`min(physical clamp, economic ceiling)` where the economic ceiling is
`round(local_large_context_economic_ceiling_ratio x ctx_size)` capped by the
absolute `local_large_context_warm_cache_threshold` (LP-0MU466L1X003RTHN).

---

## 0. Gating precondition (AC0) — do not start measuring before it is met

The arms are only comparable on a **contended** GPU. Local currently runs at
~6–10 % busy on a full night, so an A/B measured now records noise, not
saturation. Before any arm is measured:

1. Generate the standard report for a **complete night** window:

   ```bash
   python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
       --start '2026-10-26 22:00:00' --end '2026-10-27 06:00:00' \
       --output-dir ~/proxy-usage-reports/gate-LP-0MVMHNUXW0061FGN-night
   ```

2. Read its `## Local model utilization` section. **Required: busy
   (dispatch → finish) >= 60 % for the whole window.**
3. Record the report path, window start/end, busy %, turns served and session
   count as a comment on LP-0MVMHNUXW0061FGN **before** any configuration
   change.

If busy < 60 %, stop: leave the parent blocked with the measured number
recorded, change no geometry flag, and re-measure on a later night. The
reporter encodes this rule too — `compare` returns `decision: inconclusive`
(exit code 2) when either arm's window is below 60 % busy.

---

## 1. Window

| Rule | Value |
|---|---|
| Window length | **one full night: 22:00 → 06:00 local (8.0 h)**, identical for both arms |
| Preferred span | the same weekday type for both arms (e.g. both Sunday→Monday nights) |
| Tolerance | the two arms' windows must have the **same length in hours** (the reporter records `window.hours`; compare refuses nothing, but a reviewer must not compare uneven windows) |
| Restart boundary | neither window may contain the arm's own restart (see §5) |

The night window is used because that is when the machine actually has demand;
a daytime window measures a different, mostly idle load. The exact window is
recorded in every capture and on the work item, so Arm A and Arm B can be given
identical-length windows.

---

## 2. Metrics captured per arm

Every metric the parent's AC3 names, with its source:

| Metric (arm JSON path) | What it measures | Source |
|---|---|---|
| `metrics.local_busy_pct` | local model busy % of the window (dispatch → finish) | analyzer (`local_busy.busy_pct`) |
| `metrics.turns_served_locally` | turns served by the local model | analyzer (`local_requests`) |
| `metrics.remote_requests` / `metrics.sessions` | context for the local share | analyzer |
| `metrics.diverts_by_reason` | local-skip diverts per `routing_skip_local` reason (legacy `warm_cache_bypass` folded into `context_too_large`) | log scan |
| `metrics.context_ceiling_diverts` | `context_too_large` + `large_context_bypass` — the size-cap diverts Arm B targets | log scan |
| `metrics.queue_waits` | `contention_queue_dispatch` events (queue-path dispatch) | analyzer |
| `metrics.fallback_after_queue` | `contention_queue_fallback_after_queue` events | analyzer |
| `metrics.queue_wait_seconds` | total queued duration | analyzer |
| `metrics.local_dispatch_first_byte` | local `dispatch_to_first_byte_ms` p50/p90 + sample count (prefill latency) | log scan |
| `metrics.ttft_all_streams` | time-to-first-token p50/p90 over all streams | analyzer |
| `metrics.prefill_slot_seconds` | prefill phase slot-seconds | analyzer (`local_busy.prefill_vs_decode.prefill`) |
| `metrics.decode_slot_seconds` | decode phase slot-seconds | analyzer (`local_busy.prefill_vs_decode.decode`) |
| `metrics.errors.slot_save_failed` | `slot_save failed` events | analyzer (`errors_by_type.slot_save_error`) |
| `metrics.errors.http_400_lines` / `http_503_lines` | HTTP 400/503 status **lines** in the window (an upper bound on events: one event can log two lines) | log scan |
| `metrics.errors.upstream_503` | upstream 503 errors | analyzer |
| `metrics.errors.by_type` | full error taxonomy | analyzer |
| `geometry` | slots, per-slot ctx, clamp, effective warm, cold, band width **as configured at capture time** | `proxy/config.yaml` |
| `live_geometry_probe` | slots / per-slot `n_ctx` **as measured on the running server** | llama-server `/slots` → `/props` |

Both arms are measured by the same code (the standard proxy-usage-analysis
pipeline plus this reporter's log scan), so a difference between arms is a
geometry difference, not a tooling difference.

---

## 3. Running an arm

### 3.1 Apply the geometry

```bash
# Show both arm definitions and what the live config currently matches
python3 scripts/slot_geometry_arm.py show

# Prepare (dry-run) the config patch for one arm; add --write to apply it
python3 scripts/slot_geometry_arm.py apply --arm B          # dry-run, prints the diff
python3 scripts/slot_geometry_arm.py apply --arm B --write  # patch proxy/config.yaml
```

`apply` patches only these keys and refuses if any of them is missing (config
drift is a loud failure):

| Key | Arm A | Arm B |
|---|---|---|
| `session_slot_pool_size` (⇒ `LLAMA_PARALLEL`) | `2` | `1` |
| `local_large_context_warm_cache_threshold` | `120000` | `258048` |
| `local_large_context_economic_ceiling_ratio` | `0.4578` | `0.9844` |
| `local_large_context_cold_cache_threshold` | `38000` | `38000` (unchanged) |

The warm cap and the economic-ceiling ratio must move **together**: the
absolute warm value is only an upper bound on the ratio-derived ceiling, so
raising it alone is a no-op (LP-0MVHSET0A008VU95). Cold stays at 38,000 so the
`(cold, warm]` band does not collapse to zero width — a collapsed band makes
the cached-ratio routing check unreachable (LP-0MSI2M5BT004BCDP).

`local_model_ctx_size` stays `262144` in both arms (total context is the same;
only the split changes), and `local_session_tenants` is left alone because it
defaults to the resolved slot count — "N tenants, N slots" stays aligned
automatically (LP-0MVN1IQTF0037WTG).

Arm B's warm ceiling is the single-slot physical clamp:
`262144 // 1 - 4096 = 258,048`.

### 3.2 Restart and verify

```bash
/skill:start-proxy --restart      # controlled teardown + relaunch of proxy and backends
python3 scripts/slot_geometry_arm.py verify --arm B --model Qwen3
```

`verify` reads the running server (`GET /slots?model=Qwen3`, falling back to
`GET /props`) and reports the measured slot count and per-slot `n_ctx`. It
**exits 1 on any mismatch** — wrong slot count, wrong per-slot context, slots
that disagree with each other, or (with `--config`) a collapsed routing band.
Never record an arm as started until `verify` exits 0.

### 3.3 Capture the arm

After the restart, wait out the cache warm-up (see §5), then run the arm for a
full night and capture:

```bash
python3 scripts/slot_geometry_ab_report.py capture \
    --label arm-b-1x262k --arm B \
    --start '2026-11-02 22:00:00' --end '2026-11-03 06:00:00'
```

Outputs (raw-data archive, gitignored — `benchmark-results/` in `.gitignore`):

```
benchmark-results/slot-geometry/<label>/
    arm.json              # the arm's metrics (the comparison input)
    analyzer/report.md    # the standard usage report for the same window
    analyzer/*.csv        # per-session CSVs (standard tooling)
    analyzer/summary.json # the analyzer JSON the capture was built from
```

Naming convention: `<arm>-<geometry>`, i.e. `arm-a-2x131k` / `arm-b-1x262k`
(append `-rerunN` if a window has to be repeated). Each archive is
self-describing: window, geometry, live geometry probe, metric sources, and the
analysis input it was built from.

To recompute a capture offline from an existing analyzer summary:

```bash
python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
    --log-dir /var/log/llama-proxy --start '...' --end '...' --output-dir /tmp/an --json > /tmp/summary.json
python3 scripts/slot_geometry_ab_report.py capture --label arm-a-2x131k --arm A \
    --start '...' --end '...' --analyzer-json /tmp/summary.json --out-dir /tmp/arm-a
```

---

## 4. Pre-registered decision gate (written before any measurement)

`compare` applies exactly these four criteria to the two captured arms. They are
constants in `scripts/slot_geometry_ab_report.py` and are **not** to be tuned
after seeing the numbers:

| # | Criterion | Passes when |
|---|---|---|
| (a) | `turns_served_locally` | candidate `>=` baseline (local volume must not shrink) |
| (b) | `context_ceiling_diverts` | candidate diverts fall by **>= 50 %** vs baseline — the whole point of the extra context |
| (c) | `first_byte_p90` | candidate p90 must not worsen by **> 30 %** (local `dispatch_to_first_byte_ms` p90; falls back to stream TTFT p90 when no local samples exist) |
| (d) | `local_busy_pct` | candidate must not regress by **> 10 points** (local work must not stall behind one slot) |

Decision precedence:

- **any criterion fails → `keep_2_slots`** (a measured regression is evidence of
  harm whether or not the window was contended);
- **all criteria pass and both arms >= 60 % busy → `adopt_1x262k`**;
- **all criteria pass but the busy gate is unmet → `inconclusive`** (exit code
  2) — an uncontended window cannot demonstrate a win.

```bash
python3 scripts/slot_geometry_ab_report.py compare \
    --baseline   benchmark-results/slot-geometry/arm-a-2x131k/arm.json \
    --candidate  benchmark-results/slot-geometry/arm-b-1x262k/arm.json
```

Exit codes: `0` decision reached (`adopt_1x262k` / `keep_2_slots`), `2`
inconclusive, `1` error/missing input.

A `keep_2_slots` / `inconclusive` verdict is a real result: record it with the
numbers on the work item. A third outcome — *mode-dependent* (keep 2 slots for
interactive work, use 1 x 262 K for large-context batches) — is chosen only by
recording which numbers motivated it, since the reporter's gate cannot express
it.

---

## 5. Restart / cache-loss marker procedure

A restart empties the 40 GiB `cache-ram` prompt cache, so the first requests
after a restart pay a full prefill. First-byte statistics from the two periods
must never be mixed, or Arm B (which always starts right after a restart) would
look worse for a reason unrelated to its geometry.

1. Immediately after `/skill:start-proxy --restart`, record the restart
   timestamp: `date -Is`.
2. `verify` the arm (§3.2) and record its JSON output with the arm's label.
3. **Do not start the measurement window until the cache has warmed**: require
   at least 12 local streams with `cache_state=warm` after the restart before
   the window opens, or simply start the window at 22:00 as §1 prescribes and
   keep the restart well outside it (preferred).
4. Set the window `--start` **after** the restart, and keep the restart out of
   both arms' windows.
5. If a restart lands inside a window (crash, OOM, mode switch), discard that
   window: re-run the arm on the next comparable night with a `-rerunN` archive
   label and say so on the work item.

The capture records `live_geometry_probe` (with its timing) so a reviewer can
confirm the geometry in force during the window; the arm helper's `verify`
output is the restart-time evidence.

---

## 6. Recording the outcome

1. Arm captures land in `benchmark-results/slot-geometry/<label>/arm.json`
   (gitignored raw data).
2. Record the decision comment on LP-0MVMHNUXW0061FGN: the two window spans,
   every metric side by side, the four gate criteria with their computed
   values, and the decision.
3. On `adopt_1x262k`, file the follow-on configuration change (or implement it
   if in scope); on `keep_2_slots`, record what would have to change for a
   re-test.
4. Update this page with the measured numbers, and propagate any guidance change
   to `models.ini` / `proxy/config.yaml` comments and
   `.pi/skills/proxy-usage-analysis/REFERENCE.md` (parent AC5).

---

## 7. Reproducibility

```bash
# Both arms, the live-config match, and the derived clamps
python3 scripts/slot_geometry_arm.py show

# Verify a running server against an arm (read-only)
python3 scripts/slot_geometry_arm.py verify --arm A --model Qwen3

# Tests (fixtures + a throwaway HTTP server; no production log or GPU needed)
python3 -m pytest tests/test_slot_geometry_ab_report.py tests/test_slot_geometry_arm.py -q
```

Related: LP-0MSAOQTJS000FFVM (local ctx-size evaluation), LP-0MSAZXXDY005AWA1
(per-slot clamp), LP-0MSI2M5BT004BCDP (band collapse), LP-0MVHSET0A008VU95
(warm cap vs economic ceiling), LP-0MU466L1X003RTHN (three-term warm model).
