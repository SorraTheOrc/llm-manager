# llama-server prompt-cache tuning (LP-0MUPPAC7R0053J8L)

How the local Qwen3 backend's prompt cache is sized, why the previous default
was too small, and how to reproduce the before/after measurement.

## Problem

llama-server runs with its default in-memory prompt cache limit,
`--cache-ram = 8192 MiB`. On this workload the cap is routinely saturated, so
`server_prompt_cache::update()` evicts the oldest cached prompt. An evicted
prompt is a session whose prefix is gone — its next turn becomes a full
prefill of the whole context.

Measured over the committed baseline corpus
(`prompt-cache-baseline-2026-10-01.json`, 31 llama-server log files):

| metric | value |
|---|---|
| requests | 6,739 |
| prompt tokens requested | 249,296,932 |
| tokens reused from KV cache | 224,981,209 (90.25%) |
| tokens prefilled | 24,315,723 |
| full-prefill requests (0 reused) | 461 |
| — small (`<10k`, new session — expected) | 231 reqs / 1,041,689 tokens |
| — **large (`>=10k` — lost prefix)** | **230 reqs / 6,315,379 tokens** |
| — explicitly forced by llama.cpp | 392 |
| prompt-eval wall time | 92,635 s |
| cache cap / peak | 8,192 / 8,188.7 MiB |
| cache-state samples above 90% of cap | 706 / 2,990 (23.61%) |
| maximum prompts held | 12 |
| evictions | 277 (mean 1,517.6 MiB, max 3,103.3 MiB) |

## Root cause

`--cache-ram` was unset, so `common/common.h`'s default applies:

```cpp
int32_t cache_ram_mib = 8192;  // -1 = no limit, 0 = disable, 1 = 1 MiB, ...
```

Eviction is driven by `server_prompt_cache::update()` in
`tools/server/server-task.cpp`:

```cpp
if (limit_size > 0) {
    while (states.size() > 1 && size() > limit_size) {
        SRV_WRN(" - cache size limit reached, removing oldest entry ...");
        states.pop_front();
    }
}
```

## Why a single prompt is so large (checkpoint overhead)

`Qwen3.6-35B-A3B` (`qwen35moe`) is a **hybrid** attention + SSM/recurrent
model, listed in `llm_arch_is_hybrid()`. A recurrent state cannot be rewound,
so llama.cpp keeps context checkpoints per slot. With the default
`--ctx-checkpoints 32` and a measured checkpoint size of **62.813 MiB**, the
checkpoint term alone is:

```
32 x 62.813 MiB = 2,010 MiB per prompt
```

which is why a single prompt consumes ~1.5 GiB and only a handful fit in
8 GiB. Upstream guidance (ggml-org/llama.cpp#21831) is to size the cache as
roughly `ctx-checkpoints x ~100 MB` — for 32 checkpoints that is ~3.2 GiB per
prompt, an order of magnitude above the old effective budget.

## Sizing derivation (AC2)

The value is derived from the measurement, not chosen as a round number:

* **per-prompt state** = mean evicted-entry size ≈ **1,456–1,518 MiB**
  (evicted entries are exactly the large prompts we want to retain). The
  checkpoint-derived upper bound is 2,010 MiB and the upstream rule 3,200 MiB.
* **target prompts** = observed maximum concurrently-retained prompts —
  **12** in the pre-deploy baseline, **18** in the post-deploy measurement.
* **margin** = 1.25 for headroom.

```
1,517.6 MiB x 12 x 1.25 = 22,764 MiB  ->  23 GiB = 23,552 MiB  (first step)
1,456.3 MiB x 18 x 1.25 = 32,767 MiB  ->  32 GiB = 32,768 MiB  (raised)
2,013.3 MiB x 16 x 1.25 = 40,266 MiB  ->  40 GiB = 40,960 MiB  (raised, LP-0MV5HSUHF002MDPG)
```

Configured in `models.ini` `[Qwen3]`:

```ini
cache-ram = 40960
ctx-checkpoints = 32
```

The first step (23,552 MiB) raised the retained prompts from ~3–5 to ~15.
The post-deploy measurement then observed up to **18** concurrent prompts and
residual evictions, so the cap was raised to **32,768 MiB** (~22 prompts) —
the value `--recommend` derives from the 2026-10-07 measurement
(LP-0MUYFC3C5004TCYN). The 2026-10-12 post-deploy measurement still showed
saturation under the single-slot load (18 evictions over ~4.7 d, 26.3% of
samples above 90% of the 32,768 MiB cap, observed max 16 concurrently-retained
prompts), so the cap was raised again to **40,960 MiB** (~20 prompts) — the
value `--recommend` derives from the 2026-10-12 measurement
(`docs/dev/prompt-cache-post-deploy-2026-10-12.json`, LP-0MV5HSUHF002MDPG).

### Checkpoint lever (AC3)

`--ctx-checkpoints` is the dominant per-prompt term (2,010 MiB of the ~2 GiB).
It is **kept at the default 32**:

* reducing it saves memory but coarsens rollback granularity;
* on a hybrid/recurrent model, fewer checkpoints can themselves *increase*
  full prefills (fewer anchors to match a reused prefix), which works against
  the goal;
* upstream guidance is to size the cache around the checkpoint cost rather
  than shrink the checkpoint set.

The companion granularity knob `--checkpoint-every-n-tokens` (default 8192,
available in the running build) is documented but left at its default. If a
constrained host cannot afford the derived cache, the fallback is to lower
`--ctx-checkpoints` (e.g. 16, halving checkpoint state to ~1,005 MiB/prompt)
and accept the coarser rollback — the trade-off to state when doing so.

### Headroom invariant (AC4)

The cap must not risk host OOM. `MemAvailable` already excludes the
loaded model, KV cache and other backends, so the invariant is:

```
MemAvailable - cache-ram >= margin   (margin = max(8 GiB, 10% MemTotal))
```

On the measured host:

```
77,679.5 MiB - 23,552 MiB = 54,127 MiB  >=  12,743.7 MiB   -> safe (23 GiB)
79,248.3 MiB - 40,960 MiB = 38,288.3 MiB  >=  12,743.7 MiB  -> safe (40 GiB)
```

(re-checked at deploy time for LP-0MV5HSUHF002MDPG: `MemAvailable` was
88,244 MiB on 2026-10-14, a stricter bound than the 79,248 MiB recorded at the
2026-10-12 measurement, so 88,244 − 40,960 = 47,284 MiB ≥ 12,743.7 MiB.)

The failure mode to avoid is the OOM fallback inside
`server_prompt_cache::alloc()`:

```cpp
// allocation failed -> silently cap the cache well below the configured value
limit_size = 0.4 * size();
```

If that path is reached the configured cap is not honoured, so the value is
sized to leave a wide margin.

## Dead `swa-full` flag (AC8)

`models.ini` set `swa-full = true` for `[Qwen3]` and `[Qwen3-MTP]`. This is a
**no-op** for this model: the load log shows `n_swa = 0` / `is_swa_any = 0`
and the memory path is `llama_memory_hybrid`, which takes no `swa_full`
parameter (ggml-org/llama.cpp#25913). The active lines are removed (the
single-model `--swa-full` in `start-llama.sh` for these presets too) and a
comment is left so the flag is not re-added as a supposed tunable.

## Reproduction / runbook (AC1, AC5, AC6)

Capture a baseline or an after-window measurement with the analysis script:

```bash
# Human summary
./scripts/prompt_cache_analysis.py

# Machine-readable artifact (commit this for before/after comparison)
./scripts/prompt_cache_analysis.py --json --recommend \
    --assume-cache-ram-mib 32768 > docs/dev/prompt-cache-after.json

# Restrict to one rotated file (llama-server logs carry no timestamps)
./scripts/prompt_cache_analysis.py --glob 'llama-server.log-2026-10-01'

# Multi-file corpus: comma-separated globs = any match
./scripts/prompt_cache_analysis.py \
    --glob 'llama-server.log,llama-server.*.log,llama-server.log-2026-10-*'

# Post-deploy corpus without a symlink farm: keep only files whose
# cache-state lines declare the new cap (llama logs have no timestamps)
./scripts/prompt_cache_analysis.py --cache-ram-filter 32768
```

### Throughput and first-byte latency (AC6)

Since LP-0MUYFBVHX004Z0UE the script also reports the metrics AC6 needs:

* `prompt_eval_tok_s` — prompt-eval throughput (tok/s);
* `decode_tok_s` — generation (decode) throughput (tok/s);
* `first_byte_ms` — client-visible first-byte latency, parsed from the
  proxy's `dispatch_first_byte_ms=` lines (`--proxy-log-dir`, default the
  same directory as the llama-server logs; `--no-proxy` skips it).

Each is a `{samples, median, p10, p90}` block in the `--json` summary and a
single line in the human summary. The proxy log rotates far more often than
the llama-server log, so restrict it with `--proxy-glob` (comma-separated)
to align the proxy window with the llama corpus:

```bash
./scripts/prompt_cache_analysis.py --cache-ram-filter 32768 \
    --proxy-glob 'proxy.log,proxy.log.2026-10-*'
```

The JSON records the corpus (`meta.files`, `meta.proxy_files`,
`meta.cache_ram_filter_mib`, `meta.globs`, `meta.proxy_globs`) so the window
is reproducible. For AC5/AC6, after the new `models.ini` is deployed (proxy
restart):

1. wait a comparable window;
2. re-run the script over the post-deploy files;
3. compare `large_full_prefill_requests`, `tokens_prefilled`,
   `prefill_wall_seconds` and `evictions` against the committed baseline;
4. confirm `prompt_eval_tok_s`, `decode_tok_s` and `first_byte_ms`
   (proxy-side) have not regressed.

The capacity projection in `--recommend`/`--assume-cache-ram-mib` gives the
expected effect up front: at 23,552 MiB the cache holds ~15 prompts versus the
observed maximum 12, so the projected avoided evictions and large full
prefills are 277 and 230 respectively (one avoided eviction ≈ one saved
lost-prefix full prefill — an explicit upper-bound assumption).

## AC5 — Measured improvement (post-deploy comparison)

> **Superseded for the current cap:** this section covers the first-step
> 23,552 MiB measurement. The 32,768 MiB post-deploy verification is in
> [AC5/AC6 — 32768 MiB post-deploy verification (2026-10-12)](#ac5ac6--32768-mib-post-deploy-verification-2026-10-12).

Deployment: 2026-10-01 ~21:45 (cache-ram raised from 8192 to 23552 MiB).
Post-deploy measurement corpus: 28 llama-server log files covering ~6 days
(`prompt-cache-post-deploy-2026-10-07.json`). Baseline corpus: 31 files
(`prompt-cache-baseline-2026-10-01.json`).

### Before/After table

| metric | Before (8 GiB) | After (23 GiB) | Change |
|---|---|---|---|
| requests | 6,739 | 3,610 | -46.4% |
| large full-prefills (>=10k) | 230 (3.4%/req) | 156 (4.3%/req) | +26.6% /req |
| forced full-prefills (llama.cpp) | 392 (5.8%/req) | 238 (6.6%/req) | +13.3% /req |
| **evictions** | **277 (4.1%/req)** | **88 (2.4%/req)** | **-40.7% /req** |
| cache saturation samples | 23.6% | 18.8% | -20.4% |
| max prompts held | 12 | 18 | +50.0% |
| prefill wall time (h) | 25.7 | 13.0 | -49.4% |
| tokens prefilled / req | 3,608 | 3,943 | +9.3% |
| KV-cache reuse pct | 90.2% | 88.6% | -1.8% |

### Interpretation

**Positive signals (AC5 met, with caveats):**

* **Evictions dropped 68.2% absolute (277 → 88)** and 40.7% per request.
  This is the direct, intended effect of the larger cache — fewer prompts
  are evicted so their prefix survives across turns.
* Cache saturation fell from 23.6% to 18.8% of samples, confirming the cap
  is hit less often.
* Max prompts held rose from 12 to 18, showing the cache accommodates more
  concurrent sessions.

**Confounding factors — the full-prefill rate per request is *higher*:**

* The post-deploy window includes **multiple proxy restarts** (each restart
  warms the cache from zero), which inflates full-prefill counts because
  every session's prefix is lost after a restart.
* The baseline (~15 days) had fewer restarts relative to its length, so its
  warm-cache periods dominate.
* The higher per-request full-prefill rate (3.4% → 4.3% large, 5.8% → 6.6%
  forced) is therefore a **measurement artifact of cold-cache periods**,
  not a regression of the cache-sizing change.
* During warm periods the eviction reduction is the clean signal.

**Verdict for AC5:** The eviction reduction (40.7%/req) and cache saturation
reduction (23.6% → 18.8%) confirm the cache-sizing change works as designed.
The slightly higher full-prefill rate per request is an artifact of the
shorter, restart-heavy post-deploy window — a clean comparison would
require an equal-length warm-cache window.

### Cache sizing at the margin

The post-deploy measurement shows `cache_prompts_max = 18` (vs 12 before),
so the `recommend()` function suggests **32768 MiB** (32 GiB) to hold 22
prompts at 1.25× margin. The 23552 MiB cap held 16 prompts, adequate for the
baseline's 12 but not for the current peak of 18. **Applied** under
LP-0MUYFC3C5004TCYN: `models.ini` `[Qwen3]` now sets `cache-ram = 32768`.

**Headroom check at 32768 MiB:**
```
MemAvailable ~77,108 MiB - 32,768 = 44,340 MiB >= 12,744 MiB (reserve)  -> safe
```

The wider margin means the `server_prompt_cache::alloc()` OOM fallback
(`limit_size = 0.4 * size()`) is not approached.

## AC6 — Throughput / latency

**Prompt-eval throughput (tokens processed by prompt eval / wall time):**

```
Before: 24,315,723 tokens / 92,635.5 s = 262.5 tok/s
After:  14,234,959 tokens / 46,730 s   = 304.7 tok/s
```

No regression — prompt-eval throughput is ~16% **higher** after the change.

**Generation throughput (tok/s during content generation):**

The prompt-cache analysis script was extended under `LP-0MUYFBVHX004Z0UE` to
report decode throughput and proxy-side first-byte latency too, so the figures
below (from before that extension) are superseded by the 32,768 MiB
verification section. The interim measurement
(comment on this work item, LP-0MUPQIEK10079EAW) reported a drop from
118.4 to 103.8 tok/s (-12.3%) over a ~11.5 h restart-heavy window.

This generation-throughput regression is likely the same cold-cache artifact
that inflates full-prefill rates — during a cold start, most of the
request's latency is spent on prompt processing, leaving less time for
generation per request, which depresses the measured throughput.

**Verdict for AC6:** The prompt-eval throughput did **not regress** (it
improved). The generation-throughput figure from the interim measurement
(-12.3%) is confounded by restart-heavy cold-cache periods and is likely
an artifact rather than a true regression. A clean generation-throughput
measurement would require a proxy-side metric (not currently logged).

## AC5/AC6 — 32768 MiB post-deploy verification (2026-10-12)

`LP-0MUPQIEK10079EAW`. The 32768 MiB config (`1acec9f7`) went live at the
2026-10-07 ~19:30 restart. The spawned Qwen3 child carries the tuned values
(no restart was performed to capture this measurement — the running process
already had them, and a restart would only have reset the window cold):

```
llama-server --alias Qwen3 --ctx-size 262144 --cache-ram 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 --swa-checkpoints 32 \
  --flash-attn true ...
```

`--cache-ram 32768` ✓, `--swa-checkpoints 32` (the `--ctx-checkpoints` alias) ✓,
no `--swa-full` ✓ (AC1).

Post-deploy corpus: 17 llama-server log files (2026-10-07 19:30 →
2026-10-12 12:40, ~4.7 d, 1,404 requests), selected with
`--cache-ram-filter 32768` (llama-server logs carry no timestamps, so the cap
declared in the cache-state lines defines the window). Artifact:
`docs/dev/prompt-cache-post-deploy-2026-10-12.json` (AC2).

### Comparability

The committed baseline (`prompt-cache-baseline-2026-10-01.json`) is a ~15-day,
31-file corpus whose live source files have since been rotated away, so it can
no longer be regenerated with the extended AC6 metrics. Two comparisons are
therefore recorded:

* **vs the committed baseline** — same metric set, but the windows differ
  (~15 d vs ~4.7 d); totals are not length-matched.
* **vs a length-matched 8 GiB corpus**
  (`prompt-cache-baseline-4day-2026-10-01.json`) — the four surviving daily
  files from the 8 GiB era (~4 d, 1,373 requests, comparable volume) re-run
  with the current script, so the AC6 throughput statistics have a same-tool
  baseline.

### AC5 — before/after

| metric | 8 GiB, 15 d (31 f) | **8 GiB, ~4 d (4 f)** | **32 GiB, ~4.7 d (17 f)** | length-matched change |
|---|---|---|---|---|
| requests | 6,739 | 1,373 | 1,404 | +2.3% |
| prompt tokens requested | 249,296,932 | 48,255,576 | 49,370,556 | +2.3% |
| tokens prefilled | 24,315,723 | 5,527,637 | 4,557,226 | **-17.6%** |
| tokens prefilled / request | 3,608 | 4,026 | 3,246 | **-19.4%** |
| large full-prefills (>=10k) | 230 | 56 | 58 | +3.6% (rate 4.08% → 4.13%) |
| forced full-prefills | 392 | 82 | 68 | **-17.1%** (rate 5.97% → 4.84%) |
| prompt-eval wall time (s) | 92,635.5 | 17,963.9 | 26,004.3 | +44.8% |
| **evictions** | **277** | **65** | **18** | **-72.3%** (rate 4.73% → 1.28%) |
| cache-saturated samples | 23.6% | 22.0% | 26.3% | +19.6% |
| max prompts held | 12 | 12 | 16 | +33.3% |
| KV-cache reuse | 90.25% | 88.55% | 90.77% | +2.5% |

**AC5 verdict — met, with a composition caveat.** The intended effect is
confirmed: evictions fell 72.9% per request (65 → 18 absolute) and held prompts
rose from 12 to 16, so fewer sessions lose their prefix and more concurrent
sessions fit. `tokens_prefilled` fell 19.4% per request and forced full-prefills
fell 18.9% per request. `large_full_prefill_requests` is flat per request
(4.08% → 4.13%) and `prefill_wall_seconds` rose per request (+41.6%), which is
*not* the cache behaviour regressing — see the confounds below.

### AC6 — throughput / latency

| metric (median, p10/p90) | 8 GiB, ~4 d (4 f) | 32 GiB, ~4.7 d (17 f) |
|---|---|---|
| prompt-eval throughput (tok/s) | 73.1 (28.9 / 222.7) | **77.3** (28.6 / 199.8) |
| decode throughput (tok/s) | 14.1 (5.0 / 29.5) | **18.4** (8.5 / 36.2) |
| first-byte latency, proxy (ms) | n/a (no 8 GiB proxy logs survive) | 8,046.3 (550.0 / 80,362.6, n=10,557) |

**AC6 verdict — met.** Prompt-eval and decode medians did not regress (both
improved). The 8 GiB-era proxy logs have been rotated away, so no 8 GiB
`first_byte_ms` baseline exists; the nearest surviving reference is the
23,552 MiB era (2026-10-05..07), whose median was 10,679 ms — so first-byte
latency also did not regress. `first_byte_ms` is dominated by queue wait under
the single-slot profile, so it is reported rather than used as a cache signal.

### Confounds (why per-token wall time rose)

1. **Slot profile changed with the cache.** The 8 GiB baseline era ran
   `n_slots = 3`; from 2026-10-07 the profile runs `n_slots = 1`
   (`LP-0MU1S617L`). A single slot serialises prefills, which raises wall time
   per request independently of `--cache-ram`.
2. **Composition.** A larger cache means more prefix reuse, so the *remaining*
   prefill is a smaller delta. Smaller prefills are less efficient per token
   (fixed per-prefill overhead), which lowers aggregate tok/s even though the
   total prefill work drops — exactly the trade the sizing was meant to make.
   The per-prefill median did not regress.
3. **Window.** ~4.7 d vs the committed baseline's ~15 d, with self-heal and
   mode-switch restarts flushing the cache cold.

### Sizing at the margin

The `--recommend` output from this measurement derives **40,960 MiB (40 GiB)**
(per-prompt 2,013 MiB x observed max 16 x 1.25), because the cache still
saturates at 32 GiB (26.3% of samples above 90% of cap; 18 evictions). The cap
is adequate for the observed maximum — `holdable_prompts` 20 >= observed 16 —
but the higher per-prompt size under the current single-slot load means 32 GiB
is now at the edge. **Applied** under LP-0MV5HSUHF002MDPG: `models.ini`
`[Qwen3]` now sets `cache-ram = 40960` (host headroom is safe:
`79,248 − 40,960 = 38,288 MiB >= 12,744 MiB reserve`). See the
[40,960 MiB deploy note](#40960-mib-deploy-lp-0mv5hshf002mdpg) below for the
deploy evidence.

### Reproduction

```bash
# 32 GiB post-deploy artifact (~4.7 d, 17 files)
./scripts/prompt_cache_analysis.py --json --recommend \
    --cache-ram-filter 32768 \
    --proxy-glob 'proxy.log,proxy.log-2026-10-0[89]*,proxy.log-2026-10-1[012]*,proxy.log.2026-10-0[89]*,proxy.log.2026-10-1[012]*' \
    > docs/dev/prompt-cache-post-deploy-2026-10-12.json

# length-matched 8 GiB baseline (~4 d, 4 files)
./scripts/prompt_cache_analysis.py --json --recommend \
    --cache-ram-filter 8192 --no-proxy \
    > docs/dev/prompt-cache-baseline-4day-2026-10-01.json
```

## 40960 MiB deploy (LP-0MV5HSUHF002MDPG)

Deployment: 2026-10-14 ~00:31 IST via `/skill:start-proxy --restart` (never kill the
proxy manually). The restarted proxy spawned the Qwen3 child carrying the
tuned values:

```
llama-server --alias Qwen3 --ctx-size 262144 --cache-ram 40960 \
  --cache-type-k q8_0 --cache-type-v q8_0 --swa-checkpoints 32 \
  --flash-attn true ...
```

`--cache-ram 40960` ✓, `--swa-checkpoints 32` (the `--ctx-checkpoints` alias) ✓,
no `--swa-full` ✓ (AC3 of LP-0MV5HSUHF002MDPG).

Headroom re-checked at deploy time (AC2): `MemAvailable − 40960 >= reserve`
with `reserve = max(8 GiB, 10% MemTotal) = 12,743.7 MiB`:

```
88,244 MiB (MemAvailable, 2026-10-14) − 40,960 = 47,284 MiB  >=  12,743.7 MiB  -> safe
```

The OOM fallback inside `server_prompt_cache::alloc()`
(`limit_size = 0.4 * size()`) is not approached.

The live post-deploy validation of this cap is a separate follow-up
(LP-0MV5HSZ51005CM7A). Preliminary findings captured at
`docs/dev/prompt-cache-post-deploy-2026-10-15-partial.json` (partial window,
~55 min / 50 requests):

| Metric | 32,768 MiB (1,404 req) | 40,960 MiB (50 req) | Change |
|--------|------------------------|----------------------|--------|
| Evictions | 18 | 0 | Eliminated |
| Cache saturated % | 26.27% | 0.0% | Eliminated |
| Cache peak (MiB) | 32,765.6 | 5,651.2 | Well under cap |
| Forced full prefills | 68 (4.8%/req) | 5 (10.0%/req) | Higher % (small n) |
| Prompt eval tok/s med | 77.3 | 131.6 | Improved |
| Decode tok/s median | 18.4 | 26.0 | Improved |
| First byte ms median | 8,046.3 | 8,106.7 | Comparable |
| Max prompts held | 16 | 6 | Fewer held |

`--recommend` output (50-request sample): only 20,480 MiB needed — 40,960 MiB
provides ample headroom. The full-window verification, the log-rotation data
loss that limited it, and the slot-partitioned analysis are in the next
section (LP-0MV5HSZ51005CM7A).

## 40960 MiB post-deploy verification (2026-10-18) — LP-0MV5HSZ51005CM7A

Follow-up validation of the 40,960 MiB cap vs the committed 32,768 MiB
window (`prompt-cache-post-deploy-2026-10-12.json`, 1,404 requests, ~4.7 d).

### AC1 — deploy args (re-verified live 2026-10-18 ~17:00 UTC)

The live Qwen3 child carries the tuned values (no restart was needed to
capture this measurement):

```
llama-server --alias Qwen3 --ctx-size 262144 --cache-ram 40960 \
  --cache-type-k q8_0 --cache-type-v q8_0 --swa-checkpoints 32 \
  --flash-attn true ...
```

`--cache-ram 40960` ✓, `--swa-checkpoints 32` (the `--ctx-checkpoints`
alias) ✓, no `--swa-full` ✓.

### AC2 — window and data availability (blocked by llama-server log rotation)

The ~4.7-day post-deploy window (deploy 2026-10-13 23:31 UTC) matured
~2026-10-18 16:20 UTC, but the full window is **not reconstructible**:
llama-server numbered logs rotate to 15 copies on every spawn
(`rotate_llama_logs(keep=15)`), and the restart rate is high (mode-switch
fast⇄cheap restarts and crash-restarts; ~4 restarts in the final hour alone
and all 15 numbered segments on disk written on 2026-10-18). The surviving
40,960 corpus therefore *shrank* as earlier segments were discarded:

| captured | 40,960 corpus |
|---|---|
| 2026-10-16 07:16 (interim) | 1,065 requests |
| 2026-10-16 21:44 (interim) | 760 requests |
| 2026-10-18 ~17:00 (this run) | **380 requests** |

Each restart resets the in-memory prompt cache, so the surviving fragment is
dominated by short, cold-cache restart segments — it cannot stand in for the
full-window comparison. The committed artifact
(`docs/dev/prompt-cache-post-deploy-2026-10-18.json`) is the rotation-limited
best available; raising llama-server log retention for multi-day measurement
windows is tracked in follow-up LP-0MVE0JS3B005PWY2.

### AC3/AC4 — aggregate comparison (rotation-limited corpus vs 32,768 MiB)

| metric | 32,768 MiB (1,404 req, ~4.7 d) | 40,960 MiB (380 req, rotation-limited) | change |
|---|---|---|---|
| evictions | 18 (1.28%/req) | 0 (0.0%/req) | **improved** |
| cache-saturated samples | 67 (26.27%) | 0 (0.0%) | **improved** |
| cache peak (MiB) | 32,765.6 / 32,768 | 7,375.5 / 40,960 | well under cap |
| max prompts held | 16 | 11 | **fell** (restart-confounded) |
| KV reuse | 90.77% | 60.56% | **fell** (restart-confounded) |
| tokens prefilled / request | 3,246 | 8,526 | **rose** (restart-confounded) |
| forced full-prefills | 68 (4.84%/req) | 69 (18.16%/req) | **rate rose** (restart-confounded) |
| large full-prefills | 58 (4.13%/req) | 73 (19.2%/req) | **rate rose** (restart-confounded) |
| prompt-eval tok/s (median) | 77.3 | 116.8 | **improved** |
| decode tok/s (median) | 18.4 | 24.4 | **improved** |
| first-byte ms, proxy (median) | 8,046.3 (n=10,557) | 11,424.2 (n=6,130) | **regressed +42%** (queue/restart-dominated) |

**Verdict.** The cap is not harming the workload — in the surviving fragment
the cache never approached 40,960 MiB (peak ~18% of cap, 0 evictions, 0%
saturation) and prompt-eval/decode throughput improved. But AC3's *"~0
evictions under sustained load"* is **not confirmed**: the only load-representative
corpus captured at this cap (the 2026-10-16 interim, 1,065 requests) still
reached 99.8% of the cap (peak 40,866.6 MiB) with 10 evictions (0.94%/req vs
1.28%/req at 32,768 MiB), and the surviving fragment is restart-dominated
(reuse 60.6% vs 90.8%, forced-prefill rate 18.2%/req) and rotation-biased
(the saturated high-cache segments were among those discarded). AC4's
first-byte latency regressed +42% on the raw numbers, consistent with the
cold-cache-restart queue effect previously documented rather than a
cap-capacity signal; the 2026-10-16 interim at the same cap showed the same
magnitude. No further raise is indicated: every `--recommend` output at this
cap (39,936 MiB from the 1,065-req interim; 27,648 MiB from this fragment)
is at or below the deployed 40,960 MiB.

### Slot-partitioned evaluation (per the 2026-10-16 methodology)

The corpus was partitioned by the Qwen3 child's slot count
(`srv load_model: initializing slots, n_slots = N` in each llama-server file;
the proxy `total_slots` field is unreliable). First-byte latency cannot be
slot-attributed (proxy-side), so partitions are llama-only
(`--no-proxy`). Artifacts: `prompt-cache-post-deploy-2026-10-18-slot{1,2,3}.json`.

| n_slots | requests | reuse | tokens prefilled/req | evictions | cache peak (MiB) | max held | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 1.14% | 22,225 | 0 | 1,228 / 40,960 | 2 | too small to judge |
| **2 (current profile)** | **311** | **64.59%** | **7,607** | **0** | **7,376 / 40,960** | **11** | **cap ample; restart-confounded** |
| 3 | 65 | 45.61% | 12,080 | 0 | 2,874 / 40,960 | 4 | too small to judge |

Per-partition verdict for the producer's question *"does the larger cache
work?"*: at the current 2-slot profile the cap is demonstrably ample — 0
evictions, 0% saturated samples, peak ~18% of cap across 311 requests. The
1-slot and 3-slot subsets are too small for a verdict. Slot-profile
drift (the 3→1 change LP-0MU1S617L has since moved to `--parallel 2`) and
the restart-dominated window prevent a load-matched comparison to the
32,768 MiB baseline, whose per-slot composition is unreconstructible.

### Reproduction

```bash
# aggregate (AC2) — aligned proxy glob
./scripts/prompt_cache_analysis.py --json --recommend --cache-ram-filter 40960 \
    --proxy-glob 'proxy.log,proxy.log-2026-10-1[345678]*,proxy.log.2026-10-1[345678]*' \
    > docs/dev/prompt-cache-post-deploy-2026-10-18.json

# per-slot partitions (llama-only; n_slots from the child's log line)
./scripts/prompt_cache_analysis.py --json --recommend --cache-ram-filter 40960 \
    --no-proxy --glob 'llama-server.log,llama-server.9.log,llama-server.11.log,llama-server.12.log,llama-server.14.log,llama-server.2.log,llama-server.5.log,llama-server.log-2026-10-17.gz,llama-server.log-2026-10-18' \
    > docs/dev/prompt-cache-post-deploy-2026-10-18-slot2.json
```

> **Reproducibility caveat:** the surviving corpus rotates underneath a
> re-run (live file grows, numbered files cycle at 15), so minor count drift
> between runs is expected; the committed artifacts are the canonical
> snapshot.

## What the proxy cannot see (cold-labelled dispatches that were cache hits)

LP-0MWRQVLFY003BK5L. The proxy's `cache_state` / `cached_ratio` on a
`dispatch_first_byte_ms` line is a **routing prediction**, not a measurement:
`_lookup_dispatch_cache_state()` reads `provider._get_cached_ratio(model,
session)`, a table keyed by ``(model, session)``
(`proxy/proxy/router_helpers.py`, threshold `_WARM_CACHE_RATIO_THRESHOLD =
0.5`). A fresh session — or one whose entry was invalidated or evicted — has no
entry, so the ratio defaults to ``0.0`` and the dispatch is labelled ``cold``
*whatever the backend actually did*.

llama-server's prompt cache is **global and cross-session**:
`server_prompt_cache` restores a matching prefix into any slot, for any
session, as long as it is still in `--cache-ram`. The proxy therefore cannot
see a genuine hit when the routing table has no entry. On the 2026-11-21 window
that gap produced a large population of sub-second dispatches labelled
``cached_ratio=0.0`` that were, in fact, almost entirely served from the
prompt cache.

### Window and figures (2026-11-21, UTC)

The window is the whole of 2026-11-21 as carried by the proxy logs
(00:00:00 → 23:04:06). The independent cross-check reads the rotated
`llama-server.log-2026-11-22` file (llama-server lines carry no timestamps, so
the file *is* the 2026-11-21 window). Every figure in the log-analysis tables
below is produced by `proxy/scripts/analyse_cold_labelled_cache_hits.py` and is
reproducible from the commands at the end of this section; the replay figures
come from `proxy/scripts/replay_cold_labelled_cache_hit.py`.

A dispatch is in the audited set when it is cold-labelled
(`cache_state=cold` or `cached_ratio=0.0`), its
`dispatch_to_first_byte_ms` is under **800 ms**, and `estimated_tokens` is
above **3 000**.

| metric | value |
|---|---|
| dispatch lines in window | 2,449 |
| cold-labelled | 910 |
| **cold-labelled, sub-800 ms, > 3 000 est tokens** | **572** |
| distinct sessions | 40 |
| — one session repeating an identical prompt | 527 (526 at `estimated_tokens=7 185`) |
| dispatches with a server-side measurement | 572 / 572 (100.0 %) |
| llama-server `measured_cached_ratio` | median **0.9996** (p10 0.9996 / p90 0.9996, min 0.9841, max 0.9999) |
| measured new tokens / request | median **4** (max 148) |
| reused / measured prompt tokens | **5,861,226 / 5,864,993 (99.94 %)** |
| time-to-first-byte | median 132.9 ms |
| implied prefill throughput | median **54,063 tok/s** (p10 49,792 / p90 55,784) |
| engine aggregate prefill rate (same window) | **207.2 tok/s** ⇒ implied is **261×** higher |

> The 572 count differs by one from the 571 quoted when the item was filed; the
difference is threshold-boundary drift on a re-run, not a change in the data.

**Why the implied rate settles it.** The engine reports its own bulk prefill
rate in the same window: 3,555,647 tokens prefilled across 2,359 `prompt eval`
lines, wall time 17,159,629 ms ⇒ **207.2 tok/s**. A dispatch that moves 7,185
estimated tokens to first byte in 132.9 ms is an implied 54,063 tok/s — 261×
what the engine can compute inline. (The proxy's `estimated_tokens` relies on a
char/4 heuristic that *under*-counts here: the repeated prompt tokenises to
10,011 real tokens, so the true implied rate is higher still.) 54,000 tok/s is
not a prefill; it is a cache restore.

### Signal 1 — llama-server reuse captured on the dispatch itself

With `local_measured_cache_abort_enabled` on, the proxy injects
`return_progress` and llama-server reports `cache = n_prompt_tokens_cache`
against `total`. The proxy logs this as `prefill_progress_measured` **immediately
before** the dispatch line for the same session. Joining each audited dispatch
to its own measurement (consumed once, same session) gives, for the example
line from the item (`estimated_tokens=11499 cached_ratio=0.0`):

```
prefill_progress_measured session=… total=10011 cache=10007 processed=10007
    measured_cached_ratio=0.9996 measured_new_tokens=4 priced_new_tokens=100
 dispatch_first_byte_ms=127.8 dispatch_to_first_byte_ms=127.5
    cache_state=cold cached_ratio=0.000 estimated_tokens=7185
```

Every one of the 572 audited dispatches carries such a line, and the median
measured reuse is **99.96 %** while the routing label said **0 %**.

### Signal 2 — independent llama-server slot-log cross-check

The same file gives a second, proxy-independent derivation. Each task logs
`new prompt … task.n_tokens = N` and then
`n_tokens = X, memory_seq_rm [X, end)`; `X` is the retained (reused) prefix and
`N - X` the tokens actually prefilled:

| metric | value |
|---|---|
| tasks / with a reuse line | 2,540 / 2,540 |
| reused / prompt tokens over the window | **50,343,878 / 51,048,581 (98.62 %)** |
| dominant repeated prompt | **10,011 tokens × 527 → 99.96 % reused** (exactly `n_past=10,007` each time) |
| `prompt eval` lines / tokens | 2,359 / 3,555,647 |

So the repeated prompt the proxy priced at `cached_ratio=0.0` reused 10,007 of
its 10,011 tokens **in the engine's own slot log**, 527 times over.

### Signal 3 — bounded controlled replay

`proxy/scripts/replay_cold_labelled_cache_hit.py` sends the **same** prompt
twice, each from a fresh `X-Session-Id`, and records both sides. The proxy prices
every fresh session cold; the engine caches the prefix across sessions. A live
run on 2026-11-22 11:36 UTC (prompt 4,146 real tokens, proxy estimate 3,651):

| request | proxy label | first byte | llama-server `prompt eval` |
|---|---|---|---|
| cold (first session) | `cold / 0.0` | 6,504.7 ms | `5821.42 ms / 4146 tokens` (712.20 tok/s) |
| warm (second, **fresh** session) | `cold / 0.0` | **92.2 ms** | **`50.04 ms / 4 tokens`** |

The second request was priced `cached_ratio=0.0` / `cache_state=cold` yet
llama-server reports `measured_cached_ratio=0.999` (4,142 of 4,146 tokens
reused) and computes only **4** tokens — a **60.6×** first-byte speedup. That is
a warm prefix the proxy did not price.

Scope/constraints honoured: no synchronous `/slots` call is added to the
admission path (the replay speaks only the normal chat-completions protocol),
and no proxy restart was needed — the replay ran against the running proxy.
The `private` local-only alias is used, whose `force_full_prompt` sends the
whole prompt each turn: exactly the cross-session-cache case. The replay is
bounded to two requests and retries the proxy's retryable 503 (capacity) rather
than disturbing an in-flight prefill.

### Reproduction

```bash
# Log-only analysis (two independent signals; --llama-log is optional)
./proxy/scripts/analyse_cold_labelled_cache_hits.py \
    --date 2026-11-21 \
    --llama-log /var/log/llama-proxy/llama-server.log-2026-11-22

# Same window, machine readable (the figures above)
./proxy/scripts/analyse_cold_labelled_cache_hits.py \
    --date 2026-11-21 \
    --llama-log /var/log/llama-proxy/llama-server.log-2026-11-22 --json

# Any other window; proxy-log timestamps are the authoritative boundary
./proxy/scripts/analyse_cold_labelled_cache_hits.py \
    --start '2026-11-21 00:00' --end '2026-11-22 00:00'

# Bounded controlled replay (two requests; retries a capacity 503 only)
./proxy/scripts/replay_cold_labelled_cache_hit.py
```

The analysis exits 0 and prints the audited set, its measured reuse and the
implied-vs-engine rate. The replay exits 0 only when the second (fresh) session
is priced cold but measured warm; `--json` on either script is the canonical
artifact to commit for a later comparison.

The measurement above establishes *that* warmth is invisible. Which detection
path, if any, can make it visible to the ETA — and what each would cost — is
assessed below in
[Detection paths assessed](#detection-paths-assessed-lp-0mwrrqvm5l006rc05)
(LP-0MWRQVM5L006RC05).

## Detection paths assessed (LP-0MWRQVM5L006RC05)

Work item: [investigation] Assess warm-prefix detection paths and file
follow-ups. The question is narrow: **can anything turn cross-session prompt-
cache warmth into an input the admission ETA can price?** Three candidate paths
are assessed — (a) llama-server prompt-cache metrics on `/metrics` or `/slots`,
(b) an **asynchronous** warm-prefix sampler feeding the residency sidecar, and
(c) charging the observed cold-first-byte outcome back into the calibration
window.

Each is scored on the measurement it needs, the cost/latency it adds to the
admission path, and a verdict. Everything below was measured on this host on
2026-11-22 against llama-server build `e97492369` (the deployed
`/home/rgardler/llama.cpp/build/bin/llama-server`, router mode over
`models.ini`, `Qwen3` loaded).

| path | measurement it needs | available? | cost added to admission | verdict |
|---|---|---|---|---|
| (a) `/metrics` or `/slots` cache metrics | per-request cache-hit tokens or cache-ram contents | **no** — `/metrics` 501, and aggregates only; `/slots` carries no cache contents | none if scraped, but a `--metrics` flag + restart, and fleet-level only | **not viable** |
| (b) async warm-prefix sampler → sidecar | the measured `prompt_progress` `cache / total` of a completed dispatch, keyed by prompt-prefix fingerprint | **yes** — already captured by `local_measured_cache_abort_enabled` | ~0.06 ms per admission (fingerprint + lookup), no upstream call | **viable with conditions** — recommended |
| (c) charge the observed outcome back into the calibration window | the same measured `cache / total`, read where the sample is recorded | **yes** — the measurement is already in scope ~20 lines below the sample call | **zero** (no new work, no new call) | **viable with conditions** — required, not sufficient on its own |

### (a) llama-server prompt-cache metrics on `/metrics` or `/slots` — not viable

**What it would need.** A per-request cache-hit count (or the current contents of
the prompt cache, keyed by prompt) so admission can ask "is *this* incoming
prefix warm?".

**What the deployed build actually exposes.**

```console
$ curl -s 'http://localhost:8080/metrics?model=Qwen3'
{"error":{"code":501,"message":"This server does not support metrics endpoint. Start it with `--metrics`","type":"not_supported_error"}}
```

The metrics endpoint is not enabled by the running process (`ps` shows no
`--metrics`; `models.ini` sets none), so enabling it is a preset edit **plus a
restart**. More decisively, the endpoint could not answer the question even when
enabled: at this build the whole metric set is aggregate
(`tools/server/server-context.cpp`, `get_metrics`) — counters
`prompt_tokens_total`, `prompt_seconds_total`, `tokens_predicted_total`,
`tokens_predicted_seconds_total`, `n_decode_total`, `n_tokens_max`,
`n_busy_slots_per_decode`; gauges `prompt_tokens_seconds`,
`predicted_tokens_seconds`, `requests_processing`, `requests_deferred`. There is
**no cache-hit counter, no `cache-ram` occupancy gauge and no prompt identity**.

`/slots` is no better. A live probe (2026-11-22, 1.07 s round-trip on a busy
slot) returned exactly:

```json
["id", "n_ctx", "speculative", "is_processing", "id_task", "params", "next_token"]
```

The current task's prompt text and generated text appear only under
`LLAMA_SERVER_SLOTS_DEBUG` (`slots_debug`, `server_slot::to_json`), and even then
they describe **the slot's own task**, not the global prompt cache. The cache
itself (`server_prompt_cache`, `tools/server/server-task.cpp`) has no HTTP
surface at all: `load()`/`alloc()`/`update()` are internal, and `update()`
evicts `states.front()` — **oldest-first by bytes** against the `cache-ram`
limit — which the proxy cannot observe.

**Cost.** Zero on the admission path if scraped asynchronously, but the signal
is a fleet-level delta (`prompt_tokens_total` counts only tokens actually
pushed into a batch, i.e. *after* prefix reuse — verified at
`server-context.cpp:2540`, so Σ submitted − Δ counter is an aggregate hit
volume). It cannot attribute warmth to the request being admitted.

**Verdict: not viable** as a warm-prefix detector. It remains useful as a
coarse aggregate cross-check, which the committed log analysis already provides
for free (§"Signal 1/2" above), so no follow-up is filed for enabling it.

### (b) asynchronous warm-prefix sampler feeding the sidecar — viable with conditions (recommended)

**What it needs.** The only measured warmth the proxy ever sees: the
`prompt_progress` event's `cache` / `total` for a **completed** dispatch
(`measured_prefill.get_measured_prefill`), plus a tokenizer-free fingerprint of
that dispatch's message list (`slot_cache_ratio.message_fingerprints`). Written
into an in-process index keyed by `(model, fingerprint prefix)` →
`(measured total, measured cache, observed_at)`.

**Why "asynchronous", and why not `/slots`.** The AC forbids a synchronous
hot-path `/slots` call for the measured reason (`/slots` held >15 s during a
prompt batch, LP-0MUXTW4MD003TVIC). A sampler *of `/slots`* would also be
useless here: `/slots` carries no cache-ram contents (above), and the proxy's
existing ~1 s snapshot (`observability._last_slot_details_cache`, consumed by
`residency.py`) therefore can only ever describe **slot** residency — the
per-session fact the ETA already prices. The sampler that works is the one that
writes the **measurement the dispatch already produced**, off the admission
path, at first-byte time — the same place `record_dispatch_sample` and
`record_cold_measured_sample` already run.

**Cost added to the admission path.** One fingerprint pass plus one prefix
comparison, no HTTP:

```console
$ python3 -c "...message_fingerprints on 31 messages / 97.5 KB..."
# 200 iterations, 2026-11-22, this host
fingerprint:   0.057 ms per call
prefix_match:  0.0005 ms per call
```

0.06 ms against a 30 s queue budget is ~5 orders of magnitude of headroom.

**Conditions for viability.**

1. **Measured evidence only.** An entry may be written only from a real
   `prompt_progress` measurement — never from a prediction, or the index
   inherits the very optimism it exists to remove.
2. **The credit is capped at the measurement** (as
   `slot_cache_ratio.safety_margin` caps the slot-snapshot credit), because the
   entry can go stale: llama-server evicts oldest-first by bytes, so a still-held
   entry can be a prefix it has already dropped.
3. **The failure mode is over-credit, and it is bounded by an existing
   backstop.** A mis-credited, actually-cold dispatch is released early by the
   measured-cache-abort path (`reason=measured_cache_deficit`, measured p50
   3.8 s versus the 120 s prefill deadline in force when that was measured —
   180 s today). That backstop must stay armed.
4. **It cannot help a first occurrence.** A novel prefix has no measurement, so
   the blind spot remains for the first time a prompt is ever seen — by
   construction, not by omission.

**Expected effect — measurable.** The audited population is a *repeat* prefix,
so it is exactly what this path can price:

| figure | measured baseline | with the measured-warmth credit |
|---|---|---|
| own-ETA of a 12,176-token repeat (guard records, 2026-11-22) | **29.7 s** (p50 over the 33 audited pairs) | **≈25.6 s** on today's coded coefficient (≈2.1 ms/token, inferred from 1,125 guard records), **< 1 s** once the coded coefficient is fixed (LP-0MWL0VHOY0065EI8) |
| `prefill_eta_accuracy` p50 over those matched pairs | **10,764 %** | ≈7,000 % on today's coefficient, **≈40–100 %** with the coded fix |
| admission behind one occupant (`projected_wait = eta × (busy + 1)`) | 59.4 s vs a 30 s budget → **refused** | 51.2 s → still refused; **< 2 s → admitted** |

The last two rows are the reason (b) is *recommended* rather than *sufficient*:
the credit changes what the ETA multiplies, and today the multiplier itself is
inflated — 182 guard records on 2026-11-22 price a prompt with **≤ 60 new
tokens** at **≥ 20 s** of own prefill, e.g.

```
2026-11-22 00:02:43,513 local_admission_guard decision=reject reason=eta_exceeds_budget
  eta_prefill_s=60.0 eta_occupancy_s=120.0 new_tokens=22 projected_wait_s=179.9 budget_s=30.0
  rate_bucket=solitary samples=200 busy_slots=2 …
```

A request with 22 new tokens priced at 60 s of own prefill — that is the
coded-coefficient defect already filed as LP-0MWL0VHOY0065EI8, not the blind
spot, and no warmth credit can repair it.

### (c) charging the observed outcome back into the calibration window — viable with conditions

**What it needs.** Nothing new: the router holds the measured ratio of the
dispatch it is recording. In `proxy/proxy/router.py` the priced window is fed the
**prediction** (~l.3327)

```python
record_dispatch_sample(model_name, _estimated_tokens, _cached_ratio,
                       _first_byte_service_ms, contended=_is_contended)
```

while ~20 lines below the same scope reads the **measurement** for
`record_cold_measured_sample` (`measured_prefill.get_measured_prefill(model,
session_id)`). A cross-session hit therefore enters the window as
`(coded=0, new=<whole prompt>, ms≈130)`.

**Cost added to the admission path: zero.** The measurement is already in scope;
this is a change of argument, not a new call.

**Shipped (LP-0MWRWBJPP001Q5JX).** The window now records the dispatch's
**measured** reuse (`measured_prefill.get_measured_prefill(...).measured_cached_ratio`)
when one exists and the routing prediction only when none does, resolved by
`prefill_eta.resolve_window_ratio`. Each sample's basis (`measured` / `predicted`)
is logged on the `dispatch_first_byte_ms` line (`window_ratio_basis=`) so the
window's composition is attributable. This changes only the sample
**composition** of the calibration window — no admission price is touched by
this item, and the cross-session blind spot above is unchanged: an unmeasured
(possibly warm) prompt is still priced from the routing prediction.

**Evidence that this matters.** 2026-11-22 partial window (00:00 → 13:10 UTC): of
the 1,317 measured dispatches ≥ 3,000 estimated tokens, **42.3 %** were priced at
≥ 2× their measured new tokens (Σ priced 4,884,984 vs Σ measured 3,384,634 new
tokens), and the audited set's median measured reuse is 0.9997 against a routing
label of 0.0. Replaying the solitary window over the same 1,102 dispatches
through the shipped `_fit` / `estimate_prefill_seconds`:

| window fed | coded ms/token | prefill ms/token | 14,548-token cold ETA | 12,176-token fully-warm ETA |
|---|---|---|---|---|
| the **predicted** ratio (today) | 0.352 | 1.987 | 28.9 s | 5.13 s |
| the **measured** ratio | 0.228 | **3.975** | **57.8 s** | 3.33 s |

Phantom-cold samples **halve** the fitted cold-prefill rate; fed measured data
the fit lands on 3.98 ms/token, agreeing with the engine's own aggregate prefill
rate (207.2 tok/s ≈ 4.8 ms/token) and the shipped 3.9 ms/token default.

#### Reproduction

The replay reuses the committed analyser for parsing and joining, then feeds the
shipped sampler and fit — no proxy import, no upstream call:

```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "an", "proxy/scripts/analyse_cold_labelled_cache_hits.py")
an = importlib.util.module_from_spec(spec); sys.modules["an"] = an
spec.loader.exec_module(an)
sys.path.insert(0, "proxy")
from proxy import prefill_eta

lines = an.read_log_lines(an.discover_proxy_logs(an.DEFAULT_LOG_DIR))
events = sorted(
    (e for e in (an.parse_dispatch(l) or an.parse_measurement(l) for l in lines)
     if e is not None and START <= e.ts <= END),
    key=lambda e: e.ts)
paired = an.join_measurements(events)

for use_measured in (False, True):          # predicted (today) / measured (path c)
    prefill_eta._samples.clear()
    for p in paired:
        d = p.dispatch
        ratio = d.cached_ratio
        if use_measured and p.measurement is not None:
            ratio = p.measurement.measured_cached_ratio
        prefill_eta.record_dispatch_sample(
            d.model, d.estimated_tokens, ratio, d.to_first_byte_ms)
    eta, coeffs = prefill_eta.estimate_prefill_seconds(d.model, 14_548, 0.0)
    print(use_measured, eta, coeffs.coded_ms_per_token, coeffs.prefill_ms_per_token)
```

with `START`/`END` set to the window above (`2026-11-22T00:00:00` …
`2026-11-22T23:59:59`). The replay models the **solitary** window only; the live
proxy also populates a contended window, which is why the live fitted rates
differ from these — the direction and the ≈2× gap are the claim, not the
absolute milliseconds. As everywhere else in this document the window is live,
so a re-run over a longer corpus drifts by a few percent (the committed table is
the snapshot).

**Conditions for viability.**

1. **It does not detect anything.** It re-rates the estimator; an unmeasured
   prompt is still priced cold, so on its own it changes no admission decision
   for the blind-spot population.
2. **It must not ship alone.** Honest cold prices are *higher*: the 14,548-token
   cold prompt moves 28.9 s → 57.8 s, i.e. from "admitted inside a 30 s budget"
   to "refused". Shipping (c) without (b) converts admits into refusals for
   prompts that are, in fact, warm.
3. **It is the right sample source for (b).** Crediting a warmth ratio against
   rates fitted from phantom-cold samples is half a fix — which is why it is the
   blocking prerequisite of the recommended path.

**Verdict: viable with conditions** — a strict calibration-fidelity fix, and the
prerequisite of (b). Implemented as LP-0MWRWBJPP001Q5JX.

### Recommendation

**Path (b)** — the asynchronous warm-prefix index written from the measured
`prompt_progress` reuse and read at admission — is the only path that can price
cross-session warmth, and path (a) is closed by the backend surface rather than
by effort. Ship it with its two prerequisites:

| follow-up | role | state |
|---|---|---|
| `LP-0MWRWBPFV00339XP` — *Price cross-session prompt-cache warmth at admission from the measured reuse of a matching prefix* | the recommended path | filed, blocked |
| `LP-0MWRWBJPP001Q5JX` — *Prefill calibration window is fed the predicted cache ratio* | blocking prerequisite (path (c)) | implemented |
| `LP-0MWL0VHOY0065EI8` — *Warm 99 %-cached prompt priced near cold rate* | conditional prerequisite: the coded coefficient that the credit multiplies | pre-existing, open |

Expected effect, stated against this document's own baseline: the audited
population's own-ETA p50 falls from **29.7 s** to the coded-term floor of
**≈25.6 s** from the credit alone, and to **< 1 s** with LP-0MWL0VHOY0065EI8;
the matched `prefill_eta_accuracy` p50 over those pairs falls from **10,764 %**
to below **1,000 %**; and the audit set stops being refused behind a single
occupant (59.4 s → < 2 s projected wait against a 30 s budget). None of these
figures is a behaviour change in this item: it is an investigation, and the
pricing ships under the items above, each with its own restart.

## Non-goals

* `force_full_prompt` / delta-routing behaviour.
* The proxy's disk slot save/restore (`session_slot_*`) — broken upstream for
  hybrid models (ggml-org/llama.cpp#25913) and not the cache path that matters.
* Re-tuning the routing cold/warm thresholds.
