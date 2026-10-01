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

* **per-prompt state** = mean evicted-entry size = **1,517.6 MiB**
  (evicted entries are exactly the large prompts we want to retain). The
  checkpoint-derived upper bound is 2,010 MiB and the upstream rule 3,200 MiB.
* **target prompts** = observed maximum concurrently-retained prompts = **12**.
* **margin** = 1.25 for headroom.

```
1,517.6 MiB x 12 x 1.25 = 22,764 MiB  ->  round up to 23 GiB = 23,552 MiB
```

Configured in `models.ini` `[Qwen3]`:

```ini
cache-ram = 23552
ctx-checkpoints = 32
```

This raises the retained prompts from ~3–5 to ~15, so a session prefix
survives across turns.

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
77,679.5 MiB - 23,552 MiB = 54,127 MiB  >=  12,743.7 MiB   -> safe
```

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
    --assume-cache-ram-mib 23552 > docs/dev/prompt-cache-after.json

# Restrict to one rotated file (llama-server logs carry no timestamps)
./scripts/prompt_cache_analysis.py --glob 'llama-server.log-2026-10-01'
```

The JSON records the corpus (`meta.files`) so the window is reproducible. For
AC5/AC6, after the new `models.ini` is deployed (proxy restart):

1. wait a comparable window;
2. re-run the script over the post-deploy files;
3. compare `large_full_prefill_requests`, `tokens_prefilled`,
   `prefill_wall_seconds` and `evictions` against the committed baseline;
4. confirm prefill throughput (tokens/second in the `prompt eval time` lines)
   and client first-byte latency have not regressed.

The capacity projection in `--recommend`/`--assume-cache-ram-mib` gives the
expected effect up front: at 23,552 MiB the cache holds ~15 prompts versus the
observed maximum 12, so the projected avoided evictions and large full
prefills are 277 and 230 respectively (one avoided eviction ≈ one saved
lost-prefix full prefill — an explicit upper-bound assumption).

## Non-goals

* `force_full_prompt` / delta-routing behaviour.
* The proxy's disk slot save/restore (`session_slot_*`) — broken upstream for
  hybrid models (ggml-org/llama.cpp#25913) and not the cache path that matters.
* Re-tuning the routing cold/warm thresholds.
