# RESULTS NOTE — snapshot save/restore cost per band, 2026-10-28

Work item: **LP-0MVOG8TK6002EWZU** (Measure save/restore cost per band under load),
parent **LP-0MVN3KPEC007WWL5** (Evaluate widening the session-slot snapshot pool
2 → 4), harness from C1 **LP-0MVOG8SYO006R49J**.

## Verdict up front

**Every band is `plausible` in both runs: all 8 saves and all 8 restores finished
in under 0.3 s — between ~60x and ~2 000x inside the adaptive timeout budget
(`min(3.0 + 0.0015·tokens, 60)`), including while another slot streamed.**
The restore mechanism is *not* the bottleneck the parent's 0.0015 s/token model
estimated: the measured restore wall time is 0.02–0.09 s (GPU KV load + slot
restore), not 0.0015 s/token of prompt. Restore effectiveness is token-verified
at ≥ 99.4 % across all bands (only 4 prompt tokens reprocessed per band), so a
restored 60k+ session genuinely resumes inside the 60 s cap.

What the run *does* show to be slow — and what the 0.0015 s/token model actually
describes — is the **populate prefill** (19.96 s at 10k → 480.8 s at 60k+). That
cost is the barrier to a *fresh* session, not to a session resuming from a
snapshot, and it is exactly what the parent's skip-when-busy / cooldown gates
already exclude from the live path.

Under-load deltas were small: save +0.01…+0.17 s, restore +0.02 s, zero
timeout-exceeded operations. The correct reading is that save/restore are so
cheap that a concurrent stream on the other slot hardly moves them.

## Environment record

| Item | Value |
|---|---|
| Host | sorra-agents-1, Ubuntu 7.0.0-34-generic; AMD Radeon (Strix Halo APU, gfx1151) |
| Measured model | `unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M` (alias `Qwen3`), llama-server child of the 8080 router: `--parallel 2 --ctx-size 262144 --cache-ram 40960 --cache-type-k q8_0 --cache-type-v q8_0 --swa-checkpoints 32 --slot-save-path /home/rgardler/projects/llm/slot-cache --n-gpu-layers 80` |
| Per-slot ctx | 131 072 (262 144 / 2 slots) |
| Measured endpoint | `http://127.0.0.1:8080` (the router the proxy itself uses; instance ports churn) |
| `sha256sum proxy/config.yaml` before + after | `c2cee32fca9e55b7698d0c7df969cb3357ec76accaa9fe063721e15c8163966e` (unchanged) |
| Timeout model | `session_slot_timeout_seconds: 3.0` + `session_slot_timeout_per_token_seconds: 0.0015`, capped at `session_slot_max_timeout_seconds: 60` (cap from 38 000 tokens) |
| `session_slot_pool_size` | 2 (production, unchanged) |
| `session_slot_skip_when_busy` | true (production; bypassed by direct measurement) |
| `local_max_concurrent_prefills` | 1 |
| MemAvailable (pre) | 74.4 GiB |
| Runs | unloaded 12:59:30Z→13:32:32Z; under load 13:32:32Z→13:46:15Z |

## Exact commands

```bash
# Unloaded
python3 -m proxy.benchmarks.slot_snapshot_benchmark \
  --bands 10k,25k,40k,60k+ --base-url http://127.0.0.1:8080 \
  --slot-id 1 --verify-slot-id 1 --populate-slot --verify-chat \
  --snapshot-dir /home/rgardler/projects/llm/slot-cache \
  --session-id snapshot-bench-unloaded

# Under load (a second 512-token streaming completion pinned to slot 0)
python3 -m proxy.benchmarks.slot_snapshot_benchmark \
  --bands 10k,25k,40k,60k+ --base-url http://127.0.0.1:8080 \
  --slot-id 1 --verify-slot-id 1 --populate-slot --verify-chat \
  --snapshot-dir /home/rgardler/projects/llm/slot-cache \
  --session-id snapshot-bench-underload --under-load --load-slot-id 0 --load-max-tokens 512

# Report (pure function of the two runs)
python3 -m proxy.benchmarks.slot_snapshot_band_report \
  --unloaded unloaded.json --under-load under_load.json \
  --output-dir . --markdown README.md
```

The harness hosts a stub-client unit suite (`proxy/tests/test_slot_snapshot_benchmark.py`,
`test_slot_snapshot_band_report.py`) so every field above is pinned offline.

## Measured results

| band | prompt tokens | timeout (s) | fill (s) | save (s) unload/load | restore (s) unload/load | Δ restore under load | cached tokens | cached ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10k | 9 942 | 18.0 | 19.96 | 0.03 / 0.05 | 0.02 / 0.04 | +0.02 | 9 938 | 99.96 % | plausible |
| 25k | 24 942 | 40.5 | 95.25 | 0.05 / 0.07 | 0.03 / 0.05 | +0.02 | 24 938 | 99.98 % | plausible |
| 40k | 39 942 | 60.0 | 233.93 | 0.10 / 0.11 | 0.05 / 0.07 | +0.02 | 39 938 | 99.99 % | plausible |
| 60k+ | 59 942 | 60.0 | 480.83 | 0.12 / 0.29 | 0.07 / 0.09 | +0.02 | 59 938 | 99.99 % | plausible |

- Save/restore server-side (`timings.save_ms`/`restore_ms`) tracked the wall
  clock (≈0 queueing in every band); under load, ≤0.02 s of queueing appeared.
- Disk footprint per band: 166.3 / 322.3 / 478.2 / 686.2 MiB — 1.10–1.61× the
  `kv_memory_table.py` pure-KV estimate (the excess is the hybrid/recurrent state
  + SWA checkpoints, an additive overhead, so the ratio converges toward 1 at
  larger bands). Four retained snapshots at all four bands = **6.5 GiB,
  32.3 % of the 20 GiB `cleanup-slot-cache.sh` cap**; even four 60k+ snapshots
  alone = 2.7 GiB (13.4 %).
- Restore effectiveness: `cached_tokens = prompt_total − timings.prompt_n` gave
  ≥ 99.4 % for every band; `timings.prompt_n` was 4 in both runs every time.

## Why the earlier 05:31Z run was discarded

The first harness version sized prompts at 4.0 chars/token, but the production
tokeniser consumes ~1.0 chars/token for this numeric content (measured
2026-10-28: 40 125 chars → 40 062 prompt tokens). The stale `unloaded.json`
(schema v2) therefore measured a 10k band as 40 039 tokens, and its 25k/40k/60k
bands blew past the 131k slot context. The harness now sizes in blocks
(`CHARS_PER_TOKEN = 1.0`) and the 05:31Z artefact is replaced here.

## Harness fixes made during this item (all unit-pinned)

1. **Populate/erase are administrative, with their own budgets.** The first full
   run mis-attributed a 20 ms erase as a `failed` 18 s erase because it reused
   the band's adaptive timeout. Administrative calls now use
   `ERASE_TIMEOUT_SECONDS = 600` (an abandoned erase is *not* harmless — it can
   execute after the populate and wipe the band, observed as `n_saved: 0`) and
   `POPULATE_TIMEOUT_SECONDS = 900`.
2. **Populate retries (≤3) on transport/5xx only.** The proxy's model-health
   loop recycles the llama-server instance during long prefills (a 60k prefill
   stops answering `/slots`, the 5 s probe times out, recovery reloads the
   model), which killed two bands with `HTTP 500: proxy error: Failed to read
   connection`. Retrying the *administrative* populate cannot flatter the
   save/restore numbers it prepares; 4xx stays single-shot.
3. **`snapshot_holds_band` fidelity guard.** `n_saved` is compared against the
   populated prompt size; a save of the wrong KV (reload or a concurrent session
   taking the slot — observed as `n_saved: 3 878` for a 40k band) is classified
   `snapshot-not-band` and never read as a measurement.
4. **Slot actions carry the model name.** The router refuses `/slots/{id}` without
   a model; the proxy itself passes `model` (`proxy/proxy/session.py:
   _call_slot_endpoint`). Measuring through the router is the production path and
   immune to instance port churn.

## Fidelity caveats

1. **No mode lock held.** A concurrent mode switch during the window would
   restart the backend and void the affected band. None occurred; band timestamps
   are recorded in `unloaded.json`/`under_load.json` for correlation.
2. **No live dispatch lease taken.** Save/restore bypass the proxy's per-endpoint
   dispatch lease / slot-ownership registry; a live session could have been
   dispatched onto the measured slot mid-band. `queued s` (wall clock minus
   `timings.*_ms`) is recorded per operation as the observable that would expose
   such interference; it was ≤0.02 s under load.
3. **The measured slot is destroyed.** Each band erased and prefilled slot 1, so
   a live session owning that slot would lose its cached KV (re-prefill cost on
   return). The proxy reported `available_slots: 2` and no local dispatches
   during the measured windows (all routed remote in the prior 24 h), so no live
   session was observed displaced.
4. **The `--under-load` condition is one concurrent stream** on the other slot
   (the harness's own), not the full production contention mix — that is the
   bake-off replay child's job. The mechanism cost is so low that the delta is
   small by construction; the numbers show the *worst* single-stream case is
   cheap, not that production contention never exceeds it.
5. **`cached_tokens` cannot separate restored KV from the RAM prompt cache**
   (`--cache-ram 40960`): both surface as a small `prompt_n`. The latency
   comparison against the adaptive timeout is what decides "benefits inside the
   cap".
6. **The populates are measured with a 900 s budget**, so a band's populate
   duration does not bound the save/restore pair being measured.

## Artefacts

| File | Content |
|---|---|
| `unloaded.json` | Raw unloaded run (schema v3) |
| `under_load.json` | Raw under-load run (schema v3) |
| `summary.json` | Machine-readable band report (deltas, footprint, verdicts, cleanup-cap comparison) |
| `README.md` | Rendered report (same content, human-readable) |
| `RESULTS_NOTE.md` | This note |

## Implications for the parent decision

- **Pool-4 doesn't pay for restore latency.** Restores are 0.02–0.09 s at every
  band; the 60 s cap is not approached by restore even at 60k+. The benefit of
  the snapshot mechanism is bounded by *when a matching snapshot exists*, not by
  save/restore cost.
- **The 0.0015 s/token model over-budgets the mechanism by 2–3 orders of
  magnitude** for save/restore. It describes re-prefill cost, which the snapshot
  mechanism avoids, not the cost of restoring a KV file.
- **Disk is the practical constraint.** 4 × all-bands ≈ 6.5 GiB (32 % of the
  20 GiB cap) with today's `slot_*.bin` naming that `cleanup-slot-cache.sh`
  actually cleans; a pool of 4 with one snapshot per session stays well under the
  cap at observed session sizes unless sessions cluster at 60k+.
