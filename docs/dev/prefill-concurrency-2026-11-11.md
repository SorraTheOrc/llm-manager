# Serialised vs parallel local prefill: re-measurement and parallel-prefill test

**Work item:** LP-0MWCEHDRV000B7HD (parent spike LP-0MVXY8ZBZ0003042, question AC1b)
**Date:** 2026-11-11. **Host:** Strix Halo (gfx1151), local `llama-server`
serving Qwen3.6-35B-A3B, proxy on `:8000`, engine on `:8080`.
**Register of every number below:** `docs/dev/prefill-concurrency-baseline-2026-11-11.json`
(raw artefacts under `docs/dev/measurements/prefill-concurrency-2026-11-11-*`).
**Executable fidelity check:** `tests/test_prefill_concurrency_measurement_doc.py`
recomputes the registered figures from the artefacts and fails if this document
and the register disagree.

## Verdict

**Reject parallel prefill.** Opening the gate to two concurrent prefills does
**not** raise aggregate prefill throughput: 156.1 tok/s at concurrency 2 versus
156.2 tok/s at concurrency 1 on the same 44,770-token prompt (ratio 1.00), while
the wall window doubles (2.00×) and the slowest request in a round falls to
126.6 tok/s (0.81× of the serial per-request rate 156.5 tok/s). Prefill on this
GPU is memory-bandwidth bound, so concurrency redistributes latency between
requests instead of adding capacity — exactly what W3 recorded from production
traffic.

The rollback value is the shipped behaviour: `local_max_concurrent_prefills: 1`
(`proxy/config.yaml:1967`) keeps every request at the full single-stream rate.
The lever is a one-key config edit in either direction
(`router_helpers._max_concurrent_prefills` treats `0` as "cap disabled"), so the
decision is cheap to hold and cheap to revisit — it simply has no throughput to
buy.

Consequently the spike's AC1b question ("is serialisation the binding
constraint?") answers **no**: serialising prefills is not what caps local
prefill throughput. The cap is the engine's sustained prefill rate itself.

## 1. What is shipped (the geometry being measured)

| Setting | Value | Source |
|---|---:|---|
| `session_slot_pool_size` | 2 | `proxy/config.yaml:1935` |
| `local_max_concurrent_prefills` | 1 | `proxy/config.yaml:1967` |
| `local_model_ctx_size` | 262,144 | `proxy/config.yaml:1636` |
| per-slot context | 131,072 | 262,144 / 2 (the engine's `n_ctx_slot = 131072` agrees) |
| `ubatch-size` | 1,024 | `models.ini:78` |
| `--parallel` | follows `session_slot_pool_size` | `start-llama.sh:252` |

The configuration is unchanged since 2026-11-09
(`git log --since=2026-11-09 -- proxy/config.yaml models.ini start-llama.sh`
returns only a documentation commit), so both production windows below measure
the shipped geometry.

## 2. AC1 — re-measure with the current config

Two independent production windows, parsed from `llama-server`'s own
`prompt eval time` lines (`scripts/prefill_throughput_report.py`, which costs no
GPU time and reads live traffic's own measurements), plus the serial arm of the
engine probe.

**P1 — the prior production day (2026-11-10 00:00 → 2026-11-11 00:00 UTC)**
holds 2,238 measured prefills (all prompt sizes) over 4,689,123 prompt tokens:

| Metric | Value |
|---|---:|
| aggregate throughput (tokens / engine seconds) | **152.8 tok/s** |
| per-request rate p50 / p90 | **105.7** / **266.6 tok/s** |
| 20k–40k bucket: requests / aggregate / p50 / p90 | **12** / **241.0** / **258.0** / **287.5 tok/s** |
| 40k–80k bucket: requests | **0** |

**P2 — the current day up to the probe start (2026-11-11 00:00 → 18:50:36 UTC)**
holds 2,377 measured prefills (all prompt sizes) over 4,447,277 prompt tokens:

| Metric | Value |
|---|---:|
| aggregate throughput | **166.9 tok/s** |
| per-request rate p50 / p90 | **109.4** / **266.6 tok/s** |
| 20k–40k bucket: requests / aggregate / p50 / p90 | **13** / **243.5** / **273.2** / **293.0 tok/s** |
| 40k–80k bucket: requests / aggregate | **1** / **156.3 tok/s** (this item's own probe) |

Three findings matter more than the headline aggregate:

1. **The 20k–40k regime reproduces the recorded ceiling.** Aggregate 241.0 and
   243.5 tok/s, per-request p50 258.0 and 273.2 tok/s — within ~5 % of W3's
   recorded solo rate (253.8 tok/s) and of the ~254 tok/s ceiling the spike
   quotes. The current config has not drifted.
2. **Concurrency 1 really is serialising nothing that aggregates.** With the
   gate at 1 there is at most one prefill in flight, so per-request rate and
   aggregate rate coincide for a large prompt, and the aggregate over all sizes
   (152.8–166.9 tok/s) is dominated by small prompts, not by capacity.
3. **The 40–80k regime is not served locally today.** P1 contains **0** prefills
   above 40k tokens; P2's only one is this item's own probe. The proxy diverts
   that regime before dispatch (`context_too_large`, `large_context_bypass`,
   `local_admission_guard`), which is why the
   engine-side bucket is empty and why the parallel-prefill test had to be run
   as an engine probe rather than observed in production.

## 3. AC2 — explicitly testing parallel prefill

There is no way to observe concurrent prefills in production while the gate is
closed, so the test is a controlled engine probe (the same method and endpoint
the ubatch A/B used: direct to `:8080`, unique cache-defeating prompt text,
`cache_prompt: false`, `max_tokens: 1`). Each round requests a 40,960-token
prompt and delivers **44,770** tokens; arms alternate `c=1, c=2` per round so
that drift in the shared GPU's live load affects both arms equally.

| Round | Arm | Requests in flight | Round window | Per-request rate | **Aggregate** |
|---|---|---:|---:|---:|---:|
| 1 | serial | 1 | 286.6 s | 156.3 tok/s | **156.2 tok/s** |
| 1 | parallel | 2 | 573.8 s | 126.5 / 135.1 tok/s | **156.1 tok/s** |
| 2 | serial | 1 | 286.6 s | 156.5 tok/s | **156.2 tok/s** |
| 2 | parallel | 2 | 595.5 s | 126.6 / 126.6 tok/s | **150.4 tok/s** |
| 3 | serial | 1 | 285.8 s | 156.8 tok/s | **156.6 tok/s** |
| 3 | parallel | 2 | 572.9 s | 156.6 / 156.3 tok/s | **156.3 tok/s** |

- **Aggregate does not rise.** Median aggregate 156.1 tok/s (parallel) vs
  156.2 tok/s (serial) — ratio **1.00**; the parallel range 150.4–156.3 tok/s
  stays inside the serial rounds' own spread (156.2–156.6 tok/s).
- **Wall time doubles.** Median window 573.8 s vs 286.6 s — factor **2.00**.
- **Per-request latency pays for it.** The slowest request in a round drops to
  126.6 tok/s (**0.81×** the serial 156.5 tok/s), i.e. the client that used to
  wait 286 s waits ~354–596 s.
- **The engine, not the gate, is what serialises.** In round 3 the two requests
  ran back-to-back, each at full rate (156.6 / 156.3 tok/s, finishing at 286.1 s
  and 572.9 s into the round); in rounds 1–2 they overlapped only partially.
  `llama-server` fills an 8,192-token batch from one sequence while a prompt is
  being processed, so a second prefill only advances in the gaps — concurrency
  there buys interleaving, not parallelism at the arithmetic level.

**Scheduling cost (recorded as the item's constraint requires).** The probe ran
during live traffic (GPU 100 % busy for the whole window). The one proxy local
dispatch that arrived inside the probe window took **183.2 s** to first byte
(`dispatch_first_byte_ms=183240.7`), against a p50 of 6.3 s over 189 dispatches
in the preceding 1 h 43 m. That is the honest price of the measurement and it
is also the lever's own behaviour observed from the outside: while a 44,770-token
prefill holds the engine, a live request queued behind it.

The ubatch close-out rule (LP-0MVIQ0FSX006HQOS: do not re-raise `ubatch-size`
to 2048 without measuring at 40–80k) is **not** touched by this item: no ubatch
change is proposed. The serial arm does supply the like-for-like reference the
rule was missing — 156.2 tok/s at 44,770 tokens with `ubatch-size 1,024` — so a
future 2048 arm can be judged against the same 40–80k workload instead of the
~27k batches that decided the 2026-10-22 A/B.

## 4. AC3 — reconciliation with the W3 evidence

W3 (`proxy/docs/routing.md`, 460 paired dispatches, 2026-10-19 → 10-20) recorded
per-request p50 253.8 → 156.6 → 128.8 tok/s as 0 → 1 → 2 other prefills
overlapped, with aggregate flat at ~254 tok/s and prefill wall time inflated
2.3×. This measurement reproduces that shape and explains why it holds with the
gate closed:

| Observation | W3 (production, gate open/pool-sized) | This item (engine probe) |
|---|---|---|
| Overlapping prefills | 47.8 % of them | impossible in production; forced in the probe |
| Per-request rate under overlap | 156.6 tok/s (1 overlap) | 126.6–135.1 tok/s (probe median min 126.6) |
| Aggregate under overlap | ~254 tok/s | 150.4–156.3 tok/s |
| Wall-time factor | 2.3× | 2.00× |

Three points, in the order they matter:

1. **The overlap cost W3 measured is a property of the engine, not of the
   gate.** The probe reaches the same number (aggregate flat, wall ~2×) with the
   gate bypassed entirely, so W3's "aggregate stayed flat" was never an artefact
   of serialisation — it was memory-bandwidth conservation at the GPU.
2. **Today's serialised traffic reproduces W3's *solo* rate.** The 20k–40k
   bucket's aggregate 241.0–243.5 tok/s sits within ~5 % of W3's 253.8 tok/s
   solo rate. Serialisation does not lower throughput; it removes the overlap
   that used to inflate wall time 2.3×.
3. **Nothing in the new evidence contradicts W3**, so the item's conditional
   conclusion stands: the single-gate assumption is **not** the binding
   constraint. The binding constraint is the sustained prefill rate this GPU
   delivers — 156.2 tok/s at 44,770 tokens, 241.0–243.5 tok/s at 20k–40k.

## 5. AC4 — throughput and per-slot cost, for the spike's comparison table

| Lever | Aggregate throughput | Per-slot cost | Verdict input |
|---|---:|---|---|
| **Serialised prefills (shipped)** | **156.2 tok/s** at 44,770 tokens (156.5 per-request); **241.0–243.5 tok/s** at 20k–40k | 1 prefill gate; per-slot context 131,072 tokens; ~2.0 GiB of context-checkpoint state per prefilling slot (32 checkpoints × 62.813 MiB) | one slot sustains **13.5M tokens/day** at 40–80k, **21.0M tokens/day** at 20–40k |
| **Parallel prefills (`local_max_concurrent_prefills: 0` or >1)** | **156.1 tok/s** at 44,770 tokens (median, measured) | the same per-slot context and checkpoint cost on **two** slots at once, for no additional throughput | two slots still sustain **13.5M tokens/day** — the second slot buys latency splitting, not capacity |

For scale against the spike's ~5.8 %-share arithmetic: the per-slot capacity
figures above are the same order as the register's derived single-slot capacity
(21.9M cold tokens/day at 253.8 tok/s,
`docs/dev/single-slot-throughput-baseline-2026-10-31.json` D21). Parallel
prefill does not move that number.

## 6. AC5 — adopt/reject

**Reject.** Measurement: aggregate 1.00× (best case 1.00×, worst 0.96×), wall
time 2.00×, slowest per-request rate 0.81×. There is no measurement, and no
arithmetic over these measurements, under which opening the gate raises the
aggregate above the ~254 tok/s ceiling — the probe tops out at 156.6 tok/s for
44,770-token prompts and the 20k–40k production bucket at 243.5 tok/s.

**Rollback value:** the shipped `local_max_concurrent_prefills: 1` is the
lowest-latency configuration for every individual request, and the decision is
a single config key with no code path behind it — adopting the lever later
would be one key edit (`0` disables the cap,
`router_helpers._max_concurrent_prefills`), and rolling it back is the same
edit. Since the measured benefit is zero, the correct action is to keep the key
at 1 and stop treating prefill concurrency as a throughput lever.

## 7. Method, artefacts and reproduction

| Window | Artefact | Reproduce |
|---|---|---|
| P1 (2026-11-10, 24 h) | `docs/dev/measurements/prefill-concurrency-2026-11-11-production-2026-11-10.json` | `python3 scripts/prefill_throughput_report.py --files /var/log/llama-proxy/llama-server.log-2026-11-11 --json` |
| P2 (2026-11-11, 00:00–18:50 UTC) | `docs/dev/measurements/prefill-concurrency-2026-11-11-production-2026-11-11.json` | `python3 scripts/prefill_throughput_report.py --files /var/log/llama-proxy/llama-server.log --start '2026-11-11 00:00:00' --json` |
| E1 (engine probe, 2026-11-11 18:43–19:27 UTC) | `docs/dev/measurements/prefill-concurrency-2026-11-11-c{1,2}-r{1,2,3}.json` | interleaved `scripts/ubatch_ab_benchmark.py --sizes 40960 --reps 1 --concurrency {1,2}` rounds |

Notes on validity, in the order they could bite:

- **Log windows are not exactly reproducible.** `llama-server` writes no
  timestamps, so a window is bounded by logrotate's dateext file names (a file
  named with the 2026-11-11 rotation holds 2026-11-10's content) and by the live
  file's last write. The committed JSON artefacts are the record once the
  rotation files age out. P2 is a snapshot taken at 18:50:36, before the probe's
  second round; re-running it today yields a later end and more probe rows.
- **The probe shared the GPU with live traffic.** That lowers absolute rates
  (156.2 tok/s at 44,770 tokens, vs 241.0–243.5 tok/s at 20k–40k) but cannot
  manufacture a flat aggregate ratio, and interleaving the arms keeps both arms
  under the same load. The scheduling cost is quantified in §3.
- **`0 tokens` prompt-eval lines are excluded.** They are cache hits, not
  prefills; counting them would flatten the distribution.
- **The probe bypasses the proxy's gate** exactly as the ubatch A/B does. That
  is the point: it separates the gate's behaviour from the engine's, which is
  what the hypothesis is about.
- **No ubatch change was made.** Re-raising `ubatch-size` would require a
  `llama-server` restart (cold 40 GiB prompt cache, live traffic disrupted); the
  item's "concurrency > 1 **or** larger ubatch at 40–80k" is satisfied by the
  concurrency axis, and §3 supplies the serial 40–80k reference a future ubatch
  arm would need.
