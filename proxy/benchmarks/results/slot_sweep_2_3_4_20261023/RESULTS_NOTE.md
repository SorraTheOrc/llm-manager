# RESULTS NOTE — controlled N=2/N=3 slot sweep, 2026-10-23

Work item: **LP-0MVK096GF0038FTH** (Execute N=2/N=3/N=4 slot sweep in a quiet window),
parent **LP-0MVGPEP3O000RKKD** (Run controlled N=3/N=4 slot-count sweep in a quiet window).
Predecessor: **LP-0MVK0969J004MTG3** (preflight gate + validated commands).

## Verdict up front

**Both runs are INVALID for local-latency conclusions: local stream share = 0.0 % at
N=2 and N=3.** Every completed benchmark stream was served by a remote provider
(`OpenCode Go 3/deepseek-v4.1-flash`, `MunitoAI Free/deepseek-v4-flash`), so the
reported TTFT/TPS are network numbers, not local numbers. The queue-timeout and
denial counts are recorded but are **not comparable with production session-minutes**
(see "Why local share is 0 %" below).

**N=4 was not attempted** — quantified skip reason in "N=4 decision".

The measured data therefore **cannot validate or refute** the concurrency-vs-capacity
projection (N=2 denies ~19.7 % of session-minutes, N=3 ~6.7 %, N=4 ~2.2 %); the
projection's status is **not testable with this harness as configured**. See
LP-0MVK09622002CRMR for the recorded verdict and the reopen conditions.

## Environment record

| Item | Value |
|---|---|
| Preflight gate (`scripts/slot_sweep_preflight.py --json --timeout 30`) | exit **0**, `quiet: true`, `local_idle_seconds=412.5`, `available_slots=2/total_slots=2`, `local_owner_session_id=null`, `slots_stale=false` (see `preflight_pre.json`) |
| Gate time | 2026-10-23T20:06:35Z |
| `sha256sum proxy/config.yaml` **before** | `d1a7a154e99a224484e6d431281c9d40363c7676655ce87b5743672e61378dcc` |
| `sha256sum proxy/config.yaml` **after** | `d1a7a154e99a224484e6d431281c9d40363c7676655ce87b5743672e61378dcc` (identical) |
| `session_slot_pool_size` | 2 (production value; restored after the sweep) |
| `local_model_ctx_size` | 262 144 |
| `contention_queue_max_wait_seconds` | **60** (halved from 120 by LP-0MVIQ0EXV00673HO — the value in force, not the 120 s assumed in the parent description) |
| `local_max_concurrent_prefills` | 1 (prefill is serialised: slots add decode concurrency, not prefill capacity) |
| llama-server `--parallel` before | 2 (observed on the live process) |
| System MemAvailable before | 44.3 GiB (87 GB used of 124 GB) |
| Lease holders at gate time | none (the only idle lease was the driving agent session) |

Pre-run values are also captured in `env_pre.txt`; the raw gate JSON is
`preflight_pre.json`.

## Exact commands

```bash
# 1. Gate (must exit 0)
python3 scripts/slot_sweep_preflight.py --json --timeout 30

# 2. Measured N=2 then N=3 (harness restarts llama-server between slot counts)
LLAMA_PROXY_BENCHMARK_ROOT=/home/rgardler/projects/llm \
  python3 -m proxy.benchmarks.slot_benchmark \
  --sweep --resume 2 --concurrency 3 --prompt-tokens 2000 \
  --output-dir <worktree>/proxy/benchmarks/results/slot_sweep_2_3_4_20261023/n2_n3

# 3. Report preserved before any further invocation (generate_report overwrites
#    slot_benchmark_report.json / _summary.txt on every run)
cp n2_n3/slot_benchmark_report.json n2_n3/report_2_3.json
cp n2_n3/slot_benchmark_summary.txt n2_n3/report_2_3.txt
```

The harness writes `slot_<N>.json` per run and restores `proxy/config.yaml` plus
restarts the proxy/llama-server at the original slot count in a `finally` block
(verified — see "Config restoration").

## Measured results per N

| Metric | N=2 | N=3 |
|---|---|---|
| Requested / observed llama-server `--parallel` (`observed_slots`) | 2 / **2** ✅ | 3 / **3** ✅ |
| Per-slot context clamp (`per_slot_ctx`, physical = ctx/N − headroom) | 126 976 | 83 285 |
| Warm threshold implied (95 % of per-slot ctx) | 120 627 | 79 121 |
| Completed / errors | 11 / 12 (1 error) | 11 / 12 (1 error) |
| **Local stream share** | **0.0 %** (0 / 11) | **0.0 %** (0 / 11) |
| Resolved providers observed | remote (OpenCode Go 3, MunitoAI Free — deepseek-v4.x-flash) | remote (same) |
| Avg TTFT (`time_to_first_token_seconds`) | **0.033 s** (network, not prefill) | **0.162 s** (network) |
| Avg decode tok/s | 147.3 | 161.0 |
| Avg / max request duration | 64.9 s / 451.3 s | 104.4 s / 616.6 s |
| Total completion tokens | 32 867 | 24 581 |
| Queue-timeout count / rate | 2 / **66.7 %** | 0 / **0.0 %** |
| Local quarantine episodes | 1 | 0 |
| `system_memory_bytes` (MemAvailable at run end) | 90.68 GB | 86.42 GB |
| `finish_reason=length` count | **not recorded** (harness limitation — see Limitations) | **not recorded** |
| Total run wall time | 24.5 min (20:08:29Z → 20:33:00Z) | 19.1 min (20:33:41Z → 20:52:5xZ) |

Per-slot KV memory is not sampled by the harness (llama-server exposes no per-slot
KV metric without `--metrics`); `proxy/benchmarks/kv_memory_table.py` gives the
modelled split — total KV ≈ 2.7 GiB is **constant** across slot counts, so per-slot
KV is ≈ 1.35 GiB at N=2, ≈ 0.9 GiB at N=3, and the N=4 cost is the context clamp
(61 440 / warm threshold 58 368), not VRAM. The measured MemAvailable agrees: the
model+KV footprint is ≈ 27–34 GB in every configuration.

## Why local share is 0 % (root cause, from the proxy logs of the run window)

This is **structural**, not a load artefact — the 2026-10-17 sweep also reported
`local share 0.0 %`, and the 2026-10-20 attempt stalled for the same reason.

1. Each slot-count restart reloads the model, which re-arms the proxy's
   **cold-start admission window** (`proxy/proxy/cold_start.py`, LP-0MUCEFCAT005NFNN):
   while cold, local dispatch concurrency is capped at
   `COLD_START_DEFAULT_MAX_CONCURRENT = 1` and the cap only lifts
   (`mark_warm`) once the first local prefill completes.
2. The harness fires `--concurrency 3` immediately after `Llama-server ready`, i.e.
   inside that window. At least two of the three requests are refused at the local
   gate on every cycle:
   `local_dispatch_denied session=benchmark-2-… owner=benchmark-2-… active=1 cold_start=True`,
   answered with a retryable `server_busy` + `Retry-After` so the client falls back.
3. The one admitted request holds the single cold-start local slot for a whole
   agentic turn — the harness targets up to **18 000 output tokens**
   (`output_tokens_target: 18000`, `--prompt-tokens 2000`) — so the remaining two
   exhaust the 60 s `contention_queue_max_wait_seconds` window (observed
   `contention_queue_wait max_wait=60.0s max_depth=8 saturation=prefill`, then
   `contention_queue_dispatch … queued_duration=49.29s`) or the 1 200 s client
   timeout, and complete remotely (`Fallback triggered … from=local-qwen3, to=munitoai-free, reason=HTTP 503`).
4. Result: every completed stream resolves to a remote provider, and the N=2 run
   reports one local quarantine episode from the `/health` samples taken while
   llama-server was still settling after its restart.

**Consequence for the projection:** the sweep's 3-way concurrency is *not* a proxy
for "3 concurrent production sessions". Production sessions arrive staggered and
warm (45.7 % of local dispatches are already cache-cold, but they are not 18 k-token
cold-start floods), and the projection's N=3 benefit comes from admitting three
*different* sessions — which this harness cannot exercise because its own cold-start
guard permits a single concurrent local dispatch. The measured queue-timeout rates
(66.7 % at N=2, 0 % at N=3) are therefore **not comparable** with the projection's
denied session-minutes and must not be used to justify a slot-count change.

## Validity caveats (carried forward to the verdict)

1. **INVALID runs.** Local stream share 0.0 % at N=2 and N=3 ⇒ no local-latency
   conclusion (TTFT / decode tok/s) may be drawn. Both runs are marked INVALID.
2. **Driver-session interference.** The sweep was launched from an agent session.
   That session's own requests were *denied* local dispatch during N=3
   (`local_dispatch_denied session=herdr-1792778971-1879334-27664 owner=benchmark-2-2-…`),
   i.e. they ran remotely and did not consume a local slot, but they are inside the
   measurement window and appear in the log-derived contention counts.
3. **Other host actors.** A second herdr agent session (`LP-0MVKN9IRP003XJH6`) was
   alive throughout, and a status-probe loop (`ac3-serving-*`) polled
   `/llama/local/status` every 10 s (it only fires a probe request when all slots
   are idle). Status polls add log lines to the contention window.
4. **`contention_queue_max_wait_seconds` in force = 60 s** (not 120 s): queue-timeout
   rates are only comparable with runs using the same value.
5. **Restart-induced cold window** (item 1 above) applies to every slot count: this
   is the mechanism that produces the 0 % local share.
6. **`finish_reason` is not captured** by this harness version; proxy.log is cleared
   by the restore restart, so it cannot be derived retroactively for this run.
7. **Remote-provider variance.** Remote streams were served by deepseek-v4.x-flash
   on two different providers with rate-limit fallbacks between them
   (`from=munitoai-free, to=opencode-go-3, reason=rate_limited`), so remote TTFT/TPS
   are not a stable baseline either.

## N=4 decision — skipped (quantified)

Memory gate **passed** (N=3 ended at 86.42 GB MemAvailable ≫ the 40 GiB threshold,
no viability warnings, `observed_slots=3`), but **the N=4 run was skipped** because
both completed runs are INVALID (local share 0.0 %): an N=4 run would reproduce the
same invalid measurement (cold-start cap = 1 concurrent local dispatch) at the cost
of another restart cycle plus ~20 min, and AC4 forbids drawing conclusions from an
invalid run. Re-running N=4 is only meaningful once the harness can land requests
locally (see the reopen conditions in LP-0MVK09622002CRMR).

## Config restoration (AC5 of the child)

- `sha256sum proxy/config.yaml` after the sweep: `d1a7a154e99a224484e6d431281c9d40363c7676655ce87b5743672e61378dcc`
  — **identical** to the pre-run value.
- `session_slot_pool_size: 2` restored by the harness `finally` block.
- llama-server observed `--parallel 2` after the restore; proxy `/health` →
  `{"status":"healthy","ready":true,"current_model":"Qwen3"}`.
- `/llama/local/status` after restore: `llama_server_running=true`,
  `available_slots=2`, `total_slots=2`.
- The preflight gate passes again once the host is idle for the minimum window
  (the gate was re-evaluated after restoration; the driving session's own activity
  is the only lease at that point).

## Artefacts

| File | Content |
|---|---|
| `preflight_pre.json` | Gate result before the sweep (exit 0, quiet) |
| `env_pre.txt` | Config sha256, MemAvailable, llama-server cmdline/`--parallel`, timestamp |
| `sweep_n2_n3.log` | Full harness output (per-request progress, contention + viability blocks, restore) |
| `n2_n3/slot_2.json` | N=2 run artefact (per-request rows, contention, viability) |
| `n2_n3/slot_3.json` | N=3 run artefact |
| `n2_n3/report_2_3.json` / `.txt` | Preserved combined report (N=2 + N=3) |
| `n2_n3/slot_benchmark_report.json` / `_summary.txt` | Harness report files as written |

## Limitations / follow-ups

- `finish_reason` (and therefore `finish_reason=length`) is not recorded per request —
  harness gap, filed as a discovered-from follow-up.
- A harness that can exercise *session* concurrency locally is required to test the
  projection: e.g. stagger arrivals (so the first prefill lifts the cold cap before
  the next request arrives), single-turn requests with production-sized outputs, or a
  `--concurrency N` mode that waits for `mark_warm` before launching requests.
