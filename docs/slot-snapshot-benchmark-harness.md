# Slot snapshot save/restore benchmark harness

`proxy/benchmarks/slot_snapshot_benchmark.py` — measures what a llama-server
KV **slot snapshot** costs per prompt-size band, so the "widen the snapshot
pool" evaluation (`LP-0MVN3KPEC007WWL5` AC1) rests on measurements instead of
estimates.

It is the harness half of AC1; the measurement runs (and the bake-off replay)
are separate children and add columns, not plumbing. It also serves the
virtual-slot spike (`LP-0MVGD9ZKI004ETTE` / `LP-0MVKU0RG0002WWF2`) rather than
growing a second harness.

## What it measures

Per band (default `10k`, `25k`, `40k`, `60k+`):

| Field | Meaning |
|---|---|
| `save_seconds` | Wall time of `POST /slots/{id}?action=save` |
| `restore_seconds` | Wall time of `POST /slots/{id}?action=restore` |
| `snapshot_bytes` | On-disk size of the snapshot file (when the directory is visible to the harness) |
| `save_outcome` / `restore_outcome` | `success` / `failed` / `skipped` |
| `save_error` / `restore_error` | The failure detail (HTTP status + body, or exception type + message) |
| `timeout_seconds` | The adaptive timeout the operation was issued under |
| `save_within_timeout` / `restore_within_timeout` | Whether it finished inside that timeout |
| `prompt_tokens_estimated` | Harness sizing (character heuristic) |
| `prompt_total_tokens` | `usage.prompt_tokens` from the verification completion, else the estimate |
| `prompt_processed_tokens` | llama-server `timings.prompt_n` (prompt tokens actually processed) |
| `cached_tokens`, `cached_ratio` | The token-based hit (see below) |
| `under_load` | Whether the pair ran while a second slot was streaming |

## Invocation

```bash
# Offline check — no llama-server, no proxy, no GPU required
python3 -m proxy.benchmarks.slot_snapshot_benchmark --dry-run
python3 -m proxy.benchmarks.slot_snapshot_benchmark --dry-run --json

# Measure the four bands against a live llama-server (slot 0)
python3 -m proxy.benchmarks.slot_snapshot_benchmark \
    --base-url http://localhost:8080 \
    --snapshot-dir "$(grep -m1 'session_slot_save_path:' proxy/config.yaml | awk '{print $2}')"

# AC1 "under load": a second streaming session keeps another slot processing
# while the saves issue (needs >= 2 llama-server slots).
python3 -m proxy.benchmarks.slot_snapshot_benchmark --under-load --json

# Add the token-based hit measurement: one non-streaming completion per band
# after the restore, recording usage.prompt_tokens and timings.prompt_n.
python3 -m proxy.benchmarks.slot_snapshot_benchmark --verify-chat --json

# Custom bands (names or raw token counts; 60k+ is a floor, not a ceiling)
python3 -m proxy.benchmarks.slot_snapshot_benchmark --bands 10k,25k,60k+,90000
```

`--json` writes one machine-readable document (plan + records + totals);
without `--json` a table is printed. A live run also writes
`benchmark-results/slot_snapshot_benchmark_<UTC stamp>.json`
(`LLAMA_PROXY_BENCHMARK_ROOT` selects the repo root — the same convention as
`slot_benchmark.py`).

## The hit definition is token based

```
cached_tokens = prompt_total - timings.prompt_n
```

where `prompt_total` is the request's total prompt tokens
(`usage.prompt_tokens` from the verification completion, falling back to the
harness estimate when the server does not report it) and `timings.prompt_n`
is the number of prompt tokens llama-server actually processed.

Latency is **corroboration only** and never part of the hit test: a restore
that took 40 s but still left 24k of a 25k prompt to prefill is a *miss*
however fast it was, and a restore that took 58 s but left only 500 tokens is
a *hit*.

## Timeout model

Every save/restore is issued with the live adaptive timeout from
`proxy/proxy/session.py:_build_slot_context` (the `proxy/config.yaml`
slot-persistence safeguards):

```
min(session_slot_timeout_seconds
    + session_slot_timeout_per_token_seconds * tokens,
    session_slot_max_timeout_seconds)
```

i.e. `min(3.0 + 0.0015 * tokens, 60)` with the shipped values — the cap
engages at exactly `(60 - 3) / 0.0015 = 38 000` tokens. The model is resolved
from `proxy/config.yaml` when present and from the documented shipped
defaults otherwise, and each record carries the timeout it ran under plus
whether it finished inside it. Override per run with `--timeout-base`,
`--timeout-per-token`, `--timeout-max`.

## Failure policy

A failed save or restore is a **measurement outcome, not a harness failure**:
it is recorded with its error and elapsed time, is **never retried** (a retry
would replace a timeout with a success and destroy the data AC1 needs), and
the run continues. When the save fails the restore is recorded as `skipped`
rather than issued against a snapshot that was never written.

## Snapshot-size cross-check

Measured bytes are cross-checked against `proxy/benchmarks/kv_memory_table.py`:

```
expected_bytes = KV_MIB_PER_TOKEN * prompt_tokens * 1 MiB
```

`KV_MIB_PER_TOKEN` is the measured q8_0 per-total-ctx-cell KV footprint used
by the F4 headroom table (1362.7 MiB / 131 072 cells ≈ 0.0104 MiB/token). The
recorded `snapshot_size_ratio` is `measured / expected`, and
`snapshot_size_in_tolerance` accepts a factor-of-two band either way
(`SNAPSHOT_BYTES_TOLERANCE = (0.5, 2.0)`). This is an order-of-magnitude
cross-check — a 10x deviation means the snapshot is not what we think it is,
not that the formula is off by 10x.

## Fidelity caveats

1. **Prompt sizing is a character estimate.** The harness emits numeric
   content (comma-separated integers, the same salted-prompt driver style as
   `slot_benchmark.py`) and assumes 4.0 chars/token, which *over-delivers* real
   tokens relative to the band floor (~3.5 chars/token for digits). Always use
   the recorded `prompt_total_tokens`/`prompt_tokens_estimated` rather than the
   band label when comparing runs.
2. **`--verify-chat` is best-effort.** llama-server assigns chat requests to a
   slot by availability, so `timings.prompt_n` is authoritative only when the
   verification completion lands on the slot that was just restored. On a
   multi-slot server, run the verification with a single free slot or repeat
   it; the record keeps `verified: false` when the server reports no timings.
3. **"Under load" is one concurrent stream.** `--under-load` keeps one
   streaming completion open on another slot for the whole save/restore
   window (the driver re-opens it if it finishes early). It reproduces the
   "saves issued while another slot is processing" condition, not the full
   production contention mix — that is the bake-off replay's job
   (`LP-0MVN3KPEC007WWL5` AC2).
4. **`--snapshot-dir` must match llama-server's `--slot-save-path`.** The
   harness only *stats* the file; if the path is wrong `snapshot_bytes` is
   `null` (never fabricated) and the size cross-check is `null` too.
5. **llama-server does not time itself out.** The harness applies the
   adaptive timeout to its own HTTP calls, exactly as `_call_slot_endpoint`
   does; a client-side timeout is recorded as a failure with the exception
   type, which is what the production path would log as `slot_save failed`.
6. **No authentication / no proxy hop.** Save/restore and the verification
   completion go straight to llama-server on `--base-url`. The proxy's own
   gates (`session_slot_skip_when_busy`, context cap, circuit breaker) are
   measured separately by the replay and log-analysis tools, so this harness
   isolates the server-side cost.

## Reuse (deliberately not rebuilt)

- `proxy/benchmarks/slot_benchmark.py` — result-directory convention
  (`LLAMA_PROXY_BENCHMARK_ROOT`), salted-prompt driver style, concurrency.
- `proxy/benchmarks/kv_memory_table.py` — per-token KV footprint for the size
  cross-check.
- `scripts/slot_persistence_harness.py` / `scripts/analyse_slot_ratio_restore.py`
  — the `slot_save` / `slot_restore` log vocabulary this harness reproduces
  (`success session=… slot=…`, `failed slot=… error=… elapsed=… timeout=…`), so
  benchmark records and production logs can be compared directly.
- `proxy/tests/prefill_thrash_replay_harness.py` — the fixture-driven,
  offline-testable analysis-tool pattern, including the "no production module
  imports the harness" guard (`SNAPSHOT_BENCHMARK_SENTINEL`).

## Tests

`proxy/tests/test_slot_snapshot_benchmark.py` — offline only (no GPU, no live
server): band sizing for 10k/25k/40k/60k+, the adaptive-timeout formula and
its 60 s cap boundary, the record schema, failed save/restore recording with
no retry, the under-load interleaving proof (stubbed HTTP client), the
token-based hit definition, the `kv_memory_table.py` size cross-check,
`--dry-run`/`--json`/`--bands` CLI behaviour, and the AC5 no-production-import
guard.
