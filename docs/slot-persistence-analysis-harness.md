# Slot Persistence Analysis Harness — Schema & Usage

Part of F1 (LP-0MTCMEJX2008W85X) for the KV slot save/restore evaluation
(LP-0MTAQNB7J0094X71). The harness parses llama-server.log*, proxy.log* and
the slot-cache inventory into one structured JSON corpus of baseline metrics.

## Usage

```bash
# Full corpus over all available logs (no live-proxy dependency)
python3 scripts/slot_persistence_harness.py --log-dir /var/log/llama-proxy > corpus.json

# Incident-day window (2026-08-26): proxy events filtered to the day
python3 scripts/slot_persistence_harness.py \
  --log-dir /var/log/llama-proxy \
  --start 2026-08-26 --end 2026-08-27 > corpus-2026-08-26.json

# Incident-day with day-exact llama metrics (llama logs have no timestamps;
# --llama-file restricts parsing to that day's rotated file)
python3 scripts/slot_persistence_harness.py \
  --log-dir /var/log/llama-proxy \
  --llama-file '*2026-08-27*' \
  --start 2026-08-26 --end 2026-08-27 > corpus-2026-08-26-dayexact.json

# Compact (no indentation) for programmatic consumption
python3 scripts/slot_persistence_harness.py --log-dir /var/log/llama-proxy --compact > corpus.json

# Baseline summary only (meta + baseline_metrics + per-file llama breakdown)
python3 scripts/slot_persistence_harness.py --log-dir /var/log/llama-proxy --summary \
  > baseline-summary.json

# Print the JSON schema
python3 scripts/slot_persistence_harness.py --schema
```

The harness reads the log directory as-is (live + rotated, plain and
gzip-compressed files) — there is no dependency on a running proxy or
llama-server. Copy the log files to a snapshot directory and rerun for
reproducible corpora (the `meta.generated` timestamp is the only field that
changes between runs over the same snapshot; all `baseline_metrics`,
`llama_files_seen` and event counts are deterministic).

## Corpus schema

| Top-level key | Contents |
|---|---|
| `meta` | analysis window, file counts, line counts, generation timestamp |
| `slot_save_events` | every proxy slot_save success/failure (ts, session, slot, status; failures: error, elapsed, timeout, busy_info) |
| `slot_restore_events` | every proxy slot_restore success/failure |
| `skip_events` | routing_skip_local + slot-persistence skip/cooldown events with structured reason/session/threshold fields |
| `routing_check_events` | proxy routing_check lines (per-request proxy-side token estimates) |
| `slots_status_codes` | status_request polls (slots_stale, total/available slots, server state) |
| `slots_status_summary` | poll/healthy/stale counts |
| `lease_events` | lease renewed/released events (lease churn) |
| `orphan_events` | orphan-cleanup lease events |
| `llama_checkpoint_events` | llama-server `created context checkpoint` events (native KV checkpoints, with n_tokens/size) |
| `llama_checkpoint_restore_events` | llama-server `restored context checkpoint` events |
| `llama_slots_access` | llama-server access-log GET /slots done-request lines (HTTP 200/400/500 evidence) |
| `llama_prompt_io` | prompt_save/prompt_load line counts |
| `prefill_tokens` | llama-server prompt-eval prefill token total + line count |
| `slot_cache_inventory` | slot-cache/*.bin files: size, mtime, age |
| `llama_files_seen` | per-file breakdown of llama-derived metrics (see below) |
| `baseline_metrics` | rollup numbers used for incident validation and downstream analysis |

## Per-file llama breakdown (`llama_files_seen`)

llama-server logs carry **no timestamps**, so day attribution happens at the
file level:

| llama-server log file | Likely coverage |
|---|---|
| `llama-server.log-2026-08-27.gz` | 2026-08-26 (rotated at 2026-08-27 00:00) |
| `llama-server.log-2026-08-28.gz` | 2026-08-27 |
| `llama-server.log-2026-08-29` | 2026-08-28 |
| `llama-server.log` | current (live) |

Each file's stats: `created_checkpoints`, `restored_checkpoints`,
`slots_200/400/500`, `prompt_save`, `prompt_load`, `prefill_tokens`,
`prefill_lines`, `prompt_done_tokens`, `prompt_done_events`.

## Baseline metrics vs. incident claims (2026-08-26)

The incident description (LP-0MTAQNB7J0094X71) claimed, for 2026-08-26:

| Incident claim | Harness value (incident-day file) | Status |
|---|---|---|
| 2,954 checkpoints saved vs 145 restored (~5% restore) | `llama-server.log-2026-08-27.gz`: 3,191 created, 154 restored (4.83%) | ✓ within tolerance — incident was measured mid-day (22:12); full-day file totals slightly higher; **ratio matches** |
| 6,459 of ~69.6K GET /slots polls returned 500 (9.3%) | 6,865 of 73,003 (9.40%) | ✓ within tolerance — mid-day vs full-day window; **ratio matches** |
| 527 HTTP 400s on GET /slots | 527 | ✓ **exact match** |
| 42.7M prefill tokens/day | 46.1M `prompt_done` tokens (1,701 `prompt processing done` events) | ✓ within tolerance — incident measured mid-day (22:12); full-day file totals slightly higher; **same methodology** (`prompt processing done` n_tokens sum) |

The harness records multiple prefill measures so downstream analysis (F4) can
pick the correct one per question — see the next section.

## Prefill-token measures

- `prompt_done_tokens_total` / `prompt_processing_done` — llama-server
  `prompt processing done` events: the metric the 2026-08-26 incident used
  (46.1M on the incident-day file). This is the primary corroborating
  measure for incident reproduction.
- `prefill_token_total` — llama-server `prompt eval time` lines: executed
  prefill tokens on local dispatches (a subset of the above; each request's
  final eval, not the batched progress events).
- `proxy_estimated_tokens_total` / `routing_check_tokens_total` — proxy-side
  `estimated_tokens` on routing_skip / routing_check lines: tokens the proxy
  *estimated* for all routing evaluations (larger population, includes
  requests that were routed remote).

## Exit codes

- `0` — success
- `1` — log directory missing / analysis error