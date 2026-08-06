# Error extraction summary

- Window: 2026-08-03 00:00:00 → 2026-08-04 00:00:00
- Total error events: **355**

## Counts by error type

| Error type | Count |
|---|---|
| backend_retry | 93 |
| slot_save_error | 17 |
| stream_error | 6 |
| stream_finish_error | 127 |
| upstream_http_error | 112 |

## Stream finished: reason=error split (provider/model)

| Provider | Model | Count |
|---|---|---|
| opencode-go | deepseek-v4-flash | 93 |
| opencode | deepseek-v4-flash-free | 28 |
| local | Qwen3 | 6 |

## Headline assertions

- **PASSED**

## Artifacts

- `errors.csv` — one row per error event
- `counts.csv` / `counts.json` — aggregated counts
- `evidence.txt` — raw evidence lines
