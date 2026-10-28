# Measured prefill capture from llama-server `prompt_progress`

**Work item:** LP-0MVR7DAFF006HBRX (parent LP-0MVQIT9QY007QU27, Option D).
Contract module: `proxy/proxy/prompt_progress.py` (F1, LP-0MVR7CR34000AG12).
Capture module: `proxy/proxy/measured_prefill.py`.

## Why measure at all

Every admission-time warmth signal is a prediction. In the 2026-10-26 window
all 5,456 `routing_resident_prefix` decisions carried `n_past_actual=none`, so
`resolve_resident_prefix` reported `reuse_ratio=1.0` for whatever session owned
the slot and a clobbered owner was admitted as if fully cached — then
re-prefilled the whole prompt until the 120 s prefill deadline released it
(29 of 116 evictions were ~56k-token prompts priced at 17–182 new tokens).

llama-server b8782 (build `e97492369`) exposes the ground truth: when the
request carries `return_progress: true`, it sends `prompt_progress` events
inside the existing SSE stream. The first event is emitted right after
`n_past` is assigned, so its `cache` value is the **actual reused prefix**:

```
{"total": 56000, "cache": 200, "processed": 0, "time_ms": 0}
```

`total - cache` is therefore the work llama-server is really doing: 55,800
tokens for a "fake-warm" prompt, versus ~200 for a genuinely warm one.

## What this ships

| Piece | Where | Notes |
|---|---|---|
| Upstream injection | `router.proxy_to_local` | `return_progress: true` added to a **copy** of the parsed body for local streaming dispatches only |
| Capture | `router.proxy_to_local` reframed SSE loop | every `prompt_progress` event parsed; latest `{total, cache, processed, time_ms}` recorded per dispatch |
| Dispatch exposure | `measured_prefill.DISPATCH_RECORD_KEY` on the local dispatch-lease record + `get_measured_prefill(model, session)` | exposes `measured_cached_ratio` (`cache / total`), `measured_new_tokens` (`total - cache`) and the admission-priced `priced_new_tokens` |
| Observability | `prefill_progress_measured` INFO line | `session`, `model`, `total`, `cache`, `processed`, `time_ms`, `measured_cached_ratio`, `measured_new_tokens`, `priced_new_tokens` — one line per dispatch (the first event) |
| Rollback | `server: local_measured_cache_abort_enabled` | the same key the early-abort wiring (F4) uses; `0`/`false`/`off` restores pre-change behaviour end to end (no injection, no capture, no early release) |

Decisions are **unchanged** by this item: the early-abort that consumes the
measurement is F4 (LP-0MVR7IN0O000DPXN).

## Invariants

- **Client-visible stream is byte-for-byte unchanged.** `prompt_progress`
  chunks are proxy-internal (they only exist because the proxy asked for
  them), so they are captured and withheld. A client that sets
  `return_progress` itself keeps its own progress events and is not
  re-injected.
- **No leakage into recorded payloads.** The key lives only in the upstream
  body copy; the recorded client/proxy payloads (`body_json`) never carry it.
- **First-byte path unchanged.** A read that carries only progress events is
  excluded from the client-data accounting, so it can never be mistaken for
  the first content byte, nor reset the lease/idle budgets different from a
  keepalive. A progress event split across two reads is carried as a fragment
  so the tail is not counted as data either.
- **Fail-open.** A missing event, a malformed event, a wrong-typed field or an
  exception anywhere in capture degrades to "no measurement"; nothing escapes
  the stream loop and the dispatch behaves exactly as before.

## Verifying after deploy

```bash
# one attributable line per measured dispatch
grep 'prefill_progress_measured' /var/log/llama-proxy/proxy.log

# measured-vs-predicted error for the window
python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
  --hours 24 --output-dir ~/proxy-usage-reports/measured-prefill
```

`measured_cached_ratio` near 1.0 with a small `priced_new_tokens` gap is the
healthy warm case; `measured_new_tokens` far above `priced_new_tokens` on a
large `total` is the fake-warm signature the early abort targets.

## Tests

`proxy/tests/test_measured_prefill_capture.py` — injection (local only, no
leakage), capture over a synthetic b8782 stream, dispatch-record exposure,
byte-for-byte client stream, client-requested progress passthrough, fail-open
(missing/malformed/exception), the log grammar, the rollback key, and the
remote path forwarding the client body verbatim.

`proxy/tests/test_prompt_progress_contract.py` — the parse contract itself
(parent AC1, LP-0MVR7K63C006O9ZL): all four fields extracted from the four real
b8782 event shapes (initial 0%, in progress, fully warm, fake warm), absent or
wrong-typed fields degrading to explicit `None`, derived
`measured_cached_ratio` / `measured_new_tokens`, survival of arbitrary
`SSEEventReframer` byte splits, and `PromptProgressCapture` latest-wins
semantics.
