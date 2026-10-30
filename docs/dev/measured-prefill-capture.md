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

Consumers of the store (added later): the admission ladder's
`measured_last_dispatch` rung (LP-0MVT2PY3J00077W6) reads the session's last
dispatch `total` to report a numeric reusable prefix in
`routing_resident_prefix` when the deployed build reports no per-slot
`context_tokens`; F4's early abort reads `cache`/`total` for the same dispatch.

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

`proxy/tests/test_measured_cache_abort_wiring.py` — the router wiring over a
synthetic b8782 stream: a fake-warm first event releases the dispatch with
`reason=measured_cache_deficit` in seconds (never the 120 s deadline), the
client is then served by a remote provider, a genuinely warm stream is
byte-for-byte unchanged, the rollback key restores pre-change behaviour, and
missing/malformed/erroring measurements fail open. Also pins the parent AC2
"first progress event" clause (LP-0MVR7K6CP000LW2F): only the first
measurement decides, so a later event can neither block an abort nor trigger
one after bytes have reached the client.

## Regression proof and window replay (F5, LP-0MVR7IXAI002QXXD)

The no-regression guarantees are pinned against the parent's own reference
window (2026-10-26 09:06 → 2026-10-27 09:06):

- `proxy/tests/fixtures/measured_prefill_window_2026_10_26.jsonl` — the 120
  `local_eviction_after_acceptance` events (29 "fake-warm": predicted reuse
  ≥ 0.99) plus 457 measured `n_past` / `slot.prompt.tokens.size()`
  observations from the same llama-server build.
- `proxy/scripts/capture_measured_prefill_window.py` — regenerates the
  fixture from `/var/log/llama-proxy` (deterministic; llm-server lines carry
  no timestamps, so the fixture ships the measured *evidence population*
  rather than a per-event join).
- `proxy/tests/measured_prefill_window_replay.py` — replays every event
  through the real decision in two measured cache classes derived from that
  evidence: `reprefill` (smallest measured `n_past` in the prompt-size band —
  the class the fix targets) and `best_case` (largest measured reuse in the
  band — the working warm population). `enabled=False` replays the documented
  rollback.
- `proxy/tests/test_measured_prefill_regression.py` — AC1 (the 178 measured
  large warm tasks never abort, the warm 56k stream completes locally
  byte-for-byte, the residency ladder decides identically with the feature on
  and off), AC2 (29/29 fake-warm events classify as
  `reason=measured_cache_deficit` and release in seconds through the real
  `proxy_to_local`), AC3 (the rollback replays the pre-change window exactly).

```bash
# Replay report (both measured classes + the rollback)
python3 proxy/tests/measured_prefill_window_replay.py
# Regenerate the fixture from the reference window's logs
python3 proxy/scripts/capture_measured_prefill_window.py
```

The replay measured 29/29 fake-warm events abortable at their
admission-priced `new_tokens`, reclaiming 402,201 of the window's 554,792
modelled wasted prefill tokens; with the knob set to `0` every event keeps
today's 120 s deadline (0 abortable).

## Early-abort peek invariants (LP-0MVR7IXAI002QXXD)

The abort decision must be made **before** the response is committed, so F4
peeks the SSE stream. With `return_progress` injected, the SSE headers arrive
at prefill *start*, which moves the pre-first-byte wait into that peek. The
peek therefore keeps the pre-first-byte monitor's duties:

- the dispatch lease is still extended on prefill progress (and a confirmed
  client disconnect still releases it) while the peek waits;
- the wait is bounded by the prefill ceiling **and** the stream loop's
  runtime budget; a completely hung upstream fails open to the loop's
  heartbeat + synthesised-termination behaviour instead of a 503;
- the peek uses its **own** `SSEEventReframer`; the consumed raw reads are
  replayed through a *fresh* reframer in the stream loop, so a chunk that
  carries a complete event plus the start of the next one can never be
  double-fed.

## Post-deploy measurement (parent AC4/AC5)

Runs only after F4 is live on `dev` **and** the proxy has been restarted onto
that build for ≥ 24 h (the restart, not the commit, starts the window):

```bash
# 1. confirm the build is live and the knob is on
grep -c 'prefill_progress_measured' /var/log/llama-proxy/proxy.log
python3 - <<'PY'
import yaml; c = yaml.safe_load(open('proxy/config.yaml'))
print(c['server'].get('local_measured_cache_abort_enabled'))
PY

# 2. the post-deploy window report
python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
  --hours 24 --output-dir ~/proxy-usage-reports/measured-prefill-postdeploy --json

# 3. reason counts: measured_cache_deficit vs the residual prefill_deadline
#    (eviction records only: the same reason also appears on the
#    measured_cache_abort and lease_released lines of the same event)
grep 'local_eviction_after_acceptance' /var/log/llama-proxy/proxy.log \
  | grep -o 'reason=measured_cache_deficit' | wc -l
grep 'local_eviction_after_acceptance' /var/log/llama-proxy/proxy.log \
  | grep -o 'reason=prefill_deadline' | wc -l
```

Compare against the reference window's baseline: 120 deadline evictions
(29 fake-warm) and 695,253 admission-priced new tokens discarded. A healthy
result is `measured_cache_deficit` replacing the fake-warm share of
`prefill_deadline` while the large warm completions (≥ 20k, first byte < 10 s)
hold their count.

### Measured result — first post-deploy window

Window `2026-10-29T14:08:10Z → 2026-10-30T14:08:10Z` (24 h), entirely after the
post-F4 restart at `2026-10-29T00:10:19Z` (dev tip `52085f09`, which contains
F4 `1908f431`). The first `prefill_progress_measured` line ever recorded is
`2026-10-29T00:10:30Z`, and every log file covering earlier dates contains
zero, so the restart — not the commit — is what started capture. Measured
2026-10-30 (LP-0MVS03KCV005BTGE); the reference column is the same parser run
over the 2026-10-26/27 baseline window, which reproduces the parent's figures
exactly (114 evictions, 648,008 tokens, 252 large prefills, 177 < 10 s).

| Metric (24 h) | Reference (pre-F4) | Post-deploy |
|---|---|---|
| `prefill_progress_measured` lines | 0 | 1,098 (345 distinct dispatches ≥ 20k) |
| `reason=prefill_deadline` evictions (priced new tokens) | 114 (648,008) | 60 (292,466, **−54.9 %**) |
| `reason=measured_cache_deficit` evictions | — | 248 (858,903 priced tokens released early) |
| Measured-deficit `cache/total` (p50 / max) | — | 0.324 / 0.748 — no warm abort |
| Early-release latency (p50 / p90 / max) | — | 3.8 s / 64.2 s / 118 s (51.6 % < 5 s) |
| Large local attempts / completions (≥ 20k) | 334 / 252 | 381 / 170 |
| Large warm completions (≥ 20k, first byte < 10 s) | 118 of 149 | 48 of 59 (81.4 % vs 79.2 %) |

Reading it:

- **The new reason replaces the fake-warm share of the deadline class.** The
  baseline-comparable wasted-token metric (admission-priced `new_tokens` of
  `prefill_deadline` evictions) falls 54.9 %. Counting the early releases'
  partial prefill too (Σ `min(measured_deficit, elapsed × 43.1 tok/s)` =
  208,470), all-in discarded prefill compute is 500,936 vs 648,008 (−22.7 %).
  239 of the 248 releases needed more than the whole 120 s budget at the
  43.1 tok/s prompt-eval p10, i.e. they were doomed to the deadline before F4.
- **No warm dispatch is aborted.** The measured `cache/total` of every release
  is ≤ 0.748 (the 0.25 deficit-ratio floor makes that structural); none is
  near the ≥ 0.99 warm band. Warm large latencies are unchanged to better
  (service p50 2.67 s → 3.13 s, p90 26.8 s → 15.0 s) and the share reaching a
  first byte under 10 s is flat (79.2 % → 81.4 %).
- **The absolute warm-large count falls for compositional reasons, not
  regression.** The predicted-warm label is over-optimistic: measurement shows
  only 88 of 345 large dispatches (25.5 %) are genuinely warm, while the
  pre-F4 prediction labelled 149 of 252 completions (59 %) warm. Large local
  attempts rose 334 → 381 (+14 %) on 34 % more local streams, and 211 of those
  attempts were released early (144 measured deficit, 54 deadline, 12
  `no_progress_watchdog`, 1 lease) versus 82 evictions before — so doomed
  dispatches leave the completed population instead of being counted in it.
- **The early abort is not always fast.** The tail is real: 26 of 248 releases
  took ≥ 60 s because the *first* `prompt_progress` event itself arrives late
  for very large prompts (that subset's prompt p50 is 67,849 tokens). It is
  still a bounded improvement over the 120 s deadline, but the "seconds, not
  120 s" claim only holds for the ~52 % of releases under 5 s.

Other eviction reasons in the window: `no_progress_watchdog` 60 and
`lease_evicted` 7. Reproduction: build the window with `zcat -f
/var/log/llama-proxy/proxy.log* | awk '$0 >= "2026-10-29 14:08:10" && $0 <
"2026-10-30 14:08:10"'`, then count eviction records (not raw reason
occurrences, which triple-count each event) with `grep
local_eviction_after_acceptance | grep -c 'reason=...'`; the
CSV/report artefacts are in `~/proxy-usage-reports/measured-prefill-postdeploy/`.
