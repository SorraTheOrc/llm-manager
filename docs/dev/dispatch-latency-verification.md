# AC1/AC2 dispatch-latency verification (LP-0MV6BR3PT001WBEI)

Reproducible measurement for the two remaining acceptance criteria of
[LP-0MU1S617L002Q9PR] — *Fix the proxy: large-context
sessions stall and time out instead of progressing*. This item is a
measurement/decision item, not a functional change: the structural fixes are
already deployed (AC1 ceiling bound in LP-0MV6BR3HI00957PF, AC2 metric
segmentation in LP-0MV6CFMU9005WU2E); this page defines what is measured, how,
and the current evidence.

## Tool

`scripts/dispatch_latency_verification.py` scans the proxy log stream
(`/var/log/llama-proxy`, both live and rotated files under either naming
scheme, plain or `.gz` — via `scripts/lib/proxy_logs.py`) over an arbitrary
window and reports:

**AC1 — 900 s client-deadline surfaces:**

- `dispatch_first_byte_timeout` events, split by ceiling
  (`ceiling=900s` — the legacy default that raced the client's 900 s
  queue-timeout deadline exactly; `ceiling=840s` — the post-fix default that
  answers strictly first, LP-0MV6BR3HI00957PF);
- `Chain exhausted for model ... holding <N>s` chain-hold starts and the
  cumulative hold seconds;
- client queue-timeout relays (upstream error text mentioning the 900 s /
  queue-timeout verdict being passed back to the client).

**AC2 — re-scoped first-byte latency (LP-0MV6CFMU9005WU2E):**

- local-dispatch `dispatch_to_first_byte_ms` (service time once the proxy
  slot lock is held) and `dispatch_first_byte_ms` (total), split
  `cache_state=warm|cold` (routing cached-ratio ≥ 0.5 ⇒ warm, fail-open to
  cold);
- `queue_wait_ms` (proxy-side dispatch→slot-acquired) p50/p90/p99/max.

The verdict evalates the re-scoped targets and emits `PASS`/`WARN`:

| Target | Bound |
|--------|-------|
| AC1 legacy-ceiling events in window | 0 |
| AC2 warm-cache service p99 | < 60 s |
| queue-wait p99 (context; contention cap is 60 s default / 120 s cheap / 45 s fast) | ≤ 120 s |

The AC1 verdict is a precondition only until the 24 h window has fully elapsed
(`--window-closed` asserts this on the final run).

```
$ python3 scripts/dispatch_latency_verification.py \
    --since '2026-10-14 15:48' --until '2026-10-15 15:48' \
    --window-closed --json > ac1_24h_window.json
```

Exit codes: 0 = PASS, 2 = WARN, 1 = error/invalid input.

## Understanding the metrics

- `dispatch_first_byte_timeout ... ceiling=840s` is the fix working as
  designed: a local request that produced no first byte within 840 s is
  aborted server-side with a retryable 503 + Retry-After, so the client never
  waits out its 900 s deadline. A `ceiling=900s` event is the AC1 race signal
  (pre-fix).
- The AC2 segmentation separates the *proxy-side* single-slot queue wait
  (`queue_wait_ms`, the contention queue / slot_guard hold) from service time.
  It does **not** capture llama-server-internal slot serialisation: when the
  one local slot is generating a long stream, a warm local request can sit
  inside llama-server behind the in-flight generation and still report a
  minutes-long `dispatch_to_first_byte_ms` with a tiny proxy `queue_wait_ms`.
  The initial post-deploy data shows exactly this (see Evidence).

## AC1 24 h window definition

Per the producer decision (recorded on this work item, 2026-10-13):

1. The AC1 fix (`chain_hold_total_seconds=720`,
   `local_dispatch_first_byte_timeout_seconds=840`, commit `4e8970e0` /
   LP-0MV6BR3HI00957PF) is deployed — first observed live
   `ceiling=840s` event `2026-10-13 12:05:55` local.
2. The 24 h post-change window starts at the next timed proxy restart once the
   fix is deployed. The running proxy process restarted
   `2026-10-13 15:48:19` local (`Starting LLama Proxy Server` at
   `15:48:26`), so the candidate window is
   **2026-10-13 15:48 → 2026-10-14 15:48 (local)**.
3. When the window elapses, run the tool with `--window-closed`; the verdict
   requires **zero** `ceiling=900s` first-byte timeouts and zero client
   queue-timeout relays across the whole window.
4. Any request that holds a local dispatch at or beyond the client deadline
   must also be absent: the 840 s server ceiling and 720 s chain-hold budget
   guarantee the proxy always responds before the client, which the tool's
   ceiling split makes checkable.

## Evidence — post-deploy window (2026-10-13)

Snapshot taken 2026-10-13 16:00 local with the tool; the full artifact is
committed as `docs/dev/dispatch-latency-post-deploy-2026-10-13.json`:

- **Window:** `2026-10-13 09:00` → now (post-AC2-metric-deploy; first
  `cache_state=` line at `09:02:37`).
- **AC1:** 4 `dispatch_first_byte_timeout` events, **all `ceiling=840s`**
  (12:05:55, 12:06:39, 13:43:00, 13:43:05), zero at the legacy 900 s ceiling;
  zero chain-hold starts; zero client queue-timeout relays. The 840 s ceiling
  bound is live and no request has raced the client deadline in the observed
  window.
- **AC2:** 86 warm / 100 cold local dispatches.
  - warm service p50 = 4.3 s, p90 = 88.1 s, **p99 = 307.6 s**
  - cold service p50 = 49.9 s, p90 = 186.7 s, p99 = 723.0 s
  - proxy queue wait p50 = 0.9 ms, p99 = 2.6 ms (max 3.4 ms)
- **AC2 finding:** the re-scoped warm-cache p99 target (< 60 s) is **not yet
  demonstrated**. The warm tail (13 of 86 samples > 60 s, up to 395.6 s)
  coincides with single-slot saturation: e.g. session
  `01a1350b-097b-7465-aabb-465b12315f85` was routed warm
  (`cached_ratio=1.00`, 119 new tokens at 12:52:20) but its first byte landed
  at 12:58:55 (395.6 s) while the local slot was occupied by another session;
  the proxy-side `queue_wait_ms` (1.6 ms) cannot see llama-server's internal
  queue. The AC2 re-scope excludes proxy contention, not server-internal slot
  serialisation.

## Open item

The AC2 warm-cache p99 needs one of (producer decision, following the initial
post-deploy data):

1. a further refinement so the warm metric excludes (or separately reports)
   llama-server-internal slot serialisation, or
2. an operator-approved target/scope adjustment for the single-slot profile,
   or
3. a performance change that actually bounds server-internal first-byte time
   on the 1-slot backend.

Until one of these lands, the AC1 24 h window should still complete (it is
independent of AC2) and be measured with `--window-closed`.

## Reproducibility

```bash
# Current window summary
python3 scripts/dispatch_latency_verification.py
# Machine-readable artifact
python3 scripts/dispatch_latency_verification.py \
  --since '2026-10-13 09:00:00' --json > docs/dev/dispatch-latency-post-deploy-2026-10-13.json
# Tests (synthetic corpus, no production data needed)
python3 -m pytest tests/test_dispatch_latency_verification.py -q
```
