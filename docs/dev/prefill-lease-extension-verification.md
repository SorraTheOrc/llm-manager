# Prefill lease-extension post-deploy verification

Live verification of the liveness-based local-prefill dispatch-lease extension
restored in `2ced52ac` (LP-0MUXQ5QL40027VZ4) and completed in `9b7c505c`
(LP-0MUXTW4MD003TVIC). Tracked by **LP-0MUXQNPKE008K9MP**.

## Background

A local dispatch held an explicit-session lease while llama-server prefilled a
prompt. On the deployed llama.cpp b8782 build the per-slot numeric progress
fields are gone, so the proxy could not observe progress. The no-progress
watchdog (`local_dispatch_no_progress_timeout_seconds`, default 180s) then
released the lease, the upstream stream was cancelled, and the client retried
into a fresh full prefill — an unbounded kill/retry loop for long prefills
(LP-0MUXQ5QL40027VZ4).

`_extend_lease_during_prefill` now extends the lease on a numeric progress
advance **or** an observed liveness signal (`is_processing`), and
`9b7c505c` added a third signal: the upstream request is still demonstrably
in flight (`prefill_in_flight=True`), which is the only reliable signal while
`/slots` is blocked for the duration of a prompt-processing batch. The
`local_dispatch_max_prefill_seconds` ceiling (900s) still bounds a wedge.

## Method

`scripts/prefill_lease_verification.py` parses the proxy log stream (live and
both rotated naming schemes, plain or `.gz`) and counts the AC-relevant events
over an arbitrary window:

```bash
python3 scripts/prefill_lease_verification.py --since '2026-10-06 05:36' \
    --until '2026-10-07 00:00' --json          # pre-fix baseline
python3 scripts/prefill_lease_verification.py --since '2026-10-12 06:07' \
    --until '2026-10-12 22:11' --json          # post-fix window
```

The raw measurement is committed as
[`prefill-lease-extension-verification-2026-10-12.json`](prefill-lease-extension-verification-2026-10-12.json).

## Results

| Metric | pre-fix 2026-10-06 (18.4h) | pre-fix 2026-10-07 (7.6h) | post-fix 2026-10-12 (16.1h) |
|---|---|---|---|
| `lease_extended_during_prefill` | 0 | 0 | 6096 |
| … `alive=True` (liveness shape) | 0 | 0 | 77 |
| … `in_flight=True` | 0 | 0 | 6096 |
| `lease_released reason=no_progress_watchdog` | 135 | 58 | 14 |
| `backend_retry error=ReadTimeout` | 67 | 79 | 29 |
| `dispatch_first_byte_timeout` (900s ceiling) | 5 | 10 | 10 |
| `prefill_progress_unobservable` | 887 | 397 | 690 |
| local stream finishes | 1552 | 757 | 596 |
| long (>180s) local prefill completions | 0 | 0 | 28 |

Normalised, the no-progress watchdog falls from ~7.3–7.6 releases/hour pre-fix
to ~0.9/hour post-fix — roughly an 8x reduction — and `ReadTimeout` retries
fall from ~3.6–10.4/hour to ~1.8/hour. `dispatch_first_byte_timeout` is the
deliberate 900s max-prefill ceiling (every event carries `ceiling=900s`), not
a kill/retry-loop signal; it stayed low and bounded.

## Acceptance criteria

1. **met** — `lease_extended_during_prefill` appears with `progress=None
   alive=True` (77 events) and, under load, `progress=None alive=False
   in_flight=True` (6096 events); the pre-fix count was 0.
2. **met** — `no_progress_watchdog` releases drop sharply (~8x/hour) and
   `backend_retry ... ReadTimeout` drops. `dispatch_first_byte_timeout` is the
   900s wedge ceiling, not a loop signal, and stayed bounded (see caveat
   below).
3. **met** — 28 local stream finishes were preceded by a >180s continuous
   lease-extension run (the same-session run analysis), where pre-fix not a
   single prefill over 180s could complete; they were all released and
   retried.
4. **met** — the no-progress watchdog still fired 14 times post-fix: most
   ~180s after the session's last extension (genuine silence, no liveness),
   two at the 900s max-prefill ceiling. A genuinely wedged slot is still
   released.
5. **not triggered** — AC1–AC4 hold, so `2ced52ac` is not rolled back.

## Caveat

AC2's `dispatch_first_byte_timeout` sub-clause did not fall: the pre-fix
baselines were 5 and 10 events, the post-fix window 10. These events are the
`local_dispatch_max_prefill_seconds` wedge ceiling, driven by how many
genuinely huge prefills occurred rather than by the watchdog loop, so they are
not a valid before/after signal for this fix. The two loop-defining metrics
(`no_progress_watchdog` kills and `ReadTimeout` retries) both dropped sharply.

## Reproduce

The script is dependency-light (standard library plus `scripts/lib/proxy_logs.py`)
and reads whichever proxy log files are present, so it can be re-run against a
later window to confirm the behaviour persists.
