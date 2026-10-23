# Prefill-deadline recovery — live verification (2026-10-23)

Live, post-deploy verification of the fix chain for
[LP-0MVKN9IRP003XJH6](../worklog) (wedged prefill lease loop holding the local
dispatch gate ~4 h on 2026-10-23 02:00-06:00). The measurement item is
**LP-0MVKUKGER000TW2W** (child C7).

The verification answers one operational question: after a local dispatch is
cancelled by the prefill deadline (or stalls with no observable progress), does
the local gate recover in minutes and does local serving resume — instead of
re-dispatching the same giant prompt for hours?

## Deployment under test

| Item | Value |
|---|---|
| Code | `dev` @ `a4d2b233` (C5 lease/accounting) and earlier C1-C4, `a6…` config |
| Proxy restart | 2026-10-23 **20:08:53** (PID 1903525, detached) — the running process before this (00:29) predated every fix commit |
| Restart command | `cd proxy && bash scripts/start-proxy.sh --restart` |
| Config defaults | `local_prefill_deadline_seconds: 90`, `local_prefill_thrash_cooldown_seconds: 600`, `local_admission_guard_eta_percentile: 75` |
| Post-fix window measured | 20:08:53 → 21:03:06 (54 min) plus later spot checks |

## Result summary

| AC | Claim | Result |
|---|---|---|
| AC1 | post-fix report: no thrash hour and zero `local_lease_active` fallbacks | **fallbacks 0 in every post-fix hour**; hours with >=10 dispatch attempts are the deliberate verification stalls, whose llama-server prompt-eval/eval batches are present (so not the incident's zero-eval thrash shape). Full 24 h sample follows in the device window below |
| AC2 | a deliberately stalled prefill releases the gate in <=5 min | **MET — 0 s (streaming/deadline path) and 189 s (non-streaming/watchdog path)** |
| AC3 | local serving resumes after recovery | **MET** — local stream served at 22:04:25 (first byte 2.85 s, tokens delivered) |
| AC4 | evidence + regressions filed | this document + comment; one bounded-regression item filed (see below) |

## AC2 — gate release after a deliberately stalled prefill

### Streaming dispatch (primary deadline path)

A 45k-token novel prompt (`X-Session-ID: prefill-recovery-1792785079`,
`X-Local-Preference: local-only`, `stream: true`) was admitted, because the
admission guard is inert until a (model, bucket) window holds
`local_admission_guard_min_samples` observations after the restart
(`reason: insufficient_samples` from `/llama/local/reserve`, fail-open by
design). The prefill exceeded the 90 s deadline:

```text
2026-10-23 20:51:19,xxx - INFO - [local] POST http://127.0.0.1:8000/v1/chat/completions ... session_id=prefill-recovery-1792785079 ... slot=0
2026-10-23 20:52:58,342 - WARNING - prefill_deadline_exceeded session=prefill-recovery-1792785079 deadline=90s → cancelling local dispatch and detouring to the next provider
2026-10-23 20:52:58,342 - INFO  - prefill_thrash_cooldown_started session=prefill-recovery-1792785079 cooldown_s=600
2026-10-23 20:52:58,342 - INFO  - lease_released session=prefill-recovery-1792785079 reason=prefill_deadline endpoint=default
```

- **Deadline → lease release: 0 s** (identical timestamps) — before the fix the
  deadline cancellation left the record and marker behind and the same session
  re-dispatched for ~4 h.
- Client received the retryable `503` at **+99.1 s** (deadline + detour).
- `/llama/local/status` reported the gate free (`local_prefills_in_flight=0`,
  `available_slots>0`) at **+142.9 s**.
- The 600 s `/v1` cooldown was armed for the session, so a retry skips local
  (`routing_skip_local ... reason=prefill_thrash_cooldown`).

### Non-streaming dispatch (watchdog path)

A 55k-token prompt with `stream: false` (headers return before the prefill
completes, so the deadline's first-byte wait is not the bound) was held by a
genuinely progress-less prefill and released by the no-progress watchdog:

```text
2026-10-23 20:19:17,590 - INFO - [local] POST ... session_id=prefill-recovery-1792783157 ... slot=0
2026-10-23 20:22:26,978 - WARNING - lease_released session=prefill-recovery-1792783157 reason=no_progress_watchdog endpoint=default
```

- **Dispatch → release: 189 s** (<= 300 s bound). llama-server still completed
  the prompt evaluation (`prompt eval time = 435798.21 ms / 56795 tokens`) and
  generated its 8 tokens (`eval time = 256.31 ms / 8 tokens`) — the "GPU busy,
  nothing delivered" shape that the report now labels
  `prefill-only-no-decode`.

### Live production session (unplanned)

The verifier's own agent session hit the deadline repeatedly right after the
restart and every cancellation produced the same inline release + cooldown:

```text
2026-10-23 20:11:46,488 - WARNING - prefill_deadline_exceeded session=01a16a1c-... deadline=90s → cancelling local dispatch and detouring to the next provider
2026-10-23 20:11:46,488 - INFO  - prefill_thrash_cooldown_started session=01a16a1c-... cooldown_s=600
2026-10-23 20:11:46,488 - INFO  - lease_released session=01a16a1c-... reason=prefill_deadline endpoint=http://localhost:8080
... (20:13:25, 20:15:06, 20:43:24, 20:41:44, 20:38:22 — same 0 s shape)
2026-10-23 20:17:59,084 - INFO - routing_skip_local provider=local-qwen3 ... session=01a16a1c-... cooldown_remaining=427s reason=prefill_thrash_cooldown → skipping local, routing to next remote provider
```

The incident re-dispatched 41 local POSTs for one wedged session (35 inside
02:00-03:00); post-fix the same session is kept off local for 600 s after each
deadline, and the observed local POST count for the whole post-fix window was
14 (all deliberate or deadline-cancelled dispatches).

## AC1 — post-fix report figures

`analyze_proxy_usage.py` over the post-fix window (20:08:53 → 21:03:06),
hourly activity (`report.md` in the run's output directory):

| Hour | Local dispatch attempts | Local served | Lease denials | `reason=local_lease_active` fallbacks | State |
|---|---|---|---|---|---|
| 20:00 | 14 | 0 | 3 | **0** | prefill-only-no-decode+lease-blocked |
| 21:00 | 2 | 0 | 2 | **0** | prefill-only-no-decode+lease-blocked |

For contrast, the same report over the 24 h **pre-fix** window
(2026-10-22 20:04 → 2026-10-23 20:04) contains six thrash hours and 766
`local_lease_active` fallbacks:

| Hour | Attempts | Served | `local_lease_active` |
|---|---|---|---|
| 2026-10-22 23:00 | 166 | 0 | 26 |
| 2026-10-23 02:00 | 173 | 0 | 14 |
| 2026-10-23 03:00 | 82 | 0 | 6 |
| 2026-10-23 06:00 | 79 | 0 | 27 |
| 2026-10-23 10:00 | 46 | 0 | 21 |
| 2026-10-23 11:00 | 151 | 0 | 116 |

The post-fix hours with >=10 attempts are **not** the incident's thrash shape:
each dispatch was either this verification's deliberate stall or a large-prompt
dispatch cancelled at the deadline, and llama-server evaluated real batches in
every case (e.g. `prompt eval time = 435798.21 ms / 56795 tokens` +
`eval time = 256.31 ms / 8 tokens`; `prompt eval time = 65408.55 ms / 20626
tokens` + `eval time = 190.76 ms / 8 tokens`). A zero-eval thrash hour did not
occur, and no request fell back with `reason=local_lease_active`.

The literal "24 h post-fix window" cannot exist until 24 h after the 20:08:53
restart; the follow-up window item reruns the same measurement
(`python3 scripts/prefill_recovery_verification.py --measure --since … --until …`)
over the full window.

## AC3 — local serving resumes

After the deliberate stalls and the recovery window, a fresh small request
(`stream: true`, `X-Local-Preference: local-only`, `X-Session-ID:
ac3-serving-220422`) was admitted immediately and served locally with tokens
delivered:

```text
2026-10-23 22:04:22,699 - INFO - [local] POST ... session_id=ac3-serving-220422 ... slot=0
2026-10-23 22:04:25,544 - INFO - Stream started: provider=local model=Qwen3 session=ac3-serving-220422
2026-10-23 22:04:25,545 - INFO - dispatch_first_byte_ms=2845.6 dispatch_to_first_byte_ms=2845.4 cache_state=cold cached_ratio=0.000 queue_wait_ms=0.2 session=ac3-serving-220422 model=Qwen3 estimated_tokens=5
2026-10-23 22:04:25,702 - INFO - Stream finished: reason=length session=ac3-serving-220422 provider=local model=Qwen3
```

First byte in 2.85 s, `queue_wait_ms=0.2`, no deadline detour and no lease
denial — the gate is fully usable after recovery. `llama-server.log` also shows
continued local decode after each earlier recovery (`eval time = 256.31 ms / 8
tokens` for the stalled dispatch itself).

Earlier attempts in the same window (21:00-21:20) were refused with
`reason=local_lease_active` because the backend was saturated by the concurrent
slot-sweep benchmark and by the deliberately abandoned test prefills still
being processed by llama-server — it was a capacity condition, not a leak: the
same request succeeded the moment a slot was free, and the 22:04 stream shows
the clean steady state.

## Regression observed (filed)

A dispatch whose **client dies mid-prefill** retained its
`local_prefill_in_flight` marker/lease for ~5.5-6 minutes (dispatch 20:58:36 →
marker cleared ~21:04), refusing other sessions with
`local_dispatch_denied ... reason=local_lease_active` and
`local_owner_session_id=null`. Bounded (not the incident's ~4 h), but longer
than the 5-minute bound. Filed as **LP-0MVLFVY6K003VYEL** with a blocking edge
on the parent epic.

## Full-window follow-up — coverage audit and intact-window measurement

The 24 h follow-up (**LP-0MVLGNUDJ009NRX2**) needs a window the log set can
actually cover. Audited on 2026-10-23 22:30, `/var/log/llama-proxy` does not
cover the nominal window:

| File | First timestamp | Last timestamp |
|---|---|---|
| `proxy.log.2026-10-23_12` | 2026-10-23 12:29:16 | 2026-10-23 18:29:12 |
| *(no file covers this span)* | — | — |
| `proxy.log` (birth 18:29:18, truncated) | 2026-10-23 22:04:16 | live |

**3 h 35 m (18:29:12 → 22:04:16) is missing from every rotated and live file**,
including the fix go-live (20:08:53) and the whole C7 window quoted above. The
nominal `--since '2026-10-23 20:08:53'` window therefore cannot be measured in
full. The window is **re-anchored to the 22:04:14 proxy restart** (PID 1934147,
`dev` @ 60715de4 — a superset of C1-C5), so the full 24 h sample runs
**2026-10-23 22:04:14 → 2026-10-24 22:04:14**. The log loss and the concurrent
dual `uvicorn` processes that share `proxy.log` (PID 1923906 from 21:33 plus
PID 1934147, only the latter owning `:8000`) are filed as **LP-0MVLHMAWV008EKTI**
(critical, blocks the 24 h item).

Measurement over the intact window (22:04:14 → 22:29:00):

```json
"ac1": {"min_attempts": 10, "thrash_hours": [], "lease_active_fallbacks": 9,
        "hours_checked": 1, "verdict_met": false}
"ac2": {"pairs": 5, "unreleased_sessions": [], "max_release_seconds": 0.0,
        "verdict_met": true}
"ac3": {"local_finishes_after_last_release": 15, "verdict_met": true}
```

`analyze_proxy_usage.py` over 20:08:53 → 22:12:00 (only the 22:00 hour has
log data — the gap shows up as two `idle` hours):

| Hour | State | Busy | Requests | Local attempts | Local served | Lease denials | Fallbacks |
|---|---|---|---|---|---|---|---|
| 20:08-21:00 | idle (no log data — see gap) | 0s | 0 | 0 | 0 | 0 | 0 |
| 21:00-22:00 | idle (no log data — see gap) | 0s | 0 | 0 | 0 | 0 | 0 |
| 22:00-22:12 | serving+lease-blocked | 1m 45s | 69 | 4 | 4 | 12 | 60 |

What the intact window shows:

- **No thrash hour**: `thrash_hours: []`, and llama-server evaluated real
  batches throughout (`eval time = 2318.24 ms / 81 tokens`, …), unlike the
  incident's zero-eval hours.
- **Recovery still bounded**: 5 `prefill_deadline_exceeded` events, each
  releasing its lease in **0.0 s**, none unreleased.
- **Local serving works**: 20 local `Stream finished` events (15 after the
  last release).
- **`reason=local_lease_active` denials are NOT zero**: 9 in 25 min (3
  `Fallback triggered … reason=local_lease_active` + 6
  `contention_queue_fallback_after_queue … reason=local_lease_active`), 12
  `local_dispatch_denied` against `owner=<other herdr session> active=1`.
  One session (`herdr-1792789684-1790656-13656`) held the lease ~19 min with
  continuous `lease_extended_during_prefill progress=None alive=False
  in_flight=True` renewal while three others queued 60 s and fell back. The
  holder did decode (so this is monopoly/fairness, not the wedge), but the
  literal "zero `local_lease_active` fallbacks" AC is not satisfiable under
  4-6 concurrent agent sessions on a 2-slot pool. Filed as
  **LP-0MVLHMCN6003GAMD** with a decision (restate the AC to the wedge shape,
  or scope a lease-fairness fix).

Resume command for the full window once it exists (after 2026-10-24 22:04:14):

```bash
python3 scripts/prefill_recovery_verification.py --measure \
    --log-dir /var/log/llama-proxy \
    --since '2026-10-23 22:04:14' --until '2026-10-24 22:04:14' --json
python3 .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py \
    --start '2026-10-23 22:04:14' --end '2026-10-24 22:04:14' --json
```

## Limitations

1. **24 h sample.** The fix went live at 20:08:53 on 2026-10-23, so the
   post-fix window measurable in-session is under an hour; the full-window
   check is tracked separately (see the coverage audit above — the window is
   re-anchored to 22:04:14 because the 18:29→22:04 log span was lost).
2. **Log truncation.** A concurrent benchmark restarted the proxy repeatedly
   from ~21:08, truncating `proxy.log`; the post-fix figures above were captured
   from the analysis run at 21:03, the log excerpts quoted at the time, and a
   snapshot of the stream/lease lines at 22:05. The 18:29:12 → 22:04:16 span
   is absent from every file — see LP-0MVLHMAWV008EKTI.
3. **Guard calibration.** After a restart the admission guard is inert until
   `local_admission_guard_min_samples` observations exist per (model, bucket);
   the deliberately stalled prompts were therefore admitted and exercised the
   deadline/cooldown/watchdog paths rather than the admission refusal
   (`prefill_eta_exceeds_deadline`).
4. **Concurrent workload.** From 21:08 a slot-sweep benchmark held the local
   slots and repeatedly restarted the backend, which delayed the AC3
   client-delivered completion until 22:04 and made several probe admissions
   observe a genuinely saturated pool.

## Reproduce

```bash
# Post-fix window measurement (AC1 from the report figures + AC2 pairing)
python3 scripts/prefill_recovery_verification.py --measure \
    --log-dir /var/log/llama-proxy \
    --since '2026-10-23 20:08:53' --until '<window end>' --json

# Deliberate stalled prefill (streaming: deadline path)
python3 scripts/prefill_recovery_verification.py --stall-test --stream \
    --prompt-tokens 45000

# Deliberate stalled prefill (non-streaming: watchdog path)
python3 scripts/prefill_recovery_verification.py --stall-test \
    --prompt-tokens 55000
```

Tests: `python3 -m pytest tests/test_prefill_recovery_verification.py -q`.
