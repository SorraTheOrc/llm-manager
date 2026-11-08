# Expected-reuse prefix credit for routing `cached_ratio`

**Work item:** LP-0MVI2XOTY008Q4UV (discovered from LP-0MVI2VRT2000V1AP)

## The defect

The routing `cached_ratio` (used by the cold-cache gate, the economic bypass
and the prefill-ETA admission guard) is written from the **previous** local
response's `usage.prompt_tokens_details.cached_tokens`
(`provider.update_cached_ratio`). A session's first request is genuinely
cache-cold — nothing of it is in the KV cache yet — so that first response
reports `cached_tokens=0` and the proxy records `cached_ratio = 0.0` against
the session. The next request reads that 0.0 and is costed as **fully**
cache-cold, even though the first turn's tokens are now resident in the slot.

Observed on `audit-WL-0MVFRRKKN00544PQ-phase2_child_5-6e22ef57`
(2026-10-21 12:24–12:25):

| turn | estimated tokens | measured | `routing_check` |
|---|---|---|---|
| 1 | 2,067 | genuinely cold (first request) | `cached_ratio=0.00 cache_state=cold` |
| 2 | 5,221 | 2,067 tokens actually reusable | `cached_ratio=0.00 cache_state=cold` |

The label was pessimistic, not just conservative: turn 2's prefill took
6.6 s for 5,221 tokens (790 tok/s) versus 402 tok/s on the genuinely cold
turn 1 — consistent with llama.cpp reusing turn 1's prefix while the proxy
credited nothing.

The slot-snapshot sidecar (`slot_cache_ratio.py`) already solves the same
problem for **persisted** snapshots; but snapshot saves are rare in practice
(5 in a day's log), so in the common case the zero-first-turn usage path is
the only writer.

## The fix

`provider.record_prompt_prefix()` stores, after each local dispatch that
reports real usage, the measured `prompt_tokens` together with a fingerprint
of the request's message list (reusing `slot_cache_ratio.message_fingerprints`).

`provider.expected_cached_ratio()` returns the ratio used for routing:

```
measured = _get_cached_ratio(model, session)              # previous turn
if recorded prefix is a prefix of the incoming message list:
    credit = min(1.0, margin × min(prev_prompt_tokens, estimate) / estimate)
    return max(measured, credit)                          # measured is a floor
return measured                                           # today's behaviour
```

- **Prefix match** uses `slot_cache_ratio.prefix_matches` — the same rule the
  snapshot sidecar applies, so a changed system prompt or a history truncated
  from the front earns nothing.
- **Safety margin** `server.cached_prefix_credit_margin` defaults to `0.9`
  (matching the sidecar's `session_slot_ratio_safety_margin`): llama.cpp may
  invalidate its own checkpoints on this hybrid-attention architecture, so the
  full recorded prompt is never credited. `0.0` disables the credit and
  restores the pre-fix behaviour exactly.
- **Measured ratio is a floor**, never lowered — a warm measured/restored
  ratio still wins.
- **Fail-open and bounded**: no usage data → no record (unchanged behaviour);
  invalid inputs are ignored; the record dict is capped like
  `_last_cached_ratio`; `clear_cached_ratio_for_session()` drops the record
  with the ratio so a discarded session cannot keep its credit.

Wired at the three routing decision points:

| Site | Change |
|---|---|
| `_should_skip_local` (cold-cache gate / economic bypass) | uses `expected_cached_ratio` |
| `routing_check` (the `cached_ratio=` log line) | uses `expected_cached_ratio` |
| `_prefill_admission_guard` (prefill-ETA guard) | new `messages` parameter; uses `expected_cached_ratio` |

## Effect on the observed case

Turn 2 changes from `new_tokens=5221 ratio=0.00` to
`new_tokens≈3361 ratio≈0.36` — truthfully partial. Because
`_WARM_CACHE_RATIO_THRESHOLD` is `0.5`, that turn still *labels* as `cold`
(a 40 % credit on a prompt that grew 2.5× cannot cross the boundary); the
label flips on turn 3 once turn 2's full prompt is resident. The number that
matters for routing and the ETA guard — `new_tokens` — is correct
immediately.

## Large-append rule

**Work item:** LP-0MW81DLWY003XOZ2 (merged from LP-0MW0KZUUS002LICZ AC9).

The prefix credit above is only as good as its assumption: that the prompt grew
by a small turn on top of the recorded prefix. A single message can append a
large block of genuinely new work, which the measured ratio cannot see — it
describes the **previous** response's hit fraction.

Measured 2026-11-03 08:41:54: `routing_check estimated_tokens=18516
new_tokens=173 cached_ratio=0.99 cache_state=warm`, then llama-server prefilled
13,002 tokens; `measured_cache_abort` recorded `measured_new_tokens=12936
priced_new_tokens=173 deficit_ratio=0.659` and the turn took **53.2 s** to first
byte. The append that broke the estimate was a single tool-role message of
**51,945 bytes** (a whole `SKILL.md` pasted into history).

`provider.expected_cached_ratio()` now measures the content appended after the
recorded prefix (`slot_cache_ratio.appended_content_bytes`, the same role +
content fields the fingerprint digests) and, when it exceeds
`server.local_large_append_bytes_threshold` bytes, returns `0.0`:

```
if recorded prefix matches incoming:
    if appended_content_bytes(messages, fingerprints) > threshold:
        return 0.0                                  # full prefill
    credit = min(1.0, margin × min(prev_prompt_tokens, estimate) / estimate)
    return max(measured, credit)
return measured                                     # unchanged
```

Consequences and limits, all deliberate:

- **The cold threshold then governs.** With `cached_ratio = 0`, `new_tokens` is
  the whole estimate, so a prompt above
  `local_large_context_cold_cache_threshold` (38,000) falls through the chain as
  `large_context_bypass` instead of being admitted on a ratio that was never
  about the appended content.
- **Only the append counts.** A large *resident* prefix plus a small new turn
  keeps the measured ratio; the threshold is not a prompt-size cap.
- **No recorded prefix → unchanged.** A ratio restored from the slot-snapshot
  sidecar has no known append boundary, so the measured ratio is used as before
  rather than treating the whole prompt as an append (that would re-price every
  large prompt of a restarted warm session cold — a behaviour change this rule
  does not make).
- **Default 32 KiB** (≈8k tokens): the incident's 51,945-byte append is caught,
  an ordinary turn's few-hundred-byte append is not.
- **Sentinel `0` disables the rule** and restores the previous price
  byte-for-byte; a missing or malformed value keeps the default (a typo must
  not silently disable the rule).
- **Threaded through the router.** `_should_skip_local` takes `server_config`
  and passes it to `expected_cached_ratio`, so the predicted-ratio price and the
  prefix-credit margin read the same `server:` section as the rest of the
  routing pipeline.

Tests: `proxy/tests/test_large_append_full_prefill.py` — the 51,945-byte tool
append prices `cached_ratio = 0` and falls under the cold threshold, a small
append keeps the measured ratio, a resident prefix is not counted as an append,
threshold configurability (flat and nested), the `0`/negative sentinel, the
malformed-value fallback and the size helper's prefix arithmetic.

## Rollback

Set `server.cached_prefix_credit_margin: 0.0` in `proxy/config.yaml`. No
code change needed; the credit becomes zero and `expected_cached_ratio`
returns the measured ratio, i.e. the previous behaviour.

## Tests

`proxy/tests/test_cached_ratio_prefix_bootstrap.py` — 13 cases: cold-start
second turn earns credit (AC1), brand-new session stays cold (AC2), changed
head and truncated history earn nothing (AC3), the wired prefill-ETA guard's
ETA drops below the budget (AC4), measured/restored ratios are preserved,
clearing the session forgets the prefix, margin scaling and the 0.0 rollback
(AC5), and invalid records are ignored.

## Where the prediction still matters — and where it no longer does

**Work item:** LP-0MVN1IQTF0037WTG.

`expected_cached_ratio()` is a *prediction*: it is written from the previous
local response's usage (above), from the persisted slot sidecar, or from the
recorded prefix fingerprint. Under slot contention it is structurally wrong —
the 2026-10-24 window measured `cached_ratio ≈ 0.99` predicted for requests that
then prefilled from `n_past=738`, and 27 sessions cycled through 2 slots.

Since LP-0MVN1IQTF0037WTG the **resident-prefix gate and the cold-token pricing
of large prompts no longer read this prediction**: they price from the *measured*
reusable prefix in the cached `/slots` snapshot
(`proxy.residency.get_residency_view()` → `n_past_actual`), which is what makes
the divergence case decidable at all (task 92383: predicted 0.99, measured
1.5 % → refused; task 92422: measured 100 % → admitted).

The prediction is still used, deliberately, in two places:

1. **Fail-open fallback.** When there is no fresh snapshot *and* the session owns
   no slot (`residency_source=unknown`), admission degrades to the predicted
   ratio rather than denying local on absent data.
2. **Everything below the budget.** Prompts at or under
   `local_large_context_cold_cache_threshold` (and small contexts generally) are
   routed local as before, as is the cheap-mode economic recovery, the
   capacity-admission path and the prefill-ETA guard's ETA input.

So the prefix-credit machinery documented above remains the mechanism that
keeps *sub-threshold* multi-turn sessions cheap, while the residency gate
decides the large-prompt band from measurement. The knob that disables the new
gate (`server.local_resident_prefix_min_ratio: 0`) restores the predicted-ratio
decision everywhere — the documented rollback for the residency work, exactly
as `cached_prefix_credit_margin: 0.0` is the rollback for the credit itself.

**Do not add a synchronous `/slots` call to make the prediction "more accurate".**
`/slots` is held for >15 s during prompt processing
(`LP-0MUXTW4MD003TVIC`); measurement comes from the ~1 s cached snapshot.
