# Retry-outcome join: what each post-acceptance abort cost the caller

Work item: **LP-0MWM0CPQN009SHJR** (epic LP-0MWM0B7IQ008W2FP, child 3).
Emission: `proxy/proxy/retry_outcome.py`. Analysis:
`proxy/scripts/analyse_prefill_deadline_residual.py` (`retry_join`).

The group-2 verdict established that **user stall cost rose ×3.81**
(60.572 → 230.556 s/h), and V1 measured **88.567 s/h** against a **6.06 s/h**
gate. A stall-weighted figure says how long the client *waited*; it does not
say what the client *got*. A 15 s abort followed by an immediate successful
retry is cheap to the user; the same abort leaving the request unanswered is
expensive. This change adds the join that answers it — a **measurement-only**
change (no admission decision, abort policy or threshold moves, AC4).

## Join semantics

An abort (`local_eviction_after_acceptance`) is joined to the caller's **next
dispatch of the same session** on two keys:

- `session` — the client/session id the eviction line already carried;
- `prompt_fp` — a stable **prompt identity**: the SHA-256 of the request's
  `messages` list serialised as canonical JSON (the same digest the
  single-flight dedup coordinator uses,
  `proxy.router_helpers._request_dedup_hash`).

The join is **cross-turn**: a retry is usually a new client request (a new
`turn`), but the provider chain can also re-dispatch inside the same turn
(a local refusal that escalates remotely, or a second local admit after the
abort). The analyser orders by timestamp and, when a turn is present, collapses
that turn's several outcome lines to a single **terminal outcome** with the
precedence `remote > local_accepted > local_refused`.

### "Same prompt" tolerance

Two dispatches are the *same prompt* when their `prompt_fp` values are equal.
Because the fingerprint hashes only the `messages` list, **sampling and
generation parameters are ignored** (a retry that changes only `temperature`,
`max_tokens`, `top_p`, … still joins); a changed message list (new turn
content) changes the fingerprint and is reported as a different prompt.

## Field names

Appended to the existing `local_eviction_after_acceptance` line (append-only —
no pre-existing key moves, so a deployed positional parser keeps its prefix):

| Field | Meaning |
|---|---|
| `prompt_fp` | Stable prompt identity (SHA-256 of `messages`); `unknown` when the record carried none (a pre-change dispatch). |
| `turn` | Per-request correlation id stamped on the record; `unknown` when absent. Extra provenance, not the join key. |

New line, one per local admission/refusal or remote serve:

```
local_dispatch_outcome session=S prompt_fp=F outcome=O endpoint=E [dispatch_token=T] [<caller evidence>] [turn=X]
```

`outcome` is one of:

| Outcome | Meaning |
|---|---|
| `local_accepted` | A local dispatch was admitted for this request (emitted at dispatch start, before first byte). |
| `local_refused` | Local was refused on this attempt (gate, lease/occupancy, or slot availability). |
| `remote` | The request was served by a remote provider. |

The analyser additionally reports `none` — *no next dispatch observed in the
window* — which is a join result, never an emitted outcome.

The retry's **first byte** is joined from two pre-existing lines, both already
carrying `session` and `turn`:

- local: `dispatch_first_byte_ms=… session=… turn=…`;
- remote: `ttft_seconds=… session=… provider=… turn=…`.

## The analyser block

`analyse_prefill_deadline_residual.py` publishes `report["retry_join"]`:

```
total / by_reason[reason]:
  aborts
  retried              {count, share}
  outcome              {local_accepted, local_refused, remote, none}
  same_prompt          {count, share_of_retried, unknown}
  abort_to_first_byte_s{count, p50, p95, min, max, mean}
  completed            {count, share}
  never_completed      {count, share}   # share that never completed in the window
rows[]                 # one per abort: ts, session, reason, elapsed_s,
                       # prompt_fp, turn, outcome, retried, same_prompt,
                       # completed, abort_to_first_byte_s, retry_turn,
                       # retry_prompt_fp, first_byte_provider
```

`retry_join.tolerance` and `retry_join.caveat` are published in the payload so a
reader never has to guess the join rules. Run it like the rest of the residual
tool:

```bash
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
    --start "2026-11-27 10:18:00" --end "2026-11-29 10:18:00" --json \
  | jq '.retry_join.by_reason'
```

The text (non-`--json`) summary prints a `Retry outcome:` line and a per-reason
breakdown beside the stall-cost line.

## Caveats

- **No correlation across a client restart or across sessions.** The session id
  is half the join key, so a client that restarts with a new session id (or an
  anonymous request without a stable `X-Session-Id`) cannot be joined to its
  retry; the abort is reported as `none`.
- **Window-bounded.** A retry that lands after the window close is reported as
  never-completed, not as escaped. Choose the window to exceed the largest
  observed abort→retry gap.
- **First byte only when measurable.** `abort_to_first_byte_s` is populated only
  when the retry produced a local or remote first byte inside the window;
  otherwise the abort is in `never_completed`.
- **Terminal-outcome collapse.** When one turn emits several outcome lines, only
  its terminal outcome is attributed (precedence `remote > local_accepted >
  local_refused`); the per-attempt detail is not preserved in the aggregate.
- **Log-only.** Nothing here reads or changes proxy configuration, so the block
  cannot alter an admission decision or a deadline.

## Verifying no behaviour change (AC4)

The emission is best-effort and wrapped (`record_dispatch_outcome` never raises;
both `proxy_to_local`/`proxy_to_remote` call sites swallow failures). The
dispatch record only gains a `prompt_fp` field; no decision reads it. The
analysis reads only log lines. The changed-proxy test suite
(`proxy/tests`, 8 543 tests) and the new
`proxy/tests/test_retry_outcome.py` pin this contract.
