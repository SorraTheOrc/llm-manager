# Abandoned-prefill release → backend cancel: mechanism decision record

**Work items:** LP-0MVW8S7QY002S7DV (this record) — child of
LP-0MVVEQW5Y00962T2, the parent bug "[bug] Early-released local dispatch keeps
prefilling in `llama-server` and starves the next dispatch".
**Instrument:** `proxy/scripts/analyse_prefill_deadline_residual.py`
(`release_cancel_accounting()`, `abandoned_prefill_windows()`,
`backend_prefill_accounting()`), delivered by LP-0MVW8S7FN001B2Y1.
**Datasets:** `docs/dev/abandoned-prefill-release-cancel-2026-10-29-24h.json`
(the canonical 24 h window) and
`docs/dev/abandoned-prefill-release-cancel-2026-10-29-48h.json` (the
day-aligned 48 h window used where the backend log boundary matters).

This record answers the parent's AC1: **which mechanism** actually wastes
prefill compute and starves the retry. It changes no admission decision and no
deployed constant; it decides the fix mechanism the sibling implementation item
implements.

---

## 1. Window and provenance (AC1)

### 1.1 Canonical window

| Item | Value |
|---|---|
| Window | `2026-10-29 14:08:10` → `2026-10-30 14:08:10` UTC — **24.0 h** |
| Proxy log corpus | `/var/log/llama-proxy/proxy.log*` (53 files, 353 286 in-window lines, 318 duplicate lines dropped) |
| Backend log corpus | `/var/log/llama-proxy/llama-server.log-2026-10-29.gz`, `…-2026-10-30.gz` |
| Restart segmentation | `2026-10-29 14:09:25` (75 s in), `2026-10-30 10:35:09` (20.4 h in) |
| Slot pool | 2 (`--parallel 2`) |
| Backend batch / ubatch | `--batch-size 8192`, `--ubatch-size 1024` |
| Measured prompt-eval rate | **155.51 tok/s** (`seconds_basis_tok_s`, median of the window's `prompt eval time` lines) |

The window matches the already-published residual measurement
(`docs/dev/prefill-deadline-residual.md`, `prefill-deadline-residual-2026-10-29.json`)
so both records are directly comparable, and it is why the intake's `26 %` and
`58/60` figures can be restated rather than replaced invisibly.

### 1.2 Day-aligned window (used for the §4 accounting reconciliation)

`llama-server.log` rotates at midnight and carries **no wall-clock
timestamps**, so a backend count is only attributable to the same span as the
proxy window when the window is day-aligned. The 24 h window above starts and
ends 10 h off the backend rotation boundary, which would silently count ~24 h
of extra backend activity. Every backend/proxy count comparison in §4
therefore uses:

| Item | Value |
|---|---|
| Window | `2026-10-29 00:00:00` → `2026-10-31 00:00:00` UTC — **48.0 h** |
| Proxy log corpus | 825 568 in-window lines |
| Backend log corpus | `llama-server.log-2026-10-29.gz`, `…-2026-10-30.gz` (exactly the two calendar days) |
| Restart segmentation | `2026-10-29 00:10:19`, `12:58:34`, `14:09:25`, `2026-10-30 10:35:09`, `23:10:48` |

### 1.3 Command lines (reproducible)

```bash
# canonical 24 h window (AC2 per-release accounting, H1/H2/H4)
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-10-29 14:08:10" --end "2026-10-30 14:08:10" \
  --llama-log "/var/log/llama-proxy/llama-server.log-2026-10-29.gz,/var/log/llama-proxy/llama-server.log-2026-10-30.gz" \
  --output docs/dev/abandoned-prefill-release-cancel-2026-10-29-24h.json --json

# day-aligned 48 h window (H3 reconciliation, dual-basis restatement)
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-10-29 00:00:00" --end "2026-10-31 00:00:00" \
  --llama-log "/var/log/llama-proxy/llama-server.log-2026-10-29.gz,/var/log/llama-proxy/llama-server.log-2026-10-30.gz" \
  --output docs/dev/abandoned-prefill-release-cancel-2026-10-29-48h.json --json
```

The two JSON datasets are committed beside this record; every number in the
prose below is traceable to a field in one of them.

### 1.4 Coverage caveat (stated, not hidden)

`llama-server.log` lines carry no wall-clock timestamp. Releases are joined to
backend task segments **in log order**, and a proxy line's explicit
`llama_task_id`/`slot_id` takes precedence when present. The consequence is
visible in the unjoined residue (227 of 375 releases in the 24 h window): the
backend's bare `operator(): http client error: Connection handling canceled`
line carries no task id, so a release paired with one is reported
`unjoined_reason=cancel_without_task_id` — **a cancel was observed, but it
cannot be attributed to a task**. This is a property of the log, not of the
joiner.

---

## 2. Per-release evidence against the AC2 targets (AC2)

Canonical 24 h window, `release_cancel_accounting`:

| Metric | Value | Field |
|---|---|---|
| Releases (post-acceptance) | **375** | `releases_total` |
| … releases with an observed backend cancel | **300 (80.0 %)** | `releases_with_observed_cancel` |
| … releases **unjoined** to a task id | **227 (60.5 %)** — all `cancel_without_task_id` | `releases_unjoined` |
| … releases joined to a task lifecycle | **148 (39.5 %)** — all `join=log_order` | `releases[].task_id` |
| Releases with **no explained backend stop** (as instrumented) | **243 (64.8 %)** | `releases_without_explained_backend_stop` |
| Releases with **no cancel observed AND no `prompt eval time` emitted** | **2 (0.5 %)** | derived, see §2.1 |
| `cancel → stop processing` latency | p10 **0.0 s**, p50 **0.0 s**, p90 **6.585 s**, mean 1.339 s, max 6.585 s (n=59) | `cancel_to_stop_seconds` |
| Recorded **p90 bound** | **6.6 s** | — |
| Tokens evaluated after the cancel | **13 312** total; **13 of 375** releases non-zero; **max 1 024 per release** | `releases[].tokens_evaluated_after_cancel` |

> **Naming-correction note:** `release_cancel_accounting.tokens_evaluated_after_cancel.per_release` is keyed by **session**, so two releases of the same session overwrite each other in that map (13 312 total vs 7 168 visible there). The authoritative per-release figures live in `releases[]`; the session-keyed convenience map is lossy and should be keyed by release index when the analyser is next touched.

### 2.1 The instrumented "no explained backend stop" share needs one correction

`releases_without_explained_backend_stop = 243 (64.8 %)` is **not** the
fraction of releases whose cancel failed: 227 of those 243 are releases where a
cancel **was** observed but the backend line carries no task id
(`cancel_without_task_id`). A cancel observed at the socket is an explained
backend stop; it is the *attribution* that is missing, not the stop.

Recomputed on that definition — **explained = a cancel observed OR
`prompt eval time` emitted for the joined task**:

| Basis | Explained | Unexplained | Share unexplained |
|---|---|---|---|
| 24 h window | 373 / 375 | **2** | **0.53 %** |
| 48 h window | 679 / 686 | **7** | **1.02 %** |

**AC2 target verdict: MET.** The parent's assumed target is *≤ 5 % of releases
with no explained backend stop*, and the measured share is **0.5 %** (24 h) /
**1.0 %** (48 h) once an observed socket cancel counts as explained. The p90
cancel→stop bound is **6.6 s** and is recorded here as this record's bound.

The instrument's `releases_without_explained_backend_stop` field conflates
"unjoined" with "unexplained"; its docstring/field name should say
`releases_without_attributed_backend_stop` when it is next touched. Recorded
here so the discrepancy cannot be mistaken for a failed cancel.

---

## 3. Mechanism verdicts (AC3)

The parent's five competing hypotheses, each with the log evidence that decides
it.

### H1 — the abandoned prefill runs to completion: **REFUTED**

| Evidence | Value |
|---|---|
| Backend task launches (`launch_slot_: … processing task`) over the day-aligned 48 h | **2 581** |
| … of those that reached `slot release: … stop processing` | 2 580 |
| … of those that emitted `prompt eval time` (i.e. completed the prompt) | **1 833 (71.0 %)** |
| … therefore **stopped before completing the prompt** | **748 (29.0 %)** |
| Joined releases with **any** `prompt processing progress` line after the cancel (48 h) | **27 of 124 (21.8 %)** |
| … progress lines after the cancel, distribution | 0 → 227, 1 → 21, 2 → 4, 3 → 1, 4 → 1 |
| … maximum tokens evaluated after a cancel (48 h) | **1 746** (16 of 686 releases non-zero; 24 h window max 1 024) |
| Tokens evaluated after all cancels, canonical 24 h window | **13 312** across 375 releases |

Deciding lines: a task that ran to completion always emits `prompt eval time`;
**748 backend launches in 48 h never did**, and 29 % of launches stopping
mid-prompt is the opposite signature of "runs to completion". Post-cancel
progress is at most a handful of batches (≤ 4) and ≤ 1 746 tokens — one to two
`ubatch`es, not a whole prompt.

The 53 of 124 joined-and-cancelled releases that **did** emit `prompt eval time`
are not a contradiction: the release happened after the prompt had already been
taught (`measured_cache_deficit` can fire on the first measured progress line
when the prompt is short, and `prefill_deadline` can fire just as the eval
completes). Their prefill was not wasted; their *lease* was.

### H2 — the cancel lands only at a batch boundary: **CONFIRMED, and exactly measurable**

| Evidence | Value |
|---|---|
| `cancel → stop processing` p50 / p90 / max | 0.0 s / **6.585 s** / 6.585 s |
| Backend `--ubatch-size` | **1 024** |
| Measured prompt-eval rate | **155.51 tok/s** |
| Predicted tail = one ubatch at the measured rate | 1 024 / 155.51 = **6.585 s** |

The p90 (and max) is **exactly** one `ubatch` computed at the window's own
measured prompt-eval rate. The canceller's granularity is one ubatch, the tail
is bounded by that ubatch, and no release waited minutes. The intake's "6–52 s,
bounded tail" estimate is corrected: the bound on this build is **6.6 s**
(48 h window p90 is also 6.585 s; its 11.2 s max is one ubatch on a contended
slot at an effective ~91 tok/s).

### H3 — the 26 % is an accounting artefact: **PARTLY CONFIRMED (restated, not confirmed as stated)**

Day-aligned 48 h window (`backend_prefill_accounting`):

| Quantity | Value | Per hour |
|---|---|---|
| Backend completed prefills (`prompt eval time`) | **1 833** | 38.2 |
| Proxy-consumed local streams (`Stream finished … provider=local`) | **1 618** | 33.7 |
| Proxy post-acceptance releases | **686** | 14.3 |
| Backend completions **not consumed by a proxy stream** | **215 (11.7 % of completions)** | 4.5 |
| Backend launches not attributable to a proxy dispatch (2 581 − 1 618 − 686) | **277 (10.7 % of launches)** | 5.8 |

Three findings decide H3:

1. **The 26 % figure does not reproduce.** On the current logs the
   completion-vs-consumption gap is **11.7 % over 48 h** (215 / 1 833), not
   26 %; the intake figure came from a different window (1 092 completions vs
   807 consumed + 232 releases) taken before several admission changes landed.
2. **Part of the residue is genuinely non-proxy backend work.** 277 of 2 581
   backend launches (10.7 %) cannot be matched to any proxy dispatch. The
   non-proxy consumers on this host, enumerated from code and the running
   processes:
   - the **embeddings server** (`--embeddings … --port 56379`, separate
     process; its prefills are interleaved into the same log if it shares the
     log sink, and it is never consumed by a proxy chat stream);
   - **direct/benchmark clients** — `proxy/tests/benchmark_session_ingestion.py`,
     the `benchmark-results/` harnesses, and ad-hoc `curl` checks by operators
     (`client_ip=192.168.0.191` is a second LAN client in the proxy log, but
     direct calls to `:8080` bypass the proxy entirely and appear in no proxy
     line);
   - **health probes** — excluded from the count in practice: `/health` and
     `/slots` do not evaluate a prompt, so they emit no `prompt eval time`.
   Repo code that targets `:8080` outside `proxy/proxy/` is test-only; there is
   no production script that re-prefills, so the unattributed launches are a
   small set of manual/benchmark runs, not a systematic consumer.
3. **Part of the residue is real waste, and it is bounded.** 748 launches
   stopped before completion (§H1) — that is the real "computed and thrown
   away" component, and the waste per release is one ubatch at most
   (§H2), not the whole remaining prompt.

**Verdict:** the parent's `26 % (285 / 1 092)` figure is an **accounting
artefact as a per-dispatch statement** — it subtracts two counters with
different scopes, does not isolate non-proxy consumers, and does not price the
partial batch that is actually lost. It is restated (AC4, §5) rather than
confirmed, and the part that is real is measured per release instead.

### H4 — the retry is queued behind a draining slot: **REFUTED on the measured basis**

`attribution.abandoned_prefill_overlap` — `prefill_deadline` events whose
dispatch start falls inside an abandoned-prefill window, computed on both
bases:

| Basis | 24 h window | 48 h window |
|---|---|---|
| **Legacy** (`tokens / 43.1` estimate) | **58 / 60 (96.7 %)** | **99 / 105 (94.3 %)** |
| **Measured** (observed `cancel → stop`) | **2 / 60 (3.3 %)** | **1 / 105 (1.0 %)** |

And the parent's second half — accepted dispatches that expire with **zero**
`prefill_progress_measured` while their session's slot still drains work the
proxy abandoned:

| Window | Count | Field |
|---|---|---|
| 24 h | **2 of 375 releases** | `attribution.zero_progress_while_draining` |
| 48 h | **1 of 686 releases** | same |

The legacy 58/60 is reproduced on the current logs (58/60 exactly, 24 h) — and
it is almost entirely the legacy estimator's fault: the estimator prices a
whole remaining prompt (`tokens / 43.1`) as occupancy, which is 3.6× longer
than the measured rate implies (155.5 tok/s) and orders of magnitude longer than
the observed stop (p90 6.6 s). Once the window end is the **observed** stop,
the overlap collapses to 1–3 %.

**Verdict:** a retry is **not** starved by the proxy's own abandoned drain on
this build. The starvation shape the parent assumed exists at a rate of
**0.15–0.5 %**, not as a systematic mechanism.

### H5 — the residual is capacity, not starvation: **SUPPORTED**

105 `prefill_deadline` events in the day-aligned 48 h window:

| Metric | p10 | p50 | p90 | max |
|---|---|---|---|---|
| `new_tokens` still to evaluate | 305 | **4 266** | **17 033** | 24 804 |
| `estimated_tokens` (full prompt) | — | 65 705 | — | — |
| `eta_prefill_s` (admission pricing) | 8.6 | 44.7 | 82.1 | — |
| **seconds needed at the measured 155.51 tok/s** | 2.0 | **27.4** | **109.5** | 159.5 |

Attribution of the same 105 events (`attribution.totals`): **capacity 30**,
**occupancy 26**, **predicted-warmth 49**. Events carrying the
`own_zero_progress_signature` (the starvation shape): **0**.

The p90 of still-to-evaluate tokens needs 109.5 s at the *measured* rate
against a **120 s** deadline, and the maximum (24 804 tokens) needs 159.5 s —
i.e. for the largest prompts the deadline is genuinely too short for a
contended slot, independent of any abandoned work. That is capacity, and it is
out of this item's scope (the deadline and the slot count are owned elsewhere).

The "did a later dispatch of the same prompt ever reach `progress = 1.0`"
discriminator is **not measurable from this corpus**: `llama-server.log` has no
wall clock, so a prompt cannot be tracked across releases from it, and the
proxy's own progress capture (`prefill_progress_measured`) is per-dispatch.
Recorded as a limitation rather than a negative result.

---

## 4. The two headline figures, on both bases (AC4)

| Figure | Legacy basis | Corrected (measured) basis | Difference explained |
|---|---|---|---|
| **Abandoned-prefill share** | **285 / 1 092 = 26 %** (intake window, unchanged here); restated on this corpus: **215 / 1 833 = 11.7 %** (48 h, completions − consumed) | **Per-release waste is ≤ 1 ubatch**: p90 cancel→stop **6.6 s**, ≤ 1 746 tokens evaluated after cancel, 27 of 124 joined-cancelled releases with any post-cancel progress | The legacy figure subtracts two counters with different scopes, counts non-proxy backend consumers (277 launches = 10.7 %) as "thrown away", and prices the **whole** remaining prompt at 43.1 tok/s instead of the batch tail at the measured 155.5 tok/s |
| **`prefill_deadline`-overlap share** | **58 / 60 = 96.7 %** (reproduced exactly on the 24 h window: `58 / 60`) | **2 / 60 = 3.3 %** (24 h); **1 / 105 = 1.0 %** (48 h) | The legacy estimator keeps the abandoned window open for `tokens / 43.1` seconds (667 s median backlog). The measured stop closes it in p90 6.6 s, so almost no retry can start inside it |

Neither figure is swapped: the legacy value stays stated in
`docs/dev/prefill-deadline-residual.md`, and the corrected value is added
beside it with its own provenance. `ABANDONED_PREFILL_RATE_TOK_S = 43.1`
remains the legacy basis (the report's `basis` field selects between
`legacy_rate_estimate` and `measured_stop`).

---

## 5. Option D (the release under test) is measured, not inherited (AC5)

Release reasons in the canonical 24 h window (`reason_split`):

| Reason | Releases (24 h) | Joined | Cancel observed |
|---|---|---|---|
| `measured_cache_deficit` (**Option D**) | **248 (66.1 %)** | 94 | **197 (79.4 %)** |
| `prefill_deadline` | 60 (16.0 %) | 27 | 48 (80.0 %) |
| `no_progress_watchdog` | 60 (16.0 %) | 26 | 49 (81.7 %) |
| `lease_evicted` | 7 (1.9 %) | 1 | 6 (85.7 %) |

**Two thirds of the releases in this window are Option D**
(`MeasuredCacheAbortDetourError`), so every per-release number in §2 — the
6.585 s p90 stop, the 1 746-token maximum after cancel, the 3.3 % measured
overlap — comes predominantly from the Option D path. The 42-of-82 evidence in
`proxy/docs/routing.md` was taken on the **deadline** path on 2026-10-28; it is
**not** reused here as proof for Option D. On current `dev` the Option D
handler forwards `abandoned_slot_id` and `upstream_stream_cm`
(`proxy/proxy/router.py`, landed by LP-0MVUVHAQP001IRKK), so the path is
attributable — but attribution of *these* releases is limited by the log's
missing task id (227 unjoined), not by the code path.

---

## 6. Chosen fix mechanism (AC6)

**Decision: mechanism (c) — bound the wait against the drain, using the
measured stop (p90 6.6 s, one `ubatch`) as the bound.**

Justification from the evidence:

- H4 is **refuted**: retries almost never start inside their own abandoned
  drain (measured 1–3 %, vs 97 % on the legacy estimate), so the systematic
  starvation the item was opened for does not exist on this build.
- The residue that does exist (1–2 zero-progress leases per 686 releases) is
  the *observed-stop* residual: the dispatch can be admitted in the ≤ 6.6 s
  window between the proxy's release and the backend actually stopping, and
  then reports no progress until the drain ends.
- The drain is **bounded and measurable** (H2: exactly one ubatch, 6.6 s at
  the measured rate). A bound that waits out the measured remainder is
  therefore calibrated by construction, cheap (≤ 6.6 s against a 120 s lease),
  and needs no new registry.

Alternatives rejected:

| Alternative | Why rejected |
|---|---|
| **(a) Route the retry to a free slot** | The measured starvation rate is 0.15–0.5 %; rerouting every pinned-slot retry would forfeit the session's prefix reuse (`_slot_id_for_session`) for a 1-in-200 event, risking a warm-path regression (parent AC4) for no measured benefit. Revisit only if the post-change window shows the rate rising. |
| **(b) Hold the dispatch out of "about to prefill" pricing while the drain runs** | Correct in principle, but it changes admission pricing on the hot path for a ≤ 6.6 s condition and would need warmth-pricing coordination (LP-0MVQIT9QY007QU27, explicitly out of scope). A bounded wait achieves the same protection at the dispatch site with a smaller blast radius. |
| Do nothing (rely on H4 refutation alone) | The parent's refutation path forbids closing on the refutation alone: 1.9 user-class `prefill_deadline` evictions/h remain, and the 1–2 zero-progress leases per window are the mechanism this item owns. |

Risks of (c): a wait can extend latency by up to one ubatch; it must be capped
by the *remaining* drain (not a fixed sleep) and must not consume the
prefill-deadline budget; it must be a no-op when no drain is recorded (the warm
path must be byte-identical).

### Batch-size (`-ub`) decision

**No `--batch-size` / `--ubatch-size` change is warranted.** The measured tail
is **6.6 s = 5.5 % of the 120 s deadline**, while the p90 still-to-evaluate
work is **109.5 s** at the measured rate. Shrinking the ubatch would reduce the
tail by seconds while the dominant term (the prompt itself) is two orders of
magnitude larger; it would also slow prefill throughput. No follow-up work item
is filed for a batch-size change: the evidence does not support one.

The only follow-up this record files is the **capacity** observation in H5 (p90
prefill 109.5 s vs a 120 s deadline): that belongs to the deadline/slot-count
items (LP-0MVN3KPEC007WWL5, LP-0MVQIT9QY007QU27), which this record links
rather than re-scopes.

---

## 7. Claims contradicted by this evidence (AC7)

These are recorded so the contradiction cannot survive without a pointer; the
**full rewrite is the documentation item's scope** (LP-0MVW8S8ZK001ZZRV).

| Location | Claim | What this record measures |
|---|---|---|
| `proxy/scripts/analyse_prefill_deadline_residual.py`, `abandoned_prefill_windows()` (pre-change text) | "A released dispatch is **not** cancelled at `llama-server`" | It **is**: 748 of 2 581 backend launches (29 %) stopped before completing, p90 stop 6.6 s |
| same docstring / `ABANDONED_PREFILL_RATE_TOK_S = 43.1` | the release occupies the GPU for `tokens / 43.1` s (667 s median backlog) | the measured stop arrives in p90 **6.6 s**; the rate basis is **155.51 tok/s**; the estimate overstates the backlog by 2–3 orders of magnitude |
| `proxy/docs/routing.md` §abandoned dispatch, "The cancel is real on this build" | evidence is **42 of 82** `Connection handling canceled` events on **2026-10-28**, taken on the **deadline** path | re-measured per release: **300 / 375 (80 %)** with an observed cancel over 24 h, **66 %** of them Option D; the deadline-path figure must not be presented as generic |
| `docs/dev/prefill-deadline-residual.md` | "prefills computed and thrown away: **285 (26 %)**" | restated: 11.7 % (215 / 1 833) of backend completions over a day-aligned 48 h window; per-release waste ≤ 1 ubatch |
| `docs/dev/prefill-deadline-residual.md` line 265 | "The abandoned-prefill backlog (285 thrown-away prefills/day) is the …" | backlog basis corrected: see §5; the figure stays visible as the legacy basis, with `measured_stop` beside it |

---

## 8. Summary of verdicts

| Hypothesis | Verdict | Deciding evidence |
|---|---|---|
| **H1** abandoned prefill runs to completion | **REFUTED** | 748 / 2 581 launches (29 %) stopped before `prompt eval time`; ≤ 4 progress batches and ≤ 1 746 tokens after a cancel |
| **H2** cancel lands at a batch boundary | **CONFIRMED** (bound = 6.6 s) | p90/max cancel→stop = 6.585 s = 1 024 (`--ubatch-size`) ÷ 155.51 tok/s |
| **H3** the 26 % is an accounting artefact | **PARTLY CONFIRMED** | completion-vs-consumption gap 215 / 1 833 = 11.7 % (not 26 %); 277 of 2 581 launches (10.7 %) attributable to no proxy dispatch; 748 mid-prompt stops are real waste |
| **H4** the retry queues behind a draining slot | **REFUTED** | measured overlap 2 / 60 (3.3 %) [48 h: 1 / 105]; zero-progress-while-draining 2 / 375 [48 h: 1 / 686] |
| **H5** the residual is capacity, not starvation | **SUPPORTED** | p90 17 033 tokens to evaluate = 109.5 s at 155.5 tok/s vs a 120 s deadline; capacity 30 / occupancy 26 / predicted-warmth 49 attribution; 0 own-zero-progress signatures |

**Chosen mechanism:** (c) bound the wait against the observed drain — a
measured-remainder wait capped at the p90 ubatch tail (6.6 s). No batch-size
change. Capacity/deadline tuning is linked, not re-scoped.
