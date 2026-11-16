# Group 1 validation — prefill-deadline residual (certified ≥ 24 h window)

**Work item:** LP-0MW8CS7O7000YPDB (child of the group-1 validation LP-0MW7V0S6A006DKZ4)
**Window:** `2026-11-10 14:18:26` → `2026-11-11 15:18:26` UTC — **25.0 h**
**Dataset:** [`group1-prefill-deadline-residual.json`](./group1-prefill-deadline-residual.json)

## Window certification (AC1)

The single group-1 proxy restart was taken at **2026-11-10T14:18:26Z** (proxy PID
316897, deployed HEAD `01d89a19e`, /health all-healthy — evidence on
LP-0MW8COT80004L1SR). The window ran **25.0 h** restart-free:

- No `llama-server process died`, `self-heal attempt`, `watchdog router self-heal`,
  `TTS server exited` or `Config reload` line anywhere in the window
  (`grep -hE ... /var/log/llama-proxy/proxy.log*` → **0** matches).
- The analyser's `source.restarts_in_window` holds exactly **one** rotation line —
  the window's own boot rotation at `2026-11-10 14:18:27`
  (`Rotated previous proxy log aside: proxy.log.2026-11-10_141824 (pre-restart window preserved)`),
  which AC1 explicitly permits.
- Source corpus: 48 log files, 274 096 in-window lines, 943 duplicate lines dropped.

## Measurement (AC2)

Authoritative run, `--llama-log` supplied (so the run certifies on measured-stop /
measured prefill accounting, not `basis=legacy_rate_estimate`):

```bash
python3 proxy/scripts/analyse_prefill_deadline_residual.py \
  --start "2026-11-10 14:18:26" --end "2026-11-11 15:18:26" \
  --llama-log "/var/log/llama-proxy/llama-server*.log" \
  --json --output proxy/docs/validation/group1-prefill-deadline-residual.json
```

Backend prefill accounting: **3 566** completed backend prefill segments, **843**
abandoned prefill segments over the window.

## Post-acceptance evictions (25.0 h)

Total evictions after acceptance: **23** = **0.92/h**.

| Reason | Total | User | Audit | /h (all) | Distinct sessions |
|--------|------:|-----:|------:|---------:|------------------:|
| `measured_cache_deficit` | 14 | 14 | 0 | 0.56 | 11 |
| `prefill_deadline` | 4 | 4 | 0 | 0.16 | 3 |
| `no_progress_watchdog` | 4 | 4 | 0 | 0.16 | 3 |
| `lease_evicted` | 1 | 1 | 0 | 0.04 | 1 |

### User-class combined (the (b2) criterion)

`prefill_deadline` (user) + `measured_cache_deficit` (user)
= **4 + 14 = 18 events = 0.72/h** over 25.0 h.

| Window | User-class combined | Target |
|--------|--------------------:|--------|
| Baseline (2026-11-06 → 2026-11-08, 33.736 h) | 2.223/h | ≤ 0.5/h |
| **This window (group 1 as deployed)** | **0.72/h** | ≤ 0.5/h |

**Not met** against the ≤ 0.5/h target (0.72/h), but a **≈ 68 % reduction** from
the 2.223/h baseline. The met/not-met verdict with the full surviving-event
analysis is produced by the analysis child LP-0MW8CS862005HFDX (parent AC5); this
child records the certified window and the measurement only.

## Attribution (n = 4 `prefill_deadline` events)

| Attribution | Events |
|-------------|-------:|
| capacity | 2 |
| occupancy | 2 |
| predicted-warmth | 0 |

(Baseline attribution was capacity 22 / occupancy 9 / predicted-warmth 29, n = 60.)

`prefill_deadline` event distributions (n = 4):

| Metric | min | p50 | mean | max |
|--------|----:|----:|-----:|----:|
| `elapsed_s` | 180.1 | 181.35 | 186.05 | 201.4 |
| `estimated_tokens` | 39 892 | 53 246.5 | 56 071.75 | 77 902 |
| `eta_prefill_s` | 31.9 | 71.1 | 74.4 | 123.6 |
| `priced_cached_ratio` | 0.661 | 0.76 | 0.77 | 0.90 |

## Notes / assumptions

- Artifact location follows this child's AC3 verbatim (`proxy/docs/validation/`).
  The group-2 plan noted the older `docs/dev/prefill-deadline-residual-<variant>.{json,md}`
  convention; the dataset here is the group-1 authoritative run and is referenced
  by this path from the hand-off (LP-0MW8CS8NY0087XBV).
- `local_prefill_deadline_seconds` was not changed by this validation (out of scope).
