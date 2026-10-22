# Slot-count re-evaluation: keep N=2 pending a quiet-window N=3 sweep (LP-0MVGCQ3QN009HRE8)

**Item:** Re-evaluate `session_slot_pool_size` against measured GPU busy-time and contention history (LP-0MVGCQ3QN009HRE8).
**Decision:** **KEEP `session_slot_pool_size: 2`** (production unchanged) until the controlled N=3 sweep in LP-0MVGPEP3O000RKKD lands. The evidence below shows N=3 would absorb more concurrent demand, but the measured N=3 cost/benefit could not be established on the shared proxy; the mitigations N=2 depends on are enumerated in [Decision and mitigations](#decision-and-mitigations-ac4).
**Date:** 2026-10-20
**Method harness:** `scripts/slot_history_extract.py` (this item) + `proxy/benchmarks/slot_benchmark.py` (prior verdict).

---

## TL;DR

- **The historical record now spans 76 report-days and four slot eras** (3/3 → 1/3 → 2/3 → 2/2). It is committed below (AC1) and reproducible with one command.
- **On the 2026-10-19 07:23 → 2026-10-20 07:23 window (single N=2 config)**, the log-derived concurrency distribution is: peak **9** concurrent sessions, mean **1.27**, p95 **4**; **16.5%** of minutes have more than two sessions in flight.
- **Capacity model** (`denied = max(0, concurrent − N)`): N=2 denies **19.7%** of session-minutes; N=3 would deny **6.7%**; N=4 **2.2%**. So the *demand* case for a third slot is real.
- **But N=3 does not add prefill capacity** (prefill is serialised at `local_max_concurrent_prefills: 1`) and it **shrinks per-slot context to 83,285** (warm threshold 79,121 vs 120,627 at N=2), pushing more large prompts into cold re-prefill — the dominant source of the 86% queue-abandonment rate already observed.
- **Decision: keep N=2.** Its 126,976-token per-slot context protects the warm-cache band that a third slot would evict. Revisit when the quiet-window N=3 sweep (LP-0MVGPEP3O000RKKD) and the prefill-aware metrics item (LP-0MVGCQ421005IXGC) land.

---

## Method and data sources (AC1 / AC2)

Two independent observables, no new production instrumentation:

1. **Archived daily reports** (`~/proxy-usage-reports/YYYY-MM-DD[_n]/report.md`, 174 files → **76 distinct report-days** after de-duplication). Each row carries busy time (total/fast/cheap), queued→local, fallback-after-queue, dispatch-denied, local serve share, and decode / prompt-eval medians.
2. **Proxy logs** (`/var/log/llama-proxy/proxy.log*`, ~7-day retention). Per-session `Stream started`/`Stream finished` pairs reconstruct the in-flight interval series; `local_dispatch_denied`, `contention_queue_*`, `dispatch_first_byte_ms` and `Stream finished: reason=…` are counted directly.

**Slot config in force** is reconstructed from `git log -p` on `proxy/config*.yaml` (`SLOT_CONFIG_ERAS` in the extractor), so each report-day is attributed to the era active at its window start. Approximation: a window that straddles a change is attributed to the start-era (day resolution).

**Concurrency method.** A session is *in flight* between a `Stream started` and its matching `Stream finished` (FIFO per session, across log rotations). Intervals are clipped to the window; unpaired starts are dropped (they would otherwise invent permanent sessions). Concurrency is sampled at every complete minute boundary (distinct sessions with ≥1 in-flight stream). This measures **demand** (a session routed remote still wanted a slot), which is the quantity a larger pool is sized against.

**Fidelity caveats.** The reports' `Busy` metric counts only `Stream started → Stream finished` decode time; local prefill is logged separately as `dispatch_first_byte_ms`. Real slot occupancy is therefore higher than the reported busy % (see [Measured N=2 baseline](#measured-n2-baseline-ac3)). Early reports (2026-08) use a Day/Night split and carry no busy %/queue rows; those cells are shown as `--`. The historical eras are confounded by rising demand and larger contexts, so the table is evidence, not a controlled experiment.

### Historical report table (76 report-days)

Reproduce with:

```bash
python3 scripts/slot_history_extract.py --history --dedupe-days --markdown
```

| Date (window start) | Snapshot | Slots fast/cheap | Requests (local%) | Busy % | Busy (h) | Decode med tok/s | Prompt-eval med tok/s | Queued→local | Fallback after queue | Dispatch denied |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-02 | 2026-08-03 | 3/3 | 9,355 (13.2%) | -- | -- | 12.8 | 87.2 | -- | -- | 639 |
| 2026-08-06 | 2026-08-08 | 3/3 | 21,031 (22.5%) | 40.3% | 9h 39m | 28.9 | 152.4 | -- | -- | 642 |
| 2026-08-07 | 2026-08-08_2 | 3/3 | 21,104 (25.6%) | 47.0% | 11h 16m | 27.4 | 170.5 | -- | -- | 1,093 |
| 2026-08-08 | 2026-08-10 | 3/3 | 16,840 (27.5%) | 36.1% | 8h 39m | 26.3 | 167.2 | -- | -- | 557 |
| 2026-08-09 | 2026-08-10_2 | 3/3 | 20,439 (24.7%) | 50.8% | 12h 11m | 27.2 | 155.1 | -- | -- | 851 |
| 2026-08-10 | 2026-08-11_2 | 3/3 | 12,268 (25.7%) | 56.2% | 13h 28m | 25.9 | 114.8 | -- | -- | 807 |
| 2026-08-11 | 2026-08-12_2 | 3/3 | 13,739 (33.6%) | 59.3% | 14h 14m | 25.0 | 132.8 | -- | -- | 959 |
| 2026-08-12 | 2026-08-14 | 3/3 | 14,113 (28.1%) | 53.1% | 12h 43m | 25.6 | 128.6 | 42 | 18 | 446 |
| 2026-08-13 | 2026-08-14_2 | 3/3 | 15,136 (23.5%) | 53.6% | 12h 52m | 23.2 | 118.9 | 54 | 291 | 735 |
| 2026-08-14 | 2026-08-16 | 3/3 | 9,684 (31.1%) | 44.8% | 10h 45m | 33.1 | 117.9 | 120 | 338 | 509 |
| 2026-08-15 | 2026-08-16_2 | 3/3 | 11,288 (26.3%) | 47.5% | 11h 24m | 23.9 | 129.5 | 73 | 47 | 622 |
| 2026-08-16 | 2026-08-18 | 3/3 | 7,296 (35.3%) | 40.3% | 9h 40m | 30.0 | 118.2 | 70 | 180 | 360 |
| 2026-08-17 | 2026-08-19 | 3/3 | 9,147 (38.7%) | 55.3% | 13h 16m | 23.1 | 108.8 | 69 | 50 | 294 |
| 2026-08-18 | 2026-08-20 | 3/3 | 4,070 (45.9%) | 29.8% | 7h 08m | 30.0 | 76.9 | 1 | 4 | 38 |
| 2026-08-19 | 2026-08-20_3 | 3/3 | 7,413 (43.7%) | 56.0% | 13h 25m | 22.1 | 95.5 | 238 | 120 | 684 |
| 2026-08-20 | 2026-08-21_2 | 3/3 | 8,091 (40.7%) | 62.4% | 14h 58m | 22.3 | 96.6 | 178 | 91 | 433 |
| 2026-08-21 | 2026-08-23 | 3/3 | 10,264 (31.8%) | 69.3% | 16h 37m | 18.4 | 89.5 | 106 | 56 | 671 |
| 2026-08-22 | 2026-08-23_2 | 3/3 | 8,833 (33.4%) | 60.6% | 14h 33m | 18.4 | 89.6 | 106 | 56 | 562 |
| 2026-08-23 | 2026-08-24_2 | 3/3 | 9,263 (35.9%) | 90.1% | 21h 38m | 21.3 | 93.5 | 59 | 70 | 940 |
| 2026-08-24 | 2026-08-25_2 | 3/3 | 7,319 (39.2%) | 68.1% | 16h 20m | 25.5 | 108.5 | 91 | 58 | 794 |
| 2026-08-25 | 2026-08-26_4 | 3/3 | 5,471 (39.8%) | 63.7% | 15h 17m | 21.9 | 105.2 | 29 | 7 | 367 |
| 2026-08-26 | 2026-08-28_7 | 3/3 | 5,744 (41.1%) | 69.1% | 16h 35m | 22.4 | 106.6 | 29 | 7 | 306 |
| 2026-08-27 | 2026-08-28_4 | 3/3 | 3,551 (47.8%) | 35.2% | 8h 27m | 29.5 | 99.8 | 3 | 2 | 98 |
| 2026-08-28 | 2026-08-29_2 | 3/3 | 4,562 (40.4%) | 46.2% | 11h 04m | 21.7 | 92.3 | 27 | 20 | 188 |
| 2026-08-29 | 2026-08-31 | 3/3 | 8,657 (22.5%) | 52.6% | 12h 38m | 14.4 | 76.2 | 22 | 6 | 1,775 |
| 2026-08-30 | 2026-08-31_2 | 3/3 | 10,771 (22.5%) | 66.0% | 15h 50m | 16.2 | 78.1 | 39 | 7 | 1,592 |
| 2026-08-31 | 2026-09-02 | 3/3 | 7,727 (33.1%) | 69.5% | 16h 40m | 25.4 | 91.8 | 0 | 0 | 826 |
| 2026-09-01 | 2026-09-03 | 3/3 | 6,101 (29.2%) | 63.2% | 15h 10m | 28.4 | 78.0 | 122 | 25 | 486 |
| 2026-09-02 | 2026-09-03_2 | 3/3 | 3,832 (26.8%) | 40.0% | 9h 35m | 16.7 | 76.7 | 15 | 10 | 96 |
| 2026-09-03 | 2026-09-05 | 3/3 | 4,825 (42.2%) | 90.0% | 21h 35m | 15.7 | 81.5 | 0 | 0 | 396 |
| 2026-09-04 | 2026-09-06 | 3/3 | 3,448 (51.5%) | 52.9% | 12h 41m | 15.4 | 79.7 | 0 | 0 | 204 |
| 2026-09-05 | 2026-09-07 | 3/3 | 3,662 (23.3%) | 52.4% | 12h 34m | 15.9 | 68.6 | 2 | 0 | 82 |
| 2026-09-06 | 2026-09-08 | 3/3 | 809 (72.3%) | 18.3% | 4h 23m | 16.9 | 75.5 | 0 | 0 | 0 |
| 2026-09-07 | 2026-09-08_3 | 3/3 | 3,720 (55.1%) | 84.4% | 20h 14m | 18.9 | 70.6 | 0 | 0 | 97 |
| 2026-09-08 | 2026-09-09_2 | 3/3 | 3,032 (45.2%) | 44.4% | 10h 39m | 27.4 | 59.8 | 0 | 0 | 184 |
| 2026-09-09 | 2026-09-11 | 3/3 | 2,136 (44.6%) | 28.1% | 6h 45m | 20.3 | 59.3 | 342 | 0 | 196 |
| 2026-09-10 | 2026-09-12 | 3/3 | 2,241 (66.8%) | 32.1% | 7h 41m | 30.5 | 82.1 | 244 | 5 | 121 |
| 2026-09-11 | 2026-09-13 | 3/3 | 1,642 (53.4%) | 22.9% | 5h 29m | 27.3 | 86.5 | 45 | 0 | 31 |
| 2026-09-12 | 2026-09-14 | 3/3 | 4,102 (35.0%) | 22.7% | 5h 27m | 21.4 | 74.8 | 670 | 1 | 540 |
| 2026-09-13 | 2026-09-15 | 1/3 | 3,445 (43.2%) | 15.5% | 3h 43m | 31.3 | 88.1 | 917 | 1 | 320 |
| 2026-09-14 | 2026-09-16 | 1/3 | 2,900 (57.8%) | 17.5% | 4h 11m | 33.4 | 92.6 | 1,593 | 0 | 366 |
| 2026-09-15 | 2026-09-17 | 1/3 | 5,734 (53.8%) | 36.3% | 8h 42m | 27.7 | 79.7 | 1,757 | 6 | 449 |
| 2026-09-16 | 2026-09-18 | 1/3 | 8,249 (69.9%) | 53.6% | 12h 51m | 28.1 | 66.0 | 2,535 | 5 | 474 |
| 2026-09-17 | 2026-09-19 | 1/3 | 9,200 (93.1%) | 66.6% | 15h 59m | 43.4 | 75.3 | 1,251 | 0 | 43 |
| 2026-09-18 | 2026-09-20 | 1/3 | 4,331 (58.9%) | 42.1% | 10h 05m | 25.8 | 84.9 | 834 | 6 | 197 |
| 2026-09-19 | 2026-09-20_2 | 1/3 | 5,985 (57.9%) | 75.8% | 18h 11m | 22.6 | 80.2 | 1,844 | 6 | 350 |
| 2026-09-20 | 2026-09-21_2 | 1/3 | 6,813 (48.0%) | 60.1% | 14h 25m | 25.7 | 87.7 | 1,763 | 1 | 468 |
| 2026-09-21 | 2026-09-23 | 1/3 | 5,367 (64.0%) | 60.1% | 14h 25m | 19.8 | 84.2 | 1,985 | 4 | 583 |
| 2026-09-22 | 2026-09-24 | 1/3 | 7,303 (37.6%) | 61.9% | 14h 51m | 16.3 | 77.9 | 1,953 | 52 | 724 |
| 2026-09-23 | 2026-09-25 | 1/3 | 10,062 (28.6%) | 63.3% | 15h 11m | 18.9 | 74.7 | 2,629 | 56 | 887 |
| 2026-09-24 | 2026-09-26 | 1/3 | 9,622 (24.5%) | 56.3% | 13h 30m | 16.3 | 76.4 | 2,587 | 62 | 1,112 |
| 2026-09-25 | 2026-09-27 | 1/3 | 7,125 (26.1%) | 45.8% | 10h 59m | 26.7 | 82.0 | 1,988 | 14 | 945 |
| 2026-09-26 | 2026-09-28 | 1/3 | 16,584 (4.7%) | 43.6% | 10h 27m | 20.7 | 80.8 | 3,305 | 2 | 2,936 |
| 2026-09-27 | 2026-09-29 | 1/3 | 12,778 (10.9%) | 47.1% | 11h 18m | 25.1 | 86.1 | 5,012 | 9 | 4,182 |
| 2026-09-28 | 2026-09-30 | 1/3 | 8,412 (32.6%) | 50.9% | 12h 13m | 19.2 | 73.1 | 3,814 | 23 | 1,613 |
| 2026-09-29 | 2026-10-01 | 1/3 | 9,318 (27.4%) | 50.7% | 12h 10m | 27.4 | 70.5 | 3,664 | 11 | 1,916 |
| 2026-09-30 | 2026-10-02 | 1/3 | 6,526 (33.2%) | 62.1% | 14h 54m | 25.9 | 75.5 | 2,887 | 3 | 1,367 |
| 2026-10-01 | 2026-10-03 | 1/3 | 6,489 (20.1%) | 47.0% | 11h 17m | 24.6 | 79.7 | 2,067 | 1 | 1,453 |
| 2026-10-02 | 2026-10-04 | 1/3 | 5,967 (38.5%) | 42.8% | 10h 15m | 14.3 | 76.2 | 2,042 | 6 | 757 |
| 2026-10-03 | 2026-10-05 | 1/3 | 2,799 (40.1%) | 14.0% | 3h 21m | 17.5 | 80.4 | 1,074 | 4 | 418 |
| 2026-10-04 | 2026-10-06 | 1/3 | 8,297 (20.0%) | 37.0% | 8h 52m | 21.2 | 81.8 | 3,563 | 4 | 2,473 |
| 2026-10-05 | 2026-10-07 | 1/3 | 8,989 (26.6%) | 63.6% | 15h 16m | 20.2 | 85.7 | 4,037 | 14 | 3,454 |
| 2026-10-06 | 2026-10-08 | 1/3 | 11,587 (18.8%) | 55.0% | 13h 12m | 15.1 | 74.9 | 4,514 | 59 | 3,071 |
| 2026-10-07 | 2026-10-09 | 1/3 | 31,513 (8.5%) | 91.6% | 21h 58m | 21.2 | 57.3 | 9,556 | 124 | 7,107 |
| 2026-10-08 | 2026-10-10 | 1/3 | 24,745 (9.8%) | 68.5% | 16h 26m | 23.0 | 69.5 | 9,490 | 146 | 7,478 |
| 2026-10-09 | 2026-10-11 | 1/3 | 13,838 (15.0%) | 42.4% | 10h 10m | 16.7 | 73.2 | 2,719 | 780 | 1,469 |
| 2026-10-10 | 2026-10-12 | 1/3 | 5,934 (29.8%) | 49.6% | 11h 54m | 18.3 | 73.9 | 716 | 738 | 89 |
| 2026-10-11 | 2026-10-13 | 1/3 | 2,159 (54.7%) | 23.6% | 5h 40m | 17.6 | 77.7 | 606 | 256 | 56 |
| 2026-10-12 | 2026-10-14 | 1/3 | 19,183 (6.3%) | 30.3% | 7h 15m | 14.7 | 69.6 | 1,029 | 2,612 | 329 |
| 2026-10-13 | 2026-10-14_2 | 1/3 | 11,544 (10.4%) | 33.6% | 8h 04m | 15.3 | 57.9 | 885 | 2,106 | 202 |
| 2026-10-14 | 2026-10-16 | 1/3 | 19,602 (6.7%) | 30.3% | 7h 16m | 26.7 | 96.2 | 581 | 3,752 | 328 |
| 2026-10-15 | 2026-10-16 | 1/3 | 12,220 (7.0%) | 21.9% | 5h 15m | 17.8 | 57.4 | 773 | 1,416 | 585 |
| 2026-10-16 | 2026-10-18 | 1/3 | 3,599 (13.2%) | 16.3% | 3h 54m | 17.9 | 91.5 | 319 | 428 | 1,075 |
| 2026-10-17 | 2026-10-19 | 2/3 | 12,667 (8.5%) | 68.9% | 16h 32m | 24.1 | 101.9 | 1,025 | 1,991 | 3,075 |
| 2026-10-18 | 2026-10-20 | 2/2 | 12,810 (7.1%) | 31.9% | 7h 39m | 22.6 | 99.8 | 1,268 | 3,339 | 1,359 |
| 2026-10-19 | 2026-10-20_2 | 2/2 | 17,193 (2.9%) | 29.9% | 7h 10m | -- | -- | 912 | 5,074 | 2,505 |

### Reading the table by era

| Era | Days | Avg busy % | Avg local serve share | Queued→local | Fallback after queue | Dispatch denied |
|---|---:|---:|---:|---:|---:|---:|
| **3/3** (2026-08-02 → 09-12) | 39 | 52.0% | 36.4% | 2,796 | 1,469 | 20,221 |
| **1/3** (2026-09-13 → 10-16) | 34 | 46.4% | 32.3% | 86,289 | 12,707 | 49,276 |
| **2/3** (2026-10-17) | 1 | 68.9% | 8.5% | 1,025 | 1,991 | 3,075 |
| **2/2** (2026-10-18 → 10-19) | 2 | 30.9% | 5.0% | 2,180 | 8,413 | 3,864 |

Queue abandonment as a share of queue-path arrivals: 3/3 **34%**, 1/3 **13%**, 2/3 **66%**, 2/2 **79%**. Abandonment did **not** fall when the fast slot went 1 → 2 (2026-10-17) or when modes converged at 2/2 — it rose with total demand and context size. That is the central historical finding: **slot count alone does not govern abandonment; prefill pressure and cache-cold prompts do.**

---

## Concurrency vs capacity (AC2)

Window: **2026-10-19 07:23:12 → 2026-10-20 07:23:12** (the intake's window, single N=2 config), 1,440 complete minutes.

Reproduce with:

```bash
python3 scripts/slot_history_extract.py --concurrency \
  --window-start 2026-10-19T07:23:12 --window-end 2026-10-20T07:23:12 --markdown
```

**Concurrency distribution** (minutes with N distinct sessions streaming; window 1,440 complete minutes, 17,448 in-flight intervals):

| Concurrent sessions | Minutes | Share |
|---:|---:|---:|
| 0 | 489 | 34.0% |
| 1 | 437 | 30.3% |
| 2 | 277 | 19.2% |
| 3 | 154 | 10.7% |
| 4 | 55 | 3.8% |
| 5 | 20 | 1.4% |
| 6 | 6 | 0.4% |
| 7 | 1 | 0.1% |
| 9 | 1 | 0.1% |
| **min / mean / p50 / p95 / peak** | **0 / 1.27 / 1 / 4 / 9** | |

**Capacity model** — `denied = max(0, concurrent - N)` session-minutes per minute, projected at N=2/3/4:

| Slots N | Minutes above N | Share of minutes above N | Share of session-minutes denied |
|---:|---:|---:|---:|
| 2 | 237 | 16.5% | 19.7% |
| 3 | 83 | 5.8% | 6.7% |
| 4 | 28 | 1.9% | 2.2% |

**Peak concurrency by session class:** `audit-*` 3, `herdr-*` 9, `operator/pi/LAN` 4.

**Local-dispatch denials in window:** 2,635 (by active count: active=1 → 3, active=2 → 2,627, active=3 → 5); contention queue: 799 dispatched, 4,933 abandoned.

**Measured local prefill (dispatch → first byte):** 411 samples, median 74.5s, p90 297.0s, max 817.3s, total 13.9 h of GPU prefill time.

**Stream finish reasons:** `tool_calls` 17,612, `stop` 319, `client_disconnect` 4, `error` 1 (566 local streams finished).

**Interpretation.** Demand exceeds two slots for 16.5% of minutes, and the worst minutes reach 9 concurrent sessions (driven by `herdr-*` panes; audits peak at 3 and operator/pi/LAN at 4). Under the capacity model a third slot removes two-thirds of the denied session-minutes (19.7% → 6.7%); a fourth removes almost all of the rest. This is a strong *quantity* argument for N=3 — it is the *cost* side (prefill serialisation and per-slot context) that decides against it for now.

---

## Measured N=2 baseline (AC3, production-derived)

Measured directly from the same window's logs (no new load):

| Metric | Value |
|---|---|
| Local prefill (dispatch → first byte) | **411 samples**, median **74.5 s**, p90 **297.0 s**, max **817.3 s**, total **13.9 h** |
| Local streams finished | 566 |
| Finish reasons | `tool_calls` 17,612 · `stop` 319 · `client_disconnect` 4 · `error` 1 · **`length` 0** |
| Dispatch denied | 2,635 (99.7% at `active=2`) |
| Contention queue | 799 dispatched · **4,933 abandoned (86.1%)** |
| Reported decode busy (report, 2026-10-19) | 29.9% (7 h 12 m) |
| Real slot occupancy incl. prefill | ≈ 13.9 h prefill + 7.2 h decode = **21.1 h of 48 slot-hours ≈ 44%** |

The prefill total matches the intake's independent estimate (14.3 h / 463 requests) within log-rotation noise. The queue-abandonment rate (86%) confirms that the 120 s `contention_queue_max_wait_seconds` is shorter than the median prefill-plus-queue hold, so most queued waits are guaranteed losses — the motivation for the prefill-ETA admission guard (already live).

## Measured N=3 / N=4 sweep status (AC3, deferred)

The controlled sweep was **attempted and aborted** on 2026-10-20: `slot_benchmark --slots 3 --concurrency 3 --prompt-tokens 2000` ran against the live proxy while it was serving 5 concurrent sessions (52 distinct sessions / 5 min; 2 audits + 4 herdr panes holding local leases). Under that contention every benchmark turn-1 request stalled at 600–760 s against the 1200 s timeout and llama-server stopped answering health probes. The harness restored the config (`sha256 6eec5656…`) and a controlled `start-proxy.sh --restart` returned the proxy to healthy (N=2, `--parallel 2`).

A clean N=3/N=4 measurement needs a quiet window; it is tracked in **LP-0MVGPEP3O000RKKD** (same harness, exact commands recorded there). Until it runs, the N=3 cost/benefit is **projected**, not measured — which is why the decision is to hold N=2 rather than switch.

**Per-slot context / warm-threshold interaction** (the item that must be decided together):

| Slots N | Per-slot physical ctx (`262144/N − 4096`) | Warm threshold at 95% of per-slot |
|---:|---:|---:|
| 2 | **126,976** | **120,627** |
| 3 | 83,285 | 79,121 |
| 4 | 61,440 | 58,368 |

N=3's 83,285 clamp lands almost exactly on today's *effective* ceiling, and 45.7% of local dispatches are already cache-cold. Moving to N=3 without first fixing cache warmth would convert more 83–127k prompts into cold re-prefills — raising prefill time, queue hold time and therefore abandonment.

---

## Decision and mitigations (AC4)

**Decision: keep `session_slot_pool_size: 2`.** No production change (AC5).

Rationale:
1. N=2 preserves the 126,976-token per-slot context and therefore the warm-cache band; N=3 (83,285) and N=4 (61,440) would evict it.
2. Prefill serialisation (`local_max_concurrent_prefills: 1`) means extra slots add decode concurrency, not prefill capacity — they cannot by themselves fix the 86% queue abandonment.
3. The historical record shows abandonment rising with demand and context size, not falling when the slot count rose.
4. The N=3 demand benefit (19.7% → 6.7% denied session-minutes) is real but **unmeasured on cost**; the quiet-window sweep is the gate.

**Expected effect if N were changed (for the follow-up):**

| Outcome | Keep N=2 (chosen) | Move to N=3 |
|---|---|---|
| Local serve share | 7.1% (10-18) / 2.9% (10-19) observed | small upside on *warm* sessions, largely offset by more cold re-prefill |
| Queue abandonment | 86% (prefill-bound) | unlikely to fall materially; prefill still serialised, cold share rises |
| Per-slot context | **126,976** | 83,285 |
| Warm threshold (95%) | **120,627** | 79,121 |
| VRAM headroom | ~43 GB `MemAvailable` observed; ROCm reports unified memory | roughly neutral KV total (more slots × smaller ctx), unmeasured precisely |
| `finish_reason=length` | **0** in window | expected to rise (smaller ctx truncates) |

**Mitigations N=2 depends on (all necessary to keep abandonment and latency bounded):**

1. **Prefill serialisation** (`local_max_concurrent_prefills: 1`, LP-0MVGDUUV3001XFRJ) — protects decode from prefill contention; do not disable.
2. **Prefill-ETA admission guard** (LP-0MVGDUV5E005UWBQ) — refuses queue waits that cannot succeed within the 120 s budget, replacing 86% futile waits with immediate remote fallback.
3. **Cache warmth** (LP-0MVGD9YXH008O304) — cut the 45.7% cold share; every point of warmth directly reduces prefill hold and abandonment.
4. **Capacity-based admission** (LP-0MVGDUVS6003T2IX) — keep default-off until warmth improves and the 24 h measurement is green.
5. **Economic threshold discipline** — keep the 38k cold / ~126,976 warm routing clamp; do not widen it to compete for slots the GPU cannot prefill.
6. **Queue-wait tuning only with the ETA guard** — a longer `contention_queue_max_wait_seconds` without the guard merely lengthens guaranteed-loss waits.
7. **Re-decide slot count with LP-0MVGPEP3O000RKKD** (quiet-window sweep) and **LP-0MVGCQ421005IXGC** (prefill-aware busy metrics); decide the warm-threshold item in the same pass.

---

## Scope and follow-ups (AC5)

- **No production slot-count change was made.** The only production interaction was the aborted sweep, which restored config to `session_slot_pool_size: 2` and was verified healthy (`--parallel 2`, 2 slots).
- **LP-0MVGPEP3O000RKKD** — controlled N=3/N=4 sweep in a quiet window (AC3 completion; `discovered-from` this item).
- **LP-0MVGCQ421005IXGC** — prefill-aware busy metrics (the reported 29.9% busy understates real occupancy; this doc uses the 44% figure).
- Warm-threshold = 95%-of-per-slot item — must be decided with any future slot-count change (N=3 target becomes 79,121).

## Files / evidence

- Extraction script: `scripts/slot_history_extract.py`; tests: `tests/test_slot_history_extract.py` (14 tests).
- Raw extracted data: `docs/slot-history-LP-0MVGCQ3QN009HRE8.json` (history + concurrency).
- Prior verdict / method: `docs/slot-count-verdict-lp-0mv9ungeb009xuz5.md`.
- Sweep harness: `proxy/benchmarks/slot_benchmark.py`.
