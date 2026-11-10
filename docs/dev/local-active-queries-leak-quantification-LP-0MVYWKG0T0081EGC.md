# `local_active_queries` leak quantification (LP-0MVYWKG0T0081EGC)

Deterministic measurement of the `local_active_queries` counter leak that parent
LP-0MVWF78QK0060TZH tracks (its AC1, and the baseline the post-fix AC6 window is
compared against).

Tool: [`scripts/analyze_local_active_queries_leak.py`](../../scripts/analyze_local_active_queries_leak.py).
Tests: [`proxy/tests/test_analyze_local_active_queries_leak.py`](../../proxy/tests/test_analyze_local_active_queries_leak.py).

## What it measures

| Field | Definition |
|---|---|
| recovery warnings | `WARNING - local_active_queries counter recovered: reset from N to M (M active dispatch records)` lines in the window |
| delta distribution | the `N→M` reset pair, counted per distinct pair (`2→1`, `1→0`, …) |
| local streams | `INFO - Stream finished: … provider=local` lines in the window (the denominator AC1 names; remote providers such as `provider=OpenCode` are excluded) |
| leak ratio | warnings / local streams, reported both ways (`warnings_per_local_stream`, `local_streams_per_warning`) |
| files used | every file that contributed in-window lines, with its own first/last timestamp and per-file warning/stream counts |

The per-file rows are the point of the tool: the original hand measurement
mixed a rotated file from the previous day into the window and double-counted
it. `--files` pins the exact file set, and a line that is byte-identical in two
files (a logrotate `copytruncate` tail) is counted once and reported under
`duplicates_dropped`, so double counting cannot happen silently.

## Coverage verdict

A ratio computed over a window whose logs are incomplete is not a measurement.
The tool reuses [`scripts/lib/proxy_logs.py`](../../scripts/lib/proxy_logs.py)
for discovery (dot `proxy.log.YYYY-MM-DD_HH` and dash
`proxy.log-YYYY-MM-DD_HH`, plain or `.gz`) and fails the verdict (exit 3) when:

- two adjacent files inside the window are more than `--max-gap-seconds`
  (default 3600) apart;
- the window starts more than that before its first retained line;
- with `--window-closed`, the window ends more than that after its last
  retained line;
- the retained logs do not reach the window at all (`outside_retention`).

A window the retention *does* span but which contains no matched lines is an
empty result (exit 0), not a gap.

## Reproducible commands

Window bounds are the log's own wall clock (the proxy log carries no UTC
offset). Both windows below are fully elapsed, so `--window-closed` is used.

### Corrected 2026-10-31 baseline (parent AC1)

```bash
./scripts/analyze_local_active_queries_leak.py \
    --window '2026-10-31T00:00:00,2026-10-31T13:14:00' --window-closed
```

Files used (6): `proxy.log-2026-10-31_00.gz` (00:00:00, 3 lines),
`proxy.log.2026-10-31_003749.gz` (00:00:00–00:37:49),
`proxy.log.2026-10-31_004152.gz` (00:37:53–00:41:52),
`proxy.log.2026-10-31_00.gz` (00:41:57–06:41:53),
`proxy.log.2026-10-31_06.gz` (06:42:00–12:41:54),
`proxy.log.2026-10-31_142242.gz` (12:42:00–13:14:00).

**76 recoveries / 355 local streams ⇒ 1 per 4.67 local streams**, deltas
`2→1` 37, `1→0` 18, `3→2` 11, `3→1` 6, `4→2` 3, `2→0` 1. This confirms the
independent review update on the parent (76/355) and refutes the item's
original 136/359.

### Current (pre-fix) 24 h baseline — the AC6 comparison point

```bash
./scripts/analyze_local_active_queries_leak.py \
    --window '2026-11-01T05:00:00,2026-11-02T06:49:00' --window-closed
```

Files used (9): `proxy.log.2026-10-31_23.gz` (05:00:05–05:56:08),
`proxy.log.2026-11-01_05.gz` (05:56:09–11:56:08),
`proxy.log.2026-11-01_131248.gz` (11:56:09–13:12:48),
`proxy.log.2026-11-01_131746.gz` (13:12:52–13:17:46),
`proxy.log.2026-11-01_13.gz` (13:17:50–19:17:48),
`proxy.log.2026-11-01_212637.gz` (19:17:57–21:26:37),
`proxy.log-2026-11-02_00.gz` (21:26:40–00:00:00),
`proxy.log.2026-11-01_21.gz` (00:00:00–03:26:39),
`proxy.log.2026-11-02_03.gz` (03:26:43–06:49:00).

**159 recoveries / 1616 local streams ⇒ 1 per 10.16 local streams**, deltas
`2→1` 92, `1→0` 33, `3→2` 21, `3→1` 9, `4→2` 4.

The plan-time figure for the same window was **151**. The 8-warning difference
is entirely `proxy.log.2026-10-31_23.gz`, whose name encodes 2026-10-31 but
whose content runs 2026-11-01 05:00:05 → 05:56:08 — the "a rotated file holds
data past its name" trap. Excluding exactly that file reproduces 151 / 1500:

```bash
./scripts/analyze_local_active_queries_leak.py \
    --window '2026-11-01T05:00:00,2026-11-02T06:49:00' --window-closed \
    --files proxy.log.2026-11-01_05.gz,proxy.log.2026-11-01_131248.gz,\
proxy.log.2026-11-01_131746.gz,proxy.log.2026-11-01_13.gz,\
proxy.log.2026-11-01_212637.gz,proxy.log.2026-11-01_21.gz,\
proxy.log-2026-11-02_00.gz,proxy.log.2026-11-02_03.gz
```

### Cross-check against the AC2/AC8 audit measurement

The AC2/AC8 audit (LP-0MVYWK964006YDQ8) measured 95 recoveries in
2026-11-04 15:38 → 2026-11-05 03:50 using `Stream started` as its denominator.
This tool reproduces **95** when the end bound covers the whole `03:50`
minute (`2026-11-05T03:51:00`) and 94 for an end of exactly `03:50:00`, and it
reports 1220 local `Stream finished` events, so the two measurements agree on
the numerator and differ only in the denominator definition AC1 fixed
(`Stream finished`).

## Post-fix use (AC6)

Run the same two commands after the fix lands: the ≥60 min post-fix window must
show **0** recovery warnings, and the 24 h figure recorded above must not
recur. The per-file rows make the comparison auditable: the same file set
should contribute the same streams, with the warnings gone.
