# Group-2 proxy restart evidence (LP-0MW8EATPF00924OF)

Single group-2 validation restart, performed 2026-11-16T15:20:47Z via
`/skill:start-proxy --restart` (parent LP-0MW7V14DN007F127 AC1; group-1
serialisation proven by LP-0MW8EAT60003Z28H `docs/dev/group-2-serialisation-proof.md`).

## Restart record

| Field | Pre-restart | Post-restart |
|---|---|---|
| proxy PID | **3122239** | **3720326** |
| proxy `ps -o lstart` | Sun Nov 15 01:06:48 2026 (uptime 1d14:13) | **Mon Nov 16 15:20:47 2026** |
| restart attribution | `source=manual actor=unknown reason=manual` (2026-11-15T01:06:50Z — see note) | `source=manual actor=unknown reason=LP-0MW7V14DN007F127: group-2 validation restart (cold-regime ETA + deadline token capacity)` |
| deployed HEAD | — | **`79f53ac1a`** (LP-0MW8EAT60003Z28H: Implementation complete, 2026-11-16 15:09:39 +0000) |

Launch command:

```bash
cd /home/rgardler/projects/llm/proxy
bash scripts/start-proxy.sh --restart --reason "LP-0MW7V14DN007F127: group-2 validation restart (cold-regime ETA + deadline token capacity)"
```

Startup log: `Proxy launched detached: child pid 3720326, output in /tmp/proxy-startup-20261116T152048Z-3719311.log`.
Log line: `Logging initialised at /var/log/llama-proxy/proxy.log pid=3720326`; prior log preserved
(`Rotated previous proxy log aside: proxy.log.2026-11-16_152045 (pre-restart window preserved)`).

Raw `/health` after restart (2026-11-16T15:21Z):

```json
{"status":"healthy","ready":true,"llama_server_running":true,"backend_reachable":true,"llama_server_managed":true,"self_healing_in_progress":false,"tts_server_running":true,"tts_server_healthy":true}
```

## Group-2 ancestry — every sha is an ancestor of the deployed tree

`git merge-base --is-ancestor <sha> HEAD` where HEAD = `79f53ac1a` (deployed tree; all group-2
commits land on `dev` and are ancestors of the merge at this HEAD). **All YES:**

| sha (short) | item |
|---|---|
| 7f6033901 | LP-0MW957Q41002EXE2 — cold-only prefill rate window and fit |
| a6b843588 | LP-0MW957PGJ007MCDH — cold-regime calibration rows |
| d8aca9755 | LP-0MW957RB3000ER79 — deadline token capacity |
| ab6d50551 | LP-0MW957SKX0079QKC — honest seconds basis |
| 2c3917dbe | LP-0MW957QPH004644X — deadline token capacity rows |
| 938bb1b6e | LP-0MW957RWM0007Z3S — two-server attribution rows |
| 18dc37d43 | LP-0MW957TAE000QD9N — cold-regime replay proof |
| 4a72facb0 | LP-0MWGAEBWJ00334W2 — cold-window calibration fix |

Verified verbatim: `for s in 7f60339013e0c806dd944c362677d69e14bc9501 a6b84358802682440912fe921e9bd9b70f9befd3 d8aca9755d8f8c1b2d22fcb7451b7230bdc126de ab6d50551e3fcdc9cceea04d11cdf6c086a9e112 2c3917dbec24182478a2e5af74fd1d57fe9cf3b1 938bb1b6ec23c2bc2f200d033b21f0476c30b76f 18dc37d43 4a72facb0; do git merge-base --is-ancestor $s HEAD && echo "$s YES" || echo "$s NO"; done` → all YES.

## Group-1 ancestry — no regression by revert (14/14)

Same command against the group-1 sha set (`ff6c8996d 41b54d27d 55794658d c18c0aaa8 b16b89284
80bd39a7b 2db749595 94f8dde35 9487ef83b 01d89a19e 011f2db0c 03159b423 d5db475b1 2577fd14b`)
→ **14/14 YES**. Group-1's measured-residency routing is still present; the deployed tree loads
both groups, and exactly one group is under test per restart.

## Deployed group-2 payload (post-restart)

`GET /llama/local/status` (2026-11-16T15:21Z): the group-2 fields RB3/SKX/QPH/RWM publish now
appear (absent pre-restart):

| Field | Value |
|---|---|
| `deadline_capacity_tokens` | 46153 |
| `deadline_capacity_calibrated` | false (uncalibrated at startup) |
| `deadline_capacity_cold_ms_per_token` | 3.9 |
| `prefill_eta_kind` | reference_cold_bound |
| `prefill_eta_calibrated` | false |
| `capacity_hint` | prefill_bound |
| `status_schema_version` | 7 |

TXD hand-off expectation met: `deadline_capacity_tokens == floor(180000 / 3.9)` and the guard log
line carries `deadline_capacity_tokens=` / `capacity_basis=` / `cold_ms_per_token=`. Per
LP-0MWGAEBWJ00334W2 the 65k+ bucket calibrates from abandoned prefills; the AC5(e)
`prefill_capacity_exceeded` refusal count is measured over the window by the verdict child.

## Note: the previous unattributed restart (2026-11-15T01:06:50Z)

The pre-restart proxy (PID 3122239) had started at 2026-11-15T01:06:50Z with
`source=manual actor=unknown reason=manual`, *before* the WJ2 calibration fix (`4a72facb0`,
merged 01:10:04Z) and the TAE replay proof (`18dc37d43`, merged 05:09:55Z). That tree was an
incomplete group-2 build (the calibration fix that makes the deadline bound non-inert was not
loaded), so it is **not** treated as this group's validation restart. This restart loads the full
group-2 tree and is the single certified group-2 anchor; the ≥ 24 h restart-free window measured
by LP-0MW8EAU8P0079280 starts at this anchor (first whole minute after 15:20:47Z → earliest close
2026-11-17T15:20:47Z).
