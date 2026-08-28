# Session estimated-context distribution (2026-08-24..26)

Derived from `routing_check` log lines (`estimated_tokens`, the proxy's routing-time session-context estimate). Mode per the effective per-slot warm clamp (fast 83285 / cheap 100000). Breach caps: fast 83285, cheap 61440.

## Trend across days

| Day | Sessions | Pressure warnings | Breach fast | Breach cheap |
|---|---|---|---|---|
| 2026-08-24 | 253 | 1234 | 24/187 | 6/66 |
| 2026-08-25 | 191 | 1362 | 18/119 | 11/72 |
| 2026-08-26 | 180 | 1731 | 14/135 | 5/45 |

## 2026-08-24

- Sessions with routing checks: **253**
- `context_pressure` warnings: **1234** (39 sessions)
- routing_skip_local: context_too_large=1054, large_context_bypass=383

### Distribution (per-session max estimated context)

| Mode | Sessions | Median | Mean | p90 | p95 | Max |
|---|---|---|---|---|---|---|
| cheap | 66 | 19313 | 27992 | 57875 | 95537 | 136976 |
| fast | 187 | 32769 | 44051 | 89881 | 125804 | 248905 |

### Breach counts vs per-mode cap

| Mode | Sessions | Breach (>= cap) | % | Cap |
|---|---|---|---|---|
| cheap | 66 | 6 | 9.1% | 61440 |
| fast | 187 | 24 | 12.8% | 83285 |

## 2026-08-25

- Sessions with routing checks: **191**
- `context_pressure` warnings: **1362** (34 sessions)
- routing_skip_local: context_too_large=1191, large_context_bypass=526

### Distribution (per-session max estimated context)

| Mode | Sessions | Median | Mean | p90 | p95 | Max |
|---|---|---|---|---|---|---|
| cheap | 72 | 32362 | 34491 | 64616 | 76416 | 109098 |
| fast | 119 | 37346 | 53869 | 115435 | 140059 | 424128 |

### Breach counts vs per-mode cap

| Mode | Sessions | Breach (>= cap) | % | Cap |
|---|---|---|---|---|
| cheap | 72 | 11 | 15.3% | 61440 |
| fast | 119 | 18 | 15.1% | 83285 |

## 2026-08-26

- Sessions with routing checks: **180**
- `context_pressure` warnings: **1731** (25 sessions)
- routing_skip_local: context_too_large=1553, large_context_bypass=365

### Distribution (per-session max estimated context)

| Mode | Sessions | Median | Mean | p90 | p95 | Max |
|---|---|---|---|---|---|---|
| cheap | 45 | 30009 | 36777 | 64606 | 109169 | 162389 |
| fast | 135 | 27385 | 46234 | 92101 | 167364 | 651408 |

### Breach counts vs per-mode cap

| Mode | Sessions | Breach (>= cap) | % | Cap |
|---|---|---|---|---|
| cheap | 45 | 5 | 11.1% | 61440 |
| fast | 135 | 14 | 10.4% | 83285 |

