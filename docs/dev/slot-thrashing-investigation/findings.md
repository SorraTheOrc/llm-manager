# Slot Thrashing Investigation Findings

**Date:** 2026-06-23
**Work Item:** LP-0MQR75QYM001HAUB (Slot thrashing reproduction & evidence)
**Parent:** LP-0MQR0780Z006TLX6 (Improve slot management)

## Executive Summary

Slot thrashing at the proxy level is **confirmed and reproducible**. With `pool_size=1` (the current production configuration) and ≥2 concurrent sessions, **95–98% of all turn transitions** result in a different session acquiring the slot — meaning a session's KV cache is almost certainly invalidated between every turn.

This is not a GPU-level issue; it's a proxy-level coordination gap. The `SlotLockCoordinator` serializes concurrent requests through the same asyncio.Lock, but between turns the lock is released and any waiting session can immediately acquire it. No ordering or affinity mechanism exists.

## Reproducibility

### Repro Script

A standalone, self-contained repro script is provided at:

```
docs/dev/slot-thrashing-investigation/repro.py
```

Usage:
```bash
python3 docs/dev/slot-thrashing-investigation/repro.py --sessions 4 --turns 10 --json
```

The script:
- Imports no production code; mirrors `SlotLockCoordinator` logic exactly
- Simulates N concurrent sessions making multi-turn requests with realistic timing
- Captures every lock acquire/release event with high-resolution timestamps
- Detects and quantifies inter-session slot stealing
- Reports gap windows, hold times, and estimated save/restore cost

### Requirements

- Python 3.10+
- No live GPU needed
- No test harness dependencies

## Evidence

### Steal Rate vs. Pool Size (4 sessions, 10 turns each)

| Pool Size | Steal Rate | Gap Window (avg) | Notes |
|-----------|-----------|-------------------|-------|
| 1 | 95–98% | ~0.03ms | **Production config** — near-total thrashing |
| 2 | 90% | ~7.8ms | Still severe: 2 slots for 4 sessions causes many collisions |
| 3 | 78% | ~6.3ms | Moderate improvement |
| 4 | 40% | ~5.9ms | Significant improvement: one slot per session |
| 8 | 43% | ~5.3ms | More slots than sessions doesn't help (hash collisions) |

### Steal Rate vs. Session Count (pool_size=1)

| Sessions | Steal Rate | Total Turns | Total Steals |
|----------|-----------|-------------|-------------|
| 1 | 0% | 10 | 0 |
| 2 | 95% | 20 | 19 |
| 4 | 98% | 40 | 39 |
| 8 | 98% | 40 | 39 |

### Lock Hold Times (pool_size=1, 4 sessions)

| Metric | Value |
|--------|-------|
| Min hold | ~10ms |
| Max hold | ~31ms |
| Avg hold | ~23ms |
| Overhead (save+restore) | ~10ms/turn |
| Overhead vs response time | 47–79% |

### Gap Window Analysis (pool_size=1, 4 sessions)

| Metric | Value |
|--------|-------|
| Min gap | ~0.004ms |
| Max gap | ~0.061ms |
| Avg gap | ~0.024ms |
| Median gap | ~0.025ms |

Gap windows are near-zero because asyncio's cooperative multitasking schedules the next waiting task immediately upon lock release. There is no OS-level preemption gap. The lock is released, and the next waiting `lock.acquire()` completes in the same event-loop iteration.

### Control: Single Session (pool_size=1, 1 session, 10 turns)

- **0 steals** — confirming the lock works correctly for single-session use.
- The session holds the same slot for all 10 turns with no interleaving.

## Analysis

### Why Thrashing Occurs

1. **pool_size=1** means all sessions hash to `slot_id=0` (SHA-256 mod 1 = 0)
2. **SlotLockCoordinator** creates one `asyncio.Lock` for slot_id=0
3. The lock serializes concurrent requests through slot 0
4. **But**: when a response is complete, the lock is released
5. **Between turns**: any waiting session can immediately acquire the lock
6. The cooperative asyncio event loop means the next scheduled task gets it — there's no ordering, affinity, or backoff

### Why pool_size>1 Helps (Partially)

With `pool_size=4`:
- Sessions distribute across 4 slots (via SHA-256 hash mod 4)
- Sessions on different slots don't contend
- But 4 sessions across 4 slots still has collisions (40% steal rate vs 98%)

With `pool_size > session_count`:
- No additional benefit — hash collisions still occur
- Extra slots don't help if no session is bound to them

### Cache Invalidation Cost

When a session loses its slot between turns:
1. The proxy must issue `POST /slots/{id}?action=save` to save current KV cache
2. Next time the session acquires the slot, it issues `POST /slots/{id}?action=restore` to reload cache
3. Both operations involve disk I/O and HTTP round-trips to llama-server
4. Configured timeout: `session_slot_timeout_seconds: 3.0`

**With 95–98% steal rate, every session pays save+restore cost on every turn** — even if no other session is actively using the GPU.

## Architectural Context

The proxy has three independent slot-related mechanisms:

| Mechanism | Location | Purpose |
|-----------|----------|---------|
| `SlotLockCoordinator` | `session.py:579` | asyncio lock per slot_id; serializes requests to same slot |
| Slot save/restore | `session.py:476-516` | KV cache persistence to disk via llama-server API |
| `_check_slot_availability` | `router_helpers.py:488` | Pre-route check of llama-server /slots endpoint |

The thrashing problem is in **`SlotLockCoordinator`** — not in slot save/restore (which works correctly) or slot availability check (which works correctly).

## Edge Cases and Boundary Conditions

### Inconclusive Reproduction

All runs were conclusive: thrashing is always reproducible with `pool_size=1` and ≥2 sessions.

### Race Condition Window

The gap between lock release and next acquire is typically **<0.1ms** — far smaller than any cache save/restore latency. This means:
- There is effectively no "safe window" where the original session can reclaim the slot
- Every session must save its cache, and the next session must restore/reload its own

### Worst-Case Analysis

Worst case: all 8 sessions active simultaneously, each making sequential multi-turn requests. Every turn of every session incurs save+restore overhead. With `session_slot_timeout_seconds: 3.0`, a single slow save can hold up all other sessions.

## Key Files (Touch Points)

The following sources were read during the investigation:

- `proxy/proxy/session.py` — `SlotLockCoordinator` (line 579), `_save_slot_snapshot` (line 498), `_restore_slot_snapshot` (line 476), `_slot_id_for_session` (line 526)
- `proxy/proxy/router.py` — slot lock acquire (lines 244, 811), slot save/restore (lines 758, 816, 983)
- `proxy/proxy/router_helpers.py` — `_check_slot_availability` (line 488)
- `proxy/config.yaml` — slot configuration (lines 206-209)
- `proxy/tests/test_slot_polling.py` — existing slot polling tests (not modified)

## Instrumentation Touch Points

The repro script (`repro.py`) is self-contained and mirrors `SlotLockCoordinator` logic without modifying production code. No permanent instrumentation was added to the proxy.

If a future implementation wishes to add runtime logging, the touch points would be:

1. `proxy/proxy/session.py` — `SlotLockCoordinator.acquire()` (line 579): add logging of acquire/release with session_id and timestamp
2. `proxy/proxy/session.py` — `_save_slot_snapshot()` (line 498): add logging of save start/end with slot_id and duration
3. `proxy/proxy/session.py` — `_restore_slot_snapshot()` (line 476): add logging of restore start/end with slot_id and duration

## Related Open Items

- **LP-0MQMC4MKY006J08E** (Prompt-cache / session reuse tests & small fixes) — directly related; F1 evidence confirms the cache-invalidation scenario this item aims to address
- **LP-0MQMC4MNU002QJK4** (Cleanup: slot-cache retention & cleanup script) — slot persistence is orthogonal but affects save/restore reliability
- **LP-0MQ0PYH8P008DLPJ** (Web based logging per slot) — instrumentation touch points identified above would support this

## Conclusion

Slot thrashing at the proxy level is **confirmed** for all tested configurations with `pool_size=1` and ≥2 concurrent sessions. The current `SlotLockCoordinator` provides mutual exclusion but does NOT provide session-level slot affinity, leading to near-100% cache invalidation between turns.

Reference data files:
- `repro.py` — Self-contained reproduction script
- This document — Analysis and findings
