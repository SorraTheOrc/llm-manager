---
type: source
title: "Observation: Slot thrashing confirmed at 95-98% rate with pool_size=1"
slug: obs-2026-06-23-slot-thrashing-confirmed-at-95-98-rate-with-pool-size-1
status: observation
created: 2026-06-23
updated: 2026-06-23
relevance: high
observed_at: 2026-06-23T23:27:37.337Z
source_context: "LP-0MQR75QYM001HAUB: Slot thrashing reproduction & evidence"
---
# ⭐ Observation: Slot thrashing confirmed at 95-98% rate with pool_size=1
Investigation F1 (LP-0MQR75QYM001HAUB) confirmed slot thrashing is reproducible at the proxy level. With pool_size=1 and >=2 concurrent sessions, 95-98% of turn transitions result in a different session acquiring the slot. With pool_size=4 (matching session count), steal rate drops to ~40%. Single-session control shows 0% steals. The root cause is SlotLockCoordinator providing mutual exclusion without session-level slot affinity — the asyncio.Lock is released between turns, allowing any waiting session to acquire. Evidence committed at docs/dev/slot-thrashing-investigation/.
*Relevance: high*

*Context: LP-0MQR75QYM001HAUB: Slot thrashing reproduction & evidence*
---
*Observed: 2026-06-23T23:27:37.337Z*