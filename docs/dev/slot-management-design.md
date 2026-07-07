# Slot Management Design Proposal

**Date:** 2026-06-23
**Work Item:** LP-0MQR7AYRA006IU0P (Slot management design proposal)
**Parent:** LP-0MQR0780Z006TLX6 (Improve slot management)
**Depends on:** LP-0MQR75QYM001HAUB (Slot thrashing reproduction & evidence)

> This document is a **design-only proposal**. No implementation code is included.
> Implementation is deferred to follow-up work items created from the roadmap
> in F3.

## References

- [F1: Slot Thrashing Evidence](../slot-thrashing-investigation/findings.md) —
  empirical confirmation of 95–98% steal rate with `pool_size=1`
- Session Routing Regression Suite (LP-0MPU11WBN004MFQ8)
- Observability: slot-reject signal verification (LP-0MPYMVEYO0015J5L)
- Prompt-cache / session reuse tests & small fixes (LP-0MQMC4MKY006J08E)
- Cleanup: slot-cache retention & cleanup script (LP-0MQMC4MNU002QJK4)
- Web based logging per slot (LP-0MQ0PYH8P008DLPJ)

---

## 1. Problem Statement

### 1.1 Observed Behavior (F1 Findings)

With the current `session_slot_pool_size: 1` configuration, **95–98% of all
turn transitions** between ≥2 concurrent sessions result in a **different
session acquiring the slot** — the previous session's KV cache snapshot on
disk is immediately overwritten by the next session's save/restore cycle.

| Sessions | pool_size=1 | pool_size=2 | pool_size=4 |
|----------|-------------|-------------|-------------|
| 1        | 0%          | 0%          | 0%          |
| 2        | 95%         | —           | —           |
| 4        | 98%         | 90%         | 40%         |
| 8        | 98%         | —           | 43%         |

### 1.2 Root Cause

The `SlotLockCoordinator` (in `proxy/proxy/session.py:579`) provides
**mutual exclusion** but does **not** provide **session-level slot affinity**:

1. All sessions hash to `slot_id=0` (SHA-256 mod `pool_size=1`)
2. A single `asyncio.Lock` serializes concurrent requests through slot 0
3. When a response stream completes, the lock is released
4. Any waiting session can acquire the lock — there's no ordering guarantee
5. The gap between release and next acquire is typically **<0.1ms** due to
   asyncio's cooperative scheduling — there is **no safe window** for the
   original session to reclaim the slot

### 1.3 Cost

Every turn of every session incurs:
- **Save cost**: `POST /slots/{id}?action=save` — writes KV cache snapshot
  to disk (disk I/O + HTTP round-trip to llama-server)
- **Restore cost**: `POST /slots/{id}?action=restore` — reads KV cache
  snapshot from disk
- Configured timeout: `session_slot_timeout_seconds: 3.0`

**Every session pays this cost on every turn, even when no other session is
actively using the GPU.**

### 1.4 Design Goals

1. **Reduce cache invalidation**: Sessions should hold their slot between
   turns unless another session genuinely needs it
2. **Fairness**: All 1–8 agents should get reasonable throughput
3. **Minimal proxy changes**: No llama-server modifications
4. **Backward compatibility**: Existing 503/Retry-After clients must not break
5. **Single-GPU constraint**: Solutions must acknowledge that only one
   inference can run at a time

---

## 2. Approach A: Session-Level Slot Reservation

### 2.1 Concept

After processing a session's request, the proxy **reserves** the slot for that
session for a configurable **idle timeout** (e.g., 2–10s). During the
reservation window, if the same session's next request arrives, it reclaims
the slot with cache intact. If the timeout expires, the slot is released for
any waiting session.

The reservation is **preemptible**: if another session's request arrives and
no other slots are available, the proxy *may* preempt the reservation (with a
warning log), forcing the original session to re-restore its cache.

### 2.2 Implementation Sketch

```
SlotLockCoordinator enhancement:
  - Maintain a dict: slot_id -> {reserved_session_id, expiry_timestamp}
  - On lock acquire:
    - If slot is reserved for this session: grant immediately (cache valid)
    - If slot is reserved for another session AND timeout expired:
      release reservation, grant to new session
    - If slot is reserved for another session AND timeout NOT expired:
      block/wait until timeout or preemption
  - On lock release:
    - Set reservation for {session_id, now + idle_timeout}
    - Do not release the asyncio.Lock yet — hold it until timeout or
      preemption
  - Preemption logic:
    - If waiting_queue has items AND reservation timeout not expired:
      wait for either timeout or explicit preemption signal
    - Preemption: log warning, release reservation, grant to waiter
```

### 2.3 Pros

| Aspect | Assessment |
|--------|-----------|
| Cache preservation | **Excellent for active sessions** — cache survives between turns if next request arrives within timeout |
| Fairness | **Moderate** — depends on timeout tuning; long timeouts starve other sessions |
| Implementation complexity | **Medium** — adds reservation state machine to SlotLockCoordinator (~100–150 lines) |
| Backward compatibility | **Good** — no API change; existing clients work as-is |
| Compatibility with existing machinery | **Good** — single-flight, delta ingestion, session header resolution untouched |

### 2.4 Cons

| Aspect | Assessment |
|--------|-----------|
| Timeout tuning | **Tricky** — too short = thrashing returns, too long = wasted capacity with 1–8 agents sharing 1 GPU |
| Session-end detection | All sessions look "active" to the proxy; heuristic is needed |
| Preemption complexity | Edge cases: what if 3 sessions all arrive within timeout window? |
| GPU serialization | Reservation doesn't help if GPU is busy — queued requests still wait |

### 2.5 Effort Estimate

| Component | Effort |
|-----------|--------|
| Reservation state machine in SlotLockCoordinator | **M** (4–8h) |
| Preemption & waiting queue logic | **M** (4–8h) |
| Config: idle_timeout, preemption toggle | **S** (1–2h) |
| Unit tests | **M** (4–6h) |
| Integration tests | **S** (2–4h) |
| **Total** | **~15–28h** |

---

## 3. Approach B: Queue-Based Slot Allocation

### 3.1 Concept

Requests wait in a **FIFO (fair) queue** managed by the proxy, rather than
all retrying independently. When a slot becomes free, the next waiting request
gets it in order. Combined with the existing `_check_slot_availability` /
503/Retry-After mechanism, this provides deterministic ordering.

### 3.2 Implementation Sketch

```
SlotAllocationQueue (new module or addition to SlotLockCoordinator):
  - asyncio.Queue per slot_id (or global FIFO)
  - On slot unavailable:
    - Instead of returning 503 immediately, enqueue the request
    - Return a promise/future that resolves when a slot is free
    - If queue exceeds max_depth, return 503 immediately
  - On slot release:
    - Dequeue next waiter, grant the slot
    - Waiter proceeds with restore (or no restore if first request)
  - Queue timeout:
    - If a waiter's timeout expires, remove from queue, return 503
```

### 3.3 Pros

| Aspect | Assessment |
|--------|-----------|
| Fairness | **Excellent** — strict FIFO guarantees no starvation |
| Implementation complexity | **Low-Medium** — asyncio.Queue is standard; ~80–120 lines |
| Backward compatibility | **Good** — 503 still returned when queue is full |
| Session-end detection | **Not needed** — queue resolves automatically |

### 3.4 Cons

| Aspect | Assessment |
|--------|-----------|
| Cache preservation | **Poor** — queue doesn't prevent inter-session cache invalidation between turns (still subject to the gap) |
| Queue depth management | With 8 concurrent agents, queue may grow deep; max depth is a tuning parameter |
| Latency in queue | Requests wait in queue while another session holds the slot doing inference — adds latency |
| Memory pressure | Queue entries hold request state (body, headers, session_id) |

### 3.5 Effort Estimate

| Component | Effort |
|-----------|--------|
| FIFO queue integration with SlotLockCoordinator | **S** (2–4h) |
| Queue timeout & max-depth config | **S** (1–2h) |
| 503 fallback when queue full | **S** (1–2h) |
| Unit tests | **S** (2–4h) |
| Integration tests | **S** (2–3h) |
| **Total** | **~8–15h** |

---

## 4. Approach C: Client-Side Backoff with Jitter & Coordinated Retry

### 4.1 Concept

Agents use randomized exponential backoff (with jitter) when retrying after a
503, reducing the probability of synchronized retries. Combined with a small
Retry-After window that varies per agent (via response header), this makes
concurrent retry storms less likely.

### 4.2 Implementation Sketch

```
Proxy changes:
  - Enhance _build_slot_exhaustion_response to include:
    - Per-session retry-after: base + random(0, jitter_window)
    - Backoff hint header: X-Backoff-Base: 1.0, X-Backoff-Jitter: 0.5

Client changes (each agent):
  - On 503: sleep(RETRY_AFTER + random(0, JITTER)) before retry
  - Exponential backoff on repeated 503s
  - Cap max backoff at configurable limit
```

### 4.3 Pros

| Aspect | Assessment |
|--------|-----------|
| Implementation complexity | **Very Low** on proxy side — only header changes |
| Backward compatibility | **Excellent** — existing clients work unchanged (they already retry on 503) |
| No new state | No proxy-side queue or reservation state to manage |
| Works with existing 503 mechanism | Naturally extends the existing `Retry-After` header |

### 4.4 Cons

| Aspect | Assessment |
|--------|-----------|
| Cache preservation | **None** — purely statistical, doesn't prevent inter-session cache invalidation |
| Starvation guarantee | **None** — purely statistical; worst-case, agents can still collide |
| Client-side changes needed | Each agent must implement the backoff strategy; inconsistent across agents |
| No deterministic ordering | Retry timing is random; no fairness guarantee across sessions |

### 4.5 Effort Estimate

| Component | Effort |
|-----------|--------|
| Proxy: Retry-After with per-session jitter | **XS** (1–2h) |
| Proxy: backoff hint headers | **XS** (0.5–1h) |
| Client implementation (per agent) | **S** (2–4h each) |
| Unit tests (proxy side) | **XS** (0.5–1h) |
| **Total (proxy only)** | **~2–5h** |

---

## 5. Approach D: Hybrid — Reservation + Limited Queue

### 5.1 Concept

**Recommended approach.** Combines the best of Approaches A and B:

1. **Reservation window** (2–5s): After a session's response, the proxy
   reserves the slot for that session. If the session's next request arrives
   within the window, it reclaims the slot with cache intact.
2. **Fair queue** (fallback): When the reservation window expires, the slot
   goes to the next waiting request (FIFO queue). If the queue is empty, the
   slot becomes available for any session.
3. **Preemption** (exception): If the queue exceeds `max_waiting` depth and
   a reservation is still active, the oldest waiting request can preempt the
   reservation (with a log warning).

### 5.2 Why Hybrid Over Pure Approaches

| Concern | Pure Reservation (A) | Pure Queue (B) | Hybrid (D) |
|---------|---------------------|----------------|------------|
| Cache preservation for fast follow-ups | ✅ Excellent | ❌ Poor | ✅ Excellent |
| Fairness under load | ❌ Starvation risk | ✅ Excellent | ✅ Good |
| Backward compatibility | ✅ Good | ✅ Good | ✅ Good |
| Implementation complexity | ⚠️ Medium | ✅ Low-Medium | ⚠️ Medium-High |
| No session-end detection needed | ❌ Needs heuristic | ✅ Not needed | ✅ Not needed (reservation timeout + queue) |

### 5.3 Implementation Sketch

```
SlotReservationQueue (new module or integration into SlotLockCoordinator):
  State per slot:
    - reserved_session_id: Optional[str]
    - reservation_expiry: Optional[float]  (monotonic time)
    - waiting_queue: asyncio.Queue[tuple[str, asyncio.Future]]

  acquire(slot_id, session_id):
    1. If slot reserved for this session AND within expiry:
       → Grant immediately (cache preserved)
    2. If slot reserved for another session AND within expiry:
       → Check waiting_queue depth
         - If depth >= MAX_WAITING: preempt reservation, grant to caller
         - Else: enqueue caller with a timeout; wait for signal
    3. If slot free (no reservation or expired):
       → Grant immediately
       → Start reservation timer for current session

  release(slot_id, session_id):
    1. Set reservation: {session_id, now + RESERVATION_TIMEOUT}
    2. If waiting_queue not empty:
       → If reservation active: don't release the asyncio.Lock yet
         (held until expiry or preemption)
       → Start a background task that releases at expiry
    3. If queue has depth > 0 AND reservation active:
       → Wake the oldest waiter, grant slot
```

### 5.4 Pros

| Aspect | Assessment |
|--------|-----------|
| Cache preservation | **Excellent for fast follow-ups** — the common case for multi-turn agents |
| Fairness under load | **Good** — reservation gives way to FIFO queue on idle |
| Backward compatibility | **Good** — no API changes; existing clients work |
| Session-end detection | **Not needed** — reservation timeout + queue handles all cases |
| Works with 1–8 agents / 1 GPU | ✅ — reservation window matches typical agent "thinking time" (0.5–5s) |

### 5.5 Cons

| Aspect | Assessment |
|--------|-----------|
| Implementation complexity | **Medium-High** — combines reservation state machine + queue + preemption (~200–300 lines) |
| Timeout tuning | Reservation window is workload-dependent; may need auto-tuning or operator adjustment |
| Edge cases | Reservation + preemption + queue interactions need careful state machine design |
| Preemption cost | When preempting a reservation, the evicted session pays save+restore cost |

### 5.6 Effort Estimate

| Component | Effort |
|-----------|--------|
| SlotReservationQueue state machine | **L** (8–12h) |
| Reservation + preemption logic | **M** (4–6h) |
| FIFO queue integration | **M** (4–6h) |
| Config: reservation_timeout, max_waiting, preemption toggle | **S** (1–2h) |
| Unit tests | **L** (6–10h) |
| Integration tests | **M** (4–6h) |
| **Total** | **~27–42h** |

---

## 6. Recommendation

**Approach D: Hybrid (Reservation + Limited Queue)** is the recommended
approach.

### 6.1 Rationale

1. **Solves the primary problem**: The 95–98% cache invalidation rate drops to
   near-zero for sessions that make rapid follow-up requests (within the
   reservation window). This is the common case for multi-turn agent
   conversations where "thinking time" between turns is 0.5–5s.

2. **Degrades gracefully**: When the reservation window expires, fair-queue
   behavior takes over — no starvation, no session-end detection needed. This
   is the fallback for sessions that go idle (e.g., the agent is waiting for
   a user response).

3. **Works within constraints**: No llama-server changes, no GPU changes,
   existing session machinery intact, backward-compatible with existing
   503/Retry-After clients.

4. **Single GPU**: With 1 GPU and pool_size=1, only one session can process
   at any moment. The hybrid approach ensures the most recently active session
   retains the slot — minimizing total save/restore overhead for the active
   workload.

5. **No session-end detection needed**: The reservation timeout naturally
   handles the "when is an agent done?" question. If the agent sends another
   request within N seconds, cache is preserved. If not, the slot goes to the
   next waiter. This avoids complex heuristics or API changes.

### 6.2 Key Design Parameters

| Parameter | Suggested Default | Tuning Guidance |
|-----------|------------------|-----------------|
| `slot_reservation_timeout_seconds` | **3.0** | Should match typical agent "thinking time" between turns. With 1–8 agents, 3s balances cache preservation vs. fairness. Increase if agents have long inter-turn processing, decrease if more than 2–3 agents are actively contending. |
| `slot_queue_max_depth` | **4** | Per-slot max waiting queue depth. With 8 max sessions and 1 slot, 4 provides a reasonable buffer. Excess requests get immediate 503. |
| `slot_preemption_enabled` | **True** | When queue exceeds max_depth, preempt the reservation. Prevents head-of-line blocking. |
| `slot_preemption_warn_only` | **False** | If True, log warning but don't preempt — queue fills up and new requests get 503. Safer but less throughput under severe load. |

### 6.3 1–8 Agent Caseload Behavior

| Scenario | Expected Behavior |
|----------|-----------------|
| **1 agent, fast turns** (<3s gap) | Cache preserved on every turn; no save/restore overhead |
| **2 agents, staggered turns** | Each agent holds the slot for ~3s after its turn; if the other agent's next request arrives within the gap, cache is preserved |
| **4 agents, rapid-fire requests** | Queue starts filling; first-come-first-served with 3s reservation windows. Cache preserved for agents with <3s inter-turn gaps |
| **8 agents, all active** | Queue reaches max_depth=4; excess requests get 503 with Retry-After. Reservation windows may be preempted by queue pressure. Cache preservation rate drops but stays better than 0% |

### 6.4 Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Reservation timeout too short (thrashing returns) | Start with 3s default; operators can tune via config |
| Reservation timeout too long (agents starve) | Queue fallback ensures no starvation; preemption provides escape valve |
| Implementation bugs in state machine | Comprehensive unit tests (see roadmap); start with reservation-only, add queue in second pass |
| Interference with existing slot work items | Coordinate with LP-0MQMC4MKY006J08E and LP-0MQMC4MNU002QJK4 (see roadmap) |

---

## 7. Session-End Detection Analysis

### 7.1 The Problem

The proxy cannot directly know when an agent has "finished" its turn. An agent
may:
- Be processing (thinking) and will send the next request shortly
- Be waiting for a user response (could be seconds to hours)
- Have crashed or disconnected

### 7.2 Approaches Considered

| Approach | Assessment |
|----------|-----------|
| **Explicit API signal** (e.g., `POST /session/{id}/release-slot`) | Reliable but requires client changes; all agents must implement. Adding a new endpoint increases surface area. |
| **Heartbeat/heuristic** (e.g., session last_active timestamp) | Approximate; requires per-session activity tracking. Already partially available via session manager. |
| **Idle timeout** | Simple, configurable. The recommended approach for the hybrid model — reservation timeout naturally serves as the idle timeout. |
| **LLM stop token detection** | Fragile — depends on response content parsing; misses non-stop-token completions |

### 7.3 Recommended Strategy

**Idle timeout (reservation window)** — incorporated into the recommended
hybrid approach. No explicit API signal is needed because:

1. The reservation timeout (default 3s) covers the common case of agent
   thinking time between turns
2. When the timeout expires, the slot goes to the next waiter via FIFO queue
3. No agent changes required
4. If the agent is still active (just slow), it gets a 503 on its next request
   and retries — the existing Retry-After mechanism works as-is

### 7.4 Edge Cases

| Edge Case | Behavior |
|-----------|----------|
| Agent sends request 3.1s after previous (just past timeout) | Gets 503, retries with Retry-After=5s. Next time, request arrives faster → cache preserved |
| Agent crashes mid-conversation | Reservation times out after 3s; slot goes to next waiter. No cleanup needed |
| Agent sends request while waiting in queue (duplicate) | Single-flight prevents duplicate processing; second request gets 429 |
| All 8 agents sending requests simultaneously | Queue fills to max_depth=4; 4 get queued, 4 get 503 with backoff. Preemption kicks in if queue stays full |

---

## 8. Pool Size Analysis

### 8.1 `pool_size > 1` Viability Assessment

**Question:** Can increasing `session_slot_pool_size` beyond 1 help, given
that only one inference can run at a time on a single GPU?

**Answer: Yes, partially** — but the hybrid approach (D) is a better solution.

### 8.2 Why pool_size > 1 Helps

| pool_size | 4 sessions steal rate | Mechanism |
|-----------|----------------------|-----------|
| 1 | 98% | All sessions contend for 1 slot |
| 2 | 90% | Sessions distribute across 2 locks; collisions still high |
| 3 | 78% | 3 locks for 4 sessions → fewer collisions |
| 4 | 40% | 4 locks for 4 sessions → best case (one slot per session) |
| 8 | 43% | 8 locks for 4 sessions → hash collisions still occur |

With `pool_size=4` and 4 sessions:
- Each session gets (probabilistically) its own slot
- No lock contention between sessions on different slots
- Steal rate drops from 98% to 40%

### 8.3 Why pool_size > 1 Alone Is Insufficient

1. **40% steal rate is still high** — 4 out of 10 turns still lose cache
2. **Hash collisions are random** — you can't guarantee a session maps to a
   unique slot
3. **With 8 sessions**, even pool_size=8 gives 43% steal rate (birthday
   problem)
4. **No ordering guarantee** — pool_size doesn't add reservation/affinity

### 8.4 Recommended Approach to pool_size

- **Keep `pool_size=1`** as the default
- **Do not increase pool_size** as a primary solution — it's a partial fix
- **The hybrid approach (D) is strictly better** because it provides
  session-level affinity regardless of `pool_size`
- If pool_size is increased in the future (e.g., multiple GPUs), the hybrid
  approach scales naturally

---

## 9. Compatibility with Existing Machinery

| Component | Impact of Hybrid Approach |
|-----------|-------------------------|
| **Single-flight coordination** | Unchanged — still serializes per-session requests |
| **Delta ingestion** | Unchanged — reservation window doesn't affect message merging |
| **Slot save/restore** | Unchanged — save/restore happens at same points; reservation only adds delay before release |
| **Session header resolution** | Unchanged |
| **503/Retry-After** | Unchanged — still returned when queue is full |
| **Slot availability check** | Unchanged — pre-route check still works |
| **Polling for slot state** | Unchanged |

---

## 10. Open Questions

1. **Reservation timeout default**: Is 3s appropriate for the actual agent
   workloads running against this proxy? Should be validated with production
   timing data.

2. **Preemption behavior**: Should preemption always be allowed, or should it
   require a config toggle? The design recommends a toggle with `True` as
   default.

3. **Logging granularity**: How much detail should be logged per
   reservation/preemption event? The design recommends `INFO` for
   preemption, `DEBUG` for reservation grant/release.

4. **Queue fairness**: Strict FIFO or weighted per-session fairness? FIFO is
   simpler and recommended for initial implementation.

---

## 11. Related Work Coordination

| Item | Relationship | Coordination Notes |
|------|-------------|-------------------|
| **LP-0MQMC4MKY006J08E** (Prompt-cache / session reuse tests) | Cache preservation tests should be updated to verify reservation behavior | Share reservation timeout config; update test fixtures to simulate within-timeout and after-timeout scenarios |
| **LP-0MQMC4MNU002QJK4** (Slot-cache retention & cleanup) | Slot persistence reliability affects save/restore cost | Ensure cleanup script doesn't delete active reservation snapshot files |
| **LP-0MQ0PYH8P008DLPJ** (Web based logging per slot) | Instrumentation touch points in F1 findings overlap | Reservation events (grant, release, preempt) are natural logging events for per-slot UI |
| **LP-0MQWXX17C005BX1E** (slot_save failed) | Improved error logging for `_call_slot_endpoint` — exception type names, non-200 response logging, and debug-level stack traces | Operators can now diagnose save/restore failures from proxy logs without requiring reproduction |
| **LP-0MPU11WBN004MFQ8** (Session Routing Regression Suite) | Regression coverage for slot coordination must be updated | Add tests for reservation timeout and queue behavior |

---

## 12. Implementation Roadmap

This section breaks the recommended **Approach D (Hybrid Reservation + Queue)**
into implementable sub-tasks with effort estimates. Implementation is **deferred**
to follow-up work items; these sub-tasks are **proposals**, not created items.

### Phase 1: Core Reservation (MVP)

Minimal viable implementation — only the reservation state machine without queue
fallback or preemption.

| # | Sub-task | Effort | Dependencies | Description |
|---|----------|--------|-------------|-------------|
| 1.1 | Add `SessionSlotReservation` data structure to `SlotLockCoordinator` | **S** (2–3h) | None | Simple dataclass with `session_id`, `expiry` fields; store in dict keyed by slot_id |
| 1.2 | Modify `acquire()` to check reservation before granting | **M** (3–5h) | 1.1 | If slot reserved for requesting session_id AND expiry not passed: grant immediately. If reserved for different session: block. If free: grant + set reservation. |
| 1.3 | Add reservation timeout config key `slot_reservation_timeout_seconds` | **XS** (0.5–1h) | 1.1 | New config in `proxy/config.yaml` and `server_config` dict. Default: 3.0. |
| 1.4 | Modify `release()` to set reservation instead of releasing immediately | **M** (4–6h) | 1.2, 1.3 | After lock-release body completes, set reservation timer. Hold asyncio.Lock until timeout or preemption. |
| 1.5 | Unit tests for reservation state machine | **M** (4–6h) | 1.2 | Test: within-timeout reclaim, after-timeout release, concurrent sessions, edge cases |
| 1.6 | Integration tests with mock slot save/restore | **M** (3–5h) | 1.5 | Verify save/restore is skipped for within-timeout reclaim; verify save+restore happens on cross-session switch |
| | **Phase 1 Total** | **~17–26h** | | |

### Phase 2: Fair Queue

Add FIFO waiting queue so that when reservation expires, the next waiting
session gets the slot deterministically.

| # | Sub-task | Effort | Dependencies | Description |
|---|----------|--------|-------------|-------------|
| 2.1 | Add `waiting_queue: asyncio.Queue` per slot | **S** (2–3h) | Phase 1 | Standard FIFO queue of `(session_id, asyncio.Future)` tuples |
| 2.2 | Enqueue callers when slot is reserved for another session | **M** (3–4h) | 2.1, 1.2 | On acquire: if slot reserved for other session AND within expiry, enqueue waiter. Return future that resolves when slot free. |
| 2.3 | Dequeue and grant on reservation expiry | **M** (3–4h) | 2.2 | Background task: when reservation timer fires, dequeue next waiter, resolve their future, grant slot |
| 2.4 | Config: `slot_queue_max_depth` with overflow → 503 | **S** (1–2h) | 2.2 | If queue exceeds max_depth, return 503 immediately (existing mechanism) |
| 2.5 | Unit tests for queue behavior | **M** (4–6h) | 2.3 | Test: FIFO ordering, max_depth overflow, concurrent enqueue/dequeue |
| 2.6 | Integration tests for queue + reservation interaction | **M** (3–5h) | 2.5 | Test: reservation expiry → next waiter gets slot, multiple waiters, timeout |
| | **Phase 2 Total** | **~16–24h** | | |

### Phase 3: Preemption (Optional)

Allow queue pressure to preempt an active reservation, preventing head-of-line
blocking under severe load.

| # | Sub-task | Effort | Dependencies | Description |
|---|----------|--------|-------------|-------------|
| 3.1 | Add preemption trigger when queue exceeds max_depth | **M** (3–5h) | 2.4 | When enqueuing and queue depth > max_depth, preempt the active reservation: release lock, log warning, grant to oldest waiter |
| 3.2 | Config: `slot_preemption_enabled` (bool) and `slot_preemption_warn_only` | **S** (1–2h) | 3.1 | Toggle preemption behavior; warn-only logs instead of preempting |
| 3.3 | Preemption metrics and logging | **S** (1–2h) | 3.1 | Counter for preemption events, gauge for queue depth, logging at INFO level |
| 3.4 | Unit tests for preemption logic | **M** (3–5h) | 3.1 | Test: preemption triggers correctly, warn-only mode, edge cases |
| 3.5 | Integration tests: preemption under load | **M** (3–5h) | 3.4 | Simulate 8 sessions hammering the proxy; verify preemption prevents queue overflow |
| | **Phase 3 Total** | **~11–19h** | | |

### Phase 4: Observability & Polish

Metrics, logging, and documentation for operators.

| # | Sub-task | Effort | Dependencies | Description |
|---|----------|--------|-------------|-------------|
| 4.1 | Add Prometheus metrics for reservation state | **S** (2–3h) | Phase 1 | Gauge: reserved slots count, queue depth. Counter: preemptions, reservation grants, timeouts |
| 4.2 | Add per-slot reservation logging (INFO/DEBUG) | **XS** (1–2h) | Phase 1 | Log grant/release/timeout/preemption events with session_id and slot_id |
| 4.3 | Update `proxy/config.yaml` with new config keys | **XS** (0.5h) | Phases 1–3 | Document all new slot config keys with defaults and descriptions |
| 4.4 | Update `proxy/README.md` with new slot management behavior | **S** (1–2h) | 4.3 | Document reservation timeout, queue behavior, and tuning guidance |
| 4.5 | Update `docs/dev/slot-management-design.md` with post-implementation lessons | **XS** (0.5–1h) | After testing | Reflect any design changes discovered during implementation |
| | **Phase 4 Total** | **~5–8.5h** | | |

### Total Implementation Effort

| Phase | Effort (hours) | Scope |
|-------|---------------|-------|
| Phase 1: Core Reservation | 17–26 | MVP — solves the primary thrashing problem |
| Phase 2: Fair Queue | 16–24 | Adds fairness under load |
| Phase 3: Preemption | 11–19 | Optional escape valve for severe load |
| Phase 4: Observability | 5–9 | Production readiness |
| **Total** | **49–78h** | Full recommended approach |

**Phase 1 alone** (17–26h) delivers the core value — reducing thrashing from
95–98% to near-zero for session with <3s inter-turn gaps — and can be shipped
independently. Phases 2–4 add robustness and production readiness.

---

## 13. Open Questions & Assumptions

### Open Questions

1. **What is the actual distribution of agent 'thinking time' between turns?**
   This directly affects the optimal `slot_reservation_timeout_seconds` value.
   Without production telemetry, the default of 3.0s is a reasonable starting
   point but should be validated.

2. **Should the reservation timeout be static or adaptive?**
   A static timeout (3s default) is simple but may not suit all workloads.
   An adaptive timeout that tracks per-session inter-request timing could
   auto-tune — but adds complexity that may not be justified.

3. **Is preemption always safe?**
   When preempting a reservation, the evicted session's slot save may not have
   completed. Should preemption wait for save to finish, or force-cancel?
   Recommendation: preemption should wait for the current save to complete
   (up to `session_slot_timeout_seconds`) before granting the slot.

4. **What happens to slot save/restore during reservation?**
   The current design assumes save happens *before* lock release (already the
   case). With reservation, the save still happens before the reservation is
   set. Restore happens on next acquire. No change to save/restore timing.

### Assumptions

1. The asyncio cooperative scheduling model is sufficient — no OS-level lock
   or semaphore is needed.
2. `pool_size=1` remains the default — the hybrid approach works correctly
   regardless of pool_size.
3. Existing 503/Retry-After clients will retry naturally — no client changes
   needed.
4. Save/restore latency is acceptable (~10ms each in the repro simulation).
5. Reservation state lives in-memory only (no persistence across proxy
   restarts).

### Risks

1. **Reservation state machine correctness**: The combination of reservation +
   queue + preemption has complex interleavings. Mitigation: Phase 1 (pure
   reservation) is simpler and can be validated before adding queue/preemption.
2. **Timeout sensitivity**: If the default 3s timeout is wrong, thrashing may
   not be meaningfully reduced. Mitigation: configurable timeout; operators can
   tune.
3. **Queue memory pressure**: With 8 concurrent agents, the queue holds at
   most `max_depth` (default 4) request objects in memory. Each object is
   ~1–10KB (headers + body). Mitigation: trivial memory cost (~40KB max).

---

## 14. Follow-Up Work Item Proposals

The following work items are **proposed** (NOT created) for implementating the
recommended approach. These are one-liners suitable for creating as child work
items of the parent epic.

| ID | Title | Phase | Estimated Effort |
|----|-------|-------|-----------------|
| P1 | Add SessionSlotReservation data structure to SlotLockCoordinator | 1 | S (2–3h) |
| P2 | Implement reservation-aware lock acquire/release in SlotLockCoordinator | 1 | M (7–11h) |
| P3 | Add slot_reservation_timeout_seconds config key | 1 | XS (0.5–1h) |
| P4 | Add FIFO waiting queue to SlotLockCoordinator | 2 | S (2–3h) |
| P5 | Implement enqueue/dequeue with reservation expiry handoff | 2 | M (6–8h) |
| P6 | Add slot_queue_max_depth config and 503 overflow | 2 | S (1–2h) |
| P7 | Implement preemption trigger when queue exceeds max_depth | 3 | M (3–5h) |
| P8 | Add preemption config toggles | 3 | S (1–2h) |
| P9 | Add Prometheus metrics for reservation/queue state | 4 | S (2–3h) |
| P10 | Update config.yaml and README with new slot config | 4 | S (1–2h) |

### Interleaving with Related Open Items

| Related Item | Recommended Sequence |
|-------------|--------------------|
| **LP-0MQMC4MKY006J08E** (Prompt-cache / session reuse tests) | Execute after Phase 1 (P1–P3) — tests should verify cache preservation under reservation |
| **LP-0MQMC4MNU002QJK4** (Slot-cache retention & cleanup) | Can be done in parallel with any phase — orthogonal to reservation |
| **LP-0MQ0PYH8P008DLPJ** (Web based logging per slot) | Coordinate with Phase 4 (P9) — both touch metrics/logging infrastructure |
