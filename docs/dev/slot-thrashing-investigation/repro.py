#!/usr/bin/env python3
"""
Slot thrashing reproduction script.

Simulates ≥2 concurrent sessions through the proxy's SlotLockCoordinator
with pool_size=1, demonstrating the window between response-sent and
next-request-arrived where another session can acquire the slot.

This is a ONE-OFF repro script for investigation purposes only. It is NOT
a permanent test and should NOT be added to the test suite.

Usage (from repo root):
    python3 docs/dev/slot-thrashing-investigation/repro.py \
        [--pool-size N] [--sessions N] [--turns N] \
        [--think-min S] [--think-max S] \
        [--save-cost MS] [--restore-cost MS] \
        [--response-min S] [--response-max S] \
        [--json]

Results:
    - Steal rate: percentage of turn transitions where a different session
      acquires the slot (measures cache invalidation between turns).
    - Save/restore overhead: estimated proxy-level cost of save+restore
      per session per turn, aggregated across all sessions.
    - Gap window analysis: time between release and next acquire.

Requirements:
    - Python 3.10+ (for `match` statement if used; this script uses if/else)
    - No live GPU needed; only tests the proxy-level lock coordination.

Copyright (c) 2026. One-off investigation; not part of the permanent test suite.
"""

import asyncio
import hashlib
import json
import random
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional


# ===================================================================
# Instrumented SlotLockCoordinator (mirrors proxy/proxy/session.py:579)
# ===================================================================
# This is an instrumented copy so the repro script is self-contained
# and does not require adding/removing logging in production code.
# Changes from production: added logging callbacks for acquire/release events.


@dataclass
class LockEvent:
    """A single lock acquire or release event."""
    session_id: str
    slot_id: int
    event_type: str  # "acquire" or "release"
    turn: int
    timestamp: float  # time.monotonic()
    phase: str = ""   # "thinking", "restore", "response", "save"


@dataclass
class PhaseTiming:
    """Timing for a specific phase of a simulated turn."""
    session_id: str
    turn: int
    phase: str
    start: float
    end: float
    duration_s: float


@dataclass
class SessionTimeline:
    """Tracks the timeline of a single simulated session."""
    session_id: str
    slot_id: int
    events: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    turn_start: Optional[float] = None
    turn_end: Optional[float] = None
    steals_detected: int = 0
    total_response_time: float = 0.0
    total_thinking_time: float = 0.0
    total_save_cost: float = 0.0
    total_restore_cost: float = 0.0
    turn_completions: int = 0

    def record_lock(self, event_type: str, turn: int):
        ev = LockEvent(
            session_id=self.session_id,
            slot_id=self.slot_id,
            event_type=event_type,
            turn=turn,
            timestamp=time.monotonic(),
        )
        self.events.append(ev)

    def record_phase(self, turn: int, phase: str, start: float, end: float):
        pt = PhaseTiming(
            session_id=self.session_id,
            turn=turn,
            phase=phase,
            start=start,
            end=end,
            duration_s=end - start,
        )
        self.phases.append(pt)


class InstrumentedSlotLockCoordinator:
    """
    Mirrors proxy/proxy/session.py::SlotLockCoordinator but with
    event logging callbacks so we can measure gap windows.
    """

    def __init__(self, timelines: dict[str, SessionTimeline], pool_size: int = 1):
        self._lock = asyncio.Lock()
        self._locks: dict[int, asyncio.Lock] = {}
        self._timelines = timelines
        self._pool_size = pool_size

    @staticmethod
    def _slot_id_for_session(session_id: str, pool_size: int) -> Optional[int]:
        """Mirrors proxy/proxy/session.py::_slot_id_for_session."""
        if not session_id or pool_size <= 0:
            return None
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % int(pool_size)

    def acquire(self, slot_id: Optional[int], session_id: str, turn: int):
        @asynccontextmanager
        async def _guard():
            if slot_id is None:
                yield
                return
            async with self._lock:
                lock = self._locks.get(slot_id)
                if lock is None:
                    lock = asyncio.Lock()
                    self._locks[slot_id] = lock
            await lock.acquire()
            try:
                tl = self._timelines.get(session_id)
                if tl:
                    tl.record_lock("acquire", turn)
                yield
            finally:
                lock.release()
                tl = self._timelines.get(session_id)
                if tl:
                    tl.record_lock("release", turn)

        return _guard()


# ===================================================================
# Simulated Session
# ===================================================================


async def simulated_session(
    coordinator: InstrumentedSlotLockCoordinator,
    session_id: str,
    timeline: SessionTimeline,
    num_turns: int,
    think_min: float,
    think_max: float,
    response_time_min: float,
    response_time_max: float,
    restore_cost_s: float,
    save_cost_s: float,
):
    """
    Simulates an agent making multi-turn requests through the proxy.

    Each "turn":
    1. Agent "thinking" delay (simulates agent processing between turns)
    2. Acquire slot lock (proxy-side coordination)
    3. Slot restore (read KV cache snapshot from disk; cost: ~0.1-3s)
    4. HTTP round-trip to llama-server + inference (response delay)
    5. Slot save (write KV cache snapshot to disk; cost: ~0.1-3s)
    6. Release slot lock (lock released after response sent and slot saved)

    The gap between lock release (step 6) and next lock acquire (step 2)
    of the next turn for the same session is where other sessions can
    acquire the lock, causing cache invalidation.
    """
    for turn in range(1, num_turns + 1):
        # Step 1: Agent "thinking" between turns
        if turn > 1:
            thinking_delay = random.uniform(think_min, think_max)
            think_start = time.monotonic()
            await asyncio.sleep(thinking_delay)
            think_end = time.monotonic()
            timeline.total_thinking_time += think_end - think_start
            timeline.record_phase(turn, "thinking", think_start, think_end)

        # Step 2: Acquire slot lock
        guard = coordinator.acquire(timeline.slot_id, session_id, turn)
        async with guard:
            # Step 3: Slot restore (reading saved KV cache from disk)
            restore_start = time.monotonic()
            await asyncio.sleep(restore_cost_s)
            restore_end = time.monotonic()
            timeline.total_restore_cost += restore_end - restore_start
            timeline.record_phase(turn, "restore", restore_start, restore_end)

            # Step 4: Simulate model inference response
            response_start = time.monotonic()
            response_delay = random.uniform(response_time_min, response_time_max)
            await asyncio.sleep(response_delay)
            response_end = time.monotonic()
            timeline.total_response_time += response_end - response_start
            timeline.record_phase(turn, "response", response_start, response_end)

            # Step 5: Slot save (writing KV cache to disk)
            save_start = time.monotonic()
            await asyncio.sleep(save_cost_s)
            save_end = time.monotonic()
            timeline.total_save_cost += save_end - save_start
            timeline.record_phase(turn, "save", save_start, save_end)

            # Step 6: Lock released at block exit
            timeline.turn_completions += 1

    # Mark that this session is done
    timeline.turn_start = None


# ===================================================================
# Analysis
# ===================================================================


@dataclass
class WindowEvent:
    """A detected window where slot stealing could occur."""
    slot_id: int
    stolen_from_session: str
    stolen_from_turn: int
    stealing_session: str
    stealing_turn: int
    gap_start: float
    gap_end: float
    gap_duration: float


def detect_slot_stealing(
    timelines: list[SessionTimeline],
) -> tuple[list[WindowEvent], dict[str, float]]:
    """
    Analyze all lock events to detect inter-session slot stealing.

    A "stealing window" occurs when:
    - Session A releases the lock (turn N complete)
    - Session B acquires the lock before Session A's next request
    """
    # Collect all events sorted by timestamp
    all_events: list[LockEvent] = []
    for tl in timelines:
        all_events.extend(tl.events)

    all_events.sort(key=lambda e: e.timestamp)

    # Track per-slot last release
    last_release: dict[int, tuple[str, int, float]] = {}
    windows: list[WindowEvent] = []

    # Track per-session turn statistics
    session_turn_times: dict[str, list[float]] = {}
    for tl in timelines:
        session_turn_times[tl.session_id] = []

    for ev in all_events:
        if ev.event_type == "release":
            last_release[ev.slot_id] = (ev.session_id, ev.turn, ev.timestamp)
        elif ev.event_type == "acquire":
            # Check if someone else released this slot since this session's last acquire
            if ev.slot_id in last_release:
                prev_sess, prev_turn, release_ts = last_release[ev.slot_id]
                if prev_sess != ev.session_id:
                    window_dur = ev.timestamp - release_ts
                    windows.append(WindowEvent(
                        slot_id=ev.slot_id,
                        stolen_from_session=prev_sess,
                        stolen_from_turn=prev_turn,
                        stealing_session=ev.session_id,
                        stealing_turn=ev.turn,
                        gap_start=release_ts,
                        gap_end=ev.timestamp,
                        gap_duration=window_dur,
                    ))

    # Compute hold times (time each session holds the lock per turn)
    hold_times: dict[str, list[float]] = {}
    for tl in timelines:
        hold_times[tl.session_id] = []
        for i in range(0, len(tl.events) - 1, 2):
            if i + 1 < len(tl.events):
                acq = tl.events[i]
                rel = tl.events[i + 1]
                if acq.event_type == "acquire" and rel.event_type == "release":
                    hold_times[acq.session_id].append(rel.timestamp - acq.timestamp)

    # Compute cache-invalidation probability
    # "Cache valid" = session acquires the lock two turns in a row without
    # another session in between
    consecutive_turns: dict[str, int] = {}
    cache_invalidated: dict[str, int] = {}
    for tl in timelines:
        consecutive_turns[tl.session_id] = 0
        cache_invalidated[tl.session_id] = 0

    sorted_by_session: dict[str, list[LockEvent]] = {}
    for tl in timelines:
        sorted_by_session[tl.session_id] = [e for e in all_events if e.session_id == tl.session_id]

    for sess_id, sess_events in sorted_by_session.items():
        acquires = [e for e in sess_events if e.event_type == "acquire"]
        for acquire in acquires:
            # Check who held the lock just before this acquire
            # (reverse search for nearest release or acquire)
            idx = all_events.index(acquire)
            # Find nearest previous event for this slot
            prev_ev = None
            for j in range(idx - 1, -1, -1):
                if all_events[j].slot_id == acquire.slot_id:
                    prev_ev = all_events[j]
                    break
            if prev_ev and prev_ev.session_id != sess_id:
                cache_invalidated[sess_id] = cache_invalidated.get(sess_id, 0) + 1

    return windows, hold_times, cache_invalidated


# ===================================================================
# Report generation
# ===================================================================


def generate_report(
    timelines: list[SessionTimeline],
    windows: list[WindowEvent],
    hold_times: dict[str, list[float]],
    cache_invalidated: dict[str, int],
    pool_size: int,
    num_sessions: int,
    num_turns: int,
    think_min: float,
    think_max: float,
    save_cost_s: float,
    restore_cost_s: float,
):
    """Generate a structured report of findings."""
    print("=" * 72)
    print("SLOT THRASHING REPRODUCTION REPORT")
    print("=" * 72)
    print(f"\nConfiguration:")
    print(f"  Pool size:                     {pool_size}")
    print(f"  Sessions:                      {num_sessions}")
    print(f"  Turns per session:             {num_turns}")
    print(f"  Think time range:              [{think_min:.2f}s, {think_max:.2f}s]")
    print(f"  Simulated restore cost:        {restore_cost_s*1000:.1f}ms")
    print(f"  Simulated save cost:           {save_cost_s*1000:.1f}ms")
    print()

    total_turns = num_sessions * num_turns
    total_steals = len(windows)
    steal_rate = (total_steals / total_turns * 100) if total_turns > 0 else 0.0

    print(f"=== RESULTS ===")
    print(f"  Total turns completed:         {total_turns}")
    print(f"  Inter-session steals detected: {total_steals}")
    print(f"  Steal rate (per turn):         {steal_rate:.1f}%")
    print()

    if windows:
        gap_durations = [w.gap_duration for w in windows]
        min_gap = min(gap_durations)
        max_gap = max(gap_durations)
        avg_gap = sum(gap_durations) / len(gap_durations)
        median_gap = sorted(gap_durations)[len(gap_durations) // 2]

        print(f"  Gap window analysis (time between release and next acquire):")
        print(f"    Min gap:   {min_gap*1000:.4f}ms")
        print(f"    Max gap:   {max_gap*1000:.4f}ms")
        print(f"    Avg gap:   {avg_gap*1000:.4f}ms")
        print(f"    Median gap: {median_gap*1000:.4f}ms")
        print()

        print(f"  Steal events (first 20):")
        print(f"    {'Slot':<6} {'From Session':<28} {'Turn':<6} {'To Session':<28} {'Turn':<6} {'Gap (ms)':<10}")
        print(f"    {'-'*6} {'-'*28} {'-'*6} {'-'*28} {'-'*6} {'-'*10}")
        for w in windows[:20]:
            print(f"    {w.slot_id:<6} {w.stolen_from_session:<28} {w.stolen_from_turn:<6} {w.stealing_session:<28} {w.stealing_turn:<6} {w.gap_duration*1000:<10.4f}")
        if len(windows) > 20:
            print(f"    ... ({len(windows) - 20} more events)")
        print()
    else:
        print(f"  NO INTER-SESSION STEALING DETECTED")
        print()

    # Lock hold time statistics
    all_holds = [h for holds in hold_times.values() for h in holds]
    if all_holds:
        print(f"  Lock hold time per turn:")
        print(f"    Min hold:   {min(all_holds)*1000:.2f}ms")
        print(f"    Max hold:   {max(all_holds)*1000:.2f}ms")
        print(f"    Avg hold:   {sum(all_holds)/len(all_holds)*1000:.2f}ms")
        print()

    # Per-session statistics
    print("  Per-Session Statistics:")
    print(f"    {'Session':<28} {'Slot':<6} {'Cache Lost':<12} {'Avg Resp':<14} {'Avg Think':<12} {'Save+Restore':<14}")
    print(f"    {'-'*28} {'-'*6} {'-'*12} {'-'*14} {'-'*12} {'-'*14}")
    for tl in timelines:
        avg_resp = tl.total_response_time / num_turns if num_turns > 0 else 0
        avg_think = tl.total_thinking_time / max(num_turns - 1, 1) if num_turns > 1 else 0
        total_save_restore = tl.total_save_cost + tl.total_restore_cost
        cache_lost = cache_invalidated.get(tl.session_id, 0)
        print(f"    {tl.session_id:<28} {tl.slot_id:<6} {cache_lost:<12} {avg_resp*1000:<13.1f}ms {avg_think*1000:<11.1f}ms {total_save_restore*1000:<13.1f}ms")
    print()

    # Cost analysis
    total_save = sum(tl.total_save_cost for tl in timelines)
    total_restore = sum(tl.total_restore_cost for tl in timelines)
    total_overhead = total_save + total_restore
    total_think = sum(tl.total_thinking_time for tl in timelines)
    total_resp = sum(tl.total_response_time for tl in timelines)

    print(f"  Aggregate Overhead Analysis:")
    print(f"    Total save cost:       {total_save*1000:.1f}ms ({total_save:.3f}s)")
    print(f"    Total restore cost:    {total_restore*1000:.1f}ms ({total_restore:.3f}s)")
    print(f"    Total save+restore:    {total_overhead*1000:.1f}ms ({total_overhead:.3f}s)")
    print(f"    Total response time:   {total_resp*1000:.1f}ms ({total_resp:.3f}s)")
    print(f"    Overhead ratio:        {total_overhead/total_resp*100:.1f}% of response time")
    print(f"    Average per-turn cost: {total_overhead/total_turns*1000:.1f}ms")
    print()

    # Per-session steal count
    stolen_from: dict[str, int] = {}
    for w in windows:
        stolen_from[w.stolen_from_session] = stolen_from.get(w.stolen_from_session, 0) + 1

    print(f"  Per-Session Cache Losses (slot stolen between turns):")
    for tl in timelines:
        losses = stolen_from.get(tl.session_id, 0)
        turnover = losses / num_turns * 100 if num_turns > 0 else 0
        print(f"    {tl.session_id:<28} {losses}/{num_turns} turns ({turnover:.0f}%)")

    print()
    print("=" * 72)

    return {
        "pool_size": pool_size,
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "total_turns": total_turns,
        "total_steals": total_steals,
        "steal_rate_percent": round(steal_rate, 1),
        "gap_ms": {
            "min": round(min_gap * 1000, 4) if windows else 0,
            "max": round(max_gap * 1000, 4) if windows else 0,
            "avg": round(avg_gap * 1000, 4) if windows else 0,
            "median": round(median_gap * 1000, 4) if windows else 0,
        } if windows else None,
        "hold_time_ms": {
            "min": round(min(all_holds) * 1000, 2) if all_holds else 0,
            "max": round(max(all_holds) * 1000, 2) if all_holds else 0,
            "avg": round(sum(all_holds) / len(all_holds) * 1000, 2) if all_holds else 0,
        } if all_holds else None,
        "overhead": {
            "total_save_ms": round(total_save * 1000, 1),
            "total_restore_ms": round(total_restore * 1000, 1),
            "total_save_restore_ms": round(total_overhead * 1000, 1),
            "overhead_vs_response_pct": round(total_overhead / total_resp * 100, 1) if total_resp > 0 else 0,
            "avg_per_turn_ms": round(total_overhead / total_turns * 1000, 1) if total_turns > 0 else 0,
        },
    }


# ===================================================================
# Main
# ===================================================================


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Reproduce slot thrashing between concurrent sessions"
    )
    parser.add_argument("--pool-size", type=int, default=1, help="session_slot_pool_size (default: 1)")
    parser.add_argument("--sessions", type=int, default=4, help="Number of concurrent sessions (default: 4)")
    parser.add_argument("--turns", type=int, default=5, help="Turns per session (default: 5)")
    parser.add_argument("--think-min", type=float, default=0.01, help="Min thinking time between turns in seconds (default: 0.01)")
    parser.add_argument("--think-max", type=float, default=0.05, help="Max thinking time between turns in seconds (default: 0.05)")
    parser.add_argument("--response-min", type=float, default=0.005, help="Min simulated response time in seconds (default: 0.005)")
    parser.add_argument("--response-max", type=float, default=0.020, help="Max simulated response time in seconds (default: 0.020)")
    parser.add_argument("--restore-cost", type=float, default=0.002, help="Simulated slot restore cost in seconds (default: 0.002)")
    parser.add_argument("--save-cost", type=float, default=0.002, help="Simulated slot save cost in seconds (default: 0.002)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Create timelines for each session
    timelines: list[SessionTimeline] = []
    timeline_map: dict[str, SessionTimeline] = {}

    for i in range(args.sessions):
        session_id = f"session-{i:03d}-{random.randint(10000, 99999)}"
        slot_id = InstrumentedSlotLockCoordinator._slot_id_for_session(
            session_id, args.pool_size
        )
        tl = SessionTimeline(
            session_id=session_id,
            slot_id=slot_id or 0,
        )
        timelines.append(tl)
        timeline_map[session_id] = tl

    coordinator = InstrumentedSlotLockCoordinator(timeline_map, args.pool_size)

    # Print header
    print(f"Starting {args.sessions} concurrent sessions ({args.turns} turns each)...")
    print(f"  Pool size:        {args.pool_size}")
    print(f"  Think time:       [{args.think_min:.3f}s, {args.think_max:.3f}s]")
    print(f"  Response time:    [{args.response_min:.3f}s, {args.response_max:.3f}s]")
    print(f"  Restore cost:     {args.restore_cost*1000:.1f}ms")
    print(f"  Save cost:        {args.save_cost*1000:.1f}ms")
    sys.stdout.flush()

    start_time = time.monotonic()
    tasks = [
        simulated_session(
            coordinator,
            tl.session_id,
            tl,
            args.turns,
            args.think_min,
            args.think_max,
            args.response_min,
            args.response_max,
            args.restore_cost,
            args.save_cost,
        )
        for tl in timelines
    ]
    await asyncio.gather(*tasks)
    elapsed = time.monotonic() - start_time

    print(f"Completed in {elapsed:.2f}s")
    sys.stdout.flush()

    # Detect slot stealing
    windows, hold_times, cache_invalidated = detect_slot_stealing(timelines)

    # Generate report
    result = generate_report(
        timelines,
        windows,
        hold_times,
        cache_invalidated,
        args.pool_size,
        args.sessions,
        args.turns,
        args.think_min,
        args.think_max,
        args.save_cost,
        args.restore_cost,
    )
    result["elapsed_seconds"] = round(elapsed, 2)

    if args.json:
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    asyncio.run(main())
