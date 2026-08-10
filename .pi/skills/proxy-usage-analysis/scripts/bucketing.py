"""Fast/cheap session bucketing derived from the proxy slot schedule.

The proxy's ``slot_schedule`` (in the active config profile,
``proxy/config-fast.yaml`` or ``proxy/config-cheap.yaml``) defines
transition times and the number of GPU slots active from that time until the
next transition (midnight wrapping). This module turns those entries into
contiguous day periods and labels them "fast" (fewer slots) vs "cheap" (more
slots), so the analysis does not hardcode the split.

Terminology note (LP-0MSLMYEEU002IBH6): the operating modes are **fast**
(cloud-backed; the old "day" period) and **cheap** (1-slot local pool,
same models as fast, LP-0MSMIPPJI007GU9N; the old "night" period). The
slot-count -> label mapping is unchanged: the period(s)
with the fewest slots are labelled "fast", the period(s) with the most
"cheap"; equal counts collapse to a single "fast" bucket.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

MINUTES_PER_DAY = 24 * 60


def minute_of_day(dt: datetime) -> float:
    """Fractional minutes from midnight (seconds contribute a fraction)."""
    return dt.hour * 60 + dt.minute + dt.second / 60.0


@dataclass
class SlotPeriod:
    """A contiguous window of the day with a constant slot count.

    ``start_minutes`` is inclusive, ``end_minutes`` exclusive; when
    ``end_minutes <= start_minutes`` the period wraps past midnight.
    """

    start_minutes: float
    end_minutes: float
    slots: int
    label: str  # "fast" | "cheap"

    def contains(self, minutes: float) -> bool:
        if self.end_minutes > self.start_minutes:
            return self.start_minutes <= minutes < self.end_minutes
        # Wraps past midnight.
        return minutes >= self.start_minutes or minutes < self.end_minutes


@dataclass
class SlotSchedule:
    periods: list[SlotPeriod]
    fast_slots: int | None
    cheap_slots: int | None
    source: str  # "config" | "default"

    def period_for(self, dt: datetime) -> SlotPeriod:
        minutes = minute_of_day(dt)
        for p in self.periods:
            if p.contains(minutes):
                return p
        return self.periods[-1]


def schedule_from_entries(entries: Sequence[tuple[str, int]]) -> SlotSchedule:
    """Build a schedule from ``(HH:MM, slots)`` transition entries.

    Each entry's slot count applies from its time until the next entry
    (midnight wrapping: the last entry's count applies from 00:00 until the
    first entry). Periods are labelled by slot count: the period(s) with the
    minimum count are "fast", all others "cheap". If every period has the
    same count there is a single "fast" bucket.
    """
    parsed = []
    for time_str, slots in entries:
        hh, mm = time_str.split(":")
        parsed.append((int(hh) * 60 + int(mm), int(slots)))
    parsed.sort(key=lambda t: t[0])

    if not parsed:
        return SlotSchedule(periods=[], fast_slots=None, cheap_slots=None, source="default")

    boundaries = [t[0] for t in parsed]
    counts = [t[1] for t in parsed]

    min_count = min(counts)
    max_count = max(counts)

    periods: list[SlotPeriod] = []
    # Count active before the first boundary is the last entry's (midnight wrap).
    prev_start = 0.0
    prev_slots = counts[-1]
    for boundary, slots in zip(boundaries, counts):
        if boundary > prev_start:
            periods.append(
                SlotPeriod(prev_start, float(boundary), prev_slots, _label(prev_slots, min_count, max_count))
            )
        prev_start = float(boundary)
        prev_slots = slots
    if prev_start < MINUTES_PER_DAY:
        periods.append(
            SlotPeriod(prev_start, float(MINUTES_PER_DAY), prev_slots, _label(prev_slots, min_count, max_count))
        )

    fast_slots = min_count if min_count != max_count else min_count
    cheap_slots = max_count if min_count != max_count else None
    return SlotSchedule(periods=periods, fast_slots=fast_slots, cheap_slots=cheap_slots, source="config")


def _label(slots: int, min_count: int, max_count: int) -> str:
    if max_count == min_count:
        return "fast"
    return "fast" if slots == min_count else "cheap"


def schedule_from_config(config: dict | None, default_slots: int | None) -> SlotSchedule:
    """Derive the schedule from a parsed proxy config dict.

    If ``slot_schedule.enabled`` is false or entries are missing, fall back to
    a single "fast" bucket using ``default_slots`` (typically
    ``session_slot_pool_size``).
    """
    entries: list[tuple[str, int]] = []
    if config:
        schedule = config.get("slot_schedule") or {}
        enabled = schedule.get("enabled", True)
        if enabled:
            for e in schedule.get("entries") or []:
                if isinstance(e, dict):
                    entries.append((str(e.get("time")), int(e.get("slots", 0))))
                elif isinstance(e, (tuple, list)) and len(e) == 2:
                    entries.append((str(e[0]), int(e[1])))
    if not entries:
        slots = default_slots if default_slots is not None else 0
        return SlotSchedule(
            periods=[SlotPeriod(0.0, float(MINUTES_PER_DAY), slots, "fast")],
            fast_slots=slots if slots else None,
            cheap_slots=None,
            source="default",
        )
    return schedule_from_entries(entries)


def bucket_for_time(schedule: SlotSchedule, dt: datetime) -> SlotPeriod:
    """Return the fast/cheap period containing ``dt``."""
    return schedule.period_for(dt)
