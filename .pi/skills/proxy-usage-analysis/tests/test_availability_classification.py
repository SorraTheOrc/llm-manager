"""Hourly local-availability classification tests (LP-0MVKUKG3F0035I85).

The report must distinguish **lease-blocked** and **prefill-only-no-decode**
hours from genuinely idle hours: ``Busy`` is computed only from paired
``Stream started``/``Stream finished`` lines, so on 2026-10-23 02:00-03:00 —
when a wedged prefill lease held the local dispatch gate for ~4h while the
GPU prefilled ~2M prompt tokens and never decoded — the hourly table read
``Busy 0.0% / Local served 0``, i.e. it rendered an outage as idle.

Each hour's state is derived **only from timestamped proxy-log evidence**:
- ``serving`` — at least one local ``Stream finished`` (decode happened);
- ``prefill-only-no-decode`` — local dispatch attempts (``[local] POST``)
  without any completed local stream (prefill work, no decode);
- ``lease-blocked`` — local demand refused (``local_dispatch_denied`` /
  ``reason=local_lease_active`` fallbacks);
- ``idle`` — no dispatch attempt, no refusal and no completed stream.

States are non-exclusive: the wedged 02:00-03:00 hour is simultaneously
``prefill-only-no-decode`` (the gate's own re-dispatches) and
``lease-blocked`` (every other session refused). Fixture lines below are
copied verbatim from /var/log/llama-proxy/proxy.log.2026-10-23_00 (the proxy
truncates request bodies; the analysis must tolerate that).
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregation
import log_parser
import reporting

WINDOW_START = datetime(2026, 10, 23, 2, 0, 0)
WINDOW_END = datetime(2026, 10, 23, 3, 0, 0)
HOUR_KEY = datetime(2026, 10, 23, 2, 0, 0)

WEDGED_SESSION = "01a16636-4076-717a-b755-60bc61fc961f"
DENIED_SESSION = "01a16616-f901-753e-9297-a77062e48df1"

# --- Fixtures copied from the real 2026-10-23 02:00-03:00 window ------------

# The wedged owner re-dispatches the identical prompt to llama-server; the
# request body (truncated by the logger) contains ``session=``-looking text,
# which the parser must not treat as an event session.
LOCAL_POST_WEDGED = (
    "2026-10-23 02:00:44,564 - INFO - [local] POST http://192.168.0.199:8000/v1/chat/completions "
    'body={"model": "code", "messages": [{"role": "user", "content": [{"type": "text", "text": '
    '"<skill name=\\"implement\\" location=\\"/home/rgardler/.pi/agent/skills/implement/SKILL.md\\">"}]}] '
    f"session_id={DENIED_SESSION} session={{'x-client-request-id': '{DENIED_SESSION}', "
    f"'x-session-affinity': '{DENIED_SESSION}'}} slot=none"
)

# A further dispatch attempt later in the same hour.
LOCAL_POST_WEDGED_2 = LOCAL_POST_WEDGED.replace(
    "2026-10-23 02:00:44,564", "2026-10-23 02:01:07,424"
)

# Refusals: the wire-in retry is refused because the wedged owner holds the
# lease (``local_dispatch_denied``), and the queue path falls back with
# ``reason=local_lease_active``.
DISPATCH_DENIED_1 = (
    "2026-10-23 02:00:44,588 - INFO - local_dispatch_denied "
    f"session={DENIED_SESSION} owner={WEDGED_SESSION} active=1 cold_start=False"
)
DISPATCH_DENIED_2 = (
    "2026-10-23 02:00:52,101 - INFO - local_dispatch_denied "
    f"session={DENIED_SESSION} owner={WEDGED_SESSION} active=1 cold_start=False"
)
QUEUE_FALLBACK_LEASE = (
    "2026-10-23 02:00:39,213 - INFO - contention_queue_fallback_after_queue "
    "provider=local-qwen3 session=herdr-1792711827-616329-14270 "
    "queued_duration=60.00s reason=local_lease_active"
)
FALLBACK_LEASE = (
    "2026-10-23 02:01:03,374 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-2, reason=local_lease_active"
)

# Session-bearing variant (LP-0MVMBMSWR0036RES): the proxy now stamps the
# fallback line with ``session=``/``attempt=``; the hourly evidence counters
# must treat it exactly like the session-less shape.
FALLBACK_LEASE_SESSION_BEARING = (
    "2026-10-23 02:01:05,112 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-2, reason=local_lease_active "
    "session=herdr-1792711827-616329-14270 attempt=2"
)

# Remote traffic in the same hour (requests were attempted and served — by
# remote providers, never by the local model).
REMOTE_STREAM_STARTED = (
    "2026-10-23 02:02:00,000 - INFO - Stream started: provider=opencode-go "
    "model=deepseek-v4.1-flash session=herdr-1792711827-616329-14270 request=[]"
)
REMOTE_STREAM_FINISHED = (
    "2026-10-23 02:02:20,000 - INFO - Stream finished: reason=stop "
    "tokens=950/200/1150 session=herdr-1792711827-616329-14270 "
    "provider=opencode-go model=deepseek-v4.1-flash request=[]"
)


def _lines() -> list[str]:
    return [
        QUEUE_FALLBACK_LEASE,
        LOCAL_POST_WEDGED,
        DISPATCH_DENIED_1,
        DISPATCH_DENIED_2,
        LOCAL_POST_WEDGED_2,
        FALLBACK_LEASE,
        REMOTE_STREAM_STARTED,
        REMOTE_STREAM_FINISHED,
    ]


def _schedule():
    import bucketing

    return bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])


def _events(lines: list[str]):
    return [
        ev
        for line in lines
        if (ev := log_parser.parse_log_line(line)) is not None
    ]


def _summary(lines: list[str] | None = None):
    return aggregation.aggregate(
        _events(lines if lines is not None else _lines()),
        WINDOW_START,
        WINDOW_END,
        _schedule(),
    )


def test_classify_hour_wedged_is_lease_blocked_and_prefill_only():
    """AC1: the 2026-10-23 02:00-03:00 evidence classifies as
    ``lease-blocked`` AND ``prefill-only-no-decode`` (never falsely ``idle``).

    A wedged gate is indistinguishable from idle via busy-% alone (no local
    ``Stream finished`` exists, so busy is 0s); the availability states must
    surface it. ``states`` is a set — both labels apply to the same hour.
    """
    counts = Counter(
        {
            "local_dispatch_attempts": 2,
            "local_denials": 2,
            "lease_active_fallbacks": 2,  # queue fallback + Fallback triggered
            "local_served": 0,
        }
    )
    av = aggregation.hour_availability(HOUR_KEY, counts)
    assert av.is_idle is False
    assert set(av.states) == {
        aggregation.AVAILABILITY_PREFILL_ONLY_NO_DECODE,
        aggregation.AVAILABILITY_LEASE_BLOCKED,
    }
    assert av.label == "prefill-only-no-decode+lease-blocked"
    assert av.local_served == 0
    assert av.local_dispatch_attempts == 2
    assert av.local_denials == 2
    assert av.lease_active_fallbacks == 2


def test_wedged_hour_fixture_end_to_end():
    """AC1 (report path): aggregating the real 2026-10-23 02:00-03:00 fixture
    yields a ``lease-blocked`` + ``prefill-only-no-decode`` hour and the
    report renders it as such, with busy ''0s'' (the pre-existing blind spot).
    """
    summary = _summary()
    counts = summary.hourly_activity[HOUR_KEY]
    assert counts["local_served"] == 0                       # no decode
    assert counts["local_dispatch_attempts"] == 2            # [local] POST
    assert counts["local_denials"] == 2                      # local_dispatch_denied
    assert counts["lease_active_fallbacks"] == 2             # lease-active refusals
    av = aggregation.hour_availability(HOUR_KEY, counts)
    assert set(av.states) == {
        aggregation.AVAILABILITY_PREFILL_ONLY_NO_DECODE,
        aggregation.AVAILABILITY_LEASE_BLOCKED,
    }
    md = reporting.build_report(summary, None)
    section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
    # Hour | State | Busy | % | Requests started | Local attempts | Local served |
    # Local dispatch | Lease denials | Fallbacks | Local-skip
    assert (
        "| 02:00-03:00 | prefill-only-no-decode+lease-blocked | 0s | 0.0% | 1 | 0 | 0 | 2 | 2 | 1 | 0 |"
        in section
    ), section
    # The footnote defines the vocabulary in the same section.
    assert "_Hour state (derived only from the logs" in section
    assert "`lease-blocked`" in section
    assert "`prefill-only-no-decode`" in section
    assert "`idle`" in section


def test_genuinely_idle_hour_classifies_idle():
    """AC2: an hour with no local dispatch attempts, no denials and no local
    decode classifies ``idle`` — no false positives.

    Remote-only traffic is the closest signal to the outage: requests were
    served, but there was no local demand at all.
    """
    summary = _summary(
        [
            "2026-10-23 02:10:00,000 - INFO - Stream started: provider=opencode-go "
            "model=deepseek-v4.1-flash session=s1 request=[]",
            "2026-10-23 02:10:20,000 - INFO - Stream finished: reason=stop "
            "tokens=950/200/1150 session=s1 provider=opencode-go model=deepseek-v4.1-flash request=[]",
        ]
    )
    av = aggregation.hour_availability(HOUR_KEY, summary.hourly_activity.get(HOUR_KEY, Counter()))
    assert av.is_idle is True
    assert av.states == (aggregation.AVAILABILITY_IDLE,)
    md = reporting.build_report(summary, None)
    section = md.split("## Summary by hour", 1)[1].split("## ", 1)[0]
    assert "| 02:00-03:00 | idle | 0s | 0.0% | 1 | 0 | 0 | 0 | 0 | 0 | 0 |" in section


def test_serving_hour_with_zero_busy_never_classifies_idle():
    # A local stream that completed in the hour is ``serving`` even when the
    # busy-% figure is tiny; a serving hour that also refused sessions is
    # ``serving+lease-blocked`` (the two conditions are independent).
    summary = _summary(
        [
            "2026-10-23 02:00:05,000 - INFO - Stream started: provider=local model=Qwen3 session=l1 request=[]",
            "2026-10-23 02:00:10,000 - INFO - Stream finished: reason=stop tokens=100/10/110 "
            "session=l1 provider=local model=Qwen3 request=[]",
            DISPATCH_DENIED_1,
        ]
    )
    av = aggregation.hour_availability(HOUR_KEY, summary.hourly_activity[HOUR_KEY])
    assert set(av.states) == {
        aggregation.AVAILABILITY_SERVING,
        aggregation.AVAILABILITY_LEASE_BLOCKED,
    }
    assert av.label == "serving+lease-blocked"


def test_prefill_only_requires_dispatch_without_decode():
    # Dispatch attempts with no completed stream → prefill-only-no-decode;
    # dispatch attempts that DID complete a stream are serving (not both).
    only_dispatch = aggregation.hour_availability(
        HOUR_KEY,
        Counter({"local_dispatch_attempts": 3, "local_served": 0}),
    )
    assert only_dispatch.states == (aggregation.AVAILABILITY_PREFILL_ONLY_NO_DECODE,)

    served = aggregation.hour_availability(
        HOUR_KEY,
        Counter({"local_dispatch_attempts": 3, "local_served": 1}),
    )
    assert served.states == (aggregation.AVAILABILITY_SERVING,)


def test_lease_blocked_recognised_from_denials_and_lease_active_reasons():
    # ``local_dispatch_denied`` alone marks lease-blocked; a
    # ``reason=local_lease_active`` fallback (queue path or Fallback
    # triggered) marks it too — AC1's ``local_lease_active`` input.
    denied_only = aggregation.classify_hour(HOUR_KEY, local_denials=1)
    assert set(denied_only.states) == {aggregation.AVAILABILITY_LEASE_BLOCKED}
    lease_fallback_only = aggregation.classify_hour(HOUR_KEY, lease_active_fallbacks=1)
    assert set(lease_fallback_only.states) == {aggregation.AVAILABILITY_LEASE_BLOCKED}

    # The real fixture lines: queue fallback carries the reason, so it is
    # counted as a lease-active refusal by the parser.
    events = _events([QUEUE_FALLBACK_LEASE, FALLBACK_LEASE])
    summary = aggregation.aggregate(events, WINDOW_START, WINDOW_END, _schedule())
    counts = summary.hourly_activity[HOUR_KEY]
    assert counts["lease_active_fallbacks"] == 2
    assert set(aggregation.hour_availability(HOUR_KEY, counts).states) == {
        aggregation.AVAILABILITY_LEASE_BLOCKED
    }


def test_session_bearing_lease_fallback_counts_as_lease_active():
    """LP-0MVMBMSWR0036RES: a ``Fallback triggered`` line that carries
    ``session=``/``attempt=`` still feeds the lease-active refusal evidence —
    the hourly availability verdict is unchanged by the new fields."""
    events = _events([FALLBACK_LEASE_SESSION_BEARING])
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "fallback"
    assert ev.session == "herdr-1792711827-616329-14270"
    assert ev.attempt_index == 2
    summary = aggregation.aggregate(events, WINDOW_START, WINDOW_END, _schedule())
    counts = summary.hourly_activity[HOUR_KEY]
    assert counts["lease_active_fallbacks"] == 1
    assert set(aggregation.hour_availability(HOUR_KEY, counts).states) == {
        aggregation.AVAILABILITY_LEASE_BLOCKED
    }


def test_parser_extracts_local_post_without_fabricating_session():
    # ``[local] POST`` parses as a ``local_dispatch`` event with **no** session
    # (the echoed request body itself contains ``session=`` text, which must
    # not be misread as the event's session).
    evs = _events([LOCAL_POST_WEDGED])
    assert len(evs) == 1
    ev = evs[0]
    assert ev.kind == "local_dispatch"
    assert ev.session is None
    assert ev.ts == datetime(2026, 10, 23, 2, 0, 44, 564000)


def test_report_deterministic_across_runs_and_json_state():
    """AC3: identical inputs produce byte-identical reports, and the JSON
    summary carries the state + its raw evidence."""
    first = reporting.build_report(_summary(), None)
    second = reporting.build_report(_summary(), None)
    assert first == second

    data = reporting.summary_to_json(_summary())
    hourly = next(e for e in data["hourly_activity"] if e["hour"] == "2026-10-23 02:00:00")
    assert hourly["state"] == "prefill-only-no-decode+lease-blocked"
    assert hourly["local_dispatch_attempts"] == 2
    assert hourly["local_denials"] == 2
    assert hourly["lease_active_fallbacks"] == 2
    assert hourly["local_served"] == 0
