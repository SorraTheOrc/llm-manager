"""Shared status-poll capacity-signal derivations (LP-0MVU8O4EL005IYP9).

Two surfaces measure the same thing over the same log lines:

- ``scripts/capacity_signal_verification.py`` — the standalone measurement tool
  the local-capacity acceptance criteria are written against;
- the 05:00 nightly report (``.pi/skills/proxy-usage-analysis/``), which since
  LP-0MVU8O4EL005IYP9 publishes the ``slots_stale`` rate next to its target.

Before this module each surface carried its own copy of the derivation
(status-poll detection, ``key=value`` field extraction, the denominator that
excludes polls with no ``slots_stale`` field, the rate and the target check).
Copies drift, and a drifted copy silently publishes a figure that contradicts
the tool — the exact failure mode the nightly-report work item set out to
close. The rule therefore lives here once and both surfaces import it
(LP-0MVU8O4EL005IYP9 AC1.1): the reference tool natively (like
``scripts/lib/proxy_logs.py``), the skill through a documented path-based shim
(``.pi/skills/proxy-usage-analysis/scripts/capacity_signals.py``).

Derivation contract (identical on both surfaces)
------------------------------------------------
``is_status_poll``
    A structured status surface line: ``status_request`` that also carries a
    ``capacity_hint=`` field. One such line per ``GET /llama/local/status``
    (``proxy/proxy/handlers.py`` → ``logger.info("status_request", ...)``).
``stale_flag`` / ``tally_stale``
    ``slots_stale`` is tri-state: ``True``/``False`` from the line, or ``None``
    when the poll carries no ``slots_stale`` field at all (schema 1). A ``None``
    poll is **excluded from the denominator** rather than counted as fresh —
    the pre-fix surface never emitted the field, so counting it fresh would
    understate the rate.
``in_window``
    ``[since, until)`` — the lower bound inclusive, the upper exclusive, so
    adjacent windows never double-count a poll. The nightly analyzer's other
    sections keep their historical closed interval; only this derivation is
    half-open, because it has to reproduce the reference tool's figures
    exactly (AC1.3).
``stale_rate`` / ``stale_rate_check``
    The rate is the stale count over that denominator (``0.0`` when the
    denominator is empty), rounded to four decimals, and the target check is
    the strict comparison ``rate < target`` against the configurable target
    (default :data:`DEFAULT_STALE_RATE_TARGET`, the reference tool's 5 %).
``hint_histogram`` / ``summarise_hints``
    The ``capacity_hint`` distribution over status polls — counts and shares of
    all polls per token, token order stable — plus the
    ``saturated``-without-live-work tally with its share of all polls and of
    ``saturated`` polls. A token the report's vocabulary does not know (a new
    claim name, a case variant, ``unknown`` for an unreadable value) keeps its
    **own** row; nothing is dropped or re-bucketed.
``live_local_work``
    The single live-work rule (LP-0MVU8O9OC005YQU1 AC2.2): ``in_flight > 0``
    (schema 3, authoritative) **or** one of the ``local_active_query`` /
    ``active_query`` flags (the schema-2 fallback, the pair the pre-fix
    false-``saturated`` incident was about). ``has_live_local_work`` is the
    field-mapping entry point for the reference tool and delegates here, so the
    two surfaces cannot diverge.
``poll_coverage``
    Whether the observed status polls span the report window (within
    :data:`DEFAULT_COVERAGE_GAP_SECONDS` of each edge), so a truncated log
    retention window cannot masquerade as a complete measurement.

Standard library only, and deliberately **not** importing ``.pi/**``: this
module is owned by the repo-level ``scripts/lib/`` package (mirroring
``proxy_logs.py``) and the skill depends on it, never the other way round.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

#: Default ``slots_stale`` rate target: fewer than 5 % of eligible status polls.
#: Matches ``scripts/capacity_signal_verification.py --stale-rate-target`` and
#: the nightly analyzer's ``--stale-rate-target`` default.
DEFAULT_STALE_RATE_TARGET = 0.05

#: Token recorded for a status poll whose ``capacity_hint`` value is absent or
#: unreadable. Never dropped from the distribution: a vocabulary change must
#: stay visible rather than being silently re-bucketed (LP-0MVU8O9OC005YQU1
#: AC2.3).
UNKNOWN_HINT = "unknown"

#: The hint token that claims the local pool is out of capacity.
SATURATED_HINT = "saturated"

#: Tolerance applied when deciding whether the observed status polls span the
#: reported window. Matches the reference tool's ``--max-gap-seconds`` default
#: (1 h) so the two coverage models agree on what "spans the window" means.
DEFAULT_COVERAGE_GAP_SECONDS = 3600.0

#: The structured status surface: one line per ``GET /llama/local/status``.
STATUS_POLL_RE = re.compile(r"\bstatus_request\b")

#: Fields read off a status poll line (``key=value``, value is non-space).
STATUS_FIELD_RE = re.compile(
    r"\b(?P<key>capacity_hint|slots_stale|slots_detail_stale|available_slots|"
    r"local_active_query|active_query|local_in_flight_requests|"
    r"status_schema_version|latency_ms)=(?P<value>[^\s]+)"
)

#: The field whose presence decides whether a poll enters the rate denominator.
FIELD_SLOTS_STALE = "slots_stale"

#: ``latency_ms`` bands separating a fast fresh poll (< 1 s) from one that
#: exhausted the ``/slots`` query budget (2.0 s counts / 1.0 s detail) and
#: beyond.
LATENCY_BANDS = ("lt_1s", "1s_to_2s", "2s_to_5s", "gte_5s", "unknown")


def is_status_poll(line: str) -> bool:
    """True for a structured ``status_request`` line (the poll surface)."""
    return bool(STATUS_POLL_RE.search(line)) and "capacity_hint=" in line


def status_fields(line: str) -> dict[str, str]:
    """Return the ``key=value`` status fields present on *line*."""
    return {
        match.group("key"): match.group("value")
        for match in STATUS_FIELD_RE.finditer(line)
    }


def parse_latency_ms(value: str | None) -> int | None:
    """Parse the handler's ``latency_ms=`` field (``None`` when absent)."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_int(value: str | None) -> int | None:
    """Parse a non-negative integer log field (``None`` when absent/bogus)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def latency_band(latency_ms: int | None) -> str:
    """Bucket a status-poll ``latency_ms`` (``unknown`` when absent)."""
    if latency_ms is None:
        return "unknown"
    if latency_ms < 1000:
        return "lt_1s"
    if latency_ms < 2000:
        return "1s_to_2s"
    if latency_ms < 5000:
        return "2s_to_5s"
    return "gte_5s"


def has_live_local_work(fields: Mapping[str, str]) -> bool:
    """True when a status poll reports concurrent live local work.

    Schema 3 polls carry ``local_in_flight_requests`` (authoritative: derived
    from live dispatch records). Schema 2 polls do not, so the
    ``local_active_query`` / ``active_query`` flags are the fallback — which is
    exactly the pair the pre-fix false-``saturated`` incident was about.
    Any single positive signal counts as live work: a violating poll must show
    none of them.

    This is the field-mapping entry point used by
    ``scripts/capacity_signal_verification.py``; the rule itself lives in
    :func:`live_local_work` so the nightly analyzer — which has typed
    ``LogEvent`` fields rather than a raw field map — applies exactly the same
    rule (LP-0MVU8O9OC005YQU1 AC2.2).
    """
    return live_local_work(
        parse_int(fields.get("local_in_flight_requests")),
        fields.get("local_active_query") == "true",
        fields.get("active_query") == "true",
    )


def live_local_work(
    in_flight_requests: int | None,
    local_active_query: bool | None,
    active_query: bool | None,
) -> bool:
    """The one live-work rule, over already-parsed poll fields.

    ``in_flight_requests`` (schema 3) is authoritative when it reports live
    work; otherwise the ``local_active_query`` / ``active_query`` flags decide
    (the schema-2 fallback). ``None`` means the field was absent from the line,
    never a fabricated default.
    """
    if in_flight_requests is not None and in_flight_requests > 0:
        return True
    return bool(local_active_query) or bool(active_query)


@dataclass(frozen=True)
class HintSummary:
    """The ``capacity_hint`` distribution and the violation tallies.

    ``counts`` is the token-order-stable histogram over status polls and
    ``shares`` each token's share of those polls. ``saturated_without_live_work``
    is the number of ``saturated`` polls that report no live local work, with
    its share of all polls and of ``saturated`` polls — both shares, because a
    violation rate reads very differently against each denominator
    (LP-0MVU8O9OC005YQU1 AC2.3).
    """

    counts: dict[str, int] = field(default_factory=dict)
    shares: dict[str, float] = field(default_factory=dict)
    saturated: int = 0
    saturated_without_live_work: int = 0
    violation_share_of_polls: float = 0.0
    violation_share_of_saturated: float = 0.0


def hint_histogram(hints: Iterable[str | None]) -> dict[str, int]:
    """Counts per ``capacity_hint`` token, sorted by token.

    A value that is absent or unreadable is recorded as :data:`UNKNOWN_HINT`
    rather than dropped, so an unparsed line cannot vanish from the
    distribution. Sorted output keeps the rendered rows and the JSON keys
    stable between nights.
    """
    counter = Counter(hint if hint else UNKNOWN_HINT for hint in hints)
    return dict(sorted(counter.items()))


def share(count: int, total: int) -> float:
    """``count / total`` rounded to four decimals (``0.0`` when none)."""
    return round(count / total, 4) if total else 0.0


def summarise_hints(
    samples: Iterable[tuple[str | None, bool]],
) -> HintSummary:
    """Histogram, shares and saturated-without-live-work tallies.

    Each sample is ``(capacity_hint_token, has_live_local_work)`` from one
    status poll, so both surfaces feed the same shape into the same arithmetic
    (LP-0MVU8O9OC005YQU1 AC2.2/AC2.3).
    """
    materialised = [
        (hint if hint else UNKNOWN_HINT, bool(live)) for hint, live in samples
    ]
    counts = hint_histogram(hint for hint, _ in materialised)
    total = len(materialised)
    saturated = counts.get(SATURATED_HINT, 0)
    violations = sum(
        1 for hint, live in materialised if hint == SATURATED_HINT and not live
    )
    return HintSummary(
        counts=counts,
        shares={token: share(count, total) for token, count in counts.items()},
        saturated=saturated,
        saturated_without_live_work=violations,
        violation_share_of_polls=share(violations, total),
        violation_share_of_saturated=share(violations, saturated),
    )


def poll_coverage(
    first_poll: dt.datetime | None,
    last_poll: dt.datetime | None,
    window_start: dt.datetime,
    window_end: dt.datetime,
    max_gap_seconds: float = DEFAULT_COVERAGE_GAP_SECONDS,
) -> dict:
    """Whether the observed status polls span the reported window.

    The nightly analyzer has no file-level coverage model (the reference tool
    does), so the honest proxy is the span of the polls themselves: a window
    whose retained logs start late or end early shows up as a large gap at one
    edge. ``covered`` is ``False`` when that gap exceeds *max_gap_seconds* at
    either edge, which is what lets the report label a truncated window as a
    partial measurement instead of a complete one (LP-0MVU8O9OC005YQU1 AC2.7).
    With no polls there is nothing to measure, so ``covered`` is ``False`` and
    both gaps stay ``None`` rather than being fabricated as zero.
    """
    if first_poll is None or last_poll is None:
        return {
            "covered": False,
            "window_start_gap_seconds": None,
            "window_end_gap_seconds": None,
            "max_gap_seconds": float(max_gap_seconds),
        }
    start_gap = (first_poll - window_start).total_seconds()
    end_gap = (window_end - last_poll).total_seconds()
    return {
        "covered": start_gap <= max_gap_seconds and end_gap <= max_gap_seconds,
        "window_start_gap_seconds": float(start_gap),
        "window_end_gap_seconds": float(end_gap),
        "max_gap_seconds": float(max_gap_seconds),
    }


def in_window(
    ts: dt.datetime | None,
    since: dt.datetime | None,
    until: dt.datetime | None,
) -> bool:
    """``since`` inclusive, ``until`` exclusive (adjacent windows never overlap)."""
    if ts is None:
        return False
    if since is not None and ts < since:
        return False
    if until is not None and ts >= until:
        return False
    return True


def stale_flag(fields: Mapping[str, str]) -> bool | None:
    """Tri-state ``slots_stale`` flag from a poll's raw ``key=value`` fields.

    ``None`` means the line carried no ``slots_stale`` field (schema 1), so the
    poll is excluded from the stale-rate denominator rather than counted as
    fresh.
    """
    value = fields.get(FIELD_SLOTS_STALE)
    if value is None:
        return None
    return value == "true"


def tally_stale(flags: Iterable[bool | None]) -> tuple[int, int]:
    """Return ``(stale, denominator)`` from each poll's tri-state stale flag.

    ``None`` entries (polls with no ``slots_stale`` field) are skipped entirely:
    they neither count as stale nor enlarge the denominator.
    """
    stale = 0
    denominator = 0
    for flag in flags:
        if flag is None:
            continue
        denominator += 1
        if flag:
            stale += 1
    return stale, denominator


def stale_rate(count: int, denominator: int) -> float:
    """Stale polls as a share of eligible polls (``0.0`` when none eligible)."""
    return share(count, denominator)


def stale_rate_check(
    count: int,
    denominator: int,
    target: float = DEFAULT_STALE_RATE_TARGET,
) -> dict:
    """Evaluate the stale-rate target: ``{"target", "observed", "passed"}``.

    The comparison is strict (``rate < target``), so a rate exactly at the
    target is a miss — matching the reference tool's verdict. ``denominator``
    is returned too, so a consumer can tell an honest ``0.0`` rate from one
    computed over no eligible polls.
    """
    rate = stale_rate(count, denominator)
    return {
        "target": float(target),
        "observed": rate,
        "denominator": denominator,
        "passed": rate < float(target),
    }
