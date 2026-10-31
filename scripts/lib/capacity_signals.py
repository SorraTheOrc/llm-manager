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

Standard library only, and deliberately **not** importing ``.pi/**``: this
module is owned by the repo-level ``scripts/lib/`` package (mirroring
``proxy_logs.py``) and the skill depends on it, never the other way round.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Mapping

#: Default ``slots_stale`` rate target: fewer than 5 % of eligible status polls.
#: Matches ``scripts/capacity_signal_verification.py --stale-rate-target`` and
#: the nightly analyzer's ``--stale-rate-target`` default.
DEFAULT_STALE_RATE_TARGET = 0.05

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
    """
    in_flight = fields.get("local_in_flight_requests")
    if in_flight is not None:
        try:
            if int(in_flight) > 0:
                return True
        except ValueError:
            pass
    return (
        fields.get("local_active_query") == "true"
        or fields.get("active_query") == "true"
    )


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
    return round(count / denominator, 4) if denominator else 0.0


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
