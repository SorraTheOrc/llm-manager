"""Streaming, line-based parser for llama-proxy structured INFO log lines.

The proxy logs lines of the form::

    %Y-%m-%d %H:%M:%S,%f - LEVEL - message

Only a small set of structured message prefixes carry the signal this skill
needs (``Stream started``, ``Stream finished``, ``Fallback triggered``,
``routing_skip_local``, ``local_dispatch_denied``). Everything else is
ignored. Parsing is tolerant: missing fields default to ``None`` and a line
that cannot be parsed is skipped (never fatal), so truncated request payloads
and log-format drift do not break the analysis.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Log line prefix: "2026-08-02 13:58:32,260 - INFO - <message>"
LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - (\w+) - (.*)$")

# Field extractors (tolerant: may match nothing).
RE_PROVIDER = re.compile(r"\bprovider=(\S+)")
RE_MODEL = re.compile(r"\bmodel=(\S+)")
RE_SESSION = re.compile(r"\bsession=([A-Za-z0-9_.-]+)")
RE_TOKENS = re.compile(r"\btokens=(\d+)/(\d+)/(\d+)")
RE_ENTRY = re.compile(r"\bentry=(\S+)")
RE_ERROR_DETAIL = re.compile(r"\berror=([^\s,}]+)")
RE_FALLBACK = re.compile(
    r"Fallback triggered for model=(\S+?), from=(\S+?), to=(\S+?), reason=([\w ]+)$"
)
# routing_skip_local reason sits between "reason=" and the "→" arrow.
RE_ROUTING_SKIP_REASON = re.compile(r"\breason=([\w_]+)\s*→")
RE_DISPATCH_DENIED = re.compile(r"session=([A-Za-z0-9_.-]+) owner=([A-Za-z0-9_.-]+) active=(\d+)")

# Compaction events (LP-0MTHCTLAF00147IT): the proxy emits ``compaction_event``
# for every compaction decision (compact / remote_with_guidance / backstop_*)
# with fields session/mode/action/reason/pre_tokens/post_tokens/turns_summarized/
# turns_dropped/summary_tokens/dry_run. Churn and backstop lines are parsed
# as supplements but do not affect primary counts (tolerant, never fatal).
RE_COMPACTION_PRE_TOKENS = re.compile(r"\bpre_tokens=(\d+)")
RE_COMPACTION_POST_TOKENS = re.compile(r"\bpost_tokens=(\d+)")
RE_COMPACTION_TURNS_SUMMARIZED = re.compile(r"\bturns_summarized=(\d+)")
RE_COMPACTION_TURNS_DROPPED = re.compile(r"\bturns_dropped=(\d+)")
RE_COMPACTION_SUMMARY_TOKENS = re.compile(r"\bsummary_tokens=(\d+)")
RE_COMPACTION_DRY_RUN = re.compile(r"\bdry_run=(true|false)")
RE_COMPACTION_ACTION = re.compile(r"\baction=([A-Za-z_]+)")
RE_COMPACTION_MODE = re.compile(r"\bmode=(\w+)")
RE_COMPACTION_REASON = re.compile(r"\breason=([A-Za-z_]+)")
# Backstop-specific field names (truncate_backstop structured log).
RE_BACKSTOP_DROPPED_TURNS = re.compile(r"\bdropped_turns=(\d+)")
RE_BACKSTOP_DROPPED_MESSAGES = re.compile(r"\bdropped_messages=(\d+)")
# Churn line: "compaction_churn session=<id> count=<n> rate_per_hour=<f> exceeds_target=<bool>"
# Backstop line: "compaction_backstop action=dropped|exhausted ... dropped_turns=N dropped_messages=N"
# Example: "compaction_backstop action=dropped dropped_turns=1 dropped_messages=2 estimated_before=65000"
RE_CHURN_COUNT = re.compile(r"\bcount=(\d+)")
RE_CHURN_RATE = re.compile(r"\brate_per_hour=([\d.]+)")

# Contention-queue events (LP-0MSORQVK50012Q4D F4 AC3): the proxy emits
# ``contention_queue_dispatch`` (a queued request was dispatched local after
# a slot freed) and ``contention_queue_fallback_after_queue`` (caps exceeded
# → fell back to the next remote provider). Both carry queued_duration; the
# dispatch line additionally carries policy + depth.
RE_CONTENTION_DURATION = re.compile(r"queued_duration=([\d.]+)s")
RE_CONTENTION_DEPTH = re.compile(r"\bdepth=(\d+)")
RE_CONTENTION_POLICY = re.compile(r"\bpolicy=(\S+)")

# Enriched error payload on ``Stream finished: reason=error`` (LP-0MT6322OT00900OX).
# The proxy now logs ``error_type=<type> error_message=<msg> suggested_action=<act>``
# before the trailing session/provider/model/entry fields. Messages/actions may
# contain spaces, so values are captured non-greedily up to the next known key
# (or end of line); ``error_message`` may be followed by ``suggested_action``.
RE_ERROR_TYPE = re.compile(r"\berror_type=(\S+)")
RE_ERROR_MESSAGE = re.compile(
    r"\berror_message=(.+?)(?=\s+(?:session|provider|model|entry|request|suggested_action)=|$)"
)
RE_SUGGESTED_ACTION = re.compile(
    r"\bsuggested_action=(.+?)(?=\s+(?:session|provider|model|entry|request)=|$)"
)

# Error-line extractors.
RE_BACKEND_ATTEMPT = re.compile(r"attempt=(\d+/\d+)")
RE_BACKEND_SIGNAL = re.compile(r"signal=([\w_]+)")
RE_UPSTREAM_STATUS = re.compile(r"status=(\d+)")
RE_UPSTREAM_URL = re.compile(r"url=(\S+)")
# upstream error type appears in the JSON body as {"type":"error","error":{"type":"<Type>",...}}.
RE_UPSTREAM_BODY_TYPE = re.compile(r'"type":"(FreeUsageLimitError|[A-Za-z]+Error)"')

# Rotated log naming: proxy.log.YYYY-MM-DD_HH (the timestamp is the rotation
# time). Note: the name-encoded time does NOT reliably bound a file's content
# span in this deployment — rotated files routinely hold data past it — so
# discovery includes every rotated file and iter_events is the only boundary.

STREAM_STARTED = "Stream started"
STREAM_FINISHED = "Stream finished"
FALLBACK = "Fallback triggered"
ROUTING_SKIP = "routing_skip_local"
DISPATCH_DENIED = "local_dispatch_denied"
CONTENTION_DISPATCH = "contention_queue_dispatch"
CONTENTION_FALLBACK_AFTER_QUEUE = "contention_queue_fallback_after_queue"
COMPACTION_EVENT = "compaction_event"
COMPACTION_BACKSTOP = "compaction_backstop"
COMPACTION_CHURN = "compaction_churn"

# Reason-value normalization (backward compatibility). ``warm_cache_bypass``
# was the pre-LP-0MSF8XDG7000PERM name for the warm-cache hard-cap skip. The
# name misleads (the skip fires when the estimated prompt context exceeds the
# per-slot hard cap, regardless of cache state) and was renamed to
# ``context_too_large``. Rotated logs (6-hourly rotation, 90-day retention)
# still contain the legacy value, so it is normalized here so downstream
# analysis treats both spellings as the same reason.
CONTEXT_TOO_LARGE = "context_too_large"
LEGACY_WARM_CACHE_BYPASS = "warm_cache_bypass"


def _normalize_reason(reason: str | None) -> str | None:
    """Map legacy reason values to their current names."""
    if reason == LEGACY_WARM_CACHE_BYPASS:
        return CONTEXT_TOO_LARGE
    return reason

# Error-line prefixes (WARNING level structured lines the parser recognizes).
STREAM_ERROR = "Stream error:"
SLOT_SAVE_FAILED = "slot_save failed"
BACKEND_RETRY = "backend_retry"
UPSTREAM_ERROR = "[remote] upstream error"

# Operating-mode switch lines (LP-0MSM5K4TX004MICX): the proxy logs each
# applied mode transition as ``Mode scheduler: applied scheduled mode
# <mode>``. These reconstruct the mode active during the analysis window so
# sessions are bucketed by the mode that was actually running (fast vs cheap
# profiles), not the mode at analysis time (LP-0MSPZUD4G007IYGH).
MODE_SWITCH_PREFIX = "Mode scheduler: applied scheduled mode"
RE_MODE_SWITCH = re.compile(r"Mode scheduler: applied scheduled mode (\w+)")

# MANUAL mode switches (POST /admin/set-mode, LP-0MSMF25V9002AY1J) persist the
# new mode and restart the proxy but do NOT emit an ``applied scheduled mode``
# line; the grandfathering init that runs on every restart then reports the
# actually-active mode as ``Grandfathering: enabled; other-mode config <file>
# (current=<mode>)`` (LP-0MT1EE315007AKXG). The ``Mode scheduler: enabled with
# N entries`` announcement fires on BOTH scheduled and manual transitions and
# is therefore not a reliable signal; the ``(current=<mode>)`` field on the
# grandfathering line is. The ``restart_services: router-mode restart complete
# (N slots)`` line is corroborating evidence only (slot count -> mode mapping
# is deployment-specific and never parsed).
GRANDFATHERING_PREFIX = "Grandfathering: enabled; other-mode config"
RE_MANUAL_MODE_CURRENT = re.compile(r"\(current=(\w+)\)")

# Best-effort provider attribution for ``[remote] upstream error`` lines: the
# line carries only the target URL, so the provider is inferred from the
# endpoint path/host. These patterns mirror the remote provider endpoints in
# proxy/config.yaml (opencode.ai/zen/go → opencode-go, opencode.ai/zen →
# opencode, api.deepseek.com → deepseek, models.inference.ai.azure.com →
# github). The model is not present in the line and stays ``None``.
UPSTREAM_URL_PROVIDER_PATTERNS = (
    ("opencode.ai/zen/go", "opencode-go"),
    ("opencode.ai/zen", "opencode"),
    ("api.deepseek.com", "deepseek"),
    ("models.inference.ai.azure.com", "github"),
)


def _provider_from_upstream_url(url: str | None) -> str | None:
    """Infer a provider name from an upstream error target URL (best effort).

    Known endpoint patterns are matched first; anything else falls back to the
    bare hostname so unknown endpoints are still attributed. Returns ``None``
    when the line carried no URL.
    """
    if not url:
        return None
    for needle, provider in UPSTREAM_URL_PROVIDER_PATTERNS:
        if needle in url:
            return provider
    host = url.split("/", 3)[2] if "//" in url else None
    return host or None

# Events within this many seconds *before* a session's first remote stream are
# candidates for attributing a session-less "Fallback triggered" line to that
# session.
FALLBACK_ATTRIBUTION_WINDOW_SECONDS = 60

# Margin (each side) by which ``iter_events`` widens the window so the
# local-model busy-time analysis can pair ``Stream started``/``Stream finished``
# events that cross the window boundary. 1h generously covers long generations
# that overrun the window end; rotated logs bound the pre-window side anyway.
BUSY_WINDOW_MARGIN = timedelta(hours=1)

# Margin by which ``iter_events`` widens the window for the operating-mode
# timeline (``Mode scheduler: applied scheduled mode`` lines, see
# MODE_SWITCH_PREFIX). The mode schedule switches at most twice a day, so a
# 48h margin guarantees every in-window session finds the mode transition
# that was active at its start (the nearest prior transition in the available
# logs), even for windows that begin just after a transition. The effective
# margin used by the analysis is max(BUSY_WINDOW_MARGIN, MODE_TIMELINE_MARGIN)
# so a single streaming pass serves both the busy-time pairing and the mode
# timeline.
MODE_TIMELINE_MARGIN = timedelta(hours=48)

LOCAL_PROVIDER = "local"


@dataclass
class LogEvent:
    """One parsed structured log line.

    ``kind`` is one of ``stream_started``, ``stream_finished``, ``fallback``,
    ``routing_skip``, ``dispatch_denied``, ``contention_dispatch``,
    ``contention_fallback_after_queue``, ``compaction_event``,
    ``compaction_backstop``, ``compaction_churn``, or an error kind
    (``stream_error``, ``stream_finish_error``, ``slot_save_error``,
    ``backend_retry``, ``upstream_http_error``). Only the fields relevant to
    each kind are populated.
    """

    kind: str
    ts: datetime
    provider: str | None = None
    model: str | None = None
    session: str | None = None
    reason: str | None = None
    prompt: int | None = None
    completion: int | None = None
    total: int | None = None
    src: str | None = None
    dst: str | None = None
    owner: str | None = None
    active: int | None = None
    # Contention-queue fields (contention_dispatch /
    # contention_fallback_after_queue kinds, LP-0MSORQVK50012Q4D F4 AC3).
    queued_duration: str | None = None
    depth: str | None = None
    policy: str | None = None
    # Error-taxonomy fields (populated for error kinds only).
    error: str | None = None
    # Enriched error payload (LP-0MT6322OT00900OX): carried on
    # ``stream_finish_error`` when the log line includes the enriched
    # error fields.
    error_type: str | None = None
    error_message: str | None = None
    suggested_action: str | None = None
    entry: str | None = None
    status: int | None = None
    attempt: str | None = None
    signal: str | None = None
    src_file: str | None = None
    raw: str | None = None
    # Operating-mode field (mode_switch kind only): "fast" | "cheap".
    mode: str | None = None
    # Compaction fields (compaction_event / compaction_backstop / compaction_churn).
    action: str | None = None
    pre_tokens: int | None = None
    post_tokens: int | None = None
    turns_summarized: int | None = None
    turns_dropped: int | None = None
    summary_tokens: int | None = None
    dry_run: bool | None = None
    # Backstop-specific.
    dropped_messages: int | None = None
    # Churn-specific.
    churn_count: int | None = None
    churn_rate: float | None = None


def _first(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1) if m else None


def _session_from(text: str) -> str | None:
    token = _first(RE_SESSION, text)
    if token is None or token == "unknown":
        return None
    return token


def parse_log_line(line: str) -> LogEvent | None:
    """Parse one log line into a :class:`LogEvent`, or ``None`` if the line
    carries no relevant signal (or cannot be parsed).

    ``session=unknown`` is treated as unattributed (``session=None``).
    """
    if not line:
        return None
    m = LINE_RE.match(line)
    if m is None:
        return None
    date_part, ms_part, _level, msg = m.groups()
    try:
        ts = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S").replace(
            microsecond=int(ms_part) * 1000
        )
    except ValueError:
        return None

    if msg.startswith(STREAM_STARTED):
        return LogEvent(
            "stream_started",
            ts,
            provider=_first(RE_PROVIDER, msg),
            model=_first(RE_MODEL, msg),
            session=_session_from(msg),
        )
    if msg.startswith(STREAM_FINISHED):
        tokens = RE_TOKENS.search(msg)
        prompt = completion = total = None
        if tokens:
            prompt, completion, total = (int(v) for v in tokens.groups())
        reason = _first(RE_ROUTING_SKIP_REASON, msg) or _first(
            re.compile(r"\breason=(\w+)"), msg
        )
        # ``Stream finished: reason=error`` is the client-visible synthetic
        # error event.  Prior to LP-0MT6322OT00900OX the log line carried no
        # error payload; now it may carry ``error_type``, ``error_message``,
        # and ``suggested_action`` from the enriched error object.
        if reason == "error":
            return LogEvent(
                "stream_finish_error",
                ts,
                provider=_first(RE_PROVIDER, msg),
                model=_first(RE_MODEL, msg),
                session=_session_from(msg),
                reason=reason,
                error="finish_reason:error",
                error_type=_first(RE_ERROR_TYPE, msg),
                error_message=_first(RE_ERROR_MESSAGE, msg),
                suggested_action=_first(RE_SUGGESTED_ACTION, msg),
                entry=_first(RE_ENTRY, msg),
                raw=line,
            )
        return LogEvent(
            "stream_finished",
            ts,
            provider=_first(RE_PROVIDER, msg),
            model=_first(RE_MODEL, msg),
            session=_session_from(msg),
            reason=reason,
            prompt=prompt,
            completion=completion,
            total=total,
        )
    if msg.startswith(FALLBACK):
        m2 = RE_FALLBACK.search(msg)
        if m2 is None:
            return None
        fmodel, src, dst, reason = m2.groups()
        return LogEvent(
            "fallback", ts, reason=_normalize_reason(reason), src=src, dst=dst
        )
    if msg.startswith(ROUTING_SKIP):
        return LogEvent(
            "routing_skip",
            ts,
            session=_session_from(msg),
            reason=_normalize_reason(_first(RE_ROUTING_SKIP_REASON, msg)),
        )
    if msg.startswith(DISPATCH_DENIED):
        m2 = RE_DISPATCH_DENIED.search(msg)
        if m2 is None:
            return None
        session, owner, active = m2.groups()
        return LogEvent(
            "dispatch_denied", ts, session=session, owner=owner, active=int(active)
        )
    if msg.startswith(CONTENTION_DISPATCH):
        # "contention_queue_dispatch provider=... session=... queued_duration=1.23s policy=queue depth=0"
        return LogEvent(
            "contention_dispatch",
            ts,
            provider=_first(RE_PROVIDER, msg),
            session=_session_from(msg),
            queued_duration=_first(RE_CONTENTION_DURATION, msg),
            depth=_first(RE_CONTENTION_DEPTH, msg),
            policy=_first(RE_CONTENTION_POLICY, msg),
        )
    if msg.startswith(CONTENTION_FALLBACK_AFTER_QUEUE):
        # "contention_queue_fallback_after_queue provider=... session=... queued_duration=1.23s"
        return LogEvent(
            "contention_fallback_after_queue",
            ts,
            provider=_first(RE_PROVIDER, msg),
            session=_session_from(msg),
            queued_duration=_first(RE_CONTENTION_DURATION, msg),
        )
    if msg.startswith(MODE_SWITCH_PREFIX):
        # "Mode scheduler: applied scheduled mode cheap" — the mode that the
        # scheduler applied at this timestamp (LP-0MSM5K4TX004MICX).
        m = RE_MODE_SWITCH.search(msg)
        if m is None:
            return None
        mode = m.group(1).lower()
        if mode not in ("fast", "cheap"):
            return None
        return LogEvent("mode_switch", ts, mode=mode)
    if msg.startswith(GRANDFATHERING_PREFIX):
        # "Grandfathering: enabled; other-mode config config-fast.yaml
        # (current=cheap)" — the actually-active mode after a restart. Manual
        # mode switches emit this (and NOT an applied-scheduled-mode line), so
        # it is what reconstructs manual-switch transitions in the mode
        # timeline (LP-0MT1EE315007AKXG).
        m = RE_MANUAL_MODE_CURRENT.search(msg)
        if m is None:
            return None
        mode = m.group(1).lower()
        if mode not in ("fast", "cheap"):
            return None
        return LogEvent("mode_switch", ts, mode=mode)
    if msg.startswith(STREAM_ERROR):
        # "Stream error: session=... provider=... model=... error=NameError"
        return LogEvent(
            "stream_error",
            ts,
            provider=_first(RE_PROVIDER, msg),
            model=_first(RE_MODEL, msg),
            session=_session_from(msg),
            error=_first(RE_ERROR_DETAIL, msg),
            raw=line,
        )
    if msg.startswith(SLOT_SAVE_FAILED):
        # "slot_save failed slot=2 error=ReadTimeout/ReadTimeout"
        # Slot persistence always targets the local llama-server, so the event
        # is attributed to the local provider; the model is not in the line.
        return LogEvent(
            "slot_save_error",
            ts,
            provider=LOCAL_PROVIDER,
            error=_first(RE_ERROR_DETAIL, msg),
            raw=line,
        )
    if msg.startswith(BACKEND_RETRY):
        # "backend_retry path=... stream=True attempt=1/8 delay=... signal=... error=..."
        return LogEvent(
            "backend_retry",
            ts,
            error=_first(RE_ERROR_DETAIL, msg),
            attempt=_first(RE_BACKEND_ATTEMPT, msg),
            signal=_first(RE_BACKEND_SIGNAL, msg),
            raw=line,
        )
    if msg.startswith(UPSTREAM_ERROR):
        # "[remote] upstream error status=429 url=... body={"type":"error","error":{"type":"FreeUsageLimitError",...}}"
        status_m = RE_UPSTREAM_STATUS.search(msg)
        body_type = RE_UPSTREAM_BODY_TYPE.search(msg)
        return LogEvent(
            "upstream_http_error",
            ts,
            provider=_provider_from_upstream_url(_first(RE_UPSTREAM_URL, msg)),
            error=body_type.group(1) if body_type else None,
            status=int(status_m.group(1)) if status_m else None,
            raw=line,
        )
    if msg.startswith(COMPACTION_EVENT):
        # "compaction_event session=<8ch> mode=fast|cheap action=compact|..."
        # Tolerant: missing numeric fields default to None; never fatal.
        dry_raw = _first(RE_COMPACTION_DRY_RUN, msg)
        dry_val: bool | None = None
        if dry_raw is not None:
            dry_val = dry_raw == "true"
        pre_m = RE_COMPACTION_PRE_TOKENS.search(msg)
        post_m = RE_COMPACTION_POST_TOKENS.search(msg)
        ts_m = RE_COMPACTION_TURNS_SUMMARIZED.search(msg)
        td_m = RE_COMPACTION_TURNS_DROPPED.search(msg)
        st_m = RE_COMPACTION_SUMMARY_TOKENS.search(msg)
        return LogEvent(
            "compaction_event",
            ts,
            session=_session_from(msg),
            mode=(_first(RE_COMPACTION_MODE, msg) or "").lower() or None,
            action=_first(RE_COMPACTION_ACTION, msg),
            reason=_first(RE_COMPACTION_REASON, msg),
            pre_tokens=int(pre_m.group(1)) if pre_m else None,
            post_tokens=int(post_m.group(1)) if post_m else None,
            turns_summarized=int(ts_m.group(1)) if ts_m else None,
            turns_dropped=int(td_m.group(1)) if td_m else None,
            summary_tokens=int(st_m.group(1)) if st_m else None,
            dry_run=dry_val,
            raw=line,
        )
    if msg.startswith(COMPACTION_BACKSTOP):
        # "compaction_backstop action=dropped|exhausted dropped_turns=1 dropped_messages=2 ..."
        dt_m = RE_BACKSTOP_DROPPED_TURNS.search(msg)
        dm_m = RE_BACKSTOP_DROPPED_MESSAGES.search(msg)
        eb = re.search(r"estimated_before=(\d+)", msg)
        ea = re.search(r"estimated_after=(\d+)", msg)
        return LogEvent(
            "compaction_backstop",
            ts,
            action=_first(RE_COMPACTION_ACTION, msg),
            turns_dropped=int(dt_m.group(1)) if dt_m else None,
            dropped_messages=int(dm_m.group(1)) if dm_m else None,
            pre_tokens=int(eb.group(1)) if eb else None,
            post_tokens=int(ea.group(1)) if ea else None,
            raw=line,
        )
    if msg.startswith(COMPACTION_CHURN):
        # "compaction_churn session=<8ch> count=<n> rate_per_hour=<f> exceeds_target=<bool>"
        cc_m = RE_CHURN_COUNT.search(msg)
        cr_m = RE_CHURN_RATE.search(msg)
        churn_rate_val: float | None = None
        if cr_m:
            try:
                churn_rate_val = float(cr_m.group(1))
            except ValueError:
                churn_rate_val = None
        return LogEvent(
            "compaction_churn",
            ts,
            session=_session_from(msg),
            churn_count=int(cc_m.group(1)) if cc_m else None,
            churn_rate=churn_rate_val,
            raw=line,
        )
    return None


def iter_events(
    path: Path,
    window_start: datetime,
    window_end: datetime,
    margin: timedelta = timedelta(0),
) -> Iterator[LogEvent]:
    """Stream-parse ``path`` line by line, yielding only events whose
    timestamp falls inside ``[window_start - margin, window_end + margin]``.

    The file is never loaded into memory: lines are read one at a time and a
    cheap prefix check skips non-log lines before regex parsing.

    ``margin`` widens the yield window on both sides so the local-model
    utilization (busy-time) analysis can pair ``Stream started``/``Stream
    finished`` events that cross the analysis window boundary (the caller
    clips them back to ``[window_start, window_end]``). Defaults to zero so
    callers that only care about in-window events are unchanged.
    """
    lo = window_start - margin
    hi = window_end + margin
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 24 or not line[:4].isdigit():
                continue
            ev = parse_log_line(line)
            if ev is None:
                continue
            if lo <= ev.ts <= hi:
                ev.src_file = path.name
                yield ev


def discover_log_files(log_dir: Path, window_start: datetime) -> list[Path]:
    """Return the log files in ``log_dir`` that can overlap the analysis
    window, sorted by name.

    The live ``proxy.log`` is always included, and so is every rotated file
    (``proxy.log.YYYY-MM-DD_HH``). The name-encoded timestamp does not
    reliably bound a rotated file's content in this deployment — files
    routinely hold data well past their encoded rotation time (e.g.
    ``proxy.log.2026-08-07_03`` contains data until 09:03) — so any name- or
    mtime-based inclusion test risks silently dropping in-window data.
    ``iter_events`` per-line timestamp filtering remains the authoritative
    boundary check.

    ``window_start`` is retained for API compatibility; discovery no longer
    depends on it.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    candidates: list[Path] = []
    for p in sorted(log_dir.iterdir()):
        if p.is_file() and (p.name == "proxy.log" or p.name.startswith("proxy.log.")):
            candidates.append(p)
    return sorted(candidates, key=lambda p: p.name)
