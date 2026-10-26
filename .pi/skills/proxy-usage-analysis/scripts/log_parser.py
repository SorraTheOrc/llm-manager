"""Streaming, line-based parser for llama-proxy structured INFO log lines.

The proxy logs lines of the form::

    %Y-%m-%d %H:%M:%S,%f - LEVEL - message

Only a small set of structured message prefixes carry the signal this skill
needs (``Stream started``, ``Stream finished``, ``Fallback triggered``,
``routing_skip_local``, ``routing_skip_provider_context``,
``local_dispatch_denied``). Everything else is ignored. Parsing is tolerant:
missing fields default to ``None`` and a line that cannot be parsed is skipped
(never fatal), so truncated request payloads and log-format drift do not break
the analysis.

Attribution markers (LP-0MVAPNC2900145RH) are appended to restart and
mode-switch log lines, carrying ``source=<value>``, ``actor=<value>``, and
``reason=<value>`` key=value pairs. The parser extracts these from
``Mode scheduler``, ``Grandfathering``, ``Mode-switch restart spawned``,
``restart_services``, and ``Proxy restart attribution:`` lines and stores
them on the resulting :class:`LogEvent` as ``source``, ``actor``, and
``reason`` fields — ``None`` when no markers are present (backward
compatible with historical unattributed logs). See also
:func:`_attribution_from` for the extraction logic.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


# Log line prefix: "2026-08-02 13:58:32,260 - INFO - <message>"
def _normalise_bucket(value: str | None) -> str | None:
    """Normalise a parsed priority/mode marker to the internal bucket key.

    The proxy now logs ``priority=standard|low`` (AC6); historical logs carry
    ``mode=fast|cheap``. Both map to the internal budget buckets (``cheap``/
    ``low`` -> "cheap", everything else -> "fast").
    """
    if not value:
        return None
    v = str(value).strip().lower()
    return "cheap" if v in ("cheap", "low") else "fast"


LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - (\w+) - (.*)$")

# Field extractors (tolerant: may match nothing).
RE_PROVIDER = re.compile(r"\bprovider=(\S+)")
RE_MODEL = re.compile(r"\bmodel=(\S+)")
RE_SESSION = re.compile(r"\bsession=([A-Za-z0-9_.-]+)")

# TTFT (Time to First Token) measurement lines.
# Local stream path (router.py): "dispatch_first_byte_ms=<ms> dispatch_to_first_byte_ms=<ms> session=<id> model=<model>"
# Remote stream path (proxy_remote.py): "ttft_seconds=<s> session=<id> provider=<provider> model=<model>"
RE_DISPATCH_FIRST_BYTE = re.compile(
    r"dispatch_first_byte_ms=([\d.]+)"
)
# Service-time split of the same local dispatch line (LP-0MV6CFMU9005WU2E):
# ``dispatch_to_first_byte_ms`` = total minus the single-slot queue wait, and
# ``queue_wait_ms`` is that queued component. ``cache_state``/``cached_ratio``
# describe the prompt-cache outcome of the request.
RE_DISPATCH_TO_FIRST_BYTE = re.compile(r"dispatch_to_first_byte_ms=([\d.]+)")
RE_QUEUE_WAIT_MS = re.compile(r"queue_wait_ms=([\d.]+)")
RE_CACHE_STATE = re.compile(r"\bcache_state=(\w+)")
RE_CACHED_RATIO = re.compile(r"\bcached_ratio=([\d.]+)")
RE_TTFT_SECONDS = re.compile(
    r"ttft_seconds=([\d.]+)"
)
RE_TOKENS = re.compile(r"\btokens=(\d+)/(\d+)/(\d+)")
RE_ENTRY = re.compile(r"\bentry=(\S+)")
#: Buffered (non-streamed) reply marker on terminal ``Stream finished`` lines
#: (LP-0MVMHNVKT003PRIK AC1 non-streaming path): such lines carry measured
#: ``tokens=`` but are not paired into busy/stream counts — there is no
#: ``Stream started`` for them. Falling back to ``None`` (streaming or
#: pre-marker logs) keeps every historic consumer unchanged.
RE_STREAM_FALSE = re.compile(r"\bstream=false\b")
RE_ERROR_DETAIL = re.compile(r"\berror=([^\s,}]+)")
RE_FALLBACK = re.compile(
    r"Fallback triggered for model=(\S+?), from=(\S+?), to=(\S+?), "
    r"reason=([^=]*?)(?=\s+\w+=|\s*$)"
)
# Fallback evidence appended after ``reason=`` (LP-0MVMBMSWR0036RES):
# ``session=<uuid|unknown> attempt=<n>``, plus ``limit=``/``remaining=``/
# ``waited=`` when the reason is budget-driven (``rate_limited*``). Absent on
# pre-change rotated logs, in which case every field degrades to ``None`` and
# the legacy nearest-line attribution heuristic still applies.
RE_FALLBACK_ATTEMPT = re.compile(r"\battempt=(\d+)")
RE_FALLBACK_BUDGET_LIMIT = re.compile(r"\blimit=(\d+)")
RE_FALLBACK_BUDGET_REMAINING = re.compile(r"\bremaining=(\d+)")
RE_FALLBACK_BUDGET_WAITED = re.compile(r"\bwaited=([\d.]+)")
# routing_skip_local reason sits between "reason=" and the "→" arrow.
RE_ROUTING_SKIP_REASON = re.compile(r"\breason=([\w_]+)\s*→")
# Provider context-ceiling skip fields (LP-0MVMBM1Y3001TPRT):
#   routing_skip_provider_context provider=<p> type=remote
#     status=provider_context_exceeded estimated_tokens=<n> limit=<n>
#     reason=provider_context_exceeded session=<uuid>
# ``estimated_tokens`` is the proxy's routing-time prompt estimate and
# ``limit`` the per-entry ``max_context_tokens`` ceiling it exceeded. Both are
# optional downstream (older rotated logs may omit ``limit``).
RE_SKIP_ESTIMATED_TOKENS = re.compile(r"\bestimated_tokens=(\d+)")
RE_SKIP_LIMIT = re.compile(r"\blimit=(\d+)")
RE_PROVIDER_TYPE = re.compile(r"\btype=([A-Za-z0-9_.-]+)")
# Chain-hold cycle stamp (LP-0MU5AIAAY003KVM0): the proxy tags
# ``routing_skip_local`` lines with ``chain_hold_cycle=<N>`` (cycle 0 = first
# pre-hold cycle, incremented on every hold restart). Tolerant: absent on
# pre-chain-hold rotated logs → None (treated as cycle 0 downstream).
RE_CHAIN_HOLD_CYCLE = re.compile(r"\bchain_hold_cycle=(\d+)")
RE_DISPATCH_DENIED = re.compile(r"session=([A-Za-z0-9_.-]+) owner=([A-Za-z0-9_.-]+) active=(\d+)")
# Bare ``reason=<token>`` extractor for lines whose reason is not followed by
# the routing-skip arrow (e.g. ``contention_queue_fallback_after_queue ...
# reason=local_lease_active``).
RE_REASON_KV = re.compile(r"\breason=([\w_]+)")

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
# New proxy emits priority=<standard|low> (LP-0MVDT94F0003VHUA AC6);
# historical logs carry mode=<fast|cheap>. Both parse into the
# internal budget bucket key (cheap/low -> cheap, else fast).
RE_COMPACTION_PRIORITY = re.compile(r"\b(?:priority|mode)=(\w+)")
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

# Prefill-ETA admission-guard decision (LP-0MVGDUV5E005UWBQ): the proxy emits
# ``local_admission_guard decision=admit|reject reason=<verdict> enforced=<bool>
# eta_prefill_s=<f> budget_s=<f> ...`` at the contention-queue decision point.
# ``decision`` is the guard's verdict; ``enforced=false`` marks an ``observe``
# rollout (logged but not applied).
RE_LOCAL_ADMISSION_DECISION = re.compile(r"\bdecision=(admit|reject)\b")
RE_LOCAL_ADMISSION_ENFORCED = re.compile(r"\benforced=(true|false)\b")
RE_LOCAL_ADMISSION_ETA = re.compile(r"\beta_prefill_s=([\d.]+)")
RE_LOCAL_ADMISSION_BUDGET = re.compile(r"\bbudget_s=([\d.]+)")
RE_LOCAL_ADMISSION_REASON = re.compile(r"\breason=([A-Za-z_]+)")

# Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J): the proxy emits
# ``routing_idle_admit provider=... model=... reason=economic_bypass_idle
# idle_seconds=<f> threshold_seconds=<f> estimated_tokens=<n> ...`` when an
# economic cold-cache bypass is overridden because the GPU was demonstrably
# idle. ``idle_seconds`` is the measured idle age at admission.
RE_ROUTING_IDLE_SECONDS = re.compile(r"\bidle_seconds=([\d.]+)")
RE_IDLE_ADMIT_THRESHOLD = re.compile(r"\bthreshold_seconds=([\d.]+)")

# Residency-aware admission + eviction evidence (LP-0MVN1IQTF0037WTG AC5).
# ``none`` is a *meaningful* value for ``n_past_actual`` / ``reuse_ratio`` /
# ``slot_owner_session`` (measurement unavailable, or the session owns no
# slot), so it parses to ``None`` and is never read as zero.
RE_RESIDENCY_DECISION = re.compile(
    r"\bdecision=([A-Za-z_]+)"
)
RE_N_PAST_ACTUAL = re.compile(r"\bn_past_actual=(\d+|none)\b")
RE_REUSE_RATIO = re.compile(r"\breuse_ratio=([\d.]+|none)\b")
RE_SLOT_OWNER_SESSION = re.compile(r"\bslot_owner_session=(\S+)")
RE_RESIDENCY_SOURCE = re.compile(r"\bresidency_source=([A-Za-z_]+)")
RE_TENANT_LIMIT = re.compile(r"\btenants=(\d+|none)\b")
RE_TENANT_OCCUPANTS = re.compile(r"\boccupants=(\d+)")
RE_EVICTION_REASON = re.compile(r"\breason=([A-Za-z_]+)")
RE_SLOT_ID = re.compile(r"\bslot=(\d+|none)\b")
RE_SNAPSHOT_PLANNED = re.compile(r"\bsnapshot_planned=(true|false|unknown)\b")

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

# Rotated log naming: two mechanisms produce two naming patterns:
#   - In-process TimedRotatingFileHandler → proxy.log.YYYY-MM-DD_HH  (dot)
#   - Logrotate safety net (dateformat -%Y-%m-%d_%H) → proxy.log-YYYY-MM-DD_HH  (dash)
# Both can hold data in the same time window, so discovery includes both.
# The name-encoded time does NOT reliably bound a file's content span in
# this deployment — rotated files routinely hold data past it — so discovery
# includes every rotated file and iter_events is the only boundary.

STREAM_STARTED = "Stream started"
STREAM_FINISHED = "Stream finished"
FALLBACK = "Fallback triggered"
ROUTING_SKIP = "routing_skip_local"
# Per-provider context-ceiling skip (LP-0MVMBM1Y3001TPRT): the estimated prompt
# exceeds the remote entry's ``max_context_tokens``, so that entry is skipped
# for this request only. It is *not* a provider failure (no cooldown, no
# failure-domain exclusion) — it is a per-request size mismatch, and the line
# carries the session UUID, the provider, the estimate and the ceiling.
ROUTING_SKIP_PROVIDER_CONTEXT = "routing_skip_provider_context"
DISPATCH_DENIED = "local_dispatch_denied"
# Local dispatch attempt marker (LP-0MVKUKG3F0035I85): the proxy logs one
# ``[local] POST http://<backend>/... `` line for **every** attempt it makes at
# the local llama-server, including attempts that never yield a first byte (a
# wedged prefill lease, a cancelled dispatch). ``Stream started: provider=local``
# only appears once the request is admitted, so it cannot see work that was
# attempted-but-never-served; this marker is the per-hour local-dispatch
# attempt evidence the availability classification needs.
LOCAL_POST = "[local] POST"
CONTENTION_DISPATCH = "contention_queue_dispatch"
CONTENTION_FALLBACK_AFTER_QUEUE = "contention_queue_fallback_after_queue"
LOCAL_ADMISSION_GUARD = "local_admission_guard"
ROUTING_IDLE_ADMIT = "routing_idle_admit"
# Residency-aware admission decision (LP-0MVN1IQTF0037WTG AC5): the proxy emits
# ``routing_resident_prefix provider=local model=... decision=<verdict>
# n_past_actual=<n|none> estimated_tokens=<n> reuse_ratio=<f|none>
# slot_owner_session=<id|none> residency_source=<measured|session_slot|
# non_resident|unknown> min_ratio=<f> tenants=<n> occupants=<n> session=<id>``
# once per admission decision. ``decision`` is one of ``admit``,
# ``resident_prefix_insufficient``, ``tenant_limit`` (plus
# ``large_context_bypass`` for a live-priced cold miss). The fail-open path
# (unmeasured view — no fresh snapshot, no ownership) logs
# ``admit_predicted`` / ``skip_predicted`` with ``n_past_actual=none`` so
# residency-blind decisions stay observable (LP-0MVN4LANP0069QQM AC5).
ROUTING_RESIDENT_PREFIX = "routing_resident_prefix"
# Shared post-acceptance eviction event (LP-0MVN1IQTF0037WTG AC6,
# LP-0MVN3KMGI009BZ0D AC7): ``local_eviction session=... reason=<vocab>
# endpoint=... slot=<n|none> slot_owner_session=... tenants=...
# snapshot_planned=<bool>`` emitted at ERROR level whenever a resident
# session's lease/slot is reclaimed.
#: A committed `local_eviction` line as emitted by `_record_local_eviction`
#: (used to distinguish it from the after-acceptance event below, which shares
#: the prefix but is its OWN kind).
LOCAL_EVICTION = "local_eviction"
# Post-acceptance eviction (LP-0MVN4LCED00661MX, parent AC7): the proxy emits
# ``local_eviction_after_acceptance session=<id> reason=<vocab>
# estimated_tokens=<n> new_tokens=<n> eta_prefill_s=<f|unknown>
# elapsed_s=<f> slot_id=<n|unknown> llama_task_id=<id|unknown> endpoint=<...>``
# at ERROR level when an ACCEPTED local dispatch is released before producing a
# first byte. It shares the `local_eviction` prefix, so it must be matched
# explicitly and classified as its own kind — never folded into the
# `local_eviction` tenancy event (different fields, different meaning).
LOCAL_EVICTION_AFTER_ACCEPTANCE = "local_eviction_after_acceptance"
RE_EVICTION_AA_ESTIMATED = re.compile(r"\bestimated_tokens=(\d+)")
RE_EVICTION_AA_NEW_TOKENS = re.compile(r"\bnew_tokens=(\d+)")
RE_EVICTION_AA_ETA = re.compile(r"\beta_prefill_s=([\d.]+|unknown)")
RE_EVICTION_AA_ELAPSED = re.compile(r"\belapsed_s=([\d.]+)")
RE_EVICTION_AA_SLOT_ID = re.compile(r"\bslot_id=(\d+|unknown)")
RE_EVICTION_AA_TASK_ID = re.compile(r"\bllama_task_id=(\S+)")
# Prefill-deadline cancellation (LP-0MVOAQDB4002D56G, parent AC4): the proxy
# logs ``prefill_deadline_exceeded session=<id> deadline=<n>s → cancelling
# local dispatch and detouring to the next provider`` at WARNING level when an
# admitted local prefill overruns the deadline. Its own kind so AC4's
# deadline-rate check reads from the report instead of a raw log scan.
PREFILL_DEADLINE_EXCEEDED = "prefill_deadline_exceeded"
RE_PREFILL_DEADLINE_SECONDS = re.compile(r"\bdeadline=(\d+(?:\.\d+)?)s")
# Provider-chain exhaustion: ``All providers exhausted for model=<route>;
# unavailable={...}; usage_limit_reset=<...>`` — the client-visible failure
# the AC4 "no increase in 503s / exhaustion" check counts. Distinct from the
# upstream per-provider errors (``upstream_http_error``), which are counted by
# status code.
PROVIDER_EXHAUSTED = "All providers exhausted"
RE_PROVIDER_EXHAUSTED_MODEL = re.compile(
    r"All providers exhausted for model=(\S+?);"
)
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

# Restart markers (LP-0MUQO82LJ008F1FS). A mode switch is a FULL proxy restart
# that kills in-flight local streams, so a local stream error that lands next
# to one of these markers is a restart kill, not an upstream fault. Two log
# lines bracket a restart:
#
#   - ``Mode-switch restart spawned: <script> --restart`` (mode.py
#     ``_spawn_restart``; logged AFTER the bounded drain completes, immediately
#     before the process is killed)
#   - ``restart_services: router-mode restart complete (N slots)``
#     (lifecycle.py; the new process is up)
#
# Both are parsed into timestamped ``restart`` events so ``aggregation`` can
# correlate stream errors. The slot count and script path are corroborating
# evidence only and are never parsed.
MODE_SWITCH_RESTART_SPAWNED = "Mode-switch restart spawned:"
ROUTER_MODE_RESTART_COMPLETE = "restart_services: router-mode restart complete"

# Attribution markers (LP-0MVC0S6U5003SEDJ): LP-0MVAPNC2900145RH appended
# ``source=``/``actor=``/``reason=`` key=value markers to the restart and
# mode-switch log paths (and the startup attribution line below). Values are
# matched tolerantly: ``source``/``actor`` are single tokens (no spaces);
# ``reason`` captures non-greedily up to the next known ``key=`` token (or
# end of line) so space-containing reasons survive (e.g. the
# ``manual (no reason provided)`` start-proxy.sh default).
RE_ATTRIBUTION_SOURCE = re.compile(r"\bsource=([a-z0-9_]+)")
RE_ATTRIBUTION_ACTOR = re.compile(r"\bactor=([A-Za-z0-9_.-]+)")
RE_ATTRIBUTION_REASON = re.compile(
    r"\breason=(.*?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|$)"
)
# Tolerant ``mode=<fast|cheap>`` extractor for attribution-bearing restart
# lines (the set_mode() spawn carries the mode being switched to,
# LP-0MVAPNC2900145RH). Reused on restart kinds so the report can surface
# the requested mode (LP-0MVC144KS001PDR4); stays ``None`` when absent.
RE_MODE_KV = re.compile(r"\bmode=(\w+)")

# Startup attribution line (LP-0MVAPNC2900145RH): the restarted process
# (server.py ``_log_restart_attribution``) echoes the reason — and optional
# source/actor — exported into its environment by the starting process
# (start-proxy.sh --restart / mode.py ``_spawn_restart``) right before the
# ``Starting LLama Proxy Server`` marker. A fresh unattributed start logs
# nothing (backward compatible).
RESTART_ATTRIBUTION_PREFIX = "Proxy restart attribution:"

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
    ``routing_skip``, ``provider_context_skip``, ``dispatch_denied``,
    ``local_dispatch``, ``contention_dispatch``,
    ``contention_fallback_after_queue``, ``idle_admit``, ``mode_switch``,
    ``restart``,
    ``compaction_event``, ``compaction_backstop``, ``compaction_churn``, or an
    error kind (``stream_error``, ``stream_finish_error``, ``slot_save_error``,
    ``backend_retry``, ``upstream_http_error``). Only the fields relevant to
    each kind are populated.
    """

    kind: str
    ts: datetime
    provider: str | None = None
    model: str | None = None
    session: str | None = None
    # Attribution markers (LP-0MVAPNC2900145RH): populated on ``mode_switch``,
    # ``restart`` (all sub-kinds), and ``stream_started`` events that carry
    # attribution key=value markers. Three fields, each ``None`` when absent:
    #
    # - ``source`` (LP-0MVC0S6U5003SEDJ, LP-0MVAPNC2900145RH): lower-case token
    #   identifying the trigger origin — one of ``scheduler``, ``worker``,
    #   ``api``, or ``manual``. Set on mode-switch, restart-spawned,
    #   restart-complete, and startup-attribution lines.
    # - ``actor`` (LP-0MVC0S6U5003SEDJ, LP-0MVAPNC2900145RH): the actor
    #   identifier (session UUID or service name) that initiated the action.
    #   Always ``None`` on ``stream_started`` events.
    # - ``reason`` (LP-0MVAPNC2900145RH): free-form attribution reason.
    #   On restart events this carries the ``reason=`` value from
    #   ``start-proxy.sh --restart`` (e.g. "manual (no reason provided)").
    #   On mode-switch events this carries the scheduler/actor reason.
    #   Always ``None`` on non-attribution kinds (stream_started, fallback,
    #   stream_finished, etc.).
    source: str | None = None
    actor: str | None = None
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
    # Fallback evidence (LP-0MVMBMSWR0036RES, ``fallback`` kind only):
    # ``attempt_index`` is the per-session fallback ordinal parsed from
    # ``attempt=``; ``budget_limit``/``budget_remaining``/``budget_waited``
    # carry the budget snapshot emitted with a ``rate_limited*`` reason.
    # All ``None`` on pre-change lines (and on every other kind).
    attempt_index: int | None = None
    budget_limit: int | None = None
    budget_remaining: int | None = None
    budget_waited: float | None = None
    signal: str | None = None
    src_file: str | None = None
    # 1-based line number inside ``src_file`` (set by ``iter_events``); with
    # ``src_file`` it is the evidence pointer for a parsed line.
    line_no: int | None = None
    raw: str | None = None
    # Provider context-ceiling skip fields (``provider_context_skip`` kind,
    # LP-0MVMBM1Y3001TPRT): the routing-time prompt estimate that exceeded the
    # entry's ceiling, that ceiling (``limit``), and the entry's ``type=``
    # (``remote`` / ``local``). Each is ``None`` when the line omits it, so
    # older rotated logs keep parsing (AC8).
    estimated_tokens: int | None = None
    limit: int | None = None
    provider_type: str | None = None
    # Operating-mode field (mode_switch kind only): "fast" | "cheap".
    mode: str | None = None
    # Restart-marker field (restart kind only): "spawned" | "complete"
    # (LP-0MUQO82LJ008F1FS).
    restart_kind: str | None = None
    # Root-cause classification for stream error events
    # (stream_finish_error / stream_error), set by ``aggregation.aggregate``:
    # one of ``upstream_stall_after_content``, ``mode_switch_restart``,
    # ``remote_chain_exhaustion``, ``other``. None for non-stream errors
    # (LP-0MUQO82LJ008F1FS).
    root_cause: str | None = None
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
    # TTFT (Time to First Token) in milliseconds, from local or remote stream
    # logging (LP-0MTSSM5SO003PKU0). Populated by dispatch_first_byte_ms=
    # (local) or ttft_seconds= (remote) log lines.
    ttft_ms: float | None = None
    # Prefill breakdown carried by the local ``dispatch_first_byte_ms=`` line
    # (LP-0MVJN025B001I8AL): ``ttft_ms`` is the dispatch→first-byte *total*,
    # ``dispatch_to_first_byte_ms`` is the same interval minus the queued wait
    # (``queue_wait_ms``), and ``cache_state``/``cached_ratio`` report the
    # prompt-cache outcome. All four are ``None`` on remote ``ttft_seconds=``
    # lines and on historical logs that predate the split.
    dispatch_to_first_byte_ms: float | None = None
    queue_wait_ms: float | None = None
    cache_state: str | None = None
    cached_ratio: float | None = None
    # Chain-hold cycle stamp (LP-0MU5AIAAY003KVM0): 0 = first (pre-hold)
    # cycle, incremented on every hold restart; ``None`` when the line does
    # not carry the token (pre-chain-hold logs). Populated on
    # ``routing_skip`` events; aggregation treats ``> 0`` as hold-cycle
    # re-emissions rather than genuine demand.
    chain_hold_cycle: int | None = None
    # Prefill-ETA admission-guard fields (LP-0MVGDUV5E005UWBQ,
    # ``local_admission_guard`` kind): the guard's verdict (``admit`` /
    # ``reject``), whether it was applied (``false`` in ``observe`` mode), and
    # the projected prefill ETA / wait budget in seconds.
    decision: str | None = None
    enforced: str | None = None
    eta_prefill_s: str | None = None
    budget_s: str | None = None
    # Idle-aware local admission (LP-0MVGD9Z8Z0055Y5J, ``idle_admit`` kind):
    # the idle age at admission and the configured threshold. Populated on
    # ``idle_admit`` events only.
    idle_seconds: str | None = None
    threshold_seconds: str | None = None
    # Residency-aware admission evidence (LP-0MVN1IQTF0037WTG AC5,
    # ``residency_decision`` kind): the measured reusable prefix, the request
    # estimate, the resulting reuse ratio and which session owned the slot at
    # decision time (``residency_source`` records whether the ratio was
    # measured, inferred from slot ownership, or absent). ``n_past_actual`` /
    # ``reuse_ratio`` / ``slot_owner_session`` are ``None`` when the line
    # reports ``none`` — never coerced to zero.
    n_past_actual: int | None = None
    reuse_ratio: float | None = None
    slot_owner_session: str | None = None
    residency_source: str | None = None
    tenant_limit: int | None = None
    tenant_occupants: int | None = None
    # Eviction evidence (LP-0MVN1IQTF0037WTG AC6, ``local_eviction`` kind): the
    # reclaimed slot, whether the evicted session's KV snapshot was planned,
    # and the closed reason vocabulary in ``reason``.
    slot_id: int | None = None
    snapshot_planned: bool | None = None
    # Post-acceptance eviction evidence (LP-0MVN4LCED00661MX, parent AC7).
    # ``eta_prefill_seconds``/``elapsed_seconds`` are floats with a distinct
    # name from the admission guard's string-valued ``eta_prefill_s`` so the
    # two events can never be conflated.
    new_tokens: int | None = None
    eta_prefill_seconds: float | None = None
    elapsed_seconds: float | None = None
    llama_task_id: str | None = None
    # Prefill-deadline overrun (``prefill_deadline_exceeded`` kind): the
    # configured deadline that was exceeded, in seconds.
    deadline_seconds: float | None = None
    # Buffered-reply marker on ``stream_finished`` events
    # (LP-0MVMHNVKT003PRIK AC1 non-streaming path): ``False`` when the line
    # carried ``stream=false`` (a single JSON reply, not a chunked stream), so
    # consumers can keep it out of stream pairing while still using its measured
    # ``tokens=``. ``None`` on streaming lines and on pre-marker logs.
    streamed: bool | None = None


def _first(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1) if m else None


def _float(pattern: re.Pattern, text: str) -> float | None:
    """Tolerant float extraction for a key=value field (``None`` if absent)."""
    token = _first(pattern, text)
    if token is None:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _int(pattern: re.Pattern, text: str) -> int | None:
    """Tolerant int extraction for a key=value field (``None`` if absent)."""
    token = _first(pattern, text)
    if token is None:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _session_from(text: str) -> str | None:
    token = _first(RE_SESSION, text)
    if token is None or token == "unknown":
        return None
    return token


@dataclass(frozen=True)
class PrefillMatch:
    """One local stream paired FIFO with its ``dispatch_first_byte_ms`` event.

    ``ttft`` is ``None`` when the stream produced no first-byte line: an
    aborted request (client disconnect, stream error) or a logging gap. Such
    streams are **unknown prefill** — they must be reported as such rather
    than silently dropped from the count (LP-0MVJN025B001I8AL AC4).
    """

    stream: LogEvent
    ttft: LogEvent | None = None

    @property
    def known_prefill(self) -> bool:
        """Whether this stream's prefill duration is observable."""
        return self.ttft is not None and self.ttft.ttft_ms is not None

    @property
    def prefill_seconds(self) -> float | None:
        """Dispatch→first-byte seconds, or ``None`` when unknown."""
        if self.ttft is None or self.ttft.ttft_ms is None:
            return None
        return self.ttft.ttft_ms / 1000.0

    @property
    def dispatch_ts(self) -> datetime | None:
        """Back-calculated dispatch time (``ttft.ts - ttft_ms``), or ``None``.

        The dispatch line is written at first-byte time and carries only a
        duration, so the prefill interval is
        ``[dispatch_ts, ttft.ts]`` (LP-0MVJGCQ421005IXGC intake brief).
        """
        if self.ttft is None or self.ttft.ttft_ms is None:
            return None
        return self.ttft.ts - timedelta(milliseconds=self.ttft.ttft_ms)


def pair_ttft_to_streams(
    ttft_events: Iterable[LogEvent],
    started_events: Iterable[LogEvent],
) -> list[PrefillMatch]:
    """Pair local streams with their TTFT events per session, FIFO by time.

    TTFT lines carry ``session=``, so the Nth ``dispatch_first_byte_ms``
    event of a session pairs with that session's Nth ``stream_started`` event
    (the same FIFO convention ``aggregation.compute_busy_stats`` uses for
    start/finish pairs). Streams are returned sorted by their start timestamp;
    streams with no TTFT paired are included with ``ttft=None`` so callers can
    count them as unknown prefill instead of losing them. TTFT events beyond a
    session's stream count have no stream to attribute to and are ignored.
    Session-less events cannot be paired and are skipped.
    """
    starts: dict[str, list[LogEvent]] = {}
    for ev in started_events:
        if ev.session is not None:
            starts.setdefault(ev.session, []).append(ev)
    ttfts: dict[str, list[LogEvent]] = {}
    for ev in ttft_events:
        if ev.session is not None:
            ttfts.setdefault(ev.session, []).append(ev)

    matches: list[PrefillMatch] = []
    for session_id, session_starts in starts.items():
        ordered_starts = sorted(session_starts, key=lambda e: e.ts)
        ordered_ttfts = sorted(ttfts.get(session_id, []), key=lambda e: e.ts)
        for index, stream in enumerate(ordered_starts):
            ttft = ordered_ttfts[index] if index < len(ordered_ttfts) else None
            matches.append(PrefillMatch(stream=stream, ttft=ttft))
    matches.sort(key=lambda m: m.stream.ts)
    return matches


def _attribution_from(msg: str) -> tuple[str | None, str | None, str | None]:
    """Tolerantly extract the ``source=``/``actor=``/``reason=`` attribution
    markers (LP-0MVAPNC2900145RH) from a log message.

    Any of the three may be absent (historical logs) — each then stays
    ``None``. ``reason`` captures greedily to the next known ``key=`` token
    (or end of line) so reasons containing spaces survive.
    """
    return (
        _first(RE_ATTRIBUTION_SOURCE, msg),
        _first(RE_ATTRIBUTION_ACTOR, msg),
        _first(RE_ATTRIBUTION_REASON, msg),
    )


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
            # ``stream=false`` marks a buffered reply's terminal line
            # (LP-0MVMHNVKT003PRIK): measured tokens, no stream pairing.
            streamed=False if RE_STREAM_FALSE.search(msg) else None,
        )
    if msg.startswith(FALLBACK):
        m2 = RE_FALLBACK.search(msg)
        if m2 is None:
            return None
        fmodel, src, dst, reason = m2.groups()
        return LogEvent(
            "fallback",
            ts,
            reason=_normalize_reason(reason.strip() or None),
            src=src,
            dst=dst,
            # Session-bearing since LP-0MVMBMSWR0036RES; pre-change rotated
            # logs have no ``session=`` and parse to ``None``.
            session=_session_from(msg),
            attempt_index=_int(RE_FALLBACK_ATTEMPT, msg),
            budget_limit=_int(RE_FALLBACK_BUDGET_LIMIT, msg),
            budget_remaining=_int(RE_FALLBACK_BUDGET_REMAINING, msg),
            budget_waited=_float(RE_FALLBACK_BUDGET_WAITED, msg),
        )
    if msg.startswith(ROUTING_SKIP_PROVIDER_CONTEXT):
        # "routing_skip_provider_context provider=munitoai-free type=remote
        #  status=provider_context_exceeded estimated_tokens=195787 limit=150000
        #  reason=provider_context_exceeded session=<uuid>"
        # A per-provider context-ceiling skip: session-bearing, so it is
        # attributable evidence of *why* a remote attempt was not made. The
        # ``status=`` value duplicates ``reason=`` (both
        # ``provider_context_exceeded``) and is therefore not stored twice.
        return LogEvent(
            "provider_context_skip",
            ts,
            provider=_first(RE_PROVIDER, msg),
            provider_type=_first(RE_PROVIDER_TYPE, msg),
            session=_session_from(msg),
            reason=_first(RE_REASON_KV, msg),
            estimated_tokens=_int(RE_SKIP_ESTIMATED_TOKENS, msg),
            limit=_int(RE_SKIP_LIMIT, msg),
            raw=line,
        )
    if msg.startswith(ROUTING_SKIP):
        cycle_m = RE_CHAIN_HOLD_CYCLE.search(msg)
        return LogEvent(
            "routing_skip",
            ts,
            session=_session_from(msg),
            reason=_normalize_reason(_first(RE_ROUTING_SKIP_REASON, msg)),
            chain_hold_cycle=int(cycle_m.group(1)) if cycle_m else None,
        )
    if msg.startswith(LOCAL_POST):
        # ``[local] POST http://... body={...} session_id=<uuid> session={...}``.
        # The request body is echoed (truncated) into the line and may itself
        # contain ``session=``/``provider=`` text, so no field is extracted
        # here: the event is a per-attempt marker, counted by hour only.
        return LogEvent("local_dispatch", ts)
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
        # (optionally suffixed with ``reason=<token>``, e.g. ``local_lease_active``)
        return LogEvent(
            "contention_fallback_after_queue",
            ts,
            provider=_first(RE_PROVIDER, msg),
            session=_session_from(msg),
            queued_duration=_first(RE_CONTENTION_DURATION, msg),
            reason=_normalize_reason(_first(RE_REASON_KV, msg)),
        )
    if msg.startswith(ROUTING_IDLE_ADMIT):
        # "routing_idle_admit provider=local model=Qwen3
        #  reason=economic_bypass_idle idle_seconds=412.3 threshold_seconds=60.0
        #  estimated_tokens=64000 new_tokens=61000 cached_ratio=0.05
        #  cache_state=cold session=<id>"
        return LogEvent(
            "idle_admit",
            ts,
            provider=_first(RE_PROVIDER, msg),
            model=_first(RE_MODEL, msg),
            session=_session_from(msg),
            reason=_normalize_reason(_first(RE_LOCAL_ADMISSION_REASON, msg)),
            idle_seconds=_first(RE_ROUTING_IDLE_SECONDS, msg),
            threshold_seconds=_first(RE_IDLE_ADMIT_THRESHOLD, msg),
        )
    if msg.startswith(LOCAL_ADMISSION_GUARD):
        # "local_admission_guard decision=reject reason=eta_exceeds_budget
        #  enforced=true eta_prefill_s=468.0 budget_s=120.0 priority=low ..."
        m2 = RE_LOCAL_ADMISSION_DECISION.search(msg)
        if m2 is None:
            return None
        return LogEvent(
            "local_admission_guard",
            ts,
            provider=_first(RE_PROVIDER, msg),
            model=_first(RE_MODEL, msg),
            session=_session_from(msg),
            reason=_normalize_reason(_first(RE_LOCAL_ADMISSION_REASON, msg)),
            decision=m2.group(1),
            enforced=_first(RE_LOCAL_ADMISSION_ENFORCED, msg),
            eta_prefill_s=_first(RE_LOCAL_ADMISSION_ETA, msg),
            budget_s=_first(RE_LOCAL_ADMISSION_BUDGET, msg),
        )
    if msg.startswith(ROUTING_RESIDENT_PREFIX):
        # "routing_resident_prefix provider=local model=Qwen3 decision=admit
        #  n_past_actual=49103 estimated_tokens=49103 reuse_ratio=1.0000
        #  slot_owner_session=<id> residency_source=measured min_ratio=0.50
        #  tenants=2 occupants=1 session=<id>"
        decision = _first(RE_RESIDENCY_DECISION, msg)
        if decision is None:
            return None
        raw_n_past = _first(RE_N_PAST_ACTUAL, msg)
        raw_ratio = _first(RE_REUSE_RATIO, msg)
        owner = _first(RE_SLOT_OWNER_SESSION, msg)
        return LogEvent(
            "residency_decision",
            ts,
            provider=_first(RE_PROVIDER, msg),
            model=_first(RE_MODEL, msg),
            session=_session_from(msg),
            decision=decision,
            estimated_tokens=_int(RE_SKIP_ESTIMATED_TOKENS, msg),
            n_past_actual=(
                int(raw_n_past) if raw_n_past and raw_n_past != "none" else None
            ),
            reuse_ratio=(
                float(raw_ratio) if raw_ratio and raw_ratio != "none" else None
            ),
            slot_owner_session=None if not owner or owner == "none" else owner,
            residency_source=_first(RE_RESIDENCY_SOURCE, msg),
            tenant_limit=(
                int(_first(RE_TENANT_LIMIT, msg))
                if _first(RE_TENANT_LIMIT, msg)
                and _first(RE_TENANT_LIMIT, msg) != "none"
                else None
            ),
            tenant_occupants=(
                int(_first(RE_TENANT_OCCUPANTS, msg))
                if _first(RE_TENANT_OCCUPANTS, msg)
                else None
            ),
        )
    if msg.startswith(PREFILL_DEADLINE_EXCEEDED):
        # "prefill_deadline_exceeded session=<id> deadline=120s → cancelling
        #  local dispatch and detouring to the next provider"
        # Session-less lines are still counted (the AC4 rate does not depend on
        # attribution); only the deadline token is optional.
        return LogEvent(
            PREFILL_DEADLINE_EXCEEDED,
            ts,
            session=_session_from(msg),
            deadline_seconds=_float(RE_PREFILL_DEADLINE_SECONDS, msg),
        )
    if msg.startswith(PROVIDER_EXHAUSTED):
        # "All providers exhausted for model=v1/chat/completions;
        #  unavailable={'local-qwen3': 59}; usage_limit_reset=None"
        return LogEvent(
            "provider_exhausted",
            ts,
            model=_first(RE_PROVIDER_EXHAUSTED_MODEL, msg),
        )
    if msg.startswith(LOCAL_EVICTION_AFTER_ACCEPTANCE):
        # "local_eviction_after_acceptance session=<id> reason=prefill_deadline
        #  estimated_tokens=40000 new_tokens=40000 eta_prefill_s=148.1
        #  elapsed_s=92.4 slot_id=1 llama_task_id=unknown endpoint=default"
        # A missing reason makes the line unparseable (the whole point of the
        # event is its cause), so it is skipped rather than guessed.
        reason = _first(RE_EVICTION_REASON, msg)
        if reason is None:
            return None
        raw_eta = _first(RE_EVICTION_AA_ETA, msg)
        raw_slot_id = _first(RE_EVICTION_AA_SLOT_ID, msg)
        raw_task_id = _first(RE_EVICTION_AA_TASK_ID, msg)
        raw_elapsed = _first(RE_EVICTION_AA_ELAPSED, msg)
        return LogEvent(
            LOCAL_EVICTION_AFTER_ACCEPTANCE,
            ts,
            session=_session_from(msg),
            reason=reason,
            slot_id=(
                int(raw_slot_id)
                if raw_slot_id and raw_slot_id != "unknown"
                else None
            ),
            estimated_tokens=_int(RE_EVICTION_AA_ESTIMATED, msg),
            new_tokens=_int(RE_EVICTION_AA_NEW_TOKENS, msg),
            eta_prefill_seconds=(
                float(raw_eta) if raw_eta and raw_eta != "unknown" else None
            ),
            elapsed_seconds=float(raw_elapsed) if raw_elapsed else None,
            llama_task_id=(
                None if not raw_task_id or raw_task_id == "unknown" else raw_task_id
            ),
        )
    if msg.startswith(LOCAL_EVICTION):
        # "local_eviction session=<id> reason=lease_evicted endpoint=default
        #  slot=1 slot_owner_session=<id> tenants=2 snapshot_planned=true"
        reason = _first(RE_EVICTION_REASON, msg)
        if reason is None:
            return None
        raw_slot = _first(RE_SLOT_ID, msg)
        raw_planned = _first(RE_SNAPSHOT_PLANNED, msg)
        return LogEvent(
            "local_eviction",
            ts,
            session=_session_from(msg),
            reason=reason,
            slot_id=int(raw_slot) if raw_slot and raw_slot != "none" else None,
            slot_owner_session=(
                None
                if not _first(RE_SLOT_OWNER_SESSION, msg)
                or _first(RE_SLOT_OWNER_SESSION, msg) == "none"
                else _first(RE_SLOT_OWNER_SESSION, msg)
            ),
            tenant_limit=(
                int(_first(RE_TENANT_LIMIT, msg))
                if _first(RE_TENANT_LIMIT, msg)
                and _first(RE_TENANT_LIMIT, msg) != "none"
                else None
            ),
            snapshot_planned=(
                None
                if raw_planned in (None, "unknown")
                else raw_planned == "true"
            ),
        )
    if msg.startswith(MODE_SWITCH_PREFIX):
        # "Mode scheduler: applied scheduled mode cheap" — the mode that the
        # scheduler applied at this timestamp (LP-0MSM5K4TX004MICX). Since
        # LP-0MVAPNC2900145RH the scheduler also stamps the line with
        # ``source=scheduler actor=mode-scheduler`` (LP-0MVC0S6U5003SEDJ).
        m = RE_MODE_SWITCH.search(msg)
        if m is None:
            return None
        mode = m.group(1).lower()
        if mode not in ("fast", "cheap"):
            return None
        source, actor, reason = _attribution_from(msg)
        return LogEvent("mode_switch", ts, mode=mode, source=source, actor=actor, reason=reason)
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
        # No attribution markers are emitted on this line today; the tolerant
        # extractors pick them up if a future format adds them
        # (LP-0MVC0S6U5003SEDJ, LP-0MVAPNC2900145RH).
        source, actor, reason = _attribution_from(msg)
        return LogEvent("mode_switch", ts, mode=mode, source=source, actor=actor, reason=reason)
    if msg.startswith(MODE_SWITCH_RESTART_SPAWNED):
        # "Mode-switch restart spawned: /path/start-proxy.sh --restart" — the
        # kill is imminent (logged after the bounded drain). Since
        # LP-0MVAPNC2900145RH the line carries the attribution captured at
        # set_mode() time (``source=``/``actor=``/``mode=``/``reason=``
        # markers) when callers self-identify (LP-0MVC0S6U5003SEDJ).
        source, actor, reason = _attribution_from(msg)
        mode = (_first(RE_MODE_KV, msg) or "").lower() or None
        return LogEvent(
            "restart",
            ts,
            restart_kind="spawned",
            raw=line,
            source=source,
            actor=actor,
            reason=reason,
            mode=mode,
        )
    if msg.startswith(ROUTER_MODE_RESTART_COMPLETE):
        # "restart_services: router-mode restart complete (2 slots)" — the
        # restarted proxy is up (corroborating evidence; slot count ignored).
        # Tolerant: no attribution markers (or mode=) are emitted on this
        # line today, but the extractors pick them up if a future format adds
        # them (LP-0MVC0S6U5003SEDJ, LP-0MVAPNC2900145RH).
        source, actor, reason = _attribution_from(msg)
        mode = (_first(RE_MODE_KV, msg) or "").lower() or None
        return LogEvent(
            "restart",
            ts,
            restart_kind="complete",
            raw=line,
            source=source,
            actor=actor,
            reason=reason,
            mode=mode,
        )
    if msg.startswith(RESTART_ATTRIBUTION_PREFIX):
        # "Proxy restart attribution: source=api actor=herdr reason=manual
        # (no reason provided)" — echoed by the restarted process right
        # before the ``Starting LLama Proxy Server`` marker; carries the
        # definitive source/actor/reason of this restart
        # (LP-0MVAPNC2900145RH, LP-0MVC0S6U5003SEDJ). A ``mode=`` marker
        # (when exported by the starting process) is extracted too so the
        # report can surface the requested mode (LP-0MVC144KS001PDR4).
        source, actor, reason = _attribution_from(msg)
        mode = (_first(RE_MODE_KV, msg) or "").lower() or None
        return LogEvent(
            "restart",
            ts,
            restart_kind="startup",
            raw=line,
            source=source,
            actor=actor,
            reason=reason,
            mode=mode,
        )
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
            mode=_normalise_bucket(_first(RE_COMPACTION_PRIORITY, msg)),
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

    # TTFT (Time to First Token) lines (LP-0MTSSM5SO003PKU0).
    # Local stream path: "dispatch_first_byte_ms=<ms> dispatch_to_first_byte_ms=<ms> session=<id> model=<model>"
    # Remote stream path: "ttft_seconds=<s> session=<id> provider=<provider> model=<model>"
    dispatch_m = RE_DISPATCH_FIRST_BYTE.search(msg)
    ttft_m = RE_TTFT_SECONDS.search(msg)
    if dispatch_m or ttft_m:
        # Convert remote ttft_seconds (float) to milliseconds for consistency
        ttft_ms_val = None
        if ttft_m:
            try:
                ttft_ms_val = float(ttft_m.group(1)) * 1000.0
            except (ValueError, TypeError):
                pass
        if dispatch_m:
            try:
                ttft_ms_val = float(dispatch_m.group(1))
            except (ValueError, TypeError):
                pass
        if ttft_ms_val is not None:
            return LogEvent(
                "ttft",
                ts,
                provider=_first(RE_PROVIDER, msg),
                model=_first(RE_MODEL, msg),
                session=_session_from(msg),
                ttft_ms=ttft_ms_val,
                dispatch_to_first_byte_ms=_float(RE_DISPATCH_TO_FIRST_BYTE, msg),
                queue_wait_ms=_float(RE_QUEUE_WAIT_MS, msg),
                cache_state=_first(RE_CACHE_STATE, msg),
                cached_ratio=_float(RE_CACHED_RATIO, msg),
                raw=line,
            )

    return None


def open_log_text(path: Path):
    """Open a proxy log file for text reading, transparently decompressing ``.gz``.

    Rotation may compress older files (logrotate ``compress``, or the
    in-process handler's ``.gz`` retention). Detecting the ``.gz`` suffix here
    keeps every caller free of compression concerns; without it a compressed
    file is read as plain text, yields no parseable lines, and is silently
    counted as scanned (real case: ``proxy.log.2026-09-13_00.gz`` held 778
    stream starts that contributed nothing).
    """
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


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

    Rotated files may be gzipped (``.gz`` suffix). When detected, the file is
    transparently decompressed so callers never need to distinguish between
    plain and compressed rotated logs.
    """
    lo = window_start - margin
    hi = window_end + margin
    with open_log_text(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            if len(line) < 24 or not line[:4].isdigit():
                continue
            ev = parse_log_line(line)
            if ev is None:
                continue
            if lo <= ev.ts <= hi:
                ev.src_file = path.name
                ev.line_no = line_no
                yield ev


def discover_log_files(log_dir: Path, window_start: datetime) -> list[Path]:
    """Return the log files in ``log_dir`` that can overlap the analysis
    window, sorted by name.

    The live ``proxy.log`` is always included, and so is every rotated file.
    Two rotation mechanisms produce two naming patterns in this deployment:

    - In-process ``TimedRotatingFileHandler`` → ``proxy.log.YYYY-MM-DD_HH``
    - Logrotate safety net (``dateformat -%Y-%m-%d_%H``) → ``proxy.log-YYYY-MM-DD_HH``

    Both can hold data in the same time window, so both are included. The
    name-encoded timestamp does not reliably bound a rotated file's content
    in this deployment — files routinely hold data well past their encoded
    rotation time (e.g. ``proxy.log.2026-08-07_03`` contains data until
    09:03) — so any name- or mtime-based inclusion test risks silently
    dropping in-window data. ``iter_events`` per-line timestamp filtering
    remains the authoritative boundary check.

    ``window_start`` is retained for API compatibility; discovery no longer
    depends on it.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    candidates: list[Path] = []
    for p in sorted(log_dir.iterdir()):
        if p.is_file() and (
            p.name == "proxy.log"
            or p.name.startswith("proxy.log.")
            or p.name.startswith("proxy.log-")
        ):
            candidates.append(p)
    return sorted(candidates, key=lambda p: p.name)
