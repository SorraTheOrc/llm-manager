"""Per-prompt-fingerprint exponential backoff for deterministic local 400s.

Why this exists
---------------
A session that re-sends a byte-identical conversation and receives a
deterministic local HTTP 400 (LP-0MTXEBQ4E001BMI2's ``local_http_400_deterministic``
streak) currently routes remote again on every single turn. Nothing records
that *this exact history* is provably rejected, so the loop can only be seen,
never throttled. This module supplies the missing identity: a stable
**prompt fingerprint** for a conversation, plus an in-memory exponential
backoff keyed by that fingerprint (1 min -> 5 min -> 15 min cap).

Keyed by fingerprint, not by session
------------------------------------
The same bad history must back off for *every* session that sends it (two
herdr workers looping on the same compacted conversation share one verdict),
so the backoff dict is keyed by fingerprint alone. The entry still records the
last ``session_id`` that touched it, which is what makes "did this session's
history change?" answerable: a session that sends a different fingerprint has
moved on, so the previous fingerprint's entry is dropped and the new history
starts at the bottom of the ladder.

Hot-path cost
-------------
Hashing a 100-message history on every request would be real overhead, so the
fingerprint is a **hash chain** (``h_n = sha256(h_{n-1} || canonical(msg_n))``)
cached per session. A repeat of the same history costs nothing but a cheap
head/tail sample, and a history that merely grew costs only the new messages —
the previously hashed prefix is reused. A rewritten history (compaction: same
length, different head or tail) is detected by the sample and re-hashed from
scratch.

Terminal breaker
----------------
A history that keeps being resent *after* its windows expire is not merely
slow, it is wedged: the client will never succeed until it changes the
conversation. When the shared ``resend_count`` for a fingerprint reaches
:data:`DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD` (configurable via
``deterministic_client_error_escalation_threshold``), :func:`session_wedged_error`
reports the terminal verdict the caller must answer with. The count resets on
exactly the two events that end the loop — a successful stream
(:func:`clear_backoff`) and a changed history
(:func:`note_prompt_fingerprint`) — so the breaker can only fire on an
unbroken run of rejections of the *same* history.

State is in-memory only and bounded (stale backoff entries are pruned, the
session cache is FIFO-evicted), consistent with the sibling cooldown and
streak mechanisms. A proxy restart loses the state: acceptable, because a
restart also loses the wedged client's in-flight request.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

#: Domain separator so a fingerprint can never collide with a bare SHA-256 of
#: some other payload this process hashes.
DOMAIN_SEPARATOR = b"llama-proxy/prompt-fingerprint/v1"

#: Message fields that are regenerated on every send of the *same* history.
#: Including them would give each resend a new fingerprint and defeat the
#: whole mechanism; conversation content lives in ``role``/``content``/
#: ``tool_calls``. Stripped at every depth, because tool-call and content-part
#: ids are regenerated the same way.
VOLATILE_KEYS = frozenset({"timestamp", "request_id", "id"})

#: Exponential ladder: first repeat 1 min, then 5 min, then 15 min cap.
BACKOFF_INITIAL_SECONDS = 60.0
BACKOFF_SCHEDULE_SECONDS: tuple[float, ...] = (BACKOFF_INITIAL_SECONDS, 300.0, 900.0)
BACKOFF_MAX_SECONDS = BACKOFF_SCHEDULE_SECONDS[-1]

#: Terminal circuit breaker (LP-0MW959H72003PF67 / LP-0MW95BE7X00574OL,
#: parent AC3): after this many consecutive rejections of the same history the
#: client is told the session is wedged instead of being handed another
#: typed 400 it has already ignored.
DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD = 5

#: Proxy-config key overriding the threshold above (read at the top level or
#: inside the ``server`` section, like the other proxy knobs).
ESCALATION_THRESHOLD_CONFIG_KEY = "deterministic_client_error_escalation_threshold"

#: Terminal error vocabulary returned once the breaker trips.
SESSION_WEDGED = "session_wedged"
SESSION_WEDGED_CAUSE = "repeated_deterministic_400"
SESSION_WEDGED_HINT = "compaction may help"

#: The terminal error's advertised wait is never shorter than this: a wedged
#: session must not be retried immediately, even when the last window lapsed.
SESSION_WEDGED_MIN_RETRY_AFTER_SECONDS = 60.0

#: Upper bounds keeping the module's in-memory state finite.
MAX_BACKOFF_ENTRIES = 1000
MAX_SESSION_CACHE_ENTRIES = 1000

#: ``fingerprint -> entry``; one shared verdict per distinct bad history.
_backoff_entries: dict[str, dict[str, Any]] = {}

#: ``session_id -> last fingerprint`` so a changed history can drop the
#: previous verdict for that session.
_session_fingerprints: dict[str, str] = {}

#: ``session_id -> cached hash-chain state`` for the hot path.
_hash_chain_cache: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _strip_volatile(value: Any) -> Any:
    """Return ``value`` with :data:`VOLATILE_KEYS` removed at every depth."""
    if isinstance(value, Mapping):
        return {key: _strip_volatile(item) for key, item in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_strip_volatile(item) for item in value]
    return value


def _canonical_message(message: Any) -> bytes:
    """Canonical, key-order-independent serialisation of one message."""
    return json.dumps(
        _strip_volatile(message),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _initial_digest() -> bytes:
    return hashlib.sha256(DOMAIN_SEPARATOR).digest()


def _extend_digest(digest: bytes, message: Any) -> bytes:
    """Fold one message into the chain digest."""
    hasher = hashlib.sha256(digest)
    hasher.update(_canonical_message(message))
    return hasher.digest()


def compute_prompt_fingerprint(messages: Sequence[Any]) -> str:
    """Return the stable fingerprint (hex SHA-256) of a message array.

    Stable across processes and Python invocations: the digest is built from
    canonical JSON (sorted keys, fixed separators) with the volatile fields in
    :data:`VOLATILE_KEYS` removed, never from ``hash()`` or ``repr()``.
    """
    digest = _initial_digest()
    for message in messages:
        digest = _extend_digest(digest, message)
    return digest.hex()


def fingerprint_for_session(session_id: str | None, messages: Sequence[Any]) -> str:
    """Fingerprint ``messages``, reusing this session's earlier hash-chain work.

    Three cases, cheapest first:

    1. **Same history re-sent** (head, tail and length all match the cached
       state): the cached digest is returned without walking the chain.
    2. **History grew** (same head and same message at the old boundary, more
       messages): only the new messages are hashed and the cached prefix
       digest is reused.
    3. **History was rewritten** (compaction): the whole array is re-hashed.

    Solving the fingerprint also notes it against the session, so a session
    that moved to a different history drops its previous backoff entry — see
    :func:`note_prompt_fingerprint`.
    """
    key = session_id or "unknown"
    items = list(messages)
    digest = _chain_digest_for(key, items)
    fingerprint = digest.hex()
    note_prompt_fingerprint(key, fingerprint)
    return fingerprint


def _prefix_unchanged(cached: dict[str, Any], items: list[Any], head: bytes) -> bool:
    """Cheap sample check that ``items`` still starts with the cached prefix.

    A grown history may reuse the cached chain digest only when the message
    array really is the cached one plus new tails: the first message and the
    message at the old boundary must be unchanged. Without the boundary check,
    a rewrite that keeps the head and happens to be longer would inherit the
    old prefix digest and fingerprint a *different* history as identical.
    """
    if cached["head"] != head:
        return False
    boundary_count = int(cached["count"])
    if boundary_count == 0:
        return cached["tail"] == b""
    return cached["tail"] == _canonical_message(items[boundary_count - 1])


def _chain_digest_for(key: str, items: list[Any]) -> bytes:
    cached = _hash_chain_cache.get(key)
    head = _canonical_message(items[0]) if items else b""
    tail = _canonical_message(items[-1]) if items else b""
    count = len(items)

    if cached is not None and cached["count"] == count and cached["head"] == head and cached["tail"] == tail:
        digest = cached["digest"]
    elif cached is not None and count > cached["count"] and _prefix_unchanged(cached, items, head):
        digest = cached["digest"]
        for message in items[cached["count"] :]:
            digest = _extend_digest(digest, message)
    else:
        digest = _initial_digest()
        for message in items:
            digest = _extend_digest(digest, message)

    _hash_chain_cache.pop(key, None)
    _hash_chain_cache[key] = {"count": count, "head": head, "tail": tail, "digest": digest}
    while len(_hash_chain_cache) > MAX_SESSION_CACHE_ENTRIES:
        _hash_chain_cache.pop(next(iter(_hash_chain_cache)))
    return digest


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


def retry_after_for_resend(resend_count: int) -> float:
    """Backoff for the ``resend_count``-th consecutive rejection (1-based)."""
    index = max(int(resend_count), 1) - 1
    if index >= len(BACKOFF_SCHEDULE_SECONDS):
        return BACKOFF_MAX_SECONDS
    return BACKOFF_SCHEDULE_SECONDS[index]


def get_backoff_entry(fingerprint: str) -> dict[str, Any] | None:
    """Return a copy of the live backoff entry for ``fingerprint``, if any."""
    entry = _backoff_entries.get(fingerprint)
    return dict(entry) if entry is not None else None


def record_backoff(
    fingerprint: str,
    session_id: str | None,
    retry_after: float | None = None,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Record one more rejection of ``fingerprint`` and open its backoff window.

    ``resend_count`` advances the exponential ladder; an explicit
    ``retry_after`` (e.g. from an upstream ``Retry-After``) overrides it
    without disturbing the count.
    """
    clock = time.monotonic() if now is None else float(now)
    previous = _backoff_entries.get(fingerprint)
    resend_count = (int(previous["resend_count"]) + 1) if previous is not None else 1
    delay = float(retry_after) if retry_after is not None else retry_after_for_resend(resend_count)
    entry: dict[str, Any] = {
        "fingerprint": fingerprint,
        "session_id": session_id or "unknown",
        "retry_after": delay,
        "resend_count": resend_count,
        "recorded_at": clock,
        "until": clock + delay,
    }
    _backoff_entries.pop(fingerprint, None)
    _backoff_entries[fingerprint] = entry
    _prune_backoff_entries(clock)
    return dict(entry)


def clear_backoff(fingerprint: str) -> bool:
    """Drop the backoff entry for ``fingerprint`` (successful stream / moved on).

    Returns whether an entry was actually removed, so a caller can log the
    transition without guessing.
    """
    return _backoff_entries.pop(fingerprint, None) is not None


def check_backoff_expired(fingerprint: str, *, now: float | None = None) -> bool:
    """Return ``True`` when ``fingerprint`` may be sent again.

    An expired entry is deliberately *kept*: the next rejection must continue
    the ladder (1 -> 5 -> 15 min) rather than restart it, which is what lets
    the terminal breaker count resends.
    """
    entry = _backoff_entries.get(fingerprint)
    if entry is None:
        return True
    clock = time.monotonic() if now is None else float(now)
    return clock >= float(entry["until"])


def remaining_backoff_seconds(fingerprint: str, *, now: float | None = None) -> float:
    """Seconds left in ``fingerprint``'s window (``0.0`` when none/expired)."""
    entry = _backoff_entries.get(fingerprint)
    if entry is None:
        return 0.0
    clock = time.monotonic() if now is None else float(now)
    return max(0.0, float(entry["until"]) - clock)


def note_prompt_fingerprint(session_id: str | None, fingerprint: str) -> str | None:
    """Note the history this session is now sending.

    A session that sends a *different* history has moved on (compaction,
    truncation, a new conversation), so its previous fingerprint's backoff is
    dropped and the new history starts at the bottom of the ladder. Returns
    the cleared previous fingerprint, or ``None`` when nothing changed.
    """
    key = session_id or "unknown"
    previous = _session_fingerprints.get(key)
    _session_fingerprints[key] = fingerprint
    if previous is None or previous == fingerprint:
        return None
    clear_backoff(previous)
    return previous


# ---------------------------------------------------------------------------
# Terminal breaker
# ---------------------------------------------------------------------------


def get_resend_count(fingerprint: str) -> int:
    """Return the consecutive rejections recorded for ``fingerprint``.

    Shared across sessions by design (the verdict belongs to the history, see
    the module docstring), and ``0`` for a history with no live entry — a
    fingerprint that was cleared by a success or a changed history starts
    again from zero.
    """
    entry = _backoff_entries.get(fingerprint)
    if entry is None:
        return 0
    return max(0, int(entry.get("resend_count", 0) or 0))


def escalation_threshold(config: Mapping[str, Any] | None = None) -> int:
    """Return the escalation threshold configured in ``config``.

    Reads :data:`ESCALATION_THRESHOLD_CONFIG_KEY` at the top level and, like
    the other proxy knobs, inside a ``server`` section. Anything that is not a
    positive integer falls back to
    :data:`DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD`: a mistyped knob
    must neither disable the breaker nor wedge a session on its first 400.
    """
    raw: Any = None
    if isinstance(config, Mapping):
        raw = config.get(ESCALATION_THRESHOLD_CONFIG_KEY)
        if raw is None:
            server = config.get("server")
            if isinstance(server, Mapping):
                raw = server.get(ESCALATION_THRESHOLD_CONFIG_KEY)
    if raw is None or isinstance(raw, bool):
        return DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD
    if value < 1:
        return DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD
    return value


def session_wedged_error(
    fingerprint: str,
    *,
    config: Mapping[str, Any] | None = None,
    threshold: int | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Return the terminal ``session_wedged`` payload, or ``None``.

    ``None`` means the breaker has not tripped: the caller keeps its existing
    behaviour (the typed 400 / normal fallback). The payload is a superset of
    the documented contract — ``error``/``cause``/``hint`` plus the
    ``retry_after`` the caller turns into a ``Retry-After`` header and the
    ``resend_count``/``fingerprint`` used for logging. ``threshold`` wins over
    ``config`` when both are given, so a caller that already resolved the knob
    does not read it twice.
    """
    limit = int(threshold) if threshold is not None else escalation_threshold(config)
    if limit < 1:
        limit = DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD
    count = get_resend_count(fingerprint)
    if count < limit:
        return None
    retry_after = max(
        SESSION_WEDGED_MIN_RETRY_AFTER_SECONDS,
        remaining_backoff_seconds(fingerprint, now=now),
    )
    return {
        "error": SESSION_WEDGED,
        "cause": SESSION_WEDGED_CAUSE,
        "hint": SESSION_WEDGED_HINT,
        "retry_after": int(math.ceil(retry_after)),
        "resend_count": count,
        "fingerprint": fingerprint,
    }


def _prune_backoff_entries(now: float) -> None:
    """Keep :data:`_backoff_entries` bounded; expired entries go first."""
    if len(_backoff_entries) <= MAX_BACKOFF_ENTRIES:
        return
    for fingerprint in [fp for fp, entry in _backoff_entries.items() if now >= float(entry["until"])]:
        del _backoff_entries[fingerprint]
    while len(_backoff_entries) > MAX_BACKOFF_ENTRIES:
        oldest = min(_backoff_entries, key=lambda fp: float(_backoff_entries[fp]["recorded_at"]))
        del _backoff_entries[oldest]


def reset() -> None:
    """Clear all in-memory state (proxy restart semantics, and test isolation)."""
    _backoff_entries.clear()
    _session_fingerprints.clear()
    _hash_chain_cache.clear()
