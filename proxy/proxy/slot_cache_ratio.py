"""Persist and restore the routing ``cached_ratio`` from a slot snapshot.

LP-0MVGD9YXH008O304 — the cold-cache routing gate derives

    new_tokens = estimated_tokens * (1 - cached_ratio)

where ``cached_ratio`` used to come *only* from ``provider._last_cached_ratio``,
an in-memory dict written solely from ``prompt_tokens_details.cached_tokens`` on
a previous local response.  Two consequences:

* a session whose KV prefix is sitting in a **persisted slot snapshot** was
  scored cold (82.8 % of routing decisions reported ``cached_ratio=0.00``), so
  every session above the economic cold threshold was permanently bypassed
  (``large_context_bypass``) even though llama-server could reuse the prefix;
* the dict is lost on every proxy restart, resetting the whole fleet to cold
  even though the KV snapshots survive on disk.

This module closes both gaps with a small **sidecar** written next to the slot
snapshot file (``slot_<session>.bin`` → ``slot_<session>.bin.ratio.json``):

* after a successful ``slot_save`` the proxy records how many prompt tokens are
  now in the saved KV cache (measured from the upstream ``usage`` block) plus a
  fingerprint of the request's message list;
* at routing time — including the first request after a proxy restart — the
  sidecar is read back, validated, and used to seed the routing ratio.

Correctness guards (registered in the intake's AC2/AC3):

* **prefix match** — the recorded message fingerprints must be a prefix of the
  incoming message list, so a changed system prompt or truncated history never
  earns credit;
* **cap** — the credited ratio is ``min(1.0, cached_tokens / estimated_tokens)``
  multiplied by a safety margin (``<= 1.0``), so restoration can never
  over-credit the measured value;
* **fail-open** — a missing/corrupt/expired sidecar, or any error, degrades to
  today's ``cached_ratio = 0`` behaviour rather than breaking dispatch.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("llama-proxy.provider")

SIDECAR_SUFFIX = ".ratio.json"
"""Suffix appended to the slot snapshot filename (``slot_x.bin`` → ``slot_x.bin.ratio.json``)."""

SIDECAR_VERSION = 1
"""Schema version of the sidecar payload; unknown versions are ignored."""

DEFAULT_SAFETY_MARGIN = 0.9
"""Fraction of the measured cached tokens credited by default.

llama.cpp can invalidate its own checkpoints (``forcing full prompt
re-processing`` is expected on this hybrid-architecture Qwen3 per
``AGENTS.md``), so a restored snapshot is not a guarantee of a zero-cost
prefix.  Crediting 90 % of the measurement keeps the routing decision honest
when a few checkpoints are dropped.
"""

DEFAULT_TTL_SECONDS = 86400.0
"""Sidecar age beyond which the snapshot is no longer credited (24 h)."""

SOURCE_RESTORED = "restored"
"""Ratio-source label recorded when seeding from a slot snapshot."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def sidecar_path(slot_filename: str | Path) -> Path:
    """Return the sidecar path for *slot_filename*."""
    return Path(str(slot_filename) + SIDECAR_SUFFIX)


def _config_float(server_config: dict | None, key: str, default: float) -> float:
    """Read a non-negative float from *server_config*, falling back to *default*."""
    if not isinstance(server_config, dict):
        return default
    raw = server_config.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value < 0:
        return default
    return value


def safety_margin(server_config: dict | None) -> float:
    """Return the configured safety margin, clamped to ``[0.0, 1.0]``."""
    configured = _config_float(
        server_config, "session_slot_ratio_safety_margin", DEFAULT_SAFETY_MARGIN
    )
    return min(1.0, configured)


def ttl_seconds(server_config: dict | None) -> float:
    """Return the configured sidecar TTL in seconds (0 disables expiry)."""
    return _config_float(
        server_config, "session_slot_ratio_ttl_seconds", DEFAULT_TTL_SECONDS
    )


# ---------------------------------------------------------------------------
# Prompt prefix fingerprints
# ---------------------------------------------------------------------------


def _content_text(message: Any) -> str:
    """Return a stable text form of a message's ``content`` field.

    Plain strings pass through; multimodal part lists are JSON-serialised with
    sorted keys so the fingerprint is deterministic across processes.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(content)


def message_fingerprints(messages: Any) -> list[str]:
    """Return one SHA-256 hex digest per message (role + content).

    Message-level granularity keeps the check tokenizer-free: a stored digest
    list that is a prefix of the incoming list proves the conversation prefix
    is unchanged, which is exactly the condition under which llama-server can
    reuse the slot's KV prefix.
    """
    if not isinstance(messages, (list, tuple)):
        return []
    fingerprints: list[str] = []
    for message in messages:
        role = ""
        if isinstance(message, dict):
            role = str(message.get("role") or "")
        digest = hashlib.sha256()
        digest.update(role.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_content_text(message).encode("utf-8"))
        fingerprints.append(digest.hexdigest())
    return fingerprints


def prefix_matches(stored: Any, incoming: Any) -> bool:
    """True when *stored* fingerprints cover a prefix of *incoming*.

    A changed system prompt (first element differs) or a history truncated from
    the front fails the check and earns no credit (AC2).  An incoming list
    shorter than the stored one is accepted when the overlapping elements
    match — the caller's token cap keeps the credit proportionate.
    """
    if not isinstance(stored, (list, tuple)) or not isinstance(incoming, (list, tuple)):
        return False
    if not stored or not incoming:
        return False
    overlap = min(len(stored), len(incoming))
    if overlap <= 0:
        return False
    return list(stored[:overlap]) == list(incoming[:overlap])


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------


def write_slot_ratio_sidecar(
    slot_filename: str | Path,
    *,
    session_id: str | None,
    model: str | None,
    cached_tokens: int,
    prompt_tokens: int,
    messages: Any,
    saved_at: datetime | None = None,
) -> bool:
    """Persist the post-save cache state next to the slot snapshot.

    ``cached_tokens`` is the measured number of tokens now resident in the
    saved KV cache (the served request's ``prompt_tokens``), so a later request
    whose message list extends this one can be credited for them.

    Writes atomically (temp file + replace) and never raises: a sidecar failure
    must not fail the request or the slot save.
    """
    if not slot_filename or not session_id or not model:
        return False
    try:
        payload = {
            "version": SIDECAR_VERSION,
            "session_id": session_id,
            "model": model,
            "cached_tokens": int(cached_tokens),
            "prompt_tokens": int(prompt_tokens),
            "messages": message_fingerprints(messages),
            "saved_at": (saved_at or datetime.now(UTC)).isoformat(),
        }
        path = sidecar_path(slot_filename)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(path)
        return True
    except Exception:
        logger.debug("slot ratio sidecar write failed", exc_info=True)
        return False


def read_slot_ratio_sidecar(slot_filename: str | Path) -> dict | None:
    """Read and structurally validate the sidecar for *slot_filename*.

    Returns ``None`` when the sidecar is absent, unreadable, corrupt, or of an
    unknown schema version.
    """
    if not slot_filename:
        return None
    try:
        path = sidecar_path(slot_filename)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("slot ratio sidecar read failed", exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != SIDECAR_VERSION:
        return None
    if not isinstance(payload.get("messages"), list):
        return None
    try:
        payload["cached_tokens"] = int(payload.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return None
    return payload


def delete_slot_ratio_sidecar(slot_filename: str | Path | None) -> None:
    """Remove the sidecar for *slot_filename* (best effort)."""
    if not slot_filename:
        return
    try:
        sidecar_path(slot_filename).unlink(missing_ok=True)
    except Exception:
        logger.debug("slot ratio sidecar delete failed", exc_info=True)


# ---------------------------------------------------------------------------
# Ratio seeding
# ---------------------------------------------------------------------------


def ratio_from_sidecar(
    meta: dict | None,
    estimated_tokens: int | float | None,
    margin: float = DEFAULT_SAFETY_MARGIN,
) -> float:
    """Return the credited ratio for *meta*, capped by the measurement.

    ``ratio = min(1.0, cached_tokens / estimated_tokens) * margin`` — the
    measured fraction is the ceiling (AC3) and the margin can only reduce it.
    Returns ``0.0`` for missing/invalid measurements.
    """
    if not isinstance(meta, dict):
        return 0.0
    try:
        cached = float(meta.get("cached_tokens") or 0)
        estimate = float(estimated_tokens or 0)
    except (TypeError, ValueError):
        return 0.0
    if cached <= 0 or estimate <= 0:
        return 0.0
    measured = min(1.0, cached / estimate)
    try:
        clamped_margin = min(1.0, max(0.0, float(margin)))
    except (TypeError, ValueError):
        clamped_margin = DEFAULT_SAFETY_MARGIN
    return min(1.0, measured * clamped_margin)


def _is_expired(saved_at: Any, ttl: float, now: datetime | None = None) -> bool:
    """True when *saved_at* (ISO-8601) is older than *ttl* seconds."""
    if ttl <= 0:
        return False
    if not isinstance(saved_at, str) or not saved_at:
        return True
    try:
        saved = datetime.fromisoformat(saved_at)
    except (TypeError, ValueError):
        return True
    if saved.tzinfo is None:
        saved = saved.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return (reference - saved).total_seconds() > ttl


def _candidate_slot_filenames(
    slot_path: str | Path, session_id: str, endpoint: str | None
) -> list[str]:
    """Return the snapshot filenames a session could own, most specific first.

    Mirrors ``session._build_slot_context``: per-endpoint subdirectories
    (``{host}-{port}``) keep different llama-server instances' snapshots apart
    (LP-0MRPILSMW004T4H8).  The legacy root directory is tried as a fallback so
    snapshots written before per-endpoint layout (or by callers that resolve no
    endpoint) are still found.  Imported lazily to avoid a module-level import
    cycle (``session`` → ``provider``).
    """
    try:
        from proxy.session import _endpoint_host_port, _slot_filename_for_session
    except Exception:
        return []
    candidates: list[str] = []
    if endpoint:
        try:
            host, port = _endpoint_host_port(endpoint)
            candidates.append(
                _slot_filename_for_session(
                    session_id, Path(slot_path) / f"{host}-{port}"
                )
            )
        except Exception:
            pass
    try:
        root = _slot_filename_for_session(session_id, Path(slot_path))
    except Exception:
        return candidates
    if root not in candidates:
        candidates.append(root)
    return candidates


def seed_cached_ratio(
    server_config: dict | None,
    body_json: Any,
    model_name: str | None,
    session_id: str | None,
    estimated_tokens: int | None,
    endpoint: str | None = None,
    now: datetime | None = None,
) -> str | None:
    """Seed the routing ratio from the persisted slot snapshot.

    Returns the ratio-source label (``"restored"``) when a credit was applied,
    otherwise ``None`` — meaning the caller keeps today's behaviour
    (``cached_ratio`` unchanged, typically 0.0 for a cold session).

    Validation order (each failure degrades to ``None``):

    1. slot persistence configured (``session_slot_save_path``) and the
       snapshot file exists;
    2. a readable, versioned sidecar exists;
    3. the sidecar's model matches the local provider's ``llama_model``;
    4. the sidecar is not older than ``session_slot_ratio_ttl_seconds``;
    5. the sidecar's message fingerprints prefix-match the incoming messages
       (AC2 — changed system prompt / truncated history earns nothing).
    """
    if not session_id or not model_name or not isinstance(server_config, dict):
        return None
    slot_path = server_config.get("session_slot_save_path")
    if not slot_path:
        return None
    slot_filename = None
    for candidate in _candidate_slot_filenames(slot_path, session_id, endpoint):
        try:
            if Path(candidate).exists():
                slot_filename = candidate
                break
        except Exception:
            continue
    if not slot_filename:
        return None

    meta = read_slot_ratio_sidecar(slot_filename)
    if meta is None:
        return None
    stored_model = meta.get("model")
    if stored_model and str(stored_model) != str(model_name):
        return None
    if _is_expired(meta.get("saved_at"), ttl_seconds(server_config), now):
        return None

    messages = body_json.get("messages") if isinstance(body_json, dict) else None
    if not prefix_matches(meta.get("messages"), message_fingerprints(messages)):
        logger.debug(
            "slot_ratio_sidecar skipped reason=prefix_mismatch session=%s model=%s",
            session_id[:8] if session_id else "unknown",
            model_name,
        )
        return None

    ratio = ratio_from_sidecar(meta, estimated_tokens, safety_margin(server_config))
    if ratio <= 0.0:
        return None

    try:
        from proxy.provider import set_cached_ratio

        set_cached_ratio(
            model_name,
            session_id,
            ratio,
            source=SOURCE_RESTORED,
        )
    except Exception:
        logger.debug("slot ratio seeding failed", exc_info=True)
        return None
    logger.info(
        "slot_ratio_restored session=%s model=%s cached_tokens=%d "
        "estimated_tokens=%s cached_ratio=%.3f",
        session_id[:8] if session_id else "unknown",
        model_name,
        int(meta.get("cached_tokens") or 0),
        estimated_tokens,
        ratio,
    )
    return SOURCE_RESTORED
