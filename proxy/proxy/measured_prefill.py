"""Measured prefill capture from llama-server ``prompt_progress`` (F2).

LP-0MVR7DAFF006HBRX / parent LP-0MVQIT9QY007QU27 (Option D).

Why
---

Every admission-time warmth predicate is a *prediction*: ``routing_resident_prefix``
decisions carry ``n_past_actual=none``, so ``resolve_resident_prefix`` reports
``reuse_ratio=1.0`` for a slot-owner session even when llama-server is about
to re-prefill the prompt from scratch (measured 2026-10-26: 29 of 116 deadline
evictions were ~56k prompts with priced ``new_tokens`` of 17-182, cancelled at
120 s). The measurement that actually tells warmth apart is llama-server's own
``prompt_progress`` event, emitted inside the streaming response when the
request carries ``return_progress: true`` (contract pinned by
:mod:`proxy.prompt_progress`, F1). Its first event carries
``cache = n_prompt_tokens_cache`` — the exact reused prefix — against
``total = n_tokens()``, so ``total - cache`` is the work actually being
computed.

What this module provides (purely additive — no decision changes here)
-----------------------------------------------------------------------

``inject_return_progress``
    Build the upstream request body for a local *streaming* dispatch with
    ``return_progress: true`` added. The key is added to a copy: the parsed
    client body (``body_json``) — which is what the proxy records/stores —
    never carries it, so the internal key cannot leak into recorded payloads
    (AC1).

``MeasuredPrefill``
    One dispatch's latest measurement: ``total``, ``cache``, ``processed``,
    ``time_ms`` plus the derived ``measured_cached_ratio`` (``cache / total``)
    and ``measured_new_tokens`` (``total - cache``). ``as_dict`` is what the
    dispatch record exposes (AC3).

``set_measured_prefill`` / ``get_measured_prefill``
    Per-``(model, session)`` store of the latest measurement (capped like the
    cached-ratio table), so consumers can read the measured reuse of the most
    recent local dispatch without reaching into the transient dispatch-lease
    record (which is released at the end of the stream).

``PREFILL_PROGRESS_EVENT`` / ``MeasurePrefill.log_fields``
    The ``prefill_progress_measured`` log grammar (AC6): ``total``, ``cache``,
    ``processed``, ``measured_cached_ratio`` and ``priced_new_tokens`` (plus
    the derived ``measured_new_tokens``), so the proxy-usage-analysis parser
    can attribute measured-vs-predicted error after deploy.

``capture_enabled``
    The feature switch. It is the same ``local_measured_cache_abort_enabled``
    key the early-abort wiring (F4) uses, so disabling the feature restores
    pre-change behaviour end to end: no ``return_progress`` is injected, no
    measurement is captured and no early release can fire (parent AC4).

Fail-open by contract (AC5)
---------------------------

Nothing here raises into a stream: injection failures return ``None`` (the
request is forwarded unchanged), and the capture bookkeeping in
``router.proxy_to_local`` is wrapped so a measurement problem degrades to "no
measurement" and leaves dispatch behaviour exactly as today.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from proxy.measured_cache_abort import AbortSettings
from proxy.prompt_progress import PromptProgress

#: Request key llama-server b8782 reads to emit ``prompt_progress`` events
#: (``params_from_json_cmpl``, ``server-task.cpp``). Injected proxy-side only.
RETURN_PROGRESS_KEY = "return_progress"

#: Field name the local dispatch-lease record exposes the measurement under.
DISPATCH_RECORD_KEY = "measured_prefill"

#: Log event emitted once per dispatch on the first observed progress event.
PREFILL_PROGRESS_EVENT = "prefill_progress_measured"

#: Cap on the per-``(model, session)`` store (mirrors the cached-ratio table).
MAX_MEASURED_PREFILL_ENTRIES = 1000


@dataclass(frozen=True)
class MeasuredPrefill:
    """One dispatch's latest measured ``prompt_progress`` observation.

    ``priced_new_tokens`` is the admission-time priced compute for the same
    dispatch (``estimated_tokens * (1 - cached_ratio)`` from the admission
    snapshot). It is recorded, never used to decide anything here: the
    measured-vs-priced comparison is the early-abort wiring's job (F4) and the
    log line's attribution job (AC6).
    """

    total: int | None
    cache: int | None
    processed: int | None
    time_ms: float | None
    priced_new_tokens: float | None = None

    @classmethod
    def from_progress(
        cls,
        progress: PromptProgress,
        *,
        priced_new_tokens: float | None = None,
    ) -> MeasuredPrefill:
        """Build a record from an F1 :class:`PromptProgress` observation."""
        return cls(
            total=progress.total,
            cache=progress.cache,
            processed=progress.processed,
            time_ms=progress.time_ms,
            priced_new_tokens=priced_new_tokens,
        )

    @property
    def measured_cached_ratio(self) -> float | None:
        """``cache / total`` — the measured KV-cache reuse ratio (AC3)."""
        if self.total is None or self.cache is None or self.total <= 0:
            return None
        return self.cache / self.total

    @property
    def measured_new_tokens(self) -> int | None:
        """``total - cache`` — tokens actually computed by llama-server (AC3)."""
        if self.total is None or self.cache is None:
            return None
        return self.total - self.cache

    def as_dict(self) -> dict[str, Any]:
        """Plain mapping (derived values included) for the dispatch record."""
        return {
            "total": self.total,
            "cache": self.cache,
            "processed": self.processed,
            "time_ms": self.time_ms,
            "measured_cached_ratio": self.measured_cached_ratio,
            "measured_new_tokens": self.measured_new_tokens,
            "priced_new_tokens": self.priced_new_tokens,
        }

    def log_fields(self) -> str:
        """``key=value`` fields for the ``prefill_progress_measured`` line."""
        ratio = self.measured_cached_ratio
        measured_new = self.measured_new_tokens
        return " ".join(
            (
                f"total={_fmt_int(self.total)}",
                f"cache={_fmt_int(self.cache)}",
                f"processed={_fmt_int(self.processed)}",
                f"time_ms={_fmt_float(self.time_ms, 1)}",
                "measured_cached_ratio="
                + (f"{ratio:.4f}" if ratio is not None else "unknown"),
                f"measured_new_tokens={_fmt_int(measured_new)}",
                f"priced_new_tokens={_fmt_float(self.priced_new_tokens, 0)}",
            )
        )


def _fmt_int(value: int | None) -> str:
    return "unknown" if value is None else str(int(value))


def _fmt_float(value: float | None, digits: int) -> str:
    if value is None:
        return "unknown"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(number):
        return "unknown"
    return f"{number:.{digits}f}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def capture_enabled(server_config: Mapping[str, Any] | None) -> bool:
    """Whether measured-prefill capture is enabled.

    Shares the ``local_measured_cache_abort_enabled`` rollback switch with the
    early-abort wiring (F4) so one config key turns the whole feature off
    (parent AC4: no ``return_progress``, no capture, no early release).
    Absent/empty config keeps the target behaviour enabled; a malformed value
    fails open to pre-change behaviour, exactly like the abort settings.
    """
    try:
        return AbortSettings.from_config(server_config).enabled
    except Exception:  # pragma: no cover - defensive, from_config never raises
        return False


# ---------------------------------------------------------------------------
# Upstream request injection (AC1)
# ---------------------------------------------------------------------------


def inject_return_progress(body_json: Any) -> bytes | None:
    """Serialise *body_json* with ``return_progress: true`` for a local stream.

    Returns ``None`` (no injection) when the body is not a mapping, is not a
    streaming request, or cannot be serialised — the caller then forwards the
    original body untouched (fail-open).

    The key is added to a **copy**: ``body_json`` is the client-visible /
    recorded payload and must stay unchanged (AC1). A client that already set
    ``return_progress`` keeps its own value; the proxy does not rewrite it.
    """
    if not isinstance(body_json, Mapping):
        return None
    if not body_json.get("stream"):
        return None
    if RETURN_PROGRESS_KEY in body_json:
        return None
    try:
        upstream = dict(body_json)
        upstream[RETURN_PROGRESS_KEY] = True
        return json.dumps(upstream).encode("utf-8")
    except (TypeError, ValueError):
        return None


def priced_new_tokens(
    admission_evidence: Any,
    estimated_tokens: Any,
) -> float | None:
    """The admission-time priced compute for this dispatch (AC6 field).

    Prefers the verbatim admission snapshot stamped on the request
    (``_local_admission_evidence["new_tokens"]`` — the number the routing
    model actually used, LP-0MVODJ3P6000AIUM). Falls back to the routing
    prompt estimate (i.e. a cold-priced dispatch) when no snapshot exists, and
    to ``None`` when neither is usable.
    """
    if isinstance(admission_evidence, Mapping):
        value = admission_evidence.get("new_tokens")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
        ):
            return float(value)
    if (
        isinstance(estimated_tokens, (int, float))
        and not isinstance(estimated_tokens, bool)
        and math.isfinite(float(estimated_tokens))
        and estimated_tokens >= 0
    ):
        return float(estimated_tokens)
    return None


# ---------------------------------------------------------------------------
# Per-(model, session) measurement store (AC3)
# ---------------------------------------------------------------------------

_measured_prefill: dict[tuple[str, str], MeasuredPrefill] = {}


def set_measured_prefill(
    model_name: str | None,
    session_id: str | None,
    measured: MeasuredPrefill,
) -> None:
    """Record the latest measurement for a ``(model, session)`` pair.

    Never raises: a metric path must not break a stream. The store is capped
    like the cached-ratio table (oldest entry evicted first).
    """
    if not model_name or not session_id or measured is None:
        return
    try:
        key = (model_name, session_id)
        if (
            key not in _measured_prefill
            and len(_measured_prefill) >= MAX_MEASURED_PREFILL_ENTRIES
        ):
            oldest = next(iter(_measured_prefill))
            _measured_prefill.pop(oldest, None)
        _measured_prefill[key] = measured
    except Exception:
        pass


def get_measured_prefill(
    model_name: str | None,
    session_id: str | None,
) -> MeasuredPrefill | None:
    """Latest measurement for a ``(model, session)`` pair, or ``None``."""
    if not model_name or not session_id:
        return None
    return _measured_prefill.get((model_name, session_id))


def clear_measured_prefill(
    model_name: str | None,
    session_id: str | None,
) -> None:
    """Forget the measurement for a ``(model, session)`` pair."""
    if not model_name or not session_id:
        return
    _measured_prefill.pop((model_name, session_id), None)


def reset_measured_prefill() -> None:
    """Clear the whole store (test isolation; also usable on model switch)."""
    _measured_prefill.clear()
