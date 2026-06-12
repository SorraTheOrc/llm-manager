"""
Observability Module

Centralizes backend signal tracking (connect/read/timeout/other failures),
SSE client management for real-time status and log tail broadcasts, and
exception classification for observability.

Uses a lazy server import (_srv()) to access module-level state without
circular import issues.
"""

import asyncio
from typing import Any


# ---------------------------------------------------------------------------
# Lazy server import — avoids circular imports when server.py imports us
# ---------------------------------------------------------------------------
def _srv():
    import proxy.server as _m
    return _m


# ===================================================================
# Backend signal counters (connect/read/timeout/other/concurrency)
# ===================================================================

backend_signal_counts: dict = {
    "connect_failures": 0,
    "read_failures": 0,
    "timeout_failures": 0,
    "other_failures": 0,
    "concurrency_rejects": 0,
}


def _record_backend_signal(signal_name: str) -> None:
    """Increment a backend signal counter for observability."""
    srv = _srv()
    try:
        if signal_name in srv.backend_signal_counts:
            srv.backend_signal_counts[signal_name] = (
                int(srv.backend_signal_counts.get(signal_name, 0)) + 1
            )
    except Exception:
        pass


def _classify_backend_exception(exc: Exception) -> str:
    """Map backend transport exceptions to signal buckets."""
    import httpx

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "connect_failures"
    if isinstance(exc, (httpx.ReadError,)):
        return "read_failures"
    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
        return "timeout_failures"
    return "other_failures"


# ===================================================================
# SSE clients for real-time broadcasts
# ===================================================================

sse_clients: set[asyncio.Queue] = set()
log_tail_clients: set[asyncio.Queue] = set()
