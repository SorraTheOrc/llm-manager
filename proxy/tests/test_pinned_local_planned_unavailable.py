"""Pinned local-preferred request on a known-until-blocked remote chain.

LP-0MVWC9CZH004OZYN AC7 (test-first, LP-0MVYVWZDM007F9B3). The 2026-10-31
08:16 incident shape:

- ``X-Local-Preference: local-preferred`` (the client is pinned to local),
- the local route is refused (the prefill-ETA admission guard falls through),
- every remote provider is blocked by a *known-until* gate: three
  ``available_times`` windows closed until 10:00 UTC, one usage-limit reset
  ~16 h away, one quota-exhausted account ~17 days away.

Today the client gets the local-capacity ``429 Local admission refused`` with
the local ETA timer, which says nothing about the 10:00 UTC window edge. AC7
requires the planned-unavailable response (or an equivalent) carrying the same
``retry_at`` / ``code``, so the client is told when the chain reopens.

Red phase: these tests fail against the pre-AC7 code (a 429 with a local
capacity timer). Green once LP-0MVYVX6PC003CSLQ lands.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

#: The incident instant: 08:16 UTC, inside the 06:00-10:00 scheduled gap.
_FIXED_NOW = datetime(2026, 10, 31, 8, 16, 0, tzinfo=UTC)

#: The opencode-go windows: closed 06:00-10:00 UTC.
_OPENCODE_WINDOWS = ["00:00-01:00", "04:00-06:00", "10:00-00:00"]

#: 10:00 UTC minus 08:16 UTC, in seconds.
_WINDOW_EDGE_SECONDS = 6240


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED_NOW


class _Request:
    """Request stub with ``state`` and the pin-local header."""

    def __init__(self, headers: dict[str, str] | None = None):
        self._body = json.dumps({"model": "plan"}).encode("utf-8")
        self.headers = dict(headers or {"X-Local-Preference": "local-preferred"})
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()
        self.state = SimpleNamespace()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _incident_chain() -> dict:
    """Local + the four blocked remote entries of the incident."""
    return {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
            {"name": "opencode-go", "type": "remote",
             "endpoint": "https://opencode.ai/zen/go",
             "available_times": _OPENCODE_WINDOWS},
            {"name": "opencode-go-2", "type": "remote",
             "endpoint": "https://opencode.ai/zen/go",
             "api_key_env": "OPENCODE_2_API_KEY",
             "available_times": _OPENCODE_WINDOWS},
            {"name": "opencode-go-3", "type": "remote",
             "endpoint": "https://opencode.ai/zen/go",
             "api_key_env": "OPENCODE_3_API_KEY",
             "available_times": _OPENCODE_WINDOWS},
            {"name": "munitoai-free", "type": "remote",
             "endpoint": "https://munitoai.example/v1",
             "api_key_env": "MUNITO_API_KEY"},
        ]
    }


def _config() -> dict:
    return {
        "provider_cooldown_seconds": 60,
        "max_retry_after_seconds": 100000,
        "server": {
            "session_slot_pool_size": 1,
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": 60,
            "contention_queue_max_depth": 4,
            "local_prefill_deadline_seconds": 0,
        },
    }


def _body(response: Response) -> dict:
    return json.loads(bytes(response.body))


async def _drive_incident(chain: dict, config: dict):
    """Drive the pinned incident shape through the real fallback chain.

    The local route is refused by the ETA admission guard (the ``fall-through``
    shape, not the local-only refusal); every remote is blocked, so no remote
    dispatch may run.
    """
    async def _local(_req, _path):
        raise AssertionError("the admission guard refuses before local dispatch")

    async def _remote(_req, _path, _pc):  # pragma: no cover - must never run
        raise AssertionError("every remote is blocked; none may be dispatched")

    with (
        patch("proxy.provider._maybe_queue_for_local_slot",
              AsyncMock(return_value=("admission_reject", "eta_exceeds_budget", 0.0))),
        patch("proxy.provider._get_local_concurrency_info",
              lambda *_a, **_k: (1, 1)),
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider.datetime", _FixedDateTime),
    ):
        return await provider.proxy_with_fallback(
            _Request(), "v1/chat/completions", chain, config
        )


def _seed_usage_limit_reset(chain: dict, seconds: int) -> str:
    """Seed the munitoai-free account's usage-limit reset; return its key."""
    entry = next(p for p in chain["providers"] if p["name"] == "munitoai-free")
    key = provider._usage_limit_account_key(entry)
    provider._usage_reset_at[key] = time.time() + seconds
    return key


# ======================================================================
# AC-T7a: the pinned incident is told the 10:00 UTC window edge
# ======================================================================

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Red phase (LP-0MVYVWZDM007F9B3): the pinned path returns the "
        "local-capacity 429 today; LP-0MVYVX6PC003CSLQ lands the "
        "planned-unavailable response and removes this marker."
    ),
)
@pytest.mark.asyncio
async def test_pinned_incident_is_told_the_window_edge():
    """AC7: a pinned request on a known-until-blocked chain gets 10:00Z.

    The honest answer is the 10:00 UTC window edge (6240 s away), not the
    local-capacity ETA timer and not the ~16 h usage reset or ~17 day quota.
    """
    chain = _incident_chain()
    key = _seed_usage_limit_reset(chain, 16 * 3600)
    try:
        response = await _drive_incident(chain, _config())
    finally:
        provider._usage_reset_at.pop(key, None)

    body = _body(response)
    assert body.get("code") == "no_provider_available_in_window", (
        "a pinned request on a known-until-blocked chain must be told the "
        f"window story, got code={body.get('code')!r}"
    )
    assert body.get("retryable") is True
    assert body.get("retry_at", "").startswith("2026-10-31T10:00"), (
        f"expected the 10:00 UTC window edge, got retry_at={body.get('retry_at')!r}"
    )
    assert response.headers.get("retry-after") == str(body.get("retry_after"))
    # The prose error stays the back-compatible string.
    assert body.get("error") == "All providers exhausted"


# ======================================================================
# AC-T7b: the refusal contract is untouched when the chain is not blocked
# ======================================================================

@pytest.mark.asyncio
async def test_pinned_refusal_keeps_the_local_capacity_429_when_remotes_are_ready():
    """AC7 scope: a non-blocked remote chain keeps today's local-capacity 429.

    The planned-unavailable response applies only to the known-until-blocked
    combination; a pinned request whose remote chain is not deterministically
    blocked keeps the capacity-verdict payload and its own timer.
    """
    chain = {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-ready", "type": "remote",
             "endpoint": "https://ready.example/v1"},
        ]
    }

    async def _local(_req, _path):
        raise AssertionError("the admission guard refuses before local dispatch")

    async def _remote(_req, _path, _pc):  # pragma: no cover - pinned: never used
        raise AssertionError("a pinned request must not fall through to remote")

    with (
        patch("proxy.provider._maybe_queue_for_local_slot",
              AsyncMock(return_value=("admission_reject", "eta_exceeds_budget", 0.0))),
        patch("proxy.provider._get_local_concurrency_info",
              lambda *_a, **_k: (1, 1)),
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider.datetime", _FixedDateTime),
    ):
        response = await provider.proxy_with_fallback(
            _Request(), "v1/chat/completions", chain, _config()
        )

    # The pinned refusal is unchanged: the local-capacity 429, not the new code.
    assert response.status_code == 429
    body = _body(response)
    assert body.get("code") == "local_admission_reject"
    assert body.get("code") != "no_provider_available_in_window"
