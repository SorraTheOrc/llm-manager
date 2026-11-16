"""
Test fixtures for measured prefill scenarios (LP-0MW7V2DK400365ER / AC5, AC12).

Provides deterministic, reusable fixtures that simulate the prefill scenarios
required by the downstream feature children:

- ``concurrent_slots_fixture`` — two llama-server slots both in prefill
  simultaneously, with independent progress counters (parent AC1).
- ``stalled_task_fixture`` — a slot whose progress never advances for a
  configurable duration (parent AC2).
- ``progressing_task_fixture`` — a slot whose progress advances at a
  configurable rate (parent AC3).
- ``stderr_with_progress`` / ``stderr_without_progress`` — stubbed
  llama-server stderr output mimicking ``slot update_slots … n_tokens = N``
  lines (parent AC4).
- ``slots_timeout_fixture`` — simulates the ``/slots`` endpoint timing out
  (parent AC5, ``query_timeout`` under long prefill).
- ``stubbed_slots_payload_with_progress`` /
  ``stubbed_slots_payload_without_progress`` — stubbed ``/slots`` payloads
  for asserting the measured vs fail-safe paths (parent AC6).

All fixtures are hermetic — no live llama-server, no real HTTP.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_httpx_client(response_factory):
    """Build an ``httpx.AsyncClient``-like object that returns *response_factory*.

    *response_factory* can be a single ``httpx.Response`` or a callable
    ``(request) -> httpx.Response``.
    """
    if callable(response_factory):
        _factory = response_factory
    else:
        def _factory(_r):
            return response_factory

    class _Client:
        def __init__(self):
            self.base_url = ""

        async def get(self, url, **kwargs):
            return _factory(url)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    return _Client()


def _build_slots_response(slot_dicts, status_code=200):
    """Return a mock ``httpx.Response`` for a list of slot dicts."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = AsyncMock(return_value=slot_dicts)
    return resp


def _build_slots_payload(slot_dicts):
    """Return raw JSON bytes for a /slots response."""
    return json.dumps(slot_dicts).encode("utf-8")


# ---------------------------------------------------------------------------
# AC1 — Two concurrent local tasks fixture
# ---------------------------------------------------------------------------


class ConcurrentSlotsState:
    """State for two slots both actively prefilling.

    Each slot has its own ``progress`` counter that advances independently.
    """

    def __init__(self, progress_a=1000, progress_b=500, processing_a=True, processing_b=True):
        self.progress_a = progress_a
        self.progress_b = progress_b
        self.processing_a = processing_a
        self.processing_b = processing_b

    def to_slots_list(self):
        """Return slot list matching llama-server /slots shape."""
        return [
            {
                "id": 0,
                "prompt": "test prompt A",
                "n_past": self.progress_a,
                "is_processing": self.processing_a,
                "context_tokens": self.progress_a,
            },
            {
                "id": 1,
                "prompt": "test prompt B",
                "n_past": self.progress_b,
                "is_processing": self.processing_b,
                "context_tokens": self.progress_b,
            },
        ]

    def advance(self, delta_a=100, delta_b=50):
        """Advance both slots' progress (only if still processing)."""
        if self.processing_a:
            self.progress_a += delta_a
        if self.processing_b:
            self.progress_b += delta_b

    def stall_slot(self, slot_id):
        """Stop progress for a specific slot."""
        if slot_id == 0:
            self.processing_a = False
        else:
            self.processing_b = False


@contextmanager
def concurrent_slots_fixture(progress_a=1000, progress_b=500, processing_a=True, processing_b=True):
    """Context manager yielding a ConcurrentSlotsState for two concurrent prefilling slots.

    Usage:
        with concurrent_slots_fixture() as state:
            slots = state.to_slots_list()
            # ... use slots as /slots response
    """
    state = ConcurrentSlotsState(
        progress_a=progress_a,
        progress_b=progress_b,
        processing_a=processing_a,
        processing_b=processing_b,
    )
    yield state


# ---------------------------------------------------------------------------
# AC2 — Stalled task fixture
# ---------------------------------------------------------------------------


class StalledSlotState:
    """State for one slot that stops making progress for N seconds.

    The stalled slot's ``is_processing`` flag determines whether the
    liveness signal is still observable (True means the slot appears alive
    but isn't advancing numerically — this is the b8782 build scenario).
    """

    def __init__(self, slot_id=0, progress=5000, processing=True, stall_duration=30):
        self.slot_id = slot_id
        self.progress = progress
        self.processing = processing
        self.stall_duration = stall_duration  # seconds
        self.start_time = time.monotonic()

    def to_slots_list(self):
        """Return slot list for the stalled slot."""
        return [
            {
                "id": self.slot_id,
                "prompt": "test stalled prompt",
                "n_past": self.progress,
                "is_processing": self.processing,
                "context_tokens": None,  # b8782 doesn't report this
            },
        ]

    def elapsed(self):
        return time.monotonic() - self.start_time

    def is_stalled(self):
        return self.elapsed() < self.stall_duration


@contextmanager
def stalled_task_fixture(progress=5000, processing=True, stall_duration=30):
    """Context manager yielding a StalledSlotState.

    Usage:
        with stalled_task_fixture() as state:
            slots = state.to_slots_list()
            assert state.is_stalled()
    """
    state = StalledSlotState(
        slot_id=0,
        progress=progress,
        processing=processing,
        stall_duration=stall_duration,
    )
    yield state


# ---------------------------------------------------------------------------
# AC3 — Progressing task fixture
# ---------------------------------------------------------------------------


class ProgressingSlotState:
    """State for one slot whose progress advances at a configurable rate.

    Progress advances on each call to ``tick()`` by ``advance_per_tick``
    tokens, simulating the ``slot update_slots … n_tokens = N`` pattern.
    """

    def __init__(self, slot_id=0, progress=0, advance_per_tick=100, processing=True):
        self.slot_id = slot_id
        self.progress = progress
        self.advance_per_tick = advance_per_tick
        self.processing = processing
        self.tick_count = 0

    def to_slots_list(self):
        return [
            {
                "id": self.slot_id,
                "prompt": "test progressing prompt",
                "n_past": self.progress,
                "is_processing": self.processing,
                "context_tokens": self.progress,
            },
        ]

    def tick(self):
        """Advance progress and return updated slot list."""
        self.progress += self.advance_per_tick
        self.tick_count += 1
        return self.to_slots_list()

    @property
    def total_progress(self):
        return self.progress


@contextmanager
def progressing_task_fixture(progress=0, advance_per_tick=100, processing=True):
    """Context manager yielding a ProgressingSlotState.

    Usage:
        with progressing_task_fixture() as state:
            slots = state.tick()  # progress advances by advance_per_tick
    """
    state = ProgressingSlotState(
        slot_id=0,
        progress=progress,
        advance_per_tick=advance_per_tick,
        processing=processing,
    )
    yield state


# ---------------------------------------------------------------------------
# AC4 — llama-server stderr fixture (with and without progress lines)
# ---------------------------------------------------------------------------

STDERR_WITH_PROGRESS = [
    "slot update_slots: id 0 | task 18016 | prompt processing progress, n_tokens = 1000",
    "slot update_slots: id 0 | task 18016 | prompt processing progress, n_tokens = 2000",
    "slot update_slots: id 0 | task 18016 | prompt processing progress, n_tokens = 3500",
    "slot update_slots: id 0 | task 18016 | prompt processing progress, n_tokens = 5000",
]

STDERR_WITHOUT_PROGRESS = [
    "slot update_slots: id 0 | task 18016 | prompt processing complete",
    "slot update_slots: id 0 | task 18016 | prompt processing finished",
]

STDERR_MIXED = [
    "slot update_slots: id 0 | task 18016 | prompt processing progress, n_tokens = 1000",
    "slot update_slots: id 1 | task 18017 | prompt processing progress, n_tokens = 500",
    "slot update_slots: id 0 | task 18016 | prompt processing progress, n_tokens = 2000",
]


class StderrFixture:
    """Stubbed llama-server stderr lines.

    Provides lines that mimic the ``slot update_slots … n_tokens = N`` pattern
    used by the progress parser (when one is implemented).
    """

    def __init__(self, lines, with_progress=True):
        self.lines = lines
        self.with_progress = with_progress
        self.index = 0

    def next_line(self):
        if self.index < len(self.lines):
            line = self.lines[self.index]
            self.index += 1
            return line
        return None

    def remaining(self):
        return len(self.lines) - self.index

    def reset(self):
        self.index = 0


@contextmanager
def stderr_with_progress(lines=None):
    """Yield a StderrFixture with progress lines."""
    fixture = StderrFixture(lines or STDERR_WITH_PROGRESS, with_progress=True)
    yield fixture


@contextmanager
def stderr_without_progress(lines=None):
    """Yield a StderrFixture without progress lines."""
    fixture = StderrFixture(lines or STDERR_WITHOUT_PROGRESS, with_progress=False)
    yield fixture


@contextmanager
def stderr_mixed(lines=None):
    """Yield a StderrFixture with mixed content."""
    fixture = StderrFixture(lines or STDERR_MIXED, with_progress=True)
    yield fixture


# ---------------------------------------------------------------------------
# AC5 — /slots timeout fixture
# ---------------------------------------------------------------------------


class SlotsTimeoutError(Exception):
    """Raised when /slots times out (simulates query_timeout under load)."""
    pass


async def _timeout_slots(*args, **kwargs):
    """Simulate /slots timing out."""
    raise SlotsTimeoutError("/slots timed out during long prefill")


@contextmanager
def slots_timeout_fixture():
    """Context manager that patches /slots to raise a timeout.

    Usage:
        with slots_timeout_fixture():
            # /slots queries will raise SlotsTimeoutError
            pass
    """
    with patch("proxy.observability._query_slots", side_effect=_timeout_slots):
        yield


@asynccontextmanager
async def slots_timeout_fixture_async():
    """Async version of slots_timeout_fixture for use with async tests."""
    with patch("proxy.observability._query_slots", side_effect=_timeout_slots):
        yield


@contextmanager
def slots_timeout_fixture_detail():
    """Context manager that patches _query_slots_detail to raise a timeout."""
    async def _detail_timeout(*args, **kwargs):
        raise SlotsTimeoutError("/slots detail timed out")

    with patch("proxy.observability._query_slots_detail", side_effect=_detail_timeout):
        yield


# ---------------------------------------------------------------------------
# AC6 — Stubbed /slots payloads
# ---------------------------------------------------------------------------


def stubbed_slots_payload_with_progress(slot_id=0, progress=5000, processing=True):
    """Build a stubbed /slots payload that carries numeric progress.

    This asserts the **measured** lease-extension path: the progress value
    is present and numeric so ``_query_prefill_progress`` can return
    ``(progress, alive)`` instead of ``(None, alive)``.

    Includes fields from both older llama.cpp builds (``n_past``) and newer
    ones (``is_processing``).
    """
    return [
        {
            "id": slot_id,
            "prompt": "test prompt",
            "n_past": progress,
            "n_prompt_tokens_processed": progress,
            "is_processing": processing,
            "context_tokens": progress,
            "decoding": False,
            "has_extra": False,
        }
    ]


def stubbed_slots_payload_without_progress(slot_id=0, processing=False):
    """Build a stubbed /slots payload without numeric progress.

    This asserts the **fail-safe** path: ``n_past`` is absent, so
    ``_query_prefill_progress`` returns ``(None, processing)``. The
    liveness signal alone extends the lease (b8782 build scenario).
    """
    return [
        {
            "id": slot_id,
            "prompt": "test prompt",
            "is_processing": processing,
            # No n_past, no n_prompt_tokens_processed, no context_tokens
            # — simulating b8782 where these fields were removed
        }
    ]


def stubbed_slots_payload_multi_slot():
    """Build a /slots payload for multiple slots with mixed progress.

    Slot 0: progressing (n_past=3000, is_processing=True)
    Slot 1: idle (n_past=0, is_processing=False)
    """
    return [
        {
            "id": 0,
            "prompt": "test prompt A",
            "n_past": 3000,
            "is_processing": True,
            "context_tokens": 3000,
            "decoding": False,
        },
        {
            "id": 1,
            "prompt": "",
            "n_past": 0,
            "is_processing": False,
            "context_tokens": 0,
            "decoding": False,
        },
    ]


def stubbed_slots_payload_timeout():
    """Build a payload that simulates a /slots timeout response.

    Returns a response object that raises on json() instead of returning data.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    # Simulate a response that times out on reading
    async def _timeout_json():
        await asyncio.sleep(10)  # Will never complete in tests
        return []
    resp.json = _timeout_json
    return resp


# ---------------------------------------------------------------------------
# Fixture for patching _query_slots_progress in tests
# ---------------------------------------------------------------------------


def patch_query_slots_progress(result_dict, side_effect=None):
    """Return a context manager that patches _query_slots_progress.

    Args:
        result_dict: A dict mapping slot_id -> {"progress": int|None, "processing": bool}
                     that _query_slots_progress will return.
        side_effect: If set, raise this exception instead of returning result_dict.

    Usage:
        with patch_query_slots_progress({0: {"progress": 3000, "processing": True}}):
            # _query_slots_progress will return the dict above
            pass
    """
    async def _spy(*args, **kwargs):
        if side_effect is not None:
            raise side_effect
        return result_dict

    return patch("proxy.observability._query_slots_progress", side_effect=_spy)


# ---------------------------------------------------------------------------
# Fixture for patching query_llama_status
# ---------------------------------------------------------------------------


def patch_llama_status(result_dict, side_effect=None):
    """Return a context manager that patches query_llama_status.

    Args:
        result_dict: The dict query_llama_status will return.
        side_effect: If set, raise this exception instead.

    Usage:
        with patch_llama_status({"kv_cache_tokens": 5000, "n_past": None}):
            pass
    """
    async def _spy(*args, **kwargs):
        if side_effect is not None:
            raise side_effect
        return result_dict

    return patch("proxy.observability.query_llama_status", side_effect=_spy)


# ---------------------------------------------------------------------------
# Fixture for creating a minimal server object
# ---------------------------------------------------------------------------


def make_test_srv(records=None, config=None, logger=None):
    """Build a minimal server object for router_helpers tests.

    Mirrors the pattern in test_prefill_lease_extension.py.

    Args:
        records: Optional dict of dispatch records.
        config: Optional server config dict.
        logger: Optional logger mock.

    Returns:
        SimpleNamespace ready for router_helpers functions.
    """
    import proxy.server as server_mod

    default_config = {
        "server": {
            "llama_router_mode": False,
            "llama_server_port": 8080,
            "max_concurrent_queries": 4,
            "session_slot_pool_size": 2,
            "llama_request_timeout": 30,
            "local_dispatch_lease_timeout_seconds": 60,
            "local_dispatch_lease_per_token_seconds": 0.0,
            "local_dispatch_lease_max_seconds": 1500,
            "local_dispatch_lease_prefill_poll_seconds": 0.1,
            "local_dispatch_lease_prefill_buffer_seconds": 1.0,
            "local_dispatch_no_progress_timeout_seconds": 30,
            "local_dispatch_first_byte_timeout_seconds": 120,
            "session_single_flight_mode": "bypass",
            "disconnect_cleanup_timeout": 1,
        }
    }

    cfg = config if config is not None else default_config
    logger = logger or MagicMock()

    import asyncio

    return MagicMock(
        config=cfg,
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records=records or {},
        local_dispatch_records_lock=asyncio.Lock(),
        logger=logger,
        local_prefill_in_flight={},
    )


# ---------------------------------------------------------------------------
# pytest fixtures (auto-use when available)
# ---------------------------------------------------------------------------


@pytest.fixture
def test_srv():
    """Provide a minimal server object for tests."""
    return make_test_srv()


@pytest.fixture
def concurrent_slots():
    """Provide a ConcurrentSlotsState via context manager."""
    with concurrent_slots_fixture() as state:
        yield state


@pytest.fixture
def stalled_slot():
    """Provide a StalledSlotState via context manager."""
    with stalled_task_fixture() as state:
        yield state


@pytest.fixture
def progressing_slot():
    """Provide a ProgressingSlotState via context manager."""
    with progressing_task_fixture() as state:
        yield state


@pytest.fixture
def slots_with_progress():
    """Provide a /slots payload with numeric progress."""
    return stubbed_slots_payload_with_progress()


@pytest.fixture
def slots_without_progress():
    """Provide a /slots payload without numeric progress (b8782)."""
    return stubbed_slots_payload_without_progress()


@pytest.fixture
def multi_slot_payload():
    """Provide a multi-slot /slots payload."""
    return stubbed_slots_payload_multi_slot()


@pytest.fixture
def stderr_with_prog():
    """Provide stderr lines with progress."""
    with stderr_with_progress() as fixture:
        yield fixture


@pytest.fixture
def stderr_without_prog():
    """Provide stderr lines without progress."""
    with stderr_without_progress() as fixture:
        yield fixture
