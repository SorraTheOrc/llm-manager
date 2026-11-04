"""Tests for status_request structured logging (client IP + response payload).

Covers the client-IP resolution helper and the KeyValueFormatter that
renders structured ``extra`` fields into the plain-text proxy.log
(LP-0MSK9XXCN0077CMA).
"""
import asyncio
import contextlib
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytestmark = pytest.mark.refactor_parity


# ---------------------------------------------------------------------------
# _resolve_client_ip unit tests
# ---------------------------------------------------------------------------


def _make_request(headers=None, client_host="127.0.0.1", has_client=True, client_port=None):
    """Build a minimal mock Request for calling client-identity resolvers."""
    req = MagicMock()
    req.headers = headers or {}
    if has_client:
        req.client = MagicMock()
        req.client.host = client_host
        req.client.port = client_port
    else:
        req.client = None
    return req


def test_resolve_client_ip_direct():
    """A direct connection resolves to request.client.host with source 'direct'."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={}, client_host="192.168.0.191")
    assert _resolve_client_ip(req) == ("192.168.0.191", "direct")


def test_resolve_client_ip_x_forwarded_for():
    """X-Forwarded-For is honored with source 'header'."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={"x-forwarded-for": "203.0.113.7"})
    assert _resolve_client_ip(req) == ("203.0.113.7", "header")


def test_resolve_client_ip_x_forwarded_for_chain_takes_first():
    """A proxy chain 'client, proxy1' resolves to the first entry."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={"x-forwarded-for": "203.0.113.7, 10.0.0.1"})
    assert _resolve_client_ip(req) == ("203.0.113.7", "header")


def test_resolve_client_ip_x_real_ip():
    """X-Real-IP is honored with source 'header'."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={"x-real-ip": "198.51.100.9"})
    assert _resolve_client_ip(req) == ("198.51.100.9", "header")


def test_resolve_client_ip_prefers_xff_over_x_real_ip():
    """X-Forwarded-For takes precedence when both headers are present."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(
        headers={"x-forwarded-for": "203.0.113.7", "x-real-ip": "198.51.100.9"}
    )
    assert _resolve_client_ip(req) == ("203.0.113.7", "header")


def test_resolve_client_ip_unknown_when_no_client():
    """Without a client address or headers, resolves to ('unknown', 'direct')."""
    from proxy.handlers import _resolve_client_ip

    req = _make_request(headers={}, has_client=False)
    assert _resolve_client_ip(req) == ("unknown", "direct")


def test_resolve_client_port_direct():
    """A direct connection resolves request.client.port; IP source stays 'direct'."""
    from proxy.handlers import _resolve_client_ip, _resolve_client_port

    req = _make_request(headers={}, client_host="192.168.0.191", client_port=51842)
    assert _resolve_client_port(req) == 51842
    assert _resolve_client_ip(req) == ("192.168.0.191", "direct")


@pytest.mark.parametrize(
    "headers",
    [
        {"x-forwarded-for": "203.0.113.7"},
        {"x-real-ip": "198.51.100.9"},
    ],
)
def test_resolve_client_port_unknown_for_header_paths(headers):
    """Header-based (reverse-proxy) paths carry no source port -> 'unknown'.

    Even when a direct connection exists, the header is authoritative for the
    client identity and no port travels with it.
    """
    from proxy.handlers import _resolve_client_port

    req = _make_request(headers=headers, client_host="10.0.0.5", client_port=51842)
    assert _resolve_client_port(req) == "unknown"


def test_resolve_client_port_unknown_when_no_client():
    """Without a client address or headers, the port resolves to 'unknown'."""
    from proxy.handlers import _resolve_client_port

    req = _make_request(headers={}, has_client=False)
    assert _resolve_client_port(req) == "unknown"


def test_resolve_client_id_precedence_x_session_id():
    """x-session-id wins over the other session headers."""
    from proxy.handlers import _resolve_client_id

    req = _make_request(
        headers={
            "x-session-id": "sess-abc",
            "session_id": "sess-def",
            "x-client-request-id": "req-123",
        }
    )
    assert _resolve_client_id(req) == "sess-abc"


def test_resolve_client_id_precedence_session_id():
    """session_id wins when x-session-id is absent."""
    from proxy.handlers import _resolve_client_id

    req = _make_request(
        headers={"session_id": "sess-def", "x-client-request-id": "req-123"}
    )
    assert _resolve_client_id(req) == "sess-def"


def test_resolve_client_id_precedence_x_client_request_id():
    """x-client-request-id is used when neither session header is present."""
    from proxy.handlers import _resolve_client_id

    req = _make_request(headers={"x-client-request-id": "req-123"})
    assert _resolve_client_id(req) == "req-123"


def test_resolve_client_id_none_when_no_header():
    """No session header -> None so the field can be omitted (not 'unknown')."""
    from proxy.handlers import _resolve_client_id

    req = _make_request(headers={})
    assert _resolve_client_id(req) is None


# ---------------------------------------------------------------------------
# Endpoint logging tests (via ASGI transport)
# ---------------------------------------------------------------------------


def _status_records(caplog):
    return [r for r in caplog.records if "status_request" in r.getMessage()]


@pytest.mark.asyncio
async def test_status_request_logs_local_active_query(caplog):
    """The status_request log line carries local_active_query (LP-0MSL2ZLLS009RVKR)."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    assert hasattr(records[-1], "local_active_query")
    assert records[-1].local_active_query is False


@pytest.mark.asyncio
async def test_status_request_logs_client_ip_direct(caplog):
    """Direct pollers are attributable: client_ip comes from request.client."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    assert records[-1].client_ip == "127.0.0.1"
    assert records[-1].client_ip_source == "direct"


@pytest.mark.asyncio
async def test_status_request_logs_client_ip_from_xff_header(caplog):
    """Reverse-proxy pollers are attributable via X-Forwarded-For."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get(
                "/llama/local/status",
                headers={"X-Forwarded-For": "203.0.113.7"},
            )

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    assert records[-1].client_ip == "203.0.113.7"
    assert records[-1].client_ip_source == "header"


@pytest.mark.asyncio
async def test_status_request_logs_all_response_fields(caplog):
    """The log record carries every response field plus client_ip."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    rec = records[-1]
    for field in (
        "client_ip",
        "client_ip_source",
        "client_port",
        "latency_ms",
        "llama_server_running",
        "active_query",
        "local_active_query",
        "model_switch_in_progress",
        "current_model",
        "available_slots",
        "total_slots",
        "local_owner_session_id",
        "local_owner_lease_remaining_seconds",
    ):
        assert hasattr(rec, field), f"Missing log field {field}"

    assert rec.llama_server_running is True
    assert isinstance(rec.latency_ms, int)
    assert isinstance(rec.available_slots, int)
    assert isinstance(rec.total_slots, int)


@pytest.mark.asyncio
async def test_status_request_logs_client_port_direct(caplog):
    """Direct pollers are attributable by source port: client_port is logged.

    client_ip / client_ip_source keep their current semantics (additive
    change only, LP-0MSKV3IEQ004ZV88).
    """
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    rec = records[-1]
    assert rec.client_port == 123  # httpx ASGITransport default client port
    assert rec.client_ip == "127.0.0.1"
    assert rec.client_ip_source == "direct"


@pytest.mark.asyncio
async def test_status_request_logs_client_id_when_present(caplog):
    """status_request carries client_id when a session header is present."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get(
                "/llama/local/status",
                headers={"X-Session-Id": "sess-poll-42"},
            )

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    assert records[-1].client_id == "sess-poll-42"


@pytest.mark.asyncio
async def test_status_request_omits_client_id_when_absent(caplog):
    """status_request omits client_id entirely when no session header is sent."""
    from proxy.server import app

    from proxy import server as srv_module

    caplog.set_level(logging.INFO, logger="llama-proxy")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        with patch.object(
            srv_module, "query_llama_status", new_callable=AsyncMock
        ) as mock_qls:
            mock_qls.return_value = {"llama_server_running": True}
            resp = await ac.get("/llama/local/status")

    assert resp.status_code == 200
    records = _status_records(caplog)
    assert records, "Expected status_request log record"
    assert not hasattr(records[-1], "client_id"), (
        "client_id must be omitted (absent), not rendered as None"
    )


# ---------------------------------------------------------------------------
# KeyValueFormatter tests
# ---------------------------------------------------------------------------


def _make_status_record():
    """Build a LogRecord mimicking the status_request extra payload."""
    record = logging.LogRecord(
        name="llama-proxy",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="status_request",
        args=(),
        exc_info=None,
    )
    record.client_ip = "192.168.0.191"
    record.client_ip_source = "direct"
    record.client_port = 51842
    record.latency_ms = 12
    record.llama_server_running = True
    record.active_query = False
    record.local_active_query = False
    record.model_switch_in_progress = False
    record.current_model = "Qwen3"
    record.available_slots = 3
    record.total_slots = 3
    record.local_owner_session_id = None
    record.local_owner_lease_remaining_seconds = None
    return record


def test_key_value_formatter_renders_status_request_payload():
    """Plain-text log line includes every extra field as key=value."""
    from proxy.utils import KeyValueFormatter

    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(_make_status_record())

    assert "status_request" in text
    assert "client_ip=192.168.0.191" in text
    assert "client_ip_source=direct" in text
    assert "client_port=51842" in text
    assert "latency_ms=12" in text
    assert "llama_server_running=true" in text
    assert "active_query=false" in text
    assert "local_active_query=false" in text
    assert "model_switch_in_progress=false" in text
    assert "current_model=Qwen3" in text
    assert "available_slots=3" in text
    assert "total_slots=3" in text
    assert "local_owner_session_id=None" in text
    assert "local_owner_lease_remaining_seconds=None" in text


def test_key_value_formatter_renders_client_id():
    """client_id renders as key=value when present on the record."""
    from proxy.utils import KeyValueFormatter

    record = _make_status_record()
    record.client_id = "sess-poll-42"
    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(record)
    assert "client_id=sess-poll-42" in text


def test_key_value_formatter_renders_client_port_unknown():
    """client_port=unknown renders as key=value for header-based paths."""
    from proxy.utils import KeyValueFormatter

    record = _make_status_record()
    record.client_port = "unknown"
    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(record)
    assert "client_port=unknown" in text


def test_key_value_formatter_skips_reserved_attributes():
    """LogRecord machinery attributes are not rendered as key=value pairs."""
    from proxy.utils import KeyValueFormatter

    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(_make_status_record())

    for reserved in (
        "name=",
        "levelname=",
        "levelno=",
        "pathname=",
        "lineno=",
        "module=",
        "funcName=",
        "created=",
        "thread=",
        "process=",
        "message=",
        "msg=",
        "asctime=",
        "taskName=",
    ):
        assert reserved not in text, f"Reserved attribute rendered: {reserved}"


def test_key_value_formatter_plain_record_unchanged():
    """Records without extras render exactly as before (message only)."""
    from proxy.utils import KeyValueFormatter

    record = logging.LogRecord(
        name="llama-proxy",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Stream started: session-1",
        args=(),
        exc_info=None,
    )
    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(record)
    assert "Stream started: session-1" in text
    assert text.rstrip().endswith("Stream started: session-1")


def test_key_value_formatter_handles_non_scalar_values():
    """Non-scalar extra values are JSON-encoded so they stay readable."""
    from proxy.utils import KeyValueFormatter

    record = _make_status_record()
    record.current_model = {"name": "Qwen3", "tokens": 4096}
    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    text = formatter.format(record)
    assert 'current_model={"name": "Qwen3", "tokens": 4096}' in text


# ---------------------------------------------------------------------------
# Log-compatibility for the per-phase timing surface (LP-0MVVXGHHB0073W1B)
# ---------------------------------------------------------------------------


def _load_capacity_signals():
    """Load the shared status-poll parser (scripts/lib/capacity_signals.py)."""
    import importlib.util
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "lib"
        / "capacity_signals.py"
    )
    spec = importlib.util.spec_from_file_location(
        "capacity_signals_lib", script
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _render_status_line(extra_fields: dict) -> str:
    """Render a status_request log line from the standard extra payload plus
    *extra_fields*, exactly as the KeyValueFormatter would."""
    from proxy.utils import KeyValueFormatter

    record = _make_status_record()
    record.slots_stale = True
    record.slots_detail_stale = True
    record.slot_counts_stale = False
    record.capacity_hint = "idle"
    record.status_schema_version = 5
    record.local_in_flight_requests = 0
    record.slots = []
    for key, value in extra_fields.items():
        setattr(record, key, value)
    formatter = KeyValueFormatter("%(asctime)s - %(levelname)s - %(message)s")
    return formatter.format(record)


def test_capacity_parser_keeps_parsing_unchanged_lines_with_additive_fields():
    """Matrix 6: the per-phase timing surface is ADDITIVE — the shared
    capacity parser (``scripts/capacity_signal_verification.py``'s field regex)
    keeps parsing unchanged ``status_request`` lines and simply ignores the new
    timing fields, so pre-change windows stay measurable (F1 AC4).
    """
    signals = _load_capacity_signals()

    # The pre-change line, plus the additive per-phase timing surface the
    # instrumenting child (LP-0MVVXGHT1001I5YX) adds: unknown-to-the-parser
    # fields must not disturb the fields it extracts.
    line = _render_status_line(
        {
            "status_query_ms": 3,
            "slots_counts_ms": 4,
            "slots_detail_ms": 5,
            "slots_upstream_calls": 1,
            "slots_snapshot_age_seconds": 0.42,
            "slots_snapshot_source": "upstream",
        }
    )

    assert signals.is_status_poll(line)
    fields = signals.status_fields(line)
    # The unchanged pre-change fields keep their names and values (the
    # parser's fixed field set deliberately reads available_slots, not
    # total_slots).
    assert fields["slots_stale"] == "true"
    assert fields["slots_detail_stale"] == "true"
    assert fields["capacity_hint"] == "idle"
    assert fields["available_slots"] == "3"
    assert fields["status_schema_version"] == "5"
    assert fields["latency_ms"] == "12"
    assert signals.stale_flag(fields) is True
    assert signals.parse_latency_ms(fields["latency_ms"]) == 12
    # The additive timing fields render on the line but are unknown to the
    # parser's fixed field set — they neither appear in the extraction nor
    # break it.
    assert "slots_upstream_calls=1" in line
    assert "slots_snapshot_source" not in fields
    assert "slots_upstream_calls" not in fields

# ---------------------------------------------------------------------------
# Per-phase /slots read-path timings (LP-0MVVXGHT1001I5YX)
#
# The ``status_request`` line used to carry only an aggregate ``latency_ms``,
# so a stale (> 2 s) poll could not be attributed to the status query, the
# counts query, the detail query or a cache path. These tests pin the
# additive per-phase timing + provenance surface (AC1/AC2/AC3).
# ---------------------------------------------------------------------------

#: The six additive fields the instrumentation publishes on every
#: ``status_request`` line. The work-item text says "seven" but enumerates
#: exactly six named fields; the named list is authoritative, so AC1 asserts
#: every one of these.
PER_PHASE_FIELDS = (
    "status_query_ms",
    "slots_counts_ms",
    "slots_detail_ms",
    "slots_upstream_calls",
    "slots_snapshot_age_seconds",
    "slots_snapshot_source",
)


class _SlotsBackend:
    """Raw ``/slots`` fetch stub that records calls and answers from a script.

    *payloads* is consumed one entry per upstream call and the last entry
    repeats. Each entry is ``(status_code, data)`` or an exception to raise.

    The stub replaces :func:`proxy.observability._do_slots_fetch`, whose
    contract is ``(status_code, parsed_data)`` — a tuple, not a response
    object — so ``__call__`` returns that tuple (same shape as
    ``_FanoutBackend`` in test_slots_poll_coalescing.py).
    """

    def __init__(self, payloads, delay: float = 0.0):
        self.payloads = list(payloads)
        self.delay = delay
        self.calls: list[str] = []

    async def __call__(self, url: str, timeout: float, client=None):
        self.calls.append(url)
        if self.delay:
            await asyncio.sleep(self.delay)
        payload = (
            self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        )
        if isinstance(payload, BaseException):
            raise payload
        status, data = payload
        return status, data


class _UnlockedLock:
    """Stand-in for the server's asyncio locks (never actually contended)."""

    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@contextlib.asynccontextmanager
async def _status_poll(
    *,
    backend=None,
    llama_running: bool = True,
    current_model: str | None = "test-model",
    status_side_effect=None,
    warm_cache: dict | None = None,
    warm_counts: tuple | None = None,
    query_timeout: float = 1.0,
):
    """GET ``/llama/local/status`` with the production slots read path.

    Only the raw upstream ``/slots`` fetch (*backend*) and the status query are
    stubbed, so the phase stamps, provenance plumbing and cache decisions are
    the real ones. Every fallback cache is pinned empty first (a sibling test's
    leftovers must not decide the verdict); *warm_cache* then seeds the
    coalesced ``/slots`` cache to simulate a preceding poll, and *warm_counts*
    seeds the last-known counts cache that the graceful-degradation path serves.

    Yields ``(payload, log_record, backend)``.
    """
    from proxy.server import app

    from proxy import observability as _obs
    from proxy import server

    if status_side_effect is None:

        async def fake_query():
            return {"llama_server_running": llama_running}

        status_patch = patch(
            "proxy.server.query_llama_status", side_effect=fake_query
        )
    else:
        status_patch = patch(
            "proxy.server.query_llama_status", side_effect=status_side_effect
        )

    with (
        patch.object(_obs, "_last_slot_details_cache", []),
        patch.object(_obs, "_last_slot_details_at", 0.0),
        patch.object(_obs, "_last_slot_counts_cache", warm_counts),
        patch.object(_obs, "_slots_poll_cache", dict(warm_cache or {})),
        patch.dict(os.environ, {"STATUS_QUERY_TIMEOUT": str(query_timeout)}),
        status_patch,
        # One ``patch.multiple`` instead of one manager per attribute: the
        # original 19-manager ``with`` hit CPython's 20-block nesting limit.
        patch.multiple(
            server,
            active_queries=0,
            active_queries_lock=_UnlockedLock(),
            local_active_queries=0,
            local_active_queries_lock=_UnlockedLock(),
            model_switch_refcount=0,
            model_switch_lock=_UnlockedLock(),
            background_loads={},
            local_dispatch_records={},
            local_dispatch_records_lock=_UnlockedLock(),
            current_model=current_model,
            _http_client=MagicMock(),
            config={
                "server": {
                    "llama_server_port": 8080,
                    "session_slot_pool_size": 3,
                }
            },
        ),
        patch.object(_obs, "_do_slots_fetch", backend),
    ):
        records: list[logging.LogRecord] = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Collect()
        handler.setLevel(logging.INFO)
        log = logging.getLogger("llama-proxy")
        _prev_level = log.level
        # The package logger's effective level is above INFO in the test
        # runtime, so a handler alone would never see the record — raise it
        # for the poll and restore it afterwards.
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                resp = await ac.get("/llama/local/status")
        finally:
            log.removeHandler(handler)
            log.setLevel(_prev_level)

    assert resp.status_code == 200
    status_records = [r for r in records if r.getMessage() == "status_request"]
    assert status_records, "Expected a status_request log record"
    yield resp.json(), status_records[-1], backend


@pytest.mark.asyncio
async def test_status_request_carries_every_per_phase_field_on_success():
    """AC1: the success path publishes all six fields, never absent."""
    backend = _SlotsBackend([(200, [{"id": 0, "is_processing": False}])])

    async with _status_poll(backend=backend) as (_payload, rec, _b):
        for field in PER_PHASE_FIELDS:
            assert hasattr(rec, field), f"Missing per-phase log field {field}"
        # Phase timings are whole milliseconds; the provenance fields are a
        # count, a float-or-None age and a vocabulary token.
        assert isinstance(rec.status_query_ms, int)
        assert isinstance(rec.slots_counts_ms, int)
        assert isinstance(rec.slots_detail_ms, int)
        assert isinstance(rec.slots_upstream_calls, int)
        assert rec.slots_upstream_calls == 1
        assert isinstance(rec.slots_snapshot_age_seconds, float)
        assert rec.slots_snapshot_source == "upstream"


@pytest.mark.asyncio
async def test_status_request_carries_every_per_phase_field_when_status_times_out():
    """AC1 (timeout path): a hung status query still publishes all six fields.

    The status phase is still measured; the slots phases are never attempted,
    so their timings and the snapshot age are ``null`` and the source is
    ``none`` — never an absent field.
    """

    async def hanging_query():
        await asyncio.sleep(3600.0)

    async with _status_poll(
        status_side_effect=hanging_query, query_timeout=0.05
    ) as (payload, rec, _b):
        assert payload["llama_server_running"] is False
        for field in PER_PHASE_FIELDS:
            assert hasattr(rec, field), f"Missing per-phase log field {field}"
        assert isinstance(rec.status_query_ms, int)
        assert rec.slots_counts_ms is None
        assert rec.slots_detail_ms is None
        assert rec.slots_upstream_calls == 0
        assert rec.slots_snapshot_age_seconds is None
        assert rec.slots_snapshot_source == "none"


@pytest.mark.asyncio
async def test_status_request_carries_every_per_phase_field_when_backend_down():
    """AC1 (backend-down path): llama-server down publishes all six fields.

    No ``/slots`` attempt is made, so the counts/detail timings and the
    snapshot age are ``null``; every field is still present.
    """
    async with _status_poll(llama_running=False) as (_payload, rec, _backend):
        for field in PER_PHASE_FIELDS:
            assert hasattr(rec, field), f"Missing per-phase log field {field}"
        assert isinstance(rec.status_query_ms, int)
        assert rec.slots_counts_ms is None
        assert rec.slots_detail_ms is None
        assert rec.slots_upstream_calls == 0
        assert rec.slots_snapshot_age_seconds is None
        assert rec.slots_snapshot_source == "none"


@pytest.mark.asyncio
async def test_slots_snapshot_source_labels_an_upstream_fetch():
    """AC2: a poll that caused a real upstream ``/slots`` fetch says so."""
    backend = _SlotsBackend([(200, [{"id": 0, "is_processing": False}])])

    async with _status_poll(backend=backend) as (_payload, rec, b):
        assert b.calls, "fixture must exercise the upstream fetch"
        assert rec.slots_snapshot_source == "upstream"
        assert rec.slots_upstream_calls == 1
        # The payload was fetched during this poll, so its age is ~0.
        assert rec.slots_snapshot_age_seconds == pytest.approx(0.0, abs=0.5)


@pytest.mark.asyncio
async def test_slots_snapshot_source_labels_a_ttl_cache_hit():
    """AC2: a poll served from the coalesced TTL cache says ``ttl_cache``."""
    import time

    url = (
        "http://localhost:8080/slots?model=test-model"
    )
    # A warm 200 entry, exactly as a preceding poll inside the TTL leaves it.
    warm = {url: (time.monotonic() - 0.25, 200, [{"id": 0, "is_processing": False}])}
    backend = _SlotsBackend([(200, [{"id": 0, "is_processing": False}])])

    async with _status_poll(backend=backend, warm_cache=warm) as (_payload, rec, b):
        assert rec.slots_snapshot_source == "ttl_cache", (
            "a warm TTL entry must be labelled ttl_cache, got "
            f"{rec.slots_snapshot_source!r}"
        )
        assert rec.slots_upstream_calls == 0, "a TTL hit causes no upstream fetch"
        assert rec.slots_snapshot_age_seconds == pytest.approx(0.25, abs=0.5)


@pytest.mark.asyncio
async def test_slots_snapshot_source_labels_a_failure_backoff_hit():
    """AC2: a poll served from a cached failure says ``failure_backoff``."""
    import time

    url = "http://localhost:8080/slots?model=test-model"
    # A cached non-200 entry: the extended backoff window holds it well past
    # the plain TTL, so a 2 s old failure is still served.
    warm = {url: (time.monotonic() - 2.0, 500, None)}
    backend = _SlotsBackend([(200, [{"id": 0, "is_processing": False}])])

    async with _status_poll(backend=backend, warm_cache=warm) as (_payload, rec, b):
        assert rec.slots_snapshot_source == "failure_backoff", (
            "a cached failed fetch must be labelled failure_backoff, got "
            f"{rec.slots_snapshot_source!r}"
        )
        assert rec.slots_upstream_calls == 0
        assert rec.slots_snapshot_age_seconds == pytest.approx(2.0, abs=0.5)


@pytest.mark.asyncio
async def test_slots_snapshot_source_labels_a_last_known_snapshot():
    """The graceful-degradation path is labelled ``snapshot``.

    A transient counts failure with a fresh last-known counts cache serves the
    snapshot instead of 0/0; the provenance field names that path and the age
    of the snapshot it served.
    """
    import time

    backend = _SlotsBackend([(500, None)])

    async with _status_poll(
        backend=backend, warm_counts=(2, 3, time.monotonic() - 4.0)
    ) as (payload, rec, _b):
        # Since LP-0MW1VDZIX0011D8M a stale snapshot asserts no availability:
        # the summary publishes 0 + an explicit unknown flag rather than the
        # cached ``2`` that contradicted the detail.
        assert payload["available_slots"] == 0
        assert payload["availability_unknown"] is True
        assert payload["total_slots"] == 3
        assert rec.slots_snapshot_source == "snapshot"
        assert rec.slots_snapshot_age_seconds == pytest.approx(4.0, abs=1.0)


@pytest.mark.asyncio
async def test_phase_timings_reconcile_with_the_aggregate_latency():
    """AC3: status_query_ms + slots_counts_ms + slots_detail_ms <= latency_ms.

    Each phase is stamped around the existing awaits, so their sum can never
    exceed the aggregate the line already published.
    """
    backend = _SlotsBackend(
        [(200, [{"id": 0, "is_processing": False}])], delay=0.02
    )

    async with _status_poll(backend=backend) as (_payload, rec, _b):
        assert rec.slots_counts_ms is not None
        assert rec.slots_detail_ms is not None
        total = (
            rec.status_query_ms + rec.slots_counts_ms + rec.slots_detail_ms
        )
        assert total <= rec.latency_ms, (
            f"phase timings ({rec.status_query_ms} + {rec.slots_counts_ms} + "
            f"{rec.slots_detail_ms} = {total} ms) exceed the aggregate "
            f"latency_ms ({rec.latency_ms} ms)"
        )
        # The fixture's upstream delay is inside the counts phase, so that
        # phase must be measurably non-zero (a stamp that never advanced would
        # silently satisfy the inequality above).
        assert rec.slots_counts_ms >= 20
        assert rec.latency_ms >= 20


@pytest.mark.asyncio
async def test_phase_timings_reconcile_on_the_failure_path():
    """AC3: the reconciliation holds when the slots queries fail too."""
    backend = _SlotsBackend([(500, None)], delay=0.02)

    async with _status_poll(backend=backend) as (_payload, rec, _b):
        assert rec.slots_counts_ms is not None
        total = rec.status_query_ms + rec.slots_counts_ms
        assert total <= rec.latency_ms
