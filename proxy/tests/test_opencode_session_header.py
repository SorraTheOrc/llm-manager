"""
Tests for x-opencode-session header synthesis on opencode.ai upstream hops.

Background (RCA LP-0MTR3CHEP007S699): Console (opencode.ai/zen) and Console
Go (opencode.ai/zen/go) upstreams return HTTP 400 ``MissingSessionID`` for
requests that lack the ``x-opencode-session`` header. When pi/opencode
clients talk DIRECTLY to opencode.ai they add the header client-side
(provider-attribution.js ``getSessionHeaders``); when they talk through the
llama-proxy (base URL = proxy host) they never add it, so the proxy must
synthesize it on the upstream hop from the inbound session id.

Covers:
1. opencode.ai upstream + inbound session id  -> x-opencode-session set
2. client-supplied x-opencode-session         -> not overridden
3. no inbound session id                      -> header absent
4. non-opencode.ai upstream                   -> header never synthesized
5. CR/LF in a client-supplied session value   -> header-injection guarded
"""

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import Request
from proxy.proxy_remote import proxy_to_remote


def _make_mock_client(captured_headers, mock_response=None):
    """Build a mock httpx.AsyncClient that records upstream request headers.

    Mirrors the non-streaming path in ``_handle_remote_non_streaming``
    (``client.post(url, headers=..., content=...)``); the recorded headers
    are appended to *captured_headers* so tests can assert on the wire.
    """
    if mock_response is None:
        mock_response = MagicMock()
        type(mock_response).status_code = PropertyMock(return_value=200)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = (
            b'{"id":"test","choices":[{"finish_reason":"stop","index":0,'
            b'"message":{"role":"assistant","content":"Hello!"}}]}'
        )

    def _capture_post(url, headers=None, content=None, **kwargs):
        captured_headers.append(dict(headers or {}))
        return mock_response

    client_instance = MagicMock()
    client_instance.post = AsyncMock(side_effect=_capture_post)

    mock_client_cls = MagicMock(return_value=client_instance)
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client_instance)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_client_cls


@pytest.fixture
def mock_request():
    """Create a mock Request whose headers can be set per-test."""
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.url.path = "/v1/chat/completions"
    req.is_disconnected = AsyncMock(return_value=False)
    req.body = AsyncMock(
        return_value=b'{"model":"muse-spark-1.2-contributor",'
        b'"messages":[{"role":"user","content":"hi"}]}'
    )
    req.headers = {}
    return req


async def _run_proxy_to_remote(mock_request, captured_headers, provider_cfg):
    """Call proxy_to_remote with the standard mock patches."""
    mock_client_cls = _make_mock_client(captured_headers)
    with patch("proxy.proxy_remote.httpx.AsyncClient", mock_client_cls):
        with patch("proxy.proxy_remote._schedule_recv_token_increment", AsyncMock()):
            with patch("proxy.proxy_remote.log_response"):
                with patch("proxy.proxy_remote.log_request"):
                    return await proxy_to_remote(
                        request=mock_request,
                        path="v1/chat/completions",
                        model_config=provider_cfg,
                    )


_CONSOLE_GO_CFG = {
    "name": "opencode-go",
    "type": "remote",
    "provider": "opencode-go",
    "endpoint": "https://opencode.ai/zen/go",
    "model": "muse-spark-1.2-contributor",
    "forward_session_headers": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# AC1: opencode.ai upstream + inbound session id -> x-opencode-session set
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_opencode_upstream_synthesizes_session_header(mock_request):
    """x-opencode-session is set from the inbound x-client-request-id value."""
    captured = []
    mock_request.headers = {"x-client-request-id": "herdr-1788760948-3818499-9937"}

    await _run_proxy_to_remote(mock_request, captured, _CONSOLE_GO_CFG)

    assert len(captured) == 1
    upstream_headers = {k.lower(): v for k, v in captured[0].items()}
    assert upstream_headers.get("x-opencode-session") == "herdr-1788760948-3818499-9937", \
        f"Expected synthesized x-opencode-session, got {upstream_headers.get('x-opencode-session')}"


@pytest.mark.asyncio
async def test_opencode_upstream_synthesizes_from_x_session_affinity(mock_request):
    """x-opencode-session falls back to x-session-affinity when x-session-id
    and x-client-request-id are absent."""
    captured = []
    mock_request.headers = {"x-session-affinity": "herdr-1788760948-3818499-9937"}

    await _run_proxy_to_remote(mock_request, captured, _CONSOLE_GO_CFG)

    upstream_headers = {k.lower(): v for k, v in captured[0].items()}
    assert upstream_headers.get("x-opencode-session") == "herdr-1788760948-3818499-9937"


@pytest.mark.asyncio
async def test_opencode_upstream_synthesizes_from_x_session_id(mock_request):
    """x-opencode-session falls back to x-session-id when present."""
    captured = []
    mock_request.headers = {"x-session-id": "abc-123-session"}

    await _run_proxy_to_remote(mock_request, captured, _CONSOLE_GO_CFG)

    upstream_headers = {k.lower(): v for k, v in captured[0].items()}
    assert upstream_headers.get("x-opencode-session") == "abc-123-session"


# ═══════════════════════════════════════════════════════════════════════════════
# AC2: client-supplied x-opencode-session is never overridden
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_client_supplied_opencode_session_not_overridden(mock_request):
    """A client-supplied x-opencode-session is preserved even when the
    request also carries proxy session headers."""
    captured = []
    mock_request.headers = {
        "x-opencode-session": "client-provided-session-42",
        "x-client-request-id": "herdr-other-session",
    }

    await _run_proxy_to_remote(mock_request, captured, _CONSOLE_GO_CFG)

    upstream_headers = {k.lower(): v for k, v in captured[0].items()}
    assert upstream_headers.get("x-opencode-session") == "client-provided-session-42", \
        "Proxy must not override a client-supplied x-opencode-session"


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: no inbound session id -> no synthesized header
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_no_inbound_session_no_synthesized_header(mock_request):
    """No session id on the inbound request -> no x-opencode-session header."""
    captured = []
    mock_request.headers = {}

    await _run_proxy_to_remote(mock_request, captured, _CONSOLE_GO_CFG)

    upstream_headers = {k.lower(): v for k, v in captured[0].items()}
    assert "x-opencode-session" not in upstream_headers, \
        "x-opencode-session must not be synthesized without an inbound session id"


# ═══════════════════════════════════════════════════════════════════════════════
# AC4: non-opencode.ai upstream never synthesizes the header
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_non_opencode_upstream_never_synthesizes(mock_request):
    """DeepSeek / OpenRouter upstreams are unaffected by the synthesis."""
    captured = []
    mock_request.headers = {"x-client-request-id": "herdr-1788760948-3818499-9937"}
    deepseek_cfg = {
        "name": "deepseek",
        "type": "remote",
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    }

    await _run_proxy_to_remote(mock_request, captured, deepseek_cfg)

    upstream_headers = {k.lower(): v for k, v in captured[0].items()}
    assert "x-opencode-session" not in upstream_headers, \
        "Non-opencode.ai upstreams must never receive a synthesized x-opencode-session"


@pytest.mark.asyncio
async def test_lookalike_opencode_host_never_synthesizes(mock_request):
    """A lookalike host (e.g. opencode.ai.evil.example) is not treated as
    an opencode.ai upstream: the host check must be exact, not substring."""
    captured = []
    mock_request.headers = {"x-client-request-id": "herdr-1788760948-3818499-9937"}
    lookalike_cfg = {
        "name": "opencode-lookalike",
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai.evil.example/zen/go",
        "model": "muse-spark-1.2-contributor",
    }

    await _run_proxy_to_remote(mock_request, captured, lookalike_cfg)

    upstream_headers = {k.lower(): v for k, v in captured[0].items()}
    assert "x-opencode-session" not in upstream_headers, \
        "Lookalike hosts must never receive a synthesized x-opencode-session"


# ═══════════════════════════════════════════════════════════════════════════════
# AC5: header-injection guard on CR/LF in the session value
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_session_value_crlf_injection_guarded(mock_request):
    """CR/LF in a client-supplied session value must not reach the wire."""
    captured = []
    mock_request.headers = {
        "x-client-request-id": "evil-session\r\nX-Injected: boom",
    }

    await _run_proxy_to_remote(mock_request, captured, _CONSOLE_GO_CFG)

    upstream_headers = captured[0]
    x_opencode_session = upstream_headers.get("x-opencode-session", "")
    assert "\r" not in x_opencode_session and "\n" not in x_opencode_session, \
        f"CR/LF must be stripped from x-opencode-session, got {x_opencode_session!r}"
    # A second (injected) header must not appear either.
    injected = [k for k in upstream_headers if "injected" in k.lower()]
    assert not injected, f"Header injection must be prevented, got {injected}"
