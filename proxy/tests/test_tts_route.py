"""
Contract and mock-based route handler tests for the /v1/audio/speech endpoint.

These tests define the expected API contract (OpenAI-compatible TTS request
format) and verify that the proxy route:
  - Forwards valid requests to the tts-server backend
  - Returns audio/wav content-type on success
  - Returns 400 for invalid/missing parameters
  - Returns 502 when the tts-server is unreachable

All tests run within the existing pytest-asyncio framework without requiring
a running tts-server. A mock httpx client is used to simulate the tts-server
backend.

Structured error format (all 502 responses follow the same pattern):

.. code-block:: json

    {
        "error": {
            "type": "tts_error",
            "code": "tts_server_unreachable",
            "message": "User-friendly description with actionable guidance"
        },
        "status": 502,
        "path": "/v1/audio/speech"
    }
"""

from unittest.mock import AsyncMock

import httpx
import pytest
import struct

pytestmark = pytest.mark.refactor_parity

# ---------------------------------------------------------------------------
# WAV binary stub — a minimal valid WAV header so tests verify actual audio
# data, not just empty bytes.
# ---------------------------------------------------------------------------
_SAMPLE_RATE = 24000
_NUM_SAMPLES = int(_SAMPLE_RATE * 0.1)  # ~2400 samples
_DATA_SIZE = _NUM_SAMPLES * 2  # 16-bit = 2 bytes per sample
_RIFF_SIZE = 36 + _DATA_SIZE

FAKE_WAV_BYTES = struct.pack(
    "<4sI4s4sIHHIIHH4sI",
    b"RIFF",
    _RIFF_SIZE,
    b"WAVE",
    b"fmt ",
    16,  # chunk size
    1,  # PCM format
    1,  # mono
    _SAMPLE_RATE,
    _SAMPLE_RATE * 2,  # byte rate
    2,  # block align
    16,  # bits per sample
    b"data",
    _DATA_SIZE,
) + b"\x00" * _DATA_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tts_response(status_code: int = 200,
                       content: bytes = FAKE_WAV_BYTES,
                       headers: dict | None = None):
    """Create a real httpx.Response to simulate the tts-server backend."""
    hdrs = headers or {"content-type": "audio/wav"}
    resp = httpx.Response(status_code=status_code, content=content, headers=hdrs)
    return resp


# ---------------------------------------------------------------------------
# Config fixture — inject TTS server settings into server.config
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tts_server_config(monkeypatch):
    """Set up TTS server config in the server module."""
    import proxy.server as srv
    cfg = srv.config
    if "server" not in cfg or not isinstance(cfg.get("server"), dict):
        cfg["server"] = {}
    server_cfg = cfg.setdefault("server", {})
    server_cfg["tts_server_host"] = "localhost"
    server_cfg["tts_server_port"] = 8081
    server_cfg["tts_model_path"] = "/models/qwen3-tts.gguf"
    server_cfg["tts_tokenizer_path"] = "/models/qwen3-tts-tokenizer.gguf"
    yield


def _make_voices_response(status_code: int = 200):
    """Create a mock httpx.Response for the /v1/voices endpoint."""
    import json
    content = json.dumps({
        "voices": [
            {"name": "serena", "kind": "speaker"},
            {"name": "vivian", "kind": "speaker"},
            {"name": "uncle_fu", "kind": "speaker"},
            {"name": "ryan", "kind": "speaker"},
            {"name": "aiden", "kind": "speaker"},
            {"name": "ono_anna", "kind": "speaker"},
            {"name": "sohee", "kind": "speaker"},
            {"name": "eric", "kind": "speaker"},
            {"name": "dylan", "kind": "speaker"},
        ]
    }).encode()
    resp = httpx.Response(status_code=status_code, content=content)
    return resp


# ---------------------------------------------------------------------------
# /v1/voices tests
# ---------------------------------------------------------------------------

class TestVoicesRoute:
    """Verify the /v1/voices GET endpoint."""

    @pytest.mark.asyncio
    async def test_list_voices_returns_voices(self, monkeypatch):
        """GET /v1/voices returns the list of available voices."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)
        mock_resp = _make_voices_response()

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get = AsyncMock(return_value=mock_resp)
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.get("/v1/voices")

        assert resp.status_code == 200, \
            f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "voices" in data, "Expected 'voices' key in response"
        assert len(data["voices"]) == 9, "Expected 9 voices"
        assert data["voices"][0]["name"] == "serena"

    @pytest.mark.asyncio
    async def test_list_voices_when_tts_unreachable(self, monkeypatch):
        """GET /v1/voices returns 502 when tts-server is unreachable."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.get("/v1/voices")

        assert resp.status_code == 502, \
            f"Expected 502, got {resp.status_code}: {resp.text}"
        assert "TTS server unreachable" in resp.text


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestTtsRouteContract:
    """Verify the expected request/response contract for /v1/audio/speech."""

    @pytest.mark.asyncio
    async def test_valid_request_returns_wav(self, monkeypatch):
        """A valid POST /v1/audio/speech returns 200 with audio/wav
        content-type."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)
        mock_resp = _make_tts_response()

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_resp)
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "Hello, this is a test of the TTS system.",
                    "voice": "serena",
                    "response_format": "wav",
                },
            )

        assert resp.status_code == 200, \
            f"Expected 200, got {resp.status_code}: {resp.text}"
        content_type = resp.headers.get("content-type", "")
        assert "audio/wav" in content_type, \
            f"Expected audio/wav content-type, got: {content_type}"
        assert len(resp.content) > 0, \
            "Expected non-empty audio response body"

    @pytest.mark.asyncio
    async def test_request_passthrough_forwards_all_params(self, monkeypatch):
        """The proxy must forward model, input, voice, response_format,
        and instructions to tts-server."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)
        mock_resp = _make_tts_response()
        sent_body = None

        async def capture_post(url, *, json=None, **kwargs):
            nonlocal sent_body
            sent_body = json
            return mock_resp

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = capture_post
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "Forward test",
                    "voice": "serena",
                    "response_format": "wav",
                    "instructions": "Speak in a cheerful tone",
                },
            )

        assert sent_body is not None, \
            "Handler did not forward any body to tts-server"
        assert sent_body.get("model") == "qwen3-tts"
        assert sent_body.get("input") == "Forward test"
        assert sent_body.get("voice") == "serena"
        assert sent_body.get("response_format") == "wav"
        assert sent_body.get("instructions") == "Speak in a cheerful tone", \
            "instructions must be forwarded to tts-server"

    @pytest.mark.asyncio
    async def test_instructions_is_forwarded(self, monkeypatch):
        """The proxy must forward the instructions field when provided."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)
        mock_resp = _make_tts_response()
        sent_body = None

        async def capture_post(url, *, json=None, **kwargs):
            nonlocal sent_body
            sent_body = json
            return mock_resp

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = capture_post
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "Hello world",
                    "voice": "serena",
                    "instructions": "Speak slowly and clearly",
                },
            )

        assert sent_body is not None, \
            "Handler did not forward any body to tts-server"
        assert sent_body.get("instructions") == "Speak slowly and clearly", \
            "instructions must be forwarded when provided"

    @pytest.mark.asyncio
    async def test_instructions_is_optional(self, monkeypatch):
        """The proxy must handle requests without instructions
        (backward compatible)."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)
        mock_resp = _make_tts_response()
        sent_body = None

        async def capture_post(url, *, json=None, **kwargs):
            nonlocal sent_body
            sent_body = json
            return mock_resp

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = capture_post
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "Hello world",
                    "voice": "serena",
                },
            )

        assert resp.status_code == 200, \
            f"Expected 200, got {resp.status_code}: {resp.text}"
        assert sent_body is not None, \
            "Handler did not forward any body to tts-server"
        # instructions should not be present in forward_body when not provided
        assert "instructions" not in sent_body or not sent_body.get("instructions"), \
            "instructions should not be forwarded when not provided"

    @pytest.mark.asyncio
    async def test_request_with_minimal_params_succeeds(self, monkeypatch):
        """Only model and input are required; default values for voice and
        format."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)
        mock_resp = _make_tts_response()

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_resp)
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "Hello world",
                },
            )

        assert resp.status_code == 200, \
            f"Expected 200, got {resp.status_code}: {resp.text}"
        assert "audio/wav" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Error case tests
# ---------------------------------------------------------------------------

class TestTtsRouteErrors:
    """Verify error handling for the /v1/audio/speech endpoint."""

    @pytest.mark.asyncio
    async def test_missing_input_returns_400(self):
        """Request without input should return 400 Bad Request."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={"model": "qwen3-tts"},
            )

        assert resp.status_code == 400, \
            f"Expected 400, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_empty_input_returns_400(self):
        """Request with empty input should return 400 Bad Request."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "",
                },
            )

        assert resp.status_code == 400, \
            f"Expected 400, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_missing_model_returns_400(self):
        """Request without model should return 400 Bad Request."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={"input": "test"},
            )

        assert resp.status_code == 400, \
            f"Expected 400, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_tts_server_unreachable_returns_structured_502(self, monkeypatch):
        """When tts-server is unreachable, proxy returns structured 502 error."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError(
                "Connection refused",
                request=httpx.Request("POST",
                                      "http://localhost:8081/v1/audio/speech"),
            )
        )
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "test",
                },
            )

        assert resp.status_code == 502, \
            f"Expected 502, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "error" in data, "Expected 'error' key in structured response"
        assert data["error"]["type"] == "tts_error"
        assert data["error"]["code"] == "tts_server_unreachable"
        msg = data["error"]["message"]
        assert "unreachable" in msg.lower() or "connect" in msg.lower(), \
            f"Message should mention unreachable/connect: {msg}"
        assert data["status"] == 502
        assert data["path"] == "/v1/audio/speech"

    @pytest.mark.asyncio
    async def test_tts_server_timeout_returns_structured_502(self, monkeypatch):
        """When tts-server times out, proxy returns structured 502 with
        timeout duration."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException(
                "Timed out after 30 seconds",
                request=httpx.Request("POST",
                                      "http://localhost:8081/v1/audio/speech"),
            )
        )
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "test",
                },
            )

        assert resp.status_code == 502, \
            f"Expected 502, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "error" in data, "Expected 'error' key in structured response"
        assert data["error"]["type"] == "tts_error"
        assert data["error"]["code"] == "tts_server_timeout"
        msg = data["error"]["message"]
        assert "timeout" in msg.lower(), \
            f"Message should mention timeout: {msg}"
        assert "30" in msg, \
            f"Message should include timeout duration: {msg}"
        assert data["status"] == 502
        assert data["path"] == "/v1/audio/speech"

    @pytest.mark.asyncio
    async def test_tts_server_http_error_returns_structured_502(self, monkeypatch):
        """When tts-server returns an HTTP error, proxy returns structured
        502 with the backend's response body as root-cause context."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        # Simulate tts-server returning a 502 with its own error body
        backend_body = b'{"error":"internal timeout"}'
        mock_resp = _make_tts_response(
            status_code=502,
            content=backend_body,
            headers={"content-type": "application/json"},
        )

        import proxy.server as srv
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_resp)
        monkeypatch.setattr(srv, "_http_client", mock_client)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "test",
                },
            )

        assert resp.status_code == 502, \
            f"Expected 502, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "error" in data, "Expected 'error' key in structured response"
        assert data["error"]["type"] == "tts_error"
        assert data["error"]["code"] == "tts_server_error"
        msg = data["error"]["message"]
        assert "error" in msg.lower() or "TTS" in msg, \
            f"Message should mention error/TTS: {msg}"
        assert "root_cause" in data, \
            "Expected 'root_cause' field with tts-server's response body"
        assert data["status"] == 502
        assert data["path"] == "/v1/audio/speech"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        """Request with malformed JSON body returns 400."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                content=b"not json at all",
                headers={"content-type": "application/json"},
            )

        assert resp.status_code == 400, \
            f"Expected 400, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_input_too_long_returns_400(self):
        """Request with input exceeding max length returns 400."""
        from proxy.server import app

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/audio/speech",
                json={
                    "model": "qwen3-tts",
                    "input": "x" * 10001,
                },
            )

        assert resp.status_code == 400, \
            f"Expected 400, got {resp.status_code}: {resp.text}"
