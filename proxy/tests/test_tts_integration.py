"""
End-to-end integration tests for the TTS pipeline (tts-server + proxy).

These tests require a running qwentts.cpp tts-server on the default port
(localhost:8081).  Tests are automatically skipped when the server is not
available.

To run manually when tts-server is running:
  pytest proxy/tests/test_tts_integration.py -v
"""

import httpx
import pytest

from proxy.server import app

pytestmark = pytest.mark.tts_integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# These defaults must match proxy/config.yaml server.tts_server_*
TTS_SERVER_HOST = "localhost"
TTS_SERVER_PORT = 8081
TTS_URL = f"http://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}/v1/audio/speech"


def have_tts_server() -> bool:
    """Return True if the tts-server is reachable."""
    try:
        httpx.post(
            TTS_URL,
            json={"model": "test", "input": "ping"},
            timeout=3.0,
        )
        return True
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
        return False


# Skip entire class if tts-server is not running
skip_no_tts = pytest.mark.skipif(not have_tts_server(), reason="tts-server is not running")


@skip_no_tts
class TestTtsServerLive:
    """Integration tests that exercise a running tts-server directly.

    These tests bypass the proxy and communicate with the tts-server at
    the configured URL (default http://localhost:8081/v1/audio/speech).
    """

    @pytest.mark.asyncio
    async def test_returns_binary_audio(self):
        """A valid POST returns non-empty binary data."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                TTS_URL,
                json={"model": "test", "input": "Hello world, this is a test."},
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        content = resp.content
        assert len(content) > 44, f"Response too small to be valid WAV: {len(content)} bytes"
        assert content[:4] == b"RIFF", f"Not a WAV file: starts with {content[:4]!r}"

    @pytest.mark.asyncio
    async def test_happy_path_with_minimal_body(self):
        """Minimal valid body (model + input) succeeds."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                TTS_URL,
                json={"model": "test", "input": "Integration test."},
            )
        assert resp.status_code == 200
        assert len(resp.content) > 44

    @pytest.mark.asyncio
    async def test_can_generate_speech_for_multiple_texts(self):
        """Multiple different input texts all produce valid audio."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "This is a short phrase.",
            "Hello from the TTS integration test suite.",
        ]
        async with httpx.AsyncClient(timeout=60) as client:
            for text in texts:
                resp = await client.post(
                    TTS_URL,
                    json={"model": "test", "input": text},
                )
                assert resp.status_code == 200, f"Failed for text: {text[:40]!r}"
                assert len(resp.content) > 44, f"Audio too short for text: {text[:40]!r}"

    @pytest.mark.asyncio
    async def test_proxy_returns_wav_for_valid_request(self):
        """Proxy forwards valid TTS request and returns WAV.

        This test requires both the tts-server AND the proxy to be running,
        so it exercises the full pipeline.
        """
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60) as client:
            resp = await client.post(
                "/v1/audio/speech",
                json={"model": "test", "input": "Hello from the proxy integration test."},
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        content_type = resp.headers.get("content-type", "")
        assert "audio/wav" in content_type, f"Expected audio/wav, got: {content_type}"
        assert len(resp.content) > 44, f"Response too small: {len(resp.content)} bytes"
