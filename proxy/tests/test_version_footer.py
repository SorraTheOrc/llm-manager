"""Tests for version footer feature: version capture and template injection.

Tests cover:
- _capture_llama_server_version() with success and failure modes
- _capture_rocm_version() with success and failure modes
- Template injection of version placeholders in both index and view_logs pages
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from subprocess import TimeoutExpired


# ---------------------------------------------------------------------------
# Tests for _capture_llama_server_version
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_llama_server_version_success():
    """Mock llama-server --version and verify parsed build number."""
    from proxy.server import _capture_llama_server_version

    mock_result = MagicMock()
    mock_result.stdout = "build: 4321 (abc12345)\n"
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        version = await _capture_llama_server_version()

    assert version is not None
    assert "build 4321" in version or "4321" in version


@pytest.mark.asyncio
async def test_capture_llama_server_version_full_format():
    """Mock llama-server --version with version: X.Y.Z format."""
    from proxy.server import _capture_llama_server_version

    mock_result = MagicMock()
    mock_result.stdout = "version: 1.2.3 (deadbeef)\n"
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        version = await _capture_llama_server_version()

    assert version is not None
    # Should contain meaningful version info
    assert len(version) > 0
    assert version != "unknown"


@pytest.mark.asyncio
async def test_capture_llama_server_version_failure():
    """When llama-server --version fails, capture returns 'unknown'."""
    from proxy.server import _capture_llama_server_version

    with patch("subprocess.run", side_effect=FileNotFoundError("No such file")):
        version = await _capture_llama_server_version()

    assert version == "unknown"


@pytest.mark.asyncio
async def test_capture_llama_server_version_timeout():
    """When llama-server --version times out, capture returns 'unknown'."""
    from proxy.server import _capture_llama_server_version

    with patch("subprocess.run", side_effect=TimeoutExpired("cmd", timeout=5)):
        version = await _capture_llama_server_version()

    assert version == "unknown"


@pytest.mark.asyncio
async def test_capture_llama_server_version_nonzero_exit():
    """When llama-server --version returns non-zero, capture returns 'unknown'."""
    from proxy.server import _capture_llama_server_version

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        version = await _capture_llama_server_version()

    assert version == "unknown"


# ---------------------------------------------------------------------------
# Tests for _capture_rocm_version
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_rocm_version_success_showtag():
    """Mock rocm-smi --showtag and verify parsed version string."""
    from proxy.server import _capture_rocm_version

    mock_result = MagicMock()
    mock_result.stdout = "ROCm 7.2.4\n"
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        version = await _capture_rocm_version()

    assert version is not None
    assert "7.2.4" in version or "ROCm" in version


@pytest.mark.asyncio
async def test_capture_rocm_version_fallback_version():
    """When --showtag fails, fall back to rocm-smi --version."""
    from proxy.server import _capture_rocm_version

    # First call raises FileNotFoundError (--showtag not available),
    # second succeeds with --version
    mock_result2 = MagicMock()
    mock_result2.stdout = "ROCm version: 6.0.0\n"
    mock_result2.returncode = 0

    with patch("subprocess.run", side_effect=[
        FileNotFoundError("No such file"),
        mock_result2,
    ]):
        version = await _capture_rocm_version()

    assert version is not None
    assert "6.0.0" in version or "ROCm" in version
    assert version != "unknown"


@pytest.mark.asyncio
async def test_capture_rocm_version_failure():
    """When both rocm-smi commands fail, capture returns 'unknown'."""
    from proxy.server import _capture_rocm_version

    with patch("subprocess.run", side_effect=FileNotFoundError("No such file")):
        version = await _capture_rocm_version()

    assert version == "unknown"


@pytest.mark.asyncio
async def test_capture_rocm_version_timeout():
    """When rocm-smi --showtag times out, capture returns 'unknown'."""
    from proxy.server import _capture_rocm_version

    with patch("subprocess.run", side_effect=TimeoutExpired("cmd", timeout=5)):
        version = await _capture_rocm_version()

    assert version == "unknown"


# ---------------------------------------------------------------------------
# Tests for module-level version variable initialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_has_version_variables():
    """Module-level version variables exist and default to 'unknown'."""
    import proxy.server as srv

    # Variables should exist at module level
    assert hasattr(srv, "llama_server_version")
    assert hasattr(srv, "rocm_version")
    assert srv.llama_server_version == "unknown"
    assert srv.rocm_version == "unknown"


# ---------------------------------------------------------------------------
# Tests for template injection (index page)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_template_includes_version_footer():
    """Verify that the index page replaces __LLAMA_SERVER_VERSION__ and __ROCM_VERSION__."""
    from proxy.server import app
    import httpx

    transport = httpx.ASGITransport(app=app)

    with patch("proxy.server.llama_server_version", "build 1234 (abc123)"):
        with patch("proxy.server.rocm_version", "ROCm 7.2.4"):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                resp = await ac.get("/")

    assert resp.status_code == 200
    html = resp.text
    assert "build 1234" in html
    assert "ROCm 7.2.4" in html
    assert "llama-server" in html


@pytest.mark.asyncio
async def test_index_template_shows_unknown_when_not_captured():
    """When version data is 'unknown', the footer still renders."""
    from proxy.server import app
    import httpx

    transport = httpx.ASGITransport(app=app)

    with patch("proxy.server.llama_server_version", "unknown"):
        with patch("proxy.server.rocm_version", "unknown"):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                resp = await ac.get("/")

    assert resp.status_code == 200
    html = resp.text
    assert "unknown" in html


# ---------------------------------------------------------------------------
# Tests for template injection (view_logs page)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_view_logs_template_includes_version_footer():
    """Verify that the view_logs page replaces __LLAMA_SERVER_VERSION__ and __ROCM_VERSION__."""
    from proxy.server import app
    import httpx

    transport = httpx.ASGITransport(app=app)

    with patch("proxy.server.llama_server_version", "build 4321 (def567)"):
        with patch("proxy.server.rocm_version", "ROCm 6.0.0"):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                resp = await ac.get("/logs")

    assert resp.status_code == 200
    html = resp.text
    assert "build 4321" in html
    assert "ROCm 6.0.0" in html
    assert "llama-server" in html


@pytest.mark.asyncio
async def test_view_logs_template_shows_unknown_when_not_captured():
    """When version data is 'unknown', the view_logs footer still renders."""
    from proxy.server import app
    import httpx

    transport = httpx.ASGITransport(app=app)

    with patch("proxy.server.llama_server_version", "unknown"):
        with patch("proxy.server.rocm_version", "unknown"):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                resp = await ac.get("/logs")

    assert resp.status_code == 200
    html = resp.text
    assert "unknown" in html
