"""
Tests for gating per-chunk SSE logging behind verbose mode (LP-0MS9GAN2P002NR4M).

Verifies that:
1. By default (no verbose), ``log_response_chunk`` emits ``STREAM CHUNK`` lines
   at DEBUG level, so they are NOT written to proxy.log at the default INFO
   level.
2. When verbose is enabled (config ``logging.verbose_chunks`` or env var
   ``LLAMA_PROXY_VERBOSE``), ``STREAM CHUNK`` lines are emitted at INFO level.
3. Per-stream lifecycle lines (``Stream finished:``) remain at INFO even in
   non-verbose mode — only the per-chunk noise is demoted.
4. The console-suppression behaviour (ContentOnlyConsoleHandler) still hides
   STREAM CHUNK content from the console even in verbose mode.
"""

import logging
from pathlib import Path

import proxy.server as server


def _configure_logging(tmp_path: Path, *, verbose_chunks: bool = False) -> None:
    """Set up the llama-proxy logger with a minimal config in tmp_path."""
    cfg = {
        "logging": {
            "directory": str(tmp_path / "logs"),
            "rotation_hours": 1,
            "retention_days": 1,
            "level": "INFO",
            "verbose_chunks": verbose_chunks,
        }
    }
    logger = logging.getLogger("llama-proxy")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    server.setup_logging(cfg)


def _chunk_records(caplog) -> list:
    """Return log records whose message starts with ``STREAM CHUNK``."""
    return [
        r for r in caplog.records
        if str(r.getMessage()).startswith("STREAM CHUNK")
    ]


# ── Default (non-verbose) behaviour ───────────────────────────────────────


def test_chunk_logged_at_debug_by_default(caplog, tmp_path):
    """STREAM CHUNK lines are emitted at DEBUG when verbose is off."""
    _configure_logging(tmp_path, verbose_chunks=False)
    caplog.set_level(logging.DEBUG, logger="llama-proxy")

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    records = _chunk_records(caplog)
    assert records, "expected a STREAM CHUNK record at DEBUG"
    assert all(r.levelno == logging.DEBUG for r in records), (
        "STREAM CHUNK must be DEBUG level in non-verbose mode"
    )


def test_no_info_chunk_record_by_default(caplog, tmp_path):
    """No INFO-level STREAM CHUNK record is emitted by default."""
    _configure_logging(tmp_path, verbose_chunks=False)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    assert not any(
        r.levelno == logging.INFO and str(r.getMessage()).startswith("STREAM CHUNK")
        for r in caplog.records
    ), "STREAM CHUNK must not appear at INFO level in non-verbose mode"


# ── Verbose via config key ────────────────────────────────────────────────


def test_chunk_logged_at_info_when_config_verbose(caplog, tmp_path):
    """STREAM CHUNK lines are emitted at INFO when logging.verbose_chunks=true."""
    _configure_logging(tmp_path, verbose_chunks=True)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    records = _chunk_records(caplog)
    assert records, "expected a STREAM CHUNK record at INFO"
    assert all(r.levelno == logging.INFO for r in records), (
        "STREAM CHUNK must be INFO level in verbose mode"
    )


# ── Verbose via environment variable ──────────────────────────────────────


def test_chunk_logged_at_info_when_env_verbose(caplog, tmp_path, monkeypatch):
    """LLAMA_PROXY_VERBOSE=1 enables INFO-level STREAM CHUNK logging."""
    monkeypatch.setenv("LLAMA_PROXY_VERBOSE", "1")
    _configure_logging(tmp_path, verbose_chunks=False)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    records = _chunk_records(caplog)
    assert records, "expected a STREAM CHUNK record at INFO"
    assert all(r.levelno == logging.INFO for r in records), (
        "LLAMA_PROXY_VERBOSE=1 must produce INFO-level STREAM CHUNK records"
    )


def test_env_verbose_overrides_config_false(caplog, tmp_path, monkeypatch):
    """Env var takes precedence over verbose_chunks=false in config."""
    monkeypatch.setenv("LLAMA_PROXY_VERBOSE", "1")
    _configure_logging(tmp_path, verbose_chunks=False)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    assert any(
        r.levelno == logging.INFO and str(r.getMessage()).startswith("STREAM CHUNK")
        for r in caplog.records
    ), "Env var must enable INFO-level STREAM CHUNK logging"


# ── Lifecycle lines stay at INFO ──────────────────────────────────────────


def test_stream_finished_stays_info_without_verbose(caplog, tmp_path):
    """Per-stream Stream finished lines remain INFO even in non-verbose mode."""
    _configure_logging(tmp_path, verbose_chunks=False)
    caplog.set_level(logging.INFO, logger="llama-proxy")

    chunk = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    server.log_response_chunk(chunk, session_id="sess123", model="test-model")

    assert any(
        r.levelno == logging.INFO and "Stream finished: reason=stop" in str(r.getMessage())
        for r in caplog.records
    ), "Stream finished lifecycle line must remain at INFO"


# ── Console suppression preserved ─────────────────────────────────────────


def test_console_still_suppresses_chunk_content_in_verbose_mode(tmp_path):
    """Console handler still hides STREAM CHUNK content when verbose is on."""
    import io

    cfg = {
        "logging": {
            "directory": str(tmp_path / "logs"),
            "rotation_hours": 1,
            "retention_days": 1,
            "level": "INFO",
            "verbose_chunks": True,
        }
    }
    logger = logging.getLogger("llama-proxy")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    server.setup_logging(cfg)

    console_handler = None
    for h in logger.handlers:
        if isinstance(h, server.ContentOnlyConsoleHandler):
            console_handler = h
            break
    assert console_handler is not None, "ContentOnlyConsoleHandler not installed"

    strio = io.StringIO()
    console_handler.setStream(strio)
    strio.truncate(0)
    strio.seek(0)

    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    server.log_response_chunk(chunk)

    assert strio.getvalue() == "", (
        "STREAM CHUNK content must not appear in console even in verbose mode"
    )
