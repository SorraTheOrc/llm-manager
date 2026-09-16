"""
Tests for summarizer retry logic on transient failures (R3,
LP-0MTTPXIAB0031AC4).

Verifies:
- Connection errors, read timeouts, and 5xx / 429 responses are retried
  (default 2 retries) before failing open.
- Non-recoverable 4xx responses (auth / model-not-found etc.) are NOT
  retried — fail open immediately.
- Retry attempts are logged at WARNING; exhaustion returns "" unchanged.
- Retry count and inter-attempt delay are config-driven
  (server.compaction_summarizer_retries / ..._retry_delay_seconds).
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from proxy.compaction_summarizer import build_local_summarizer


def _success(content: str = "SUMMARY") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _status(code: int, text: str = "err") -> MagicMock:
    resp = MagicMock()
    resp.status_code = code
    resp.text = text
    return resp


def _cfg(**server):
    return {"server": {"local_model_ctx_size": 262144, **server}}


def _run(mock_client, messages=None, config=None):
    with (
        patch("proxy.compaction_summarizer.httpx.Client") as mock_cls,
        patch("proxy.compaction_summarizer.time.sleep") as mock_sleep,
    ):
        mock_cls.return_value.__enter__.return_value = mock_client
        summarizer = build_local_summarizer(config or {}, llama_port=8080)
        result = summarizer(messages or [{"role": "user", "content": "hi"}])
        return result, mock_client.post.call_count, mock_sleep.call_count


class TestRetriesOnTransientFailures:
    def test_recovers_after_connect_errors(self, caplog):
        """Two ConnectErrors then success → summary returned, 3 attempts."""
        mock_client = MagicMock()
        mock_client.post.side_effect = [
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            _success("RECOVERED"),
        ]
        with caplog.at_level("WARNING", logger="llama-proxy.compaction_summarizer"):
            result, calls, sleeps = _run(mock_client)
        assert result == "RECOVERED"
        assert calls == 3
        assert sleeps == 2
        retry_msgs = [r.message for r in caplog.records if "retrying" in r.message]
        assert len(retry_msgs) == 2

    def test_recovers_after_read_timeout(self):
        mock_client = MagicMock()
        mock_client.post.side_effect = [
            httpx.ReadTimeout("slow"),
            _success("OK"),
        ]
        result, calls, _ = _run(mock_client)
        assert result == "OK"
        assert calls == 2

    def test_recovers_after_503_then_success(self):
        mock_client = MagicMock()
        mock_client.post.side_effect = [_status(503), _success("OK")]
        result, calls, sleeps = _run(mock_client)
        assert result == "OK"
        assert calls == 2
        assert sleeps == 1

    def test_recovers_after_429_then_success(self):
        mock_client = MagicMock()
        mock_client.post.side_effect = [_status(429), _success("OK")]
        result, calls, _ = _run(mock_client)
        assert result == "OK"
        assert calls == 2

    def test_exhausted_connect_errors_return_empty(self, caplog):
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        with caplog.at_level("WARNING", logger="llama-proxy.compaction_summarizer"):
            result, calls, _ = _run(mock_client)
        assert result == ""
        assert calls == 3  # default 2 retries + initial
        assert any("after 3 attempts" in r.message for r in caplog.records)

    def test_exhausted_503_returns_empty(self):
        mock_client = MagicMock()
        mock_client.post.return_value = _status(503)
        result, calls, _ = _run(mock_client)
        assert result == ""
        assert calls == 3


class TestNoRetryOnPermanentFailures:
    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_4xx_not_retried(self, code):
        mock_client = MagicMock()
        mock_client.post.return_value = _status(code)
        result, calls, sleeps = _run(mock_client)
        assert result == ""
        assert calls == 1
        assert sleeps == 0

    def test_auth_error_not_retried(self):
        """401 (auth) is permanent — single attempt, fail open."""
        mock_client = MagicMock()
        mock_client.post.return_value = _status(401, "unauthorized")
        result, calls, _ = _run(mock_client)
        assert result == ""
        assert calls == 1


class TestRetryConfigDriven:
    def test_zero_retries_disables_retry(self):
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        result, calls, sleeps = _run(mock_client, config=_cfg(compaction_summarizer_retries=0))
        assert result == ""
        assert calls == 1
        assert sleeps == 0

    def test_custom_retry_count_and_delay(self):
        mock_client = MagicMock()
        mock_client.post.side_effect = [
            httpx.ConnectError("x"),
            httpx.ConnectError("x"),
            httpx.ConnectError("x"),
            httpx.ConnectError("x"),
            _success("FINAL"),
        ]
        result, calls, sleeps = _run(
            mock_client,
            config=_cfg(
                compaction_summarizer_retries=4,
                compaction_summarizer_retry_delay_seconds=0.25,
            ),
        )
        assert result == "FINAL"
        assert calls == 5
        assert sleeps == 4

    def test_retry_delay_applied_between_attempts(self):
        """time.sleep is called with the configured delay."""
        mock_client = MagicMock()
        mock_client.post.side_effect = [httpx.ConnectError("x"), _success("OK")]
        with (
            patch("proxy.compaction_summarizer.httpx.Client") as mock_cls,
            patch("proxy.compaction_summarizer.time.sleep") as mock_sleep,
        ):
            mock_cls.return_value.__enter__.return_value = mock_client
            summarizer = build_local_summarizer(
                _cfg(compaction_summarizer_retry_delay_seconds=1.25),
                llama_port=8080,
            )
            assert summarizer([{"role": "user", "content": "hi"}]) == "OK"
        mock_sleep.assert_called_once_with(1.25)
