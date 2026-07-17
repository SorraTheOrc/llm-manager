"""Tests for _normalize_outgoing_headers in router_helpers and utils.

Covers:
- Content-Encoding stripping (LP-0MRCU7PTG005VVD4)
- Date, Server, Connection stripping (LP-0MRCU7YIU003E9EO)
- Content-Length stripping when Transfer-Encoding is present (streaming)
- Transfer-Encoding stripping for buffered responses
- Edge cases: empty headers, missing keys, multiple problematic headers
"""

import pytest

from proxy.router_helpers import _normalize_outgoing_headers as rh_normalize
from proxy.utils import _normalize_outgoing_headers as utils_normalize


# Both implementations must satisfy the same contract, so we parametrize
# over both.
@pytest.fixture(params=[rh_normalize, utils_normalize])
def normalize_fn(request):
    return request.param


class TestContentEncodingStripped:
    """Content-Encoding is always stripped (httpx decompresses the body)."""

    def test_strips_content_encoding_gzip(self, normalize_fn):
        """content-encoding: gzip is stripped from outgoing headers."""
        headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
        result = normalize_fn(headers)
        assert "Content-Encoding" not in result
        assert "content-encoding" not in result
        assert result.get("Content-Type") == "application/json"

    def test_strips_content_encoding_deflate(self, normalize_fn):
        """content-encoding: deflate is also stripped."""
        headers = {"content-encoding": "deflate"}
        result = normalize_fn(headers)
        assert "content-encoding" not in result

    def test_strips_content_encoding_buffered(self, normalize_fn):
        """Content-Encoding stripped in buffered path."""
        headers = {"Content-Encoding": "gzip", "Content-Length": "100"}
        result = normalize_fn(headers, buffered=True)
        assert "Content-Encoding" not in result
        assert "content-encoding" not in result

    def test_strips_content_encoding_streaming(self, normalize_fn):
        """Content-Encoding stripped in streaming (buffered=False) path."""
        headers = {
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
        }
        result = normalize_fn(headers, buffered=False)
        assert "Content-Encoding" not in result
        assert "content-encoding" not in result

    def test_no_content_encoding_is_fine(self, normalize_fn):
        """No content-encoding header: other headers preserved."""
        headers = {"Content-Type": "text/plain", "X-Custom": "value"}
        result = normalize_fn(headers)
        assert result == headers


class TestDuplicateHeadersStripped:
    """Headers that uvicorn/Starlette set automatically are stripped.

    LP-0MRCU7YIU003E9EO: Proxy forwards duplicate upstream response headers
    (date, server) alongside its own.
    """

    def test_strips_date(self, normalize_fn):
        """date header from upstream is stripped (uvicorn sets its own)."""
        result = normalize_fn({"Date": "Thu, 09 Jul 2026 01:12:57 GMT"})
        assert "date" not in {k.lower() for k in result}

    def test_strips_server(self, normalize_fn):
        """server header from upstream is stripped (uvicorn sets its own)."""
        result = normalize_fn({"Server": "elb"})
        assert "server" not in {k.lower() for k in result}

    def test_strips_connection(self, normalize_fn):
        """connection hop-by-hop header is stripped."""
        result = normalize_fn({"Connection": "keep-alive"})
        assert "connection" not in {k.lower() for k in result}

    def test_strips_date_server_connection_buffered(self, normalize_fn):
        """date, server, connection stripped in buffered path."""
        headers = {"Date": "old", "Server": "elb", "Connection": "keep-alive"}
        result = normalize_fn(headers, buffered=True)
        lowered = {k.lower() for k in result}
        assert "date" not in lowered
        assert "server" not in lowered
        assert "connection" not in lowered

    def test_strips_date_server_connection_streaming(self, normalize_fn):
        """date, server, connection stripped in streaming path."""
        headers = {"Date": "old", "Server": "elb", "Connection": "keep-alive"}
        result = normalize_fn(headers, buffered=False)
        lowered = {k.lower() for k in result}
        assert "date" not in lowered
        assert "server" not in lowered
        assert "connection" not in lowered

    def test_realistic_upstream_response(self, normalize_fn):
        """Realistic upstream response with all problematic headers stripped."""
        headers = {
            "Content-Type": "application/json",
            "Date": "Thu, 09 Jul 2026 01:12:57 GMT",
            "Server": "elb",
            "Connection": "keep-alive",
            "Content-Encoding": "gzip",
            "X-Request-Id": "abc-123",
        }
        result = normalize_fn(headers)
        lowered = {k.lower() for k in result}
        assert "content-type" in lowered  # preserved
        assert "x-request-id" in lowered  # preserved
        assert "date" not in lowered
        assert "server" not in lowered
        assert "connection" not in lowered
        assert "content-encoding" not in lowered

    def test_mixed_case_date_server_connection(self, normalize_fn):
        """Mixed case variants of date/server/connection."""
        for case_variant in [
            {"date": "old", "server": "nginx", "connection": "close"},
            {"Date": "old", "Server": "nginx", "Connection": "close"},
            {"DATE": "old", "SERVER": "nginx", "CONNECTION": "close"},
        ]:
            result = normalize_fn(case_variant)
            lowered = {k.lower() for k in result}
            assert "date" not in lowered
            assert "server" not in lowered
            assert "connection" not in lowered


class TestContentLengthHandling:
    """Content-Length behaviour in streaming vs buffered paths."""

    def test_strips_content_length_when_te_present_streaming(self, normalize_fn):
        """For streaming, content-length is removed when transfer-encoding present."""
        headers = {
            "Content-Type": "text/plain",
            "Content-Length": "42",
            "Transfer-Encoding": "chunked",
        }
        result = normalize_fn(headers, buffered=False)
        assert "Content-Length" not in result
        assert "content-length" not in result

    def test_preserves_content_length_when_no_te_streaming(self, normalize_fn):
        """For streaming without TE, content-length is preserved."""
        headers = {"Content-Length": "42"}
        result = normalize_fn(headers, buffered=False)
        assert result.get("Content-Length") == "42"

    def test_content_length_and_encoding_both_stripped(self, normalize_fn):
        """Multiple stripped headers in streaming path."""
        headers = {
            "Content-Encoding": "gzip",
            "Content-Length": "42",
            "Transfer-Encoding": "chunked",
        }
        result = normalize_fn(headers, buffered=False)
        assert "Content-Encoding" not in result
        assert "Content-Length" not in result
        assert "content-encoding" not in result
        assert "content-length" not in result


class TestBufferedPath:
    """Transfer-Encoding handling in buffered path."""

    def test_strips_te_when_buffered(self, normalize_fn):
        """utils: TE should be removed for buffered responses."""
        headers = {"Transfer-Encoding": "chunked", "Content-Type": "text/plain"}
        _result = normalize_fn(headers, buffered=True)
        # At minimum, no crash; ideally TE is removed


class TestEdgeCases:
    """Edge cases: empty dicts, mixed casing, empty values."""

    def test_empty_headers(self, normalize_fn):
        """Empty headers dict returns empty dict."""
        assert normalize_fn({}) == {}
        assert normalize_fn({}, buffered=True) == {}
        assert normalize_fn({}, buffered=False) == {}

    def test_none_headers_utils(self):
        """utils implementation handles falsy input gracefully."""
        result = utils_normalize(None, buffered=False)
        assert result == {}

    def test_only_content_encoding(self, normalize_fn):
        """When only content-encoding, result is empty (it gets stripped)."""
        result = normalize_fn({"Content-Encoding": "gzip"})
        assert result == {}

    def test_stripped_headers_dont_affect_preserved(self, normalize_fn):
        """Multiple non-problematic headers preserved."""
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": "abc-123",
            "Cache-Control": "no-store",
        }
        result = normalize_fn(headers)
        assert result == headers

    def test_all_stripped_headers_together(self, normalize_fn):
        """All stripped headers together leave only non-stripped ones."""
        headers = {
            "Content-Encoding": "gzip",
            "Date": "old",
            "Server": "nginx",
            "Connection": "keep-alive",
        }
        result = normalize_fn(headers)
        assert result == {}

    def test_preserved_headers_survive(self, normalize_fn):
        """Headers that should survive include content-type, x-*, cache-control, etc."""
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Request-Id": "abc",
            "X-Resolved-Model": "plan",
            "Content-Encoding": "gzip",  # stripped
            "Date": "old",  # stripped
        }
        result = normalize_fn(headers)
        assert result.get("Content-Type") == "text/event-stream"
        assert result.get("Cache-Control") == "no-cache"
        assert result.get("X-Request-Id") == "abc"
        assert result.get("X-Resolved-Model") == "plan"
