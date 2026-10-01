"""Unit tests for the shared SSE event re-framer (LP-0MUOBUPBC002GYTL).

The proxy forwards raw upstream ``aiter_bytes()`` reads to the client and
appends synthetic/retry events. A single read may contain a complete event
plus the start of the next (or end mid-event), so forwarding raw bytes lets a
later event be concatenated with a dangling partial — the client then merges
both into one event whose ``data:`` payload is not valid JSON.

:class:`proxy.utils.SSEEventReframer` buffers trailing partial data and only
emits complete events. These tests pin its framing and discard semantics.
"""

import pytest
from proxy.utils import SSEEventReframer, SSEFrameOverflowError

EVENT_A = b'data: {"choices":[{"delta":{"content":"A"},"index":0}]}\n\n'
EVENT_B = b'data: {"choices":[{"delta":{"content":"B"},"index":0}]}\n\n'


def test_complete_event_is_emitted_verbatim():
    """A single complete event is returned unchanged by one feed call."""
    framer = SSEEventReframer()
    assert framer.feed(EVENT_A) == [EVENT_A]
    assert framer.pending == b""
    assert framer.has_pending() is False


def test_trailing_partial_is_buffered_not_emitted():
    """A read ending mid-event emits nothing and holds the partial."""
    framer = SSEEventReframer()
    assert framer.feed(b'data: {"choices":[{"delta":{"content":"Hal') == []
    assert framer.pending == b'data: {"choices":[{"delta":{"content":"Hal'
    assert framer.has_pending() is True


def test_partial_completed_across_reads_is_emitted_once():
    """A partial split across two reads surfaces as one complete event."""
    framer = SSEEventReframer()
    first = EVENT_A[:-2]  # drop the terminating "\n\n"
    assert framer.feed(first) == []
    assert framer.feed(b"\n\n" + EVENT_B) == [EVENT_A, EVENT_B]
    assert framer.pending == b""


def test_complete_event_followed_by_partial_in_same_read():
    """A read with a complete event plus a dangling partial emits only the event."""
    framer = SSEEventReframer()
    trailing = b'data: {"choices":[{"delta":{"content":"NEXT'
    assert framer.feed(EVENT_A + trailing) == [EVENT_A]
    assert framer.pending == trailing


def test_multiple_events_in_one_read_are_split():
    """Several events in a single read are emitted as separate events."""
    framer = SSEEventReframer()
    third = b"data: [DONE]\n\n"
    assert framer.feed(EVENT_A + EVENT_B + third) == [EVENT_A, EVENT_B, third]


def test_crlf_event_boundary_is_recognised():
    """CRLF-terminated events (\r\n\r\n) are framed correctly."""
    framer = SSEEventReframer()
    crlf = b'data: {"choices":[{"delta":{"content":"A"},"index":0}]}\r\n\r\n'
    assert framer.feed(crlf) == [crlf]
    assert framer.pending == b""


def test_split_crlf_boundary_is_not_emitted_prematurely():
    """A CRLF boundary split across reads is held until complete."""
    framer = SSEEventReframer()
    assert framer.feed(b'data: {"x":1}\r\n\r') == []
    assert framer.feed(b"\n") == [b'data: {"x":1}\r\n\r\n']


def test_keepalive_comment_event_is_forwarded_as_a_complete_event():
    """An SSE comment (keep-alive) is a complete event and is forwarded."""
    framer = SSEEventReframer()
    keepalive = b": keep-alive\n\n"
    assert framer.feed(keepalive) == [keepalive]


def test_empty_chunk_is_a_noop():
    """An empty chunk neither emits nor buffers anything."""
    framer = SSEEventReframer()
    assert framer.feed(b"") == []
    assert framer.pending == b""


def test_discard_pending_drops_the_partial_and_returns_it():
    """discard_pending drops the held partial so it can never be emitted."""
    framer = SSEEventReframer()
    framer.feed(EVENT_A + b'data: {"choices":[{"delta":{"content":"DANGLING')
    dropped = framer.discard_pending()
    assert dropped == b'data: {"choices":[{"delta":{"content":"DANGLING'
    assert framer.pending == b""
    assert framer.has_pending() is False
    # A subsequent stream's first event is emitted cleanly, not concatenated.
    assert framer.feed(EVENT_B) == [EVENT_B]


def test_oversized_pending_event_fails_closed():
    """A single event exceeding the cap raises rather than buffering unbounded."""
    framer = SSEEventReframer(max_pending_bytes=16)
    with pytest.raises(SSEFrameOverflowError) as excinfo:
        framer.feed(b"data: " + b"x" * 32)
    assert excinfo.value.max_bytes == 16
    # The oversized partial is dropped so the cap holds.
    assert framer.pending == b""


def test_oversized_event_with_complete_prefix_still_fails_closed():
    """A complete event followed by an oversized partial raises, not buffers."""
    complete = b"data: {}\n\n"
    framer = SSEEventReframer(max_pending_bytes=64)
    with pytest.raises(SSEFrameOverflowError):
        framer.feed(complete + b"data: " + b"y" * 128)
    assert framer.pending == b""


def test_event_within_cap_after_previous_partial_does_not_raise():
    """A large-but-bounded event that completes does not trip the cap."""
    framer = SSEEventReframer(max_pending_bytes=1024)
    event = b"data: " + b"z" * 512 + b"\n\n"
    assert framer.feed(event) == [event]
    assert framer.pending == b""
