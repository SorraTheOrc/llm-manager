"""Parse contract for b8782 ``prompt_progress`` (parent AC1, LP-0MVR7K63C006O9ZL).

Parent: LP-0MVQIT9QY007QU27 — ``local_eviction_after_acceptance
reason=prefill_deadline`` (producer-approved Option D). The deciding signal is
the *measured* reuse reported by llama-server itself: when the upstream request
carries ``return_progress: true`` (injected by the capture half, F2), llama.cpp
b8782 emits ``prompt_progress`` objects inside the streaming SSE response. The
first event carries ``cache = n_prompt_tokens_cache`` (the exact reused prefix)
against ``total`` (the full prompt).

This item (the AC1 coverage guardian) pins the *parse* half of that contract
against the four real event shapes b8782 produces, so any later change to the
extraction (or an upstream shape drift) is caught by a unit test rather than by
a silently wrong abort decision:

  initial 0%       ``cache`` set, ``processed``/``time_ms`` still 0
  in progress      ``processed``/``time_ms`` advance, ``cache`` fixed
  fully warm       ``cache == total`` (``measured_new_tokens == 0``)
  fake warm        tiny ``cache`` against a large ``total``

Only the public extraction API is exercised (``parse_prompt_progress_event``,
``PromptProgress``, ``PromptProgressCapture``) through the same
``SSEEventReframer`` byte path ``proxy_to_local`` uses; nothing here re-derives
production logic.
"""

from __future__ import annotations

import json

import pytest
from proxy.prompt_progress import (
    PROMPT_PROGRESS_FIELDS,
    PromptProgressCapture,
    parse_prompt_progress_event,
)
from proxy.utils import SSEEventReframer

#: The b8782 event shape verified live (F1): ``prompt_progress`` is a
#: top-level key next to ``choices`` — never nested inside a delta.
CONTENT_CHUNK = (
    b'data: {"choices":[{"delta":{"content":"hi"},"index":0}],'
    b'"created":0,"id":"chatcmpl-1","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)

#: The fake-warm signature from the measured window: a ~56k prompt whose slot
#: had been clobbered, so only 200 tokens are actually reused.
FAKE_WARM_EVENT = (
    b'data: {"choices":[{"delta":{"role":"assistant"}}],'
    b'"prompt_progress":{"total":56000,"cache":200,"processed":0,"time_ms":0}}\n\n'
)

#: The four real shapes, as ``(event, total, cache, processed, time_ms)``.
B8782_SHAPES = (
    # Initial 0% event: cache is already the reused prefix, nothing processed.
    (FAKE_WARM_EVENT, 56_000, 200, 0, 0.0),
    # In-progress event: cache is fixed, processed/time_ms advance.
    (
        b'data: {"choices":[{"delta":{"content":"hi"}}],'
        b'"prompt_progress":{"total":56000,"cache":200,'
        b'"processed":8192,"time_ms":410}}\n\n',
        56_000,
        200,
        8_192,
        410.0,
    ),
    # Fully warm: the whole prompt was reused, nothing to compute.
    (
        b'data: {"choices":[{"delta":{"content":"hi"}}],'
        b'"prompt_progress":{"total":30000,"cache":30000,'
        b'"processed":0,"time_ms":0}}\n\n',
        30_000,
        30_000,
        0,
        0.0,
    ),
    # One of the 252 working large prefills: cache ~= 0.999 x total.
    (
        b'data: {"choices":[{"delta":{"content":"hi"}}],'
        b'"prompt_progress":{"total":127872,"cache":127744,'
        b'"processed":0,"time_ms":0}}\n\n',
        127_872,
        127_744,
        0,
        0.0,
    ),
)


def _event(progress: dict[str, object], **body: object) -> bytes:
    """Serialise one OAI-compatible chunk carrying *progress* (b8782 layout)."""
    chunk = {
        "choices": [{"delta": {"role": "assistant"}, "index": 0}],
        "created": 0,
        "id": "chatcmpl-progress",
        "model": "Qwen3",
        "object": "chat.completion.chunk",
        "prompt_progress": progress,
        **body,
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


# ---------------------------------------------------------------------------
# AC1 — the four fields parse out of the real b8782 event shapes
# ---------------------------------------------------------------------------


def test_contract_field_set_matches_b8782_to_json():
    """The contract names exactly the four keys ``to_json()`` emits, in order."""
    assert PROMPT_PROGRESS_FIELDS == ("total", "cache", "processed", "time_ms")


@pytest.mark.parametrize(("event", "total", "cache", "processed", "time_ms"), B8782_SHAPES)
def test_real_b8782_shapes_parse_every_field(
    event: bytes,
    total: int,
    cache: int,
    processed: int,
    time_ms: float,
) -> None:
    """Each real shape yields all four fields with the values it carried."""
    progress = parse_prompt_progress_event(event)

    assert progress is not None
    assert progress.total == total
    assert progress.cache == cache
    assert progress.processed == processed
    assert progress.time_ms == time_ms


def test_derived_reuse_signals_match_the_measured_shapes():
    """``cache / total`` and ``total - cache`` describe fake-warm vs warm."""
    fake_warm = parse_prompt_progress_event(FAKE_WARM_EVENT)
    assert fake_warm is not None
    # The deciding signal: a large prompt with almost nothing reused.
    assert fake_warm.measured_cached_ratio == pytest.approx(200 / 56_000)
    assert fake_warm.measured_new_tokens == 55_800

    warm = parse_prompt_progress_event(B8782_SHAPES[2][0])
    assert warm is not None
    assert warm.measured_cached_ratio == pytest.approx(1.0)
    assert warm.measured_new_tokens == 0


# ---------------------------------------------------------------------------
# AC1 — absent/invalid fields degrade to explicit ``None``, never a default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", PROMPT_PROGRESS_FIELDS)
def test_a_missing_field_is_absent_not_defaulted(missing: str) -> None:
    """Omitting one key leaves that field ``None`` and the others measured."""
    progress = {
        "total": 56_000,
        "cache": 200,
        "processed": 8_192,
        "time_ms": 410,
    }
    progress.pop(missing)

    parsed = parse_prompt_progress_event(_event(progress))

    assert parsed is not None
    assert getattr(parsed, missing) is None
    for field in PROMPT_PROGRESS_FIELDS:
        if field != missing:
            assert getattr(parsed, field) == progress[field]


@pytest.mark.parametrize(
    "value",
    ["x", None, True, 1.5, [], {}],
    ids=["string", "null", "bool", "fractional", "list", "object"],
)
def test_non_integral_token_counts_are_absent(value: object) -> None:
    """A token count that is not a whole number is never reported as measured."""
    parsed = parse_prompt_progress_event(_event({"total": value, "cache": 200}))

    assert parsed is not None
    assert parsed.total is None
    # A bad sibling field does not discard the measurement.
    assert parsed.cache == 200
    assert parsed.measured_cached_ratio is None
    assert parsed.measured_new_tokens is None


def test_integral_floats_are_accepted_as_token_counts() -> None:
    """JSON numbers without a fraction (``56000.0``) are the same measurement."""
    parsed = parse_prompt_progress_event(_event({"total": 56000.0, "cache": 200.0, "processed": 0, "time_ms": 0}))

    assert parsed is not None
    assert parsed.total == 56_000
    assert parsed.cache == 200
    assert parsed.measured_new_tokens == 55_800


@pytest.mark.parametrize("value", ["x", None, True, [], {}], ids=["string", "null", "bool", "list", "obj"])
def test_non_numeric_time_ms_is_absent(value: object) -> None:
    """``time_ms`` is a numeric duration; anything else is "not reported"."""
    parsed = parse_prompt_progress_event(_event({"total": 56000, "cache": 200, "time_ms": value}))

    assert parsed is not None
    assert parsed.time_ms is None
    assert parsed.total == 56_000


def test_derived_signals_are_absent_without_total_or_cache() -> None:
    """No denominator or no reuse ⇒ no ratio (and no fabricated new-token count)."""
    no_total = parse_prompt_progress_event(_event({"cache": 200}))
    assert no_total is not None
    assert no_total.measured_cached_ratio is None
    assert no_total.measured_new_tokens is None

    zero_total = parse_prompt_progress_event(_event({"total": 0, "cache": 0}))
    assert zero_total is not None
    assert zero_total.measured_cached_ratio is None
    assert zero_total.measured_new_tokens == 0

    no_cache = parse_prompt_progress_event(_event({"total": 56_000}))
    assert no_cache is not None
    assert no_cache.measured_cached_ratio is None
    assert no_cache.measured_new_tokens is None


# ---------------------------------------------------------------------------
# AC1 — non-progress and unusable payloads are "no measurement"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        CONTENT_CHUNK,
        b"data: [DONE]\n\n",
        b'data: {"choices":[],"prompt_progress":"not-an-object"}\n\n',
        b'data: {"choices":[],"prompt_progress":null}\n\n',
        b"data: not-json\n\n",
        b"data: [1, 2, 3]\n\n",
        b"\xff\xfe not utf-8 \n\n",
        b"",
        "",
        None,
        123,
    ],
    ids=[
        "content-chunk",
        "done-sentinel",
        "progress-not-object",
        "progress-null",
        "invalid-json",
        "json-list",
        "invalid-utf8",
        "empty-bytes",
        "empty-str",
        "none",
        "non-text",
    ],
)
def test_unusable_event_parses_to_no_measurement(event: object) -> None:
    """Anything without a ``prompt_progress`` object is simply not measured."""
    assert parse_prompt_progress_event(event) is None  # type: ignore[arg-type]


def test_bare_json_payload_is_accepted() -> None:
    """An already-unwrapped chunk body parses the same as a reframed event."""
    bare = json.dumps({"prompt_progress": {"total": 10, "cache": 4}}).encode()

    parsed = parse_prompt_progress_event(bare)

    assert parsed is not None
    assert (parsed.total, parsed.cache) == (10, 4)


# ---------------------------------------------------------------------------
# AC1 — the measurement survives the byte path ``proxy_to_local`` uses
# ---------------------------------------------------------------------------


def test_progress_events_survive_arbitrary_stream_splits():
    """Every split boundary of a multi-event byte stream yields the same measurements."""
    stream = b"".join(shape[0] for shape in B8782_SHAPES) + CONTENT_CHUNK
    expected = [(shape[1], shape[2], shape[3], shape[4]) for shape in B8782_SHAPES]

    for split in range(1, len(stream)):
        framer = SSEEventReframer()
        parsed = [
            parse_prompt_progress_event(event) for event in framer.feed(stream[:split]) + framer.feed(stream[split:])
        ]
        measured = [(p.total, p.cache, p.processed, p.time_ms) for p in parsed if p is not None]
        assert measured == expected, f"split at byte {split} lost a measurement"


def test_capture_retains_the_latest_measurement_per_dispatch():
    """The capture exposes the newest event and counts only progress events."""
    first = parse_prompt_progress_event(FAKE_WARM_EVENT)
    capture = PromptProgressCapture()

    assert capture.observed is False
    assert capture.latest is None

    assert capture.observe(CONTENT_CHUNK) is None
    assert capture.events_seen == 0
    assert capture.observed is False

    assert capture.observe(FAKE_WARM_EVENT) == first
    assert capture.observed is True
    assert capture.events_seen == 1

    later = _event({"total": 56_000, "cache": 200, "processed": 8_192, "time_ms": 410})
    assert capture.observe(later) is not None
    assert capture.events_seen == 2
    assert capture.latest is not None
    assert (capture.latest.processed, capture.latest.time_ms) == (8_192, 410.0)
    # b8782 fixes ``cache`` before prefill, so the ratio cannot drift.
    assert capture.latest.cache == 200

    # A malformed event neither raises nor erases the last good measurement.
    assert capture.observe(b"data: {broken\n\n") is None
    assert capture.events_seen == 2
    assert (capture.latest.processed, capture.latest.time_ms) == (8_192, 410.0)

    # A further good event replaces the previous one (latest wins).
    capture.observe(_event({"total": 56_000, "cache": 55_000, "processed": 1_024, "time_ms": 80}))
    assert capture.events_seen == 3
    assert capture.latest is not None
    assert capture.latest.cache == 55_000


def test_capture_without_progress_stays_unmeasured():
    """A stream where ``return_progress`` was dropped leaves no measurement."""
    capture = PromptProgressCapture()

    for _ in range(3):
        assert capture.observe(CONTENT_CHUNK) is None

    assert capture.observed is False
    assert capture.latest is None
    assert capture.events_seen == 0
