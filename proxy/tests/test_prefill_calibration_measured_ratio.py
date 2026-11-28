"""Priced calibration window records the dispatch's MEASURED cache reuse.

LP-0MWRWBJPP001Q5JX (path (c) of the assessment in
``docs/dev/prompt-cache-tuning.md``).

The defect: ``record_dispatch_sample`` was fed the routing **prediction**
(``_cached_ratio``), which is ``0.0`` for a session with no cached-ratio entry.
A cross-session prompt-cache hit therefore entered the priced window as
``(coded=0, new=<whole prompt>, ms≈130)`` — a phantom-cold sample that halved
the fitted cold-prefill rate. The same dispatch's **measured** reuse is already
in scope at the recording site (``measured_prefill.get_measured_prefill``).

AC mapping:
- AC1  the window's ratio is the measured reuse when one exists, the prediction
       only when none does, and the sample's basis is recorded/attributable;
- AC2  a warm-labelled-cold dispatch (predicted 0.0, measured ≈0.9996, 7,185
       estimated tokens) enters the priced window as ~4 new tokens, not the
       whole prompt;
- AC3  a dispatch with no measurement keeps today's predicted ratio
       byte-for-byte (fail-open);
- AC4  a replay over a fixed fixture pins the direction and magnitude: the
       fitted cold-prefill rate from measured-ratio samples is materially
       higher than from predicted-ratio samples of the same dispatches;
- AC5  malformed / partial / raising measurements leave the sample exactly as
       today and never raise on the stream path;
- AC6  no admission price changes in this item (the window composition does).
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import proxy.measured_prefill as measured_prefill
import pytest
from fastapi.responses import StreamingResponse
from proxy.measured_prefill import MeasuredPrefill

from proxy import prefill_eta

MODEL = "Qwen3"
SESSION = "sess-prefill-calibration-measured-1"

#: A b8782 ``prompt_progress`` chunk (the shape the server emits).
_CONTENT_CHUNK = (
    b'data: {"choices":[{"delta":{"content":"hi"},"index":0}],'
    b'"created":0,"id":"chatcmpl-1","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)
_FINISH_CHUNK = (
    b'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],'
    b'"created":0,"id":"chatcmpl-finish","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)
_DONE = b"data: [DONE]\n\n"

#: The audited warm-labelled-cold dispatch (AC2): 7,185 estimated tokens,
#: predicted 0.0, measured 0.9996 reuse.
WARM_PROMPT_TOKENS = 7_185
WARM_MEASURED_CACHE = int(WARM_PROMPT_TOKENS * 0.9996)  # 7,182


def _progress_event(total: int, cache: int, processed: int = 0, time_ms: int = 0) -> bytes:
    payload = {
        "choices": [
            {
                "finish_reason": None,
                "index": 0,
                "delta": {"role": "assistant", "content": None},
            }
        ],
        "created": 0,
        "id": "chatcmpl-progress",
        "model": MODEL,
        "system_fingerprint": "b8782-e97492369",
        "object": "chat.completion.chunk",
        "prompt_progress": {
            "total": total,
            "cache": cache,
            "processed": processed,
            "time_ms": time_ms,
        },
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


@pytest.fixture(autouse=True)
def _clean_state():
    prefill_eta.reset()
    measured_prefill.reset_measured_prefill()
    yield
    prefill_eta.reset()
    measured_prefill.reset_measured_prefill()


def _window(model: str = MODEL, tokens: int = WARM_PROMPT_TOKENS):
    return prefill_eta._samples.get((model, prefill_eta.bucket_for(tokens)))


# ---------------------------------------------------------------------------
# AC2 — the warm-labelled-cold sample enters the window as a cache hit
# ---------------------------------------------------------------------------


def test_warm_labelled_cold_dispatch_records_almost_no_new_tokens():
    """AC2: predicted 0.0 / measured 0.9996 ⇒ ~4 new tokens, not 7,185."""
    measured = MeasuredPrefill(
        total=WARM_PROMPT_TOKENS,
        cache=WARM_MEASURED_CACHE,
        processed=WARM_PROMPT_TOKENS,
        time_ms=130.0,
    )
    assert measured.measured_cached_ratio == pytest.approx(0.9996, abs=1e-4)

    ratio, basis = prefill_eta.resolve_window_ratio(
        0.0, measured.measured_cached_ratio
    )
    assert basis == prefill_eta.WINDOW_RATIO_BASIS_MEASURED
    assert ratio == pytest.approx(measured.measured_cached_ratio)

    prefill_eta.record_dispatch_sample(
        MODEL, WARM_PROMPT_TOKENS, ratio, 130.0, ratio_basis=basis
    )

    window = _window()
    assert window is not None and len(window) == 1
    coded, new, ms = window[-1]
    assert ms == 130.0
    # ~4 new tokens (the exact count depends on the integer floor of the
    # cached term: 7,185 × 0.9996 = 7,182 already served).
    assert 0 < new <= 5, f"expected ~4 new tokens, got {new}"
    assert coded >= WARM_PROMPT_TOKENS - 5
    # …and the phantom-cold shape the defect produced is what it must NOT be.
    assert new != WARM_PROMPT_TOKENS


# ---------------------------------------------------------------------------
# AC3 — no measurement keeps the predicted ratio byte-for-byte
# ---------------------------------------------------------------------------


def test_no_measurement_keeps_the_predicted_ratio_byte_for_byte():
    """AC3: fail-open — an unmeasured dispatch is recorded exactly as today."""
    ratio, basis = prefill_eta.resolve_window_ratio(0.42, None)
    assert ratio == 0.42
    assert basis == prefill_eta.WINDOW_RATIO_BASIS_PREDICTED

    prefill_eta.record_dispatch_sample(
        MODEL, WARM_PROMPT_TOKENS, ratio, 130.0, ratio_basis=basis
    )
    _coded, new, ms = _window()[-1]
    assert ms == 130.0
    # The whole un-cached share is new — the pre-change behaviour exactly.
    assert new == WARM_PROMPT_TOKENS - int(WARM_PROMPT_TOKENS * 0.42)


def test_default_basis_is_the_predicted_one():
    """The kwarg default preserves every existing caller's behaviour."""
    prefill_eta.record_dispatch_sample(MODEL, 1_000, 0.0, 4_000.0)
    _coded, new, _ms = _window(tokens=1_000)[-1]
    assert new == 1_000


# ---------------------------------------------------------------------------
# AC5 — malformed / partial / raising measurements fall back, never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "measured",
    [None, float("nan"), float("inf"), float("-inf"), "not-a-number", object()],
)
def test_malformed_measurements_fall_back_to_the_prediction(measured):
    """AC5: any unusable measurement leaves the sample exactly as today."""
    ratio, basis = prefill_eta.resolve_window_ratio(0.25, measured)
    assert ratio == 0.25
    assert basis == prefill_eta.WINDOW_RATIO_BASIS_PREDICTED


def test_partial_measurement_without_a_ratio_is_ignored():
    """A record with ``total``/``cache`` missing reports ``None`` ⇒ prediction."""
    partial = MeasuredPrefill(total=None, cache=None, processed=10, time_ms=5.0)
    assert partial.measured_cached_ratio is None

    ratio, basis = prefill_eta.resolve_window_ratio(
        0.0, partial.measured_cached_ratio
    )
    assert (ratio, basis) == (0.0, prefill_eta.WINDOW_RATIO_BASIS_PREDICTED)


def test_resolve_never_raises_and_clamps_out_of_range_measurements():
    assert prefill_eta.resolve_window_ratio(0.1, 5.0) == (
        1.0,
        prefill_eta.WINDOW_RATIO_BASIS_MEASURED,
    )
    assert prefill_eta.resolve_window_ratio(0.1, -2.0) == (
        0.0,
        prefill_eta.WINDOW_RATIO_BASIS_MEASURED,
    )


def test_the_sample_basis_is_logged_for_attribution(caplog):
    """AC1: the promoted sample's basis is attributable from its log line."""
    with caplog.at_level(logging.DEBUG, logger="llama-proxy"):
        prefill_eta.record_dispatch_sample(
            MODEL, 8_000, 0.25, 1_000.0, ratio_basis="measured"
        )
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("prefill_calibration_sample")
    ]
    assert lines, "expected the prefill_calibration_sample debug line"
    assert "ratio_basis=measured" in lines[-1]
    assert "new=6000" in lines[-1]


# ---------------------------------------------------------------------------
# AC4 — replay: measured samples raise the fitted cold-prefill rate
# ---------------------------------------------------------------------------

#: A fixed fixture of the audited shape: six genuinely-cold dispatches at
#: 4.0 ms/token interleaved with six cross-session cache hits that the routing
#: prediction labelled cold (0.0) but llama-server served warm, in 130 ms.
_REPLAY_FIXTURE = [
    # (prompt_tokens, predicted_ratio, measured_ratio, service_ms)
    *[(8_000, 0.0, 0.0, 8_000 * 4.0)] * 6,
    *[(8_000, 0.0, 0.9996, 130.0)] * 6,
]


def _replay(use_measured: bool):
    prefill_eta.reset()
    for tokens, predicted, measured, service_ms in _REPLAY_FIXTURE:
        ratio = measured if use_measured else predicted
        prefill_eta.record_dispatch_sample(MODEL, tokens, ratio, service_ms)
    # 8,000 tokens is bucket 0 and below the conservative large-prompt floor,
    # so this reads the solitary window the samples were just written to.
    return prefill_eta.coefficients(MODEL, 8_000, {"server": {}})


def test_replay_measured_samples_materially_raise_the_fitted_cold_rate():
    """AC4: the phantom-cold samples halve the fitted rate; measured ones don't.

    Reference (2026-11-22 replay in ``docs/dev/prompt-cache-tuning.md``):
    predicted 1.987 ms/token vs measured 3.975 ms/token — the direction and
    the ≈2× gap are the claim, not the absolute milliseconds.
    """
    predicted = _replay(use_measured=False)
    measured = _replay(use_measured=True)

    assert predicted.calibrated is True
    assert measured.calibrated is True
    assert measured.prefill_ms_per_token > predicted.prefill_ms_per_token * 1.8, (
        "measured-ratio samples must fit a materially higher cold-prefill "
        f"rate (predicted={predicted.prefill_ms_per_token:.3f}, "
        f"measured={measured.prefill_ms_per_token:.3f})"
    )
    # The measured fit recovers the true 4.0 ms/token within the conservatism
    # factor (×1.2), while the predicted fit is dragged towards the warm
    # service time.
    assert measured.prefill_ms_per_token == pytest.approx(4.0 * 1.2, rel=0.05)
    assert predicted.prefill_ms_per_token < 3.0


# ---------------------------------------------------------------------------
# Router integration harness (AC1/AC2/AC3 on the real first-byte path)
# ---------------------------------------------------------------------------


def _mock_server():
    srv = MagicMock()
    srv.config = {
        "server": {
            "max_concurrent_queries": 4,
            "session_single_flight_mode": "queue",
            "session_single_flight_max_queue_depth": 1,
        }
    }
    srv.active_queries = 0
    srv.local_active_queries = 0
    srv.backend_ready = True
    srv.llama_process = MagicMock()
    srv.current_model = MODEL
    srv.logger = MagicMock()
    srv.session_manager = AsyncMock()
    srv.local_dispatch_records = {}
    srv.local_dispatch_records_lock = AsyncMock()
    srv.local_prefill_in_flight = {}
    srv.local_prefill_in_flight_lock = AsyncMock()
    return srv


def _mock_request(body: dict):
    req = MagicMock()
    req.method = "POST"
    req.headers = {"host": "localhost"}
    req.url = type("U", (), {"path": "/v1/chat/completions"})()
    req.body = AsyncMock(return_value=json.dumps(body).encode())
    req.is_disconnected = AsyncMock(return_value=False)
    req._local_admission_evidence = {
        "estimated_tokens": WARM_PROMPT_TOKENS,
        "cached_ratio": 0.0,
        "new_tokens": WARM_PROMPT_TOKENS,
    }
    return req


class _StreamResponse:
    def __init__(self, chunks: list[bytes]):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self.content = b""
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b""


class _CM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


async def _drive_local_stream(
    chunks: list[bytes],
    *,
    predicted_ratio: float,
    estimated_tokens: int = WARM_PROMPT_TOKENS,
):
    """Run the real ``proxy_to_local`` first-byte path; return ``(srv, received)``."""
    import proxy.router as router
    import proxy.router_helpers as helpers

    srv = _mock_server()
    srv.local_dispatch_records[SESSION] = {"active": True}
    response = _StreamResponse(chunks)
    cm = _CM(response)
    client = MagicMock()
    client.stream = MagicMock(return_value=cm)
    client.aclose = AsyncMock()

    async def _session(*args, **_kwargs):
        body_json = args[1] if len(args) > 1 else {}
        return {
            "session_id": SESSION,
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": None,
            "body_override": None,
            "body_json": body_json,
            "original_message_count": 1,
            "session_explicit": False,
        }

    async def _retry(factory, **_kwargs):
        return await factory()

    stack = ExitStack()
    patches = [
        patch("proxy.router._srv", return_value=srv),
        patch("proxy.router_helpers._srv", return_value=srv),
        # The routing prediction is 0.0 for a session with no cached-ratio
        # entry — the audited warm-labelled-cold scenario.
        patch.object(
            helpers,
            "_lookup_dispatch_cache_state",
            MagicMock(return_value=("cold", predicted_ratio)),
        ),
        patch("proxy.router._is_self_healing_active", return_value=False),
        patch("proxy.router._check_slot_availability", return_value=None),
        patch("proxy.router._handle_session", side_effect=_session),
        patch("proxy.router._build_slot_context", return_value=(None, None, 3.0)),
        patch("proxy.router.httpx.AsyncClient", return_value=client),
        patch("proxy.router._schedule_token_increment"),
        patch("proxy.router._schedule_recv_token_increment"),
        patch("proxy.router._schedule_traffic_recording"),
        patch("proxy.router.count_text_tokens", return_value=5),
        patch("proxy.router._extract_delta_text_from_sse_chunk", return_value=""),
        patch("proxy.router.evaluate_stream_guardrail", return_value=None),
        patch("proxy.router.log_request"),
        patch(
            "proxy.router._detect_restore_signal_from_llama_log", return_value=False
        ),
        patch("proxy.session._resolve_log_path", return_value=Path("/nonexistent")),
        patch("proxy.router._update_session_and_slot", new=AsyncMock()),
        patch("proxy.router._cleanup_after_request", new=AsyncMock()),
        patch("proxy.router._increment_local_active_queries", new=AsyncMock()),
        patch("proxy.router._increment_active_queries", new=AsyncMock()),
        patch("proxy.router._increment_per_model_query", new=AsyncMock()),
        patch("proxy.router_helpers._decrement_generating_only_slot", new=AsyncMock()),
        patch("proxy.router_helpers._increment_generating_only_slot", new=AsyncMock()),
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=estimated_tokens),
        ),
        patch("proxy.router._call_with_backend_retries", side_effect=_retry),
    ]
    try:
        for p in patches:
            stack.enter_context(p)
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = await router.proxy_to_local(_mock_request(body), "v1/chat/completions")
        assert isinstance(result, StreamingResponse)
        received = b""
        async for chunk in result.body_iterator:
            received += chunk
    finally:
        stack.close()
    return srv, received


def _info_lines(srv) -> list[str]:
    lines: list[str] = []
    for call in srv.logger.info.call_args_list:
        template = str(call.args[0]) if call.args else ""
        args = call.args[1:]
        if args and "%" in template:
            try:
                lines.append(template % tuple(args))
            except (TypeError, ValueError):
                pass
    return lines


@pytest.mark.asyncio
async def test_router_feeds_the_measured_reuse_to_the_priced_window():
    """AC1/AC2: the real first-byte path records the measured ratio."""
    _srv, received = await _drive_local_stream(
        [_progress_event(WARM_PROMPT_TOKENS, WARM_MEASURED_CACHE),
         _CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        predicted_ratio=0.0,
    )
    # The dispatch completed normally (a genuinely warm measurement never
    # trips the cache-deficit abort).
    assert _FINISH_CHUNK in received

    window = _window()
    assert window is not None and len(window) == 1, "expected one priced sample"
    _coded, new, _ms = window[-1]
    assert 0 < new <= 5, f"expected ~4 new tokens from the measured reuse, got {new}"


@pytest.mark.asyncio
async def test_router_records_the_basis_on_the_dispatch_line():
    """AC1: the sample's basis is attributable from the dispatch log line."""
    srv, _received = await _drive_local_stream(
        [_progress_event(WARM_PROMPT_TOKENS, WARM_MEASURED_CACHE),
         _CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        predicted_ratio=0.0,
    )
    dispatch = [
        line for line in _info_lines(srv) if line.startswith("dispatch_first_byte_ms=")
    ]
    assert dispatch, "expected the dispatch_first_byte_ms line"
    assert "window_ratio_basis=measured" in dispatch[-1]
    assert "window_cached_ratio=" in dispatch[-1]


@pytest.mark.asyncio
async def test_router_without_a_measurement_keeps_the_prediction():
    """AC3: no prompt_progress ⇒ the whole un-cached prompt is recorded."""
    srv, _received = await _drive_local_stream(
        [_CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        predicted_ratio=0.0,
    )

    window = _window()
    assert window is not None and len(window) == 1
    _coded, new, _ms = window[-1]
    assert new == WARM_PROMPT_TOKENS, (
        f"an unmeasured dispatch must keep the predicted ratio (got {new})"
    )
    dispatch = [
        line for line in _info_lines(srv) if line.startswith("dispatch_first_byte_ms=")
    ]
    assert dispatch and "window_ratio_basis=predicted" in dispatch[-1]


@pytest.mark.asyncio
async def test_router_survives_a_raising_measurement_lookup(monkeypatch):
    """AC5: a raising measurement read leaves today's sample and the stream intact."""
    import proxy.measured_prefill as mp

    monkeypatch.setattr(
        mp,
        "get_measured_prefill",
        MagicMock(side_effect=RuntimeError("measurement store exploded")),
    )
    _srv, received = await _drive_local_stream(
        [_progress_event(WARM_PROMPT_TOKENS, WARM_MEASURED_CACHE),
         _CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        predicted_ratio=0.0,
    )
    assert _FINISH_CHUNK in received and _DONE in received

    window = _window()
    assert window is not None and len(window) == 1
    _coded, new, _ms = window[-1]
    assert new == WARM_PROMPT_TOKENS
