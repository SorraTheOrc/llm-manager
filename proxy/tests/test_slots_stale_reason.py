"""Enumerated, machine-readable ``slots_stale_reason`` (LP-0MW1VVIPG008896N).

Slice 2 of LP-0MW1VDWY0007EMH7. ``/llama/local/status`` published
``slots_stale`` as a bare boolean, so during the 2026-11-04 freeze an operator
could only learn *why* by decoding counter names in ``/metrics``
(``llama_slots_child_port_fallbacks_total{reason="no_child_port_router_mode"}``,
``llama_slots_query_failures_total{reason="timeout"}``, ``…{reason="unknown"}``).
The status payload itself carried no cause, so every consumer could only fail
closed blindly.

This module pins the contract:

* one canonical vocabulary constant, so no bare string literals leak to call
  sites and a test can iterate the whole set;
* every member is produced by a covered code path (exhaustiveness);
* a documented precedence — the detail phase dominates the counts phase
  (detail is what capacity consumes), and an over-age snapshot dominates every
  transport-level reason;
* fail-open: a recorder fault degrades to ``None`` and never breaks the
  payload;
* the existing metric label vocabulary is *unchanged*, so
  ``monitoring/slots_query_alerts.yaml`` keeps matching.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from unittest.mock import patch

import httpx
import pytest

from proxy import observability as obs

# ======================================================================
# The vocabulary
# ======================================================================


class TestVocabulary:
    """One canonical constant is the single source of truth."""

    def test_vocabulary_is_the_documented_five(self):
        """Exactly the five reasons the work item enumerates."""
        assert obs.SLOTS_STALE_REASONS == (
            "router_model_missing",
            "child_port_undiscovered",
            "query_timeout",
            "query_error",
            "snapshot_too_old",
        )

    def test_vocabulary_is_immutable_and_iterable(self):
        """A tuple: iteration order is stable and nothing can append to it."""
        assert isinstance(obs.SLOTS_STALE_REASONS, tuple)
        assert len(set(obs.SLOTS_STALE_REASONS)) == len(obs.SLOTS_STALE_REASONS)

    def test_metric_label_vocabulary_is_unchanged(self):
        """The /metrics labels keep their spelling (alert rules depend on them).

        ``slots_stale_reason`` is an additive status-payload field; it must not
        rename the counter labels that
        ``monitoring/slots_query_alerts.yaml`` matches on.
        """
        for label in (
            "timeout",
            "connection_error",
            "invalid_payload",
            "unknown",
            "http_400",
            "http_500",
        ):
            reason = obs.classify_slots_failure_reason(label)
            assert reason in obs.SLOTS_STALE_REASONS, (
                f"metric label {label!r} did not map onto the vocabulary"
            )


class TestClassification:
    """Existing failure labels map deterministically onto the vocabulary."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("timeout", "query_timeout"),
            ("connection_error", "query_error"),
            ("invalid_payload", "query_error"),
            ("unknown", "query_error"),
            ("http_500", "query_error"),
            ("http_502", "query_error"),
            ("http_404", "query_error"),
        ],
    )
    def test_transport_labels_map(self, label, expected):
        assert obs.classify_slots_failure_reason(label) == expected

    @pytest.mark.parametrize("label", [None, "", "not_a_label"])
    def test_unrecognised_labels_are_query_error(self, label):
        """An unknown label is still an error, never a silent ``None``."""
        assert obs.classify_slots_failure_reason(label) == "query_error"

    def test_query_timeout_only_for_timeout(self):
        assert (
            obs.classify_slots_failure_reason("timeout") == "query_timeout"
        )
        assert "query_timeout" not in {
            obs.classify_slots_failure_reason("http_500"),
            obs.classify_slots_failure_reason("invalid_payload"),
        }


class TestPrecedence:
    """The documented precedence between phases and over-age snapshots."""

    def test_detail_phase_dominates_counts_phase(self):
        """Detail is what capacity consumes, so its failure is reported first."""
        assert obs.select_slots_stale_reason(
            detail_reason="query_error",
            counts_reason="query_timeout",
        ) == "query_error"
        assert obs.select_slots_stale_reason(
            detail_reason="query_timeout",
            counts_reason="query_error",
        ) == "query_timeout"

    def test_counts_phase_reported_when_detail_is_healthy(self):
        assert obs.select_slots_stale_reason(
            detail_reason=None, counts_reason="query_timeout"
        ) == "query_timeout"

    def test_snapshot_too_old_dominates_every_transport_reason(self):
        for detail in (None, "query_error", "query_timeout"):
            for counts in (None, "query_error"):
                assert obs.select_slots_stale_reason(
                    detail_reason=detail,
                    counts_reason=counts,
                    snapshot_too_old=True,
                ) == "snapshot_too_old"

    def test_healthy_phases_yield_no_reason(self):
        assert obs.select_slots_stale_reason(
            detail_reason=None, counts_reason=None
        ) is None
        assert obs.select_slots_stale_reason(
            detail_reason=None, counts_reason=None, snapshot_too_old=False
        ) is None

    def test_unknown_values_are_rejected_to_query_error(self):
        """A reason outside the vocabulary is not published verbatim."""
        assert obs.select_slots_stale_reason(
            detail_reason="something_new", counts_reason=None
        ) == "query_error"


class TestExhaustiveness:
    """Every vocabulary member is reachable from a covered code path."""

    def test_every_member_is_produced(self):
        produced = {
            obs.classify_slots_failure_reason("timeout"),
            obs.classify_slots_failure_reason("http_500"),
            obs.SLOTS_STALE_REASON_ROUTER_MODEL_MISSING,
            obs.SLOTS_STALE_REASON_CHILD_PORT_UNDISCOVERED,
            obs.select_slots_stale_reason(
                detail_reason=None,
                counts_reason=None,
                snapshot_too_old=True,
            ),
        }
        assert produced == set(obs.SLOTS_STALE_REASONS), (
            "a vocabulary member has no covered code path"
        )

    @pytest.mark.asyncio
    async def test_query_timeout_and_query_error_from_real_fetches(self):
        """The two transport reasons come out of the real fetch paths."""

        class _TimeoutClient:
            async def get(self, url, **kwargs):
                raise TimeoutError()

        class _ErrorClient:
            async def get(self, url, **kwargs):
                return _Resp(500)

        obs.reset_slots_stale_reason()
        await obs._query_slots(_TimeoutClient(), 8080, timeout=0.1, model="m")
        assert obs.last_slots_stale_reason() == "query_timeout"

        obs.reset_slots_stale_reason()
        await obs._query_slots(_ErrorClient(), 8080, timeout=0.1, model="m")
        assert obs.last_slots_stale_reason() == "query_error"


# ======================================================================
# Recorder behaviour
# ======================================================================


class TestRecorder:
    """The in-process recorder is fail-open and reset-friendly."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        obs.reset_slots_stale_reason()
        yield
        obs.reset_slots_stale_reason()

    def test_records_and_reads_back(self):
        obs.record_slots_stale_reason("query_timeout")
        assert obs.last_slots_stale_reason() == "query_timeout"

    def test_none_clears(self):
        obs.record_slots_stale_reason("query_timeout")
        obs.record_slots_stale_reason(None)
        assert obs.last_slots_stale_reason() is None

    def test_invalid_reason_is_not_recorded(self):
        obs.record_slots_stale_reason("query_timeout")
        obs.record_slots_stale_reason("bogus")
        assert obs.last_slots_stale_reason() == "query_timeout"

    def test_recorder_fault_never_raises(self, monkeypatch):
        """A recorder fault degrades to ``None``; it never breaks the caller."""

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(obs, "_STALE_REASON_LOCK", _boom)
        # Neither writer nor reader may propagate.
        obs.record_slots_stale_reason("query_timeout")
        assert obs.last_slots_stale_reason() in (
            None,
            "query_timeout",
        )


class TestDefinitiveModelAbsent:
    """A definitive 400/404 is not a staleness reason (it is fresh truth)."""

    def test_model_absent_is_not_classified_as_stale(self):
        assert obs.classify_slots_failure_reason("http_400") == "query_error"
        # ...but the handler's definitive-absent path takes precedence: it
        # never serves cache, so it publishes no stale reason at all.
        assert obs.slots_failure_is_definitive_model_absent(
            {"status": 400, "reason": "http_400"}
        )


# ======================================================================
# Payload integration
# ======================================================================


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    async def json(self):
        return self._payload


class _NoopLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


class _FailingClient:
    def __init__(self, mode="error"):
        self.mode = mode

    # httpx.AsyncClient is an async context manager and ``_do_slots_fetch``
    # opens it as one on the detail path (which builds its own client), so the
    # shim must support the protocol or the detail read fails with a TypeError
    # instead of the simulated transport error (LP-0MW3ED73W007JYGI).
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kwargs):
        if self.mode == "timeout":
            raise TimeoutError()
        return _Resp(500)


class _HttpxShim:
    def __init__(self, client):
        self.AsyncClient = lambda **kwargs: client

    def __getattr__(self, name):
        return getattr(httpx, name)


async def _get_status(client, resolver=(None, 41395), router_mode=True):
    from proxy.server import app

    from proxy import server

    async def fake_query_llama_status():
        return {"llama_server_running": True}

    transport = httpx.ASGITransport(app=app)
    config = {
        "server": {
            "llama_server_port": 8080,
            "llama_router_mode": router_mode,
        }
    }
    with patch(
        "proxy.server.query_llama_status", side_effect=fake_query_llama_status
    ):
        with patch.object(server, "current_model", "Qwen3"):
            with patch.object(server, "model_switch_refcount", 0):
                with patch.object(server, "model_switch_lock", _NoopLock()):
                    with patch.object(server, "background_loads", {}):
                        with patch.object(server, "local_dispatch_records", {}):
                            with patch.object(
                                server, "local_dispatch_records_lock", _NoopLock()
                            ):
                                with patch.object(server, "config", config):
                                    with patch.object(
                                        server, "_http_client", client
                                    ):
                                        with patch(
                                            "proxy.router_helpers"
                                            "._resolve_local_slots_target",
                                            lambda *a, **k: resolver,
                                        ):
                                            with patch.object(
                                                obs, "httpx", _HttpxShim(client)
                                            ):
                                                async with httpx.AsyncClient(
                                                    transport=transport,
                                                    base_url="http://test",
                                                ) as ac:
                                                    resp = await ac.get(
                                                        "/llama/local/status"
                                                    )
    assert resp.status_code == 200
    return resp.json()


class TestPayload:
    """``/llama/local/status`` publishes ``slots_stale_reason`` correctly."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        obs.reset_slots_poll_cache()
        obs.reset_slots_stale_reason()
        obs._last_slot_counts_cache = None
        obs._last_slot_details_cache[:] = []
        obs._last_slot_details_at = 0.0
        yield
        obs.reset_slots_poll_cache()
        obs.reset_slots_stale_reason()
        obs._last_slot_counts_cache = None
        obs._last_slot_details_cache[:] = []

    @pytest.mark.asyncio
    async def test_fresh_payload_has_null_reason(self):
        class _Ok:
            async def get(self, url, **kwargs):
                return _Resp(200, [{"id": 0, "is_processing": False}])

        payload = await _get_status(_Ok())
        assert payload["slots_stale"] is False
        assert payload["slots_stale_reason"] is None
        assert payload["status_schema_version"] == 7

    @pytest.mark.asyncio
    async def test_query_error_is_published(self):
        """The 2026-11-04 shape: a failed fetch served from the fossil cache."""
        obs._last_slot_details_cache[:] = [
            {"slot_id": 0, "is_processing": False, "n_decoded": None}
        ]
        payload = await _get_status(_FailingClient("error"))
        assert payload["slots_stale"] is True
        assert payload["slots_stale_reason"] == "query_error"

    @pytest.mark.asyncio
    async def test_query_timeout_is_published(self):
        obs._last_slot_details_cache[:] = [
            {"slot_id": 0, "is_processing": False, "n_decoded": None}
        ]
        payload = await _get_status(_FailingClient("timeout"))
        assert payload["slots_stale"] is True
        assert payload["slots_stale_reason"] == "query_timeout"

    @pytest.mark.asyncio
    async def test_reason_published_even_with_no_cache_to_serve(self):
        """Diagnosability does not depend on there being a fossil to serve.

        With no cached snapshot ``slots_stale`` stays false (nothing stale is
        being *served* — the counts are simply 0/0 and consumers already see
        no capacity), but the cause is still published so an operator is not
        left guessing.
        """
        payload = await _get_status(_FailingClient("error"))
        assert payload["slots_stale"] is False
        assert payload["slots_stale_reason"] == "query_error"

    @pytest.mark.asyncio
    async def test_child_port_undiscovered_is_published(self):
        """The router-port fallback (no child port) is its own reason."""
        payload = await _get_status(
            _FailingClient("error"), resolver=(None, 8080)
        )
        assert payload["slots_stale_reason"] == "child_port_undiscovered"

    @pytest.mark.asyncio
    async def test_router_model_missing_when_last_resort_rejected(self):
        """Model resolution fell through to the last resort and was rejected."""

        class _RejectsEverything:
            async def get(self, url, **kwargs):
                return _Resp(
                    400,
                    {"error": {"code": 400, "message": "model not found"}},
                )

        with patch.object(
            obs, "_resolve_slots_model_id", lambda model=None: "Qwen3"
        ), patch(
            "proxy.router_helpers.resolve_local_model_id",
            lambda srv, model=None: ("Qwen3", "last_resort"),
        ):
            payload = await _get_status(_RejectsEverything())
        assert payload["slots_stale_reason"] == "router_model_missing"

    @pytest.mark.asyncio
    async def test_reason_is_not_published_when_fresh(self):
        """A healthy poll never carries a leftover reason from an earlier poll."""
        class _Ok:
            async def get(self, url, **kwargs):
                return _Resp(200, [{"id": 0, "is_processing": False}])

        await _get_status(_FailingClient("error"))
        obs.reset_slots_poll_cache()
        payload = await _get_status(_Ok())
        assert payload["slots_stale"] is False
        assert payload["slots_stale_reason"] is None


class TestStatusRequestLogLine:
    """The reason is also on the ``status_request`` log line."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        obs.reset_slots_poll_cache()
        obs.reset_slots_stale_reason()
        obs._last_slot_counts_cache = None
        yield
        obs.reset_slots_poll_cache()
        obs.reset_slots_stale_reason()
        obs._last_slot_counts_cache = None

    @pytest.mark.asyncio
    async def test_reason_appears_in_log_extras(self):
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                if record.getMessage() == "status_request":
                    records.append(record)

        logger = logging.getLogger("llama-proxy")
        previous = logger.level
        logger.setLevel(logging.INFO)
        handler = _Capture()
        logger.addHandler(handler)
        obs._last_slot_details_cache[:] = [
            {"slot_id": 0, "is_processing": False, "n_decoded": None}
        ]
        try:
            await _get_status(_FailingClient("error"))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

        assert records, "no status_request line was logged"
        assert records[-1].slots_stale_reason == "query_error"
