"""Contract tests for the shared ``/slots`` GPU-telemetry fixtures.

LP-0MVT1NVAJ007KUGO (parent LP-0MVNWUEY8008XM6W). This is the test-first slice:
it ships the payload builders in ``gpu_fixtures.py`` and proves, against the
**real consumers** in ``proxy.proxy.observability``, that each fixture has the
shape and the failure behaviour its consumer expects. Sibling children
(sampler, recording rule, CLI) then start from a failing test instead of from
code.

Deliberately *not* tested here: the sampler's gauge wiring. That lands with
LP-0MVT1O0VG00905EL and would be a vacuous placeholder today.
"""

from __future__ import annotations

import proxy.observability as obs
import pytest
from gpu_fixtures import (
    FAILURE_MODES,
    LEAN_SLOT_KEYS,
    RICH_SLOT_KEYS,
    empty_slots_payload,
    expected_busy_slots,
    expected_in_flight_tasks,
    failing_slots_client,
    mixed_slots_payload,
    rich_slots_payload,
    slot,
    slots_payload,
    successful_slots_client,
)

URL = "http://localhost:8080/slots?model=mxbai-embed"
TIMEOUT = 0.05


def _fresh() -> None:
    """Drop the coalesced /slots cache so each case hits its own client."""
    obs.reset_slots_poll_cache()


# ---------------------------------------------------------------------------
# Healthy payload — shape and what _query_slots derives from it
# ---------------------------------------------------------------------------


def test_mixed_payload_is_a_lean_build_payload():
    """The healthy fixture has exactly the shape the deployed build returns.

    Verified live 2026-10-29: no ``id_task`` and no ``next_token``, even for a
    processing slot. A consumer that keys off ``id_task`` must therefore fall
    back to ``is_processing``.
    """
    payload = mixed_slots_payload(total=4, busy_ids=(0, 2))

    assert isinstance(payload, list)
    assert len(payload) == 4
    for index, entry in enumerate(payload):
        assert set(entry) == set(LEAN_SLOT_KEYS), (
            f"slot {index} keys {sorted(entry)} != lean build keys "
            f"{sorted(LEAN_SLOT_KEYS)}"
        )
        assert entry["id"] == index
        assert isinstance(entry["is_processing"], bool)
        assert entry["n_ctx"] == 256
    assert [s["is_processing"] for s in payload] == [True, False, True, False]
    assert expected_busy_slots(payload) == 2


@pytest.mark.asyncio
async def test_query_slots_derives_available_and_total_from_mixed_payload():
    """``_query_slots`` turns the healthy fixture into (available, total)."""
    _fresh()
    client = successful_slots_client(mixed_slots_payload(total=4, busy_ids=(0, 2)))
    status_out: dict = {}

    available, total = await obs._query_slots(
        client, 8080, timeout=TIMEOUT, model="mxbai-embed", status_out=status_out
    )

    assert (available, total) == (2, 4)
    # A definitive answer, not a failure: the sampler may publish these counts.
    assert status_out == {"status": 200, "reason": None}
    assert client.calls == [URL]


@pytest.mark.asyncio
async def test_query_slots_detail_derives_per_slot_state_from_mixed_payload():
    """``_query_slots_detail`` is the consumer of the lean per-slot state."""
    client = successful_slots_client(mixed_slots_payload(total=3, busy_ids=(1,)))

    details = await obs._query_slots_detail(
        llama_port=8080, timeout=TIMEOUT, model="mxbai-embed", _client=client
    )

    assert [(d["slot_id"], d["is_processing"]) for d in details] == [
        (0, False),
        (1, True),
        (2, False),
    ]
    # The deployed build publishes no next_token, so n_decoded/context_tokens
    # are honestly None rather than invented.
    assert all(d["n_decoded"] is None for d in details)
    assert all(d["context_tokens"] is None for d in details)


# ---------------------------------------------------------------------------
# Rich payload — id_task / next_token variant
# ---------------------------------------------------------------------------


def test_rich_payload_carries_id_task_and_next_token():
    """The rich fixture carries the extra fields the sampler prefers."""
    payload = rich_slots_payload()

    for entry in payload:
        assert set(entry) == set(RICH_SLOT_KEYS)
        assert isinstance(entry["id_task"], int)
        assert isinstance(entry["next_token"], dict)

    busy = [s for s in payload if s["is_processing"]]
    idle = [s for s in payload if not s["is_processing"]]
    # Busy slots carry a real task id; idle slots report the sentinel -1.
    assert {s["id_task"] for s in busy} == {1001, 1003}
    assert {s["id_task"] for s in idle} == {-1}
    assert expected_in_flight_tasks(payload) == 2
    assert expected_busy_slots(payload) == 2


def test_in_flight_count_falls_back_to_is_processing_on_lean_build():
    """``id_task`` is absent on the deployed build; the fallback still counts.

    This is the exact split the sampler sibling must implement, so the fixture
    pins both branches: rich payload → count of ``id_task >= 0``; lean payload
    → count of ``is_processing``.
    """
    lean = mixed_slots_payload(total=4, busy_ids=(0, 2))
    assert not any("id_task" in entry for entry in lean)
    assert expected_in_flight_tasks(lean) == 2

    rich = rich_slots_payload()
    # Card: an explicit -1 must NOT be counted even though the key exists.
    assert expected_in_flight_tasks(rich) == len(
        [s for s in rich if s["id_task"] >= 0]
    ) == 2


@pytest.mark.asyncio
async def test_rich_payload_next_token_feeds_n_decoded():
    """A rich payload makes ``_query_slots_detail`` report a real ``n_decoded``."""
    details = await obs._query_slots_detail(
        llama_port=8080,
        timeout=TIMEOUT,
        model="mxbai-embed",
        _client=successful_slots_client(rich_slots_payload()),
    )

    by_id = {d["slot_id"]: d for d in details}
    assert by_id[1]["is_processing"] is True
    assert by_id[1]["n_decoded"] == 5  # from next_token.n_decoded
    assert by_id[1]["context_tokens"] is None  # this build has no n_past
    assert by_id[0]["n_decoded"] == 0


def test_slot_builder_rejects_out_of_range_busy_ids():
    """A typo in a case definition fails loudly instead of silently idling."""
    with pytest.raises(ValueError, match="outside range"):
        slots_payload(total=2, busy_ids=(2,))


def test_slot_builder_rich_idle_defaults_are_the_sentinels():
    """An explicitly built idle rich slot has ``id_task = -1`` by default."""
    entry = slot(0, rich=True)
    assert entry["id_task"] == -1
    assert entry["next_token"] == {
        "has_next_token": False,
        "n_remain": -1,
        "n_decoded": 0,
    }


# ---------------------------------------------------------------------------
# Empty payload — a truthful "no slots", not a failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_payload_is_reported_as_a_definitive_empty_backend():
    """``[]`` with HTTP 200 is a real answer — the opposite of a failed fetch."""
    _fresh()
    client = successful_slots_client(empty_slots_payload())
    status_out: dict = {}

    available, total = await obs._query_slots(
        client, 8080, timeout=TIMEOUT, status_out=status_out
    )

    assert empty_slots_payload() == []
    assert (available, total) == (0, 0)
    assert status_out == {"status": 200, "reason": None}, (
        "an empty-but-successful payload must not be classified as a failure"
    )


# ---------------------------------------------------------------------------
# Failure modes — no fixture may look like "zero busy slots"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", FAILURE_MODES)
async def test_failure_fixtures_never_masquerade_as_an_empty_healthy_payload(mode):
    """Every failure mode is distinguishable from ``(200, [])``.

    The parent brief's trap: a failed ``/slots`` query must produce *no* gauge
    write, never ``llama_slots_busy 0``. The fetch layer must therefore never
    hand back a successful empty payload on failure — it either returns a
    non-200 status or raises.
    """
    _fresh()
    client = failing_slots_client(mode, timeout=TIMEOUT)

    try:
        status, data = await obs._fetch_slots_payload(URL, TIMEOUT, client)
    except (TimeoutError, ConnectionError):
        return  # transport failure: the caller sees an exception, not a payload
    assert status != 200, f"{mode} returned a 200 payload: {data!r}"
    assert data is None


@pytest.mark.asyncio
async def test_http_500_fixture_yields_a_definitive_failure_reason():
    """HTTP 500 (model reloading) → (0, 0) with ``reason="http_500"``."""
    _fresh()
    client = failing_slots_client("http_500")
    status_out: dict = {}

    counts = await obs._query_slots(client, 8080, timeout=TIMEOUT, status_out=status_out)

    assert counts == (0, 0)
    assert status_out == {"status": 500, "reason": "http_500"}
    assert not obs.slots_failure_is_definitive_model_absent(status_out)


@pytest.mark.asyncio
async def test_timeout_fixture_yields_a_timeout_reason():
    """A slow backend → ``reason="timeout"``, classified as transient."""
    _fresh()
    client = failing_slots_client("timeout", timeout=TIMEOUT)
    status_out: dict = {}

    counts = await obs._query_slots(client, 8080, timeout=TIMEOUT, status_out=status_out)

    assert counts == (0, 0)
    assert status_out == {"status": None, "reason": "timeout"}
    assert not obs.slots_failure_is_definitive_model_absent(status_out)


@pytest.mark.asyncio
async def test_connection_error_fixture_yields_a_connection_error_reason():
    """Nothing listening → ``reason="connection_error"``, classified as transient."""
    _fresh()
    client = failing_slots_client("connection_error")
    status_out: dict = {}

    counts = await obs._query_slots(client, 8080, timeout=TIMEOUT, status_out=status_out)

    assert counts == (0, 0)
    assert status_out == {"status": None, "reason": "connection_error"}
    assert not obs.slots_failure_is_definitive_model_absent(status_out)


@pytest.mark.asyncio
async def test_failure_reasons_map_to_the_existing_failure_counter():
    """A failed fetch still increments ``llama_slots_query_failures_total``.

    Reconciled by reason so the assertion holds when another test has already
    incremented the counter, and so a wrong label fails rather than passing.
    """
    from proxy import metrics as metrics_module

    metric = getattr(metrics_module, "llama_slots_query_failures_total", None)
    if metric is None:
        pytest.skip("prometheus_client not installed: counters are no-ops")

    before = metric.labels(reason="http_500")._value.get()
    _fresh()
    await obs._query_slots(failing_slots_client("http_500"), 8080, timeout=TIMEOUT)
    after = metric.labels(reason="http_500")._value.get()

    assert after == before + 1


def test_failing_client_factory_rejects_unknown_mode():
    """A typo'd failure mode is an error, not a silently healthy fixture."""
    with pytest.raises(ValueError, match="unknown failure mode"):
        failing_slots_client("http_503")
