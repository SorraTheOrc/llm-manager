"""Builder-level retry contract for every client-facing admission refusal.

LP-0MVUXY1CL005KWCC AC3. The reason-aware retry contract
(LP-0MVNTT4X8009IO2X) is pinned end-to-end elsewhere
(``test_local_only_refusal_matrix.py``, ``test_local_non_admission_retry_timing.py``);
this module pins the **builder-level** invariant that made the slot-exhaustion
``429`` an outlier: no client-facing admission-refusal builder may silently
omit *both* a truthful timer and an explicit non-retryable verdict.

Each row below is one (builder x reason) pair and states which of the two
contracts the refusal must honour:

- **timer** — ``Retry-After`` header, body ``retry_after`` (the same value),
  ``retry_at`` (RFC 3339 UTC) and ``retryable: true``; the value is >= 1 s and
  capped by ``server.max_retry_after_seconds``;
- **no timer** — ``retryable: false`` plus compaction guidance, and *no*
  ``Retry-After`` at all (a delay header on a request waiting cannot heal is
  what invited the 2026-10-25 retry storm).

The shipped per-reason rows (``eta_exceeds_budget``,
``eta_exceeds_budget_direct``, ``prefill_eta_exceeds_deadline``,
``prefill_eta_exceeds_safety_budget``, ``prefill_budget_exceeded``,
``context_too_large``) are pinned here so they cannot regress, and the
slot-exhaustion row is the one this change added.

The chain-hold regression at the end proves the slot ``429``'s advertised
``retry_after`` now bounds the hold (``_bounded_hold_seconds``) instead of
leaving ``_exhaustion_retry_after`` to return ``None`` and fall back to the
legacy full ``chain_hold_seconds`` hold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response

#: Cap used by the contract rows; wide enough that the ETA rows are not clamped.
CONFIG: dict = {"server": {"max_retry_after_seconds": 3600}}

#: ETA/budget pair the ETA-derived rows advertise: ceil(60 - 10) = 50 s.
ETA_SECONDS = 60.0
BUDGET_SECONDS = 10.0
ETA_RETRY_AFTER = 50

SLOT_CONFIG: dict = {
    "server": {
        "max_retry_after_seconds": 3600,
        "slot_unavailable_retry_after": 7,
    }
}


@dataclass(frozen=True)
class _Row:
    """One (builder x reason) row of the refusal retry contract."""

    builder: str
    reason: str | None
    #: Expected ``Retry-After`` when the row must carry a timer, else ``None``.
    retry_after: int | None


ROWS: tuple[_Row, ...] = tuple(
    _Row(builder, reason, retry_after)
    for builder in ("admission_reject", "local_only_refusal")
    for reason, retry_after in (
        ("eta_exceeds_budget", ETA_RETRY_AFTER),
        ("eta_exceeds_budget_direct", ETA_RETRY_AFTER),
        ("prefill_eta_exceeds_deadline", ETA_RETRY_AFTER),
        ("prefill_eta_exceeds_safety_budget", ETA_RETRY_AFTER),
        ("prefill_budget_exceeded", None),
        ("context_too_large", None),
    )
) + (_Row("slot_exhaustion", None, 7),)


def _attempt(reason: str | None) -> dict:
    return {
        "provider": "local-llama",
        "reason": reason,
        "eta_prefill_seconds": ETA_SECONDS,
        "budget_seconds": BUDGET_SECONDS,
    }


def _build(row: _Row):
    """Build the refusal for *row* through its public builder."""
    if row.builder == "slot_exhaustion":
        # Deterministic: no live occupant is measurable, so the configured
        # ``server.slot_unavailable_retry_after`` fallback owns the timer.
        with patch.object(
            provider, "_local_occupant_prefill_seconds", return_value=0.0
        ):
            return provider._build_exhausted_response(
                all_local_slot_exhaustion=True,
                total_slots=4,
                config=SLOT_CONFIG,
            )
    if row.builder == "admission_reject":
        return provider._build_local_admission_reject_response(
            _attempt(row.reason), CONFIG
        )
    if row.builder == "local_only_refusal":
        return provider._build_local_only_refusal_response(
            _attempt(row.reason), CONFIG
        )
    raise AssertionError(f"unknown builder {row.builder!r}")


def _assert_truthful_timer(response, expected: int) -> None:
    """The refusal answers *when* to retry, consistently on body and header."""
    body = json.loads(response.body)
    assert body["retryable"] is True
    assert body["retry_after"] == expected
    assert expected >= 1
    assert expected <= provider._max_retry_after_seconds(CONFIG)
    assert response.headers["retry-after"] == str(expected)
    retry_at = datetime.fromisoformat(body["retry_at"])
    assert retry_at.utcoffset() == timedelta(0), "retry_at must be UTC"
    advertised = (retry_at - datetime.now(UTC)).total_seconds()
    assert abs(advertised - expected) < 5


def _assert_no_timer(response) -> None:
    """The refusal says waiting cannot help, and offers the remedy instead."""
    body = json.loads(response.body)
    assert body["retryable"] is False
    assert body["retry_after"] is None
    assert "retry_at" not in body
    assert "retry-after" not in response.headers
    assert body["suggested_action"] == provider._COMPACTION_GUIDANCE


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.builder}-{r.reason}")
def test_refusal_builder_never_omits_the_retry_contract(row: _Row):
    """Every refusal is either truthfully timed or explicitly not retryable."""
    response = _build(row)

    assert response.status_code == 429, "an admission refusal is a 429"
    if row.retry_after is None:
        _assert_no_timer(response)
    else:
        _assert_truthful_timer(response, row.retry_after)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.builder}-{r.reason}")
def test_refusal_builder_keeps_its_stable_code(row: _Row):
    """The additive timer never disturbs the client-classifying ``code``."""
    body = json.loads(_build(row).body)
    expected = (
        provider.EXHAUSTION_CODE_SLOTS
        if row.builder == "slot_exhaustion"
        else provider.EXHAUSTION_CODE_LOCAL_ADMISSION
    )
    assert body["code"] == expected


def test_slot_exhaustion_keeps_its_legacy_body_keys_and_values():
    """Additive only: the pre-existing keys keep their exact values."""
    response = provider._build_exhausted_response(
        all_local_slot_exhaustion=True, total_slots=4, config=SLOT_CONFIG
    )
    body = json.loads(response.body)
    assert body["error"] == "All providers exhausted"
    assert body["code"] == "all_slots_exhausted"
    assert body["message"] == (
        "Model server busy: 0/4 slots available. Retry later."
    )


def test_slot_exhaustion_advertises_the_measured_occupant_service():
    """A measured remainder owns the timer, rounded up (never optimistic)."""
    with patch.object(
        provider, "_local_occupant_prefill_seconds", return_value=42.5
    ):
        response = provider._build_exhausted_response(
            all_local_slot_exhaustion=True, total_slots=4, config=SLOT_CONFIG
        )
    body = json.loads(response.body)
    assert body["retry_after"] == 43, "the measured wait must round up"
    assert response.headers["retry-after"] == "43"
    assert body["retryable"] is True
    assert "retry_at" in body


def test_slot_exhaustion_falls_back_to_the_slot_unavailable_knob():
    """With no measurement the sibling router ``503``'s knob is advertised."""
    with patch.object(
        provider, "_local_occupant_prefill_seconds", return_value=0.0
    ):
        response = provider._build_exhausted_response(
            all_local_slot_exhaustion=True, total_slots=4, config=SLOT_CONFIG
        )
    assert json.loads(response.body)["retry_after"] == 7
    assert provider._get_slot_unavailable_retry_after(SLOT_CONFIG) == 7.0


def test_slot_exhaustion_defaults_to_five_seconds_without_config():
    """The documented default (5 s) applies when the knob is unset."""
    with patch.object(
        provider, "_local_occupant_prefill_seconds", return_value=0.0
    ):
        response = provider._build_exhausted_response(
            all_local_slot_exhaustion=True, total_slots=2
        )
    assert json.loads(response.body)["retry_after"] == 5
    assert response.headers["retry-after"] == "5"


def test_slot_exhaustion_timer_is_capped():
    """A long measured occupant is clamped by ``max_retry_after_seconds``."""
    capped = {"server": {"max_retry_after_seconds": 60}}
    with patch.object(
        provider, "_local_occupant_prefill_seconds", return_value=9999.0
    ):
        response = provider._build_exhausted_response(
            all_local_slot_exhaustion=True, total_slots=1, config=capped
        )
    body = json.loads(response.body)
    assert body["retry_after"] == 60
    assert response.headers["retry-after"] == "60"


def test_slot_exhaustion_timer_is_floored_at_one_second():
    """A sub-second remainder still advertises the 1 s floor."""
    with patch.object(
        provider, "_local_occupant_prefill_seconds", return_value=0.2
    ):
        response = provider._build_exhausted_response(
            all_local_slot_exhaustion=True, total_slots=1, config=SLOT_CONFIG
        )
    assert json.loads(response.body)["retry_after"] == 1


class _Request:
    """Minimal request stub for the chain cycle."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


@pytest.fixture
def _clean_provider_state():
    provider._provider_unavailable_until.clear()
    provider._usage_reset_at.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._usage_reset_at.clear()


@pytest.mark.asyncio
async def test_remote_429_chain_reports_generic_exhaustion_not_slot_exhaustion(
    _clean_provider_state,
):
    """The slot branch is reserved for *local* slot pressure.

    LP-0MVUXY1CL005KWCC: a chain whose only provider is remote and answers a
    bare quota ``429`` is a rate limit, not a busy local slot. It must end in
    the generic ``all_exhausted`` 503 (with the provider cooldown as its
    ``retry_after``), never in the ``all_slots_exhausted`` 429 whose message
    and timer describe local slots.
    """
    model_config = {
        "providers": [
            {
                "name": "remote-quota",
                "type": "remote",
                "endpoint": "https://example.invalid/v1",
            }
        ]
    }

    async def _quota_429(_req, _path, _cfg):
        return Response(
            content=(
                b'{"code":"rate_limit_exceeded",'
                b'"type":"rate_limit_error"}'
            ),
            status_code=429,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _quota_429):
        result = await provider.proxy_with_remote_fallback(
            _Request(),
            "v1/chat/completions",
            model_config,
            {"provider_cooldown_seconds": 30},
        )

    body = json.loads(result.body)
    assert result.status_code == 503
    assert body["code"] == provider.EXHAUSTION_CODE_ALL
    assert body["retry_after"] >= 1
    assert "slots available" not in json.dumps(body)


def test_slot_429_timer_bounds_the_chain_hold():
    """AC3: the slot ``429`` timer now bounds ``_bounded_hold_seconds``.

    Before this change ``_exhaustion_retry_after`` returned ``None`` for the
    timer-free slot ``429``, so the chain preserved the legacy full
    ``chain_hold_seconds`` hold. With the timer present the hold is
    ``min(chain_hold_seconds, retry_after)``.
    """
    with patch.object(
        provider, "_local_occupant_prefill_seconds", return_value=0.0
    ):
        response = provider._build_exhausted_response(
            all_local_slot_exhaustion=True, total_slots=4, config=SLOT_CONFIG
        )

    assert provider._response_is_chain_exhaustion(response)
    assert provider._exhaustion_retry_after(response) == 7
    assert provider._bounded_hold_seconds(300.0, response, 3) == 7.0
