"""Atomic local-admission reservations + explicit fallback intent
(LP-0MVIKPCSG002B0NL).

Covers:

- AC1: reserve returns a ticket when the occupancy-aware projected wait fits
  the budget, and a follow-up request with the ticket dispatches local without
  queueing (no ``contention_queue_wait`` log for that request).
- AC2: reserve returns 429 with ``reason``/``queue_eta_s``/``budget_s``/
  ``retry_after_s`` when the projected wait exceeds the budget; the refused
  request never enters the contention queue.
- AC3: an expired or cancelled ticket is not honoured (falls back to the
  ordinary path); a cancelled reservation releases the gate for the next
  ticket.
- AC4: ``X-Local-Preference: local-only`` turns a would-be remote fallback
  into the retryable 429, even while remote providers are available.
- AC5: ``remote-ok`` keeps the local route ("local allowed, remote
  acceptable") and never waits for it.
- AC6: route headers ``X-Local-Route: local|remote|refused`` and
  ``X-Local-Fallback-Reason``.
- AC7: the reserve verdict and the admission-guard verdict are computed by the
  same function — identical outcomes for identical inputs.
- AC8: fail-open/backcompat — unknown preference values, missing fields and a
  missing ticket behave exactly as today; an uncalibrated reserve admits.

The single-implementation contract (LP-0MVSMROLM005RO71, parent AC1) is pinned
at the end of this module: ``provider.py`` must call
``admission_reserve.local_capacity_verdict`` once per refusal — the remote-first
cycle, the local-first cycle (choke point and ``refused_local_only`` branch)
and the refusal builder all route through it — and must never re-derive the
verdict. A spy replaces the helper with a sentinel-returning wrapper, so a
second reason table in ``provider.py`` shows up as a missing call and a 429
body that does not carry the sentinel.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response
from starlette.datastructures import Headers

from proxy import admission_reserve, contention_queue, prefill_eta, priority

#: Tokens whose calibrated own-ETA (~56.2 s) fits the 60 s budget while the
#: projected wait with one occupant (~112.4 s) does not (same fixture as the
#: occupancy test).
_TOKENS = 12_000
_BUDGET = 60.0


class _DummyRequest:
    """Minimal request stub (matches the admission-guard test helpers)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers: dict | None = None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


class _MutableConcurrency:
    """Mutable ``_get_local_concurrency_info`` stand-in."""

    def __init__(self, active: int = 0, max_: int = 1):
        self.active = active
        self.max = max_

    def __call__(self, config, endpoint=None) -> tuple[int, int]:
        return (self.active, self.max)


def _ok_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


def _cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 4,
        # This module pins the BUDGET refusal (60 s) and its payload/reason.
        # The deadline bound is occupancy-inclusive too (LP-0MVOFVCA2008DQVE),
        # so keep it clear of the ~112 s projected wait: otherwise the
        # deadline would dominate the reason and these assertions would stop
        # testing the budget path they name.
        "local_prefill_deadline_seconds": 300,
    }
    server.update(server_overrides)
    return {"provider_cooldown_seconds": 60, "server": server}


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    admission_reserve.reset()
    yield
    contention_queue.reset()
    prefill_eta.reset()
    admission_reserve.reset()


def _seed_calibration(
    tokens: int = _TOKENS, ms_per_token: float = 3.9, count: int = 6
) -> None:
    for _ in range(count):
        prefill_eta.record_dispatch_sample("Qwen3", tokens, 0.0, tokens * ms_per_token)


@contextmanager
def _estimate_patches(tokens: int = _TOKENS, cached_ratio: float = 0.0):
    with (
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=tokens),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        patch("proxy.provider._should_skip_local", lambda *a, **k: (False, None)),
        patch("proxy.provider._get_cached_ratio", lambda *_: cached_ratio),
    ):
        yield


# ---------------------------------------------------------------------------
# AC1 — reserve end-to-end (ticket → follow-up dispatch, no queueing)
# ---------------------------------------------------------------------------

def test_reserve_returns_ticket_when_projected_wait_fits_budget():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-1"},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )

    assert status == 200
    assert payload["admit"] is True
    assert payload["reservation_id"]
    assert payload["ttl_s"] == admission_reserve.DEFAULT_RESERVE_TTL_SECONDS


@pytest.mark.asyncio
async def test_follow_up_with_ticket_dispatches_local_without_queueing(caplog):
    """AC1 end-to-end: the ticket bypasses admission re-check and the queue."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()
    # Gate is occupied (busy=1 ⇒ projected 112 s > budget), so without the
    # ticket this request would be refused. The ticket was reserved for an
    # idle gate (busy=0 ⇒ own ETA 56.2 s < budget) and is honoured now.
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-1"},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )
    assert status == 200

    request = _DummyRequest(headers={"X-Local-Reservation": payload["reservation_id"]})
    with (
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        action, reason, elapsed = await provider._maybe_queue_for_local_slot(
            _cfg(),
            1,
            1,
            request,
            {},
            {},
            {"name": "local-llama", "llama_model": "Qwen3"},
            "s-1",
            priority="low",
        )

    assert action == "dispatch"
    assert reason == "reserved"
    assert elapsed == 0.0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_reserve_claimed" in messages
    assert "contention_queue_wait" not in messages, "reserved dispatch must not queue"
    # Single-use: the ticket is consumed by the honoured follow-up.
    assert admission_reserve.peek(payload["reservation_id"]) is None


# ---------------------------------------------------------------------------
# AC2 — reserve refusal (429 with reasons; never enters the queue)
# ---------------------------------------------------------------------------

def test_reserve_refuses_with_full_payload_when_wait_exceeds_budget():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-1"},
        _cfg(),
        model="Qwen3",
        busy_slots=1,  # occupied gate ⇒ projected ~112 s > 60 s budget
    )

    assert status == 429
    assert payload["admit"] is False
    assert payload["reason"] == "eta_exceeds_budget"
    assert payload["queue_eta_s"] > _BUDGET
    assert payload["budget_s"] == _BUDGET
    assert isinstance(payload["retry_after_s"], int)
    assert payload["retry_after_s"] >= 1


@pytest.mark.asyncio
async def test_refused_reserve_request_never_enters_the_contention_queue(caplog):
    """AC2 end-to-end: a refused refund request has no queue side effects."""
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS}, _cfg(), model="Qwen3", busy_slots=1
    )
    assert status == 429
    assert "reservation_id" not in payload
    assert contention_queue.queue_depth() == 0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "local_admission_reserve admit=false reason=eta_exceeds_budget" in messages


# ---------------------------------------------------------------------------
# AC3 — TTL and cancellation
# ---------------------------------------------------------------------------

def test_expired_ticket_is_not_honoured(monkeypatch):
    import proxy.admission_reserve as ar_mod

    # Ticket created at monotonic 0 with a 2 s TTL.
    admission_reserve._reservations["t-expired"] = admission_reserve.Reservation(
        ticket_id="t-expired",
        estimated_tokens=_TOKENS,
        cached_ratio=0.0,
        priority=None,
        provider="Qwen3",
        session_id=None,
        created_at=0.0,
        ttl_seconds=2.0,
    )
    monkeypatch.setattr(ar_mod.time, "monotonic", lambda: 1.0)
    assert ar_mod.peek("t-expired") is not None, "ticket valid inside its TTL"

    # Fast-forward past the TTL: the ticket must not be honoured.
    monkeypatch.setattr(ar_mod.time, "monotonic", lambda: 3.0)
    assert ar_mod.peek("t-expired") is None, "expired ticket must not be honoured"
    assert ar_mod.claim("t-expired") is None, "expired claim falls back (AC3)"


def test_claim_of_expired_ticket_counts_expiry_without_raising(monkeypatch):
    """LP-0MVJXC9MK003UET8: the expired-claim branch must not raise F823.

    ``claim()`` incremented the module-global ``_expired`` without declaring
    it ``global``, so the branch made ``_expired`` a local read-before-
    assignment (``UnboundLocalError``). The race is real: a ticket can expire
    between ``_prune_expired()`` and the ``is_expired()`` check, because both
    sample ``time.monotonic()``. Drive that exact interleaving by having the
    first monotonic read (inside ``_prune_expired``) land inside the TTL and
    the second (inside ``claim``) land past it.
    """
    import proxy.admission_reserve as ar_mod

    admission_reserve._reservations["t-race"] = admission_reserve.Reservation(
        ticket_id="t-race",
        estimated_tokens=_TOKENS,
        cached_ratio=0.0,
        priority=None,
        provider="Qwen3",
        session_id=None,
        created_at=0.0,
        ttl_seconds=2.0,
    )
    before = admission_reserve.stats()["reserved_expired"]
    # monotonic() is called by _prune_expired (t=1.0, inside the 2 s TTL) and
    # then by Reservation.is_expired() inside claim (t=3.0, past it).
    readings = iter([1.0, 3.0])
    monkeypatch.setattr(ar_mod.time, "monotonic", lambda: next(readings, 3.0))

    assert admission_reserve.claim("t-race") is None, (
        "an expired claim must fall back to the ordinary path"
    )
    assert admission_reserve.stats()["reserved_expired"] == before + 1, (
        "the expiry must be counted in the reserved_expired stat"
    )


def test_cancelled_ticket_is_not_honoured_and_releases_the_gate():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-1"},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )
    ticket = payload["reservation_id"]

    cancel_status, cancel_payload = admission_reserve.cancel(ticket)
    assert cancel_status == 200
    assert cancel_payload["cancelled"] is True
    assert admission_reserve.peek(ticket) is None

    # The next client can reserve again (the gate claim was released).
    status2, payload2 = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "session_id": "s-2"},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )
    assert status2 == 200
    assert payload2["reservation_id"] != ticket


def test_cancel_unknown_ticket_returns_404():
    status, payload = admission_reserve.cancel("does-not-exist")
    assert status == 404
    assert payload["cancelled"] is False


def test_missing_ticket_header_takes_the_ordinary_path():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS}, _cfg(), model="Qwen3", busy_slots=0
    )
    # A follow-up WITHOUT the header has no reservation.
    assert admission_reserve.claim(None) is None


# ---------------------------------------------------------------------------
# AC7 — single estimator: reserve verdict == guard verdict
# ---------------------------------------------------------------------------

def test_reserve_verdict_matches_admission_guard_for_identical_inputs():
    _seed_calibration()
    for busy in (0, 1):
        reserve_status, reserve_payload = admission_reserve.reserve(
            {"estimated_tokens": _TOKENS}, _cfg(), model="Qwen3", busy_slots=busy
        )
        guard = prefill_eta.evaluate_admission(
            _cfg(), "Qwen3", _TOKENS, 0.0, "medium", busy_slots=busy
        )
        assert reserve_status == (200 if guard.admit else 429)
        if guard.admit:
            assert reserve_payload["admit"] is True
        else:
            assert reserve_payload["reason"] == guard.reason
            assert reserve_payload["queue_eta_s"] == pytest.approx(
                guard.projected_wait_seconds, abs=0.1
            )


# ---------------------------------------------------------------------------
# AC8 — fail-open / backcompat
# ---------------------------------------------------------------------------

def test_reserve_missing_or_invalid_fields_are_400():
    status, payload = admission_reserve.reserve({}, _cfg())
    assert status == 400
    assert "estimated_tokens" in payload.get("error", "")

    status, payload = admission_reserve.reserve(
        {"estimated_tokens": -5}, _cfg()
    )
    assert status == 400


def test_uncalibrated_reserve_admits_even_with_full_occupancy():
    # No samples seeded ⇒ inert guard ⇒ admit (never reject on a guess).
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS}, _cfg(), model="Qwen3", busy_slots=8
    )
    assert status == 200
    assert payload["admit"] is True
    assert payload["reservation_id"]
    # The verdict's reason shows the guard was inert, not calibrated.
    decision = prefill_eta.evaluate_admission(
        _cfg(), "Qwen3", _TOKENS, 0.0, None, busy_slots=8
    )
    assert decision.admit is True
    assert decision.reason == "insufficient_samples"


def test_unknown_preference_value_falls_back_to_prefer_local():
    assert (
        admission_reserve.parse_preference({"X-Local-Preference": "purple"})
        == "prefer-local"
    )
    assert (
        admission_reserve.parse_preference({"X-Local-Preference": "PURPLE"})
        == "prefer-local"
    )
    assert admission_reserve.parse_preference({}) == "prefer-local"
    assert (
        admission_reserve.parse_preference({"X-Local-Preference": "remote-ok"})
        == "remote-ok"
    )


# ---------------------------------------------------------------------------
# Preference vocabulary: local-preferred is canonical, local-only is the alias
# (LP-0MVSHWYVE005MS16)
# ---------------------------------------------------------------------------

def test_local_preferred_parses():
    """AC1: the new canonical pin-local value parses to itself."""
    assert admission_reserve.PREF_LOCAL_PREFERRED == "local-preferred"
    assert (
        admission_reserve.parse_preference(
            {"x-local-preference": "local-preferred"}
        )
        == "local-preferred"
    )
    # AC3: the *value* is case-insensitive, exactly as before the rename.
    assert (
        admission_reserve.parse_preference(
            {"X-Local-Preference": "LOCAL-PREFERRED"}
        )
        == "local-preferred"
    )


def test_legacy_local_only_alias_normalises():
    """AC1/AC3: the deployed wire value normalises, in any header case."""
    assert (
        admission_reserve.parse_preference({"x-local-preference": "local-only"})
        == "local-preferred"
    )
    assert (
        admission_reserve.parse_preference({"X-Local-Preference": "Local-Only"})
        == "local-preferred"
    )
    # Real requests carry Starlette's case-insensitive Headers: both the header
    # name and the value case must keep working (AC3).
    assert (
        admission_reserve.parse_preference(
            Headers({"X-LOCAL-PREFERENCE": "LOCAL-ONLY"})
        )
        == "local-preferred"
    )


def test_alias_table_and_canonical_set_cannot_drift():
    """The alias is a table entry, not a stray literal (AC4)."""
    assert admission_reserve.LEGACY_PREFERENCE_ALIASES == {
        "local-only": "local-preferred"
    }
    assert "local-preferred" in admission_reserve.VALID_PREFERENCES
    assert "local-only" not in admission_reserve.VALID_PREFERENCES, (
        "local-only is a legacy alias, not a canonical value"
    )
    assert set(admission_reserve.LEGACY_PREFERENCE_ALIASES.values()) <= set(
        admission_reserve.VALID_PREFERENCES
    )


def test_legacy_alias_logs_one_deprecation_per_request(caplog):
    """AC2: one WARNING per request, naming both sides of the alias.

    The deprecation is emitted by ``parse_preference()`` and latched by the
    per-request ``_local_preference`` cache, so a request that is parsed for
    the dispatch loop, the remote cycle and the refusal path still logs once.
    """
    request = _DummyRequest(headers={"x-local-preference": "local-only"})

    with caplog.at_level(logging.WARNING, logger="llama-proxy"):
        first = provider._request_local_preference(request)
        second = provider._request_local_preference(request)
        third = priority.local_preference_for_request(request)

    assert first == second == third == "local-preferred"
    deprecated = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(deprecated) == 1, (
        "the legacy alias must log exactly one deprecation per request; got "
        f"{deprecated!r}"
    )
    assert "local-only" in deprecated[0], (
        f"the deprecation must name the legacy value; got {deprecated[0]!r}"
    )
    assert "local-preferred" in deprecated[0], (
        f"the deprecation must name the canonical value; got {deprecated[0]!r}"
    )


def test_canonical_preference_logs_no_deprecation(caplog):
    """AC2: only the legacy alias is deprecated — canonical values stay quiet."""
    for value in ("local-preferred", "prefer-local", "remote-ok"):
        request = _DummyRequest(headers={"x-local-preference": value})
        with caplog.at_level(logging.WARNING, logger="llama-proxy"):
            provider._request_local_preference(request)
        deprecations = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert deprecations == [], (
            f"{value!r} is canonical and must not log a deprecation; got "
            f"{deprecations!r}"
        )
        caplog.clear()


#: The preference vocabulary values that must exist in exactly one module.
_PREFERENCE_LITERALS = {
    "local-only",
    "local-preferred",
    "prefer-local",
    "remote-ok",
}

_PROXY_PACKAGE = Path(__file__).resolve().parents[1] / "proxy"


def test_single_preference_vocabulary_source():
    """AC4: one module owns the vocabulary and the parser.

    Every string constant that *is* a preference value (equality, not
    substring — log and refusal messages legitimately mention the wire values)
    must live in ``admission_reserve.py``, and no other module may define a
    second ``parse_preference``. ``test_*.py`` modules inside the package are
    fixtures, not production code (``test_home_page.py`` uses ``local-only`` as
    a *provider name*).
    """
    sources = [
        path
        for path in sorted(_PROXY_PACKAGE.glob("*.py"))
        if not path.name.startswith("test_")
    ]
    assert sources, f"no production modules found under {_PROXY_PACKAGE}"

    stray_literals: list[str] = []
    stray_parsers: list[str] = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in _PREFERENCE_LITERALS
                and source.name != "admission_reserve.py"
            ):
                stray_literals.append(f"{source.name}:{node.lineno} {node.value!r}")
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "parse_preference"
                and source.name != "admission_reserve.py"
            ):
                stray_parsers.append(f"{source.name}:{node.lineno}")

    assert stray_literals == [], (
        "preference literals must only exist in admission_reserve.py (import "
        f"the constants instead); found {stray_literals}"
    )
    assert stray_parsers == [], (
        "admission_reserve.parse_preference() is the only parser; found a "
        f"second definition at {stray_parsers}"
    )


def test_dry_run_returns_verdict_without_claiming():
    _seed_calibration()
    status, payload = admission_reserve.reserve(
        {"estimated_tokens": _TOKENS, "dry_run": True},
        _cfg(),
        model="Qwen3",
        busy_slots=0,
    )
    assert status == 200
    assert payload["dry_run"] is True
    assert "reservation_id" not in payload


# ---------------------------------------------------------------------------
# AC4 — local-only: 429 instead of remote fallback (remotes available)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preference",
    ["local-only", "local-preferred"],
    ids=["legacy-alias", "canonical"],
)
async def test_pin_local_refuses_429_when_guard_would_fall_back_remote(
    preference, caplog
):
    """AC4/AC5: with remotes available, either pin-local spelling yields 429.

    ``local-only`` is the deployed legacy alias and must reach the *same*
    choke point as the canonical ``local-preferred`` once normalised — no
    demotion to ``prefer-local``, no silent remote fall-through.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy")
    _seed_calibration()
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-fallback", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ],
        "aliases": ["hybrid*"],
    }
    request = _DummyRequest(headers={"X-Local-Preference": preference})
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _cfg()
        )

    # Occupied gate + 20k tokens ⇒ the guard refuses; a pin-local client must
    # NOT proceed to the remote provider.
    assert calls == []
    assert result.status_code == 429
    assert result.headers.get("X-Local-Route") == "refused"
    assert result.headers.get("X-Local-Fallback-Reason") == "eta_exceeds_budget"
    assert result.headers.get("Retry-After")
    assert result.headers.get("X-Local-Admission") == "reject"
    body = json.loads(result.body)
    assert body["code"] == "local_admission_reject"
    assert body["retry_after"] >= 1
    # LP-0MVNTJJGX00375VQ AC1/AC2/AC6: a pin-local refusal is a capacity/intent
    # outcome, not chain exhaustion — the remote provider above is available
    # and was never consulted (``calls == []``), so "All providers exhausted"
    # would be factually wrong.
    assert body["error"] == "Local admission refused", (
        f"the queue-path refusal must not claim exhaustion; got {body['error']!r}"
    )
    assert b"All providers exhausted" not in result.body
    assert "local-only" in body["detail"], (
        f"detail must name the pin intent; got {body['detail']!r}"
    )
    assert "refused" in body["detail"], (
        f"detail must name the guard's refusal; got {body['detail']!r}"
    )


async def _issue_pin_local_refusal(preference: str) -> tuple:
    """Issue a pin-local request the guard refuses; return ``(response, calls)``.

    The occupied gate plus a calibrated 12k-token prompt projects past the 60 s
    wait budget, so the *only* variable between calls is the wire spelling of
    the pin-local preference.
    """
    _seed_calibration()
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers={"X-Local-Preference": preference}),
            "v1/chat/completions",
            config,
            _cfg(),
        )
    return result, calls


@pytest.mark.asyncio
async def test_legacy_alias_refusal_is_field_for_field_the_canonical_one(caplog):
    """Row 7: the alias is the same refusal, plus exactly one deprecation line.

    ``local-only`` is a *wire* alias, not a second behaviour: the two spellings
    must produce an identical refusal — status, route/reason headers and every
    stable body field (including the retry story) — and the alias must announce
    itself exactly once per request, never once per routing decision
    (LP-0MVSHWYVE005MS16 AC1/AC2). ``retry_at`` is excluded from the comparison
    because it is stamped from the wall clock at build time.
    """
    caplog.set_level(logging.WARNING, logger="llama-proxy")

    canonical, canonical_calls = await _issue_pin_local_refusal("local-preferred")
    canonical_deprecations = [
        record.getMessage()
        for record in caplog.records
        if "deprecated X-Local-Preference" in record.getMessage()
    ]
    caplog.clear()
    legacy, legacy_calls = await _issue_pin_local_refusal("local-only")
    legacy_deprecations = [
        record.getMessage()
        for record in caplog.records
        if "deprecated X-Local-Preference" in record.getMessage()
    ]

    assert canonical_calls == legacy_calls == [], (
        "neither spelling may reach a remote provider (dispatched="
        f"{canonical_calls!r}/{legacy_calls!r})"
    )
    assert canonical.status_code == legacy.status_code == 429
    for header in ("X-Local-Route", "X-Local-Fallback-Reason", "Retry-After"):
        assert legacy.headers.get(header) == canonical.headers.get(header), (
            f"the alias must not change {header}: canonical="
            f"{canonical.headers.get(header)!r}, legacy="
            f"{legacy.headers.get(header)!r}"
        )
    canonical_body = json.loads(canonical.body)
    legacy_body = json.loads(legacy.body)
    for key in ("code", "reason", "verdict", "retryable", "retry_after"):
        assert legacy_body.get(key) == canonical_body.get(key), (
            f"the alias must not change body[{key!r}]: canonical="
            f"{canonical_body.get(key)!r}, legacy={legacy_body.get(key)!r}"
        )

    assert canonical_deprecations == [], (
        "the canonical value is not deprecated; got "
        f"{canonical_deprecations!r}"
    )
    assert len(legacy_deprecations) == 1, (
        "the legacy alias must log exactly one deprecation per request; got "
        f"{legacy_deprecations!r}"
    )


@pytest.mark.asyncio
async def test_prefer_local_still_falls_back_remote_when_refused(caplog):
    """Backcompat (AC8): prefer-local (and absent header) keep the fallback."""
    _seed_calibration()
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-fallback", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ],
        "aliases": ["hybrid*"],
    }
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _cfg()
        )

    assert calls == ["remote"], "prefer-local keeps today's remote fallback"
    assert result.status_code == 200


# ---------------------------------------------------------------------------
# AC5 — remote-ok keeps the local route and never waits for it
#
# LP-0MVUH9OYZ007GMWF AC1: the value's name is historic. It used to strip every
# local entry from the chain; it now means "local allowed, remote acceptable",
# so a free local slot is used and a busy one falls through without queueing.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remote_ok_attempts_the_local_route_when_a_slot_is_free(caplog):
    """AC5/AC1: a free local slot is served locally under ``remote-ok``."""
    _seed_calibration()
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):  # pragma: no cover - must never run
        calls.append("remote")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-fallback", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ],
        "aliases": ["hybrid*"],
    }
    request = _DummyRequest(headers={"X-Local-Preference": "remote-ok"})
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _cfg()
        )

    assert calls == ["local"], (
        f"remote-ok must attempt the local route (parent AC15); got {calls}"
    )
    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") == "local"
    assert "X-Local-Fallback-Reason" not in result.headers, (
        "a locally served request has nothing to fall through to"
    )


@pytest.mark.asyncio
async def test_remote_ok_on_a_remote_less_chain_is_served_locally(caplog):
    """AC5/AC3: a remote-less chain is never emptied by a preference."""
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
        ],
        "aliases": ["hybrid*"],
    }
    request = _DummyRequest(headers={"X-Local-Preference": "remote-ok"})
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(),
        patch("proxy.router_helpers.evaluate_and_apply_compaction", AsyncMock()),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", config, _cfg()
        )

    assert calls == ["local"], (
        f"no remote tier exists to step to; the local route serves, got {calls}"
    )
    assert result.status_code == 200, (
        f"an empty chain must never be manufactured from a preference, got "
        f"{result.status_code}"
    )
    assert result.headers.get("X-Local-Route") == "local"


# ---------------------------------------------------------------------------
# AC6 — route headers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_success_carries_route_local(caplog):
    _seed_calibration()
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-fallback", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ],
        "aliases": ["hybrid*"],
    }
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        _estimate_patches(tokens=2_000),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config, _cfg()
        )

    assert calls == ["local"]
    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") == "local"


@pytest.mark.asyncio
async def test_remote_success_carries_route_remote_and_reason(caplog):
    """AC6: remote-served responses carry the recorded fallback reason."""
    _seed_calibration()
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    config = {
        "providers": [
            {"name": "local-llama", "type": "local", "llama_model": "Qwen3"},
            {"name": "remote-fallback", "type": "remote",
             "endpoint": "https://api.example.com/v1"},
        ],
        "aliases": ["hybrid*"],
    }
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(1, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        # fallback policy: local slot exhausted ⇒ fall through immediately
        _estimate_patches(),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config,
            _cfg(contention_queue_policy="fallback"),
        )

    assert calls == ["remote"]
    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") == "remote"
    assert result.headers.get("X-Local-Fallback-Reason")


# ---------------------------------------------------------------------------
# Route registration (the endpoint exists and is additive)
# ---------------------------------------------------------------------------

def test_reserve_routes_are_registered():
    from proxy.handlers import router

    paths = {route.path for route in router.routes}
    assert "/llama/local/reserve" in paths
    assert "/llama/local/reserve/{reservation_id}" in paths


# ---------------------------------------------------------------------------
# Local-capacity verdict: ``deferrable`` vs ``too_large``
# (LP-0MVSHX47F003XEIZ / LP-0MVSL7QJW006OYPB)
# ---------------------------------------------------------------------------

def _admission_decision(**overrides) -> prefill_eta.AdmissionDecision:
    """Build an ``AdmissionDecision`` row without invoking the estimator.

    Defaults reproduce the observed 2026-10-28 incident shape: the priced
    own-ETA (31.7 s) fits the 60 s budget, the projection with one occupant
    (63.4 s) does not, and the prompt (6 627 new tokens) is well inside the
    25 000-token cold-prefill budget.
    """
    fields = {
        "admit": False,
        "reason": "eta_exceeds_budget",
        "eta_prefill_seconds": 31.7,
        "budget_seconds": 60.0,
        "policy": "reject",
        "calibrated": True,
        "samples": 5,
        "projected_wait_seconds": 63.4,
        "new_tokens": 6627,
        "budget_tokens": 25000,
    }
    fields.update(overrides)
    return prefill_eta.AdmissionDecision(**fields)


def test_capacity_verdict_incident_shape_is_deferrable():
    """AC1: the observed incident shape is temporal, not a size refusal."""
    decision = _admission_decision()

    verdict = admission_reserve.local_capacity_verdict(decision)
    assert verdict.verdict == admission_reserve.CAPACITY_DEFERRABLE
    assert verdict.reason == "eta_exceeds_budget"

    # The equivalent raw-input form (the direct free-slot guard path, which
    # labels the same decision ``eta_exceeds_budget_direct``).
    raw = admission_reserve.local_capacity_verdict(
        reason="eta_exceeds_budget_direct",
        new_tokens=6627,
        budget_tokens=25000,
    )
    assert raw.verdict == admission_reserve.CAPACITY_DEFERRABLE
    assert raw.reason == "eta_exceeds_budget_direct"


#: Full classification table (AC3): ``(id, reason, new_tokens,
#: budget_tokens, estimated_tokens, context_limit, expected_verdict)``.
_CAPACITY_VERDICT_ROWS = [
    # Guard temporal reasons — waiting can help.
    ("guard-budget", "eta_exceeds_budget", 6627, 25000, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("guard-budget-direct", "eta_exceeds_budget_direct", 6627, 25000, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("guard-deadline", "prefill_eta_exceeds_deadline", 6627, 25000, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("guard-safety-budget", "prefill_eta_exceeds_safety_budget", 6627, 25000,
     None, None, admission_reserve.CAPACITY_DEFERRABLE),
    # Non-guard temporal exits (lease / cooldown / occupancy / queue / load).
    ("lease-active", "local_lease_active", None, None, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("provider-cooldown", "local_provider_cooldown", None, None, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("busy-slots", "slot_exhaustion", None, None, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("queue-expiry", "fallback_after_queue", None, None, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("model-loading", "model_loading", None, None, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("stall-cooldown", "prefill_thrash_cooldown", None, None, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("guard-admitted", "within_budget", 6627, 25000, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    # Too large — a property of the request; waiting cannot help.
    ("cold-prefill-cap", "prefill_budget_exceeded", 30000, 25000, None, None,
     admission_reserve.CAPACITY_TOO_LARGE),
    ("context-cap", "context_too_large", None, None, None, None,
     admission_reserve.CAPACITY_TOO_LARGE),
    ("tokens-over-budget", None, 25001, 25000, None, None,
     admission_reserve.CAPACITY_TOO_LARGE),
    ("tokens-over-budget-unknown-reason", "some_new_temporal_reason", 25001,
     25000, None, None, admission_reserve.CAPACITY_TOO_LARGE),
    ("context-over-limit", None, None, None, 40000, 32768,
     admission_reserve.CAPACITY_TOO_LARGE),
    # Disabled bounds never classify too_large (0 = cap/bound disabled).
    ("token-cap-disabled", None, 99999, 0, None, None,
     admission_reserve.CAPACITY_DEFERRABLE),
    ("context-bound-disabled", None, None, None, 99999, 0,
     admission_reserve.CAPACITY_DEFERRABLE),
]


@pytest.mark.parametrize(
    "row",
    _CAPACITY_VERDICT_ROWS,
    ids=[row[0] for row in _CAPACITY_VERDICT_ROWS],
)
def test_capacity_verdict_classification_table(row):
    """AC3: every table row asserts its own outcome and preserved reason."""
    _row_id, reason, new_tokens, budget_tokens, estimated_tokens, context_limit, expected = row

    verdict = admission_reserve.local_capacity_verdict(
        reason=reason,
        new_tokens=new_tokens,
        budget_tokens=budget_tokens,
        estimated_tokens=estimated_tokens,
        context_limit=context_limit,
    )

    assert verdict.verdict == expected
    assert verdict.reason == reason


def test_capacity_verdict_decision_form_classifies_too_large():
    """AC2: the guard decision alone classifies both too-large triggers."""
    for reason in ("prefill_budget_exceeded", "context_too_large"):
        verdict = admission_reserve.local_capacity_verdict(
            _admission_decision(reason=reason)
        )
        assert verdict.verdict == admission_reserve.CAPACITY_TOO_LARGE
        assert verdict.reason == reason


def test_capacity_verdict_raw_inputs_override_decision_fields():
    """AC2: raw values win over the decision they were derived from."""
    too_large = admission_reserve.local_capacity_verdict(
        _admission_decision(), new_tokens=30000
    )
    assert too_large.verdict == admission_reserve.CAPACITY_TOO_LARGE

    # An explicit raw reason also wins over the decision's reason.
    verdict = admission_reserve.local_capacity_verdict(
        _admission_decision(), reason="context_too_large"
    )
    assert verdict.verdict == admission_reserve.CAPACITY_TOO_LARGE
    assert verdict.reason == "context_too_large"


def test_capacity_verdict_fail_safe_defaults_to_deferrable():
    """AC4: unknown/absent reasons and missing inputs stay pin-local."""
    empty = admission_reserve.local_capacity_verdict()
    assert empty == admission_reserve.CapacityVerdict(
        verdict=admission_reserve.CAPACITY_DEFERRABLE, reason=None
    )

    unknown = admission_reserve.local_capacity_verdict(reason="never_seen_before")
    assert unknown.verdict == admission_reserve.CAPACITY_DEFERRABLE
    assert unknown.reason == "never_seen_before"

    # A value present without its bound cannot be compared: deferrable.
    incomplete = admission_reserve.local_capacity_verdict(
        new_tokens=30000, estimated_tokens=40000
    )
    assert incomplete.verdict == admission_reserve.CAPACITY_DEFERRABLE


def test_capacity_verdict_is_pure_and_never_calls_the_estimator():
    """AC5: pure helper — no estimator call, identical inputs ⇒ same verdict."""
    with patch(
        "proxy.admission_reserve.evaluate_admission",
        side_effect=AssertionError("the verdict must not re-price the prompt"),
    ):
        first = admission_reserve.local_capacity_verdict(_admission_decision())
        second = admission_reserve.local_capacity_verdict(_admission_decision())

    assert first == second
    assert first == admission_reserve.CapacityVerdict(
        verdict=admission_reserve.CAPACITY_DEFERRABLE, reason="eta_exceeds_budget"
    )


def test_capacity_verdict_vocabulary_and_module_boundary():
    """AC5: exactly two verdict values, and no import of ``provider``."""
    assert admission_reserve.CAPACITY_DEFERRABLE == "deferrable"
    assert admission_reserve.CAPACITY_TOO_LARGE == "too_large"
    assert admission_reserve.CAPACITY_VERDICTS == frozenset(
        {"deferrable", "too_large"}
    )

    source = Path(admission_reserve.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(
                f"{node.module}.{alias.name}" for alias in node.names
            )
    providers = sorted(name for name in imported if name.split(".")[-1] == "provider")
    assert providers == [], (
        "admission_reserve must not import provider (import cycle); found "
        f"{providers}"
    )


# ---------------------------------------------------------------------------
# Single-implementation contract: one verdict call per refusal
# (LP-0MVSMROLM005RO71, child of LP-0MVSMI7JK002FKUF — parent AC1)
#
# ``provider.py`` must consume ``admission_reserve.local_capacity_verdict`` and
# never re-derive the capacity split. The spy below replaces the helper with a
# recording wrapper returning a sentinel verdict, so a re-derived verdict (or a
# second reason table in ``provider.py``) shows up as a missing call *and* a
# 429 body that fails to carry the sentinel.
#
# TDD red phase (LP-0MVSMROLM005RO71): the call-site assertions started
# ``xfail(strict=True)`` while ``provider.py`` did not consume the helper. The
# wiring child (LP-0MVSMINPR006SWT6) flipped them green in the same change that
# added the behaviour, so a red assertion now means the contract broke.
# ---------------------------------------------------------------------------

#: A verdict value from *outside* the vocabulary: only a verdict that actually
#: travelled through the helper can carry it into the 429 body.
_SPY_SENTINEL = "spy-sentinel-verdict"

_LOCAL_PROVIDER_NAME = "local-llama"
_REMOTE_PROVIDER_NAME = "remote-fallback"


class _CapacityVerdictSpy:
    """Recording stand-in for the single capacity-verdict helper."""

    def __init__(self, sentinel: str = _SPY_SENTINEL) -> None:
        self.sentinel = sentinel
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs) -> admission_reserve.CapacityVerdict:
        self.calls.append({"args": args, "kwargs": kwargs})
        return admission_reserve.CapacityVerdict(
            verdict=self.sentinel,
            reason=kwargs.get("reason"),
        )


def _spy_capacity_verdict(sentinel: str = _SPY_SENTINEL):
    """Patch the helper; return ``(spy, patcher)``.

    Patched at the module ``provider.py`` imports it from — never at the
    refusal builder, which would let the contract break while the test passes.
    """
    spy = _CapacityVerdictSpy(sentinel)
    return spy, patch("proxy.admission_reserve.local_capacity_verdict", new=spy)


def _refusal_builder_calls(spy: _CapacityVerdictSpy) -> list[dict]:
    """Return the spy calls made by the refusal builder (one per refusal).

    LP-0MVSLZY470051ZB2 added a *pre-dispatch* consultation of the same helper
    from the pin predicate (``_local_preference_pins_local``), so a pin-local
    request may now consult the helper more than once. The invariant that still
    holds — and that these tests pin — is one computation **per decision
    point**: the 429 payload's verdict comes from exactly one builder call.

    The builder's consultation is distinguishable structurally: it passes the
    refusal evidence by keyword (``reason`` / ``new_tokens`` /
    ``budget_tokens``), while the pin predicate passes the routing evidence
    positionally (the guard decision) or as a sub-set of the same keywords.
    """
    return [
        call
        for call in spy.calls
        if call["args"] == ()
        and set(call["kwargs"]) == {"reason", "new_tokens", "budget_tokens"}
    ]


def _local_then_remote_chain() -> dict:
    return {
        "providers": [
            {"name": _LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"},
            {
                "name": _REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }


def _remote_only_chain() -> dict:
    return {
        "providers": [
            {
                "name": _REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }


async def _issue_local_first_cycle(
    preference: str | None,
    *,
    tokens: int = _TOKENS,
    cooldown: bool = False,
    concurrency: tuple[int, int] = (0, 1),
) -> tuple:
    """Drive ``proxy_with_fallback`` for *preference*; return ``(response, calls)``.

    *concurrency* is the ``(active, max)`` local slot state: busy slots are what
    route the request through the contention queue (and thus the
    ``refused_local_only`` branch) instead of the free-slot direct path.
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        return _ok_response()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    headers = {} if preference is None else {"X-Local-Preference": preference}
    if cooldown:
        provider._provider_unavailable_until[_LOCAL_PROVIDER_NAME] = time.time() + 60
    try:
        with (
            patch("proxy.router.proxy_to_local", _local),
            patch("proxy.server.proxy_to_remote", _remote),
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(*concurrency),
            ),
            patch("proxy.provider.request_priority", return_value="low"),
            _estimate_patches(tokens),
        ):
            result = await provider.proxy_with_fallback(
                _DummyRequest(headers=headers),
                "v1/chat/completions",
                _local_then_remote_chain(),
                _cfg(),
            )
    finally:
        provider._provider_unavailable_until.pop(_LOCAL_PROVIDER_NAME, None)
    return result, calls


@pytest.mark.asyncio
async def test_remote_first_cycle_routes_through_the_helper_once():
    """AC1: the remote-first cycle's refusal consults the helper exactly once.

    The pin predicate may consult the same helper pre-dispatch
    (LP-0MVSLZY470051ZB2); what AC1 pins here is that the echoed verdict comes
    from a single builder computation.
    """
    spy, patcher = _spy_capacity_verdict()
    calls: list[str] = []

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return _ok_response()

    with patcher, patch("proxy.server.proxy_to_remote", _remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(headers={"X-Local-Preference": "local-only"}),
            "v1/chat/completions",
            _remote_only_chain(),
            _cfg(),
        )

    assert result.status_code == 429
    assert calls == [], "the refusal must precede any remote dispatch"
    assert spy.calls, "the pin predicate must consult the shared helper"
    builder_calls = _refusal_builder_calls(spy)
    assert len(builder_calls) == 1, (
        f"the refusal verdict must be computed exactly once; got {spy.calls!r}"
    )
    assert json.loads(result.body)["verdict"] == spy.sentinel


@pytest.mark.asyncio
async def test_local_first_choke_point_refusal_routes_through_the_helper_once():
    """AC1: the local-first choke point's refusal computes it exactly once."""
    spy, patcher = _spy_capacity_verdict()

    with patcher:
        result, calls = await _issue_local_first_cycle("local-only", cooldown=True)

    assert result.status_code == 429
    assert calls == [], "a pin-local client is never moved to remote"
    assert result.headers["X-Local-Fallback-Reason"] == "local_provider_cooldown"
    assert spy.calls, "the pin predicate must consult the shared helper"
    builder_calls = _refusal_builder_calls(spy)
    assert len(builder_calls) == 1, (
        f"the refusal verdict must be computed exactly once; got {spy.calls!r}"
    )
    assert json.loads(result.body)["verdict"] == spy.sentinel


@pytest.mark.asyncio
async def test_refused_local_only_branch_routes_through_the_helper_once():
    """AC1: the ``refused_local_only`` branch's refusal computes it once."""
    spy, patcher = _spy_capacity_verdict()
    _seed_calibration(100_000)

    with patcher:
        # Busy slot + queue policy: this is the contention-queue admission
        # denial (``refused_local_only``), not the free-slot direct path that
        # deliberately keeps the exhausted response (out of scope here).
        result, calls = await _issue_local_first_cycle(
            "local-only", tokens=100_000, concurrency=(1, 1)
        )

    assert result.status_code == 429
    assert calls == [], "a pin-local client is never moved to remote"
    assert spy.calls, "the pin predicate must consult the shared helper"
    builder_calls = _refusal_builder_calls(spy)
    assert len(builder_calls) == 1, (
        f"the refusal verdict must be computed exactly once; got {spy.calls!r}"
    )
    assert json.loads(result.body)["verdict"] == spy.sentinel


def test_refusal_builder_routes_through_the_helper_once():
    """AC1: the builder itself calls the helper exactly once."""
    spy, patcher = _spy_capacity_verdict()

    with patcher:
        result = provider._build_local_only_refusal_response(
            {
                "reason": "eta_exceeds_budget_direct",
                "eta_prefill_seconds": 31.7,
                "budget_seconds": 60.0,
            }
        )

    assert result.status_code == 429
    assert len(spy.calls) == 1, f"exactly one helper call; got {spy.calls!r}"
    assert json.loads(result.body)["verdict"] == spy.sentinel


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", ["prefer-local", "remote-ok", None])
async def test_non_refusal_paths_never_call_the_helper(preference):
    """AC4: no verdict is produced for a request that is not refused."""
    spy, patcher = _spy_capacity_verdict()

    with patcher:
        result, calls = await _issue_local_first_cycle(preference)

    assert result.status_code == 200
    assert calls, "the request must be served, not refused"
    assert spy.calls == [], f"no helper call on a served request; got {spy.calls!r}"


@pytest.mark.asyncio
async def test_fall_through_on_a_size_refusal_never_calls_the_helper():
    """AC4: a fall-through (prefer-local over budget) is not a refusal verdict."""
    spy, patcher = _spy_capacity_verdict()
    _seed_calibration(100_000)

    with patcher:
        result, calls = await _issue_local_first_cycle(
            "prefer-local", tokens=100_000
        )

    assert result.status_code == 200
    assert calls == ["remote"]
    assert spy.calls == [], f"no helper call on a fall-through; got {spy.calls!r}"
