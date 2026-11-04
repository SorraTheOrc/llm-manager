"""Fail fast on a provably dead provider chain (LP-0MW0KZR7A003UFQK).

On 2026-11-03 08:41:42 every remote provider of a model was skipped for a
deterministic, long-horizon reason — ``munitoai-free``
``usage_limit_reset_pending`` (reset ~15.4 h away), ``opencode-go-3``
``quota_exhausted`` (~14 days), and four entries
``outside_available_window`` — yet the proxy logged ``Chain exhausted ...
holding 300s before restarting cycle`` three times. The hold bound
(``_bounded_hold_seconds``) is driven by the exhaustion response's
``retry_after``, and that timing was lost because the exhaustion tail
substitutes the *first provider's raw error* (at 08:40 a bare ``400`` with no
``Retry-After``) for the generic exhaustion payload.

At 08:49:40 the same class of defect hit the other half of the request: the
local prefill-ETA guard refused a ``remote-ok`` request on an enforced ETA
bound without checking whether *any* remote provider could serve it, so the
request was pushed onto the provably dead remote chain.

The contracts pinned here:

- AC1 — when every provider of the residual chain is deterministically skipped
  or terminally refused for this request, the chain is not held: the error is
  returned immediately with ``Retry-After`` (header) == ``retry_after`` (body)
  from the **earliest** known reset, and no ``chain_hold_cycle`` is consumed.
  Both the "nothing tried" and the "one terminal 400" shapes are covered.
- AC2 — a remote provider "has capacity" only when nothing deterministically
  skips it, it is not in cooldown/quarantine/hard-stop and it is inside its
  ``available_times`` window; a chain with no such provider must not receive a
  refused local request. (The end-to-end admission rows live in
  ``test_local_admission_guard.py``, which owns that harness.)
- AC3 — the chain-end line distinguishes fail-fast from
  ``Chain exhausted ... holding`` on one ``chain_planned_unavailable``
  vocabulary.

Hermetic: a stub request object, module-level state reset per test, and
``retry_after`` derivations monkeypatched only where the wall clock would
otherwise decide the assertion.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import proxy.provider as provider
import pytest
from starlette.responses import Response


class _StubRequest:
    """Request stub carrying ``state`` (the fail-fast verdict's home)."""

    def __init__(self, body: dict | None = None):
        self._body = json.dumps(body or {"model": "test"}).encode("utf-8")
        self.headers: dict[str, str] = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()
        self.state = SimpleNamespace()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _chain(*names: str) -> dict:
    return {
        "providers": [
            {"name": name, "type": "remote", "endpoint": f"https://{name}.example/v1"}
            for name in names
        ]
    }


def _exhaustion_response(retry_after: int | None = 600) -> Response:
    payload: dict = {"error": "All providers exhausted", "code": "all_exhausted"}
    headers = {"Cache-Control": "no-store"}
    if retry_after is not None:
        payload["retry_after"] = retry_after
        headers["Retry-After"] = str(retry_after)
    return Response(
        content=json.dumps(payload).encode("utf-8"),
        status_code=503,
        media_type="application/json",
        headers=headers,
    )


def _body(response: Response) -> dict:
    return json.loads(bytes(response.body))


# ======================================================================
# AC1: the residual-chain verdict
# ======================================================================


def test_deterministic_skips_make_the_chain_provably_dead():
    """AC1 row 1: nothing was tried, every skip is deterministic."""
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "a", "usage_limit_reset_pending")
    provider._note_chain_resolution_skip(request, "b", "outside_available_window")

    verdict = provider._residual_chain_verdict(
        request, _chain("a", "b", "c"), [], {}
    )

    # ``c`` was never skipped and never refused: the chain is not proved dead.
    assert verdict is None

    provider._note_chain_resolution_skip(request, "c", "quota_hard_stop")
    verdict = provider._residual_chain_verdict(request, _chain("a", "b", "c"), [], {})

    assert verdict is not None
    assert verdict["providers"] == 3
    assert verdict["skips"] == {
        "a": "usage_limit_reset_pending",
        "b": "outside_available_window",
        "c": "quota_hard_stop",
    }


def test_terminal_400_alongside_deterministic_skips_is_provably_dead():
    """AC1 row 2: the 08:40 shape — a real first-provider 400 plus skips.

    A terminal request-scoped refusal does not by itself defeat the verdict:
    holding cannot change a 400 the prompt will always reproduce.
    """
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "b", "usage_limit_reset_pending")
    provider._note_chain_resolution_skip(request, "c", "quota_exhausted")
    attempts = [
        {
            "provider": "a",
            "type": "remote",
            "status": "http_exception",
            "status_code": 400,
            "error": "invalid_request",
        }
    ]

    verdict = provider._residual_chain_verdict(
        request, _chain("a", "b", "c"), attempts, {}
    )

    assert verdict is not None
    assert verdict["terminal"] == ["a"]


@pytest.mark.parametrize(
    "attempt_status_code",
    [500, 429, 408, None],
)
def test_a_transient_attempt_keeps_the_bounded_hold(attempt_status_code):
    """AC1 constraint 1: a 5xx/429/timeout/absent status is not terminal."""
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "b", "usage_limit_reset_pending")
    attempts = [
        {
            "provider": "a",
            "type": "remote",
            "status": "http_exception" if attempt_status_code else "connection_error",
            "status_code": attempt_status_code,
        }
    ]

    assert provider._residual_chain_verdict(request, _chain("a", "b"), attempts, {}) is None


def test_a_cooldown_skip_keeps_the_bounded_hold():
    """AC1 constraint 1: cooldowns stay transient — the hold can heal them."""
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "a", "usage_limit_reset_pending")
    provider._note_chain_resolution_skip(request, "b", "provider_cooldown")

    assert provider._residual_chain_verdict(request, _chain("a", "b"), [], {}) is None


def test_an_unknown_skip_reason_defaults_to_transient():
    """AC1: a new reason must never silently enable the fail-fast path."""
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "a", "usage_limit_reset_pending")
    provider._note_chain_resolution_skip(request, "b", "some_future_reason")

    assert provider._residual_chain_verdict(request, _chain("a", "b"), [], {}) is None
    assert "some_future_reason" not in provider.DETERMINISTIC_CHAIN_SKIP_REASONS


def test_a_derived_skip_is_neutral_not_disqualifying():
    """AC1: a sibling skipped for a shared failure domain does not veto the verdict.

    The root provider's own attempt decides it — here the root is accounted for
    by a terminal refusal, so the derived skip on its sibling is neutral.
    """
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "root", "quota_hard_stop")
    provider._note_chain_resolution_skip(request, "sibling", "failure_domain")

    verdict = provider._residual_chain_verdict(
        request, _chain("root", "sibling"), [], {}
    )

    assert verdict is not None
    assert verdict["skips"]["sibling"] == "failure_domain"


def test_no_recorded_skips_means_no_verdict():
    """The verdict is never inferred from an empty record (conservative)."""
    request = _StubRequest()

    assert provider._residual_chain_verdict(request, _chain("a", "b"), [], {}) is None


def test_earliest_known_reset_wins_over_the_maximum():
    """AC1: the advertised wait is the EARLIEST reset, not the slowest one."""
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "soon", "usage_limit_reset_pending")
    provider._note_chain_resolution_skip(request, "far", "outside_available_window")
    model_config = _chain("soon", "far")

    with (
        patch("proxy.provider._usage_reset_remaining", lambda *_: 90.0),
        patch("proxy.provider._seconds_until_next_window", lambda *a, **k: 900000.0),
    ):
        verdict = provider._residual_chain_verdict(request, model_config, [], {})

    assert verdict is not None
    assert verdict["retry_after_seconds"] == 90, (
        "the earliest known reset must win: advertising ~10 days would park the "
        "client far past the moment capacity returns"
    )


def test_unknown_reset_leaves_the_timing_to_the_exhaustion_payload():
    """AC1: an underivable instant is not a licence to invent one."""
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "a", "quota_hard_stop")

    with patch("proxy.provider._quota_hard_stop_wait_seconds", lambda *a, **k: 0.0):
        verdict = provider._residual_chain_verdict(request, _chain("a"), [], {})

    assert verdict is not None
    assert verdict["retry_after_seconds"] is None


# ======================================================================
# AC1: no hold, and honest retry timing on the preserved error
# ======================================================================


def test_stamped_verdict_refuses_the_hold():
    """AC1: a stamped verdict makes ``_bounded_hold_seconds`` return ``None``."""
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "a", "usage_limit_reset_pending")
    verdict = provider._residual_chain_verdict(request, _chain("a"), [], {})
    provider._set_chain_fail_fast_verdict(request, verdict)

    assert provider._bounded_hold_seconds(300, _exhaustion_response(), 3, request) is None
    # An unstamped request keeps today's bounded hold byte-identically.
    assert (
        provider._bounded_hold_seconds(
            300, _exhaustion_response(retry_after=600), 3, _StubRequest()
        )
        == 300.0
    ), "600 s is inside the 900 s budget, so the hold is bounded by hold_seconds"
    assert (
        provider._bounded_hold_seconds(300, _exhaustion_response(), 3, None) == 300.0
    ), "an unknown retry_after keeps the legacy full hold"


def test_retry_timing_is_attached_to_the_preserved_error():
    """AC1: keep the first provider's 400, add header + body ``retry_after``."""
    original = Response(
        content=json.dumps(
            {"error": {"type": "backend_error", "code": "backend_error"}, "status": 400}
        ).encode("utf-8"),
        status_code=400,
        media_type="application/json",
    )

    timed = provider._with_retry_timing(original, 90, {"server": {}})

    assert timed.status_code == 400, "the real provider status is preserved"
    assert timed.headers["Retry-After"] == "90"
    assert _body(timed)["retry_after"] == 90
    assert _body(timed)["error"]["code"] == "backend_error", "body is preserved"


def test_retry_timing_respects_the_advertisement_cap():
    """AC1 constraint: ``max_retry_after_seconds`` still bounds the advertisement."""
    timed = provider._with_retry_timing(
        _exhaustion_response(), 86400, {"server": {"max_retry_after_seconds": 120}}
    )

    assert timed.headers["Retry-After"] == "120"
    assert _body(timed)["retry_after"] == 120


def test_retry_timing_is_a_no_op_without_a_known_reset():
    """A preserved error with no derivable reset keeps its bytes and headers."""
    original = _exhaustion_response(retry_after=None)

    assert provider._with_retry_timing(original, None, {}) is original


@pytest.mark.asyncio
async def test_run_chain_cycles_does_not_hold_a_provably_dead_chain(monkeypatch):
    """AC1: a fully-unavailable chain consumes no hold cycle and returns at once."""
    request = _StubRequest()
    model_config = _chain("a", "b")
    config = {
        "server": {
            "chain_hold_seconds": 0.05,
            "chain_hold_max_cycles": 3,
            "chain_hold_total_seconds": 720,
        }
    }
    starts: list[tuple] = []
    monkeypatch.setattr(
        provider, "_log_chain_hold_start", lambda *args: starts.append(args)
    )
    exhaustion = _exhaustion_response()

    async def cycle_fn(req, path, mdl, cfg):
        provider._note_chain_resolution_skip(req, "a", "usage_limit_reset_pending")
        provider._note_chain_resolution_skip(req, "b", "outside_available_window")
        verdict = provider._residual_chain_verdict(req, mdl, [])
        provider._set_chain_fail_fast_verdict(req, verdict)
        raise provider.ChainExhaustedError(exhaustion)

    result = await provider._run_chain_cycles(
        request, "v1/chat/completions", model_config, config, cycle_fn
    )

    assert result is exhaustion, "the exhaustion response is returned unchanged"
    assert starts == [], "no hold may start for a provably dead chain"
    assert provider._get_chain_hold_consumed(request) == 0.0
    assert provider._get_chain_hold_cycle(request) == 0


@pytest.mark.asyncio
async def test_run_chain_cycles_still_holds_a_transient_chain(monkeypatch):
    """AC1 constraint 1: the transient shape keeps today's bounded hold."""
    request = _StubRequest()
    model_config = _chain("a")
    config = {
        "server": {
            "chain_hold_seconds": 0.01,
            "chain_hold_max_cycles": 1,
            "chain_hold_total_seconds": 0,
        }
    }
    starts: list[tuple] = []
    monkeypatch.setattr(
        provider, "_log_chain_hold_start", lambda *args: starts.append(args)
    )
    calls = {"n": 0}

    async def cycle_fn(req, path, mdl, cfg):
        calls["n"] += 1
        provider._note_chain_resolution_skip(req, "a", "provider_cooldown")
        verdict = provider._residual_chain_verdict(req, mdl, [])
        provider._set_chain_fail_fast_verdict(req, verdict)
        # ``retry_after=0`` is the documented "retry immediately" case: the
        # hold still happens (the chain may have healed), it just costs no
        # wall-clock time.
        raise provider.ChainExhaustedError(_exhaustion_response(retry_after=0))

    await provider._run_chain_cycles(
        request, "v1/chat/completions", model_config, config, cycle_fn
    )

    assert starts, "a cooldown-skipped chain must still be held so it can heal"
    assert calls["n"] == 2, "one pre-hold cycle plus one hold-retry cycle"


# ======================================================================
# AC3: one chain-end vocabulary for fail-fast and hold
# ======================================================================


def test_chain_end_line_distinguishes_fail_fast_from_a_hold(caplog):
    """AC3: ``chain_planned_unavailable`` carries an explicit hold marker."""
    verdict = {
        "skips": {"a": "usage_limit_reset_pending", "b": "outside_available_window"},
        "terminal": ["c"],
        "providers": 3,
        "retry_after_seconds": 90,
    }

    with caplog.at_level("INFO", logger="llama-proxy"):
        provider._log_chain_planned_unavailable("v1/chat/completions", verdict)

    line = next(
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("chain_planned_unavailable")
    )
    assert "reason=deterministic_residual_chain" in line
    assert "hold=skipped" in line
    assert "retry_after=90" in line
    assert "a:usage_limit_reset_pending" in line
    assert "terminal=c" in line
    assert "holding" not in line


# ======================================================================
# AC2: what "a remote provider has capacity" means
# ======================================================================


def test_remote_capacity_ignores_a_deterministically_skipped_entry():
    """AC2: an entry already skipped for a deterministic reason has no capacity."""
    request = _StubRequest()
    provider._note_chain_resolution_skip(request, "dead", "usage_limit_reset_pending")

    with (
        patch("proxy.provider._provider_in_allowed_window", lambda *a, **k: True),
        patch("proxy.provider._usage_reset_remaining", lambda *_: 0.0),
        patch("proxy.provider._quota_exhausted", lambda *a, **k: (False, "")),
        patch("proxy.provider._quota_unavailable_reason", lambda *a, **k: None),
        patch("proxy.provider._provider_cooldown_remaining", lambda *_: 0),
    ):
        assert provider._remote_chain_has_capacity(request, _chain("dead"), {}) is False


def test_remote_capacity_is_true_for_a_ready_entry():
    """AC2: one healthy remote entry is enough."""
    with (
        patch("proxy.provider._provider_in_allowed_window", lambda *a, **k: True),
        patch("proxy.provider._usage_reset_remaining", lambda *_: 0.0),
        patch("proxy.provider._quota_exhausted", lambda *a, **k: (False, "")),
        patch("proxy.provider._quota_unavailable_reason", lambda *a, **k: None),
        patch("proxy.provider._provider_cooldown_remaining", lambda *_: 0),
    ):
        assert provider._remote_chain_has_capacity(_StubRequest(), _chain("ready"), {}) is True


@pytest.mark.parametrize(
    "patch_target, patch_value",
    [
        ("_provider_in_allowed_window", lambda *a, **k: False),
        ("_provider_cooldown_remaining", lambda *_: 120),
        ("_usage_reset_remaining", lambda *_: 3600.0),
        ("_quota_exhausted", lambda *a, **k: (True, "quota_exhausted")),
        ("_quota_unavailable_reason", lambda *a, **k: "quota_exhausted"),
    ],
)
def test_remote_capacity_is_false_for_every_unavailable_state(patch_target, patch_value):
    """AC2: the probe uses the resolution predicate's own vocabulary."""
    base = {
        "_provider_in_allowed_window": lambda *a, **k: True,
        "_provider_cooldown_remaining": lambda *_: 0,
        "_usage_reset_remaining": lambda *_: 0.0,
        "_quota_exhausted": lambda *a, **k: (False, ""),
        "_quota_unavailable_reason": lambda *a, **k: None,
    }
    base[patch_target] = patch_value
    with (
        patch("proxy.provider._provider_in_allowed_window", base["_provider_in_allowed_window"]),
        patch("proxy.provider._provider_cooldown_remaining", base["_provider_cooldown_remaining"]),
        patch("proxy.provider._usage_reset_remaining", base["_usage_reset_remaining"]),
        patch("proxy.provider._quota_exhausted", base["_quota_exhausted"]),
        patch("proxy.provider._quota_unavailable_reason", base["_quota_unavailable_reason"]),
    ):
        assert provider._remote_chain_has_capacity(_StubRequest(), _chain("x"), {}) is False


def test_a_local_only_chain_has_no_remote_capacity():
    """AC2: with no remote provider there is no remote chain to bounce to."""
    model_config = {"providers": [{"name": "local-llama", "type": "local"}]}

    assert provider._remote_chain_has_capacity(_StubRequest(), model_config, {}) is False


def test_remote_capacity_ignores_the_requested_provider():
    """AC2: the producer of the refusal is not counted as its destination."""
    with (
        patch("proxy.provider._provider_in_allowed_window", lambda *a, **k: True),
        patch("proxy.provider._usage_reset_remaining", lambda *_: 0.0),
        patch("proxy.provider._quota_exhausted", lambda *a, **k: (False, "")),
        patch("proxy.provider._quota_unavailable_reason", lambda *a, **k: None),
        patch("proxy.provider._provider_cooldown_remaining", lambda *_: 0),
    ):
        assert (
            provider._remote_chain_has_capacity(
                _StubRequest(), _chain("self"), {}, exclude_name="self"
            )
            is False
        )
