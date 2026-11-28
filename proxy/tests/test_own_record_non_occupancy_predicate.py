"""Own-record non-occupancy predicate, keyed on the streaming state.

Parent LP-0MWMSQR3O002YNPW (AC1, AC2, AC3); implementation item
LP-0MWNHBG7D002NH5S.

The routing concurrency gate must not count the requesting session's own record
whenever **none of that session's dispatches is still streaming upstream**. That
single rule subsumes the two shapes the gate has to admit:

- an **idle lease** — ``active: False`` (LP-0MW0VH2LB008JSKQ);
- **finished-but-not-yet-released** — ``active: True`` with an empty
  ``streaming_dispatch_tokens`` set: the terminal chunk has been marked
  (LP-0MWNHBFN6000GH2J) but teardown has not released the credit.

It must keep counting genuinely in-flight work — the same session's record
holding at least one streaming token (AC2) — and it must stay a pure read
(AC5). Records **without** the marker predate it: they keep the pre-change
``active``-only behaviour, so nothing changes for legacy/hand-built records.

Pinned here:

- AC1 — the predicate itself: idle lease, finished-but-not-released, and the
  alias that keeps the pre-change name working.
- AC2 — a streaming own record counts; a foreign session's record is never
  excluded, whatever its streaming state.
- AC3 — ``session_id=None`` / omitted excludes nothing.
- AC4 — both routing-gate call sites thread the requesting session into the
  occupancy read (AC by AST, so a new anonymous gate call fails).
- Legacy fallback — a record with no ``streaming_dispatch_tokens`` key.
"""

from __future__ import annotations

import ast
import inspect
import time

import proxy.provider as provider
import proxy.server as server
import pytest

ENDPOINT = "http://localhost:8080"
POOL = 2
ME = "session-me"
OTHER = "session-other"

ME_TOKEN = 9201
OTHER_TOKEN = 9202


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _config(**server_overrides) -> dict:
    server_cfg = {"llama_server_port": 8080, "session_slot_pool_size": POOL}
    server_cfg.update(server_overrides)
    return {"server": server_cfg, "provider_cooldown_seconds": 60}


def _record(
    *,
    active: bool = True,
    streaming: list[int] | None = None,
    credits: list[int] | None = None,
    expires_in: float = 60.0,
) -> dict:
    """A dispatch record in the production shape.

    *streaming* ``None`` builds a **legacy** record with no
    ``streaming_dispatch_tokens`` key at all (the pre-marker shape).
    """
    now = time.monotonic()
    record = {
        "backend": "local",
        "started_at": now,
        "active": active,
        "expires_at": now + expires_in,
        "model_name": "Qwen3",
        "endpoint": ENDPOINT,
        "session": ME,
    }
    if credits is not None:
        record["dispatch_credits"] = list(credits)
        record["dispatch_token"] = credits[0] if credits else None
    if streaming is not None:
        record["streaming_dispatch_tokens"] = list(streaming)
    return record


@pytest.fixture
def install_records(monkeypatch):
    """Install a dispatch registry on the ``proxy.server`` module state."""

    def _install(records: dict, *, generating: int = 0) -> dict:
        monkeypatch.setattr(server, "local_dispatch_records", records, raising=False)
        monkeypatch.setattr(server, "local_generating_queries", generating, raising=False)
        monkeypatch.setattr(server, "local_active_queries", 0, raising=False)
        return records

    return _install


def _predicate(record: dict, session_id: str | None = ME, key=None) -> bool:
    return provider._is_own_record_non_occupancy(
        key if key is not None else (ENDPOINT, session_id), record, session_id
    )


# --------------------------------------------------------------------------
# AC1 — the two non-occupying own shapes
# --------------------------------------------------------------------------


def test_own_idle_lease_is_non_occupancy():
    """AC1: ``active: False`` with no streaming token is the requester's own."""
    assert _predicate(_record(active=False, streaming=[])) is True


def test_own_finished_but_not_released_record_is_non_occupancy():
    """AC1: ``active: True`` + empty streaming set is finished, not in flight.

    This is the shape teardown has not reached yet: the credit is still held
    (``dispatch_credits`` still lists the token) and ``active`` is still true,
    only the streaming marker says the turn finished upstream.
    """
    record = _record(active=True, streaming=[], credits=[ME_TOKEN])

    assert record["dispatch_credits"] == [ME_TOKEN]
    assert _predicate(record) is True


def test_finished_record_with_empty_list_is_non_occupancy_for_plain_key():
    """AC1: the legacy plain-key form obeys the same rule."""
    assert _predicate(_record(active=True, streaming=[]), key=ME) is True


def test_pre_change_predicate_name_is_an_alias():
    """AC1: the renamed predicate keeps the old name callable (compat alias)."""
    assert provider._is_own_idle_lease is provider._is_own_record_non_occupancy


# --------------------------------------------------------------------------
# AC2 — genuine in-flight work keeps counting
# --------------------------------------------------------------------------


def test_own_streaming_record_is_occupancy():
    """AC2: one streaming token ⇒ the own record still counts."""
    assert _predicate(_record(streaming=[ME_TOKEN], credits=[ME_TOKEN])) is False


def test_own_streaming_record_counts_even_with_a_finished_sibling_dispatch():
    """AC2: a concurrent second dispatch keeps the whole record counting."""
    record = _record(active=True, streaming=[OTHER_TOKEN], credits=[ME_TOKEN, OTHER_TOKEN])

    assert _predicate(record) is False


def test_foreign_session_record_is_never_excluded():
    """AC2/AC1: another session's record counts whatever its streaming state.

    The predicate is *own*-record scoped: widening it to any finished record
    would hide real concurrency.
    """
    finished = _record(active=True, streaming=[])

    assert _predicate(finished, session_id=ME, key=(ENDPOINT, OTHER)) is False
    assert _predicate(finished, session_id=ME, key=OTHER) is False


# --------------------------------------------------------------------------
# AC3 — anonymous callers exclude nothing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("session_id", [None, ""])
def test_no_requester_excludes_nothing(install_records, session_id):
    """AC3: ``session_id`` of ``None``/empty excludes no record at all."""
    records = install_records(
        {
            (ENDPOINT, ME): _record(active=False, streaming=[]),
            (ENDPOINT, OTHER): _record(active=False, streaming=[]),
        }
    )

    assert _predicate(records[(ENDPOINT, ME)], session_id=session_id) is False
    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=session_id
    ) == (POOL, POOL)


def test_omitted_session_keeps_the_count(install_records):
    """AC3: the pool-description readers (no session) see today's count."""
    install_records({(ENDPOINT, ME): _record(active=True, streaming=[])})

    assert provider._get_local_concurrency_info(_config(), endpoint=ENDPOINT) == (
        1,
        POOL,
    )


# --------------------------------------------------------------------------
# Legacy fallback — no streaming marker at all
# --------------------------------------------------------------------------


def test_legacy_record_without_the_marker_keeps_the_active_only_rule():
    """Legacy record: no ``streaming_dispatch_tokens`` key ⇒ ``active`` decides.

    Fallback is deliberately *not* "finished because we cannot see a stream":
    a legacy record with no marker and ``active: True`` might be mid-stream, so
    it keeps counting exactly as it did before this slice.
    """
    legacy_active = _record(active=True)
    legacy_idle = _record(active=False)

    assert "streaming_dispatch_tokens" not in legacy_active
    assert _predicate(legacy_active) is False
    assert _predicate(legacy_idle) is True


def test_non_list_marker_is_treated_as_legacy():
    """A malformed marker degrades to the legacy rule, never to "finished"."""
    record = _record(active=True)
    record["streaming_dispatch_tokens"] = None

    assert _predicate(record) is False


# --------------------------------------------------------------------------
# Counts through the gate
# --------------------------------------------------------------------------


def test_own_finished_record_does_not_saturate_the_pool(install_records):
    """AC1: own finished record + one genuinely streaming session reads 1/2."""
    install_records(
        {
            (ENDPOINT, ME): _record(active=True, streaming=[], credits=[ME_TOKEN]),
            (ENDPOINT, OTHER): _record(
                active=True, streaming=[OTHER_TOKEN], credits=[OTHER_TOKEN]
            ),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, POOL)


def test_two_other_sessions_still_saturate_the_pool(install_records):
    """AC2: the exclusion never weakens genuine occupancy elsewhere."""
    install_records(
        {
            (ENDPOINT, ME): _record(active=True, streaming=[], credits=[ME_TOKEN]),
            (ENDPOINT, OTHER): _record(active=True, streaming=[OTHER_TOKEN]),
            (ENDPOINT, "session-third"): _record(active=True, streaming=[3]),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (2, POOL)


def test_mixed_legacy_and_marked_records(install_records):
    """A legacy own record and a marked own record are read independently."""
    install_records(
        {
            (ENDPOINT, ME): _record(active=False),
            (ENDPOINT, OTHER): _record(active=True, streaming=[]),
        }
    )

    assert provider._get_local_concurrency_info(
        _config(), endpoint=ENDPOINT, session_id=ME
    ) == (1, POOL)


# --------------------------------------------------------------------------
# AC5 — the predicate is a read
# --------------------------------------------------------------------------


def test_predicate_is_read_only():
    """AC5: the predicate never mutates the record it inspects."""
    record = _record(active=True, streaming=[ME_TOKEN], credits=[ME_TOKEN])
    before = {k: (list(v) if isinstance(v, list) else v) for k, v in record.items()}

    _predicate(record)

    assert record == before


# --------------------------------------------------------------------------
# AC4 — both routing-gate call sites thread the session
# --------------------------------------------------------------------------

#: The routing cycle that owns the two local-admission gate call sites
#: (``provider.py`` — the concurrency-limit gate and the lease-denial retry).
ROUTING_CYCLE = "_proxy_with_fallback_cycle"


def _occupancy_calls(function_name: str) -> list[ast.Call]:
    tree = ast.parse(inspect.getsource(provider))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == function_name
        ):
            calls = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and (
                    sub.func.id in {"_local_concurrency_info", "_get_local_concurrency_info"}
                ):
                    calls.append(sub)
            return calls
    raise AssertionError(f"{function_name} not found in proxy.provider")


def test_both_routing_gate_sites_thread_the_session():
    """AC4: every occupancy read on the routing path passes ``session_id``.

    The extended predicate is reachable only through the ``session_id``
    keyword, so an anonymous gate call silently reverts to counting the
    requester's own finished record. Pinned structurally: a new anonymous
    ``_local_concurrency_info(...)`` call inside the routing cycle fails here.
    """
    calls = _occupancy_calls(ROUTING_CYCLE)

    assert len(calls) == 2, (
        "the routing cycle owns exactly two occupancy reads "
        f"(local concurrency gate + lease-denial retry), found {len(calls)}"
    )
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "session_id" in keywords, (
            f"occupancy read at provider.py:{call.lineno} drops the requesting "
            "session, reverting to counting its own finished record"
        )
        session_kw = next(kw for kw in call.keywords if kw.arg == "session_id")
        assert isinstance(session_kw.value, ast.Name) and (
            session_kw.value.id == "_session_id"
        ), f"provider.py:{call.lineno} must thread the request's own session id"


def test_gate_adjacent_occupancy_reads_stay_anonymous():
    """AC4/AC5: the admission guard's occupancy read deliberately has no session.

    ``_prefill_admission_guard`` reads occupancy anonymously so the own-record
    exclusion cannot change its verdicts (LP-0MWNHCJDS006IDGN); threading the
    session there would be a behaviour change outside this slice.
    """
    for call in _occupancy_calls("_local_gate_occupancy") + _occupancy_calls(
        "_local_slot_saturation"
    ):
        keywords = {kw.arg for kw in call.keywords}
        assert "session_id" not in keywords, (
            f"provider.py:{call.lineno}: the guard's occupancy read must stay "
            "anonymous — the own-record predicate is routing-only"
        )
