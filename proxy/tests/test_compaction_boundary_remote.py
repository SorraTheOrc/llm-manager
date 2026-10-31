"""Local→remote boundary compaction contract — test-first specification.

Feature: LP-0MVRBQBDF0037OZ5 ("Compact before moving from local to remote").
This module: LP-0MVTDU4P8001L4II — the TDD-red contract for the boundary
decision (LP-0MVTDU6Y1004ZCGX), the slot pin plus immediate lease release
(LP-0MVTDU91C003Q9YZ) and the dispatch hook that consumes them
(LP-0MVTDUBR00038ZRL), plus LP-0MVTDUEF5007A1SP — the end-to-end integration
and default-off regression tests at the bottom of the file.

The point of this module is that the *trigger*, the *slot pin* and the
*release ordering* were specified by executable assertions before any
production code existed: every test here failed only because the boundary
symbols were absent (``missing boundary-compaction symbol …``), never because
of an unrelated import error, and it turned green only as each child landed.

Contract pinned by this module
------------------------------

1. Boundary decision — ``proxy.compaction.decide_boundary_compaction``::

       decide_boundary_compaction(
           *, provider_type, has_local_provenance, estimated_tokens, config
       ) -> {"should_compact", "reason", "est_before", "guard_tokens", "enabled"}

   Fires **only** for (``provider_type == "remote"`` ∧
   ``has_local_provenance`` ∧ ``estimated_tokens > guard_tokens`` ∧ the
   ``server.remote_boundary_compaction`` knob enabled). The guard is resolved
   from ``server.remote_boundary_compaction_min_tokens`` (default 20 000) and
   the comparison is strict — an estimate *equal* to the guard does not fire.

   Provenance is a recorded-local-turn signal from the session store, never a
   live slot assignment, so a released / evicted / save-restored slot does not
   disqualify a session.

2. Config resolution — ``proxy.provider.compaction_config`` carries
   ``remote_boundary_compaction`` (bool, default ``False``) and
   ``remote_boundary_compaction_min_tokens`` (int, default ``20000``). The
   knob is independent of the assembly-time ``compaction_trigger_ratio``,
   which is neither read nor re-enabled by it.

3. Slot-pinned summariser —
   ``proxy.compaction_summarizer.build_slot_pinned_summarizer``::

       build_slot_pinned_summarizer(config, slot_id, llama_port=8080) -> Summarizer

   The summariser POSTs to the **local** llama-server chat-completions
   endpoint and the request body carries ``id_slot == slot_id`` — the session
   reuses its own disposable slot. It must not use the dedicated summariser
   model (``server.summarizer_model.llama_model``): slot ids are per
   llama-server instance, so the session's own local model is the target.

4. Slot lookup + release —
   ``proxy.router_helpers.summarise_on_session_slot``::

       summarise_on_session_slot(
           srv, session_id, messages, *, config=None,
           summarizer_builder=None, endpoint=None
       ) -> {"slot_id", "summary", "reason", "released"}

   * The assigned slot comes from the **read-only**
     ``proxy.session._assigned_slot_for_session`` (looked up at call time, so
     this module monkeypatches ``proxy.session``).
   * ``summarizer_builder`` is called **once** with the resolved slot id and
     returns the ``Summarizer``; production defaults to
     ``build_slot_pinned_summarizer(config, slot_id, llama_port=…)``.
   * ``_release_local_dispatch`` runs **after** the summariser returns
     (success, failure or timeout) and **before** the helper returns — so the
     release always precedes the remote stream the dispatcher starts next.
   * No assigned slot → the summariser is never called or built, the skip is
     reported as ``boundary_no_slot`` and the release still runs (no stale
     lease).

Machine-readable reason tokens (parent AC4 — logged at WARNING by the hook)::

    boundary_compact            fired (decision + helper success)
    boundary_disabled           knob ``server.remote_boundary_compaction`` off
    boundary_not_remote         provider about to be dispatched is not remote
    boundary_no_local_provenance  no recorded locally-served turn
    boundary_below_guard        estimate at or below the minimum-size guard
    boundary_no_slot            no assigned local slot for the session
    boundary_slot_unavailable   the assigned slot could not be reached at all
                                (transport/connection-level failure)
    boundary_summarizer_failed  the summariser ran and failed open
                                (blank / ``EmptySummary`` / any other raise)

TDD-red convention (retired)
----------------------------

The contract was pinned with ``pytest.mark.xfail(strict=True)`` — the
established test-first convention in this suite
(``test_local_only_refusal_matrix.py``, ``test_compaction_headers_contract.py``,
``test_local_resident_prefix_admission.py``). Every marker has since been
flipped off with the behaviour it pinned: LP-0MVTDU6Y1004ZCGX (decision and
config plus ``TestLocalProvenance``, ``TestBoundaryConfigValidation`` and
``TestLocalCompletionProvenanceWiring``), LP-0MVTDU91C003Q9YZ
(``TestSlotPinnedSummariser``, ``TestSlotReleaseOrdering``) and
LP-0MVTDUBR00038ZRL (``TestBoundaryCompactionHook``,
``TestBoundaryCompactionDispatchWiring``,
``TestBoundaryHookDirectRemoteDispatch``). No ``xfail`` marker remains, so a
red row is a real failure.

The one config assertion that already holds (``compaction_trigger_ratio`` stays
``0`` — the boundary knob never re-enables assembly-time compaction) is
**not** marked: it is the regression guard that stops the knob being folded
into the assembly trigger.

Constraints honoured here: injected fakes and stubbed ``httpx`` transports
only — no live llama-server, GPU or network access, no sleeps, deterministic.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import proxy.provider as provider
import proxy.router_helpers as router_helpers
import pytest
from fastapi import Response

# Minimum-size guard default agreed in LP-0MVTDU6Y1004ZCGX (guarded by the
# ``remote_boundary_compaction_min_tokens`` knob).
_DEFAULT_GUARD_TOKENS = 20000
# The dedicated summariser model, which the slot-pinned builder must NOT use.
_DEDICATED_SUMMARISER_MODEL = "Qwen2.5-7B"


def _require(module_name: str, symbol: str):
    """Return *symbol* from *module_name*, failing the test when it is absent.

    Importing lazily keeps this module collecting cleanly while the boundary
    contract is unimplemented, so the red state is one clear assertion per
    test instead of a collection error.
    """
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol)
    except AttributeError:
        pytest.fail(
            f"missing boundary-compaction symbol {module_name}.{symbol} — "
            "contract not implemented yet (TDD-red by design, LP-0MVTDU4P8001L4II)"
        )


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _boundary_config(
    *, enabled: bool = True, min_tokens: int | None = None, **server_overrides
) -> dict:
    """Proxy config with the boundary knob set (and the guard optionally so)."""
    server: dict = {
        "llama_server_port": 8080,
        "compaction_trigger_ratio": 0,
        "remote_boundary_compaction": enabled,
        "summarizer_model": {"type": "local", "llama_model": _DEDICATED_SUMMARISER_MODEL},
    }
    if min_tokens is not None:
        server["remote_boundary_compaction_min_tokens"] = min_tokens
    server.update(server_overrides)
    return {
        "server": server,
        # The session's own local model — the slot is that model's instance.
        "models": {
            "plan": {
                "providers": [
                    {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"}
                ]
            }
        },
    }


def _decide(**kwargs) -> dict:
    decide = _require("proxy.compaction", "decide_boundary_compaction")
    return decide(**kwargs)


def _messages(turns: int) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "SYS"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def _fake_srv(session_id: str) -> SimpleNamespace:
    """Minimal srv exposing the lease registry ``_release_local_dispatch`` uses."""
    return SimpleNamespace(
        local_dispatch_records={
            router_helpers._dispatch_lease_key(None, session_id): {
                "session_id": session_id
            }
        },
        local_dispatch_records_lock=asyncio.Lock(),
        local_prefill_in_flight={},
        logger=logging.getLogger("test.boundary.compaction"),
        config=_boundary_config(),
    )


class _SlotEnv(SimpleNamespace):
    """Recorded boundary events plus the mutable fake slot assignment."""

    events: list[tuple]
    slot: int | None

    def kinds(self) -> list[str]:
        return [event[0] for event in self.events]

    def order(self, kind: str) -> int:
        return self.kinds().index(kind)


@pytest.fixture
def slot_env(monkeypatch) -> _SlotEnv:
    """Wire the read-only slot lookup and record every lease release.

    The real ``_release_local_dispatch`` still runs, so the "no leaked lease"
    assertions are genuine rather than a mocked no-op.
    """
    import proxy.session as session_mod

    env = _SlotEnv(events=[], slot=3)

    def fake_assigned_slot(session_id, endpoint=None):
        env.events.append(("slot_lookup", session_id))
        return env.slot

    monkeypatch.setattr(session_mod, "_assigned_slot_for_session", fake_assigned_slot)

    real_release = router_helpers._release_local_dispatch

    async def recording_release(srv, session_id, request=None, endpoint=None):
        env.events.append(("release", session_id))
        return await real_release(
            srv, session_id, request=request, endpoint=endpoint
        )

    monkeypatch.setattr(router_helpers, "_release_local_dispatch", recording_release)
    return env


def _helper():
    return _require("proxy.router_helpers", "summarise_on_session_slot")


def _ok_response(content: str = "SUMMARY TEXT", status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


# ---------------------------------------------------------------------------
# AC1 — the boundary decision fires only on the agreed gate
# ---------------------------------------------------------------------------


class TestBoundaryDecision:
    """`decide_boundary_compaction` — remote + provenance + above guard + knob."""

    def test_fires_for_remote_provider_with_local_provenance_above_guard(self):
        """The one firing case, with the full result shape asserted."""
        out = _decide(
            provider_type="remote",
            has_local_provenance=True,
            estimated_tokens=90000,
            config=_boundary_config(),
        )

        assert out["should_compact"] is True
        assert out["reason"] == "boundary_compact"
        assert out["est_before"] == 90000
        assert out["guard_tokens"] == _DEFAULT_GUARD_TOKENS
        assert out["enabled"] is True

    def test_does_not_fire_without_recorded_local_provenance(self):
        """Remote-only models (and local never used) are unchanged."""
        out = _decide(
            provider_type="remote",
            has_local_provenance=False,
            estimated_tokens=90000,
            config=_boundary_config(),
        )

        assert out["should_compact"] is False
        assert out["reason"] == "boundary_no_local_provenance"

    def test_does_not_fire_for_a_non_remote_provider(self):
        """A local dispatch is never compacted at the boundary."""
        out = _decide(
            provider_type="local",
            has_local_provenance=True,
            estimated_tokens=90000,
            config=_boundary_config(),
        )

        assert out["should_compact"] is False
        assert out["reason"] == "boundary_not_remote"

    @pytest.mark.parametrize("estimated_tokens", [0, 1, _DEFAULT_GUARD_TOKENS - 1, _DEFAULT_GUARD_TOKENS])
    def test_does_not_fire_at_or_below_the_size_guard(self, estimated_tokens):
        """Trivial sessions are left alone; the guard comparison is strict."""
        out = _decide(
            provider_type="remote",
            has_local_provenance=True,
            estimated_tokens=estimated_tokens,
            config=_boundary_config(),
        )

        assert out["should_compact"] is False
        assert out["reason"] == "boundary_below_guard"
        assert out["guard_tokens"] == _DEFAULT_GUARD_TOKENS
        assert out["est_before"] == estimated_tokens

    def test_fires_just_above_the_size_guard(self):
        """One token over the guard is enough: the boundary is not off-by-one."""
        out = _decide(
            provider_type="remote",
            has_local_provenance=True,
            estimated_tokens=_DEFAULT_GUARD_TOKENS + 1,
            config=_boundary_config(),
        )

        assert out["should_compact"] is True
        assert out["reason"] == "boundary_compact"

    def test_guard_defaults_to_twenty_thousand_tokens(self):
        """No guard key → the agreed 20 000-token default."""
        cfg = _boundary_config()  # no guard key → default
        assert "remote_boundary_compaction_min_tokens" not in cfg["server"]

        out = _decide(
            provider_type="remote",
            has_local_provenance=True,
            estimated_tokens=90000,
            config=cfg,
        )

        assert out["guard_tokens"] == _DEFAULT_GUARD_TOKENS

    @pytest.mark.parametrize(
        ("min_tokens", "estimated_tokens", "should_compact"),
        [(5000, 5000, False), (5000, 5001, True), (120000, 119999, False)],
    )
    def test_guard_is_configurable(self, min_tokens, estimated_tokens, should_compact):
        """The operator can tune the minimum-size guard per deployment."""
        out = _decide(
            provider_type="remote",
            has_local_provenance=True,
            estimated_tokens=estimated_tokens,
            config=_boundary_config(min_tokens=min_tokens),
        )

        assert out["guard_tokens"] == min_tokens
        assert out["should_compact"] is should_compact

    def test_does_not_fire_when_the_knob_is_disabled(self):
        """Default-off: the knob alone suppresses the whole boundary path."""
        out = _decide(
            provider_type="remote",
            has_local_provenance=True,
            estimated_tokens=90000,
            config=_boundary_config(enabled=False),
        )

        assert out["should_compact"] is False
        assert out["reason"] == "boundary_disabled"
        assert out["enabled"] is False


# ---------------------------------------------------------------------------
# AC2 — the independent, default-off knob and its guard resolve in config
# ---------------------------------------------------------------------------


class TestBoundaryConfig:
    """`compaction_config` resolves the boundary knob independently."""

    def test_defaults_to_disabled_with_the_agreed_guard(self):
        from proxy.provider import compaction_config

        cfg = compaction_config({"server": {}})

        assert cfg["remote_boundary_compaction"] is False
        assert cfg["remote_boundary_compaction_min_tokens"] == _DEFAULT_GUARD_TOKENS

    def test_reads_explicit_knob_and_guard_values(self):
        from proxy.provider import compaction_config

        cfg = compaction_config(
            {
                "server": {
                    "remote_boundary_compaction": True,
                    "remote_boundary_compaction_min_tokens": 5000,
                }
            }
        )

        assert cfg["remote_boundary_compaction"] is True
        assert cfg["remote_boundary_compaction_min_tokens"] == 5000


class TestBoundaryKnobIndependence:
    """Green today: the regression guard for AC5's independence clause."""

    def test_does_not_re_enable_assembly_time_compaction(self):
        """The boundary knob is independent of `compaction_trigger_ratio`.

        Deliberately not ``xfail``-marked: this assertion already holds, and it
        must keep holding once the boundary knob exists — folding the new knob
        into the assembly trigger would silently re-enable prompt-assembly
        compaction (the operator's 2026-09-17 decision).
        """
        from proxy.provider import compaction_config

        cfg = compaction_config(
            {
                "server": {
                    "compaction_trigger_ratio": 0,
                    "remote_boundary_compaction": True,
                }
            }
        )

        assert cfg["trigger_ratio"] == 0.0


# ---------------------------------------------------------------------------
# AC2 — local-served provenance lives in the session store
# ---------------------------------------------------------------------------


class TestLocalProvenance:
    """`SessionManager` records — never infers — local-served provenance."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_fresh_session_has_no_provenance(self):
        """A session that has never been served locally is unchanged."""
        from proxy.session_manager import SessionManager

        mgr = SessionManager()

        async def scenario():
            session, _created = await mgr.get_or_create("sess-fresh")
            return session.locally_served_turns, await mgr.has_local_provenance(
                "sess-fresh"
            )

        turns, has_provenance = self._run(scenario())
        assert turns == 0
        assert has_provenance is False

    def test_marking_a_turn_records_provenance_and_counts_up(self):
        """One local completion is enough; repeated turns accumulate."""
        from proxy.session_manager import SessionManager

        mgr = SessionManager()

        async def scenario():
            await mgr.get_or_create("sess-marked")
            first = await mgr.mark_locally_served("sess-marked")
            after_one = await mgr.has_local_provenance("sess-marked")
            await mgr.mark_locally_served("sess-marked")
            info = mgr.get_session_info("sess-marked")
            return first, after_one, info

        first, after_one, info = self._run(scenario())
        assert first is True
        assert after_one is True
        assert info is not None
        assert info["locally_served_turns"] == 2

    def test_unknown_session_never_gains_provenance(self):
        """Marking an evicted/unknown session is a no-op, not a phantom turn."""
        from proxy.session_manager import SessionManager

        mgr = SessionManager()

        async def scenario():
            marked = await mgr.mark_locally_served("sess-unknown")
            return marked, await mgr.has_local_provenance("sess-unknown")

        marked, has_provenance = self._run(scenario())
        assert marked is False
        assert has_provenance is False

    def test_non_positive_turn_counts_are_rejected(self):
        """A zero/negative increment must not fabricate provenance."""
        from proxy.session_manager import SessionManager

        mgr = SessionManager()

        async def scenario():
            await mgr.get_or_create("sess-nonpositive")
            for bad in (0, -3, "nope"):
                assert await mgr.mark_locally_served("sess-nonpositive", bad) is False
            return await mgr.has_local_provenance("sess-nonpositive")

        assert self._run(scenario()) is False

    def test_provenance_survives_slot_release_and_save_restore(self):
        """Provenance is a session-store fact, never a live slot assignment.

        The boundary decision must still fire for a session whose slot was
        released, evicted or save/restored: those events touch the slot/KV
        machinery, not this recorded counter.
        """
        from proxy.session_manager import SessionManager

        mgr = SessionManager()

        async def scenario():
            await mgr.get_or_create("sess-survivor")
            await mgr.mark_locally_served("sess-survivor")
            # Slot lifecycle churn: history updates, compaction anchors and a
            # save/restore confirmation all run after the local turn.
            await mgr.update_messages(
                "sess-survivor", [{"role": "user", "content": "u"}]
            )
            await mgr.mark_compacted("sess-survivor", from_count=1)
            await mgr.set_restore_confirmed("sess-survivor", True)
            await mgr.set_restore_confirmed("sess-survivor", False)
            return await mgr.has_local_provenance("sess-survivor")

        assert self._run(scenario()) is True

    def test_a_new_session_after_removal_starts_clean(self):
        """Provenance belongs to the session, not to the session id forever."""
        from proxy.session_manager import SessionManager

        mgr = SessionManager()

        async def scenario():
            await mgr.get_or_create("sess-recycled")
            await mgr.mark_locally_served("sess-recycled")
            await mgr.remove("sess-recycled")
            await mgr.get_or_create("sess-recycled")
            return await mgr.has_local_provenance("sess-recycled")

        assert self._run(scenario()) is False


# ---------------------------------------------------------------------------
# AC2 (config half) — startup validation of the knob and its guard
# ---------------------------------------------------------------------------


class TestBoundaryConfigValidation:
    """`validate_compaction_config` rejects an unusable boundary config."""

    def _problems(self, server: dict) -> list[str]:
        from proxy.provider import validate_compaction_config

        return validate_compaction_config({"server": server})

    def test_shipped_disabled_default_passes(self):
        """Absent keys (the default state) are valid."""
        assert self._problems({}) == []

    @pytest.mark.parametrize(
        ("enabled", "guard"), [(True, 20000), (False, 5000), (True, 200000)]
    )
    def test_valid_knob_and_guard_pass(self, enabled, guard):
        problems = self._problems(
            {
                "remote_boundary_compaction": enabled,
                "remote_boundary_compaction_min_tokens": guard,
            }
        )
        assert problems == []

    @pytest.mark.parametrize("bad", ["yes", 1, 0, [True]])
    def test_non_boolean_knob_is_fatal(self, bad):
        """A truthy string/int is a config typo, not an opt-in."""
        problems = self._problems({"remote_boundary_compaction": bad})
        assert any(
            p.startswith("FATAL:") and "remote_boundary_compaction" in p
            for p in problems
        )

    @pytest.mark.parametrize("bad", [0, -1, -20000])
    def test_non_positive_guard_is_fatal(self, bad):
        """A non-positive guard would compact every session: refuse it."""
        problems = self._problems({"remote_boundary_compaction_min_tokens": bad})
        assert any(
            p.startswith("FATAL:")
            and "remote_boundary_compaction_min_tokens" in p
            for p in problems
        )

    @pytest.mark.parametrize("bad", ["20000", 1.5, True])
    def test_non_integer_guard_is_fatal(self, bad):
        problems = self._problems({"remote_boundary_compaction_min_tokens": bad})
        assert any(
            p.startswith("FATAL:")
            and "remote_boundary_compaction_min_tokens" in p
            for p in problems
        )

    def test_validation_does_not_disturb_the_assembly_trigger(self):
        """Validating the boundary keys never re-enables assembly compaction."""
        from proxy.provider import compaction_config

        problems = self._problems(
            {
                "compaction_trigger_ratio": 0,
                "remote_boundary_compaction": True,
            }
        )
        assert problems == []
        assert (
            compaction_config(
                {"server": {"compaction_trigger_ratio": 0}}
            )["trigger_ratio"]
            == 0.0
        )


# ---------------------------------------------------------------------------
# AC2 (wiring half) — the successful local-completion path records provenance
# ---------------------------------------------------------------------------


class TestLocalCompletionProvenanceWiring:
    """`proxy.router` marks provenance only on a successful local completion."""

    def _srv(self, manager):
        srv = MagicMock()
        srv.session_manager = manager
        # No current model → the cached-ratio and slot-save blocks are skipped;
        # ``slot_enabled=False`` additionally skips the snapshot save.
        srv.current_model = None
        srv.logger = MagicMock()
        return srv

    def _response(self) -> MagicMock:
        import json

        body = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        }
        response = MagicMock()
        response.content = json.dumps(body).encode("utf-8")
        response.headers = {}
        return response

    def _has_provenance_after_completion(self, monkeypatch, status: int) -> bool:
        import proxy.router as router
        from proxy.session_manager import SessionManager

        monkeypatch.setattr(
            router, "_detect_restore_signal_from_llama_log", lambda *a, **k: False
        )
        manager = SessionManager()
        session_id = "sess-local-completion"
        messages = _messages(2)

        async def scenario():
            await manager.get_or_create(session_id)
            await router._update_session_and_slot(
                self._srv(manager),
                session_id,
                {"messages": messages},
                False,
                [],
                len(messages),
                self._response(),
                llama_port=8080,
                slot_id=0,
                slot_filename=None,
                slot_timeout=1.0,
                slot_model_payload="Qwen3",
                slot_enabled=False,
                upstream_status=status,
            )
            return await manager.has_local_provenance(session_id)

        return asyncio.run(scenario())

    def test_successful_local_completion_records_provenance(self, monkeypatch):
        assert self._has_provenance_after_completion(monkeypatch, 200) is True

    @pytest.mark.parametrize("status", [400, 429, 500, 503])
    def test_failed_local_completion_records_nothing(self, monkeypatch, status):
        """A 4xx/5xx local attempt served no turn — no false provenance."""
        assert self._has_provenance_after_completion(monkeypatch, status) is False


# ---------------------------------------------------------------------------
# AC3 (shipped config) — the knob ships off, with the agreed guard
# ---------------------------------------------------------------------------


class TestShippedBoundaryConfig:
    """`proxy/config.yaml` ships the boundary trigger disabled by default."""

    def _server(self) -> dict:
        import pathlib

        import yaml

        config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)["server"]

    def test_knob_ships_disabled(self):
        """A shipped opt-in would change live request handling unannounced."""
        assert self._server()["remote_boundary_compaction"] is False

    def test_guard_ships_at_the_agreed_default(self):
        assert (
            self._server()["remote_boundary_compaction_min_tokens"]
            == _DEFAULT_GUARD_TOKENS
        )

    def test_shipped_values_do_not_fire_the_decision(self):
        """The shipped profile decides nothing even for a huge session."""
        from proxy.compaction import decide_boundary_compaction

        out = decide_boundary_compaction(
            provider_type="remote",
            has_local_provenance=True,
            estimated_tokens=500000,
            config={"server": self._server()},
        )
        assert out["should_compact"] is False
        assert out["reason"] == "boundary_disabled"


# ---------------------------------------------------------------------------
# AC3 (slot pin half) — the summariser request carries the assigned id_slot
# ---------------------------------------------------------------------------


class TestSlotPinnedSummariser:
    """`build_slot_pinned_summarizer` pins the call to the session's own slot."""

    def test_request_body_carries_the_assigned_id_slot(self):
        builder = _require(
            "proxy.compaction_summarizer", "build_slot_pinned_summarizer"
        )
        cfg = _boundary_config()

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_client_cls:
            client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = client
            client.post.return_value = _ok_response("hello summary")

            summariser = builder(cfg, slot_id=3, llama_port=8123)
            result = summariser(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                None,
            )

            assert result == "hello summary"
            args, kwargs = client.post.call_args
            assert "http://localhost:8123/v1/chat/completions" in args[0]
            body = kwargs["json"]
            assert body["id_slot"] == 3

    def test_targets_the_sessions_own_local_model_not_the_summariser_model(self):
        """Slot ids are per llama-server instance: the local model runs the call."""
        builder = _require(
            "proxy.compaction_summarizer", "build_slot_pinned_summarizer"
        )
        cfg = _boundary_config()

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_client_cls:
            client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = client
            client.post.return_value = _ok_response("summary")

            builder(cfg, slot_id=7, llama_port=8080)([{"role": "user", "content": "u"}], None)

            body = client.post.call_args.kwargs["json"]
            assert body["model"] != _DEDICATED_SUMMARISER_MODEL
            assert body["model"]

    def test_no_slot_builds_a_summariser_that_never_calls_http(self):
        """A ``None`` slot makes the summariser inert (and falsy) — no HTTP."""
        builder = _require(
            "proxy.compaction_summarizer", "build_slot_pinned_summarizer"
        )

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_client_cls:
            summariser = builder(_boundary_config(), slot_id=None, llama_port=8080)
            result = summariser([{"role": "user", "content": "u"}], None)

            mock_client_cls.assert_not_called()
            assert not result

    def test_no_slot_reports_the_boundary_no_slot_reason_token(self):
        """The inert summariser carries the machine-readable skip reason."""
        from proxy.compaction import EmptySummary

        builder = _require(
            "proxy.compaction_summarizer", "build_slot_pinned_summarizer"
        )

        with patch("proxy.compaction_summarizer.httpx.Client"):
            summariser = builder(_boundary_config(), slot_id=None, llama_port=8080)
            result = summariser([{"role": "user", "content": "u"}], None)

        assert isinstance(result, EmptySummary)
        assert result.kind == "boundary_no_slot"
        assert result.attempts == 0

    def test_legacy_top_level_local_model_is_used(self):
        """A legacy ``type: local`` model entry names the pinned call's model."""
        builder = _require(
            "proxy.compaction_summarizer", "build_slot_pinned_summarizer"
        )
        cfg = {
            "server": {
                "summarizer_model": {
                    "type": "local",
                    "llama_model": _DEDICATED_SUMMARISER_MODEL,
                }
            },
            "models": {"legacy-local": {"type": "local", "llama_model": "Qwen3-legacy"}},
        }

        with patch("proxy.compaction_summarizer.httpx.Client") as mock_client_cls:
            client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = client
            client.post.return_value = _ok_response("summary")

            builder(cfg, slot_id=1, llama_port=8080)(
                [{"role": "user", "content": "u"}], None
            )

            body = client.post.call_args.kwargs["json"]
            assert body["model"] == "Qwen3-legacy"
            assert body["id_slot"] == 1


# ---------------------------------------------------------------------------
# AC3 (ordering half) — release precedes the remote stream start
# ---------------------------------------------------------------------------


class TestSlotReleaseOrdering:
    """`summarise_on_session_slot` releases the session's lease before returning."""

    @pytest.mark.asyncio
    async def test_release_precedes_the_remote_stream_start(self, slot_env):
        helper = _helper()
        srv = _fake_srv("sess-boundary-1")

        def builder(slot_id):
            assert slot_id == 3

            def summarise(messages):
                slot_env.events.append(("summarise", slot_id))
                return "BOUNDARY SUMMARY"

            return summarise

        out = await helper(
            srv,
            "sess-boundary-1",
            _messages(6),
            config=_boundary_config(),
            summarizer_builder=builder,
        )
        # Everything the dispatcher does next — the remote stream — happens
        # strictly after the helper returns.
        slot_env.events.append(("remote_stream_start",))

        assert out["slot_id"] == 3
        assert out["summary"] == "BOUNDARY SUMMARY"
        assert out["reason"] == "boundary_compact"
        assert slot_env.order("summarise") < slot_env.order("release")
        assert slot_env.order("release") < slot_env.order("remote_stream_start")
        # No leaked local lease for a session that is now served remotely.
        assert "sess-boundary-1" not in srv.local_dispatch_records

    @pytest.mark.asyncio
    async def test_summariser_runs_off_the_event_loop(self, slot_env):
        """The blocking local call runs in a worker thread, not the loop.

        The summariser can wait up to ``compaction_summarizer_timeout`` for the
        session's own slot (which may still be finishing an abandoned request),
        and boundary compaction runs before the remote dispatch — an inline
        call would freeze the loop, including the stream whose slot is being
        waited on (LP-0MU1RXEY10075TUU).
        """
        import threading

        helper = _helper()
        srv = _fake_srv("sess-boundary-thread")
        seen: dict[str, threading.Thread] = {}

        def builder(slot_id):
            def summarise(messages):
                seen["thread"] = threading.current_thread()
                return "BOUNDARY SUMMARY"

            return summarise

        await helper(
            srv,
            "sess-boundary-thread",
            _messages(4),
            config=_boundary_config(),
            summarizer_builder=builder,
        )

        assert seen["thread"] is not threading.main_thread()

    @pytest.mark.asyncio
    async def test_no_assigned_slot_skips_the_summariser_but_still_releases(
        self, slot_env
    ):
        helper = _helper()
        slot_env.slot = None
        srv = _fake_srv("sess-boundary-2")
        built: list[int | None] = []

        def builder(slot_id):
            built.append(slot_id)
            return lambda messages: "never called"

        out = await helper(
            srv,
            "sess-boundary-2",
            _messages(4),
            config=_boundary_config(),
            summarizer_builder=builder,
        )

        assert out["reason"] == "boundary_no_slot"
        assert out["slot_id"] is None
        assert out["summary"] is None
        assert built == []
        assert "release" in slot_env.kinds()
        assert "summarise" not in slot_env.kinds()
        assert "sess-boundary-2" not in srv.local_dispatch_records

    @pytest.mark.asyncio
    async def test_summariser_exception_releases_and_reports_failure(self, slot_env):
        helper = _helper()
        srv = _fake_srv("sess-boundary-3")

        def builder(slot_id):
            def summarise(messages):
                slot_env.events.append(("summarise", slot_id))
                raise RuntimeError("summariser blew up")

            return summarise

        out = await helper(
            srv,
            "sess-boundary-3",
            _messages(4),
            config=_boundary_config(),
            summarizer_builder=builder,
        )

        assert out["reason"] == "boundary_summarizer_failed"
        assert out["summary"] is None
        assert slot_env.order("release") > slot_env.order("summarise")
        assert "sess-boundary-3" not in srv.local_dispatch_records

    @pytest.mark.asyncio
    async def test_blank_summary_maps_to_boundary_summarizer_failed(self, slot_env):
        """An empty summary is never injected — it is a failure, not a result."""
        helper = _helper()
        srv = _fake_srv("sess-boundary-4")

        out = await helper(
            srv,
            "sess-boundary-4",
            _messages(4),
            config=_boundary_config(),
            summarizer_builder=lambda slot_id: (lambda messages: "   "),
        )

        assert out["reason"] == "boundary_summarizer_failed"
        assert out["summary"] is None
        assert "sess-boundary-4" not in srv.local_dispatch_records

    @pytest.mark.asyncio
    async def test_empty_summary_sentinel_maps_to_boundary_summarizer_failed(
        self, slot_env
    ):
        """The existing fail-open sentinel is preserved, not silently applied."""
        from proxy.compaction import EmptySummary

        helper = _helper()
        srv = _fake_srv("sess-boundary-5")

        out = await helper(
            srv,
            "sess-boundary-5",
            _messages(4),
            config=_boundary_config(),
            summarizer_builder=lambda slot_id: (
                lambda messages: EmptySummary("timeout", 2)
            ),
        )

        assert out["reason"] == "boundary_summarizer_failed"
        assert out["summary"] is None
        assert "sess-boundary-5" not in srv.local_dispatch_records

    @pytest.mark.asyncio
    async def test_unreachable_slot_maps_to_boundary_slot_unavailable(self, slot_env):
        """A connection-level failure means the assigned slot is unavailable."""
        helper = _helper()
        srv = _fake_srv("sess-boundary-6")

        def builder(slot_id):
            def summarise(messages):
                raise httpx.ConnectError("connection refused")

            return summarise

        out = await helper(
            srv,
            "sess-boundary-6",
            _messages(4),
            config=_boundary_config(),
            summarizer_builder=builder,
        )

        assert out["reason"] == "boundary_slot_unavailable"
        assert out["summary"] is None
        assert "release" in slot_env.kinds()
        assert "sess-boundary-6" not in srv.local_dispatch_records


# ---------------------------------------------------------------------------
# LP-0MVTDUBR00038ZRL — the dispatch hook that consumes the boundary contract
#
# The hook rewrites ``body_json["messages"]`` and refreshes the cached
# ``request._body`` before the remote POST, so the assertions below run
# against ``await request.body()`` — the exact bytes the remote dispatcher
# reads — rather than against the in-memory dict alone.
# ---------------------------------------------------------------------------

#: Provider entry about to be dispatched remotely.
_REMOTE_PROVIDER = {"name": "remote-primary", "type": "remote", "endpoint": "https://r.example.com/v1"}
#: Deterministic per-character estimator — no tokenizer, no network.
_CHARS_PER_TURN = 4000


def _hook():
    return _require("proxy.router_helpers", "evaluate_boundary_compaction")


def _chars_estimate(messages) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def _big_messages(turns: int, size: int = _CHARS_PER_TURN) -> list[dict]:
    """A history large enough that the boundary span folds the oldest turns."""
    messages: list[dict] = [{"role": "system", "content": "SYS" * (size // 3)}]
    for i in range(turns):
        messages.append({"role": "user", "content": f"u{i}-" + "x" * size})
        messages.append({"role": "assistant", "content": f"a{i}-" + "y" * size})
    return messages


class _FakeRequest:
    """Minimal request: cached body, session header and the hook's flags."""

    def __init__(self, body_json: dict, session_id: str = "sess-boundary-hook"):
        self._body = json.dumps(body_json).encode("utf-8")
        self.headers = {"x-session-id": session_id}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self) -> bytes:
        return self._body

    def dispatched(self) -> dict:
        return json.loads(self._body)


class _FakeSessionManager:
    """Session store double recording provenance reads and history writes."""

    def __init__(self, has_provenance: bool = True):
        self.has_provenance = has_provenance
        self.provenance_reads = 0
        self.updated: list[list[dict]] = []
        self.compacted: list[tuple[int, int | None]] = []

    async def has_local_provenance(self, session_id: str) -> bool:
        self.provenance_reads += 1
        return self.has_provenance

    async def update_messages(self, session_id: str, messages) -> bool:
        self.updated.append(list(messages))
        return True

    async def mark_compacted(
        self, session_id: str, from_count: int, base_count: int | None = None
    ) -> bool:
        self.compacted.append((from_count, base_count))
        return True


def _hook_srv(session_id: str, manager=None) -> SimpleNamespace:
    srv = _fake_srv(session_id)
    srv.session_manager = manager if manager is not None else _FakeSessionManager()
    return srv


def _warning_records(caplog, token: str):
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and token in record.getMessage()
    ]


def _boundary_event_records(caplog):
    return [
        record
        for record in caplog.records
        if "boundary_compaction " in record.getMessage()
    ]


class TestBoundaryCompactionHook:
    """`evaluate_boundary_compaction` — fire once, fail open, log every pass."""

    @pytest.mark.asyncio
    async def test_compacted_history_is_the_body_the_remote_dispatcher_sends(
        self, slot_env, caplog
    ):
        """AC1/AC4: the compacted list is what the dispatcher reads, stored too."""
        hook = _hook()
        messages = _big_messages(12)
        body_json = {"model": "test", "messages": list(messages)}
        request = _FakeRequest(body_json)
        manager = _FakeSessionManager()
        srv = _hook_srv("sess-hook-apply", manager)
        calls: list[dict] = []

        def builder(slot_id):
            def summarise(middle_messages, previous_summary=None):
                calls.append(
                    {
                        "slot": slot_id,
                        "messages": list(middle_messages),
                        "previous": previous_summary,
                    }
                )
                return "BOUNDARY SUMMARY"

            return summarise

        with caplog.at_level("INFO"):
            out = await hook(
                srv,
                _boundary_config(),
                body_json,
                "sess-hook-apply",
                _REMOTE_PROVIDER,
                request=request,
                estimate_tokens=_chars_estimate,
                summarizer_builder=builder,
            )

        # Exactly one summarisation, pinned to the session's own slot.
        assert len(calls) == 1
        assert calls[0]["slot"] == 3
        assert calls[0]["previous"] is None
        assert out["applied"] is True
        assert out["reason"] == "boundary_compact"
        assert out["slot_id"] == 3

        # The wire body — what ``proxy_to_remote`` reads — carries the
        # compacted list, not the original history.
        dispatched = request.dispatched()
        assert dispatched["messages"] == out["messages"]
        assert dispatched["cache_prompt"] is True
        assert dispatched["messages"] != messages
        assert out["estimated_after"] < out["estimated_before"]

        # Retention invariant: system prompt + first user message verbatim,
        # then the marked summary, then whole recent turns.
        assert dispatched["messages"][0] == messages[0]
        assert dispatched["messages"][1] == messages[1]
        assert "BOUNDARY SUMMARY" in dispatched["messages"][2]["content"]
        assert dispatched["messages"][2]["role"] == "user"
        assert dispatched["messages"][-2:] == messages[-2:]
        # The folded turns are the middle ones — the summariser never saw the
        # retained first user message.
        assert messages[1] not in calls[0]["messages"]
        assert out["turns_summarized"] == 9
        assert len(calls[0]["messages"]) > 0

        # Stored history matches what went on the wire, with the anchors the
        # post-compaction delta heal needs.
        assert manager.updated == [out["messages"]]
        assert manager.compacted == [(len(messages), len(out["messages"]))]

        # AC4: one structured record with session, slot, estimates, turns,
        # duration and outcome.
        events = _boundary_event_records(caplog)
        assert len(events) == 1
        line = events[0].getMessage()
        assert "session=sess-hook-apply" in line
        assert "slot=3" in line
        assert "est_before=" in line and "est_after=" in line
        assert "turns_summarized=" in line
        assert "duration_ms=" in line
        assert "outcome=boundary_compact" in line
        assert _warning_records(caplog, "boundary_") == []

        # The lease release (inside the summariser helper) precedes the
        # remote stream this dispatcher starts next.
        slot_env.events.append(("remote_stream_start",))
        assert slot_env.order("release") < slot_env.order("remote_stream_start")
        assert "sess-hook-apply" not in srv.local_dispatch_records

    @pytest.mark.asyncio
    async def test_second_evaluation_in_the_same_cycle_is_a_no_op(
        self, slot_env, caplog
    ):
        """AC5: one summarisation per request — a retry never re-compacts."""
        hook = _hook()
        body_json = {"model": "test", "messages": _big_messages(12)}
        request = _FakeRequest(body_json)
        srv = _hook_srv("sess-hook-once")
        calls: list[int] = []

        def builder(slot_id):
            def summarise(messages, previous_summary=None):
                calls.append(slot_id)
                return "BOUNDARY SUMMARY"

            return summarise

        first = await hook(
            srv,
            _boundary_config(),
            body_json,
            "sess-hook-once",
            _REMOTE_PROVIDER,
            request=request,
            estimate_tokens=_chars_estimate,
            summarizer_builder=builder,
        )
        assert first["applied"] is True
        compacted_body = request._body

        second = await hook(
            srv,
            _boundary_config(),
            body_json,
            "sess-hook-once",
            _REMOTE_PROVIDER,
            request=request,
            estimate_tokens=_chars_estimate,
            summarizer_builder=builder,
        )

        assert second["evaluated"] is False
        assert second["applied"] is False
        assert second["reason"] == "already_evaluated"
        assert calls == [3], "the summariser must run exactly once per request"
        assert request._body == compacted_body

    @pytest.mark.asyncio
    async def test_knob_disabled_returns_before_any_work(self, slot_env, caplog):
        """AC5: the shipped default adds no provenance read, estimate or call."""
        hook = _hook()
        body_json = {"model": "test", "messages": _big_messages(12)}
        request = _FakeRequest(body_json)
        original_body = request._body
        manager = _FakeSessionManager()
        srv = _hook_srv("sess-hook-disabled", manager)
        built: list[int] = []

        with caplog.at_level("INFO"):
            out = await hook(
                srv,
                _boundary_config(enabled=False),
                body_json,
                "sess-hook-disabled",
                _REMOTE_PROVIDER,
                request=request,
                estimate_tokens=_chars_estimate,
                summarizer_builder=lambda slot_id: built.append(slot_id),
            )

        assert out["evaluated"] is False
        assert out["applied"] is False
        assert out["reason"] == "boundary_disabled"
        assert manager.provenance_reads == 0
        assert built == []
        assert request._body == original_body
        assert _boundary_event_records(caplog) == []

    @pytest.mark.asyncio
    async def test_non_remote_provider_returns_before_any_work(self, slot_env):
        """Boundary compaction never applies to a local dispatch."""
        hook = _hook()
        body_json = {"model": "test", "messages": _big_messages(12)}
        request = _FakeRequest(body_json)
        original_body = request._body
        manager = _FakeSessionManager()
        srv = _hook_srv("sess-hook-local", manager)

        out = await hook(
            srv,
            _boundary_config(),
            body_json,
            "sess-hook-local",
            {"name": "local-qwen3", "type": "local"},
            request=request,
            estimate_tokens=_chars_estimate,
        )

        assert out["reason"] == "boundary_not_remote"
        assert out["applied"] is False
        assert manager.provenance_reads == 0
        assert request._body == original_body

    @pytest.mark.asyncio
    async def test_session_without_local_provenance_dispatches_unchanged(
        self, slot_env, caplog
    ):
        """AC1: a never-locally-served session is untouched, silently."""
        manager = _FakeSessionManager(has_provenance=False)
        hook = _hook()
        body_json = {"model": "test", "messages": _big_messages(12)}
        request = _FakeRequest(body_json)
        original_body = request._body
        srv = _hook_srv("sess-hook-no-prov", manager)
        built: list[int] = []

        with caplog.at_level("INFO"):
            out = await hook(
                srv,
                _boundary_config(),
                body_json,
                "sess-hook-no-prov",
                _REMOTE_PROVIDER,
                request=request,
                estimate_tokens=_chars_estimate,
                summarizer_builder=lambda slot_id: built.append(slot_id),
            )

        assert out["reason"] == "boundary_no_local_provenance"
        assert out["applied"] is False
        assert built == []
        assert request._body == original_body
        assert _warning_records(caplog, "boundary_") == []

    @pytest.mark.asyncio
    async def test_below_the_size_guard_warns_and_dispatches_unchanged(
        self, slot_env, caplog
    ):
        """AC3: a session under the guard is left alone with a WARNING token."""
        hook = _hook()
        body_json = {"model": "test", "messages": _big_messages(2)}
        request = _FakeRequest(body_json)
        original_body = request._body
        srv = _hook_srv("sess-hook-small")

        with caplog.at_level("INFO"):
            out = await hook(
                srv,
                _boundary_config(),
                body_json,
                "sess-hook-small",
                _REMOTE_PROVIDER,
                request=request,
                estimated_tokens=1000,
                estimate_tokens=_chars_estimate,
            )

        assert out["reason"] == "boundary_below_guard"
        assert out["applied"] is False
        assert request._body == original_body
        warnings = _warning_records(caplog, "boundary_below_guard")
        assert len(warnings) == 1, "exactly one WARNING per skip"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mode", "token"),
        [
            ("raise", "boundary_summarizer_failed"),
            ("blank", "boundary_summarizer_failed"),
            ("empty_sentinel", "boundary_summarizer_failed"),
            ("transport", "boundary_slot_unavailable"),
        ],
    )
    async def test_summariser_failures_fail_open_byte_identically(
        self, slot_env, caplog, mode, token
    ):
        """AC2/AC3: every summariser failure leaves the body untouched."""
        from proxy.compaction import EmptySummary

        hook = _hook()
        body_json = {"model": "test", "messages": _big_messages(12)}
        request = _FakeRequest(body_json)
        original_body = request._body
        srv = _hook_srv(f"sess-hook-{mode}")

        def summarise(messages, previous_summary=None):
            if mode == "raise":
                raise RuntimeError("summariser blew up")
            if mode == "transport":
                raise httpx.ConnectError("connection refused")
            if mode == "blank":
                return "   "
            return EmptySummary("timeout", 2)

        def builder(slot_id):
            def wrapper(messages, previous_summary=None):
                slot_env.events.append(("summarise", slot_id))
                return summarise(messages, previous_summary)

            return wrapper

        with caplog.at_level("INFO"):
            out = await hook(
                srv,
                _boundary_config(),
                body_json,
                f"sess-hook-{mode}",
                _REMOTE_PROVIDER,
                request=request,
                estimate_tokens=_chars_estimate,
                summarizer_builder=builder,
            )

        assert out["applied"] is False
        assert out["reason"] == token
        assert out["messages"] is None
        # AC2: the outgoing history is byte-identical to today's.
        assert request._body == original_body
        assert _warning_records(caplog, token), "the skip must be logged as a WARNING"
        assert len(_warning_records(caplog, token)) == 1
        # The slot is released even on failure (parent AC3).
        assert "release" in slot_env.kinds()

    @pytest.mark.asyncio
    async def test_no_assigned_slot_warns_once_and_dispatches_unchanged(
        self, slot_env, caplog, monkeypatch
    ):
        """AC3: no slot means skip, WARNING token, and no summariser build."""
        from proxy import compaction_summarizer as summarizer_mod

        hook = _hook()
        slot_env.slot = None
        body_json = {"model": "test", "messages": _big_messages(12)}
        request = _FakeRequest(body_json)
        original_body = request._body
        srv = _hook_srv("sess-hook-no-slot")
        built: list[int | None] = []
        remote_summaries: list = []

        # AC3: there is no remote-summariser fallback on this path.
        monkeypatch.setattr(
            summarizer_mod,
            "build_compact_summarizer",
            lambda *a, **k: remote_summaries.append((a, k)),
        )

        def builder(slot_id):
            built.append(slot_id)
            return lambda messages, previous_summary=None: "never called"

        with caplog.at_level("INFO"):
            out = await hook(
                srv,
                _boundary_config(),
                body_json,
                "sess-hook-no-slot",
                _REMOTE_PROVIDER,
                request=request,
                estimate_tokens=_chars_estimate,
                summarizer_builder=builder,
            )

        assert out["reason"] == "boundary_no_slot"
        assert out["applied"] is False
        assert built == []
        assert request._body == original_body
        assert len(_warning_records(caplog, "boundary_no_slot")) == 1
        assert remote_summaries == [], "no remote summariser fallback"

    @pytest.mark.asyncio
    async def test_hook_never_raises_into_the_dispatch_loop(
        self, slot_env, caplog, monkeypatch
    ):
        """AC2: an unexpected error is swallowed, logged, and dispatch continues."""

        async def exploding(*args, **kwargs):
            raise RuntimeError("slot helper exploded")

        monkeypatch.setattr(router_helpers, "summarise_on_session_slot", exploding)
        hook = _hook()
        body_json = {"model": "test", "messages": _big_messages(12)}
        request = _FakeRequest(body_json)
        original_body = request._body
        srv = _hook_srv("sess-hook-explode")

        with caplog.at_level("INFO"):
            out = await hook(
                srv,
                _boundary_config(),
                body_json,
                "sess-hook-explode",
                _REMOTE_PROVIDER,
                request=request,
                estimate_tokens=_chars_estimate,
            )

        assert out["applied"] is False
        assert out["reason"] == "boundary_evaluation_failed"
        assert request._body == original_body
        assert len(
            _warning_records(caplog, "boundary_evaluation_failed")
        ) == 1


# ---------------------------------------------------------------------------
# AC1 (dispatch half) — both remote dispatch points run the hook pre-POST
# ---------------------------------------------------------------------------


class _WiringRequest(_FakeRequest):
    """Request stub with the shape the fallback cycles expect."""


def _allow_budget():
    from proxy.provider_budget import BudgetDecision

    async def _budget(provider_cfg, config, session_id, request=None, model_config=None):
        return BudgetDecision(allowed=True, reason="unlimited")

    return _budget


class TestBoundaryCompactionDispatchWiring:
    """The remote POST is preceded by the boundary hook in both cycles."""

    def _record_hook(self, monkeypatch, events: list):
        async def fake_hook(
            srv, config, body_json, session_id, provider_cfg, **kwargs
        ):
            events.append(("hook", session_id, provider_cfg.get("name")))
            return {"evaluated": False, "applied": False, "reason": "boundary_disabled"}

        monkeypatch.setattr(router_helpers, "evaluate_boundary_compaction", fake_hook)

    def _record_remote(self, monkeypatch, events: list, expected_body: dict):
        async def fake_remote(request, path, provider_cfg):
            events.append(("dispatch", provider_cfg.get("name")))
            assert json.loads(await request.body())["messages"] == expected_body[
                "messages"
            ]
            return Response(
                content=json.dumps(
                    {"choices": [{"message": {"content": "ok"}}]}
                ),
                status_code=200,
                media_type="application/json",
            )

        monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: fake_remote)

    @pytest.fixture(autouse=True)
    def _reset_provider_state(self, monkeypatch):
        provider._provider_unavailable_until.clear()
        provider._provider_failure_count.clear()
        provider._usage_reset_at.clear()
        provider._sibling_failure_count.clear()
        provider._sibling_failure_streak_start.clear()
        monkeypatch.setattr(provider, "request_priority", lambda *a, **k: "medium")
        monkeypatch.setattr("proxy.priority.request_priority", lambda *a, **k: "medium")
        monkeypatch.setattr(provider, "acquire_provider_budget", _allow_budget())
        yield

    @pytest.mark.asyncio
    async def test_remote_only_cycle_evaluates_the_hook_before_the_post(
        self, monkeypatch
    ):
        events: list = []
        body_json = {"model": "wiring", "messages": _messages(2)}
        request = _WiringRequest(body_json, session_id="sess-wiring-remote-only")
        self._record_hook(monkeypatch, events)
        self._record_remote(monkeypatch, events, body_json)

        response = await provider._proxy_with_remote_fallback_cycle(
            request,
            "v1/chat/completions",
            {"name": "wiring", "providers": [dict(_REMOTE_PROVIDER)]},
            {"provider_cooldown_seconds": 60, "server": {"llama_server_port": 8080}},
        )

        assert response.status_code == 200
        assert events == [
            ("hook", "sess-wiring-remote-only", "remote-primary"),
            ("dispatch", "remote-primary"),
        ]

    @pytest.mark.asyncio
    async def test_mixed_cycle_evaluates_the_hook_before_the_post(self, monkeypatch):
        events: list = []
        body_json = {"model": "wiring", "messages": _messages(2)}
        request = _WiringRequest(body_json, session_id="sess-wiring-mixed")
        self._record_hook(monkeypatch, events)
        self._record_remote(monkeypatch, events, body_json)
        # Local is skipped on size, and the assembly-time compaction gate is
        # disabled (the shipped ``compaction_trigger_ratio: 0``), so the cycle
        # continues to the remote provider exactly as today.
        monkeypatch.setattr(
            provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large")
        )
        monkeypatch.setattr(
            provider,
            "_evaluate_compaction_for_bypass",
            AsyncMock(
                return_value={
                    "evaluated": True,
                    "action": "noop",
                    "applied": False,
                    "estimated_before": 0,
                    "estimated_after": 0,
                    "reason": "compaction_disabled",
                    "messages": None,
                }
            ),
        )

        local_calls: list = []

        async def fake_local(request, path, endpoint=None):
            local_calls.append(endpoint)
            return Response(content=b"{}", status_code=200)

        monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: fake_local)

        response = await provider._proxy_with_fallback_cycle(
            request,
            "v1/chat/completions",
            {
                "name": "wiring",
                "providers": [
                    {"name": "local-a", "type": "local", "llama_model": "Qwen3"},
                    dict(_REMOTE_PROVIDER),
                ],
            },
            {
                "provider_cooldown_seconds": 60,
                "server": {
                    "llama_server_port": 8080,
                    "compaction_trigger_ratio": 0,
                },
            },
        )

        assert response.status_code == 200
        assert local_calls == [], "the bypassed local entry must not be dispatched"
        assert events == [
            ("hook", "sess-wiring-mixed", "remote-primary"),
            ("dispatch", "remote-primary"),
        ]


# ---------------------------------------------------------------------------
# The direct remote dispatch sites in ``proxy/ui.py``
#
# ``provider``'s two fallback cycles cover most remote traffic; ``ui.py`` also
# dispatches remote directly (unknown-model ``default_remote`` and a remote
# model with no ``providers`` list). Those sites run the same shared hook.
# ---------------------------------------------------------------------------


class TestBoundaryHookDirectRemoteDispatch:
    """`ui._evaluate_boundary_compaction_before_remote` delegates, fail-open."""

    def _srv(self, config: dict | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            config=config if config is not None else _boundary_config(),
            logger=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_forwards_to_the_shared_hook_with_the_resolved_session(
        self, monkeypatch
    ):
        import proxy.ui as ui

        calls: list[tuple] = []

        async def fake_hook(
            request, provider_name, provider_cfg, body_json, session_id, config
        ):
            calls.append((provider_name, body_json, session_id, config))
            return {"applied": True, "reason": "boundary_compact"}

        monkeypatch.setattr(
            provider, "_evaluate_boundary_compaction_for_remote", fake_hook
        )
        config = _boundary_config()
        srv = self._srv(config)
        provider_cfg = {"name": "remote-direct", "type": "remote"}
        body_json = {"messages": [{"role": "user", "content": "hi"}]}
        request = MagicMock()
        request.headers = {"x-session-id": "sess-ui-remote"}

        await ui._evaluate_boundary_compaction_before_remote(
            srv, request, provider_cfg, body_json
        )

        assert calls == [
            ("remote-direct", body_json, "sess-ui-remote", config)
        ]

    @pytest.mark.asyncio
    async def test_no_session_id_means_no_evaluation(self, monkeypatch):
        import proxy.ui as ui

        calls: list[tuple] = []

        async def fake_hook(*args, **kwargs):
            calls.append(args)
            return {}

        monkeypatch.setattr(
            provider, "_evaluate_boundary_compaction_for_remote", fake_hook
        )
        request = MagicMock()
        request.headers = {}

        await ui._evaluate_boundary_compaction_before_remote(
            self._srv(), request, {"type": "remote"}, {"messages": []}
        )

        assert calls == []

    @pytest.mark.asyncio
    async def test_a_failing_hook_never_blocks_the_remote_dispatch(
        self, monkeypatch
    ):
        import proxy.ui as ui

        async def exploding(*args, **kwargs):
            raise RuntimeError("hook exploded")

        monkeypatch.setattr(
            provider, "_evaluate_boundary_compaction_for_remote", exploding
        )
        request = MagicMock()
        request.headers = {"x-session-id": "sess-ui-boom"}

        # Must not raise: the caller dispatches remote immediately afterwards.
        await ui._evaluate_boundary_compaction_before_remote(
            self._srv(), request, {"type": "remote"}, {"messages": []}
        )


# ---------------------------------------------------------------------------
# LP-0MVTDUEF5007A1SP — integration: a locally-served session escalates remote
#
# Everything below drives production code end to end: the mixed and
# remote-only dispatch cycles, ``evaluate_boundary_compaction``, the real
# slot-pinned summariser (``build_slot_pinned_summarizer``), the real lease
# release and the real session store. Only the two HTTP transports are
# stubbed — the local summariser POST and the remote POST each run through an
# ``httpx.MockTransport``, so every body assertion below reads the bytes that
# would leave the process rather than an in-memory dict.
# ---------------------------------------------------------------------------

_INT_SESSION = "sess-boundary-integration"
_INT_LOCAL_MODEL = "Qwen3"
_INT_SUMMARY = "INTEGRATION BOUNDARY SUMMARY"
_INT_LOCAL_PROVIDER = {
    "name": "local-qwen3",
    "type": "local",
    "llama_model": _INT_LOCAL_MODEL,
}
_INT_REMOTE_PROVIDER = {
    "name": "remote-integration",
    "type": "remote",
    "endpoint": "https://remote.invalid/v1",
}
#: Sibling remote used only by the in-cycle retry case.
_INT_SECOND_REMOTE_PROVIDER = {
    "name": "remote-integration-2",
    "type": "remote",
    "endpoint": "https://remote2.invalid/v1",
}
#: Turns/size that clear the planner's 38 000-token fast target with room to
#: fold several turns (deterministic: only the routing estimator is involved,
#: which falls back to a character heuristic for an unknown model).
_INT_TURNS = 10
_INT_TURN_CHARS = 12000
#: A guard small enough that the large history always clears it.
_INT_GUARD_TOKENS = 200


def _integration_config(*, enabled: bool = True, min_tokens: int = _INT_GUARD_TOKENS) -> dict:
    """Boundary config wired for the integration harness."""
    cfg = _boundary_config(enabled=enabled, min_tokens=min_tokens)
    cfg["server"].update(
        {
            "session_slot_pool_size": 4,
            "compaction_summarizer_timeout": 5,
            # No retry waits: the harness must stay in the sub-second range
            # and a single summariser attempt is what the assertions count.
            "compaction_summarizer_retries": 0,
            "compaction_summarizer_retry_delay_seconds": 0,
        }
    )
    cfg["provider_cooldown_seconds"] = 60
    return cfg


class _BoundaryTrace:
    """Ordered record of the boundary: local turn, summariser, remote POST."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, dict]] = []

    def add(self, kind: str, **payload) -> None:
        self.entries.append((kind, payload))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.entries]

    def payloads(self, kind: str) -> list[dict]:
        return [payload for observed, payload in self.entries if observed == kind]

    def order(self, kind: str) -> int:
        return self.kinds().index(kind)


@pytest.fixture
def boundary_world(monkeypatch):
    """A proxy runtime with real session, slot and lease registries.

    ``proxy.server``'s module globals are what the boundary hook reads at
    dispatch time, so they are swapped for fresh per-test state: the store
    that records provenance, the dispatch-lease registry the release removes
    from, and the slot registry the summariser is pinned to.
    """
    import proxy.router as router
    import proxy.server as server
    import proxy.session as session_mod
    from proxy.session_manager import SessionManager

    manager = SessionManager()
    monkeypatch.setattr(server, "session_manager", manager)
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_prefill_in_flight", {})
    # No local llama-server is running: the cached-ratio/save blocks are
    # skipped (paid only when ``current_model`` is set) and the restore-signal
    # probe has no log to read.
    monkeypatch.setattr(server, "current_model", None)
    monkeypatch.setattr(
        router, "_detect_restore_signal_from_llama_log", lambda *a, **k: False
    )
    monkeypatch.setattr(session_mod, "_slot_owners", {})

    # Routing/provider side effects that are orthogonal to the boundary path.
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    monkeypatch.setattr(provider, "request_priority", lambda *a, **k: "medium")
    monkeypatch.setattr("proxy.priority.request_priority", lambda *a, **k: "medium")
    monkeypatch.setattr(provider, "acquire_provider_budget", _allow_budget())
    # Assembly-time compaction is off in the shipped profile; keep the bypass
    # evaluation a no-op so only the boundary trigger can act.
    monkeypatch.setattr(
        provider,
        "_evaluate_compaction_for_bypass",
        AsyncMock(
            return_value={
                "evaluated": True,
                "action": "noop",
                "applied": False,
                "estimated_before": 0,
                "estimated_after": 0,
                "reason": "compaction_disabled",
                "messages": None,
            }
        ),
    )

    return SimpleNamespace(
        server=server,
        manager=manager,
        session=session_mod,
        trace=_BoundaryTrace(),
    )


def _install_local_summariser_transport(
    monkeypatch, trace: _BoundaryTrace, *, status: int = 200, content: str = _INT_SUMMARY
) -> None:
    """Route the local summariser POST through a stubbed httpx transport.

    The real ``build_slot_pinned_summarizer`` → ``build_local_summarizer``
    code path runs; only the socket is replaced, so the recorded request is
    the exact body the summariser would send to the session's own slot.
    """
    import proxy.compaction_summarizer as summarizer_mod

    real_client = httpx.Client

    def _factory(*args, **kwargs):
        def _handler(request: httpx.Request) -> httpx.Response:
            trace.add(
                "summariser_request",
                url=str(request.url),
                body=json.loads(request.content or b"{}"),
            )
            if status != 200:
                return httpx.Response(status, json={"error": "summariser down"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": content}}
                    ]
                },
            )

        kwargs["transport"] = httpx.MockTransport(_handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(summarizer_mod.httpx, "Client", _factory)


def _install_remote_transport(
    monkeypatch,
    trace: _BoundaryTrace,
    world,
    session_id: str,
    *,
    status_by_name: dict[str, int] | None = None,
) -> None:
    """Dispatch the remote provider over a stubbed httpx transport.

    The recorded request is the post-boundary body — i.e. what the remote
    provider would receive. Lease/slot state is snapshotted *inside* the
    transport handler, which is the moment the remote stream would start.
    """

    async def _dispatch(request, path, provider_cfg):
        body = await request.body()

        def _handler(remote_request: httpx.Request) -> httpx.Response:
            trace.add(
                "remote_request",
                url=str(remote_request.url),
                provider=provider_cfg.get("name"),
                raw=remote_request.content,
                body=json.loads(remote_request.content or b"{}"),
                leases_at_start=dict(world.server.local_dispatch_records),
                slot_at_start=world.session._assigned_slot_for_session(session_id),
            )
            status = (status_by_name or {}).get(provider_cfg.get("name"), 200)
            if status != 200:
                return httpx.Response(status, json={"error": f"upstream {status}"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "remote ok"}}
                    ]
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_handler)
        ) as client:
            response = await client.post(
                f"{provider_cfg['endpoint']}/{path}",
                content=body,
                headers={"content-type": "application/json"},
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type="application/json",
        )

    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _dispatch)


def _install_local_dispatch(monkeypatch, trace: _BoundaryTrace) -> None:
    """Stub the local backend (no llama-server in the integration harness)."""

    async def _dispatch(request, path, endpoint=None):
        trace.add("local_dispatch", endpoint=endpoint)
        return Response(
            content=json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "local ok"}}]}
            ),
            status_code=200,
            media_type="application/json",
        )

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: _dispatch)


async def _serve_local_turn(
    world, monkeypatch, session_id: str, messages: list[dict]
) -> None:
    """Serve one turn locally and record its provenance in the session store.

    The turn goes through the production mixed cycle with the local backend
    stubbed; the router's own completion hook then records the served turn,
    which is the provenance signal the boundary decision reads. The slot and
    dispatch lease the local dispatch would hold are seeded afterwards — the
    real bookkeeping lives in ``proxy_to_local``, which is the one piece
    stubbed out.
    """
    import proxy.router as router

    # The proxy's request path creates the session; the cycle below is driven
    # directly, so create it here before its completion hook runs.
    await world.manager.get_or_create(session_id)
    _install_local_dispatch(monkeypatch, world.trace)
    monkeypatch.setattr(provider, "_should_skip_local", lambda *a, **k: (False, None))
    request = _WiringRequest(
        {"model": "integration", "messages": messages}, session_id=session_id
    )
    response = await provider._proxy_with_fallback_cycle(
        request,
        "v1/chat/completions",
        {"name": "integration", "providers": [dict(_INT_LOCAL_PROVIDER), dict(_INT_REMOTE_PROVIDER)]},
        _integration_config(),
    )
    assert response.status_code == 200

    await router._update_session_and_slot(
        world.server,
        session_id,
        {"messages": messages},
        False,
        [],
        len(messages),
        response,
        llama_port=8080,
        slot_id=None,
        slot_filename=None,
        slot_timeout=1.0,
        slot_model_payload=_INT_LOCAL_MODEL,
        slot_enabled=False,
        upstream_status=200,
    )
    assert await world.manager.has_local_provenance(session_id) is True

    # The session still owns its slot + dispatch lease at the boundary.
    slot_id = world.session._slot_id_for_session(session_id, pool_size=4)
    assert slot_id is not None
    world.server.local_dispatch_records[
        router_helpers._dispatch_lease_key(None, session_id)
    ] = {"session_id": session_id}
    world.trace.add("local_turn_complete", slot=slot_id)


async def _dispatch_remote_turn(
    world,
    monkeypatch,
    session_id: str,
    messages: list[dict],
    *,
    config: dict | None = None,
    remote_only: bool = False,
    remote_providers: list[dict] | None = None,
    status_by_name: dict[str, int] | None = None,
):
    """Send the next request remote, bypassing local.

    Returns ``(response, request, raw_body_before)`` so a regression test can
    compare the dispatched bytes against the pre-change body.
    """
    _install_remote_transport(
        monkeypatch, world.trace, world, session_id, status_by_name=status_by_name
    )
    if remote_providers is not None:
        # A remote-only chain with more than one entry exercises the
        # in-cycle provider retry (the second provider is the fallback).
        monkeypatch.setattr(
            provider, "_should_skip_local", lambda *a, **k: (True, "remote_only_chain")
        )
        model_config = {
            "name": "integration",
            "providers": [dict(cfg) for cfg in remote_providers],
        }
    elif remote_only:
        monkeypatch.setattr(
            provider, "_should_skip_local", lambda *a, **k: (True, "remote_only_chain")
        )
        model_config = {"name": "integration", "providers": [dict(_INT_REMOTE_PROVIDER)]}
    else:
        # The real local→remote boundary: local is skipped on size and the
        # chain continues to the remote provider.
        monkeypatch.setattr(
            provider, "_should_skip_local", lambda *a, **k: (True, "context_too_large")
        )
        model_config = {
            "name": "integration",
            "providers": [dict(_INT_LOCAL_PROVIDER), dict(_INT_REMOTE_PROVIDER)],
        }
    request = _WiringRequest(
        {"model": "integration", "messages": list(messages)}, session_id=session_id
    )
    raw_before = request._body
    # A remote-only chain — whether declared as one provider or several —
    # runs the remote-only cycle; the mixed cycle is only for a chain that
    # also contains a local provider.
    cycle = (
        provider._proxy_with_remote_fallback_cycle
        if remote_only or remote_providers is not None
        else provider._proxy_with_fallback_cycle
    )
    response = await cycle(
        request,
        "v1/chat/completions",
        model_config,
        config if config is not None else _integration_config(),
    )
    return response, request, raw_before


def _big_integration_messages() -> list[dict]:
    return _big_messages(_INT_TURNS, size=_INT_TURN_CHARS)


class TestBoundaryCompactionIntegration:
    """Local→remote escalation, proven on the wire with stubbed transports."""

    @pytest.mark.asyncio
    async def test_locally_served_session_compacts_once_before_the_remote_post(
        self, boundary_world, monkeypatch, caplog
    ):
        """AC1/AC2/AC3: one summarisation, compacted wire body, released lease."""
        world = boundary_world
        messages = _big_integration_messages()
        _install_local_summariser_transport(monkeypatch, world.trace)

        # (b) served locally first — that is what makes provenance real.
        await _serve_local_turn(world, monkeypatch, _INT_SESSION, _messages(2))
        slot_id = world.trace.payloads("local_turn_complete")[0]["slot"]

        # (c) the next request bypasses local and routes remote.
        with caplog.at_level("INFO"):
            response, _request, _raw_before = await _dispatch_remote_turn(
                world, monkeypatch, _INT_SESSION, messages
            )

        assert response.status_code == 200

        # AC1 — exactly one summarisation call for the boundary evaluation.
        summariser_calls = world.trace.payloads("summariser_request")
        assert len(summariser_calls) == 1
        # AC2 — pinned to the session's own local slot and its own model.
        assert summariser_calls[0]["url"].startswith("http://localhost:8080/")
        assert summariser_calls[0]["body"]["id_slot"] == slot_id
        assert summariser_calls[0]["body"]["model"] == _INT_LOCAL_MODEL
        assert summariser_calls[0]["body"]["stream"] is False
        # The retention set is never handed to the summariser.
        assert messages[1]["content"] not in json.dumps(summariser_calls[0]["body"])

        # AC2 — the compacted body is what reached the remote provider.
        remote_calls = world.trace.payloads("remote_request")
        assert len(remote_calls) == 1
        remote = remote_calls[0]
        assert remote["url"].startswith("https://remote.invalid/v1/")
        dispatched = remote["body"]
        assert dispatched["cache_prompt"] is True
        assert dispatched["messages"] != messages
        assert len(dispatched["messages"]) < len(messages)
        # Retention invariant: system prompt + first user message verbatim.
        assert dispatched["messages"][0] == messages[0]
        assert dispatched["messages"][1] == messages[1]
        # The marked summary replaces the folded turns.
        summary_message = dispatched["messages"][2]
        assert summary_message["role"] == "user"
        assert _INT_SUMMARY in summary_message["content"]
        assert (
            "The conversation history before this point was compacted"
            in summary_message["content"]
        )
        # The most recent turns are kept whole and verbatim.
        assert dispatched["messages"][-2:] == messages[-2:]

        # AC3 — the release precedes the remote stream start.
        assert world.trace.order("summariser_request") < world.trace.order(
            "remote_request"
        )
        assert remote["leases_at_start"] == {}
        assert remote["slot_at_start"] is None
        # ... and no lease leaks afterwards.
        assert _INT_SESSION not in world.server.local_dispatch_records
        assert world.session._assigned_slot_for_session(_INT_SESSION) is None

        # The stored history is what went on the wire (delta heal sees it).
        session = await world.manager.get(_INT_SESSION)
        assert session is not None
        assert session.messages == dispatched["messages"]

        # AC4 — exactly one structured boundary event for the evaluation.
        events = _boundary_event_records(caplog)
        assert len(events) == 1
        line = events[0].getMessage()
        assert f"session={_INT_SESSION}" in line
        assert f"slot={slot_id}" in line
        assert "outcome=boundary_compact" in line
        assert "est_before=" in line and "est_after=" in line
        assert "turns_summarized=" in line and "duration_ms=" in line

    @pytest.mark.asyncio
    async def test_remote_only_chain_compacts_at_the_same_boundary(
        self, boundary_world, monkeypatch
    ):
        """The second remote dispatch point runs the same hook end to end."""
        world = boundary_world
        _install_local_summariser_transport(monkeypatch, world.trace)
        await _serve_local_turn(world, monkeypatch, _INT_SESSION, _messages(2))

        response, _request, _raw = await _dispatch_remote_turn(
            world,
            monkeypatch,
            _INT_SESSION,
            _big_integration_messages(),
            remote_only=True,
        )

        assert response.status_code == 200
        assert len(world.trace.payloads("summariser_request")) == 1
        remote = world.trace.payloads("remote_request")[0]
        assert _INT_SUMMARY in json.dumps(remote["body"])
        assert remote["leases_at_start"] == {}
        assert world.session._assigned_slot_for_session(_INT_SESSION) is None

    @pytest.mark.asyncio
    async def test_an_in_cycle_provider_retry_never_re_summarises(
        self, boundary_world, monkeypatch
    ):
        """AC1: a failed remote provider retries with the same compacted history."""
        world = boundary_world
        _install_local_summariser_transport(monkeypatch, world.trace)
        await _serve_local_turn(world, monkeypatch, _INT_SESSION, _messages(2))

        response, request, _raw = await _dispatch_remote_turn(
            world,
            monkeypatch,
            _INT_SESSION,
            _big_integration_messages(),
            remote_providers=[
                dict(_INT_REMOTE_PROVIDER),
                dict(_INT_SECOND_REMOTE_PROVIDER),
            ],
            status_by_name={_INT_REMOTE_PROVIDER["name"]: 500},
        )

        assert response.status_code == 200, "the sibling provider serves the retry"
        # Exactly one summarisation for the whole cycle, not one per attempt.
        assert len(world.trace.payloads("summariser_request")) == 1
        posts = world.trace.payloads("remote_request")
        assert [post["provider"] for post in posts] == [
            _INT_REMOTE_PROVIDER["name"],
            _INT_SECOND_REMOTE_PROVIDER["name"],
        ]
        # Both attempts carry the identical compacted history.
        assert posts[0]["body"]["messages"] == posts[1]["body"]["messages"]
        assert _INT_SUMMARY in posts[1]["body"]["messages"][2]["content"]
        assert request._boundary_compaction_result["reason"] == "boundary_compact"

    @pytest.mark.asyncio
    async def test_knob_off_dispatches_a_byte_identical_body_without_summarising(
        self, boundary_world, monkeypatch, caplog
    ):
        """AC4 regression: the shipped default changes nothing on the wire."""
        world = boundary_world
        _install_local_summariser_transport(monkeypatch, world.trace)
        await _serve_local_turn(world, monkeypatch, _INT_SESSION, _messages(2))

        with caplog.at_level("INFO"):
            response, _request, raw_before = await _dispatch_remote_turn(
                world,
                monkeypatch,
                _INT_SESSION,
                _big_integration_messages(),
                config=_integration_config(enabled=False),
            )

        assert response.status_code == 200
        # No summariser call, no boundary event, no lease release.
        assert world.trace.payloads("summariser_request") == []
        assert _boundary_event_records(caplog) == []
        remote = world.trace.payloads("remote_request")[0]
        assert remote["raw"] == raw_before, "the dispatched body must be unchanged"
        assert remote["body"]["messages"] == json.loads(raw_before)["messages"]
        assert remote["leases_at_start"] != {}, "no side effect when the knob is off"
        assert _INT_SESSION in world.server.local_dispatch_records

    @pytest.mark.asyncio
    async def test_session_without_local_service_dispatches_unchanged(
        self, boundary_world, monkeypatch
    ):
        """AC5: a session never served locally is untouched, even knob-on."""
        world = boundary_world
        _install_local_summariser_transport(monkeypatch, world.trace)
        # No local turn: the session store records no local provenance.
        session_id = "sess-boundary-never-local"
        await world.manager.get_or_create(session_id)

        response, request, raw_before = await _dispatch_remote_turn(
            world, monkeypatch, session_id, _big_integration_messages()
        )

        assert response.status_code == 200
        assert world.trace.payloads("summariser_request") == []
        assert world.trace.payloads("remote_request")[0]["raw"] == raw_before
        assert request._boundary_compaction_result["reason"] == (
            "boundary_no_local_provenance"
        )

    @pytest.mark.asyncio
    async def test_below_the_size_guard_dispatches_unchanged(
        self, boundary_world, monkeypatch
    ):
        """AC5: a locally-served session under the guard is left alone."""
        world = boundary_world
        _install_local_summariser_transport(monkeypatch, world.trace)
        await _serve_local_turn(world, monkeypatch, _INT_SESSION, _messages(2))

        response, request, raw_before = await _dispatch_remote_turn(
            world,
            monkeypatch,
            _INT_SESSION,
            _big_integration_messages(),
            config=_integration_config(min_tokens=10_000_000),
        )

        assert response.status_code == 200
        assert world.trace.payloads("summariser_request") == []
        assert world.trace.payloads("remote_request")[0]["raw"] == raw_before
        assert request._boundary_compaction_result["reason"] == "boundary_below_guard"
        # A below-guard skip is not a boundary evaluation: the lease survives.
        assert _INT_SESSION in world.server.local_dispatch_records

    @pytest.mark.asyncio
    async def test_summariser_failure_fails_open_and_still_releases_the_lease(
        self, boundary_world, monkeypatch
    ):
        """AC4: a failing summariser leaves the body untouched, dispatch continues."""
        world = boundary_world
        _install_local_summariser_transport(
            monkeypatch, world.trace, status=500
        )
        await _serve_local_turn(world, monkeypatch, _INT_SESSION, _messages(2))

        response, request, raw_before = await _dispatch_remote_turn(
            world, monkeypatch, _INT_SESSION, _big_integration_messages()
        )

        assert response.status_code == 200
        assert len(world.trace.payloads("summariser_request")) == 1
        remote = world.trace.payloads("remote_request")[0]
        assert remote["raw"] == raw_before, "history is byte-identical on failure"
        assert request._boundary_compaction_result["reason"] == (
            "boundary_summarizer_failed"
        )
        # Subsequent turns still release: the slot must not stay pinned.
        assert _INT_SESSION not in world.server.local_dispatch_records
        assert world.session._assigned_slot_for_session(_INT_SESSION) is None
