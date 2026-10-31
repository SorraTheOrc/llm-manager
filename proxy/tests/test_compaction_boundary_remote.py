"""Local→remote boundary compaction contract — test-first specification.

Feature: LP-0MVRBQBDF0037OZ5 ("Compact before moving from local to remote").
This module: LP-0MVTDU4P8001L4II — the TDD-red contract for the boundary
decision (LP-0MVTDU6Y1004ZCGX), the slot pin plus immediate lease release
(LP-0MVTDU91C003Q9YZ) and the dispatch hook that consumes them
(LP-0MVTDUBR00038ZRL).

The point of this module is that the *trigger*, the *slot pin* and the
*release ordering* are specified by executable assertions before any
production code exists: every test here fails only because the boundary
symbols are absent (``missing boundary-compaction symbol …``), never because
of an unrelated import error, and it turns green only when the contract below
is implemented.

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

TDD-red convention
------------------

The not-yet-implemented contract is pinned with ``pytest.mark.xfail(
strict=True)`` — the established test-first convention in this suite
(``test_local_only_refusal_matrix.py``, ``test_compaction_headers_contract.py``,
``test_local_resident_prefix_admission.py``): the rows stay red without
breaking the green tree, and ``strict`` means the implementing children cannot
land without flipping the marker off with the behaviour. LP-0MVTDU6Y1004ZCGX
has removed the decision/config markers (see ``TestBoundaryDecision``,
``TestBoundaryConfig``, plus this session's ``TestLocalProvenance``,
``TestBoundaryConfigValidation`` and ``TestLocalCompletionProvenanceWiring``
below); LP-0MVTDU91C003Q9YZ removed the slot-pin and release-ordering markers
(``TestSlotPinnedSummariser``, ``TestSlotReleaseOrdering``) when
``build_slot_pinned_summarizer`` and ``summarise_on_session_slot`` landed, so
the only remaining markers belong to the not-yet-wired dispatch hook. A red
row afterwards is a real failure.

The one config assertion that already holds (``compaction_trigger_ratio`` stays
``0`` — the boundary knob never re-enables assembly-time compaction) is
**not** marked: it is the regression guard that stops the knob being folded
into the assembly trigger.

Constraints honoured here: injected fakes only — no live llama-server, GPU or
network access, no sleeps, deterministic.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import proxy.router_helpers as router_helpers
import pytest

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
