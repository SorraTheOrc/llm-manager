"""Oversized-session enforcement: one path, one knob (LP-0MTVXP7DG00613ZB).

Covers the four acceptance criteria:

- AC1: ``compaction_remote_with_guidance`` is enforced — non-compactable
  oversized sessions route remote with guidance, never local near-full-slot.
- AC2: a live compaction survives the next request — the session is NOT
  invalidated with ``history_mismatch``; the compacted history persists and
  the delta protocol heals.
- AC3: a single detection knob remains (``compaction_trigger_ratio``);
  ``context_pressure_warn_ratio`` and ``local_hard_routing_cap_ratio_*`` are
  removed from the live configuration surface.
- AC4: these tests cover guidance enforcement, post-compaction sync, and the
  unified trigger.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_request(body: dict, session_id: str | None = "sess-oversized"):
    class _DummyRequest:
        def __init__(self):
            self._body = json.dumps(body).encode()
            self.headers = {
                "content-type": "application/json",
                "x-session-id": session_id or "",
            }
            self.method = "POST"
            self.url = type("U", (), {"path": "/v1/chat/completions"})()

        async def body(self):
            return self._body

    return _DummyRequest()


def _session_result(**overrides):
    result = {
        "session_id": "sess-oversized",
        "session_id_header": "sess-oversized",
        "session_explicit": True,
        "session_created": False,
        "is_delta_request": False,
        "session_fallback_reason": None,
        "delta_messages": None,
        "original_message_count": 1,
        "body_json": {"model": "Qwen3"},
        "body_override": None,
    }
    result.update(overrides)
    return result


def _patch_router_harness(monkeypatch, server_cfg):
    """Patch the router/server globals and return the proxy_to_local callable."""
    import proxy.router as router_mod
    from proxy.router import proxy_to_local

    from proxy import server as srv

    monkeypatch.setattr(srv, "config", {"server": server_cfg})
    proc = MagicMock()
    proc.poll.return_value = None
    monkeypatch.setattr(srv, "llama_process", proc)
    monkeypatch.setattr(srv, "backend_ready", True)
    monkeypatch.setattr(srv, "current_model", "Qwen3")
    monkeypatch.setattr(srv, "active_queries", 0)
    monkeypatch.setattr(srv, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_active_queries", 0)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_dispatch_records", {})
    monkeypatch.setattr(srv, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "backend_signal_counts", {})

    monkeypatch.setattr(router_mod, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(router_mod, "_build_slot_context", lambda *_: (None, None, 3.0))
    monkeypatch.setattr(router_mod, "_resolve_slot_model_name", lambda model, *_: model)
    monkeypatch.setattr(router_mod, "_check_slot_availability", AsyncMock(return_value=None))
    return proxy_to_local, router_mod


def _make_server(**overrides):
    srv = MagicMock()
    srv.config = {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
            "compaction_dry_run": False,
            "summarizer_model": {"type": "local", "llama_model": "Qwen3"},
            **overrides,
        }
    }
    srv.logger = MagicMock()
    srv.session_manager = MagicMock()
    return srv


# ---------------------------------------------------------------------------
# AC3 — unified detection knob
# ---------------------------------------------------------------------------


class TestUnifiedDetectionKnob:
    """AC3: only ``compaction_trigger_ratio`` remains as a detection knob."""

    def test_warn_ratio_is_compaction_trigger(self):
        from proxy.provider import _get_context_pressure_warn_ratio

        config = {"server": {"compaction_trigger_ratio": 0.55}}
        assert _get_context_pressure_warn_ratio(config) == pytest.approx(0.55)

    def test_legacy_warn_ratio_key_is_ignored(self):
        """A stale ``context_pressure_warn_ratio`` key must not re-create the knob."""
        from proxy.provider import _get_context_pressure_warn_ratio

        config = {
            "server": {
                "compaction_trigger_ratio": 0.70,
                "context_pressure_warn_ratio": 0.10,
            }
        }
        assert _get_context_pressure_warn_ratio(config) == pytest.approx(0.70)

    def test_warning_fires_at_the_compaction_trigger(self):
        """The advisory threshold equals the compaction trigger (0.70 × clamp)."""
        from proxy.provider import should_warn_context_pressure

        # ctx 262144, 4 slots -> per-slot clamp 61440; trigger 0.70 -> 43008.
        config = {
            "server": {
                "local_model_ctx_size": 262144,
                "session_slot_pool_size": 4,
                "compaction_trigger_ratio": 0.70,
            }
        }
        assert should_warn_context_pressure(43008, config) is False
        assert should_warn_context_pressure(43009, config) is True

    def test_warning_disabled_when_compaction_disabled(self):
        from proxy.provider import should_warn_context_pressure

        config = {
            "server": {
                "local_model_ctx_size": 262144,
                "session_slot_pool_size": 4,
                "compaction_trigger_ratio": 0,
            }
        }
        assert should_warn_context_pressure(999999, config) is False

    @pytest.mark.parametrize(
        "config_file",
        ["config.yaml", "config-fast.yaml", "config-cheap.yaml"],
    )
    def test_live_configs_expose_one_detection_knob(self, config_file):
        import yaml

        proxy_dir = Path(__file__).resolve().parent.parent
        with open(proxy_dir / config_file) as fh:
            cfg = yaml.safe_load(fh)
        server = cfg["server"]
        assert "context_pressure_warn_ratio" not in server, config_file
        assert "local_hard_routing_cap_ratio_fast" not in server, config_file
        assert "local_hard_routing_cap_ratio_cheap" not in server, config_file
        assert "compaction_trigger_ratio" in server, config_file


# ---------------------------------------------------------------------------
# AC1 — guidance enforcement
# ---------------------------------------------------------------------------


class TestRemoteWithGuidanceEnforcement:
    """AC1: ``remote_with_guidance`` never dispatches local near-full-slot."""

    @pytest.mark.asyncio
    async def test_proxy_to_local_routes_remote_with_guidance(self, monkeypatch):
        import proxy.provider as provider_mod
        from fastapi.responses import JSONResponse

        server_cfg = {
            "llama_server_port": 8080,
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "max_concurrent_queries": 16,
        }
        proxy_to_local, router_mod = _patch_router_harness(monkeypatch, server_cfg)

        session_result = _session_result(
            compaction_remote_with_guidance=True,
            compaction_estimated_before=90000,
            compaction_reason="summarizer_unavailable",
        )
        monkeypatch.setattr(router_mod, "_handle_session", AsyncMock(return_value=session_result))

        from proxy import server as srv

        monkeypatch.setattr(
            srv,
            "get_model_config",
            lambda name: {
                "providers": [
                    {"name": "local-qwen3", "type": "local"},
                    {"name": "opencode-go", "type": "remote"},
                ]
            },
        )

        captured = {}

        async def fake_remote(request, path, model_config, config):
            captured["providers"] = model_config["providers"]
            captured["path"] = path
            return JSONResponse(status_code=200, content={"ok": True})

        monkeypatch.setattr(provider_mod, "proxy_with_remote_fallback", fake_remote)

        resp = await proxy_to_local(_dummy_request({"model": "Qwen3", "messages": []}), "v1/chat/completions")

        assert captured["providers"] == [{"name": "opencode-go", "type": "remote"}]
        assert captured["path"] == "v1/chat/completions"
        guidance = resp.headers.get("X-Session-Compaction-Guidance")
        assert guidance is not None and "context_pressure" in guidance
        router_mod._check_slot_availability.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_remote_provider_returns_compaction_gate(self, monkeypatch):
        server_cfg = {
            "llama_server_port": 8080,
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "max_concurrent_queries": 16,
        }
        proxy_to_local, router_mod = _patch_router_harness(monkeypatch, server_cfg)

        session_result = _session_result(
            compaction_remote_with_guidance=True,
            compaction_estimated_before=90000,
            compaction_reason="summarizer_failed",
        )
        monkeypatch.setattr(router_mod, "_handle_session", AsyncMock(return_value=session_result))

        from proxy import server as srv

        monkeypatch.setattr(
            srv,
            "get_model_config",
            lambda name: {"providers": [{"name": "local-qwen3", "type": "local"}]},
        )

        resp = await proxy_to_local(_dummy_request({"model": "Qwen3", "messages": []}), "v1/chat/completions")

        assert resp.status_code == 429
        assert resp.headers.get("X-Compaction-Gate") == "true"
        assert "context_pressure" in resp.headers.get("X-Session-Compaction-Guidance", "")
        router_mod._check_slot_availability.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_session_sets_flag_only_in_live_mode(self, monkeypatch):
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        mock_session = MagicMock(
            session_id="sess-live", message_count=200, restore_confirmed=False,
        )
        mock_session.messages = [{"role": "user", "content": "hi"}]
        srv.session_manager.get_or_create = AsyncMock(return_value=(mock_session, False))
        srv.session_manager.compute_delta = MagicMock(return_value=([{"role": "user", "content": "new"}], True))
        srv.session_manager.update_messages = AsyncMock()
        srv.session_manager.mark_compacted = AsyncMock()

        def fake_eval(*a, **kw):
            return {
                "action": "remote_with_guidance",
                "applied": False,
                "dry_run": False,
                "messages": [],
                "reason": "summarizer_unavailable",
                "estimated_before": 90000,
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("proxy.router_helpers._evaluate_session_compaction", fake_eval)
            mp.setattr("proxy.compaction_summarizer.build_local_summarizer", lambda *a, **k: MagicMock(return_value="s"))
            result = await _handle_session(
                srv,
                {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
                srv.config["server"],
                {"x-session-id": "sess-live"},
            )

        assert result.get("compaction_remote_with_guidance") is True
        assert result.get("compaction_reason") == "summarizer_unavailable"

    @pytest.mark.asyncio
    async def test_handle_session_dry_run_does_not_set_flag(self):
        from proxy.router_helpers import _handle_session

        srv = _make_server(compaction_dry_run=True)
        mock_session = MagicMock(session_id="sess-dry", message_count=200, restore_confirmed=False)
        mock_session.messages = [{"role": "user", "content": "hi"}]
        srv.session_manager.get_or_create = AsyncMock(return_value=(mock_session, False))
        srv.session_manager.compute_delta = MagicMock(return_value=([], True))
        srv.session_manager.update_messages = AsyncMock()
        srv.session_manager.mark_compacted = AsyncMock()

        def fake_eval(*a, **kw):
            return {
                "action": "remote_with_guidance",
                "applied": False,
                "dry_run": True,
                "messages": [],
                "reason": "summarizer_unavailable",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("proxy.router_helpers._evaluate_session_compaction", fake_eval)
            mp.setattr("proxy.compaction_summarizer.build_local_summarizer", lambda *a, **k: MagicMock(return_value="s"))
            result = await _handle_session(
                srv,
                {"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}]},
                srv.config["server"],
                {"x-session-id": "sess-dry"},
            )

        assert "compaction_remote_with_guidance" not in result


# ---------------------------------------------------------------------------
# AC2 — post-compaction sync heal
# ---------------------------------------------------------------------------


_SUMMARY = (
    "The conversation history before this point was compacted into the following summary:\n\n"
    "<summary>\nfolded middle turns\n</summary>"
)


def _compacted_base():
    """A compacted base: system + first user + summary + two recent turns."""
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "FIRST"},
        {"role": "user", "content": _SUMMARY},
        {"role": "assistant", "content": "recent-a"},
        {"role": "user", "content": "recent-u"},
    ]


def _pre_compaction_history():
    """The client's full pre-compaction history (length 6, anchors the recent tail)."""
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "FIRST"},
        {"role": "assistant", "content": "old-a"},
        {"role": "user", "content": "old-u"},
        {"role": "assistant", "content": "recent-a"},
        {"role": "user", "content": "recent-u"},
    ]


def _heal_session(messages, from_count, base_count):
    return SimpleNamespace(
        messages=list(messages),
        compacted_from_count=from_count,
        compacted_base_count=base_count,
    )


class TestPostCompactionHeal:
    """AC2: live compaction survives the next request via the delta heal."""

    def test_compute_post_compaction_delta_skips_appended_response(self):
        """The assistant appended after compaction is not re-sent as a delta."""
        from proxy.session_manager import compute_post_compaction_delta

        base = _compacted_base()
        stored = base + [{"role": "assistant", "content": "assistant-after"}]
        session = _heal_session(stored, from_count=6, base_count=len(base))
        # Client still sends its full pre-compaction history + the assistant
        # it received + the genuinely new turn.
        incoming = _pre_compaction_history() + [
            {"role": "assistant", "content": "assistant-after"},
            {"role": "assistant", "content": "new"},
        ]
        delta = compute_post_compaction_delta(session, incoming)
        assert delta == [{"role": "assistant", "content": "new"}]

    def test_compute_post_compaction_delta_heals_repeatedly(self):
        """The anchors are stable, so every subsequent request heals."""
        from proxy.session_manager import compute_post_compaction_delta

        base = _compacted_base()
        stored = base + [{"role": "assistant", "content": "assistant-after"}]
        session = _heal_session(stored, from_count=6, base_count=len(base))
        # Second post-compaction request: more appended history + a new turn.
        stored2 = stored + [
            {"role": "assistant", "content": "healed-1"},
            {"role": "assistant", "content": "assistant-after-2"},
        ]
        session.messages = stored2
        incoming = _pre_compaction_history() + [
            {"role": "assistant", "content": "assistant-after"},
            {"role": "assistant", "content": "healed-1"},
            {"role": "assistant", "content": "assistant-after-2"},
            {"role": "assistant", "content": "new-2"},
        ]
        assert compute_post_compaction_delta(session, incoming) == [
            {"role": "assistant", "content": "new-2"}
        ]

    def test_compute_post_compaction_delta_requires_marker(self):
        from proxy.session_manager import compute_post_compaction_delta

        session = _heal_session(_pre_compaction_history(), from_count=6, base_count=3)
        incoming = _pre_compaction_history() + [{"role": "assistant", "content": "new"}]
        assert compute_post_compaction_delta(session, incoming) is None

    def test_compute_post_compaction_delta_rejects_unrelated_history(self):
        from proxy.session_manager import compute_post_compaction_delta

        base = _compacted_base()
        session = _heal_session(base, from_count=6, base_count=len(base))
        # The tail at the anchored offset does not match the compacted turns.
        incoming = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "FIRST"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "y"},
            {"role": "assistant", "content": "DIFFERENT"},
            {"role": "user", "content": "ALSO-DIFFERENT"},
            {"role": "assistant", "content": "new"},
        ]
        assert compute_post_compaction_delta(session, incoming) is None

    def test_compute_post_compaction_delta_requires_anchors(self):
        from proxy.session_manager import compute_post_compaction_delta

        incoming = _pre_compaction_history() + [{"role": "assistant", "content": "new"}]
        base = _compacted_base()
        # No recorded offset (never compacted).
        no_offset = _heal_session(base, from_count=None, base_count=len(base))
        assert compute_post_compaction_delta(no_offset, incoming) is None
        # No recorded base length.
        no_base = _heal_session(base, from_count=6, base_count=None)
        assert compute_post_compaction_delta(no_base, incoming) is None
        # No new messages beyond the anchor.
        no_new = _heal_session(base, from_count=6, base_count=len(base))
        assert compute_post_compaction_delta(no_new, _pre_compaction_history()) is None

    @pytest.mark.asyncio
    async def test_live_compaction_records_anchors(self):
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        pre = _pre_compaction_history()
        compacted = _compacted_base()
        session = MagicMock(session_id="sess-compact", message_count=len(pre), restore_confirmed=False)
        session.messages = list(pre)
        srv.session_manager.get_or_create = AsyncMock(return_value=(session, False))
        srv.session_manager.compute_delta = MagicMock(return_value=([], True))
        srv.session_manager.update_messages = AsyncMock()
        srv.session_manager.mark_compacted = AsyncMock()

        def fake_eval(*a, **kw):
            return {
                "action": "compact",
                "applied": True,
                "dry_run": False,
                "messages": compacted,
                "estimated_before": 90000,
                "estimated_after": 10000,
                "mode": "fast",
                "reason": "compacted_within_target",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("proxy.router_helpers._evaluate_session_compaction", fake_eval)
            mp.setattr("proxy.compaction_summarizer.build_local_summarizer", lambda *a, **k: MagicMock(return_value="s"))
            result = await _handle_session(
                srv, {"model": "Qwen3", "messages": list(pre)}, srv.config["server"],
                {"x-session-id": "sess-compact"},
            )

        assert result["compaction_applied"] is True
        srv.session_manager.mark_compacted.assert_awaited_once_with(
            "sess-compact", len(pre), len(compacted),
        )

    @pytest.mark.asyncio
    async def test_compaction_includes_the_current_turn(self):
        """A delta request's new turn is part of the compacted dispatch body."""
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        pre = _pre_compaction_history()
        new_turn = [{"role": "user", "content": "brand-new-turn"}]
        session = MagicMock(session_id="sess-delta", message_count=len(pre), restore_confirmed=False)
        session.messages = list(pre)
        srv.session_manager.get_or_create = AsyncMock(return_value=(session, False))
        # Real delta classification: the new turn extends the stored history.
        srv.session_manager.compute_delta = MagicMock(return_value=(list(new_turn), True))
        srv.session_manager.update_messages = AsyncMock()
        srv.session_manager.mark_compacted = AsyncMock()
        captured = {}

        def fake_eval(srv_arg, sid, msgs, mode, **kw):
            captured["msgs"] = list(msgs)
            return {
                "action": "compact", "applied": True, "dry_run": False,
                "messages": _compacted_base(), "estimated_before": 90000,
                "estimated_after": 10000, "mode": "fast",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("proxy.router_helpers._evaluate_session_compaction", fake_eval)
            mp.setattr("proxy.compaction_summarizer.build_local_summarizer", lambda *a, **k: MagicMock(return_value="s"))
            await _handle_session(
                srv, {"model": "Qwen3", "messages": list(pre + new_turn)}, srv.config["server"],
                {"x-session-id": "sess-delta"},
            )

        assert captured["msgs"] == pre + new_turn

    @pytest.mark.asyncio
    async def test_next_request_heals_instead_of_invalidating(self, monkeypatch):
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        base = _compacted_base()
        # Session state after a live compaction: compacted base + assistant, and
        # the client still resending its full pre-compaction history.
        session = MagicMock(session_id="sess-heal", message_count=len(base) + 1, restore_confirmed=False)
        session.messages = list(base) + [{"role": "assistant", "content": "assistant-after"}]
        session.compacted_from_count = 6
        session.compacted_base_count = len(base)
        srv.session_manager.get_or_create = AsyncMock(return_value=(session, False))
        incoming = _pre_compaction_history() + [
            {"role": "assistant", "content": "assistant-after"},
            {"role": "assistant", "content": "new"},
        ]
        srv.session_manager.compute_delta = MagicMock(return_value=(list(incoming), False))
        srv.session_manager.update_messages = AsyncMock()
        srv.session_manager.mark_compacted = AsyncMock()

        invalidate = AsyncMock()
        monkeypatch.setattr("proxy.session._invalidate_session_and_slot", invalidate)

        def fake_eval(*a, **kw):
            return {"action": "noop", "applied": False, "dry_run": False, "messages": [], "reason": "below_trigger"}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("proxy.router_helpers._evaluate_session_compaction", fake_eval)
            mp.setattr("proxy.compaction_summarizer.build_local_summarizer", lambda *a, **k: MagicMock(return_value="s"))
            result = await _handle_session(
                srv, {"model": "Qwen3", "messages": list(incoming)}, srv.config["server"],
                {"x-session-id": "sess-heal"},
            )

        invalidate.assert_not_awaited()
        assert result["is_delta_request"] is True
        assert result["delta_messages"] == [{"role": "assistant", "content": "new"}]
        assert json.loads(result["body_override"])["messages"] == [
            {"role": "assistant", "content": "new"}
        ]

    @pytest.mark.asyncio
    async def test_end_to_end_compaction_then_heal(self, monkeypatch):
        """A real SessionManager: live compaction, then the next request heals."""
        from proxy.router_helpers import _handle_session
        from proxy.session_manager import SessionManager

        manager = SessionManager()
        sid = "sess-e2e"
        pre = _pre_compaction_history()  # 6 messages
        new_turn = [{"role": "user", "content": "u-new"}]
        first_incoming = pre + new_turn  # the delta request's produced history
        await manager.update_messages(sid, list(pre))

        srv = _make_server()
        srv.session_manager = manager
        compacted = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "FIRST"},
            {"role": "user", "content": _SUMMARY},
            {"role": "assistant", "content": "recent-a"},
            {"role": "user", "content": "recent-u"},
            {"role": "user", "content": "u-new"},
        ]

        def fake_eval(*a, **kw):
            return {
                "action": "compact", "applied": True, "dry_run": False,
                "messages": compacted, "estimated_before": 90000,
                "estimated_after": 10000, "mode": "fast",
                "reason": "compacted_within_target",
            }

        invalidate = AsyncMock()
        monkeypatch.setattr("proxy.session._invalidate_session_and_slot", invalidate)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("proxy.router_helpers._evaluate_session_compaction", fake_eval)
            mp.setattr("proxy.compaction_summarizer.build_local_summarizer", lambda *a, **k: MagicMock(return_value="s"))
            r1 = await _handle_session(
                srv, {"model": "Qwen3", "messages": list(first_incoming)},
                srv.config["server"], {"x-session-id": sid},
            )

        assert r1["compaction_applied"] is True
        assert r1["is_delta_request"] is False
        stored = await manager.get(sid)
        assert stored.compacted_from_count == len(first_incoming)
        assert stored.compacted_base_count == len(compacted)

        # The response-time merge appends the assistant to the compacted base.
        await manager.append_messages(sid, [{"role": "assistant", "content": "assistant-after"}])

        # Next request: the client still resends its pre-compaction history.
        second_incoming = first_incoming + [
            {"role": "assistant", "content": "assistant-after"},
            {"role": "assistant", "content": "next-new"},
        ]

        def fake_noop(*a, **kw):
            return {"action": "noop", "applied": False, "dry_run": False, "messages": [], "reason": "below_trigger"}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("proxy.router_helpers._evaluate_session_compaction", fake_noop)
            mp.setattr("proxy.compaction_summarizer.build_local_summarizer", lambda *a, **k: MagicMock(return_value="s"))
            r2 = await _handle_session(
                srv, {"model": "Qwen3", "messages": list(second_incoming)},
                srv.config["server"], {"x-session-id": sid},
            )

        invalidate.assert_not_awaited()
        assert r2["is_delta_request"] is True
        assert r2["delta_messages"] == [{"role": "assistant", "content": "next-new"}]
        # The compacted history persisted across the heal.
        stored2 = await manager.get(sid)
        assert stored2.messages[: len(compacted)] == compacted

    @pytest.mark.asyncio
    async def test_genuine_edit_still_invalidates(self, monkeypatch):
        """A true history edit (non-compaction mismatch) keeps the old behavior."""
        from proxy.router_helpers import _handle_session

        srv = _make_server()
        session = MagicMock(session_id="sess-edit", message_count=3, restore_confirmed=False)
        session.messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "FIRST"},
            {"role": "assistant", "content": "old"},
        ]
        session.compacted_from_count = None
        session.compacted_base_count = None
        srv.session_manager.get_or_create = AsyncMock(return_value=(session, False))
        incoming = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "EDITED"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "new"},
        ]
        srv.session_manager.compute_delta = MagicMock(return_value=(list(incoming), False))
        srv.session_manager.update_messages = AsyncMock()
        srv.session_manager.mark_compacted = AsyncMock()

        invalidate = AsyncMock()
        monkeypatch.setattr("proxy.session._invalidate_session_and_slot", invalidate)

        def fake_eval(*a, **kw):
            return {"action": "noop", "applied": False, "dry_run": False, "messages": [], "reason": "below_trigger"}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("proxy.router_helpers._evaluate_session_compaction", fake_eval)
            mp.setattr("proxy.compaction_summarizer.build_local_summarizer", lambda *a, **k: MagicMock(return_value="s"))
            await _handle_session(
                srv, {"model": "Qwen3", "messages": list(incoming)}, srv.config["server"],
                {"x-session-id": "sess-edit"},
            )

        invalidate.assert_awaited()
