"""Seed the routing ``cached_ratio`` from the persisted slot snapshot.

LP-0MVGD9YXH008O304. The proxy persists KV snapshots per session
(``slot_<session>.bin``) and restores them before a local dispatch, but the
routing cold-cache gate only knew about an in-memory ratio written by a
previous local response. A session with a warm prefix on disk was therefore
scored cold (82.8 % of routing decisions reported ``cached_ratio=0.00``) and
permanently bypassed local — and every proxy restart reset the whole fleet to
cold.

These tests cover the sidecar that closes the gap:

- AC1: a session with a persisted snapshot covering the cold threshold no
  longer produces ``large_context_bypass``;
- AC2: a snapshot whose message prefix does not match earns no credit;
- AC3: the credited ratio is capped by ``cached_tokens / estimated_tokens``;
- AC4: logs/labels distinguish ``cache_state=restored``;
- AC6: seeding works with no in-memory state (proxy restart).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from proxy.provider import (
    _cached_ratio_source,
    _get_cached_ratio,
    _get_cached_ratio_source,
    _last_cached_ratio,
    _should_skip_local,
    clear_cached_ratio_for_session,
)
from proxy.router_helpers import _classify_cache_state, _lookup_dispatch_cache_state
from proxy.session import _endpoint_host_port, _slot_filename_for_session
from proxy.slot_cache_ratio import (
    DEFAULT_SAFETY_MARGIN,
    delete_slot_ratio_sidecar,
    message_fingerprints,
    prefix_matches,
    ratio_from_sidecar,
    read_slot_ratio_sidecar,
    seed_cached_ratio,
    sidecar_path,
    write_slot_ratio_sidecar,
)

MODEL = "Qwen3"
SESSION = "sess-restore-1"


@pytest.fixture(autouse=True)
def _clean_provider_cache_state():
    """Isolate the process-global ratio dicts between tests."""
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    yield
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()


def _messages(count: int = 3, system: str = "You are a helpful agent.") -> list[dict]:
    out = [{"role": "system", "content": system}]
    for i in range(1, count):
        out.append({"role": "user", "content": f"turn {i}"})
    return out


def _write_snapshot(
    tmp_path,
    *,
    session_id: str = SESSION,
    model: str = MODEL,
    cached_tokens: int = 95_000,
    prompt_tokens: int = 95_000,
    messages: list[dict] | None = None,
    saved_at=None,
    endpoint: str | None = None,
    with_sidecar: bool = True,
    server_overrides: dict | None = None,
):
    """Create a slot snapshot (+ sidecar) and return the server config."""
    slot_root = tmp_path / "slot-cache"
    base = slot_root
    if endpoint:
        host, port = _endpoint_host_port(endpoint)
        base = slot_root / f"{host}-{port}"
    base.mkdir(parents=True, exist_ok=True)
    filename = _slot_filename_for_session(session_id, base)
    with open(filename, "wb") as handle:
        handle.write(b"KV-SNAPSHOT")
    if with_sidecar:
        assert write_slot_ratio_sidecar(
            filename,
            session_id=session_id,
            model=model,
            cached_tokens=cached_tokens,
            prompt_tokens=prompt_tokens,
            messages=messages if messages is not None else _messages(),
            saved_at=saved_at,
        )
    server = {"session_slot_save_path": str(slot_root)}
    server.update(server_overrides or {})
    return {"server": server}


class TestSidecarRoundTrip:
    def test_write_then_read(self, tmp_path):
        messages = _messages()
        config = _write_snapshot(tmp_path, messages=messages)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )

        meta = read_slot_ratio_sidecar(filename)

        assert meta is not None
        assert meta["session_id"] == SESSION
        assert meta["model"] == MODEL
        assert meta["cached_tokens"] == 95_000
        assert meta["messages"] == message_fingerprints(messages)

    def test_sidecar_sits_next_to_snapshot(self, tmp_path):
        config = _write_snapshot(tmp_path)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )

        assert sidecar_path(filename).exists()
        assert sidecar_path(filename).name == f"slot_{SESSION}.bin.ratio.json"

    def test_corrupt_sidecar_is_ignored(self, tmp_path):
        config = _write_snapshot(tmp_path)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )
        sidecar_path(filename).write_text("{not json", encoding="utf-8")

        assert read_slot_ratio_sidecar(filename) is None

    def test_unknown_version_is_ignored(self, tmp_path):
        config = _write_snapshot(tmp_path)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )
        sidecar_path(filename).write_text(
            json.dumps({"version": 999, "messages": []}), encoding="utf-8"
        )

        assert read_slot_ratio_sidecar(filename) is None

    def test_missing_sidecar_returns_none(self, tmp_path):
        config = _write_snapshot(tmp_path, with_sidecar=False)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )

        assert read_slot_ratio_sidecar(filename) is None

    def test_delete_removes_sidecar_only(self, tmp_path):
        config = _write_snapshot(tmp_path)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )

        delete_slot_ratio_sidecar(filename)

        assert not sidecar_path(filename).exists()
        assert Path(filename).exists()


class TestPrefixMatch:
    def test_unchanged_conversation_prefix_matches(self):
        stored = message_fingerprints(_messages(3))
        incoming = message_fingerprints(_messages(5))

        assert prefix_matches(stored, incoming) is True

    def test_changed_system_prompt_does_not_match(self):
        stored = message_fingerprints(_messages(3, system="old system prompt"))
        incoming = message_fingerprints(_messages(3, system="new system prompt"))

        assert prefix_matches(stored, incoming) is False

    def test_front_truncated_history_does_not_match(self):
        stored = message_fingerprints(_messages(4))
        incoming = message_fingerprints(_messages(4)[1:])

        assert prefix_matches(stored, incoming) is False

    def test_shorter_incoming_prefix_matches(self):
        stored = message_fingerprints(_messages(4))
        incoming = message_fingerprints(_messages(2))

        assert prefix_matches(stored, incoming) is True

    def test_empty_inputs_do_not_match(self):
        assert prefix_matches([], message_fingerprints(_messages())) is False
        assert prefix_matches(message_fingerprints(_messages()), []) is False


class TestRatioCap:
    """AC3 — restoration can never over-credit the measured tokens."""

    def test_measured_fraction_is_the_cap(self):
        meta = {"cached_tokens": 30_000}

        assert ratio_from_sidecar(meta, 100_000, margin=1.0) == pytest.approx(0.3)

    def test_safety_margin_only_reduces(self):
        meta = {"cached_tokens": 30_000}
        measured = ratio_from_sidecar(meta, 100_000, margin=1.0)
        margined = ratio_from_sidecar(meta, 100_000, margin=DEFAULT_SAFETY_MARGIN)

        assert margined < measured
        assert margined == pytest.approx(0.3 * DEFAULT_SAFETY_MARGIN)

    def test_ratio_never_exceeds_one(self):
        meta = {"cached_tokens": 500_000}

        assert ratio_from_sidecar(meta, 100_000, margin=1.0) == 1.0
        assert ratio_from_sidecar(meta, 100_000, margin=0.9) == pytest.approx(0.9)

    def test_missing_measurement_is_zero(self):
        assert ratio_from_sidecar({}, 100_000) == 0.0
        assert ratio_from_sidecar({"cached_tokens": 0}, 100_000) == 0.0
        assert ratio_from_sidecar({"cached_tokens": 10}, 0) == 0.0
        assert ratio_from_sidecar(None, 100_000) == 0.0


class TestSeedCachedRatio:
    def test_seeds_from_persisted_snapshot_with_no_memory(self, tmp_path):
        """AC6 — a proxy restart loses all in-memory state; disk wins."""
        messages = _messages()
        config = _write_snapshot(tmp_path, messages=messages, cached_tokens=95_000)

        assert _get_cached_ratio(MODEL, SESSION) == 0.0  # nothing in memory

        source = seed_cached_ratio(
            config["server"], {"messages": messages}, MODEL, SESSION, 100_000
        )

        assert source == "restored"
        assert _get_cached_ratio_source(MODEL, SESSION) == "restored"
        assert _get_cached_ratio(MODEL, SESSION) == pytest.approx(
            0.95 * DEFAULT_SAFETY_MARGIN
        )

    def test_missing_snapshot_earns_no_credit(self, tmp_path):
        config = _write_snapshot(tmp_path)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )
        Path(filename).unlink()

        assert (
            seed_cached_ratio(
                config["server"], {"messages": _messages()}, MODEL, SESSION, 100_000
            )
            is None
        )
        assert _get_cached_ratio(MODEL, SESSION) == 0.0

    def test_missing_sidecar_earns_no_credit(self, tmp_path):
        config = _write_snapshot(tmp_path, with_sidecar=False)

        assert (
            seed_cached_ratio(
                config["server"], {"messages": _messages()}, MODEL, SESSION, 100_000
            )
            is None
        )
        assert _get_cached_ratio(MODEL, SESSION) == 0.0

    def test_changed_system_prompt_earns_no_credit(self, tmp_path):
        """AC2 — the incoming prompt must prefix-match the snapshot."""
        config = _write_snapshot(
            tmp_path, messages=_messages(3, system="original system prompt")
        )
        incoming = _messages(3, system="a completely different system prompt")

        assert (
            seed_cached_ratio(
                config["server"], {"messages": incoming}, MODEL, SESSION, 100_000
            )
            is None
        )
        assert _get_cached_ratio(MODEL, SESSION) == 0.0

    def test_other_model_earns_no_credit(self, tmp_path):
        config = _write_snapshot(tmp_path, model="SomeOtherModel")

        assert (
            seed_cached_ratio(
                config["server"], {"messages": _messages()}, MODEL, SESSION, 100_000
            )
            is None
        )
        assert _get_cached_ratio(MODEL, SESSION) == 0.0

    def test_expired_sidecar_earns_no_credit(self, tmp_path):
        from datetime import UTC, datetime, timedelta

        stale = datetime.now(UTC) - timedelta(days=3)
        config = _write_snapshot(tmp_path, saved_at=stale)

        assert (
            seed_cached_ratio(
                config["server"], {"messages": _messages()}, MODEL, SESSION, 100_000
            )
            is None
        )
        assert _get_cached_ratio(MODEL, SESSION) == 0.0

    def test_ttl_zero_disables_expiry(self, tmp_path):
        from datetime import UTC, datetime, timedelta

        stale = datetime.now(UTC) - timedelta(days=30)
        config = _write_snapshot(
            tmp_path, saved_at=stale, server_overrides={"session_slot_ratio_ttl_seconds": 0}
        )

        assert (
            seed_cached_ratio(
                config["server"], {"messages": _messages()}, MODEL, SESSION, 100_000
            )
            == "restored"
        )

    def test_configured_safety_margin_is_honoured(self, tmp_path):
        config = _write_snapshot(
            tmp_path,
            cached_tokens=50_000,
            server_overrides={"session_slot_ratio_safety_margin": 0.5},
        )

        seed_cached_ratio(
            config["server"], {"messages": _messages()}, MODEL, SESSION, 100_000
        )

        assert _get_cached_ratio(MODEL, SESSION) == pytest.approx(0.25)

    def test_no_slot_persistence_configured_is_a_no_op(self, tmp_path):
        assert (
            seed_cached_ratio(
                {}, {"messages": _messages()}, MODEL, SESSION, 100_000
            )
            is None
        )

    def test_endpoint_specific_slot_directory(self, tmp_path):
        endpoint = "http://192.168.0.199:8080"
        config = _write_snapshot(tmp_path, endpoint=endpoint)

        source = seed_cached_ratio(
            config["server"],
            {"messages": _messages()},
            MODEL,
            SESSION,
            100_000,
            endpoint=endpoint,
        )

        assert source == "restored"

    def test_root_layout_snapshot_found_with_resolved_endpoint(self, tmp_path):
        """The router resolves a default endpoint (``localhost:8080``) even
        when the provider entry declares none, so a snapshot written at the
        slot-cache root must still be found."""
        config = _write_snapshot(tmp_path)  # root layout

        source = seed_cached_ratio(
            config["server"],
            {"messages": _messages()},
            MODEL,
            SESSION,
            100_000,
            endpoint="http://localhost:8080",
        )

        assert source == "restored"

    def test_seeding_never_overwrites_with_zero(self, tmp_path):
        """A rejected seed leaves an existing ratio untouched."""
        config = _write_snapshot(tmp_path, messages=_messages(3, system="old"))
        from proxy.provider import set_cached_ratio

        set_cached_ratio(MODEL, SESSION, 0.42)

        assert (
            seed_cached_ratio(
                config["server"],
                {"messages": _messages(3, system="new")},
                MODEL,
                SESSION,
                100_000,
            )
            is None
        )
        assert _get_cached_ratio(MODEL, SESSION) == pytest.approx(0.42)


class TestRoutingGate:
    """AC1 — a restored snapshot stops the cold-cache bypass."""

    COLD = 38_000

    def test_snapshot_credit_removes_large_context_bypass(self, tmp_path):
        messages = _messages()
        config = _write_snapshot(tmp_path, messages=messages, cached_tokens=95_000)
        body = {"messages": messages}
        estimated = 120_000

        # Baseline (today's behaviour): no ratio → economic bypass.
        skip, reason = _should_skip_local(
            MODEL, SESSION, body, self.COLD, estimated_tokens=estimated
        )
        assert (skip, reason) == (True, "large_context_bypass")

        seed_cached_ratio(config["server"], body, MODEL, SESSION, estimated)

        skip, reason = _should_skip_local(
            MODEL, SESSION, body, self.COLD, estimated_tokens=estimated
        )
        assert (skip, reason) == (False, None)

    def test_small_partial_prefix_still_bypasses(self, tmp_path):
        """A weak credit must not mask a genuinely expensive prefill."""
        messages = _messages()
        config = _write_snapshot(tmp_path, messages=messages, cached_tokens=10_000)
        body = {"messages": messages}
        estimated = 120_000

        seed_cached_ratio(config["server"], body, MODEL, SESSION, estimated)

        skip, reason = _should_skip_local(
            MODEL, SESSION, body, self.COLD, estimated_tokens=estimated
        )
        assert (skip, reason) == (True, "large_context_bypass")


class TestCacheStateLabel:
    """AC4 — restored-prefix wins are labelled separately."""

    def test_restored_label_only_when_warm(self):
        assert _classify_cache_state(0.9, "restored") == "restored"
        assert _classify_cache_state(0.9) == "warm"
        assert _classify_cache_state(0.2, "restored") == "cold"

    def test_dispatch_lookup_reports_restored(self, tmp_path):
        messages = _messages()
        config = _write_snapshot(tmp_path, messages=messages)

        seed_cached_ratio(
            config["server"], {"messages": messages}, MODEL, SESSION, 100_000
        )

        state, ratio = _lookup_dispatch_cache_state(MODEL, SESSION)
        assert state == "restored"
        assert ratio > 0.5


class TestSessionInvalidation:
    def test_invalidation_discards_sidecar_and_ratio(self, tmp_path):
        import asyncio

        from proxy.session import _invalidate_session_and_slot

        messages = _messages()
        config = _write_snapshot(tmp_path, messages=messages)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )
        seed_cached_ratio(
            config["server"], {"messages": messages}, MODEL, SESSION, 100_000
        )
        assert _get_cached_ratio(MODEL, SESSION) > 0.0

        asyncio.run(_invalidate_session_and_slot(SESSION, "history_mismatch", filename))

        assert not sidecar_path(filename).exists()
        assert _get_cached_ratio(MODEL, SESSION) == 0.0
        assert _get_cached_ratio_source(MODEL, SESSION) is None

    def test_clear_cached_ratio_for_session_leaves_other_sessions(self):
        from proxy.provider import set_cached_ratio

        set_cached_ratio(MODEL, SESSION, 0.8, source="restored")
        set_cached_ratio(MODEL, "other-session", 0.5)

        clear_cached_ratio_for_session(SESSION)

        assert _get_cached_ratio(MODEL, SESSION) == 0.0
        assert _get_cached_ratio_source(MODEL, SESSION) is None
        assert _get_cached_ratio(MODEL, "other-session") == pytest.approx(0.5)


class TestSidecarWriterOnSave:
    """The writer half: a successful slot save persists the measured state."""

    def _srv(self):
        srv = MagicMock()
        srv.current_model = MODEL
        srv.session_manager = MagicMock()
        srv.session_manager.set_restore_confirmed = AsyncMock()
        srv.session_manager.update_messages = AsyncMock()
        srv.session_manager.get = AsyncMock(return_value=None)
        srv.logger = MagicMock()
        return srv

    def _response(self, prompt_tokens: int, cached_tokens: int):
        body = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        }
        response = MagicMock()
        response.content = json.dumps(body).encode("utf-8")
        response.headers = {}
        return response

    def test_successful_save_writes_sidecar_with_prompt_tokens(
        self, tmp_path, monkeypatch
    ):
        import asyncio

        import proxy.router as router

        messages = _messages()
        config = _write_snapshot(tmp_path, messages=messages, with_sidecar=False)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )

        async def _fake_save(*_args, **_kwargs):
            return True

        monkeypatch.setattr(router, "_save_slot_snapshot", _fake_save, raising=False)
        monkeypatch.setattr(
            router, "_detect_restore_signal_from_llama_log", lambda *a, **k: False
        )

        asyncio.run(
            router._update_session_and_slot(
                self._srv(),
                SESSION,
                {"messages": messages},
                False,
                [],
                len(messages),
                self._response(prompt_tokens=90_000, cached_tokens=30_000),
                llama_port=8080,
                slot_id=0,
                slot_filename=filename,
                slot_timeout=3.0,
                slot_model_payload=MODEL,
                slot_enabled=True,
                upstream_status=200,
            )
        )

        meta = read_slot_ratio_sidecar(filename)
        assert meta is not None
        # The whole served prompt is now in the saved KV cache, not just the
        # fraction llama-server reported as cached for that request.
        assert meta["cached_tokens"] == 90_000
        assert meta["messages"] == message_fingerprints(messages)

    def test_failed_save_writes_no_sidecar(self, tmp_path, monkeypatch):
        import asyncio

        import proxy.router as router

        messages = _messages()
        config = _write_snapshot(tmp_path, messages=messages, with_sidecar=False)
        filename = _slot_filename_for_session(
            SESSION, config["server"]["session_slot_save_path"]
        )

        async def _fake_save(*_args, **_kwargs):
            return False

        monkeypatch.setattr(router, "_save_slot_snapshot", _fake_save, raising=False)
        monkeypatch.setattr(
            router, "_detect_restore_signal_from_llama_log", lambda *a, **k: False
        )

        asyncio.run(
            router._update_session_and_slot(
                self._srv(),
                SESSION,
                {"messages": messages},
                False,
                [],
                len(messages),
                self._response(prompt_tokens=90_000, cached_tokens=30_000),
                llama_port=8080,
                slot_id=0,
                slot_filename=filename,
                slot_timeout=3.0,
                slot_model_payload=MODEL,
                slot_enabled=True,
                upstream_status=200,
            )
        )

        assert read_slot_ratio_sidecar(filename) is None
