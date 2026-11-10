"""Identifying fields on the ``local_active_queries`` recovery WARNING.

Parent LP-0MVWF78QK0060TZH **AC5** (this item, LP-0MVYWLHNI000AQ7X): the
self-healing clamp ``_recover_stuck_local_active_queries`` is *the* detector
for the counter leak, but its WARNING named only the counts — so the 12 % of
recoveries in the AC2 evidence with no antecedent log signal in the preceding
30 s could not be attributed to a request without a fresh investigation. The
line now names the leaking request's identifying fields when they are
determinable:

``session=`` / ``turn=`` / ``record=`` / ``release_reason=`` / ``release_age_s=``

Why the fields are captured where they are
------------------------------------------

The clamp runs on a 10 s background tick, *outside* any request context, so it
cannot read the leaking turn from the context var. The identity is therefore
captured where it is already known and free to record:

* on the dispatch record at admission (``session`` / ``turn``), and
* in a bounded in-memory release history next to the existing
  ``lease_released session=... reason=...`` lines.

The clamp only *reads* those snapshots — no new lock, no new hot-path work
(AC2) — and the reset values are byte-for-byte what they were (AC4).

Acceptance-criterion map
------------------------

- **AC1** — ``TestRecoveryWarningIdentifyingFields`` asserts the WARNING text
  contains the session, turn, dispatch-record id and release reason when a
  record/release is present, and still fires with ``unknown`` placeholders when
  nothing is determinable.
- **AC2** — ``TestAdmissionAndReleaseCapture`` asserts the identity comes from
  state that already exists at admission/release (the record stamp and the
  ``lease_released`` reason), not from work done on the clamp tick.
- **AC3** — this file: a simulated leak (counter above the active-record count)
  driven through the real recovery function.
- **AC4** — the clamp assertions pin the unchanged ``reset from N to M`` text
  and counter value in every case.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import proxy.router_helpers as helpers
import proxy.server as server
import pytest
from proxy.turn_id import bind_turn_id, reset_turn_id


@pytest.fixture(autouse=True)
def _clean_turn_context():
    """Every test starts and ends with no bound turn (context vars leak)."""
    reset_turn_id()
    yield
    reset_turn_id()


def _warning_lines(logger: MagicMock) -> list[str]:
    """Render ``logger.warning`` calls back into their formatted lines."""
    lines: list[str] = []
    for call in getattr(logger, "warning").call_args_list:
        template = str(call.args[0]) if call.args else ""
        args = call.args[1:]
        if args and "%" in template:
            try:
                lines.append(template % args)
                continue
            except (TypeError, ValueError):
                pass
        lines.append(template + (" " + " ".join(str(a) for a in args) if args else ""))
    return lines


def _recovery_lines(srv) -> list[str]:
    return [
        line
        for line in _warning_lines(srv.logger)
        if "local_active_queries counter recovered" in line
    ]


def _server(**overrides) -> SimpleNamespace:
    """Minimal server state for the recovery clamp (records + counter only)."""
    state = {
        "local_active_queries": 0,
        "local_active_queries_lock": asyncio.Lock(),
        "local_dispatch_records": {},
        "local_dispatch_records_lock": asyncio.Lock(),
        "config": {"server": {}},
        "logger": MagicMock(),
    }
    state.update(overrides)
    return SimpleNamespace(**state)


def _active_record(session: str, turn: str, endpoint: str = "local-qwen3") -> dict:
    return {
        "backend": endpoint,
        "started_at": time.monotonic(),
        "active": True,
        "expires_at": time.monotonic() + 600,
        "model_name": "qwen3",
        "endpoint": endpoint,
        "session": session,
        "turn": turn,
        "dispatch_token": 1,
        "dispatch_credits": [1],
    }


class TestRecoveryWarningIdentifyingFields:
    """AC1/AC3/AC4 — the WARNING names the request it could not account for."""

    @pytest.mark.asyncio
    async def test_names_active_dispatch_and_release_reason(self):
        """A leak beside one live dispatch names that dispatch and the release."""
        key = ("local-qwen3", "sess-leak")
        srv = _server(
            local_active_queries=2,
            local_dispatch_records={key: _active_record("sess-leak", "abc12345")},
        )
        helpers._note_local_dispatch_release(
            srv,
            "sess-released",
            reason="measured_cache_abort",
            record={"session": "sess-released", "turn": "def67890"},
            record_key=("local-qwen3", "sess-released"),
        )

        await helpers._recover_stuck_local_active_queries(srv)

        lines = _recovery_lines(srv)
        assert len(lines) == 1, f"expected exactly one recovery WARNING, got {lines}"
        line = lines[0]
        # AC4: the reset values are unchanged, only the payload grew.
        assert "reset from 2 to 1 (1 active dispatch records)" in line
        assert srv.local_active_queries == 1
        # AC1: the live dispatch is named ...
        assert "session=sess-leak" in line
        assert "turn=abc12345" in line
        assert "record=local-qwen3|sess-leak" in line
        # ... and the recent release explains the extra increment.
        assert "release_reason=measured_cache_abort" in line
        assert "release_age_s=" in line and "unknown" not in line.split("release_age_s=")[1]

    @pytest.mark.asyncio
    async def test_names_most_recent_released_record_when_no_record_is_active(self):
        """With no active record, the just-released request is the only identity."""
        srv = _server(
            local_active_queries=1,
            local_dispatch_records={
                ("local-qwen3", "sess-gone"): {
                    "active": False,
                    "expires_at": time.monotonic() + 60,
                }
            },
        )
        helpers._note_local_dispatch_release(
            srv,
            "sess-gone",
            reason="client_disconnect",
            record={"session": "sess-gone", "turn": "0123abcd"},
            record_key=("local-qwen3", "sess-gone"),
        )

        await helpers._recover_stuck_local_active_queries(srv)

        line = _recovery_lines(srv)[0]
        assert "reset from 1 to 0 (0 active dispatch records)" in line
        assert srv.local_active_queries == 0
        assert "session=sess-gone" in line
        assert "turn=0123abcd" in line
        assert "record=local-qwen3|sess-gone" in line
        assert "release_reason=client_disconnect" in line

    @pytest.mark.asyncio
    async def test_degrades_to_unknown_placeholders_and_still_fires(self):
        """AC1: nothing determinable → the detector still fires, with `unknown`."""
        srv = _server(local_active_queries=3, local_dispatch_records={})

        await helpers._recover_stuck_local_active_queries(srv)

        line = _recovery_lines(srv)[0]
        assert "reset from 3 to 0 (0 active dispatch records)" in line
        assert srv.local_active_queries == 0
        assert "session=unknown" in line
        assert "turn=unknown" in line
        assert "record=unknown" in line
        assert "release_reason=unknown" in line
        assert "release_age_s=unknown" in line

    @pytest.mark.asyncio
    async def test_stale_release_is_not_attributed(self):
        """A release older than the window is churn, not the leak's cause."""
        srv = _server(local_active_queries=1, local_dispatch_records={})
        srv.local_dispatch_release_history = [
            {
                "session": "sess-old",
                "turn": "deadbeef",
                "record": "local-qwen3|sess-old",
                "reason": "idle_timeout",
                "released_at": time.monotonic()
                - (helpers.LOCAL_DISPATCH_RELEASE_MAX_AGE_SECONDS + 60),
            }
        ]

        await helpers._recover_stuck_local_active_queries(srv)

        line = _recovery_lines(srv)[0]
        assert "release_reason=unknown" in line
        assert "idle_timeout" not in line
        assert "sess-old" not in line
        assert srv.local_active_queries == 0


class TestAdmissionAndReleaseCapture:
    """AC2 — the fields come from state that already exists at admission/release."""

    @pytest.mark.asyncio
    async def test_explicit_admission_stamps_session_and_turn(self, monkeypatch):
        monkeypatch.setattr(server, "local_dispatch_records", {})
        monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
        monkeypatch.setattr(server, "local_active_queries", 0)
        monkeypatch.setattr(server, "local_active_queries_lock", asyncio.Lock())
        monkeypatch.setattr(server, "logger", MagicMock())
        monkeypatch.setattr(
            server, "config", {"server": {"local_dispatch_lease_timeout_seconds": 180}}
        )
        turn = bind_turn_id()

        acquired, _owner, _count, _retry = await helpers._try_acquire_local_dispatch(
            server,
            max_local=1,
            session_key="sess-explicit",
            backend="local-qwen3",
        )

        assert acquired is True
        record = server.local_dispatch_records[("local-qwen3", "sess-explicit")]
        assert record["session"] == "sess-explicit"
        assert record["turn"] == turn

    @pytest.mark.asyncio
    async def test_anonymous_admission_stamps_session_and_turn(self):
        srv = _server()
        turn = bind_turn_id()

        await helpers._increment_local_active_queries(
            srv,
            session_key="sess-anon",
            backend="local-qwen3",
            body_json={"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]},
        )

        record = srv.local_dispatch_records[("local-qwen3", "sess-anon")]
        assert record["session"] == "sess-anon"
        assert record["turn"] == turn

    @pytest.mark.asyncio
    async def test_measured_cache_abort_release_is_attributed(self):
        """The dominant leaking path records its reason for the next clamp tick."""
        key = ("local-qwen3", "sess-abort")
        srv = _server(local_dispatch_records={key: _active_record("sess-abort", "cafe0001")})

        removed, _record = await helpers._release_unserved_dispatch(
            srv,
            "sess-abort",
            endpoint="local-qwen3",
            release_reason="measured_cache_deficit",
            eviction_reason="measured_cache_deficit",
        )

        assert removed is True
        history = srv.local_dispatch_release_history
        assert history[-1]["session"] == "sess-abort"
        assert history[-1]["turn"] == "cafe0001"
        assert history[-1]["record"] == "local-qwen3|sess-abort"
        assert history[-1]["reason"] == "measured_cache_deficit"

    @pytest.mark.asyncio
    async def test_explicit_release_is_attributed(self):
        key = ("local-qwen3", "sess-explicit-release")
        srv = _server(
            local_dispatch_records={
                key: _active_record("sess-explicit-release", "beef0002")
            }
        )

        released = await helpers._release_local_dispatch(
            srv, "sess-explicit-release", endpoint="local-qwen3"
        )

        assert released is True
        history = srv.local_dispatch_release_history
        assert history[-1]["reason"] == "explicit_release"
        assert history[-1]["turn"] == "beef0002"
        assert history[-1]["record"] == "local-qwen3|sess-explicit-release"

    def test_release_history_is_bounded(self):
        """The snapshot is a fixed-size ring: no unbounded growth on the hot path."""
        srv = _server()
        for index in range(helpers.LOCAL_DISPATCH_RELEASE_HISTORY_LIMIT + 6):
            helpers._note_local_dispatch_release(
                srv, f"sess-{index}", reason=f"reason-{index}"
            )

        history = srv.local_dispatch_release_history
        assert len(history) == helpers.LOCAL_DISPATCH_RELEASE_HISTORY_LIMIT
        assert history[-1]["reason"] == (
            f"reason-{helpers.LOCAL_DISPATCH_RELEASE_HISTORY_LIMIT + 5}"
        )

    def test_release_capture_never_raises_on_a_partial_server(self):
        """A broken server object must not turn a release into a failure."""

        class _Hostile:
            def __setattr__(self, name, value):  # pragma: no cover - defensive
                raise RuntimeError("no attributes here")

        helpers._note_local_dispatch_release(
            _Hostile(), "sess", reason="client_disconnect"
        )
