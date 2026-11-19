"""Own-record attribution of ``local_concurrency_limit`` fall-throughs.

LP-0MWNHBI6A001V5DD (parent LP-0MWMSQR3O002YNPW AC5). The parent returns the
requesting session's own non-streaming dispatch record to the *requester* at
routing time. A raw ``reason=local_concurrency_limit`` count cannot show the
fix worked, because the line is still emitted for genuine contention. These
tests pin the classification rules of ``scripts/attribute_local_concurrency.py``
against synthetic log lines shaped exactly like the proxy's own:

* an own **idle lease** (``lease_renewed``) — LP-0MW0VH2LB008JSKQ;
* an own **finished-but-not-released** record (terminal chunk before teardown)
  — LP-0MWNHBG7D002NH5S, the parent's race;
* an own **streaming** record (a genuinely concurrent same-session request);
* **other sessions'** occupancy.

They also pin rotation (gzip), exact-duplicate de-duplication, the window
filter and the warm-up that establishes state admitted before the window.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import attribute_local_concurrency as attribution  # noqa: E402

WINDOW_START = datetime(2026, 11, 18, 12, 0, 0)
WINDOW_END = datetime(2026, 11, 18, 12, 30, 0)


def _ts(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S") + ",000"


def _fallback(
    moment: datetime, session: str, *, turn: str = "turn0001", attempt: int = 1
) -> str:
    return (
        f"{_ts(moment)} - INFO - Fallback triggered for model=v1/chat/completions, "
        f"from=local-qwen3, to=opencode-go-3, reason=local_concurrency_limit "
        f"session={session} attempt={attempt} turn={turn}"
    )


def _admit_local(moment: datetime, session: str) -> str:
    return (
        f"{_ts(moment)} - INFO - local_preference_resolved preference=remote-ok "
        f"outcome=local reason=none session={session} model=unknown"
    )


def _stream_started_local(moment: datetime, session: str, *, turn: str = "turn0001") -> str:
    return (
        f"{_ts(moment)} - INFO - Stream started: provider=local model=Qwen3 "
        f"session={session} request=[{{'type': 'text', 'text': 'x'}}] turn={turn}"
    )


def _stream_finished_local(
    moment: datetime, session: str, *, turn: str = "turn0001"
) -> str:
    return (
        f"{_ts(moment)} - INFO - Stream finished: reason=tool_calls "
        f"tokens=100/10/110 session={session} provider=local model=Qwen3 "
        f"request=[{{'type': 'text', 'text': 'x'}}] turn={turn}"
    )


def _stream_finished_remote(moment: datetime, session: str) -> str:
    return (
        f"{_ts(moment)} - INFO - Stream finished: reason=tool_calls "
        f"tokens=100/10/110 session={session} provider=OpenCode Go 3 "
        f"model=deepseek-v4.1-flash entry=opencode-go-3 "
        f"request=[{{'type': 'text', 'text': 'x'}}] turn=turn0001"
    )


def _lease_renewed(moment: datetime, session: str) -> str:
    return f"{_ts(moment)} - INFO - lease_renewed session={session} timeout=30s"


def _lease_released(moment: datetime, session: str, *, reason: str = "request_completed") -> str:
    return (
        f"{_ts(moment)} - WARNING - lease_released session={session} "
        f"reason={reason} endpoint=http://localhost:8080"
    )


def _eviction(moment: datetime, session: str) -> str:
    return (
        f"{_ts(moment)} - ERROR - local_eviction session={session} "
        f"reason=tenancy_lease_expired endpoint=http://localhost:8080 "
        f"session_id={session}"
    )


def _write_log(directory: Path, name: str, lines: list[str], *, gzipped: bool = False) -> Path:
    path = directory / name
    payload = "\n".join(lines) + "\n"
    if gzipped:
        with gzip.open(path, "wt") as handle:
            handle.write(payload)
    else:
        path.write_text(payload)
    return path


def _attribute(directory: Path, **kwargs) -> dict:
    return attribution.attribute(
        str(directory),
        kwargs.pop("window_start", WINDOW_START),
        kwargs.pop("window_end", WINDOW_END),
        include_rows=kwargs.pop("include_rows", True),
        **kwargs,
    )


def _rows(summary: dict) -> list[dict]:
    return summary["rows"]


# ---------------------------------------------------------------------------
# Own-record shapes
# ---------------------------------------------------------------------------

def test_own_idle_lease_is_attributable(tmp_path: Path) -> None:
    """Parent shape: own local turn finished, lease retained, one other idle.

    The requester's own idle lease is the reason ``cur_local`` reads 2/2 with
    nothing generating; excluding it leaves 1 < 2 and would admit.
    """
    moment = WINDOW_START + timedelta(minutes=5)
    requester = "sess-requester"
    other = "sess-other"
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(moment - timedelta(seconds=20), requester),
            _stream_started_local(moment - timedelta(seconds=20), requester),
            _stream_finished_local(moment - timedelta(seconds=3), requester),
            _lease_renewed(moment - timedelta(seconds=2, milliseconds=-100), requester),
            _admit_local(moment - timedelta(seconds=10), other),
            _stream_finished_local(moment - timedelta(seconds=1), other),
            _lease_renewed(moment - timedelta(seconds=1), other),
            _fallback(moment, requester, turn="turn-own-lease"),
        ],
    )
    summary = _attribute(tmp_path)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_IDLE_LEASE
    assert row["others_present"] == 1
    assert row["attribution"] == attribution.OWN_NON_STREAMING_ATTRIBUTABLE
    assert summary["own_record_attributable"] == 1


def test_own_finished_unreleased_is_attributable(tmp_path: Path) -> None:
    """Parent race: terminal chunk has landed, teardown has not released.

    The record is still ``active`` with no streaming token, so pre-fix it
    counted as occupancy; the fix excludes it. This is the exact shape of the
    2026-11-18 incident (``Stream finished`` 145 ms before the next turn).
    """
    moment = WINDOW_START + timedelta(minutes=6)
    requester = "sess-unreleased"
    other = "sess-other"
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(moment - timedelta(seconds=30), requester),
            _stream_started_local(moment - timedelta(seconds=30), requester),
            _stream_finished_local(moment - timedelta(milliseconds=150), requester),
            # No lease_renewed yet: the record is finished-but-not-released.
            _admit_local(moment - timedelta(seconds=15), other),
            _stream_finished_local(moment - timedelta(seconds=2), other),
            _lease_renewed(moment - timedelta(seconds=2), other),
            _fallback(moment, requester, turn="turn-race"),
        ],
    )
    summary = _attribute(tmp_path)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_FINISHED_UNRELEASED
    assert row["attribution"] == attribution.OWN_NON_STREAMING_ATTRIBUTABLE


def test_own_streaming_is_gated_not_attributed(tmp_path: Path) -> None:
    """A genuinely concurrent same-session request keeps counting (AC2)."""
    moment = WINDOW_START + timedelta(minutes=7)
    requester = "sess-streaming"
    other = "sess-other"
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(moment - timedelta(seconds=20), requester),
            _stream_started_local(moment - timedelta(seconds=20), requester),
            _admit_local(moment - timedelta(seconds=10), other),
            _stream_finished_local(moment - timedelta(seconds=5), other),
            _lease_renewed(moment - timedelta(seconds=5), other),
            _fallback(moment, requester, turn="turn-concurrent"),
        ],
    )
    summary = _attribute(tmp_path)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_STREAMING
    assert row["attribution"] == attribution.OWN_STREAMING_GATED
    assert summary["own_record_attributable"] == 0


def test_own_non_streaming_but_others_fill_the_pool_is_contention(tmp_path: Path) -> None:
    """Excluding an idle own lease still leaves a full pool: genuine."""
    moment = WINDOW_START + timedelta(minutes=8)
    requester = "sess-requester"
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(moment - timedelta(seconds=20), requester),
            _stream_finished_local(moment - timedelta(seconds=3), requester),
            _lease_renewed(moment - timedelta(seconds=2), requester),
            _admit_local(moment - timedelta(seconds=15), "other-a"),
            _stream_finished_local(moment - timedelta(seconds=1), "other-a"),
            _lease_renewed(moment - timedelta(seconds=1), "other-a"),
            _admit_local(moment - timedelta(seconds=12), "other-b"),
            _stream_finished_local(moment - timedelta(seconds=1), "other-b"),
            _lease_renewed(moment - timedelta(seconds=1), "other-b"),
            _fallback(moment, requester),
        ],
    )
    summary = _attribute(tmp_path)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_IDLE_LEASE
    assert row["others_present"] == 2
    assert row["attribution"] == attribution.OTHER_SESSIONS_CONTENTION
    assert summary["own_record_attributable"] == 0
    assert summary["genuine_contention"] == 1


def test_own_absent_is_other_session_occupancy(tmp_path: Path) -> None:
    moment = WINDOW_START + timedelta(minutes=9)
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(moment - timedelta(seconds=15), "other-a"),
            _stream_finished_local(moment - timedelta(seconds=1), "other-a"),
            _lease_renewed(moment - timedelta(seconds=1), "other-a"),
            _admit_local(moment - timedelta(seconds=12), "other-b"),
            _stream_finished_local(moment - timedelta(seconds=1), "other-b"),
            _lease_renewed(moment - timedelta(seconds=1), "other-b"),
            _fallback(moment, "sess-absent"),
        ],
    )
    summary = _attribute(tmp_path)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_ABSENT_STATE
    assert row["attribution"] == attribution.OTHER_SESSIONS_CONTENTION


def test_release_clears_a_leaked_record(tmp_path: Path) -> None:
    """A release after a finish must not leave a ghost "present" record.

    Non-streaming local serves and prefill-aborted dispatches can lack a
    matching ``Stream finished``; the release is the authoritative teardown.
    """
    moment = WINDOW_START + timedelta(minutes=10)
    requester = "sess-released"
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(moment - timedelta(seconds=400), requester),
            _lease_released(moment - timedelta(seconds=399), requester),
            _fallback(moment, requester),
        ],
    )
    summary = _attribute(tmp_path)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_ABSENT_STATE
    assert row["attribution"] == attribution.UNATTRIBUTABLE


def test_finished_unreleased_is_bounded_by_teardown_grace(tmp_path: Path) -> None:
    """Without a teardown line the finished state must not leak forever."""
    moment = WINDOW_START + timedelta(minutes=11)
    requester = "sess-no-teardown"
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(moment - timedelta(seconds=400), requester),
            _stream_finished_local(moment - timedelta(seconds=399), requester),
            # No lease_renewed; 399 s exceeds the 60 s teardown grace.
            _fallback(moment, requester),
        ],
    )
    summary = _attribute(tmp_path, teardown_grace_seconds=60.0)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_ABSENT_STATE


# ---------------------------------------------------------------------------
# Window, warm-up and rotation
# ---------------------------------------------------------------------------

def test_warmup_establishes_state_from_before_the_window(tmp_path: Path) -> None:
    """A stream admitted before the window is still live at the first event."""
    in_window = WINDOW_START + timedelta(seconds=10)
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(WINDOW_START - timedelta(seconds=300), "sess-long"),
            _stream_started_local(WINDOW_START - timedelta(seconds=300), "sess-long"),
            _fallback(in_window, "sess-reader"),
        ],
    )
    summary = _attribute(tmp_path, warmup_seconds=3600.0)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_ABSENT_STATE
    assert row["others_present"] == 1


def test_events_before_the_window_are_not_recorded(tmp_path: Path) -> None:
    before = WINDOW_START - timedelta(seconds=5)
    inside = WINDOW_START + timedelta(seconds=5)
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _fallback(before, "sess-before"),
            _fallback(inside, "sess-inside"),
        ],
    )
    summary = _attribute(tmp_path)
    assert summary["events"] == 1
    assert _rows(summary)[0]["ts"] == inside.isoformat(sep=" ")


def test_gzip_rotation_and_duplicate_lines_are_counted_once(tmp_path: Path) -> None:
    moment = WINDOW_START + timedelta(minutes=12)
    line = _fallback(moment, "sess-dupe")
    _write_log(tmp_path, "proxy.log", [line])
    _write_log(tmp_path, "proxy.log.1.gz", [line], gzipped=True)
    summary = _attribute(tmp_path)
    assert summary["events"] == 1


def test_remote_stream_finished_does_not_clear_a_local_record(tmp_path: Path) -> None:
    """A remote terminal chunk is not a local dispatch's teardown."""
    moment = WINDOW_START + timedelta(minutes=13)
    requester = "sess-mixed"
    _write_log(
        tmp_path,
        "proxy.log",
        [
            _admit_local(moment - timedelta(seconds=20), requester),
            _stream_started_local(moment - timedelta(seconds=20), requester),
            _stream_finished_remote(moment - timedelta(seconds=5), requester),
            _fallback(moment, requester),
        ],
    )
    summary = _attribute(tmp_path)
    row = _rows(summary)[0]
    assert row["own_state"] == attribution.OWN_STREAMING


# ---------------------------------------------------------------------------
# Summary and CLI
# ---------------------------------------------------------------------------

def test_summary_reports_headline_counts_and_method(tmp_path: Path) -> None:
    moment = WINDOW_START + timedelta(minutes=14)
    _write_log(tmp_path, "proxy.log", [_fallback(moment, "sess-x")])
    summary = _attribute(tmp_path)
    assert summary["events"] == 1
    assert summary["own_record_attributable"] == 0
    assert summary["genuine_contention"] == 0
    assert summary["unattributable"] == 1
    assert summary["method"]["max_local"] == 2
    assert summary["method"]["lease_timeout_seconds"] == 30.0
    assert "local_concurrency_limit" in summary["method"]["filter"]


def test_main_json_is_machine_readable(tmp_path: Path, capsys) -> None:
    moment = WINDOW_START + timedelta(minutes=15)
    _write_log(tmp_path, "proxy.log", [_fallback(moment, "sess-cli")])
    code = attribution.main(
        [
            "--log-dir",
            str(tmp_path),
            "--start",
            WINDOW_START.strftime("%Y-%m-%d %H:%M:%S"),
            "--until",
            WINDOW_END.strftime("%Y-%m-%d %H:%M:%S"),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["events"] == 1
    assert payload["by_attribution"]["unattributable"] == 1


def test_main_rejects_an_inverted_window(tmp_path: Path) -> None:
    code = attribution.main(
        [
            "--log-dir",
            str(tmp_path),
            "--start",
            WINDOW_END.strftime("%Y-%m-%d %H:%M:%S"),
            "--until",
            WINDOW_START.strftime("%Y-%m-%d %H:%M:%S"),
            "--json",
        ]
    )
    assert code == 2
