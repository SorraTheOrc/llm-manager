"""Per-prompt-fingerprint exponential backoff (LP-0MW959G0C008U9AI).

Targets parent LP-0MW0KZU86008PD8J AC2 — "repeat requests with the same prompt
fingerprint are backed off on an exponential schedule (1 -> 5 -> 15 min cap);
the backoff resets on any successful stream for that session or on a changed
history".

These tests pin the contract of ``proxy.fingerprint_backoff``:

- TAC1: a fingerprint is a stable hash of the message array that ignores the
  non-deterministic fields (``timestamp``, ``request_id``, ``id``).
- TAC2: the ladder is 1 min -> 5 min -> 15 min, capped at 15 min.
- TAC3: a changed history clears the previous backoff for that session.
- TAC4: a successful stream clears the backoff and restarts the ladder.
- TAC5: the verdict is shared — the same bad history is one backoff entry for
  every session that sends it.
- TAC6: a re-sent history is not re-hashed on the hot path (per-session cache;
  only appended messages are hashed).

The module itself lands here: this child's deliverables list the module code
(or an inline equivalent) so that its tests can be green before I1
(LP-0MW95BCVV004LO0K) wires the mechanism into ``provider.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from proxy import fingerprint_backoff as fb

_MODULE_PATH = Path(__file__).resolve().parents[1] / "proxy" / "fingerprint_backoff.py"


@pytest.fixture(autouse=True)
def _reset_state():
    """Isolate the module's in-memory state between tests."""
    fb.reset()
    yield
    fb.reset()


def _history(*contents: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content} for content in contents]


# =====================================================================
# TAC1 — stable hash of the message array
# =====================================================================


def test_fingerprint_is_stable_for_the_same_history():
    history = [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "continue"},
    ]

    first = fb.compute_prompt_fingerprint(history)
    assert first == fb.compute_prompt_fingerprint(history)
    # A structural copy (fresh dicts) is the same history.
    assert first == fb.compute_prompt_fingerprint([dict(message) for message in history])
    # An empty history has a stable fingerprint too.
    assert fb.compute_prompt_fingerprint([]) == fb.compute_prompt_fingerprint([])


def test_fingerprint_ignores_non_deterministic_fields():
    base = [
        {
            "role": "user",
            "content": "hello",
            "timestamp": 1_700_000_000,
            "request_id": "req-1",
            "id": "msg-1",
        }
    ]
    resent = [
        {
            "role": "user",
            "content": "hello",
            "timestamp": 1_700_000_042,
            "request_id": "req-2",
            "id": "msg-2",
        }
    ]
    assert fb.compute_prompt_fingerprint(base) == fb.compute_prompt_fingerprint(resent)

    # Nested tool-call metadata is regenerated per send as well.
    tool_call = {"id": "call-1", "timestamp": 10, "function": {"name": "read", "arguments": "{}"}}
    tool_call_resent = {"id": "call-9", "timestamp": 99, "function": {"name": "read", "arguments": "{}"}}
    with_tool_call = [{"role": "assistant", "content": "", "tool_calls": [tool_call]}]
    with_tool_call_resent = [{"role": "assistant", "content": "", "tool_calls": [tool_call_resent]}]
    assert fb.compute_prompt_fingerprint(with_tool_call) == fb.compute_prompt_fingerprint(with_tool_call_resent)


def test_fingerprint_changes_with_content_role_and_order():
    distinct = [
        fb.compute_prompt_fingerprint(_history("hi")),
        fb.compute_prompt_fingerprint(_history("hello")),
        fb.compute_prompt_fingerprint(_history("hi", "hi")),
        fb.compute_prompt_fingerprint(_history("a", "b")),
        fb.compute_prompt_fingerprint(_history("b", "a")),
        fb.compute_prompt_fingerprint([{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]),
    ]
    assert len(set(distinct)) == len(distinct), "each distinct history must get its own fingerprint"


def test_fingerprint_is_stable_across_processes():
    """A ``hash()``/``repr()``-based implementation would fail this."""
    payload = [
        {"role": "user", "content": "hi", "timestamp": 1},
        {"role": "assistant", "content": "hello"},
    ]
    script = textwrap.dedent(
        f"""
        import importlib.util

        spec = importlib.util.spec_from_file_location("fingerprint_backoff_under_test", {str(_MODULE_PATH)!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(module.compute_prompt_fingerprint({payload!r}))
        """
    )

    digests = set()
    for seed in ("1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.add(completed.stdout.strip())

    assert len(digests) == 1, f"fingerprint must not depend on the hash seed: {digests}"
    assert digests.pop() == fb.compute_prompt_fingerprint(payload)


# =====================================================================
# TAC2 — 1 min -> 5 min -> 15 min cap
# =====================================================================


def test_backoff_ladder_is_one_five_fifteen_minutes_with_a_cap():
    fingerprint = fb.compute_prompt_fingerprint(_history("poisoned history"))
    delays = []

    for attempt in range(1, 5):
        now = 1_000.0 * attempt
        entry = fb.record_backoff(fingerprint, "sess-wedged", now=now)
        delays.append(entry["retry_after"])
        assert entry["resend_count"] == attempt
        assert entry["fingerprint"] == fingerprint
        assert entry["session_id"] == "sess-wedged"
        assert fb.check_backoff_expired(fingerprint, now=now) is False
        assert fb.remaining_backoff_seconds(fingerprint, now=now) == entry["retry_after"]

    assert delays == [60.0, 300.0, 900.0, 900.0], "the ladder must be 1 -> 5 -> 15 min, capped at 15"


def test_backoff_window_expires_without_restarting_the_ladder():
    fingerprint = fb.compute_prompt_fingerprint(_history("bad"))
    fb.record_backoff(fingerprint, "sess", now=100.0)

    assert fb.check_backoff_expired(fingerprint, now=159.9) is False
    assert fb.remaining_backoff_seconds(fingerprint, now=159.9) == pytest.approx(0.1)
    assert fb.check_backoff_expired(fingerprint, now=160.0) is True
    assert fb.remaining_backoff_seconds(fingerprint, now=160.0) == 0.0

    # The expired entry survives so the next rejection escalates instead of
    # restarting at 1 min (the terminal breaker counts these resends).
    assert fb.get_backoff_entry(fingerprint)["resend_count"] == 1
    assert fb.record_backoff(fingerprint, "sess", now=161.0)["retry_after"] == 300.0


def test_explicit_retry_after_overrides_the_ladder():
    fingerprint = fb.compute_prompt_fingerprint(_history("bad"))
    entry = fb.record_backoff(fingerprint, "sess", retry_after=7.0, now=50.0)

    assert entry["retry_after"] == 7.0
    assert fb.check_backoff_expired(fingerprint, now=56.9) is False
    assert fb.check_backoff_expired(fingerprint, now=57.0) is True


# =====================================================================
# TAC3 — a changed history clears the previous backoff for that session
# =====================================================================


def test_changed_history_clears_the_previous_backoff_for_the_session():
    session = "sess-compacted"
    old_fingerprint = fb.fingerprint_for_session(session, _history("old compacted history"))
    fb.record_backoff(old_fingerprint, session, now=100.0)
    assert fb.check_backoff_expired(old_fingerprint, now=101.0) is False

    new_fingerprint = fb.fingerprint_for_session(session, _history("rewritten after compaction"))

    assert new_fingerprint != old_fingerprint
    assert fb.get_backoff_entry(old_fingerprint) is None
    assert fb.check_backoff_expired(old_fingerprint, now=101.0) is True
    # The rewritten history starts a fresh ladder rather than inheriting one.
    assert fb.record_backoff(new_fingerprint, session, now=102.0)["resend_count"] == 1


def test_repeating_the_same_history_keeps_the_backoff_ladder():
    session = "sess-still-stuck"
    fingerprint = fb.fingerprint_for_session(session, _history("same", "history"))
    fb.record_backoff(fingerprint, session, now=10.0)

    assert fb.fingerprint_for_session(session, _history("same", "history")) == fingerprint
    assert fb.get_backoff_entry(fingerprint)["resend_count"] == 1

    entry = fb.record_backoff(fingerprint, session, now=11.0)
    assert entry["resend_count"] == 2
    assert entry["retry_after"] == 300.0


def test_note_prompt_fingerprint_reports_the_cleared_history():
    old = fb.compute_prompt_fingerprint(_history("old"))
    new = fb.compute_prompt_fingerprint(_history("new"))
    fb.record_backoff(old, "sess")

    assert fb.note_prompt_fingerprint("sess", old) is None, "unchanged history clears nothing"
    assert fb.note_prompt_fingerprint("sess", new) == old
    assert fb.get_backoff_entry(old) is None


# =====================================================================
# TAC4 — a successful stream clears the backoff
# =====================================================================


def test_successful_stream_clears_the_backoff_and_restarts_the_ladder():
    fingerprint = fb.compute_prompt_fingerprint(_history("bad"))
    fb.record_backoff(fingerprint, "sess", now=1.0)
    fb.record_backoff(fingerprint, "sess", now=2.0)
    assert fb.get_backoff_entry(fingerprint)["resend_count"] == 2

    assert fb.clear_backoff(fingerprint) is True
    assert fb.get_backoff_entry(fingerprint) is None
    assert fb.check_backoff_expired(fingerprint, now=3.0) is True

    entry = fb.record_backoff(fingerprint, "sess", now=3.0)
    assert entry["resend_count"] == 1, "a success must reset the ladder"
    assert entry["retry_after"] == 60.0
    assert fb.clear_backoff(fingerprint) is True
    assert fb.clear_backoff(fingerprint) is False, "clearing again must be a no-op"


# =====================================================================
# TAC5 — one verdict shared across sessions
# =====================================================================


def test_same_history_shares_one_backoff_verdict_across_sessions():
    history = _history("identical", "bad history")
    fingerprint_a = fb.fingerprint_for_session("sess-a", history)
    fingerprint_b = fb.fingerprint_for_session("sess-b", history)
    assert fingerprint_a == fingerprint_b, "the same history must fingerprint identically for any session"

    first = fb.record_backoff(fingerprint_a, "sess-a", now=100.0)
    second = fb.record_backoff(fingerprint_b, "sess-b", now=110.0)

    assert first["resend_count"] == 1
    assert second["resend_count"] == 2, "a second session on the same history continues the shared ladder"
    assert second["retry_after"] == 300.0
    assert second["session_id"] == "sess-b"
    # Session A sees the entry session B just escalated (one shared window).
    assert fb.get_backoff_entry(fingerprint_a) == fb.get_backoff_entry(fingerprint_b)
    assert fb.get_backoff_entry(fingerprint_a)["resend_count"] == 2
    assert fb.check_backoff_expired(fingerprint_a, now=120.0) is False


def test_clearing_from_one_session_frees_every_session_on_the_same_history():
    fingerprint = fb.compute_prompt_fingerprint(_history("shared bad history"))
    fb.record_backoff(fingerprint, "sess-a", now=1.0)
    fb.record_backoff(fingerprint, "sess-b", now=2.0)
    assert fb.check_backoff_expired(fingerprint, now=3.0) is False

    fb.clear_backoff(fingerprint)

    assert fb.check_backoff_expired(fingerprint, now=3.0) is True
    assert fb.record_backoff(fingerprint, "sess-c", now=4.0)["resend_count"] == 1


# =====================================================================
# TAC6 — no whole-history re-hashing on the hot path
# =====================================================================


def test_resending_the_same_history_does_not_walk_the_hash_chain():
    session = "sess-hot-path"
    history = _history(*[f"turn-{index}" for index in range(50)])
    first = fb.fingerprint_for_session(session, history)

    with patch.object(fb, "_extend_digest", wraps=fb._extend_digest) as extend:
        again = fb.fingerprint_for_session(session, list(history))

    assert again == first
    assert extend.call_count == 0, "a re-sent history must reuse the cached chain digest"


def test_appending_a_turn_hashes_only_the_new_messages():
    session = "sess-append"
    history = _history(*[f"turn-{index}" for index in range(20)])
    base = fb.fingerprint_for_session(session, history)
    grown = history + _history("next turn")

    with patch.object(fb, "_extend_digest", wraps=fb._extend_digest) as extend:
        incremental = fb.fingerprint_for_session(session, grown)

    assert extend.call_count == 1, "only the appended message should be hashed"
    assert incremental != base
    assert incremental == fb.compute_prompt_fingerprint(grown)


def test_rewritten_history_with_the_same_length_is_refingerprinted():
    session = "sess-rewrite"
    original = _history(*[f"turn-{index}" for index in range(10)])
    fb.fingerprint_for_session(session, original)

    rewritten = _history(*[f"compacted-{index}" for index in range(10)])

    assert fb.fingerprint_for_session(session, rewritten) == fb.compute_prompt_fingerprint(rewritten)


def test_grown_history_with_a_rewritten_prefix_is_refingerprinted():
    session = "sess-grown-rewrite"
    original = _history("head", "a", "b")
    fb.fingerprint_for_session(session, original)

    # Same head and longer, but the previously hashed prefix was rewritten:
    # reusing the cached prefix would fingerprint a different history.
    rewritten = _history("head", "x", "y", "z")

    assert fb.fingerprint_for_session(session, rewritten) == fb.compute_prompt_fingerprint(rewritten)


def test_evicted_session_cache_still_returns_the_correct_fingerprint(monkeypatch):
    monkeypatch.setattr(fb, "MAX_SESSION_CACHE_ENTRIES", 1)
    expected = fb.compute_prompt_fingerprint(_history("x"))

    fb.fingerprint_for_session("sess-a", _history("x"))
    fb.fingerprint_for_session("sess-b", _history("y"))  # evicts sess-a

    assert fb.fingerprint_for_session("sess-a", _history("x")) == expected


def test_backoff_state_is_bounded(monkeypatch):
    monkeypatch.setattr(fb, "MAX_BACKOFF_ENTRIES", 3)
    fingerprints = [fb.compute_prompt_fingerprint(_history(f"bad-{index}")) for index in range(6)]

    for index, fingerprint in enumerate(fingerprints):
        fb.record_backoff(fingerprint, "sess", now=1_000.0 + index)

    retained = [fingerprint for fingerprint in fingerprints if fb.get_backoff_entry(fingerprint) is not None]
    assert fingerprints[-1] in retained, "the newest verdict must survive pruning"
    assert len(retained) <= 3


def test_module_state_is_json_serialisable():
    """Entries are handed to logs/metrics, so they must serialise cleanly."""
    fingerprint = fb.compute_prompt_fingerprint(_history("bad"))
    entry = fb.record_backoff(fingerprint, "sess", now=5.0)

    assert json.loads(json.dumps(entry)) == entry


# =====================================================================
# LP-0MW959H72003PF67 (T3) — terminal circuit breaker (session_wedged)
#
# Parent AC3: after N consecutive identical resends (configurable via
# ``deterministic_client_error_escalation_threshold``, default 5) the client is
# told the session is wedged — ``{error: "session_wedged", cause:
# "repeated_deterministic_400", hint: "compaction may help"}`` — instead of
# being throttled forever. The counter resets on a successful stream and on a
# changed history, so the breaker can only fire on an unbroken run of
# rejections of the *same* history.
#
# These tests pin ``proxy.fingerprint_backoff``'s half of that contract: the
# verdict and the configurable threshold. Wiring the verdict into the typed-400
# response (HTTP status, ``Retry-After`` header, metric) belongs to
# LP-0MW95BE7X00574OL.
# =====================================================================

_ESCALATION_KEY = fb.ESCALATION_THRESHOLD_CONFIG_KEY


def _reject(fingerprint: str, times: int, session: str = "sess-wedged", start: float = 1_000.0):
    """Record ``times`` consecutive rejections of ``fingerprint``."""
    entry = None
    for index in range(times):
        entry = fb.record_backoff(fingerprint, session, now=start + index)
    return entry


def test_no_terminal_verdict_without_any_recorded_rejection():
    fingerprint = fb.compute_prompt_fingerprint(_history("untried history"))

    assert fb.get_resend_count(fingerprint) == 0
    assert fb.session_wedged_error(fingerprint) is None


def test_five_consecutive_identical_resends_escalate_to_session_wedged():
    """TAC1: the 5th identical resend is terminal, the 4th is not."""
    session = "sess-wedged"
    fingerprint = fb.fingerprint_for_session(session, _history("poisoned", "history"))

    for attempt in range(1, 5):
        _reject(fingerprint, 1, session, start=1_000.0 + attempt)
        assert fb.session_wedged_error(fingerprint) is None, (
            f"resend {attempt} must still be a typed 400, not a terminal verdict"
        )

    _reject(fingerprint, 1, session, start=1_005.0)
    payload = fb.session_wedged_error(fingerprint)

    assert payload is not None
    assert payload["error"] == "session_wedged"
    assert payload["cause"] == "repeated_deterministic_400"
    assert payload["hint"] == "compaction may help"
    # Extra fields feed the response header and the log line; the documented
    # contract above is what clients match on.
    assert payload["resend_count"] == 5
    assert payload["fingerprint"] == fingerprint
    assert payload["retry_after"] >= 1


def test_escalation_threshold_defaults_to_five():
    """TAC2: the shipped default is 5 (parent AC3's N)."""
    assert fb.DETERMINISTIC_CLIENT_ERROR_ESCALATION_THRESHOLD == 5
    assert _ESCALATION_KEY == "deterministic_client_error_escalation_threshold"
    assert fb.escalation_threshold() == 5
    assert fb.escalation_threshold({}) == 5


def test_escalation_threshold_is_configurable_from_proxy_config():
    """TAC2: the knob is honoured at the top level and inside ``server``."""
    assert fb.escalation_threshold({_ESCALATION_KEY: 3}) == 3
    assert fb.escalation_threshold({_ESCALATION_KEY: "7"}) == 7
    assert fb.escalation_threshold({"server": {_ESCALATION_KEY: 2}}) == 2
    # A top-level value wins over the section, matching the other knobs.
    assert fb.escalation_threshold({_ESCALATION_KEY: 4, "server": {_ESCALATION_KEY: 9}}) == 4
    # An explicit threshold wins over the config (a caller that already resolved it).
    fingerprint = fb.compute_prompt_fingerprint(_history("bad"))
    _reject(fingerprint, 2)
    assert fb.session_wedged_error(fingerprint, config={_ESCALATION_KEY: 2}) is not None
    assert fb.session_wedged_error(fingerprint, config={_ESCALATION_KEY: 9}) is None
    assert fb.session_wedged_error(fingerprint, threshold=2, config={_ESCALATION_KEY: 9}) is not None


def test_invalid_escalation_threshold_falls_back_to_the_default():
    """A mistyped knob must neither disable the breaker nor wedge on resend 1."""
    for invalid in ("many", 0, -1, True, [], {}, 0.4):
        assert fb.escalation_threshold({_ESCALATION_KEY: invalid}) == 5, (
            f"{invalid!r} must fall back to the default threshold"
        )


def test_configured_threshold_escalates_at_its_own_count():
    """TAC2: lowering the knob trips the breaker earlier."""
    session = "sess-tuned"
    fingerprint = fb.fingerprint_for_session(session, _history("poisoned"))
    config = {_ESCALATION_KEY: 3}

    _reject(fingerprint, 2, session)
    assert fb.session_wedged_error(fingerprint, config=config) is None

    _reject(fingerprint, 1, session, start=1_010.0)
    payload = fb.session_wedged_error(fingerprint, config=config)

    assert payload is not None
    assert payload["resend_count"] == 3, "the default of 5 would still be throttling here"


def test_successful_stream_between_resends_resets_the_terminal_counter():
    """TAC3: a success clears the count, so the breaker cannot fire later."""
    session = "sess-recovers"
    fingerprint = fb.fingerprint_for_session(session, _history("bad", "history"))
    _reject(fingerprint, 4, session)

    assert fb.clear_backoff(fingerprint) is True  # the successful stream
    assert fb.get_resend_count(fingerprint) == 0

    _reject(fingerprint, 1, session, start=1_010.0)

    assert fb.get_resend_count(fingerprint) == 1
    assert fb.session_wedged_error(fingerprint) is None, (
        "one rejection after a success is a fresh ladder, not a wedged session"
    )


def test_changed_fingerprint_between_resends_resets_the_terminal_counter():
    """TAC4: a rewritten (compacted) history is a fresh count."""
    session = "sess-compacts"
    old_fingerprint = fb.fingerprint_for_session(session, _history("old", "compacted"))
    _reject(old_fingerprint, 4, session)

    new_fingerprint = fb.fingerprint_for_session(session, _history("rebuilt", "history"))
    assert new_fingerprint != old_fingerprint
    assert fb.get_resend_count(old_fingerprint) == 0, "the previous verdict is dropped"
    _reject(new_fingerprint, 1, session, start=1_010.0)

    assert fb.get_resend_count(new_fingerprint) == 1
    assert fb.session_wedged_error(old_fingerprint) is None
    assert fb.session_wedged_error(new_fingerprint) is None


def test_backoff_expired_resends_still_accumulate_toward_escalation():
    """The breaker counts resends, not wall-clock: expiry is not a reset."""
    fingerprint = fb.compute_prompt_fingerprint(_history("bad"))

    for round_index in range(5):
        now = 1_000.0 + round_index * 3_600.0  # each window long expired
        fb.record_backoff(fingerprint, "sess", now=now)

    assert fb.check_backoff_expired(fingerprint, now=100_000.0) is True
    payload = fb.session_wedged_error(fingerprint, now=100_000.0)

    assert payload is not None
    assert payload["resend_count"] == 5
    assert payload["retry_after"] >= int(fb.SESSION_WEDGED_MIN_RETRY_AFTER_SECONDS), (
        "an expired window must not advertise an immediate retry"
    )


def test_terminal_verdict_is_shared_by_every_session_on_the_same_history():
    """The verdict belongs to the history, so a second session sees it too."""
    history = _history("identical", "bad history")
    fingerprint_a = fb.fingerprint_for_session("sess-a", history)
    fingerprint_b = fb.fingerprint_for_session("sess-b", history)
    _reject(fingerprint_a, 4, "sess-a")

    assert fb.session_wedged_error(fingerprint_b) is None

    _reject(fingerprint_b, 1, "sess-b", start=1_010.0)

    payload = fb.session_wedged_error(fingerprint_a)
    assert payload is not None
    assert payload["resend_count"] == 5, "the ladder is shared across sessions"


def test_terminal_payload_is_json_serialisable():
    """The payload is returned to the client, so it must serialise cleanly."""
    fingerprint = fb.compute_prompt_fingerprint(_history("bad"))
    _reject(fingerprint, 5)

    payload = fb.session_wedged_error(fingerprint)

    assert payload is not None
    assert json.loads(json.dumps(payload)) == payload
