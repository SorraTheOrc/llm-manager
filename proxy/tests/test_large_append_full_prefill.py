"""Large-append rule: force full-prefill pricing (LP-0MW81DLWY003XOZ2).

The routing ``cached_ratio`` is measured on the **previous** local response and
credited forward from the session's recorded prompt prefix
(``expected_cached_ratio``).  That is a good estimate for a conversation that
grows by a small turn, and a bad one when a single message appends a large
block of new content: on 2026-11-03 08:41:54 a **51,945-byte** tool-role
message was appended, the price came out as ``cached_ratio=0.99
new_tokens=173``, and llama-server then prefilled 13,002 tokens (53.2 s to
first byte, ``measured_cache_abort deficit_ratio=0.659``).

The rule under test: when the content **appended after the recorded prefix**
exceeds ``server.local_large_append_bytes_threshold`` bytes, the pre-dispatch
price forces a full prefill (``cached_ratio = 0``) instead of trusting the
in-memory residency ratio, so the cold threshold is evaluated against the real
prefill work.  ``0`` disables the rule (the sentinel) and restores the previous
price byte-for-byte.

AC coverage:

- AC1 (rule): a large append forces ``cached_ratio = 0`` and therefore falls
  under the cold threshold (``large_context_bypass``);
- AC2 (the incident): the 51,945-byte tool-append case is priced as a full
  prefill;
- AC3 (configurability): the threshold is configurable (flat and nested),
  documented (``proxy/config.yaml`` + ``proxy/docs/routing.md``) and disabled
  by the ``0`` sentinel without changing any other behaviour.
"""

from __future__ import annotations

import logging

import pytest
from proxy.provider import (
    DEFAULT_LARGE_APPEND_BYTES_THRESHOLD,
    LARGE_APPEND_BYTES_THRESHOLD_KEY,
    _cached_ratio_source,
    _last_cached_ratio,
    _last_prompt_prefix,
    _should_skip_local,
    expected_cached_ratio,
    large_append_bytes_threshold,
    record_prompt_prefix,
    set_cached_ratio,
)
from proxy.slot_cache_ratio import appended_content_bytes, message_fingerprints

MODEL = "Qwen3"
SESSION = "sess-large-append"
COLD_THRESHOLD = 38_000
WARM_THRESHOLD = 120_000

#: The append measured in the 2026-11-03 incident (a full SKILL.md pasted into
#: history as one tool-role message).
INCIDENT_APPEND_BYTES = 51_945

#: A prompt whose total estimate is above the cold threshold (38 000), so the
#: cold threshold governs the admission decision (``_should_skip_local``
#: short-circuits below it).
LARGE_PROMPT_TOKENS = 43_000


@pytest.fixture(autouse=True)
def _clean_state():
    """Isolate the process-global ratio/prefix state between tests."""
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    _last_prompt_prefix.clear()
    yield
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    _last_prompt_prefix.clear()


def _base_messages() -> list[dict]:
    """The already-dispatch prefix of the session (recorded last turn)."""
    return [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "read the skill file"},
    ]


def _append_message(total_bytes: int) -> dict:
    """A tool-role message whose ``role + content`` occupies *total_bytes*."""
    overhead = len("tool") + 1  # role + the fingerprint separator
    assert total_bytes > overhead
    return {"role": "tool", "content": "x" * (total_bytes - overhead)}


def _prime_warm_session(*, prev_prompt_tokens: int = 6_622) -> list[dict]:
    """Record a warm measured ratio plus the prompt prefix it belongs to."""
    base = _base_messages()
    set_cached_ratio(MODEL, SESSION, 0.99)
    record_prompt_prefix(MODEL, SESSION, prev_prompt_tokens, base)
    return base


def _price(messages, *, server_config=None, estimated_tokens=LARGE_PROMPT_TOKENS):
    return expected_cached_ratio(
        MODEL,
        SESSION,
        estimated_tokens,
        messages,
        server_config=server_config,
    )


def _route(messages, *, server_config=None, estimated_tokens=LARGE_PROMPT_TOKENS):
    """Run the pre-dispatch price path exactly as the router calls it.

    ``resident_prefix_min_ratio=0`` disables the residency ladder, which is the
    configuration in which the predicted-ratio price (check 5) decides — the
    path this rule protects.  (With the ladder enabled an *unmeasured* prompt
    above the cold threshold is already denied; the rule keeps the predicted
    path honest where it still decides.)
    """
    return _should_skip_local(
        MODEL,
        SESSION,
        {"messages": messages},
        COLD_THRESHOLD,
        estimated_tokens=estimated_tokens,
        warm_cache_threshold=WARM_THRESHOLD,
        resident_prefix_min_ratio=0.0,
        server_config=server_config,
    )


# ---------------------------------------------------------------------------
# AC1/AC2 — a large append forces a full-prefill price
# ---------------------------------------------------------------------------


class TestLargeAppendForcesFullPrefill:
    def test_incident_append_is_priced_as_a_full_prefill(self):
        """AC2: the 51,945-byte tool append prices ``cached_ratio = 0``."""
        base = _prime_warm_session()
        assert _price(base) == pytest.approx(0.99)  # no append: ratio trusted

        incoming = base + [_append_message(INCIDENT_APPEND_BYTES)]
        assert _price(incoming) == 0.0

    def test_large_append_falls_under_the_cold_threshold(self):
        """AC1: full-prefill pricing sends the large prompt remote."""
        base = _prime_warm_session()
        incoming = base + [_append_message(INCIDENT_APPEND_BYTES)]

        # With the rule disabled the same request is admitted on the warm
        # ratio (today's behaviour) — this is the miss the rule closes.
        assert _route(incoming, server_config={LARGE_APPEND_BYTES_THRESHOLD_KEY: 0}) == (
            False,
            None,
        )

        skip, reason = _route(incoming)
        assert skip is True
        assert reason == "large_context_bypass"

    def test_small_append_keeps_the_measured_ratio(self):
        """The rule only fires for a large append; a normal turn is unchanged."""
        base = _prime_warm_session()
        incoming = base + [_append_message(2_048)]

        assert _price(incoming) == pytest.approx(0.99)
        assert _route(incoming) == (False, None)

    def test_resident_prefix_messages_do_not_count_towards_the_append(self):
        """Only content *after* the recorded prefix is the append."""
        long_prefix = [
            {"role": "system", "content": "you are a coding agent"},
            {"role": "tool", "content": "p" * INCIDENT_APPEND_BYTES},
        ]
        set_cached_ratio(MODEL, SESSION, 0.99)
        record_prompt_prefix(MODEL, SESSION, 40_000, long_prefix)

        # A big *resident* prefix plus a small append: the prefix is reusable,
        # so the measured ratio still stands.
        incoming = long_prefix + [{"role": "user", "content": "next"}]
        assert _price(incoming) == pytest.approx(0.99)

        # The same session with a big append beyond the prefix is forced cold.
        forced = long_prefix + [_append_message(INCIDENT_APPEND_BYTES)]
        assert _price(forced) == 0.0

    def test_append_exactly_at_the_threshold_is_not_forced(self):
        """The rule fires when the append *exceeds* the threshold."""
        base = _prime_warm_session()
        threshold = 4_096
        at_threshold = base + [_append_message(threshold)]
        assert _price(at_threshold, server_config={LARGE_APPEND_BYTES_THRESHOLD_KEY: threshold}) == pytest.approx(0.99)

        over_threshold = base + [_append_message(threshold + 1)]
        assert (
            _price(
                over_threshold,
                server_config={LARGE_APPEND_BYTES_THRESHOLD_KEY: threshold},
            )
            == 0.0
        )

    def test_large_append_price_is_logged(self, caplog):
        """The forced price is attributable from the log alone."""
        base = _prime_warm_session()
        incoming = base + [_append_message(INCIDENT_APPEND_BYTES)]

        with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
            assert _price(incoming) == 0.0

        records = [r.getMessage() for r in caplog.records if "routing_large_append_full_prefill" in r.getMessage()]
        assert records, "forced full-prefill price must be logged"
        assert f"appended_bytes={INCIDENT_APPEND_BYTES}" in records[0]
        assert f"threshold={DEFAULT_LARGE_APPEND_BYTES_THRESHOLD}" in records[0]

    def test_no_recorded_prefix_keeps_the_measured_ratio(self):
        """With no recorded prefix the append boundary is unknown: unchanged.

        A proxy restart restores a ratio from the slot-snapshot sidecar without
        a recorded prompt prefix.  The rule must not treat the *whole* prompt
        as an append there, or a restored warm session would be re-priced cold
        for every large prompt (a behaviour change this rule does not make).
        """
        set_cached_ratio(MODEL, SESSION, 0.9, source="restored")
        incoming = _base_messages() + [_append_message(INCIDENT_APPEND_BYTES)]
        assert _price(incoming) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# AC3 — configurable threshold, documented, with a disable sentinel
# ---------------------------------------------------------------------------


class TestThresholdConfiguration:
    def test_threshold_is_configurable(self):
        base = _prime_warm_session()
        incoming = base + [_append_message(8_192)]

        # Default threshold (32 KiB): 8 KiB is not a large append.
        assert _price(incoming) == pytest.approx(0.99)

        # A lower configured threshold makes the same append large.
        assert _price(incoming, server_config={LARGE_APPEND_BYTES_THRESHOLD_KEY: 4_096}) == 0.0

    def test_nested_server_section_is_read(self):
        base = _prime_warm_session()
        incoming = base + [_append_message(INCIDENT_APPEND_BYTES)]
        server_config = {"server": {LARGE_APPEND_BYTES_THRESHOLD_KEY: 4_096}}
        assert _price(incoming, server_config=server_config) == 0.0

    @pytest.mark.parametrize("sentinel", [0, -1])
    def test_sentinel_disables_the_rule(self, sentinel):
        base = _prime_warm_session()
        incoming = base + [_append_message(INCIDENT_APPEND_BYTES)]
        server_config = {LARGE_APPEND_BYTES_THRESHOLD_KEY: sentinel}

        assert large_append_bytes_threshold(server_config) == 0
        assert _price(incoming, server_config=server_config) == pytest.approx(0.99)
        assert _route(incoming, server_config=server_config) == (False, None)

    def test_missing_config_keeps_the_shipped_default(self):
        assert large_append_bytes_threshold(None) == DEFAULT_LARGE_APPEND_BYTES_THRESHOLD
        assert large_append_bytes_threshold({}) == DEFAULT_LARGE_APPEND_BYTES_THRESHOLD

    @pytest.mark.parametrize("malformed", ["not-a-number", None, [], {}])
    def test_malformed_value_keeps_the_shipped_default(self, malformed):
        """A typo must not silently disable the rule (fail to the default)."""
        assert (
            large_append_bytes_threshold({LARGE_APPEND_BYTES_THRESHOLD_KEY: malformed})
            == DEFAULT_LARGE_APPEND_BYTES_THRESHOLD
        )

    def test_default_threshold_catches_the_incident_append(self):
        assert DEFAULT_LARGE_APPEND_BYTES_THRESHOLD < INCIDENT_APPEND_BYTES


# ---------------------------------------------------------------------------
# The size measurement helper
# ---------------------------------------------------------------------------


class TestAppendedContentBytes:
    def test_counts_only_messages_after_the_recorded_prefix(self):
        prefix = _base_messages()
        stored = message_fingerprints(prefix)
        assert appended_content_bytes(prefix, stored) == 0

        appended = prefix + [_append_message(INCIDENT_APPEND_BYTES)]
        assert appended_content_bytes(appended, stored) == INCIDENT_APPEND_BYTES

    def test_no_prefix_means_the_whole_prompt_is_appended(self):
        messages = _base_messages() + [_append_message(1_024)]
        assert appended_content_bytes(messages, []) == sum(len(m["role"]) + 1 + len(m["content"]) for m in messages)

    def test_truncated_history_has_no_append(self):
        stored = message_fingerprints(_base_messages() + [_append_message(1_024)])
        assert appended_content_bytes(_base_messages(), stored) == 0

    @pytest.mark.parametrize("messages", [None, "not-a-list", 7])
    def test_non_list_input_is_zero(self, messages):
        assert appended_content_bytes(messages, []) == 0
