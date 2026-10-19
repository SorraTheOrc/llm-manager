"""Tests for honouring upstream ``x-ratelimit-*`` headers (LP-0MVBHDNLB007ALLJ).

MunitoAI answers HTTP 429 with custom ``x-ratelimit-*`` headers and no
standard ``Retry-After``. These tests cover:

- duration parsing (``52s``, ``1m``, ``1h30m``, ``250ms``, bare seconds);
- reset aggregation, including preferring exhausted dimensions;
- the ``x-ratelimit-over-limit`` flag;
- opt-in gating (per provider entry vs global ``server`` config);
- the header-derived cooldown taking precedence over exponential backoff;
- fallback to backoff when headers are absent/unparseable;
- persistence of the header-derived cooldown in ``provider-state.json``;
- backward compatibility when the feature is not enabled.
"""

import json
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response

# Reset headers mirroring a live MunitoAI free-tier 429 (LP-0MVBHDNLB007ALLJ).
_MUNITO_429_HEADERS = {
    "x-ratelimit-limit-requests": "10",
    "x-ratelimit-remaining-requests": "0",
    "x-ratelimit-reset-requests": "52s",
    "x-ratelimit-over-limit": "yes",
}


@pytest.fixture(autouse=True)
def _isolate_provider_state(tmp_path, monkeypatch):
    """Isolate cooldown maps and the state-file path for every test here."""
    monkeypatch.setenv(
        provider._PROVIDER_STATE_FILE_ENV,
        str(tmp_path / "provider-state.json"),
    )
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()


def _enabled_config(**extra):
    return {"respect_ratelimit_headers": True, **extra}


def _munito_entry(**extra):
    return {"name": "munitoai-free", "respect_ratelimit_headers": True, **extra}


class TestParseRatelimitDuration:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("52s", 52.0),
            ("0s", 0.0),
            ("1m", 60.0),
            ("1h", 3600.0),
            ("1h30m", 5400.0),
            ("1h30m52s", 5452.0),
            ("250ms", 0.25),
            ("1d", 86400.0),
            ("90", 90.0),
            ("0.5s", 0.5),
            ("52 s", 52.0),
            ("1H30M", 5400.0),
        ],
    )
    def test_parses_supported_forms(self, raw, expected):
        assert provider._parse_ratelimit_duration(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "   ", "abc", "1x2s", "s", "-5s", "1h x"])
    def test_rejects_unparseable_values(self, raw):
        assert provider._parse_ratelimit_duration(raw) is None


class TestRatelimitOverLimit:
    @pytest.mark.parametrize("value", ["yes", "YES", "true", "1", "on"])
    def test_truthy(self, value):
        assert provider._ratelimit_over_limit({"x-ratelimit-over-limit": value})

    @pytest.mark.parametrize("value", ["no", "false", "0", "off", "maybe"])
    def test_falsy(self, value):
        assert not provider._ratelimit_over_limit({"x-ratelimit-over-limit": value})

    def test_absent_header_is_false(self):
        assert not provider._ratelimit_over_limit({})
        assert not provider._ratelimit_over_limit(None)


class TestParseRatelimitResetSeconds:
    def test_single_reset_header(self):
        assert provider._parse_ratelimit_reset_seconds(
            {"x-ratelimit-reset-requests": "52s"}
        ) == 52.0

    def test_no_reset_headers_returns_none(self):
        assert provider._parse_ratelimit_reset_seconds(
            {"x-ratelimit-over-limit": "yes"}
        ) is None
        assert provider._parse_ratelimit_reset_seconds(None) is None

    def test_unparseable_reset_returns_none(self):
        assert provider._parse_ratelimit_reset_seconds(
            {"x-ratelimit-reset-requests": "soon"}
        ) is None

    def test_exhausted_dimension_wins_over_unexhausted_longer_reset(self):
        # Requests exhausted after 52s; tokens still available with a long
        # reset that is irrelevant to the current 429.
        headers = {
            "x-ratelimit-reset-requests": "52s",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-tokens": "1h",
            "x-ratelimit-remaining-tokens": "5000",
        }
        assert provider._parse_ratelimit_reset_seconds(headers) == 52.0

    def test_longest_exhausted_reset_wins(self):
        headers = {
            "x-ratelimit-reset-requests": "52s",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-tokens": "3m",
            "x-ratelimit-remaining-tokens": "0",
        }
        assert provider._parse_ratelimit_reset_seconds(headers) == 180.0

    def test_without_remaining_headers_longest_reset_is_used(self):
        headers = {
            "x-ratelimit-reset-requests": "52s",
            "x-ratelimit-reset-input-tokens": "2m",
        }
        assert provider._parse_ratelimit_reset_seconds(headers) == 120.0


class TestRespectRatelimitGating:
    def test_default_off(self):
        assert not provider._get_respect_ratelimit_headers({}, {})
        assert not provider._get_respect_ratelimit_headers(None, None)

    def test_entry_flag_enables(self):
        assert provider._get_respect_ratelimit_headers(
            {"respect_ratelimit_headers": False},
            {"respect_ratelimit_headers": True},
        )

    def test_entry_false_overrides_global_true(self):
        assert not provider._get_respect_ratelimit_headers(
            {"server": {"respect_ratelimit_headers": True}},
            {"respect_ratelimit_headers": False},
        )

    def test_global_server_config_enables(self):
        assert provider._get_respect_ratelimit_headers(
            {"server": {"respect_ratelimit_headers": True}}, {"name": "p"}
        )

    def test_flat_config_enables(self):
        assert provider._get_respect_ratelimit_headers(
            {"respect_ratelimit_headers": True}, {"name": "p"}
        )

    def test_string_forms_are_coerced(self):
        assert provider._get_respect_ratelimit_headers(
            {"server": {"respect_ratelimit_headers": "yes"}}, {"name": "p"}
        )
        assert not provider._get_respect_ratelimit_headers(
            {"respect_ratelimit_headers": "off"}, {"name": "p"}
        )


class TestCooldownIntegration:
    def test_header_reset_sets_cooldown_over_backoff(self):
        """AC2/AC4: 429 reset header sets the cooldown and attempt record."""
        attempts = []
        response = Response(
            status_code=429, content=b"Too Many Requests", headers=_MUNITO_429_HEADERS
        )
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "Too Many Requests",
                _enabled_config(),
                provider_cfg=_munito_entry(),
            )
        # max(backoff 1s, reset 52s + margin 2s) = 54s
        assert cooldown == 54.0
        assert attempts[-1]["cooldown_seconds"] == 54.0
        assert attempts[-1]["ratelimit_reset_seconds"] == 52.0
        assert attempts[-1]["ratelimit_over_limit"] is True
        assert provider._provider_unavailable_until["munitoai-free"] == 1054.0

    def test_safety_margin_is_configurable(self):
        attempts = []
        response = Response(
            status_code=429, content=b"r", headers=_MUNITO_429_HEADERS
        )
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "r",
                _enabled_config(ratelimit_cooldown_margin_seconds=10),
                provider_cfg=_munito_entry(),
            )
        assert cooldown == 62.0

    def test_backoff_extended_beyond_short_reset(self):
        """AC2: per-provider failure counters still extend a short reset."""
        provider._provider_failure_count["munitoai-free"] = 4
        attempts = []
        response = Response(
            status_code=429,
            content=b"r",
            headers={"x-ratelimit-reset-requests": "5s"},
        )
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "r",
                _enabled_config(),
                provider_cfg=_munito_entry(),
            )
        # backoff 1*2^4 = 16s > reset 5s + 2s margin
        assert cooldown == 16.0

    def test_retry_after_still_wins_when_longer(self):
        attempts = []
        response = Response(
            status_code=429,
            content=b"r",
            headers={**_MUNITO_429_HEADERS, "Retry-After": "120"},
        )
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                600.0,
                attempts,
                "r",
                _enabled_config(),
                provider_cfg=_munito_entry(),
            )
        assert cooldown == 120.0

    def test_over_limit_without_parseable_reset_falls_back(self):
        """AC3: over-limit flag alone leaves exponential backoff unchanged."""
        attempts = []
        response = Response(
            status_code=429,
            content=b"r",
            headers={"x-ratelimit-over-limit": "yes"},
        )
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "r",
                _enabled_config(),
                provider_cfg=_munito_entry(),
            )
        assert cooldown == 1.0
        assert "ratelimit_reset_seconds" not in attempts[-1]

    def test_disabled_feature_ignores_headers(self):
        """AC6: default-off behaviour is byte-for-byte the backoff path."""
        attempts = []
        response = Response(
            status_code=429, content=b"r", headers=_MUNITO_429_HEADERS
        )
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "r",
                {},
                provider_cfg={"name": "munitoai-free"},
            )
        assert cooldown == 1.0
        assert "ratelimit_reset_seconds" not in attempts[-1]

    def test_non_429_ignores_ratelimit_headers(self):
        attempts = []
        response = Response(
            status_code=503, content=b"r", headers=_MUNITO_429_HEADERS
        )
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "r",
                _enabled_config(),
                provider_cfg=_munito_entry(),
            )
        assert cooldown == 1.0

    def test_global_config_enables_for_entry_without_flag(self):
        attempts = []
        response = Response(
            status_code=429, content=b"r", headers=_MUNITO_429_HEADERS
        )
        with patch("time.time", return_value=1000.0):
            cooldown = provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "r",
                {"server": {"respect_ratelimit_headers": True}},
                provider_cfg={"name": "munitoai-free"},
            )
        assert cooldown == 54.0

    def test_skip_log_reports_header_derived_remaining(self, caplog):
        """AC4: the resolver skip line reflects the header-derived remaining."""
        attempts = []
        response = Response(
            status_code=429, content=b"r", headers=_MUNITO_429_HEADERS
        )
        with patch("time.time", return_value=1000.0):
            provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "r",
                _enabled_config(),
                provider_cfg=_munito_entry(),
            )
            model_config = {
                "providers": [{"name": "munitoai-free", "type": "remote"}]
            }
            with caplog.at_level("INFO", logger="llama-proxy.provider"):
                selected = provider.resolve_provider(model_config)
        assert selected is None
        assert "munitoai-free" in caplog.text
        # LP-0MVFA1VRT007T9HT (83dcdd75): a quota 429 carrying an
        # ``x-ratelimit-reset-*`` header now quarantines the failing ACCOUNT for
        # the header-derived reset plus the 2-minute usage-limit margin, and the
        # resolver's account quarantine is checked BEFORE the entry cooldown. So
        # the skip line reports ``usage_limit_reset_pending`` with that window
        # (52s header + 120s margin = 172s) instead of the 54s entry cooldown —
        # the header-derived remaining still reaches the operator, through the
        # longer-lived gate that now supersedes the in-memory cooldown.
        assert "usage_limit_reset_pending" in caplog.text
        assert "reset_in=172s" in caplog.text


class TestPersistence:
    def test_header_derived_cooldown_reaches_provider_state_file(
        self, tmp_path, monkeypatch
    ):
        """AC5: the header-derived cooldown is persisted across restarts."""
        state_path = tmp_path / "state.json"
        monkeypatch.setenv(provider._PROVIDER_STATE_FILE_ENV, str(state_path))
        attempts = []
        response = Response(
            status_code=429, content=b"r", headers=_MUNITO_429_HEADERS
        )
        with patch("time.time", return_value=1000.0):
            provider._handle_http_error_with_cooldown(
                response,
                "munitoai-free",
                "remote",
                60.0,
                attempts,
                "r",
                _enabled_config(),
                provider_cfg=_munito_entry(),
            )
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        assert raw["provider_unavailable_until"]["munitoai-free"] == 1054.0

        # Simulate a restart at the same instant: the cooldown is restored.
        provider._provider_unavailable_until.clear()
        with patch("time.time", return_value=1000.0):
            restored, _ = provider.load_provider_state(state_path)
        assert restored == 1
        assert provider._provider_unavailable_until["munitoai-free"] == 1054.0
