"""Documentation-drift tests for the quota-surge configuration keys.

Work item LP-0MVF7KRH30000KOA (epic AC12 / AC12c). The design is documented in
``docs/dev/quota-surge.md`` and inline in ``proxy/config.yaml``; this module
keeps that documentation from drifting away from the shipped behaviour:

- AC12c (present): every documented ``server.quota_*`` key is present in
  ``proxy/config.yaml`` with its documented default value, so the shipped
  config file and the design doc cannot disagree.
- AC12c (fallback): when a key is absent the readers fall back to exactly the
  documented default, checked against both an empty config and the real,
  key-absent ``proxy/config_test.yaml`` fixture.
- ``proxy/config_test.yaml`` stays valid YAML with a ``server`` section.

Existing reader tests (``test_quota_pacing.py::test_config_defaults``,
``test_quota_state.py::test_config_defaults_and_overrides``) already cover the
override/clamping behaviour of the readers themselves. These tests deliberately
do **not** duplicate that — they cover the *documentation* axis: the on-disk
config values and the doc↔code default cross-check.

Doc-claim cross-check (epic AC12d)
----------------------------------

Every behavioural claim in ``docs/dev/quota-surge.md`` maps to an executed
test; the gaps column is intentionally empty (no claim is documented without
test coverage). The most load-bearing claims:

| Doc claim | Test(s) |
|---|---|
| Poller polls at most once per ``quota_poll_interval_seconds`` per account | ``test_quota_state.py::test_interval_gating_allows_one_poll_per_interval`` |
| Concurrent pollers coalesce into one request per account | ``test_quota_state.py::test_concurrent_refreshes_coalesce_into_one_request`` |
| A poll failure keeps the previous values, warns, never delays a request | ``test_quota_state.py::test_poll_failure_keeps_previous_values_and_warns``; ``test_quota_state.py::test_cache_read_does_not_wait_for_inflight_poll`` |
| ``effective_headroom`` is the minimum of the three windows | ``test_quota_state.py::test_derivation_reports_windows_and_minimum_headroom`` |
| Non-``ok`` / missing windows are blocking; missing reset ⇒ 0 | ``test_quota_state.py::test_non_ok_window_status_is_blocking``; ``test_quota_state.py::test_missing_window_is_treated_as_blocking``; ``test_quota_state.py::test_missing_or_invalid_resets_at_yields_zero`` |
| Quota is per ``api_key_env`` (shared key ⇒ one snapshot/poll) | ``test_quota_state.py::test_account_key_uses_api_key_env_not_model``; ``test_quota_state.py::test_shared_api_key_env_shares_one_snapshot_and_poll`` |
| Score is monotonic in headroom and as the reset approaches | ``test_quota_pacing.py::test_urgency_score_is_non_decreasing_in_headroom``; ``test_quota_pacing.py::test_urgency_score_is_non_decreasing_as_reset_approaches`` |
| Score formula / baseline outside the window | ``test_quota_pacing.py::test_urgency_score_inside_window_equals_headroom_over_time_with_target``; ``test_quota_pacing.py::test_baseline_outside_surge_window`` |
| Achievability from the documented 20 %/5 h and 50 %/7 d caps | ``test_quota_pacing.py::test_achievable_fraction_follows_the_documented_cap_relationship``; ``test_quota_pacing.py::test_cap_rate_is_the_binding_sustainable_rate`` |
| Ramp starts before the final week when the residual is unspendable | ``test_quota_pacing.py::test_ramp_started_achievability_table``; ``test_quota_pacing.py::test_ramp_started_is_reported_outside_the_surge_window`` |
| No score implies exceeding a rolling/weekly cap | ``test_quota_pacing.py::test_no_score_exceeds_a_window_cap``; ``test_quota_pacing.py::test_score_sweep_never_breaches_a_window_cap`` |
| Peak burns at 2× and escalation only when otherwise unspendable | ``test_quota_pacing.py::test_burn_multiplier_represents_the_2x_peak_rate``; ``test_quota_pacing.py::test_peak_escalation_only_when_otherwise_unspendable``; ``test_quota_pacing.py::test_assessment_exposes_peak_weighting`` |
| Ordering prefers the higher-scoring account; ties keep config order | ``test_quota_ordering.py::test_prefers_account_with_more_headroom``; ``test_quota_ordering.py::test_preference_reverses_with_percentages``; ``test_quota_ordering.py::test_equal_headroom_preserves_config_order`` |
| Final-week bypass at 6 days, not at 8 days; clears after reset | ``test_quota_surge_bypass.py::test_six_days_out_bypasses_window_at_0200``; ``test_quota_surge_bypass.py::test_eight_days_out_remains_ineligible``; ``test_quota_surge_bypass.py::test_bypass_clears_once_the_reset_passes`` |
| Bypass is ``available_times``-only (rate limit, circuit breaker, quarantine, failure domain still apply) | ``test_quota_surge_bypass.py::test_rate_limited_entry_still_skipped_under_surge``; ``test_quota_surge_bypass.py::test_circuit_broken_entry_still_skipped_under_surge``; ``test_quota_surge_bypass.py::test_usage_limit_quarantined_entry_still_skipped_under_surge``; ``test_quota_surge_bypass.py::test_failure_domain_sibling_still_excluded_under_surge`` |
| Hard stop skips exhausted accounts and applies with surge off/dry-run | ``test_quota_hard_stop.py::test_quota_skip_table``; ``test_quota_hard_stop.py::test_skip_applies_when_surge_disabled_or_dry_run``; ``test_quota_hard_stop.py::test_single_dispatch_followed_by_crossing_is_the_only_request`` |
| Hard stop is cumulative with the usage-limit quarantine | ``test_quota_hard_stop.py::test_gousagelimiterror_quarantine_and_hard_stop_are_cumulative``; ``test_quota_hard_stop.py::test_quota_skip_never_overrides_an_active_quarantine`` |
| Kill switch is byte-identical to today; dry-run logs without applying | ``test_quota_kill_switch.py::test_kill_switch_is_byte_identical_to_today``; ``test_quota_kill_switch.py::test_dry_run_leaves_routing_unchanged``; ``test_quota_kill_switch.py::test_dry_run_logs_the_bypass_it_would_apply`` |
| Stale state suspends surge (fail-safe); missing state never suspends | ``test_quota_kill_switch.py::test_stale_state_suspends_the_bypass_and_warns_once``; ``test_quota_kill_switch.py::test_stale_state_reverts_ordering_to_config_order``; ``test_quota_kill_switch.py::test_missing_state_never_suspends`` |
| Metrics, ``/events`` and home page expose derived state; no key leaks | ``test_quota_observability.py::test_metrics_gauges_carry_the_exposed_values``; ``test_quota_observability.py::test_status_events_payload_carries_the_quota_section``; ``test_quota_observability.py::test_home_page_rows_show_quota_fields``; ``test_quota_observability.py::test_no_api_key_value_is_exposed_on_any_surface`` |
| End-to-end flow and unaffected pre-existing gates | ``test_quota_surge_integration.py::test_full_flow_poll_order_surge_hard_stop_stale``; ``test_quota_surge_integration.py::test_pre_existing_routing_gates_unaffected_under_surge`` |
| ``/v1/usage`` request contract (path, Bearer, ``no-store``) | this file: ``test_usage_transport_requests_the_documented_contract`` |
| Documented config defaults match the shipped config and absent-key fallback | this file: ``test_documented_key_is_present_in_config_yaml_with_documented_default``; ``test_absent_key_falls_back_to_documented_default``; ``test_real_key_absent_config_uses_documented_default`` |

The upstream *discovery facts* in section 2 (headers carry no quota data,
no model selector, no credits endpoint) are observed properties of the
third-party endpoint, not guarantees of the proxy: the proxy's own contract
here is graceful degradation, which the poller tests cover. They are
re-verified only by the on-demand live probes, not by the unit suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from proxy import quota_pacing, quota_state

_PROXY_DIR = Path(__file__).resolve().parents[1]

#: The two config files this documentation contract covers.
CONFIG_YAML = _PROXY_DIR / "config.yaml"
CONFIG_TEST_YAML = _PROXY_DIR / "config_test.yaml"

#: ``/v1/usage`` path appended to a provider endpoint by the default poller
#: transport (documented in ``docs/dev/quota-surge.md`` section 2).
USAGE_PATH = "/v1/usage"


class _FakeResponse:
    """Minimal httpx response stand-in (``json`` + ``raise_for_status``)."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    """Records the documented ``/v1/usage`` request the poller would make."""

    requests: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url: str, headers: dict | None = None):
        _FakeAsyncClient.requests.append(
            {"url": url, "headers": headers or {}, "timeout": self.timeout}
        )
        return _FakeResponse({"usage": {}})


#: ``server.quota_*`` key → ``(documented default in config.yaml,
#: reader, documented reader default)``.
#:
#: ``quota_surge_accounts`` is intentionally absent: it is an optional
#: allow-list, commented out in ``config.yaml``, whose documented "default" is
#: *unset* (all ``opencode-*`` accounts). It is covered separately.
DOCUMENTED_KEYS: dict[str, tuple[object, object, object]] = {
    "quota_surge_enabled": (True, quota_pacing.surge_enabled, True),
    "quota_surge_dry_run": (False, quota_pacing.surge_dry_run, False),
    "quota_poll_interval_seconds": (
        900,
        quota_state.poll_interval_seconds,
        900.0,
    ),
    "quota_surge_days": (7, quota_pacing.surge_days, 7.0),
    "quota_exhaustion_percent": (
        99.5,
        quota_state.quota_exhaustion_percent,
        99.5,
    ),
    "quota_state_max_age_seconds": (
        3600,
        quota_state.quota_state_max_age_seconds,
        3600.0,
    ),
    "quota_pacing_target": (0.95, quota_pacing.pacing_target, 0.95),
}


def _load(path: Path) -> dict:
    """Parse a YAML config file, failing the test on invalid YAML."""
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    assert isinstance(parsed, dict), f"{path} must parse to a mapping"
    return parsed


def test_config_yaml_parses_with_a_server_section():
    """``proxy/config.yaml`` is valid YAML with a ``server`` mapping."""
    config = _load(CONFIG_YAML)
    assert isinstance(config.get("server"), dict), "config.yaml lacks a server section"


@pytest.mark.parametrize("key", sorted(DOCUMENTED_KEYS))
def test_documented_key_is_present_in_config_yaml_with_documented_default(key):
    """AC12/AC12c: the shipped config carries every documented default."""
    documented, _reader, _reader_default = DOCUMENTED_KEYS[key]
    server = _load(CONFIG_YAML)["server"]
    assert key in server, f"documented key server.{key} is missing from config.yaml"
    assert server[key] == documented, (
        f"config.yaml server.{key}={server[key]!r} does not match the documented "
        f"default {documented!r}"
    )


@pytest.mark.parametrize("key", sorted(DOCUMENTED_KEYS))
def test_absent_key_falls_back_to_documented_default(key):
    """AC12c: an absent key is read with exactly the documented default."""
    _documented, reader, reader_default = DOCUMENTED_KEYS[key]
    assert reader({}) == reader_default
    assert reader({"server": {}}) == reader_default


@pytest.mark.parametrize("key", sorted(DOCUMENTED_KEYS))
def test_real_key_absent_config_uses_documented_default(key):
    """AC12c: the shipped ``config_test.yaml`` (no quota keys) uses defaults."""
    _documented, reader, reader_default = DOCUMENTED_KEYS[key]
    config = _load(CONFIG_TEST_YAML)
    assert key not in config.get("server", {})
    assert reader(config) == reader_default


def test_config_test_yaml_stays_valid():
    """The test fixture stays valid YAML with a ``server`` section."""
    config = _load(CONFIG_TEST_YAML)
    assert isinstance(config.get("server"), dict)


def test_optional_surge_accounts_documents_all_accounts_when_absent():
    """``quota_surge_accounts`` unset means "all ``opencode-*``" (documented)."""
    assert quota_state.surge_accounts({}) is None
    assert quota_state.surge_accounts(_load(CONFIG_TEST_YAML)) is None
    quoted = {"server": {"quota_surge_accounts": ["opencode-go-3"]}}
    assert quota_state.surge_accounts(quoted) == ["opencode-go-3"]


@pytest.mark.asyncio
async def test_usage_transport_requests_the_documented_contract(monkeypatch):
    """The default poller transport makes the request section 2 documents.

    ``GET {endpoint}/v1/usage`` with a Bearer key and ``Cache-Control:
    no-store`` — the sentence in ``docs/dev/quota-surge.md`` section 2 that
    describes *proxy* behaviour (as opposed to observed upstream facts), so it
    must be pinned here to keep the doc honest (epic AC12d).
    """
    _FakeAsyncClient.requests.clear()
    monkeypatch.setattr(quota_state.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setenv("OPENCODE_DOCS_TEST_KEY", "sk-docs-test")
    entry = {
        "name": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_DOCS_TEST_KEY",
    }
    body = await quota_state._default_quota_usage_transport(entry)
    assert body == {"usage": {}}
    assert len(_FakeAsyncClient.requests) == 1
    request = _FakeAsyncClient.requests[0]
    assert request["url"] == "https://opencode.ai/zen/go" + USAGE_PATH
    assert request["headers"]["Authorization"] == "Bearer sk-docs-test"
    assert request["headers"]["Cache-Control"] == "no-store"
    assert request["timeout"] == quota_state.DEFAULT_USAGE_TIMEOUT_SECONDS
