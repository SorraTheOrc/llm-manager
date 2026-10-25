"""Off-peak low-priority fallback gating (LP-0MVF1FM9500070U1).

Parent item: LP-0MVF17VP70053JXX — "Enable low-priority fallback to
opencode-go during its off-peak windows".

The ``low`` priority chain for ``plan``/``author``/``code`` gains the three
``opencode-go`` entries (after the free tier) so that local-slot overflow
during the cheap priority window can spill onto idle paid capacity that is
already confined to its off-peak ``available_times`` windows.

This module pins the *mechanism* before the config change lands (the exact
``proxy/config.yaml`` chain assertion is owned by the config item
LP-0MVF1FQJR000NH2I, which is the only item allowed to touch the file):

- target chain shape: ``local-qwen3 → munitoai-free → opencode-go-3 →
  opencode-go-2 → opencode-go`` (AC2), with ``medium``/``high`` untouched.
- UTC window gating (AC1/AC3): the opencode entries are eligible inside
  ``available_times`` (e.g. 00:00–01:00, 04:00–06:00) and skipped outside
  them (e.g. 01:00–04:00, 06:00–10:00 gaps), falling through exactly as the
  two-entry chain does today.
- behaviour neutrality while the free tier still has budget: the paid tiers
  are never promoted ahead of ``munitoai-free``.
- per-API-key 429 failover (AC4): a 429 on ``opencode-go-3`` does **not**
  exclude ``opencode-go-2``/``opencode-go`` — usage-limit quarantine is keyed
  per account (``_usage_limit_account_key``), so the three entries carry
  independent quota (LP-0MSMBWB23009XYPW, LP-0MVCX0E2V001VB5N).
- same-failure-domain 5xx caveat (AC5): a 5xx/stall on ``opencode-go-3``
  **does** exclude its same-gateway siblings as one failure domain
  (``_failure_domain_key``), so the three entries are not 3× outage
  resilience (LP-0MSG45I8Q0020N1F).
- no paid-tier leak (AC5): ``munitoai``/``deepseek-flash`` never appear in a
  ``low`` chain, and any non-free remote tier in a ``low`` chain must carry
  ``available_times`` (off-peak-only spend).

The chain-order/window/failover tests are driven through the real routing
seams (``provider_chain_for_priority`` + ``_resolve_provider_with_exclusions``)
against provider entries taken from the live ``proxy/config.yaml``, so the
test cannot drift from the shipped entries.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import proxy.provider as provider
import pytest
import yaml
from proxy.priority import (
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    priority_routing_profile,
    provider_chain_for_priority,
    proxy_dir,
)

# Target low chain (parent AC1). Kept as data so the per-model loop and the
# mechanism assertions can never disagree.
LOCAL_TIER = "local-qwen3"
FREE_TIER = "munitoai-free"
OPENCODE_NAMES = ["opencode-go-3", "opencode-go-2", "opencode-go"]
TARGET_LOW_CHAIN = [LOCAL_TIER, FREE_TIER, *OPENCODE_NAMES]
LOW_CHAIN_MODELS = ("plan", "author", "code")

# The off-peak windows every opencode-go entry must keep (LP-0MSXPHTD9004DZ4P).
# Asserted verbatim so this test also guards against accidental widening.
OPENCODE_WINDOWS = ["00:00-01:00", "04:00-06:00", "10:00-00:00"]

# Inside the UTC 04:00-06:00 window, and inside the 01:00-04:00 gap.
INSIDE_WINDOW_UTC = datetime(2026, 1, 5, 4, 30, tzinfo=UTC)
OUTSIDE_WINDOW_UTC = datetime(2026, 1, 5, 2, 0, tzinfo=UTC)


class _FakeDatetime:
    """Stand-in for ``proxy.provider.datetime`` (deterministic UTC clock).

    Mirrors the convention already used in ``test_provider_fallback.py`` so the
    window gate inside ``_resolve_provider_with_exclusions`` observes a fixed
    ``now`` without touching the production seam.
    """

    _now = INSIDE_WINDOW_UTC

    @classmethod
    def now(cls, tz=None):
        return cls._now

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        return datetime.fromtimestamp(ts, tz)


def _patch_utc_now(dt):
    _FakeDatetime._now = dt
    return patch("proxy.provider.datetime", _FakeDatetime)


@pytest.fixture(autouse=True)
def reset_provider_state():
    """Clear cross-request quarantine state before and after each test."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_limit_last_probe.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_limit_last_probe.clear()


# ---------------------------------------------------------------------------
# Fixtures built from the live config
# ---------------------------------------------------------------------------


def _live_config() -> dict:
    with open(proxy_dir() / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _live_model(model_name: str) -> dict:
    return _live_config()["models"][model_name]


def _live_entry(model_name: str, entry_name: str) -> dict:
    for entry in _live_model(model_name)["providers"]:
        if entry.get("name") == entry_name:
            return entry
    raise AssertionError(f"{entry_name!r} not found in model {model_name!r}")


def _target_chain_entries(model_name: str) -> list[dict]:
    """Resolve the target low chain from the live provider ``priorities`` labels.

    Since LP-0MVGPZWY0001JD1D the chain is a plain filter over the model's own
    ``providers`` list, so the live labels are the only source of truth: no
    synthetic route table is injected here.
    """
    return provider_chain_for_priority(_live_model(model_name), PRIORITY_LOW)


def _next_provider(chain: list[dict], named: set[str], domains: set[str]):
    return provider._resolve_provider_with_exclusions({"providers": chain}, set(named), set(domains))


# ---------------------------------------------------------------------------
# AC2 — chain shape and ordering
# ---------------------------------------------------------------------------


class TestTargetChainShape:
    @pytest.mark.parametrize("model_name", LOW_CHAIN_MODELS)
    def test_low_chain_is_local_free_then_all_three_opencode_entries(self, model_name):
        names = [e["name"] for e in _target_chain_entries(model_name)]
        assert names == TARGET_LOW_CHAIN

    @pytest.mark.parametrize("model_name", LOW_CHAIN_MODELS)
    def test_medium_and_high_chains_are_the_full_provider_list(self, model_name):
        # AC5: medium/high semantics must stay byte-identical to before — the
        # full ``providers`` list, in order.
        model = _live_model(model_name)
        expected = [p["name"] for p in model["providers"]]
        for priority in (PRIORITY_MEDIUM, PRIORITY_HIGH):
            names = [p["name"] for p in provider_chain_for_priority(model, priority)]
            assert names == expected


class TestLiveConfigInvariants:
    """Invariants that hold before *and* after the config item lands."""

    @pytest.mark.parametrize("model_name", LOW_CHAIN_MODELS)
    def test_low_chain_starts_local_then_free_tier(self, model_name):
        names = [e["name"] for e in provider_chain_for_priority(_live_model(model_name), PRIORITY_LOW)]
        assert names[0] == LOCAL_TIER
        assert FREE_TIER in names

    @pytest.mark.parametrize("model_name", LOW_CHAIN_MODELS)
    def test_every_paid_tier_in_a_low_chain_is_window_gated(self, model_name):
        # AC6/off-peak-only spend: anything after the free tier that is not
        # free/local must carry ``available_times``.
        chain = provider_chain_for_priority(_live_model(model_name), PRIORITY_LOW)
        for entry in chain:
            name = entry.get("name", "")
            if name in (LOCAL_TIER, FREE_TIER):
                continue
            assert entry.get("available_times"), (
                f"paid tier {name!r} appears in a low chain without "
                "available_times — off-peak-only spend would be violated"
            )

    def test_fully_paid_tiers_never_enter_a_low_chain(self):
        # AC6: ``munitoai`` and ``deepseek-flash`` have no window discipline,
        # so they must never be reachable from a ``low`` route. Membership is
        # declared by each entry's own ``priorities`` label
        # (LP-0MVGPZWY0001JD1D).
        config = _live_config()
        banned = {"munitoai", "deepseek-flash"}
        for model_name, model in config["models"].items():
            low_labels = [
                e.get("name")
                for e in (model or {}).get("providers") or []
                if PRIORITY_LOW in (e.get("priorities") or [])
            ]
            resolved = provider_chain_for_priority(model, PRIORITY_LOW)
            leaked = banned & {e.get("name", "") for e in resolved}
            assert not leaked, (
                f"model {model_name!r} low chain {low_labels!r} "
                f"leaks paid tiers {sorted(leaked)}"
            )

    @pytest.mark.parametrize("model_name", LOW_CHAIN_MODELS)
    def test_opencode_entries_keep_their_off_peak_windows(self, model_name):
        # AC2 (parent): no widening. All three entries share the documented
        # off-peak windows, byte-identical to the pre-change file.
        for entry_name in OPENCODE_NAMES:
            entry = _live_entry(model_name, entry_name)
            assert entry["available_times"] == OPENCODE_WINDOWS

    @pytest.mark.parametrize("model_name", LOW_CHAIN_MODELS)
    def test_live_low_chain_is_exactly_the_target_order(self, model_name):
        # AC1 (parent): the shipped config's ``low`` chain is exactly
        # ``[local-qwen3, munitoai-free, opencode-go-3, opencode-go-2,
        # opencode-go]`` — free tier ordered before any paid entry and the
        # three keys in providers-list order.
        names = [e["name"] for e in provider_chain_for_priority(_live_model(model_name), PRIORITY_LOW)]
        assert names == TARGET_LOW_CHAIN


# ---------------------------------------------------------------------------
# AC3 — UTC window gating
# ---------------------------------------------------------------------------


class TestWindowGating:
    @pytest.mark.parametrize("entry_name", OPENCODE_NAMES)
    def test_entry_is_eligible_inside_its_window(self, entry_name):
        entry = _live_entry("plan", entry_name)
        assert provider._is_within_allowed_window(entry, now_utc=INSIDE_WINDOW_UTC) is True

    @pytest.mark.parametrize("entry_name", OPENCODE_NAMES)
    def test_entry_is_skipped_inside_the_utc_gap(self, entry_name):
        entry = _live_entry("plan", entry_name)
        assert provider._is_within_allowed_window(entry, now_utc=OUTSIDE_WINDOW_UTC) is False

    def test_in_window_low_chain_falls_through_to_the_paid_tiers(self):
        # local + free already spent → the paid off-peak tier absorbs the
        # overflow while its window is open.
        chain = _target_chain_entries("plan")
        with _patch_utc_now(INSIDE_WINDOW_UTC):
            picked = _next_provider(chain, {LOCAL_TIER, FREE_TIER}, set())
        assert picked is not None
        assert picked["name"] == "opencode-go-3"

    def test_out_of_window_low_chain_behaves_exactly_as_today(self):
        # During the UTC 01:00-04:00 / 06:00-10:00 gaps the paid entries are
        # skipped, so an exhausted chain resolves to ``None`` just like the
        # two-entry ``[local-qwen3, munitoai-free]`` chain before this change.
        chain = _target_chain_entries("plan")
        with _patch_utc_now(OUTSIDE_WINDOW_UTC):
            picked = _next_provider(chain, {LOCAL_TIER, FREE_TIER}, set())
        assert picked is None

    def test_out_of_window_low_chain_still_serves_local_and_free(self):
        chain = _target_chain_entries("plan")
        with _patch_utc_now(OUTSIDE_WINDOW_UTC):
            first = _next_provider(chain, set(), set())
            second = _next_provider(chain, {LOCAL_TIER}, set())
        assert first is not None and first["name"] == LOCAL_TIER
        assert second is not None and second["name"] == FREE_TIER

    @pytest.mark.parametrize("model_name", LOW_CHAIN_MODELS)
    def test_free_tier_is_never_preempted_by_a_paid_tier(self, model_name):
        # Behaviour neutrality: as long as the free tier has budget it is
        # chosen ahead of every paid opencode entry, window or not.
        chain = _target_chain_entries(model_name)
        with _patch_utc_now(INSIDE_WINDOW_UTC):
            picked = _next_provider(chain, {LOCAL_TIER}, set())
        assert picked is not None
        assert picked["name"] == FREE_TIER


# ---------------------------------------------------------------------------
# AC4 — per-API-key 429 failover vs same-domain 5xx
# ---------------------------------------------------------------------------


class TestFailoverScope:
    def test_each_opencode_entry_has_its_own_account_key(self):
        # ``_usage_limit_account_key`` = ``api_key_env@endpoint:model`` — the
        # three entries differ only by key, hence independent quota.
        keys = [provider._usage_limit_account_key(_live_entry("plan", name)) for name in OPENCODE_NAMES]
        assert len(set(keys)) == len(OPENCODE_NAMES)
        for name, key in zip(OPENCODE_NAMES, keys):
            entry = _live_entry("plan", name)
            assert key.startswith(f"{entry['api_key_env']}@")

    def test_429_on_one_entry_keeps_its_sibling_accounts_eligible(self):
        chain = _target_chain_entries("plan")
        go3 = _live_entry("plan", "opencode-go-3")
        excluded = {provider._usage_limit_account_key(go3)}
        with _patch_utc_now(INSIDE_WINDOW_UTC):
            picked = _next_provider(chain, {LOCAL_TIER, FREE_TIER}, excluded)
        assert picked is not None
        assert picked["name"] == "opencode-go-2", (
            "a 429 on opencode-go-3 must not exclude the independent opencode-go-2 account"
        )

    def test_429_on_all_accounts_exhausts_the_chain(self):
        chain = _target_chain_entries("plan")
        excluded = {provider._usage_limit_account_key(_live_entry("plan", name)) for name in OPENCODE_NAMES}
        with _patch_utc_now(INSIDE_WINDOW_UTC):
            picked = _next_provider(chain, {LOCAL_TIER, FREE_TIER}, excluded)
        assert picked is None

    def test_opencode_siblings_share_one_failure_domain(self):
        domains = {provider._failure_domain_key(_live_entry("plan", name)) for name in OPENCODE_NAMES}
        assert len(domains) == 1, (
            "the three opencode-go entries share one gateway+model and are therefore ONE failure domain"
        )

    def test_5xx_on_one_entry_excludes_its_siblings(self):
        # Honest caveat (AC5): a 5xx/stall is recorded against the shared
        # failure domain, so the chain does NOT gain 3x outage resilience.
        chain = _target_chain_entries("plan")
        go3 = _live_entry("plan", "opencode-go-3")
        excluded = {provider._failure_domain_key(go3)}
        with _patch_utc_now(INSIDE_WINDOW_UTC):
            picked = _next_provider(chain, {LOCAL_TIER, FREE_TIER}, excluded)
        assert picked is None, "same-gateway siblings must be skipped after a 5xx on opencode-go-3 (one failure domain)"


# ---------------------------------------------------------------------------
# AC7 — the paid-tier invariant is documented where routing knobs are
# ---------------------------------------------------------------------------


class TestRoutingProfileDocs:
    def test_low_profile_docstring_records_the_off_peak_exception(self):
        # The routing-knob docstring must state that ``low`` may spend on
        # paid tiers inside their off-peak ``available_times`` windows, and
        # must carry the per-key-429 / same-domain-5xx rationale so nobody
        # assumes 3x outage resilience.
        doc = priority_routing_profile.__doc__ or ""
        assert "off-peak" in doc.lower()
        assert "available_times" in doc
        assert "3x rate-limit" in doc
        assert "outage resilience" in doc
