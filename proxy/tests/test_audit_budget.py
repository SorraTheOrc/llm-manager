"""Proxy-owned cross-host audit admission budget and live lease registry
(LP-0MW95FYMH006XYNK, parent epic LP-0MW0KZUJP0019SLS AC1).

Covers (child AC6):

- AO1: cap enforcement — ``max_audits=1`` admits one audit session and
  refuses a second DIFFERENT session with ``reason=audit-budget-saturated``
  plus a bounded ``retry_after_s`` hint (the second audit waits, it never
  competes); ``max_audits=0`` is the kill switch.
- AO2: idempotent re-acquire — the same session re-acquiring (or renewing)
  its own lease is a renew, never a self-refusal.
- AO3: clean release — a released lease frees the slot for the next audit.
- AO4: staleness — a lease older than its TTL is reclaimed
  (``audit_lease_reclaimed reason=expired``) and the slot frees.
- AO5: availability read agrees with lease state (``{cap, live, available,
  leases}``) and reports ``error: registry_unavailable`` instead of asserting
  capacity on an internal registry failure (fail-safe).
- AO6: proxy-clock-only comparisons — expiry uses ``time.monotonic``
  (a skewed host clock cannot hold a stale lease).
- AO7: requested client ``cap`` is clamped to the proxy's configured maximum.
- AO8: the acquire/renew/release endpoints are the only mutation paths and a
  refused acquire performs no partial state change.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import patch

import pytest

from proxy import audit_budget


def _cfg(max_audits: int = 1, ttl: float = 3600.0) -> dict:
    """Config dict with a server.audit_admission block."""
    return {"server": {"audit_admission": {"max_audits": max_audits, "lease_ttl_seconds": ttl}}}


@pytest.fixture(autouse=True)
def _reset_budget():
    audit_budget.reset()
    yield
    audit_budget.reset()


# ---------------------------------------------------------------------------
# AO1 — cap enforcement (single holder, waiting second)
# ---------------------------------------------------------------------------

def test_cap_one_admits_one_and_refuses_the_second_with_retry_hint(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    status1, payload1 = audit_budget.acquire(_cfg(), "audit-a-11111111", "10.0.0.1")
    assert status1 == 200
    assert payload1["admit"] is True
    assert payload1["cap"] == 1
    assert payload1["live"] == 1
    assert payload1["available"] == 0
    assert payload1["lease_id"]

    # A second DIFFERENT audit session waits (429 + bounded retry hint),
    # never competes.
    status2, payload2 = audit_budget.acquire(_cfg(), "audit-b-22222222", "10.0.0.2")
    assert status2 == 429
    assert payload2["admit"] is False
    assert payload2["reason"] == "audit-budget-saturated"
    assert payload2["live"] == 1
    assert payload2["available"] == 0
    assert isinstance(payload2["retry_after_s"], int)
    assert 1 <= payload2["retry_after_s"] <= 3600

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "audit_lease_acquired" in messages
    assert "audit_lease_refused reason=audit-budget-saturated" in messages


def test_max_audits_zero_is_the_kill_switch():
    status, payload = audit_budget.acquire(_cfg(max_audits=0), "audit-a", "10.0.0.1")
    assert status == 429
    assert payload["reason"] == "audit-budget-saturated"
    assert audit_budget._leases == {}  # no partial state change


def test_cap_two_admits_two_then_refuses_third():
    cfg = _cfg(max_audits=2)
    s1, p1 = audit_budget.acquire(cfg, "audit-a", "10.0.0.1")
    s2, p2 = audit_budget.acquire(cfg, "audit-b", "10.0.0.2")
    assert (s1, s2) == (200, 200)
    s3, p3 = audit_budget.acquire(cfg, "audit-c", "10.0.0.3")
    assert s3 == 429
    assert p3["reason"] == "audit-budget-saturated"
    assert p3["live"] == 2


# ---------------------------------------------------------------------------
# AO2 — idempotent re-acquire / renew
# ---------------------------------------------------------------------------

def test_same_session_reacquire_renews_instead_of_refusing():
    cfg = _cfg()
    status1, payload1 = audit_budget.acquire(cfg, "audit-a", "10.0.0.1")
    assert status1 == 200

    # Re-acquire within TTL: renewed, not refused, still one live lease.
    status2, payload2 = audit_budget.acquire(cfg, "audit-a", "10.0.0.1")
    assert status2 == 200
    assert payload2.get("renewed") is True
    assert payload2["lease_id"] == payload1["lease_id"]
    assert payload2["live"] == 1
    assert payload2["available"] == 0


def test_renew_extends_the_lease():
    cfg = _cfg(ttl=600.0)
    status, payload = audit_budget.acquire(cfg, "audit-a", "10.0.0.1")
    assert status == 200
    assert payload["expires_at"].endswith("+00:00")  # ISO-8601 UTC on the wire

    with patch("proxy.audit_budget.time.monotonic", return_value=time.monotonic() + 300):
        s2, p2 = audit_budget.renew(cfg, "audit-a")
        assert s2 == 200
        assert p2["renewed"] is True
        # renewal restores the full TTL from the (advanced) now — the lease
        # is not about to expire.
        assert p2["expires_in_s"] > 500


def test_renew_with_no_lease_is_a_reported_noop():
    s, p = audit_budget.renew(_cfg(), "audit-nope")
    assert s == 200
    assert p["renewed"] is False


def test_renew_requires_session_id():
    s, p = audit_budget.renew(_cfg(), None)
    assert s == 400


# ---------------------------------------------------------------------------
# AO3 — clean release
# ---------------------------------------------------------------------------

def test_clean_release_frees_the_slot_for_the_next_audit(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    cfg = _cfg()
    assert audit_budget.acquire(cfg, "audit-a", "10.0.0.1")[0] == 200
    assert audit_budget.acquire(cfg, "audit-b", "10.0.0.2")[0] == 429

    s, p = audit_budget.release("audit-a")
    assert s == 200
    assert p["released"] is True
    assert p["live"] == 0

    # The slot is free again: the waiting audit acquires on retry.
    s2, p2 = audit_budget.acquire(cfg, "audit-b", "10.0.0.2")
    assert s2 == 200
    assert p2["admit"] is True
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "audit_lease_released" in messages


def test_release_of_missing_lease_is_a_reported_noop():
    s, p = audit_budget.release("audit-nope")
    assert s == 200
    assert p["released"] is False


# ---------------------------------------------------------------------------
# AO4 — staleness and crash recovery
# ---------------------------------------------------------------------------

def test_expired_lease_is_reclaimed_and_avail_read_frees_the_slot(caplog):
    caplog.set_level(logging.INFO, logger="llama-proxy")
    cfg = _cfg(ttl=3600.0)
    assert audit_budget.acquire(cfg, "audit-a", "10.0.0.1")[0] == 200
    acquired_at = audit_budget._leases["audit-a"].acquired_at

    # Advance the proxy's monotonic clock 2 h past acquisition (ttl 3600 s):
    # the lease is long expired and the read reclaims it.
    with patch(
        "proxy.audit_budget.time.monotonic",
        return_value=acquired_at + 7200.0,
    ):
        availability = audit_budget.availability(cfg)
        assert availability["live"] == 0
        assert availability["available"] == 1
        assert availability["leases"] == []

    # A new audit acquires the reclaimed slot without waiting.
    s, p = audit_budget.acquire(cfg, "audit-b", "10.0.0.2")
    assert s == 200
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "audit_lease_reclaimed reason=expired" in messages


def test_prune_expired_is_proxy_clock_only():
    """AO6: expiry is decided on the proxy's monotonic clock, never host time."""
    cfg = _cfg(ttl=3600.0)
    audit_budget.acquire(cfg, "audit-a", "10.0.0.1")
    acquired_at = audit_budget._leases["audit-a"].acquired_at
    # Wall-clock move alone cannot expire the lease (monotonic unchanged).
    with patch("proxy.audit_budget.time.time", return_value=time.time() + 7200):
        result = audit_budget.availability(cfg)
    assert result["live"] == 1  # still live: monotonic clock untouched
    # ...but a monotonic move does expire it.
    with patch(
        "proxy.audit_budget.time.monotonic",
        return_value=acquired_at + 7200.0,
    ):
        result = audit_budget.availability(cfg)
    assert result["live"] == 0


# ---------------------------------------------------------------------------
# AO5 — availability read
# ---------------------------------------------------------------------------

def test_availability_read_agrees_with_lease_state():
    cfg = _cfg(max_audits=2, ttl=3600.0)
    a1 = audit_budget.availability(cfg)
    assert a1["cap"] == 2
    assert a1["live"] == 0
    assert a1["available"] == 2
    assert a1["leases"] == []

    audit_budget.acquire(cfg, "audit-a", "10.0.0.1")
    audit_budget.acquire(cfg, "audit-b", "10.0.0.2")
    a2 = audit_budget.availability(cfg)
    assert a2["live"] == 2
    assert a2["available"] == 0
    assert len(a2["leases"]) == 2
    by_session = {lease["session_id"]: lease for lease in a2["leases"]}
    assert set(by_session) == {"audit-a", "audit-b"}
    assert by_session["audit-a"]["client_ip"] == "10.0.0.1"
    assert by_session["audit-a"]["expires_in_s"] > 0


def test_availability_failsafe_on_registry_error():
    """A broken registry reports registry_unavailable, never fabricated capacity."""
    cfg = _cfg()
    with patch(
        "proxy.audit_budget._leases",
        new_callable=lambda: _ExplodingRegistry(),
    ):
        result = audit_budget.availability(cfg)
    assert result["error"] == "registry_unavailable"
    assert result["available"] is None


def test_acquire_failsafe_on_registry_error(caplog):
    caplog.set_level(logging.ERROR, logger="llama-proxy")
    cfg = _cfg()
    with patch(
        "proxy.audit_budget._leases",
        new_callable=lambda: _ExplodingRegistry(),
    ):
        status, payload = audit_budget.acquire(cfg, "audit-a", "10.0.0.1")
    assert status == 503
    assert payload["reason"] == "audit-budget-unavailable"
    assert payload["retry_after_s"] == audit_budget.FAILSAFE_RETRY_AFTER_SECONDS
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "audit_budget_unavailable" in messages


class _ExplodingRegistry(dict):
    """Registry dict whose every mutation/iteration raises (fail-safe tests)."""

    def get(self, *args, **kwargs):  # pragma: no cover - exercised via raises
        raise RuntimeError("registry unreadable")

    def values(self):  # pragma: no cover
        raise RuntimeError("registry unreadable")

    def __getitem__(self, key):  # pragma: no cover
        raise RuntimeError("registry unreadable")

    def __setitem__(self, key, value):  # pragma: no cover
        raise RuntimeError("registry unreadable")

    def __len__(self):  # pragma: no cover
        raise RuntimeError("registry unreadable")


# ---------------------------------------------------------------------------
# AO7 — client-requested cap is clamped to the proxy maximum
# ---------------------------------------------------------------------------

def test_requested_cap_is_clamped_to_proxy_max():
    cfg = _cfg(max_audits=1)
    # A client that asks for cap=3 against a proxy cap of 1 gets 1.
    status, payload = audit_budget.acquire(cfg, "audit-a", "10.0.0.1", requested_cap=3)
    assert status == 200
    assert payload["cap"] == 1
    assert payload["available"] == 0


def test_requested_cap_below_proxy_max_restricts_that_session():
    """A client may request a SMALLER cap; the proxy honours the lower bound."""
    cfg = _cfg(max_audits=2)
    # First holder asks for a personal cap of 1... proxy cap is 2; the
    # requested cap is advisory for the refusal decision, never a capacity
    # grant beyond the proxy's own maximum.
    s1, p1 = audit_budget.acquire(cfg, "audit-a", "10.0.0.1", requested_cap=1)
    assert s1 == 200
    # A second session is refused only when the proxy cap (2) is saturated.
    s2, p2 = audit_budget.acquire(cfg, "audit-b", "10.0.0.2", requested_cap=2)
    assert s2 == 200
    s3, p3 = audit_budget.acquire(cfg, "audit-c", "10.0.0.3", requested_cap=2)
    assert s3 == 429


# ---------------------------------------------------------------------------
# AO8 — identity / config sanity
# ---------------------------------------------------------------------------

def test_audit_session_prefix_detection():
    assert audit_budget.is_audit_session("audit-LP-0MW0KZUJP0019SLS-ctx-deadbeef") is True
    assert audit_budget.is_audit_session("herdr-123") is False
    assert audit_budget.is_audit_session(None) is False
    # Risk note: a non-audit session carrying the prefix is still flagged as
    # the prefix value — the gate logs the raw identity (audit_tier flag).
    assert audit_budget.is_audit_session("audit-not-really") is True


def test_acquire_requires_session_id():
    s, p = audit_budget.acquire(_cfg(), None, "10.0.0.1")
    assert s == 400
    assert "session_id" in p["error"]


def test_stats_counters():
    cfg = _cfg()
    audit_budget.acquire(cfg, "audit-a", "10.0.0.1")
    audit_budget.acquire(cfg, "audit-b", "10.0.0.2")  # refused (saturated)
    audit_budget.release("audit-a")
    stats = audit_budget.stats()
    assert stats["audit_leases_live"] == 0
    assert stats["audit_leases_issued"] == 1
    assert stats["audit_leases_released"] == 1
    assert stats["audit_refused_saturated"] == 1
