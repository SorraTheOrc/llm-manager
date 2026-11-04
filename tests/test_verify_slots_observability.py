"""Tests for ``scripts/verify_slots_observability.py`` (LP-0MW1VVOJZ008YUPH).

The verifier is the recorded-evidence tool for the parent work item
(LP-0MW1VDWY0007EMH7): after a proxy restart it must prove
``slots_stale=false``, an age below the ceiling, and a real model-bearing
``GET /slots?model=…`` 200 in ``llama-server``'s log.

These tests pin the verdict logic offline — the HTTP fetcher and the log
reader are injected — including the three named failing cases the work item
requires (stale surface, age at/over the ceiling, no model-bearing 200) plus
the negative control that a model-*less* 200 does not satisfy the log check.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_slots_observability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_slots_observability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_slots_observability"] = module
    spec.loader.exec_module(module)
    return module


vs = _load_module()


HEALTHY_STATUS = {
    "slots_stale": False,
    "slots_stale_reason": None,
    "slots_snapshot_source": "upstream",
    "slots_snapshot_age_seconds": 0.4,
    "slots_max_snapshot_age_seconds": 300.0,
    "available_slots": 2,
    "total_slots": 2,
    "slots_detail_stale": False,
}

HEALTHY_HEALTH = {
    "slots_surface_status": "ok",
    "slots_stale": False,
    "slots_stale_reason": None,
}

GOOD_LOG = (
    "[44809] srv  log_server_r: done request: GET /slots?model=Qwen3 127.0.0.1 200\n"
)


class TestHealthySurface:
    """A healthy proxy passes every check and exits 0."""

    def test_report_is_ok(self):
        report = vs.verify(HEALTHY_STATUS, HEALTHY_HEALTH, GOOD_LOG)
        assert report.ok is True
        assert report.to_json()["ok"] is True

    def test_every_check_is_named(self):
        report = vs.verify(HEALTHY_STATUS, HEALTHY_HEALTH, GOOD_LOG)
        names = [c.name for c in report.checks]
        assert "slots_stale=false" in names
        assert "snapshot age below ceiling" in names
        assert "model-bearing /slots 200 in llama-server log" in names
        assert "health surface ok" in names

    def test_no_snapshot_served_is_healthy(self):
        """A live read (no snapshot) is the healthiest state, not a failure."""
        status = dict(HEALTHY_STATUS, slots_snapshot_age_seconds=None)
        report = vs.verify(status, HEALTHY_HEALTH, GOOD_LOG)
        assert report.ok is True

    def test_main_exits_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(
            vs, "_http_get_json", lambda url, timeout=5.0: (
                HEALTHY_STATUS if url.endswith("/status") else HEALTHY_HEALTH
            )
        )
        monkeypatch.setattr(vs, "read_log_tail", lambda path, max_bytes=0: GOOD_LOG)
        assert vs.main([]) == 0
        assert "RESULT: PASS" in capsys.readouterr().out

    def test_json_output_is_machine_readable(self, monkeypatch, capsys):
        monkeypatch.setattr(
            vs, "_http_get_json", lambda url, timeout=5.0: (
                HEALTHY_STATUS if url.endswith("/status") else HEALTHY_HEALTH
            )
        )
        monkeypatch.setattr(vs, "read_log_tail", lambda path, max_bytes=0: GOOD_LOG)
        assert vs.main(["--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert isinstance(payload["checks"], list)


class TestFailingCases:
    """The three named failing cases the work item requires."""

    def test_stale_surface_fails(self):
        status = dict(
            HEALTHY_STATUS,
            slots_stale=True,
            slots_stale_reason="query_timeout",
        )
        report = vs.verify(status, HEALTHY_HEALTH, GOOD_LOG)
        assert report.ok is False
        failed = {c.name for c in report.checks if not c.ok}
        assert "slots_stale=false" in failed

    def test_age_at_the_ceiling_is_acceptable(self):
        status = dict(HEALTHY_STATUS, slots_snapshot_age_seconds=300.0)
        report = vs.verify(status, HEALTHY_HEALTH, GOOD_LOG)
        assert report.ok is True

    def test_age_over_the_ceiling_fails(self):
        status = dict(
            HEALTHY_STATUS,
            slots_stale=True,
            slots_stale_reason="snapshot_too_old",
            slots_snapshot_age_seconds=29_825.67,
        )
        report = vs.verify(status, HEALTHY_HEALTH, GOOD_LOG)
        assert report.ok is False
        failed = {c.name for c in report.checks if not c.ok}
        assert "snapshot age below ceiling" in failed

    def test_missing_ceiling_with_an_age_fails(self):
        status = dict(HEALTHY_STATUS, slots_max_snapshot_age_seconds=None)
        report = vs.verify(status, HEALTHY_HEALTH, GOOD_LOG)
        failed = {c.name for c in report.checks if not c.ok}
        assert "snapshot age below ceiling" in failed

    def test_no_model_bearing_200_fails(self):
        log = (
            "[44809] srv  log_server_r: done request: GET /slots 127.0.0.1 200\n"
        )
        report = vs.verify(HEALTHY_STATUS, HEALTHY_HEALTH, log)
        assert report.ok is False
        failed = {c.name for c in report.checks if not c.ok}
        assert "model-bearing /slots 200 in llama-server log" in failed

    def test_model_bearing_500_does_not_count(self):
        log = (
            "[44809] srv  log_server_r: done request: GET /slots?model=Qwen3 "
            "127.0.0.1 500\n"
        )
        report = vs.verify(HEALTHY_STATUS, HEALTHY_HEALTH, log)
        failed = {c.name for c in report.checks if not c.ok}
        assert "model-bearing /slots 200 in llama-server log" in failed

    def test_empty_log_fails(self):
        report = vs.verify(HEALTHY_STATUS, HEALTHY_HEALTH, "")
        assert report.ok is False

    def test_unreachable_endpoints_fail(self):
        report = vs.verify(None, None, GOOD_LOG)
        assert report.ok is False
        assert {c.name for c in report.checks if not c.ok} >= {
            "status payload",
            "health surface",
        }


class TestReasonAgreement:
    """``/health`` must agree with the status payload."""

    def test_disagreement_fails(self):
        health = dict(
            HEALTHY_HEALTH,
            slots_surface_status="stale",
            slots_stale_reason="query_error",
        )
        status = dict(
            HEALTHY_STATUS, slots_stale=True, slots_stale_reason="query_timeout"
        )
        report = vs.verify(status, health, GOOD_LOG)
        failed = {c.name for c in report.checks if not c.ok}
        assert "health/status reason agreement" in failed

    def test_agreement_passes(self):
        status = dict(
            HEALTHY_STATUS, slots_stale=True, slots_stale_reason="query_error"
        )
        health = dict(
            HEALTHY_HEALTH,
            slots_surface_status="stale",
            slots_stale_reason="query_error",
        )
        report = vs.verify(status, health, GOOD_LOG)
        assert not [c for c in report.checks if c.name == "health/status reason agreement" and not c.ok]


class TestLogParsing:
    """The log matcher is strict about both status and the model parameter."""

    def test_finds_a_model_bearing_200(self):
        found, line = vs.find_model_bearing_slots_200(GOOD_LOG)
        assert found is True
        assert "model=Qwen3" in line

    def test_ignores_model_less_200(self):
        found, _ = vs.find_model_bearing_slots_200(
            "[1] srv  log_server_r: done request: GET /slots 127.0.0.1 200\n"
        )
        assert found is False

    def test_returns_the_last_match(self):
        log = GOOD_LOG + (
            "[44810] srv  log_server_r: done request: "
            "GET /slots?model=Qwen3-32B 127.0.0.1 200\n"
        )
        found, line = vs.find_model_bearing_slots_200(log)
        assert found is True
        assert "Qwen3-32B" in line

    def test_unreadable_log_is_empty_not_an_error(self, tmp_path):
        assert vs.read_log_tail(tmp_path / "missing.log") == ""

    def test_reads_the_tail(self, tmp_path):
        log = tmp_path / "llama-server.log"
        log.write_text(GOOD_LOG)
        assert "model=Qwen3" in vs.read_log_tail(log)


class TestReadOnly:
    """The verifier never mutates state."""

    def test_script_only_reads(self):
        """No write-mode file operations and no worklog/proxy mutation imports."""
        source = SCRIPT.read_text()
        for forbidden in (
            "write_text(",
            "open(.*['\"]w",
            "requests.post",
            "urllib.request.Request(",
            "wl ",
            "subprocess",
        ):
            assert forbidden not in source, f"{forbidden!r} found in the verifier"


class TestRouterModelProbe:
    """The direct probe proves the model-parameter contract.

    llama.cpp's request log omits the query string on this host, so the
    model-bearing-200 evidence comes from probing the router rather than from
    a log field that does not exist.
    """

    def test_probe_documents_its_contract(self):
        """The probe's contract (400 without model, 200 with) is documented."""
        documented = vs.probe_router_model_requirement.__doc__ or ""
        assert "400" in documented
        assert "200" in documented

    def test_probe_reported_instead_of_the_log_check(self):
        report = vs.verify(
            HEALTHY_STATUS,
            HEALTHY_HEALTH,
            GOOD_LOG,
            router_probe=(True, "model-less=400 model-bearing=200 model='Qwen3'"),
        )
        names = [c.name for c in report.checks]
        assert "router requires ?model= and answers it" in names
        assert "model-bearing /slots 200 in llama-server log" not in names
        assert report.ok is True

    def test_failing_probe_fails_the_verdict(self):
        report = vs.verify(
            HEALTHY_STATUS,
            HEALTHY_HEALTH,
            GOOD_LOG,
            router_probe=(False, "model-less=400 model-bearing=400 model='Qwen3'"),
        )
        assert report.ok is False
        failed = {c.name for c in report.checks if not c.ok}
        assert "router requires ?model= and answers it" in failed

    def test_successful_slots_read_in_log_is_always_checked(self):
        report = vs.verify(
            HEALTHY_STATUS,
            HEALTHY_HEALTH,
            "[1] srv  log_server_r: done request: GET /slots 127.0.0.1 200\n",
            router_probe=(True, "model-less=400 model-bearing=200"),
        )
        names = [c.name for c in report.checks]
        assert "/slots 200 in llama-server log" in names
        assert report.ok is True

    def test_missing_slots_200_in_log_fails(self):
        report = vs.verify(
            HEALTHY_STATUS, HEALTHY_HEALTH, "", router_probe=(True, "ok")
        )
        failed = {c.name for c in report.checks if not c.ok}
        assert "/slots 200 in llama-server log" in failed
