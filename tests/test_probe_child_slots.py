#!/usr/bin/env python3
"""Unit tests for ``scripts/probe_child_slots.py``.

The probe is the AC2 evidence tool for LP-0MVVXGI490089OX5: it times
``GET /slots?model=…`` directly against the llama-server child port so backend
cost can be separated from the proxy's own serialisation. Every test here drives
the public functions against a synthetic log or a fake HTTP opener, so nothing
touches the live proxy, the live child or the production log directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import probe_child_slots as probe  # noqa: E402

# ---------------------------------------------------------------------------
# fake HTTP helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``urllib.request.urlopen``'s response object."""

    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _opener(payload: bytes, status: int = 200):
    def opener(url: str, timeout: float | None = None):
        return _FakeResponse(payload, status)

    return opener


def _timeout_opener(url: str, timeout: float | None = None):
    raise TimeoutError("timed out")


def _processed_opener(processing: int, total: int = 2):
    slots = [{"id": i, "is_processing": i < processing} for i in range(total)]
    return _opener(json.dumps(slots).encode())


def _no_sleep(_seconds: float) -> None:
    return None


def _status_line(**overrides: object) -> str:
    fields = {
        "capacity_hint": "prefill_bound",
        "local_in_flight_requests": 1,
        "measured_in_flight_requests": 2,
        "slots_stale": "true",
        "slots_stale_reason": "query_timeout",
        "slots_snapshot_age_seconds": 11.925,
        "slots_snapshot_source": "snapshot",
        "status_query_ms": 4,
        "slots_counts_ms": 2001,
        "slots_detail_ms": 1003,
        "slots_upstream_calls": 2,
        "available_slots": 1,
    }
    fields.update(overrides)
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    return f"2026-11-17 15:20:35,000 - INFO - status_request {rendered}"


# ---------------------------------------------------------------------------
# child-port resolution
# ---------------------------------------------------------------------------


class TestChildPortResolution:
    def test_last_proxying_line_for_the_model_wins(self):
        text = (
            "2026-11-17 15:00:00 - spawning server instance with name=Qwen3 on port 40000\n"
            "2026-11-17 15:01:00 - proxying request to model mxbai-embed on port 53437\n"
            "2026-11-17 15:02:00 - proxying request to model Qwen3 on port 44409\n"
            "2026-11-17 15:03:00 - proxying request to model mxbai-embed on port 53437\n"
            "2026-11-17 15:04:00 - proxying request to model Qwen3 on port 44410\n"
        )
        assert probe.resolve_child_port_from_text(text, "Qwen3") == 44410

    def test_other_model_lines_are_ignored(self):
        text = "proxying request to model mxbai-embed on port 53437\n"
        assert probe.resolve_child_port_from_text(text, "Qwen3") is None

    def test_spawn_line_is_the_fallback_when_no_request_line_exists(self):
        text = "spawning server instance with name=Qwen3 on port 58113\n"
        assert probe.resolve_child_port_from_text(text, "Qwen3") == 58113

    def test_proxying_line_prefers_over_an_older_spawn_line(self):
        text = (
            "spawning server instance with name=Qwen3 on port 40000\n"
            "proxying request to model Qwen3 on port 44409\n"
        )
        assert probe.resolve_child_port_from_text(text, "Qwen3") == 44409

    def test_unknown_model_resolves_to_none(self):
        assert probe.resolve_child_port_from_text("some unrelated log line\n", "Qwen3") is None

    def test_resolution_reads_the_log_tail(self, tmp_path):
        log = tmp_path / "llama-server.log"
        log.write_text("x" * 100 + "\nproxying request to model Qwen3 on port 44409\n", encoding="utf-8")
        assert probe.resolve_child_port(log, "Qwen3") == 44409

    def test_missing_log_file_resolves_to_none_rather_than_raising(self, tmp_path):
        assert probe.resolve_child_port(tmp_path / "absent.log", "Qwen3") is None


# ---------------------------------------------------------------------------
# sampling and summarising
# ---------------------------------------------------------------------------


class TestSampling:
    def test_successful_sample_counts_processing_slots(self):
        sample = probe.fetch_sample(0, "http://127.0.0.1:44409/slots?model=Qwen3", 5.0, opener=_processed_opener(1))
        assert sample.ok is True
        assert sample.timed_out is False
        assert sample.http_status == 200
        assert sample.processing_slots == 1
        assert sample.total_slots == 2
        assert sample.latency_ms is not None

    def test_timeout_is_recorded_as_a_timeout_not_a_latency(self):
        sample = probe.fetch_sample(0, "http://127.0.0.1:44409/slots?model=Qwen3", 35.0, opener=_timeout_opener)
        assert sample.ok is False
        assert sample.timed_out is True
        assert "timeout after 35s" == sample.error

    def test_http_error_records_the_status_code(self):
        import urllib.error

        def opener(url: str, timeout: float | None = None):
            raise urllib.error.HTTPError(url, 500, "boom", None, None)

        sample = probe.fetch_sample(0, "http://127.0.0.1:44409/slots?model=Qwen3", 5.0, opener=opener)
        assert sample.ok is False
        assert sample.timed_out is False
        assert sample.http_status == 500

    def test_run_probe_takes_the_requested_number_of_samples(self):
        samples = probe.run_probe(
            "http://127.0.0.1:44409/slots?model=Qwen3",
            samples=4,
            interval=0.0,
            timeout=1.0,
            opener=_processed_opener(1),
            sleep=_no_sleep,
        )
        assert [sample.index for sample in samples] == [0, 1, 2, 3]

    def test_summary_excludes_timeouts_from_latency_and_counts_them_separately(self):
        samples = [
            probe.Sample(0, "t", 4.0, ok=True, processing_slots=1, total_slots=2),
            probe.Sample(1, "t", 900.0, ok=True, processing_slots=1, total_slots=2),
            probe.Sample(2, "t", 35000.0, ok=False, timed_out=True, error="timeout after 35s"),
            probe.Sample(3, "t", 2500.0, ok=True, processing_slots=0, total_slots=2),
        ]
        summary = probe.summarise(samples, budget_ms=1000)
        assert summary["n"] == 4
        assert summary["ok"] == 3
        assert summary["timeouts"] == 1
        assert summary["errors"] == 0
        assert summary["busy_samples"] == 2
        assert summary["idle_samples"] == 1
        assert summary["min_ms"] == 4.0
        assert summary["max_ms"] == 2500.0
        # Nearest-rank over [4, 900, 2500]: p50 -> 900, p95 -> 2500.
        assert summary["p50_ms"] == 900.0
        assert summary["p95_ms"] == 2500.0
        assert summary["over_budget"] == 1

    def test_empty_summary_reports_none_rather_than_a_fabricated_zero(self):
        summary = probe.summarise([])
        assert summary["ok"] == 0
        assert summary["p50_ms"] is None
        assert summary["p95_ms"] is None
        assert summary["min_ms"] is None
        assert summary["max_ms"] is None


# ---------------------------------------------------------------------------
# same-moment proxy facts
# ---------------------------------------------------------------------------


class TestProxyFacts:
    def test_last_status_line_ignores_non_status_lines(self):
        log_text = (
            "2026-11-17 15:20:00,000 - INFO - status_request capacity_hint=idle slots_stale=false\n"
            "2026-11-17 15:20:01,000 - INFO - Stream finished: reason=stop tokens=1/2/3\n"
            "2026-11-17 15:20:02,000 - INFO - status_request capacity_hint=saturated slots_stale=true\n"
            "2026-11-17 15:20:03,000 - INFO - Fallback triggered for model=v1/chat/completions\n"
        )
        line = probe.last_status_line_from_text(log_text)
        assert line is not None
        assert "capacity_hint=saturated" in line

    def test_last_status_line_reads_the_log_tail(self, tmp_path):
        path = tmp_path / "proxy.log"
        path.write_text(_status_line() + "\nunrelated line\n", encoding="utf-8")
        line = probe.last_status_line(path)
        assert line is not None
        assert "capacity_hint=prefill_bound" in line
        assert probe.last_status_line(tmp_path / "absent.log") is None

    def test_status_facts_parse_the_read_path_fields(self):
        facts = probe.status_facts(_status_line())
        assert facts["capacity_hint"] == "prefill_bound"
        assert facts["local_in_flight_requests"] == 1
        assert facts["measured_in_flight_requests"] == 2
        assert facts["slots_stale"] == "true"
        assert facts["slots_stale_reason"] == "query_timeout"
        assert facts["slots_snapshot_age_seconds"] == 11.925
        assert facts["slots_snapshot_source"] == "snapshot"
        assert facts["status_query_ms"] == 4
        assert facts["slots_counts_ms"] == 2001
        assert facts["slots_detail_ms"] == 1003
        assert facts["slots_upstream_calls"] == 2

    def test_literal_none_fields_are_normalised_to_none(self):
        facts = probe.status_facts(
            _status_line(**{"slots_stale_reason": "None", "slots_snapshot_source": "None", "slots_stale": "false"})
        )
        assert facts["slots_stale_reason"] is None
        assert facts["slots_snapshot_source"] is None
        assert facts["slots_stale"] == "false"

    def test_missing_line_yields_no_facts(self):
        assert probe.status_facts(None) == {}


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


class TestCli:
    def test_pinned_port_probe_reports_json_and_writes_artifacts(self, tmp_path, capsys):
        artifact_dir = tmp_path / "probe"
        exit_code = probe.main(
            ["--port", "44409", "--samples", "3", "--interval", "0", "--json", "--write-artifact", str(artifact_dir)],
            opener=_processed_opener(1),
            sleep=_no_sleep,
        )
        assert exit_code == 0
        report = json.loads(capsys.readouterr().out)
        assert report["meta"]["child_port"] == 44409
        assert report["meta"]["work_item"] == "LP-0MVVXGI490089OX5"
        assert report["summary"]["n"] == 3
        assert report["summary"]["busy_samples"] == 3
        assert report["meta"]["url"].endswith("/slots?model=Qwen3")
        written = sorted(path.name for path in artifact_dir.iterdir())
        assert written == ["probe.json", "probe.txt"]

    def test_unresolved_port_exits_three(self, tmp_path, capsys):
        exit_code = probe.main(["--llama-log", str(tmp_path / "absent.log")], sleep=_no_sleep)
        assert exit_code == 3
        assert "no child port resolved" in capsys.readouterr().err

    def test_non_positive_sample_count_is_a_usage_error(self):
        assert probe.main(["--port", "1", "--samples", "0"], sleep=_no_sleep) == 2

    def test_rendered_report_names_the_separation_of_costs(self):
        samples = [
            probe.Sample(0, "t", 5.0, ok=True, processing_slots=1, total_slots=2),
            probe.Sample(1, "t", 35000.0, ok=False, timed_out=True, error="timeout after 35s"),
        ]
        report = probe.build_report(
            model="Qwen3",
            port=44409,
            url="http://127.0.0.1:44409/slots?model=Qwen3",
            samples=samples,
            summary=probe.summarise(samples),
            before=probe.status_facts(_status_line()),
            after=probe.status_facts(_status_line()),
            llama_log=Path("/var/log/llama-proxy/llama-server.log"),
            proxy_log=Path("/var/log/llama-proxy/proxy.log"),
            timeout=35.0,
        )
        text = probe.render(report)
        assert "p50=5.0 ms" in text
        assert "timeouts=1" in text
        assert "slots_counts_ms=2001" in text
        assert "timeout" in text.split("per-sample:")[1]
