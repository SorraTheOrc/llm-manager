#!/usr/bin/env python3
"""Warm-start priming gate + staggered-arrival tests (LP-0MVLHNEDO008FTBH).

The 2026-10-23 controlled N=2/N=3 slot sweep produced **local stream share
0.0 %**: every measured request launched inside the proxy's post-restart
cold-start window, where ``proxy/cold_start.py`` caps local dispatch at one
concurrent request until the first local prefill completes and calls
``mark_warm``. A fixed 60 s inter-request gap could not fix that, because the
*first* request never completed a local prefill at all.

These tests pin the two harness changes that make local concurrency
measurable:

* **warm-start gate** — one small priming request must COMPLETE ON THE LOCAL
  backend (``X-Resolved-Model`` local, e.g. ``local-qwen3/…``) before any
  measured request is sent. The gate retries within a bounded budget and, on
  persistent failure, raises ``WarmStartError`` so the run fails fast instead
  of silently measuring remote fallback.
* **``--stagger-seconds``** — measured starts are spaced by
  ``stagger_seconds * idx``; the default 0 preserves the all-at-once launch
  and the hard-coded ``INTER_REQUEST_GAP_S`` 60 s gap is gone.

Also pinned: the peak local concurrency metric
(``contention.max_concurrent_local_streams``) that makes the AC3 evidence
("at least 2 concurrent local streams observed") auditable from the run JSON.

All tests run against a scripted fake client — no live proxy, no restarts and
no network are involved.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "proxy" / "benchmarks" / "slot_benchmark.py"

_spec = importlib.util.spec_from_file_location("slot_benchmark_warm_start", MODULE_PATH)
sb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sb
_spec.loader.exec_module(sb)


# ---------------------------------------------------------------------------
# Scripted doubles
# ---------------------------------------------------------------------------

LOCAL = "local-qwen3/Qwen3"
REMOTE = "deepseek/v4-flash"


class _FakeResponse:
    """Minimal httpx.Response stand-in with the headers the harness reads."""

    def __init__(self, resolved_model: str | None, status_code: int = 200):
        self.status_code = status_code
        self.headers = {"X-Resolved-Model": resolved_model} if resolved_model else {}
        self.text = "upstream unavailable"
        self._payload = {"usage": {"prompt_tokens": 11, "completion_tokens": 2}}

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Scripted httpx.AsyncClient stand-in.

    Each ``post`` pops the next scripted item: a string is a resolved model
    served with HTTP 200, an exception instance is raised (transport error).
    """

    def __init__(self, script: list):
        self._script = list(script)
        self.calls: list[dict] = []

    async def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append(
            {"url": url, "json": json, "timeout": timeout, "headers": headers}
        )
        item = self._script.pop(0) if self._script else LOCAL
        if isinstance(item, BaseException):
            raise item
        return _FakeResponse(item)


def _recording_sleep(delays: list):
    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    return _sleep


# ---------------------------------------------------------------------------
# Warm-start priming gate
# ---------------------------------------------------------------------------

class TestWarmStartGate:
    def test_priming_gate_passes_on_first_local_completion(self):
        delays: list[float] = []
        client = _FakeClient([LOCAL])

        gate = asyncio.run(
            sb.warm_start_local_backend(client, sleep=_recording_sleep(delays))
        )

        assert gate.enabled is True
        assert gate.completed_locally is True
        assert gate.attempts == 1
        assert gate.resolved_model == LOCAL
        assert gate.errors == []
        assert delays == []
        assert len(client.calls) == 1

    def test_priming_request_is_small_and_single_turn(self):
        """AC1: priming is a single short turn so it fits the cold grace window."""
        client = _FakeClient([LOCAL])

        asyncio.run(sb.warm_start_local_backend(client, sleep=_recording_sleep([])))

        call = client.calls[0]
        assert call["json"]["max_tokens"] == sb.WARM_START_MAX_TOKENS
        assert call["json"]["max_tokens"] <= 20
        messages = call["json"]["messages"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == "Say Hi"
        assert call["timeout"] == sb.WARM_START_TIMEOUT_S
        assert call["timeout"] <= 120.0
        assert call["url"].endswith("/v1/chat/completions")

    def test_priming_request_uses_an_explicit_session(self):
        """The cold-start cap only applies to explicitly-sessioned requests.

        Without an ``X-Session-Id`` the priming request bypasses the
        ``_try_acquire_local_dispatch`` cold cap entirely, so a local success
        would NOT prove the cap lifted.
        """
        delays: list[float] = []
        client = _FakeClient([REMOTE, LOCAL])

        gate = asyncio.run(
            sb.warm_start_local_backend(
                client, max_attempts=2, retry_delay_s=1.0,
                sleep=_recording_sleep(delays),
            )
        )

        headers = [c["headers"] for c in client.calls]
        assert all(h and h.get("X-Session-Id") for h in headers)
        assert headers[0]["X-Session-Id"].startswith(sb.WARM_START_SESSION_PREFIX)
        # One logical client: retries keep the same session id.
        assert headers[0]["X-Session-Id"] == headers[1]["X-Session-Id"]
        assert gate.session_id == headers[0]["X-Session-Id"]

    def test_priming_gate_retries_until_request_lands_locally(self):
        delays: list[float] = []
        client = _FakeClient([REMOTE, LOCAL])

        gate = asyncio.run(
            sb.warm_start_local_backend(
                client, max_attempts=3, retry_delay_s=5.0,
                sleep=_recording_sleep(delays),
            )
        )

        assert gate.completed_locally is True
        assert gate.attempts == 2
        assert delays == [5.0]
        # The remote attempt is recorded as evidence, not silently dropped.
        assert len(gate.errors) == 1
        assert REMOTE in gate.errors[0]

    def test_priming_gate_retries_transport_errors(self):
        delays: list[float] = []
        client = _FakeClient([sb.httpx.ConnectError("backend not ready"), LOCAL])

        gate = asyncio.run(
            sb.warm_start_local_backend(
                client, max_attempts=2, retry_delay_s=1.0,
                sleep=_recording_sleep(delays),
            )
        )

        assert gate.attempts == 2
        assert gate.completed_locally is True
        assert len(gate.errors) == 1

    def test_priming_gate_fails_fast_when_no_attempt_lands_locally(self):
        """AC1: persistent remote fallback aborts the run with a clear error."""
        delays: list[float] = []
        client = _FakeClient([REMOTE, REMOTE, REMOTE])

        with pytest.raises(sb.WarmStartError) as excinfo:
            asyncio.run(
                sb.warm_start_local_backend(
                    client, max_attempts=3, retry_delay_s=2.0,
                    sleep=_recording_sleep(delays),
                )
            )

        message = str(excinfo.value)
        # Diagnostic must name the mechanism being worked around so an operator
        # can find the cause without re-reading the harness source.
        assert "cold-start" in message
        assert "local_dispatch_denied" in message
        assert "in cooldown" in message
        assert "--no-warm-start" in message
        # Bounded retry budget: exactly max_attempts requests, max_attempts-1 waits.
        assert len(client.calls) == 3
        assert delays == [2.0, 2.0]


# ---------------------------------------------------------------------------
# Warm-start gate inside run_slot_benchmark (ordering + abort semantics)
# ---------------------------------------------------------------------------

def _patch_run(monkeypatch, handler):
    """Patch send_request/_capture_gpu_memory, returning recorded measured indices."""
    calls: list[dict] = []
    measured: list[int] = []

    async def _fake_send_request(client, messages, index, **kwargs):
        calls.append({"index": index, "messages": messages, "kwargs": kwargs})
        return handler(index, kwargs, measured)

    monkeypatch.setattr(sb, "send_request", _fake_send_request)
    monkeypatch.setattr(sb, "_capture_gpu_memory", lambda: None)
    return calls, measured


def _is_priming(kwargs: dict) -> bool:
    return kwargs.get("max_tokens") == sb.WARM_START_MAX_TOKENS


def test_gate_completes_locally_before_measured_requests(monkeypatch):
    """AC1: nothing measured is sent until the priming request lands locally."""

    def handler(index, kwargs, measured):
        if not _is_priming(kwargs):
            measured.append(index)
        return sb.RequestResult(index, "completed", 0.5, resolved_model=LOCAL)

    calls, measured = _patch_run(monkeypatch, handler)

    result = asyncio.run(
        sb.run_slot_benchmark(
            slot_count=2, shared_content="shared", user_prompt="count",
            num_requests=2, warmup=0, warm_start=True,
        )
    )

    assert calls[0]["kwargs"]["max_tokens"] == sb.WARM_START_MAX_TOKENS
    assert len(measured) == 2 * sb.TOTAL_SESSION_TURNS
    assert result.warm_start is not None
    assert result.warm_start.completed_locally is True
    # The gate outcome is recorded in the run artefact so the local share is
    # attributable to a warm backend, not to luck.
    artefact = result.to_dict()
    assert artefact["warm_start"]["completed_locally"] is True
    assert artefact["warm_start"]["enabled"] is True


def test_gate_failure_aborts_before_any_measured_request(monkeypatch):
    """AC1: the run fails fast; not one measured request is sent remotely."""

    def handler(index, kwargs, measured):
        if not _is_priming(kwargs):
            measured.append(index)
        return sb.RequestResult(index, "completed", 0.5, resolved_model=REMOTE)

    calls, measured = _patch_run(monkeypatch, handler)

    with pytest.raises(sb.WarmStartError):
        asyncio.run(
            sb.run_slot_benchmark(
                slot_count=2, shared_content="shared", user_prompt="count",
                num_requests=2, warmup=0, warm_start=True,
                warm_start_attempts=2, warm_start_retry_delay=0.0,
            )
        )

    assert measured == []
    assert len(calls) == 2  # both priming attempts, nothing else


def test_gate_disabled_by_default_in_leaf_run(monkeypatch):
    def handler(index, kwargs, measured):
        if not _is_priming(kwargs):
            measured.append(index)
        return sb.RequestResult(index, "completed", 0.5, resolved_model=LOCAL)

    calls, measured = _patch_run(monkeypatch, handler)

    result = asyncio.run(
        sb.run_slot_benchmark(
            slot_count=2, shared_content="shared", user_prompt="count",
            num_requests=1, warmup=0,
        )
    )

    assert all(not _is_priming(c["kwargs"]) for c in calls)
    assert len(measured) == sb.TOTAL_SESSION_TURNS
    assert result.warm_start is not None
    assert result.warm_start.enabled is False


# ---------------------------------------------------------------------------
# Stagger math
# ---------------------------------------------------------------------------

class TestStaggerMath:
    def test_delta_between_successive_starts_is_stagger_seconds(self):
        offsets = [sb.stagger_delay_seconds(idx, 15.0) for idx in range(4)]
        assert offsets == [0.0, 15.0, 30.0, 45.0]
        assert [b - a for a, b in zip(offsets, offsets[1:])] == [15.0, 15.0, 15.0]

    def test_default_zero_means_all_at_once(self):
        assert [sb.stagger_delay_seconds(idx, 0.0) for idx in range(4)] == [0, 0, 0, 0]

    def test_warmup_and_negative_indices_never_delay(self):
        assert sb.stagger_delay_seconds(-1, 60.0) == 0.0
        assert sb.stagger_delay_seconds(-5, 60.0) == 0.0

    def test_negative_stagger_is_clamped_to_zero(self):
        assert sb.stagger_delay_seconds(2, -30.0) == 0.0

    def test_run_applies_stagger_to_measured_starts(self, monkeypatch):
        """The launch path really sleeps stagger_seconds × idx, not 60 s."""

        def handler(index, kwargs, measured):
            if not _is_priming(kwargs):
                measured.append(index)
            return sb.RequestResult(index, "completed", 0.5, resolved_model=LOCAL)

        _patch_run(monkeypatch, handler)
        delays: list[float] = []

        asyncio.run(
            sb.run_slot_benchmark(
                slot_count=3, shared_content="shared", user_prompt="count",
                num_requests=3, warmup=0, stagger_seconds=7.0,
                sleep=_recording_sleep(delays),
            )
        )

        # idx 0 starts immediately; idx 1/2 wait one/two stagger intervals.
        assert delays == [7.0, 14.0]

    def test_run_default_stagger_has_no_delays(self, monkeypatch):
        def handler(index, kwargs, measured):
            if not _is_priming(kwargs):
                measured.append(index)
            return sb.RequestResult(index, "completed", 0.5, resolved_model=LOCAL)

        _patch_run(monkeypatch, handler)
        delays: list[float] = []

        asyncio.run(
            sb.run_slot_benchmark(
                slot_count=2, shared_content="shared", user_prompt="count",
                num_requests=2, warmup=0, sleep=_recording_sleep(delays),
            )
        )

        assert delays == []


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

class TestCli:
    def test_stagger_seconds_defaults_to_zero(self):
        assert sb.parse_args(["--slots", "2"]).stagger_seconds == 0.0

    def test_stagger_seconds_is_parsed_as_float(self):
        args = sb.parse_args(["--sweep", "--stagger-seconds", "12.5"])
        assert args.stagger_seconds == 12.5

    def test_warm_start_defaults_on_for_sweep_only(self):
        """AC1: `--sweep` warms by default; single/all runs stay opt-in."""
        assert sb.resolve_warm_start(sb.parse_args(["--sweep"])) is True
        assert sb.resolve_warm_start(sb.parse_args(["--all"])) is False
        assert sb.resolve_warm_start(sb.parse_args(["--slots", "2"])) is False

    def test_explicit_flags_override_the_mode_default(self):
        assert sb.resolve_warm_start(
            sb.parse_args(["--sweep", "--no-warm-start"])
        ) is False
        assert sb.resolve_warm_start(
            sb.parse_args(["--slots", "2", "--warm-start"])
        ) is True

    def test_warm_start_tuning_flags_have_defaults(self):
        args = sb.parse_args(["--sweep"])
        assert args.warm_start_attempts == sb.WARM_START_MAX_ATTEMPTS
        assert args.warm_start_timeout == sb.WARM_START_TIMEOUT_S

    def test_dry_run_reports_warm_start_and_stagger(self, capsys):
        sb.main([
            "--sweep", "--resume", "2", "--concurrency", "3",
            "--prompt-tokens", "2000", "--stagger-seconds", "0", "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Slot counts: 2, 3" in out
        assert "Warm start: enabled" in out
        assert "Stagger: 0s" in out

    def test_negative_stagger_is_rejected(self):
        with pytest.raises(SystemExit) as excinfo:
            sb.main(["--slots", "2", "--stagger-seconds", "-1", "--dry-run"])
        assert excinfo.value.code == 2

    def test_help_documents_new_flags(self, capsys):
        with pytest.raises(SystemExit):
            sb.parse_args(["--help"])
        out = capsys.readouterr().out
        assert "--warm-start" in out
        assert "--no-warm-start" in out
        assert "--stagger-seconds" in out


# ---------------------------------------------------------------------------
# Peak local concurrency metric (AC3 auditability)
# ---------------------------------------------------------------------------

class TestPeakLocalConcurrency:
    def test_two_overlapping_local_streams_are_counted(self):
        results = [
            sb.RequestResult(0, "completed", 10.0, resolved_model=LOCAL,
                             started_at=100.0, ended_at=110.0),
            sb.RequestResult(1, "completed", 10.0, resolved_model="local/Qwen3",
                             started_at=105.0, ended_at=115.0),
            sb.RequestResult(2, "completed", 10.0, resolved_model=REMOTE,
                             started_at=105.0, ended_at=115.0),
        ]
        peak, reason = sb.derive_max_concurrent_local_streams(results)
        assert peak == 2
        assert reason is None

    def test_sequential_local_streams_count_one(self):
        results = [
            sb.RequestResult(0, "completed", 10.0, resolved_model=LOCAL,
                             started_at=100.0, ended_at=110.0),
            sb.RequestResult(1, "completed", 10.0, resolved_model=LOCAL,
                             started_at=110.0, ended_at=120.0),
        ]
        assert sb.derive_max_concurrent_local_streams(results) == (1, None)

    def test_remote_only_run_counts_zero(self):
        results = [
            sb.RequestResult(0, "completed", 10.0, resolved_model=REMOTE,
                             started_at=100.0, ended_at=110.0),
        ]
        assert sb.derive_max_concurrent_local_streams(results) == (0, None)

    def test_missing_timestamps_report_reason(self):
        results = [sb.RequestResult(0, "completed", 10.0, resolved_model=LOCAL)]
        peak, reason = sb.derive_max_concurrent_local_streams(results)
        assert peak is None
        assert reason

    def test_metric_surfaces_in_contention_summary_and_artefact(self):
        results = [
            sb.RequestResult(0, "completed", 10.0, resolved_model=LOCAL,
                             started_at=100.0, ended_at=110.0),
            sb.RequestResult(1, "completed", 10.0, resolved_model=LOCAL,
                             started_at=101.0, ended_at=111.0),
        ]
        metrics = sb.build_contention_metrics(results, log_files=[])

        assert metrics.max_concurrent_local_streams == 2
        assert metrics.local_stream_share == 1.0
        run = sb.SlotRunResult(slot_count=2, results=results, contention=metrics)
        assert run.summary()["contention"]["max_concurrent_local_streams"] == 2
        assert run.to_dict()["contention"]["max_concurrent_local_streams"] == 2

    def test_request_timestamps_are_recorded_in_artefact(self):
        r = sb.RequestResult(0, "completed", 1.0, started_at=100.0, ended_at=101.5)
        d = r.to_dict()
        assert d["started_at"] == 100.0
        assert d["ended_at"] == 101.5
