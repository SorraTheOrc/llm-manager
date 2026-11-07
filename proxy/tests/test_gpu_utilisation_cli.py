#!/usr/bin/env python3
"""Tests for ``scripts/gpu_utilisation.py`` (LP-0MVT1O52G000LMC6).

The CLI answers "how busy was the GPU between 17:00 and 19:00?" from the
deployed Prometheus store, so every test here drives the *query path* rather
than a live scrape: recorded ``/api/v1/query_range`` payloads
(``monitoring/tests/fixtures/``) are served by an injected getter or by a
loopback HTTP stub. That keeps the assertions deterministic and offline, and it
proves the figure is a direct measurement — nothing here reads ``proxy.log``.

The honesty contract these tests pin (parent LP-0MVNWUEY8008XM6W, AC4):

* a window with missing samples reports the missing share as **unknown** —
  never as idle, never renormalised away;
* computing + idle + unknown is 100 % by construction;
* an instant only counts as *observed* when a sentinel proves the pipeline was
  reporting (``up{job=...} == 1`` or a slot heartbeat fresher than one scrape
  interval) **and** the computing-workload rule can actually classify it.

Arithmetic (the 90 W threshold, the 0/1 definitions, the percentage split) is
asserted against the public functions, so a regression shows up as a wrong
number rather than a ``None``.
"""

import contextlib
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import gpu_utilisation as gu  # noqa: E402

FIXTURES = PROJECT_ROOT / "monitoring" / "tests" / "fixtures"
BUNDLE = json.loads((FIXTURES / "window_with_gap.json").read_text())
#: Live capture of the 2026-11-02 05:25-05:45 slice of the measured worst-case
#: window: 15 of its 81 instants break a naive same-instant rule comparison.
JITTER_BUNDLE = json.loads((FIXTURES / "window_rules_jitter.json").read_text())
START = BUNDLE["capture"]["window"]["start_unix"]
END = BUNDLE["capture"]["window"]["end_unix"]
STEP = 15
#: Inclusive 15 s grid across the two-hour window (7200 / 15 + 1).
EXPECTED = 481
#: Every stream in the bundle carries the same 10-minute hole.
OBSERVED = 441
COMPUTING = 68
IDLE = 373
UNKNOWN = 40


# ---------------------------------------------------------------------------
# Helpers: recorded payloads, fake getters, loopback Prometheus stubs
# ---------------------------------------------------------------------------


def matrix(points: dict[int, float]) -> dict:
    """A Prometheus ``query_range`` matrix payload holding one series."""
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": "synthetic"},
                    "values": [[str(ts), str(value)] for ts, value in sorted(points.items())],
                }
            ],
        },
    }


def empty_matrix() -> dict:
    return {"status": "success", "data": {"resultType": "matrix", "result": []}}


def bundle_getter(responses=None, calls=None):
    """Return a getter that answers queries from recorded payloads."""
    payloads = BUNDLE["responses"] if responses is None else responses

    def getter(url: str) -> dict:
        query = parse_qs(urlparse(url).query)["query"][0]
        if calls is not None:
            calls.append(query)
        return payloads[query]

    return getter


def querying_getter(points_by_query, calls=None):
    """A getter that answers every query with one synthetic series."""

    def getter(url: str) -> dict:
        query = parse_qs(urlparse(url).query)["query"][0]
        if calls is not None:
            calls.append(query)
        points = points_by_query.get(query, {})
        return matrix(points) if points else empty_matrix()

    return getter


def closed_port_url() -> str:
    """A loopback URL nothing listens on (connection refused)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@contextlib.contextmanager
def prometheus_stub(responses, status: int = 200):
    """Serve recorded payloads over real HTTP on a loopback port."""
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query).get("query", [""])[0]
            requests.append(query)
            if status != 200:
                body = json.dumps({"status": "error", "error": "boom"}).encode()
            else:
                body = json.dumps(responses[query]).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep pytest output clean
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_cli(argv, *, getter=None, now=None, capsys=None):
    """Run ``main`` and return ``(exit_code, stdout, stderr)``."""
    code = gu.main(argv, getter=getter, now=now)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# Window and step selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("15s", 15),
        ("90m", 5400),
        ("2h", 7200),
        ("1h30m", 5400),
        ("2m30s", 150),
        ("3600", 3600),
    ],
)
def test_parse_duration_handles_every_supported_unit(text, expected):
    assert gu.parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "2 h", "abc", "-5m", "1h30", "h"])
def test_parse_duration_rejects_unparseable_text(text):
    with pytest.raises(ValueError):
        gu.parse_duration(text)


def test_parse_timestamp_accepts_iso_zulu_and_unix_seconds():
    assert gu.parse_timestamp("2026-10-31T02:13:00Z") == START
    assert gu.parse_timestamp(str(END)) == END


def test_parse_timestamp_rejects_unparseable_text():
    with pytest.raises(ValueError):
        gu.parse_timestamp("yesterday")


def test_resolve_window_derives_bounds_from_a_duration_ending_now():
    window = gu.resolve_window(now=END, window="2h")

    assert (window.start, window.end) == (START, END)
    assert window.duration_seconds == 7200


def test_resolve_window_defaults_to_the_documented_two_hour_window():
    window = gu.resolve_window(now=END)

    assert (window.start, window.end) == (START, END)


def test_resolve_window_accepts_explicit_since_and_until():
    window = gu.resolve_window(
        now=END + 3600, since="2026-10-31T02:13:00Z", until=str(END)
    )

    assert (window.start, window.end) == (START, END)


def test_resolve_window_rejects_window_combined_with_absolute_bounds():
    with pytest.raises(ValueError, match="cannot be combined"):
        gu.resolve_window(now=END, window="2h", since=str(START))
    with pytest.raises(ValueError, match="cannot be combined"):
        gu.resolve_window(now=END, window="2h", until=str(END))


def test_resolve_window_rejects_until_without_since():
    with pytest.raises(ValueError, match="requires --since"):
        gu.resolve_window(now=END, until=str(END))


def test_resolve_window_rejects_an_empty_or_inverted_window():
    with pytest.raises(ValueError, match="after"):
        gu.resolve_window(now=END, since=str(END + 60), until=str(END))
    with pytest.raises(ValueError, match="after"):
        gu.resolve_window(now=END, since=str(END))


def test_expected_sample_count_covers_both_window_ends():
    assert gu.expected_sample_count(START, END, STEP) == EXPECTED
    # Longer than the store's resolution: one sample per minute.
    assert gu.expected_sample_count(0, 7200, 60) == 121
    # A window shorter than a single step still has to ask for one instant.
    assert gu.expected_sample_count(0, 5, 15) == 1


def test_a_window_shorter_than_one_step_still_reports_one_instant():
    series = _series(
        {
            "computing_workload_bool": {0: 1.0},
            "computing_bool": {0: 1.0},
            "power_watts": {0: 120.0},
            "slots_busy": {0: 1.0},
            "slot_heartbeat": {0: 0.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=5),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.expected_samples == 1
    assert result.computing_samples == 1
    assert result.computing_pct == pytest.approx(100.0)


def test_step_defaults_to_the_scrape_interval():
    assert gu.resolve_step(None, 15) == 15
    assert gu.resolve_step(None, 30) == 30


def test_step_larger_than_the_scrape_interval_is_allowed():
    assert gu.resolve_step("60s", 15) == 60


def test_step_finer_than_the_scrape_interval_is_rejected():
    with pytest.raises(ValueError, match="scrape interval"):
        gu.resolve_step("1s", 15)


def test_step_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        gu.resolve_step("0s", 0)


def test_scrape_interval_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        gu.resolve_scrape_interval("0s")


# ---------------------------------------------------------------------------
# Threshold and classification arithmetic
# ---------------------------------------------------------------------------


def test_power_threshold_is_inclusive_at_exactly_90_watts():
    assert gu.power_at_compute(90.0) is True
    assert gu.power_at_compute(89.999) is False
    assert gu.power_at_compute(90.001) is True


def test_workload_definition_needs_both_compute_power_and_a_busy_slot():
    assert gu.classify_instant(120.0, 1) is True
    assert gu.classify_instant(90.0, 1) is True  # boundary, inclusive
    assert gu.classify_instant(90.0, 0) is False  # power alone is not our work
    assert gu.classify_instant(40.0, 3) is False  # busy slots at idle power
    assert gu.classify_instant(89.9, 2) is False


def test_threshold_constant_matches_the_recording_rule():
    """The CLI constant is the rule's threshold, not a second definition."""
    rules = (PROJECT_ROOT / "monitoring" / "gpu_compute_rules.yaml").read_text()

    assert f">= bool {gu.THRESHOLD_WATTS:g}" in rules


# ---------------------------------------------------------------------------
# Response parsing and timestamp alignment
# ---------------------------------------------------------------------------


def test_parse_matrix_keeps_the_peak_across_multiple_series():
    payload = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {"metric": {}, "values": [[100, "1"], [115, "2"]]},
                {"metric": {}, "values": [[100, "3"], [130, "0"]]},
            ],
        },
    }

    assert gu.parse_matrix(payload) == {100: 3.0, 115: 2.0, 130: 0.0}


def test_parse_matrix_ignores_values_that_are_not_numbers():
    payload = {
        "status": "success",
        "data": {"result": [{"metric": {}, "values": [[100, "NaN-ish"]]}]},
    }

    assert gu.parse_matrix(payload) == {}


def test_align_to_grid_snaps_jitter_and_drops_out_of_window_samples():
    aligned = gu.align_to_grid(
        {START + 7: 1.0, START + 22: 2.0, END + 300: 9.0},
        start=START,
        step=STEP,
        expected=EXPECTED,
    )

    assert aligned == {START: 1.0, START + STEP: 2.0}


# ---------------------------------------------------------------------------
# Gap honesty: missing samples are unknown, never idle
# ---------------------------------------------------------------------------


def _series(points_by_key, window_start=0, step=STEP):
    return {key: dict(points) for key, points in points_by_key.items()}


def test_missing_samples_are_reported_as_unknown_not_idle():
    """Four instants, one computing, one idle, two unsampled."""
    series = _series(
        {
            "computing_workload_bool": {0: 0.0, 15: 1.0},
            "computing_bool": {0: 0.0, 15: 1.0},
            "power_watts": {0: 40.0, 15: 120.0},
            "slots_busy": {0: 0.0, 15: 2.0},
            "slot_heartbeat": {0: 0.0, 15: 15.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=45),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert (result.computing_samples, result.idle_samples, result.unknown_samples) == (
        1,
        1,
        2,
    )
    assert result.computing_pct == pytest.approx(25.0)
    assert result.idle_pct == pytest.approx(25.0)
    assert result.unknown_pct == pytest.approx(50.0)
    assert sum(
        (result.computing_pct, result.idle_pct, result.unknown_pct)
    ) == pytest.approx(100.0)


def test_gap_bundle_yields_the_documented_816_percent_unknown_share():
    """The recorded two-hour bundle with its 40-sample hole."""
    window = gu.Window(start=START, end=END)
    series = gu.load_series(
        getter=bundle_getter(),
        prometheus_url=gu.DEFAULT_PROMETHEUS_URL,
        window=window,
        step_seconds=STEP,
        expected_samples=EXPECTED,
    )

    result = gu.compute_utilisation(
        series,
        window=window,
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.expected_samples == EXPECTED
    assert result.observed_samples == OBSERVED
    assert result.unknown_samples == UNKNOWN
    assert result.computing_samples == COMPUTING
    assert result.idle_samples == IDLE
    assert result.unknown_pct == pytest.approx(100.0 * UNKNOWN / EXPECTED, abs=1e-3)
    assert sum(
        (result.computing_pct, result.idle_pct, result.unknown_pct)
    ) == pytest.approx(100.0)
    # The gap is never counted as idle.
    assert result.idle_samples != EXPECTED - result.computing_samples


def test_stale_heartbeat_makes_an_instant_unknown():
    """A heartbeat older than one scrape interval proves nothing."""
    series = _series(
        {
            "computing_workload_bool": {0: 0.0},
            "computing_bool": {0: 0.0},
            "power_watts": {0: 40.0},
            "slots_busy": {0: 0.0},
            "slot_heartbeat": {0: -100.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=15),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.unknown_samples == 2
    assert result.idle_samples == 0


def test_heartbeat_freshness_follows_the_configured_scrape_interval():
    """The same heartbeat is fresh at a 120 s bound and stale at 15 s."""
    series = _series(
        {
            "computing_workload_bool": {0: 0.0},
            "computing_bool": {0: 0.0},
            "slot_heartbeat": {0: -100.0},
        }
    )

    configured = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=0),
        step_seconds=STEP,
        scrape_interval_seconds=120,
    )
    default = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=0),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert (configured.idle_samples, configured.unknown_samples) == (1, 0)
    assert (default.idle_samples, default.unknown_samples) == (0, 1)


def test_a_down_target_does_not_prove_the_pipeline_was_alive():
    series = _series(
        {
            "computing_workload_bool": {0: 0.0},
            "computing_bool": {0: 0.0},
            "up_llama_proxy": {0: 0.0},
            "up_node_exporter": {0: 0.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=15),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.unknown_pct == pytest.approx(100.0)
    assert result.observed_samples == 0


def test_an_alive_pipeline_without_a_classifiable_sample_is_unknown():
    """``up`` proves the scrape happened, not that the GPU work is knowable."""
    series = _series(
        {
            "up_node_exporter": {0: 1.0},
            "power_watts": {0: 40.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=15),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.unknown_samples == 2
    assert result.idle_samples == 0


def test_all_unknown_window_reports_no_power_or_slot_figures():
    result = gu.compute_utilisation(
        {},
        window=gu.Window(start=0, end=30),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.observed_samples == 0
    assert result.unknown_pct == pytest.approx(100.0)
    assert result.power_mean_watts is None
    assert result.power_max_watts is None
    assert result.peak_busy_slots is None


def test_power_and_slot_figures_describe_the_observed_window():
    series = _series(
        {
            "computing_workload_bool": {0: 0.0, 15: 1.0},
            "computing_bool": {0: 0.0, 15: 1.0},
            "power_watts": {0: 40.0, 15: 120.0},
            "slots_busy": {0: 0.0, 15: 2.0},
            "slot_heartbeat": {0: 0.0, 15: 15.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=15),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.power_mean_watts == pytest.approx(80.0)
    assert result.power_max_watts == pytest.approx(120.0)
    assert result.peak_busy_slots == 2


def test_power_only_fraction_uses_the_same_honest_denominator():
    """``gpu:computing:bool`` counts any GPU consumer, over the same window."""
    series = _series(
        {
            "computing_workload_bool": {0: 0.0, 15: 0.0},
            "computing_bool": {0: 0.0, 15: 1.0},
            "slots_busy": {0: 0.0, 15: 0.0},
            "slot_heartbeat": {0: 0.0, 15: 15.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=15),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.computing_pct == pytest.approx(0.0)
    assert result.power_only_computing_samples == 1
    assert result.power_only_computing_pct == pytest.approx(50.0)


def test_cross_check_flags_a_recorded_rule_that_disagrees_with_the_recomputation():
    series = _series(
        {
            # Says "our work" while the GPU drew 40 W and no slot was busy.
            "computing_workload_bool": {0: 1.0},
            "computing_bool": {0: 0.0},
            "power_watts": {0: 40.0},
            "slots_busy": {0: 0.0},
            "slot_heartbeat": {0: 0.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=0),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.cross_check_samples == 1
    assert result.cross_check_mismatches == 1
    assert result.warnings


def test_cross_check_warning_blames_the_join_when_only_the_workload_rule_drifted():
    """AC5: the power half agrees, so the warning must not name the threshold."""
    series = _series(
        {
            "computing_workload_bool": {0: 1.0},  # says "our work" at 40 W
            "computing_bool": {0: 0.0},  # agrees: 40 W is below the threshold
            "power_watts": {0: 40.0},
            "slots_busy": {0: 0.0},
            "slot_heartbeat": {0: 0.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=0),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    warning = result.warnings[0]
    assert "gpu:computing_workload:bool" in warning
    assert "gpu:computing:bool" not in warning
    assert "THRESHOLD_WATTS" not in warning


def test_cross_check_ignores_a_recorded_rule_one_sample_behind_the_slots():
    """AC4: jitter, not drift — the recorded rule used the previous slot sample.

    ``llama_slots_busy`` steps to 1 at t=15 but the recording rule evaluated a
    moment earlier and saw 0, so at t=15 and t=45 the recorded
    ``gpu:computing_workload:bool`` is one sample behind the slot series. A
    same-instant comparison calls both a definition drift; the ±1-sample
    staleness window recognises them as timestamp jitter.
    """
    series = _series(
        {
            "computing_workload_bool": {0: 0.0, 15: 0.0, 30: 1.0, 45: 1.0},
            "computing_bool": {0: 1.0, 15: 1.0, 30: 1.0, 45: 1.0},
            "power_watts": {0: 120.0, 15: 120.0, 30: 120.0, 45: 120.0},
            "slots_busy": {0: 0.0, 15: 1.0, 30: 1.0, 45: 1.0},
            "slot_heartbeat": {0: 0.0, 15: 15.0, 30: 30.0, 45: 45.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=45),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.cross_check_samples == 4
    assert result.cross_check_mismatches == 0
    assert result.warnings == ()


def test_cross_check_ignores_a_recorded_power_rule_one_sample_behind_the_power():
    """The same staleness tolerance applies to the power input itself."""
    series = _series(
        {
            # Power steps 80 -> 120 W at t=15 and 30; the recorded rules at
            # t=15 saw the previous (80 W) sample.
            "computing_workload_bool": {0: 0.0, 15: 0.0, 30: 1.0},
            "computing_bool": {0: 0.0, 15: 0.0, 30: 1.0},
            "power_watts": {0: 80.0, 15: 120.0, 30: 120.0},
            "slots_busy": {0: 1.0, 15: 1.0, 30: 1.0},
            "slot_heartbeat": {0: 0.0, 15: 15.0, 30: 30.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=30),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.cross_check_samples == 3
    assert result.cross_check_mismatches == 0
    assert result.warnings == ()


def test_cross_check_still_fires_when_the_recorded_threshold_drifted():
    """AC3 mutation test: a rule recorded at 100 W must not pass a 90 W check.

    Every instant draws 95 W with one busy slot, so the recorded rule (written
    for a 100 W threshold) says 0 while the CLI's 90 W classification says 1.
    No staleness window can explain that, so the check must still fire — and it
    must name ``THRESHOLD_WATTS``, because the power half is what diverged.
    """
    series = _series(
        {
            "computing_workload_bool": {0: 0.0, 15: 0.0, 30: 0.0},
            "computing_bool": {0: 0.0, 15: 0.0, 30: 0.0},
            "power_watts": {0: 95.0, 15: 95.0, 30: 95.0},
            "slots_busy": {0: 1.0, 15: 1.0, 30: 1.0},
            "slot_heartbeat": {0: 0.0, 15: 15.0, 30: 30.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=30),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.cross_check_samples == 3
    assert result.cross_check_mismatches == 3
    warning = result.warnings[0]
    assert "THRESHOLD_WATTS" in warning
    assert "gpu:computing:bool" in warning


def test_cross_check_does_not_report_a_dead_sampler_as_rule_drift():
    """AC5: frozen inputs are a sampler outage, not a rule definition drift.

    After the pipeline dies every series returns the same last sample, so the
    recorded rule (frozen on the slot state it last saw) and the raw series
    (frozen on the next scrape) stay a sample apart forever. That must not
    produce a warning telling the operator to check the rules.
    """
    series = _series(
        {
            # The sampler stopped at t=15: the heartbeat repeats from t=30 and
            # the recorded rule froze on the slot state it last saw (1).
            "computing_workload_bool": {0: 1.0, 15: 1.0, 30: 1.0, 45: 1.0},
            "computing_bool": {0: 1.0, 15: 1.0, 30: 1.0, 45: 1.0},
            "power_watts": {0: 120.0, 15: 120.0, 30: 120.0, 45: 120.0},
            "slots_busy": {0: 1.0, 15: 1.0, 30: 0.0, 45: 0.0},
            "slot_heartbeat": {0: 0.0, 15: 15.0, 30: 15.0, 45: 15.0},
        }
    )

    result = gu.compute_utilisation(
        series,
        window=gu.Window(start=0, end=45),
        step_seconds=STEP,
        scrape_interval_seconds=STEP,
    )

    assert result.cross_check_samples == 3  # t=45 is frozen and skipped
    assert result.cross_check_mismatches == 0
    assert result.warnings == ()


def test_live_capture_of_the_worst_case_window_reports_no_rule_drift():
    """AC2/AC4: 2026-11-02 05:25-05:45, the measured jitter hotspot.

    Fifteen instants in this recorded slice are flagged by a naive
    same-instant comparison — the recording rules were evaluated on their own
    schedule against the neighbouring power/slot sample. The staleness-aware
    cross-check must report none of them, and it must still compare every
    instant in the slice (nothing is skipped by the frozen-input guard).
    """
    capture = JITTER_BUNDLE["capture"]
    window = gu.Window(
        start=capture["window"]["start_unix"], end=capture["window"]["end_unix"]
    )
    step = capture["step_seconds"]
    expected = gu.expected_sample_count(window.start, window.end, step)
    series = gu.load_series(
        getter=bundle_getter(JITTER_BUNDLE["responses"]),
        prometheus_url=gu.DEFAULT_PROMETHEUS_URL,
        window=window,
        step_seconds=step,
        expected_samples=expected,
    )

    naive_divergences = [
        timestamp
        for timestamp, workload in series["computing_workload_bool"].items()
        if timestamp in series["power_watts"]
        and timestamp in series["slots_busy"]
        and gu.classify_instant(
            series["power_watts"][timestamp], series["slots_busy"][timestamp]
        )
        != (workload >= 0.5)
    ]

    result = gu.compute_utilisation(
        series,
        window=window,
        step_seconds=step,
        scrape_interval_seconds=STEP,
    )

    # The slice must actually exercise the staleness path, or it proves nothing.
    assert len(naive_divergences) == 15
    assert result.cross_check_samples == expected == 81
    assert result.cross_check_mismatches == 0
    assert result.warnings == ()


def test_staleness_candidates_prefer_the_instant_then_its_neighbours():
    points = {0: 1.0, 15: 2.0, 30: 3.0}

    assert gu.staleness_candidates(points, 15, 15) == [2.0, 1.0, 3.0]
    assert gu.staleness_candidates(points, 0, 15) == [1.0, 2.0]
    assert gu.staleness_candidates({}, 0, 15) == []
    assert gu.staleness_candidates({0: 5.0, 15: 5.0}, 15, 15) == [5.0]


# ---------------------------------------------------------------------------
# CLI: queries, output, exit codes
# ---------------------------------------------------------------------------


def test_cli_issues_exactly_the_documented_queries(capsys):
    calls: list[str] = []

    code, _out, _err = run_cli(
        ["--since", str(START), "--until", str(END), "--json"],
        getter=bundle_getter(calls=calls),
        capsys=capsys,
    )

    assert code == gu.EXIT_OK
    assert calls == list(gu.QUERIES.values())
    assert "gpu_busy_percent" not in " ".join(calls)


def test_the_cli_is_the_single_producer_of_the_attributable_share():
    """C6 / epic AC2 (LP-0MW5K90EQ001W0R1): the CLI *is* the producer.

    It queries the attributable rule series itself and does the grid-honest
    share arithmetic locally, so no PromQL aggregation over
    ``gpu:computing_workload:bool`` may appear in its query mapping — that is
    what a second derivation definition would look like. The one-producer guard
    in ``test_monitoring_deployment.py`` enforces the other side (no rule,
    report or dashboard may aggregate it).
    """
    assert gu.QUERIES["computing_workload_bool"] == "gpu:computing_workload:bool"
    for key, expression in gu.QUERIES.items():
        assert "(" not in expression, (
            f"QUERIES[{key!r}] = {expression!r} is a PromQL expression, not a "
            "bare series selector; the CLI computes the share from the raw "
            "series locally"
        )


def test_human_output_names_the_scope_of_the_power_only_derivation(capsys):
    """AC1: a figure's name states its scope — the human output keeps the
    'any GPU consumer, not just ours' label on the power-only derivation."""
    code, out, _err = run_cli(
        ["--since", str(START), "--until", str(END)],
        getter=bundle_getter(),
        capsys=capsys,
    )

    assert code == gu.EXIT_OK
    assert "(gpu:computing:bool: any GPU consumer, not just ours)" in out


def test_json_output_carries_the_documented_schema(capsys):
    code, out, _err = run_cli(
        ["--since", str(START), "--until", str(END), "--json"],
        getter=bundle_getter(),
        capsys=capsys,
    )

    assert code == gu.EXIT_OK
    payload = json.loads(out)

    assert payload["window"]["start_unix"] == START
    assert payload["window"]["end_unix"] == END
    assert payload["window"]["duration_seconds"] == 7200
    assert payload["resolution"]["step_seconds"] == STEP
    assert payload["resolution"]["scrape_interval_seconds"] == 15
    assert payload["resolution"]["expected_samples"] == EXPECTED
    assert payload["coverage"] == {
        "expected_samples": EXPECTED,
        "observed_samples": OBSERVED,
        "computing_samples": COMPUTING,
        "idle_samples": IDLE,
        "unknown_samples": UNKNOWN,
    }
    assert payload["shares_pct"]["computing"] == pytest.approx(
        100.0 * COMPUTING / EXPECTED, abs=1e-3
    )
    assert payload["shares_pct"]["unknown"] == pytest.approx(
        100.0 * UNKNOWN / EXPECTED, abs=1e-3
    )
    assert sum(
        (
            payload["shares_pct"]["computing"],
            payload["shares_pct"]["idle"],
            payload["shares_pct"]["unknown"],
        )
    ) == pytest.approx(100.0)
    assert payload["power_watts"]["samples"] == OBSERVED
    assert payload["power_watts"]["max"] == pytest.approx(111.075, abs=1e-3)
    assert payload["slots"]["peak_busy"] == 2
    assert payload["rule_cross_check"] == {"compared_samples": OBSERVED, "mismatches": 0}


def test_human_output_states_shares_power_slots_and_resolution(capsys):
    code, out, _err = run_cli(
        ["--since", str(START), "--until", str(END)],
        getter=bundle_getter(),
        capsys=capsys,
    )

    assert code == gu.EXIT_OK
    assert "2026-10-31T02:13:00Z" in out
    assert "2026-10-31T04:13:00Z" in out
    assert "computing 14.1%" in out
    assert "idle 77.5%" in out
    assert "unknown 8.3%" in out
    assert "power mean 50 W" in out
    assert "max 111 W" in out
    assert "peak 2 busy slots" in out
    assert "step 15s" in out
    assert "scrape floor 15s" in out
    assert "481 expected samples" in out


def test_cli_reports_a_rule_drift_on_stderr_and_keeps_the_stable_json_keys(capsys):
    """AC6: a firing cross-check keeps ``compared_samples`` / ``mismatches``.

    The rules were recorded for a 100 W threshold, the CLI classifies at 90 W,
    so every 95 W instant diverges. The JSON shape must not change and the
    diagnosis must reach stderr, where an operator (or cron) will see it.
    """
    power = {0: 95.0, 15: 95.0, 30: 95.0}
    slots = {0: 1.0, 15: 1.0, 30: 1.0}
    points_by_query = {
        gu.QUERIES["power_watts"]: power,
        gu.QUERIES["slots_busy"]: slots,
        gu.QUERIES["computing_bool"]: {0: 0.0, 15: 0.0, 30: 0.0},
        gu.QUERIES["computing_workload_bool"]: {0: 0.0, 15: 0.0, 30: 0.0},
        gu.QUERIES["slot_heartbeat"]: {0: 0.0, 15: 15.0, 30: 30.0},
        gu.QUERIES["up_llama_proxy"]: {0: 1.0, 15: 1.0, 30: 1.0},
    }

    code, out, err = run_cli(
        ["--since", "0", "--until", "30", "--json"],
        getter=querying_getter(points_by_query),
        capsys=capsys,
    )

    assert code == gu.EXIT_OK
    payload = json.loads(out)
    assert payload["schema_version"] == 1
    assert payload["rule_cross_check"] == {"compared_samples": 3, "mismatches": 3}
    assert payload["warnings"]
    assert "THRESHOLD_WATTS" in err


def test_cli_reports_a_rule_drift_without_the_threshold_advice_for_a_join_drift(
    capsys,
):
    """AC5 at the CLI boundary: the workload-rule warning never names the threshold."""
    points_by_query = {
        gu.QUERIES["power_watts"]: {0: 40.0, 15: 40.0, 30: 40.0},
        gu.QUERIES["slots_busy"]: {0: 0.0, 15: 0.0, 30: 0.0},
        gu.QUERIES["computing_bool"]: {0: 0.0, 15: 0.0, 30: 0.0},
        gu.QUERIES["computing_workload_bool"]: {0: 1.0, 15: 1.0, 30: 1.0},
        gu.QUERIES["slot_heartbeat"]: {0: 0.0, 15: 15.0, 30: 30.0},
        gu.QUERIES["up_llama_proxy"]: {0: 1.0, 15: 1.0, 30: 1.0},
    }

    code, _out, err = run_cli(
        ["--since", "0", "--until", "30"],
        getter=querying_getter(points_by_query),
        capsys=capsys,
    )

    assert code == gu.EXIT_OK
    assert "gpu:computing_workload:bool" in err
    assert "THRESHOLD_WATTS" not in err


def test_a_coarser_step_is_queried_and_reported(capsys):
    code, out, _err = run_cli(
        ["--since", str(START), "--until", str(END), "--step", "60s", "--json"],
        getter=bundle_getter(),
        capsys=capsys,
    )

    assert code == gu.EXIT_OK
    payload = json.loads(out)
    assert payload["resolution"]["step_seconds"] == 60
    assert payload["resolution"]["step"] == "1m"
    assert payload["resolution"]["expected_samples"] == 121


def test_all_unknown_window_exits_non_zero_with_a_clear_message(capsys):
    code, out, err = run_cli(
        ["--since", str(START), "--until", str(END), "--json"],
        getter=querying_getter({}),
        capsys=capsys,
    )

    assert code == gu.EXIT_NO_DATA
    assert "no observed samples" in err
    payload = json.loads(out)
    assert payload["shares_pct"]["unknown"] == pytest.approx(100.0)


def test_unparseable_duration_exits_with_usage_error(capsys):
    code, out, err = run_cli(["--window", "banana"], getter=bundle_getter(), capsys=capsys)

    assert code == gu.EXIT_USAGE
    assert out == ""
    assert "banana" in err


def test_window_combined_with_since_exits_with_usage_error(capsys):
    code, _out, err = run_cli(
        ["--window", "2h", "--since", str(START)],
        getter=bundle_getter(),
        capsys=capsys,
    )

    assert code == gu.EXIT_USAGE
    assert "cannot be combined" in err


def test_step_finer_than_the_scrape_interval_exits_with_usage_error(capsys):
    code, _out, err = run_cli(
        ["--since", str(START), "--until", str(END), "--step", "1s"],
        getter=bundle_getter(),
        capsys=capsys,
    )

    assert code == gu.EXIT_USAGE
    assert "scrape interval" in err


def test_unreachable_prometheus_url_exits_with_an_actionable_message(capsys):
    url = closed_port_url()

    code, _out, err = run_cli(
        ["--since", str(START), "--until", str(END), "--prometheus-url", url],
        capsys=capsys,
    )

    assert code == gu.EXIT_PROMETHEUS_UNAVAILABLE
    assert url in err


def test_prometheus_error_response_is_reported(capsys):
    with prometheus_stub(BUNDLE["responses"], status=500) as (url, _requests):
        code, _out, err = run_cli(
            ["--since", str(START), "--until", str(END), "--prometheus-url", url],
            capsys=capsys,
        )

    assert code == gu.EXIT_PROMETHEUS_UNAVAILABLE
    assert "500" in err


def test_cli_measures_without_any_proxy_log(tmp_path, monkeypatch, capsys):
    """The figure is a direct store measurement, not a log inference.

    Served over real HTTP from recorded payloads, in a working directory that
    contains no ``proxy.log`` at all.
    """
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "proxy.log").exists()

    with prometheus_stub(BUNDLE["responses"]) as (url, requests_seen):
        code, out, _err = run_cli(
            ["--since", str(START), "--until", str(END), "--prometheus-url", url, "--json"],
            capsys=capsys,
        )

    assert code == gu.EXIT_OK
    assert not (tmp_path / "proxy.log").exists()
    assert len(requests_seen) == len(gu.QUERIES)
    payload = json.loads(out)
    assert payload["coverage"]["unknown_samples"] == UNKNOWN
    assert payload["coverage"]["computing_samples"] == COMPUTING
    assert payload["prometheus_url"] == url
