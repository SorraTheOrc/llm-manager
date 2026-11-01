#!/usr/bin/env python3
"""Contract tests for the recorded Prometheus payload fixtures.

LP-0MVT1NVAJ007KUGO (parent LP-0MVNWUEY8008XM6W). The fixtures under
``monitoring/tests/fixtures/`` are the offline stand-in for a live Prometheus
that the `gpu_utilisation` CLI (LP-0MVT1O52G000LMC6) is tested against. This
module pins the *shape* every consumer may rely on:

- the fixture envelope documents the endpoint, query and window it came from
  (an unlabelled blob is unusable six months later);
- every stream is a Prometheus ``query_range`` matrix with numeric
  string-encoded values and strictly increasing timestamps;
- the deliberate gap is real, exactly 40 samples wide across every authored
  stream, and therefore yields the documented 8.316 % unknown share;
- the two ``up`` sentinels and the power series carry *different* ``instance``
  labels (the label-matching trap the recording-rule child must respect).

No network access, no live Prometheus: these tests only read the files.
"""

import json
from datetime import UTC, datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "monitoring" / "tests" / "fixtures"

STEP_SECONDS = 15
WINDOW_SECONDS = 7200
#: 481 inclusive 15 s samples across a two-hour window.
EXPECTED_SAMPLES = WINDOW_SECONDS // STEP_SECONDS + 1
#: The deliberate 10-minute hole authored into every gapped stream.
GAP_SAMPLES = 60 * 10 // STEP_SECONDS  # 40
EXPECTED_UNKNOWN_SHARE = GAP_SAMPLES / EXPECTED_SAMPLES  # 0.08316...

#: Files that must each document one query and hold a real-shaped response.
SINGLE_QUERY_FIXTURES = {
    "gpu_computing_bool.json": "gpu:computing:bool",
    "gpu_computing_workload_bool.json": "gpu:computing_workload:bool",
    "gpu_power_watts_amdgpu.json": "gpu:power_watts:amdgpu",
    "llama_slots_busy.json": "llama_slots_busy",
    "llama_slots_sample_timestamp_seconds.json": "llama_slots_sample_timestamp_seconds",
    'up_llama_proxy.json': 'up{job="llama-proxy"}',
    'up_node_exporter.json': 'up{job="node-exporter"}',
}

#: The three authored streams share the deliberate hole; the live captures do
#: not, and the clipper in ``window_with_gap.json`` does not either.
GAPPED_FIXTURES = (
    "gpu_computing_workload_bool.json",
    "llama_slots_busy.json",
    "llama_slots_sample_timestamp_seconds.json",
)

#: Queries the CLI issues for one window, all present in the bundle.
BUNDLE_QUERIES = (
    "gpu:computing:bool",
    "gpu:computing_workload:bool",
    "gpu:power_watts:amdgpu",
    "llama_slots_busy",
    "llama_slots_sample_timestamp_seconds",
    'up{job="llama-proxy"}',
    'up{job="node-exporter"}',
)


def _load(name: str) -> dict:
    path = FIXTURES / name
    assert path.exists(), f"missing fixture {path}"
    return json.loads(path.read_text())


def _values(response: dict) -> list[list[str]]:
    """Return the single series' values from a matrix response."""
    result = response["data"]["result"]
    assert len(result) == 1, f"expected one series, got {len(result)}"
    return result[0]["values"]


def _timestamps(values: list[list[str]]) -> list[int]:
    return [int(float(ts)) for ts, _ in values]


def _gap_size(values: list[list[str]]) -> int:
    """Number of missing 15 s samples inside the window."""
    stamps = _timestamps(values)
    span = (stamps[-1] - stamps[0]) // STEP_SECONDS + 1
    return span - len(stamps)


# ---------------------------------------------------------------------------
# Envelope: every fixture documents how it was captured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SINGLE_QUERY_FIXTURES))
def test_fixture_documents_endpoint_query_and_window(name):
    """An undocumented payload is unusable: endpoint, query and window required."""
    capture = _load(name)["capture"]

    assert capture["endpoint"].endswith("/api/v1/query_range")
    assert capture["query"] == SINGLE_QUERY_FIXTURES[name]
    window = capture["window"]
    assert datetime.strptime(window["start"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    )
    assert datetime.strptime(window["end"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    )
    assert window["end_unix"] - window["start_unix"] == WINDOW_SECONDS
    assert capture["step_seconds"] == STEP_SECONDS
    assert capture["captured_at"]
    # Provenance must be explicit, so nobody mistakes an authored series for a
    # recorded hardware measurement.
    assert capture["source"] in {
        "captured live from the Prometheus HTTP API on this host "
        "(scrape_interval 15s, evaluation_interval 15s)",
        "authored",
    }
    assert capture["notes"].strip(), f"{name} must explain how values were produced"


@pytest.mark.parametrize("name", sorted(SINGLE_QUERY_FIXTURES))
def test_fixture_covers_the_full_documented_window(name):
    """Start/end timestamps match the documented window exactly."""
    data = _load(name)
    stamps = _timestamps(_values(data["response"]))
    window = data["capture"]["window"]

    assert stamps[0] == window["start_unix"]
    assert stamps[-1] == window["end_unix"]


@pytest.mark.parametrize("name", sorted(SINGLE_QUERY_FIXTURES))
def test_fixture_is_a_prometheus_matrix_with_numeric_values(name):
    """Consumer-visible shape: matrix result, string-encoded floats, ordered ts."""
    response = _load(name)["response"]

    assert response["status"] == "success"
    assert response["data"]["resultType"] == "matrix"
    stamps = _timestamps(_values(response))
    assert stamps == sorted(stamps), "timestamps must be strictly increasing"
    assert len(set(stamps)) == len(stamps), "duplicate timestamps"
    for ts, value in _values(response):
        float(value)  # Prometheus returns values as strings; must parse
        assert float(ts).is_integer()
    assert _values(response), f"{name} has no samples"


# ---------------------------------------------------------------------------
# Values: what each stream must be able to express
# ---------------------------------------------------------------------------


def test_computing_stream_is_binary():
    """``gpu:computing:bool`` is 0/1 — the CLI's power-only definition."""
    values = _values(_load("gpu_computing_bool.json")["response"])

    assert {float(v) for _, v in values} == {0.0, 1.0}
    # The fixture must exercise both branches, otherwise a CLI that ignores the
    # bound would still pass.
    assert any(float(v) == 1.0 for _, v in values)
    assert any(float(v) == 0.0 for _, v in values)


def test_workload_stream_is_binary_and_absent_in_the_gap():
    """``gpu:computing_workload:bool`` is 0/1 and never fabricates a 0 in a gap."""
    data = _load("gpu_computing_workload_bool.json")
    values = _values(data["response"])

    assert {float(v) for _, v in values} <= {0.0, 1.0}
    assert _gap_size(values) == GAP_SAMPLES


def test_power_stream_reaches_and_crosses_the_90w_threshold():
    """The power series must contain samples on both sides of 90 W."""
    values = [float(v) for _, v in _values(_load("gpu_power_watts_amdgpu.json")["response"])]

    assert min(values) < 90.0
    assert max(values) >= 90.0


def test_slot_stream_exposes_both_idle_and_busy_occupancy():
    """``llama_slots_busy`` must be able to express 0 (idle) and >0 (busy)."""
    values = [float(v) for _, v in _values(_load("llama_slots_busy.json")["response"])]

    assert 0.0 in values, "an idle sample is required to test the idle share"
    assert max(values) >= 1.0, "a busy sample is required to test peak concurrency"
    assert max(values) >= 2.0, "peak concurrency needs a >1 sample to be meaningful"


def test_slot_heartbeat_is_the_sample_timestamp():
    """The heartbeat proves liveness; its value is the instant it was sampled."""
    values = _values(_load("llama_slots_sample_timestamp_seconds.json")["response"])

    for ts, value in values:
        assert float(value) == float(ts)


def test_up_sentinels_are_1_and_carry_their_own_job_label():
    """``up`` is 1 for a scrapeable target and labelled with the scrape job."""
    for name, job in (
        ("up_llama_proxy.json", "llama-proxy"),
        ("up_node_exporter.json", "node-exporter"),
    ):
        response = _load(name)["response"]
        series = response["data"]["result"][0]
        assert series["metric"]["job"] == job
        assert {float(v) for _, v in series["values"]} == {1.0}


def test_power_and_up_sentinels_carry_different_instance_labels():
    """The label-matching trap: power and slots live on different instances.

    A ``on(instance)`` join between the node_exporter power series and a
    proxy-published slot series matches nothing. The fixture must keep the two
    ``instance`` values distinct so the recording-rule child cannot pass with a
    label-blind test.
    """
    power = _load("gpu_power_watts_amdgpu.json")["response"]["data"]["result"][0]
    proxy_up = _load("up_llama_proxy.json")["response"]["data"]["result"][0]

    assert power["metric"]["instance"] == "localhost:9100"
    assert power["metric"]["job"] == "node-exporter"
    assert proxy_up["metric"]["instance"] == "localhost:8000"
    assert proxy_up["metric"]["instance"] != power["metric"]["instance"]


# ---------------------------------------------------------------------------
# The deliberate gap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GAPPED_FIXTURES)
def test_authored_stream_carries_the_documented_10_minute_gap(name):
    """Every authored stream has the same 40-sample hole (600 s at 15 s)."""
    values = _values(_load(name)["response"])

    assert _gap_size(values) == GAP_SAMPLES
    stamps = _timestamps(values)
    biggest_step = max(b - a for a, b in zip(stamps, stamps[1:]))
    assert biggest_step == (GAP_SAMPLES + 1) * STEP_SECONDS


@pytest.mark.parametrize("name", sorted(set(SINGLE_QUERY_FIXTURES) - set(GAPPED_FIXTURES)))
def test_live_captures_have_no_gap(name):
    """The recorded live series are continuous, so the gap is unambiguously ours."""
    assert _gap_size(_values(_load(name)["response"])) == 0


def test_gap_unknown_share_is_the_documented_figure():
    """40 missing of 481 expected samples is 8.316 % — the CLI's asserted split."""
    values = _values(_load("llama_slots_busy.json")["response"])

    present = len(values)
    unknown = _gap_size(values)

    assert present + unknown == EXPECTED_SAMPLES
    assert unknown / EXPECTED_SAMPLES == pytest.approx(0.08316, abs=5e-5)


# ---------------------------------------------------------------------------
# The combined bundle the CLI consumes for one window
# ---------------------------------------------------------------------------


def test_bundle_serves_every_query_the_cli_issues():
    bundle = _load("window_with_gap.json")

    assert set(bundle["responses"]) == set(BUNDLE_QUERIES)
    for query, response in bundle["responses"].items():
        assert response["status"] == "success", query
        assert response["data"]["resultType"] == "matrix", query


def test_bundle_documents_its_window_and_gap():
    capture = _load("window_with_gap.json")["capture"]

    assert capture["endpoint"].endswith("/api/v1/query_range")
    assert capture["window"]["end_unix"] - capture["window"]["start_unix"] == WINDOW_SECONDS
    assert capture["step_seconds"] == STEP_SECONDS
    assert str(GAP_SAMPLES) in capture["notes"] or "10-minute" in capture["notes"]


def test_bundle_gap_covers_every_stream_so_unknown_is_honest():
    """All streams share the hole: that models "nobody was looking", not idle.

    If only the slot series had a hole while ``up`` kept reporting, the covered
    share would be 100 % and the gap would have to be attributed to *something*
    (idle or computing) rather than reported as unknown.
    """
    responses = _load("window_with_gap.json")["responses"]

    for query, response in responses.items():
        values = response["data"]["result"][0]["values"]
        assert _gap_size(values) == GAP_SAMPLES, f"{query} gap differs"


def test_bundle_slot_occupancy_matches_the_workload_definition():
    """Outside the gap, workload=1 exactly where power ≥ 90 W and a slot is busy.

    This is the definition the parent brief requires to be expressed once (as a
    recording rule) and shared with the CLI; the fixture is the executable
    reference the CLI test compares against.
    """
    responses = _load("window_with_gap.json")["responses"]

    power = {
        int(float(ts)): float(v)
        for ts, v in responses["gpu:power_watts:amdgpu"]["data"]["result"][0]["values"]
    }
    busy = {
        int(float(ts)): float(v)
        for ts, v in responses["llama_slots_busy"]["data"]["result"][0]["values"]
    }
    workload = {
        int(float(ts)): float(v)
        for ts, v in responses["gpu:computing_workload:bool"]["data"]["result"][0]["values"]
    }

    assert set(power) == set(busy) == set(workload)
    for ts, recorded in workload.items():
        expected = 1.0 if power[ts] >= 90.0 and busy[ts] >= 1.0 else 0.0
        assert recorded == expected, f"t={ts} power={power[ts]} busy={busy[ts]}"


def test_bundle_contains_a_computing_and_an_idle_stretch():
    """The CLI test needs both outcomes in the same stream to be meaningful."""
    values = [
        float(v)
        for _, v in _load("window_with_gap.json")["responses"][
            "gpu:computing_workload:bool"
        ]["data"]["result"][0]["values"]
    ]

    assert 1.0 in values and 0.0 in values
