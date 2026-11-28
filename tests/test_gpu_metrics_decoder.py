#!/usr/bin/env python3
"""Tests for the pinned ``gpu_metrics`` v3.0 decoder.

``scripts/lib/gpu_metrics_decoder.py`` turns the 264-byte
``/sys/class/drm/card0/device/gpu_metrics`` blob into named fields. Because the
structure is not shipped in this host's ``linux-headers``, every offset is
pinned in-repo, so these tests exist to make a transcription slip or a silent
kernel/firmware layout change fail loudly instead of misparsing live data
(LP-0MVZEW6K6007BPY0, parent LP-0MVSQT7YC0045HFW).

Coverage:

* the committed real capture decodes to the recorded values, and its
  ``average_socket_power`` cross-checks the hwmon ``power1_average`` value
  recorded alongside it to within 1 % (AC4, AC5),
* the pinned table is internally consistent with the ``#pragma pack(4)``
  geometry (offsets aligned, only the documented interior gaps, members end at
  260, 4 bytes of tail padding) so a typo cannot silently shift every field,
* truncated, empty and oversized buffers are rejected with the observed and
  expected sizes (AC3),
* unknown ``format_revision``, ``structure_size`` and ``content_revision``
  values are rejected with the observed and expected values (AC2),
* the four tail-padding bytes are never decoded, and semantic absurdity inside
  the field region is *not* second-guessed (fail closed on layout, not values).

Capture provenance and re-capture recipe
----------------------------------------
The fixture is one real read of ``/sys/class/drm/card0/device/gpu_metrics``
paired with a read of the amdgpu hwmon ``power1_average`` within the same
millisecond; ``gpu_metrics_v3_0.meta.json`` records the capture timestamp,
kernel, SHA-256, header values and the hwmon reading in microwatts. To
re-capture after a kernel, firmware or driver change (then update both files):

.. code-block:: shell

    python3 - <<'PY'
    from pathlib import Path
    Path("tests/fixtures/gpu_metrics_v3_0.bin").write_bytes(
        Path("/sys/class/drm/card0/device/gpu_metrics").read_bytes()
    )
    PY

The paired hwmon value lives in the sidecar rather than in this file because
hwmon is live: the cross-check needs the reading taken *with* the capture. The
intake-observed bands the fixture must still satisfy (42–123 W socket power,
2 711–2 900 MHz GFX clock) are asserted below, so a replacement capture from a
different device is caught rather than silently redefining the layout.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import gpu_metrics_decoder as gmd  # noqa: E402
from lib.gpu_metrics_decoder import (  # noqa: E402
    CONTENT_REVISION,
    FIELD_REGION_BYTES,
    FIELD_TABLE,
    FORMAT_REVISION,
    HEADER_FIELD_NAMES,
    HEADER_SIZE_BYTES,
    STRUCTURE_SIZE_BYTES,
    TAIL_PADDING_BYTES,
    GpuMetricsError,
    GpuMetricsHeaderError,
    GpuMetricsSizeError,
    decode,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "gpu_metrics_v3_0.bin"
META_PATH = FIXTURE_DIR / "gpu_metrics_v3_0.meta.json"

# Values decoded from the committed capture (2026-11-18T18:52:30Z, kernel
# 7.0.0-38-generic). Asserting the whole record, not a sample of it, is the
# point: any offset shift breaks at least one member. Array members are tuples.
EXPECTED_FIELDS: dict[str, int | tuple[int, ...]] = {
    "temperature_gfx": 6675,
    "temperature_soc": 5925,
    "temperature_core": (0,) * 16,
    "temperature_skin": 0,
    "average_gfx_activity": 100,
    "average_vcn_activity": 0,
    "average_ipu_activity": (0,) * 8,
    "average_core_c0_activity": (3, 1, 0, 0, 2, 21, 4, 7, 0, 1, 1, 1, 0, 0, 9, 2),
    "average_dram_reads": 593,
    "average_dram_writes": 275,
    "average_ipu_reads": 0,
    "average_ipu_writes": 0,
    "system_clock_counter": 1571074192854016,
    "average_socket_power": 42111,
    "average_ipu_power": 0,
    "average_apu_power": 42111,
    "average_gfx_power": 11878,
    "average_dgpu_power": 0,
    "average_all_core_power": 1383,
    "average_core_power": (11, 11, 11, 12, 29, 4873, 11, 371, 14, 14, 14, 14, 14, 14, 14, 14),
    "average_sys_power": 63541,
    "stapm_power_limit": 65535,
    "current_stapm_power_limit": 65535,
    "average_gfxclk_frequency": 2900,
    "average_socclk_frequency": 1041,
    "average_vpeclk_frequency": 800,
    "average_ipuclk_frequency": 0,
    "average_fclk_frequency": 1446,
    "average_vclk_frequency": 800,
    "average_uclk_frequency": 922,
    "average_mpipu_frequency": 0,
    "current_coreclk": (1782, 1782, 1782, 1782, 1782, 3565, 1782, 1782, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000),
    "current_core_maxfreq": 3569,
    "current_gfx_maxfreq": 2900,
    "throttle_residency_prochot": 0,
    "throttle_residency_spl": 8,
    "throttle_residency_fppt": 75481927,
    "throttle_residency_sppt": 3885104,
    "throttle_residency_thm_core": 977847959,
    "throttle_residency_thm_gfx": 246613473,
    "throttle_residency_thm_soc": 0,
    "time_filter_alphavalue": 1000000,
}


@pytest.fixture(scope="module")
def fixture_bytes() -> bytes:
    return FIXTURE_PATH.read_bytes()


@pytest.fixture(scope="module")
def fixture_meta() -> dict:
    return json.loads(META_PATH.read_text())


@pytest.fixture(scope="module")
def decoded(fixture_bytes: bytes) -> dict[str, int | tuple[int, ...]]:
    return decode(fixture_bytes)


# --- The committed capture -------------------------------------------------


def test_fixture_is_one_real_264_byte_capture(fixture_bytes: bytes, fixture_meta: dict) -> None:
    assert len(fixture_bytes) == STRUCTURE_SIZE_BYTES == 264
    assert hashlib.sha256(fixture_bytes).hexdigest() == fixture_meta["sha256"]
    assert fixture_meta["structure_size_bytes"] == len(fixture_bytes)
    assert fixture_meta["source_path"] == "/sys/class/drm/card0/device/gpu_metrics"
    # Provenance travels with the fixture so a decode is reproducible.
    assert fixture_meta["captured_at"]
    assert fixture_meta["host_kernel"]
    assert fixture_meta["header"] == {
        "structure_size": STRUCTURE_SIZE_BYTES,
        "format_revision": FORMAT_REVISION,
        "content_revision": CONTENT_REVISION,
    }


def test_fixture_decodes_to_the_recorded_values(decoded: dict, fixture_meta: dict) -> None:
    assert decoded["structure_size"] == fixture_meta["header"]["structure_size"]
    assert decoded["format_revision"] == fixture_meta["header"]["format_revision"]
    assert decoded["content_revision"] == fixture_meta["header"]["content_revision"]
    assert {name: decoded[name] for name in EXPECTED_FIELDS} == EXPECTED_FIELDS


def test_fixture_socket_power_cross_checks_hwmon_within_one_percent(decoded: dict, fixture_meta: dict) -> None:
    hwmon_watts = fixture_meta["hwmon"]["power1_average_uw"] / 1_000_000
    socket_watts = decoded["average_socket_power"] / 1000
    deviation = abs(socket_watts - hwmon_watts) / hwmon_watts
    assert deviation <= 0.01, (
        f"average_socket_power {socket_watts:.3f} W vs hwmon power1_average "
        f"{hwmon_watts:.3f} W: {deviation:.2%} apart"
    )


def test_fixture_is_inside_the_intake_observation_bands(decoded: dict) -> None:
    """The fixture must be this host's record, not a relabelled strangers' blob."""
    # Intake (2026-10-29) observed socket power 42-123 W and GFX clock
    # 2711-2900 MHz across idle and compute bands on this APU.
    assert 42_000 <= decoded["average_socket_power"] <= 123_000
    assert 2711 <= decoded["average_gfxclk_frequency"] <= 2900
    # The values are SMU time-filtered, so no claim finer than the sample holds.
    assert decoded["time_filter_alphavalue"] == 1_000_000


def test_tail_padding_bytes_are_recorded_and_never_decoded(fixture_bytes: bytes, fixture_meta: dict, decoded: dict) -> None:
    tail = fixture_bytes[FIELD_REGION_BYTES:]
    assert len(tail) == TAIL_PADDING_BYTES == 4
    assert tail.hex() == fixture_meta["tail_padding_hex"]
    # Rewriting the padding must not change a single decoded value, which is
    # what "padding" means: those four bytes carry no member.
    mutated = bytearray(fixture_bytes)
    mutated[FIELD_REGION_BYTES:] = b"\x00\x00\x00\x00"
    assert decode(mutated) == decoded
    assert set(decoded) & {"tail_padding"} == set()


# --- The pinned field table ------------------------------------------------


def test_field_table_matches_the_pack4_geometry() -> None:
    assert HEADER_SIZE_BYTES == 4
    assert len({field.name for field in FIELD_TABLE}) == len(FIELD_TABLE)
    assert set(HEADER_FIELD_NAMES) & {field.name for field in FIELD_TABLE} == set()

    previous_end = HEADER_SIZE_BYTES
    gaps: list[int] = []
    for field in FIELD_TABLE:
        assert field.offset >= previous_end, f"{field.name} overlaps the previous member"
        assert field.offset % min(field.element_size, 4) == 0, f"{field.name} is misaligned for pack(4)"
        assert field.end <= FIELD_REGION_BYTES, f"{field.name} runs past the field region"
        gap = field.offset - previous_end
        assert gap in (0, 2, 4), f"{field.name} has an unexpected {gap}-byte gap"
        if gap:
            gaps.append(gap)
        previous_end = field.end

    # The members end exactly at the documented field-region boundary; the
    # only interior gaps are the three 2-byte pack(4) alignments.
    assert previous_end == FIELD_REGION_BYTES == 260
    assert gaps == [2, 2, 2]
    assert STRUCTURE_SIZE_BYTES - FIELD_REGION_BYTES == TAIL_PADDING_BYTES == 4


def test_decoded_keys_are_the_header_then_every_member_in_order(decoded: dict) -> None:
    assert list(decoded) == list(HEADER_FIELD_NAMES) + [field.name for field in FIELD_TABLE]


def test_arrays_decode_as_tuples_and_scalars_as_ints(decoded: dict) -> None:
    for field in FIELD_TABLE:
        value = decoded[field.name]
        if field.is_array:
            assert isinstance(value, tuple), field.name
            assert len(value) == field.count, field.name
            assert all(isinstance(item, int) for item in value), field.name
        else:
            assert isinstance(value, int), field.name


# --- Size rejection --------------------------------------------------------


def test_truncated_buffer_is_rejected(fixture_bytes: bytes) -> None:
    for size in (STRUCTURE_SIZE_BYTES - 1, 100, HEADER_SIZE_BYTES):
        with pytest.raises(GpuMetricsSizeError) as excinfo:
            decode(fixture_bytes[:size])
        error = excinfo.value
        assert error.field == "structure_size"
        assert error.observed == size
        assert error.expected == STRUCTURE_SIZE_BYTES
        message = str(error)
        assert "truncated" in message
        assert str(size) in message
        assert str(STRUCTURE_SIZE_BYTES) in message


def test_zero_length_buffer_is_rejected() -> None:
    with pytest.raises(GpuMetricsSizeError) as excinfo:
        decode(b"")
    error = excinfo.value
    assert error.observed == 0
    assert error.expected == STRUCTURE_SIZE_BYTES
    assert "empty" in str(error)


def test_oversized_buffer_is_rejected(fixture_bytes: bytes) -> None:
    for extra in (b"\x00", b"\x00" * 3832):
        blob = fixture_bytes + extra
        with pytest.raises(GpuMetricsSizeError) as excinfo:
            decode(blob)
        error = excinfo.value
        assert error.observed == len(blob)
        assert error.expected == STRUCTURE_SIZE_BYTES
        assert "oversized" in str(error)


# --- Header rejection ------------------------------------------------------


@pytest.mark.parametrize("observed", [0, 1, 2, 4, 255])
def test_unknown_format_revision_is_rejected(fixture_bytes: bytes, observed: int) -> None:
    blob = bytearray(fixture_bytes)
    blob[2] = observed
    with pytest.raises(GpuMetricsHeaderError) as excinfo:
        decode(blob)
    error = excinfo.value
    assert error.field == "format_revision"
    assert error.observed == observed
    assert error.expected == FORMAT_REVISION
    assert "format_revision" in str(error)
    assert str(observed) in str(error)
    assert str(FORMAT_REVISION) in str(error)


@pytest.mark.parametrize("observed", [0, 256, 260, 268, 4096])
def test_unknown_structure_size_is_rejected(fixture_bytes: bytes, observed: int) -> None:
    blob = bytearray(fixture_bytes)
    struct.pack_into("<H", blob, 0, observed)
    with pytest.raises(GpuMetricsHeaderError) as excinfo:
        decode(blob)
    error = excinfo.value
    assert error.field == "structure_size"
    assert error.observed == observed
    assert error.expected == STRUCTURE_SIZE_BYTES
    assert "structure_size" in str(error)
    assert str(observed) in str(error)
    assert str(STRUCTURE_SIZE_BYTES) in str(error)


@pytest.mark.parametrize("observed", [1, 2, 255])
def test_unknown_content_revision_is_rejected(fixture_bytes: bytes, observed: int) -> None:
    blob = bytearray(fixture_bytes)
    blob[3] = observed
    with pytest.raises(GpuMetricsHeaderError) as excinfo:
        decode(blob)
    error = excinfo.value
    assert error.field == "content_revision"
    assert error.observed == observed
    assert error.expected == CONTENT_REVISION


# --- Error ergonomics ------------------------------------------------------


def test_rejections_are_value_errors_so_callers_catch_one_base() -> None:
    assert issubclass(GpuMetricsSizeError, GpuMetricsError)
    assert issubclass(GpuMetricsHeaderError, GpuMetricsError)
    assert issubclass(GpuMetricsError, ValueError)
    with pytest.raises(ValueError):
        decode(b"")


def test_buffer_types_are_accepted_interchangeably(fixture_bytes: bytes) -> None:
    expected = decode(fixture_bytes)
    assert decode(bytearray(fixture_bytes)) == expected
    assert decode(memoryview(fixture_bytes)) == expected


def test_only_the_layout_is_validated_not_the_values(fixture_bytes: bytes) -> None:
    """Fail closed on an unknown layout, not on implausible field values."""
    blob = bytearray(fixture_bytes)
    blob[HEADER_SIZE_BYTES:FIELD_REGION_BYTES] = b"\xff" * (FIELD_REGION_BYTES - HEADER_SIZE_BYTES)
    decoded = decode(blob)
    assert decoded["average_socket_power"] == 0xFFFFFFFF
    assert decoded["temperature_gfx"] == 0xFFFF


def test_errors_carry_no_indirection_to_the_input(fixture_bytes: bytes) -> None:
    """A rejected decode must not leak a partially built dict to the caller."""
    blob = bytearray(fixture_bytes)
    blob[2] = 99
    with pytest.raises(GpuMetricsHeaderError) as excinfo:
        decode(blob)
    assert not hasattr(excinfo.value, "decoded")
