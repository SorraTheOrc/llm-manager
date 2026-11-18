#!/usr/bin/env python3
"""Decode the ``amd_gpu_metrics_v3_0`` blob published at ``gpu_metrics``.

Why this exists (LP-0MVZEW6K6007BPY0, parent LP-0MVSQT7YC0045HFW)
-----------------------------------------------------------------
``/sys/class/drm/card0/device/gpu_metrics`` is a world-readable 264-byte binary
record (the interface ``amdgpu_top`` reads). No installed ``linux-headers``
package on this host ships the structure, so the layout is pinned here as a
literal field table: transcribing it once, with byte offsets that a test can
check, is the only way to decode the blob without ROCm userspace, ``amd-smi``
or root.

Source
------
``struct gpu_metrics_v3_0`` in
``drivers/gpu/drm/amd/include/kgd_pp_interface.h`` (the kernel's
``metrics_table_header`` plus the APU-oriented v3.0 member list), which lives
inside ``#pragma pack(push, 4)``. The same structure is vendored verbatim in
``libdrm_amdgpu_sys`` 0.8.16 (``wrapper/wrapper_gpu_metrics.h``), which is what
the offsets below were transcribed from and cross-checked against.

Layout, and why the record is 264 bytes
---------------------------------------
``<HBB`` header (4 B: ``structure_size``, ``format_revision``,
``content_revision``) followed by a 256-byte field region, so the members end at
offset 260 (``FIELD_REGION_BYTES``). The kernel reports ``structure_size`` as
**264**, leaving 4 bytes of tail padding (``TAIL_PADDING_BYTES``) that carry no
field: on the capture committed as ``tests/fixtures/gpu_metrics_v3_0.bin`` those
bytes read ``ffffffff``. Three interior gaps also come from ``pack(4)``
alignment — 2 bytes before the ``uint64_t`` timestamp at offset 104, and 2 bytes
before each of the two ``uint32_t`` blocks at offsets 120 and 228. The
field-table test asserts this arithmetic, so a transcription slip fails there
rather than silently misparsing live data.

Fail closed
-----------
:func:`decode` accepts **exactly** :data:`STRUCTURE_SIZE_BYTES` bytes and
requires ``format_revision`` :data:`FORMAT_REVISION` and ``content_revision``
:data:`CONTENT_REVISION`. Any other size or revision raises
:class:`GpuMetricsError` naming the observed and expected values instead of
returning a partial or misparsed record: a kernel or firmware change must break
loudly, because every offset below would otherwise be a guess. A bumped
revision is re-validated against the live blob before the pinned values change
(the re-measure recipe is documented with the probe CLI, LP-0MVZEW6UL0054D8W).

Values are SMU **time filtered** (``time_filter_alphavalue`` reads 1 000 000 on
this host) with an undocumented window, so they support no claim finer than the
sample interval. Activity fields are percentages; power is milliwatts; clocks
are megahertz; bandwidths are megabytes per second.

Example
-------
>>> from pathlib import Path
>>> decoded = decode(Path("/sys/class/drm/card0/device/gpu_metrics").read_bytes())
>>> decoded["format_revision"]
3
>>> decoded["average_socket_power"] > 0
True
>>> len(decoded["average_core_c0_activity"])
16
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# --- Pinned header and record geometry --------------------------------------

HEADER_STRUCT_FORMAT = "<HBB"
"""``struct metrics_table_header``: u16 size, u8 format, u8 content revision."""

HEADER_SIZE_BYTES = struct.calcsize(HEADER_STRUCT_FORMAT)
"""Bytes occupied by the header (4)."""

STRUCTURE_SIZE_BYTES = 264
"""``structure_size`` of ``struct gpu_metrics_v3_0``, as the kernel reports it."""

FIELD_REGION_BYTES = 260
"""Bytes of decoded members: header plus fields, ending before tail padding."""

TAIL_PADDING_BYTES = STRUCTURE_SIZE_BYTES - FIELD_REGION_BYTES
"""Trailing bytes in the sysfs record that carry no member (4)."""

FORMAT_REVISION = 3
"""Pinned ``format_revision`` — the v3.0 layout, the only one decoded here."""

CONTENT_REVISION = 0
"""Pinned ``content_revision``; a bump may redefine existing member semantics."""

HEADER_FIELD_NAMES = ("structure_size", "format_revision", "content_revision")
"""Keys the decoded dict always carries before the member keys."""


# --- Decoded members --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One member of ``struct gpu_metrics_v3_0`` at a pinned byte offset."""

    name: str
    offset: int
    struct_format: str
    """Single little-endian integer code: ``H`` (u16), ``I`` (u32) or ``Q`` (u64)."""
    count: int = 1
    unit: str = ""
    description: str = ""

    @property
    def element_size(self) -> int:
        """Bytes one element occupies (2, 4 or 8)."""
        return struct.calcsize(f"<{self.struct_format}")

    @property
    def size(self) -> int:
        """Bytes this member occupies, with no alignment padding."""
        return self.element_size * self.count

    @property
    def end(self) -> int:
        """Offset one past this member's last byte."""
        return self.offset + self.size

    @property
    def is_array(self) -> bool:
        return self.count > 1


FIELD_TABLE: tuple[FieldSpec, ...] = (
    # Temperature
    FieldSpec("temperature_gfx", 4, "H", unit="degC", description="gfx temperature on APUs"),
    FieldSpec("temperature_soc", 6, "H", unit="degC", description="soc temperature on APUs"),
    FieldSpec("temperature_core", 8, "H", 16, unit="degC", description="CPU core temperature on APUs"),
    FieldSpec("temperature_skin", 40, "H", unit="degC", description="skin temperature on APUs"),
    # Utilization
    FieldSpec("average_gfx_activity", 42, "H", unit="%", description="time filtered GFX busy % [0-100]"),
    FieldSpec("average_vcn_activity", 44, "H", unit="%", description="time filtered VCN busy % [0-100]"),
    FieldSpec("average_ipu_activity", 46, "H", 8, unit="%", description="time filtered IPU per-column busy % [0-100]"),
    FieldSpec(
        "average_core_c0_activity",
        62,
        "H",
        16,
        unit="%",
        description="time filtered per-core C0 residency % [0-100]",
    ),
    FieldSpec("average_dram_reads", 94, "H", unit="MB/s", description="time filtered DRAM read bandwidth [MB/sec]"),
    FieldSpec("average_dram_writes", 96, "H", unit="MB/s", description="time filtered DRAM write bandwidth [MB/sec]"),
    FieldSpec("average_ipu_reads", 98, "H", unit="MB/s", description="time filtered IPU read bandwidth [MB/sec]"),
    FieldSpec("average_ipu_writes", 100, "H", unit="MB/s", description="time filtered IPU write bandwidth [MB/sec]"),
    # Driver attached timestamp
    FieldSpec("system_clock_counter", 104, "Q", unit="ns", description="driver attached timestamp (in ns)"),
    # Power/Energy
    FieldSpec(
        "average_socket_power",
        112,
        "I",
        unit="mW",
        description="time filtered power used for PPT/STAPM [APU+dGPU] [mW]",
    ),
    FieldSpec("average_ipu_power", 116, "H", unit="mW", description="time filtered IPU power [mW]"),
    FieldSpec("average_apu_power", 120, "I", unit="mW", description="time filtered APU power [mW]"),
    FieldSpec("average_gfx_power", 124, "I", unit="mW", description="time filtered GFX power [mW]"),
    FieldSpec("average_dgpu_power", 128, "I", unit="mW", description="time filtered dGPU power [mW]"),
    FieldSpec(
        "average_all_core_power",
        132,
        "I",
        unit="mW",
        description="time filtered sum of core power across all cores in the socket [mW]",
    ),
    FieldSpec(
        "average_core_power",
        136,
        "H",
        16,
        unit="mW",
        description="calculated core power [mW]",
    ),
    FieldSpec("average_sys_power", 168, "H", unit="mW", description="time filtered total system power [mW]"),
    FieldSpec("stapm_power_limit", 170, "H", unit="mW", description="maximum IRM defined STAPM power limit [mW]"),
    FieldSpec("current_stapm_power_limit", 172, "H", unit="mW", description="time filtered STAPM power limit [mW]"),
    # Average clocks
    FieldSpec("average_gfxclk_frequency", 174, "H", unit="MHz", description="time filtered GFX clock [MHz]"),
    FieldSpec("average_socclk_frequency", 176, "H", unit="MHz", description="time filtered SOC clock [MHz]"),
    FieldSpec("average_vpeclk_frequency", 178, "H", unit="MHz", description="time filtered VPE clock [MHz]"),
    FieldSpec("average_ipuclk_frequency", 180, "H", unit="MHz", description="time filtered IPU clock [MHz]"),
    FieldSpec("average_fclk_frequency", 182, "H", unit="MHz", description="time filtered F clock [MHz]"),
    FieldSpec("average_vclk_frequency", 184, "H", unit="MHz", description="time filtered VCLK [MHz]"),
    FieldSpec("average_uclk_frequency", 186, "H", unit="MHz", description="time filtered UCLK [MHz]"),
    FieldSpec("average_mpipu_frequency", 188, "H", unit="MHz", description="time filtered MPIPU clock [MHz]"),
    # Current clocks
    FieldSpec("current_coreclk", 190, "H", 16, unit="MHz", description="target core frequency [MHz]"),
    FieldSpec(
        "current_core_maxfreq",
        222,
        "H",
        unit="MHz",
        description="CCLK frequency limit enforced on classic cores [MHz]",
    ),
    FieldSpec("current_gfx_maxfreq", 224, "H", unit="MHz", description="GFXCLK frequency limit enforced on GFX [MHz]"),
    # Throttle residency (ASIC dependent)
    FieldSpec("throttle_residency_prochot", 228, "I", description="throttle residency prochot"),
    FieldSpec("throttle_residency_spl", 232, "I", description="throttle residency SPL"),
    FieldSpec("throttle_residency_fppt", 236, "I", description="throttle residency FPPT"),
    FieldSpec("throttle_residency_sppt", 240, "I", description="throttle residency SPPT"),
    FieldSpec("throttle_residency_thm_core", 244, "I", description="throttle residency thermal (core)"),
    FieldSpec("throttle_residency_thm_gfx", 248, "I", description="throttle residency thermal (GFX)"),
    FieldSpec("throttle_residency_thm_soc", 252, "I", description="throttle residency thermal (SOC)"),
    # Metrics table alpha filter time constant
    FieldSpec(
        "time_filter_alphavalue",
        256,
        "I",
        unit="us",
        description="metrics table alpha filter time constant [us]",
    ),
)


# --- Errors -----------------------------------------------------------------


class GpuMetricsError(ValueError):
    """Base class for a rejected ``gpu_metrics`` record.

    Carries the mismatching ``field``, the ``observed`` value and the
    ``expected`` value so a caller can log or act on the failure without
    re-parsing the message.
    """

    def __init__(self, message: str, *, field: str, observed: int, expected: int) -> None:
        super().__init__(message)
        self.field = field
        self.observed = observed
        self.expected = expected


class GpuMetricsSizeError(GpuMetricsError):
    """The buffer is not exactly :data:`STRUCTURE_SIZE_BYTES` bytes long."""


class GpuMetricsHeaderError(GpuMetricsError):
    """The header reports a structure size or revision this decoder does not know."""


# --- Decoding ---------------------------------------------------------------


def _size_mismatch(size: int) -> GpuMetricsSizeError:
    if size == 0:
        shape = "empty"
    elif size < STRUCTURE_SIZE_BYTES:
        shape = "truncated"
    else:
        shape = "oversized"
    return GpuMetricsSizeError(
        f"gpu_metrics blob is {shape}: {size} bytes, expected exactly {STRUCTURE_SIZE_BYTES} "
        f"({FIELD_REGION_BYTES}-byte field region + {TAIL_PADDING_BYTES}-byte tail padding for "
        "struct gpu_metrics_v3_0); refusing to return a partial or unknown record",
        field="structure_size",
        observed=size,
        expected=STRUCTURE_SIZE_BYTES,
    )


def _header_mismatch(field: str, observed: int, expected: int) -> GpuMetricsHeaderError:
    return GpuMetricsHeaderError(
        f"gpu_metrics {field} mismatch: observed {observed}, expected {expected}; refusing to decode "
        "an unknown layout — re-verify the pinned field table against struct gpu_metrics_v3_0 before "
        "accepting a new revision",
        field=field,
        observed=observed,
        expected=expected,
    )


def decode(raw: bytes | bytearray | memoryview) -> dict[str, int | tuple[int, ...]]:
    """Decode one ``gpu_metrics`` record into named fields.

    Parameters
    ----------
    raw:
        The ``gpu_metrics`` contents, read with no offset or seek. Must be
        exactly :data:`STRUCTURE_SIZE_BYTES` (264) bytes.

    Returns
    -------
    dict
        ``structure_size``, ``format_revision``, ``content_revision`` and one
        key per :data:`FIELD_TABLE` member, in the table's order. Array members
        are tuples in declaration order (e.g. ``temperature_core`` has 16
        entries); scalar members are plain ``int``. Units are the raw firmware
        units documented on each :class:`FieldSpec`.

    Raises
    ------
    GpuMetricsSizeError
        The buffer is empty, truncated or oversized.
    GpuMetricsHeaderError
        ``structure_size``, ``format_revision`` or ``content_revision`` is not
        the pinned value.
    """
    blob = bytes(raw)
    if len(blob) != STRUCTURE_SIZE_BYTES:
        raise _size_mismatch(len(blob))

    structure_size, format_revision, content_revision = struct.unpack_from(HEADER_STRUCT_FORMAT, blob, 0)
    if structure_size != STRUCTURE_SIZE_BYTES:
        raise _header_mismatch("structure_size", structure_size, STRUCTURE_SIZE_BYTES)
    if format_revision != FORMAT_REVISION:
        raise _header_mismatch("format_revision", format_revision, FORMAT_REVISION)
    if content_revision != CONTENT_REVISION:
        raise _header_mismatch("content_revision", content_revision, CONTENT_REVISION)

    decoded: dict[str, int | tuple[int, ...]] = {
        "structure_size": structure_size,
        "format_revision": format_revision,
        "content_revision": content_revision,
    }
    for field in FIELD_TABLE:
        values = struct.unpack_from(f"<{field.count}{field.struct_format}", blob, field.offset)
        decoded[field.name] = values if field.is_array else values[0]
    return decoded
