#!/usr/bin/env python3
"""Tests for the shared ggml CPU-backend artifact census.

``scripts/lib/ggml_cpu_census.py`` is the single source of truth for answering
"is this ``libggml-cpu`` build the one we intend to deploy?" — the question the
LP-0MVZVZLCS000LQNB diagnosis answered with a symbol-name heuristic that can
never be satisfied by an upstream build (upstream names the arch kernel
``ggml_vec_dot_q4_K_q8_K``; only the fallback carries a ``_generic`` suffix).

Covers LP-0MW3K7Y3C003WEXW AC1 (arch-specific quant kernels present, with the
census quoted) and the ``GGML_CPU_REPACK`` half of AC3 (the
``ggml_cpu_extra_work_size`` crash path is absent from the artifact).

The fixtures under ``tests/fixtures/ggml_cpu_census/`` are real ``nm``/``objdump``
excerpts captured from the deployed artifact (``-march=native``, repack on) plus
a synthetic generic-only/trampoline body for the defect class.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import ggml_cpu_census  # noqa: E402
from lib.ggml_cpu_census import classify, locate_offsets  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "ggml_cpu_census"
SCRIPT = REPO_ROOT / "scripts" / "lib" / "ggml_cpu_census.py"

HOT = ("ggml_vec_dot_q4_K_q8_K", "ggml_vec_dot_q6_K_q8_K", "ggml_vec_dot_q4_0_q8_0")


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


EXTRA_BUFFER_TYPES = "ggml_backend_cpu_get_extra_buffer_types"


def _arch_disassembly(registration: str = "on") -> dict[str, str]:
    return {
        "ggml_vec_dot_q4_K_q8_K": _fixture("objdump_vec_q4_K_avx2.txt"),
        "ggml_vec_dot_q6_K_q8_K": _fixture("objdump_vec_q6_K_avx2.txt"),
        "ggml_vec_dot_q4_0_q8_0": _fixture("objdump_vec_q4_0_avx2.txt"),
        EXTRA_BUFFER_TYPES: _fixture(f"objdump_extra_buffer_types_repack_{registration}.txt"),
    }


def _generic_disassembly(registration: str = "off") -> dict[str, str]:
    trampoline = _fixture("objdump_vec_q4_K_generic_only.txt")
    bodies = {name: trampoline for name in HOT}
    bodies[EXTRA_BUFFER_TYPES] = _fixture(f"objdump_extra_buffer_types_repack_{registration}.txt")
    return bodies


# ---------------------------------------------------------------------------
# classify(): the verdict logic
# ---------------------------------------------------------------------------


def test_deployed_style_build_is_arch_specific_with_repack_compiled_in():
    """The live artifact shape: -march=native kernels AND the repack path.

    This is the build that crashes: it ships vectorised quant kernels, so the
    "missing arch kernels" theory is refuted for this artifact, and the repack
    extra-buffer path (the crashing code) is present.
    """
    report = classify(
        _fixture("nm_repack_on_arch.txt"),
        _arch_disassembly(),
        expect_repack="on",
        library="libggml-cpu.so.0.9.11",
    )

    assert report["arch_specific"] is True
    assert report["repack_registered"] is True
    assert report["generic_kernels"] == 3
    assert report["verdict"] == "ok", report["errors"]
    # Every hot kernel keeps its generic fallback symbol; that is normal and
    # must never be reported as the defect.
    for name in HOT:
        assert report["hot_kernels"][name]["vector_insns"] > 0
        assert report["hot_kernels"][name]["generic_tail_call"] is False


def test_registered_repack_fails_the_repack_off_expectation():
    """A built artifact that still registers the crash path must be rejected."""
    report = classify(
        _fixture("nm_repack_on_arch.txt"),
        _arch_disassembly("on"),
        expect_repack="off",
    )

    assert report["verdict"] == "fail"
    assert any("repack" in err.lower() for err in report["errors"])
    assert any("ggml_cpu_extra_work_size" in err for err in report["errors"]), (
        "the repack error must name the crashing function so the operator "
        f"does not have to rediscover it: {report['errors']!r}"
    )


def test_generic_only_build_fails_the_arch_kernel_gate():
    """A build whose arch entry point just tail-calls ``_generic`` is a defect.

    This is the LP-0MVZVZLCS000LQNB §2 class (no arch-specific quant kernels):
    every quantised dot product falls back to the compiler-vectorised reference
    loop. The census must fail it loudly rather than quote a passing count.
    """
    report = classify(
        _fixture("nm_generic_only.txt"),
        _generic_disassembly(),
        expect_repack="off",
    )

    assert report["arch_specific"] is False
    assert report["repack_registered"] is False
    assert report["verdict"] == "fail"
    assert any("arch" in err.lower() for err in report["errors"])
    assert report["hot_kernels"]["ggml_vec_dot_q4_K_q8_K"]["generic_tail_call"] is True


def test_repack_free_arch_build_passes_both_gates():
    """The intended deployment artifact: arch kernels, no repack path."""
    report = classify(
        _fixture("nm_repack_off_arch.txt"),
        _arch_disassembly("off"),
        expect_repack="off",
    )

    assert report["arch_specific"] is True
    assert report["repack_registered"] is False
    assert report["repack_markers"] == []
    assert report["verdict"] == "ok", report["errors"]
    assert report["errors"] == []


def test_repack_code_linked_but_unregistered_passes_the_repack_gate():
    """The shipped fix: repack.cpp is still linked, but nothing registers it.

    ``GGML_CPU_REPACK=OFF`` does not remove ``repack.cpp`` from the build (it is
    compiled unconditionally); it removes the ``#ifdef`` block that registers the
    repack buffer type, which is what decides whether
    ``ggml_cpu_extra_work_size`` can enter its faulting dispatch. A gate based on
    symbol presence would wrongly reject this artifact.
    """
    report = classify(
        _fixture("nm_repack_on_arch.txt"),   # symbols: repack code IS linked
        _arch_disassembly("off"),            # ... but nothing registers it
        expect_repack="off",
    )

    assert report["repack_linked"] is True
    assert report["repack_registered"] is False
    assert report["verdict"] == "ok", report["errors"]


def test_undisassembled_registration_site_cannot_be_certified_as_repack_free():
    """Unknown registration state must fail closed, never pass by default."""
    report = classify(
        _fixture("nm_repack_off_arch.txt"),
        {name: _fixture("objdump_vec_q4_K_avx2.txt") for name in HOT},
        expect_repack="off",
    )

    assert report["repack_registered"] is None
    assert report["verdict"] == "fail"
    assert any("cannot determine" in err for err in report["errors"])


def test_registration_state_is_irrelevant_when_repack_expectation_is_any():
    report = classify(
        _fixture("nm_repack_off_arch.txt"),
        {name: _fixture("objdump_vec_q4_K_avx2.txt") for name in HOT},
        expect_repack="any",
    )

    assert report["repack_registered"] is None
    assert report["verdict"] == "ok", report["errors"]


def test_arch_kernel_without_vector_codegen_is_not_arch_specific():
    """Vector instruction count, not the symbol name, decides arch specificity."""
    scalar_body = (
        "\n00000000000f22c0 <ggml_vec_dot_q4_K_q8_K>:\n"
        "   f22c0:\t48 83 ec 08          \tsub    $0x8,%rsp\n"
        "   f22c4:\te8 00 00 00 00       \tcall   f22c9 <other>\n"
    )
    bodies = {name: scalar_body for name in HOT}
    bodies[EXTRA_BUFFER_TYPES] = _fixture("objdump_extra_buffer_types_repack_off.txt")
    report = classify(
        _fixture("nm_repack_off_arch.txt"),
        bodies,
        expect_repack="off",
    )

    assert report["arch_specific"] is False
    assert report["verdict"] == "fail"


def test_require_arch_kernels_can_be_relaxed_for_a_repack_only_check():
    """``--no-require-arch-kernels`` keeps the repack gate but drops the arch gate."""
    report = classify(
        _fixture("nm_generic_only.txt"),
        _generic_disassembly(),
        expect_repack="off",
        require_arch_kernels=False,
    )

    assert report["arch_specific"] is False
    assert report["verdict"] == "ok", report["errors"]


# ---------------------------------------------------------------------------
# locate_offsets(): crash file offsets -> enclosing symbol
# ---------------------------------------------------------------------------


def test_crash_offsets_locate_the_repack_dispatch():
    """The kernel's bracketed crash offsets must resolve to the repack dispatch.

    ``ggml_cpu_extra_work_size`` is at 0x51010 and its three faulting
    instructions are +0x38/+0x41/+0x47, so the offsets the kernel prints for
    this artifact must land inside it — reproducing the LP-0MW3K7Y3C003WEXW
    re-diagnosis from the log line alone.
    """
    nm = (
        "0000000000050f90 T ggml_cpu_extra_compute_forward\n"
        "0000000000051010 T ggml_cpu_extra_work_size\n"
        "0000000000051090 T ggml_compute_forward_add_non_quantized\n"
    )

    located = locate_offsets(nm, [0x51048, 0x51051, 0x51057, 0x51090])

    assert [entry["symbol"] for entry in located] == [
        "ggml_cpu_extra_work_size",
        "ggml_cpu_extra_work_size",
        "ggml_cpu_extra_work_size",
        "ggml_compute_forward_add_non_quantized",
    ]
    assert located[0]["offset_into_symbol"] == 0x38
    assert located[2]["offset_into_symbol"] == 0x47


def test_locate_offsets_below_the_first_symbol_is_reported_as_unknown():
    located = locate_offsets("0000000000051010 T ggml_cpu_extra_work_size\n", [0x10])

    assert located == [{"offset": 0x10, "symbol": None, "offset_into_symbol": None}]


# ---------------------------------------------------------------------------
# CLI: end-to-end against stub nm/objdump tools
# ---------------------------------------------------------------------------


def _elf_stub(tmp_path: Path) -> Path:
    """A minimal file carrying the ELF magic the CLI checks before censusing."""
    lib = tmp_path / "libggml-cpu.so.0.9.11"
    lib.write_bytes(b"\x7fELF" + b"\x02\x01\x01" + b"\x00" * 64)
    return lib


def _tool(tmp_path: Path, name: str, body: str) -> Path:
    tool = tmp_path / name
    tool.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    tool.chmod(0o755)
    return tool


def _census_tools(
    tmp_path: Path, nm_fixture: str, use_arch_disassembly: bool, registration: str = "off"
) -> tuple[Path, Path]:
    nm = _tool(tmp_path, "nm", f'cat "{FIXTURES / nm_fixture}"\n')
    if use_arch_disassembly:
        body = (
            f'case "$*" in\n'
            f'  *extra_buffer_types*) cat "{FIXTURES / f"objdump_extra_buffer_types_repack_{registration}.txt"}" ;;\n'
            f'  *q4_K_q8_K*) cat "{FIXTURES / "objdump_vec_q4_K_avx2.txt"}" ;;\n'
            f'  *q6_K_q8_K*) cat "{FIXTURES / "objdump_vec_q6_K_avx2.txt"}" ;;\n'
            f'  *q4_0_q8_0*) cat "{FIXTURES / "objdump_vec_q4_0_avx2.txt"}" ;;\n'
            f'  *) cat "{FIXTURES / "objdump_vec_q4_K_generic_only.txt"}" ;;\n'
            f"esac\n"
        )
    else:
        body = f'cat "{FIXTURES / "objdump_vec_q4_K_generic_only.txt"}"\n'
    objdump = _tool(tmp_path, "objdump", body)
    return nm, objdump


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**os.environ},
    )


def test_cli_reports_ok_for_a_repack_free_arch_artifact(tmp_path):
    lib = _elf_stub(tmp_path)
    nm, objdump = _census_tools(tmp_path, "nm_repack_off_arch.txt", use_arch_disassembly=True)

    proc = _run_cli(
        str(lib), "--json", "--expect-repack", "off",
        "--nm", str(nm), "--objdump", str(objdump),
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    report = json.loads(proc.stdout)
    assert report["verdict"] == "ok"
    assert report["library"].endswith("libggml-cpu.so.0.9.11")
    assert report["arch_specific"] is True
    assert report["repack_registered"] is False


def test_cli_exits_non_zero_when_the_repack_gate_fails(tmp_path):
    lib = _elf_stub(tmp_path)
    nm, objdump = _census_tools(
        tmp_path, "nm_repack_on_arch.txt", use_arch_disassembly=True, registration="on"
    )

    proc = _run_cli(
        str(lib), "--json", "--expect-repack", "off",
        "--nm", str(nm), "--objdump", str(objdump),
    )

    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["verdict"] == "fail"
    assert report["repack_registered"] is True
    # The operator sees the reason on stderr as well as in the JSON report.
    assert "ggml_cpu_extra_work_size" in proc.stderr


def test_cli_locates_crash_offsets_from_the_log_line(tmp_path):
    """``--locate-offset`` takes the hex the kernel prints in ``lib[…[51057,…]``."""
    lib = _elf_stub(tmp_path)
    nm = _tool(
        tmp_path, "nm",
        'printf "%s\\n" "0000000000050f90 T ggml_cpu_extra_compute_forward" '
        '"0000000000051010 T ggml_cpu_extra_work_size" '
        '"0000000000051090 T ggml_compute_forward_add_non_quantized"\n',
    )
    objdump = _tool(tmp_path, "objdump", ": nothing\n")

    proc = _run_cli(
        str(lib), "--json", "--no-require-arch-kernels", "--expect-repack", "any",
        "--nm", str(nm), "--objdump", str(objdump), "--locate-offset", "0x51057",
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    report = json.loads(proc.stdout)
    assert report["offsets"][0]["symbol"] == "ggml_cpu_extra_work_size"


def test_cli_refuses_a_non_elf_artifact(tmp_path):
    """A non-ELF file is a usage error (exit 2), never a silent pass."""
    bogus = tmp_path / "libggml-cpu.so"
    bogus.write_text("not an elf\n", encoding="utf-8")

    proc = _run_cli(str(bogus), "--json")

    assert proc.returncode == 2
    assert "ELF" in proc.stderr or "ELF" in proc.stdout


def test_cli_refuses_an_artifact_without_symbols(tmp_path):
    """A stripped library cannot be censused: fail loudly rather than guess."""
    lib = _elf_stub(tmp_path)
    nm = _tool(tmp_path, "nm", "echo 'nm: no symbols' >&2\nexit 0\n")

    proc = _run_cli(str(lib), "--json", "--nm", str(nm), "--objdump", str(nm))

    assert proc.returncode == 2
    assert "symbol" in (proc.stderr + proc.stdout).lower()


def test_hot_kernel_list_covers_the_repack_quant_types():
    """The census must look at the Q4_K/Q6_K/Q4_0 hot paths the crash implicates."""
    assert set(HOT) <= set(ggml_cpu_census.HOT_KERNELS)


@pytest.mark.parametrize("expect", ["on", "off", "any"])
def test_expect_repack_accepts_only_the_documented_vocabulary(expect):
    report = classify(
        _fixture("nm_repack_off_arch.txt"), _arch_disassembly(), expect_repack=expect
    )
    assert report["verdict"] in ("ok", "fail")


def test_invalid_expect_repack_value_is_rejected():
    with pytest.raises(ValueError):
        classify(
            _fixture("nm_repack_off_arch.txt"), _arch_disassembly(), expect_repack="maybe"
        )
