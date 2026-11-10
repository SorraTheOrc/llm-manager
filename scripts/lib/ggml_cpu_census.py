#!/usr/bin/env python3
"""ggml CPU-backend artifact census — one verdict per ``libggml-cpu`` build.

Why this exists (LP-0MW3K7Y3C003WEXW)
------------------------------------
The SIGSEGV/GPF investigation of the deployed ``libggml-cpu.so.0.9.11``
produced two *false* conclusions because each surface rolled its own heuristic:

1. "39 ``_generic`` kernels / 0 arch-specific kernels" — upstream llama.cpp
   **always** ships the ``_generic`` fallback symbols and never suffixes the
   arch kernel with ``_avx*`` (it is plain ``ggml_vec_dot_q4_K_q8_K``). A
   healthy ``-march=native`` build therefore "measured" as broken, while a
   genuinely broken build (arch entry point reduced to a tail-call into
   ``_generic``) would have "measured" as fine.
2. The artifact actually being measured was not the artifact that was running
   (``/usr/local/lib64`` vs the deployed ``/home/rgardler/llama.cpp/build/bin``).

The census replaces both with verifiable, artifact-level facts:

* **arch-specific codegen** — disassemble the hot quant kernels and count
  ``ymm``/``zmm`` instructions in the symbol body; a bare tail-call into
  ``<kernel>_generic`` is *not* arch-specific, whatever the symbol is called.
* **repack path** — is ``GGML_CPU_REPACK`` compiled in? Its extra buffer type
  is the only thing that gives ``ggml_cpu_extra_work_size`` a body to execute,
  and that function is where the deployed artifact faults (see
  ``docs/build-record-llama-cpu-LP-0MW3K7Y3C003WEXW.md``).
* **crash-offset location** — map the hex offset the kernel prints in
  ``libggml-cpu.so.0.9.11[51057,ff000]`` onto the enclosing symbol, so the
  attribution is reproduced from the log line instead of re-derived by hand.

The module is a pure classifier plus a thin CLI so both the build
(``scripts/rebuild-llama.sh``) and the diagnosis harness
(``scripts/diagnose_libggml_segfault.sh``) can share it.

Exit codes: ``0`` verdict ok, ``1`` verdict fail, ``2`` usage/tooling error
(missing library, non-ELF, stripped library, unusable ``nm``/``objdump``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: Quant dot-product kernels whose codegen decides "does this build use the
#: hand-written arch paths?". Q4_K/Q6_K are the presets actually in play on
#: this host (``Qwen3-MTP`` Q4_K_S, ``Qwen2.5-7B`` Q4_K_M); Q4_0 covers the
#: repack source type.
HOT_KERNELS: tuple[str, ...] = (
    "ggml_vec_dot_q4_K_q8_K",
    "ggml_vec_dot_q6_K_q8_K",
    "ggml_vec_dot_q4_0_q8_0",
)

#: A body with fewer vector instructions than this is a wrapper/scalar build,
#: not an arch-specific kernel (real bodies carry 14–90 on x86-64).
MIN_VECTOR_INSNS = 4

#: Function whose body decides whether the crash path can execute: with
#: ``GGML_CPU_REPACK=OFF`` the ``#ifdef`` block that registers the repack buffer
#: type is compiled out, so the vector it returns is empty and
#: ``ggml_cpu_extra_work_size`` never enters its faulting loop body. The repack
#: *code* (``repack.cpp``) is always linked, so symbol presence cannot decide
#: this — only the registration call can.
EXTRA_BUFFER_TYPES_SYMBOL = "ggml_backend_cpu_get_extra_buffer_types"

#: Symbols that indicate the repack implementation is linked (informational:
#: the loader is compiled unconditionally, so their presence is NOT a verdict).
REPACK_MARKERS: tuple[str, ...] = (
    "ggml_backend_cpu_repack",
    "ggml::cpu::repack",
    "repack::",
    "ggml_gemv_",
    "ggml_gemm_",
    "ggml_repack_get_optimal_repack_type",
    "repack_q",
)

_EXPECT_REPACK_VALUES = ("on", "off", "any")
_VECTOR_RE = re.compile(r"\b(?:ymm|zmm)\d+\b")
_HEADER_RE = re.compile(r"^([0-9a-fA-F]+) <(.+)>:$")
_KERNEL_GENERIC_RE = re.compile(r"^ggml_(?:vec_dot|gemv|gemm)_.*_generic$")
_ARCH_KERNEL_RE = re.compile(r"^ggml_vec_dot_[A-Za-z0-9_]+$")
_ELF_MAGIC = b"\x7fELF"


@dataclass(frozen=True)
class Symbol:
    """One ``nm`` entry: address (``None`` when undefined), type char, name."""

    address: int | None
    kind: str
    name: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_symbols(nm_output: str) -> list[Symbol]:
    """Parse ``nm`` output, tolerating demangled C++ names with spaces."""
    symbols: list[Symbol] = []
    for raw in nm_output.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isalnum() and _is_hex(parts[0]):
            symbols.append(Symbol(int(parts[0], 16), parts[1], parts[2]))
        elif len(parts) == 2 and len(parts[0]) == 1 and not _is_hex(parts[0]):
            # undefined symbol, e.g. "U ggml_backend_cpu_repack_buffer_type"
            symbols.append(Symbol(None, parts[0], parts[1]))
    return symbols


def _is_hex(value: str) -> bool:
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _symbol_block(disassembly: str, symbol: str) -> list[str]:
    """Instructions of one symbol, taken from ``objdump -d`` output.

    The header match is a containment test so a caller can name a function by
    its source name even when ``objdump`` printed the mangled one.
    """
    lines = disassembly.splitlines()
    if not any(_HEADER_RE.match(line.strip()) for line in lines):
        return [line for line in lines if line.strip()]

    block: list[str] = []
    inside = False
    for line in lines:
        header = _HEADER_RE.match(line.strip())
        if header:
            name = header.group(2).strip()
            inside = name == symbol or symbol in name
            continue
        if inside:
            block.append(line)
    return block


def count_vector_insns(disassembly: str, symbol: str) -> int:
    return sum(1 for line in _symbol_block(disassembly, symbol) if _VECTOR_RE.search(line))


def repack_registered(disassembly: str) -> bool | None:
    """Is the repack extra buffer type registered? ``None`` when unknowable."""
    body = _symbol_block(disassembly, EXTRA_BUFFER_TYPES_SYMBOL)
    if not body:
        return None
    target = re.compile(r"\b(?:call|jmp)\w*\b[^\n]*repack", re.IGNORECASE)
    return any(target.search(line) for line in body)


def is_generic_tail_call(disassembly: str, symbol: str) -> bool:
    """True when the body just forwards to ``<symbol>_generic``."""
    target = re.compile(r"\b(?:jmp|call)\b[^\n]*\b" + re.escape(symbol) + r"_generic\b")
    return any(target.search(line) for line in _symbol_block(disassembly, symbol))


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def classify(
    nm_output: str,
    disassembly: Mapping[str, str],
    *,
    require_arch_kernels: bool = True,
    expect_repack: str = "any",
    library: str = "",
) -> dict:
    """Classify one artifact.

    ``disassembly`` maps a hot-kernel symbol name to ``objdump -d`` output.
    Missing entries are treated as "not disassembled" (a warning), so the
    caller decides whether a missing body is fatal.
    """
    if expect_repack not in _EXPECT_REPACK_VALUES:
        raise ValueError(
            f"expect_repack must be one of {_EXPECT_REPACK_VALUES}, got {expect_repack!r}"
        )

    symbols = parse_symbols(nm_output)
    names = [symbol.name for symbol in symbols]

    generic_kernels = sum(1 for name in names if _KERNEL_GENERIC_RE.match(name))
    arch_kernels = [
        name for name in names if _ARCH_KERNEL_RE.match(name) and not name.endswith("_generic")
    ]

    hot_kernels: dict[str, dict] = {}
    for kernel in HOT_KERNELS:
        body = disassembly.get(kernel)
        hot_kernels[kernel] = {
            "present": kernel in names,
            "disassembled": body is not None,
            "vector_insns": count_vector_insns(body, kernel) if body is not None else 0,
            "generic_tail_call": is_generic_tail_call(body, kernel) if body is not None else False,
        }

    arch_specific = any(
        entry["present"]
        and entry["vector_insns"] >= MIN_VECTOR_INSNS
        and not entry["generic_tail_call"]
        for entry in hot_kernels.values()
    )

    repack_markers = sorted({marker for name in names for marker in REPACK_MARKERS if marker in name})
    repack_linked = bool(repack_markers)
    registration_body = disassembly.get(EXTRA_BUFFER_TYPES_SYMBOL)
    registered = repack_registered(registration_body) if registration_body is not None else None

    errors: list[str] = []
    warnings: list[str] = []

    for kernel, entry in hot_kernels.items():
        if not entry["present"]:
            warnings.append(f"{kernel}: symbol not found in the artifact")
        elif not entry["disassembled"]:
            warnings.append(f"{kernel}: not disassembled (objdump unavailable?)")

    if require_arch_kernels and not arch_specific:
        detail = ", ".join(
            f"{kernel}: {entry['vector_insns']} vector insns"
            f"{' (tail-calls _generic)' if entry['generic_tail_call'] else ''}"
            for kernel, entry in hot_kernels.items()
        )
        errors.append(
            "no arch-specific quant kernel codegen detected — the hot quant "
            f"dot products would run the generic reference loop ({detail}); "
            "build with GGML_NATIVE=ON (or explicit GGML_AVX2/GGML_AVX512)"
        )

    if registered is None and expect_repack != "any":
        errors.append(
            f"cannot determine whether the CPU repack path is registered: "
            f"{EXTRA_BUFFER_TYPES_SYMBOL}() was not disassembled (stripped or "
            "missing symbol), so the ggml_cpu_extra_work_size crash path cannot "
            "be ruled out"
        )
    elif expect_repack == "off" and registered:
        errors.append(
            "CPU repack extra-buffer path is REGISTERED (GGML_CPU_REPACK=ON): "
            f"{EXTRA_BUFFER_TYPES_SYMBOL}() calls ggml_backend_cpu_repack_buffer_type(), "
            "so ggml_cpu_extra_work_size executes its faulting dispatch (offsets "
            "+0x38/+0x41/+0x47) — rebuild with -DGGML_CPU_REPACK=OFF"
        )
    elif expect_repack == "on" and registered is False:
        errors.append(
            "expected the CPU repack extra-buffer path to be registered "
            f"(GGML_CPU_REPACK=ON) but {EXTRA_BUFFER_TYPES_SYMBOL}() does not call "
            "ggml_backend_cpu_repack_buffer_type() (GGML_CPU_REPACK=OFF)"
        )

    return {
        "library": library,
        "generic_kernels": generic_kernels,
        "arch_kernels": arch_kernels,
        "hot_kernels": hot_kernels,
        "arch_specific": arch_specific,
        "repack_registered": registered,
        "repack_linked": repack_linked,
        "repack_markers": repack_markers,
        "require_arch_kernels": require_arch_kernels,
        "expect_repack": expect_repack,
        "verdict": "fail" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
    }


def locate_offsets(nm_output: str, offsets: Iterable[int]) -> list[dict]:
    """Map file offsets onto the enclosing symbol (last symbol at or below it)."""
    defined = [symbol for symbol in parse_symbols(nm_output) if symbol.address is not None]
    defined.sort(key=lambda symbol: symbol.address or 0)

    located: list[dict] = []
    for offset in offsets:
        enclosing: Symbol | None = None
        for symbol in defined:
            if (symbol.address or 0) <= offset:
                enclosing = symbol
            else:
                break
        located.append(
            {
                "offset": offset,
                "symbol": enclosing.name if enclosing else None,
                "offset_into_symbol": (offset - (enclosing.address or 0)) if enclosing else None,
            }
        )
    return located


# ---------------------------------------------------------------------------
# Human formatting
# ---------------------------------------------------------------------------


def format_report(report: Mapping) -> str:
    lines = [
        f"artifact:                {report['library'] or '(unspecified)'}",
        f"generic kernels:         {report['generic_kernels']}",
        f"arch kernels:            {len(report['arch_kernels'])}",
        f"arch-specific codegen:   {'yes' if report['arch_specific'] else 'NO'}",
    ]
    for kernel, entry in report["hot_kernels"].items():
        lines.append(
            f"  - {kernel}: present={entry['present']} vector_insns={entry['vector_insns']}"
            f" generic_tail_call={entry['generic_tail_call']}"
        )
    registered = report["repack_registered"]
    lines.append(
        f"CPU repack registered:   "
        f"{'unknown' if registered is None else ('YES' if registered else 'no')} "
        f"({report['expect_repack']} expected)"
        + (f"; repack code linked: {'yes' if report['repack_linked'] else 'no'}")
    )
    if report.get("offsets"):
        for entry in report["offsets"]:
            symbol = entry["symbol"] or "(below the first symbol)"
            lines.append(
                f"offset 0x{entry['offset']:x} -> {symbol}"
                + (
                    f" +0x{entry['offset_into_symbol']:x}"
                    if entry["offset_into_symbol"] is not None
                    else ""
                )
            )
    for warning in report["warnings"]:
        lines.append(f"WARNING: {warning}")
    for error in report["errors"]:
        lines.append(f"ERROR: {error}")
    lines.append(f"verdict:                 {report['verdict'].upper()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_tool(command: str, args: Sequence[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [*command.split(), *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise SystemExit(_tool_error(f"{command}: {exc}"))
    return proc.returncode, proc.stdout


def _tool_error(message: str) -> int:
    print(f"ggml_cpu_census: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Census one libggml-cpu artifact (arch kernels, CPU repack path)."
    )
    parser.add_argument("library", help="path to libggml-cpu.so*")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--expect-repack",
        choices=list(_EXPECT_REPACK_VALUES),
        default="any",
        help="require the GGML_CPU_REPACK path to be on, off, or don't care (default)",
    )
    parser.add_argument(
        "--no-require-arch-kernels",
        dest="require_arch_kernels",
        action="store_false",
        help="skip the arch-specific codegen gate (repack-only check)",
    )
    parser.add_argument(
        "--locate-offset",
        action="append",
        default=[],
        metavar="OFFSET",
        help="file offset from a kernel crash line (libggml-cpu.so[51057,...]); repeatable",
    )
    parser.add_argument("--nm", default=os.environ.get("GGML_CENSUS_NM", "nm"))
    parser.add_argument("--objdump", default=os.environ.get("GGML_CENSUS_OBJDUMP", "objdump"))
    args = parser.parse_args(argv)

    try:
        with open(args.library, "rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        return _tool_error(f"cannot read {args.library}: {exc}")
    if magic != _ELF_MAGIC:
        return _tool_error(f"{args.library} is not an ELF object (magic={magic!r})")

    rc, nm_output = _run_tool(args.nm, ["-C", "--defined-only", args.library])
    if rc != 0:
        return _tool_error(f"{args.nm} failed (rc={rc}) on {args.library}")
    symbols = parse_symbols(nm_output)
    if not symbols:
        return _tool_error(
            f"no symbols in {args.library} (stripped artifact?): the census needs an "
            "unstripped libggml-cpu"
        )

    names = {symbol.name for symbol in symbols}
    disassembly: dict[str, str] = {}

    def disassemble(target: str) -> None:
        rc, output = _run_tool(args.objdump, ["-d", f"--disassemble={target}", args.library])
        if rc == 0 and output.strip():
            disassembly[target] = output

    for kernel in HOT_KERNELS:
        if kernel in names:
            disassemble(kernel)

    # The repack-registration site lives in a C++ function, so objdump needs the
    # mangled symbol: resolve it from the plain (undemangled) symbol table.
    if any(EXTRA_BUFFER_TYPES_SYMBOL in name for name in names):
        rc, raw_nm = _run_tool(args.nm, ["--defined-only", args.library])
        mangled = None
        if rc == 0:
            for line in raw_nm.splitlines():
                parts = line.strip().split(None, 2)
                if len(parts) == 3 and _is_hex(parts[0]) and EXTRA_BUFFER_TYPES_SYMBOL in parts[2]:
                    mangled = parts[2]
                    break
        disassemble(mangled or f"{EXTRA_BUFFER_TYPES_SYMBOL}()")
        if mangled:
            disassembly[EXTRA_BUFFER_TYPES_SYMBOL] = disassembly.pop(mangled)

    report = classify(
        nm_output,
        disassembly,
        require_arch_kernels=args.require_arch_kernels,
        expect_repack=args.expect_repack,
        library=args.library,
    )

    if args.locate_offset:
        offsets = []
        for raw in args.locate_offset:
            try:
                offsets.append(int(raw, 16))
            except ValueError:
                return _tool_error(f"--locate-offset expects hex, got {raw!r}")
        report["offsets"] = locate_offsets(nm_output, offsets)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["verdict"] != "ok":
            # Keep the readable reasons next to the machine-readable report so a
            # caller that captures stdout still shows the operator why it refused.
            print(format_report(report), file=sys.stderr)
    else:
        print(format_report(report))

    return 0 if report["verdict"] == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
