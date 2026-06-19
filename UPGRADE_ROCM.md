ROCm Upgrade: Preflight and quick runbook

This document describes the preflight checks for upgrading ROCm on the host and validating container compatibility.

Preflight script

- scripts/rocm-preflight.sh --dry-run --json
  - Produces a JSON summary with the following keys:
    - ok: boolean — overall pass/fail for basic checks
    - repo_present: boolean — whether /etc/apt/sources.list.d/rocm.list exists
    - repo_versions: array — candidate ROCm versions parsed from the repo file
    - rocm_smi_present: boolean — whether rocm-smi is available
    - rocm_smi_version: string — captured rocm-smi version/tag (if present)
    - uname: string — kernel uname -a
    - gpu_detected: boolean — whether an AMD GPU was detected via lspci
    - gpu_lines: array — matching lspci lines
    - errors: array — non-fatal issues observed during preflight

Usage examples

- Human-readable check:
  scripts/rocm-preflight.sh --dry-run

- Machine-readable JSON summary:
  scripts/rocm-preflight.sh --dry-run --json

Notes

- The preflight script is deliberately non-destructive and will not attempt to modify APT configuration or install packages.
- The script is intended to be run locally on the host where the upgrade will be performed.
- For full upgrade automation see scripts/install-rocm.sh (TBD).
