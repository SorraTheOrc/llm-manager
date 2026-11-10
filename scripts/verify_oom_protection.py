#!/usr/bin/env python3
"""Verify (and optionally reapply) the backend OOM ranking from ``/proc``.

LP-0MW3K7XJ30027RV6 (parent AC5). The proxy configures
``server.llama_server_oom_score_adj: 800`` so the model backend dies before the
front door if the kernel ever picks a victim. On 2026-11-05 the live router and
its per-model child both still read ``oom_score_adj=0``:

    $ for p in $(pgrep -x llama-server); do echo "$p $(cat /proc/$p/oom_score_adj)"; done
    2089209 0
    2089312 0

The cause was the start-script wrapper: ``start-llama.sh`` forks
``llama-server`` as a foreground child, and the kernel copies ``oom_score_adj``
at fork time, so a score written to the wrapper afterwards never reached the
backend. The proxy now scores the whole process tree at spawn time and
re-asserts it periodically; this tool is the operator-side proof, reading the
same ``/proc`` files ``ps``/``cat`` would.

Usage::

    # read-only: what does the kernel hold right now?
    python3 scripts/verify_oom_protection.py
    python3 scripts/verify_oom_protection.py --json

    # reapply the configured score to every llama-server process of a tree
    python3 scripts/verify_oom_protection.py --pid <spawn-wrapper-pid> --apply

Exit status: ``0`` when every discovered ``llama-server`` carries the configured
score, ``1`` otherwise (an unverified backend — the runbook documents the gap
and its remedy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The ``proxy`` package lives one level down (``<repo>/proxy/proxy``), as the
# repository's conftest does for the test suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy"))

from proxy.host_pressure import (
    LLAMA_SERVER_OOM_SCORE_ADJ,
    inspect_backend_oom_protection,
    reapply_backend_oom_protection,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Verify the llama-server oom_score_adj ranking from /proc "
            "(LP-0MW3K7XJ30027RV6)."
        )
    )
    parser.add_argument(
        "--pid",
        type=int,
        action="append",
        default=[],
        help=(
            "root pid to walk (repeatable) — the tracked spawn handle or a "
            "wrapper such as start-llama.sh. Default: discover the router by "
            "listening port."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="llama_server_port used for discovery when --pid is not given (default: 8080).",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=LLAMA_SERVER_OOM_SCORE_ADJ,
        help=(
            f"expected oom_score_adj (default: {LLAMA_SERVER_OOM_SCORE_ADJ}, "
            "matching server.llama_server_oom_score_adj)"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the expected score to every discovered llama-server process",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser.parse_args(argv)


def _roots(args) -> list[int]:
    if args.pid:
        return list(args.pid)
    try:
        from proxy.backend_rss import discover_llama_pids

        return list(discover_llama_pids(args.port))
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"warning: could not discover llama-server pids on port {args.port} "
            f"({exc})",
            file=sys.stderr,
        )
        return []


def main(argv=None) -> int:
    args = _parse_args(argv)
    roots = _roots(args)
    config = {"server": {"llama_server_oom_score_adj": args.expected}}
    if args.apply:
        report = reapply_backend_oom_protection(roots, config, require_backend=False)
    else:
        report = inspect_backend_oom_protection(roots, config)

    payload = report.as_dict()
    payload["roots"] = roots
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        walked = ", ".join(str(pid) for pid in roots) or "none"
        print(f"configured oom_score_adj: {report.configured}")
        print(f"roots walked: {walked}")
        if report.rows:
            print()
            print(f"{'pid':<8} {'comm':<16} {'score':<10} role")
            for row in report.rows:
                score = "unreadable" if row.applied is None else str(row.applied)
                role = "llama-server" if row.is_backend else "root/wrapper"
                comm = row.comm or "-"
                print(f"{row.pid:<8} {comm:<16} {score:<10} {role}")
            print()
        if report.protected:
            print(
                "PROTECTED: every llama-server process carries "
                f"oom_score_adj={report.configured}"
            )
        else:
            print(f"GAP: {report.gap or 'no backend process found'}")
            print(
                "remedy: python3 scripts/verify_oom_protection.py --apply "
                "(then re-check), or restart the backend so the proxy scores it "
                "at spawn time — see docs/dev/oom-recovery-runbook.md"
            )
    return 0 if report.protected else 1


if __name__ == "__main__":
    raise SystemExit(main())
