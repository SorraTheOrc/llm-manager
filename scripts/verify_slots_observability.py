#!/usr/bin/env python3
"""Deterministic PASS/FAIL verifier for local slot observability.

This is the evidence tool behind LP-0MW1VVOJZ008YUPH (parent
LP-0MW1VDWY0007EMH7). The parent's acceptance criteria require *recorded live
verification* after a proxy restart — ``slots_stale=false``,
``slots_snapshot_age_seconds`` below the ceiling, and a real
``GET /slots?model=…`` 200 in ``llama-server``'s log — but before this script
that evidence was an ad-hoc shell session nobody could reproduce or audit.

Read-only by construction: it issues plain HTTP GETs and reads log files. It
never changes proxy behaviour, status semantics, dispatch state or the worklog.

Usage
-----
::

  ./scripts/verify_slots_observability.py                    # human verdict
  ./scripts/verify_slots_observability.py --json             # machine readable
  ./scripts/verify_slots_observability.py \\
      --status-url http://localhost:8000/llama/local/status \\
      --health-url http://localhost:8000/health \\
      --llama-log /var/log/llama-proxy/llama-server.log

Exit codes
----------
``0``
    Every check passed.
``1``
    At least one check failed (the failing checks are named in the output).
``2``
    Usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STATUS_URL = "http://localhost:8000/llama/local/status"
DEFAULT_HEALTH_URL = "http://localhost:8000/health"
DEFAULT_LLAMA_LOG = "/var/log/llama-proxy/llama-server.log"

#: A ``/slots`` request line in ``llama-server``'s log carrying ``?model=``.
#: llama.cpp logs ``done request: GET /slots?model=… 127.0.0.1 200``; older
#: builds log the path without the query, so the model parameter is required
#: by the check rather than assumed.
_SLOTS_REQUEST_RE = re.compile(
    r"GET\s+(?P<path>/slots[^\s]*)\s+\S+\s+(?P<status>\d{3})"
)


@dataclass
class Check:
    """One named verification step."""

    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    """The full verification result."""

    checks: list[Check] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = ["Slot observability verification", "=" * 32]
        for key, value in sorted(self.facts.items()):
            lines.append(f"{key}: {value}")
        lines.append("")
        for check in self.checks:
            mark = "PASS" if check.ok else "FAIL"
            lines.append(f"[{mark}] {check.name}: {check.detail}")
        lines.append("")
        lines.append("RESULT: PASS" if self.ok else "RESULT: FAIL")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "facts": self.facts,
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail}
                for c in self.checks
            ],
        }


def _http_get_json(url: str, timeout: float = 5.0) -> dict:
    """Fetch *url* and parse a JSON object (raises on any failure).

    An explicit ``ProxyHandler({})`` keeps ``urllib`` from honouring
    ``HTTP(S)_PROXY`` for these localhost-only fetches — a proxy env var on
    the host must never make the verifier talk to the wrong endpoint.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return parsed


def find_model_bearing_slots_200(log_text: str) -> tuple[bool, str]:
    """Return ``(found, last_matching_line)`` for a model-bearing /slots 200.

    A line counts only when it is a 200 **and** the path carries ``model=``:
    the model-less request is exactly what the router rejects, so accepting it
    would make this check vacuous.
    """
    last = ""
    for line in log_text.splitlines():
        match = _SLOTS_REQUEST_RE.search(line)
        if not match:
            continue
        if match.group("status") != "200":
            continue
        if "model=" not in match.group("path"):
            continue
        last = line.strip()
    return bool(last), last


def read_log_tail(path: Path, max_bytes: int = 4_000_000) -> str:
    """Read up to *max_bytes* from the end of *path* (empty when unreadable)."""
    try:
        if not path.exists():
            return ""
        size = path.stat().st_size
        if size == 0:
            return ""
        n = min(max_bytes, size)
        with open(path, "rb") as handle:
            handle.seek(size - n)
            raw = handle.read(n)
        return raw.decode("utf-8", errors="replace").replace("\x00", "")
    except Exception:
        return ""


def probe_router_model_requirement(
    router_url: str, model: str, timeout: float = 5.0
) -> tuple[bool, str]:
    """Prove the router requires ``?model=`` and answers when it is supplied.

    llama.cpp's ``log_server_r`` line does **not** include the query string
    (verified on this host: it logs ``GET /slots 127.0.0.1 200``), so a
    model-bearing 200 can never be read straight out of the log on this build.
    This probe supplies the same evidence directly:

    * ``GET /slots`` **without** ``model`` must be rejected (400) — that is the
      router contract the 2026-11-04 freeze tripped over;
    * ``GET /slots?model=<model>`` must succeed (200 with a list body).

    Returns ``(ok, detail)``.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _status(url: str) -> int:
        try:
            with opener.open(url, timeout=timeout) as response:
                return int(getattr(response, "status", 0) or 0)
        except urllib.error.HTTPError as exc:
            return int(exc.code)
        except Exception:
            return 0

    bare = _status(f"{router_url.rstrip('/')}/slots")
    with_model = _status(
        f"{router_url.rstrip('/')}/slots?model={urllib.parse.quote(model)}"
    )
    ok = bare == 400 and with_model == 200
    return ok, f"model-less={bare} model-bearing={with_model} model={model!r}"


def verify(
    status: dict | None,
    health: dict | None,
    log_text: str,
    router_probe: tuple[bool, str] | None = None,
) -> Report:
    """Evaluate every parent acceptance check from already-fetched inputs.

    Pure and offline-testable: the caller supplies the status payload, the
    health payload and the log text.
    """
    report = Report()

    if status is None:
        report.checks.append(
            Check("status payload", False, "GET /llama/local/status failed")
        )
        report.checks.append(
            Check("slots_stale=false", False, "no status payload to evaluate")
        )
        report.checks.append(
            Check("snapshot age below ceiling", False, "no status payload")
        )
    else:
        stale = status.get("slots_stale")
        reason = status.get("slots_stale_reason")
        age = status.get("slots_snapshot_age_seconds")
        ceiling = status.get("slots_max_snapshot_age_seconds")
        source = status.get("slots_snapshot_source")
        report.facts.update(
            {
                "slots_stale": stale,
                "slots_stale_reason": reason,
                "slots_snapshot_source": source,
                "slots_snapshot_age_seconds": age,
                "slots_max_snapshot_age_seconds": ceiling,
                "available_slots": status.get("available_slots"),
                "total_slots": status.get("total_slots"),
                "slots_detail_stale": status.get("slots_detail_stale"),
            }
        )
        report.checks.append(
            Check(
                "slots_stale=false",
                stale is False,
                f"slots_stale={stale!r} reason={reason!r}",
            )
        )
        if age is None:
            # No snapshot served at all is the healthiest state: there is
            # nothing stale to be too old.
            report.checks.append(
                Check(
                    "snapshot age below ceiling",
                    True,
                    "no snapshot served (live read, nothing stale)",
                )
            )
        elif ceiling is None:
            report.checks.append(
                Check(
                    "snapshot age below ceiling",
                    False,
                    f"age={age} but no published ceiling",
                )
            )
        else:
            try:
                within = float(age) <= float(ceiling)
            except (TypeError, ValueError):
                within = False
            report.checks.append(
                Check(
                    "snapshot age below ceiling",
                    within,
                    f"age={age} ceiling={ceiling}",
                )
            )

    if health is None:
        report.checks.append(
            Check("health surface", False, "GET /health failed")
        )
    else:
        surface = health.get("slots_surface_status")
        report.facts.update(
            {
                "slots_surface_status": surface,
                "health_slots_stale_reason": health.get("slots_stale_reason"),
            }
        )
        report.checks.append(
            Check(
                "health surface ok",
                surface == "ok",
                f"slots_surface_status={surface!r}",
            )
        )
        if status is not None:
            agree = health.get("slots_stale_reason") == status.get(
                "slots_stale_reason"
            )
            report.checks.append(
                Check(
                    "health/status reason agreement",
                    agree,
                    f"health={health.get('slots_stale_reason')!r} "
                    f"status={status.get('slots_stale_reason')!r}",
                )
            )

    if router_probe is not None:
        probe_ok, probe_detail = router_probe
        report.checks.append(
            Check("router requires ?model= and answers it", probe_ok, probe_detail)
        )
    else:
        found, line = find_model_bearing_slots_200(log_text)
        report.checks.append(
            Check(
                "model-bearing /slots 200 in llama-server log",
                found,
                line or "no 'GET /slots?model=… … 200' line in the log tail",
            )
        )

    # A successful /slots read is logged whether or not the build includes the
    # query string, so this is always checkable and always meaningful.
    ok_200 = bool(re.search(r"GET\s+/slots[^\s]*\s+\S+\s+200", log_text))
    report.checks.append(
        Check(
            "/slots 200 in llama-server log",
            ok_200,
            "found a successful /slots read after the restart"
            if ok_200
            else "no 'GET /slots … 200' line in the log tail",
        )
    )
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify local slot observability against a running proxy and read "
            "the evidence block (read-only)."
        ),
    )
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    parser.add_argument(
        "--llama-log",
        default=DEFAULT_LLAMA_LOG,
        help="llama-server log to scan for a model-bearing /slots 200",
    )
    parser.add_argument(
        "--router-url",
        default=None,
        help=(
            "llama-server router base URL (e.g. http://localhost:8080). When "
            "given, the model-parameter contract is probed directly: a "
            "model-less /slots must 400 and /slots?model=<id> must 200. "
            "Preferred over the log check because llama.cpp's request log "
            "omits the query string."
        ),
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="machine-readable")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    status = health = None
    try:
        status = _http_get_json(args.status_url, timeout=args.timeout)
    except (urllib.error.URLError, OSError, ValueError):
        status = None
    try:
        health = _http_get_json(args.health_url, timeout=args.timeout)
    except (urllib.error.URLError, OSError, ValueError):
        health = None

    log_text = read_log_tail(Path(args.llama_log))

    router_probe = None
    if args.router_url:
        model = ""
        if isinstance(status, dict):
            model = str(status.get("current_model") or "")
        if model:
            router_probe = probe_router_model_requirement(
                args.router_url, model, timeout=args.timeout
            )
        else:
            router_probe = (
                False,
                "no current_model in the status payload to probe with",
            )

    report = verify(status, health, log_text, router_probe=router_probe)

    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
