#!/usr/bin/env python3
"""Apply / verify a slot-geometry A/B arm (docs/dev/slot-geometry-ab.md).

This is the control side of the slot-geometry experiment (child
LP-0MVNCMDW50043RSP of LP-0MVMHNUXW0061FGN). Arm A is the current 2 x
131,072 geometry; Arm B is the candidate 1 x 262,144. Per-slot context is
derived, never configured directly: llama-server divides ``--ctx-size`` by
``--parallel``; the proxy lifecycle exports ``session_slot_pool_size`` as
``LLAMA_PARALLEL`` (proxy/proxy/lifecycle.py), so changing the slot count
means patching that config key and restarting the proxy.

Usage:
  slot_geometry_arm.py show [--config proxy/config.yaml] [--json]   # both arms + live config
  slot_geometry_arm.py apply --arm A|B [--config proxy/config.yaml] [--write]
  slot_geometry_arm.py verify --arm A|B [--llama-url http://127.0.0.1:8080] \\
      [--model Qwen3] [--config ...] [--json]

Exit codes:
  0 - ok (apply dry-run prepared / verify matched)
  1 - mismatch, missing key, unknown arm, unreachable server
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

#: Output headroom reserved for generated tokens — mirrors
#: ``proxy/provider.py:_LOCAL_ROUTING_OUTPUT_HEADROOM`` (LP-0MSAZXXDY005AWA1);
#: a prompt above ``ctx_size // slots - headroom`` cannot fit the KV slot.
OUTPUT_HEADROOM = 4096

# The two arm definitions (the only place the geometry is encoded; the
# reporter and the protocol doc read from here so they cannot drift).
# Warm threshold for Arm B is the physical single-slot clamp (258,048) with
# the economic-ceiling ratio raised alongside it (LP-0MVHSET0A008VU95:
# setting the absolute warm cap above the ratio-derived ceiling is a no-op).
ARMS: dict[str, dict] = {
    "A": {
        "label": "2 x 131072 (current)",
        "parallel": 2,
        "ctx_size": 262144,
        "config": {
            "session_slot_pool_size": "2",
            "local_large_context_warm_cache_threshold": "120000",
            "local_large_context_economic_ceiling_ratio": "0.4578",
            "local_large_context_cold_cache_threshold": "38000",
        },
    },
    "B": {
        "label": "1 x 262144",
        "parallel": 1,
        "ctx_size": 262144,
        "config": {
            "session_slot_pool_size": "1",
            "local_large_context_warm_cache_threshold": "258048",
            "local_large_context_economic_ceiling_ratio": "0.9844",
            "local_large_context_cold_cache_threshold": "38000",
        },
    },
}

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "proxy" / "config.yaml"


def arm_geometry(arm: str) -> dict:
    """Resolve one arm's derived geometry (raises KeyError for unknown arms)."""
    if arm not in ARMS:
        raise KeyError(
            f"unknown arm {arm!r} (choose from {', '.join(sorted(ARMS))})"
        )
    spec = ARMS[arm]
    per_slot = spec["ctx_size"] // spec["parallel"]
    clamp = per_slot - OUTPUT_HEADROOM
    ratio = float(spec["config"]["local_large_context_economic_ceiling_ratio"])
    warm = int(spec["config"]["local_large_context_warm_cache_threshold"])
    cold = int(spec["config"]["local_large_context_cold_cache_threshold"])
    economic = min(warm, round(ratio * spec["ctx_size"]))
    effective_warm = min(clamp, economic)
    return {
        "label": spec["label"],
        "ctx_size": spec["ctx_size"],
        "slots": spec["parallel"],
        "per_slot_ctx": per_slot,
        "per_slot_route_clamp": clamp,
        "effective_warm_threshold": effective_warm,
        "cold_threshold": cold,
        "band_width": max(effective_warm - cold, 0),
        "config": dict(spec["config"]),
        "llama_server_flags": ["--parallel", str(spec["parallel"]), "--ctx-size",
                               str(spec["ctx_size"])],
    }


class GeometryProbeError(RuntimeError):
    """llama-server geometry could not be read (unreachable or unparseable)."""


def probe_llama_geometry(
    base_url: str, model: str | None = None, timeout: float = 5.0
) -> dict:
    """Read the slot count and per-slot context from a running llama-server.

    Tries ``GET /slots`` first (per-slot ``n_ctx`` is the ground truth; if the
    slots disagree the probe reports the mismatch instead of guessing), then
    falls back to ``GET /props`` (``total_slots`` +
    ``default_generation_settings.n_ctx``) for builds that hide ``/slots``.

    Returns ``{"slots": N, "per_slot_ctx": M|None, "source": "slots"|"props",
    "per_slot_ctx_list": [...]}``. Raises :class:`GeometryProbeError` when the
    server cannot be read.
    """
    base = base_url.rstrip("/")
    try:
        slots_url = f"{base}/slots"
        if model:
            slots_url += f"?model={urllib_request.quote(model)}"
        with urllib_request.urlopen(slots_url, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        if isinstance(payload, list) and payload:
            n_ctx_list = [int(s.get("n_ctx") or 0) for s in payload]
            per_slot = n_ctx_list[0] if len(set(n_ctx_list)) == 1 else None
            return {
                "slots": len(payload),
                "per_slot_ctx": per_slot,
                "per_slot_ctx_list": n_ctx_list,
                "source": "slots",
            }
    except (urllib_error.URLError, urllib_error.HTTPError, OSError,
            ValueError, TypeError, KeyError):
        pass

    try:
        props_url = f"{base}/props"
        if model:
            props_url += f"?model={urllib_request.quote(model)}"
        with urllib_request.urlopen(props_url, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        total = int(payload.get("total_slots") or 0)
        n_ctx = int((payload.get("default_generation_settings") or {}).get("n_ctx") or 0)
        if total <= 0:
            raise GeometryProbeError(
                f"{props_url} returned total_slots={total}"
            )
        return {
            "slots": total,
            "per_slot_ctx": n_ctx if n_ctx > 0 else None,
            "per_slot_ctx_list": [n_ctx] * total,
            "source": "props",
        }
    except (urllib_error.URLError, urllib_error.HTTPError, OSError,
            ValueError, TypeError) as exc:
        raise GeometryProbeError(
            f"could not read llama-server geometry from {base}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# config.yaml patching (text-preserving: comments and other keys survive)
# ---------------------------------------------------------------------------

_CONFIG_KEY_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z0-9_]+):(?P<rest>[ \t]*)(?P<value>\S.*)$")


def _patch_config(text: str, arm_spec: dict) -> list[dict]:
    """Return (and apply to a copy of) the changes the arm needs.

    Only the declared geometry keys are touched; a key missing from the file
    is a loud failure (config drift must never be papered over).
    """
    changes: list[dict] = []
    for key, new_value in arm_spec["config"].items():
        pattern = re.compile(rf"^([ \t]*{re.escape(key)}:[ \t]*)(\S.*)$", re.MULTILINE)
        match = pattern.search(text)
        if not match:
            raise KeyError(
                f"geometry key {key!r} not found in config.yaml — refusing to "
                f"apply {arm_spec['label']} to a config that drifted"
            )
        old_value = match.group(2)
        changes.append({"key": key, "from": old_value, "to": new_value,
                        "change": old_value != new_value})
    return changes


def apply_arm(arm: str, config_path: Path, write: bool) -> dict:
    """Prepare (or, with ``write``, apply) an arm's config patch."""
    geometry = arm_geometry(arm)
    text = config_path.read_text(encoding="utf-8")
    changes = _patch_config(text, ARMS[arm])
    result = {
        "arm": arm,
        "arm_label": geometry["label"],
        "written": False,
        "changes": changes,
        "llama_server_flags": geometry["llama_server_flags"],
        "restart_command": "Restart the proxy to activate the new geometry: "
        "/skill:start-proxy --restart is the controlled teardown + relaunch.",
        "revert_hint": "git checkout -- proxy/config.yaml  (or apply --arm A)",
    }
    if write:
        patched = text
        for change in changes:
            if not change["change"]:
                continue
            patched = re.sub(
                rf"^([ \t]*{re.escape(change['key'])}:[ \t]*)\S.*$",
                lambda m, c=change: m.group(1) + c["to"],
                patched,
                count=1,
                flags=re.MULTILINE,
            )
        config_path.write_text(patched, encoding="utf-8")
        result["written"] = True
    return result


def verify_arm(
    arm: str, probe: dict, config_path: Path | None, base_url: str, model: str
) -> dict:
    """Verify measured reality against the arm's expected geometry.

    Refuses success on any mismatch: slot count, per-slot context, and (when a
    config is given) the effective warm threshold must exceed the cold
    threshold, otherwise the routing band is collapsed and the cached-ratio
    check is unreachable (LP-0MSI2M5BT004BCDP).
    """
    expect = arm_geometry(arm)
    problems: list[str] = []

    if probe["slots"] != expect["slots"]:
        problems.append(
            f"slot count {probe['slots']} != expected {expect['slots']}"
        )
    if len(set(probe.get("per_slot_ctx_list") or [])) > 1:
        problems.append(
            f"per-slot n_ctx differs between slots: {probe['per_slot_ctx_list']}"
        )
    if probe["per_slot_ctx"] is not None:
        if probe["per_slot_ctx"] != expect["per_slot_ctx"]:
            problems.append(
                f"per-slot n_ctx {probe['per_slot_ctx']} != expected "
                f"{expect['per_slot_ctx']}"
            )

    if config_path is not None:
        text = config_path.read_text(encoding="utf-8")
        warm = None
        ratio = 0.0
        cold = None
        for key in (
            "local_large_context_warm_cache_threshold",
            "local_large_context_economic_ceiling_ratio",
            "local_large_context_cold_cache_threshold",
        ):
            m = re.search(rf"^[ \t]*{re.escape(key)}:[ \t]*(\S+)", text, re.MULTILINE)
            if not m:
                continue
            value = _as_number(m.group(1))
            if key.endswith("warm_cache_threshold"):
                warm = int(value)
            elif key.endswith("economic_ceiling_ratio"):
                ratio = float(value)
            elif key.endswith("cold_cache_threshold"):
                cold = int(value)
        if warm is None or cold is None:
            problems.append("config.yaml is missing warm/cold threshold keys")
        else:
            economic = min(warm, round(ratio * expect["ctx_size"]))
            effective = min(expect["per_slot_route_clamp"], economic)
            if effective <= cold:
                problems.append(
                    f"routing band is collapsed: effective warm {effective} "
                    f"<= cold {cold} (thresholds must be changed together, "
                    f"LP-0MSI2M5BT004BCDP)"
                )

    return {
        "arm": arm,
        "arm_label": expect["label"],
        "expected": {"slots": expect["slots"],
                     "per_slot_ctx": expect["per_slot_ctx"],
                     "per_slot_route_clamp": expect["per_slot_route_clamp"]},
        "measured": {"slots": probe["slots"],
                     "per_slot_ctx": probe["per_slot_ctx"]},
        "source": probe["source"],
        "base_url": base_url,
        "model": model,
        "problems": problems,
        "verified": not problems,
    }


def _as_number(text: str) -> float | int:
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return 0


def _render_verify(data: dict) -> str:
    flag = "OK" if data["verified"] else "MISMATCH"
    lines = [
        f"verify arm {data['arm']} ({data['arm_label']}) against {data['base_url']}",
        f"  expected: {data['expected']['slots']} slot(s) x "
        f"{data['expected']['per_slot_ctx']} ctx (clamp "
        f"{data['expected']['per_slot_route_clamp']})",
        f"  measured: {data['measured']['slots']} slot(s) x "
        f"{data['measured']['per_slot_ctx']} ctx (source: {data['source']})",
    ]
    for problem in data["problems"]:
        lines.append(f"  FAIL - {problem}")
    lines.append(f"  verdict: {flag}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    show = sub.add_parser("show", help="print both arm definitions and the live config geometry")
    show.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    show.add_argument("--json", action="store_true")

    apply = sub.add_parser("apply", help="patch config.yaml for an arm (dry-run by default)")
    apply.add_argument("--arm", choices=sorted(ARMS), required=True)
    apply.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    apply.add_argument("--write", action="store_true", help="actually write the patch")
    apply.add_argument("--json", action="store_true")

    verify = sub.add_parser("verify", help="verify live llama-server geometry matches an arm")
    verify.add_argument("--arm", choices=sorted(ARMS), required=True)
    verify.add_argument("--config", type=Path)
    verify.add_argument("--llama-url", default="http://127.0.0.1:8080")
    verify.add_argument("--model", default="Qwen3")
    verify.add_argument("--timeout", type=float, default=5.0)
    verify.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.mode == "show":
        live = None
        if args.config.is_file():
            text = args.config.read_text(encoding="utf-8")
            slots = _read_int(text, "session_slot_pool_size")
            ctx = _read_int(text, "local_model_ctx_size")
            if slots:
                live = {
                    "slots": slots,
                    "per_slot_ctx": ctx // slots if ctx else None,
                }
        data = {
            "arms": {name: arm_geometry(name) for name in sorted(ARMS)},
            "live": live,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if live is not None:
            data["live"]["matching_arm"] = next(
                (name for name in sorted(ARMS) if arm_geometry(name)["slots"] == live["slots"]),
                None,
            )
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            for name, geometry in data["arms"].items():
                print(
                    f"arm {name}: {geometry['slots']} slot(s) x "
                    f"{geometry['per_slot_ctx']} ctx  (clamp "
                    f"{geometry['per_slot_route_clamp']}, warm "
                    f"{geometry['effective_warm_threshold']}, cold "
                    f"{geometry['cold_threshold']}, band {geometry['band_width']})"
                )
            print(f"live config: {live}")
        return 0

    if args.mode == "apply":
        if not args.config.is_file():
            print(f"error: config not found: {args.config}", file=sys.stderr)
            return 1
        try:
            data = apply_arm(args.arm, args.config, args.write)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            verb = "APPLIED" if data["written"] else "DRY-RUN (no changes written)"
            print(f"arm {data['arm']} ({data['arm_label']}): {verb}")
            for c in data["changes"]:
                arrow = "->" if c["change"] else "=="
                print(f"  {c['key']}: {c['from']} {arrow} {c['to']}")
            print(f"  llama-server flags: {' '.join(data['llama_server_flags'])}")
            print(f"  restart: {data['restart_command']}")
            print(f"  revert:  {data['revert_hint']}")
        return 0

    if args.mode == "verify":
        try:
            probe = probe_llama_geometry(
                args.llama_url, model=args.model, timeout=args.timeout
            )
        except GeometryProbeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        data = verify_arm(
            args.arm, probe, args.config, args.llama_url, args.model
        )
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(_render_verify(data))
        return 0 if data["verified"] else 1

    return 1


def _read_int(text: str, key: str) -> int | None:
    m = re.search(rf"^[ \t]*{re.escape(key)}:[ \t]*(\S+)", text, re.MULTILINE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
