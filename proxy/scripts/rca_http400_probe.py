#!/usr/bin/env python3
"""RCA probe: reproduce remote HTTP 400 rejections on real endpoints.

Work item LP-0MSC4UJXU008HVV5 (parent LP-0MSC1BNP90017L9K).

Probes the real remote chat-completions endpoints used in the plan/author/code
fallback chains with minimal synthetic payloads that mirror the recorded
tool-call-turn message shapes:

  - assistant with ``content: null`` + ``tool_calls`` (+ ``reasoning_content``)
  - tool messages with / without matching ``tool_call_id``
  - tool_calls entries with missing ``id``/``type``
  - truncated ``function.arguments`` JSON

Bound: ~20 requests total (respecting provider rate limits). No live recorded
traffic is replayed — only synthetic minimal payloads (operator decision).

API keys are resolved exactly like the proxy: from ``~/.pi/agent/auth.json``
(via the same resolution order as ``proxy_remote._try_pi_auth_json``).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

AUTH_PATH = Path.home() / ".pi" / "agent" / "auth.json"

# Endpoints mirroring proxy/config.yaml plan/author/code remote providers.
# key_lookup mirrors the api_key_env names used in config.
ENDPOINTS = [
    {
        "name": "opencode-deepseek-free",
        "provider": "opencode",
        "url": "https://opencode.ai/zen/v1/chat/completions",
        "key_env": "OPENCODE_API_KEY",
        "model": "deepseek-v4-flash-free",
    },
    {
        "name": "opencode-go-deepseek",
        "provider": "opencode-go",
        "url": "https://opencode.ai/zen/go/v1/chat/completions",
        "key_env": "OPENCODE_API_KEY",
        "model": "deepseek-v4-flash",
    },
    {
        "name": "deepseek-v4-flash",
        "provider": "deepseek",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-v4-flash",
    },
]

# Minimal synthetic payload shapes mirroring recorded tool-call turns.
# "control" is a well-formed OpenAI-compatible sequence (should pass).
def _shape_control() -> dict:
    return {
        "model": "REPLACE_MODEL",
        "messages": [
            {"role": "user", "content": "Say hi."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_01", "type": "function",
                 "function": {"name": "say_hi", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_01", "content": "hi!"},
            {"role": "user", "content": "Thanks."},
        ],
        "max_tokens": 16,
    }


def _shape_content_null_toolcalls() -> dict:
    """Recorded shape: assistant content=null + tool_calls (no reasoning_content)."""
    payload = _shape_control()
    payload["messages"][1]["content"] = None
    return payload


def _shape_content_null_reasoning_toolcalls() -> dict:
    """Exact recorded shape: content=null + reasoning_content + tool_calls."""
    payload = _shape_control()
    payload["messages"][1]["content"] = None
    payload["messages"][1]["reasoning_content"] = "The user says hi, so I will call say_hi."
    return payload


def _shape_missing_tool_call_id() -> dict:
    """Tool message without tool_call_id (dangling reference)."""
    payload = _shape_control()
    tool_msg = payload["messages"][2]
    del tool_msg["tool_call_id"]
    return payload


def _shape_mismatched_tool_call_id() -> dict:
    """Tool message referencing a tool_call_id that no assistant message declares."""
    payload = _shape_control()
    payload["messages"][2]["tool_call_id"] = "call_NOPE"
    return payload


def _shape_missing_id_type() -> dict:
    """tool_calls entries missing id and type (function object only)."""
    payload = _shape_control()
    tc = payload["messages"][1]["tool_calls"][0]
    del tc["id"]
    del tc["type"]
    return payload


def _shape_truncated_arguments() -> dict:
    """function.arguments JSON truncated mid-string (invalid JSON)."""
    payload = _shape_control()
    payload["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"x": "unterminated'
    return payload


def _shape_empty_tool_calls() -> dict:
    """assistant message with empty tool_calls list."""
    payload = _shape_control()
    payload["messages"][1]["tool_calls"] = []
    return payload


SHAPES = {
    "control": _shape_control,
    "content_null_toolcalls": _shape_content_null_toolcalls,
    "content_null_reasoning_toolcalls": _shape_content_null_reasoning_toolcalls,
    "missing_tool_call_id": _shape_missing_tool_call_id,
    "mismatched_tool_call_id": _shape_mismatched_tool_call_id,
    "missing_id_type": _shape_missing_id_type,
    "truncated_arguments": _shape_truncated_arguments,
    "empty_tool_calls": _shape_empty_tool_calls,
}


def _load_auth_keys() -> dict:
    """Load API keys from ~/.pi/agent/auth.json (mirror of proxy resolution)."""
    if not AUTH_PATH.exists():
        return {}
    try:
        data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    keys = {}
    for name, entry in (data or {}).items():
        if isinstance(entry, dict) and entry.get("type") == "api_key" and entry.get("key"):
            keys[name.lower()] = str(entry["key"])
    return keys


def _resolve_key(key_env: str, auth_keys: dict) -> str | None:
    """Mirror proxy_remote._try_pi_auth_json resolution."""
    if key_env:
        val = os.environ.get(key_env)
        if val:
            return val
    lookup = key_env.lower() if key_env else ""
    if lookup == "opencode_api_key":
        for preferred in ("opencode-go", "opencode"):
            if preferred in auth_keys:
                return auth_keys[preferred]
    if lookup in auth_keys:
        return auth_keys[lookup]
    if lookup.endswith("_api_key"):
        stem = lookup[:-8]
        if stem in auth_keys:
            return auth_keys[stem]
    return None


def _redact(text: str) -> str:
    """Redact anything that looks like a bearer token in a response body."""
    for token in ("Bearer ", "sk-", "ghu_"):
        text = text.replace(token, token[:1] + "***")
    return text


def main() -> int:
    auth_keys = _load_auth_keys()
    if not auth_keys:
        print("ERROR: no API keys found in %s" % AUTH_PATH, file=sys.stderr)
        return 2

    out = {"endpoints": [], "results": [], "rejected_matrix": {}}
    total_requests = 0
    max_requests = 20

    for ep in ENDPOINTS:
        key = _resolve_key(ep["key_env"], auth_keys)
        if not key:
            print(f"SKIP {ep['name']}: no key for {ep['key_env']}")
            continue
        ep_record = {"name": ep["name"], "url": ep["url"], "model": ep["model"], "probes": []}
        out["endpoints"].append(ep_record)
        print(f"\n### {ep['name']} ({ep['url']}) model={ep['model']}")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=15.0)) as client:
            for shape_name, shape_fn in SHAPES.items():
                if total_requests >= max_requests:
                    print("Reached probe bound; stopping.", file=sys.stderr)
                    break
                payload = shape_fn()
                payload["model"] = ep["model"]
                payload["stream"] = False
                total_requests += 1
                try:
                    resp = client.post(ep["url"], json=payload, headers=headers)
                    body_text = _redact(resp.text[:600])
                    record = {
                        "endpoint": ep["name"],
                        "shape": shape_name,
                        "status": resp.status_code,
                        "body": body_text,
                    }
                    ep_record["probes"].append(record)
                    out["results"].append(record)
                    marker = "OK " if resp.status_code == 200 else "FAIL"
                    print(f"  [{marker}] {shape_name:38s} -> {resp.status_code}")
                    if resp.status_code != 200:
                        print(f"         body: {body_text[:300]}")
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "endpoint": ep["name"],
                        "shape": shape_name,
                        "status": "error",
                        "body": f"{type(exc).__name__}: {exc}",
                    }
                    ep_record["probes"].append(record)
                    out["results"].append(record)
                    print(f"  [ERR ] {shape_name:38s} -> {type(exc).__name__}: {exc}")
                time.sleep(1.0)  # respect rate limits between probes

    # Build rejected-shape matrix: shape -> set of endpoints that rejected it
    for r in out["results"]:
        if r["status"] != 200:
            out["rejected_matrix"].setdefault(r["shape"], []).append(r["endpoint"])

    report_path = Path(__file__).with_name("rca_http400_probe_report.json")
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(f"Total requests: {total_requests}")
    print("Rejected shapes:", json.dumps(out["rejected_matrix"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
