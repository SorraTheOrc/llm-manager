#!/usr/bin/env python3
"""Controlled verification of slot save/restore + real cached_ratio after the
GPU-wedge fix (LP-0MS9GAN2P009KK6G / child LP-0MSATT52O008LD0U).

Runs a 5+ turn local session (fresh X-Session-Id, model=plan, modest context
under session_slot_max_prompt_tokens so slot persistence is enabled), captures
per-turn latency and the final usage chunk (prompt_tokens + cached_tokens).

Hard requirements enforced here:
  * EVERY turn must be served by the LOCAL provider (retries with fresh session
    ids until true — concurrent agent sessions can saturate the local pool).
  * The client echoes back the REAL assistant content so the proxy's session
    history matches and slot persistence / cache reuse stays enabled.
  * Requests include ``stream_options.include_usage`` so llama-server emits the
    final usage chunk carrying ``prompt_tokens_details.cached_tokens``.

Usage:
    python verify_session_caching.py [--url URL] [--turns TURNS] [--max-wait SECONDS]
"""

import argparse
import json
import subprocess
import sys
import time
import uuid

import httpx

LOG_PATH = "/var/log/llama-proxy/proxy.log"
SLOTS_URL = "http://127.0.0.1:8080/slots?model=Qwen3"


def free_slots() -> int:
    """Number of idle llama-server slots right now (via /slots endpoint)."""
    try:
        import httpx
        r = httpx.get(SLOTS_URL, timeout=5)
        d = r.json()
        slots = d if isinstance(d, list) else d.get("slots", [])
        return sum(1 for s in slots if not s.get("is_processing"))
    except Exception:
        return -1


def wait_for_free_slot(timeout_s: float = 900.0) -> bool:
    """Block until at least one llama-server slot is idle."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        n = free_slots()
        if n >= 1:
            return True
        if n < 0:
            # Endpoint unreachable — assume OK rather than spin forever
            return True
        time.sleep(10)
    return False


def local_streams_for(session_id: str) -> int:
    """Count 'Stream started: provider=local' lines for this session."""
    try:
        out = subprocess.run(
            ["grep", "Stream started: provider=local", LOG_PATH],
            capture_output=True, text=True, timeout=10,
        )
        return sum(1 for line in out.stdout.splitlines() if session_id in line)
    except Exception:
        return 0


def remote_streams_for(session_id: str) -> int:
    """Count 'Stream started:' lines for this session from non-local providers."""
    try:
        out = subprocess.run(
            ["grep", "Stream started: provider=", LOG_PATH],
            capture_output=True, text=True, timeout=10,
        )
        return sum(
            1 for line in out.stdout.splitlines()
            if session_id in line and "provider=local" not in line
        )
    except Exception:
        return 0


def run_session(url: str, model: str, turns: int) -> dict:
    session_id = str(uuid.uuid4())
    conversation = [
        {"role": "system", "content": "You are a helpful assistant. Be brief."}
    ]
    per_turn: list[dict] = []

    with httpx.Client(timeout=900.0) as client:
        for turn in range(1, turns + 1):
            # Wait until a slot is free so this turn actually hits local.
            if not wait_for_free_slot():
                raise RuntimeError(f"no free slot for turn {turn} within window")
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"Turn {turn}: Reply with exactly one short word: "
                        f"the number {turn} squared is what?"
                    ),
                }
            )
            payload = {
                "model": model,
                "messages": conversation,
                "max_tokens": 600,
                "stream": True,
                "stream_options": {"include_usage": True},
                # Qwen3 is a reasoning model; disable thinking so it emits
                # real content (keeps session history echo consistent).
                "chat_template_kwargs": {"enable_thinking": False},
            }
            headers = {"X-Session-Id": session_id}
            start = time.monotonic()
            usage = None
            content = ""
            with client.stream(
                "POST", f"{url}/v1/chat/completions", json=payload, headers=headers
            ) as resp:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except Exception:
                        continue
                    if isinstance(chunk, dict):
                        if isinstance(chunk.get("usage"), dict):
                            usage = chunk["usage"]
                        for choice in chunk.get("choices", []) or []:
                            delta = choice.get("delta") or {}
                            content += delta.get("content") or ""
            elapsed = time.monotonic() - start

            cached = 0
            if isinstance(usage, dict):
                details = usage.get("prompt_tokens_details") or {}
                cached = int(details.get("cached_tokens", 0) or 0)
            prompt = int(usage.get("prompt_tokens", 0) or 0) if usage else 0
            ratio = (cached / prompt) if prompt > 0 else 0.0

            per_turn.append(
                {
                    "turn": turn,
                    "elapsed_s": round(elapsed, 3),
                    "prompt_tokens": prompt,
                    "cached_tokens": cached,
                    "cached_ratio": round(ratio, 4),
                    "assistant_chars": len(content),
                    "messages_sent": len(conversation) - 1,
                }
            )
            # Echo REAL content so proxy history matches and slot cache stays valid.
            conversation.append(
                {"role": "assistant", "content": content.strip() or "ok"}
            )
            print(
                f"turn={turn} elapsed={elapsed:.2f}s prompt={prompt} "
                f"cached={cached} ratio={ratio:.4f} chars={len(content)}"
            )
            time.sleep(2)

    return {"session_id": session_id, "per_turn": per_turn}


def run_session_with_retry(url: str, model: str, turns: int, max_attempts: int = 40) -> dict:
    """Run the session until EVERY turn was served by the LOCAL provider."""
    for attempt in range(1, max_attempts + 1):
        results = run_session(url, model, turns)
        sid = results["session_id"]
        local_n = local_streams_for(sid)
        remote_n = remote_streams_for(sid)
        print(f"attempt {attempt}: local streams={local_n} remote streams={remote_n}")
        if local_n == turns and remote_n == 0:
            print(f"attempt {attempt}: ALL {turns} turns served by LOCAL provider")
            return results
        print(f"attempt {attempt}: fell to remote or partial local (local busy); retrying")
        time.sleep(30)
    raise RuntimeError("gave up: no local capacity within retry window")


def main():
    parser = argparse.ArgumentParser(description="Verify session caching (LP-0MS9GAN2P009KK6G)")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="plan")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run_session_with_retry(args.url, args.model, args.turns)
    if args.json:
        print(json.dumps(results, indent=2))
    print(f"\nSESSION_ID={results['session_id']}")


if __name__ == "__main__":
    main()
