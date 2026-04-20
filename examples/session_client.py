#!/usr/bin/env python3
"""
Session-based client example for llama-server via the proxy.

Demonstrates how to use X-Session-Id headers for incremental prompt
ingestion, reducing CPU and latency for multi-turn conversations.

Usage:
    python session_client.py [--url URL] [--model MODEL] [--turns TURNS]

Examples:
    # 3-turn conversation with default model
    python session_client.py

    # 5 turns with specific model
    python session_client.py --model qwen3 --turns 5

    # Specify proxy URL
    python session_client.py --url http://localhost:8000
"""

import argparse
import json
import sys
import time
from typing import Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install with: pip install httpx")
    sys.exit(1)


def make_request(
    url: str,
    messages: list[dict],
    model: str,
    session_id: Optional[str] = None,
    stream: bool = False,
) -> tuple[dict, dict, float]:
    """Send a chat completion request with optional session header.

    Args:
        url: Base URL of the proxy (e.g., http://localhost:8000)
        messages: List of message dicts (role, content)
        model: Model name to use
        session_id: Optional session ID for incremental ingestion
        stream: Whether to stream the response

    Returns:
        Tuple of (response_json, response_headers, elapsed_seconds)
    """
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["X-Session-Id"] = session_id

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 100,
        "stream": stream,
    }

    endpoint = f"{url}/v1/chat/completions"
    start = time.monotonic()

    with httpx.Client(timeout=120.0) as client:
        response = client.post(endpoint, json=payload, headers=headers)

    elapsed = time.monotonic() - start
    response_headers = dict(response.headers)
    try:
        response_json = response.json()
    except Exception:
        response_json = {"error": response.text}

    return response_json, response_headers, elapsed


def run_session_conversation(
    url: str, model: str, turns: int, session_id: Optional[str] = None
) -> None:
    """Run a multi-turn conversation using session-based incremental ingestion.

    Args:
        url: Base URL of the proxy
        model: Model name to use
        turns: Number of conversation turns
        session_id: Optional session ID (auto-generated if None)
    """
    conversation: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."}
    ]

    print(f"\n{'='*60}")
    print(f"Session Client Example")
    print(f"URL: {url}  Model: {model}  Turns: {turns}")
    print(f"{'='*60}\n")

    # First request: no session, so proxy generates one
    session_header = session_id
    total_tokens_sent = 0
    total_time = 0.0

    for turn in range(1, turns + 1):
        user_msg = f"This is turn {turn}. Please summarize what we've discussed so far in one sentence."
        conversation.append({"role": "user", "content": user_msg})

        # Count message tokens (heuristic)
        msg_count = len(conversation)
        approx_tokens = sum(len(m["content"]) // 4 for m in conversation)
        total_tokens_sent += approx_tokens

        print(f"--- Turn {turn} ---")
        print(f"Sending {msg_count} messages (~{approx_tokens} tokens)")
        if session_header:
            print(f"Session ID: {session_header[:8]}...")

        response, headers, elapsed = make_request(
            url, conversation, model, session_id=session_header
        )

        # Check for session ID in response
        returned_session = headers.get("x-session-id")
        if returned_session and not session_header:
            session_header = returned_session
            print(f"New session created: {session_header[:8]}...")

        is_created = headers.get("x-session-created") == "true"
        is_delta = headers.get("x-session-delta") == "true"
        print(f"Session-Created: {is_created}  Session-Delta: {is_delta}")
        print(f"Time: {elapsed:.2f}s")

        # Extract assistant response
        try:
            assistant_content = response["choices"][0]["message"]["content"]
            conversation.append({"role": "assistant", "content": assistant_content})
            print(f"Response: {assistant_content[:100]}...")
        except (KeyError, IndexError):
            print(f"Error: {json.dumps(response)[:200]}")

        total_time += elapsed
        print()

    # Summary
    print(f"{'='*60}")
    print(f"Session Summary")
    print(f"  Session ID: {session_header}")
    print(f"  Total turns: {turns}")
    print(f"  Total messages in conversation: {len(conversation)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Approximate tokens sent: {total_tokens_sent}")
    print(f"{'='*60}")


def run_comparison(url: str, model: str, turns: int = 3) -> None:
    """Run a comparison: with-session vs without-session for the same conversation.

    This demonstrates the latency benefit of session-based incremental ingestion.
    """
    print(f"\n{'#'*60}")
    print(f"# COMPARISON: With Session vs Without Session")
    print(f"# {turns} turns each, model: {model}")
    print(f"{'#'*60}\n")

    # Without session (full history each time)
    print(">>> WITHOUT SESSION (re-sends full history each turn)")
    print("-" * 40)
    conversation: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."}
    ]
    times_no_session = []
    for turn in range(1, turns + 1):
        conversation.append({"role": "user", "content": f"Turn {turn}: Tell me about number {turn}."})
        response, _, elapsed = make_request(url, conversation, model, session_id=None)
        times_no_session.append(elapsed)
        try:
            assistant_content = response["choices"][0]["message"]["content"]
            conversation.append({"role": "assistant", "content": assistant_content})
        except (KeyError, IndexError):
            conversation.append({"role": "assistant", "content": "Error"})
        print(f"  Turn {turn}: {elapsed:.2f}s ({len(conversation)} messages)")

    # Reset conversation for session-based
    print(f"\n>>> WITH SESSION (incremental ingestion)")
    print("-" * 40)
    conversation2: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."}
    ]
    session_id = None
    times_with_session = []
    for turn in range(1, turns + 1):
        conversation2.append({"role": "user", "content": f"Turn {turn}: Tell me about number {turn}."})
        response, headers, elapsed = make_request(url, conversation2, model, session_id=session_id)
        if not session_id:
            session_id = headers.get("x-session-id")
        times_with_session.append(elapsed)
        try:
            assistant_content = response["choices"][0]["message"]["content"]
            conversation2.append({"role": "assistant", "content": assistant_content})
        except (KeyError, IndexError):
            conversation2.append({"role": "assistant", "content": "Error"})
        is_delta = headers.get("x-session-delta") == "true"
        print(f"  Turn {turn}: {elapsed:.2f}s (delta={is_delta}, {len(conversation2)} messages)")

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"  Without session: {', '.join(f'{t:.2f}s' for t in times_no_session)}")
    print(f"  With session:    {', '.join(f'{t:.2f}s' for t in times_with_session)}")
    avg_no = sum(times_no_session) / len(times_no_session)
    avg_with = sum(times_with_session) / len(times_with_session)
    print(f"  Avg without session: {avg_no:.2f}s")
    print(f"  Avg with session:    {avg_with:.2f}s")
    if avg_no > 0:
        reduction = ((avg_no - avg_with) / avg_no) * 100
        print(f"  Latency reduction:   {reduction:.1f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Session-based client for llama-server proxy")
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Base URL of the proxy (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model", default="gemma4",
        help="Model name to use (default: gemma4)"
    )
    parser.add_argument(
        "--turns", type=int, default=3,
        help="Number of conversation turns (default: 3)"
    )
    parser.add_argument(
        "--session-id", default=None,
        help="Custom session ID (default: auto-generated)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run comparison: with-session vs without-session"
    )

    args = parser.parse_args()

    if args.compare:
        run_comparison(args.url, args.model, args.turns)
    else:
        run_session_conversation(args.url, args.model, args.turns, args.session_id)


if __name__ == "__main__":
    main()