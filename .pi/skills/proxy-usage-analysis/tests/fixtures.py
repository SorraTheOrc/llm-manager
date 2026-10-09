"""Fixtures copied from real lines in /var/log/llama-proxy/proxy.log.

These lines preserve the exact structured INFO format the proxy emits
(``%Y-%m-%d %H:%M:%S,%f - LEVEL - message``). Request payloads are truncated
in the same way the log truncates them (the proxy cuts payloads off; the
analysis must tolerate this).
"""

from __future__ import annotations

# --- Stream started ---------------------------------------------------------

STREAM_STARTED_LOCAL = (
    "2026-08-02 13:58:32,260 - INFO - Stream started: provider=local model=Qwen3 "
    "session=019fc284-dcb8-74ca-9a64-9306b6f9d286 "
    "request=[{'type': 'text', 'text': '/plan SA-0MSAS108O009DYKT'}]"
)

STREAM_STARTED_REMOTE = (
    "2026-08-02 13:57:06,975 - INFO - Stream started: provider=opencode-go "
    "model=deepseek-v4-flash session=019fbfea-40b0-74a5-8cc9-bf3723f2b458 "
    "request=[{'type': 'text', 'text': 'The conversation history before this point was compac..."
)

STREAM_STARTED_DEEPSEEK = (
    "2026-08-02 14:52:41,214 - INFO - Stream started: provider=deepseek "
    "model=deepseek-v4-flash session=019fc2be-eb73-7830-a55e-c8bd5d21c927 "
    "request=[{{'type': 'text', 'text': 'Summarize the current state of the proxy...'}}]"
)

STREAM_STARTED_SESSION_UNKNOWN = (
    "2026-08-02 14:00:05,000 - INFO - Stream started: provider=opencode-go "
    "model=deepseek-v4-flash session=unknown request=[{'type': 'text', 'text': '...'}]"
)

# --- Stream finished --------------------------------------------------------

# tokens before session
STREAM_FINISHED_TOKENS_SESSION = (
    "2026-08-02 14:02:19,920 - INFO - Stream finished: reason=tool_calls "
    "tokens=43550/460/44010 session=019fc27d-3a46-7e5c-871e-57ab32f875f3 "
    "provider=opencode-go model=deepseek-v4-flash "
    "request=[{'type': 'text', 'text': '<skill name=\"implement\" location=\"/home/rgardler/.pi/..."
)

# session before tokens (different field order)
STREAM_FINISHED_SESSION_TOKENS = (
    "2026-08-02 14:03:08,814 - INFO - Stream finished: reason=tool_calls "
    "session=019fc27d-3a46-7e5c-871e-57ab32f875f3 "
    "tokens=52031/56/52087 provider=opencode-go model=deepseek-v4-flash "
    "request=[{'type': 'text', 'text': '<skill name=\"implement\" location=\"/home/rgardler/.pi/..."
)

# tokens, no session (remote-only responses are logged without a session UUID)
STREAM_FINISHED_TOKENS_NO_SESSION = (
    "2026-08-02 14:00:33,338 - INFO - Stream finished: reason=stop "
    "tokens=49364/6444/55808 provider=opencode-go model=deepseek-v4-flash "
    "request=[{'type': 'text', 'text': '<conversation>\\n[Assistant thinking]: The child OSL-0..."
)

# no tokens, with session (token counts are not always logged)
STREAM_FINISHED_NO_TOKENS = (
    "2026-08-02 13:59:04,893 - INFO - Stream finished: reason=tool_calls "
    "session=019fc27d-3a46-7e5c-871e-57ab32f875f3 provider=local model=Qwen3 "
    "request=[{'type': 'text', 'text': '<skill name=\"implement\" location=\"/home/rgardler/.pi/..."
)

STREAM_FINISHED_LOCAL = (
    "2026-08-02 13:59:07,344 - INFO - Stream finished: reason=tool_calls "
    "tokens=2430/120/2550 session=019fc284-dcb8-74ca-9a64-9306b6f9d286 "
    "provider=local model=Qwen3 request=[{'type': 'text', 'text': '/plan SA-0MSAS108O009DYKT'}]"
)

# --- Fallback triggered (never carries a session UUID) ----------------------

FALLBACK_CONCURRENCY = (
    "2026-08-02 13:58:04,301 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-deepseek, reason=local_concurrency_limit"
)

# Legacy reason value (pre-LP-0MSF8XDG7000PERM): rotated logs may still carry
# ``warm_cache_bypass``; the parser normalizes it to ``context_too_large``.
FALLBACK_WARM_CACHE = (
    "2026-08-02 13:57:06,974 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-deepseek, reason=warm_cache_bypass"
)

# Current reason value emitted by the proxy.
FALLBACK_CONTEXT_TOO_LARGE = (
    "2026-08-02 13:57:06,974 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-deepseek, reason=context_too_large"
)

FALLBACK_HTTP_400 = (
    "2026-08-02 14:02:13,126 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-deepseek, reason=HTTP 400"
)

# --- routing_skip_local (session + reason, arrow before the explanation) ----

# Legacy routing_skip line (pre-LP-0MSF8XDG7000PERM): ``warm_cache_bypass``
# is normalized to ``context_too_large`` by the parser.
ROUTING_SKIP_WARM = (
    "2026-08-02 14:02:20,157 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=40000 cold_threshold=39594 warm_threshold=39594 new_tokens=323 "
    "cached_ratio=0.99 reason=warm_cache_bypass → skipping local, routing to next remote "
    "provider session=019fc27d-3a46-7e5c-871e-57ab32f875f3"
)

# Current reason value emitted by the proxy.
ROUTING_SKIP_CONTEXT_TOO_LARGE = (
    "2026-08-02 14:02:20,157 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=40000 cold_threshold=39594 warm_threshold=39594 new_tokens=323 "
    "cached_ratio=0.99 reason=context_too_large → skipping local, routing to next remote "
    "provider session=019fc27d-3a46-7e5c-871e-57ab32f875f3"
)

ROUTING_SKIP_LARGE_CONTEXT = (
    "2026-08-02 14:30:00,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=150000 cold_threshold=39594 warm_threshold=39594 new_tokens=500 "
    "cached_ratio=0.00 reason=large_context_bypass → skipping local, routing to next remote "
    "provider session=aaaaaaaa-1111-2222-3333-444444444444"
)

# --- local_dispatch_denied --------------------------------------------------

DISPATCH_DENIED = (
    "2026-08-02 14:03:06,616 - INFO - local_dispatch_denied session=019fc245 "
    "owner=019fc27d active=4"
)

# --- Contention-queue events (LP-0MSORQVK50012Q4D F4 AC3) -------------------

CONTENTION_DISPATCH = (
    "2026-08-02 14:10:00,000 - INFO - contention_queue_dispatch provider=local-qwen3 "
    "session=019fc245-aaaa-7e5c-871e-57ab32f875f3 queued_duration=1.25s policy=queue depth=0"
)

CONTENTION_FALLBACK_AFTER_QUEUE = (
    "2026-08-02 14:11:00,000 - INFO - contention_queue_fallback_after_queue "
    "provider=local-qwen3 session=019fc245-bbbb-7e5c-871e-57ab32f875f3 "
    "queued_duration=60.00s"
)

# --- Error events (real lines from /var/log/llama-proxy, Aug 3 window) ------

# Stream finished with reason=error (client-visible synthetic error; carries
# session, provider, model, and the config entry name).
STREAM_FINISHED_ERROR = (
    "2026-08-03 10:13:14,159 - INFO - Stream finished: reason=error "
    "session=019fc52e-05a0-78d5-b59d-bcb91055b787 provider=opencode "
    "model=deepseek-v4-flash-free entry=opencode-deepseek-free "
    "request=[{'type': 'text', 'text': 'The conversation history before this point was compac..."
)

# Stream error (proxy-side stream exception; WARNING level).
STREAM_ERROR_LINE = (
    "2026-08-03 12:47:13,378 - WARNING - Stream error: "
    "session=019fc754-d847-75af-86ea-991480e799d0 provider=local model=Qwen3 error=NameError"
)

# slot_save failure (local llama-server slot persistence ReadTimeout).
SLOT_SAVE_FAILED = (
    "2026-08-03 13:39:43,255 - WARNING - slot_save failed slot=2 error=ReadTimeout/ReadTimeout"
)

# backend_retry (upstream connect/read timeout during retry backoff).
BACKEND_RETRY_TIMEOUT = (
    "2026-08-03 12:37:15,723 - WARNING - backend_retry path=v1/chat/completions stream=True "
    "attempt=1/8 delay=0.216s signal=connect_failures error=ConnectTimeout"
)

# Upstream HTTP 429 (FreeUsageLimitError).
UPSTREAM_429 = (
    "2026-08-03 13:58:04,053 - WARNING - [remote] upstream error status=429 "
    "url=https://opencode.ai/zen/v1/chat/completions "
    "body={\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\","
    "\"message\":\"Rate limit exceeded. Please try again later.\"},\"metadata\":{}}"
)

# Upstream HTTP 429 (GoUsageLimitError, account-level quota). The proxy takes
# the usage-limit reset quarantine path (_usage_limit_reset_seconds ->
# _usage_reset_at, LP-0MSLJPOCC0001ROJ) rather than the 3-hour per-model
# cooldown (LP-0MRGU0I91006ODFD). The parser extracts only error.type; the
# message reset duration / metadata.limitName are not parsed today.
UPSTREAM_429_GO_USAGE_LIMIT = (
    "2026-09-26 12:00:00,000 - WARNING - [remote] upstream error status=429 "
    "url=https://opencode.ai/zen/v1/chat/completions "
    "body={\"type\":\"error\",\"error\":{\"type\":\"GoUsageLimitError\","
    "\"message\":\"Monthly usage limit reached. Resets in 29 days.\"},"
    "\"metadata\":{\"limitName\":\"monthly\"}}"
)

# --- Lines the parser must ignore -------------------------------------------

# Operating-mode scheduler lines (LP-0MSM5K4TX004MICX): the applied-mode
# lines are parsed into a mode timeline (LP-0MSPZUD4G007IYGH); the
# ``enabled with N entries`` announcement carries no applied mode and is
# ignored.
MODE_SWITCH_CHEAP = (
    "2026-08-15 01:00:08,679 - INFO - Mode scheduler: applied scheduled mode cheap"
)
MODE_SWITCH_FAST = (
    "2026-08-15 10:00:18,041 - INFO - Mode scheduler: applied scheduled mode fast"
)
MODE_SCHEDULER_ENABLED_IGNORED = (
    "2026-08-15 10:00:26,165 - INFO - Mode scheduler: enabled with 2 entries: "
    "[('01:00', 'cheap'), ('10:00', 'fast')]"
)

# Manual mode switch (LP-0MT1EE315007AKXG): ``POST /admin/set-mode`` persists
# the new mode and restarts the proxy; grandfathering init then reports the
# ACTUALLY active mode as ``Grandfathering: enabled; other-mode config
# <file> (current=<mode>)`` (real lines from /var/log/llama-proxy at a
# 18:20 manual switch to cheap). The ``enabled with 2 entries`` announcement
# fires on BOTH scheduled and manual transitions, so it is NOT the signal;
# the ``(current=<mode>)`` field on the grandfathering line is. The
# ``restart_services: router-mode restart complete (N slots)`` line is
# corroborating evidence (never parsed on its own).
MANUAL_MODE_SWITCH_CHEAP = (
    "2026-08-19 18:20:16,661 - INFO - Mode scheduler: enabled with 2 entries: "
    "[('01:00', 'cheap'), ('10:00', 'fast')]"
)
MANUAL_MODE_SWITCH_GRANDFATHERING_CHEAP = (
    "2026-08-19 18:20:16,684 - INFO - Grandfathering: enabled; other-mode config "
    "config-fast.yaml (current=cheap)"
)
MANUAL_MODE_SWITCH_RESTART = (
    "2026-08-19 18:20:23,374 - INFO - restart_services: router-mode restart complete (2 slots)"
)

# status_request lines expose the running mode's total_slots and contention
# policy (real lines from /var/log/llama-proxy): used as corroborating
# evidence in the mode-bucketing regression fixture (never parsed).
STATUS_REQUEST_CHEAP = (
    "2026-08-15 07:00:19,968 - INFO - status_request active_query=true available_slots=0 "
    "client_ip=192.168.0.199 client_ip_source=direct client_port=57414 "
    "contention_queue_depth=1 contention_queue_policy=queue "
    "current_model=Qwen3 latency_ms=3024 "
    "llama_server_running=true local_active_query=true "
    "local_owner_lease_remaining_seconds=304.8219530270435 "
    "local_owner_session_id=01a003ed-2bce-7dc2-bf15-21fcba2411c9 "
    "model_switch_in_progress=false total_slots=2"
)
STATUS_REQUEST_FAST = (
    "2026-08-15 10:05:00,000 - INFO - status_request active_query=true available_slots=2 "
    "client_ip=192.168.0.191 client_ip_source=direct client_port=54852 "
    "contention_queue_depth=0 contention_queue_policy=fallback "
    "current_model=Qwen3 latency_ms=3025 "
    "llama_server_running=true local_active_query=true "
    "local_owner_lease_remaining_seconds=298.6892733310815 "
    "local_owner_session_id=01a003ed-2bce-7dc2-bf15-21fcba2411c9 "
    "model_switch_in_progress=false total_slots=3"
)

ROUTING_CHECK_IGNORED = (
    "2026-08-02 13:57:33,529 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=12217 cold_threshold=39594 warm_threshold=39594 new_tokens=12217 "
    "cached_ratio=0.00 messages=2 session=019fc28d-051b-75fd-9f88-1c336f6779e0"
)

SESSION_HEADER_IGNORED = (
    "2026-08-02 13:57:33,529 - INFO - Session header resolved: source=session_id "
    "session=019fc28d..."
)

LEASE_RENEWED_IGNORED = "2026-08-02 13:59:04,901 - INFO - lease_renewed session=019fc27d timeout=60s"

REQUEST_ROUTING_IGNORED = "2026-08-02 13:57:06,974 - INFO - Request routing: model=plan → local-qwen3"

WARNING_LINE_IGNORED = (
    "2026-08-02 14:02:11,371 - WARNING - Response truncated: finish_reason=length "
    "session=019fc27d-3a46-7e5c-871e-57ab32f875f3 model=Qwen3"
)

MALFORMED_LINE = "garbage without a timestamp prefix"

# Error-like lines that are NOT errors and must be ignored.
SLOT_SAVE_SUCCESS = (
    "2026-08-03 11:14:46,530 - INFO - slot_save success session=ad21516e slot=0"
)
STREAM_FINISHED_STOP = (
    "2026-08-03 11:14:46,531 - INFO - Stream finished: reason=stop "
    "session=019fc52e-05a0-78d5-b59d-bcb91055b787 provider=opencode "
    "model=deepseek-v4-flash-free entry=opencode-deepseek-free request=[]"
)

# --- llama-server eval-timing lines (real lines from /var/log/llama-proxy) ---
#
# Router-mode parent/child log: child lines carry a ``[<port>]`` prefix and
# the Qwen3 child port changes on every restart (discovered from the
# ``name=Qwen3 on port <port>`` spawn line). Eval timing lines carry no
# timestamp.

QWEN3_PORT = 32999

QWEN3_SPAWN_LINE = (
    "srv          load: spawning server instance with name=Qwen3 on port 32999"
)

PROMPT_EVAL_REAL = (
    "[32999] prompt eval time =   29504.01 ms / 11449 tokens "
    "(    2.58 ms per token,   388.05 tokens per second)"
)

DECODE_EVAL_REAL = (
    "[32999]        eval time =    3776.71 ms /   153 tokens "
    "(   24.68 ms per token,    40.51 tokens per second)"
)

PROMPT_EVAL_REAL2 = (
    "[32999] prompt eval time =     190.91 ms /    19 tokens "
    "(   10.05 ms per token,    99.53 tokens per second)"
)

DECODE_EVAL_REAL2 = (
    "[32999]        eval time =    1868.57 ms /    77 tokens "
    "(   24.27 ms per token,    41.21 tokens per second)"
)

# No port prefix (non-router / standalone llama-server output).
PROMPT_EVAL_NO_PORT = (
    "prompt eval time =     190.91 ms /    19 tokens "
    "(   10.05 ms per token,    99.53 tokens per second)"
)

# ``total time`` lines carry no tok/s and must be ignored.
TOTAL_TIME_LINE = (
    "[32999]       total time =   33280.72 ms / 11602 tokens"
)

# Non-eval llama-server lines the parser must ignore.
SLOT_PRINT_TIMING_IGNORED = "[32999] slot print_timing: id  3 | task 2 | "
SLOT_RELEASE_IGNORED = (
    "[32999] slot      release: id  3 | task 2 | stop processing: "
    "n_tokens = 11601, truncated = 0"
)
SRV_LOAD_IGNORED = "srv          load: spawning server instance with name=mxbai-embed on port 51973"
MALFORMED_EVAL_LINE = "[32999] eval time = not-a-number ms / many tokens"


# --- Config fixture (proxy/config.yaml fragment, real values) ---------------

CONFIG_FRAGMENT = """\
# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
default_model: code

server:
  # --- Routing thresholds -----------------------------------------------------
  llama_router_mode: true
  max_concurrent_queries: 16
  local_large_context_cold_cache_threshold: 60000
  local_large_context_warm_cache_threshold: 100000
  local_model_ctx_size: 262144
  local_dispatch_lease_timeout_seconds: 60
  session_single_flight_mode: queue
  session_slot_save_path: /home/rgardler/projects/llm/slot-cache
  # Server restart required to change this value (no hot reload).
  session_slot_pool_size: 6

  # -----------------------------------------------------------------------
  # Slot schedule (LP-0MRXZU90M007WNWT)
  # -----------------------------------------------------------------------
  slot_schedule:
    enabled: true
    drain_minutes: 3
    entries:
      - time: "23:59"
        slots: 8
      - time: "10:00"
        slots: 6

  session_slot_timeout_seconds: 3.0
  session_slot_max_prompt_tokens: 12288
  session_slot_timeout_per_token_seconds: 0.001
  session_slot_max_timeout_seconds: 60
"""

# ---------------------------------------------------------------------------
# Per-mode config profiles (mirror the real deployment, LP-0MSMZOAJW002UR2A):
# fast = 3 slots @ 131072 ctx; cheap = 2 slots @ 262144 ctx (per-period ctx
# pinned via slot_schedule ctx_size).
# ---------------------------------------------------------------------------

MODE_FAST_CONFIG = {
    "local_model_ctx_size": 131072,
    "session_slot_pool_size": 3,
    "slot_schedule": {
        "enabled": True,
        "entries": [("23:59", 3), ("10:00", 3)],
        "ctx_by_time": {"23:59": 131072, "10:00": 131072},
    },
}

MODE_CHEAP_CONFIG = {
    "local_model_ctx_size": 131072,
    "session_slot_pool_size": 2,
    "slot_schedule": {
        "enabled": True,
        "entries": [("23:59", 2), ("10:00", 2)],
        "ctx_by_time": {"23:59": 262144, "10:00": 262144},
    },
}


def mode_map_fixture() -> object:
    """Mode schedule map over the fixture profiles (imported lazily to avoid
    a hard dependency at module load)."""
    import bucketing

    return bucketing.ModeScheduleMap.from_profiles(
        {"fast": MODE_FAST_CONFIG, "cheap": MODE_CHEAP_CONFIG},
        analysis_mode="fast",
        default_slots=None,
    )


# --- Synthetic end-to-end fixture: two sessions across one rotated file boundary
# ---------------------------------------------------------------------------

# Session S1: local-only, starts 13:30 (before window start at 14:00), active in window.
S1 = "11111111-1111-1111-1111-111111111111"
# Session S2: local -> remote fallback, fully inside the window.
S2 = "22222222-2222-2222-2222-222222222222"

# --- Compaction events (LP-0MTHCTLAF00147IT) ------------------------------

COMPACTION_EVENT_FAST = (
    "2026-08-02 14:12:00,000 - INFO - compaction_event "
    "session=019fc2be-eb73-7830-a55e-c8bd5d21c927 "
    "mode=fast action=compact reason=context_window "
    "pre_tokens=55000 post_tokens=42000 turns_summarized=3 turns_dropped=1 "
    "summary_tokens=800 dry_run=false"
)

COMPACTION_EVENT_CHEAP = (
    "2026-08-02 14:12:30,000 - INFO - compaction_event "
    "session=019fc2be-eb73-7830-a55e-c8bd5d21c928 "
    "mode=cheap action=remote_with_guidance reason=context_window "
    "pre_tokens=60000 post_tokens=35000 turns_summarized=2 turns_dropped=0 "
    "summary_tokens=1200 dry_run=true"
)

COMPACTION_BACKSTOP = (
    "2026-08-02 14:13:00,000 - INFO - compaction_backstop "
    "action=dropped dropped_turns=1 dropped_messages=2 estimated_before=65000 estimated_after=48000"
)

COMPACTION_CHURN = (
    "2026-08-02 14:13:30,000 - INFO - compaction_churn "
    "session=019fc2be-eb73-7830-a55e-c8bd5d21c927 count=5 rate_per_hour=2.3 exceeds_target=true"
)

E2E_LINES = [
    # Rotated file content (13:29:00 - 14:00:00): S1 starts before the window.
    f"2026-08-02 13:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session={S1} request=[{{'type': 'text', 'text': 'first'}}]",
    f"2026-08-02 13:30:05,000 - INFO - Stream finished: reason=stop tokens=1200/50/1250 session={S1} provider=local model=Qwen3 request=[{{'type': 'text', 'text': 'first'}}]",
    # Live file content (14:00:00+): S1 continues, S2 starts local then falls back.
    f"2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session={S1} request=[{{'type': 'text', 'text': 'second'}}]",
    f"2026-08-02 14:00:12,000 - INFO - Stream finished: reason=stop tokens=1300/60/1360 session={S1} provider=local model=Qwen3 request=[{{'type': 'text', 'text': 'second'}}]",
    f"2026-08-02 14:01:00,000 - INFO - Stream started: provider=local model=Qwen3 session={S2} request=[{{'type': 'text', 'text': 'hello'}}]",
    f"2026-08-02 14:01:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session={S2} provider=local model=Qwen3 request=[{{'type': 'text', 'text': 'hello'}}]",
    FALLBACK_CONCURRENCY.replace("13:58:04,301", "14:01:06,000"),
    f"2026-08-02 14:01:06,100 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
    f"estimated_tokens=5000 cold_threshold=39594 warm_threshold=39594 new_tokens=50 cached_ratio=0.50 "
    f"reason=local_concurrency_limit → skipping local, routing to next remote provider session={S2}",
    f"2026-08-02 14:01:06,200 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session={S2} request=[{{'type': 'text', 'text': 'hello'}}]",
    f"2026-08-02 14:01:09,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session={S2} provider=opencode-go model=deepseek-v4-flash request=[{{'type': 'text', 'text': 'hello'}}]",
    f"2026-08-02 14:02:00,000 - INFO - local_dispatch_denied session=33333333-3333-3333-3333-333333333333 owner={S2} active=6",
]


# --- Restart markers and root-cause window (LP-0MUQO82LJ008F1FS) -------------
#
# A mode switch is a full proxy restart that kills in-flight local streams.
# These two markers bracket a restart (real log formats from /var/log/llama-proxy):
#   - ``Mode-switch restart spawned:`` (mode.py, immediately before the kill)
#   - ``restart_services: router-mode restart complete (N slots)`` (lifecycle.py)
from datetime import datetime  # noqa: E402  (kept beside its fixtures below)

RESTART_SCRIPT = "/home/rgardler/projects/llm/proxy/scripts/start-proxy.sh --restart"


def _restart_spawned(ts: str) -> str:
    return f"{ts} - INFO - Mode-switch restart spawned: {RESTART_SCRIPT}"


def _restart_complete(ts: str, slots: int = 2) -> str:
    return f"{ts} - INFO - restart_services: router-mode restart complete ({slots} slots)"


RESTART_SPAWNED_LINE = _restart_spawned("2026-09-25 17:27:58,000")
RESTART_COMPLETE_LINE = _restart_complete("2026-09-25 17:28:05,000")
RESTART_SERVICES_OTHER_LINE = (
    "2026-09-25 17:27:58,100 - INFO - restart_services: restarting llama-server "
    "with 2 slots (reason=mode-switch)"
)

# Real window: 2026-09-25 05:00 -> 2026-09-26 05:00 (archive
# ~/proxy-usage-reports/2026-09-27/). ``stream_finish_error`` split =
# opencode-go stall_after_content 24, local/Qwen3 stream_exception 6,
# deepseek/deepseek-flash stream_exception 1. The error events below are the
# real ones from that archive's errors.csv; the restart markers are synthetic
# (the source logs have rotated out) but mirror the documented correlation:
# the 10:00 scheduled switch killed 3, a 14:10:19 restart killed 1, and a
# 17:27:58 restart killed 2 at 17:28:00.
WINDOW_2026_09_25_START = datetime(2026, 9, 25, 5, 0, 0)
WINDOW_2026_09_25_END = datetime(2026, 9, 26, 5, 0, 0)

_STALL_MESSAGE = "Upstream idle timeout after content delivered (120s no data)"
_STALL_ACTION = "Retry the request with full context, or route to a healthier provider"
_LOCAL_MESSAGE = "Local stream error (RemoteProtocolError); llama-server may be unhealthy"
_LOCAL_ACTION = "Check llama-server logs; the request may be retried"
_REMOTE_MESSAGE = "Proxy stream error (ReadError); upstream may be unhealthy"
_REMOTE_ACTION = "Check proxy/upstream logs; the next provider in the chain may be used"


def _stream_finish_error_line(
    ts: str,
    session: str,
    provider: str,
    model: str,
    error_type: str,
    message: str,
    action: str,
    entry: str | None = None,
) -> str:
    parts = [
        f"{ts} - INFO - Stream finished: reason=error",
        f"error_type={error_type}",
        f"error_message={message}",
        f"suggested_action={action}",
        f"session={session}",
        f"provider={provider}",
        f"model={model}",
    ]
    if entry:
        parts.append(f"entry={entry}")
    parts.append("request=[{'type': 'text', 'text': 'Run /skill:implement ...'}]")
    return " ".join(parts)


# The 24 opencode-go stall_after_content errors: (timestamp, session).
STALL_AFTER_CONTENT_EVENTS_2026_09_25 = [
    ("2026-09-25 05:05:10,376", "herdr-1790290312-625941-2722"),
    ("2026-09-25 10:21:03,000", "herdr-1790326920-2116641-18897"),
    ("2026-09-25 10:22:04,000", "herdr-1790296145-891526-21621"),
    ("2026-09-25 12:10:01,000", "herdr-1790288091-387518-21192"),
    ("2026-09-25 13:42:27,000", "herdr-1790335197-2669902-22136"),
    ("2026-09-25 14:47:23,000", "herdr-1790341558-2918819-31343"),
    ("2026-09-25 14:50:25,000", "herdr-1790340629-2874806-25757"),
    ("2026-09-25 15:45:54,000", "herdr-1790341789-2937318-13409"),
    ("2026-09-25 16:29:17,000", "herdr-1790341789-2937318-13409"),
    ("2026-09-25 17:23:45,000", "herdr-1790342944-2985357-510"),
    ("2026-09-25 17:31:50,000", "herdr-1790345194-3185084-30107"),
    ("2026-09-25 17:33:51,000", "herdr-1790351245-3373146-17771"),
    ("2026-09-25 17:39:54,000", "herdr-1790350921-3365705-20893"),
    ("2026-09-25 18:13:13,000", "herdr-1790342944-2985357-510"),
    ("2026-09-25 18:39:26,000", "herdr-1790357073-3901802-23458"),
    ("2026-09-25 19:13:44,000", "herdr-1790357814-3908773-29464"),
    ("2026-09-25 19:22:48,000", "herdr-1790357881-3948427-30122"),
    ("2026-09-25 20:01:05,000", "herdr-1790357881-3948427-30122"),
    ("2026-09-25 20:20:17,000", "herdr-1790357881-3948427-30122"),
    ("2026-09-26 00:58:41,000", "herdr-1790358777-3972773-3112"),
    ("2026-09-26 01:13:49,000", "herdr-1790358777-3972773-3112"),
    ("2026-09-26 01:15:50,000", "herdr-1790381248-340648-20627"),
    ("2026-09-26 01:43:04,000", "herdr-1790382998-3974226-9682"),
    ("2026-09-26 02:01:14,000", "herdr-1790380029-315542-29623"),
]

# The 6 local/Qwen3 stream_exception errors, each within seconds of a restart.
LOCAL_RESTART_KILL_EVENTS_2026_09_25 = [
    ("2026-09-25 10:00:36,989", "herdr-1790326461-2110867-20865"),
    ("2026-09-25 10:00:36,997", "herdr-1790324216-2083810-13994"),
    ("2026-09-25 10:00:37,001", "herdr-1790325268-2096257-939"),
    ("2026-09-25 14:10:19,141", "herdr-1790340629-2874806-25757"),
    ("2026-09-25 17:28:00,333", "herdr-1790353495-3459907-22589"),
    ("2026-09-25 17:28:00,333", "herdr-1790350921-3365705-20893"),
]

# The single remote (deepseek/deepseek-flash) stream_exception, no restart near.
REMOTE_STREAM_EXCEPTION_EVENT_2026_09_25 = (
    "2026-09-26 00:53:09,679",
    "herdr-1790358777-3972773-3112",
)

RESTART_MARKERS_2026_09_25 = [
    _restart_spawned("2026-09-25 10:00:35,500"),
    _restart_spawned("2026-09-25 14:10:18,500"),
    _restart_spawned("2026-09-25 17:27:58,000"),
    _restart_complete("2026-09-25 17:28:05,000"),
]


def window_2026_09_25_lines() -> list[str]:
    """Raw log lines reproducing the 2026-09-25 -> 2026-09-26 window's
    ``stream_finish_error`` population (31 events: 24 / 6 / 1) plus the
    restart markers they correlate with."""
    lines = list(RESTART_MARKERS_2026_09_25)
    for ts, session in STALL_AFTER_CONTENT_EVENTS_2026_09_25:
        lines.append(
            _stream_finish_error_line(
                ts, session, "opencode-go", "deepseek-v4.1-flash",
                "stall_after_content", _STALL_MESSAGE, _STALL_ACTION,
                entry="opencode-go-3",
            )
        )
    for ts, session in LOCAL_RESTART_KILL_EVENTS_2026_09_25:
        lines.append(
            _stream_finish_error_line(
                ts, session, "local", "Qwen3",
                "stream_exception", _LOCAL_MESSAGE, _LOCAL_ACTION,
            )
        )
    ts, session = REMOTE_STREAM_EXCEPTION_EVENT_2026_09_25
    lines.append(
        _stream_finish_error_line(
            ts, session, "deepseek", "deepseek-flash",
            "stream_exception", _REMOTE_MESSAGE, _REMOTE_ACTION,
            entry="deepseek-flash",
        )
    )
    return lines


# A lone local stream error with no restart / mode-switch marker anywhere near
# it: classification must degrade to ``other`` rather than fabricate a cause.
LOCAL_ERROR_NO_MARKER_2026_09_25 = _stream_finish_error_line(
    "2026-09-25 08:30:00,000", "herdr-1790300000-000000-00000",
    "local", "Qwen3", "stream_exception", _LOCAL_MESSAGE, _LOCAL_ACTION,
)
