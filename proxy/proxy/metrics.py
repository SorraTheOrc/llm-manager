"""
Prometheus metrics helpers for the proxy server.

This module exposes a small set of metrics required by LP-0MNA7G5JB004P5O6:
- llm_process_rss_bytes (gauge)
- llm_model_rss_bytes{model="..."} (gauge)
- llm_model_load_events_total{model="...",event="load|unload"} (counter)
- llm_models_loaded (gauge)

Extended by LP-0MQ1HDY1N00502S7:
- proxy_http_errors_total{endpoint="...",status="...",reason="..."} (counter)

Extended by LP-0MVF7KQED005TG9Z (quota observability surface):
- quota_rolling_percent{account,entry} (gauge)
- quota_weekly_percent{account,entry} (gauge)
- quota_monthly_percent{account,entry} (gauge)
- quota_effective_headroom{account,entry} (gauge)
- quota_seconds_to_reset{account,entry} (gauge)
- quota_resets_at_timestamp_seconds{account,entry} (gauge)
- quota_surge_active{account,entry} (gauge, 1 = active)
- quota_exhausted{account,entry} (gauge, 1 = exhausted)
- quota_last_poll_timestamp_seconds{account,entry} (gauge)
- quota_poll_failures_total{account,reason} (counter)

The implementation is best-effort: when router-mode exposes multiple models in a
single process we estimate per-model RSS by dividing the process RSS equally
across loaded models (documented). If prometheus_client is not installed the
functions are no-ops and generate_metrics() returns an explanatory payload.

Quota labels never carry an API key **value**: ``account`` is the account key
derived from the provider entry's ``api_key_env`` name (e.g.
``OPENCODE_2_API_KEY``) and ``entry`` is the configured provider entry name.
"""

from collections.abc import Iterable

_enabled = False
Gauge = None
Counter = None
Histogram = None
generate_latest = None
CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
try:
    import prometheus_client as _pc
    Gauge = getattr(_pc, 'Gauge', None)
    Counter = getattr(_pc, 'Counter', None)
    Histogram = getattr(_pc, 'Histogram', None)
    generate_latest = getattr(_pc, 'generate_latest', None)
    CONTENT_TYPE_LATEST = getattr(_pc, 'CONTENT_TYPE_LATEST', CONTENT_TYPE_LATEST)
    # Consider metrics enabled when we can create basic metrics
    if Gauge is not None and Counter is not None and generate_latest is not None:
        _enabled = True
except Exception:
    _enabled = False

# Metric objects (initialized only if prometheus_client is available)
llama_process_rss_bytes = None
llama_model_rss_bytes = None
llama_model_load_events_total = None
llama_models_loaded = None
proxy_http_errors_total = None
# Token-rate observation metrics
llama_token_rate_gauge = None
llama_token_rate_histogram = None
# /slots query failure metric (LP-0MSVP7XJ6008QPKX): count of llama-server
# /slots query failures by reason, so a sustained /slots outage (e.g. a
# stuck model reload after cheap-mode restart) surfaces as an alert instead
# of silently wedging orchestrators (herdr downtime worker) into fail-closed
# busy via total_slots=0.
llama_slots_query_failures_total = None
# Remote-stream watchdog terminations (LP-0MSVP7ZML003XZTJ): a remote
# stream terminated by the max-duration or activity-timeout watchdog (a
# "connected but idle" upstream that never goes silent). Counts by reason
# so a runaway stream — e.g. a pi-agent pane held for 13+ hours — surfaces
# as an alert instead of silently holding proxy state (local_active_query,
# slots) indefinitely.
llama_remote_stream_terminated_total = None
# Quota observability surface (LP-0MVF7KQED005TG9Z): per-account derived quota
# state from the proactive poller, so "how much is left, when does it reset, is
# surge active" is answerable from /metrics instead of from the logs. Labels are
# the account key (the ``api_key_env`` NAME, never the key value) plus the
# configured provider entry name.
quota_rolling_percent = None
quota_weekly_percent = None
quota_monthly_percent = None
quota_effective_headroom = None
quota_seconds_to_reset = None
quota_resets_at_timestamp_seconds = None
quota_surge_active = None
quota_exhausted = None
quota_last_poll_timestamp_seconds = None
#: Counts failed quota polls by reason, so a dead /v1/usage endpoint surfaces as
#: an alert instead of only a rate-limited WARN.
quota_poll_failures_total = None

#: Label names shared by every quota gauge.
_QUOTA_GAUGE_LABELS = ['account', 'entry']

if _enabled:
    try:
        llama_process_rss_bytes = Gauge(
            'llama_process_rss_bytes', 'Resident Set Size (RSS) in bytes for the llama-server process'
        )
        llama_model_rss_bytes = Gauge(
            'llama_model_rss_bytes', 'Estimated RSS bytes attributed to a specific model', ['model']
        )
        llama_model_load_events_total = Counter(
            'llama_model_load_events_total', 'Total model load/unload events', ['model', 'event']
        )
        llama_models_loaded = Gauge(
            'llama_models_loaded', 'Number of models currently loaded in the llama-server'
        )
        proxy_http_errors_total = Counter(
            'proxy_http_errors_total',
            'Total HTTP errors by endpoint, status class, and reason',
            ['endpoint', 'status', 'reason']
        )
        # Token-rate metrics: gauge for current tokens/sec, histogram for distribution
        llama_token_rate_gauge = Gauge(
            'llama_token_rate_gauge', 'Observed token generation rate (tokens/sec) per session', ['session_id']
        )
        if Histogram is not None:
            try:
                llama_token_rate_histogram = Histogram(
                    'llama_token_rate_histogram', 'Histogram of token generation rates (tokens/sec) per session', ['session_id']
                )
            except Exception:
                # If histogram creation fails, leave it as None but keep metrics enabled
                llama_token_rate_histogram = None
        else:
            llama_token_rate_histogram = None
        llama_slots_query_failures_total = Counter(
            'llama_slots_query_failures_total',
            'Total llama-server /slots query failures by reason',
            ['reason'],
        )
        llama_remote_stream_terminated_total = Counter(
            'llama_remote_stream_terminated_total',
            'Remote streams terminated by the duration/activity watchdog by reason',
            ['reason'],
        )
        # Quota observability surface (LP-0MVF7KQED005TG9Z). One series per
        # (account, entry): the account is the api_key_env NAME (never the key
        # value) and the entry is the configured provider entry name.
        quota_rolling_percent = Gauge(
            'quota_rolling_percent',
            'Rolling-window quota percentage consumed, per account and entry',
            _QUOTA_GAUGE_LABELS,
        )
        quota_weekly_percent = Gauge(
            'quota_weekly_percent',
            'Weekly-window quota percentage consumed, per account and entry',
            _QUOTA_GAUGE_LABELS,
        )
        quota_monthly_percent = Gauge(
            'quota_monthly_percent',
            'Monthly-window quota percentage consumed, per account and entry',
            _QUOTA_GAUGE_LABELS,
        )
        quota_effective_headroom = Gauge(
            'quota_effective_headroom',
            'Remaining quota percentage across the binding (minimum) window',
            _QUOTA_GAUGE_LABELS,
        )
        quota_seconds_to_reset = Gauge(
            'quota_seconds_to_reset',
            'Seconds until the account monthly allowance resets',
            _QUOTA_GAUGE_LABELS,
        )
        quota_resets_at_timestamp_seconds = Gauge(
            'quota_resets_at_timestamp_seconds',
            'Unix timestamp of the account monthly allowance reset',
            _QUOTA_GAUGE_LABELS,
        )
        quota_surge_active = Gauge(
            'quota_surge_active',
            'Whether quota-driven surge is active for the account (1 = active)',
            _QUOTA_GAUGE_LABELS,
        )
        quota_exhausted = Gauge(
            'quota_exhausted',
            'Whether the account quota is exhausted (1 = exhausted)',
            _QUOTA_GAUGE_LABELS,
        )
        quota_last_poll_timestamp_seconds = Gauge(
            'quota_last_poll_timestamp_seconds',
            'Unix timestamp of the last successful /v1/usage poll for the account',
            _QUOTA_GAUGE_LABELS,
        )
        quota_poll_failures_total = Counter(
            'quota_poll_failures_total',
            'Total /v1/usage poll failures by account and reason',
            ['account', 'reason'],
        )
    except Exception:  # pragma: no cover - defensive
        _enabled = False


def update_metrics(process_rss: int | None, loaded_models: Iterable[str] | None):
    """Update gauges for process RSS and per-model RSS.

    - process_rss: integer bytes or None
    - loaded_models: iterable of model ids or None

    Behavior:
    - Set llama_process_rss_bytes to process_rss if available
    - Set llama_models_loaded to count of loaded_models (or 1 if current_model present and loaded_models None)
    - When multiple models are loaded in a single process, per-model RSS is estimated
      by dividing process_rss equally across models if process_rss is set.
    """
    if not _enabled:
        return
    try:
        # Process RSS
        if process_rss is None:
            llama_process_rss_bytes.set(0)
        else:
            llama_process_rss_bytes.set(int(process_rss))

        models = list(loaded_models) if loaded_models is not None else []
        count = len(models)
        if count <= 0:
            # No explicit list — leave models gauge at 0
            llama_models_loaded.set(0)
            return
        llama_models_loaded.set(count)

        # Estimate per-model RSS if we have a process RSS; otherwise set 0
        if process_rss is None or process_rss <= 0:
            # zero-out previous values
            for m in models:
                llama_model_rss_bytes.labels(model=m).set(0)
        else:
            # Heuristic: divide the process RSS equally between loaded models
            try:
                per = int(process_rss // count)
            except Exception:
                per = 0
            for m in models:
                llama_model_rss_bytes.labels(model=m).set(per)
    except Exception:
        # Best-effort: do not propagate metric errors
        return


def record_model_loaded(model: str):
    """Record a load event for the model."""
    if not _enabled or not model:
        return
    try:
        llama_model_load_events_total.labels(model=model, event='load').inc()
    except Exception:
        pass


def record_model_unloaded(model: str):
    """Record an unload event for the model."""
    if not _enabled or not model:
        return
    try:
        llama_model_load_events_total.labels(model=model, event='unload').inc()
    except Exception:
        pass


def record_http_error(endpoint: str, status: str, reason: str):
    """Increment proxy_http_errors_total with the given label values.

    Args:
        endpoint: The API endpoint path, e.g. "v1/chat/completions".
        status: The HTTP status class, e.g. "5xx".
        reason: A short identifier for the error cause, e.g. "backend_error".

    This is a best-effort no-op when prometheus_client is not installed
    or the counter is unavailable.
    """
    if not _enabled or proxy_http_errors_total is None:
        return
    try:
        proxy_http_errors_total.labels(endpoint=endpoint, status=status, reason=reason).inc()
    except Exception:
        pass


def record_slots_query_failure(reason: str):
    """Increment llama_slots_query_failures_total with the given reason.

    Args:
        reason: A short identifier for the /slots failure cause, e.g.
            "http_500", "http_400", "timeout", "connection_error",
            "invalid_payload".

    LP-0MSVP7XJ6008QPKX: a sustained /slots failure during a model reload
    (cheap-mode restart) previously surfaced as total_slots=0 with no
    signal; this counter feeds a Prometheus alert on the failure rate.

    Best-effort no-op when prometheus_client is unavailable.
    """
    if not _enabled or llama_slots_query_failures_total is None:
        return
    try:
        llama_slots_query_failures_total.labels(reason=reason).inc()
    except Exception:
        pass


def record_remote_stream_terminated(reason: str):
    """Increment llama_remote_stream_terminated_total with the given reason.

    Args:
        reason: A short identifier for the watchdog termination, e.g.
            "stream_max_duration", "stream_activity_timeout".

    LP-0MSVP7ZML003XZTJ: a remote stream that never terminates (connected
    but idle) previously held proxy state indefinitely; this counter feeds
    a Prometheus alert so runaway streams surface.

    Best-effort no-op when prometheus_client is unavailable.
    """
    if not _enabled or llama_remote_stream_terminated_total is None:
        return
    try:
        llama_remote_stream_terminated_total.labels(reason=reason).inc()
    except Exception:
        pass


def record_quota_poll_failure(account: str, reason: str):
    """Increment ``quota_poll_failures_total`` for a failed ``/v1/usage`` poll.

    Args:
        account: The account key (the provider entry's ``api_key_env`` NAME —
            never the API key value).
        reason: A short, stable failure identifier, e.g. ``QuotaPollError``,
            ``HTTPStatusError``, ``ReadTimeout``.

    The previous gauges are deliberately left untouched: a failed poll is a
    fail-open event that keeps the last known-good quota values
    (LP-0MVF7KEG7007C8MF) and only increments the failure counter.

    Best-effort no-op when prometheus_client is unavailable.
    """
    if not _enabled or quota_poll_failures_total is None:
        return
    try:
        quota_poll_failures_total.labels(account=account, reason=reason).inc()
    except Exception:
        pass


def _quota_gauge_value(observation: dict, *path, default: float = 0.0) -> float:
    """Read a numeric field from a nested observation dict, never raising."""
    node = observation
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    try:
        return float(node)
    except (TypeError, ValueError):
        return default


def record_quota_state(observation: dict):
    """Publish one account's derived quota state on the Prometheus gauges.

    Args:
        observation: A single-account observation as returned by
            :func:`proxy.quota_state.snapshot_observation` (the same fields the
            ``/events`` payload and the home page render): ``account``,
            ``entry``, the three window dicts (``percent``), ``resetsAt`` /
            ``resets_at_timestamp_seconds``, ``seconds_to_reset``,
            ``effective_headroom``, ``surge_active``, ``exhausted`` and
            ``last_poll_timestamp_seconds``.

    Best-effort: a missing field defaults to ``0`` and any metric error is
    swallowed, so a metrics failure can never break the poller or routing.
    """
    if not _enabled or not isinstance(observation, dict):
        return
    gauges = (
        quota_rolling_percent,
        quota_weekly_percent,
        quota_monthly_percent,
        quota_effective_headroom,
        quota_seconds_to_reset,
        quota_resets_at_timestamp_seconds,
        quota_surge_active,
        quota_exhausted,
        quota_last_poll_timestamp_seconds,
    )
    if any(gauge is None for gauge in gauges):
        return
    account = str(observation.get('account') or 'unknown')
    entry = str(observation.get('entry') or account)
    labels = {'account': account, 'entry': entry}
    try:
        quota_rolling_percent.labels(**labels).set(
            _quota_gauge_value(observation, 'rolling', 'percent')
        )
        quota_weekly_percent.labels(**labels).set(
            _quota_gauge_value(observation, 'weekly', 'percent')
        )
        quota_monthly_percent.labels(**labels).set(
            _quota_gauge_value(observation, 'monthly', 'percent')
        )
        quota_effective_headroom.labels(**labels).set(
            _quota_gauge_value(observation, 'effective_headroom')
        )
        quota_seconds_to_reset.labels(**labels).set(
            _quota_gauge_value(observation, 'seconds_to_reset')
        )
        quota_resets_at_timestamp_seconds.labels(**labels).set(
            _quota_gauge_value(observation, 'resets_at_timestamp_seconds')
        )
        quota_surge_active.labels(**labels).set(
            1.0 if observation.get('surge_active') else 0.0
        )
        quota_exhausted.labels(**labels).set(
            1.0 if observation.get('exhausted') else 0.0
        )
        quota_last_poll_timestamp_seconds.labels(**labels).set(
            _quota_gauge_value(observation, 'last_poll_timestamp_seconds')
        )
    except Exception:
        # Best-effort: never propagate metric errors into the poller.
        return


def generate_metrics_payload() -> tuple[bytes, str]:
    """Return the Prometheus exposition payload bytes and content-type.

    If prometheus_client is unavailable, return a short plaintext message and
    text/plain content type.
    """
    if not _enabled:
        body = b"Prometheus client library not installed; metrics disabled\n"
        return body, CONTENT_TYPE_LATEST
    try:
        payload = generate_latest()
        return payload, CONTENT_TYPE_LATEST
    except Exception:
        return b"failed to generate metrics\n", CONTENT_TYPE_LATEST
