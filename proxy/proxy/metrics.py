"""
Prometheus metrics helpers for the proxy server.

This module exposes a small set of metrics required by LP-0MNA7G5JB004P5O6:
- llm_process_rss_bytes (gauge)
- llm_model_rss_bytes{model="..."} (gauge)
- llm_model_load_events_total{model="...",event="load|unload"} (counter)
- llm_models_loaded (gauge)

Extended by LP-0MQ1HDY1N00502S7:
- proxy_http_errors_total{endpoint="...",status="...",reason="..."} (counter)

The implementation is best-effort: when router-mode exposes multiple models in a
single process we estimate per-model RSS by dividing the process RSS equally
across loaded models (documented). If prometheus_client is not installed the
functions are no-ops and generate_metrics() returns an explanatory payload.
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
# Contention-queue metrics (LP-0MSORQVK50012Q4D F4 AC3)
llama_contention_queued_total = None
llama_contention_queued_duration_seconds = None
llama_contention_fallback_after_queue_total = None
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
        # Contention-queue metrics (LP-0MSORQVK50012Q4D F4 AC3): counters for
        # queued requests, queued duration (sum), and fallback-after-queue.
        llama_contention_queued_total = Counter(
            'llama_contention_queued_total',
            'Requests queued on local slot contention (cheap-mode queue policy)',
        )
        llama_contention_queued_duration_seconds = Counter(
            'llama_contention_queued_duration_seconds',
            'Total seconds requests waited in the local-slot contention queue',
        )
        llama_contention_fallback_after_queue_total = Counter(
            'llama_contention_fallback_after_queue_total',
            'Requests that fell back to remote after the contention-queue caps',
        )
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


def record_contention_queued(duration_seconds: float):
    """Record a request queued on local slot contention (LP-0MSORQVK50012Q4D F4).

    Increments the queued counter and adds *duration_seconds* to the total
    queued duration. Best-effort no-op when prometheus_client is unavailable.
    """
    if not _enabled:
        return
    try:
        if llama_contention_queued_total is not None:
            llama_contention_queued_total.inc()
        if llama_contention_queued_duration_seconds is not None:
            llama_contention_queued_duration_seconds.inc(max(0.0, float(duration_seconds)))
    except Exception:
        pass


def record_contention_fallback_after_queue():
    """Record a fallback-after-queue event (LP-0MSORQVK50012Q4D F4).

    Best-effort no-op when prometheus_client is unavailable.
    """
    if not _enabled or llama_contention_fallback_after_queue_total is None:
        return
    try:
        llama_contention_fallback_after_queue_total.inc()
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
