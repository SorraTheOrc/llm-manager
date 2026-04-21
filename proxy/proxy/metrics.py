"""
Prometheus metrics helpers for the proxy server.

This module exposes a small set of metrics required by LP-0MNA7G5JB004P5O6:
- llm_process_rss_bytes (gauge)
- llm_model_rss_bytes{model="..."} (gauge)
- llm_model_load_events_total{model="...",event="load|unload"} (counter)
- llm_models_loaded (gauge)

The implementation is best-effort: when router-mode exposes multiple models in a
single process we estimate per-model RSS by dividing the process RSS equally
across loaded models (documented). If prometheus_client is not installed the
functions are no-ops and generate_metrics() returns an explanatory payload.
"""

from typing import Iterable, Optional

_enabled = True
try:
    from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
    # Use the default registry — simple and compatible with Prometheus exposition
except Exception:  # pragma: no cover - fallback path when dependency missing
    _enabled = False
    Gauge = None
    Counter = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

# Metric objects (initialized only if prometheus_client is available)
llama_process_rss_bytes = None
llama_model_rss_bytes = None
llama_model_load_events_total = None
llama_models_loaded = None

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
    except Exception:  # pragma: no cover - defensive
        _enabled = False


def update_metrics(process_rss: Optional[int], loaded_models: Optional[Iterable[str]]):
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
