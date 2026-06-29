# Runbook: Proxy 5xx Errors on `/v1/chat/completions`

**Alert**: `ProxyHttpErrorsHigh` – fires when `rate(proxy_http_errors_total{...}[5m]) > 5` for 5 minutes.
**Severity**: Critical
**Metric**: `proxy_http_errors_total{endpoint="/v1/chat/completions",status="5xx",reason="..."}`
**Alert rule**: `monitoring/proxy_5xx_alerts.yaml`

---

## 1. Proxy-side Investigation

1. **Check proxy logs** for error signatures:
   ```bash
   grep -E '503|backend_error|backend_unavailable|self_healing|no_slots_available' /var/log/llama-proxy/proxy.log
   ```
   Look for:
   - `backend_error` — backend connection/read/timeout failures
   - `backend_unavailable` — backend process not running or `backend_ready=False`
   - `self_healing` — self-healing was active when request arrived
   - `no_slots_available` — all backend slots were busy

2. **Check session-manager logs** for invalidation patterns:
   ```bash
   grep -E 'history_mismatch|session_invalidated|session_fallback' /var/log/llama-proxy/proxy.log
   ```
   Session invalidation or `history_mismatch` may trigger a fallback path that can expose backend errors.

3. **Check backend signal counts** (from `/admin/metrics`):
   ```bash
   curl -s http://localhost:8000/admin/metrics | grep -E 'backend_signal|connect_failures|read_failures|timeout_failures|concurrency_rejects'
   ```
   Elevated counts indicate backend instability.

4. **Check Prometheus metric directly**:
   ```bash
   curl -s http://localhost:8000/metrics | grep proxy_http_errors_total
   ```
   Shows the raw counter values broken down by reason.

---

## 2. Upstream Backend Investigation

1. **Check llama-server health**:
   ```bash
   curl -s http://localhost:8080/health
   ```
   Look for `"status":"ok"` or error details. Non-ok status suggests the backend process is unhealthy.

2. **Check model state**:
   ```bash
   curl -s http://localhost:8080/slots
   ```
   Verify slots are available (not all processing). If all slots are busy, the proxy returns `slot_exhaustion` 503s.

3. **Check GPU/memory usage**:
   ```bash
   nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
   ```
   High memory or GPU utilization may cause backend timeouts or crashes.

4. **Check llama-server process status**:
   ```bash
   ps aux | grep llama-server
   ```
   Verify the process is running and not in a zombie/defunct state.

5. **Check proxy health endpoint**:
   ```bash
   curl -s http://localhost:8000/health | jq .
   ```
   Review `backend_ready`, `backend_reachable`, `self_healing_in_progress`, and `backend_recovery` fields.

---

## 3. Remediation Steps

1. **Restart the proxy** (if proxy process is stuck or misbehaving):
   ```bash
   sudo systemctl restart llama-proxy
   ```
   Or via proxyctl:
   ```bash
   proxyctl stop && proxyctl start
   ```

2. **Restart llama-server** (if the backend is unresponsive or crashed):
   ```bash
   sudo systemctl restart llama-server
   ```
   Or the proxy will self-heal automatically within the watchdog interval (~5 seconds).

3. **Rollback recent changes** — if the 5xx spike correlates with a recent deployment, rollback to the previous known-good version:
   ```bash
   git log --oneline -10
   git revert <suspect-commit-hash>
   ```

4. **Verify recovery**:
   ```bash
   curl -s http://localhost:8000/health
   curl -s http://localhost:8000/metrics | grep proxy_http_errors_total
   ```
   Confirm 5xx rate is dropping and backend is healthy.

---

## 4. Escalation Path

If the issue cannot be resolved at the proxy layer:

1. **Escalate to SRE** with the following information:
   - Timestamp range of the spike
   - Proxy log excerpts showing error signatures
   - Backend health check results
   - GPU/memory utilization snapshots
   - Any recent deployment or configuration changes

2. **Provide correlation IDs** from proxy logs to help trace specific failing requests.

3. **Create a new work item** in Worklog (wl) documenting the incident and any remaining unknowns.

---

## Appendix: Alert Details

- **Expression**: `rate(proxy_http_errors_total{endpoint="/v1/chat/completions",status="5xx"}[5m]) > 5`
- **For**: 5m (dampens transient spikes)
- **Severity**: critical
- **Metric labels**:
  - `endpoint`: `"/v1/chat/completions"` (currently the only tracked endpoint)
  - `status`: `"5xx"` (HTTP status class)
  - `reason`: `"backend_error"`, `"backend_unavailable"`, `"self_healing"`, or `"slot_exhaustion"`
- **Dashboard**: See `monitoring/grafana_llama_memory_dashboard.json` — "Proxy 5xx Errors (/v1/chat/completions)" panel.
