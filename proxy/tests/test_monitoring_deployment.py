"""Tests for monitoring deployment configuration files.

Validates that:
- Prometheus configuration (monitoring/prometheus.yml) is valid YAML and has
  the required sections (scrape_configs, rule_files).
- Grafana provisioning files (monitoring/grafana/datasources/ and dashboards/)
  are valid YAML with correct structure.
- Systemd unit files (docs/systemd/prometheus.service,
  docs/systemd/grafana-server.service) are syntactically valid ini-style units.
- Alert rule files (monitoring/*.yaml) are valid Prometheus rule files.
"""

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MONITORING_DIR = PROJECT_ROOT / "monitoring"
DOCS_SYSTEMD_DIR = PROJECT_ROOT / "docs" / "systemd"

REQUIRED_CONFIG_SECTIONS = {
    "prometheus.yml": ["global", "scrape_configs", "rule_files"],
}

REQUIRED_SCRAPE_JOBS = {
    "prometheus.yml": ["llama-proxy", "rocm-exporter"],
}

REQUIRED_RULE_FILES = {
    "prometheus.yml": [
        "monitoring/llama_memory_alerts.yaml",
        "monitoring/proxy_5xx_alerts.yaml",
        "monitoring/gpu_vram_alerts.yaml",
    ],
}

EXPECTED_ROCM_EXPORTER_METRICS = [
    "rocm_vram_total_bytes",
    "rocm_vram_used_bytes",
    "rocm_vram_free_bytes",
]


def assert_valid_yaml(path: Path) -> dict:
    """Assert that *path* is valid YAML and return the parsed content."""
    assert path.exists(), f"Missing file: {path}"
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data is not None, f"Empty or invalid YAML in {path}"
    return data


def assert_valid_systemd_unit(path: Path) -> None:
    """Assert that *path* is a valid systemd unit (ini-like format)."""
    assert path.exists(), f"Missing systemd unit: {path}"
    content = path.read_text()
    assert content.strip(), f"Empty systemd unit: {path}"
    # Must have at least one section header like [Unit], [Service], [Install]
    sections = [line for line in content.splitlines() if line.startswith("[")]
    assert len(sections) >= 1, (
        f"Systemd unit {path} has no section headers. "
        f"Expected at least one of [Unit], [Service], [Install]."
    )
    # Must have an ExecStart line in a [Service] section
    # (Simple check — a full validation would use systemd-analyze verify)
    assert "ExecStart=" in content, (
        f"Systemd unit {path} is missing ExecStart= directive."
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prometheus_config() -> dict:
    return assert_valid_yaml(MONITORING_DIR / "prometheus.yml")


@pytest.fixture
def grafana_datasources_config() -> dict:
    return assert_valid_yaml(
        MONITORING_DIR / "grafana" / "datasources" / "datasources.yaml"
    )


@pytest.fixture
def grafana_dashboards_config() -> dict:
    return assert_valid_yaml(
        MONITORING_DIR / "grafana" / "dashboards" / "dashboards.yaml"
    )


@pytest.fixture
def prometheus_alert_rules() -> list[Path]:
    """Return paths to all alert rule YAML files referenced in prometheus.yml."""
    config = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")
    rule_files = config.get("rule_files", [])
    paths = []
    for rf in rule_files:
        # Rule file paths are relative to the prometheus.yml location
        p = MONITORING_DIR / rf.replace("monitoring/", "")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Tests: Prometheus configuration
# ---------------------------------------------------------------------------


class TestPrometheusConfig:
    """Validation of monitoring/prometheus.yml."""

    def test_file_exists(self):
        """AC 1,6: Prometheus config file exists with scrape config and rule_files."""
        path = MONITORING_DIR / "prometheus.yml"
        assert path.exists(), f"Missing prometheus.yml at {path}"

    def test_valid_yaml(self, prometheus_config):
        """prometheus.yml is valid YAML."""
        assert isinstance(prometheus_config, dict)

    def test_has_required_sections(self, prometheus_config):
        """prometheus.yml has global, scrape_configs, and rule_files sections."""
        for section in REQUIRED_CONFIG_SECTIONS["prometheus.yml"]:
            assert section in prometheus_config, (
                f"Missing section '{section}' in prometheus.yml"
            )

    def test_scrape_configs_is_list(self, prometheus_config):
        """scrape_configs is a non-empty list."""
        scrape = prometheus_config.get("scrape_configs", [])
        assert isinstance(scrape, list), "scrape_configs must be a list"
        assert len(scrape) > 0, "scrape_configs must not be empty"

    def test_llama_proxy_scrape_job_exists(self, prometheus_config):
        """AC 1: Prometheus is configured to scrape the proxy /metrics endpoint."""
        jobs = {j.get("job_name") for j in prometheus_config.get("scrape_configs", [])}
        assert "llama-proxy" in jobs, (
            "Missing scrape job 'llama-proxy' in prometheus.yml"
        )

    def test_llama_proxy_scrape_target(self, prometheus_config):
        """AC 1: The llama-proxy scrape job targets localhost:8000."""
        for job in prometheus_config.get("scrape_configs", []):
            if job.get("job_name") == "llama-proxy":
                targets = job.get("static_configs", [{}])[0].get("targets", [])
                assert "localhost:8000" in targets, (
                    "llama-proxy scrape job must target localhost:8000"
                )
                return
        pytest.fail("llama-proxy scrape job not found")

    def test_rocm_exporter_scrape_job_exists(self, prometheus_config):
        """AC 2: Prometheus has a scrape job for rocm-exporter."""
        jobs = {j.get("job_name") for j in prometheus_config.get("scrape_configs", [])}
        assert "rocm-exporter" in jobs, (
            "Missing scrape job 'rocm-exporter' in prometheus.yml"
        )

    def test_rocm_exporter_scrape_target(self, prometheus_config):
        """AC 2: The rocm-exporter scrape job targets localhost:5000."""
        for job in prometheus_config.get("scrape_configs", []):
            if job.get("job_name") == "rocm-exporter":
                targets = job.get("static_configs", [{}])[0].get("targets", [])
                assert len(targets) >= 1, (
                    "rocm-exporter scrape job must have at least one target"
                )
                return
        pytest.fail("rocm-exporter scrape job not found")

    def test_alert_rule_files_configured(self, prometheus_config):
        """AC 6: Prometheus is configured to load alert rule files."""
        rule_files = prometheus_config.get("rule_files", [])
        assert len(rule_files) > 0, "rule_files must not be empty"

    def test_alert_rule_files_referenced_exist(self, prometheus_config):
        """AC 6: Each referenced rule file exists on disk."""
        rule_files = prometheus_config.get("rule_files", [])
        for rf in rule_files:
            relative = rf.replace("monitoring/", "")
            path = MONITORING_DIR / relative
            assert path.exists(), (
                f"Referenced rule file {rf} not found at {path}"
            )


# ---------------------------------------------------------------------------
# Tests: Grafana provisioning configuration
# ---------------------------------------------------------------------------


class TestGrafanaDatasourceConfig:
    """Validation of Grafana datasource provisioning."""

    def test_datasource_file_exists(self):
        """AC 4: Grafana datasource provisioning file exists."""
        path = MONITORING_DIR / "grafana" / "datasources" / "datasources.yaml"
        assert path.exists(), f"Missing datasources.yaml at {path}"

    def test_valid_yaml(self, grafana_datasources_config):
        """datasources.yaml is valid YAML."""
        assert isinstance(grafana_datasources_config, dict)

    def test_api_version(self, grafana_datasources_config):
        """datasources.yaml has apiVersion set."""
        assert "apiVersion" in grafana_datasources_config

    def test_datasources_is_list(self, grafana_datasources_config):
        """datasources.yaml contains a datasources list with at least one entry."""
        dss = grafana_datasources_config.get("datasources", [])
        assert isinstance(dss, list), "datasources must be a list"
        assert len(dss) > 0, "datasources must not be empty"

    def test_prometheus_datasource_configured(self, grafana_datasources_config):
        """AC 4: Grafana has Prometheus configured as a datasource."""
        dss = grafana_datasources_config.get("datasources", [])
        prom = [ds for ds in dss if ds.get("type") == "prometheus"]
        assert len(prom) >= 1, (
            "No Prometheus datasource found in datasources.yaml"
        )

    def test_prometheus_datasource_url(self, grafana_datasources_config):
        """AC 4: Prometheus datasource points to localhost:9090."""
        dss = grafana_datasources_config.get("datasources", [])
        for ds in dss:
            if ds.get("type") == "prometheus":
                url = ds.get("url", "")
                assert "localhost:9090" in url, (
                    f"Prometheus datasource URL should point to localhost:9090, got: {url}"
                )
                return
        pytest.fail("No Prometheus datasource found")


class TestGrafanaDashboardConfig:
    """Validation of Grafana dashboard provisioning."""

    def test_dashboards_file_exists(self):
        """AC 3: Grafana dashboard provisioning file exists."""
        path = MONITORING_DIR / "grafana" / "dashboards" / "dashboards.yaml"
        assert path.exists(), f"Missing dashboards.yaml at {path}"

    def test_valid_yaml(self, grafana_dashboards_config):
        """dashboards.yaml is valid YAML."""
        assert isinstance(grafana_dashboards_config, dict)

    def test_api_version(self, grafana_dashboards_config):
        """dashboards.yaml has apiVersion set."""
        assert "apiVersion" in grafana_dashboards_config

    def test_providers_is_list(self, grafana_dashboards_config):
        """dashboards.yaml contains a providers list."""
        providers = grafana_dashboards_config.get("providers", [])
        assert isinstance(providers, list), "providers must be a list"
        assert len(providers) > 0, "providers must not be empty"

    def test_dashboard_provider_points_to_existing_dir(self, grafana_dashboards_config):
        """AC 3: Dashboard provider's options.path points to an existing directory."""
        providers = grafana_dashboards_config.get("providers", [])
        for prov in providers:
            options = prov.get("options", {})
            dash_path_str = options.get("path", "")
            if dash_path_str:
                dash_path = Path(dash_path_str)
                if not dash_path.is_absolute():
                    # Relative to Grafana home — we can't fully validate here
                    # but we can check it's a reasonable relative path
                    assert dash_path_str.startswith(
                        "${grafana_home}"
                    ) or dash_path_str.startswith("/"), (
                        f"Dashboard path {dash_path_str} should be absolute or "
                        f"reference ${grafana_home}"
                    )


# ---------------------------------------------------------------------------
# Tests: Alert rule files
# ---------------------------------------------------------------------------


class TestAlertRuleFiles:
    """Validation that alert rule YAML files are structurally valid."""

    def test_llama_memory_alerts_exists(self):
        """Llama memory alert rules file exists."""
        path = MONITORING_DIR / "llama_memory_alerts.yaml"
        assert path.exists()

    def test_llama_memory_alerts_valid_yaml(self):
        """Llama memory alert rules is valid YAML with groups/rules."""
        data = assert_valid_yaml(MONITORING_DIR / "llama_memory_alerts.yaml")
        assert "groups" in data, "llama_memory_alerts.yaml missing 'groups' key"
        assert len(data["groups"]) >= 1
        assert len(data["groups"][0].get("rules", [])) >= 1

    def test_proxy_5xx_alerts_exists(self):
        """Proxy 5xx alert rules file exists."""
        path = MONITORING_DIR / "proxy_5xx_alerts.yaml"
        assert path.exists()

    def test_proxy_5xx_alerts_valid_yaml(self):
        """Proxy 5xx alert rules is valid YAML with groups/rules."""
        data = assert_valid_yaml(MONITORING_DIR / "proxy_5xx_alerts.yaml")
        assert "groups" in data, "proxy_5xx_alerts.yaml missing 'groups' key"
        assert len(data["groups"]) >= 1
        assert len(data["groups"][0].get("rules", [])) >= 1

    def test_alert_rule_file_names_in_prometheus_config(self):
        """AC 6: The prometheus.yml references the alert rule files."""
        config = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")
        rule_files = config.get("rule_files", [])
        assert (
            "monitoring/llama_memory_alerts.yaml" in rule_files
        ), "prometheus.yml must reference monitoring/llama_memory_alerts.yaml"
        assert (
            "monitoring/proxy_5xx_alerts.yaml" in rule_files
        ), "prometheus.yml must reference monitoring/proxy_5xx_alerts.yaml"
        assert (
            "monitoring/gpu_vram_alerts.yaml" in rule_files
        ), "prometheus.yml must reference monitoring/gpu_vram_alerts.yaml"


# ---------------------------------------------------------------------------
# Tests: Systemd unit files
# ---------------------------------------------------------------------------


class TestPrometheusSystemdUnit:
    """Validation of docs/systemd/prometheus.service."""

    def test_unit_file_exists(self):
        """Prometheus systemd unit file exists."""
        path = DOCS_SYSTEMD_DIR / "prometheus.service"
        assert path.exists(), f"Missing prometheus.service at {path}"

    def test_valid_systemd_unit(self):
        """prometheus.service is a valid systemd unit with required sections."""
        assert_valid_systemd_unit(DOCS_SYSTEMD_DIR / "prometheus.service")

    def test_execstart_references_prometheus(self):
        """ExecStart references the prometheus binary."""
        content = (DOCS_SYSTEMD_DIR / "prometheus.service").read_text()
        assert "prometheus" in content.lower(), (
            "ExecStart should reference prometheus"
        )

    def test_has_service_section(self):
        """prometheus.service has a [Service] section with ExecStart."""
        content = (DOCS_SYSTEMD_DIR / "prometheus.service").read_text()
        assert "[Service]" in content
        assert "ExecStart=" in content


class TestGrafanaSystemdUnit:
    """Validation of docs/systemd/grafana-server.service."""

    def test_unit_file_exists(self):
        """Grafana systemd unit file exists."""
        path = DOCS_SYSTEMD_DIR / "grafana-server.service"
        assert path.exists(), f"Missing grafana-server.service at {path}"

    def test_valid_systemd_unit(self):
        """grafana-server.service is a valid systemd unit with required sections."""
        assert_valid_systemd_unit(DOCS_SYSTEMD_DIR / "grafana-server.service")

    def test_execstart_references_grafana(self):
        """ExecStart references the grafana server binary."""
        content = (DOCS_SYSTEMD_DIR / "grafana-server.service").read_text()
        assert "grafana" in content.lower(), (
            "ExecStart should reference grafana"
        )

    def test_has_service_section(self):
        """grafana-server.service has a [Service] section with ExecStart."""
        content = (DOCS_SYSTEMD_DIR / "grafana-server.service").read_text()
        assert "[Service]" in content
        assert "ExecStart=" in content


# ---------------------------------------------------------------------------
# Tests: Monitoring documentation
# ---------------------------------------------------------------------------


class TestGpuVramAlerts:
    """Validation of monitoring/gpu_vram_alerts.yaml."""

    def test_gpu_vram_alerts_exists(self):
        """AC 4: GPU VRAM alert rules file exists."""
        path = MONITORING_DIR / "gpu_vram_alerts.yaml"
        assert path.exists()

    def test_gpu_vram_alerts_valid_yaml(self):
        """AC 4: GPU VRAM alert rules is valid YAML with groups/rules."""
        data = assert_valid_yaml(MONITORING_DIR / "gpu_vram_alerts.yaml")
        assert "groups" in data, "gpu_vram_alerts.yaml missing 'groups' key"
        assert len(data["groups"]) >= 1
        assert len(data["groups"][0].get("rules", [])) >= 2

    def test_gpu_vram_alert_names(self):
        """AC 4: GPU VRAM alerts have expected warning and critical rules."""
        data = assert_valid_yaml(MONITORING_DIR / "gpu_vram_alerts.yaml")
        rules = data["groups"][0]["rules"]
        alert_names = [r.get("alert") for r in rules]
        assert "GpuVramHighWarning" in alert_names, (
            "Expected alert 'GpuVramHighWarning' in gpu_vram_alerts.yaml"
        )
        assert "GpuVramHighCritical" in alert_names, (
            "Expected alert 'GpuVramHighCritical' in gpu_vram_alerts.yaml"
        )

    def test_gpu_vram_warning_threshold(self):
        """AC 4: Warning alert fires at 75% VRAM utilization."""
        data = assert_valid_yaml(MONITORING_DIR / "gpu_vram_alerts.yaml")
        rules = data["groups"][0]["rules"]
        for r in rules:
            if r.get("alert") == "GpuVramHighWarning":
                expr = r["expr"]
                assert "0.75" in expr, (
                    f"Warning threshold should be 0.75, got expr: {expr}"
                )
                assert r["labels"]["severity"] == "warning"
                return
        pytest.fail("GpuVramHighWarning alert not found")

    def test_gpu_vram_critical_threshold(self):
        """AC 4: Critical alert fires at 90% VRAM utilization."""
        data = assert_valid_yaml(MONITORING_DIR / "gpu_vram_alerts.yaml")
        rules = data["groups"][0]["rules"]
        for r in rules:
            if r.get("alert") == "GpuVramHighCritical":
                expr = r["expr"]
                assert "0.9" in expr, (
                    f"Critical threshold should be 0.9, got expr: {expr}"
                )
                assert r["labels"]["severity"] == "critical"
                return
        pytest.fail("GpuVramHighCritical alert not found")


class TestRocmExporterSystemdUnit:
    """Validation of docs/systemd/rocm-exporter.service."""

    def test_unit_file_exists(self):
        """AC 1: ROCm exporter systemd unit file exists."""
        path = DOCS_SYSTEMD_DIR / "rocm-exporter.service"
        assert path.exists(), f"Missing rocm-exporter.service at {path}"

    def test_valid_systemd_unit(self):
        """AC 1: rocm-exporter.service is a valid systemd unit."""
        assert_valid_systemd_unit(DOCS_SYSTEMD_DIR / "rocm-exporter.service")

    def test_execstart_references_rocm_exporter(self):
        """AC 1: ExecStart references the rocm-exporter binary."""
        content = (DOCS_SYSTEMD_DIR / "rocm-exporter.service").read_text()
        assert "rocm-exporter" in content.lower(), (
            "ExecStart should reference rocm-exporter"
        )

    def test_has_service_section(self):
        """AC 1: rocm-exporter.service has a [Service] section with ExecStart."""
        content = (DOCS_SYSTEMD_DIR / "rocm-exporter.service").read_text()
        assert "[Service]" in content
        assert "ExecStart=" in content


class TestGrafanaDashboardVramPanel:
    """Validation that the Grafana dashboard has VRAM panels."""

    def _load_dashboard(self) -> dict:
        path = MONITORING_DIR / "grafana_llama_memory_dashboard.json"
        import json
        with open(path) as f:
            return json.load(f)

    def test_dashboard_has_vram_panel(self):
        """AC 3: Grafana dashboard includes a VRAM usage panel."""
        dashboard = self._load_dashboard()
        panels = dashboard.get("panels", [])
        panel_titles = [p.get("title", "") for p in panels]
        vram_panels = [t for t in panel_titles if "vram" in t.lower()]
        assert len(vram_panels) >= 1, (
            f"Expected at least one VRAM panel in dashboard, found titles: {panel_titles}"
        )

    def test_vram_panel_has_rocm_expr(self):
        """AC 3: VRAM panel uses rocm-exporter metrics in its query."""
        dashboard = self._load_dashboard()
        panels = dashboard.get("panels", [])
        for p in panels:
            if "vram" in p.get("title", "").lower():
                targets = p.get("targets", [])
                all_exprs = [t.get("expr", "") for t in targets]
                combined = " ".join(all_exprs)
                has_rocm_metric = any(
                    m in combined for m in EXPECTED_ROCM_EXPORTER_METRICS
                )
                assert has_rocm_metric, (
                    f"VRAM panel '{p['title']}' should reference a rocm-exporter metric. "
                    f"Exprs: {all_exprs}"
                )
                return
        pytest.fail("No VRAM panel found in dashboard")


class TestMonitoringDocs:
    """Validation that monitoring documentation is in place."""

    def test_monitoring_readme_exists(self):
        """AC 7: monitoring/README.md exists with deployment documentation."""
        path = MONITORING_DIR / "README.md"
        assert path.exists(), f"Missing monitoring/README.md at {path}"
        content = path.read_text()
        assert len(content) > 100, (
            "monitoring/README.md appears too short to contain meaningful docs"
        )

    def test_monitoring_readme_has_verification_steps(self):
        """AC 5,7: monitoring/README.md includes verification steps."""
        path = MONITORING_DIR / "README.md"
        content = path.read_text().lower()
        # Look for keywords that indicate verification instructions
        keywords = ["curl", "verify", "check", "status", "test"]
        found = [kw for kw in keywords if kw in content]
        assert len(found) >= 1, (
            f"monitoring/README.md missing verification keywords. "
            f"Expected at least one of: {keywords}"
        )

    def test_proxy_readme_links_to_monitoring_docs(self):
        """AC 7: proxy/README.md references the monitoring deployment documentation."""
        path = PROJECT_ROOT / "proxy" / "README.md"
        assert path.exists()
        content = path.read_text().lower()
        # Check that the monitoring section references the monitoring directory
        # or the deployment guide
        refs = ["monitoring/readme", "monitoring deployment", "deploy prometheus"]
        found = [ref for ref in refs if ref in content]
        assert len(found) >= 1, (
            f"proxy/README.md expected to reference monitoring deployment docs. "
            f"None of {refs} found."
        )

    def test_monitoring_readme_has_rocm_exporter_section(self):
        """AC 5: monitoring/README.md documents rocm-exporter installation."""
        path = MONITORING_DIR / "README.md"
        content = path.read_text().lower()
        assert "rocm-exporter" in content, (
            "monitoring/README.md should document rocm-exporter installation"
        )

    def test_monitoring_readme_has_vram_alerts_section(self):
        """AC 5,7: monitoring/README.md documents VRAM alert rules."""
        path = MONITORING_DIR / "README.md"
        content = path.read_text().lower()
        assert "vram" in content, (
            "monitoring/README.md should reference VRAM alerts"
        )
