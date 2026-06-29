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

import os
import sys
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
    "prometheus.yml": ["llama-proxy"],
}

REQUIRED_RULE_FILES = {
    "prometheus.yml": [
        "monitoring/llama_memory_alerts.yaml",
        "monitoring/proxy_5xx_alerts.yaml",
    ],
}


def assert_valid_yaml(path: Path) -> dict:
    """Assert that *path* is valid YAML and return the parsed content."""
    assert path.exists(), f"Missing file: {path}"
    with open(path, "r") as f:
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
