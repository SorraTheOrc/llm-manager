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

import re
import shutil
import subprocess
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
    "prometheus.yml": ["llama-proxy", "node-exporter"],
}

REQUIRED_RULE_FILES = {
    "prometheus.yml": [
        "gpu_compute_rules.yaml",
        "gpu_memory_alerts.yaml",
        "proxy_5xx_alerts.yaml",
        "slots_query_alerts.yaml",
        "remote_stream_alerts.yaml",
    ],
}

# Series that must never appear in a rule expression:
# - rocm_vram_*: the AMD rocm-exporter publishes no release artifacts, so no
#   exporter ever served these series and rules using them could never fire.
# - gpu_busy_percent: pinned at 100% regardless of GPU state on this Strix Halo
#   APU (LP-0MS9GA6QL007XTBP), so it is not a utilisation signal.
# - llama_process_rss_bytes: published as 0 by the proxy (llama-server runs in a
#   podman container, so there is no host process handle to read RSS from).
FORBIDDEN_RULE_SERIES = [
    "rocm_vram_",
    "gpu_busy_percent",
    "llama_process_rss_bytes",
]

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
        # Rule file paths are relative to the prometheus.yml location, so they
        # must not repeat the `monitoring/` prefix (they did once, and a config
        # with unresolvable rule files is rejected by Prometheus outright).
        assert not rf.startswith("monitoring/"), (
            f"rule_files entry {rf!r} must be relative to monitoring/ and must not "
            f"repeat the 'monitoring/' prefix"
        )
        paths.append(MONITORING_DIR / rf)
    return paths


def _rule_expressions(path: Path) -> list[str]:
    """Return every expr string in a Prometheus rule file (recording + alerting)."""
    data = assert_valid_yaml(path)
    exprs = []
    for group in data.get("groups", []):
        for rule in group.get("rules", []):
            expr = rule.get("expr")
            if expr:
                exprs.append(expr)
    return exprs


def _find_key(node, key: str):
    """Recursively find *key* in a nested config mapping (config.yaml blocks)."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def _threshold_fraction(expr: str) -> float | None:
    """Parse the `> 0.75 * ...` fraction out of an alert expression."""
    m = re.search(r"[<>]=?\s*(\d+(?:\.\d+)?)\s*\*", expr)
    return float(m.group(1)) if m else None


def _promtool() -> str | None:
    """Locate promtool, or None when it is not installed on this host."""
    for candidate in (
        shutil.which("promtool"),
        str(Path.home() / "bin" / "prometheus" / "promtool"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


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

    def test_node_exporter_scrape_job_exists(self, prometheus_config):
        """GPU/host metrics come from node_exporter (rocm-exporter has no releases)."""
        jobs = {j.get("job_name") for j in prometheus_config.get("scrape_configs", [])}
        assert "node-exporter" in jobs, (
            "Missing scrape job 'node-exporter' in prometheus.yml - GPU power "
            "and GTT series are sourced from it"
        )

    def test_node_exporter_scrape_target(self, prometheus_config):
        """The node-exporter scrape job targets localhost:9100."""
        for job in prometheus_config.get("scrape_configs", []):
            if job.get("job_name") == "node-exporter":
                targets = job.get("static_configs", [{}])[0].get("targets", [])
                assert "localhost:9100" in targets, (
                    "node-exporter scrape job must target localhost:9100"
                )
                return
        pytest.fail("node-exporter scrape job not found")

    def test_no_scrape_job_for_unpublished_exporter(self, prometheus_config):
        """No active job scrapes an exporter that publishes no release artifacts."""
        jobs = {j.get("job_name") for j in prometheus_config.get("scrape_configs", [])}
        assert "rocm-exporter" not in jobs, (
            "rocm-exporter has no release artifacts, so this job can only ever be "
            "a permanently DOWN target"
        )

    def test_alert_rule_files_configured(self, prometheus_config):
        """AC 6: Prometheus is configured to load alert rule files."""
        rule_files = prometheus_config.get("rule_files", [])
        assert len(rule_files) > 0, "rule_files must not be empty"

    def test_alert_rule_files_referenced_exist(self, prometheus_config):
        """AC 6: Each referenced rule file exists on disk."""
        rule_files = prometheus_config.get("rule_files", [])
        for rf in rule_files:
            # Paths are resolved relative to prometheus.yml's own directory.
            path = MONITORING_DIR / rf
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
                        f"reference ${{{{grafana_home}}}}"
                    )


# ---------------------------------------------------------------------------
# Tests: Alert rule files
# ---------------------------------------------------------------------------


class TestAlertRuleFiles:
    """Validation that alert rule YAML files are structurally valid."""

    def test_gpu_compute_rules_exists(self):
        """GPU compute rules file exists."""
        path = MONITORING_DIR / "gpu_compute_rules.yaml"
        assert path.exists()

    def test_gpu_memory_alerts_valid_yaml(self):
        """GPU memory alert rules is valid YAML with groups/rules."""
        data = assert_valid_yaml(MONITORING_DIR / "gpu_memory_alerts.yaml")
        assert "groups" in data, "gpu_memory_alerts.yaml missing 'groups' key"
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
        """The prometheus.yml references the GPU and proxy alert rule files."""
        config = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")
        rule_files = config.get("rule_files", [])
        for expected in (
            "gpu_compute_rules.yaml",
            "gpu_memory_alerts.yaml",
            "proxy_5xx_alerts.yaml",
        ):
            assert expected in rule_files, (
                f"prometheus.yml must reference {expected} (relative to monitoring/)"
            )

    def test_removed_dead_rule_files_are_not_referenced(self):
        """Dead rule files must not come back: they could never fire."""
        config = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")
        rule_files = config.get("rule_files", [])
        for dead in ("gpu_vram_alerts.yaml", "llama_memory_alerts.yaml"):
            assert dead not in rule_files, (
                f"{dead} was removed because its expressions can never be "
                f"satisfied on this host; do not re-add it"
            )
            assert not (MONITORING_DIR / dead).exists(), (
                f"{dead} should have been deleted"
            )

    def test_no_rule_uses_a_dead_or_pinned_series(self, prometheus_alert_rules):
        """No rule may select a series that is unpublished or pinned."""
        offenders = []
        for rule_file in prometheus_alert_rules:
            for expr in _rule_expressions(rule_file):
                for forbidden in FORBIDDEN_RULE_SERIES:
                    if forbidden in expr:
                        offenders.append(f"{rule_file.name}: {expr.strip()}")
        assert not offenders, (
            "Rules reference series that are unpublished or meaningless on this "
            "host (rocm_vram_* has no exporter, gpu_busy_percent is pinned at "
            "100%, llama_process_rss_bytes is always 0):\n" + "\n".join(offenders)
        )


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


class TestGpuComputeRules:
    """GPU utilisation rules derive from power draw, not the pinned busy counter."""

    def _rules(self) -> list[dict]:
        data = assert_valid_yaml(MONITORING_DIR / "gpu_compute_rules.yaml")
        return [r for g in data["groups"] for r in g.get("rules", [])]

    def _recording_rule(self, name: str) -> dict:
        for r in self._rules():
            if r.get("record") == name:
                return r
        pytest.fail(f"Recording rule {name} not found in gpu_compute_rules.yaml")

    def test_gpu_compute_rules_exists(self):
        assert (MONITORING_DIR / "gpu_compute_rules.yaml").exists()

    def test_recording_rules_present(self):
        """The recording rules the docs/queries promise must exist."""
        records = {r.get("record") for r in self._rules()}
        for expected in ("gpu:power_watts:amdgpu", "gpu:computing:bool"):
            assert expected in records, (
                f"Missing recording rule {expected}; got {sorted(records)}"
            )

    def test_power_series_is_restricted_to_the_amdgpu_chip(self):
        """The power series must select the GPU hwmon device, not CPU/other."""
        expr = self._recording_rule("gpu:power_watts:amdgpu")["expr"]
        assert "node_hwmon_power_average_watt" in expr
        assert 'chip_name="amdgpu"' in expr, (
            "Power series must join node_hwmon_chip_names on chip_name=\"amdgpu\" "
            "so a CPU/other hwmon power sensor cannot be mistaken for GPU power"
        )

    def test_computing_bool_uses_a_boolean_comparison(self):
        """Without `bool` the comparison filters series instead of yielding 0/1."""
        expr = self._recording_rule("gpu:computing:bool")["expr"]
        assert ">= bool" in expr, (
            "gpu:computing:bool must use a `bool` comparison so idle samples are "
            "recorded as 0; otherwise averaging over a window drops idle samples "
            "and overstates utilisation"
        )

    def test_threshold_matches_the_proxy_wedge_power_threshold(self):
        """The rule threshold must not drift from the proxy's configured value."""
        cfg = assert_valid_yaml(PROJECT_ROOT / "proxy" / "config.yaml")
        # llama_gpu_wedge_power_low_threshold_watts lives in the server block.
        threshold = _find_key(
            cfg, "llama_gpu_wedge_power_low_threshold_watts"
        )
        assert threshold is not None, (
            "proxy/config.yaml no longer defines "
            "llama_gpu_wedge_power_low_threshold_watts"
        )
        expr = self._recording_rule("gpu:computing:bool")["expr"]
        assert f">= bool {threshold}" in expr, (
            f"gpu:computing:bool must use the proxy's configured compute "
            f"threshold ({threshold} W) so the two layers agree; got: {expr}"
        )

    def test_sustained_compute_alert_fires_on_the_power_series(self):
        """A sustained-compute alert must exist and use the power-derived series."""
        alerts = [r for r in self._rules() if r.get("alert")]
        assert alerts, "gpu_compute_rules.yaml defines no alerting rule"
        alert = alerts[0]
        assert "gpu:computing:bool" in alert["expr"], (
            "The sustained-compute alert must be driven by the power-derived "
            f"recording rule, not an ad-hoc expression: {alert['expr']}"
        )
        assert alert.get("for"), "A sustained-compute alert must set a `for` duration"


class TestGpuMemoryAlerts:
    """GPU memory pressure is measured on GTT, the pool models actually use."""

    def _rules(self) -> list[dict]:
        data = assert_valid_yaml(MONITORING_DIR / "gpu_memory_alerts.yaml")
        return [r for g in data["groups"] for r in g.get("rules", [])]

    def test_alerts_target_the_published_gtt_series(self):
        """Every GPU memory alert must key off gpu_gtt_used_bytes."""
        for rule in self._rules():
            assert "gpu_gtt_used_bytes" in rule["expr"], (
                f"Alert {rule.get('alert')} must use gpu_gtt_used_bytes "
                f"(published by the node_exporter textfile sampler); got: "
                f"{rule['expr']}"
            )

    def test_warning_and_critical_severities_present(self):
        severities = {r["labels"]["severity"] for r in self._rules()}
        assert severities == {"warning", "critical"}, (
            f"Expected one warning and one critical GPU memory alert, got {severities}"
        )

    def test_critical_threshold_is_higher_than_warning(self):
        """Warning must fire before critical, or the pair is useless."""
        by_sev = {r["labels"]["severity"]: r["expr"] for r in self._rules()}
        warn = _threshold_fraction(by_sev["warning"])
        crit = _threshold_fraction(by_sev["critical"])
        assert warn is not None and crit is not None, (
            f"Could not parse thresholds from {by_sev}"
        )
        assert warn < crit, (
            f"Warning threshold {warn} must be below critical threshold {crit}"
        )


class TestPrometheusRuleValidation:
    """promtool is the authority on rule syntax and behaviour; use it directly."""

    def test_config_and_rules_validate(self):
        promtool = _promtool()
        if promtool is None:
            pytest.skip("promtool not installed")
        result = subprocess.run(
            [promtool, "check", "config", str(MONITORING_DIR / "prometheus.yml")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "promtool check config failed:\n"
            f"{result.stdout}\n{result.stderr}\n"
            "Note: rule_files paths are resolved relative to prometheus.yml's own "
            "directory, so they must not repeat the 'monitoring/' prefix."
        )

    def test_rule_unit_tests_pass(self):
        """Behavioural tests: threshold semantics, alert firing, duration."""
        promtool = _promtool()
        if promtool is None:
            pytest.skip("promtool not installed")
        test_file = MONITORING_DIR / "tests" / "gpu_rules_test.yml"
        assert test_file.exists(), f"Missing {test_file}"
        result = subprocess.run(
            [promtool, "test", "rules", str(test_file)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, (
            "promtool test rules failed:\n" f"{result.stdout}\n{result.stderr}"
        )


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
