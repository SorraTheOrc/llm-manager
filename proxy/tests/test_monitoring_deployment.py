"""Tests for monitoring deployment configuration files.

Validates that:
- Prometheus configuration (monitoring/prometheus.yml) is valid YAML and has
  the required sections (scrape_configs, rule_files).
- Grafana provisioning files (monitoring/grafana/datasources/ and dashboards/)
  are valid YAML with correct structure.
- Systemd unit files (docs/systemd/prometheus.service,
  docs/systemd/grafana-server.service) are syntactically valid ini-style units.
- Alert rule files (monitoring/*.yaml) are valid Prometheus rule files.
- The documented units match the deployed stack: tailnet-only Prometheus binds,
  loopback-only exporters, a working sampler script, and install steps that
  work on a clean host (LP-0MVO8DORX004P05G).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

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
# Helpers: documented systemd units
#
# docs/systemd/<name>.service are *documents* that wrap the unit in a fenced
# `ini` block, and monitoring/README.md installs a unit by extracting that
# block. These helpers reproduce the documented extraction so the tests can
# assert the documented units, not the markdown around them.
# ---------------------------------------------------------------------------

FENCE = "```"


def _norm(text: str) -> str:
    """Collapse whitespace and lowercase, so prose assertions survive wrapping.

    Documentation is hard-wrapped, so a phrase like "publishes no releases" can
    arrive as "publishes no\n  releases". Assertions on documented wording must
    not depend on where the author wrapped a line.
    """
    return " ".join(text.split()).lower()


def _fenced_blocks(path: Path) -> list[tuple[str, str]]:
    """Return (language, body) for every fenced code block in a markdown file."""
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    lang = ""
    for line in path.read_text().splitlines():
        if line.startswith(FENCE):
            if current is None:
                lang = line[len(FENCE):].strip()
                current = []
            else:
                blocks.append((lang, "\n".join(current) + "\n"))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def _unit_text(unit_name: str) -> str:
    """Return the unit documented in docs/systemd/<unit_name>.service."""
    path = DOCS_SYSTEMD_DIR / f"{unit_name}.service"
    assert path.exists(), f"Missing unit document: {path}"
    blocks = [body for lang, body in _fenced_blocks(path) if lang == "ini"]
    assert len(blocks) == 1, (
        f"{path} must contain exactly one fenced `ini` block (the unit itself); "
        f"found {len(blocks)}. monitoring/README.md installs units by extracting "
        f"that single block."
    )
    return blocks[0]


def _sampler_script() -> str:
    """Return the GPU sysfs sampler script documented with gpu-metrics.service."""
    path = DOCS_SYSTEMD_DIR / "gpu-metrics.service"
    for lang, body in _fenced_blocks(path):
        if lang == "bash" and body.lstrip().startswith("#!"):
            return body
    raise AssertionError(
        f"{path} does not document the sampler script in a fenced bash block"
    )


def _exec_start_args(unit_text: str) -> list[str]:
    """Return the argv of ExecStart, joining systemd line continuations."""
    parts: list[str] = []
    collecting = False
    for line in unit_text.splitlines():
        if line.startswith("ExecStart="):
            line = line[len("ExecStart="):]
            collecting = True
        elif not collecting:
            continue
        stripped = line.rstrip()
        continued = stripped.endswith("\\")
        parts.append(stripped.rstrip("\\").strip())
        if not continued:
            break
    argv = " ".join(parts).split()
    assert argv, "ExecStart is empty in the documented unit"
    return argv


def _flag_values(unit_text: str, flag: str) -> list[str]:
    """Return every value passed to *flag* in ExecStart."""
    prefix = f"{flag}="
    return [a[len(prefix):] for a in _exec_start_args(unit_text) if a.startswith(prefix)]


def _split_host_port(address: str) -> tuple[str, str]:
    """Split a listen address into (host, port); bracketed IPv6 supported."""
    if address.startswith("["):
        host, _, port = address.partition("]:")
        return host + "]", port
    host, _, port = address.rpartition(":")
    return host, port


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

    def test_promtool_loads_every_configured_rule_file(self):
        """AC 1/4 (LP-0MVO8DO7100309LH): the config *and* all its rule files load.

        `promtool check config` exits 0 even when a `rule_files` entry matches
        nothing (a glob that resolves to no files is silently skipped), so the
        exit code alone does not prove the rule paths are right. Assert what
        promtool actually loaded against what the config lists: the historical
        bug (rule paths repeating the `monitoring/` prefix) then fails here.

        This is the regression guard for rule-path drift: `rule_files` paths are
        resolved relative to prometheus.yml's own directory, so they must stay
        bare filenames rather than being listed once per file with a prefix.
        """
        promtool = _promtool()
        if promtool is None:
            pytest.skip("promtool not installed")
        configured = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")["rule_files"]
        result = subprocess.run(
            [promtool, "check", "config", str(MONITORING_DIR / "prometheus.yml")],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, (
            "promtool check config failed:\n" f"{result.stdout}\n{result.stderr}"
        )
        match = re.search(r"SUCCESS: (\d+) rule files? found", result.stdout)
        assert match, (
            "promtool did not report how many rule files it loaded:\n"
            f"{result.stdout}"
        )
        assert int(match.group(1)) == len(configured), (
            f"promtool loaded {match.group(1)} rule file(s) but prometheus.yml "
            f"lists {len(configured)} ({configured}); an entry that matches no "
            f"file is silently skipped, so its alerts would never be evaluated"
        )
        unchecked = [rf for rf in configured if rf not in result.stdout]
        assert not unchecked, (
            f"promtool never checked {unchecked}; rule_files paths are resolved "
            f"relative to prometheus.yml's own directory (monitoring/), so they "
            f"must not repeat the 'monitoring/' prefix"
        )

    def test_every_rule_file_on_disk_is_loaded(self):
        """The inverse drift: a rule file present but unloaded can never fire."""
        loaded = set(assert_valid_yaml(MONITORING_DIR / "prometheus.yml")["rule_files"])
        unloaded = []
        for path in sorted(MONITORING_DIR.glob("*.yaml")):
            data = assert_valid_yaml(path)
            if isinstance(data, dict) and data.get("groups") and path.name not in loaded:
                unloaded.append(path.name)
        assert not unloaded, (
            f"rule file(s) {unloaded} live in monitoring/ but prometheus.yml does "
            f"not load them, so their alerts would never be evaluated"
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


class TestRocmExporterRouteUnavailable:
    """docs/systemd/rocm-exporter.service must record the dead end, not offer a unit."""

    def _doc(self) -> str:
        path = DOCS_SYSTEMD_DIR / "rocm-exporter.service"
        assert path.exists(), f"Missing rocm-exporter.service at {path}"
        return path.read_text()

    def test_document_marks_the_route_unavailable(self):
        """AC 2: the rocm-exporter route is marked unavailable."""
        content = _norm(self._doc())
        assert "unavailable" in content or "not installable" in content, (
            "rocm-exporter.service must state that the route is unavailable"
        )
        assert "do not install" in content, (
            "rocm-exporter.service must tell the reader not to install it"
        )

    def test_document_states_why_it_is_unavailable(self):
        """The reason must be recorded, or the dead end gets retried."""
        content = _norm(self._doc())
        assert "no release" in content or "no published artefact" in content, (
            "rocm-exporter.service must record that amd/rocm-exporter publishes "
            "no release artefacts (that is why it cannot be installed)"
        )

    def test_document_offers_no_installable_unit(self):
        """A `ini` block here would be extractable/installable — it must not exist."""
        langs = [lang for lang, _ in _fenced_blocks(DOCS_SYSTEMD_DIR / "rocm-exporter.service")]
        assert "ini" not in langs, (
            "rocm-exporter.service must not contain an `ini` block: an "
            "installable unit for an exporter that has no artefacts is a trap"
        )

    def test_document_points_at_the_supported_route(self):
        """The replacement route must be linked from the dead end."""
        content = self._doc()
        assert "node_exporter.service" in content, (
            "rocm-exporter.service must point readers at node_exporter"
        )
        assert "gpu-metrics.service" in content, (
            "rocm-exporter.service must point readers at the textfile sampler"
        )

    def test_prometheus_does_not_scrape_the_unavailable_exporter(self):
        """Downstream config must stay consistent with the documented dead end."""
        config = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")
        jobs = {j.get("job_name") for j in config.get("scrape_configs", [])}
        assert "rocm-exporter" not in jobs, (
            "prometheus.yml must not scrape rocm-exporter: nothing can listen on "
            ":5000 because the exporter publishes no release artefacts"
        )


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

    def test_monitoring_readme_marks_rocm_exporter_unavailable(self):
        """AC 2,3: the README must not present rocm-exporter as installable."""
        path = MONITORING_DIR / "README.md"
        content = _norm(path.read_text())
        assert "rocm-exporter" in content, (
            "monitoring/README.md should record why rocm-exporter is not used"
        )
        assert "not installable" in content or "unavailable" in content, (
            "monitoring/README.md must mark the rocm-exporter route unavailable"
        )
        assert "rocm-exporter/releases" not in content, (
            "monitoring/README.md must not instruct downloading a rocm-exporter "
            "release: no such artefacts are published"
        )

    def test_monitoring_readme_has_vram_alerts_section(self):
        """AC 5,7: monitoring/README.md documents VRAM alert rules."""
        path = MONITORING_DIR / "README.md"
        content = path.read_text().lower()
        assert "vram" in content, (
            "monitoring/README.md should reference VRAM alerts"
        )


# ---------------------------------------------------------------------------
# Tests: the documented Prometheus unit matches the deployed stack
# (LP-0MVO8DORX004P05G AC 1)
# ---------------------------------------------------------------------------


class TestPrometheusUnitMatchesDeployment:
    """The documented unit must describe the deployment, not a template."""

    def _unit(self) -> str:
        return _unit_text("prometheus")

    def test_extracted_unit_is_well_formed(self):
        """The extractable unit has the sections systemd needs."""
        unit = self._unit()
        for section in ("[Unit]", "[Service]", "[Install]"):
            assert section in unit, f"prometheus.service is missing {section}"

    def test_binds_loopback_and_tailnet_addresses_only(self):
        """AC 1: binds are loopback + the two tailnet addresses, never wildcard."""
        binds = _flag_values(self._unit(), "--web.listen-address")
        assert len(binds) == 3, (
            f"expected exactly three --web.listen-address values (loopback, "
            f"tailnet IPv4, tailnet IPv6), got {binds}"
        )
        hosts = {}
        for bind in binds:
            host, port = _split_host_port(bind)
            assert port == "9090", f"{bind!r} is not a :9090 listener"
            assert host, (
                f"wildcard bind {bind!r} publishes an unauthenticated Prometheus "
                f"on every interface, including the LAN"
            )
            hosts[host] = bind
        assert set(hosts) == {
            "127.0.0.1",
            "100.79.231.101",
            "[fd7a:115c:a1e0::9b33:e765]",
        }, f"unexpected bind addresses: {sorted(hosts)}"

    def test_no_wildcard_bind_regression(self):
        """The old `--web.listen-address=:9090` must not come back."""
        args = _exec_start_args(self._unit())
        assert "--web.listen-address=:9090" not in args, (
            "a wildcard bind puts an unauthenticated Prometheus (with a working "
            "POST /-/quit) on the LAN; bind loopback and the tailnet addresses"
        )

    def test_external_url_points_at_a_bind_address(self):
        """AC 1: --web.external-url must name one of the addresses we bind.

        Prometheus returns absolute links (graph, targets, alerts) built from
        --web.external-url, so a value that is not bound produces links to a
        dead address — which is what happens when the node is re-registered and
        only one of the two flags is updated.
        """
        urls = _flag_values(self._unit(), "--web.external-url")
        assert len(urls) == 1, f"expected one external URL, got {urls}"
        parts = urlsplit(urls[0])
        assert parts.scheme == "http"
        bound = {
            (_split_host_port(bind)[0].strip("[]"), _split_host_port(bind)[1])
            for bind in _flag_values(self._unit(), "--web.listen-address")
        }
        assert (parts.hostname, str(parts.port)) in bound, (
            f"--web.external-url host {parts.hostname}:{parts.port} is not one of "
            f"the bound addresses {sorted(bound)}"
        )

    def test_tailscaled_is_ordered_before_the_service(self):
        """AC 1: the bind targets only exist once tailscale0 is up."""
        unit = self._unit()
        after = re.search(r"^After=(.*)$", unit, re.MULTILINE)
        wants = re.search(r"^Wants=(.*)$", unit, re.MULTILINE)
        assert after and wants, "unit must declare After= and Wants="
        assert "tailscaled.service" in after.group(1).split(), (
            "tailscaled.service must be in After= or the first bind attempt "
            "races the tailnet interface"
        )
        assert "tailscaled.service" in wants.group(1).split(), (
            "tailscaled.service must be in Wants= so it is pulled in"
        )

    def test_retention_is_ninety_days(self):
        """AC 1: retention matches the deployed 90 days."""
        assert _flag_values(self._unit(), "--storage.tsdb.retention.time") == ["90d"]

    def test_re_registration_note_names_both_flags(self):
        """AC 1: the comment explains what to change on Tailscale re-registration."""
        comments = "\n".join(
            line for line in self._unit().splitlines()
            if line.lstrip().startswith("#")
        )
        lowered = comments.lower()
        assert "re-registered" in lowered or "re-registration" in lowered, (
            "the unit comment must explain what happens when the node is "
            "re-registered in Tailscale"
        )
        assert "--web.listen-address" in comments and "--web.external-url" in comments, (
            "the re-registration note must name both flags that need updating"
        )

    def test_lifecycle_and_admin_apis_stay_disabled(self):
        """Exposure decision: no unauthenticated reload/delete on the tailnet."""
        args = _exec_start_args(self._unit())
        for flag in ("--web.enable-lifecycle", "--web.enable-admin-api"):
            assert flag not in args, (
                f"{flag} would expose config reload / destructive TSDB deletion to "
                f"every tailnet peer with no authentication; keep it disabled"
            )

    def test_config_file_is_the_repository_config(self):
        """The unit reads monitoring/prometheus.yml in place (no copy step)."""
        values = _flag_values(self._unit(), "--config.file")
        assert len(values) == 1
        assert values[0].endswith("monitoring/prometheus.yml")


@pytest.mark.parametrize("unit", ["prometheus", "node_exporter", "gpu-metrics"])
def test_documented_units_match_the_deployed_units(unit: str) -> None:
    """Drift guard: on a host running the stack, doc and deployment must agree.

    Skipped where the stack is not deployed (CI), so the guard costs nothing
    off-host while catching the real failure mode — a repository that documents
    a stack nobody is running.
    """
    deployed = Path.home() / ".config" / "systemd" / "user" / f"{unit}.service"
    if not deployed.exists():
        pytest.skip(f"{unit}.service is not installed on this host")
    assert _unit_text(unit) == deployed.read_text(), (
        f"docs/systemd/{unit}.service no longer matches the deployed unit; update "
        f"the doc (or reinstall the unit from the doc) so the repository describes "
        f"the running stack"
    )


# ---------------------------------------------------------------------------
# Tests: node_exporter and gpu-metrics unit docs
# (LP-0MVO8DORX004P05G AC 2)
# ---------------------------------------------------------------------------


class TestNodeExporterUnitDoc:
    """node_exporter is the only exporter; it must be documented as such."""

    def test_extracted_unit_is_well_formed(self):
        unit = _unit_text("node_exporter")
        for section in ("[Unit]", "[Service]", "[Install]"):
            assert section in unit, f"node_exporter.service is missing {section}"

    def test_binds_loopback_only(self):
        """An unauthenticated exporter is scraped locally, so loopback suffices."""
        binds = _flag_values(_unit_text("node_exporter"), "--web.listen-address")
        assert binds == ["127.0.0.1:9100"], (
            f"node_exporter must bind loopback only, got {binds}"
        )

    def test_textfile_and_hwmon_collectors_are_enabled(self):
        """AC 2: GPU memory arrives via textfile, GPU power via hwmon."""
        args = _exec_start_args(_unit_text("node_exporter"))
        assert "--collector.hwmon" in args, (
            "the hwmon collector supplies node_hwmon_power_average_watt, which is "
            "the GPU utilisation signal"
        )
        textfile_dirs = _flag_values(_unit_text("node_exporter"), "--collector.textfile.directory")
        assert len(textfile_dirs) == 1, (
            f"expected one textfile directory, got {textfile_dirs}"
        )
        assert textfile_dirs[0].endswith("node_exporter/textfile"), (
            "the textfile directory must be the one gpu-metrics.service writes to"
        )

    def test_documents_the_pinned_busy_counter_caveat(self):
        """AC 2: the doc must not present gpu_busy_percent as utilisation."""
        content = _norm((DOCS_SYSTEMD_DIR / "node_exporter.service").read_text())
        assert "gpu_busy_percent" in content
        assert "not a utilisation" in content or "pinned at 100%" in content, (
            "node_exporter.service must warn that gpu_busy_percent is pinned at "
            "100% on this APU and is not a utilisation signal"
        )


class TestGpuMetricsSamplerAndUnit:
    """The sampler must be reproducible from the docs and behave as documented."""

    def _sampler_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "gpu-metrics-textfile.sh"
        path.write_text(_sampler_script())
        return path

    def _fake_sysfs(self, tmp_path: Path, **attributes: str) -> Path:
        device = tmp_path / "sys" / "class" / "drm" / "card0" / "device"
        device.mkdir(parents=True)
        for name, value in attributes.items():
            (device / name).write_text(f"{value}\n")
        return device

    def _run_sampler(self, tmp_path: Path, device: Path) -> Path:
        out_dir = tmp_path / "textfile"
        out_dir.mkdir()
        result = subprocess.run(
            ["bash", str(self._sampler_path(tmp_path)), str(out_dir)],
            capture_output=True,
            text=True,
            env={**os.environ, "GPU_SYSFS": str(device)},
        )
        assert result.returncode == 0, f"sampler failed: {result.stderr}"
        return out_dir

    def _emitted_series(self, out_dir: Path) -> dict[str, str]:
        text = (out_dir / "gpu.prom").read_text()
        return dict(
            re.findall(r"^(gpu_\w+)\{gpu=\"card0\"\} (\S+)$", text, re.MULTILINE)
        )

    def test_unit_runs_the_documented_sampler_every_ten_seconds(self):
        """AC 2: the 10 s cadence is in the unit, not just in prose."""
        command = " ".join(_exec_start_args(_unit_text("gpu-metrics")))
        assert ".local/bin/gpu-metrics-textfile.sh" in command, (
            "the unit must run the sampler script documented alongside it"
        )
        match = re.search(r"sleep (\d+)", command)
        assert match, "the unit must re-run the sampler on an interval"
        assert int(match.group(1)) == 10, (
            f"the documented interval is 10 s, the unit sleeps {match.group(1)} s"
        )

    def test_sampler_script_is_valid_bash(self, tmp_path):
        result = subprocess.run(
            ["bash", "-n", str(self._sampler_path(tmp_path))],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"documented sampler is not valid bash:\n{result.stderr}"

    def test_sampler_publishes_the_documented_series(self, tmp_path):
        """Behaviour: run the documented script against fixture sysfs attributes."""
        device = self._fake_sysfs(
            tmp_path,
            gpu_busy_percent="100",
            mem_info_vram_used="309604352",
            mem_info_vram_total="1073741824",
            mem_info_gtt_used="36573331456",
        )
        out_dir = self._run_sampler(tmp_path, device)
        assert self._emitted_series(out_dir) == {
            "gpu_busy_percent": "100",
            "gpu_vram_used_bytes": "309604352",
            "gpu_vram_total_bytes": "1073741824",
            "gpu_gtt_used_bytes": "36573331456",
        }

    def test_sampler_metric_names_are_documented_in_both_units(self, tmp_path):
        """AC 2: the names the sampler emits appear in its unit doc and upstream."""
        names = set(self._emitted_series(
            self._run_sampler(
                tmp_path,
                self._fake_sysfs(
                    tmp_path,
                    gpu_busy_percent="100",
                    mem_info_vram_used="1",
                    mem_info_vram_total="2",
                    mem_info_gtt_used="3",
                ),
            )
        ))
        assert names, "sampler emitted no series"
        for doc in ("gpu-metrics.service", "node_exporter.service"):
            content = (DOCS_SYSTEMD_DIR / doc).read_text()
            missing = sorted(n for n in names if n not in content)
            assert not missing, (
                f"{doc} does not document the series the sampler publishes: {missing}"
            )

    def test_sampler_writes_nothing_when_busy_percent_is_unreadable(self, tmp_path):
        """Publishing zeros for an unreadable sensor would look like an idle GPU."""
        device = self._fake_sysfs(tmp_path, mem_info_gtt_used="42")
        out_dir = self._run_sampler(tmp_path, device)
        assert not (out_dir / "gpu.prom").exists(), (
            "the sampler must not publish fabricated values when the driver "
            "attributes are unreadable"
        )

    def test_sampler_publishes_vram_without_gtt_when_gtt_is_unreadable(self, tmp_path):
        """A partial attribute set yields the series that are real, not zeros."""
        device = self._fake_sysfs(
            tmp_path,
            gpu_busy_percent="100",
            mem_info_vram_used="10",
            mem_info_vram_total="20",
        )
        out_dir = self._run_sampler(tmp_path, device)
        assert self._emitted_series(out_dir) == {
            "gpu_busy_percent": "100",
            "gpu_vram_used_bytes": "10",
            "gpu_vram_total_bytes": "20",
        }

    def test_output_is_written_atomically(self, tmp_path):
        """node_exporter must never read a half-written textfile."""
        device = self._fake_sysfs(tmp_path, gpu_busy_percent="7")
        out_dir = self._run_sampler(tmp_path, device)
        leftovers = sorted(p.name for p in out_dir.iterdir() if p.name != "gpu.prom")
        assert not leftovers, f"sampler left temporary files behind: {leftovers}"
        assert (out_dir / "gpu.prom").read_text().endswith("\n")


# ---------------------------------------------------------------------------
# Tests: monitoring/README.md install steps and exposure decision
# (LP-0MVO8DORX004P05G AC 3)
# ---------------------------------------------------------------------------


def _readme() -> str:
    path = MONITORING_DIR / "README.md"
    assert path.exists(), f"Missing monitoring/README.md at {path}"
    return path.read_text()


def _install_helper() -> str:
    """The unit-extraction helper documented in monitoring/README.md."""
    for lang, body in _fenced_blocks(MONITORING_DIR / "README.md"):
        if lang == "bash" and "unit_from_doc()" in body:
            return body
    raise AssertionError(
        "monitoring/README.md does not document a `unit_from_doc` helper; copying "
        "docs/systemd/*.service installs markdown, not a systemd unit"
    )


class TestMonitoringReadmeInstallSteps:
    """The install steps must work on a clean host (AC 3)."""

    def test_install_steps_do_not_copy_markdown_as_a_unit(self):
        """`cp docs/systemd/prometheus.service` installs a markdown file."""
        content = _norm(_readme())
        assert "cp docs/systemd/prometheus.service" not in content, (
            "docs/systemd/*.service are documents wrapping the unit in an `ini` "
            "block; copying one installs markdown that systemd cannot parse"
        )
        assert "cp docs/systemd/grafana-server.service" not in content

    @pytest.mark.parametrize("unit", ["prometheus", "node_exporter", "gpu-metrics"])
    def test_install_helper_installs_the_documented_unit(self, unit: str, tmp_path):
        """AC 3/5: run the documented command and compare with the documented unit.

        This executes the README's own helper rather than re-implementing the
        extraction, so the documented procedure is what gets verified.
        """
        result = subprocess.run(
            ["bash", "-c", f"{_install_helper()}\nunit_from_doc {unit}\n"],
            cwd=PROJECT_ROOT,
            env={**os.environ, "HOME": str(tmp_path)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"documented install helper failed:\n{result.stderr}"
        installed = tmp_path / ".config" / "systemd" / "user" / f"{unit}.service"
        assert installed.exists(), (
            f"the documented helper did not install {unit}.service"
        )
        assert installed.read_text() == _unit_text(unit), (
            f"the installed {unit}.service differs from the unit in "
            f"docs/systemd/{unit}.service"
        )
        for section in ("[Unit]", "[Service]", "[Install]"):
            assert section in installed.read_text(), (
                f"installed {unit}.service is not a valid unit (missing {section})"
            )

    def test_no_removed_prometheus_console_directories(self):
        """Prometheus 3.x ships no consoles/, so nothing may copy them."""
        raw = _readme()
        assert not re.search(r"^\s*cp\b.*console", raw, re.MULTILINE), (
            "the install steps must not copy consoles/ or console_libraries/: "
            "Prometheus 3.x does not ship them"
        )
        assert "no longer ship" in _norm(raw) or "ships no" in _norm(raw), (
            "the guide should say why consoles/ is no longer copied"
        )

    def test_rule_path_resolution_rule_is_documented(self):
        """AC 3: the rule-path gotcha that blocks every cold start."""
        lowered = _norm(_readme())
        assert "relative to" in lowered and "rule_files" in lowered, (
            "the guide must state that rule_files paths are resolved relative to "
            "the config file's directory"
        )
        assert "monitoring/monitoring" in lowered, (
            "the guide must show the concrete failure mode (repeating the "
            "`monitoring/` prefix resolves to monitoring/monitoring/...)"
        )

    def test_exposure_boundary_is_documented_as_a_decision(self):
        """AC 3: tailnet-only is a decision, with its consequences spelled out."""
        lowered = _norm(_readme())
        assert "tailnet" in lowered
        assert "no built-in authentication" in lowered or "no authentication" in lowered, (
            "the guide must state that Prometheus has no authentication"
        )
        assert "/-/quit" in lowered, (
            "the guide must state that any tailnet peer can POST /-/quit and stop "
            "the server"
        )
        assert "/-/reload" in lowered and "not enabled" in lowered, (
            "the guide must record that the lifecycle API is disabled"
        )
        assert "admin apis disabled" in lowered or "admin api" in lowered, (
            "the guide must record that the admin (delete-series) API is disabled"
        )

    def test_lan_is_documented_as_refused(self):
        """The boundary is only meaningful if it is verifiable."""
        lowered = _norm(_readme())
        assert "connection refused" in lowered, (
            "the guide must show how to verify the LAN address refuses connections"
        )

    def test_retention_matches_the_documented_unit(self):
        """AC 1/4: every place that states retention agrees with the unit flag."""
        assert "90 d" in _norm(_readme()) or "90 days" in _norm(_readme())
        assert _flag_values(_unit_text("prometheus"), "--storage.tsdb.retention.time") == ["90d"]

    def test_every_referenced_unit_document_exists(self):
        """Links from the guide must resolve, or a fresh host hits a dead end."""
        content = _readme()
        referenced = set(re.findall(r"(?:\.\./)?docs/systemd/([a-z0-9_.-]+\.service)", content))
        assert referenced, "monitoring/README.md references no unit documents"
        for name in sorted(referenced):
            assert (DOCS_SYSTEMD_DIR / name).exists(), (
                f"monitoring/README.md references docs/systemd/{name}, which does "
                f"not exist"
            )

    def test_installed_stack_is_documented_as_a_table(self):
        """AC 3: the reader must be able to see what is supposed to be running."""
        content = _readme()
        for unit in ("prometheus.service", "node_exporter.service", "gpu-metrics.service"):
            assert unit in content, f"monitoring/README.md does not mention {unit}"


# ---------------------------------------------------------------------------
# Tests: monitoring/README.md query cookbook
# (LP-0MVO8DORX004P05G AC 4)
# ---------------------------------------------------------------------------


def _recording_rule_names() -> set[str]:
    names: set[str] = set()
    for path in sorted(MONITORING_DIR.glob("*.yaml")):
        data = assert_valid_yaml(path)
        if not isinstance(data, dict):
            continue
        for group in data.get("groups", []) or []:
            for rule in group.get("rules", []) or []:
                if rule.get("record"):
                    names.add(rule["record"])
    return names


def _sampler_series_names() -> set[str]:
    """Series names the documented sampler publishes."""
    return set(re.findall(r"echo \"(gpu_\w+)\{", _sampler_script())) | set(
        re.findall(r"# TYPE (gpu_\w+) gauge", _sampler_script())
    )


class TestMonitoringReadmeCookbook:
    """The cookbook must answer the four operational questions asked of it."""

    def test_covers_gpu_computing_fraction_from_power(self):
        content = _readme()
        assert "avg_over_time(gpu:computing:bool" in content, (
            "the cookbook must show how to get the GPU computing fraction from "
            "the power-derived recording rule"
        )
        assert "gpu:power_watts:avg5m" in content

    def test_covers_gpu_memory_gtt_and_vram(self):
        content = _readme()
        assert "gpu_gtt_used_bytes" in content
        assert "gpu_vram_used_bytes" in content and "gpu_vram_total_bytes" in content

    def test_covers_host_metrics(self):
        content = _readme()
        assert "node_" in content
        assert "node_load1" in content or "node_memory_MemAvailable_bytes" in content

    def test_covers_the_retention_window(self):
        content = _readme()
        assert "storage.tsdb.retention.time" in content
        assert "min(timestamp(" in content, (
            "the cookbook must show how to measure the history actually retained, "
            "not just the configured window"
        )

    def test_recording_rules_used_in_the_cookbook_exist(self):
        """Anti-drift: the guide must not query a recording rule nobody defines."""
        mentioned = set(re.findall(r"\bgpu:[a-z0-9_:]+", _readme()))
        assert mentioned, "the cookbook mentions no recording rules"
        undefined = sorted(mentioned - _recording_rule_names())
        assert not undefined, (
            f"monitoring/README.md queries recording rules that do not exist in "
            f"monitoring/*.yaml: {undefined}"
        )

    def test_gpu_series_used_in_the_cookbook_are_published(self):
        """Anti-drift: every gpu_* series in the guide comes from the sampler.

        File names such as `gpu_memory_alerts.yaml` are excluded by the negative
        lookahead on `.`, so only bare series identifiers are compared.
        """
        mentioned = set(re.findall(r"\bgpu_[a-z_]+\b(?!\.)", _readme()))
        assert mentioned, "the cookbook mentions no gpu_* series"
        unpublished = sorted(mentioned - _sampler_series_names())
        assert not unpublished, (
            f"monitoring/README.md references gpu series that no exporter "
            f"publishes: {unpublished} (published: {sorted(_sampler_series_names())})"
        )
