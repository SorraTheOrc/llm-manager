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
- The Grafana dashboard's GPU memory panel selects only series the documented
  sampler actually publishes (LP-0MVOHRLPB0086Q5T).

Also enforces the `FORBIDDEN_RULE_SERIES` contract on **every** PromQL source —
rule files (YAML) and the GPU utilisation CLI (Python, `QUERIES` mapping) — so a
`gpu_busy_percent` query cannot be introduced in either place
(LP-0MVT1NVAJ007KUGO).
"""

import ast
import os
import re
import shutil
import subprocess
import sys
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
        "host_memory_alerts.yaml",
        "llama_memory_alerts.yaml",
        "proxy_5xx_alerts.yaml",
        "slots_query_alerts.yaml",
        "remote_stream_alerts.yaml",
    ],
}

# Series that must never appear in a rule expression:
# - rocm_vram_*: the AMD rocm-exporter publishes no release artifacts, so no
#   exporter ever served these series and rules using them could never fire.
#   The dashboard is covered separately by TestGrafanaDashboardVramPanel, which
#   rejects any series the documented sampler does not publish
#   (LP-0MVOHRLPB0086Q5T).
# - gpu_busy_percent: a two-state detector whose step sits at ~30 W on this
#   Strix Halo APU (1-10 % below 30 W, 92-99 % from 30 W upward), which is below
#   the idle-with-display band (p50 45 W), so idle-with-context and computing
#   both read ~99 % and the counter inverts across 40-60 W / 90-100 W, i.e. it
#   is not a utilisation signal (LP-0MS9GA6QL007XTBP, quantified in
#   LP-0MVSQQO910077LYL and re-measured in LP-0MW5K7WKZ005VJXA).
# llama_process_rss_bytes is deliberately NOT forbidden: LP-0MVO9F877008ZCP7
# fixed the always-0 publication (llama-server runs in a podman container, so
# the proxy now resolves RSS via the container process tree / cgroup
# memory.current instead of a host process handle) and restored
# llama_memory_alerts.yaml on the series.
FORBIDDEN_RULE_SERIES = [
    "rocm_vram_",
    "gpu_busy_percent",
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


# Non-rule PromQL consumers that must obey the same forbidden-series contract
# (LP-0MVT1NVAJ007KUGO). A rule file is scanned as YAML; a CLI is scanned as
# Python, because its expressions live in a ``QUERIES`` mapping. Copy-pasting a
# ``gpu_busy_percent`` query into the CLI would otherwise be a silent
# utilisation regression that no rule-file check could see.
CLI_QUERY_SOURCES = [PROJECT_ROOT / "scripts" / "gpu_utilisation.py"]

#: Assignment/argument names whose string values are PromQL, by convention.
_QUERY_NAME_SUFFIXES = ("QUERY", "QUERIES", "EXPR", "EXPRESSIONS")
_QUERY_ARGUMENT_NAMES = ("query", "queries", "expr")


def _promql_expressions_in_python(source: str, filename: str = "<string>") -> list[str]:
    """Return the PromQL expressions held in a Python query consumer.

    Only *query* strings are extracted, never prose, so a module may document
    why ``gpu_busy_percent`` must not be used without tripping its own guard.
    A string counts as a query expression when it is:

    * a value in an assignment whose target name ends in ``QUERY``, ``QUERIES``,
      ``EXPR`` or ``EXPRESSIONS`` (the CLI's ``QUERIES`` dict is the canonical
      case — nested dicts/lists are walked);
    * a literal passed as the ``query``/``queries``/``expr`` argument of a call.

    Docstrings and ``#`` comments are never scanned. Duplicates are collapsed
    in first-seen order so an offender is reported once.
    """
    tree = ast.parse(source, filename=filename)

    def collect(node: ast.AST) -> None:
        """Collect string *values* below *node*, never mapping keys.

        A ``QUERIES`` mapping's keys are human labels (``"computing_workload"``);
        only the values are PromQL. f-string literals are collected part-wise so
        an expression that interpolates the step still exposes its series names.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append(node.value)
        elif isinstance(node, ast.Dict):
            for value in node.values:
                collect(value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                collect(element)
        elif isinstance(node, ast.JoinedStr):
            for value in node.values:
                collect(value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            collect(node.left)
            collect(node.right)
        elif isinstance(node, ast.FormattedValue):
            collect(node.value)

    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.upper().endswith(
                    _QUERY_NAME_SUFFIXES
                ):
                    collect(node.value)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id.upper().endswith(_QUERY_NAME_SUFFIXES)
                and node.value is not None
            ):
                collect(node.value)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in _QUERY_ARGUMENT_NAMES:
                    collect(keyword.value)

    deduped: list[str] = []
    for expr in found:
        if expr not in deduped:
            deduped.append(expr)
    return deduped


def _forbidden_series_offenders(
    expressions, *, source_label: str, forbidden=None
) -> list[str]:
    """Return ``<source>: <expr>`` for every expression selecting a forbidden series."""
    series_list = FORBIDDEN_RULE_SERIES if forbidden is None else forbidden
    offenders: list[str] = []
    for expr in expressions:
        for series in series_list:
            if series in expr:
                offenders.append(f"{source_label}: {' '.join(expr.split())}")
                break
    return offenders


def _consumer_guard_offenders(path: Path) -> list[str]:
    """Return forbidden-series offenders for a Python PromQL consumer.

    Fails loudly when no query expression can be found: a rename from
    ``QUERIES`` to something unrecognised must not turn the guard vacuously
    green.
    """
    expressions = _promql_expressions_in_python(path.read_text(), filename=str(path))
    assert expressions, (
        f"{path.name}: no PromQL query expressions found. The "
        f"FORBIDDEN_RULE_SERIES guard would be vacuous — keep query expressions "
        f"in a mapping named QUERIES/QUERY/EXPR/EXPRESSIONS, or pass them as a "
        f"query=/expr= argument."
    )
    return _forbidden_series_offenders(expressions, source_label=path.name)


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


def _fake_gpu_sysfs(tmp_path: Path, **attributes: str) -> Path:
    """Create a fixture sysfs tree holding the amdgpu attributes given."""
    device = tmp_path / "sys" / "class" / "drm" / "card0" / "device"
    device.mkdir(parents=True)
    for name, value in attributes.items():
        (device / name).write_text(f"{value}\n")
    return device


def _run_documented_sampler(tmp_path: Path, device: Path) -> Path:
    """Run the documented sampler against *device* and return its textfile dir."""
    sampler = tmp_path / "gpu-metrics-textfile.sh"
    sampler.write_text(_sampler_script())
    out_dir = tmp_path / "textfile"
    out_dir.mkdir()
    result = subprocess.run(
        ["bash", str(sampler), str(out_dir)],
        capture_output=True,
        text=True,
        env={**os.environ, "GPU_SYSFS": str(device)},
    )
    assert result.returncode == 0, f"sampler failed: {result.stderr}"
    return out_dir


def _sampler_emitted_series(out_dir: Path) -> dict[str, str]:
    """Return the {series: value} pairs the sampler wrote to gpu.prom."""
    text = (out_dir / "gpu.prom").read_text()
    return dict(
        re.findall(r'^(gpu_\w+)\{gpu="card0"\} (\S+)$', text, re.MULTILINE)
    )


#: Fixture readings for a fully readable amdgpu device: every documented series
#: is published, so the returned set is the sampler's whole output surface.
#: ``gpu_busy_percent`` stays in the fixture on purpose: the device still exposes
#: the attribute, so the sampler's output proves it ignores a readable one
#: (retired by LP-0MW5K8ZDZ007YSIX).
FULL_GPU_SYSFS = {
    "gpu_busy_percent": "100",
    "mem_info_vram_used": "309604352",
    "mem_info_vram_total": "1073741824",
    "mem_info_gtt_used": "36573331456",
}


def _published_gpu_series(tmp_path: Path) -> set[str]:
    """Run the documented sampler and return every series name it publishes.

    The sampler in `docs/systemd/gpu-metrics.service` is the single source of
    truth for which `gpu_*` series this host actually serves: executing it
    against a fully readable fixture yields exactly the set node_exporter's
    textfile collector exposes. Anything a panel queries outside this set can
    only ever render "No data".
    """
    device = _fake_gpu_sysfs(tmp_path, **FULL_GPU_SYSFS)
    published = set(_sampler_emitted_series(_run_documented_sampler(tmp_path, device)))
    assert published, "the documented sampler published no series"
    return published


#: PromQL keywords/aggregation modifiers that are not series selectors.
PROMQL_NON_SERIES_WORDS = frozenset(
    {
        "and",
        "bool",
        "by",
        "group_left",
        "group_right",
        "ignoring",
        "inf",
        "nan",
        "offset",
        "on",
        "or",
        "unless",
        "without",
    }
)


def _selected_series(expr: str) -> set[str]:
    """Return the series selectors of a PromQL expression.

    Label values and label matchers are stripped first, so `status="5xx"` or
    `endpoint="/v1/chat/completions"` cannot be mistaken for a series; names
    followed by `(` are function calls (`rate(...)`), not selectors.
    """
    cleaned = re.sub(r'"[^"]*"', '""', expr)
    cleaned = re.sub(r"\{[^{}]*\}", "{}", cleaned)
    return {
        name
        for name in re.findall(
            r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()", cleaned
        )
        if name not in PROMQL_NON_SERIES_WORDS
    }


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

    def test_llama_memory_alerts_registered(self):
        """AC5: llama_memory_alerts.yaml is restored, valid, references the
        now-working llama_process_rss_bytes series, and is registered in both
        prometheus.yml rule_files and REQUIRED_RULE_FILES."""
        path = MONITORING_DIR / "llama_memory_alerts.yaml"
        data = assert_valid_yaml(path)
        assert "groups" in data, "llama_memory_alerts.yaml missing 'groups' key"
        groups = data["groups"]
        assert len(groups) >= 1
        rules = groups[0].get("rules", [])
        assert len(rules) >= 1
        expressions = "\n".join(str(r.get("expr", "")) for r in rules)
        assert "llama_process_rss_bytes" in expressions, (
            "restored rule must select the (now working) llama_process_rss_bytes series"
        )
        config = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")
        assert "llama_memory_alerts.yaml" in config.get("rule_files", [])
        assert "llama_memory_alerts.yaml" in REQUIRED_RULE_FILES["prometheus.yml"]

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
        # gpu_vram_alerts.yaml stays dead (rocm_vram_* has no exporter).
        # llama_memory_alerts.yaml was removed alongside it but is RESTORED by
        # LP-0MVO9F877008ZCP7 (the llama_process_rss_bytes series it selects
        # now carries a real value) — see test_llama_memory_alerts_registered.
        for dead in ("gpu_vram_alerts.yaml",):
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
            offenders.extend(
                _forbidden_series_offenders(
                    _rule_expressions(rule_file), source_label=rule_file.name
                )
            )
        assert not offenders, (
            "Rules reference series that are unpublished or meaningless on this "
            "host (rocm_vram_* has no exporter, gpu_busy_percent is pinned at "
            "100%):\n" + "\n".join(offenders)
        )


class TestForbiddenSeriesGuard:
    """The forbidden-series guard must fail on violations, not be vacuously green.

    The contract is enforced on three source kinds: Prometheus rule files (YAML,
    scanned in ``TestAlertRuleFiles``), the GPU utilisation CLI (Python, whose
    PromQL lives in a ``QUERIES`` mapping) and the sysfs sampler script (bash,
    documented inside ``docs/systemd/gpu-metrics.service``). These tests prove
    the guard would have caught a ``gpu_busy_percent`` query while still
    allowing the documentation that explains the caveat
    (LP-0MVT1NVAJ007KUGO AC3/AC4, sampler coverage LP-0MW5K8ZDZ007YSIX).
    """

    def test_gpu_utilisation_cli_uses_no_forbidden_series(self):
        """The GPU utilisation CLI must not query a forbidden series either.

        Skipped until the CLI lands (LP-0MVT1O52G000LMC6); the guard then
        activates automatically, and fails if it can no longer find any query
        expression (non-vacuous).
        """
        offenders: list[str] = []
        checked: list[str] = []
        for path in CLI_QUERY_SOURCES:
            if not path.exists():
                continue
            checked.append(path.name)
            offenders.extend(_consumer_guard_offenders(path))

        if not checked:
            pytest.skip(
                "scripts/gpu_utilisation.py not implemented yet "
                "(LP-0MVT1O52G000LMC6); guard activates when it lands"
            )
        assert not offenders, (
            "The GPU utilisation CLI queries a series that is unpublished or "
            "meaningless on this host (gpu_busy_percent saturates at ~20 W, "
            "rocm_vram_* has no exporter):\n" + "\n".join(offenders)
        )

    def test_guard_flags_a_forbidden_series_in_a_queries_mapping(self):
        source = (
            "QUERIES = {\n"
            '            "computing": "avg_over_time(gpu:computing:bool[2h])",\n'
            '            "busy": "avg_over_time(gpu_busy_percent[2h])",\n'
            "        }\n"
        )

        expressions = _promql_expressions_in_python(source)
        offenders = _forbidden_series_offenders(expressions, source_label="synthetic")

        assert expressions == [
            "avg_over_time(gpu:computing:bool[2h])",
            "avg_over_time(gpu_busy_percent[2h])",
        ]
        assert offenders == ["synthetic: avg_over_time(gpu_busy_percent[2h])"]

    def test_guard_flags_a_forbidden_series_in_a_nested_query_expression(self):
        source = (
            "QUERIES = {\n"
            '            "gpu": {"vram": "rocm_vram_used_bytes / rocm_vram_total_bytes"},\n'
            "        }\n"
        )

        offenders = _forbidden_series_offenders(
            _promql_expressions_in_python(source), source_label="synthetic"
        )

        assert len(offenders) == 1
        assert "rocm_vram_" in offenders[0]

    def test_guard_scans_expressions_passed_as_a_query_argument(self):
        source = (
            "def run(getter):\n"
            '            return getter(query="avg_over_time(gpu_busy_percent[1h])")\n'
        )

        offenders = _forbidden_series_offenders(
            _promql_expressions_in_python(source), source_label="synthetic"
        )

        assert len(offenders) == 1
        assert "gpu_busy_percent" in offenders[0]

    def test_guard_allows_documentation_and_comment_mentions(self):
        source = (
            '"""GPU utilisation. gpu_busy_percent is pinned and must not be used."""\n'
            "# gpu_busy_percent saturates at ~20 W: never query it here.\n"
            'NOTE = "gpu_busy_percent reads ~100% above 20 W"\n'
            "QUERIES = {\n"
            '            "computing": "avg_over_time(gpu:computing:bool[2h])",\n'
            "        }\n"
        )

        expressions = _promql_expressions_in_python(source)

        assert expressions == ["avg_over_time(gpu:computing:bool[2h])"]
        assert _forbidden_series_offenders(expressions, source_label="synthetic") == []

    def test_guard_fails_when_no_query_expression_is_found(self, tmp_path):
        """A rename must not silently disable the guard (non-vacuous green)."""
        path = tmp_path / "cli.py"
        path.write_text('EXPRESSIONS_RENAMED = {"a": "up"}\n')

        with pytest.raises(AssertionError, match="no PromQL query expressions"):
            _consumer_guard_offenders(path)

    def test_guard_passes_on_a_clean_consumer(self, tmp_path):
        path = tmp_path / "cli.py"
        path.write_text(
            'QUERIES = {"computing": "avg_over_time(gpu:computing:bool[2h])"}\n'
        )

        expressions = _promql_expressions_in_python(path.read_text())
        assert expressions == ["avg_over_time(gpu:computing:bool[2h])"]
        assert _consumer_guard_offenders(path) == []

    def test_sampler_script_is_guarded_against_the_retired_series(self):
        """AC 3: re-adding the attribute to the sampler fails a test.

        Retiring the series removed its only publisher, but the guard that kept
        it out of rules was a lint of the rule files — a sampler that started
        reading the attribute again would publish a plausible-looking series
        with no consumer and no failure. The sampler script is now its own
        source kind under the same ``FORBIDDEN_RULE_SERIES`` contract.
        """
        offenders = _forbidden_series_offenders(
            [_sampler_script()], source_label="gpu-metrics-textfile.sh"
        )
        assert not offenders, (
            "the documented sampler script reads a forbidden series; the "
            "retirement (LP-0MW5K8ZDZ007YSIX) is not a rule-file lint only:\n"
            + "\n".join(offenders)
        )

    def test_sampler_guard_flags_a_re_added_attribute(self):
        """Non-vacuous: a re-added read/emit line must be reported."""
        restored = (
            _sampler_script()
            + 'busy=$(read_attr gpu_busy_percent)\n'
            + 'echo "gpu_busy_percent{gpu=\\"card0\\"} $busy"\n'
        )
        offenders = _forbidden_series_offenders(
            [restored], source_label="gpu-metrics-textfile.sh"
        )
        assert len(offenders) == 1
        assert "gpu_busy_percent" in offenders[0]


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


class TestHostMemoryAlerts:
    """Pre-OOM alerts: MemAvailable and GTT-vs-RAM, not GTT-vs-pool.

    LP-0MVZ810NG008A3EW AC3. At the 2026-11-02T12:03:08Z global OOM, GTT was
    53 % of its 128 GiB pool — below `gpu_memory_alerts.yaml`'s 75 % warning —
    while only 0.2 % of the host's 124 GiB of RAM was available and the kernel
    killed the proxy. These rules must therefore key off `MemAvailable` and off
    GTT relative to *physical RAM*; anything pool-relative cannot fire in time.
    """

    def _rules(self) -> list[dict]:
        data = assert_valid_yaml(MONITORING_DIR / "host_memory_alerts.yaml")
        return [r for g in data["groups"] for r in g.get("rules", [])]

    def test_rule_file_is_loaded_by_prometheus(self):
        loaded = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")["rule_files"]
        assert "host_memory_alerts.yaml" in loaded

    def test_available_memory_rules_use_memavailable_over_memtotal(self):
        exprs = [r["expr"] for r in self._rules() if "node_memory_MemAvailable_bytes" in r["expr"]]
        assert len(exprs) == 2, f"expected a warning+critical pair, got {exprs}"
        for expr in exprs:
            assert "node_memory_MemTotal_bytes" in expr, (
                "a MemAvailable rule must be a fraction of MemTotal, not an "
                f"absolute byte count that depends on the host: {expr}"
            )

    def test_gtt_rules_compare_against_physical_ram(self):
        gtt_rules = [r for r in self._rules() if "gpu_gtt_used_bytes" in r["expr"]]
        assert len(gtt_rules) == 2, f"expected a warning+critical pair, got {gtt_rules}"
        for rule in gtt_rules:
            assert "node_memory_MemTotal_bytes" in rule["expr"], (
                "GTT must be compared with physical RAM: the GTT pool (128 GiB) "
                "is larger than RAM (124 GiB) on this host, so a pool-relative "
                f"threshold cannot fire before the kernel watermark: {rule['expr']}"
            )

    @staticmethod
    def _comparison_value(expr: str) -> float | None:
        """The threshold literal a ratio rule compares against (< 0.12, > 0.50)."""
        match = re.search(r"[<>]=?\s*(\d+(?:\.\d+)?)\s*$", expr.strip())
        return float(match.group(1)) if match else None

    def test_warning_fires_before_critical_for_each_pair(self):
        for series, fragment, usage_rule in (
            ("MemAvailable", "node_memory_MemAvailable_bytes", False),
            ("GPU GTT", "gpu_gtt_used_bytes", True),
        ):
            by_sev = {
                r["labels"]["severity"]: r["expr"]
                for r in self._rules()
                if fragment in r["expr"]
            }
            assert set(by_sev) == {"warning", "critical"}, (series, by_sev)
            warn = self._comparison_value(by_sev["warning"])
            crit = self._comparison_value(by_sev["critical"])
            assert warn is not None and crit is not None, by_sev
            if usage_rule:
                assert warn < crit, (
                    f"{series}: the warning must fire at a LOWER usage fraction "
                    f"than the critical (warn={warn}, crit={crit})"
                )
            else:
                assert warn > crit, (
                    f"{series}: the warning must fire at a HIGHER available "
                    f"fraction than the critical (warn={warn}, crit={crit})"
                )

    def test_incident_value_trips_the_warning_thresholds(self):
        """The measured 2026-11-02 ratios must have been alarming."""
        snapshot = {
            "available_fraction": 245_576 / 130_494_996,
            "gtt_of_ram": (68_482 * 1024 * 1024) / (130_494_996 * 1024),
        }
        by_series = {
            "node_memory_MemAvailable_bytes": 0.12,
            "gpu_gtt_used_bytes": 0.50,
        }
        assert snapshot["available_fraction"] < by_series["node_memory_MemAvailable_bytes"]
        assert snapshot["gtt_of_ram"] > by_series["gpu_gtt_used_bytes"]


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


DASHBOARD_PATH = MONITORING_DIR / "grafana_llama_memory_dashboard.json"
PROVISIONED_DASHBOARD_PATH = (
    MONITORING_DIR / "grafana" / "dashboards" / "grafana_llama_memory_dashboard.json"
)


class TestGrafanaDashboardVramPanel:
    """The GPU memory panel must chart memory pools this host really publishes.

    LP-0MVOHRLPB0086Q5T: the panel used to query `rocm_vram_*`, series no process
    on this host serves (`amd/rocm-exporter` publishes no release artefacts), so
    it could only ever render "No data". The published set is now derived by
    running the documented sampler, not from a hand-maintained list, so a panel
    selecting an unpublished series fails here instead of shipping.
    """

    def _load_dashboard(self) -> dict:
        import json

        with open(DASHBOARD_PATH) as f:
            return json.load(f)

    def _gpu_memory_panel(self, dashboard: dict) -> dict:
        panels = dashboard.get("panels", [])
        for panel in panels:
            if "vram" in panel.get("title", "").lower():
                return panel
        pytest.fail(
            f"No GPU memory (VRAM) panel found in dashboard, "
            f"found titles: {[p.get('title', '') for p in panels]}"
        )

    def test_dashboard_has_vram_panel(self):
        """AC 2: the GPU memory panel survives, and its title still names VRAM."""
        panel = self._gpu_memory_panel(self._load_dashboard())
        assert "vram" in panel["title"].lower()

    def test_panel_identity_is_preserved(self):
        """Retyping the panel would orphan it in a deployed Grafana instance."""
        dashboard = self._load_dashboard()
        panel = self._gpu_memory_panel(dashboard)
        assert panel["id"] == 4
        assert panel["type"] == "graph"
        assert panel["datasource"] == "Prometheus"
        assert panel["gridPos"] == {"h": 8, "w": 24, "x": 0, "y": 18}
        assert dashboard["title"] == "LLama Server Memory"
        assert dashboard["schemaVersion"] == 16

    def test_title_names_both_pools(self):
        """AC 2: GTT is where models live; VRAM is the BIOS carve-out."""
        title = self._gpu_memory_panel(self._load_dashboard())["title"].lower()
        assert "gtt" in title, "the title must name the GTT pool models map into"
        assert "vram" in title, "the title must name the VRAM carve-out"
        assert "carve-out" in title, (
            "the title must label VRAM as a carve-out, not as the model pool"
        )
        assert "model" in title, (
            "the title must say which pool the models occupy, or a reader will "
            "assume the 1 GiB carve-out holds them"
        )

    def test_every_target_legend_names_its_series_and_vram_is_the_carve_out(self):
        """AC 2: legends identify the series and mark the 1 GiB carve-out."""
        panel = self._gpu_memory_panel(self._load_dashboard())
        targets = panel.get("targets", [])
        assert targets, "the GPU memory panel has no targets"
        for target in targets:
            selectors = _selected_series(target.get("expr", ""))
            assert len(selectors) == 1, (
                f"each target should select exactly one series, {target.get('expr')!r} "
                f"selects {sorted(selectors)}"
            )
            series = next(iter(selectors))
            legend = target.get("legendFormat", "")
            assert series in legend, (
                f"legendFormat {legend!r} does not name its series {series!r}; a "
                f"reader cannot tell which exporter series a line comes from"
            )
            if "vram" in series:
                assert "carve-out" in legend.lower(), (
                    f"VRAM legend {legend!r} must mark the ~1 GiB BIOS carve-out"
                )

    def test_every_target_selects_a_series_the_sampler_publishes(self, tmp_path):
        """AC 1/AC 3: every expression resolves to a series this host serves.

        The published set comes from executing the documented sampler against a
        fully readable fixture sysfs, so a `gpu_*` name the sampler does not
        emit fails just as loudly as a dead `rocm_*` one.
        """
        published = _published_gpu_series(tmp_path)
        panel = self._gpu_memory_panel(self._load_dashboard())
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            selectors = _selected_series(expr)
            assert selectors, f"target selects no series: {expr!r}"
            unpublished = sorted(selectors - published)
            assert not unpublished, (
                f"panel target {expr!r} selects series no exporter publishes: "
                f"{unpublished} (the sampler publishes {sorted(published)})"
            )

    @pytest.mark.parametrize(
        "path", [DASHBOARD_PATH, PROVISIONED_DASHBOARD_PATH], ids=["review", "provisioned"]
    )
    def test_dashboards_no_longer_query_rocm_series(self, path: Path):
        """AC 1: the dead rocm-exporter route must not come back, in either copy."""
        assert "rocm_" not in path.read_text(), (
            f"{path.name} still queries rocm_* series, which nothing publishes; "
            f"grep -c 'rocm_' must report 0 for both dashboard copies"
        )

    def test_dashboard_copies_are_byte_identical(self):
        """The provisioning copy must not drift from the reviewed copy."""
        assert PROVISIONED_DASHBOARD_PATH.read_bytes() == DASHBOARD_PATH.read_bytes(), (
            f"{DASHBOARD_PATH} and {PROVISIONED_DASHBOARD_PATH} have diverged; "
            f"edit both together"
        )

    def test_dashboard_version_was_bumped(self):
        """AC 2: Grafana only picks up a changed dashboard when `version` moves."""
        version = self._load_dashboard().get("version")
        assert isinstance(version, int) and version >= 2, (
            f"the dashboard version must be bumped past its original 1 after the "
            f"GPU panel rewrite, got {version!r}"
        )


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
        assert "not a utilisation" in content, (
            "node_exporter.service must warn that gpu_busy_percent saturates on "
            "this APU and is not a utilisation signal"
        )


# Glyphs and prose that carry the measured GPU-utilisation-source decision
# (LP-0MVSQQO910077LYL).  The counter saturates at ~20 W; `amd-smi` reports N/A;
# `amdtop`/`rocm-smi` only surface the same counter.
USAGE_ANALYSIS_SCRIPTS = (
    PROJECT_ROOT / ".pi" / "skills" / "proxy-usage-analysis" / "scripts"
)
USAGE_ANALYSIS_SKILL_DIR = PROJECT_ROOT / ".pi" / "skills" / "proxy-usage-analysis"

# The stale premise: "pinned at 100% regardless of GPU state".  Kernel 7.0.0-34
# responds to load within seconds, so the claimed mechanism was wrong even
# though the conclusion (not a utilisation measure) holds — the counter
# saturates above ~30 W.  See LP-0MVSQQO910077LYL, LP-0MW5K7WKZ005VJXA.
STALE_PINNED_CLAIM = "pinned at 100% regardless"

AMD_SHELLOUT_TOOLS = ("amd-smi", "amdsmi", "rocm-smi", "amdtop")

# ---------------------------------------------------------------------------
# The re-measured `gpu_busy_percent` discrimination evidence (LP-0MW5K7WKZ005VJXA)
#
# The retirement of the counter used to be justified with a pre-2026-10-31
# measurement (2026-10-28, kernel 7.0.0-34: 94.5 % mean busy, 45.2 % false-busy,
# "saturates at ~20 W").  That figure no longer describes the host, so it was
# re-measured on the kernel the host actually runs and every quote replaced.
# These constants pin the published evidence: a re-quote that drifts from the
# capture (or an ageing claim nobody re-ran) fails the suite instead of ageing
# silently.
# ---------------------------------------------------------------------------

#: Capture date and kernel of the published re-measurement.
GPU_BUSY_CAPTURE_DATE = "2026-11-06"
GPU_BUSY_CAPTURE_KERNEL = "7.0.0-38"

#: Observed 15 s samples per window (not the grid size: the series has gaps).
GPU_BUSY_POST_SAMPLES = "25,463"
GPU_BUSY_PRE_SAMPLES = "28,655"

#: (published band label, pre-reboot mean busy %, post-reboot mean busy %).
#: Kernel 7.0.0-38 post-reboot, kernel 7.0.0-34 pre-reboot, same recipe both
#: ways so the two columns are like-for-like.
GPU_BUSY_BANDS = (
    ("0\u201310 W", "2.0 %", "1.7 %"),
    ("10\u201320 W", "4.7 %", "3.6 %"),
    ("20\u201330 W", "10.2 %", "7.2 %"),
    ("30\u201335 W", "99.4 %", "92.1 %"),
    ("35\u201340 W", "99.8 %", "98.4 %"),
    ("40\u201360 W", "99.5 %", "99.2 %"),
    ("60\u201380 W", "98.6 %", "97.0 %"),
    ("80\u201390 W", "99.6 %", "98.6 %"),
    ("90\u2013100 W", "99.4 %", "98.7 %"),
    ("100\u2013125 W", "98.4 %", "97.3 %"),
)

#: Window-summary shares: (label, pre-reboot %, post-reboot %).
GPU_BUSY_WINDOW_SUMMARY = (
    ("mean `gpu_busy_percent`", "95.6 %", "80.0 %"),
    ("samples at compute power (\u2265 90 W)", "31.5 %", "51.0 %"),
    ("false-busy (`busy \u2265 90 %` \u2227 `power < 40 W`)", "35.1 %", "7.9 %"),
    ("false-idle (`busy < 90 %` \u2227 `power \u2265 90 W`)", "0.5 %", "1.7 %"),
)

#: The mechanism, stated once in the canonical place.  The step moved: it is a
#: two-state detector whose transition sits at ~30 W, which is *below* the
#: idle-with-display band (p50 45 W), so idle and computing both read ~99 %.
GPU_BUSY_CANONICAL_MECHANISM = "two-state detector"
GPU_BUSY_STEP_WATTS = "~30 W"

#: The failure mode the fresh evidence actually shows: it inverts across the
#: band that matters (40-60 W idle-with-display reads *higher* than 90-100 W
#: computing), so no threshold inside the working range can rescue it.
GPU_BUSY_INVERSION_PHRASE = "inverts"

#: Source files that quote the counter's mechanism (AC 3 of
#: LP-0MW5K7WKZ005VJXA enumerates them).  The retired mechanism must be gone
#: from every one, and every one must point at the canonical measurement.
GPU_BUSY_QUOTE_SOURCES = (
    MONITORING_DIR / "README.md",
    MONITORING_DIR / "gpu_compute_rules.yaml",
    PROJECT_ROOT / "proxy" / "config.yaml",
    PROJECT_ROOT / "proxy" / "proxy" / "backends" / "llama.py",
    PROJECT_ROOT / "proxy" / "proxy" / "host_pressure.py",
    USAGE_ANALYSIS_SCRIPTS / "gpu_busy.py",
    USAGE_ANALYSIS_SKILL_DIR / "REFERENCE.md",
    USAGE_ANALYSIS_SKILL_DIR / "SKILL.md",
    DOCS_SYSTEMD_DIR / "gpu-metrics.service",
    DOCS_SYSTEMD_DIR / "node_exporter.service",
    DOCS_SYSTEMD_DIR / "rocm-exporter.service",
)

#: The retired "~20 W latch" claims.  Normalised (whitespace collapsed,
#: lowercased) before matching, so wrapping and spacing do not hide a survivor.
STALE_GPU_BUSY_CLAIMS = (
    "saturates at ~20 w",
    "saturates at ~20w",
    "~20w latch",
    "~20 w latch",
    "latches to ~100 %",
    "reads ~100 % once the gpu draws more than ~20 w",
    "reads ~100% for any draw above ~20 w",
    "90 % whenever the gpu draws more than ~20 w",
)


def _promql_recipe(readme: str) -> str:
    """Return the published re-measurement recipe block from the README."""
    marker = "re-measuring `gpu_busy_percent` discrimination"
    lowered = readme.lower()
    assert marker in lowered, (
        "monitoring/README.md must carry a 'Re-measuring `gpu_busy_percent` "
        "discrimination' subsection: the bands have to be recomputed from the "
        "Prometheus store with exact PromQL, not from remembered numbers"
    )
    section = readme[lowered.index(marker) :]
    blocks = [
        body
        for lang, body in _fenced_blocks_text(section)
        if lang in ("promql", "text", "bash")
    ]
    assert blocks, (
        "the re-measurement subsection must publish its PromQL in a fenced "
        "block so an operator can re-run it"
    )
    return "\n".join(blocks)


def _fenced_blocks_text(text: str) -> list[tuple[str, str]]:
    """(language, body) for every fenced code block in a markdown *text*."""
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    lang = ""
    for line in text.splitlines():
        if line.startswith(FENCE):
            if current is None:
                lang = line[len(FENCE) :].strip()
                current = []
            else:
                blocks.append((lang, "\n".join(current)))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


class TestGpuBusyDiscriminationRemeasurement:
    """The `gpu_busy_percent` verdict is quoted from a fresh measurement.

    LP-0MW5K7WKZ005VJXA: the conclusion (not a utilisation signal) survives, but
    the stated reason was measured before the 2026-10-31 reboot on kernel
    7.0.0-34.  The counter's step moved and the observed failure mode on kernel
    7.0.0-38 is a *inversion* across 40-60 W (idle with display) against
    90-100 W (computing) — not a "~20 W latch".  These guards pin the capture
    (date, kernel, window, sample counts) alongside the figures, so a stale
    quote fails the suite rather than ageing silently.
    """

    def _readme(self) -> str:
        return (MONITORING_DIR / "README.md").read_text()

    def test_capture_is_dated_and_kernel_pinned(self):
        """AC 1: the published evidence names its date, kernel and window."""
        content = _norm(self._readme())
        assert GPU_BUSY_CAPTURE_DATE in content, (
            "monitoring/README.md must state the re-measurement capture date "
            f"({GPU_BUSY_CAPTURE_DATE}) next to the figures it quotes"
        )
        assert GPU_BUSY_CAPTURE_KERNEL in content, (
            "monitoring/README.md must state the kernel the re-measurement ran "
            f"on ({GPU_BUSY_CAPTURE_KERNEL}); a figure without its kernel is "
            "exactly what made the old claim age silently"
        )
        assert GPU_BUSY_POST_SAMPLES.lower() in content, (
            f"monitoring/README.md must publish the post-reboot sample count "
            f"({GPU_BUSY_POST_SAMPLES})"
        )
        assert GPU_BUSY_PRE_SAMPLES.lower() in content, (
            f"monitoring/README.md must publish the pre-reboot sample count "
            f"({GPU_BUSY_PRE_SAMPLES}) it compares against"
        )
        # The window must be the whole post-boot window (or longer), and it must
        # say so: a shorter window would be a different measurement.
        assert "post-boot window" in content, (
            "monitoring/README.md must name the post-boot window as the "
            "measurement window"
        )

    def test_banded_table_matches_the_published_measurement(self):
        """AC 1: every band's pre/post mean busy is the measured value."""
        content = _norm(self._readme())
        for label, pre, post in GPU_BUSY_BANDS:
            row_prefix = _norm(label)
            assert row_prefix in content, (
                f"monitoring/README.md has no row for the {label} power band"
            )
            for value in (pre, post):
                assert _norm(value) in content, (
                    f"monitoring/README.md does not quote {value} for the "
                    f"{label} band; the table must hold the measured figure"
                )
            # The two figures must sit on the *same* table row as the band, so
            # a copy-paste slip between columns is caught, not just a stray
            # number somewhere in the file.
            rows = [
                line
                for line in self._readme().splitlines()
                if _norm(line).startswith(f"| {row_prefix}")
            ]
            assert len(rows) == 1, (
                f"expected exactly one table row for {label}, found {len(rows)}"
            )
            row = _norm(rows[0])
            assert _norm(pre) in row and _norm(post) in row, (
                f"the {label} row must carry both the pre-reboot ({pre}) and "
                f"post-reboot ({post}) mean busy; got {rows[0]!r}"
            )

    def test_window_summary_matches_the_published_measurement(self):
        """AC 1: the false-busy share and the rest of the summary are quoted."""
        content = _norm(self._readme())
        for label, pre, post in GPU_BUSY_WINDOW_SUMMARY:
            assert _norm(label) in content, (
                f"monitoring/README.md must publish the '{label}' summary figure"
            )
            assert _norm(pre) in content, (
                f"monitoring/README.md must quote the pre-reboot {label} ({pre})"
            )
            assert _norm(post) in content, (
                f"monitoring/README.md must quote the post-reboot {label} ({post})"
            )

    def test_mechanism_is_the_thirty_watt_step_and_the_inversion(self):
        """AC 2: the mechanism is the observed step + inversion, stated once.

        The old text said the counter "saturates at ~20 W".  The fresh numbers
        put the step at ~30 W *and* show the failure that actually matters: it
        cannot separate idle-with-context from compute, because 40-60 W (idle
        with a display) reads higher than 90-100 W (computing).
        """
        content = _norm(self._readme())
        assert GPU_BUSY_CANONICAL_MECHANISM in content, (
            "monitoring/README.md must state that the counter is a \"two-state "
            "detector\", not a percentage"
        )
        assert GPU_BUSY_STEP_WATTS.lower() in content, (
            f"monitoring/README.md must state the measured step ({GPU_BUSY_STEP_WATTS})"
        )
        assert GPU_BUSY_INVERSION_PHRASE in content, (
            "monitoring/README.md must name the observed failure mode: the "
            "counter inverts across the 40-60 W / 90-100 W bands"
        )

    def test_false_busy_share_is_explained_as_band_occupancy(self):
        """AC 2: the 35.1 % -> 7.9 % drop is occupancy, not a counter change.

        Without this the fresh numbers invite the opposite error — someone
        reads "7.9 % false-busy" as "the counter got better" and re-adopts it.
        The drop tracks the share of time spent in the 30-40 W band exactly.
        """
        content = _norm(self._readme())
        assert "band occupancy" in content, (
            "monitoring/README.md must explain that the false-busy share tracks "
            "how long the host sat in the 30-40 W band, not a change in the counter"
        )
        assert "7.7 %" in content, (
            "the 30-40 W occupancy that explains the post-reboot false-busy "
            "share (7.7 %) must be published"
        )

    def test_recipe_is_published_as_exact_promql(self):
        """AC 4: the bands are reproducible from the store, not remembered."""
        recipe = _promql_recipe(self._readme())
        for fragment in (
            "gpu_busy_percent",
            "gpu:power_watts:amdgpu",
            "avg_over_time",
            "count_over_time",
            "and on ()",
        ):
            assert fragment in recipe, (
                f"the published recipe is missing {fragment!r}; the bands must "
                "be recomputed from the Prometheus store with exact PromQL"
            )
        # The window must be anchored, otherwise the recipe silently measures a
        # different window every time it is run and can never reproduce a number.
        assert "@ " in recipe, (
            "the recipe must pin its evaluation instant with a PromQL `@` "
            "modifier, so re-running it reproduces the published figures"
        )

    def test_no_source_restates_the_twenty_watt_latch(self):
        """AC 3: every stale quote of the retired mechanism is gone."""
        survivors: list[str] = []
        for path in GPU_BUSY_QUOTE_SOURCES:
            assert path.exists(), f"expected {path} to exist"
            content = _norm(path.read_text())
            for claim in STALE_GPU_BUSY_CLAIMS:
                if claim in content:
                    survivors.append(f"{path.relative_to(PROJECT_ROOT)}: {claim!r}")
        assert not survivors, (
            "the retired ~20 W latch mechanism is still quoted; replace it with "
            "the measured ~30 W step and the 40-60 W / 90-100 W inversion "
            "(LP-0MW5K7WKZ005VJXA):\n" + "\n".join(survivors)
        )

    def test_every_source_points_at_the_canonical_measurement(self):
        """AC 2: the mechanism is stated once and cross-referenced elsewhere.

        Each source that quotes the counter's behaviour must either carry the
        canonical ~30 W step itself or point at monitoring/README.md, so a
        reader never lands on an unqualified version of the claim.
        """
        stale: list[str] = []
        for path in GPU_BUSY_QUOTE_SOURCES:
            content = _norm(path.read_text())
            if "gpu_busy" not in content:
                continue
            cross_referenced = "monitoring/readme.md" in content
            states_the_step = GPU_BUSY_STEP_WATTS.lower() in content
            if not (cross_referenced or states_the_step):
                stale.append(str(path.relative_to(PROJECT_ROOT)))
        assert not stale, (
            "these sources quote gpu_busy_percent without either the measured "
            "~30 W step or a pointer to monitoring/README.md:\n"
            + "\n".join(stale)
        )


class TestGpuBusyPercentSeriesRetirement:
    """C4 (LP-0MW5K8ZDZ007YSIX): the counter is no longer a published series.

    The guard keeping the counter out of rules was a lint of the rule files
    while the sampler kept publishing it — a plausible-looking series with a
    real reader nowhere. These tests pin the retirement: the sampler no longer
    reads the attribute, the metric tables no longer advertise the series, the
    decision carries its evidence, and the cookbook says the samples in the
    store are historical only.
    """

    SERIES = "gpu_busy_percent"

    def _unit_doc(self) -> str:
        return (DOCS_SYSTEMD_DIR / "gpu-metrics.service").read_text()

    def test_unit_doc_records_the_retirement_and_its_evidence(self):
        """AC 1: the decision and the measurement it cites live with the sampler."""
        content = _norm(self._unit_doc())
        assert "retire" in content, (
            "gpu-metrics.service must record that the series was retired, not "
            "just quietly stop emitting it"
        )
        assert "no longer published" in content or "not published" in content, (
            "gpu-metrics.service must say the series is no longer published"
        )
        for fact in (
            GPU_BUSY_CAPTURE_DATE,
            GPU_BUSY_CAPTURE_KERNEL,
            GPU_BUSY_STEP_WATTS.lower(),
            GPU_BUSY_INVERSION_PHRASE,
        ):
            assert fact in content, (
                f"the retirement decision must cite the current measurement; "
                f"missing {fact!r}"
            )

    def test_metric_tables_no_longer_advertise_the_series(self):
        """AC 4: a metric-table row reads as \"this is published\"; it must go."""
        for path in (
            DOCS_SYSTEMD_DIR / "gpu-metrics.service",
            DOCS_SYSTEMD_DIR / "node_exporter.service",
        ):
            metric_sections = [
                body
                for heading, body in _markdown_sections(path.read_text())
                if "metric names" in heading.lower()
            ]
            assert metric_sections, f"{path.name} must keep its metric-names table"
            table = "\n".join(metric_sections)
            assert f"| `{self.SERIES}`" not in table, (
                f"{path.name} still lists {self.SERIES} in its metric table; "
                "record the retirement in prose instead of a metric-table row"
            )
            assert self.SERIES in path.read_text(), (
                f"{path.name} must still name the retired series so a reader "
                "who greps for it learns why it stopped"
            )

    def test_sampler_script_no_longer_reads_the_attribute(self):
        """AC 3: the retirement lives in the executable sampler, not in prose."""
        assert self.SERIES not in _sampler_script(), (
            "the documented sampler still mentions the retired counter; the "
            "script must neither read nor emit it (LP-0MW5K8ZDZ007YSIX)"
        )

    def test_sampler_publishes_only_the_memory_series(self, tmp_path):
        """AC 2: GTT/VRAM survive the retirement exactly as before.

        The fixture device exposes a fully readable `gpu_busy_percent`, so this
        asserts behaviour against a device that still has the attribute.
        """
        device = _fake_gpu_sysfs(tmp_path, **FULL_GPU_SYSFS)
        published = _sampler_emitted_series(_run_documented_sampler(tmp_path, device))
        assert set(published) == {
            "gpu_gtt_used_bytes",
            "gpu_vram_used_bytes",
            "gpu_vram_total_bytes",
        }
        assert published["gpu_gtt_used_bytes"] == FULL_GPU_SYSFS["mem_info_gtt_used"]

    def test_readme_documents_the_samples_as_historical_only(self):
        """AC 5: the cookbook must explain a series that stops."""
        sections = [
            (heading, body)
            for heading, body in _markdown_sections(_readme())
            if self.SERIES in body and "historical only" in _norm(body)
        ]
        assert sections, (
            "monitoring/README.md must document the retired series as historical "
            "only, in the query cookbook a reader lands on"
        )
        heading, body = sections[0]
        text = _norm(body)
        for fact in ("90 d", "no longer", "queryable"):
            assert fact in text, (
                f"the {heading!r} section must say the samples are historical "
                f"only; missing {fact!r}"
            )

    def test_readme_does_not_present_the_retired_series_as_live(self):
        """AC 4: the install-time series list must not promise the series."""
        content = _norm(_readme())
        assert "no longer published" in content, (
            "monitoring/README.md must say the series is no longer published "
            "where it lists the GPU series the sampler serves"
        )


class TestGpuUtilisationSourceDecision:
    """The GPU-utilisation source decision is documented and not shelled out.

    LP-0MVSQQO910077LYL: keep the power-derived `gpu:computing:bool` measure and
    record, with measurement dates, that `amd-smi` returns N/A on this APU and
    that `amdtop`/`rocm-smi` surface the same saturating `gpu_busy_percent`
    counter the monitoring stack used to sample for continuity only. That series
    is retired (LP-0MW5K8ZDZ007YSIX), so the prohibition below now documents a
    series that must not come back either.
    """

    def _readme(self) -> str:
        return _norm((MONITORING_DIR / "README.md").read_text())

    def test_monitoring_readme_documents_the_amd_smi_finding(self):
        """AC 2: the operator's question is answered where the measure lives."""
        content = self._readme()
        assert "amd-smi" in content, (
            "monitoring/README.md must explain why amd-smi is not the source"
        )
        assert "amdtop" in content, (
            "monitoring/README.md must record that amdtop surfaces the same "
            "gpu_busy_percent counter"
        )
        assert "n/a" in content, (
            "monitoring/README.md must state that amd-smi reports N/A for its "
            "metrics on this APU"
        )
        assert "saturat" in content, (
            "monitoring/README.md must describe the measured saturating "
            "behaviour of gpu_busy_percent"
        )
        assert "~30 w" in content, (
            "monitoring/README.md must record the measured ~30 W step "
            "(re-measured 2026-11-06, LP-0MW5K7WKZ005VJXA); the retired claim "
            "said ~20 W"
        )

    def test_node_exporter_unit_forbids_selecting_the_busy_counter(self):
        """AC 3: the unit file carries the explicit rule-level prohibition.

        `gpu_busy_percent` saturates above ~30 W on this APU, so a rule
        selecting it fires while the GPU is idle. The prohibition has to sit
        next to the series table a rule author copies from, not only in the
        README.
        """
        content = _norm((DOCS_SYSTEMD_DIR / "node_exporter.service").read_text())
        assert "gpu_busy_percent" in content
        assert "no longer published" in content, (
            "node_exporter.service must state that gpu_busy_percent is no "
            "longer published (retired by LP-0MW5K8ZDZ007YSIX)"
        )
        assert "must not be used in any rule" in content, (
            "node_exporter.service must explicitly forbid selecting "
            "gpu_busy_percent in a rule, next to the series it describes"
        )
        assert "saturates above ~30 w" in content, (
            "node_exporter.service must state the measured saturation threshold "
            "(~30 W, re-measured 2026-11-06 in LP-0MW5K7WKZ005VJXA)"
        )

    def test_docs_replace_the_stale_pinned_claim_with_a_measurement(self):
        """AC 2/AC 3: no doc restates the unexplained 'pinned' premise."""
        for path in (
            MONITORING_DIR / "README.md",
            MONITORING_DIR / "gpu_compute_rules.yaml",
            DOCS_SYSTEMD_DIR / "node_exporter.service",
        ):
            content = _norm(path.read_text())
            assert STALE_PINNED_CLAIM not in content, (
                f"{path.name} still asserts gpu_busy_percent is 'pinned at 100% "
                "regardless' of GPU state; record the measured step instead "
                "(re-measured 2026-11-06 in LP-0MW5K7WKZ005VJXA)"
            )

    def test_usage_analysis_docs_name_the_retained_power_source(self):
        """AC 4: the report docs state the decision and cross-link monitoring."""
        reference = _norm((USAGE_ANALYSIS_SKILL_DIR / "REFERENCE.md").read_text())
        assert "amd-smi" in reference, (
            "REFERENCE.md must state that amd-smi is not the source on this APU"
        )
        assert "n/a" in reference
        assert "amdtop" in reference, (
            "REFERENCE.md must record that amdtop surfaces the same counter"
        )
        assert "monitoring/readme.md" in reference, (
            "REFERENCE.md must cross-link monitoring/README.md for the rationale"
        )

        skill = _norm((USAGE_ANALYSIS_SKILL_DIR / "SKILL.md").read_text())
        assert "amd-smi" in skill
        assert "monitoring/readme.md" in skill

    def test_gpu_busy_module_reads_prometheus_not_amd_tools(self):
        """AC 5: the report path must never shell out to an AMD tool.

        AST-based rather than textual: comments and docstrings may mention the
        tools (they explain why they are not used), while an import of
        ``subprocess`` or a call passing an ``amd-smi``/``amdtop``/``rocm-smi``
        argument would mean the decision had been reversed.
        """
        import ast

        source = (USAGE_ANALYSIS_SCRIPTS / "gpu_busy.py").read_text()
        tree = ast.parse(source)

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "subprocess" not in imported, (
            "gpu_busy.py must measure GPU busy time from Prometheus, never by "
            "invoking an external tool"
        )

        # Docstrings legitimately name the tools (they explain why they are not
        # used); only executable string literals are a reversal of the decision.
        docstrings = set()
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ]
        for tool in AMD_SHELLOUT_TOOLS:
            assert not any(tool in literal for literal in literals), (
                f"gpu_busy.py carries a {tool!r} command string; the source "
                "decision (LP-0MVSQQO910077LYL) is the Prometheus power signal"
            )
        assert "os.system" not in source, (
            "gpu_busy.py must not spawn processes with os.system"
        )


class TestGpuMetricsSamplerAndUnit:
    """The sampler must be reproducible from the docs and behave as documented."""

    def _sampler_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "gpu-metrics-textfile.sh"
        path.write_text(_sampler_script())
        return path

    def _fake_sysfs(self, tmp_path: Path, **attributes: str) -> Path:
        return _fake_gpu_sysfs(tmp_path, **attributes)

    def _run_sampler(self, tmp_path: Path, device: Path) -> Path:
        return _run_documented_sampler(tmp_path, device)

    def _emitted_series(self, out_dir: Path) -> dict[str, str]:
        return _sampler_emitted_series(out_dir)

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
        """Behaviour: run the documented script against fixture sysfs attributes.

        The fixture device still exposes `gpu_busy_percent`; the sampler must not
        publish it (retired by LP-0MW5K8ZDZ007YSIX).
        """
        device = self._fake_sysfs(
            tmp_path,
            gpu_busy_percent="100",
            mem_info_vram_used="309604352",
            mem_info_vram_total="1073741824",
            mem_info_gtt_used="36573331456",
        )
        out_dir = self._run_sampler(tmp_path, device)
        assert self._emitted_series(out_dir) == {
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

    def test_sampler_publishes_memory_when_the_busy_counter_is_unreadable(self, tmp_path):
        """The retired counter must not gate the memory series.

        Before LP-0MW5K8ZDZ007YSIX the sampler exited without writing whenever
        `gpu_busy_percent` was unreadable, so a counter nobody consumed decided
        whether GTT telemetry existed at all.
        """
        device = self._fake_sysfs(tmp_path, mem_info_gtt_used="42")
        out_dir = self._run_sampler(tmp_path, device)
        assert self._emitted_series(out_dir) == {"gpu_gtt_used_bytes": "42"}

    def test_sampler_writes_nothing_when_no_memory_attribute_is_readable(self, tmp_path):
        """A readable retired attribute alone must not keep a file alive."""
        device = self._fake_sysfs(tmp_path, gpu_busy_percent="100")
        out_dir = self._run_sampler(tmp_path, device)
        assert not (out_dir / "gpu.prom").exists(), (
            "the sampler must not publish fabricated values when the memory "
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
            "gpu_vram_used_bytes": "10",
            "gpu_vram_total_bytes": "20",
        }

    def test_output_is_written_atomically(self, tmp_path):
        """node_exporter must never read a half-written textfile."""
        device = self._fake_sysfs(tmp_path, mem_info_gtt_used="7")
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


#: Series that once were published and are now retired (LP-0MW5K8ZDZ007YSIX).
#: The cookbook still names them so a reader who queries one is not surprised
#: when it stops: the samples in the store are historical only, no longer
#: advanced. The anti-drift check below has to subtract this set explicitly —
#: and proves the sampler does not publish it any more — so keeping the mention
#: can never be mistaken for the series still being emitted.
HISTORICAL_GPU_SERIES = {"gpu_busy_percent"}


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
        assert "min_over_time(timestamp(up)[90d:" in content, (
            "the cookbook must show how to measure the history actually retained, "
            "not just the configured window — and as a subquery: an instant "
            "selector cannot see history"
        )

    def test_recording_rules_used_in_the_cookbook_exist(self):
        """Anti-drift: the guide must not query a recording rule nobody defines."""
        # Trailing punctuation is not part of an identifier: the CLI's own
        # output quotes the rule as `gpu:computing:bool:`.
        mentioned = {
            token.rstrip(":") for token in re.findall(r"\bgpu:[a-z0-9_:]+", _readme())
        }
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
        # `gpu_utilisation:` is the CLI's stderr program tag, not a series.
        mentioned = {
            token.rstrip(":") for token in re.findall(r"\bgpu_[a-z_]+\b(?!\.)", _readme())
        } - {"gpu_utilisation"}
        assert mentioned, "the cookbook mentions no gpu_* series"
        # A retired series may be documented as historical, never as published.
        published = _sampler_series_names()
        still_published = sorted(published & HISTORICAL_GPU_SERIES)
        assert not still_published, (
            "these series are documented as no longer published, so the sampler "
            f"must not emit them: {still_published}"
        )
        unpublished = sorted(mentioned - published - HISTORICAL_GPU_SERIES)
        assert not unpublished, (
            f"monitoring/README.md references gpu series that no exporter "
            f"publishes: {unpublished} (published: {sorted(published)}; "
            f"historical: {sorted(HISTORICAL_GPU_SERIES)})"
        )

# ---------------------------------------------------------------------------
# The decision-facing busy figure (LP-0MVUIMMX3005KYQ6, parent AC4/AC7)
# ---------------------------------------------------------------------------

REFERENCE_DOC = USAGE_ANALYSIS_SKILL_DIR / "REFERENCE.md"
SKILL_DOC = USAGE_ANALYSIS_SKILL_DIR / "SKILL.md"

#: Heading of the decision rule and of the single divergence-cause explanation.
AUTHORITATIVE_BUSY_HEADING = "#### Which busy figure is authoritative for decisions"
DIVERGENCE_HEADING = "#### Why the two figures diverge"
DECISION_ANCHOR = "#which-busy-figure-is-authoritative-for-decisions"
DIVERGENCE_ANCHOR = "#why-the-two-figures-diverge"

#: Labels the shipped report renders; the docs must quote these, not invent any.
GPU_BUSY_ROW_LABEL = "GPU busy — power-based (hardware)"
BUSY_UPPER_ROW_LABEL = "Busy time — upper bound (dispatch → finish)"

#: Docs carrying a busy-figure claim; every anchor link in them must resolve.
BUSY_FIGURE_DOCS = (REFERENCE_DOC, SKILL_DOC, MONITORING_DIR / "README.md")


def _documented_anchors(text: str) -> set[str]:
    """GitHub-style heading slugs plus explicit ``<a id=...>`` anchors.

    Fenced code blocks are skipped, so shell comments (``# 1. ...``) in the
    skill docs are not mistaken for headings.
    """
    anchors: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(FENCE):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if heading:
            slug = re.sub(r"[^a-z0-9 \-_]", "", heading.group(1).lower())
            anchors.add(slug.replace(" ", "-"))
    anchors.update(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    return anchors


def _reference_section(*, start: str, end: str) -> str:
    """Normalised ``REFERENCE.md`` prose between two heading lines."""
    content = (REFERENCE_DOC.read_text()).split(start, 1)
    assert len(content) == 2, f"REFERENCE.md has no {start!r} heading"
    return _norm(content[1].split(end, 1)[0])


class TestAuthoritativeBusyFigureIsDocumented:
    """The docs name the decision figure, not just the hardware one.

    Parent AC4/AC7 of LP-0MVSPWLLP008AQXZ, delivered by LP-0MVUIMMX3005KYQ6:
    the log-derived ``local_busy`` (dispatch → finish slot union) is the figure
    gates and alerts must quote; the power-based ``gpu_busy`` answers hardware
    questions only and is never local model utilisation. The causes of the gap
    stay in one place — the pre-existing divergence explanation — and are
    linked rather than restated.
    """

    def test_reference_names_the_log_derived_figure_as_the_decision_figure(self):
        """AC1: the rule, the consumers and the ``--json`` key are named."""
        section = _reference_section(
            start=AUTHORITATIVE_BUSY_HEADING, end=DIVERGENCE_HEADING
        )
        assert "decisions use the log-derived `local_busy`" in section, (
            "REFERENCE.md must state which busy figure is authoritative for "
            "decisions, not only which is hardware-side"
        )
        assert "gates, alerts and threshold comparisons" in section, (
            "the guidance must name the consumers that quote the figure"
        )
        assert "dispatch → finish" in section, (
            "the guidance must tie the decision figure to its definition"
        )

    def test_reference_limits_the_power_figure_to_hardware_questions(self):
        """AC2: prefer ``gpu_busy`` only for hardware questions."""
        section = _reference_section(
            start=AUTHORITATIVE_BUSY_HEADING, end=DIVERGENCE_HEADING
        )
        assert "prefer `gpu_busy` for hardware questions only" in section
        assert "hardware activity, not local model utilisation" in section, (
            "the power figure must be explicitly labelled non-local"
        )
        assert "was the gpu computing at all" in section, (
            "the hardware questions the power figure does answer must be named"
        )

    def test_reference_keeps_one_divergence_explanation_and_links_it(self):
        """AC3: the causes live once, in the explanation the rule links to."""
        reference = _norm(REFERENCE_DOC.read_text())
        causes = "gpu work outside the proxy's slot accounting"
        assert reference.count(causes) == 1, (
            "the gap causes must have a single source in REFERENCE.md"
        )
        divergence = _reference_section(
            start=DIVERGENCE_HEADING, end="## Legacy reason normalisation"
        )
        assert causes in divergence, (
            "the divergence explanation must keep the gap-cause list"
        )
        section = _reference_section(
            start=AUTHORITATIVE_BUSY_HEADING, end=DIVERGENCE_HEADING
        )
        assert DIVERGENCE_ANCHOR in section, (
            "the decision rule must link the divergence explanation instead of "
            "repeating the causes"
        )
        assert causes not in section

    def test_skill_states_the_matching_rule_and_cross_links(self):
        """AC4: SKILL.md carries the same rule and links the reference."""
        skill = _norm(SKILL_DOC.read_text())
        assert "hardware activity, not local model utilisation" in skill
        assert "decisions use the log-derived `local_busy`" in skill, (
            "SKILL.md must say decisions use the log-derived figure"
        )
        assert f"reference.md{DECISION_ANCHOR}" in skill, (
            "SKILL.md must cross-link the REFERENCE.md decision guidance"
        )
        json_row = next(
            _norm(line)
            for line in SKILL_DOC.read_text().splitlines()
            if line.startswith("| `--json` |")
        )
        assert "decision figure" in json_row, (
            "the --json key table must mark local_busy as the decision figure"
        )
        assert "**not** local model utilisation" in json_row, (
            "the --json key table must flag gpu_busy as hardware-only"
        )

    def test_monitoring_readme_does_not_leave_power_as_local_utilisation(self):
        """AC5: the monitoring guide points at the decision figure."""
        content = _norm((MONITORING_DIR / "README.md").read_text())
        assert "never local model utilisation" in content, (
            "monitoring/README.md must not leave the power figure readable as "
            "local model utilisation"
        )
        assert "local_busy" in content
        assert "gates and alerts quote" in content
        assert DECISION_ANCHOR in content, (
            "monitoring/README.md must link the usage-analysis decision rule"
        )

    def test_documented_labels_and_keys_are_the_shipped_ones(self):
        """AC6: the labels/keys the docs quote are what the code emits.

        Renders the shipped ``## Local model utilization`` section from the
        reporting module itself, so a documented label can never drift from the
        table the operator sees.
        """
        sys_path_added = str(USAGE_ANALYSIS_SCRIPTS)
        if sys_path_added not in sys.path:
            sys.path.insert(0, sys_path_added)
        from types import SimpleNamespace

        import reporting

        busy = SimpleNamespace(
            busy_seconds=453.6,
            window_seconds=86400.0,
            fast_busy_seconds=300.0,
            cheap_busy_seconds=100.0,
            fast_window_seconds=43200.0,
            cheap_window_seconds=43200.0,
            decode_busy_seconds=400.0,
            decode_fast_busy_seconds=260.0,
            decode_cheap_busy_seconds=90.0,
            prefill_busy_seconds=53.6,
            idle_seconds=85946.4,
            prefill_extended_streams=0,
            streams=1,
            avg_stream_duration=12.0,
            total_compute_seconds=12.0,
            avg_concurrency=1.0,
            peak_concurrency=1,
            unfinished_streams=0,
            pre_window_unfinished=0,
        )
        summary = SimpleNamespace(
            busy=busy,
            gpu_busy=SimpleNamespace(busy_seconds=33955.2, total_seconds=86400.0),
        )
        lines: list[str] = []
        reporting._append_busy_section(lines.append, summary, None)
        rendered = "\n".join(lines)

        assert GPU_BUSY_ROW_LABEL in rendered
        assert BUSY_UPPER_ROW_LABEL in rendered
        assert "local_busy" in rendered and "not local model utilisation" in rendered

        for doc in (REFERENCE_DOC, SKILL_DOC):
            content = doc.read_text()
            assert GPU_BUSY_ROW_LABEL in content, (
                f"{doc.name} must quote the report's hardware row label verbatim"
            )
        reference = REFERENCE_DOC.read_text()
        assert BUSY_UPPER_ROW_LABEL in reference, (
            "REFERENCE.md must name the table row that carries the decision figure"
        )
        for key in ("local_busy", "gpu_busy"):
            assert key in reference and key in SKILL_DOC.read_text()

    def test_busy_figure_doc_anchors_resolve(self):
        """AC7: every in-repo anchor link in the busy-figure docs resolves."""
        unresolved: list[str] = []
        for doc in BUSY_FIGURE_DOCS:
            text = doc.read_text()
            for target, fragment in re.findall(
                r"\]\(([^)\s]*)#([^)\s]+)\)", text
            ):
                linked = doc if not target else (doc.parent / target)
                if linked.suffix != ".md" or not linked.exists():
                    continue
                if fragment not in _documented_anchors(linked.read_text()):
                    unresolved.append(f"{doc.name} -> {target or doc.name}#{fragment}")
        assert not unresolved, (
            f"anchor links that resolve to no heading: {unresolved}"
        )


# ---------------------------------------------------------------------------
# Tests: GPU telemetry, calibration and retention documentation
# (LP-0MVT1O7AU002CM7P; parent LP-0MVNWUEY8008XM6W AC7/AC8)
#
# The docs this class guards are operator-facing, so every assertion here is
# tied to something executable: the CLI's own argument parser, its own renderer
# and the real sysfs sensor. A doc that drifts from the shipped behaviour fails
# rather than describing an intention.
# ---------------------------------------------------------------------------

GPU_CLI = PROJECT_ROOT / "scripts" / "gpu_utilisation.py"

#: Slot-occupancy series published by the proxy (LP-0MVT1O0VG00905EL); the
#: README table is the one place an operator copies these names from.
SLOT_SERIES = (
    "llama_slots_total",
    "llama_slots_busy",
    "llama_slot_busy",
    "llama_slots_in_flight_tasks",
    "llama_slots_sample_timestamp_seconds",
)

#: Markers that put a `gpu_busy_percent` mention in the "not a utilisation
#: signal" context the AC requires; one must appear in every README section
#: that names the counter.
NOT_A_SIGNAL_MARKERS = (
    "not a utilisation",
    "not a utilization",
    "never a utilisation",
    "never a utilization",
    "saturat",
    "continuity only",
    "must not",
)


def _gpu_cli():
    """Import ``scripts/gpu_utilisation.py`` so docs are checked against it."""
    if str(GPU_CLI.parent) not in sys.path:
        sys.path.insert(0, str(GPU_CLI.parent))
    # Deliberate late import: the module path is only needed by these tests.
    import gpu_utilisation

    return gpu_utilisation


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) sections, one per heading.

    Fenced code blocks are skipped: a bash comment (``# 1. ...``) starts with
    ``#`` but is not a heading, and mistaking it for one truncates the section.
    """
    sections: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(FENCE):
            in_fence = not in_fence
        elif not in_fence and re.match(r"^#{1,6}\s+\S", line):
            sections.append((line, []))
            continue
        if sections:
            sections[-1][1].append(line)
        else:
            sections.append(("", [line]))
    return [(heading, "\n".join(body)) for heading, body in sections]


def _prose(text: str) -> str:
    """Normalise markdown prose, dropping emphasis so wording matches."""
    return _norm(text.replace("*", "").replace("`", ""))


def _documented_cli_commands(text: str) -> list[str]:
    """Extract `python3 scripts/gpu_utilisation.py ...` invocations from docs.

    Handles shell line continuations (a trailing backslash) and strips a shell
    pipeline suffix (`| jq ...`), so each returned string is exactly the
    argument list a reader would paste.
    """
    commands: list[str] = []
    pending: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("python3 scripts/gpu_utilisation.py"):
            pending = line
        elif pending is not None and line.startswith("--"):
            pending = f"{pending.rstrip(chr(92))} {line}"
        else:
            if pending is not None:
                commands.append(pending)
                pending = None
            continue
        if not pending.endswith("\\"):
            commands.append(pending)
            pending = None
    if pending is not None:
        commands.append(pending)
    return commands


def _number_shape(line: str) -> str:
    """Mask digits so two runs of the same layout compare equal."""
    return re.sub(r"\d+(?:\.\d+)?", "#", line.strip())


class TestGpuTelemetryDocumentation:
    """The README documents what shipped, verified against the shipped code."""

    def _readme(self) -> str:
        return (MONITORING_DIR / "README.md").read_text()

    def test_slot_occupancy_series_are_documented(self):
        """AC: every published slot series is named in the README table."""
        readme = self._readme()
        assert "| Series |" in readme, (
            "monitoring/README.md must document the slot series in a table"
        )
        tables = "\n".join(
            body for _heading, body in _markdown_sections(readme) if "| Series |" in body
        )
        missing = [s for s in SLOT_SERIES if s not in tables]
        assert not missing, (
            f"slot series missing from the README series table: {missing}; an "
            "operator cannot query a series the docs do not name"
        )
        # Each row must also say what the series means, not just name it.
        for series in SLOT_SERIES:
            row = [ln for ln in tables.splitlines() if f"`{series}" in ln]
            assert row, f"no table row for {series}"
            assert len(row[0].strip("|").split("|")) >= 3, (
                f"the {series} row must carry a type and a meaning column"
            )

    def test_documented_cli_commands_parse_with_the_shipped_parser(self):
        """AC: the documented example is copy-pasteable, not aspirational.

        Every `python3 scripts/gpu_utilisation.py ...` line in the README is fed
        to the CLI's real argparse parser: an invented or renamed flag fails
        here instead of in an operator's terminal.
        """
        import shlex

        gu = _gpu_cli()
        commands = _documented_cli_commands(self._readme())
        assert len(commands) >= 3, (
            "monitoring/README.md must show the CLI with real invocations "
            f"(window, absolute window, --json); found {commands}"
        )
        parser = gu.build_arg_parser()
        for command in commands:
            argv = shlex.split(command.split("|", 1)[0])
            assert argv[:2] == ["python3", "scripts/gpu_utilisation.py"], argv
            try:
                parsed = parser.parse_args(argv[2:])
            except SystemExit as exc:  # argparse exits on an unknown flag
                raise AssertionError(
                    f"documented CLI command does not parse: {command!r} ({exc})"
                ) from None
            # The example must actually ask for something.
            assert parsed.window or parsed.since, (
                f"documented CLI command selects no window: {command!r}"
            )

    def test_documented_example_output_matches_the_renderer(self):
        """AC: the output block is a real run, compared with the real renderer.

        The example block's provenance is the CLI's own ``render_human`` — the
        numbers are masked, so only the layout is asserted (docs must not claim
        a column the CLI does not print).
        """
        gu = _gpu_cli()
        documented = [
            body
            for lang, body in _fenced_blocks(MONITORING_DIR / "README.md")
            if "GPU utilisation" in body
        ]
        assert documented, (
            "monitoring/README.md must include a real CLI output block "
            "(a fenced block containing 'GPU utilisation')"
        )
        lines = documented[0].splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("GPU utilisation"))
        rendered = gu.GpuUtilisation(
            window=gu.Window(1783900000, 1783907200),
            step_seconds=gu.SCRAPE_INTERVAL_SECONDS,
            scrape_interval_seconds=gu.SCRAPE_INTERVAL_SECONDS,
            expected_samples=481,
            observed_samples=0,
            computing_samples=0,
            idle_samples=0,
            unknown_samples=481,
            power_only_computing_samples=63,
            power_mean_watts=55.0,
            power_max_watts=120.0,
            power_samples=481,
            peak_busy_slots=None,
            slot_samples=0,
            cross_check_samples=0,
            cross_check_mismatches=0,
        ).render_human().splitlines()
        got = [_number_shape(ln) for ln in lines[start:start + len(rendered)]]
        want = [_number_shape(ln) for ln in rendered]
        assert got == want, (
            "monitoring/README.md's CLI output block no longer matches the "
            f"columns the CLI prints.\n documented: {got}\n CLI:        {want}"
        )

    def test_cli_exit_codes_and_unknown_semantics_are_documented(self):
        """AC: an operator can tell "unknown" from "idle" without reading code."""
        gu = _gpu_cli()
        content = _norm(self._readme())
        for code in (gu.EXIT_OK, gu.EXIT_USAGE, gu.EXIT_PROMETHEUS_UNAVAILABLE, gu.EXIT_NO_DATA):
            assert f"`{code}`" in content, f"README must document exit code {code}"
        assert "no observed samples" in content, (
            "README must quote the all-unknown message so the exit-4 run is "
            "recognisable instead of looking like a broken host"
        )
        assert "never" in content and "unknown" in content, (
            "README must state that missing data is unknown, not idle"
        )

    def test_resolution_floor_is_stated_and_matches_the_deployment(self):
        """AC: no figure implies finer fidelity than the store holds."""
        gu = _gpu_cli()
        content = _norm(self._readme())
        assert "resolution floor is 15 s" in content, (
            "README must state the resolution floor (the 15 s scrape interval)"
        )
        assert "10 s" in content, (
            "README must state the 10 s cadences (sysfs sampler and proxy slot "
            "sample) that the 15 s scrape observes"
        )
        assert "resolution: step 15s (scrape floor 15s)" in content, (
            "README must show the CLI reporting the step it actually queried and "
            "the scrape floor"
        )

        # The documented floor is the CLI's own constant, not a coincidence.
        assert gu.SCRAPE_INTERVAL_SECONDS == 15, (
            "the CLI's scrape-interval constant changed; update the README's "
            "resolution statement in the same commit"
        )
        assert gu.resolve_step(None, gu.SCRAPE_INTERVAL_SECONDS) == 15, (
            "the CLI's default step must remain the scrape interval"
        )
        scrape_cfg = assert_valid_yaml(MONITORING_DIR / "prometheus.yml")
        configured = {
            job["job_name"]: job.get("scrape_interval", "15s")
            for job in scrape_cfg["scrape_configs"]
        }
        assert set(configured.values()) == {"15s"}, (
            f"monitoring/prometheus.yml scrape intervals drifted from the "
            f"documented 15 s floor: {configured}"
        )

    def test_retention_statement_names_the_tsdb_creation_date(self):
        """AC: 90 d is a ceiling, and history starts when the TSDB was created."""
        content = _norm(self._readme())
        assert "90 d" in content or "90 days" in content, (
            "README must state the 90-day retention"
        )
        assert "2026-10-25" in content, (
            "README must state when this host's history starts, so an operator "
            "does not read '90 d' as '90 days of data available'"
        )
        assert "tsdb" in content, (
            "README must say that history starts when the TSDB was created"
        )
        # The documented retention query must be a subquery: an instant vector
        # cannot see history, so min(timestamp(up)) reports *now*.
        assert "min_over_time(timestamp(up)[90d:1h])" in content, (
            "the documented 'oldest sample' query must be a subquery; "
            "min(timestamp(up)) returns the current scrape, not the oldest one"
        )
        flags_doc = _norm((DOCS_SYSTEMD_DIR / "prometheus.service").read_text())
        assert "retention" in flags_doc, (
            "the documented Prometheus unit must mention its retention flag"
        )

    def test_calibration_section_states_the_shared_threshold(self):
        """AC: the bands, the sensor, and both keys that carry the threshold."""
        gu = _gpu_cli()
        content = _norm(self._readme())
        for phrase in (
            "power1_average",
            "microwatts",
            "server.llama_gpu_wedge_power_low_threshold_watts",
            "gpu:computing:bool",
        ):
            assert phrase in content, f"calibration section is missing {phrase!r}"
        assert "idle" in content and "computing (local generation)" in content, (
            "the calibration section must name both measured bands"
        )

        # The documented threshold is the shipped threshold, in both places.
        threshold = _find_key(
            assert_valid_yaml(PROJECT_ROOT / "proxy" / "config.yaml"),
            "llama_gpu_wedge_power_low_threshold_watts",
        )
        assert threshold is not None
        assert f"{int(threshold)} w" in content, (
            f"README must quote the shipped threshold ({threshold} W)"
        )
        assert float(gu.THRESHOLD_WATTS) == float(threshold), (
            "the CLI, the rule and the proxy must share one threshold; README "
            f"documents {threshold} W but the CLI uses {gu.THRESHOLD_WATTS} W"
        )
        rules = assert_valid_yaml(MONITORING_DIR / "gpu_compute_rules.yaml")
        exprs = " ".join(
            rule.get("expr", "")
            for group in rules["groups"]
            for rule in group["rules"]
        )
        assert f">= bool {int(threshold)}" in exprs, (
            "gpu_compute_rules.yaml no longer carries the documented threshold"
        )

    def test_calibration_remeasure_recipe_uses_the_real_sensor(self):
        """AC: a clean host can reproduce the numbers from the documented recipe."""
        content = _norm(self._readme())
        assert "cat /sys/class/drm/card0/device/hwmon/hwmon*/power1_average" in content, (
            "the re-measure recipe must name the exact sysfs file"
        )
        assert "divide by" in content and "1000000" in content, (
            "the recipe must say the file is in microwatts, or every reading is "
            "off by six orders of magnitude"
        )
        glob = sorted(
            (Path("/sys/class/drm/card0/device/hwmon")).glob("hwmon*/power1_average")
        )
        if not glob:
            pytest.skip("no amdgpu hwmon power sensor on this host")
        raw = int(glob[0].read_text().strip())
        assert raw > 1_000_000, (
            f"{glob[0]} reads {raw}: the documented microwatt conversion assumes "
            "hwmon's microwatt units; re-check the recipe"
        )
        watts = raw / 1_000_000
        assert 0 < watts < 1000, f"implausible GPU package power from sysfs: {watts} W"

    def test_gpu_busy_percent_is_only_mentioned_as_a_non_signal(self):
        """AC: a grep assertion keeps the counter out of utilisation claims."""
        offenders: list[str] = []
        mentions = 0
        for heading, body in _markdown_sections(self._readme()):
            if "gpu_busy_percent" not in body:
                continue
            mentions += body.count("gpu_busy_percent")
            normalised = _prose(body)
            if not any(marker in normalised for marker in NOT_A_SIGNAL_MARKERS):
                offenders.append(heading or "<intro>")
        assert mentions, (
            "the README still has to explain why gpu_busy_percent is not used"
        )
        assert not offenders, (
            "gpu_busy_percent may only appear in a 'not a utilisation signal' "
            f"context; these sections name it without one: {offenders}"
        )

    def test_computing_share_is_distinguished_from_the_log_derived_busy_figure(self):
        """AC: neither busy figure may be presented as replacing the other."""
        content = _norm(self._readme())
        for phrase in (
            "local_busy.busy_pct",
            "gpu:computing_workload:bool",
            "12.6",
            "39.3",
            "none replaces another",
        ):
            assert phrase in content, (
                f"README must distinguish the hardware and log-derived figures; "
                f"missing {phrase!r}"
            )
        assert "proxy-inferred" in content, (
            "the usage report's figure must be named as proxy-inferred (log "
            "derived), not as a hardware measurement"
        )

    def test_gpu_metrics_unit_still_describes_the_shipped_cadence(self):
        """AC: docs/systemd/gpu-metrics.service stays accurate."""
        content = _norm((DOCS_SYSTEMD_DIR / "gpu-metrics.service").read_text())
        assert "sleep 10" in content, "the sampler cadence documented is not 10 s"
        assert "node_hwmon_power_average_watt" in content, (
            "the unit document must keep naming the power series it feeds"
        )
        for series in SLOT_SERIES:
            assert series in content, (
                f"gpu-metrics.service must keep pointing slot occupancy at the "
                f"proxy; missing {series}"
            )
        assert "not a utilisation" in content, (
            "the unit document must keep saying gpu_busy_percent is not a "
            "utilisation signal"
        )
