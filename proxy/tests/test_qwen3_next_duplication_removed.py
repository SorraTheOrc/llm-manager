"""
Tests verifying that the Qwen3-Next model duplication has been removed.

The Qwen3-Next section in models.ini was an alias that loaded the exact
same GGUF as Qwen3 (unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M), causing VRAM
waste when the router loaded both. This test suite verifies:

1. models.ini has no [Qwen3-Next] section
2. proxy/config.yaml qwen3-next and code entries point to llama_model: Qwen3
3. start-llama.sh has no qwen3-next case and Qwen3 has --jinja
"""

import configparser
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_INI_PATH = os.path.join(REPO_ROOT, "models.ini")
CONFIG_YAML_PATH = os.path.join(REPO_ROOT, "proxy", "config.yaml")
START_LLAMA_PATH = os.path.join(REPO_ROOT, "start-llama.sh")


# ---------------------------------------------------------------------------
# models.ini checks
# ---------------------------------------------------------------------------


def test_models_ini_has_no_qwen3_next_section():
    """The [Qwen3-Next] section MUST NOT exist in models.ini (duplicate GGUF)."""
    config = configparser.ConfigParser()
    parsed = config.read(MODELS_INI_PATH)
    assert parsed, f"Could not read {MODELS_INI_PATH}"
    assert "Qwen3-Next" not in config.sections(), (
        "[Qwen3-Next] should be removed (it duplicated the same GGUF as [Qwen3])"
    )


def test_models_ini_qwen3_section_exists():
    """The [Qwen3] section must still exist with the correct GGUF."""
    config = configparser.ConfigParser()
    parsed = config.read(MODELS_INI_PATH)
    assert parsed, f"Could not read {MODELS_INI_PATH}"
    assert "Qwen3" in config.sections(), "[Qwen3] section should exist"
    assert "hf-repo" in config["Qwen3"], "[Qwen3] should have hf-repo"
    assert "unsloth" in config["Qwen3"]["hf-repo"], (
        "[Qwen3] should reference the unsloth Qwen3 GGUF"
    )


def test_models_ini_qwen3_coder_next_section_exists():
    """The [Qwen3-Coder-Next] section should remain (real Qwen3-Coder-Next model)."""
    config = configparser.ConfigParser()
    parsed = config.read(MODELS_INI_PATH)
    assert parsed, f"Could not read {MODELS_INI_PATH}"
    assert "Qwen3-Coder-Next" in config.sections(), (
        "[Qwen3-Coder-Next] section should remain (points to actual Qwen3-Coder-Next model)"
    )


# ---------------------------------------------------------------------------
# proxy/config.yaml checks
# ---------------------------------------------------------------------------


def _load_config_yaml():
    """Load config.yaml and return the raw parsed dict."""
    import yaml
    with open(CONFIG_YAML_PATH) as f:
        return yaml.safe_load(f)


def test_config_code_llama_model_is_qwen3():
    """code entry in config.yaml must point to llama_model: Qwen3 for the local provider."""
    cfg = _load_config_yaml()
    code = cfg.get("models", {}).get("code")
    assert code is not None, "code model entry should exist"
    providers = code.get("providers", [])
    assert len(providers) > 0, "code should have providers"
    assert providers[0].get("llama_model") == "Qwen3", (
        f"code first provider should use llama_model: Qwen3, got {providers[0].get('llama_model')!r}"
    )


def test_config_code_keeps_local_fallback_chain():
    """code entry should still have its local -> remote fallback chain."""
    cfg = _load_config_yaml()
    code = cfg.get("models", {}).get("code")
    assert code is not None
    providers = code.get("providers", [])
    assert len(providers) >= 2, "code should have at least 2 providers (local + remote fallback)"
    # First provider is local, rest should be remote
    assert providers[0].get("type") == "local", "code first provider should be local"
    for p in providers[1:]:
        assert p.get("type") == "remote", (
            f"code fallback provider should be remote, got {p.get('type')!r}"
        )


# ---------------------------------------------------------------------------
# start-llama.sh checks
# ---------------------------------------------------------------------------


def test_start_llama_qwen3_has_jinja():
    """The Qwen3 case block in start-llama.sh should include --jinja."""
    with open(START_LLAMA_PATH) as f:
        content = f.read()

    # Find the qwen3 case block and check for --jinja
    assert "qwen3)" in content, (
        "start-llama.sh should have a 'qwen3)' case block"
    )
    assert "--jinja" in content, (
        "start-llama.sh should include --jinja in the Qwen3 case block "
        "(moved from the removed Qwen3-Next)"
    )


def test_start_llama_no_qwen3_next_case():
    """start-llama.sh should NOT have a combined qwen3|qwen3-next case."""
    with open(START_LLAMA_PATH) as f:
        content = f.read()

    assert "qwen3-next)" not in content, (
        "start-llama.sh should not reference qwen3-next in a case block"
    )
