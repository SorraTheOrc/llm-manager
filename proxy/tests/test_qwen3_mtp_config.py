"""
Tests for the Qwen3 MTP evaluation configuration (LP-0MSNI1B68001VE6C).

The MTP (Multi-Token Prediction) evaluation adds a COEXISTING model entry:

1. start-llama.sh: a `qwen3-mtp` case block pointing at the MTP-converted
   GGUF (unsloth/Qwen3.6-35B-A3B-MTP-GGUF) with `--spec-type draft-mtp` and
   `--spec-draft-n-max 2` in llama-server flags, plus single-slot mode.
2. models.ini: a [Qwen3-MTP] router preset for the same MTP GGUF.
3. proxy configs (config.yaml, config-fast.yaml, config-cheap.yaml): a
   `local-qwen3-mtp` model entry whose first provider is the local MTP model,
   wired into the same remote fallback chain as the production entries.

The production plan/author/code chains must remain UNCHANGED (first local
provider still llama_model: Qwen3) so the MTP evaluation never affects live
traffic until a rollout decision is made.
"""

import configparser
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_INI_PATH = os.path.join(REPO_ROOT, "models.ini")
START_LLAMA_PATH = os.path.join(REPO_ROOT, "start-llama.sh")
CONFIG_PATHS = [
    os.path.join(REPO_ROOT, "proxy", "config.yaml"),
    os.path.join(REPO_ROOT, "proxy", "config-fast.yaml"),
    os.path.join(REPO_ROOT, "proxy", "config-cheap.yaml"),
]

MTP_HF_REPO = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"


def _load_config(path):
    """Load a YAML config and return the raw parsed dict."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# start-llama.sh checks
# ---------------------------------------------------------------------------


def test_start_llama_has_qwen3_mtp_case():
    """start-llama.sh should have a 'qwen3-mtp)' case block."""
    with open(START_LLAMA_PATH) as f:
        content = f.read()
    assert "qwen3-mtp)" in content, (
        "start-llama.sh should have a 'qwen3-mtp)' case block"
    )


def test_start_llama_qwen3_mtp_uses_mtp_gguf():
    """The qwen3-mtp block must reference the MTP-converted GGUF."""
    with open(START_LLAMA_PATH) as f:
        content = f.read()
    # Find the qwen3-mtp block
    idx = content.index("qwen3-mtp)")
    block = content[idx:idx + 1200]
    assert MTP_HF_REPO in block, (
        f"qwen3-mtp block should reference {MTP_HF_REPO}"
    )


def test_start_llama_qwen3_mtp_enables_draft_mtp():
    """The qwen3-mtp block must set --spec-type draft-mtp and --spec-draft-n-max 2."""
    with open(START_LLAMA_PATH) as f:
        content = f.read()
    idx = content.index("qwen3-mtp)")
    block = content[idx:idx + 1200]
    assert "--spec-type draft-mtp" in block, (
        "qwen3-mtp block must include --spec-type draft-mtp"
    )
    assert "--spec-draft-n-max 2" in block, (
        "qwen3-mtp block must include --spec-draft-n-max 2"
    )


def test_start_llama_qwen3_mtp_documented_single_slot():
    """The qwen3-mtp block should document the -np 1 (single-slot) constraint."""
    with open(START_LLAMA_PATH) as f:
        content = f.read()
    idx = content.index("qwen3-mtp)")
    block = content[idx:idx + 1200]
    low = block.lower()
    assert ("single-slot" in low) or ("single slot" in low) or ("np 1" in low), (
        "qwen3-mtp block should document the MTP single-slot (-np 1) requirement"
    )


def test_start_llama_qwen3_mtp_listed_in_supported_models():
    """qwen3-mtp should appear in the 'Supported models' help text."""
    with open(START_LLAMA_PATH) as f:
        content = f.read()
    assert "qwen3-mtp" in content.lower()
    # The unrecognised-model message names the supported models.
    idx = content.index("Unrecognized model")
    msg = content[idx:idx + 400]
    assert "Qwen3-MTP" in msg, (
        "Unrecognized-model message should list Qwen3-MTP"
    )


# ---------------------------------------------------------------------------
# models.ini checks
# ---------------------------------------------------------------------------


def test_models_ini_has_qwen3_mtp_section():
    """models.ini should have a [Qwen3-MTP] router preset."""
    config = configparser.ConfigParser()
    parsed = config.read(MODELS_INI_PATH)
    assert parsed, f"Could not read {MODELS_INI_PATH}"
    assert "Qwen3-MTP" in config.sections(), (
        "[Qwen3-MTP] section should exist (router preset)"
    )


def test_models_ini_qwen3_mtp_hf_repo():
    """The [Qwen3-MTP] preset should reference the MTP-converted GGUF."""
    config = configparser.ConfigParser()
    config.read(MODELS_INI_PATH)
    assert "hf-repo" in config["Qwen3-MTP"], "[Qwen3-MTP] should have hf-repo"
    assert MTP_HF_REPO in config["Qwen3-MTP"]["hf-repo"], (
        f"[Qwen3-MTP] should reference {MTP_HF_REPO}"
    )


def test_models_ini_qwen3_mtp_same_ctx_as_qwen3():
    """[Qwen3-MTP] ctx-size should match [Qwen3] (comparable benchmark)."""
    config = configparser.ConfigParser()
    config.read(MODELS_INI_PATH)
    assert "ctx-size" in config["Qwen3-MTP"], "[Qwen3-MTP] should have ctx-size"
    assert "ctx-size" in config["Qwen3"], "[Qwen3] should have ctx-size"
    assert config["Qwen3-MTP"]["ctx-size"] == config["Qwen3"]["ctx-size"], (
        "Qwen3-MTP ctx-size should match Qwen3 for a fair A/B benchmark"
    )


def test_models_ini_reasoning_format_set_for_qwen3_presets():
    """[Qwen3] and [Qwen3-MTP] must set reasoning-format = deepseek.

    LP-0MSYLL3LY004CANG: the llama.cpp build deployed for the MTP
    experiment (10480/01818e495) ends turns with a bare finish_reason=stop
    after reasoning when reasoning-format is unset, so the model never emits
    tool_calls. The old build kept generating until the tool call or the
    length cap. A/B verification: with reasoning-format=deepseek the bare-stop
    failure drops from 2/6 to 0/6 and tool-call rate is >= the old build.
    """
    config = configparser.ConfigParser()
    config.read(MODELS_INI_PATH)
    for section in ("Qwen3", "Qwen3-MTP"):
        assert section in config.sections(), f"[{section}] section should exist"
        assert "reasoning-format" in config[section], (
            f"[{section}] should have reasoning-format"
        )
        assert config[section]["reasoning-format"] == "deepseek", (
            f"[{section}] reasoning-format should be deepseek"
        )


# ---------------------------------------------------------------------------
# proxy config checks
# ---------------------------------------------------------------------------


def test_config_production_chains_unaffected():
    """The plan/author/code chains must STILL route local-first to Qwen3 (coexist)."""
    for path in CONFIG_PATHS:
        cfg = _load_config(path)
        models = cfg.get("models", {})
        for chain in ("plan", "author", "code"):
            entry = models.get(chain)
            assert entry is not None, f"{os.path.basename(path)} missing {chain} chain"
            providers = entry.get("providers", [])
            assert providers[0].get("type") == "local", (
                f"{os.path.basename(path)} {chain} first provider should stay local"
            )
            assert providers[0].get("llama_model") == "Qwen3", (
                f"{os.path.basename(path)} {chain} first provider should stay llama_model: "
                f"Qwen3, got {providers[0].get('llama_model')!r}"
            )
