"""
Prompt resolution tests.

Unit tests for the prompt resolver (proxy/prompt_resolver.py) verifying
local override vs repo default precedence, alias exact vs wildcard
precedence, size and encoding handling, and config validation for missing
mode.
"""

import logging
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompt_file(path: Path, content: str = "system prompt content") -> Path:
    """Write a UTF-8 prompt file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_binary_file(path: Path, size: int = 100) -> Path:
    """Write a binary (non-UTF-8) file of given size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write raw bytes that are not valid UTF-8
    path.write_bytes(b"\xff\xfe\xfd" + b"\x00" * (size - 3))
    return path


def _make_oversized_file(path: Path, target_size: int) -> Path:
    """Write a UTF-8 file larger than target_size bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ("x" * 1024 + "\n") * ((target_size // 1025) + 1)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Test: Resolver function exists and can be imported
# ---------------------------------------------------------------------------

def test_resolver_module_importable():
    """The prompt_resolver module should be importable."""
    import proxy.prompt_resolver as pr
    assert hasattr(pr, "resolve_system_prompt")


# ---------------------------------------------------------------------------
# Test: No prompt configured returns None
# ---------------------------------------------------------------------------

def test_no_system_prompt_config_returns_none(monkeypatch):
    """When a model entry has no system_prompt, resolver returns None."""
    from proxy.prompt_resolver import resolve_system_prompt

    model_cfg = {"type": "local", "aliases": ["test-model"]}
    result = resolve_system_prompt("test-model", model_cfg)
    assert result is None


# ---------------------------------------------------------------------------
# Test: Repo default prompt file is loaded
# ---------------------------------------------------------------------------

def test_repo_default_prompt_loaded(monkeypatch, tmp_path):
    """When no local override exists, resolve from repo default path."""
    from proxy.prompt_resolver import resolve_system_prompt

    repo_prompts = tmp_path / "proxy" / "prompts"
    prompt_file = _make_prompt_file(repo_prompts / "assistant.txt")

    model_cfg = {
        "type": "local",
        "aliases": ["assistant"],
        "system_prompt": {
            "mode": "prepend",
            "file": str(prompt_file),
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")

    result = resolve_system_prompt("assistant", model_cfg)
    assert result is not None
    assert result["content"] == "system prompt content"
    assert result["mode"] == "prepend"
    assert result["source"] == str(prompt_file)


# ---------------------------------------------------------------------------
# Test: Repo default relative to repo root
# ---------------------------------------------------------------------------

def test_repo_default_relative_path(monkeypatch, tmp_path):
    """When file is relative, it's resolved against the repo root."""
    from proxy.prompt_resolver import resolve_system_prompt

    repo_root = tmp_path
    repo_prompts = repo_root / "proxy" / "prompts"
    _prompt_file = _make_prompt_file(repo_prompts / "code.txt", "You write code.")

    model_cfg = {
        "type": "local",
        "aliases": ["code"],
        "system_prompt": {
            "mode": "override",
            "file": "proxy/prompts/code.txt",
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", repo_root)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", repo_root / ".sorraAgents" / "prompts")

    result = resolve_system_prompt("code", model_cfg)
    assert result is not None
    assert result["content"] == "You write code."
    assert result["mode"] == "override"


# ---------------------------------------------------------------------------
# Test: Local override wins over repo default
# ---------------------------------------------------------------------------

def test_local_override_wins(monkeypatch, tmp_path):
    """When a local override file exists, it takes precedence."""
    from proxy.prompt_resolver import resolve_system_prompt

    repo_prompts = tmp_path / "proxy" / "prompts"
    _make_prompt_file(repo_prompts / "assistant.txt", "repo default")

    override_dir = tmp_path / ".sorraAgents" / "prompts"
    _make_prompt_file(override_dir / "assistant.txt", "local override")

    model_cfg = {
        "type": "local",
        "aliases": ["assistant"],
        "system_prompt": {
            "mode": "prepend",
            "file": "proxy/prompts/assistant.txt",
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", override_dir)

    result = resolve_system_prompt("assistant", model_cfg)
    assert result is not None
    assert result["content"] == "local override"
    assert "override" in result["source"]


# ---------------------------------------------------------------------------
# Test: Local override for alias (by lookup name)
# ---------------------------------------------------------------------------

def test_local_override_by_alias(monkeypatch, tmp_path):
    """Override files are looked up by the alias or model name."""
    from proxy.prompt_resolver import resolve_system_prompt

    override_dir = tmp_path / ".sorraAgents" / "prompts"
    _make_prompt_file(override_dir / "gemma4.txt", "gemma4 override")

    model_cfg = {
        "type": "local",
        "aliases": ["gemma4", "Gemma4"],
        "system_prompt": {
            "mode": "override",
            "file": "proxy/prompts/default.txt",
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", override_dir)

    # Resolve using the model name (not the alias) — should still match
    result = resolve_system_prompt("gemma4", model_cfg)
    assert result is not None
    assert result["content"] == "gemma4 override"


# ---------------------------------------------------------------------------
# Test: Oversized file returns None and logs warning
# ---------------------------------------------------------------------------

def test_oversized_file_ignored(monkeypatch, tmp_path, caplog):
    """Files larger than MAX_PROMPT_SIZE (64KB) should be ignored."""
    from proxy.prompt_resolver import resolve_system_prompt, MAX_PROMPT_SIZE

    repo_prompts = tmp_path / "proxy" / "prompts"
    oversized = _make_oversized_file(repo_prompts / "large.txt", MAX_PROMPT_SIZE + 1)

    model_cfg = {
        "type": "local",
        "aliases": ["large-model"],
        "system_prompt": {
            "mode": "prepend",
            "file": str(oversized),
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")

    caplog.set_level(logging.WARNING)

    result = resolve_system_prompt("large-model", model_cfg)
    assert result is None

    # Check warning was logged
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("size" in msg.lower() or "large" in msg.lower() or "64KB" in msg for msg in warning_messages)


# ---------------------------------------------------------------------------
# Test: Non-UTF-8 file is safely skipped
# ---------------------------------------------------------------------------

def test_non_utf8_file_skipped(monkeypatch, tmp_path, caplog):
    """Non-UTF-8 files should be ignored and logged."""
    from proxy.prompt_resolver import resolve_system_prompt

    repo_prompts = tmp_path / "proxy" / "prompts"
    binary_file = _make_binary_file(repo_prompts / "binary.txt")

    model_cfg = {
        "type": "local",
        "aliases": ["binary-model"],
        "system_prompt": {
            "mode": "prepend",
            "file": str(binary_file),
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")

    caplog.set_level(logging.WARNING)

    result = resolve_system_prompt("binary-model", model_cfg)
    assert result is None

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("utf" in msg.lower() or "encoding" in msg.lower() or "skip" in msg.lower() for msg in warning_messages)


# ---------------------------------------------------------------------------
# Test: Missing system_prompt.mode raises validation error
# ---------------------------------------------------------------------------

def test_missing_mode_validation_error():
    """When a model includes system_prompt but omits mode, validation fails."""
    from proxy.prompt_resolver import validate_prompt_config

    invalid_cfg = {
        "type": "local",
        "system_prompt": {
            "file": "proxy/prompts/test.txt",
            # missing "mode"
        },
    }

    with pytest.raises(ValueError, match="mode"):
        validate_prompt_config(invalid_cfg)


def test_invalid_mode_validation_error():
    """When mode is not 'override' or 'prepend', validation fails."""
    from proxy.prompt_resolver import validate_prompt_config

    invalid_cfg = {
        "type": "local",
        "system_prompt": {
            "file": "proxy/prompts/test.txt",
            "mode": "replace",
        },
    }

    with pytest.raises(ValueError, match="mode"):
        validate_prompt_config(invalid_cfg)


def test_valid_mode_passes_validation():
    """Both 'override' and 'prepend' modes pass validation."""
    from proxy.prompt_resolver import validate_prompt_config

    cfg_override = {"type": "local", "system_prompt": {"file": "test.txt", "mode": "override"}}
    cfg_prepend = {"type": "local", "system_prompt": {"file": "test.txt", "mode": "prepend"}}

    # Should not raise
    validate_prompt_config(cfg_override)
    validate_prompt_config(cfg_prepend)


# ---------------------------------------------------------------------------
# Test: No system_prompt block means no validation error
# ---------------------------------------------------------------------------

def test_no_system_prompt_block_is_valid():
    """A model entry without system_prompt should pass validation."""
    from proxy.prompt_resolver import validate_prompt_config

    cfg = {"type": "local", "aliases": ["test"]}
    # Should not raise
    validate_prompt_config(cfg)


# ---------------------------------------------------------------------------
# Test: Exact alias takes precedence over wildcard
# ---------------------------------------------------------------------------

def test_exact_alias_precedence_over_wildcard(monkeypatch, tmp_path):
    """When both exact and wildcard aliases match, exact should win."""
    from proxy.prompt_resolver import resolve_system_prompt

    repo_prompts = tmp_path / "proxy" / "prompts"
    _make_prompt_file(repo_prompts / "exact.txt", "exact match")
    _make_prompt_file(repo_prompts / "wildcard.txt", "wildcard match")

    # Model config for the exact alias - match by 'gpt-4'
    model_cfg = {
        "type": "remote",
        "aliases": ["gpt-4", "gpt-*", "gpt4"],
        "system_prompt": {
            "mode": "prepend",
            "file": str(repo_prompts / "exact.txt"),
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")

    result = resolve_system_prompt("gpt-4", model_cfg)
    assert result is not None
    assert result["content"] == "exact match"


# ---------------------------------------------------------------------------
# Test: Resolver returns None when file does not exist
# ---------------------------------------------------------------------------

def test_missing_prompt_file_returns_none(monkeypatch, tmp_path, caplog):
    """When the configured prompt file does not exist, resolver returns None."""
    from proxy.prompt_resolver import resolve_system_prompt

    model_cfg = {
        "type": "local",
        "aliases": ["missing"],
        "system_prompt": {
            "mode": "prepend",
            "file": "proxy/prompts/nonexistent.txt",
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")

    caplog.set_level(logging.WARNING)

    result = resolve_system_prompt("missing", model_cfg)
    assert result is None

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("not found" in msg.lower() or "missing" in msg.lower() or "exist" in msg.lower() for msg in warning_messages)


# ---------------------------------------------------------------------------
# Test: Override lookup uses multiple alias names
# ---------------------------------------------------------------------------

def test_override_lookup_tries_all_aliases(monkeypatch, tmp_path):
    """Override lookup should try all aliases (model name + each alias) and return first match."""
    from proxy.prompt_resolver import resolve_system_prompt

    override_dir = tmp_path / ".sorraAgents" / "prompts"
    _make_prompt_file(override_dir / "code-alias.txt", "code alias override")

    model_cfg = {
        "type": "local",
        "aliases": ["code", "code-alias", "coder"],
        "system_prompt": {
            "mode": "override",
            "file": "proxy/prompts/default.txt",
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", override_dir)

    result = resolve_system_prompt("code-alias", model_cfg)
    assert result is not None
    assert result["content"] == "code alias override"


# ---------------------------------------------------------------------------
# Test: Result dict has expected keys
# ---------------------------------------------------------------------------

def test_result_dict_has_expected_keys(monkeypatch, tmp_path):
    """The resolver should return a dict with content, mode, source keys."""
    from proxy.prompt_resolver import resolve_system_prompt

    repo_prompts = tmp_path / "proxy" / "prompts"
    prompt_file = _make_prompt_file(repo_prompts / "test.txt", "hello world")

    model_cfg = {
        "type": "local",
        "aliases": ["test-model"],
        "system_prompt": {
            "mode": "prepend",
            "file": str(prompt_file),
        },
    }

    monkeypatch.setattr("proxy.prompt_resolver.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proxy.prompt_resolver.OVERRIDE_DIR", tmp_path / ".sorraAgents" / "prompts")

    result = resolve_system_prompt("test-model", model_cfg)
    assert isinstance(result, dict)
    assert "content" in result
    assert "mode" in result
    assert "source" in result
    assert result["content"] == "hello world"
    assert result["mode"] == "prepend"
    assert result["source"] == str(prompt_file)
