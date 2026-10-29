"""Tests for the structured prompt log (LP-0MVP79OBP002G6Z1).

Covers the module contract (one complete JSONL entry per handled request with
timestamp/session/prompt/requested+routed model), prompt extraction and
truncation, session-ID fallback, routed-model fill for local/remote/error
dispatches, disabled mode, bounded tail reads, rotation, the ``GET
/logs/prompt-log`` endpoint and the Prompt Log tab on ``/logs``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from fastapi import HTTPException
from proxy.router_helpers import _first_message_content, _get_request_preview
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from proxy import prompt_log

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_prompt_log(tmp_path):
    """Install a prompt log writing into *tmp_path* and clean up afterwards."""

    instances: list[prompt_log.PromptLog] = []

    def _make(configure=None, **section):
        cfg: dict = {"enabled": True}
        cfg.update(section)
        if configure is not None:
            cfg.update(configure)
        log = prompt_log.configure({"prompt_log": cfg}, tmp_path)
        assert log is not None
        instances.append(log)
        path = Path(cfg.get("path") or (tmp_path / prompt_log.FILENAME))
        return log, path

    yield _make

    for log in instances:
        log.close()
    prompt_log.reset()


def _chat_body(model: str = "qwen3.6", prompt: str = "Refactor the parser") -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "SECRET SYSTEM PROMPT"},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode()


def _read_entries(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# AC1: one entry per request, with the required fields
# ---------------------------------------------------------------------------


def test_chat_request_appends_one_complete_entry(make_prompt_log):
    log, path = make_prompt_log()
    response = Response(status_code=200, headers={"X-Resolved-Model": "local/qwen3.6-35b"})

    entry = prompt_log.record_exchange(
        headers={"x-session-id": "sess-123"},
        path="chat/completions",
        body=_chat_body(),
        response=response,
    )

    assert entry is not None
    assert log.flush()
    entries = _read_entries(path)
    assert len(entries) == 1

    stored = entries[0]
    assert stored["session_id"] == "sess-123"
    assert stored["endpoint"] == "v1/chat/completions"
    assert stored["requested_model"] == "qwen3.6"
    assert stored["routed_model"] == "local/qwen3.6-35b"
    assert stored["prompt"] == "Refactor the parser"
    assert stored["truncated"] is False
    assert stored["status_code"] == 200
    # ISO-8601 UTC timestamp
    assert stored["timestamp"].endswith("Z")
    parsed = datetime.fromisoformat(stored["timestamp"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    # System-prompt content is never written (same policy as the log preview)
    assert "SECRET SYSTEM PROMPT" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-session-id": "from-session-id"}, "from-session-id"),
        ({"session_id": "from-session-underscore"}, "from-session-underscore"),
        ({"x-client-request-id": "from-client-request"}, "from-client-request"),
        ({"x-session-affinity": "from-affinity"}, "from-affinity"),
        ({}, "unknown"),
    ],
)
def test_session_id_resolution_uses_shared_chain(make_prompt_log, headers, expected):
    log, path = make_prompt_log()
    prompt_log.record_exchange(
        headers=headers,
        path="chat/completions",
        body=_chat_body(),
        response=Response(status_code=200),
    )
    assert log.flush()
    assert _read_entries(path)[0]["session_id"] == expected


def test_completions_request_records_prompt_field(make_prompt_log):
    log, path = make_prompt_log()
    body = json.dumps({"model": "qwen3.6", "prompt": "Once upon a time"}).encode()

    prompt_log.record_exchange(
        headers={"x-session-id": "sess-c"},
        path="completions",
        body=body,
        response=Response(status_code=200),
    )

    assert log.flush()
    stored = _read_entries(path)[0]
    assert stored["endpoint"] == "v1/completions"
    assert stored["prompt"] == "Once upon a time"


def test_only_chat_and_completions_are_recorded(make_prompt_log):
    log, path = make_prompt_log()
    for ignored in ("embeddings", "models", ""):
        assert (
            prompt_log.record_exchange(
                headers={},
                path=ignored,
                body=_chat_body(),
                response=Response(status_code=200),
            )
            is None
        )
    assert log.flush()
    assert not path.exists()


# ---------------------------------------------------------------------------
# AC2: routed model reflects the provider that actually served the request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response_factory", "expected"),
    [
        (
            lambda: Response(status_code=200, headers={"X-Resolved-Model": "local/qwen3.6-35b"}),
            "local/qwen3.6-35b",
        ),
        (
            lambda: Response(
                status_code=200,
                headers={"X-Resolved-Model": "OpenCode Go/deepseek-v4.1-flash"},
            ),
            "OpenCode Go/deepseek-v4.1-flash",
        ),
        (
            # X-Provider is the fallback when no resolved model is present.
            lambda: Response(status_code=200, headers={"X-Provider": "deepseek"}),
            "deepseek",
        ),
        (
            # Rejected before routing: the failure reason is recorded.
            lambda: JSONResponse(
                status_code=503,
                content={"error": {"type": "startup_ramp", "code": "startup_ramp"}},
            ),
            "error:startup_ramp",
        ),
        (
            lambda: JSONResponse(
                status_code=503,
                content={"error": {"type": "model_loading", "code": "model_loading"}},
            ),
            "error:model_loading",
        ),
        (
            lambda: JSONResponse(status_code=500, content={"error": {"message": "boom"}}),
            "error:http_500",
        ),
        (lambda: Response(status_code=200), "unknown"),
    ],
)
def test_routed_model_fill(make_prompt_log, response_factory, expected):
    log, path = make_prompt_log()
    prompt_log.record_exchange(
        headers={},
        path="chat/completions",
        body=_chat_body(),
        response=response_factory(),
    )
    assert log.flush()
    assert _read_entries(path)[0]["routed_model"] == expected


def test_routed_model_marks_unknown_model_rejection(make_prompt_log):
    log, path = make_prompt_log()
    prompt_log.record_exchange(
        headers={},
        path="chat/completions",
        body=_chat_body(model="not-a-model"),
        exc=HTTPException(status_code=400, detail="Unknown model: not-a-model. No default remote configured."),
    )
    assert log.flush()
    stored = _read_entries(path)[0]
    assert stored["routed_model"] == "error:unknown_model"
    assert stored["requested_model"] == "not-a-model"


def test_routed_model_marks_unexpected_exception(make_prompt_log):
    log, path = make_prompt_log()
    prompt_log.record_exchange(
        headers={},
        path="chat/completions",
        body=_chat_body(),
        exc=RuntimeError("backend exploded"),
    )
    assert log.flush()
    assert _read_entries(path)[0]["routed_model"] == "error:exception"


# ---------------------------------------------------------------------------
# AC5: prompt extraction + configurable truncation
# ---------------------------------------------------------------------------


def test_prompt_truncation_sets_flag(make_prompt_log):
    log, path = make_prompt_log(max_prompt_chars=10)
    prompt_log.record_exchange(
        headers={},
        path="chat/completions",
        body=_chat_body(prompt="0123456789abcdefghij"),
        response=Response(status_code=200),
    )
    assert log.flush()
    stored = _read_entries(path)[0]
    assert stored["prompt"] == "0123456789"
    assert stored["truncated"] is True
    assert log.counters()["truncated"] == 1


def test_prompt_extraction_skips_system_and_ignores_missing_content():
    body = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user"},
            {"role": "user", "content": "first real content"},
            {"role": "user", "content": "second"},
        ]
    }
    assert _first_message_content(body) == "first real content"
    assert _first_message_content({"messages": [{"role": "system", "content": "x"}]}) == ""
    assert _first_message_content({}) == ""
    # The shared preview helper keeps its 80-char cap on the same extraction.
    long_body = {"messages": [{"role": "user", "content": "y" * 120}]}
    assert _get_request_preview(long_body) == "y" * 80 + "..."


# ---------------------------------------------------------------------------
# AC3: disabled mode and never-fail append
# ---------------------------------------------------------------------------


def test_disabled_mode_writes_nothing(make_prompt_log):
    log, path = make_prompt_log(enabled=False)

    assert (
        prompt_log.record_exchange(
            headers={},
            path="chat/completions",
            body=_chat_body(),
            response=Response(status_code=200),
        )
        is None
    )
    payload = prompt_log.recent_payload()
    assert payload["enabled"] is False
    assert payload["entries"] == []
    assert not path.exists()
    assert log.counters()["recorded"] == 0


def test_append_failure_is_counted_and_never_raised(make_prompt_log, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    log, _path = make_prompt_log(path=str(blocker / "prompt-log.jsonl"))

    entry = prompt_log.record_exchange(
        headers={},
        path="chat/completions",
        body=_chat_body(),
        response=Response(status_code=200),
    )

    assert entry is not None, "the request path must still see a recorded entry"
    assert log.flush()
    assert log.counters()["write_errors"] >= 1
    assert prompt_log.recent_payload()["entries"] == []


def test_serialisation_failure_is_counted(make_prompt_log):
    log, _path = make_prompt_log()
    assert log.record({"unserialisable": {1, 2, 3}}) is False
    assert log.counters()["write_errors"] == 1


# ---------------------------------------------------------------------------
# AC5: bounded recent window (UI reads a bounded tail, never the whole file)
# ---------------------------------------------------------------------------


def test_recent_entries_are_newest_first_and_bounded(make_prompt_log):
    log, _path = make_prompt_log(max_entries=10)
    for index in range(6):
        prompt_log.record_exchange(
            headers={},
            path="chat/completions",
            body=_chat_body(prompt=f"prompt-{index}"),
            response=Response(status_code=200),
        )
    assert log.flush()

    payload = prompt_log.recent_payload(limit=3)
    prompts = [entry["prompt"] for entry in payload["entries"]]
    assert prompts == ["prompt-5", "prompt-4", "prompt-3"]
    assert payload["max_entries"] == 10


def test_recent_entries_never_exceed_max_entries(make_prompt_log):
    log, _path = make_prompt_log(max_entries=2)
    for index in range(5):
        prompt_log.record_exchange(
            headers={},
            path="chat/completions",
            body=_chat_body(prompt=f"p-{index}"),
            response=Response(status_code=200),
        )
    assert log.flush()
    assert [entry["prompt"] for entry in log.recent_entries()] == ["p-4", "p-3"]


def test_missing_file_yields_empty_window(make_prompt_log):
    _log, path = make_prompt_log()
    assert not path.exists()
    payload = prompt_log.recent_payload()
    assert payload["enabled"] is True
    assert payload["entries"] == []


def test_tail_reader_skips_malformed_lines(tmp_path):
    path = tmp_path / prompt_log.FILENAME
    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "one"}),
                "not json at all",
                json.dumps({"prompt": "two"}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    entries = prompt_log.read_recent_entries(path, 10)
    assert [entry["prompt"] for entry in entries] == ["two", "one"]


def test_file_rotation_bounds_total_disk_growth(make_prompt_log):
    log, path = make_prompt_log(max_file_bytes=200, max_rotated_files=2)
    for index in range(8):
        prompt_log.record_exchange(
            headers={},
            path="chat/completions",
            body=_chat_body(prompt=f"rot-{index}" + "x" * 150),
            response=Response(status_code=200),
        )
    assert log.flush()

    # Rotation keeps the newest entry in the live file...
    live = _read_entries(path)
    assert live[-1]["prompt"].startswith("rot-7")
    # ...shifts history into .1/.2 and drops everything past max_rotated_files.
    backups = [Path(str(path) + f".{index}") for index in range(1, 5)]
    existing = [backup for backup in backups if backup.exists()]
    assert [backup.name for backup in existing] == [
        f"{path.name}.1",
        f"{path.name}.2",
    ]
    assert sum(len(_read_entries(backup)) for backup in existing) + len(live) == 3

    # The recent window spans the rotation boundary, still newest first.
    prompts = [entry["prompt"] for entry in log.recent_entries(limit=100)]
    assert prompts[0].startswith("rot-7")
    assert prompts[1].startswith("rot-6")
    assert prompts[-1].startswith("rot-5")


# ---------------------------------------------------------------------------
# Config resolution / validation
# ---------------------------------------------------------------------------


def test_resolve_config_defaults_and_path_override(tmp_path):
    resolved = prompt_log.resolve_config({}, tmp_path)
    assert resolved.enabled is True
    assert resolved.path == tmp_path / prompt_log.FILENAME
    assert resolved.max_prompt_chars == prompt_log.DEFAULT_MAX_PROMPT_CHARS
    assert resolved.max_entries == prompt_log.DEFAULT_MAX_ENTRIES

    overridden = prompt_log.resolve_config(
        {
            "prompt_log": {
                "enabled": False,
                "path": str(tmp_path / "custom.jsonl"),
                "max_prompt_chars": 50,
                "max_entries": 5,
                "max_file_bytes": 0,
            }
        },
        tmp_path,
    )
    assert overridden.enabled is False
    assert overridden.path == tmp_path / "custom.jsonl"
    assert (overridden.max_prompt_chars, overridden.max_entries, overridden.max_file_bytes) == (50, 5, 0)


def test_validate_prompt_log_config_accepts_the_shipped_section():
    with open(CONFIG_PATH) as handle:
        config = yaml.safe_load(handle)
    assert prompt_log.validate_prompt_log_config(config) == []
    section = config["prompt_log"]
    assert section["enabled"] is True
    assert isinstance(section["max_prompt_chars"], int)
    assert isinstance(section["max_entries"], int)
    assert isinstance(section["max_file_bytes"], int)
    assert isinstance(section["max_rotated_files"], int)


@pytest.mark.parametrize(
    "section",
    [
        "not-a-mapping",
        {"enabled": "yes"},
        {"path": 42},
        {"max_prompt_chars": 0},
        {"max_prompt_chars": "2000"},
        {"max_entries": 0},
        {"max_file_bytes": -1},
        {"max_rotated_files": 0},
    ],
)
def test_validate_prompt_log_config_reports_invalid_values(section):
    problems = prompt_log.validate_prompt_log_config({"prompt_log": section})
    assert problems, f"expected validation problems for {section!r}"


def test_invalid_prompt_log_section_fails_startup_validation():
    from proxy.utils import _validate_prompt_log_config

    _validate_prompt_log_config({"prompt_log": {"enabled": True}})
    with pytest.raises(ValueError, match="Prompt log config validation failed"):
        _validate_prompt_log_config({"prompt_log": {"max_entries": 0}})


# ---------------------------------------------------------------------------
# Instrumentation wiring (ui.proxy_openai_api)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_openai_api_records_exactly_one_entry(make_prompt_log, monkeypatch):
    log, path = make_prompt_log()
    from proxy import ui

    body = _chat_body()

    async def fake_do_proxy_openai_api(request, endpoint, raw_body, srv):
        return Response(status_code=200, headers={"X-Resolved-Model": "local/qwen3.6"})

    class _ImmediateCoalescer:
        async def coalesce_or_execute(self, endpoint, raw_body, factory):
            return await factory()

    monkeypatch.setattr(ui, "_do_proxy_openai_api", fake_do_proxy_openai_api)
    monkeypatch.setattr(
        "proxy.request_coalescer.get_coalescer", lambda: _ImmediateCoalescer()
    )

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"x-session-id", b"sess-hook")],
        },
        receive=AsyncMock(return_value={"type": "http.request", "body": body, "more_body": False}),
    )

    response = await ui.proxy_openai_api(request, "chat/completions")

    assert response.status_code == 200
    assert log.flush()
    entries = _read_entries(path)
    assert len(entries) == 1
    assert entries[0]["session_id"] == "sess-hook"
    assert entries[0]["routed_model"] == "local/qwen3.6"


# ---------------------------------------------------------------------------
# AC4: endpoint + Prompt Log tab on /logs
# ---------------------------------------------------------------------------


@pytest.fixture
def transport():
    from proxy.server import app

    return httpx.ASGITransport(app=app)


@pytest.mark.asyncio
async def test_prompt_log_endpoint_returns_newest_first(make_prompt_log, transport):
    log, _path = make_prompt_log()
    for index in range(3):
        prompt_log.record_exchange(
            headers={"x-session-id": f"sess-{index}"},
            path="chat/completions",
            body=_chat_body(prompt=f"endpoint-{index}"),
            response=Response(status_code=200, headers={"X-Resolved-Model": "local/qwen3.6"}),
        )
    assert log.flush()

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/logs/prompt-log", params={"limit": 2})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["enabled"] is True
    assert [entry["prompt"] for entry in payload["entries"]] == ["endpoint-2", "endpoint-1"]
    assert payload["file"] == prompt_log.FILENAME
    assert resp.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_prompt_log_endpoint_reports_disabled(make_prompt_log, transport):
    make_prompt_log(enabled=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/logs/prompt-log")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["enabled"] is False
    assert payload["entries"] == []


@pytest.mark.asyncio
async def test_logs_page_renders_the_prompt_log_tab(transport):
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/logs")

    assert resp.status_code == 200
    html = resp.text
    assert 'id="tabPromptLog"' in html
    assert 'id="tabPromptLogContent"' in html
    assert "/logs/prompt-log" in html
    # The existing tabs are unchanged.
    assert 'id="tabSlots"' in html
    assert 'id="tabAllLogs"' in html
