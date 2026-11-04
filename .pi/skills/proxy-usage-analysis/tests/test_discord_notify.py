"""Delivery contract for the Discord webhook (LP-0MW2BD6G8001E960).

The nightly report is only useful if it reaches the operator, so these tests
pin the four behaviours the work item asks for:

- a successful run posts ``report.md`` as an attachment (multipart, never a
  pasted body);
- a missing webhook is *noted in the report* rather than failing silently;
- a failed run announces itself on the same webhook;
- the webhook URL is a credential and never reaches the report, the
  ``--json`` summary or an error message.

No test performs network I/O: every send injects a fake ``poster``, and
``tests/conftest.py`` blocks the real one for the whole suite.
"""

from __future__ import annotations

import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_proxy_usage  # noqa: E402
import discord_notify  # noqa: E402

WEBHOOK = "https://discord.com/api/webhooks/123456789/secret-token-value"
WINDOW_START = datetime(2026, 11, 3, 15, 0, 0)
WINDOW_END = datetime(2026, 11, 4, 15, 0, 0)


def _report(tmp_path: Path, name: str = "report.md") -> Path:
    path = tmp_path / name
    path.write_text(
        "# Proxy Usage Analysis Report\n"
        "\n"
        "- Generated: 2026-11-04 15:35:01\n"
        "- Window: 2026-11-03 15:00:00 → 2026-11-04 15:00:00 (24.0h)\n"
        "- Lines parsed: 100 | lines skipped: 0\n"
        "\n"
        "## Performance metrics summary\n"
        "\n"
        "body\n",
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, webhook: str | None = WEBHOOK, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    lines = ["projectName: LLM Proxy", "prefix: LP"]
    if webhook is not None:
        lines += ["discord:", f"  webhook_url: {webhook}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class Recorder:
    """Fake poster capturing every (url, body, headers, timeout) tuple."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, bytes, dict, float]] = []
        self.error = error

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, body, headers, timeout))
        if self.error is not None:
            raise self.error

    @property
    def body(self) -> bytes:
        assert self.calls, "the poster was never called"
        return self.calls[-1][1]

    @property
    def headers(self) -> dict:
        return self.calls[-1][2]

    def payload(self) -> dict:
        """Extract the JSON ``payload_json`` part from the multipart body."""
        body = self.body.decode("utf-8", "replace")
        marker = 'name="payload_json"'
        start = body.index(marker)
        start = body.index("\r\n\r\n", start) + 4
        end = body.index("\r\n--", start)
        return json.loads(body[start:end])


# ---------------------------------------------------------------------------
# Webhook discovery
# ---------------------------------------------------------------------------


class TestWebhookResolution:
    def test_reads_webhook_from_the_worklog_config(self, tmp_path):
        config = _config(tmp_path)

        source = discord_notify.resolve_webhook(config_path=config)

        assert source.configured is True
        assert source.url == WEBHOOK
        assert source.config_path == config
        assert str(config) in source.searched

    def test_explicit_flag_wins_over_the_config(self, tmp_path):
        config = _config(tmp_path, webhook="https://discord.com/api/webhooks/9/other")

        source = discord_notify.resolve_webhook(
            "https://discord.com/api/webhooks/1/override", config_path=config
        )

        assert source.url == "https://discord.com/api/webhooks/1/override"
        assert source.searched == "--discord-webhook-url"

    def test_config_without_a_discord_section_is_not_configured(self, tmp_path):
        config = _config(tmp_path, webhook=None)

        source = discord_notify.resolve_webhook(config_path=config)

        assert source.configured is False
        assert source.url is None

    def test_missing_config_names_the_search_in_the_note(self, tmp_path, monkeypatch):
        monkeypatch.setattr(discord_notify, "find_worklog_config", lambda **_: None)

        source = discord_notify.resolve_webhook(start=tmp_path)

        assert source.configured is False
        assert "no `.worklog/config.yaml` found" in source.searched
        assert str(tmp_path) in source.searched

    def test_discovery_walks_up_from_the_working_directory(self, tmp_path):
        (tmp_path / ".worklog").mkdir()
        config = _config(tmp_path / ".worklog")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)

        found = discord_notify.find_worklog_config(start=nested)

        assert found == config


# ---------------------------------------------------------------------------
# Multipart payload
# ---------------------------------------------------------------------------


class TestMultipartPayload:
    def test_report_is_sent_as_a_file_attachment(self, tmp_path):
        report = _report(tmp_path)
        recorder = Recorder()

        name = discord_notify.send_report(
            WEBHOOK, report, "hello", attachment_name="proxy-usage-report-2026-11-03.md",
            poster=recorder,
        )

        assert name == "proxy-usage-report-2026-11-03.md"
        headers = recorder.headers
        assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
        body = recorder.body
        assert b'name="files[0]"; filename="proxy-usage-report-2026-11-03.md"' in body
        assert b"# Proxy Usage Analysis Report" in body  # the full report travels
        assert recorder.payload()["content"] == "hello"

    def test_every_request_carries_a_discord_user_agent(self, tmp_path):
        """Cloudflare rejects the default ``Python-urllib`` agent with HTTP 403
        error 1010, which reads like a dead webhook. The header is the fix."""
        recorder = Recorder()

        discord_notify.send_message(WEBHOOK, "hello", poster=recorder)

        assert recorder.headers["User-Agent"] == discord_notify.USER_AGENT
        assert not recorder.headers["User-Agent"].lower().startswith("python-urllib")

    def test_message_body_is_truncated_below_the_discord_limit(self):
        long_title = "x" * 5000

        content = discord_notify.build_message(
            {"sessions": 1, "total_requests": 1, "local_requests": 1, "remote_requests": 0},
            [],
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        )
        trimmed = discord_notify._truncate(long_title)
        assert len(content) <= discord_notify.MAX_CONTENT_CHARS
        assert len(trimmed) == discord_notify.MAX_CONTENT_CHARS
        assert trimmed.endswith("… (truncated)")

    def test_message_lists_the_top_findings_by_severity(self):
        class Rec:
            def __init__(self, severity, title):
                self.severity = severity
                self.title = title

        recs = [
            Rec("info", "informational"),
            Rec("high", "high one"),
            Rec("medium", "medium one"),
            Rec("critical", "critical one"),
        ]

        content = discord_notify.build_message(
            {"sessions": 2, "total_requests": 3, "local_requests": 1, "remote_requests": 2},
            recs,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            attachment_name="report.md",
            top_n=3,
        )

        assert "[CRITICAL] critical one" in content
        assert "[HIGH] high one" in content
        assert "[MEDIUM] medium one" in content
        assert "informational" not in content
        assert "report.md" in content


# ---------------------------------------------------------------------------
# Delivery outcomes
# ---------------------------------------------------------------------------


class TestDeliver:
    def test_success_posts_the_attachment_and_notes_it_in_the_report(self, tmp_path):
        report = _report(tmp_path)
        recorder = Recorder()

        result = discord_notify.deliver(
            report,
            "**report**",
            window_start=WINDOW_START,
            config_path=_config(tmp_path),
            poster=recorder,
        )

        assert result.configured is True
        assert result.sent is True
        assert result.attachment_name == "proxy-usage-report-2026-11-03.md"
        assert len(recorder.calls) == 1
        text = report.read_text()
        assert "- Discord delivery: report sent as an attachment" in text
        assert "proxy-usage-report-2026-11-03.md" in text

    def test_missing_webhook_is_noted_in_the_report_and_nothing_is_sent(self, tmp_path):
        report = _report(tmp_path)
        recorder = Recorder()

        result = discord_notify.deliver(
            report,
            "**report**",
            config_path=_config(tmp_path, webhook=None),
            poster=recorder,
        )

        assert result.configured is False
        assert result.sent is False
        assert "not configured" in result.note
        assert recorder.calls == []
        text = report.read_text()
        assert "- Discord delivery: not configured" in text
        assert "config.yaml" in text  # names the config it searched

    def test_failed_send_is_reported_not_raised(self, tmp_path):
        report = _report(tmp_path)
        recorder = Recorder(error=discord_notify.DiscordDeliveryError("HTTP 500 from the Discord webhook"))

        result = discord_notify.deliver(
            report, "**report**", config_path=_config(tmp_path), poster=recorder
        )

        assert result.configured is True
        assert result.sent is False
        assert result.note.startswith("failed — ")
        assert "HTTP 500" in (result.error or "")
        assert "- Discord delivery: failed —" in report.read_text()

    def test_disabled_delivery_sends_nothing_and_says_so(self, tmp_path):
        report = _report(tmp_path)
        recorder = Recorder()

        result = discord_notify.deliver(
            report, "**report**", enabled=False, config_path=_config(tmp_path), poster=recorder
        )

        assert result.sent is False
        assert result.note == "disabled (--no-discord)"
        assert recorder.calls == []
        assert "- Discord delivery: disabled (--no-discord)" in report.read_text()

    def test_the_webhook_credential_never_reaches_the_report(self, tmp_path):
        report = _report(tmp_path)

        discord_notify.deliver(
            report,
            "**report**",
            config_path=_config(tmp_path),
            poster=Recorder(error=RuntimeError("boom")),
        )

        text = report.read_text()
        assert "secret-token-value" not in text
        assert WEBHOOK not in text

    def test_the_note_is_replaced_not_duplicated(self, tmp_path):
        report = _report(tmp_path)

        discord_notify.set_delivery_note(report, "not configured")
        discord_notify.set_delivery_note(report, "report sent as an attachment")

        text = report.read_text()
        assert text.count("- Discord delivery:") == 1
        assert "report sent as an attachment" in text
        assert "not configured" not in text

    def test_the_note_lands_in_the_header_before_the_first_section(self, tmp_path):
        report = _report(tmp_path)

        discord_notify.set_delivery_note(report, "report sent as an attachment")

        lines = report.read_text().splitlines()
        note = next(i for i, line in enumerate(lines) if line.startswith("- Discord delivery:"))
        first_section = next(i for i, line in enumerate(lines) if line.startswith("## "))
        assert note < first_section


# ---------------------------------------------------------------------------
# Failure notification
# ---------------------------------------------------------------------------


class TestFailureNotification:
    def test_failure_message_names_host_error_and_log(self, tmp_path):
        recorder = Recorder()

        result = discord_notify.notify_failure(
            "RuntimeError: boom",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            config_path=_config(tmp_path),
            poster=recorder,
        )

        assert result.sent is True
        content = recorder.payload()["content"]
        assert "proxy-usage-analysis failed" in content
        assert socket.gethostname() in content
        assert "RuntimeError: boom" in content
        assert "cron.log" in content
        assert b'name="files[0]"' not in recorder.body  # a notice, not a report

    def test_failure_notification_never_raises(self, tmp_path):
        result = discord_notify.notify_failure(
            "boom",
            config_path=_config(tmp_path),
            poster=Recorder(error=RuntimeError("webhook down")),
        )

        assert result.sent is False
        assert "webhook down" in (result.error or "")

    def test_failure_notification_is_skipped_when_not_configured(self, tmp_path):
        recorder = Recorder()

        result = discord_notify.notify_failure(
            "boom", config_path=_config(tmp_path, webhook=None), poster=recorder
        )

        assert result.configured is False
        assert recorder.calls == []


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestCliWiring:
    """The unchanged cron invocation must deliver without any new flag."""

    def _args(self, tmp_path: Path, *extra: str) -> list[str]:
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "proxy.log").write_text(
            "2026-11-03 16:00:00,000 - INFO - Stream started: provider=local model=Qwen3 "
            "session=s1 request=[]\n"
            "2026-11-03 16:00:05,000 - INFO - Stream finished: reason=stop "
            "tokens=100/10/110 session=s1 provider=local model=Qwen3 request=[]\n",
            encoding="utf-8",
        )
        return [
            "--log-dir",
            str(log_dir),
            "--start",
            "2026-11-03 15:00:00",
            "--end",
            "2026-11-04 15:00:00",
            "--output-dir",
            str(tmp_path / "out"),
            "--discord-config",
            str(_config(tmp_path)),
            "--json",
            *extra,
        ]

    def test_a_default_run_delivers_the_report(self, tmp_path, monkeypatch, capsys):
        recorder = Recorder()
        monkeypatch.setattr(discord_notify, "_post_multipart", recorder)

        assert analyze_proxy_usage.main(self._args(tmp_path)) == 0

        summary = json.loads(capsys.readouterr().out)
        assert summary["discord_delivery"]["sent"] is True
        assert b"# Proxy Usage Analysis Report" in recorder.body
        report = (tmp_path / "out" / "report.md").read_text()
        assert "- Discord delivery: report sent as an attachment" in report

    def test_no_discord_flag_keeps_the_report_local(self, tmp_path, monkeypatch, capsys):
        recorder = Recorder()
        monkeypatch.setattr(discord_notify, "_post_multipart", recorder)

        assert analyze_proxy_usage.main(self._args(tmp_path, "--no-discord")) == 0

        assert recorder.calls == []
        summary = json.loads(capsys.readouterr().out)
        assert summary["discord_delivery"]["sent"] is False
        assert summary["discord_delivery"]["note"] == "disabled (--no-discord)"

    def test_soft_failure_is_announced_and_the_exit_code_preserved(
        self, tmp_path, monkeypatch
    ):
        recorder = Recorder()
        monkeypatch.setattr(discord_notify, "_post_multipart", recorder)

        code = analyze_proxy_usage.main(
            [
                "--start",
                "2026-11-04 15:00:00",
                "--end",
                "2026-11-03 15:00:00",
                "--output-dir",
                str(tmp_path / "out"),
                "--discord-config",
                str(_config(tmp_path)),
            ]
        )

        assert code == 2
        assert len(recorder.calls) == 1
        content = recorder.payload()["content"]
        assert "proxy-usage-analysis failed" in content
        assert "window start must be before window end" in content

    def test_unexpected_exception_is_announced_then_re_raised(self, tmp_path, monkeypatch):
        recorder = Recorder()
        monkeypatch.setattr(discord_notify, "_post_multipart", recorder)

        def _boom(**_kwargs):
            raise RuntimeError("analysis exploded")

        monkeypatch.setattr(analyze_proxy_usage.reporting, "run_analysis", _boom)

        with pytest.raises(RuntimeError, match="analysis exploded"):
            analyze_proxy_usage.main(self._args(tmp_path))

        assert len(recorder.calls) == 1
        assert "analysis exploded" in recorder.payload()["content"]

    def test_failure_notification_survives_a_dead_webhook(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            discord_notify,
            "_post_multipart",
            Recorder(error=discord_notify.DiscordDeliveryError("HTTP 500")),
        )

        code = analyze_proxy_usage.main(
            [
                "--start",
                "2026-11-04 15:00:00",
                "--end",
                "2026-11-03 15:00:00",
                "--output-dir",
                str(tmp_path / "out"),
                "--discord-config",
                str(_config(tmp_path)),
            ]
        )

        assert code == 2  # the real failure still decides the exit code
