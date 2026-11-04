"""Deliver the proxy usage report to the LLM-projects Discord webhook.

The nightly report is written to disk by :mod:`reporting`; this module is the
delivery layer that pushes it to the operator instead of leaving it parked in
``~/proxy-usage-reports/`` (LP-0MW2BD6G8001E960):

- **Report delivery.** A successful run posts ``report.md`` to the webhook as a
  multipart file attachment, with a compact message body (window, headline
  figures and the top recommendations) so the channel is readable without
  opening the attachment.
- **Missing webhook.** When no webhook is configured the report itself says so
  — a ``- Discord delivery:`` bullet is patched into the header naming the
  config path that was searched — and nothing is sent. The operator therefore
  learns about the gap from the report, not from an absence of messages.
- **Failed runs.** A run that fails (the 05:00 cron is unattended) posts a
  short failure message to the same webhook, so silence on the channel means
  "ran and was fine", never "crashed at 05:00".

The webhook URL is a credential: it is **never** written to the report, the
``--json`` summary, the logs or an error message. Only the *source* of the
credential (the config path, or the ``--discord-webhook-url`` flag) is
reported.

Two design constraints shape the API:

- **The unchanged 05:00 cron line must keep working** (``python3
  .pi/skills/proxy-usage-analysis/scripts/analyze_proxy_usage.py >> cron.log``),
  so delivery is on by default and needs no new flag.
- **Every network call is injectable** through the ``poster`` argument, so the
  test suite can assert the exact payload without reaching Discord. The skill's
  ``tests/conftest.py`` additionally disables delivery for the whole suite.

Nothing here is fatal: delivery is fail-soft, and a dead webhook never masks
the analysis error that triggered the notification.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config_loader

#: Worklog config (relative to the project root) that carries
#: ``discord.webhook_url``.
WORKLOG_CONFIG_RELATIVE = Path(".worklog") / "config.yaml"

#: Prefix of the report header bullet this module owns. The bullet is replaced
#: in place, never duplicated, so a re-rendered or re-delivered report carries
#: exactly one delivery line.
NOTE_PREFIX = "- Discord delivery:"

#: Discord rejects a message body longer than this (bytes are not the limit;
#: characters are).
MAX_CONTENT_CHARS = 2000

#: Seconds before a stalled webhook POST is abandoned.
DEFAULT_TIMEOUT_SECONDS = 30.0

#: Where the unattended daily run logs its output; named in failure messages so
#: the operator knows where the traceback is.
DEFAULT_LOG_HINT = "~/proxy-usage-reports/cron.log"

#: Discord sits behind Cloudflare, which rejects the default
#: ``Python-urllib/x.y`` agent with ``HTTP 403 error code: 1010`` (browser
#: signature ban) — the failure looks like a dead webhook but is only a missing
#: header. Discord's own docs recommend a ``DiscordBot (<url>, <version>)``
#: agent, so every request sends one.
USER_AGENT = "DiscordBot (https://github.com/rgardler/llm-proxy, 1.0)"

#: An injected HTTP poster: ``poster(url, body, headers, timeout) -> None``,
#: raising on any non-2xx response or transport error.
Poster = Callable[[str, bytes, dict, float], None]


class DiscordDeliveryError(RuntimeError):
    """A delivery attempt failed (transport error or a non-2xx response)."""


@dataclass
class WebhookSource:
    """Where the webhook credential came from (never the credential itself)."""

    url: str | None
    config_path: Path | None
    searched: str

    @property
    def configured(self) -> bool:
        return bool(self.url)


@dataclass
class DeliveryResult:
    """Outcome of one delivery attempt, safe to serialise into ``--json``."""

    configured: bool = False
    sent: bool = False
    note: str = ""
    error: str | None = None
    report_path: str | None = None
    attachment_name: str | None = None
    webhook_source: str | None = None

    def to_json(self) -> dict:
        return {
            "configured": self.configured,
            "sent": self.sent,
            "note": self.note,
            "error": self.error,
            "report_path": self.report_path,
            "attachment_name": self.attachment_name,
            "webhook_source": self.webhook_source,
        }


# ---------------------------------------------------------------------------
# Webhook discovery
# ---------------------------------------------------------------------------


def find_worklog_config(
    explicit: str | Path | None = None, start: str | Path | None = None
) -> Path | None:
    """Locate ``.worklog/config.yaml``.

    ``explicit`` (``--discord-config``) wins outright, existing or not. Without
    it the search walks upward from ``start`` (the working directory by
    default), then falls back to the repository root derived from this file so
    a run from an unrelated directory still finds the project config.
    """
    if explicit:
        return Path(explicit).expanduser()
    candidates: list[Path] = []
    base = Path(start or os.getcwd()).resolve()
    for directory in (base, *base.parents):
        candidates.append(directory / WORKLOG_CONFIG_RELATIVE)
    # <repo>/.pi/skills/proxy-usage-analysis/scripts/discord_notify.py
    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root / WORKLOG_CONFIG_RELATIVE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _webhook_from_config(config_path: Path) -> str | None:
    mapping = config_loader.load_yaml_mapping(config_path)
    discord = mapping.get("discord") if isinstance(mapping, dict) else None
    url = discord.get("webhook_url") if isinstance(discord, dict) else None
    if not isinstance(url, str):
        return None
    return url.strip() or None


def resolve_webhook(
    webhook_url: str | None = None,
    config_path: str | Path | None = None,
    start: str | Path | None = None,
) -> WebhookSource:
    """Resolve the webhook URL and record where it came from.

    ``--discord-webhook-url`` wins, then ``discord.webhook_url`` in the worklog
    config. ``searched`` describes the search for the *report* note, so it must
    stay free of the credential.
    """
    if webhook_url and webhook_url.strip():
        return WebhookSource(
            url=webhook_url.strip(),
            config_path=None,
            searched="--discord-webhook-url",
        )
    config = find_worklog_config(explicit=config_path, start=start)
    if config is None:
        base = Path(start or os.getcwd()).resolve()
        return WebhookSource(
            url=None,
            config_path=None,
            searched=(
                f"no `{WORKLOG_CONFIG_RELATIVE.as_posix()}` found from "
                f"{base} upward"
            ),
        )
    return WebhookSource(
        url=_webhook_from_config(config),
        config_path=config,
        searched=f"`{config}`",
    )


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def _truncate(content: str, limit: int = MAX_CONTENT_CHARS) -> str:
    """Trim *content* below Discord's limit, marking the cut."""
    if len(content) <= limit:
        return content
    marker = "… (truncated)"
    return content[: limit - len(marker)].rstrip() + marker


def _fmt_window(window_start: datetime | None, window_end: datetime | None) -> str:
    if window_start is None or window_end is None:
        return "window unknown"
    hours = (window_end - window_start).total_seconds() / 3600.0
    return (
        f"{window_start.strftime('%Y-%m-%d %H:%M')} → "
        f"{window_end.strftime('%Y-%m-%d %H:%M')} ({hours:.1f}h)"
    )


def build_message(
    data: dict,
    recommendations: list | None = None,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    attachment_name: str | None = None,
    top_n: int = 3,
) -> str:
    """Compose the Discord message body for a successful run.

    *data* is the ``--json`` summary, *recommendations* the generated
    ``Recommendation`` objects. The body stays short: the full report travels
    as the attachment, so the message only has to make the channel readable.
    """
    lines: list[str] = [f"**Proxy usage report — {_fmt_window(window_start, window_end)}**"]
    fallback = data.get("fallback_rate")
    busy = (data.get("local_busy") or {}).get("busy_pct")
    decode = (data.get("decode_speed") or {}).get("median_tok_s")
    parts = [
        f"Sessions **{data.get('sessions', 0)}**",
        f"requests **{data.get('total_requests', 0)}** "
        f"(local {data.get('local_requests', 0)} / remote {data.get('remote_requests', 0)})",
    ]
    lines.append(" · ".join(parts))
    figures = []
    if isinstance(fallback, (int, float)):
        figures.append(f"fallback **{fallback * 100:.1f}%**")
    if isinstance(busy, (int, float)):
        figures.append(f"local busy **{busy:.1f}%**")
    if isinstance(decode, (int, float)):
        figures.append(f"decode median **{decode:.1f} tok/s**")
    figures.append(f"**{data.get('errors', 0)}** error events")
    lines.append(" · ".join(figures))

    ranked = sorted(
        recommendations or [],
        key=lambda r: _severity_rank(getattr(r, "severity", "info")),
    )[:top_n]
    if ranked:
        lines.append("")
        lines.append("**Top findings**")
        for rec in ranked:
            severity = str(getattr(rec, "severity", "info")).upper()
            title = str(getattr(rec, "title", "")).strip()
            lines.append(f"• [{severity}] {title}")
    lines.append("")
    lines.append(
        f"Full report attached (`{attachment_name}`)."
        if attachment_name
        else "Full report attached."
    )
    return _truncate("\n".join(lines))


_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _severity_rank(severity: str) -> int:
    return _SEVERITY_ORDER.get(str(severity).lower(), len(_SEVERITY_ORDER))


def build_failure_message(
    detail: str,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    log_hint: str | None = DEFAULT_LOG_HINT,
) -> str:
    """Compose the Discord message body for a failed run."""
    lines = [
        f"⚠️ **proxy-usage-analysis failed** on `{socket.gethostname()}` "
        f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Window: {_fmt_window(window_start, window_end)}",
        f"Error: `{detail}`",
    ]
    if log_hint:
        lines.append(f"Log: `{log_hint}`")
    return _truncate("\n".join(lines))


def attachment_name_for(window_start: datetime | None, report_path: Path) -> str:
    """Dated attachment name so archived copies stay distinguishable."""
    if window_start is None:
        return report_path.name
    return f"proxy-usage-report-{window_start.strftime('%Y-%m-%d')}.md"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _multipart_body(payload: dict, files: list[tuple[str, bytes]]) -> tuple[bytes, dict]:
    """Encode a Discord webhook POST as ``multipart/form-data``.

    Discord expects the JSON payload in a ``payload_json`` part and each
    attachment in a ``files[N]`` part; the boundary is random per request. The
    ``User-Agent`` is part of the contract, not a nicety: Cloudflare answers the
    default Python agent with ``HTTP 403 error code: 1010``.
    """
    boundary = f"----proxyusage{uuid.uuid4().hex}"
    chunks: list[bytes] = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="payload_json"\r\n'
            "Content-Type: application/json\r\n\r\n"
        ).encode()
        + json.dumps(payload).encode("utf-8")
    ]
    for index, (filename, data) in enumerate(files):
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="files[{index}]"; '
                f'filename="{filename}"\r\n'
                "Content-Type: text/markdown; charset=utf-8\r\n\r\n"
            ).encode()
            + data
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"\r\n".join(chunks)
    return body, {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": USER_AGENT,
    }


def _post_multipart(url: str, body: bytes, headers: dict, timeout: float) -> None:
    """POST a prepared multipart body to the webhook (the default poster).

    The URL is deliberately absent from every error message: it carries the
    webhook token.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read()[:200].decode("utf-8", "replace").strip()
        except Exception:  # pragma: no cover - best-effort body read
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise DiscordDeliveryError(f"HTTP {exc.code} from the Discord webhook{suffix}") from exc
    except urllib.error.URLError as exc:
        raise DiscordDeliveryError(
            f"cannot reach the Discord webhook: {exc.reason}"
        ) from exc
    except OSError as exc:  # socket timeouts and connection resets
        raise DiscordDeliveryError(f"cannot reach the Discord webhook: {exc}") from exc


def send_report(
    webhook_url: str,
    report_path: str | Path,
    content: str,
    *,
    attachment_name: str | None = None,
    poster: Poster | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """POST *report_path* to the webhook as an attachment; return its name."""
    path = Path(report_path)
    name = attachment_name or path.name
    body, headers = _multipart_body(
        {"content": _truncate(content)}, [(name, path.read_bytes())]
    )
    (poster or _post_multipart)(webhook_url, body, headers, timeout)
    return name


def send_message(
    webhook_url: str,
    content: str,
    *,
    poster: Poster | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """POST a plain (attachment-free) message — used for failure notices."""
    body, headers = _multipart_body({"content": _truncate(content)}, [])
    (poster or _post_multipart)(webhook_url, body, headers, timeout)


# ---------------------------------------------------------------------------
# The report's own delivery bullet
# ---------------------------------------------------------------------------


def set_delivery_note(report_path: str | Path, text: str) -> bool:
    """Write the ``- Discord delivery:`` bullet into the report header.

    Idempotent: an existing bullet is replaced in place, so re-delivering or
    re-rendering never accumulates lines. Returns ``True`` when the report was
    updated, ``False`` when it could not be read (delivery must not fail
    because the report is unreadable — that failure is reported separately).
    """
    path = Path(report_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    note = f"{NOTE_PREFIX} {text}"
    for index, line in enumerate(lines):
        if line.startswith(NOTE_PREFIX):
            lines[index] = note
            break
    else:
        anchors = ("- Lines parsed:", "- Generated:")
        for anchor in anchors:
            position = next(
                (i for i, line in enumerate(lines) if line.startswith(anchor)), None
            )
            if position is not None:
                lines.insert(position + 1, note)
                break
        else:  # pragma: no cover - a report without a header is not a report
            lines.append(note)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def deliver(
    report_path: str | Path,
    content: str,
    *,
    window_start: datetime | None = None,
    enabled: bool = True,
    webhook_url: str | None = None,
    config_path: str | Path | None = None,
    poster: Poster | None = None,
    start: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> DeliveryResult:
    """Send the report and record the outcome in the report itself.

    Never raises: a failed send is reported through :class:`DeliveryResult`
    and the report bullet, so the analysis result is never lost to a delivery
    problem.
    """
    path = Path(report_path)
    if not enabled:
        result = DeliveryResult(
            configured=False,
            sent=False,
            note="disabled (--no-discord)",
            report_path=str(path),
        )
        set_delivery_note(path, result.note)
        return result

    source = resolve_webhook(webhook_url, config_path, start=start)
    if not source.configured:
        result = DeliveryResult(
            configured=False,
            sent=False,
            note=(
                f"not configured — no `discord.webhook_url` in {source.searched}; "
                "report not sent to Discord"
            ),
            report_path=str(path),
            webhook_source=source.searched,
        )
        set_delivery_note(path, result.note)
        return result

    name = attachment_name_for(window_start, path)
    try:
        send_report(
            source.url or "",
            path,
            content,
            attachment_name=name,
            poster=poster,
            timeout=timeout,
        )
    except Exception as exc:  # fail-soft: the report is already written
        detail = str(exc) or type(exc).__name__
        result = DeliveryResult(
            configured=True,
            sent=False,
            note=f"failed — {detail}",
            error=detail,
            report_path=str(path),
            attachment_name=name,
            webhook_source=source.searched,
        )
        set_delivery_note(path, result.note)
        return result

    result = DeliveryResult(
        configured=True,
        sent=True,
        note=(
            f"report sent as an attachment (`{name}`) at "
            f"{datetime.now().strftime('%H:%M:%S')}"
        ),
        report_path=str(path),
        attachment_name=name,
        webhook_source=source.searched,
    )
    set_delivery_note(path, result.note)
    return result


def notify_failure(
    detail: str,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    log_hint: str | None = DEFAULT_LOG_HINT,
    enabled: bool = True,
    webhook_url: str | None = None,
    config_path: str | Path | None = None,
    poster: Poster | None = None,
    start: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> DeliveryResult:
    """Announce a failed run on the webhook.

    Fail-soft by construction: this runs on the error path, so it must never
    raise, never mask the original failure and never change the exit code.
    """
    if not enabled:
        return DeliveryResult(
            configured=False, sent=False, note="disabled (--no-discord)"
        )
    source = resolve_webhook(webhook_url, config_path, start=start)
    if not source.configured:
        return DeliveryResult(
            configured=False,
            sent=False,
            note=f"not configured — no `discord.webhook_url` in {source.searched}",
            webhook_source=source.searched,
        )
    message = build_failure_message(
        detail, window_start=window_start, window_end=window_end, log_hint=log_hint
    )
    try:
        send_message(source.url or "", message, poster=poster, timeout=timeout)
    except Exception as exc:
        failure = str(exc) or type(exc).__name__
        return DeliveryResult(
            configured=True,
            sent=False,
            note=f"failed — {failure}",
            error=failure,
            webhook_source=source.searched,
        )
    return DeliveryResult(
        configured=True,
        sent=True,
        note="failure notice sent",
        webhook_source=source.searched,
    )
