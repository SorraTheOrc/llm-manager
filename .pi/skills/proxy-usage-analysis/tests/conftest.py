"""Pytest configuration: make the skill's scripts package importable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import discord_notify  # noqa: E402


@pytest.fixture(autouse=True)
def _block_real_discord_delivery(monkeypatch):
    """No test may reach the live Discord webhook (LP-0MW2BD6G8001E960).

    The skill's CLI delivers the report by default, so a test that drives
    ``analyze_proxy_usage.main()`` would otherwise post to the webhook in
    ``.worklog/config.yaml``. Only the network seam is replaced here: delivery
    tests inject their own ``poster=`` and so are unaffected, while any run
    that forgets to is refused loudly instead of reaching Discord.

    Tests that drive the CLI keep their report clean by passing
    ``--no-discord``; the two CLI helpers that do not need delivery state it
    explicitly.
    """

    def _refuse(url, body, headers, timeout):
        raise discord_notify.DiscordDeliveryError(
            "the test suite blocks real Discord deliveries"
        )

    monkeypatch.setattr(discord_notify, "_post_multipart", _refuse)
