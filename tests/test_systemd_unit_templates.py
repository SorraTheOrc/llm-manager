#!/usr/bin/env python3
"""Regression tests for the systemd unit templates in ``scripts/systemd/``.

Covers LP-0MVWIA33O004RKP7. The templates in ``scripts/systemd/`` are the
**user-service** deployment path: ``scripts/systemd/README.md`` tells the
operator to copy them into ``~/.config/systemd/user/``. A ``systemd --user``
manager already runs as the target user and cannot call ``setgroups``, so a
user unit that declares ``User=``/``Group=`` fails at exec time with::

    Failed to determine supplementary groups: Operation not permitted
    Main process exited, code=exited, status=216/GROUP

Both templates previously carried ``User=rgardler``/``Group=rgardler``. Because
``llama-proxy.service`` declares ``Requires=llama-podman-wrapper.service``, both
units then failed, leaving the whole systemd-managed stack dead on every boot.

These tests parse the units into their directives (the same key/section view
systemd uses) rather than grepping the raw text, so they assert the property
that actually matters: *which directives a unit declares*. They also assert the
system-service variant documented under ``docs/systemd/`` still declares
``User=``/``Group=``, so the fix cannot regress into "nobody can install this as
a system service any more".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_UNITS = Path("scripts/systemd")
DOCS_UNIT = Path("docs/systemd/llama-proxy.service")

#: Templates installed into ``~/.config/systemd/user/`` per the README.
USER_SERVICE_UNITS = ["llama-podman-wrapper.service", "llama-proxy.service"]

#: Directives that only make sense in a *system* unit. A user manager cannot
#: change user/group membership; declaring them aborts the exec with 216/GROUP.
FORBIDDEN_IN_USER_UNIT = {"User", "Group"}


def parse_unit_directives(text: str) -> dict[str, list[tuple[str, str]]]:
    """Parse an ini-style unit file into ``{section: [(key, value), ...]}``.

    ``configparser`` cannot be used directly: unit files legitimately repeat
    keys (``Environment=`` appears several times) and it would reject them as
    duplicates. Directives are returned as an ordered list so repeats survive.
    """
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, [])
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        sections.setdefault(current, []).append((key.strip(), value.strip()))
    return sections


def directive_keys(sections: dict[str, list[tuple[str, str]]]) -> set[str]:
    """Every directive name declared anywhere in the unit."""
    return {key for entries in sections.values() for key, _ in entries}


def service_directives(path: Path) -> dict[str, list[tuple[str, str]]]:
    return parse_unit_directives(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("unit_name", USER_SERVICE_UNITS)
def test_user_service_unit_declares_no_user_or_group(unit_name: str) -> None:
    """A user-service unit must not declare User=/Group= (216/GROUP)."""
    path = REPO_ROOT / USER_UNITS / unit_name
    assert path.is_file(), f"missing user-service template: {path}"

    declared = directive_keys(service_directives(path))
    offenders = sorted(FORBIDDEN_IN_USER_UNIT & declared)
    assert not offenders, (
        f"{unit_name} is installed as a *user* service but declares {offenders}. "
        "The systemd user manager cannot set supplementary groups, so the unit "
        "fails with 'Failed to determine supplementary groups' / status=216/GROUP. "
        "Remove these directives; a system-service installation needs them instead "
        "(see docs/systemd/)."
    )


@pytest.mark.parametrize("unit_name", USER_SERVICE_UNITS)
def test_user_service_unit_keeps_its_service_contract(unit_name: str) -> None:
    """Stripping User=/Group= must not damage the rest of the unit."""
    path = REPO_ROOT / USER_UNITS / unit_name
    sections = service_directives(path)

    assert "Unit" in sections and "Service" in sections
    assert "Install" in sections

    service_keys = {key for key, _ in sections["Service"]}
    # The unit must still name something to run and forbid silent death.
    assert "ExecStart" in service_keys
    assert "WorkingDirectory" in service_keys
    assert "Restart" in service_keys


def test_proxy_unit_requires_the_wrapper_unit() -> None:
    """The proxy depends on the podman wrapper for its local backend.

    This dependency is why a broken wrapper takes the proxy down with it, so it
    must stay explicit rather than being quietly dropped.
    """
    path = REPO_ROOT / USER_UNITS / "llama-proxy.service"
    unit_keys = {key for key, _ in service_directives(path)["Unit"]}
    assert "Requires" in unit_keys


def _ini_blocks(markdown: str) -> list[str]:
    return re.findall(r"```ini\n(.*?)```", markdown, flags=re.DOTALL)


def test_documented_system_service_variant_declares_user_and_group() -> None:
    """The system-service guidance must still exist and set User=/Group=.

    Removing User=/Group= from the user templates must not remove the operator's
    route to a *system* service, where those directives are required (and where
    omitting them would run the proxy as root).
    """
    doc = (REPO_ROOT / DOCS_UNIT).read_text(encoding="utf-8")
    blocks = [directive_keys(parse_unit_directives(b)) for b in _ini_blocks(doc)]
    assert blocks, f"no ini unit examples found in {DOCS_UNIT}"

    system_variants = [b for b in blocks if {"User", "Group"} <= b]
    user_variants = [b for b in blocks if not ({"User", "Group"} & b)]

    assert system_variants, (
        "docs/systemd/llama-proxy.service must keep a system-service variant "
        "that declares User= and Group=."
    )
    assert user_variants, (
        "docs/systemd/llama-proxy.service must keep a user-service variant that "
        "declares neither User= nor Group=."
    )


def test_readme_warns_against_user_group_in_user_units() -> None:
    """The install guide must state the rule, not leave it to trial and error."""
    readme = (REPO_ROOT / USER_UNITS / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert "216/group" in lowered, "README should name the 216/GROUP failure mode"
    # The old, ambiguous instruction that let the bug through.
    assert "adjust user/group and paths in the unit files if your setup differs" not in lowered
