"""Checks on the published docs in docs/source."""

import importlib
import re
import tomllib
from pathlib import Path

import click

from aw_of3_finetune.cleanup import CleanupSettings
from aw_of3_finetune.validate import CHECKS

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "docs" / "source"
PRIVATE_PATTERNS = (
    "/home/",
    "~/workspace",
    "~/data",
    "~/scratch",
    "atomworks-dev",
    "baker-laboratory",
)
FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]+")
NOT_OUR_FLAGS = {"--manifest-path", "--runner-yaml"}  # pixi's and run_openfold's
TUTORIAL_IDS = {"1a3n", "1fdl", "1ubq", "2crb", "2q2k", "3u8v", "4x1f", "7riv"}


def _pages() -> list[Path]:
    return sorted(PAGES.rglob("*.md"))


def _private_hits() -> list[str]:
    hits = []
    for page in _pages():
        text = page.read_text()
        hits += [f"{page.name}: {p}" for p in PRIVATE_PATTERNS if p in text]
    return hits


def _command_flags() -> set[str]:
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    flags: set[str] = set()
    for target in scripts.values():
        module, name = target.split(":")
        command: click.Command = getattr(importlib.import_module(module), name)
        for param in command.params:
            flags.update(opt for opt in [*param.opts, *param.secondary_opts] if opt.startswith("--"))
    return flags


def _page_flags() -> set[str]:
    found: set[str] = set()
    for page in _pages():
        found.update(FLAG.findall(page.read_text()))
    return found - NOT_OUR_FLAGS


def test_pages_have_no_private_references():
    actual_hits = _private_hits()

    assert actual_hits == []


def test_every_cli_flag_in_the_docs_exists():
    actual_unknown_flags = _page_flags() - _command_flags()

    assert actual_unknown_flags == set()


def test_licence_is_declared():
    actual_first_line = (REPO / "LICENSE").read_text().strip().splitlines()[0].strip()

    assert actual_first_line == "Apache License"


def test_tutorial_inputs_are_the_eight_structures():
    actual_ids = {path.stem for path in (REPO / "examples" / "tutorial").glob("*.cif")}

    assert actual_ids == TUTORIAL_IDS


def test_every_check_is_in_the_reference():
    page = (PAGES / "reference" / "report-checks.md").read_text()

    actual_missing = [check for check in CHECKS if f"| {check} " not in page]

    assert actual_missing == []


def test_every_cleanup_switch_is_in_the_reference():
    page = (PAGES / "reference" / "config-files.md").read_text()

    actual_missing = [name for name in CleanupSettings.model_fields if f"`{name}`" not in page]

    assert actual_missing == []
