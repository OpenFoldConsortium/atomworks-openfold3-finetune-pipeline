"""User-provided MSAs: the sources manifest, file checks, the source index and import.

A sources manifest (YAML or JSON) maps each OpenFold3 MSA source name to a directory
(searched recursively) or a glob of MSA files. Files are matched to dataset chains by
their query (first) sequence, so they can be named anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from atomworks.enums import MSAFileExtension

# Source names with an OpenFold3-internal or reserved meaning.
REJECTED_SOURCE_NAMES: frozenset[str] = frozenset({"colabfold_paired", "dummy", "templates"})
# The A3M members of atomworks' MSAFileExtension; only these are imported.
A3M_EXTENSIONS: tuple[str, ...] = tuple(e.value for e in MSAFileExtension if e.value.startswith(".a3m"))

_SOURCE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_GLOB_CHARS = set("*?[")


class MsaSourcesError(ValueError):
    """A problem with user-provided MSAs; the message lists every problem found."""


@dataclass(frozen=True)
class SourceSpec:
    """One manifest entry.

    Attributes:
        name: Source name; becomes the OpenFold3 alignment file stem.
        spec: The manifest value as written (directory or glob).
        files: Every file it selects, absolute and sorted, accepted or not.
    """

    name: str
    spec: str
    files: list[Path]


def source_name_problem(name: object) -> str | None:
    """Say what is wrong with a source name, or None if it is valid.

    Args:
        name: Candidate source name.

    Returns:
        A one-line problem description, or None.
    """
    if not isinstance(name, str) or not _SOURCE_NAME.match(name):
        return f"source name {name!r} must be letters, digits, '_' or '-'"
    if name in REJECTED_SOURCE_NAMES:
        return f"source name {name!r} is reserved (not allowed: {', '.join(sorted(REJECTED_SOURCE_NAMES))})"
    return None


def _select_files(value: str, base: Path) -> tuple[list[Path], str | None]:
    """Expand one manifest value into files.

    Args:
        value: Directory or glob, absolute or relative to ``base``.
        base: The manifest's folder.

    Returns:
        (sorted absolute files, problem or None).
    """
    if _GLOB_CHARS & set(value):
        pattern = Path(value)
        if pattern.is_absolute():
            anchor = Path(pattern.anchor)
            matches = anchor.glob(str(pattern.relative_to(anchor)))
        else:
            matches = base.glob(value)
        files = sorted(p.resolve() for p in matches if p.is_file())
        return files, None if files else f"glob {value!r} matches no files"
    directory = (base / value).resolve()
    if not directory.is_dir():
        return [], f"directory {value!r} ({directory}) does not exist"
    return sorted(p for p in directory.rglob("*") if p.is_file()), None


def load_msa_sources_manifest(path: Path) -> dict[str, SourceSpec]:
    """Load a sources manifest.

    Args:
        path: YAML or JSON file mapping source name to a directory or glob. Relative
            values resolve against the manifest's folder.

    Returns:
        ``{name: SourceSpec}`` in manifest order.

    Raises:
        MsaSourcesError: Listing every problem (bad names, missing directories, globs
            matching nothing, a manifest that is not a name-to-string mapping).
    """
    try:
        content = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise MsaSourcesError(f"{path}: not valid YAML/JSON: {exc}") from exc
    if content is None or content == {}:
        raise MsaSourcesError(f"{path}: no sources listed")
    if not isinstance(content, dict):
        raise MsaSourcesError(f"{path}: must map each source name to a directory or glob")
    base = path.resolve().parent
    problems: list[str] = []
    specs: dict[str, SourceSpec] = {}
    for name, value in content.items():
        if (problem := source_name_problem(name)) is not None:
            problems.append(problem)
            continue
        if not isinstance(value, str):
            problems.append(f"{name}: value must be a directory or glob string, got {value!r}")
            continue
        files, problem = _select_files(value, base)
        if problem is not None:
            problems.append(f"{name}: {problem}")
            continue
        specs[name] = SourceSpec(name=name, spec=value, files=files)
    if problems:
        raise MsaSourcesError(f"{path}:\n  " + "\n  ".join(problems))
    return specs
