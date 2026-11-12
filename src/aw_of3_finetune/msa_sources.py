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
from atomworks.io.utils.io_utils import apply_sharding_pattern
from atomworks.ml.preprocessing.msa.organizing import extract_first_sequence_from_msa
from atomworks.ml.utils.io import open_file
from openfold3.core.data.io.sequence.msa import parse_a3m
from openfold3.core.data.primitives.sequence.msa import process_msa_pairing_metadata
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings

from aw_of3_finetune.constants import MSA_SOURCES_DIRNAME
from aw_of3_finetune.msa import alignment_rep_id

# Source names with an OpenFold3-internal or reserved meaning.
REJECTED_SOURCE_NAMES: frozenset[str] = frozenset({"colabfold_paired", "dummy", "templates"})
# The A3M members of atomworks' MSAFileExtension; only these are imported.
A3M_EXTENSIONS: tuple[str, ...] = tuple(e.value for e in MSAFileExtension if e.value.startswith(".a3m"))

# atomworks' default sharding (make_msas_colabfold_server, organize_msas) and store extension.
STORE_SHARDING_PATTERN = "/0:2/"
STORE_EXTENSION = ".a3m.gz"

_SOURCE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_GLOB_CHARS = set("*?[")
# ColabFold's complex (multi-chain) A3Ms start with "#<lengths>\t<cardinalities>".
_COMPLEX_A3M_HEADER = re.compile(r"^#\d+(,\d+)*\t\d+(,\d+)*")


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


@dataclass(frozen=True)
class ScannedFile:
    """A user MSA file matched to a dataset sequence.

    Attributes:
        path: The user's file.
        source: Its source name.
        rep_id: Hash of its query sequence (the dataset's alignment_representative_id).
    """

    path: Path
    source: str
    rep_id: str


@dataclass(frozen=True)
class SourceScan:
    """What one source's files hold.

    Attributes:
        source: Source name.
        files: A3M files whose query is a dataset sequence.
        unmatched: A3M files whose query matches no dataset sequence.
        ignored: Selected files that are not A3M (by extension).
    """

    source: str
    files: list[ScannedFile]
    unmatched: list[Path]
    ignored: list[Path]


def store_path(store_root: Path, source: str, rep_id: str) -> Path:
    """Where a source's MSA for rep_id lives: the path atomworks' organize_msas would use.

    Args:
        store_root: ``<organized-msa-dir>/msa_sources``.
        source: Source name.
        rep_id: Query sequence hash.

    Returns:
        ``<store_root>/<source>/<shard>/<rep_id>.a3m.gz``.
    """
    sharded = apply_sharding_pattern(rep_id, STORE_SHARDING_PATTERN)
    return store_root / source / sharded.parent / f"{sharded.name}{STORE_EXTENSION}"


def read_msa_text(path: Path) -> str:
    """Read an MSA file as text, decompressing .gz / .zst."""
    with open_file(path) as f:
        return f.read()


def _is_a3m(path: Path) -> bool:
    return path.name.endswith(A3M_EXTENSIONS)


def _format_problem(text: str, source: str) -> str | None:
    """Run OpenFold3's own readers over a file's text, as training will.

    Args:
        text: Full A3M text.
        source: Source name; sources OpenFold3 pairs on also get their headers parsed.

    Returns:
        OpenFold3's error, or None if the file reads.
    """
    try:
        msa = parse_a3m(text)
        if source in MSASettings().msas_to_pair:
            process_msa_pairing_metadata(list(msa.metadata))
    except Exception as exc:  # noqa: BLE001 -- any parser failure is reported to the user
        return f"{type(exc).__name__}: {exc}"
    return None


def scan_sources(
    manifest: dict[str, SourceSpec], queries: dict[str, str], store_root: Path
) -> list[SourceScan]:
    """Check every user file before anything is copied.

    Per A3M file: refuse ColabFold complex A3Ms; hash the query; refuse two files with
    one query in a source; run OpenFold3's A3M parser (and, for sources OpenFold3 pairs
    on, its species-header parser); refuse a file whose store copy exists with other
    content. Files whose query is not a dataset sequence are listed, not refused.

    Args:
        manifest: From load_msa_sources_manifest.
        queries: ``{rep_id: sequence}`` for the dataset (collect_msa_queries).
        store_root: ``<organized-msa-dir>/msa_sources``.

    Returns:
        One SourceScan per source, in manifest order.

    Raises:
        MsaSourcesError: Listing every problem across all sources.
    """
    problems: list[str] = []
    scans: list[SourceScan] = []
    for name, spec in manifest.items():
        files: list[ScannedFile] = []
        unmatched: list[Path] = []
        ignored = [p for p in spec.files if not _is_a3m(p)]
        seen: dict[str, Path] = {}
        a3ms = [p for p in spec.files if _is_a3m(p)]
        if not a3ms:
            problems.append(f"{name}: no A3M files ({', '.join(A3M_EXTENSIONS)}) in {spec.spec!r}")
        for path in a3ms:
            try:
                text = read_msa_text(path)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}: {path}: cannot read: {exc}")
                continue
            if _COMPLEX_A3M_HEADER.match(text):
                problems.append(
                    f"{name}: {path}: multi-chain (complex) A3M; only per-sequence MSAs are accepted"
                )
                continue
            try:
                rep_id = alignment_rep_id(extract_first_sequence_from_msa(path))
            except ValueError as exc:
                problems.append(f"{name}: {path}: {exc}")
                continue
            if rep_id in seen:
                problems.append(f"{name}: {seen[rep_id]} and {path} have the same query sequence")
                continue
            seen[rep_id] = path
            if (problem := _format_problem(text, name)) is not None:
                problems.append(f"{name}: {path}: OpenFold3 cannot read it: {problem}")
                continue
            if rep_id not in queries:
                unmatched.append(path)
                continue
            stored = store_path(store_root, name, rep_id)
            if stored.exists() and read_msa_text(stored) != text:
                problems.append(
                    f"{name}: {path}: a different MSA for this sequence is already imported "
                    f"({stored}); delete {MSA_SOURCES_DIRNAME}/{name}/ to replace it"
                )
                continue
            files.append(ScannedFile(path=path, source=name, rep_id=rep_id))
        scans.append(SourceScan(source=name, files=files, unmatched=unmatched, ignored=ignored))
    if problems:
        raise MsaSourcesError("Problems with the user MSAs:\n  " + "\n  ".join(problems))
    return scans
