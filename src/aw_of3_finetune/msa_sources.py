"""User-provided MSAs: the sources manifest, file checks, the source index and import.

A sources manifest (YAML or JSON) maps each OpenFold3 MSA source name to a directory
(searched recursively) or a glob of MSA files. Files are matched to dataset chains by
their query (first) sequence, so they can be named anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from atomworks.enums import MSAFileExtension
from atomworks.io.utils.compression import transfer_with_compression
from atomworks.io.utils.io_utils import apply_sharding_pattern
from atomworks.ml.utils.io import open_file
from openfold3.core.data.io.sequence.msa import parse_a3m
from openfold3.core.data.primitives.sequence.msa import process_msa_pairing_metadata
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings

from aw_of3_finetune.constants import (
    COLABFOLD_UNPAIRED_DIRNAME,
    MSA_INDEX_NAME,
    MSA_SOURCES_DIRNAME,
    OF3_ALIGNMENTS_DIRNAME,
)
from aw_of3_finetune.msa import alignment_rep_id, collect_msa_queries

if TYPE_CHECKING:
    from aw_of3_finetune.pipeline import ReorganizeSummary

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


class MixedMsaDirError(MsaSourcesError):
    """An MSA folder would hold both ColabFold and user-provided MSAs."""


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


def _load_msa_folder(folder: Path) -> dict[str, SourceSpec]:
    """Use an earlier run's MSA folder as a manifest: its source index lists the sources.

    Args:
        folder: An organized MSA folder (written by generate-msas or import-msas). One
            without msa_index.yml but with ColabFold's unpaired/ reads as colabfold_main.

    Returns:
        ``{name: SourceSpec}`` with every file in each source's store.

    Raises:
        MsaSourcesError: If the folder holds no MSA sources or a listed store is missing.
    """
    index = read_msa_index(folder)
    if not index:
        raise MsaSourcesError(
            f"{folder}: no MSA sources found (no {MSA_INDEX_NAME} and no {COLABFOLD_UNPAIRED_DIRNAME}/). "
            "Pass an MSA folder from an earlier run, or a sources manifest file."
        )
    problems = [
        f"{name}: store {store} listed in {folder / MSA_INDEX_NAME} does not exist"
        for name, store in index.items() if not store.is_dir()
    ]
    if problems:
        raise MsaSourcesError(f"{folder}:\n  " + "\n  ".join(problems))
    return {
        name: SourceSpec(
            name=name,
            spec=str(store.relative_to(folder)),
            files=sorted(p.resolve() for p in store.rglob("*") if p.is_file()),
        )
        for name, store in index.items()
    }


def load_msa_sources_manifest(path: Path) -> dict[str, SourceSpec]:
    """Load a sources manifest, or the source index of an earlier run's MSA folder.

    Args:
        path: YAML or JSON file mapping source name to a directory or glob (relative
            values resolve against the manifest's folder), or an organized MSA folder
            from an earlier run (generate-msas or import-msas), whose sources are read
            from its msa_index.yml.

    Returns:
        ``{name: SourceSpec}`` in manifest order.

    Raises:
        MsaSourcesError: Listing every problem (bad names, missing directories, globs
            matching nothing, a manifest that is not a name-to-string mapping).
    """
    if path.is_dir():
        return _load_msa_folder(path)
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
        unmatched: A3M files whose query matches no dataset sequence; skipped (not checked, not copied).
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


def query_sequence(text: str) -> str:
    """The first (query) sequence of A3M/FASTA text, joining wrapped lines as OpenFold3 does.

    atomworks' extract_first_sequence_from_msa reads only the first line, which drops
    the rest of a wrapped query.

    Args:
        text: A3M text.

    Returns:
        The query sequence.

    Raises:
        ValueError: If the text has no sequence after a header.
    """
    lines: list[str] = []
    started = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if started and lines:
                break
            started = True
        elif started and line:
            lines.append(line)
    if not lines:
        raise ValueError("no sequence found")
    return "".join(lines)


def _read_head(path: Path) -> str:
    """The text of an MSA file up to the end of its query sequence (enough to identify it).

    Args:
        path: MSA file (.gz / .zst decompressed).

    Returns:
        The first line through the query's last line.
    """
    lines: list[str] = []
    headers = 0
    with open_file(path) as f:
        for line in f:
            if line.startswith(">"):
                headers += 1
                if headers == 2:
                    break
            lines.append(line)
    return "".join(lines)


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
    content. Files whose query is not a dataset sequence are only listed: they are not
    parsed or copied, so a large library of unrelated MSAs costs one header read per file.

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
                head = _read_head(path)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}: {path}: cannot read: {exc}")
                continue
            if _COMPLEX_A3M_HEADER.match(head):
                problems.append(
                    f"{name}: {path}: multi-chain (complex) A3M; only per-sequence MSAs are accepted"
                )
                continue
            try:
                rep_id = alignment_rep_id(query_sequence(head))
            except ValueError as exc:
                problems.append(f"{name}: {path}: {exc}")
                continue
            if rep_id not in queries:  # skipped: not parsed, not copied
                unmatched.append(path)
                continue
            text = read_msa_text(path)
            if rep_id in seen:
                problems.append(f"{name}: {seen[rep_id]} and {path} have the same query sequence")
                continue
            seen[rep_id] = path
            if (problem := _format_problem(text, name)) is not None:
                problems.append(f"{name}: {path}: OpenFold3 cannot read it: {problem}")
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


def _validated_index(msa_dir: Path, raw: object, index_path: Path) -> dict[str, Path]:
    """Check a raw index mapping and resolve its store folders.

    Args:
        msa_dir: The organized MSA folder.
        raw: Parsed index content.
        index_path: The index file, for messages.

    Returns:
        ``{name: absolute store folder}``, sorted by name.

    Raises:
        MsaSourcesError: If the content is not a valid index.
        MixedMsaDirError: If it lists both ColabFold's store and user stores.
    """
    if not isinstance(raw, dict):
        raise MsaSourcesError(f"{index_path}: must map each source name to its store folder")
    problems: list[str] = []
    index: dict[str, Path] = {}
    root = msa_dir.resolve()
    for name, value in raw.items():
        if (problem := source_name_problem(name)) is not None:
            problems.append(problem)
            continue
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            problems.append(f"{name}: store folder must be a path relative to {msa_dir}, got {value!r}")
            continue
        if not (root / value).resolve().is_relative_to(root):
            problems.append(f"{name}: store folder {value!r} is outside {msa_dir}")
            continue
        index[name] = msa_dir / value
    if problems:
        raise MsaSourcesError(f"{index_path}:\n  " + "\n  ".join(problems))
    unpaired = msa_dir / COLABFOLD_UNPAIRED_DIRNAME
    if unpaired in index.values() and len(index) > 1:
        raise MixedMsaDirError(
            f"{index_path} lists both ColabFold MSAs ({COLABFOLD_UNPAIRED_DIRNAME}/) and user MSAs; "
            "one MSA folder holds one or the other. Use a fresh --organized-msa-dir."
        )
    return dict(sorted(index.items()))


def _raw_index(msa_dir: Path) -> dict:
    """The index as stored, or the implicit one of a folder written before the index existed."""
    index_path = msa_dir / MSA_INDEX_NAME
    if index_path.exists():
        try:
            return yaml.safe_load(index_path.read_text())
        except yaml.YAMLError as exc:
            raise MsaSourcesError(f"{index_path}: not valid YAML: {exc}") from exc
    if (msa_dir / COLABFOLD_UNPAIRED_DIRNAME).is_dir():
        return {"colabfold_main": COLABFOLD_UNPAIRED_DIRNAME}
    return {}


def read_msa_index(msa_dir: Path) -> dict[str, Path]:
    """Read the source index of an organized MSA folder.

    The only reader of ``msa_index.yml`` and the only place that names ColabFold's
    ``unpaired/`` as a source. A folder with no index but an ``unpaired/`` directory
    (written before the index existed) reads as ``{"colabfold_main": unpaired/}``;
    reading never writes. A listed store missing on disk is kept; callers warn.

    Args:
        msa_dir: The organized MSA folder.

    Returns:
        ``{source name: store folder}``, sorted by name; empty if the folder has no MSAs.

    Raises:
        MsaSourcesError: If the index is malformed.
        MixedMsaDirError: If it lists both ColabFold's store and user stores.
    """
    return _validated_index(msa_dir, _raw_index(msa_dir), msa_dir / MSA_INDEX_NAME)


def msa_dir_kind(msa_dir: Path) -> Literal["colabfold", "user", "empty"]:
    """Say whether an MSA folder holds ColabFold MSAs, user MSAs, or none, from its index.

    Args:
        msa_dir: The organized MSA folder.

    Returns:
        "colabfold", "user" or "empty".
    """
    index = read_msa_index(msa_dir)
    if not index:
        return "empty"
    if msa_dir / COLABFOLD_UNPAIRED_DIRNAME in index.values():
        return "colabfold"
    return "user"


def add_to_msa_index(msa_dir: Path, entries: dict[str, str]) -> None:
    """Merge entries into the source index (newer wins), refusing to write an invalid or mixed index.

    Args:
        msa_dir: The organized MSA folder (created if missing).
        entries: ``{source name: store folder relative to msa_dir}``.

    Raises:
        MsaSourcesError: If the merged index is invalid (nothing is written).
        MixedMsaDirError: If it would mix ColabFold and user MSAs (nothing is written).
    """
    read_msa_index(msa_dir)  # refuse to build on a malformed or mixed index
    merged = {**_raw_index(msa_dir), **entries}
    index_path = msa_dir / MSA_INDEX_NAME
    _validated_index(msa_dir, merged, index_path)
    msa_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(yaml.safe_dump(dict(sorted(merged.items())), sort_keys=False))


@dataclass
class ImportSummary:
    """What import_user_msas did.

    Attributes:
        scans: Per source, which files matched the dataset, which did not, which were ignored.
        covered: Per source, dataset sequences with an MSA from that source in this import.
        reorganize: The of3_alignments sync that followed.
        other_sources: Sources in the index from earlier imports, not in this manifest;
            they stay in of3_alignments/ and train.yml.
    """

    scans: list[SourceScan]
    covered: dict[str, int]
    reorganize: ReorganizeSummary
    other_sources: list[str]


def import_user_msas(output_dir: Path, msa_dir: Path, manifest_path: Path) -> ImportSummary:
    """Import user MSAs into atomworks-layout stores and sync them into of3_alignments/.

    Steps: refuse a ColabFold MSA folder; load the manifest; check every file
    (scan_sources) before copying anything; copy each file to
    ``<msa_dir>/msa_sources/<source>/<shard>/<rep_id>.a3m.gz`` (the layout
    organize_msas writes); add the sources to the source index; reorganize.

    Args:
        output_dir: Pipeline output directory; the dataset's sequences come from it.
        msa_dir: The organized MSA folder.
        manifest_path: Sources manifest (YAML/JSON).

    Returns:
        What was imported and what of3_alignments/ now holds.

    Raises:
        MixedMsaDirError: If msa_dir holds ColabFold MSAs.
        MsaSourcesError: For any problem with the manifest or files (nothing is copied).
    """
    from aw_of3_finetune.pipeline import reorganize_msas_for_of3  # pipeline imports this module

    if msa_dir_kind(msa_dir) == "colabfold":
        raise MixedMsaDirError(
            f"{msa_dir} holds ColabFold MSAs ({COLABFOLD_UNPAIRED_DIRNAME}/); one MSA folder holds "
            "ColabFold or user MSAs, not both. Use a fresh --organized-msa-dir."
        )
    manifest = load_msa_sources_manifest(manifest_path)
    queries = collect_msa_queries(output_dir)
    store_root = msa_dir / MSA_SOURCES_DIRNAME
    scans = scan_sources(manifest, queries, store_root)

    for scan in scans:
        for scanned in scan.files:
            dest = store_path(store_root, scanned.source, scanned.rep_id)
            if dest.exists():  # identical content, checked by scan_sources
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            transfer_with_compression(scanned.path, dest, move=False)
    add_to_msa_index(msa_dir, {name: f"{MSA_SOURCES_DIRNAME}/{name}" for name in manifest})
    reorganize = reorganize_msas_for_of3(output_dir, msa_dir, msa_dir / OF3_ALIGNMENTS_DIRNAME)
    return ImportSummary(
        scans=scans,
        covered={scan.source: len(scan.files) for scan in scans},
        reorganize=reorganize,
        other_sources=[name for name in read_msa_index(msa_dir) if name not in manifest],
    )
