"""Validation split: read the user's id,split CSV and split the dataset cache into train and val caches."""

import csv
import dataclasses
import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal, get_sequence_identity
from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json
from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.primitives.caches.filtering import (
    JOINT_LIGAND_EXCLUSION_SET,
    check_chain_metric_eligibility,
)
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ValidationDatasetCache,
    ValidationDatasetChainData,
    ValidationDatasetInterfaceData,
    ValidationDatasetReferenceMoleculeData,
    ValidationDatasetStructureData,
)
from openfold3.core.data.primitives.structure.tokenization import get_token_count
from openfold3.core.data.resources.residues import MoleculeType
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from aw_of3_finetune.msa import collect_msa_queries
from aw_of3_finetune.pipeline import get_entity_component_info
from aw_of3_finetune.templates import _GAP_PENALTY

PairKey = tuple[str, str]
Pair = tuple[PairKey, PairKey, float]
SPLIT_VALUES = ("val", "test")
REQUIRED_COLUMNS = ("id", "split")
SPLIT_DIRNAME = "split"
DATASET_CACHE_NAME = "dataset_cache.json"
METHOD_LINE = (
    "Similarity: proteins by global alignment (BLOSUM62, gap -10/-1), identity over the shorter sequence; "
    "ligands by Morgan fingerprints (radius 2, 2048 bits), Tanimoto"
)


class SplitCsvError(click.ClickException):
    """The split CSV has problems; the message lists every one."""


@dataclass(frozen=True)
class SplitAssignment:
    """Dataset-key IDs the user put in the val and test sets; all others are train."""

    val: frozenset[str]
    test: frozenset[str]


def read_split_csv(
    path: Path, known_ids: Iterable[str], dropped_ids: Iterable[str] = ()
) -> SplitAssignment:
    """Read an `id,split` CSV and check every row before returning.

    Messages cite the file's own line numbers (the header is line 1), as an editor shows them.

    Args:
        path: CSV with `id` and `split` columns; split is `val` or `test`.
        known_ids: Valid dataset keys (cache keys, or CIF stems for a pre-check).
        dropped_ids: CIF stems the pipeline dropped; reported with their own message.

    Returns:
        The val and test IDs, spelled as in `known_ids`.

    Raises:
        SplitCsvError: One line per problem row, or per file-level problem.
    """
    known = {i.lower(): i for i in known_ids}
    dropped = {i.lower() for i in dropped_ids}
    problems: list[str] = []
    val: set[str] = set()
    test: set[str] = set()
    seen: set[str] = set()

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [(name or "").strip().lower() for name in reader.fieldnames or []]
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            quoted = ", ".join(f"'{column}'" for column in missing)
            raise SplitCsvError(
                f"split CSV {path} has 1 problem(s):\n"
                f"  header: missing column {quoted} (expected 'id,split')"
            )
        for row in reader:
            line = reader.line_num
            extra = row.pop(None, None)
            cells = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if not any(cells.values()) and not extra:
                continue
            if extra:
                problems.append(f"line {line}: has extra cells; expected id,split")
                continue
            raw_id, split = cells["id"], cells["split"].lower()
            key = raw_id.lower()
            if key in seen:
                problems.append(f"line {line}: '{raw_id}' is listed more than once")
                continue
            seen.add(key)
            if key in dropped:
                problems.append(
                    f"line {line}: '{raw_id}' was dropped by the pipeline; see the finetune-prep log"
                )
            elif key not in known:
                problems.append(f"line {line}: '{raw_id}' is not in the dataset")
            if split not in SPLIT_VALUES:
                problems.append(f"line {line}: '{cells['split']}' is not a valid split (use val or test)")
            elif key in known:
                (val if split == "val" else test).add(known[key])

    if not val and not problems:
        problems.append("no val rows: at least one structure must be assigned to val")

    if problems:
        lines = "\n".join(f"  {problem}" for problem in problems)
        raise SplitCsvError(f"split CSV {path} has {len(problems)} problem(s):\n{lines}")
    return SplitAssignment(val=frozenset(val), test=frozenset(test))


def split_csv_ids(path: Path) -> set[str]:
    """Read the lowercased ids listed in an `id,split` CSV, without checking them.

    Args:
        path: CSV with an `id` column.

    Returns:
        Every non-empty id, lowercased.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [{(k or "").strip().lower(): (v or "") for k, v in row.items()} for row in reader]
    return {row.get("id", "").strip().lower() for row in rows} - {""}


@dataclass(frozen=True)
class SplitPaths:
    """Locations of the files in `<output-dir>/split/`."""

    dir: Path
    train_cache: Path
    val_cache: Path
    report: Path
    csv: Path
    source: Path


@dataclass
class SplitResult:
    """Outcome of split_dataset_cache."""

    n_train: int
    n_val: int
    n_test: int
    paths: SplitPaths
    warnings: list[str]
    report: str = ""


def split_paths(output_dir: Path) -> SplitPaths:
    """Locate the split files under an output directory.

    Args:
        output_dir: The pipeline output directory.

    Returns:
        The paths (which may not exist yet).
    """
    split_dir = Path(output_dir) / SPLIT_DIRNAME
    return SplitPaths(
        dir=split_dir,
        train_cache=split_dir / "train_cache.json",
        val_cache=split_dir / "val_cache.json",
        report=split_dir / "split_report.txt",
        csv=split_dir / "split.csv",
        source=split_dir / "source.json",
    )


def dataset_cache_sha256(output_dir: Path) -> str:
    """Hash the full dataset cache, to detect a split made from an older one.

    Args:
        output_dir: The pipeline output directory.

    Returns:
        Hex SHA-256 of `dataset_cache.json`.
    """
    return hashlib.sha256((Path(output_dir) / DATASET_CACHE_NAME).read_bytes()).hexdigest()


def _residue_counts_and_mol_ids(atom_array, pdb_id: str) -> tuple[dict[str, int], set[str]]:
    """Residue counts of the non-standard ligands, and every molecule id the structure may use."""
    info = get_entity_component_info(atom_array)
    counts = {
        f"{entity_id}_{pdb_id}": count
        for entity_id, count in info.non_standard_ligands_to_rescount.items()
    }
    mol_ids = {str(name) for name in np.unique(atom_array.res_name)} | set(counts)
    return counts, mol_ids


def _metric_eligible(chain: ValidationDatasetChainData) -> bool:
    """OF3's check_chain_metric_eligibility, run on a copy with molecule_type parsed to MoleculeType.

    Validation caches hold molecule_type as a string; OF3 parses it the same way before this check.
    """
    parsed = dataclasses.replace(chain, molecule_type=MoleculeType[chain.molecule_type])
    return check_chain_metric_eligibility(parsed, lig_exclusion_list=JOINT_LIGAND_EXCLUSION_SET)


def _scored(flags: dict, scored: bool) -> dict:
    """The validation flags with metric_eligible and use_metrics set to `scored`."""
    return {**flags, "metric_eligible": scored, "use_metrics": scored}


def to_validation_cache(
    cache: ClusteredDatasetCache, val_ids: set[str], output_dir: Path
) -> ValidationDatasetCache:
    """Build an OF3 validation cache from the val entries of a clustered cache.

    Every chain and interface is scored except ligands on OF3's metric exclusion list
    (AF3 SI Tables 9, 10, 12) and interfaces involving them; token and residue counts
    come from each entry's NPZ under `<output_dir>/structure_files`.

    Args:
        cache: The full dataset cache.
        val_ids: Dataset keys to include.
        output_dir: The pipeline output directory holding the NPZs.

    Returns:
        The validation cache, named `<name>-val`.
    """
    flags = {
        "low_homology": True,
        "metric_eligible": True,
        "use_metrics": True,
        "source_subset": "base",
    }
    structure_data = {}
    residue_counts: dict[str, int] = {}
    mol_ids: set[str] = set()
    for pdb_id in sorted(val_ids):
        entry = cache.structure_data[pdb_id]
        npz_path = Path(output_dir) / "structure_files" / pdb_id / f"{pdb_id}.npz"
        atom_array = read_atomarray_from_npz(npz_path)
        counts, entry_mol_ids = _residue_counts_and_mol_ids(atom_array, pdb_id)
        residue_counts.update(counts)
        mol_ids |= entry_mol_ids | {c.reference_mol_id for c in entry.chains.values() if c.reference_mol_id}
        chains = {
            chain_id: ValidationDatasetChainData(
                **dataclasses.asdict(chain), **flags, ranking_model_fit=None, sabdab_annotation=None
            )
            for chain_id, chain in entry.chains.items()
        }
        for chain in chains.values():
            chain.metric_eligible = chain.use_metrics = _metric_eligible(chain)
        structure_data[pdb_id] = ValidationDatasetStructureData(
            release_date=entry.release_date,
            resolution=entry.resolution,
            token_count=get_token_count(atom_array),
            chains=chains,
            interfaces={
                key: ValidationDatasetInterfaceData(
                    **dataclasses.asdict(interface), **_scored(flags, all(chains[c].use_metrics for c in key.split("_")))
                )
                for key, interface in entry.interfaces.items()
            },
        )
    reference_molecule_data = {
        mol_id: ValidationDatasetReferenceMoleculeData(
            **dataclasses.asdict(mol), residue_count=residue_counts.get(mol_id, 1)
        )
        for mol_id, mol in cache.reference_molecule_data.items()
        if mol_id in mol_ids
    }
    return ValidationDatasetCache(
        name=f"{cache.name}-val",
        structure_data=structure_data,
        reference_molecule_data=reference_molecule_data,
    )


def _protein_sequence(sequence: str) -> ProteinSequence:
    """Build a sequence, mapping letters outside the BLOSUM62 alphabet (e.g. U, O) to X."""
    allowed = set(SubstitutionMatrix.std_protein_matrix().get_alphabet1())
    return ProteinSequence("".join(c if c in allowed else "X" for c in sequence.upper()))


def _identity_percent(a: str, b: str) -> float:
    """Global-alignment identity over the shorter sequence, in percent; the gap penalty is the template filter's."""
    if a == b:
        return 100.0
    alignment = align_optimal(
        _protein_sequence(a), _protein_sequence(b), SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=_GAP_PENALTY, max_number=1
    )[0]
    return 100.0 * get_sequence_identity(alignment, mode="shortest")


def top_protein_pair(val: dict[PairKey, str], train: dict[PairKey, str]) -> Pair | None:
    """Find the val and train protein chains with the highest sequence identity.

    Unknown letters (e.g. U) are aligned as X. Ties go to the smallest (val, train) keys.

    Args:
        val: `(entry, chain_id)` to sequence, for val chains.
        train: The same, for train chains.

    Returns:
        `(val key, train key, identity percent)`, or None if either side is empty.
    """
    best: Pair | None = None
    for val_key in sorted(val):
        for train_key in sorted(train):
            identity = _identity_percent(val[val_key], train[train_key])
            if best is None or identity > best[2]:
                best = (val_key, train_key, identity)
    return best


def _eligible(key: PairKey, mol: Chem.Mol | None) -> bool:
    """Whether a ligand is compared: off OF3's metric exclusion list, with more than one heavy atom."""
    _, mol_id = key
    return mol_id not in JOINT_LIGAND_EXCLUSION_SET and mol is not None and mol.GetNumHeavyAtoms() > 1


def top_ligand_pair(val: dict[PairKey, Chem.Mol], train: dict[PairKey, Chem.Mol]) -> Pair | None:
    """Find the val and train ligands with the highest Morgan-fingerprint Tanimoto.

    Ligands on OF3's metric exclusion list (AF3 SI Tables 9, 10, 12) and molecules with at
    most one heavy atom are skipped, as validation doesn't score them. Ties go to the smallest keys.

    Args:
        val: `(entry, mol_id)` to molecule, for val ligands.
        train: The same, for train ligands.

    Returns:
        `(val key, train key, Tanimoto)`, or None if either side has no eligible ligand.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprints(mols):
        return {
            key: generator.GetFingerprint(m)
            for key, m in sorted(mols.items())
            if _eligible(key, m)
        }

    val_prints, train_prints = fingerprints(val), fingerprints(train)
    best: Pair | None = None
    for val_key, val_print in val_prints.items():
        for train_key in train_prints:
            score = DataStructs.TanimotoSimilarity(val_print, train_prints[train_key])
            if best is None or score > best[2]:
                best = (val_key, train_key, score)
    return best


def format_split_report(
    result: SplitResult,
    proteins: tuple[dict[PairKey, str], dict[PairKey, str]],
    ligands: tuple[dict[PairKey, Chem.Mol], dict[PairKey, Chem.Mol]],
) -> str:
    """Write the split report: method, counts, and the closest train/val protein and ligand.

    Args:
        result: Split counts.
        proteins: `(val, train)` chain sequences keyed by `(entry, chain_id)`.
        ligands: `(val, train)` molecules keyed by `(entry, mol_id)`.

    Returns:
        The report text, one line per item, ending in a newline.
    """
    total = result.n_train + result.n_val + result.n_test
    lines = [
        METHOD_LINE,
        f"Split: {result.n_train} train, {result.n_val} val, {result.n_test} test ({total} entries)",
    ]
    protein_pair = top_protein_pair(*proteins)
    if protein_pair:
        (v_entry, v_chain), (t_entry, t_chain), identity = protein_pair
        lines.append(
            f"Most similar proteins: val {v_entry} chain {v_chain} <-> "
            f"train {t_entry} chain {t_chain}, {identity:.1f}% identity"
        )
    else:
        side = "val" if not proteins[0] else "train"
        lines.append(f"Most similar proteins: none (no protein chains in {side})")
    ligand_pair = top_ligand_pair(*ligands)
    if ligand_pair:
        (v_entry, v_mol), (t_entry, t_mol), score = ligand_pair
        lines.append(f"Most similar ligands:  val {v_entry} {v_mol} <-> train {t_entry} {t_mol}, Tanimoto {score:.2f}")
    else:
        side = "val" if not any(_eligible(k, m) for k, m in ligands[0].items()) else "train"
        lines.append(f"Most similar ligands:  none (no ligands in {side})")
    return "\n".join(lines) + "\n"


def _report_inputs(
    cache: ClusteredDatasetCache, val_ids: set[str], train_ids: set[str], output_dir: Path, warnings: list[str]
):
    """Gather protein sequences and ligand molecules of the val and train entries."""
    queries = collect_msa_queries(output_dir)
    proteins: tuple[dict, dict] = ({}, {})
    ligands: tuple[dict, dict] = ({}, {})
    molecules: dict[str, Chem.Mol | None] = {}
    for side, ids in enumerate((val_ids, train_ids)):
        seen_reps: set[str] = set()
        seen_mols: set[str] = set()
        for entry_id in sorted(ids):
            for chain_id, chain in sorted(cache.structure_data[entry_id].chains.items()):
                rep_id = chain.alignment_representative_id
                if chain.molecule_type == "PROTEIN" and rep_id in queries and rep_id not in seen_reps:
                    seen_reps.add(rep_id)
                    proteins[side][(entry_id, chain_id)] = queries[rep_id]
                mol_id = chain.reference_mol_id
                if chain.molecule_type == "LIGAND" and mol_id and mol_id not in seen_mols:
                    seen_mols.add(mol_id)
                    if mol_id not in molecules:
                        sdf = output_dir / "reference_mols" / f"{mol_id}.sdf"
                        molecules[mol_id] = Chem.MolFromMolFile(str(sdf)) if sdf.exists() else None
                        if molecules[mol_id] is None:
                            warnings.append(f"reference molecule {mol_id} has no readable SDF; left out of the report")
                    if molecules[mol_id] is not None:
                        ligands[side][(entry_id, mol_id)] = molecules[mol_id]
    return proteins, ligands


def split_dataset_cache(output_dir: Path, split_csv: Path) -> SplitResult:
    """Split the full dataset cache into train and validation caches under `split/`.

    Builds everything in a sibling temp directory and swaps it in on success, so a
    failure leaves any earlier split intact. The similarity report is written with it.

    Args:
        output_dir: The pipeline output directory (holds `dataset_cache.json`).
        split_csv: The user's `id,split` CSV.

    Returns:
        Counts, the file locations, and any warnings.

    Raises:
        SplitCsvError: The CSV has problems; nothing is written.
        click.ClickException: `dataset_cache.json` is missing.
    """
    output_dir = Path(output_dir)
    paths = split_paths(output_dir)
    previous = paths.dir.with_name(f"{SPLIT_DIRNAME}.old")
    if previous.is_dir() and not paths.dir.exists():
        # A hard kill between the two renames below left the old split only here.
        previous.rename(paths.dir)
    cache_path = output_dir / DATASET_CACHE_NAME
    if not cache_path.is_file():
        raise click.ClickException(
            f"{cache_path} not found: run materialize-dataset-cache (or finetune-pipeline) first."
        )
    cache = read_datacache(cache_path)
    assignment = read_split_csv(split_csv, cache.structure_data)
    train_ids = set(cache.structure_data) - assignment.val - assignment.test

    train_cache = ClusteredDatasetCache(
        name=cache.name,
        structure_data={i: cache.structure_data[i] for i in cache.structure_data if i in train_ids},
        reference_molecule_data=cache.reference_molecule_data,
    )
    val_cache = to_validation_cache(cache, set(assignment.val), output_dir)

    warnings = []
    if not any(entry.interfaces for entry in cache.structure_data.values()):
        warnings.append(
            "the dataset cache has no interface metrics; rerun finetune-prep to add them"
        )

    result = SplitResult(
        n_train=len(train_ids),
        n_val=len(assignment.val),
        n_test=len(assignment.test),
        paths=paths,
        warnings=warnings,
    )
    proteins, ligands = _report_inputs(cache, set(assignment.val), train_ids, output_dir, warnings)
    result.report = format_split_report(result, proteins, ligands)

    staging = paths.dir.with_name(f"{SPLIT_DIRNAME}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.rmtree(previous, ignore_errors=True)
    try:
        write_datacache_to_json(train_cache, staging / paths.train_cache.name)
        write_datacache_to_json(val_cache, staging / paths.val_cache.name)
        (staging / paths.report.name).write_text(result.report)
        shutil.copyfile(split_csv, staging / paths.csv.name)
        (staging / paths.source.name).write_text(
            json.dumps({"dataset_cache_sha256": dataset_cache_sha256(output_dir)}, indent=2)
        )
        if paths.dir.exists():
            paths.dir.rename(previous)
        staging.rename(paths.dir)
        shutil.rmtree(previous, ignore_errors=True)
    except BaseException:
        if previous.exists() and not paths.dir.exists():
            previous.rename(paths.dir)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return result
