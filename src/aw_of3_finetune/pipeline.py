"""Structure preparation and manifest building for fine-tuning datasets.

Chain metadata is always built from OF3's parse of this pipeline's own augmented CIF
(see prepare_augmented_cif), so it matches the NPZ by construction.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from atomworks.ml.preprocessing.msa.finding import find_msas
from openfold3.core.data.io.dataset_cache import write_datacache_to_json
from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.io.structure.mol import write_annotated_sdf
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetInterfaceData,
    ClusteredDatasetStructureData,
    DatasetReferenceMoleculeData,
)
from openfold3.core.data.primitives.structure.component import (
    get_component_info,
    get_reference_molecule_metadata,
    mol_from_atomarray,
    mol_from_ccd_entry,
)
from openfold3.core.data.primitives.structure.conformer import (
    resolve_and_format_fallback_conformer,
)
from openfold3.core.data.primitives.structure.interface import get_interface_chain_id_pairs
from openfold3.core.data.primitives.structure.metadata import get_release_date
from openfold3.core.data.resources.residues import MoleculeType

from aw_of3_finetune.augmented_cif import read_augmented_cif, write_augmented_cif
from aw_of3_finetune.cif_reader import parse_cif_chain_data
from aw_of3_finetune.cleanup import CleanupSettings, clean_structure
from aw_of3_finetune.constants import DUMMY_CLUSTER_ID, MSA_INDEX_NAME
from aw_of3_finetune.ligand_names import SignaturesByName, record_ligand_signatures
from aw_of3_finetune.msa import alignment_rep_id, chain_canonical_sequences, collect_msa_queries
from aw_of3_finetune.msa_sources import STORE_EXTENSION, read_msa_index, read_msa_text
from aw_of3_finetune.parsing import parse_structure
from aw_of3_finetune.steplog import StepRecorder, read_input_atoms
from aw_of3_finetune.templates import find_template_ids

if TYPE_CHECKING:
    from biotite.structure.io.pdbx import CIFFile

logger = logging.getLogger(__name__)


def _read_release_date(block) -> date:
    """Return the entry's release date: its earliest revision date, as OF3 defines it.

    Delegates to OF3's get_release_date; like OF3, there is no deposition-date fallback.

    Args:
        block: The source mmCIF's data block.

    Returns:
        The minimum ``pdbx_audit_revision_history.revision_date`` as a date.

    Raises:
        ValueError: If the block has no ``pdbx_audit_revision_history.revision_date``.
    """
    if (
        "pdbx_audit_revision_history" not in block
        or "revision_date" not in block["pdbx_audit_revision_history"]
    ):
        raise ValueError(
            "mmCIF has no pdbx_audit_revision_history.revision_date; OF3 defines the "
            "release date as the earliest revision date and has no fallback."
        )
    return get_release_date(block).date()


def _read_resolution(block) -> float | None:
    """Return the entry's high-resolution limit, if recorded.

    Args:
        block: The source mmCIF's data block.

    Returns:
        ``refine.ls_d_res_high`` as a float, or None if absent or ``?``/``.``.
    """
    if "refine" not in block:
        return None
    val = str(block["refine"]["ls_d_res_high"].as_array()[0])
    return None if val in ("?", ".") else float(val)


def _source_auth_asym_id(label_asym_id: str, chain_data: dict[str, dict]) -> str:
    """The source CIF's author chain id for a chain, as OF3 records it in the cache.

    The augmented CIF keeps auth_asym_id equal to label_asym_id: OF3 re-parses it with author
    fields, so they must stay unique per chain. Assembly copies ("A_2") map to their source chain.

    Args:
        label_asym_id: The chain's label_asym_id in the augmented CIF.
        chain_data: parse_cif_chain_data of the source CIF, keyed by label_asym_id.

    Returns:
        The source auth_asym_id, or label_asym_id if the chain is not in the source.
    """
    source_label = label_asym_id if label_asym_id in chain_data else label_asym_id.rsplit("_", 1)[0]
    return chain_data.get(source_label, {}).get("auth_asym_id", label_asym_id)


def build_interface_keys(atom_array: struc.AtomArray) -> list[str]:
    """Interface keys ``"<chain_1>_<chain_2>"`` for chain pairs within 5 A, in OF3's order.

    Args:
        atom_array: OF3-parsed atom array with integer-string chain IDs.

    Returns:
        One key per chain pair, as returned by OF3's get_interface_chain_id_pairs.
    """
    pairs = get_interface_chain_id_pairs(atom_array, distance_threshold=5.0)
    return [f"{chain_1}_{chain_2}" for chain_1, chain_2 in pairs.tolist()]


def build_structure_data(
    cif_path: Path,
    pdb_id: str,
    atom_array,
    template_cache_dir: Path | None = None,
) -> ClusteredDatasetStructureData:
    """Build ClusteredDatasetStructureData for the chains in ``atom_array``.

    Protein chains get a sequence-hash rep_id and template IDs; other chains get None
    for both (MSA/template search is protein-only). Interfaces are chain pairs within 5 A.
    Every chain and interface gets cluster_size = chains + interfaces of the structure,
    which makes OF3's sampling weight every structure equally.

    Args:
        cif_path: Source mmCIF; only its tables are read (release date, resolution,
            SEQRES sequences, CCD codes).
        pdb_id: Structure ID.
        atom_array: OF3's parse of this pipeline's own augmented CIF
            (parse_augmented_cif_for_training), never an independent parse of the
            source CIF -- the two parsers disagree on which chains exist (2q2k).
        template_cache_dir: OF3 template cache; protein chains get ``[]`` if None.

    Returns:
        Structure metadata with one ClusteredDatasetChainData per chain, keyed by
        OF3's integer chain ID.
    """
    block = pdbx.CIFFile.read(cif_path).block
    release_date = _read_release_date(block)
    resolution = _read_resolution(block)

    # Source tables are keyed by label_asym_id; re-key them by entity_id, which is
    # the annotation that survives assembly expansion and chain relabeling intact
    # (after chain_iid disambiguation the atoms' labels are "A_1"/"A_2").
    chain_data = parse_cif_chain_data(cif_path)
    label_to_entity = {label: data["entity_id"] for label, data in chain_data.items()}
    entity_seq: dict[int, str] = {
        label_to_entity[label]: seq
        for label, seq in chain_canonical_sequences(cif_path).items()
        if label in label_to_entity
    }
    entity_ref_mol: dict[int, str | None] = {
        data["entity_id"]: data["reference_mol_id"] for data in chain_data.values()
    }

    chains: dict[str, ClusteredDatasetChainData] = {}
    for idx in struc.get_chain_starts(atom_array):
        chain_id = str(atom_array.chain_id[idx])           # integer key: "1", "2", …
        label_asym_id = str(atom_array.label_asym_id[idx]) # "A", or "A_2" for symmetry copies
        auth_asym_id = _source_auth_asym_id(label_asym_id, chain_data)
        entity_id = int(atom_array.entity_id[idx])
        mol_type = MoleculeType(int(atom_array.molecule_type_id[idx])).name  # "PROTEIN", …

        is_protein = mol_type == "PROTEIN"
        seq = entity_seq.get(entity_id) if is_protein else None

        if seq is not None:
            rep_id = alignment_rep_id(seq)
            template_ids: list[str] | None = (
                find_template_ids(rep_id, template_cache_dir)
                if template_cache_dir is not None
                else []
            )
        else:
            rep_id = None
            template_ids = None

        chains[chain_id] = ClusteredDatasetChainData(
            label_asym_id=label_asym_id,
            auth_asym_id=auth_asym_id,
            entity_id=entity_id,
            molecule_type=mol_type,
            reference_mol_id=entity_ref_mol.get(entity_id),
            alignment_representative_id=rep_id,
            template_ids=template_ids,
            cluster_id=DUMMY_CLUSTER_ID,
            cluster_size=0,  # set below, once the interface count is known
        )

    interface_keys = build_interface_keys(atom_array)
    n_datapoints = len(chains) + len(interface_keys)
    for chain in chains.values():
        chain.cluster_size = n_datapoints
    interfaces = {
        key: ClusteredDatasetInterfaceData(cluster_id=DUMMY_CLUSTER_ID, cluster_size=n_datapoints)
        for key in interface_keys
    }

    return ClusteredDatasetStructureData(
        release_date=release_date,
        resolution=resolution,
        chains=chains,
        interfaces=interfaces,
    )


def _write_step_log(
    recorder: StepRecorder | None, pdb_id: str, step_log_dir: Path, write_step_cifs: bool
) -> None:
    """Write the step log JSON and, optionally, one plain mmCIF per recorded step.

    Without a usable recorder (None, or recording failed) no log is written and a stale
    ``<pdb_id>.json`` from an earlier run is removed.

    Args:
        recorder: Recorder filled by parse_structure and clean_structure, or None.
        pdb_id: Structure ID, used for file and directory names.
        step_log_dir: Directory for ``<pdb_id>.json`` and ``<pdb_id>/``.
        write_step_cifs: Write ``<pdb_id>/<NN>_<step>.cif`` after clearing old ones.
    """
    json_path = step_log_dir / f"{pdb_id}.json"
    if recorder is None or recorder.failed:
        json_path.unlink(missing_ok=True)
        return
    step_log_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(recorder.log().to_json())
    if not write_step_cifs:
        return
    cif_dir = step_log_dir / pdb_id
    shutil.rmtree(cif_dir, ignore_errors=True)
    cif_dir.mkdir()
    for i, (name, atoms) in enumerate(recorder.snapshots()):
        if atoms.array_length() == 0:
            continue  # the step that emptied the structure has nothing to show
        try:
            cif = pdbx.CIFFile()
            pdbx.set_structure(cif, atoms, data_block=pdb_id)
            cif.write(cif_dir / f"{i:02d}_{name}.cif")
        except Exception as exc:  # debug output must never mask the real error
            logger.warning("%s: could not write step CIF %s: %s", pdb_id, name, exc)


def _new_recorder(cif_path: Path, pdb_id: str, keep_snapshots: bool) -> StepRecorder | None:
    """Build a step recorder from the input atom table; None (with a warning) if that fails.

    Args:
        cif_path: Source mmCIF.
        pdb_id: Structure ID.
        keep_snapshots: Keep per-step atoms for step CIFs.

    Returns:
        The recorder, or None when the input atom table can't be read.
    """
    try:
        return StepRecorder(read_input_atoms(cif_path), pdb_id, keep_snapshots=keep_snapshots)
    except Exception as exc:  # the log must never fail the pipeline
        logger.warning(
            "%s: no cleanup step log; could not read the input atom table (%s: %s)",
            pdb_id, type(exc).__name__, exc,
        )
        return None


def prepare_augmented_cif(
    cif_path: Path,
    pdb_id: str,
    out_cif: Path,
    *,
    ccd: "CIFFile | None" = None,
    ref_mol_dir: Path | None = None,
    mol_cache: dict[str, DatasetReferenceMoleculeData] | None = None,
    template_cache_dir: Path | None = None,
    cleanup_settings: CleanupSettings | None = None,
    step_log_dir: Path | None = None,
    write_step_cifs: bool = False,
    ligand_signatures: SignaturesByName | None = None,
) -> tuple[ClusteredDatasetStructureData, dict[str, DatasetReferenceMoleculeData]]:
    """Write the augmented CIF for a source mmCIF, with metadata derived from its atoms.

    Two passes so OF3 only ever ingests what atomworks produced: write atoms with a
    metadata stub, OF3-parse that file, build metadata from it, rewrite out_cif.

    Args:
        cif_path: Source mmCIF.
        pdb_id: Structure ID.
        out_cif: Path to write the augmented CIF.
        ccd: Parsed Chemical Component Dictionary; needed for reference molecules.
        ref_mol_dir: Directory to write reference molecule SDFs.
        mol_cache: Shared mol_id -> reference molecule cache across structures.
        template_cache_dir: OF3 template cache, passed to build_structure_data.
        cleanup_settings: Structure cleanup switches; None means OF3's cleanup.
        step_log_dir: If given, write ``<pdb_id>.json`` (the cleanup step log) here,
            also when cleanup empties the structure. Logging failures only warn; the
            log is then not written.
        write_step_cifs: Also write one CIF per step to ``<step_log_dir>/<pdb_id>/``.
        ligand_signatures: If given with ``ccd``, collects this structure's non-CCD ligands
            for the one-name-one-molecule check. Mutated in place.

    Returns:
        ``(structure_data, reference_molecule_data)``; the latter is empty unless
        ccd and ref_mol_dir are both given.
    """
    recorder = None
    if step_log_dir is not None:
        recorder = _new_recorder(cif_path, pdb_id, keep_snapshots=write_step_cifs)
    try:
        atom_array_aw = parse_structure(cif_path, cleanup_settings, recorder)["assemblies"]["1"][0]
        atom_array_aw, _ = clean_structure(
            atom_array_aw, cif_path, pdb_id, ccd, cleanup_settings, recorder
        )
    finally:
        if step_log_dir is not None:
            try:
                _write_step_log(recorder, pdb_id, step_log_dir, write_step_cifs)
            except Exception as exc:  # never fail the entry, never mask EmptyStructureError
                logger.warning("%s: could not write the cleanup step log: %s", pdb_id, exc)

    block = pdbx.CIFFile.read(cif_path).block
    stub = ClusteredDatasetStructureData(
        release_date=_read_release_date(block),
        resolution=_read_resolution(block),
        chains={},
        interfaces={},
    )
    # Pass 1: chains-less stub. Nothing else in the file changes before the rewrite
    # below, so the chain numbering OF3 assigns here is the numbering the NPZ gets.
    write_augmented_cif(atom_array_aw, stub, {}, out_cif, pdb_id=pdb_id, source_cif=cif_path, ccd=ccd)
    atom_array_of3 = parse_augmented_cif_for_training(out_cif).atom_array

    structure_data = build_structure_data(cif_path, pdb_id, atom_array_of3, template_cache_dir)

    ref_mol_data: dict[str, DatasetReferenceMoleculeData] = {}
    if ccd is not None and ref_mol_dir is not None:
        ref_mol_data, overrides = build_reference_mol_data(
            atom_array_of3, pdb_id, ccd, ref_mol_dir, mol_cache
        )
        for chain_id, mol_id in overrides.items():
            if chain_id in structure_data.chains:
                structure_data.chains[chain_id].reference_mol_id = mol_id
        if ligand_signatures is not None:
            record_ligand_signatures(
                atom_array_of3, structure_data.chains, pdb_id, ccd, ligand_signatures
            )

    write_augmented_cif(
        atom_array_aw, structure_data, ref_mol_data, out_cif,
        pdb_id=pdb_id, source_cif=cif_path, ccd=ccd,
    )
    return structure_data, ref_mol_data


def get_entity_component_info(atom_array):
    """Run OF3's get_component_info on the first chain of each entity.

    get_component_info requires all chains of an entity to have identical atom topology,
    which fails for bioassembly-expanded structures (e.g. waters); reference molecules
    are entity-level anyway.

    Args:
        atom_array: OF3-parsed atom array of the structure.

    Returns:
        OF3's PDBComponentInfo for the restricted array.
    """
    seen_entities: set[int] = set()
    entity_first_chain_mask = np.zeros(len(atom_array), dtype=bool)
    for idx in struc.get_chain_starts(atom_array):
        eid = int(atom_array.entity_id[idx])
        if eid not in seen_entities:
            seen_entities.add(eid)
            entity_first_chain_mask |= (atom_array.chain_id == atom_array.chain_id[idx])
    return get_component_info(atom_array[entity_first_chain_mask])


def build_reference_mol_data(
    atom_array,
    pdb_id: str,
    ccd: "CIFFile",
    sdf_out_dir: Path,
    mol_cache: dict[str, DatasetReferenceMoleculeData] | None = None,
) -> tuple[dict[str, DatasetReferenceMoleculeData], dict[str, str]]:
    """Generate reference molecule SDFs and metadata for all components in atom_array.

    Polymer residues and standard ligands use CCD geometry (``{ccd_id}.sdf``);
    non-standard multi-residue ligands use crystal geometry (``{entity_id}_{pdb_id}.sdf``).

    Args:
        atom_array: OF3-parsed atom array of the structure.
        pdb_id: Structure ID, used to qualify non-standard ligand mol_ids.
        ccd: Parsed Chemical Component Dictionary.
        sdf_out_dir: Directory to write ``{mol_id}.sdf`` files.
        mol_cache: Shared mol_id -> metadata dict across structures; cached mol_ids
            skip SDF writing but are still returned. Mutated in place.

    Returns:
        ``(ref_mol_data, chain_mol_id_overrides)``: mol_id -> DatasetReferenceMoleculeData
        for all components, and chain_id -> mol_id for true multi-residue non-standard
        ligand chains whose reference_mol_id must be set after construction.
    """
    if mol_cache is None:
        mol_cache = {}

    sdf_out_dir.mkdir(parents=True, exist_ok=True)

    component_info = get_entity_component_info(atom_array)
    residue_components = component_info.residue_components
    std_ligands_to_chains = component_info.standard_ligands_to_chains
    non_std_ligands_to_chains = component_info.non_standard_ligands_to_chains
    non_std_ligands_to_rescount = component_info.non_standard_ligands_to_rescount

    # mol_id for non-standard ligands is qualified with pdb_id so structures don't
    # share SDFs for what are structurally different molecules
    non_std_mol_ids: dict[int, str] = {
        entity_id: f"{entity_id}_{pdb_id}"
        for entity_id in non_std_ligands_to_chains
    }

    mols_to_build: dict[str, object] = {}  # mol_id → mol (built below)

    # Polymer residues: CCD ideal geometry
    for ccd_id in residue_components:
        if ccd_id in mol_cache:
            continue
        try:
            mols_to_build[ccd_id] = mol_from_ccd_entry(ccd_id, ccd)
        except Exception as exc:
            logger.warning(f"{pdb_id}: mol_from_ccd_entry failed for {ccd_id}: {exc}")

    # Standard single-residue ligands: try CCD, fall back to crystal geometry
    for ccd_id, chains in std_ligands_to_chains.items():
        if ccd_id in mol_cache:
            continue
        try:
            mols_to_build[ccd_id] = mol_from_ccd_entry(ccd_id, ccd)
        except Exception:
            first_chain = chains[0]
            lig_aa = atom_array[atom_array.chain_id == first_chain]
            if lig_aa.array_length() > 0:
                try:
                    mols_to_build[ccd_id] = mol_from_atomarray(lig_aa)
                except Exception as exc:
                    logger.warning(f"{pdb_id}: mol_from_atomarray failed for {ccd_id}: {exc}")

    # Non-standard multi-residue ligands: use crystal geometry for heterogeneous
    # chains (different res_names), fall back to CCD for homogeneous chains (e.g. HOH×N).
    # Homogeneous chains use mol_id = res_name so the shared CCD SDF is reused, and
    # chain_mol_id_overrides is NOT populated (each residue is looked up individually).
    for entity_id, chains in non_std_ligands_to_chains.items():
        mol_id = non_std_mol_ids[entity_id]
        first_chain = chains[0]
        lig_aa = atom_array[
            (atom_array.entity_id == entity_id) & (atom_array.chain_id == first_chain)
        ]
        if lig_aa.array_length() == 0:
            logger.warning(f"{pdb_id}: empty atom array for non-std ligand entity {entity_id}; skipping")
            continue
        unique_resnames = list(np.unique(lig_aa.res_name))
        if len(unique_resnames) == 1:
            # Homogeneous chain (e.g. water, repeated ligand copies): treat as CCD entry.
            # mol_id becomes the shared CCD ID so deduplication works across structures.
            ccd_id = str(unique_resnames[0])
            non_std_mol_ids[entity_id] = ccd_id
            mol_id = ccd_id
            if mol_id not in mol_cache:
                try:
                    mols_to_build[mol_id] = mol_from_ccd_entry(ccd_id, ccd)
                except Exception as exc:
                    logger.warning(f"{pdb_id}: mol_from_ccd_entry failed for {ccd_id}: {exc}")
        else:
            if mol_id in mol_cache:
                continue
            try:
                mols_to_build[mol_id] = mol_from_atomarray(lig_aa)
            except Exception as exc:
                logger.warning(f"{pdb_id}: mol_from_atomarray failed for entity {entity_id}: {exc}")

    # Resolve conformers and write SDFs for new mols
    for mol_id, mol in mols_to_build.items():
        residue_count = non_std_ligands_to_rescount.get(mol_id, 1)
        try:
            mol, strategy, source = resolve_and_format_fallback_conformer(mol)
        except Exception as exc:
            logger.warning(f"{pdb_id}: resolve_and_format_fallback_conformer failed for {mol_id}: {exc}")
            continue
        meta = get_reference_molecule_metadata(mol, strategy, residue_count, source)
        mol_cache[mol_id] = DatasetReferenceMoleculeData(
            conformer_gen_strategy=meta["conformer_gen_strategy"],
            fallback_conformer_pdb_id=meta.get("fallback_conformer_pdb_id"),
            canonical_smiles=meta["canonical_smiles"],
            set_fallback_to_nan=False,
            fallback_conformer_source=meta.get("fallback_conformer_source"),
        )
        write_annotated_sdf(mol, sdf_out_dir / f"{mol_id}.sdf")

    # Collect ref_mol_data for all components in this structure (including cached)
    all_mol_ids = (
        set(residue_components)
        | set(std_ligands_to_chains)
        | set(non_std_mol_ids.values())
    )
    ref_mol_data = {
        mol_id: mol_cache[mol_id]
        for mol_id in all_mol_ids
        if mol_id in mol_cache
    }

    # chain_id → mol_id for truly multi-residue non-standard ligands only.
    # Homogeneous chains (resolved to a CCD ID) keep reference_mol_id=None so the
    # training pipeline decomposes them into individual residues as expected.
    original_non_std_mol_ids = {
        entity_id: f"{entity_id}_{pdb_id}"
        for entity_id in non_std_ligands_to_chains
    }
    chain_mol_id_overrides: dict[str, str] = {
        chain_id: non_std_mol_ids[entity_id]
        for entity_id, chains in non_std_ligands_to_chains.items()
        # Only override when the mol_id is the per-structure qualified form
        # (i.e. it is a true multi-residue ligand, not a CCD-resolved homogeneous chain)
        if non_std_mol_ids[entity_id] == original_non_std_mol_ids[entity_id]
        and non_std_mol_ids[entity_id] in mol_cache
        for chain_id in chains
    }

    return ref_mol_data, chain_mol_id_overrides


def write_slim_manifest(pdb_ids: list[str], output_path: Path, cache_name: str) -> None:
    """Write finetune-prep's slim manifest: dataset name and sorted structure IDs.

    The full ClusteredDatasetCache JSON is derived from this plus the CIFs by build_manifest.

    Args:
        pdb_ids: Structure IDs in the dataset.
        output_path: Path to write the manifest JSON.
        cache_name: Dataset name.
    """
    manifest = {"name": cache_name, "structures": sorted(pdb_ids)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def build_manifest(
    cif_dir: Path,
    output_path: Path,
    cache_name: str,
) -> ClusteredDatasetCache:
    """Assemble the of3_* categories of every augmented CIF into a ClusteredDatasetCache JSON.

    pdb_id is each file's stem; duplicate mol_ids across structures keep the first-seen entry.

    Args:
        cif_dir: Directory of augmented CIFs.
        output_path: Path to write the dataset cache JSON.
        cache_name: Dataset name.

    Returns:
        The cache that was written, in the format WeightedPDBDataset expects.
    """
    all_structure_data: dict[str, ClusteredDatasetStructureData] = {}
    all_ref_mol_data: dict = {}

    for cif_path in sorted(cif_dir.glob("*.cif")):
        pdb_id = cif_path.stem
        per_cache = read_augmented_cif(cif_path, pdb_id, cache_name)
        all_structure_data.update(per_cache.structure_data)
        for mol_id, rm in per_cache.reference_molecule_data.items():
            all_ref_mol_data.setdefault(mol_id, rm)

    cache = ClusteredDatasetCache(
        name=cache_name,
        structure_data=all_structure_data,
        reference_molecule_data=all_ref_mol_data,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        write_datacache_to_json(cache, f)
    return cache


@dataclass
class ReorganizeSummary:
    """What reorganize_msas_for_of3 changed.

    Attributes:
        written: Per source, ``.a3m`` files written or rewritten this run.
        removed: ``.a3m`` files removed because their source no longer has that sequence.
        missing: Dataset sequences with no MSA from any source.
    """

    written: dict[str, int]
    removed: int
    missing: int


def reorganize_msas_for_of3(output_dir: Path, msa_dir: Path, of3_msa_dir: Path) -> ReorganizeSummary:
    """Sync every indexed MSA store into OF3's directory-per-rep_id layout.

    OF3's alignments_directory can't read gzip or hash-sharded dirs. The sources come
    from the MSA folder's source index (``msa_index.yml``; ColabFold's ``unpaired/`` is
    source ``colabfold_main``). For each dataset sequence, ``{of3_msa_dir}/{rep_id}/``
    ends up holding exactly one ``<source>.a3m`` per source that has its MSA: missing
    files are written, changed ones rewritten, ones whose source no longer has the
    sequence removed. Directories of sequences outside the dataset are left alone. Safe
    to rerun.

    Args:
        output_dir: Pipeline output directory; the dataset's sequences come from it.
        msa_dir: The organized MSA folder.
        of3_msa_dir: Destination, ``{of3_msa_dir}/{rep_id}/{source}.a3m``.

    Returns:
        Counts of what changed and of sequences with no MSA.

    Raises:
        MsaSourcesError: If the source index is malformed or mixes ColabFold and user MSAs.
    """
    index = read_msa_index(msa_dir)
    written = {name: 0 for name in index}
    queries = collect_msa_queries(output_dir)  # rep_id -> sequence
    if not queries:
        return ReorganizeSummary(written=written, removed=0, missing=0)

    seqs = list(queries.values())
    found_by_source: dict[str, dict[str, Path]] = {}
    for name, store in index.items():
        if not store.is_dir():
            logger.warning(
                "MSA source %s: its store %s (listed in %s) does not exist; it provides no MSAs",
                name, store, msa_dir / MSA_INDEX_NAME,
            )
            found_by_source[name] = {}
            continue
        lookup = {"dir": str(store), "extension": STORE_EXTENSION, "directory_depth": 1}
        found_by_source[name] = find_msas(seqs, [lookup])[1]

    removed = missing = 0
    for rep_id, seq in queries.items():
        rep_dir = of3_msa_dir / rep_id
        provided = {name: found[seq] for name, found in found_by_source.items() if seq in found}
        if not provided:
            missing += 1
        for name, src in provided.items():
            text = read_msa_text(src)  # OF3 has no gzip support; write plain text
            dest = rep_dir / f"{name}.a3m"
            if dest.exists() and dest.read_text() == text:
                continue
            rep_dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            written[name] += 1
        if rep_dir.is_dir():
            for stale in rep_dir.glob("*.a3m"):
                if stale.stem not in provided:
                    stale.unlink()
                    removed += 1

    return ReorganizeSummary(written=written, removed=removed, missing=missing)


class NotAugmentedCifError(ValueError):
    """A CIF handed to the training-side parse wasn't written by this pipeline."""


def parse_augmented_cif_for_training(cif_path: Path):
    """Parse an augmented CIF with OF3 -- the parse that defines the NPZ's chain IDs.

    Anything that needs to agree with the NPZ must go through this exact call.

    Args:
        cif_path: Augmented CIF written by write_augmented_cif.

    Returns:
        OF3's parse_mmcif result; its atom_array has integer chain IDs ("1", "2", ...).

    Raises:
        NotAugmentedCifError: If the file has no ``of3_structure_meta`` category. A raw
            PDB entry would otherwise silently parse to an asymmetric-unit NPZ.
    """
    if "of3_structure_meta" not in pdbx.CIFFile.read(cif_path).block:
        raise NotAugmentedCifError(
            f"{cif_path}: no of3_structure_meta category -- not an augmented CIF from "
            "finetune-prep. This parse only makes sense on this pipeline's own output."
        )
    return parse_mmcif(
        cif_path,
        expand_bioassembly=False,  # atomworks already expanded the assembly
        include_bonds=True,
        # Integer chain IDs the NPZ carries and WeightedPDBDataset looks up in the cache.
        renumber_chain_ids=True,
    )


class ChainMismatchError(ValueError):
    """The chain metadata in an augmented CIF disagrees with its own atoms."""


def check_chain_consistency(augmented_cif: Path, structure_data: ClusteredDatasetStructureData) -> None:
    """Check an augmented CIF's chain metadata against the chains OF3 parses from it.

    Compares chain-ID sets and per-chain entity_id/molecule_type, so permutations are
    caught too. A mismatch fails at train time (get_reference_conformer_data_of3).

    Args:
        augmented_cif: Augmented CIF to re-parse for the NPZ's view of its chains.
        structure_data: The chain metadata written into it.

    Raises:
        ChainMismatchError: If any chain is missing from either side or disagrees.
    """
    atom_array = parse_augmented_cif_for_training(augmented_cif).atom_array
    meta = structure_data.chains

    actual: dict[str, tuple[int, str]] = {}
    for idx in struc.get_chain_starts(atom_array):
        cid = str(atom_array.chain_id[idx])
        actual[cid] = (
            int(atom_array.entity_id[idx]),
            MoleculeType(int(atom_array.molecule_type_id[idx])).name,
        )

    problems: list[str] = []
    for cid in sorted(set(meta) - set(actual), key=int):
        c = meta[cid]
        problems.append(f"chain {cid}: in metadata ({c.molecule_type} {c.label_asym_id}) but no atoms")
    for cid in sorted(set(actual) - set(meta), key=int):
        ent, mt = actual[cid]
        problems.append(f"chain {cid}: has atoms ({mt}, entity {ent}) but no metadata")
    for cid in sorted(set(meta) & set(actual), key=int):
        c = meta[cid]
        ent, mt = actual[cid]
        if (c.entity_id, c.molecule_type) != (ent, mt):
            problems.append(
                f"chain {cid}: metadata says {c.molecule_type} entity {c.entity_id} "
                f"({c.label_asym_id}), atoms are {mt} entity {ent}"
            )

    if problems:
        raise ChainMismatchError(
            f"{augmented_cif.name}: chain metadata disagrees with the structure's atoms "
            f"({len(meta)} chains in metadata, {len(actual)} in atoms). This structure "
            f"would fail at training time.\n  " + "\n  ".join(problems)
        )


def save_structures_as_npz(cif_dir: Path, out_dir: Path) -> None:
    """Convert a directory of augmented CIFs to NPZ structure files for training.

    Args:
        cif_dir: Directory of augmented CIFs.
        out_dir: Output root; writes ``{out_dir}/{pdb_id}/{pdb_id}.npz``, the layout
            WeightedPDBDataset expects under target_structures_directory.
    """
    for cif_path in sorted(cif_dir.glob("*.cif")):
        pdb_id = cif_path.stem
        parsed = parse_augmented_cif_for_training(cif_path)
        npz_dir = out_dir / pdb_id
        npz_dir.mkdir(parents=True, exist_ok=True)
        write_atomarray_to_npz(parsed.atom_array, npz_dir / f"{pdb_id}.npz")


def patch_manifest_ref_mol_data(
    manifest_path: Path,
    preprocessing_metadata_path: Path,
) -> None:
    """Merge reference_molecule_data from a preprocessing metadata.json into a manifest.

    Converts PreprocessingReferenceMoleculeData to DatasetReferenceMoleculeData (drops
    residue_count, adds set_fallback_to_nan). Existing manifest entries are kept.

    Args:
        manifest_path: Manifest JSON to update in place.
        preprocessing_metadata_path: metadata.json from preprocess_cif_dir_of3.
    """
    with manifest_path.open() as f:
        manifest = json.load(f)

    with preprocessing_metadata_path.open() as f:
        preproc_meta = json.load(f)

    existing = manifest.get("reference_molecule_data", {})
    for mol_id, mol in preproc_meta.get("reference_molecule_data", {}).items():
        existing.setdefault(
            mol_id,
            {
                "conformer_gen_strategy": mol["conformer_gen_strategy"],
                "fallback_conformer_pdb_id": mol.get("fallback_conformer_pdb_id"),
                "canonical_smiles": mol["canonical_smiles"],
                "set_fallback_to_nan": False,  # NaN-fallback is only for validation sets
            },
        )
    manifest["reference_molecule_data"] = existing

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=4)
