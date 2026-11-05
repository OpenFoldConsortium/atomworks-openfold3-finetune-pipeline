"""Structure cleanup that makes atomworks output match OF3's cleanup_structure_of3.

Unresolved atoms, hydrogens and arginine naming are handled by the atomworks parse; the
steps here (waters and MSE included, so they are logged) are OF3's own functions.
CleanupSettings switches individual steps off; its defaults are OF3's cleanup, except the
altloc choice (atomworks' first letter per chain; OF3's highest occupancy is an option).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from biotite.structure import AtomArray, BondList, index_distance
from biotite.structure.io.pdbx import CIFFile
from openfold3.core.data.primitives.structure.cleanup import (
    convert_MSE_to_MET,
    prefilter_bonds,
    remove_chains_with_CA_gaps,
    remove_clashing_chains,
    remove_crystallization_aids,
    remove_fully_unknown_polymers,
    remove_non_CCD_atoms,
    remove_small_polymers,
    remove_std_residue_terminal_atoms,
    remove_waters,
)
from openfold3.core.data.primitives.structure.component import get_covalent_component_chain_ids
from openfold3.core.data.primitives.structure.labels import assign_molecule_type_ids
from openfold3.core.data.primitives.structure.metadata import get_experimental_method
from openfold3.core.data.resources.residues import STANDARD_RESIDUES_3, MoleculeType
from pydantic import BaseModel, ConfigDict

from aw_of3_finetune.steplog import StepRecorder

logger = logging.getLogger(__name__)

_XRAY = "X-RAY DIFFRACTION"

# Parameters as passed in OF3's cleanup_structure_of3.
_BOND_FILTER_ARGS = {
    "remove_inter_chain_dative": True,
    "remove_inter_chain_poly_links": True,
    "remove_intra_chain_poly_links": True,
    "remove_longer_than": 2.4,
}


_STEP_DESCRIPTIONS = {
    "remove_waters": "Remove water molecules.",
    "remove_crystallization_aids": "Remove crystallization aids (X-ray only).",
    "remove_small_polymers": "Remove polymer chains of 3 or fewer residues.",
    "remove_fully_unknown_polymers": "Remove polymers made only of unknown residues.",
    "remove_clashing_chains": "Remove chains that clash with others.",
    "remove_non_CCD_atoms": "Remove atoms not in the CCD definition of their residue.",
    "remove_chains_with_CA_gaps": "Remove polymer chains with CA-CA gaps over 10 A.",
}


class CleanupSettings(BaseModel):
    """Per-step switches for structure cleanup; all on (OF3's cleanup) by default.

    Unresolved atoms, terminal-atom removal and hydrogen removal are not switchable: the
    model needs them (fixed atoms per residue type, full-length chains).
    """

    model_config = ConfigDict(extra="forbid")

    # Alternate conformers: "first" = atomworks' default, the first letter per chain (cheapest);
    # "clash_aware" = atomworks' random_clash_aware with a fixed seed, one conformer per residue
    # chosen so the kept conformers do not clash; "highest_occupancy" = OF3's choice, applied
    # before parsing (for direct comparison with OF3; warns).
    altloc: Literal["first", "clash_aware", "highest_occupancy"] = "first"
    remove_waters: bool = True
    convert_mse_to_met: bool = True
    remove_crystallization_aids: bool = True  # X-ray structures only, as in OF3
    remove_small_polymers: bool = True
    remove_fully_unknown_polymers: bool = True
    remove_clashing_chains: bool = True
    remove_non_ccd_atoms: bool = True  # also needs a CCD
    remove_chains_with_ca_gaps: bool = True
    prefilter_bonds: bool = True

    @classmethod
    def from_file(cls, path: Path) -> "CleanupSettings":
        """Load settings from a YAML or JSON file; unlisted steps keep their defaults.

        Args:
            path: YAML/JSON mapping of step name to its value (true/false; altloc a choice).

        Returns:
            The settings. Raises pydantic.ValidationError on unknown steps or bad values.
        """
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})

    def non_default(self) -> dict[str, bool | str]:
        """Steps whose value differs from the default.

        Returns:
            ``{step: value}`` for every switched step.
        """
        return {k: v for k, v in self.model_dump().items() if v != type(self).model_fields[k].default}

    def altloc_warning(self) -> str | None:
        """Why a non-default altloc choice is not recommended.

        Returns:
            The warning text for highest_occupancy, else None.
        """
        if self.altloc != "highest_occupancy":
            return None
        return (
            "altloc: highest_occupancy is not the recommended choice: it reproduces OpenFold3 for "
            "direct comparison but ignores clashes between the kept conformers. Use first (the "
            "default) or clash_aware."
        )

    def is_default(self) -> bool:
        """Whether every step is at its default.

        Returns:
            True if no step is switched.
        """
        return not self.non_default()


@dataclass(frozen=True)
class Removal:
    """Atoms one cleanup step removed."""

    step: str
    chain_iids: tuple[str, ...]
    n_atoms: int


class EmptyStructureError(Exception):
    """Cleanup removed every atom of a structure."""

    def __init__(self, pdb_id: str, removals: list[Removal]):
        super().__init__(f"{pdb_id}: no atoms left after cleanup ({[r.step for r in removals]})")
        self.pdb_id = pdb_id
        self.removals = removals


def _remove_terminal_atoms(atom_array: AtomArray) -> AtomArray:
    """Drop terminal atoms (OXT, OP3) of standard residues (OF3 structure formatting).

    Args:
        atom_array: Structure with a ``molecule_type_id`` annotation.

    Returns:
        The structure without those atoms.
    """
    # Bioassembly symmetry copies share chain_id; OF3 splits chains on it, so run on
    # chain_iid, then keep the surviving atoms of the original array.
    work = atom_array.copy()
    work.chain_id = work.chain_iid
    work.set_annotation("_row", np.arange(len(work)))
    kept_rows = remove_std_residue_terminal_atoms(work).get_annotation("_row")
    return atom_array[kept_rows]


def _remove_unresolved_covalent_ligand_atoms(atom_array: AtomArray) -> AtomArray:
    """Drop unresolved atoms of covalent ligands, which OF3 never adds (add_unresolved_atoms_within_residue).

    atomworks adds every missing CCD atom; for a ligand bonded to another chain, or made of
    several residues, the missing atom may be a leaving atom (7axf S6T: CL replaced by CYS SG).

    Args:
        atom_array: Structure with ``molecule_type_id`` and a bond list (after prefilter_bonds).

    Returns:
        The structure without NaN-coordinate atoms in covalent ligand chains.
    """
    work = atom_array.copy()
    work.chain_id = work.chain_iid
    covalent = np.isin(work.chain_iid, get_covalent_component_chain_ids(work))
    unresolved = np.isnan(atom_array.coord).any(axis=1)
    return atom_array[~(covalent & unresolved)]


def _apply_filter(atom_array: AtomArray, step: Callable[[AtomArray], AtomArray]) -> tuple[AtomArray, Removal | None]:
    """Run an OF3 atom-removing filter on the resolved atoms of an atomworks array.

    OF3 filters see only resolved atoms and split chains on chain_id, so run them on
    chain_iid; atoms they drop (and chains left with no resolved atoms) leave the full array.

    Args:
        atom_array: Structure with a ``molecule_type_id`` annotation.
        step: OF3 filter taking and returning an AtomArray.

    Returns:
        ``(filtered structure, Removal or None if nothing was removed)``.
    """
    work = atom_array.copy()
    work.chain_id = work.chain_iid
    work.set_annotation("_orig_idx", np.arange(len(work)))
    resolved = ~np.isnan(work.coord).any(axis=1)
    kept_idx = step(work[resolved]).get_annotation("_orig_idx")

    keep = np.ones(len(work), dtype=bool)
    keep[np.nonzero(resolved)[0][~np.isin(np.nonzero(resolved)[0], kept_idx)]] = False
    # A chain whose resolved atoms are all gone loses its unresolved atoms too.
    lost = np.unique(work.chain_iid[resolved & ~keep])
    still_resolved = np.unique(work.chain_iid[resolved & keep])
    keep[np.isin(work.chain_iid, np.setdiff1d(lost, still_resolved))] = False

    if keep.all():
        return atom_array, None
    return atom_array[keep], Removal(
        step=getattr(step, "step_name", None) or getattr(step, "func", step).__name__,
        chain_iids=tuple(str(c) for c in np.unique(work.chain_iid[~keep])),
        n_atoms=int((~keep).sum()),
    )


def _ccd_residues(res_names: np.ndarray, ccd: CIFFile) -> set[str]:
    """Residue names of ``res_names`` that have a CCD entry.

    Args:
        res_names: Residue names.
        ccd: Parsed CCD.

    Returns:
        The names found in the CCD.
    """
    return {str(n) for n in np.unique(res_names) if n in ccd}


def _remove_non_ccd_atoms(atom_array: AtomArray, ccd: CIFFile) -> AtomArray:
    """OF3's remove_non_CCD_atoms, restricted to residues the CCD has an entry for.

    OF3's function raises KeyError on a residue missing from the CCD (custom ligands);
    those residues are kept as they are. Needs the ``_orig_idx`` annotation _apply_filter sets.

    Args:
        atom_array: Structure (resolved atoms) with ``_orig_idx`` and ``molecule_type_id``.
        ccd: Parsed CCD.

    Returns:
        The structure without non-CCD atoms in CCD residues.
    """
    known = np.isin(atom_array.res_name, list(_ccd_residues(atom_array.res_name, ccd)))
    kept_idx = remove_non_CCD_atoms(atom_array[known], ccd=ccd).get_annotation("_orig_idx")
    orig_idx = atom_array.get_annotation("_orig_idx")
    return atom_array[np.isin(orig_idx, kept_idx) | ~known]


def _with_step_name(step: partial, name: str) -> partial:
    """Name a filter for Removal records and logs (wrapped OF3 functions keep OF3's name).

    Args:
        step: The filter.
        name: Step name to report.

    Returns:
        The same filter, carrying ``step_name``.
    """
    step.step_name = name
    return step


def _nonstandard_polymer_links(atom_array: AtomArray, bonds: np.ndarray) -> np.ndarray:
    """Mask of bonds linking consecutive residues of one polymer chain where either residue is non-standard.

    Args:
        atom_array: Structure with ``molecule_type_id`` and ``chain_iid`` annotations.
        bonds: Bond array, atom index pairs in the first two columns.

    Returns:
        One bool per bond.
    """
    a, b = bonds[:, 0], bonds[:, 1]
    polymer = [MoleculeType.PROTEIN, MoleculeType.DNA, MoleculeType.RNA]
    is_polymer = np.isin(atom_array.molecule_type_id[a], polymer) & np.isin(atom_array.molecule_type_id[b], polymer)
    is_consecutive = (atom_array.chain_iid[a] == atom_array.chain_iid[b]) & (
        np.abs(atom_array.res_id[a] - atom_array.res_id[b]) == 1
    )
    is_nonstandard = ~np.isin(atom_array.res_name[a], STANDARD_RESIDUES_3) | ~np.isin(
        atom_array.res_name[b], STANDARD_RESIDUES_3
    )
    return is_polymer & is_consecutive & is_nonstandard


def _filter_bonds(atom_array: AtomArray) -> AtomArray:
    """Drop OF3's disallowed bonds (inter-chain dative/polymer links, bonds > 2.4 A).

    One deviation from OF3: a link between consecutive polymer residues where either is
    non-standard (e.g. 3ogo's chromophore CRO) is kept at any length, so such a residue is
    not left bonded at one end only when the deposited geometry stretches a link.

    Args:
        atom_array: Structure with a ``molecule_type_id`` annotation.

    Returns:
        The structure with the same atoms and a filtered bond list.
    """
    work = atom_array.copy()
    work.chain_id = work.chain_iid
    bonds = prefilter_bonds(work, **{**_BOND_FILTER_ARGS, "remove_longer_than": None}).bonds.as_array()
    lengths = index_distance(atom_array, bonds[:, :2])
    too_long = (lengths > _BOND_FILTER_ARGS["remove_longer_than"]) & ~np.isnan(lengths)
    keep = ~too_long | _nonstandard_polymer_links(atom_array, bonds)
    atom_array.bonds = BondList(len(atom_array), bonds[keep])
    return atom_array


def _filters(
    is_xray: bool, ccd: CIFFile | None, settings: CleanupSettings
) -> list[Callable[[AtomArray], AtomArray]]:
    """OF3's atom-removing filters that are switched on, in OF3's order.

    Args:
        is_xray: Whether the structure is X-ray; crystallization aids are removed only then.
        ccd: Parsed CCD; the non-CCD-atom filter is skipped when None.
        settings: Which steps are on.

    Returns:
        The filters to apply, each named after its OF3 function.
    """
    candidates = [
        (settings.remove_waters, remove_waters),
        (settings.remove_crystallization_aids and is_xray, remove_crystallization_aids),
        (settings.remove_small_polymers, partial(remove_small_polymers, max_residues=3)),
        (settings.remove_fully_unknown_polymers, remove_fully_unknown_polymers),
        (
            settings.remove_clashing_chains,
            partial(remove_clashing_chains, clash_distance=1.7, clash_percentage=0.3),
        ),
        (settings.remove_non_ccd_atoms and ccd is not None, _with_step_name(partial(_remove_non_ccd_atoms, ccd=ccd), "remove_non_CCD_atoms")),
        (
            settings.remove_chains_with_ca_gaps,
            partial(remove_chains_with_CA_gaps, distance_threshold=10.0),
        ),
    ]
    return [step for enabled, step in candidates if enabled]


def clean_structure(
    atom_array: AtomArray,
    source_cif: Path,
    pdb_id: str,
    ccd: CIFFile | None,
    settings: CleanupSettings | None = None,
    recorder: StepRecorder | None = None,
) -> tuple[AtomArray, list[Removal]]:
    """Apply OF3's cleanup steps to an atomworks-parsed structure.

    Args:
        atom_array: Structure from atomworks' parse.
        source_cif: Source mmCIF, read for molecule types and the experimental method.
        pdb_id: Structure ID, for logging and errors.
        ccd: Parsed Chemical Component Dictionary; None skips the non-CCD-atom filter.
        settings: Which steps to run; None means OF3's cleanup (all on).
        recorder: Step recorder; None records nothing.

    Returns:
        ``(cleaned structure, removals)``; removals lists the filtered-out atoms.

    Raises:
        ValueError: If the source CIF has no ``chem_comp`` category or ``_exptl.method``.
        EmptyStructureError: If the filters remove every atom.
    """
    cif = CIFFile.read(source_cif)
    if "chem_comp" not in cif.block:
        raise ValueError(f"{pdb_id}: {source_cif} has no chem_comp category; cannot assign molecule types")
    if "exptl" not in cif.block or "method" not in cif.block["exptl"]:
        raise ValueError(f"{pdb_id}: {source_cif} has no _exptl.method; cannot tell whether it is X-ray")
    settings = settings or CleanupSettings()
    arr = atom_array.copy()
    # Molecule types first, as in OF3 (assigned at parse, MSE typed as MSE).
    assign_molecule_type_ids(arr, cif)
    if settings.convert_mse_to_met:
        is_mse = arr.res_name == "MSE"
        n_mse = len(set(zip(arr.chain_iid[is_mse], arr.res_id[is_mse])))
        convert_MSE_to_MET(arr)
        if n_mse:
            logger.info(f"{pdb_id}: convert_MSE_to_MET converted {n_mse} residue(s)")
        if recorder is not None:
            recorder.record("convert_MSE_to_MET", "Convert selenomethionine (MSE) to methionine.", arr)
    removals: list[Removal] = []
    if settings.remove_non_ccd_atoms and ccd is not None:
        unknown = sorted(set(map(str, np.unique(arr.res_name))) - _ccd_residues(arr.res_name, ccd))
        if unknown:
            logger.info(
                f"{pdb_id}: remove_non_CCD_atoms skipped residue(s) not in the CCD: "
                f"{', '.join(unknown)}"
            )
    for step in _filters(get_experimental_method(cif.block) == _XRAY, ccd, settings):
        arr, removal = _apply_filter(arr, step)
        if recorder is not None:
            name = getattr(step, "step_name", None) or getattr(
                getattr(step, "func", step), "__name__", repr(step)
            )
            recorder.record(name, _STEP_DESCRIPTIONS.get(name, name), arr)
        if removal is None:
            continue
        removals.append(removal)
        logger.info(f"{pdb_id}: {removal.step} removed {removal.chain_iids} ({removal.n_atoms} atoms)")
        if len(arr) == 0:
            raise EmptyStructureError(pdb_id, removals)
    if settings.prefilter_bonds:
        n_bonds = len(arr.bonds.as_array())
        arr = _filter_bonds(arr)
        if recorder is not None:
            recorder.record(
                "prefilter_bonds",
                "Drop disallowed bonds (inter-chain dative/polymer links, longer than 2.4 A).",
                arr,
                bonds_removed=n_bonds - len(arr.bonds.as_array()),
            )
    arr = _remove_unresolved_covalent_ligand_atoms(arr)
    if recorder is not None:
        recorder.record(
            "remove_unresolved_covalent_ligand_atoms",
            "Remove unresolved atoms of covalent ligands (OF3 adds none there).",
            arr,
        )
    n_before = len(arr)
    arr = _remove_terminal_atoms(arr)
    if recorder is not None:
        recorder.record("remove_terminal_atoms", "Remove OXT/OP3 terminal atoms of standard residues.", arr)
    arr.del_annotation("molecule_type_id")
    n_nan = int(np.isnan(arr.coord).any(axis=1).sum())
    logger.info(
        f"{pdb_id}: cleanup done, {len(arr)} atoms ({n_nan} unresolved (NaN coordinates)), "
        f"{n_before - len(arr)} terminal atom(s) removed"
    )
    return arr, removals
