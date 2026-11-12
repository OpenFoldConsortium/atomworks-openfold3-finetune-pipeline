"""Where a finding comes from: the user's input CIF, the pipeline's processing, or a check that could not run.

A geometry finding is in the input when every atom it involves still has its input coordinates (identity
copies) or its operator applied to its input atom (assembly copies); the pipeline never moves atoms on
purpose. A residue-bond finding is in the input when each differing residue has exactly the bonds
the input CIF's own chem_comp_bond defines for it. Findings about pipeline logic are processing findings. See the user-report spec, "Input vs processing".
"""

from dataclasses import replace

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from openfold3.core.data.primitives.structure.metadata import get_release_date

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity import VALIDITY_CHECKS
from aw_of3_finetune.report.validity.bonds import BOND_LENGTHS, RESIDUE_BONDS, bonds_follow_input_definitions
from aw_of3_finetune.report.validity.copies import find_copies, operators
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.report.validity.ligands import (
    COULD_NOT_RUN,
    LIGAND_CHEMISTRY,
    LIGAND_CONTACTS,
    LIGAND_GEOMETRY,
    LIGAND_IDENTITY,
    _nearby_protein,
)
from aw_of3_finetune.steplog import read_input_atoms
from aw_of3_finetune.validate.check_ids import RELEASE_DATE as RELEASE_DATE_CHECK

INPUT = "input"
PROCESSING = "processing"
NOT_CHECKED = "not_checked"

_REASONS = {INPUT: "already in the input CIF", NOT_CHECKED: "check could not run"}
_LIGAND_CHECKS = (LIGAND_CHEMISTRY, LIGAND_IDENTITY, LIGAND_GEOMETRY, LIGAND_CONTACTS)


def input_positions(inputs: ValidityInputs, pdb_id: str) -> set[int]:
    """Final-atom positions at their input atom's coordinates, or at an operator applied to them.

    Args:
        inputs: The pipeline output.
        pdb_id: Structure id.

    Returns:
        Positions of identity-pair atoms and of matched assembly-copy atoms.
    """
    cif_path = inputs.input_cif(pdb_id)
    copies = find_copies(read_input_atoms(cif_path), inputs.final_atoms(pdb_id), operators(cif_path))
    return {position for _, position in copies.pairs} | set(copies.source_of)


def _resolved(atoms: AtomArray) -> np.ndarray:
    """True per atom with coordinates."""
    return ~np.isnan(atoms.coord).any(axis=1)


def bond_atoms(atoms: AtomArray, row: dict) -> list[int] | None:
    """Final positions of a bond row's two atoms.

    Args:
        atoms: The final atoms.
        row: One row of a bond-lengths finding.

    Returns:
        The positions of both atoms; None if either cannot be found.
    """
    resolved = _resolved(atoms)
    copy = atoms.label_asym_id.astype(str) == row.get("label_asym_id", row["chain"])
    residue = copy & (atoms.res_id == row["res_id"]) & (atoms.res_name.astype(str) == row["res_name"])
    first = np.flatnonzero(residue & (atoms.atom_name.astype(str) == row["atom_1"]) & resolved)
    if ":" in row["atom_2"]:
        chain, res, name = row["atom_2"].split(":")
        res_labels = np.char.add(atoms.res_name.astype(str), atoms.res_id.astype(str))
        other = (atoms.chain_id.astype(str) == chain) & (res_labels == res)
    else:
        name, other = row["atom_2"], residue
    second = np.flatnonzero(other & (atoms.atom_name.astype(str) == name) & resolved)
    if len(first) == 0 or len(second) == 0:
        return None
    return [*first.tolist(), *second.tolist()]


def ligand_atoms(inputs: ValidityInputs, pdb_id: str, atoms: AtomArray, finding: Finding) -> list[int] | None:
    """Resolved atoms of a ligand finding's chain, plus the compared protein atoms for contacts.

    Args:
        inputs: The pipeline output.
        pdb_id: Structure id.
        atoms: The entry's final atoms.
        finding: A ligand finding whose rows name one chain.

    Returns:
        Final positions; None if the rows name no single chain or it has no resolved atoms.
    """
    chains = {row["chain"] for row in finding.data.get("rows", [])}
    if len(chains) != 1:
        return None
    [chain] = chains
    involved = np.flatnonzero((atoms.chain_id.astype(str) == chain) & _resolved(atoms)).tolist()
    if finding.check == LIGAND_CONTACTS:
        involved += _nearby_protein(inputs, pdb_id, atoms, chain).tolist()
    return involved or None


def _involved_atoms(inputs: ValidityInputs, finding: Finding, atoms: AtomArray) -> list[int] | None:
    """Final positions a geometry finding involves; None if they cannot be resolved."""
    if finding.check == BOND_LENGTHS:
        involved = []
        for row in finding.data.get("rows", []):
            positions = bond_atoms(atoms, row)
            if positions is None:
                return None
            involved += positions
        return involved or None
    return ligand_atoms(inputs, finding.entry, atoms, finding)


def _origin(inputs: ValidityInputs, finding: Finding, cache: dict) -> str:
    """The origin of one non-matching finding."""
    if finding.data.get("error") and finding.check in VALIDITY_CHECKS:
        return NOT_CHECKED
    failing = [row for row in finding.data.get("rows", []) if not row.get("passed", True)]
    if finding.check in _LIGAND_CHECKS and failing and all(row.get("reason") == COULD_NOT_RUN for row in failing):
        return NOT_CHECKED
    if finding.check == LIGAND_CHEMISTRY:
        return INPUT
    if finding.check == LIGAND_IDENTITY:
        input_failed = any(row.get("reference", "ccd") == INPUT and row.get("reason") != COULD_NOT_RUN for row in failing)
        return PROCESSING if input_failed else INPUT  # an input comparison that could not run is never red
    if finding.check == RESIDUE_BONDS and finding.entry is not None:
        return INPUT if bonds_follow_input_definitions(inputs, finding.entry) else PROCESSING
    if finding.check in (BOND_LENGTHS, LIGAND_GEOMETRY, LIGAND_CONTACTS) and finding.entry is not None:
        if finding.entry not in cache:
            cache[finding.entry] = (inputs.final_atoms(finding.entry), input_positions(inputs, finding.entry))
        atoms, at_input = cache[finding.entry]
        involved = _involved_atoms(inputs, finding, atoms)
        return INPUT if involved is not None and set(involved) <= at_input else PROCESSING
    return PROCESSING


def _input_release_date(inputs: ValidityInputs, pdb_id: str) -> str | None:
    """The input CIF's release date as OF3 reads it (earliest revision date); None if it has none."""
    block = pdbx.CIFFile.read(inputs.input_cif(pdb_id)).block
    if "pdbx_audit_revision_history" not in block:
        return None
    return get_release_date(block).date().isoformat()


def v_check_origin(inputs: ValidityInputs, item: dict) -> str:
    """Origin of one of the pipeline's own validate findings (a validation.json item).

    Args:
        inputs: The pipeline output.
        item: A validation.json finding (check, severity, entry, message, detail).

    Returns:
        INPUT for a V17 (real release date) warning whose date is the input CIF's own (e.g. a 1970-01-01
        placeholder there); PROCESSING for every other finding, or when the input cannot be read.
    """
    if item["check"] != RELEASE_DATE_CHECK or item["entry"] is None:
        return PROCESSING
    try:
        released = _input_release_date(inputs, item["entry"])
    except Exception:  # noqa: BLE001 — an unreadable input keeps the finding red
        return PROCESSING
    return INPUT if released is not None and item["message"].endswith(released) else PROCESSING


def classify_origins(inputs: ValidityInputs, findings: list[Finding]) -> list[Finding]:
    """Mark each non-matching finding with its origin; input and not-checked ones become expected.

    Args:
        inputs: The pipeline output the findings are about.
        findings: Validity findings.

    Returns:
        New findings in the same order; matches unchanged, others with data["origin"] (and label, reason).
        A finding whose origin cannot be determined (e.g. an unreadable input CIF) is not checked.
    """
    cache: dict = {}
    result = []
    for finding in findings:
        if finding.status == Status.MATCH:
            result.append(finding)
            continue
        try:
            origin, reason = _origin(inputs, finding, cache), None
        except Exception as error:  # noqa: BLE001 — one unreadable input must not lose the report
            origin, reason = NOT_CHECKED, f"origin could not be determined: {type(error).__name__}: {error}"
        data = finding.data | {"origin": origin}
        if origin == PROCESSING:
            result.append(replace(finding, status=Status.UNEXPLAINED, data=data))
        else:
            data["label"] = origin
            result.append(replace(finding, status=Status.EXPECTED, data=data, reason=reason or _REASONS[origin]))
    return result
