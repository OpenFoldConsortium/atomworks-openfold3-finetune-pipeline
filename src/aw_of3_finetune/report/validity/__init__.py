"""Reference-free validity checks of a pipeline output (nothing here may import the parity side)."""

import traceback
from collections.abc import Callable

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity.audit import CHECK as CLEANUP_AUDIT, check_cleanup_audit
from aw_of3_finetune.report.validity.bonds import (
    BOND_LENGTHS,
    RESIDUE_BONDS,
    check_bond_lengths,
    check_residue_bonds,
)
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.report.validity.ligands import (
    LIGAND_CHEMISTRY,
    LIGAND_CONTACTS,
    LIGAND_GEOMETRY,
    LIGAND_IDENTITY,
    check_ligand_chemistry,
    check_ligand_contacts,
    check_ligand_geometry,
    check_ligand_identity,
)
from aw_of3_finetune.report.validity.ligand_smiles import LIGAND_SMILES, check_ligand_smiles
from aw_of3_finetune.report.validity.links import COVALENT_LINKS, check_covalent_links
from aw_of3_finetune.report.validity.tracks import CHECK as CHAIN_PLACEMENT, check_chain_placement

ENTRY_MISSING = "validity.entry_missing"

VALIDITY_CHECKS: dict[str, Callable[[ValidityInputs, str], list[Finding]]] = {
    CLEANUP_AUDIT: check_cleanup_audit,
    RESIDUE_BONDS: check_residue_bonds,
    BOND_LENGTHS: check_bond_lengths,
    LIGAND_CHEMISTRY: check_ligand_chemistry,
    LIGAND_IDENTITY: check_ligand_identity,
    LIGAND_GEOMETRY: check_ligand_geometry,
    LIGAND_CONTACTS: check_ligand_contacts,
    CHAIN_PLACEMENT: check_chain_placement,
    COVALENT_LINKS: check_covalent_links,
    LIGAND_SMILES: check_ligand_smiles,
}


def entry_missing_findings(inputs: ValidityInputs) -> list[Finding]:
    """One unexplained finding per validation.json entries_in id that is not in the manifest.

    Args:
        inputs: The pipeline output to validate.

    Returns:
        The findings, in entries_in order; the detail names the E findings of an entry the pipeline dropped.
    """
    report = inputs.validation_report()
    listed = set(inputs.entries())
    dropped = set(report.get("dropped", []))
    findings = []
    for pdb_id in report.get("entries_in", []):
        if pdb_id in listed:
            continue
        if pdb_id in dropped:
            errors = [f"{item['check']} {item['message']}" for item in inputs.validation_findings(pdb_id)
                      if item["severity"] == "E"]
            detail = f"dropped by the pipeline's validate stage: {', '.join(errors) or 'no E finding recorded'}"
        else:
            detail = "in validation.json entries_in but not in the manifest"
        findings.append(Finding(ENTRY_MISSING, pdb_id, Status.UNEXPLAINED, detail, {"dropped": pdb_id in dropped}))
    return findings


def run_validity(inputs: ValidityInputs) -> list[Finding]:
    """Run every validity check on every manifest entry, then report input entries the output lacks.

    A check that raises becomes one unexplained finding for that entry.

    Args:
        inputs: The pipeline output to validate.

    Returns:
        All findings, entry by entry in manifest order, then the validity.entry_missing ones.
    """
    findings = []
    for pdb_id in inputs.entries():
        for check, function in VALIDITY_CHECKS.items():
            try:
                findings += function(inputs, pdb_id)
            except Exception as error:  # noqa: BLE001 — one bad check must not stop the others
                detail = f"check failed: {type(error).__name__}: {error}"
                data = {"error": True, "traceback": traceback.format_exc()}
                findings.append(Finding(check, pdb_id, Status.UNEXPLAINED, detail, data))
    return findings + entry_missing_findings(inputs)
