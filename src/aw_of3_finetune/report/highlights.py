"""The atoms of each input or processing finding, for the 3D view to mark: one issue per bad bond or ligand finding."""

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.origin import INPUT, PROCESSING, bond_atoms, ligand_atoms
from aw_of3_finetune.report.validity.bonds import BOND_LENGTHS
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.report.validity.ligands import LIGAND_CHEMISTRY, LIGAND_CONTACTS, LIGAND_GEOMETRY, LIGAND_IDENTITY

_LIGAND_CHECKS = (LIGAND_CHEMISTRY, LIGAND_IDENTITY, LIGAND_GEOMETRY, LIGAND_CONTACTS)
_XYZ_DECIMALS = 2


def issue_highlights(inputs: ValidityInputs, pdb_id: str, findings: list[Finding]) -> list[dict]:
    """The issues of one entry with the final coordinates of the atoms each involves.

    Args:
        inputs: The pipeline output.
        pdb_id: Structure id.
        findings: Findings after classify_origins (other entries' findings are ignored).

    Returns:
        {"origin", "check", "label", "xyz"} per bad bond (label as in the finding's detail) and per
        ligand finding (label = its detail); findings without atoms, matches and not-checked ones are left out.
    """
    flagged = [
        finding for finding in findings
        if finding.entry == pdb_id and finding.status != Status.MATCH
        and finding.data.get("origin") in (INPUT, PROCESSING) and finding.check in (BOND_LENGTHS, *_LIGAND_CHECKS)
    ]
    if not flagged:
        return []
    atoms = inputs.final_atoms(pdb_id)

    def issue(finding: Finding, label: str, positions: list[int] | None) -> list[dict]:
        if not positions:
            return []
        xyz = [[round(float(v), _XYZ_DECIMALS) for v in atoms.coord[p]] for p in positions]
        return [{"origin": finding.data["origin"], "check": finding.check, "label": label, "xyz": xyz}]

    issues = []
    for finding in flagged:
        if finding.check == BOND_LENGTHS:
            # check_bond_lengths joins one detail part per row, in row order.
            for row, label in zip(finding.data.get("rows", []), finding.detail.split("; ")):
                issues += issue(finding, label, bond_atoms(atoms, row))
        else:
            issues += issue(finding, finding.detail, ligand_atoms(inputs, pdb_id, atoms, finding))
    return issues
