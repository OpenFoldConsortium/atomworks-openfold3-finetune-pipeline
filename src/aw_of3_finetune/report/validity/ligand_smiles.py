"""Ligands defined by prepare-inputs must have a reference molecule that is the SMILES the user gave."""

import json

from rdkit import Chem

from aw_of3_finetune.prepare_inputs import LIGANDS_FILE
from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.report.validity.ligands import _label, _ligand_chains

LIGAND_SMILES = "validity.ligand_smiles"


def _sdf_smiles(path) -> str:
    """Stereo-free canonical SMILES of the first molecule of an SDF (hydrogens removed)."""
    mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)))
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)


def check_ligand_smiles(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """Each ligand chain defined in the cif dir's ligands.json: its reference SDF is the given SMILES.

    Args:
        inputs: The pipeline output to validate.
        pdb_id: Structure id (compared case-insensitively with the table's structure stems).

    Returns:
        One finding per table ligand chain; empty if the cif dir has no ligands.json.
    """
    table_path = inputs.cif_dir / LIGANDS_FILE
    if not table_path.exists():
        return []
    table = json.loads(table_path.read_text())
    atoms = inputs.final_atoms(pdb_id)
    chains = inputs.cache_entry(pdb_id)["chains"]
    findings = []
    for chain in _ligand_chains(inputs, pdb_id):
        res_name, label = _label(atoms, chain)
        record = table.get(res_name)
        if record is None or pdb_id.lower() not in {stem.lower() for stem in record["structures"]}:
            continue
        sdf = inputs.out_dir / "reference_mols" / f"{chains[chain]['reference_mol_id']}.sdf"
        data = {"chain": chain, "res_name": res_name, "rows": []}
        if not sdf.exists():
            findings.append(Finding(LIGAND_SMILES, pdb_id, Status.UNEXPLAINED, f"{label}: no reference SDF", data))
            continue
        actual, expected = _sdf_smiles(sdf), Chem.CanonSmiles(record["smiles"], useChiral=False)
        if actual == expected:
            findings.append(Finding(LIGAND_SMILES, pdb_id, Status.MATCH, f"{label}: reference is the given SMILES", data))
        else:
            detail = f"{label}: reference {actual} != given {record['smiles']}"
            findings.append(Finding(LIGAND_SMILES, pdb_id, Status.UNEXPLAINED, detail, data))
    return findings
