import json
import shutil

import pytest
from click.testing import CliRunner

from aw_of3_finetune.cli import prepare_inputs
from aw_of3_finetune.prepare_inputs import LIGANDS_FILE
from aw_of3_finetune.report.findings import Status
from aw_of3_finetune.report.origin import PROCESSING, classify_origins
from aw_of3_finetune.report.validity import VALIDITY_CHECKS, ValidityInputs
from aw_of3_finetune.report.validity.data import CHECK_COLUMNS
from aw_of3_finetune.report.validity.ligand_smiles import LIGAND_SMILES, check_ligand_smiles
from ligand_helpers import (
    ACID_SMILES,
    INDOLE_SMILES,
    write_pxr_constructs,
    X00242_PDB,
    X00242_RESTRAINT,
    X00358_PDB,
    X00358_RESTRAINT,
)
from validate_helpers import BuiltDataset, build_dataset

GEM_DIOL_SMILES = "OC(O)Cc1cc(Cl)cc(Cl)c1"
ACID_ID = "x00242-1"


@pytest.fixture(scope="module")
def prepared(ccd_path, tmp_path_factory) -> BuiltDataset:
    """prepare-inputs on both fixture structures, then finetune-prep on its output dir."""
    root = tmp_path_factory.mktemp("ligand_smiles")
    input_dir = root / "inputs"
    input_dir.mkdir()
    shutil.copy(X00242_PDB, input_dir)
    shutil.copy(X00358_PDB, input_dir)
    table = root / "ligands.csv"
    table.write_text(
        "structure_id,res_name,smiles,restraint_cif\n"
        f"x00242-1,LIG,{ACID_SMILES},{X00242_RESTRAINT}\n"
        f"x00358-1,LIG,{INDOLE_SMILES},{X00358_RESTRAINT}\n"
    )
    prepared_dir = root / "prepared"
    args = ["--input-dir", str(input_dir), "--ligands", str(table), "--output-dir", str(prepared_dir),
            "--ccd-path", str(ccd_path), "--constructs", str(write_pxr_constructs(root / "constructs.fasta"))]  # fmt: skip
    result = CliRunner().invoke(prepare_inputs, args)
    assert result.exit_code == 0, result.output
    built = build_dataset(root / "built", sorted(prepared_dir.glob("*.cif")), ccd_path)
    shutil.copy(prepared_dir / LIGANDS_FILE, built.cif_dir)
    return built


def _inputs(built: BuiltDataset, ccd_path) -> ValidityInputs:
    return ValidityInputs(built.output_dir, built.cif_dir, ccd_path)


def _acid_mol_id(inputs: ValidityInputs) -> str:
    [finding] = check_ligand_smiles(inputs, ACID_ID)
    return inputs.cache_entry(ACID_ID)["chains"][finding.data["chain"]]["reference_mol_id"]


def _copy(built: BuiltDataset, dest) -> BuiltDataset:
    shutil.copytree(built.output_dir, dest / "output")
    shutil.copytree(built.cif_dir, dest / "cif_input")
    return BuiltDataset(dest / "output", dest / "cif_input", built.msa_dir)


def test_check_is_registered_with_a_grid_column():
    assert VALIDITY_CHECKS[LIGAND_SMILES] is check_ligand_smiles
    assert CHECK_COLUMNS[LIGAND_SMILES] == "ligand_smiles"


def test_matching_ligands_are_match(prepared, ccd_path):
    for pdb_id in ("x00242-1", "x00358-1"):
        findings = check_ligand_smiles(_inputs(prepared, ccd_path), pdb_id)

        actual_statuses = {f.status for f in findings}
        assert findings
        assert actual_statuses == {Status.MATCH}


def test_gem_diol_reference_is_unexplained_and_processing(prepared, ccd_path, tmp_path):
    from rdkit import Chem

    built = _copy(prepared, tmp_path)
    inputs = _inputs(built, ccd_path)
    mol_id = _acid_mol_id(inputs)
    writer = Chem.SDWriter(str(built.output_dir / "reference_mols" / f"{mol_id}.sdf"))
    writer.write(Chem.MolFromSmiles(GEM_DIOL_SMILES))
    writer.close()

    [finding] = check_ligand_smiles(inputs, ACID_ID)

    assert finding.status == Status.UNEXPLAINED
    expected_detail = f"reference {Chem.CanonSmiles(GEM_DIOL_SMILES)} != given {ACID_SMILES}"
    assert expected_detail in finding.detail
    [classified] = classify_origins(inputs, [finding])
    assert classified.data["origin"] == PROCESSING


def test_missing_reference_sdf_is_unexplained(prepared, ccd_path, tmp_path):
    built = _copy(prepared, tmp_path)
    inputs = _inputs(built, ccd_path)
    mol_id = _acid_mol_id(inputs)
    (built.output_dir / "reference_mols" / f"{mol_id}.sdf").unlink()

    [finding] = check_ligand_smiles(inputs, ACID_ID)

    assert finding.status == Status.UNEXPLAINED
    assert "no reference SDF" in finding.detail


def test_structure_ids_compare_case_insensitively(prepared, ccd_path, tmp_path):
    built = _copy(prepared, tmp_path)
    path = built.cif_dir / LIGANDS_FILE
    table = json.loads(path.read_text())
    for record in table.values():
        record["structures"] = [stem.upper() for stem in record["structures"]]
    path.write_text(json.dumps(table))

    findings = check_ligand_smiles(_inputs(built, ccd_path), ACID_ID)

    assert [f.status for f in findings] == [Status.MATCH]


def test_no_ligands_file_means_no_findings(prepared, ccd_path, tmp_path):
    built = _copy(prepared, tmp_path)
    (built.cif_dir / LIGANDS_FILE).unlink()

    assert check_ligand_smiles(_inputs(built, ccd_path), ACID_ID) == []
