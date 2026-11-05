"""The prepare-inputs command: a directory of in-house structures to a finetune-prep cif dir."""

import json
import shutil
from pathlib import Path

from biotite.structure.io.pdbx import CIFFile
from click.testing import CliRunner
from rdkit import Chem

from aw_of3_finetune.cli import finetune_prep, prepare_inputs
from aw_of3_finetune.ligand_inputs import LigandNamer, LigandRow, ccd_codes, read_ligand_table
from aw_of3_finetune.prepare_inputs import (
    DEFAULT_RELEASE_DATE,
    LIGANDS_FILE,
    PREPARE_DROPPED_FILE,
    PREPARE_RECORD_FILE,
    _read_input,
    _row_definition,
    prepare_directory,
    prepare_structure,
    read_constructs,
)
from ligand_helpers import (
    ACID_SMILES,
    INDOLE_SMILES,
    PXR_CONSTRUCT,
    PXR_CONSTRUCTS,
    X00242_PDB,
    X00242_RESTRAINT,
    X00358_PDB,
    X00358_RESTRAINT,
    reference_smiles_by_chain,
    write_pxr_constructs,
)


def _write_inputs(tmp_path: Path, acid_smiles: str = ACID_SMILES) -> tuple[Path, Path]:
    """An input dir with both fixture .pdb files and a ligand table; returns (dir, table)."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copy(X00242_PDB, input_dir)
    shutil.copy(X00358_PDB, input_dir)
    table = tmp_path / "ligands.csv"
    table.write_text(
        "structure_id,res_name,smiles,restraint_cif\n"
        f"x00242-1,LIG,{acid_smiles},{X00242_RESTRAINT}\n"
        f"x00358-1,LIG,{INDOLE_SMILES},{X00358_RESTRAINT}\n"
    )
    return input_dir, table


def _run(input_dir: Path, table: Path, out: Path, ccd_path: Path, *extra: str):
    args = [
        "--input-dir", str(input_dir), "--ligands", str(table),
        "--output-dir", str(out), "--ccd-path", str(ccd_path),
        "--constructs", str(write_pxr_constructs(out.parent / "constructs.fasta")), *extra,
    ]  # fmt: skip
    return CliRunner().invoke(prepare_inputs, args)


def test_writes_one_cif_per_input_and_the_records(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    out = tmp_path / "out"

    result = _run(input_dir, table, out, ccd_path)

    assert result.exit_code == 0, result.output
    expected_cifs = {"x00242-1.cif", "x00358-1.cif"}
    assert {p.name for p in out.glob("*.cif")} == expected_cifs
    actual_record = json.loads((out / PREPARE_RECORD_FILE).read_text())
    expected_record = {
        "release_date": DEFAULT_RELEASE_DATE,
        "release_date_defaulted": ["x00242-1", "x00358-1"],
        "inputs": ["x00242-1", "x00358-1"],
    }
    assert actual_record == expected_record
    assert json.loads((out / PREPARE_DROPPED_FILE).read_text()) == {}
    actual_ligands = json.loads((out / LIGANDS_FILE).read_text())
    actual_structures = sorted(s for v in actual_ligands.values() for s in v["structures"])
    assert actual_structures == ["x00242-1", "x00358-1"]
    assert "prepared 2, dropped 0 (see prepare_dropped.json)" in result.output


def test_release_date_option_is_recorded(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)

    result = _run(input_dir, table, tmp_path / "out", ccd_path, "--release-date", "2025-05-05")

    assert result.exit_code == 0, result.output
    actual = json.loads((tmp_path / "out" / PREPARE_RECORD_FILE).read_text())["release_date"]
    assert actual == "2025-05-05"


def test_failing_structure_is_dropped_with_reason_and_others_continue(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path, acid_smiles="c1ccccc1")
    out = tmp_path / "out"

    result = _run(input_dir, table, out, ccd_path)

    assert result.exit_code == 0, result.output
    actual_dropped = json.loads((out / PREPARE_DROPPED_FILE).read_text())
    assert set(actual_dropped) == {"x00242-1"}
    assert actual_dropped["x00242-1"].startswith("LigandMatchError: ")
    assert "\n" not in actual_dropped["x00242-1"]
    assert (out / "x00358-1.cif").exists()
    assert not (out / "x00242-1.cif").exists()
    actual_inputs = json.loads((out / PREPARE_RECORD_FILE).read_text())["inputs"]
    assert actual_inputs == ["x00358-1"]
    assert "prepared 1, dropped 1" in result.output


def test_unreadable_structure_is_dropped_not_fatal(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    (input_dir / "broken.pdb").write_text("not a pdb\n")
    out = tmp_path / "out"

    result = _run(input_dir, table, out, ccd_path)

    assert result.exit_code == 0, result.output
    assert set(json.loads((out / PREPARE_DROPPED_FILE).read_text())) == {"broken"}
    assert (out / "x00242-1.cif").exists()


def test_unknown_structure_id_in_table_is_a_usage_error(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    with table.open("a") as handle:
        handle.write(f"nosuch,LIG,{ACID_SMILES},\n")

    result = _run(input_dir, table, tmp_path / "out", ccd_path)

    assert result.exit_code == 2
    assert "nosuch" in result.output


def test_stem_present_as_pdb_and_cif_is_a_usage_error(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    (input_dir / "x00242-1.cif").write_text("data_x\n")

    result = _run(input_dir, table, tmp_path / "out", ccd_path)

    assert result.exit_code == 2
    assert "x00242-1" in result.output


def test_invalid_ligand_table_is_a_usage_error(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    table.write_text("structure_id,res_name,smiles\nx00242-1,LIG,\n")

    result = _run(input_dir, table, tmp_path / "out", ccd_path)

    assert result.exit_code == 2
    assert "smiles" in result.output


def test_ccd_collision_is_a_hard_error(tmp_path, ccd_path, monkeypatch):
    input_dir, table = _write_inputs(tmp_path)
    ccd = CIFFile.read(ccd_path)
    # Make every code the namer would pick a "CCD code" by listing the acid's own code.
    acid_code = prepare_structure(
        X00242_PDB, _acid_rows(), LigandNamer(frozenset()), ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS
    ).ligand_codes["LIG"]
    real_codes = ccd_codes(ccd)
    monkeypatch.setattr(
        "aw_of3_finetune.prepare_inputs.ccd_codes", lambda _ccd: real_codes | {acid_code}
    )

    result = _run(input_dir, table, tmp_path / "out", ccd_path)

    assert result.exit_code == 1
    assert acid_code in result.output
    assert "CCD" in result.output


def _acid_rows():
    return [LigandRow("x00242-1", "LIG", ACID_SMILES, X00242_RESTRAINT)]


def test_output_dir_works_as_finetune_prep_cif_dir(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    assert _run(input_dir, table, prepared, ccd_path).exit_code == 0
    prep_out = tmp_path / "prep"

    result = CliRunner().invoke(
        finetune_prep,
        ["--cif-dir", str(prepared), "--output-dir", str(prep_out), "--ccd-path", str(ccd_path)],
    )

    assert result.exit_code == 0, result.output
    actual_ids = json.loads((prep_out / "manifest.json").read_text())
    assert "x00242-1" in json.dumps(actual_ids) and "x00358-1" in json.dumps(actual_ids)
    ccd = CIFFile.read(ccd_path)
    ligands = json.loads((prepared / LIGANDS_FILE).read_text())
    expected_smiles = {"x00242-1": ACID_SMILES, "x00358-1": INDOLE_SMILES}
    for stem, smiles in expected_smiles.items():
        codes = {code for code, v in ligands.items() if stem in v["structures"]}
        actual = reference_smiles_by_chain(prepared / f"{stem}.cif", codes, tmp_path / stem, ccd)
        actual_smiles = {value[1] for value in actual.values()}
        assert actual_smiles == {Chem.CanonSmiles(smiles)}


def test_rerun_removes_the_cif_of_a_structure_that_now_fails(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    out = tmp_path / "out"
    (out).mkdir()
    (out / "notes.txt").write_text("keep me")
    assert _run(input_dir, table, out, ccd_path).exit_code == 0
    assert (out / "x00242-1.cif").exists()

    bad_table = tmp_path / "bad.csv"
    bad_table.write_text(table.read_text().replace(ACID_SMILES, "c1ccccc1"))
    result = _run(input_dir, bad_table, out, ccd_path)

    assert result.exit_code == 0, result.output
    assert not (out / "x00242-1.cif").exists()
    assert "x00242-1" in json.loads((out / PREPARE_DROPPED_FILE).read_text())
    assert (out / "x00358-1.cif").exists()
    assert (out / "notes.txt").read_text() == "keep me"


def test_ligands_json_drops_codes_left_without_structures(tmp_path, ccd_path, monkeypatch):
    input_dir, table = _write_inputs(tmp_path)
    rows = read_ligand_table(table)
    ccd = CIFFile.read(ccd_path)

    def named_then_failed(path, stem_rows, namer, ccd, release_date):
        atoms, _ = _read_input(path)
        row = stem_rows[0]
        namer.name_for(_row_definition(atoms, row), row.smiles)
        raise RuntimeError("fails after naming the ligand")

    real_prepare = prepare_structure

    def fail_acid_only(path, stem_rows, namer, ccd, release_date, constructs):
        if path.stem == "x00242-1":
            return named_then_failed(path, stem_rows, namer, ccd, release_date)
        return real_prepare(path, stem_rows, namer, ccd, release_date, constructs)

    monkeypatch.setattr("aw_of3_finetune.prepare_inputs.prepare_structure", fail_acid_only)
    out = tmp_path / "out"

    prepare_directory(input_dir, rows, out, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    actual_records = json.loads((out / LIGANDS_FILE).read_text())
    actual_structures = [record["structures"] for record in actual_records.values()]
    assert actual_structures == [["x00358-1"]]


def test_row_for_a_residue_absent_from_the_structure_drops_the_structure(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    with table.open("a") as handle:
        handle.write(f"x00242-1,ZZZ,{ACID_SMILES},\n")
    out = tmp_path / "out"

    result = _run(input_dir, table, out, ccd_path)

    assert result.exit_code == 0, result.output
    actual_reason = json.loads((out / PREPARE_DROPPED_FILE).read_text())["x00242-1"]
    assert actual_reason.startswith("ValueError: ")
    assert "'ZZZ'" in actual_reason
    assert not (out / "x00242-1.cif").exists()


def test_constructs_give_the_prepared_cif_its_whole_sequence(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    out = tmp_path / "out"

    result = _run(input_dir, table, out, ccd_path)

    assert result.exit_code == 0, result.output
    block = CIFFile.read(out / "x00242-1.cif").block
    actual_sequence = block["entity_poly"]["pdbx_seq_one_letter_code_can"].as_array(str)[0].replace("\n", "")
    assert actual_sequence == PXR_CONSTRUCT


def test_gapped_structure_without_a_construct_is_dropped_with_the_reason(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    out = tmp_path / "out"
    args = ["--input-dir", str(input_dir), "--ligands", str(table), "--output-dir", str(out),
            "--ccd-path", str(ccd_path)]  # fmt: skip

    result = CliRunner().invoke(prepare_inputs, args)

    assert result.exit_code == 0, result.output
    actual_reasons = json.loads((out / PREPARE_DROPPED_FILE).read_text())
    assert sorted(actual_reasons) == ["x00242-1", "x00358-1"]
    assert "--constructs" in actual_reasons["x00242-1"]


def test_read_constructs_joins_wrapped_lines(tmp_path):
    path = tmp_path / "constructs.fasta"
    path.write_text(">one description\nACDE\nFGHI\n\n>two\nklmn\n")

    actual = read_constructs(path)

    assert actual == ("ACDEFGHI", "KLMN")


def test_constructs_file_without_a_sequence_is_a_usage_error(tmp_path, ccd_path):
    input_dir, table = _write_inputs(tmp_path)
    empty = tmp_path / "empty.fasta"
    empty.write_text("not fasta\n")

    result = _run(input_dir, table, tmp_path / "out", ccd_path, "--constructs", str(empty))

    assert result.exit_code == 2
    assert "empty.fasta" in result.output
