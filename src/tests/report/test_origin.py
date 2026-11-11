import json

import numpy as np
from biotite.structure import Atom, BondList, array
from biotite.structure.io import pdbx
from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz

from aw_of3_finetune.ligand_inputs import LigandNamer, LigandRow, ccd_codes
from aw_of3_finetune.prepare_inputs import DEFAULT_RELEASE_DATE, prepare_structure
from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.origin import INPUT, classify_origins, input_positions, v_check_origin
from aw_of3_finetune.report.validity import ValidityInputs
from aw_of3_finetune.report.validity.bonds import check_bond_lengths
from aw_of3_finetune.report.validity.ligands import COULD_NOT_RUN
from ligand_helpers import ACID_SMILES, PXR_CONSTRUCTS, X00242_PDB, X00242_RESTRAINT

PDB_ID = "tiny"
SINGLE = 1


def _atoms(xs, bonds):
    """Atoms A, B, ... of one residue XXX on chain A at the given x, joined by (i, j, type) bonds."""
    atoms = [
        Atom([x, 0.0, 0.0], chain_id="A", res_id=1, res_name="XXX", atom_name=chr(ord("A") + i), element="C",
             label_asym_id="A", hetero=True)
        for i, x in enumerate(xs)
    ]
    structure = array(atoms)
    structure.bonds = BondList(len(atoms), np.array(bonds, dtype=int).reshape(-1, 3))
    return structure


def write_origin_run(tmp_path, final_xs, input_xs, bonds=((0, 1, SINGLE),)):
    """A one-residue run whose NPZ has final_xs and whose input CIF cifs/tiny.cif has input_xs."""
    out_dir = tmp_path / "out"
    (out_dir / "of3_compressed_structures" / PDB_ID).mkdir(parents=True)
    (out_dir / "manifest.json").write_text(json.dumps({"structures": [PDB_ID]}))
    write_atomarray_to_npz(_atoms(final_xs, bonds), out_dir / "of3_compressed_structures" / PDB_ID / f"{PDB_ID}.npz")
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, _atoms(input_xs, bonds))
    (tmp_path / "cifs").mkdir()
    cif.write(tmp_path / "cifs" / f"{PDB_ID}.cif")
    return ValidityInputs(out_dir, tmp_path / "cifs", tmp_path / "ccd.cif")


def _ligand_finding(check, rows):
    """An unexplained ligand finding of chain A with the given rows."""
    return Finding(check, PDB_ID, Status.UNEXPLAINED, "XXX chain A: failed", {"rows": rows})


def _row(passed=False, **extra):
    """One ligand check row of chain A."""
    return {"chain": "A", "res_name": "XXX", "test": "some test", "passed": passed, "value": None,
            "threshold": None} | extra


def test_unmoved_atoms_are_input_positions(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])

    actual_positions = input_positions(inputs, PDB_ID)

    expected_positions = {0, 1}
    assert actual_positions == expected_positions


def test_deposited_long_bond_is_input(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0], [0.0, 3.0])
    findings = check_bond_lengths(inputs, PDB_ID)

    [actual] = classify_origins(inputs, findings)

    assert actual.data["origin"] == "input"
    assert actual.status == Status.EXPECTED
    assert actual.data["label"] == "input"
    assert actual.reason == "already in the input CIF"


def test_moved_atom_bond_is_processing(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0], [0.0, 1.5])
    findings = check_bond_lengths(inputs, PDB_ID)

    [actual] = classify_origins(inputs, findings)

    assert actual.data["origin"] == "processing"
    assert actual.status == Status.UNEXPLAINED


def test_match_findings_are_unchanged(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    findings = check_bond_lengths(inputs, PDB_ID)

    [actual] = classify_origins(inputs, findings)

    expected = findings[0]
    assert actual == expected
    assert "origin" not in actual.data


def test_ligand_chemistry_is_always_input(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    finding = _ligand_finding("validity.ligand_chemistry", [_row()])

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "input"
    assert actual.status == Status.EXPECTED


def test_ligand_identity_against_ccd_is_input(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    finding = _ligand_finding("validity.ligand_identity", [_row()])

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "input"


def test_ligand_identity_against_input_is_processing(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    finding = _ligand_finding("validity.ligand_identity", [_row(reference="ccd"), _row(reference="input")])

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "processing"
    assert actual.status == Status.UNEXPLAINED


def test_could_not_run_rows_are_not_checked(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    rows = [_row(passed=True), _row(label="unexplained", reason=COULD_NOT_RUN)]
    finding = _ligand_finding("validity.ligand_geometry", rows)

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "not_checked"
    assert actual.status == Status.EXPECTED
    assert actual.data["label"] == "not_checked"
    assert actual.reason == "check could not run"


def test_ligand_geometry_at_input_positions_is_input(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    finding = _ligand_finding("validity.ligand_geometry", [_row()])

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "input"


def test_crashed_check_is_not_checked(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    finding = Finding("validity.bond_lengths", PDB_ID, Status.UNEXPLAINED, "check failed: KeyError", {"error": True})

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "not_checked"


def test_validity_data_error_is_processing(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    finding = Finding("validity_data.error", PDB_ID, Status.UNEXPLAINED, "KeyError", {"error": True})

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "processing"


def test_cleanup_audit_failure_is_processing(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    finding = Finding("validity.cleanup_audit", PDB_ID, Status.UNEXPLAINED, "1 atom moved", {"rows": []})

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "processing"
    assert actual.status == Status.UNEXPLAINED


def test_real_cleanup_keeps_every_atom_at_input(built_7riv, ccd_path):
    inputs = ValidityInputs(built_7riv.output_dir, built_7riv.cif_dir, ccd_path)
    final = inputs.final_atoms("7riv")
    resolved = ~np.isnan(final.coord).any(axis=1)
    final_resolved_positions = set(np.flatnonzero(resolved).tolist())

    actual_missing = final_resolved_positions - input_positions(inputs, "7riv")

    expected_missing = set()
    assert actual_missing == expected_missing


def test_ligand_geometry_with_moved_atom_is_processing(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0], [0.0, 1.5])
    finding = _ligand_finding("validity.ligand_geometry", [_row()])

    [actual] = classify_origins(inputs, [finding])

    assert actual.data["origin"] == "processing"


def test_unreadable_input_cif_is_not_checked(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0], [0.0, 3.0])
    findings = check_bond_lengths(inputs, PDB_ID)
    (tmp_path / "cifs" / f"{PDB_ID}.cif").write_text("not a cif\n")

    [actual] = classify_origins(inputs, findings)

    expected_reason_start = "origin could not be determined: "
    assert actual.data["origin"] == "not_checked"
    assert actual.status == Status.EXPECTED
    assert actual.reason.startswith(expected_reason_start)


def test_release_date_defaulted_by_prepare_inputs_is_an_input_finding(tmp_path, ccd_path):
    ccd = pdbx.CIFFile.read(ccd_path)
    row = LigandRow("x00242-1", "LIG", ACID_SMILES, X00242_RESTRAINT)
    prepared = prepare_structure(X00242_PDB, [row], LigandNamer(ccd_codes(ccd)), ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "x00242-1.cif").write_text(prepared.cif_text)
    inputs = ValidityInputs(tmp_path, cif_dir, ccd_path)
    message = f"release_date defaulted by prepare-inputs: {DEFAULT_RELEASE_DATE}"
    item = {"check": "V17", "severity": "W", "entry": "x00242-1", "message": message, "detail": {}}

    actual_origin = v_check_origin(inputs, item)

    assert actual_origin == INPUT
