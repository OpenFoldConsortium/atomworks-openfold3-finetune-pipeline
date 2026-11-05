import json

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.highlights import issue_highlights
from aw_of3_finetune.report.origin import classify_origins
from aw_of3_finetune.report.user import _write_3d_data
from aw_of3_finetune.report.validity.bonds import check_bond_lengths
from report.test_origin import PDB_ID, _ligand_finding, _row, write_origin_run


def test_input_long_bond_is_one_issue_at_its_two_atoms(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0], [0.0, 3.0])
    findings = classify_origins(inputs, check_bond_lengths(inputs, PDB_ID))

    actual = issue_highlights(inputs, PDB_ID, findings)

    expected = [{
        "origin": "input",
        "check": "validity.bond_lengths",
        "label": "A:XXX1:A-A:XXX1:B 3.000 Å, too long (max 2.4 Å)",
        "xyz": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    }]
    assert actual == expected


def test_each_bad_bond_is_its_own_issue(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0, 6.0], [0.0, 3.0, 6.0], bonds=((0, 1, 1), (1, 2, 1)))
    findings = classify_origins(inputs, check_bond_lengths(inputs, PDB_ID))

    actual = issue_highlights(inputs, PDB_ID, findings)

    actual_labels = [issue["label"] for issue in actual]
    expected_labels = ["A:XXX1:A-A:XXX1:B 3.000 Å, too long (max 2.4 Å)", "A:XXX1:B-A:XXX1:C 3.000 Å, too long (max 2.4 Å)"]
    assert actual_labels == expected_labels


def test_processing_bond_keeps_its_origin(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0], [0.0, 1.5])
    findings = classify_origins(inputs, check_bond_lengths(inputs, PDB_ID))

    [actual] = issue_highlights(inputs, PDB_ID, findings)

    assert actual["origin"] == "processing"


def test_ligand_finding_highlights_the_ligand_chain(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    findings = classify_origins(inputs, [_ligand_finding("validity.ligand_chemistry", [_row()])])

    actual = issue_highlights(inputs, PDB_ID, findings)

    expected = [{
        "origin": "input",
        "check": "validity.ligand_chemistry",
        "label": "XXX chain A: failed",
        "xyz": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
    }]
    assert actual == expected


def test_matches_and_findings_without_atoms_are_left_out(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    audit = Finding("validity.cleanup_audit", PDB_ID, Status.UNEXPLAINED, "atom count differs", {})
    findings = classify_origins(inputs, [*check_bond_lengths(inputs, PDB_ID), audit])

    actual = issue_highlights(inputs, PDB_ID, findings)

    assert actual == []


def test_not_checked_findings_are_left_out(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 1.5], [0.0, 1.5])
    crashed = Finding("validity.ligand_geometry", PDB_ID, Status.UNEXPLAINED, "crashed", {"error": True})
    findings = classify_origins(inputs, [crashed])

    actual = issue_highlights(inputs, PDB_ID, findings)

    assert actual == []


def test_other_entries_findings_are_ignored(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0], [0.0, 3.0])
    findings = classify_origins(inputs, check_bond_lengths(inputs, PDB_ID))

    actual = issue_highlights(inputs, "other", findings)

    assert actual == []


def test_3d_data_carries_the_entry_issues(tmp_path):
    inputs = write_origin_run(tmp_path, [0.0, 3.0], [0.0, 3.0])
    findings = classify_origins(inputs, check_bond_lengths(inputs, PDB_ID))

    _write_3d_data(inputs, [PDB_ID], tmp_path / "validate_3d", findings)

    view = json.loads((tmp_path / "validate_3d" / f"{PDB_ID}.json").read_text())
    actual_labels = [issue["label"] for issue in view["issues"]]
    expected_labels = ["A:XXX1:A-A:XXX1:B 3.000 Å, too long (max 2.4 Å)"]
    assert actual_labels == expected_labels
